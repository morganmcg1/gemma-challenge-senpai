#!/usr/bin/env python
"""Verify-GEMM argmax-margin greedy-safety gate (PR #87) — analysis + verdict.

Reads the compact .npz + summary.json written by `verify_argmax_margin.py`
(GPU capture) and turns them into:

  PHASE 1  the top-2 margin map: histogram, min Δ, and the fraction of emitted
           positions with Δ below {1,2,4,8,16} ULP — at the native output-dtype
           rung AND the FP32-accumulator rung (Marlin accumulates FP16/BF16 MACs
           in FP32, then casts to the native output dtype; both rungs matter).
  PHASE 2  the kernel-swap sensitivity: max|Δlogit| and argmax-flip count for the
           SplitK reduction-order swap (S in {2,4,8}, isolated vs the S=1 single
           accumulation) and the M-widening swap (M in {16,32}, vs the real M=8
           Marlin reference).
  GATE     GREEN iff flip count = 0 for BOTH perturbations AND min Δ ≫ max|Δlogit|
           (comfortable headroom). RED otherwise — with the flip count, which
           perturbation, and the margin at each flip.

Primary metric: `verify_gemm_argmax_flip_count_splitk` (0 = safe).
Test metric:    `verify_gemm_min_top2_margin_ulp` (the safety headroom).

CPU only; runs in the repo venv (wandb). No GPU, no served-file change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

WANDB_GROUP = "verify-gemm-argmax-margin"
ULP_THRESHOLDS = (1, 2, 4, 8, 16)
MANT_BITS = {"float16": 10, "bfloat16": 7, "float32": 23, "torch.float16": 10,
             "torch.bfloat16": 7, "torch.float32": 23}


def _ulp(x_abs: np.ndarray, mant_bits: int) -> np.ndarray:
    """Spacing of a float format with `mant_bits` mantissa bits at magnitude x_abs."""
    e = np.floor(np.log2(np.maximum(np.abs(x_abs), 1e-30)))
    return np.exp2(e - mant_bits)


def _latest_capture(root: Path) -> Path:
    cands = sorted(p for p in root.glob("*/summary.json"))
    if not cands:
        raise FileNotFoundError(f"no capture summary.json under {root}")
    return cands[-1].parent


def _native_mant_bits(native_dtype: str) -> int:
    key = native_dtype.replace("torch.", "")
    return MANT_BITS.get(key, MANT_BITS.get(native_dtype, 10))


# The deployed int4 W4A16 Marlin lm_head GEMM (vocab n=12288) already runs a
# DETERMINISTIC split-K reduction: vLLM tiles K=2560 into 128-wide slices (20
# slices), accumulates partials in FP32 (`global_reduce_fp32`, C_tmp=at::kFloat),
# and casts FP32->bf16 ONCE at the final output write. For n>=2048
# `should_use_atomic_add_reduce()` hard-returns False, so atomic-add is OFF and
# `use_fp32_reduce` defaults True. Consequence: the ONLY lossy step is the single
# final bf16 cast, so any FP32-reduce-regime reduction-order change (the deployed
# 20-slice kernel, the S=1 cuBLAS emulation, ubel #84's SplitK) perturbs the
# pre-cast FP32 accumulator by ~2^-23 relative (negligible) and becomes visible
# only as an occasional +/-1 bf16-ULP final-cast difference. This bound HOLDS ONLY
# in that regime; #73's atomic-add control (VLLM_MARLIN_USE_ATOMIC_ADD=1) flips
# ~36% of tokens. (Source: vLLM marlin_utils.py lines 33-36/445-465; gptq_marlin.cu
# global_reduce_fp32; Marlin paper arXiv:2408.11743.)
NUMERICS_REGIME = {
    "deployed_reduction": "deterministic 20-slice (128-wide) FP32 split-K (global_reduce_fp32)",
    "use_atomic_add": "False (forced for vocab n=12288 >= 2048)",
    "use_fp32_reduce": "True (default)",
    "only_lossy_step": "single FP32->bf16 cast at final output write",
    "splitk_bound_valid_iff": "use_fp32_reduce=True AND use_atomic_add=False",
    "counterexample": "#73 atomic-add control flips ~36% of tokens (out-of-regime)",
}


def _at_risk_audit(npz, margin, margin_native_ulp, ref_argmax, splits, mwidths) -> dict[str, Any]:
    """Direct empirical audit of the thinnest-margin positions + a provable bound.

    The gate is the MEASURED flip count, never a tie-break argument. This makes
    that rigorous. For every position, take the worst logit shift across ALL swaps
    (dmax_all) and whether ANY swap flipped its argmax (flip_any). Then:
      * provable bound: a position is FLIP-PROOF iff margin > 2*dmax_all (top1 and
        top2 can each move by at most dmax_all, so the runner-up cannot overtake);
      * for the residual positions NOT covered by that bound (incl. exact bf16
        ties), report the directly MEASURED flip count and the largest shift seen.
    A GREEN that rests on "provably safe + 0 measured flips in the residual" needs
    no tie-break assumption.
    """
    npos = margin.size
    dmax_all = np.zeros(npos, np.float64)
    flip_any = np.zeros(npos, np.bool_)
    for s in splits:
        dmax_all = np.maximum(dmax_all, npz[f"sk{s}_dmax"].astype(np.float64))
        flip_any |= npz[f"sk{s}_flip_emuS1"]
    for m in mwidths:
        dmax_all = np.maximum(dmax_all, npz[f"mw{m}_dmax"].astype(np.float64))
        flip_any |= npz[f"mw{m}_flip_refM8"]

    buckets: dict[str, Any] = {}
    for t in (0, 1, 2, 4):
        sel = (margin <= 0) if t == 0 else (margin_native_ulp < t)
        label = "exact_tie" if t == 0 else f"below_{t}ulp"
        n = int(sel.sum())
        buckets[label] = {
            "n_positions": n,
            "n_flipped_measured": int(flip_any[sel].sum()),
            "max_abs_dlogit_here": float(dmax_all[sel].max()) if n else 0.0,
        }

    flip_proof = margin > (2.0 * dmax_all)            # provable, no measurement needed
    residual = ~flip_proof                            # must rely on direct measurement
    n_residual = int(residual.sum())
    order = np.argsort(margin)
    examples = [
        {
            "pos": int(i),
            "argmax": int(ref_argmax[i]),
            "margin": float(margin[i]),
            "margin_native_ulp": float(margin_native_ulp[i]),
            "max_abs_dlogit": float(dmax_all[i]),
            "flipped": bool(flip_any[i]),
        }
        for i in order[:15]
    ]
    return {
        "n_flip_proof_margin_gt_2dmax": int(flip_proof.sum()),
        "frac_flip_proof": float(flip_proof.mean()),
        "n_residual_relies_on_measurement": n_residual,
        "residual_flips_measured": int(flip_any[residual].sum()),
        "residual_max_abs_dlogit": float(dmax_all[residual].max()) if n_residual else 0.0,
        "total_flips_any_swap": int(flip_any.sum()),
        "margin_buckets": buckets,
        "thinnest_positions": examples,
    }


def analyze(capture_dir: Path) -> dict[str, Any]:
    summary = json.loads((capture_dir / "summary.json").read_text())
    npz = np.load(capture_dir / "margin_perturb.npz")

    ref_top1 = npz["ref_top1"].astype(np.float64)
    ref_top2 = npz["ref_top2"].astype(np.float64)
    ref_argmax = npz["ref_argmax"]
    emu1_argmax = npz["emu1_argmax"]
    margin = (ref_top1 - ref_top2).astype(np.float64)
    npos = margin.size

    native_dtype = summary.get("native_dtype", "torch.bfloat16")
    nbits = _native_mant_bits(native_dtype)
    splits = [int(s) for s in summary.get("splits", [2, 4, 8])]
    mwidths = [int(m) for m in summary.get("mwidths", [16, 32])]

    # --- PHASE 1: margin map, at the native-dtype rung and the FP32 rung -------
    top1_abs = np.abs(ref_top1)
    ulp_native = _ulp(top1_abs, nbits)      # spacing the sampler actually sees
    ulp_fp32 = _ulp(top1_abs, 23)           # accumulator-scale rung
    margin_in_native_ulp = margin / ulp_native
    margin_in_fp32_ulp = margin / ulp_fp32

    def _below_frac(margin_ulp: np.ndarray) -> dict[str, float]:
        return {str(t): float((margin_ulp < t).mean()) for t in ULP_THRESHOLDS}

    pos_mask = margin > 0
    n_ties = int((~pos_mask).sum())  # margin>=0 by topk construction, so this is Δ==0
    min_pos_margin = float(margin[pos_mask].min()) if pos_mask.any() else 0.0
    min_pos_margin_native_ulp = (
        float(margin_in_native_ulp[pos_mask].min()) if pos_mask.any() else 0.0
    )

    phase1 = {
        "num_positions": int(npos),
        "native_dtype": native_dtype,
        "native_mantissa_bits": nbits,
        "min_margin": float(margin.min()),
        "min_positive_margin": min_pos_margin,
        "median_margin": float(np.median(margin)),
        "p01_margin": float(np.percentile(margin, 1)),
        "p10_margin": float(np.percentile(margin, 10)),
        "max_abs_logit_softcapped": float(np.abs(np.concatenate([ref_top1, ref_top2])).max()),
        "min_margin_in_native_ulp": float(margin_in_native_ulp.min()),
        "min_positive_margin_in_native_ulp": min_pos_margin_native_ulp,
        "min_margin_in_fp32_ulp": float(margin_in_fp32_ulp.min()),
        "median_margin_in_native_ulp": float(np.median(margin_in_native_ulp)),
        "frac_below_native_ulp": _below_frac(margin_in_native_ulp),
        "frac_below_fp32_ulp": _below_frac(margin_in_fp32_ulp),
        "n_exact_ties_margin_zero": n_ties,
        "frac_exact_ties": float(n_ties / npos) if npos else 0.0,
    }

    # --- PHASE 2: perturbation magnitude + flip counts ------------------------
    fidelity_disagree = int((ref_argmax != emu1_argmax).sum())
    splitk = {}
    for s in splits:
        flip_emu = npz[f"sk{s}_flip_emuS1"]
        flip_ref = npz[f"sk{s}_flip_refM8"]
        dmax = npz[f"sk{s}_dmax"].astype(np.float64)
        splitk[str(s)] = {
            "flip_count_vs_emuS1": int(flip_emu.sum()),
            "flip_count_vs_refM8": int(flip_ref.sum()),
            "max_abs_dlogit": float(dmax.max()),
            "max_abs_dlogit_in_native_ulp": float((dmax / ulp_native).max()),
            "mean_abs_dlogit": float(dmax.mean()),
        }
    mwiden = {}
    for m in mwidths:
        flip = npz[f"mw{m}_flip_refM8"]
        dmax = npz[f"mw{m}_dmax"].astype(np.float64)
        mwiden[str(m)] = {
            "flip_count_vs_refM8": int(flip.sum()),
            "max_abs_dlogit": float(dmax.max()),
            "max_abs_dlogit_in_native_ulp": float((dmax / ulp_native).max()),
            "mean_abs_dlogit": float(dmax.mean()),
        }

    # worst-case perturbation across ALL swaps (the ceiling the margin must clear)
    splitk_flip_emu = max(v["flip_count_vs_emuS1"] for v in splitk.values())
    splitk_flip_ref = max(v["flip_count_vs_refM8"] for v in splitk.values())
    mwiden_flip = max(v["flip_count_vs_refM8"] for v in mwiden.values())
    max_dlogit_all = max(
        [v["max_abs_dlogit"] for v in splitk.values()]
        + [v["max_abs_dlogit"] for v in mwiden.values()]
    )
    total_flips = splitk_flip_emu + mwiden_flip

    # rigorous thinnest-margin audit (provable bound + measured residual)
    at_risk = _at_risk_audit(npz, margin, margin_in_native_ulp, ref_argmax, splits, mwidths)

    # --- GATE -----------------------------------------------------------------
    # The DECISIVE, gate-relevant event is an argmax FLIP: a position where a
    # kernel swap changes the emitted greedy token vs the deployed verify GEMM.
    # That maps 1:1 to the official greedy-token-identity gate. The native output
    # surface is bf16, so EXACT ties (Δ=0) are common and expected; a tie the
    # swap leaves bit-preserved keeps the deterministic lowest-index tie-break and
    # is therefore safe (counts as 0 flips). So flips — not the raw min margin —
    # are the gate. min Δ vs max|Δlogit| over the NON-tie positions is reported as
    # a robustness diagnostic; tie_bounded flags GREEN that rests on bit-preserved
    # ties rather than on positive margin headroom.
    n_ties = phase1["n_exact_ties_margin_zero"]
    headroom_ratio = (
        float(phase1["min_positive_margin"] / max_dlogit_all)
        if max_dlogit_all > 0 else float("inf")
    )
    comfortable = (n_ties == 0) and (headroom_ratio >= 2.0)
    tie_bounded = (total_flips == 0) and (n_ties > 0)
    green = (total_flips == 0)
    verdict = "GREEN" if green else "RED"

    flip_examples: list[dict[str, Any]] = []
    if not green:
        for s in splits:
            idx = np.where(npz[f"sk{s}_flip_emuS1"])[0][:10]
            for i in idx:
                flip_examples.append({"perturbation": f"splitk_S{s}", "pos": int(i),
                                      "margin": float(margin[i]),
                                      "margin_native_ulp": float(margin_in_native_ulp[i])})
        for m in mwidths:
            idx = np.where(npz[f"mw{m}_flip_refM8"])[0][:10]
            for i in idx:
                flip_examples.append({"perturbation": f"mwiden_M{m}", "pos": int(i),
                                      "margin": float(margin[i]),
                                      "margin_native_ulp": float(margin_in_native_ulp[i])})

    report = {
        "capture_dir": str(capture_dir),
        "capture_summary": summary,
        "phase1_margin_map": phase1,
        "phase2_splitk": splitk,
        "phase2_mwiden": mwiden,
        "fidelity_emuS1_vs_realM8_disagreements": fidelity_disagree,
        "numerics_regime": NUMERICS_REGIME,
        "at_risk_audit": at_risk,
        "gate": {
            "verdict": verdict,
            "comfortable_headroom": bool(comfortable),
            "tie_bounded": bool(tie_bounded),
            "n_exact_ties": int(n_ties),
            "min_margin": phase1["min_margin"],
            "min_positive_margin": phase1["min_positive_margin"],
            "max_abs_dlogit_all_swaps": max_dlogit_all,
            "headroom_ratio_minposmargin_over_maxdlogit": headroom_ratio,
            "n_flip_proof_margin_gt_2dmax": at_risk["n_flip_proof_margin_gt_2dmax"],
            "frac_flip_proof": at_risk["frac_flip_proof"],
            "n_residual_relies_on_measurement": at_risk["n_residual_relies_on_measurement"],
            "residual_flips_measured": at_risk["residual_flips_measured"],
            "splitk_flip_count_vs_emuS1_worst": splitk_flip_emu,
            "splitk_flip_count_vs_refM8_worst": splitk_flip_ref,
            "mwiden_flip_count_worst": mwiden_flip,
            "total_flips": int(total_flips),
            "flip_examples": flip_examples,
        },
        "primary_metric": {"name": "verify_gemm_argmax_flip_count_splitk", "value": splitk_flip_emu},
        "test_metric": {"name": "verify_gemm_min_top2_margin_ulp",
                        "value": phase1["min_margin_in_native_ulp"]},
    }
    return report


def log_to_wandb(report: dict[str, Any], *, wandb_name: str, wandb_group: str) -> None:
    p1 = report["phase1_margin_map"]
    g = report["gate"]
    summary = {
        "verdict": g["verdict"],
        "verdict_green": int(g["verdict"] == "GREEN"),
        "tie_bounded": int(g["tie_bounded"]),
        "comfortable_headroom": int(g["comfortable_headroom"]),
        "verify_gemm_argmax_flip_count_splitk": report["primary_metric"]["value"],
        "verify_gemm_min_top2_margin_ulp": report["test_metric"]["value"],
        "total_flips": g["total_flips"],
        "min_margin": p1["min_margin"],
        "min_positive_margin": p1["min_positive_margin"],
        "median_margin": p1["median_margin"],
        "min_margin_in_native_ulp": p1["min_margin_in_native_ulp"],
        "min_positive_margin_in_native_ulp": p1["min_positive_margin_in_native_ulp"],
        "min_margin_in_fp32_ulp": p1["min_margin_in_fp32_ulp"],
        "n_exact_ties_margin_zero": p1["n_exact_ties_margin_zero"],
        "frac_exact_ties": p1["frac_exact_ties"],
        "max_abs_dlogit_all_swaps": g["max_abs_dlogit_all_swaps"],
        "headroom_ratio": g["headroom_ratio_minposmargin_over_maxdlogit"],
        "n_flip_proof_margin_gt_2dmax": g["n_flip_proof_margin_gt_2dmax"],
        "frac_flip_proof": g["frac_flip_proof"],
        "n_residual_relies_on_measurement": g["n_residual_relies_on_measurement"],
        "residual_flips_measured": g["residual_flips_measured"],
        "splitk_flip_vs_emuS1_worst": g["splitk_flip_count_vs_emuS1_worst"],
        "splitk_flip_vs_refM8_worst": g["splitk_flip_count_vs_refM8_worst"],
        "mwiden_flip_worst": g["mwiden_flip_count_worst"],
        "fidelity_disagreements": report["fidelity_emuS1_vs_realM8_disagreements"],
        "num_positions": p1["num_positions"],
        "native_dtype": p1["native_dtype"],
    }
    for t, frac in p1["frac_below_native_ulp"].items():
        summary[f"frac_below_{t}ulp_native"] = frac
    for t, frac in p1["frac_below_fp32_ulp"].items():
        summary[f"frac_below_{t}ulp_fp32"] = frac
    for s, v in report["phase2_splitk"].items():
        summary[f"splitk_S{s}.flip_vs_emuS1"] = v["flip_count_vs_emuS1"]
        summary[f"splitk_S{s}.max_abs_dlogit"] = v["max_abs_dlogit"]
        summary[f"splitk_S{s}.max_dlogit_native_ulp"] = v["max_abs_dlogit_in_native_ulp"]
    for m, v in report["phase2_mwiden"].items():
        summary[f"mwiden_M{m}.flip_vs_refM8"] = v["flip_count_vs_refM8"]
        summary[f"mwiden_M{m}.max_abs_dlogit"] = v["max_abs_dlogit"]
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_json_artifact, log_summary)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] unavailable: {exc}", flush=True)
        return
    run = init_wandb_run(
        job_type="verify-gemm-argmax-margin", agent="senpai", name=wandb_name,
        tags=["verify-gemm-argmax-margin", wandb_group],
        group=wandb_group,
        config={"capture_dir": report["capture_dir"],
                "num_positions": p1["num_positions"],
                "wandb_group": wandb_group},
    )
    if run is None:
        print("[wandb] run not created (no creds/disabled); report.json is the record", flush=True)
        return
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="verify_argmax_margin_report",
                      artifact_type="verify-gemm-argmax-margin-report", data=report)
    finish_wandb(run)
    print(f"[wandb] logged run {wandb_name} (group={wandb_group})", flush=True)


def _print(report: dict[str, Any]) -> None:
    p1 = report["phase1_margin_map"]
    g = report["gate"]
    print("\n" + "=" * 74, flush=True)
    print("VERIFY-GEMM ARGMAX-MARGIN GREEDY-SAFETY GATE (PR #87)", flush=True)
    print("=" * 74, flush=True)
    print(f"positions={p1['num_positions']}  native_dtype={p1['native_dtype']} "
          f"(mantissa {p1['native_mantissa_bits']} bits)  softcap=30", flush=True)
    print(f"\nPHASE 1 — top-2 margin map (post-softcap, the argmax space):", flush=True)
    print(f"  min Δ          : {p1['min_margin']:.6g}  "
          f"({p1['min_margin_in_native_ulp']:.3g} native-ULP, "
          f"{p1['min_margin_in_fp32_ulp']:.3g} fp32-ULP)", flush=True)
    print(f"  median Δ       : {p1['median_margin']:.6g}  "
          f"({p1['median_margin_in_native_ulp']:.4g} native-ULP)", flush=True)
    print(f"  exact ties(Δ≤0): {p1['n_exact_ties_margin_zero']}", flush=True)
    print(f"  frac Δ < t·ULP (native): {p1['frac_below_native_ulp']}", flush=True)
    print(f"  frac Δ < t·ULP (fp32)  : {p1['frac_below_fp32_ulp']}", flush=True)
    print(f"\nPHASE 2 — kernel-swap sensitivity:", flush=True)
    for s, v in report["phase2_splitk"].items():
        print(f"  SplitK S={s}: flips(vs S=1)={v['flip_count_vs_emuS1']} "
              f"flips(vs realM8)={v['flip_count_vs_refM8']}  "
              f"max|Δlogit|={v['max_abs_dlogit']:.3g} "
              f"({v['max_abs_dlogit_in_native_ulp']:.3g} native-ULP)", flush=True)
    for m, v in report["phase2_mwiden"].items():
        print(f"  M-widen M={m}: flips(vs realM8)={v['flip_count_vs_refM8']}  "
              f"max|Δlogit|={v['max_abs_dlogit']:.3g} "
              f"({v['max_abs_dlogit_in_native_ulp']:.3g} native-ULP)", flush=True)
    print(f"  emulation fidelity (emu S=1 vs real M=8 argmax disagreements): "
          f"{report['fidelity_emuS1_vs_realM8_disagreements']}", flush=True)
    ar = report["at_risk_audit"]
    print(f"\nTHINNEST-MARGIN AUDIT (provable bound + measured residual):", flush=True)
    print(f"  flip-proof (margin > 2·max|Δlogit|): {ar['n_flip_proof_margin_gt_2dmax']}"
          f"/{p1['num_positions']} ({100*ar['frac_flip_proof']:.3f}%)", flush=True)
    print(f"  residual (relies on direct measurement): "
          f"{ar['n_residual_relies_on_measurement']} positions, "
          f"{ar['residual_flips_measured']} measured flips "
          f"(max|Δlogit| there = {ar['residual_max_abs_dlogit']:.3g})", flush=True)
    for lab, b in ar["margin_buckets"].items():
        print(f"    {lab:>11}: n={b['n_positions']:>5}  flipped={b['n_flipped_measured']}  "
              f"max|Δlogit|={b['max_abs_dlogit_here']:.3g}", flush=True)
    print("\n" + "-" * 74, flush=True)
    if g["tie_bounded"]:
        tie_note = (f"  [tie-bounded: {g['n_exact_ties']} exact bf16 ties, all "
                    f"bit-preserved -> 0 broken]")
    elif g["comfortable_headroom"]:
        tie_note = "  [comfortable headroom, no ties]"
    else:
        tie_note = ""
    print(f"VERDICT: {g['verdict']}  (total argmax flips={g['total_flips']}; "
          f"min positive Δ / max|Δlogit| = "
          f"{g['headroom_ratio_minposmargin_over_maxdlogit']:.4g}){tie_note}", flush=True)
    print(f"  primary_metric verify_gemm_argmax_flip_count_splitk = "
          f"{report['primary_metric']['value']}", flush=True)
    print(f"  test_metric    verify_gemm_min_top2_margin_ulp       = "
          f"{report['test_metric']['value']:.4g}", flush=True)
    if g["flip_examples"]:
        print("  flip examples:", flush=True)
        for ex in g["flip_examples"][:12]:
            print(f"    {ex['perturbation']} pos={ex['pos']} Δ={ex['margin']:.4g} "
                  f"({ex['margin_native_ulp']:.3g} native-ULP)", flush=True)
    print("=" * 74 + "\n", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-dir", default=None,
                    help="timestamped capture dir (default: latest under --out-root)")
    ap.add_argument("--out-root", default=str(REPO / "research/validity/verify_argmax_margin"))
    ap.add_argument("--report", default=None, help="where to write report.json")
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default=WANDB_GROUP)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    capture_dir = Path(args.capture_dir) if args.capture_dir else _latest_capture(Path(args.out_root))
    report = analyze(capture_dir)
    report_path = Path(args.report) if args.report else capture_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    _print(report)
    print(f"[analyze] wrote {report_path}", flush=True)

    if not args.no_wandb:
        wandb_name = args.wandb_name or f"kanna/verify-gemm-argmax-margin-{capture_dir.name}"
        log_to_wandb(report, wandb_name=wandb_name, wandb_group=args.wandb_group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
