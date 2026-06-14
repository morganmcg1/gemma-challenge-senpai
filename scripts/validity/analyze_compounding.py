#!/usr/bin/env python
"""Network-wide greedy-compounding gate (PR #96) — analysis + verdict.

Reads the compact .npz + summary.json written by `greedy_compounding.py` (GPU
capture) and turns them into the gate that CLOSES the one residual PR #87 left
open: upstream network-wide compounding of per-GEMM ≤1-bf16-ULP reduction-order
perturbations through ~30 residual-stream layers + RMSNorms + lm_head.

  STEP 1  in-process determinism control: the deployed stack must reproduce the
          same 65,536 greedy argmax run-to-run with perturbation OFF (N≥3, 0
          divergent). A non-deterministic baseline would make any perturbed
          comparison meaningless, so this is a HARD precondition for GREEN.
  STEP 2  compounded argmax-flip count vs the control, for the two injection
          regimes the capture ran network-wide:
            (A) realistic   GENUINE SplitK reduction-order emulation per op
                            (delta = emu_S - emu_1 over the recovered exact W,
                            S∈{2,4,8}) — the true in-regime ≤1-ULP compounding
                            bound #87 measured (primary metric; max over splits).
            (B) adversarial sign(y)·ULP per op + targeted lm_head shift — the
                            sign-aligned linear-compounding upper bound (test).
          Each regime reports BOTH the full-frontier flip (lm_head re-tiled too)
          and the upstream-only flip (native lm_head over the propagated h —
          isolates the new network-wide-h compounding #96 actually closes).
  STEP 3  the GATE. GREEN iff realistic flips = 0 (and control deterministic);
          AMBER iff a small enumerable set (< ~0.1%); RED iff ≥ ~0.1%. Any flips
          are CROSS-TABULATED against #87's thin-margin watch-set (the 907 exact
          bf16 ties + the 1226 measurement-dependent residual positions): flips
          that land only on already-thin positions are the expected tie wobble;
          a flip on a #87 *flip-proof* position is a genuine compounding signal.

Primary metric: `compounded_argmax_flip_count_realistic` (0 = residual closed).
Test metric:    `compounded_argmax_flip_count_adversarial` (worst-case bound).

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

WANDB_GROUP = "greedy-compounding-gate"
ULP_THRESHOLDS = (1, 2, 4, 8, 16)
MANT_BITS = {"float16": 10, "bfloat16": 7, "float32": 23, "torch.float16": 10,
             "torch.bfloat16": 7, "torch.float32": 23}
RED_FRAC = 0.001          # ≥ ~0.1% of emitted positions flipping -> RED
AMBER_ENUMERATE_CAP = 64  # list at most this many flip positions in the report
# Minimum argmax agreement between this capture's greedy trajectory and #87's for
# the positional cross-tab to be valid. #87 decoded with max_num_seqs=1 and this
# run with max_num_seqs=32; that decode-batch (reduction-order) difference cascades
# through the autoregressive loop, so the two position->token maps can diverge.
# Below this, a position i in #87's margin map is a DIFFERENT token context than
# position i here, and the cross-tab is positional coincidence, not signal.
TRAJ_ALIGN_MIN = 0.90

# The default #87 margin map: its ref_top1/ref_top2 + per-swap dmax/flip arrays
# define the thin-margin watch-set (907 exact ties, 1226 measurement-dependent).
DEFAULT_REF_NPZ = (REPO / "research/validity/verify_argmax_margin/"
                          "20260614T041541Z/margin_perturb.npz")


def _ulp(x_abs: np.ndarray, mant_bits: int) -> np.ndarray:
    """Spacing of a float format with `mant_bits` mantissa bits at magnitude x_abs."""
    e = np.floor(np.log2(np.maximum(np.abs(x_abs), 1e-30)))
    return np.exp2(e - mant_bits)


def _native_mant_bits(native_dtype: str) -> int:
    key = native_dtype.replace("torch.", "")
    return MANT_BITS.get(key, MANT_BITS.get(native_dtype, 10))


def _latest_capture(root: Path) -> Path:
    cands = sorted(p for p in root.glob("*/summary.json") if "smoke-" not in p.parent.name)
    if not cands:
        raise FileNotFoundError(f"no capture summary.json under {root}")
    return cands[-1].parent


def _load_87_watchset(ref_npz_path: str | Path, npos: int) -> dict[str, Any] | None:
    """Rebuild #87's thin-margin watch-set from its margin_perturb.npz.

    flip-proof  : margin > 2·max|Δlogit| over ALL #87 isolated swaps (provably
                  cannot flip from a single-GEMM reduction-order change).
    residual    : NOT flip-proof — relies on direct measurement (1226 in #87).
    tie         : margin ≤ 0 — exact bf16 ties (907 in #87), a subset of residual.
    Returns None if the file is missing or its position count does not match the
    capture (different config -> cross-tab would be meaningless).
    """
    p = Path(ref_npz_path)
    if not p.exists():
        return None
    npz = np.load(p)
    files = set(npz.files)
    if not {"ref_top1", "ref_top2"} <= files:
        return None
    t1 = npz["ref_top1"].astype(np.float64)
    t2 = npz["ref_top2"].astype(np.float64)
    if t1.size != npos:
        return None
    margin = t1 - t2
    dmax_all = np.zeros(margin.size, np.float64)
    isolated_flip = np.zeros(margin.size, np.bool_)
    for k in files:
        if k.endswith("_dmax"):
            dmax_all = np.maximum(dmax_all, npz[k].astype(np.float64))
        if k.startswith("sk") and k.endswith("_flip_emuS1"):
            isolated_flip |= npz[k]
        if k.startswith("mw") and k.endswith("_flip_refM8"):
            isolated_flip |= npz[k]
    flip_proof = margin > (2.0 * dmax_all)
    residual = ~flip_proof
    tie = margin <= 0
    # #87's teacher-forced argmax over ITS OWN greedy trajectory; used to verify the
    # two runs decoded the same token contexts before trusting a positional cross-tab.
    ref_argmax = npz["ref_argmax"] if "ref_argmax" in files else None
    return {
        "path": str(p),
        "margin": margin,
        "dmax_all": dmax_all,
        "flip_proof": flip_proof,
        "residual": residual,
        "tie": tie,
        "isolated_flip": isolated_flip,
        "ref_argmax": ref_argmax,
        "n_tie": int(tie.sum()),
        "n_residual": int(residual.sum()),
        "n_flip_proof": int(flip_proof.sum()),
        "n_isolated_flip": int(isolated_flip.sum()),
    }


def _bucket_flips(margin: np.ndarray, margin_native_ulp: np.ndarray,
                  flip_union: np.ndarray) -> dict[str, Any]:
    """Per-thinness-bucket position counts + how many of each flipped (capture's
    own self-consistent margin map: ctrl_top1 - ctrl_top2)."""
    buckets: dict[str, Any] = {}
    for t in (0, 1, 2, 4, 8, 16):
        sel = (margin <= 0) if t == 0 else (margin_native_ulp < t)
        label = "exact_tie" if t == 0 else f"below_{t}ulp"
        n = int(sel.sum())
        buckets[label] = {
            "n_positions": n,
            "n_flipped": int(flip_union[sel].sum()),
            "frac_of_bucket_flipped": float(flip_union[sel].mean()) if n else 0.0,
        }
    return buckets


def _crosstab(flip_union: np.ndarray, ws: dict[str, Any] | None,
              aligned: bool = True) -> dict[str, Any]:
    """Cross-tabulate compounding flips against #87's watch-set classes.

    `aligned` is the trajectory-alignment verdict: when False, #87 and this run
    decoded different greedy token sequences (decode-batch cascade), so a position
    i here is NOT the same token context as position i in #87's map. The raw counts
    are then positional coincidence, not signal, and the decisive flip-proof field
    is reported as None so it is never mistaken for genuine compounding.
    """
    if ws is None:
        return {"available": False,
                "note": "#87 margin_perturb.npz unavailable or shape-mismatched; "
                        "cross-tab uses the capture's own margin map only"}
    nflip = int(flip_union.sum())
    on_tie = int((flip_union & ws["tie"]).sum())
    on_residual = int((flip_union & ws["residual"]).sum())
    on_flip_proof = int((flip_union & ws["flip_proof"]).sum())
    out = {
        "available": True,
        "valid": bool(aligned),
        "ref_npz": ws["path"],
        "watchset_n_tie": ws["n_tie"],
        "watchset_n_residual": ws["n_residual"],
        "watchset_n_flip_proof": ws["n_flip_proof"],
        "watchset_n_isolated_flip_87": ws["n_isolated_flip"],
        "compounding_flips_total": nflip,
        "compounding_flips_on_87_exact_tie": on_tie,
        "compounding_flips_on_87_residual": on_residual,
        "compounding_flips_on_87_flip_proof": on_flip_proof,
        # the decisive signal: a flip on a position #87 PROVED safe in isolation is
        # genuine network-wide compounding, not the expected near-tie wobble — BUT
        # only if the two trajectories align; otherwise it is meaningless.
        "all_flips_within_87_residual": (bool(nflip > 0 and on_flip_proof == 0)
                                         if aligned else None),
        "genuine_compounding_flips_on_flip_proof": (on_flip_proof if aligned else None),
    }
    if not aligned:
        out["note"] = ("INVALID positional cross-tab: #87 trajectory misaligned with "
                       "this run (see trajectory_alignment). Counts are positional "
                       "coincidence; use this run's own margin_buckets instead.")
    return out


def _enumerate(flip_union: np.ndarray, margin: np.ndarray, margin_native_ulp: np.ndarray,
               ctrl_argmax: np.ndarray, pert_argmax: np.ndarray | None,
               dlogit_max: np.ndarray | None, ws: dict[str, Any] | None,
               output_len: int, cap: int) -> list[dict[str, Any]]:
    """List the flipped positions (thinnest margin first) for the AMBER set."""
    idx = np.where(flip_union)[0]
    if idx.size == 0:
        return []
    idx = idx[np.argsort(margin[idx])][:cap]
    out = []
    for i in idx:
        rec = {
            "pos": int(i),
            "prompt": int(i) // output_len,
            "tok": int(i) % output_len,
            "ctrl_argmax": int(ctrl_argmax[i]),
            "margin": float(margin[i]),
            "margin_native_ulp": float(margin_native_ulp[i]),
        }
        if pert_argmax is not None:
            rec["pert_argmax"] = int(pert_argmax[i])
        if dlogit_max is not None:
            rec["max_abs_dlogit"] = float(dlogit_max[i])
        if ws is not None:
            rec["in_87_exact_tie"] = bool(ws["tie"][i])
            rec["in_87_residual"] = bool(ws["residual"][i])
            rec["in_87_flip_proof"] = bool(ws["flip_proof"][i])
        out.append(rec)
    return out


def analyze(capture_dir: Path, ref_npz: str | Path) -> dict[str, Any]:
    summary = json.loads((capture_dir / "summary.json").read_text())
    npz = np.load(capture_dir / "compounding.npz")

    npos = int(summary["num_positions"])
    output_len = int(summary["output_len"])
    native_dtype = summary.get("native_dtype", "torch.bfloat16")
    nbits = _native_mant_bits(native_dtype)

    ctrl_argmax = npz["ctrl_argmax"]
    ctrl_top1 = npz["ctrl_top1"].astype(np.float64)
    ctrl_top2 = npz["ctrl_top2"].astype(np.float64)
    margin = ctrl_top1 - ctrl_top2
    ulp_native = _ulp(np.abs(ctrl_top1), nbits)
    margin_native_ulp = margin / ulp_native

    pos_mask = margin > 0
    n_ties = int((~pos_mask).sum())
    min_pos_margin = float(margin[pos_mask].min()) if pos_mask.any() else 0.0
    min_pos_margin_ulp = float(margin_native_ulp[pos_mask].min()) if pos_mask.any() else 0.0

    # --- STEP 1: determinism control ------------------------------------------
    control_divergent = list(summary.get("control_divergent_tokens", []))
    control_divergent_max = int(summary.get("control_divergent_max",
                                            max(control_divergent) if control_divergent else 0))
    control_deterministic = (control_divergent_max == 0)
    fidelity = int(summary.get("reconstruction_fidelity_disagreements", 0))

    # --- regimes from the capture --------------------------------------------
    # regime A is emitted with mode=="emu" (genuine SplitK emulation); regime B
    # with mode=="adversarial". A is keyed by SplitK width (tags realistic_s{2,4,8}).
    regimes = summary.get("regimes", [])
    realistic = [r for r in regimes if r["mode"] == "emu"]
    adversarial = [r for r in regimes if r["mode"] == "adversarial"]

    def _flip(tag: str) -> np.ndarray:
        return npz[f"{tag}_flip"]

    def _flip_up(tag: str) -> np.ndarray:
        return npz[f"{tag}_flip_upstream"]

    # full-frontier (lm_head re-tiled too) and upstream-only (native lm_head over
    # the propagated perturbed h) flip counts per SplitK width.
    realistic_per_split = {r["tag"]: int(_flip(r["tag"]).sum()) for r in realistic}
    realistic_up_per_split = {r["tag"]: int(_flip_up(r["tag"]).sum()) for r in realistic}
    realistic_union = np.zeros(npos, np.bool_)
    realistic_union_up = np.zeros(npos, np.bool_)
    for r in realistic:
        realistic_union |= _flip(r["tag"])
        realistic_union_up |= _flip_up(r["tag"])
    realistic_flip_count = max(realistic_per_split.values()) if realistic_per_split else 0
    realistic_flip_count_up = max(realistic_up_per_split.values()) if realistic_up_per_split else 0
    realistic_union_count = int(realistic_union.sum())

    adversarial_union = np.zeros(npos, np.bool_)
    adversarial_union_up = np.zeros(npos, np.bool_)
    for r in adversarial:
        adversarial_union |= _flip(r["tag"])
        adversarial_union_up |= _flip_up(r["tag"])
    adversarial_flip_count = int(adversarial_union.sum())
    adversarial_flip_count_up = int(adversarial_union_up.sum())

    # worst-case propagated drift across the realistic regimes (diagnostics)
    def _agg(items, key, fn=max):
        vals = [r[key] for r in items if key in r]
        return float(fn(vals)) if vals else 0.0

    max_dlogit_realistic = _agg(realistic, "max_abs_dlogit")
    max_dlogit_adversarial = _agg(adversarial, "max_abs_dlogit")
    max_dh_inf_realistic = _agg(realistic, "max_abs_dh_inf")
    max_dh_inf_adversarial = _agg(adversarial, "max_abs_dh_inf")
    mean_dh_l2_realistic = _agg(realistic, "mean_dh_l2", fn=lambda v: sum(v) / len(v))
    mean_dh_l2_adversarial = _agg(adversarial, "mean_dh_l2", fn=lambda v: sum(v) / len(v))
    headroom_ratio = (min_pos_margin / max_dlogit_realistic
                      if max_dlogit_realistic > 0 else float("inf"))

    # --- STEP 3: cross-tab + buckets + gate -----------------------------------
    ws = _load_87_watchset(ref_npz, npos)
    # Trajectory alignment gate for the #87 cross-tab. #87 decoded with
    # max_num_seqs=1, this run with max_num_seqs=32; that decode-batch reduction-
    # order difference cascades through the autoregressive loop and can move the
    # per-position token context. Compare this run's teacher-forced argmax against
    # #87's: if they disagree materially, position i is a different token here than
    # in #87, so the positional cross-tab is invalid and only this run's own
    # self-consistent margin map (margin_buckets below) is trustworthy.
    argmax_agreement_87: float | None = None
    traj_aligned = True
    if ws is not None and ws.get("ref_argmax") is not None:
        ref_am = np.asarray(ws["ref_argmax"])
        if ref_am.size == ctrl_argmax.size:
            argmax_agreement_87 = float((ctrl_argmax == ref_am).mean())
            traj_aligned = argmax_agreement_87 >= TRAJ_ALIGN_MIN
    crosstab_realistic = _crosstab(realistic_union, ws, aligned=traj_aligned)
    crosstab_adversarial = _crosstab(adversarial_union, ws, aligned=traj_aligned)
    buckets_realistic = _bucket_flips(margin, margin_native_ulp, realistic_union)
    buckets_adversarial = _bucket_flips(margin, margin_native_ulp, adversarial_union)

    # per-seed pert_argmax/dlogit only needed for enumeration; pick the first
    # realistic tag with the max flip count for representative pert argmax.
    rep_tag = None
    if realistic:
        rep_tag = max(realistic, key=lambda r: realistic_per_split[r["tag"]])["tag"]
    rep_pert = npz[f"{rep_tag}_pert_argmax"] if rep_tag else None
    rep_dlogit = npz[f"{rep_tag}_dlogit_max"] if rep_tag else None
    realistic_examples = _enumerate(realistic_union, margin, margin_native_ulp, ctrl_argmax,
                                    rep_pert, rep_dlogit, ws, output_len, AMBER_ENUMERATE_CAP)
    adv_tag = adversarial[0]["tag"] if adversarial else None
    adversarial_examples = _enumerate(
        adversarial_union, margin, margin_native_ulp, ctrl_argmax,
        npz[f"{adv_tag}_pert_argmax"] if adv_tag else None,
        npz[f"{adv_tag}_dlogit_max"] if adv_tag else None,
        ws, output_len, AMBER_ENUMERATE_CAP)

    frac_realistic = realistic_flip_count / npos if npos else 0.0
    red_threshold_positions = int(np.ceil(RED_FRAC * npos))

    if not control_deterministic:
        # baseline itself wobbles -> the perturbed comparison is unreliable.
        verdict = "RED" if frac_realistic >= RED_FRAC else "AMBER"
        verdict_reason = ("control non-deterministic in-process (Step 1 failed): "
                          f"max {control_divergent_max} divergent tokens run-to-run")
    elif realistic_flip_count == 0:
        verdict = "GREEN"
        verdict_reason = ("0 compounded argmax flips under realistic network-wide "
                          "≤1-ULP perturbation; deployed stack deterministic")
    elif frac_realistic >= RED_FRAC:
        verdict = "RED"
        verdict_reason = (f"{realistic_flip_count}/{npos} "
                          f"({100 * frac_realistic:.3f}%) ≥ {100 * RED_FRAC:.1f}% flip under realistic")
    else:
        verdict = "AMBER"
        loc = ("all within #87 thin-margin residual" if crosstab_realistic.get(
            "all_flips_within_87_residual") else "INCLUDES #87 flip-proof positions")
        verdict_reason = (f"small enumerable set: {realistic_flip_count}/{npos} flips "
                          f"(< {100 * RED_FRAC:.1f}%), {loc}")

    adversarial_clean = (adversarial_flip_count == 0)
    # the strongest statement: even the sign-aligned worst case never flips.
    comfortable = (verdict == "GREEN" and adversarial_clean and n_ties == 0)

    report = {
        "capture_dir": str(capture_dir),
        "capture_summary": summary,
        "step1_determinism_control": {
            "control_runs": int(summary.get("control_runs", 1)),
            "control_divergent_tokens": control_divergent,
            "control_divergent_max": control_divergent_max,
            "control_deterministic": control_deterministic,
            "reconstruction_fidelity_disagreements": fidelity,
            "note": ("in-process enforce_eager re-runs of the deployed stack with "
                     "perturbation OFF; complements #73's served N=10 result"),
        },
        "margin_map": {
            "num_positions": npos,
            "native_dtype": native_dtype,
            "native_mantissa_bits": nbits,
            "min_margin": float(margin.min()),
            "min_positive_margin": min_pos_margin,
            "min_positive_margin_in_native_ulp": min_pos_margin_ulp,
            "median_margin": float(np.median(margin)),
            "n_exact_ties_margin_zero": n_ties,
            "frac_exact_ties": float(n_ties / npos) if npos else 0.0,
            "max_abs_logit_softcapped": float(np.abs(np.concatenate([ctrl_top1, ctrl_top2])).max()),
        },
        "step2_realistic": {
            "flip_count_max_over_splits": realistic_flip_count,
            "flip_count_per_split": realistic_per_split,
            "flip_count_union_over_splits": realistic_union_count,
            "flip_count_upstream_max_over_splits": realistic_flip_count_up,
            "flip_count_upstream_per_split": realistic_up_per_split,
            "frac_flipped": frac_realistic,
            "max_abs_dlogit": max_dlogit_realistic,
            "max_abs_dh_inf": max_dh_inf_realistic,
            "mean_dh_l2": mean_dh_l2_realistic,
            "headroom_ratio_minposmargin_over_maxdlogit": headroom_ratio,
            "crosstab_vs_87_watchset": crosstab_realistic,
            "margin_buckets": buckets_realistic,
            "flip_examples": realistic_examples,
        },
        "step2_adversarial": {
            "flip_count": adversarial_flip_count,
            "flip_count_upstream": adversarial_flip_count_up,
            "adversarial_clean": adversarial_clean,
            "max_abs_dlogit": max_dlogit_adversarial,
            "max_abs_dh_inf": max_dh_inf_adversarial,
            "mean_dh_l2": mean_dh_l2_adversarial,
            "crosstab_vs_87_watchset": crosstab_adversarial,
            "margin_buckets": buckets_adversarial,
            "flip_examples": adversarial_examples,
        },
        "trajectory_alignment_vs_87": {
            "argmax_agreement": argmax_agreement_87,
            "aligned": bool(traj_aligned),
            "threshold": TRAJ_ALIGN_MIN,
            "note": ("this run self-decodes greedy targets with max_num_seqs=32; #87 "
                     "used max_num_seqs=1. Same prompts/seed, but the decode-batch "
                     "reduction-order difference cascades through the autoregressive "
                     "loop, so the two greedy trajectories — and their per-position "
                     "token contexts — diverge. When aligned=False the #87 positional "
                     "cross-tab is invalid; rely on this run's own margin_buckets. "
                     "(The divergence is itself an instance of the very phenomenon "
                     "#96 measures: reduction-order flips at near-ties, cascaded.)"),
        },
        "isolated_reference_87": {
            "note": ("PR #87 verified 0/65536 isolated lm_head argmax flips under "
                     "SplitK S∈{2,4,8} + M-widen M≤32; #96 closes the upstream "
                     "network-wide compounding residual it flagged"),
            "watchset_available": ws is not None,
            "n_tie": ws["n_tie"] if ws else None,
            "n_residual": ws["n_residual"] if ws else None,
            "n_flip_proof": ws["n_flip_proof"] if ws else None,
            "n_isolated_flip_87": ws["n_isolated_flip"] if ws else None,
        },
        "gate": {
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "comfortable_headroom": bool(comfortable),
            "control_deterministic": control_deterministic,
            "compounded_argmax_flip_count_realistic": realistic_flip_count,
            "compounded_argmax_flip_count_realistic_upstream": realistic_flip_count_up,
            "compounded_argmax_flip_count_adversarial": adversarial_flip_count,
            "compounded_argmax_flip_count_adversarial_upstream": adversarial_flip_count_up,
            "frac_flipped_realistic": frac_realistic,
            "red_threshold_frac": RED_FRAC,
            "red_threshold_positions": red_threshold_positions,
            "adversarial_clean": adversarial_clean,
            "all_realistic_flips_within_87_residual":
                crosstab_realistic.get("all_flips_within_87_residual", None),
            "genuine_compounding_flips_on_flip_proof":
                crosstab_realistic.get("genuine_compounding_flips_on_flip_proof", None),
        },
        "primary_metric": {"name": "compounded_argmax_flip_count_realistic",
                           "value": realistic_flip_count},
        "test_metric": {"name": "compounded_argmax_flip_count_adversarial",
                        "value": adversarial_flip_count},
    }
    return report


def log_to_wandb(report: dict[str, Any], *, wandb_name: str, wandb_group: str) -> None:
    g = report["gate"]
    mm = report["margin_map"]
    s1 = report["step1_determinism_control"]
    sa = report["step2_realistic"]
    sb = report["step2_adversarial"]
    summary = {
        "verdict": g["verdict"],
        "verdict_green": int(g["verdict"] == "GREEN"),
        "verdict_red": int(g["verdict"] == "RED"),
        "comfortable_headroom": int(g["comfortable_headroom"]),
        "control_deterministic": int(g["control_deterministic"]),
        "control_divergent_max": s1["control_divergent_max"],
        "reconstruction_fidelity_disagreements": s1["reconstruction_fidelity_disagreements"],
        "compounded_argmax_flip_count_realistic": g["compounded_argmax_flip_count_realistic"],
        "compounded_argmax_flip_count_realistic_upstream":
            g["compounded_argmax_flip_count_realistic_upstream"],
        "compounded_argmax_flip_count_adversarial": g["compounded_argmax_flip_count_adversarial"],
        "compounded_argmax_flip_count_adversarial_upstream":
            g["compounded_argmax_flip_count_adversarial_upstream"],
        "frac_flipped_realistic": g["frac_flipped_realistic"],
        "adversarial_clean": int(g["adversarial_clean"]),
        "realistic_flip_union": sa["flip_count_union_over_splits"],
        "realistic_max_abs_dlogit": sa["max_abs_dlogit"],
        "realistic_max_abs_dh_inf": sa["max_abs_dh_inf"],
        "realistic_mean_dh_l2": sa["mean_dh_l2"],
        "headroom_ratio": sa["headroom_ratio_minposmargin_over_maxdlogit"],
        "adversarial_max_abs_dlogit": sb["max_abs_dlogit"],
        "adversarial_max_abs_dh_inf": sb["max_abs_dh_inf"],
        "adversarial_mean_dh_l2": sb["mean_dh_l2"],
        "min_margin": mm["min_margin"],
        "min_positive_margin": mm["min_positive_margin"],
        "median_margin": mm["median_margin"],
        "n_exact_ties_margin_zero": mm["n_exact_ties_margin_zero"],
        "frac_exact_ties": mm["frac_exact_ties"],
        "num_positions": mm["num_positions"],
        "native_dtype": mm["native_dtype"],
    }
    ta = report["trajectory_alignment_vs_87"]
    summary["argmax_agreement_vs_87"] = (
        ta["argmax_agreement"] if ta["argmax_agreement"] is not None else -1.0)
    summary["trajectory_aligned_vs_87"] = int(ta["aligned"])
    ct = sa["crosstab_vs_87_watchset"]
    if ct.get("available"):
        summary["crosstab_vs_87_valid"] = int(ct.get("valid", True))
        # only surface the #87 positional cross-tab when the trajectories align;
        # otherwise the counts are coincidence and would mislead dashboards.
        if ct.get("valid", True):
            summary["realistic_flips_on_87_exact_tie"] = ct["compounding_flips_on_87_exact_tie"]
            summary["realistic_flips_on_87_residual"] = ct["compounding_flips_on_87_residual"]
            summary["realistic_flips_on_87_flip_proof"] = ct["compounding_flips_on_87_flip_proof"]
        summary["watchset_n_tie"] = ct["watchset_n_tie"]
        summary["watchset_n_residual"] = ct["watchset_n_residual"]
    # self-consistent margin buckets (the valid view) — log flips per thinness band.
    for label, b in sa["margin_buckets"].items():
        summary[f"bucket.{label}.positions"] = b["n_positions"]
        summary[f"bucket.{label}.flipped"] = b["n_flipped"]
    for tag, c in sa["flip_count_per_split"].items():
        summary[f"realistic.{tag}.flips"] = c
    for tag, c in sa["flip_count_upstream_per_split"].items():
        summary[f"realistic.{tag}.flips_upstream"] = c
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_json_artifact, log_summary)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] unavailable: {exc}", flush=True)
        return
    run = init_wandb_run(
        job_type="greedy-compounding-gate", agent="senpai", name=wandb_name,
        tags=["greedy-compounding-gate", wandb_group],
        group=wandb_group,
        config={"capture_dir": report["capture_dir"],
                "num_positions": mm["num_positions"],
                "scale": report["capture_summary"].get("scale", 1.0),
                "include_ple": report["capture_summary"].get("include_ple", False),
                "wandb_group": wandb_group},
    )
    if run is None:
        print("[wandb] run not created (no creds/disabled); report.json is the record", flush=True)
        return
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="greedy_compounding_report",
                      artifact_type="greedy-compounding-gate-report", data=report)
    finish_wandb(run)
    print(f"[wandb] logged run {wandb_name} (group={wandb_group})", flush=True)


def _print(report: dict[str, Any]) -> None:
    g = report["gate"]
    mm = report["margin_map"]
    s1 = report["step1_determinism_control"]
    sa = report["step2_realistic"]
    sb = report["step2_adversarial"]
    ref87 = report["isolated_reference_87"]
    print("\n" + "=" * 74, flush=True)
    print("NETWORK-WIDE GREEDY-COMPOUNDING GATE (PR #96)", flush=True)
    print("=" * 74, flush=True)
    print(f"positions={mm['num_positions']}  native_dtype={mm['native_dtype']} "
          f"(mantissa {mm['native_mantissa_bits']} bits)  softcap=30", flush=True)
    print(f"hooks={report['capture_summary'].get('n_gemm_hooks')} GEMM ops  "
          f"suffixes={report['capture_summary'].get('perturb_suffixes')}", flush=True)

    print("\nSTEP 1 — in-process determinism control (perturbation OFF):", flush=True)
    print(f"  control runs        : {s1['control_runs']}", flush=True)
    print(f"  divergent run-to-run: {s1['control_divergent_tokens']} "
          f"(max {s1['control_divergent_max']})  -> deterministic={s1['control_deterministic']}",
          flush=True)
    print(f"  reconstruction fid. : {s1['reconstruction_fidelity_disagreements']} "
          f"argmax_ctrl vs fed greedy token", flush=True)

    print("\nMARGIN MAP (control top-2, post-softcap):", flush=True)
    print(f"  min Δ={mm['min_margin']:.6g}  min positive Δ={mm['min_positive_margin']:.6g} "
          f"({mm['min_positive_margin_in_native_ulp']:.3g} native-ULP)  "
          f"median Δ={mm['median_margin']:.6g}", flush=True)
    print(f"  exact ties(Δ≤0)={mm['n_exact_ties_margin_zero']}  "
          f"max|logit|={mm['max_abs_logit_softcapped']:.4g}", flush=True)

    print("\nSTEP 2A — REALISTIC genuine SplitK reduction-order emu (primary):", flush=True)
    print(f"  full-frontier flips/split={sa['flip_count_per_split']}  "
          f"max={sa['flip_count_max_over_splits']}  union={sa['flip_count_union_over_splits']}  "
          f"({100 * sa['frac_flipped']:.4f}%)", flush=True)
    print(f"  upstream-only flips/split={sa['flip_count_upstream_per_split']}  "
          f"max={sa['flip_count_upstream_max_over_splits']}  "
          f"(native lm_head over propagated h — the pure network-wide-h signal)", flush=True)
    print(f"  max|Δlogit|={sa['max_abs_dlogit']:.4g}  max|Δh|inf={sa['max_abs_dh_inf']:.4g}  "
          f"mean‖Δh‖₂={sa['mean_dh_l2']:.4g}  headroom(minposΔ/maxΔlogit)="
          f"{sa['headroom_ratio_minposmargin_over_maxdlogit']:.4g}", flush=True)

    print("\nSTEP 2B — ADVERSARIAL sign-aligned ±ULP + targeted lm_head (test):", flush=True)
    print(f"  full-frontier flips={sb['flip_count']}  upstream-only={sb['flip_count_upstream']}  "
          f"clean={sb['adversarial_clean']}  "
          f"max|Δlogit|={sb['max_abs_dlogit']:.4g}  max|Δh|inf={sb['max_abs_dh_inf']:.4g}  "
          f"mean‖Δh‖₂={sb['mean_dh_l2']:.4g}", flush=True)

    ta = report["trajectory_alignment_vs_87"]
    print("\nTRAJECTORY ALIGNMENT vs #87 (decode-batch cascade check):", flush=True)
    agree = ta["argmax_agreement"]
    print(f"  argmax agreement={'n/a' if agree is None else f'{100 * agree:.2f}%'}  "
          f"threshold={100 * ta['threshold']:.0f}%  -> aligned={ta['aligned']}", flush=True)

    ct = sa["crosstab_vs_87_watchset"]
    print("\nCROSS-TAB vs #87 thin-margin watch-set:", flush=True)
    if ct.get("available"):
        if not ct.get("valid", True):
            print("  *** INVALID: #87 trajectory misaligned — positional cross-tab is "
                  "coincidence, not signal. Use this run's margin_buckets. ***", flush=True)
        print(f"  watch-set: {ct['watchset_n_tie']} exact ties, {ct['watchset_n_residual']} "
              f"measurement-dependent, {ct['watchset_n_flip_proof']} flip-proof "
              f"(#87 isolated flips={ct['watchset_n_isolated_flip_87']})", flush=True)
        print(f"  realistic flips on: exact-tie={ct['compounding_flips_on_87_exact_tie']}  "
              f"residual={ct['compounding_flips_on_87_residual']}  "
              f"FLIP-PROOF={ct['compounding_flips_on_87_flip_proof']}"
              f"{'  (coincidental — misaligned)' if not ct.get('valid', True) else ''}",
              flush=True)
    else:
        print(f"  {ct.get('note')}", flush=True)

    print("\nSELF-CONSISTENT margin buckets (this run's own top-2 map — the valid view):",
          flush=True)
    for label, b in sa["margin_buckets"].items():
        print(f"    {label:<12} positions={b['n_positions']:<6} flipped={b['n_flipped']:<5} "
              f"({100 * b['frac_of_bucket_flipped']:.1f}% of bucket)", flush=True)
    if ref87["watchset_available"]:
        print(f"  isolated #87 reference: ties={ref87['n_tie']} residual={ref87['n_residual']} "
              f"flip_proof={ref87['n_flip_proof']} isolated_flips={ref87['n_isolated_flip_87']}",
              flush=True)

    if sa["flip_examples"]:
        print(f"\n  realistic flip set (thinnest first, ≤{AMBER_ENUMERATE_CAP}):", flush=True)
        for ex in sa["flip_examples"][:16]:
            tag = ""
            if "in_87_flip_proof" in ex and ex["in_87_flip_proof"]:
                tag = "  [#87 FLIP-PROOF!]"
            elif "in_87_exact_tie" in ex and ex["in_87_exact_tie"]:
                tag = "  [#87 tie]"
            print(f"    p{ex['prompt']}/t{ex['tok']} pos={ex['pos']} ctrl={ex['ctrl_argmax']}"
                  f"->pert={ex.get('pert_argmax')} Δ={ex['margin']:.4g} "
                  f"({ex['margin_native_ulp']:.3g} ULP){tag}", flush=True)

    print("\n" + "-" * 74, flush=True)
    note = ""
    if g["comfortable_headroom"]:
        note = "  [comfortable: even adversarial 0 flips, no ties]"
    elif g["verdict"] == "GREEN":
        note = f"  [{mm['n_exact_ties_margin_zero']} ties bit-preserved; adversarial clean={g['adversarial_clean']}]"
    print(f"VERDICT: {g['verdict']}  — {g['verdict_reason']}{note}", flush=True)
    print(f"  primary  compounded_argmax_flip_count_realistic  = "
          f"{report['primary_metric']['value']}", flush=True)
    print(f"  test     compounded_argmax_flip_count_adversarial = "
          f"{report['test_metric']['value']}", flush=True)
    print("=" * 74 + "\n", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-dir", default=None,
                    help="timestamped capture dir (default: latest under --out-root)")
    ap.add_argument("--out-root", default=str(REPO / "research/validity/greedy_compounding"))
    ap.add_argument("--ref-npz", default=str(DEFAULT_REF_NPZ),
                    help="#87 margin_perturb.npz for the thin-margin watch-set cross-tab")
    ap.add_argument("--report", default=None, help="where to write report.json")
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default=WANDB_GROUP)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    capture_dir = Path(args.capture_dir) if args.capture_dir else _latest_capture(Path(args.out_root))
    report = analyze(capture_dir, args.ref_npz)
    report_path = Path(args.report) if args.report else capture_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    _print(report)
    print(f"[analyze] wrote {report_path}", flush=True)

    if not args.no_wandb:
        wandb_name = args.wandb_name or f"kanna/greedy-compounding-gate-{capture_dir.name}"
        log_to_wandb(report, wandb_name=wandb_name, wandb_group=args.wandb_group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
