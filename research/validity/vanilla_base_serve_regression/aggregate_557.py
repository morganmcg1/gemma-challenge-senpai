#!/usr/bin/env python3
"""PR #557 — aggregate the vanilla-base serve-regression root-cause + recovered denominator.

ROOT CAUSE (source-level, confirmed empirically by the ple_fold arm): dev307's
vllm/model_executor/models/gemma4.py::get_per_layer_inputs DROPS the Gemma4 Per-Layer-
Embedding embed_scale (sqrt(256)=16.0) at runtime — the "Challenge fast path" removed the
multiply and folds it at load time ONLY when env PLE_FOLD_EMBED_SCALE=1 is set
(model_loader/utils.py:130). A plain vanilla serve never sets it, so every decoder layer
gets 16x-too-small per-layer embeddings -> subtly corrupted long-CoT decode -> repetition
loops -> max_tokens truncation -> MMLU 0.432 / GPQA 0.3131. NOT an attention-backend issue.

Arms (only the config moves; ckpt/dtype/seqs/greedy harness byte-identical to #542):
  triton_default (#542 banked) : plain vanilla, no fold -> BROKEN (the regression).
  surgical_attn (Stage-3 mech) : forced-TRITON + 2D order-preserving reduction
      (SURGICAL_ATTN_USE_3D_OFF=1), NO MTP, NO fold. Tests "does attention alone fix it" ->
      expected STILL broken (the 2D path does not touch the PLE scale).
  ple_fold (Stage-1/2 RECOVERY) : plain vanilla serve (speculative_config=None, default
      TRITON, default 3D reduction) + ONLY PLE_FOLD_EMBED_SCALE=1 -> the literal missing-scale
      fix -> the recovered healthy denominator.
  global_fa (Stage-1 control)  : VLLM_ATTENTION_BACKEND=FLASH_ATTN -> FA rejects the 512-dim
      full layers -> fails at init (a stock global override is NOT the fix). Load-fail marker.

Verdicts bind to the documented ubel #511 anchor (MMLU 0.668 / GPQA 0.470) + Morgan #515
floors (0.601 / 0.423). The strict CI-lb test is re-run against the LIVE recovered fresh
base (base_fullhead CI-lb >= 0.90 x recovered point) — the test #542 could only run vs the
noise-limited n=198 anchor.

Usage:  aggregate_557.py [--no-wandb]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
HERE = ROOT / "research/validity/vanilla_base_serve_regression"
F542 = ROOT / "research/validity/base_fullhead_shortchain_quality"
sys.path.insert(0, str(F542))
import failmodes  # noqa: E402  (classify_eval re-reads .eval logs for truncation)

ANCHOR_MMLU = 0.668
ANCHOR_MMLU_K, ANCHOR_MMLU_N = 334, 500
ANCHOR_GPQA = 0.470
ANCHOR_GPQA_MEAS = 0.4444
ANCHOR_GPQA_K, ANCHOR_GPQA_N = 88, 198
FLOOR_MMLU = 0.601
FLOOR_GPQA = 0.423
BUILD = "vllm-0.22.1rc1.dev307+g3e8afdf78"
TRUNC_COLLAPSE = 0.20  # broken base sits at 33.8/36.6%; recovered must be well below


def wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def two_prop_z(k1, n1, k2, n2):
    if not n1 or not n2:
        return float("nan")
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return (p1 - p2) / se if se else float("nan")


def _load(path: Path):
    if not path.exists():
        return None
    try:
        return json.load(open(path))
    except (json.JSONDecodeError, ValueError):
        return None


def prompt_identical(a, b):
    if a is None or b is None:
        return (False, -1)
    am = {r["id"]: r.get("prompt_sha") for r in a["per_sample"]}
    bm = {r["id"]: r.get("prompt_sha") for r in b["per_sample"]}
    common = set(am) & set(bm)
    mism = [i for i in common if am[i] != bm[i]]
    return (len(mism) == 0, len(mism))


def _trunc(json_path: Path):
    try:
        c = failmodes.classify_eval(json_path)
        return {"truncation_rate": c["truncation_rate"], "n_truncation": c["n_truncation"],
                "empty_eos_rate": c["empty_eos_rate"], "n_empty_eos": c["n_empty_eos"],
                "extract_fail_rate": c["extract_fail_rate"], "n_error": c["n_error"]}
    except SystemExit as exc:
        print(f"[aggregate-557] failmodes skipped for {json_path.name}: {exc}", file=sys.stderr)
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[aggregate-557] failmodes error for {json_path.name}: {exc!r}", file=sys.stderr)
        return None


def _pt(d):
    return (d["n_correct"] / d["n_scored"]) if d and d["n_scored"] else None


def _arm(json_path: Path, d, floor, anchor_pt):
    """Generic per-arm summary: acc, Wilson, truncation, floor/anchor relation, recovered?"""
    k, n = d["n_correct"], d["n_scored"]
    p, lo, hi = wilson(k, n)
    fm = _trunc(json_path) or {}
    tr = fm.get("truncation_rate")
    point_meets_floor = bool(p >= floor)
    cilb_meets_floor = bool(lo >= floor)
    trunc_collapsed = bool(tr is not None and tr < TRUNC_COLLAPSE)
    recovered = bool(trunc_collapsed and point_meets_floor)
    return {
        "acc": p, "n": n, "correct": k, "wilson_lo": lo, "wilson_hi": hi,
        "pct_of_anchor": (p / anchor_pt) if anchor_pt else float("nan"),
        "truncation_rate": tr, "n_truncation": fm.get("n_truncation"),
        "empty_eos_rate": fm.get("empty_eos_rate"), "extract_fail_rate": fm.get("extract_fail_rate"),
        "n_error": fm.get("n_error"),
        "point_meets_floor": point_meets_floor, "cilb_meets_floor": cilb_meets_floor,
        "trunc_collapsed": trunc_collapsed, "recovered": recovered,
    }


def _axis(name, rec_json, rec, brk_json, brk, fh, anchor_k, anchor_n, anchor_pt, floor):
    """Recovered (ple_fold) vs broken (triton_default) vs anchor + live strict CI-lb."""
    a = _arm(rec_json, rec, floor, anchor_pt)
    rk, rn = rec["n_correct"], rec["n_scored"]
    ancp, anclo, anchi = wilson(anchor_k, anchor_n)
    brk_fm = _trunc(brk_json) if brk_json else None
    brk_trunc = (brk_fm or {}).get("truncation_rate")
    z_vs_anchor = two_prop_z(rk, rn, anchor_k, anchor_n)
    within_anchor_ci = bool(abs(z_vs_anchor) < 1.96)

    strict = None
    if fh is not None:
        fhk, fhn = fh["n_correct"], fh["n_scored"]
        fhp, fhlo, fhhi = wilson(fhk, fhn)
        gate = 0.90 * a["acc"]
        strict = {"fullhead_acc": fhp, "fullhead_n": fhn, "fullhead_wilson_lo": fhlo,
                  "fullhead_wilson_hi": fhhi, "live_floor_0p90x_recovered": gate,
                  "fullhead_cilb_clears_live_floor": bool(fhlo >= gate)}

    pid_brk, mis_brk = prompt_identical(rec, brk)
    pid_fh, mis_fh = prompt_identical(rec, fh)
    return {
        "axis": name, "anchor_pt": anchor_pt, "anchor_meas_acc": ancp,
        "anchor_meas_wilson_lo": anclo, "anchor_meas_wilson_hi": anchi,
        "anchor_k": anchor_k, "anchor_n": anchor_n, "floor": floor,
        "recovered_acc": a["acc"], "recovered_n": a["n"], "recovered_correct": a["correct"],
        "recovered_wilson_lo": a["wilson_lo"], "recovered_wilson_hi": a["wilson_hi"],
        "recovered_pct_of_anchor": a["pct_of_anchor"],
        "recovered_truncation_rate": a["truncation_rate"],
        "recovered_extract_fail_rate": a["extract_fail_rate"], "recovered_n_error": a["n_error"],
        "broken_acc": _pt(brk), "broken_n": brk["n_scored"] if brk else None,
        "broken_truncation_rate": brk_trunc,
        "z_vs_anchor": z_vs_anchor, "within_anchor_ci": within_anchor_ci,
        "point_meets_floor": a["point_meets_floor"], "cilb_meets_floor": a["cilb_meets_floor"],
        "trunc_collapsed_vs_broken": a["trunc_collapsed"], "axis_recovered": a["recovered"],
        "strict_cilb_live": strict,
        "prompt_identical_to_broken": pid_brk, "n_prompt_mismatch_broken": mis_brk,
        "prompt_identical_to_fullhead": pid_fh, "n_prompt_mismatch_fullhead": mis_fh,
    }


def _pct(x):
    return "NA" if x is None else f"{x*100:.1f}%"


def _runid():
    p = HERE / "wandb_run_id.txt"
    return p.read_text().strip() if p.exists() else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-name", default="stark/vanilla-base-serve-regression")
    ap.add_argument("--wandb-group", default="vanilla-base-serve-regression")
    a = ap.parse_args()

    # recovered arm = ple_fold (plain vanilla + PLE embed-scale fold)
    rec_mp, rec_gp = HERE / "ple_fold_mmlu_pro.json", HERE / "ple_fold_gpqa_diamond.json"
    rec_m, rec_g = _load(rec_mp), _load(rec_gp)
    # #542 banked broken triton_default + base_fullhead (byte-identical prompts)
    brk_mp, brk_gp = F542 / "base_mmlu_pro.json", F542 / "base_gpqa.json"
    brk_m, brk_g = _load(brk_mp), _load(brk_gp)
    fh_m, fh_g = _load(F542 / "fullhead_mmlu_pro.json"), _load(F542 / "fullhead_gpqa.json")
    # Stage-3 mechanism arm = surgical_attn (2D attn only, no fold)
    surg_mp, surg_gp = HERE / "surgical_attn_mmlu_pro.json", HERE / "surgical_attn_gpqa_diamond.json"
    surg_m, surg_g = _load(surg_mp), _load(surg_gp)
    # side probes
    gfa = _load(HERE / "global_fa_loadtest.json")
    sd = _load(HERE / "selfdet.json")

    if rec_m is None or rec_g is None:
        print("[aggregate-557] MISSING ple_fold (recovered) arms — run not complete", file=sys.stderr)
        return 2

    mmlu = _axis("mmlu_pro", rec_mp, rec_m, (brk_mp if brk_m else None), brk_m, fh_m,
                 ANCHOR_MMLU_K, ANCHOR_MMLU_N, ANCHOR_MMLU, FLOOR_MMLU)
    gpqa = _axis("gpqa_diamond", rec_gp, rec_g, (brk_gp if brk_g else None), brk_g, fh_g,
                 ANCHOR_GPQA_K, ANCHOR_GPQA_N, ANCHOR_GPQA, FLOOR_GPQA)

    recovered_mmlu, recovered_gpqa = mmlu["recovered_acc"], gpqa["recovered_acc"]

    # ---- Stage-3 mechanism arm (surgical_attn = 2D attn only, no fold) ----
    surg = {}
    if surg_m is not None:
        surg["mmlu"] = _arm(surg_mp, surg_m, FLOOR_MMLU, ANCHOR_MMLU)
    if surg_g is not None:
        surg["gpqa"] = _arm(surg_gp, surg_g, FLOOR_GPQA, ANCHOR_GPQA)
    surgical_attn_only_mmlu = surg.get("mmlu", {}).get("acc")
    surgical_attn_only_gpqa = surg.get("gpqa", {}).get("acc")
    surgical_attn_alone_fixes_longcot = bool(
        surg.get("mmlu", {}).get("recovered") and surg.get("gpqa", {}).get("recovered")
    ) if ("mmlu" in surg and "gpqa" in surg) else None

    # ---- Stage-1/2 top-line verdicts ----
    healthy_fresh_base_recoverable = bool(mmlu["axis_recovered"] and gpqa["axis_recovered"])
    recovered_base_reproduces_anchor = bool(mmlu["within_anchor_ci"] and gpqa["within_anchor_ci"])
    sm, sg = mmlu["strict_cilb_live"], gpqa["strict_cilb_live"]
    strict_cilb_passes_vs_healthy_fresh_base = bool(
        sm and sg and sm["fullhead_cilb_clears_live_floor"] and sg["fullhead_cilb_clears_live_floor"]
    )
    global_fa_load_failed = bool(gfa.get("load_failed")) if gfa else None
    regression_root_caused = healthy_fresh_base_recoverable

    surg_note = ""
    if surgical_attn_alone_fixes_longcot is False:
        surg_note = (f" The 2D order-preserving attention path alone (surgical_attn, no fold) does "
                     f"NOT fix it ({_fmt(surgical_attn_only_mmlu)}/{_fmt(surgical_attn_only_gpqa)}, "
                     f"trunc {_pct(surg.get('mmlu',{}).get('truncation_rate'))}/"
                     f"{_pct(surg.get('gpqa',{}).get('truncation_rate'))}) — the attention backend is "
                     f"NOT the cause.")
    # global_fa control: setting VLLM_ATTENTION_BACKEND=FLASH_ATTN. Two honest outcomes:
    #   load_failed=True  -> FA rejects the head_dim=512 full layers (can't globally swap attn).
    #   load_failed=False -> dev307's config.py:100 force-overrides the request back to TRITON_ATTN,
    #                        so the serve comes up on TRITON anyway. Either way attention is NOT a
    #                        usable lever and is held CONSTANT (TRITON) across every arm.
    if global_fa_load_failed is None:
        gfa_note = ""
    elif global_fa_load_failed:
        gfa_note = (" A stock global FLASH_ATTN override (VLLM_ATTENTION_BACKEND=FLASH_ATTN) fails at "
                    "init — FlashAttention rejects the head_dim=512 full layers — so the attention "
                    "backend cannot be globally swapped to fix it.")
    else:
        gfa_note = (" A stock global FLASH_ATTN override (VLLM_ATTENTION_BACKEND=FLASH_ATTN) does NOT "
                    "take effect: dev307's config.py:100 force-selects TRITON_ATTN for the heterogeneous "
                    "head dims regardless (server comes up on TRITON, load_failed=False). Attention is "
                    "thus held CONSTANT (forced TRITON_ATTN) across the broken base, surgical_attn, and "
                    "ple_fold arms — the PLE embed-scale fold is the sole differentiator.")
    regression_cause = (
        "vLLM dev307 (vllm-0.22.1rc1.dev307+g3e8afdf78) drops the Gemma4 Per-Layer-Embedding "
        "embed_scale (sqrt(256)=16.0) on the plain serve path: gemma4.py::get_per_layer_inputs no "
        "longer multiplies the per-layer embeddings by embed_scale_per_layer at runtime (the "
        "'Challenge fast path'), and the scale is instead applied by a LOAD-TIME fold gated behind "
        "env PLE_FOLD_EMBED_SCALE=1 (model_loader/utils.py:130). A vanilla serve never sets that env "
        "var, so every decoder layer receives 16x-too-small per-layer embeddings -> subtly corrupted "
        f"long-CoT greedy decode -> repetition loops -> max_tokens truncation "
        f"({_pct(mmlu['broken_truncation_rate'])}/{_pct(gpqa['broken_truncation_rate'])} MMLU/GPQA) -> "
        "0.432/0.3131. Setting PLE_FOLD_EMBED_SCALE=1 on the SAME plain vanilla serve (no MTP, no 2D "
        "attn, no split-KV, no onegraph) folds the scale into the embedding weights ('Folded Gemma4 "
        f"PLE embed scale 16.0 into weight') and recovers {recovered_mmlu:.3f}/{recovered_gpqa:.3f} with "
        f"truncation collapsed to {_pct(mmlu['recovered_truncation_rate'])}/"
        f"{_pct(gpqa['recovered_truncation_rate'])}." + surg_note + gfa_note
    ) if regression_root_caused else (
        "INCONCLUSIVE: PLE_FOLD_EMBED_SCALE=1 did not recover the base on this ckpt; see per-axis."
    )

    self_det = sd.get("self_det") if sd else None

    marker = {
        "pr": 557, "analysis_only": True, "official_tps": 0, "no_hf_job": True,
        "vllm_build": BUILD,
        "checkpoint": "google/gemma-4-E4B-it-qat-w4a16-ct (stock int4, native 262k head)",
        # Stage 1
        "regression_root_caused": regression_root_caused,
        "regression_cause": regression_cause,
        "regression_cause_class": "ple_embed_scale_fold_gated_off_on_vanilla_serve",
        "global_fa_load_failed": global_fa_load_failed,
        # Stage 2
        "recovered_vanilla_base_mmlu": recovered_mmlu,
        "recovered_vanilla_base_gpqa": recovered_gpqa,
        "recovered_base_reproduces_anchor": recovered_base_reproduces_anchor,
        "strict_cilb_passes_vs_healthy_fresh_base": strict_cilb_passes_vs_healthy_fresh_base,
        "healthy_fresh_base_recoverable": healthy_fresh_base_recoverable,
        # Stage 3
        "surgical_attn_only_mmlu": surgical_attn_only_mmlu,
        "surgical_attn_only_gpqa": surgical_attn_only_gpqa,
        "surgical_attn_alone_fixes_longcot": surgical_attn_alone_fixes_longcot,
        # anchors / mechanism
        "anchor_mmlu": ANCHOR_MMLU, "anchor_gpqa": ANCHOR_GPQA,
        "floor_mmlu": FLOOR_MMLU, "floor_gpqa": FLOOR_GPQA,
        "mmlu_recovered_pct_of_anchor": mmlu["recovered_pct_of_anchor"],
        "gpqa_recovered_pct_of_anchor": gpqa["recovered_pct_of_anchor"],
        "mmlu_z_vs_anchor": mmlu["z_vs_anchor"], "gpqa_z_vs_anchor": gpqa["z_vs_anchor"],
        "mmlu_within_anchor_ci": mmlu["within_anchor_ci"], "gpqa_within_anchor_ci": gpqa["within_anchor_ci"],
        "mmlu_recovered_truncation_rate": mmlu["recovered_truncation_rate"],
        "gpqa_recovered_truncation_rate": gpqa["recovered_truncation_rate"],
        "mmlu_broken_truncation_rate": mmlu["broken_truncation_rate"],
        "gpqa_broken_truncation_rate": gpqa["broken_truncation_rate"],
        "mmlu_recovered_wilson_ci": [mmlu["recovered_wilson_lo"], mmlu["recovered_wilson_hi"]],
        "gpqa_recovered_wilson_ci": [gpqa["recovered_wilson_lo"], gpqa["recovered_wilson_hi"]],
        "mmlu_point_meets_floor": mmlu["point_meets_floor"], "gpqa_point_meets_floor": gpqa["point_meets_floor"],
        "mmlu_cilb_meets_floor": mmlu["cilb_meets_floor"], "gpqa_cilb_meets_floor": gpqa["cilb_meets_floor"],
        "surgical_attn_mmlu_truncation_rate": surg.get("mmlu", {}).get("truncation_rate"),
        "surgical_attn_gpqa_truncation_rate": surg.get("gpqa", {}).get("truncation_rate"),
        "mmlu_n_prompt_mismatch_broken": mmlu["n_prompt_mismatch_broken"],
        "gpqa_n_prompt_mismatch_broken": gpqa["n_prompt_mismatch_broken"],
        "mmlu_n_prompt_mismatch_fullhead": mmlu["n_prompt_mismatch_fullhead"],
        "gpqa_n_prompt_mismatch_fullhead": gpqa["n_prompt_mismatch_fullhead"],
        "self_det": self_det,
    }
    report = {
        "pr": 557, "vllm_build": BUILD,
        "checkpoint": "google/gemma-4-E4B-it-qat-w4a16-ct (stock int4, native 262k head)",
        "mmlu": mmlu, "gpqa": gpqa, "surgical_attn": surg, "marker": marker,
        "global_fa": gfa, "selfdet": sd,
        "anchors": {"mmlu": ANCHOR_MMLU, "gpqa": ANCHOR_GPQA, "gpqa_measured": ANCHOR_GPQA_MEAS},
        "floors": {"mmlu": FLOOR_MMLU, "gpqa": FLOOR_GPQA},
        "analysis_only": True, "no_hf_job": True, "official_tps": 0,
    }
    (HERE / "aggregate_557.json").write_text(json.dumps(report, indent=2))

    def fmt(ax):
        ba = "NA" if ax["broken_acc"] is None else f"{ax['broken_acc']:.4f}"
        strict_line = ""
        if ax["strict_cilb_live"]:
            s = ax["strict_cilb_live"]
            strict_line = (f"\n      strict CI-lb (live): fullhead_lo={s['fullhead_wilson_lo']:.3f} "
                           f">= 0.90x recovered({0.90*ax['recovered_acc']:.3f}) -> "
                           f"{s['fullhead_cilb_clears_live_floor']}")
        return (
            f"  {ax['axis']:13s} floor={ax['floor']:.3f} anchor={ax['anchor_pt']:.3f}\n"
            f"    recovered(ple_fold) = {ax['recovered_acc']:.4f} "
            f"(Wilson {ax['recovered_wilson_lo']:.3f}-{ax['recovered_wilson_hi']:.3f}, "
            f"n={ax['recovered_n']}, {ax['recovered_pct_of_anchor']*100:.1f}% of anchor, "
            f"trunc={_pct(ax['recovered_truncation_rate'])})\n"
            f"    broken(triton_default) = {ba} "
            f"(trunc={_pct(ax['broken_truncation_rate'])})\n"
            f"      z_vs_anchor={ax['z_vs_anchor']:+.2f} within_ci:{ax['within_anchor_ci']} | "
            f"point>=floor:{ax['point_meets_floor']} cilb>=floor:{ax['cilb_meets_floor']} | "
            f"trunc_collapsed:{ax['trunc_collapsed_vs_broken']} recovered:{ax['axis_recovered']}\n"
            f"      prompts==broken:{ax['prompt_identical_to_broken']}(mis={ax['n_prompt_mismatch_broken']}) "
            f"==fullhead:{ax['prompt_identical_to_fullhead']}(mis={ax['n_prompt_mismatch_fullhead']})"
            + strict_line
        )

    print("\n==== PR #557 VANILLA-BASE SERVE REGRESSION (build " + BUILD + ") ====")
    print(fmt(mmlu)); print(fmt(gpqa))
    if surg:
        for ax in ("mmlu", "gpqa"):
            s = surg.get(ax)
            if s:
                print(f"  surgical_attn(2D, no fold) {ax}: acc={s['acc']:.4f} "
                      f"trunc={_pct(s['truncation_rate'])} recovered={s['recovered']}")
    print("  -- Stage 1: root cause --")
    print(f"  regression_root_caused = {regression_root_caused} "
          f"(class=ple_embed_scale_fold_gated_off_on_vanilla_serve)")
    print(f"  global_fa_load_failed  = {global_fa_load_failed}")
    print("  -- Stage 2: recovered denominator (ple_fold) --")
    print(f"  recovered_vanilla_base_mmlu/gpqa = {recovered_mmlu:.4f} / {recovered_gpqa:.4f}")
    print(f"  recovered_base_reproduces_anchor = {recovered_base_reproduces_anchor}")
    print(f"  strict_cilb_passes_vs_healthy_fresh_base = {strict_cilb_passes_vs_healthy_fresh_base}")
    print(f"  healthy_fresh_base_recoverable   = {healthy_fresh_base_recoverable}")
    print("  -- Stage 3: surgical-attn-only mechanism --")
    print(f"  surgical_attn_alone_fixes_longcot = {surgical_attn_alone_fixes_longcot}")
    print(f"  self_det = {self_det}")
    print("MARKER:", json.dumps(marker))

    senpai_result = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [_runid()],
        "primary_metric": {"name": "recovered_vanilla_base_mmlu", "value": recovered_mmlu},
        "test_metric": {"name": "recovered_vanilla_base_gpqa", "value": recovered_gpqa},
    }
    print("SENPAI-RESULT:", json.dumps(senpai_result))

    if not a.no_wandb:
        rid = _log_wandb(report, marker, a)
        if rid:
            report["wandb_run_id"] = rid
            (HERE / "aggregate_557.json").write_text(json.dumps(report, indent=2))
            print(f"[wandb] run id={rid}")
    return 0


def _fmt(x):
    return "NA" if x is None else f"{x:.3f}"


def _log_wandb(report, marker, a):
    rid = _runid()
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] import failed: {exc!r}; JSON saved only")
        return None
    import os
    if not (os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE")):
        print("[wandb] no key; JSON only")
        return None
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            id=rid or None, resume="allow" if rid else None,
            name=a.wandb_name, group=a.wandb_group, job_type="analysis",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] resume init failed: {exc!r}")
        return None
    for k, v in marker.items():
        if isinstance(v, (int, float, bool, str)):
            run.summary[k] = v
    for tag, ax in (("mmlu", report["mmlu"]), ("gpqa", report["gpqa"])):
        for kk in ("recovered_acc", "recovered_wilson_lo", "recovered_wilson_hi",
                   "recovered_pct_of_anchor", "recovered_truncation_rate", "broken_acc",
                   "broken_truncation_rate", "z_vs_anchor", "within_anchor_ci",
                   "point_meets_floor", "cilb_meets_floor", "trunc_collapsed_vs_broken",
                   "axis_recovered", "n_prompt_mismatch_broken", "n_prompt_mismatch_fullhead"):
            v = ax.get(kk)
            if isinstance(v, (int, float, bool)):
                run.summary[f"{tag}/{kk}"] = v
        s = ax.get("strict_cilb_live")
        if s:
            run.summary[f"{tag}/fullhead_cilb_clears_live_floor"] = s["fullhead_cilb_clears_live_floor"]
            run.summary[f"{tag}/fullhead_wilson_lo"] = s["fullhead_wilson_lo"]
    for ax in ("mmlu", "gpqa"):
        s = report["surgical_attn"].get(ax)
        if s:
            run.summary[f"surgical_attn/{ax}_acc"] = s["acc"]
            run.summary[f"surgical_attn/{ax}_truncation_rate"] = s["truncation_rate"]
            run.summary[f"surgical_attn/{ax}_recovered"] = s["recovered"]
    try:
        run.finish()
    except Exception:
        pass
    return rid or getattr(run, "id", None)


if __name__ == "__main__":
    raise SystemExit(main())
