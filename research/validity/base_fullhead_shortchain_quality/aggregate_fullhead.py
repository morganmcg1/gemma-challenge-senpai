#!/usr/bin/env python3
"""PR #542 — aggregate the base vs base_fullhead 2x2 (MMLU-Pro + GPQA-Diamond).

Reads the four per-arm run_eval.py JSONs, verifies base/base_fullhead saw
BYTE-IDENTICAL prompts (per-question prompt_sha), computes Wilson 95% score
intervals, and decides the verdict the PR specifies:

  per axis: does base_fullhead's Wilson CI LOWER BOUND clear 0.90 x (this run's
  freshly-measured base on the identical set)?

Top-line base_fullhead_shortchain_quality_safe = both axes clear. Emits the
machine-readable terminal marker land #534 ingests + logs W&B (analysis_only).

Usage:
  aggregate_fullhead.py --dir <here> [--no-wandb]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))

# Documented gate floors (Morgan #515): 90% of the ubel #511 banked base anchors.
FLOOR_MMLU = 0.601  # 0.90 * 0.668
FLOOR_GPQA = 0.423  # 0.90 * 0.470 (stricter 0.42 anchor)
# ubel #511/#527 banked anchors (cross-check only; verdict binds to FRESH base).
ANCHOR_BASE_MMLU = 0.668
ANCHOR_BASE_GPQA = 0.470
ANCHOR_SHIP_MMLU = 0.274  # the live-12k-ship collapse base_fullhead must NOT reproduce
ANCHOR_SHIP_GPQA = 0.232


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score 95% interval for k successes in n trials."""
    if not n:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def _load(path: Path):
    return json.load(open(path)) if path.exists() else None


def prompt_identical(base, full) -> tuple[bool, int]:
    if base is None or full is None:
        return (False, -1)
    b = {r["id"]: r.get("prompt_sha") for r in base["per_sample"]}
    f = {r["id"]: r.get("prompt_sha") for r in full["per_sample"]}
    common = set(b) & set(f)
    mism = [i for i in common if b[i] != f[i]]
    return (len(mism) == 0, len(mism))


def axis(name: str, base, full, floor: float) -> dict:
    bk, bn = base["n_correct"], base["n_scored"]
    fk, fn = full["n_correct"], full["n_scored"]
    bp, blo, bhi = wilson(bk, bn)
    fp, flo, fhi = wilson(fk, fn)
    pct = (fp / bp) if bp else float("nan")
    gate90 = 0.90 * bp  # fresh-base 90% gate (the verdict denominator)
    # PR verdict: base_fullhead's Wilson CI LOWER BOUND clears 0.90 x fresh base.
    meets_90pct = bool(flo >= gate90)
    point_meets_90pct = bool(fp >= gate90)
    meets_floor = bool(flo >= floor)
    pid, n_mis = prompt_identical(base, full)
    return {
        "axis": name,
        "base_acc": bp, "base_n": bn, "base_correct": bk,
        "base_wilson_lo": blo, "base_wilson_hi": bhi,
        "base_fullhead_acc": fp, "base_fullhead_n": fn, "base_fullhead_correct": fk,
        "base_fullhead_wilson_lo": flo, "base_fullhead_wilson_hi": fhi,
        "pct_of_base": pct,
        "fresh_base_90pct_gate": gate90,
        "meets_90pct": meets_90pct,             # CI-lower-bound verdict (PR-specified)
        "point_meets_90pct": point_meets_90pct,  # point-estimate (secondary)
        "documented_floor": floor,
        "meets_documented_floor": meets_floor,
        "prompt_identical": pid, "n_prompt_mismatch": n_mis,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "research/validity/base_fullhead_shortchain_quality"))
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-name", default="stark/base-fullhead-shortchain-quality")
    ap.add_argument("--wandb-group", default="base-fullhead-shortchain-quality")
    a = ap.parse_args()
    d = Path(a.dir)

    bm = _load(d / "base_mmlu_pro.json")
    fm = _load(d / "fullhead_mmlu_pro.json")
    bg = _load(d / "base_gpqa.json")
    fg = _load(d / "fullhead_gpqa.json")
    missing = [n for n, x in [("base_mmlu", bm), ("fullhead_mmlu", fm),
                              ("base_gpqa", bg), ("fullhead_gpqa", fg)] if x is None]
    if missing:
        print(f"[aggregate] MISSING arms: {missing}", file=sys.stderr)
        return 2

    mmlu = axis("mmlu_pro", bm, fm, FLOOR_MMLU)
    gpqa = axis("gpqa_diamond", bg, fg, FLOOR_GPQA)
    quality_safe = bool(mmlu["meets_90pct"] and gpqa["meets_90pct"])

    marker = {
        "concurrency": a.conc,
        # MMLU-Pro
        "mmlu_pro_base": mmlu["base_acc"],
        "mmlu_pro_base_fullhead": mmlu["base_fullhead_acc"],
        "mmlu_pro_pct_of_base": mmlu["pct_of_base"],
        "mmlu_pro_meets_90pct": mmlu["meets_90pct"],
        "mmlu_pro_base_wilson_ci": [mmlu["base_wilson_lo"], mmlu["base_wilson_hi"]],
        "mmlu_pro_base_fullhead_wilson_ci": [mmlu["base_fullhead_wilson_lo"], mmlu["base_fullhead_wilson_hi"]],
        "mmlu_pro_fresh_base_90pct_gate": mmlu["fresh_base_90pct_gate"],
        # GPQA-Diamond
        "gpqa_d_base": gpqa["base_acc"],
        "gpqa_d_base_fullhead": gpqa["base_fullhead_acc"],
        "gpqa_d_pct_of_base": gpqa["pct_of_base"],
        "gpqa_d_meets_90pct": gpqa["meets_90pct"],
        "gpqa_d_base_wilson_ci": [gpqa["base_wilson_lo"], gpqa["base_wilson_hi"]],
        "gpqa_d_base_fullhead_wilson_ci": [gpqa["base_fullhead_wilson_lo"], gpqa["base_fullhead_wilson_hi"]],
        "gpqa_d_fresh_base_90pct_gate": gpqa["fresh_base_90pct_gate"],
        # top-line
        "base_fullhead_shortchain_quality_safe": quality_safe,
        "analysis_only": True,
        "official_tps": 0,
    }

    report = {
        "pr": 542,
        "checkpoint": "google/gemma-4-E4B-it-qat-w4a16-ct (stock int4, native 262k BF16 head, 42L)",
        "mmlu": mmlu, "gpqa": gpqa,
        "marker": marker,
        "documented_floors": {"mmlu_pro": FLOOR_MMLU, "gpqa_d": FLOOR_GPQA},
        "banked_anchors": {
            "base_mmlu": ANCHOR_BASE_MMLU, "base_gpqa": ANCHOR_BASE_GPQA,
            "ship_collapse_mmlu": ANCHOR_SHIP_MMLU, "ship_collapse_gpqa": ANCHOR_SHIP_GPQA,
        },
        "analysis_only": True, "no_hf_job": True, "official_tps": 0,
    }
    (d / "aggregate.json").write_text(json.dumps(report, indent=2))

    def fmt(ax):
        return (f"  {ax['axis']:13s} base={ax['base_acc']:.4f} (n={ax['base_n']}, "
                f"CI {ax['base_wilson_lo']:.3f}-{ax['base_wilson_hi']:.3f})  "
                f"fullhead={ax['base_fullhead_acc']:.4f} (n={ax['base_fullhead_n']}, "
                f"CI {ax['base_fullhead_wilson_lo']:.3f}-{ax['base_fullhead_wilson_hi']:.3f})  "
                f"pct={ax['pct_of_base']*100:.1f}%  gate(0.9*base)={ax['fresh_base_90pct_gate']:.4f}  "
                f"CI-lb>=gate: {ax['meets_90pct']}  prompts_identical={ax['prompt_identical']}")

    print("\n==== base_fullhead SHORT-CHAIN QUALITY 2x2 (conc=32) ====")
    print(fmt(mmlu))
    print(fmt(gpqa))
    print(f"  base_fullhead_shortchain_quality_safe = {quality_safe}")
    print("MARKER:", json.dumps(marker))

    test_val = gpqa["base_fullhead_acc"]
    senpai_result = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [],
        "primary_metric": {"name": "mmlu_pro_base_fullhead", "value": mmlu["base_fullhead_acc"]},
        "test_metric": {"name": "gpqa_d_base_fullhead", "value": test_val},
    }
    print("SENPAI-RESULT:", json.dumps(senpai_result))

    if not a.no_wandb:
        rid = _log_wandb(report, marker, a)
        if rid:
            report["wandb_run_id"] = rid
            (d / "aggregate.json").write_text(json.dumps(report, indent=2))
            print(f"[wandb] run id={rid}")
    return 0


def _log_wandb(report, marker, a):
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] import failed: {exc!r}; JSON saved only")
        return None
    run = init_wandb_run(
        job_type="local_profiling", agent="stark",
        name=a.wandb_name, group=a.wandb_group,
        notes="PR#542 base_fullhead short-chain quality: does the full fast stack on the "
              "stock int4 ckpt (native 262k BF16 head, no osoi5 bake, no prune) clear "
              ">=90%-of-fresh-base on MMLU-Pro + GPQA-Diamond at conc=32?",
        config={"pr": 542, "analysis_only": True, "official_tps": 0, "concurrency": a.conc,
                "checkpoint": "google/gemma-4-E4B-it-qat-w4a16-ct",
                "floor_mmlu": FLOOR_MMLU, "floor_gpqa": FLOOR_GPQA},
    )
    if run is None:
        print("[wandb] disabled (no key); JSON only")
        return None
    for k, v in marker.items():
        if isinstance(v, (int, float, bool, str)):
            run.summary[k] = v
    for tag, ax in (("mmlu", report["mmlu"]), ("gpqa", report["gpqa"])):
        for kk in ("base_acc", "base_fullhead_acc", "pct_of_base", "meets_90pct",
                   "point_meets_90pct", "meets_documented_floor", "base_n",
                   "base_fullhead_n", "prompt_identical", "n_prompt_mismatch"):
            run.summary[f"{tag}/{kk}"] = ax[kk]
    try:
        finish_wandb(run)
    except Exception:
        pass
    return getattr(run, "id", None)


if __name__ == "__main__":
    raise SystemExit(main())
