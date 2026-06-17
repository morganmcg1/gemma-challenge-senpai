#!/usr/bin/env python3
"""PR #542 — aggregate the base vs base_fullhead 2x2 (MMLU-Pro + GPQA-Diamond),
honoring the advisor's min_tokens=8 EOS-guard steer.

Three columns per axis:
  base                  : plain stock int4 served vanilla (fresh denominator)
  base_fullhead (asserved): full fast stack on the SAME ckpt, as-served
  base_fullhead (mintok): the as-served fullhead with the immediate-EOS empties
                          recovered under request-level min_tokens=8 (recover_min_tokens.py)

The verdict binds to the **min_tokens-adjusted** fullhead: it removes a recoverable
serving artifact (wirbel #541) and degrades GRACEFULLY to the as-served number when
there are zero empties (mintok == as-served), so it never needs an arbitrary
"elevated" threshold. We report the as-served verdict alongside for transparency,
plus the per-arm failure-mode breakdown (empty-EOS vs max_tokens truncation) so the
gate's binding failure mode is attributable.

  per axis: does base_fullhead's Wilson CI LOWER BOUND clear 0.90 x (this run's
  freshly-measured base on the identical set)?

Top-line base_fullhead_shortchain_quality_safe = both axes clear (min_tokens-bound).

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
ANCHOR_BASE_MMLU = 0.668
ANCHOR_BASE_GPQA = 0.470
ANCHOR_SHIP_MMLU = 0.274  # live-12k-ship collapse base_fullhead must NOT reproduce
ANCHOR_SHIP_GPQA = 0.232


def wilson(k: int, n: int, z: float = 1.96):
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


def _fm(failmodes, fname):
    c = (failmodes or {}).get(fname, {})
    return {
        "empty_eos": c.get("n_empty_eos", 0),
        "truncation": c.get("n_truncation", 0),
        "other_fail": c.get("n_other_fail", 0),
        "empty_eos_rate": c.get("empty_eos_rate", 0.0),
        "truncation_rate": c.get("truncation_rate", 0.0),
        "extract_fail_rate": c.get("extract_fail_rate", 0.0),
        "n_error": c.get("n_error", 0),
    }


def axis(name, base, full_as, full_mt, floor, fm_base_name, fm_full_name, failmodes) -> dict:
    bk, bn = base["n_correct"], base["n_scored"]
    ak, an = full_as["n_correct"], full_as["n_scored"]
    mk, mn = full_mt["n_correct"], full_mt["n_scored"]
    bp, blo, bhi = wilson(bk, bn)
    ap, alo, ahi = wilson(ak, an)
    mp, mlo, mhi = wilson(mk, mn)
    gate90 = 0.90 * bp  # fresh-base 90% gate (verdict denominator)

    pid, n_mis = prompt_identical(base, full_as)
    return {
        "axis": name,
        "base_acc": bp, "base_n": bn, "base_correct": bk,
        "base_wilson_lo": blo, "base_wilson_hi": bhi,
        # as-served fullhead
        "fullhead_asserved_acc": ap, "fullhead_asserved_n": an, "fullhead_asserved_correct": ak,
        "fullhead_asserved_wilson_lo": alo, "fullhead_asserved_wilson_hi": ahi,
        "pct_of_base_asserved": (ap / bp) if bp else float("nan"),
        "meets_90pct_asserved": bool(alo >= gate90),
        "point_meets_90pct_asserved": bool(ap >= gate90),
        # min_tokens-adjusted fullhead (the BINDING column)
        "base_fullhead_acc": mp, "base_fullhead_n": mn, "base_fullhead_correct": mk,
        "base_fullhead_wilson_lo": mlo, "base_fullhead_wilson_hi": mhi,
        "pct_of_base": (mp / bp) if bp else float("nan"),
        "meets_90pct": bool(mlo >= gate90),                # binding verdict (CI-lb)
        "point_meets_90pct": bool(mp >= gate90),
        "recovered_n": full_mt.get("recovered_n", 0),
        "recovered_flipped_to_correct": full_mt.get("recovered_flipped_to_correct", 0),
        # gate plumbing
        "fresh_base_90pct_gate": gate90,
        "documented_floor": floor,
        "meets_documented_floor": bool(mlo >= floor),
        "prompt_identical": pid, "n_prompt_mismatch": n_mis,
        # failure-mode breakdown (advisor steer)
        "fullhead_failmodes": _fm(failmodes, fm_full_name),
        "base_failmodes": _fm(failmodes, fm_base_name),
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

    bm, fm_as = _load(d / "base_mmlu_pro.json"), _load(d / "fullhead_mmlu_pro.json")
    bg, fg_as = _load(d / "base_gpqa.json"), _load(d / "fullhead_gpqa.json")
    # min_tokens-adjusted fullhead; fall back to as-served if recovery not yet run.
    fm_mt = _load(d / "fullhead_mmlu_pro.mintok.json") or fm_as
    fg_mt = _load(d / "fullhead_gpqa.mintok.json") or fg_as
    failmodes = _load(d / "failmodes.json")

    missing = [n for n, x in [("base_mmlu", bm), ("fullhead_mmlu", fm_as),
                              ("base_gpqa", bg), ("fullhead_gpqa", fg_as)] if x is None]
    if missing:
        print(f"[aggregate] MISSING arms: {missing}", file=sys.stderr)
        return 2
    recovery_applied = (d / "fullhead_mmlu_pro.mintok.json").exists() and \
                       (d / "fullhead_gpqa.mintok.json").exists()

    mmlu = axis("mmlu_pro", bm, fm_as, fm_mt, FLOOR_MMLU,
                "base_mmlu_pro.json", "fullhead_mmlu_pro.json", failmodes)
    gpqa = axis("gpqa_diamond", bg, fg_as, fg_mt, FLOOR_GPQA,
                "base_gpqa.json", "fullhead_gpqa.json", failmodes)
    quality_safe = bool(mmlu["meets_90pct"] and gpqa["meets_90pct"])
    quality_safe_asserved = bool(mmlu["meets_90pct_asserved"] and gpqa["meets_90pct_asserved"])

    def fmgroup(ax, who):
        f = ax[f"{who}_failmodes"]
        return [f["empty_eos"], f["truncation"], f["other_fail"]]

    marker = {
        "concurrency": a.conc,
        "recovery_applied": recovery_applied,
        # MMLU-Pro
        "mmlu_pro_base": mmlu["base_acc"],
        "mmlu_pro_base_fullhead": mmlu["base_fullhead_acc"],            # binding (min_tokens)
        "mmlu_pro_base_fullhead_asserved": mmlu["fullhead_asserved_acc"],
        "mmlu_pro_pct_of_base": mmlu["pct_of_base"],
        "mmlu_pro_pct_of_base_asserved": mmlu["pct_of_base_asserved"],
        "mmlu_pro_meets_90pct": mmlu["meets_90pct"],
        "mmlu_pro_meets_90pct_asserved": mmlu["meets_90pct_asserved"],
        "mmlu_pro_base_wilson_ci": [mmlu["base_wilson_lo"], mmlu["base_wilson_hi"]],
        "mmlu_pro_base_fullhead_wilson_ci": [mmlu["base_fullhead_wilson_lo"], mmlu["base_fullhead_wilson_hi"]],
        "mmlu_pro_base_fullhead_asserved_wilson_ci": [mmlu["fullhead_asserved_wilson_lo"], mmlu["fullhead_asserved_wilson_hi"]],
        "mmlu_pro_fresh_base_90pct_gate": mmlu["fresh_base_90pct_gate"],
        "mmlu_pro_fullhead_empty_eos": mmlu["fullhead_failmodes"]["empty_eos"],
        "mmlu_pro_fullhead_truncation": mmlu["fullhead_failmodes"]["truncation"],
        "mmlu_pro_fullhead_empty_eos_rate": mmlu["fullhead_failmodes"]["empty_eos_rate"],
        "mmlu_pro_base_empty_eos": mmlu["base_failmodes"]["empty_eos"],
        "mmlu_pro_base_truncation": mmlu["base_failmodes"]["truncation"],
        "mmlu_pro_recovered_flipped": mmlu["recovered_flipped_to_correct"],
        # GPQA-Diamond
        "gpqa_d_base": gpqa["base_acc"],
        "gpqa_d_base_fullhead": gpqa["base_fullhead_acc"],
        "gpqa_d_base_fullhead_asserved": gpqa["fullhead_asserved_acc"],
        "gpqa_d_pct_of_base": gpqa["pct_of_base"],
        "gpqa_d_pct_of_base_asserved": gpqa["pct_of_base_asserved"],
        "gpqa_d_meets_90pct": gpqa["meets_90pct"],
        "gpqa_d_meets_90pct_asserved": gpqa["meets_90pct_asserved"],
        "gpqa_d_base_wilson_ci": [gpqa["base_wilson_lo"], gpqa["base_wilson_hi"]],
        "gpqa_d_base_fullhead_wilson_ci": [gpqa["base_fullhead_wilson_lo"], gpqa["base_fullhead_wilson_hi"]],
        "gpqa_d_base_fullhead_asserved_wilson_ci": [gpqa["fullhead_asserved_wilson_lo"], gpqa["fullhead_asserved_wilson_hi"]],
        "gpqa_d_fresh_base_90pct_gate": gpqa["fresh_base_90pct_gate"],
        "gpqa_d_fullhead_empty_eos": gpqa["fullhead_failmodes"]["empty_eos"],
        "gpqa_d_fullhead_truncation": gpqa["fullhead_failmodes"]["truncation"],
        "gpqa_d_fullhead_empty_eos_rate": gpqa["fullhead_failmodes"]["empty_eos_rate"],
        "gpqa_d_base_empty_eos": gpqa["base_failmodes"]["empty_eos"],
        "gpqa_d_base_truncation": gpqa["base_failmodes"]["truncation"],
        "gpqa_d_recovered_flipped": gpqa["recovered_flipped_to_correct"],
        # top-line
        "base_fullhead_shortchain_quality_safe": quality_safe,                  # min_tokens-bound
        "base_fullhead_shortchain_quality_safe_asserved": quality_safe_asserved,
        "analysis_only": True,
        "official_tps": 0,
    }

    report = {
        "pr": 542,
        "checkpoint": "google/gemma-4-E4B-it-qat-w4a16-ct (stock int4, native 262k BF16 head, 42L)",
        "mmlu": mmlu, "gpqa": gpqa,
        "marker": marker,
        "recovery_applied": recovery_applied,
        "documented_floors": {"mmlu_pro": FLOOR_MMLU, "gpqa_d": FLOOR_GPQA},
        "banked_anchors": {
            "base_mmlu": ANCHOR_BASE_MMLU, "base_gpqa": ANCHOR_BASE_GPQA,
            "ship_collapse_mmlu": ANCHOR_SHIP_MMLU, "ship_collapse_gpqa": ANCHOR_SHIP_GPQA,
        },
        "analysis_only": True, "no_hf_job": True, "official_tps": 0,
    }
    (d / "aggregate.json").write_text(json.dumps(report, indent=2))

    def fmt(ax):
        ff = ax["fullhead_failmodes"]; bf = ax["base_failmodes"]
        return (
            f"  {ax['axis']:13s} base={ax['base_acc']:.4f} (n={ax['base_n']}, "
            f"CI {ax['base_wilson_lo']:.3f}-{ax['base_wilson_hi']:.3f})\n"
            f"    fullhead as-served={ax['fullhead_asserved_acc']:.4f} "
            f"(CI {ax['fullhead_asserved_wilson_lo']:.3f}-{ax['fullhead_asserved_wilson_hi']:.3f}, "
            f"{ax['pct_of_base_asserved']*100:.1f}% of base, CI-lb>=gate:{ax['meets_90pct_asserved']})\n"
            f"    fullhead min_tokens={ax['base_fullhead_acc']:.4f} "
            f"(CI {ax['base_fullhead_wilson_lo']:.3f}-{ax['base_fullhead_wilson_hi']:.3f}, "
            f"{ax['pct_of_base']*100:.1f}% of base, CI-lb>=gate:{ax['meets_90pct']})  "
            f"recovered={ax['recovered_n']} flipped+={ax['recovered_flipped_to_correct']}\n"
            f"    gate(0.9*base)={ax['fresh_base_90pct_gate']:.4f}  prompts_identical={ax['prompt_identical']}\n"
            f"    fullhead failmodes: emptyEOS={ff['empty_eos']} trunc={ff['truncation']} other={ff['other_fail']}  "
            f"| base failmodes: emptyEOS={bf['empty_eos']} trunc={bf['truncation']} other={bf['other_fail']}"
        )

    print("\n==== base_fullhead SHORT-CHAIN QUALITY 2x2 (conc=32) ====")
    print(f"  recovery_applied={recovery_applied}")
    print(fmt(mmlu))
    print(fmt(gpqa))
    print(f"  base_fullhead_shortchain_quality_safe (min_tokens-bound) = {quality_safe}")
    print(f"  base_fullhead_shortchain_quality_safe (as-served)        = {quality_safe_asserved}")
    print("MARKER:", json.dumps(marker))

    senpai_result = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [],
        "primary_metric": {"name": "mmlu_pro_base_fullhead", "value": mmlu["base_fullhead_acc"]},
        "test_metric": {"name": "gpqa_d_base_fullhead", "value": gpqa["base_fullhead_acc"]},
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
              ">=90%-of-fresh-base on MMLU-Pro + GPQA-Diamond at conc=32? Reports as-served "
              "AND min_tokens=8 EOS-guarded (advisor steer) + per-arm failure-mode breakdown.",
        config={"pr": 542, "analysis_only": True, "official_tps": 0, "concurrency": a.conc,
                "checkpoint": "google/gemma-4-E4B-it-qat-w4a16-ct",
                "min_tokens_guard": 8, "recovery_applied": report["recovery_applied"],
                "floor_mmlu": FLOOR_MMLU, "floor_gpqa": FLOOR_GPQA},
    )
    if run is None:
        print("[wandb] disabled (no key); JSON only")
        return None
    for k, v in marker.items():
        if isinstance(v, (int, float, bool, str)):
            run.summary[k] = v
    for tag, ax in (("mmlu", report["mmlu"]), ("gpqa", report["gpqa"])):
        for kk in ("base_acc", "base_fullhead_acc", "fullhead_asserved_acc",
                   "pct_of_base", "pct_of_base_asserved", "meets_90pct",
                   "meets_90pct_asserved", "point_meets_90pct", "meets_documented_floor",
                   "base_n", "base_fullhead_n", "recovered_n", "recovered_flipped_to_correct",
                   "prompt_identical", "n_prompt_mismatch"):
            run.summary[f"{tag}/{kk}"] = ax[kk]
        for who in ("fullhead", "base"):
            for kk, vv in ax[f"{who}_failmodes"].items():
                run.summary[f"{tag}/{who}_{kk}"] = vv
    try:
        finish_wandb(run)
    except Exception:
        pass
    return getattr(run, "id", None)


if __name__ == "__main__":
    raise SystemExit(main())
