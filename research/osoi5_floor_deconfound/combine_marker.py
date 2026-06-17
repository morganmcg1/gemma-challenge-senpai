#!/usr/bin/env python3
"""PR #548 terminal marker: de-confound the shipped osoi5 quality collapse into
{body-bake, EOS-artifact} by comparing as-served vs min_tokens=8-floored quality
on MMLU-Pro / GSM8K / GPQA-Diamond, with per-arm empty/EOS rate.

Reads the per-(arm,axis) result files written by run_osoi5_floor.py and emits the
single-line SENPAI-RESULT marker + W&B summary. Tolerant of partial data (one
arm / missing axis) so interim checks work; the gate booleans require all three
floored axes present.

LOCAL ONLY. analysis_only=true, official_tps=0.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# ---- public-evidence anchors -----------------------------------------------------
# ubel #538 intact-body full-head control (un-collapsed ceiling) == PR #511 base arm.
BASE = {"mmlu_pro": 0.668, "gpqa_d": 0.444, "gsm8k": 0.878}
# Morgan #524 gate floors (GSM8K floor = 90% of vanilla base).
GATE = {"mmlu_pro": 0.601, "gpqa_d": 0.400, "gsm8k": round(0.90 * 0.878, 4)}
# Shipped osoi5 PRE-floor historical (min_tokens=null), for narrative context.
HIST_PREFLOOR = {"mmlu_pro": 0.274, "gpqa_d": 0.232}

ARMS = ["as_served", "floored"]
AXES = ["mmlu_pro", "gsm8k", "gpqa_d"]

FILES = {
    "mmlu_pro": "osoi5_{arm}_mmlu_pro.json",
    "gpqa_d": "osoi5_{arm}_gpqa_diamond.json",
    "gsm8k": "osoi5_{arm}_sampled.json",
}


def _load(name: str):
    p = HERE / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _axis_arm(axis: str, arm: str):
    """Return (acc, empty_rate, n, extra) for one axis/arm, or None if missing."""
    d = _load(FILES[axis].format(arm=arm))
    if d is None:
        return None
    if axis in ("mmlu_pro", "gpqa_d"):
        acc = float(d["accuracy"])
        n = int(d.get("n_scored") or d.get("n_samples") or 0)
        empty = float(d.get("empty_rate", float("nan")))
        extra = {
            "n_samples": d.get("n_samples"),
            "n_scored": d.get("n_scored"),
            "n_correct": d.get("n_correct"),
            "n_empty": d.get("n_empty"),
            "n_error": d.get("n_error"),
            "max_tokens": d.get("max_tokens"),
        }
    else:  # gsm8k
        acc = float(d["accuracy"])
        pp = d.get("per_problem", [])
        n = len(pp)
        n_empty = sum(1 for e in pp
                      if int(e.get("sample_chars", 0)) == 0 and e.get("finish_reason") != "error")
        empty = (n_empty / n) if n else float("nan")
        extra = {
            "n_problems": d.get("n_problems"),
            "n_correct": d.get("n_correct"),
            "n_empty": n_empty,
            "extract_fail_rate": d.get("extract_fail_rate"),
            "truncation_rate": d.get("truncation_rate"),
            "regime": d.get("regime"),
            "sampling": d.get("sampling"),
        }
    return {"acc": acc, "empty_rate": empty, "n": n, "extra": extra}


def _se_diff(p1, n1, p2, n2):
    if not n1 or not n2:
        return float("nan")
    return math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)


def main() -> int:
    by_arm = {axis: {arm: _axis_arm(axis, arm) for arm in ARMS} for axis in AXES}

    table = {}
    for axis in AXES:
        a = by_arm[axis]["as_served"]
        f = by_arm[axis]["floored"]
        row = {
            "base": BASE[axis],
            "gate": GATE[axis],
            "acc_as_served": (round(a["acc"], 4) if a else None),
            "acc_floored": (round(f["acc"], 4) if f else None),
            "empty_rate_as_served": (round(a["empty_rate"], 4) if a else None),
            "empty_rate_floored": (round(f["empty_rate"], 4) if f else None),
            "n_as_served": (a["n"] if a else None),
            "n_floored": (f["n"] if f else None),
            "pct_of_base_floored": (round(f["acc"] / BASE[axis], 4) if f else None),
            "pct_of_base_as_served": (round(a["acc"] / BASE[axis], 4) if a else None),
            "floored_meets_gate": (bool(f["acc"] >= GATE[axis]) if f else None),
        }
        if a and f:
            delta = f["acc"] - a["acc"]
            se = _se_diff(a["acc"], a["n"], f["acc"], f["n"])
            row["delta_acc_floor_minus_served"] = round(delta, 4)
            row["se_diff"] = (round(se, 5) if se == se else None)
            row["material_recovery_2sigma"] = bool(se == se and delta > 2 * se)
        if a:
            row["extra_as_served"] = a["extra"]
        if f:
            row["extra_floored"] = f["extra"]
        table[axis] = row

    # ---- key-output booleans ----
    have_floored_all = all(by_arm[ax]["floored"] is not None for ax in AXES)
    have_served_all = all(by_arm[ax]["as_served"] is not None for ax in AXES)

    floored_meets_gate = (
        have_floored_all and all(table[ax]["floored_meets_gate"] for ax in AXES)
    )
    recovers = any(
        table[ax].get("material_recovery_2sigma") and table[ax].get("delta_acc_floor_minus_served", 0) > 0
        for ax in AXES
    )
    moat_is_genuine = bool(have_floored_all and not floored_meets_gate)

    osoi5_empty_rate_mmlu = table["mmlu_pro"]["empty_rate_as_served"]
    osoi5_empty_rate_gsm8k = table["gsm8k"]["empty_rate_as_served"]
    osoi5_empty_rate_gpqa = table["gpqa_d"]["empty_rate_as_served"]

    peak_vram = 0.0
    for arm in ARMS:
        s = _load(f"arm_summary_{arm}.json")
        if s and s.get("peak_vram_gb"):
            peak_vram = max(peak_vram, float(s["peak_vram_gb"]))

    marker = {
        "terminal": bool(have_served_all and have_floored_all),
        "analysis_only": True,
        "official_tps": 0,
        "shipped_osoi5_is_secretly_quality_safe": bool(floored_meets_gate),
        "moat_is_genuine": moat_is_genuine,
        "osoi5_recovers_with_floor": bool(recovers),
        "osoi5_floored_meets_gate": bool(floored_meets_gate),
        "osoi5_empty_rate_mmlu_as_served": osoi5_empty_rate_mmlu,
        "osoi5_empty_rate_gsm8k_as_served": osoi5_empty_rate_gsm8k,
        "osoi5_empty_rate_gpqa_as_served": osoi5_empty_rate_gpqa,
        "mmlu_pro_acc_as_served": table["mmlu_pro"]["acc_as_served"],
        "mmlu_pro_acc_floored": table["mmlu_pro"]["acc_floored"],
        "gsm8k_acc_as_served": table["gsm8k"]["acc_as_served"],
        "gsm8k_acc_floored": table["gsm8k"]["acc_floored"],
        "gpqa_d_acc_as_served": table["gpqa_d"]["acc_as_served"],
        "gpqa_d_acc_floored": table["gpqa_d"]["acc_floored"],
        "self_det": True,  # MMLU-Pro & GPQA greedy (temp=0 argmax) deterministic by construction
        "peak_vram_gb": round(peak_vram, 3),
    }

    report = {
        "pr": 548,
        "wandb_group": "osoi5-floor-deconfound",
        "analysis_only": True,
        "official_tps": 0,
        "marker": marker,
        "osoi5_quality_by_arm": table,
        "anchors": {
            "base_intact_fullhead_ubel538": BASE,
            "gate_morgan524": GATE,
            "shipped_osoi5_preflood_historical_min_tokens_null": HIST_PREFLOOR,
            "shipped_local_tps": 353.73,
            "shipped_official_tps": 375.857,
        },
        "interpretation": {
            "have_served_all_axes": have_served_all,
            "have_floored_all_axes": have_floored_all,
            "floored_meets_gate": floored_meets_gate,
            "moat_is_genuine": moat_is_genuine,
            "recovers_with_floor": recovers,
            "self_det_basis": "MMLU-Pro & GPQA greedy temperature=0 argmax (deterministic); GSM8K seeded sampled per lewtun generation_config",
        },
    }

    out = HERE / "osoi5_floor_deconfound_marker.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nSENPAI-RESULT " + json.dumps(marker), flush=True)
    print(f"\n[wrote] {out}", flush=True)

    if "--no-wandb" not in sys.argv:
        try:
            sys.path.insert(0, str(ROOT))
            from scripts.wandb_logging import (
                init_wandb_run, log_summary, log_json_artifact, finish_wandb,
            )
            run = init_wandb_run(
                job_type="downstream-quality",
                agent="wirbel",
                name="wirbel/osoi5-floor-deconfound-marker",
                group="osoi5-floor-deconfound",
                tags=["osoi5", "min-tokens-floor", "eos-deconfound", "baked-body",
                      "mmlu-pro", "gsm8k", "gpqa", "pr548"],
                notes="PR #548: is the shipped osoi5 secretly quality-safe once the "
                      "first-token-EOS empties are floored? Per-arm empty-rate de-confound.",
                config={"pr": 548, "analysis_only": True, "official_tps": 0},
            )
            if run is not None:
                flat = dict(marker)
                for ax in AXES:
                    for k in ("acc_as_served", "acc_floored", "empty_rate_as_served",
                              "empty_rate_floored", "pct_of_base_floored",
                              "delta_acc_floor_minus_served"):
                        v = table[ax].get(k)
                        if isinstance(v, (int, float)):
                            flat[f"{ax}/{k}"] = v
                log_summary(run, flat, step=0)
                log_json_artifact(run, name="osoi5_floor_deconfound_marker",
                                  artifact_type="quality-marker", data=report)
                finish_wandb(run)
                print(f"[wandb] logged run={run.id}", flush=True)
            else:
                print("[wandb] skipped (disabled / no key)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] error (non-fatal): {exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
