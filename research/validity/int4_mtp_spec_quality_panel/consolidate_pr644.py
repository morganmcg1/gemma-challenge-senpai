#!/usr/bin/env python3
"""PR #644 — Complete the Reading-A panel: Option-B MMLU-Pro + GSM8K %-of-base.

ANALYSIS-ONLY. Reads the 4 served-eval JSONs (GSM8K + MMLU-Pro, each greedy &
sampled) produced by run_pr644_panel.sh against the Option-B stack
(int4_g128_lmhead body + Gemma4-MTP K=7 drafter, BI=1, vLLM 0.22.0, gb6144,
min_tokens=8), computes Wilson 95% CIs and %-of-base vs ubel #628's gb6144 base
denominators, assigns the Reading-A verdicts, and logs everything to W&B.

Base denominators (ubel #628, GREEDY, gb6144, BI=1, mt=6144, mintok=8):
  MMLU-Pro 0.7180 (run 367i9s0t) ; GSM8K 0.9280 (run 4cxd1gfx).
Bars:
  MMLU-Pro : absolute floor 0.605 ; >=90%-of-base 0.6462
  GSM8K    : absolute floor 0.807 ; >=90%-of-base 0.8352

Run with /usr/bin/python3 (only python with a working wandb 0.27.0).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/senpai/target")
RES = ROOT / "research/validity/int4_mtp_spec_quality_panel/results-pr644"

# ubel #628 base denominators (GREEDY) and run ids.
BASE = {
    "mmlu_pro": {"acc": 0.7180, "run": "367i9s0t", "floor": 0.605, "bar90": 0.6462},
    "gsm8k":    {"acc": 0.9280, "run": "4cxd1gfx", "floor": 0.807, "bar90": 0.8352},
}

# (task, decode) -> result json filename
FILES = {
    ("gsm8k", "greedy"):    "optionb_pr644_gb6144_greedy_s1234.json",
    ("gsm8k", "sampled"):   "optionb_pr644_gb6144_sampled_s1234.json",
    ("mmlu_pro", "greedy"): "mmlu_pro_greedy.json",
    ("mmlu_pro", "sampled"):"mmlu_pro_sampled.json",
}


def wilson(k: int, n: int, z: float = 1.95996398454) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return center - half, center + half


def load_cell(task: str, decode: str) -> dict[str, Any] | None:
    fp = RES / FILES[(task, decode)]
    if not fp.exists():
        return None
    d = json.load(open(fp))
    # GSM8K harness uses n_problems; MMLU (run_eval.py) uses n_scored.
    n = d.get("n_scored")
    if n is None:
        n = d.get("n_problems")
    k = d.get("n_correct")
    acc = d.get("accuracy")
    if k is None and acc is not None and n is not None:
        k = round(acc * n)
    lo, hi = wilson(int(k), int(n))
    base = BASE[task]
    out = {
        "task": task, "decode": decode, "file": str(fp.relative_to(ROOT)),
        "n": int(n), "n_correct": int(k), "acc": float(acc),
        "wilson_lo": lo, "wilson_hi": hi,
        "base_acc": base["acc"], "base_run": base["run"],
        "pct_of_base": float(acc) / base["acc"],
        "pct_of_base_lo": lo / base["acc"], "pct_of_base_hi": hi / base["acc"],
        "floor": base["floor"], "bar90": base["bar90"],
        # diagnostics (present where harness records them)
        "extract_fail_rate": d.get("extract_fail_rate", d.get("empty_rate")),
        "truncation_rate": d.get("truncation_rate", d.get("finish_length_rate")),
        "length_stop_rate": d.get("length_stop_rate"),
        "strict_rate": d.get("strict_rate"),
        "wall_s": d.get("wall_s"),
        "max_tokens": d.get("sampling", {}).get("max_tokens") if isinstance(d.get("sampling"), dict) else d.get("max_tokens"),
        "min_tokens": d.get("sampling", {}).get("min_tokens") if isinstance(d.get("sampling"), dict) else d.get("min_tokens"),
        "created_at": d.get("created_at"),
        "eval_log": d.get("eval_log"),
    }
    return out


def ci_verdict(acc: float, lo: float, hi: float, bar: float) -> str:
    """CI-aware verdict against a single bar (consistent w/ GPQA precedent)."""
    if lo >= bar:
        return "PASSES"
    if hi < bar:
        return "FAILS"
    return "KNIFE_EDGE"


def bench_verdict(cell: dict[str, Any]) -> dict[str, Any]:
    """Reading-A verdict for a bench, keyed on the binding 90%-of-base bar,
    with the absolute floor reported as a secondary gate."""
    acc, lo, hi = cell["acc"], cell["wilson_lo"], cell["wilson_hi"]
    bar90, floor = cell["bar90"], cell["floor"]
    v90 = ci_verdict(acc, lo, hi, bar90)
    vfloor = ci_verdict(acc, lo, hi, floor)
    return {
        "verdict_bar90": v90,           # binding (bar90 > floor for both benches)
        "verdict_floor": vfloor,
        "point_ge_bar90": acc >= bar90,
        "point_ge_floor": acc >= floor,
        "ci_lo_ge_bar90": lo >= bar90,
    }


def build() -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    for (task, decode) in FILES:
        c = load_cell(task, decode)
        if c is not None:
            c.update(bench_verdict(c))
            cells[f"{task}_{decode}"] = c
    # Primary verdict per bench keys on GREEDY (apples-to-apples w/ greedy base).
    summary = {}
    for task in ("mmlu_pro", "gsm8k"):
        g = cells.get(f"{task}_greedy")
        s = cells.get(f"{task}_sampled")
        tag = "MMLU" if task == "mmlu_pro" else "GSM8K"
        if g is not None:
            summary[task] = {
                "greedy_acc": g["acc"], "greedy_ci": [g["wilson_lo"], g["wilson_hi"]],
                "greedy_pct_of_base": g["pct_of_base"],
                "greedy_pct_of_base_ci": [g["pct_of_base_lo"], g["pct_of_base_hi"]],
                "sampled_acc": (s["acc"] if s else None),
                "sampled_ci": ([s["wilson_lo"], s["wilson_hi"]] if s else None),
                "sampled_pct_of_base": (s["pct_of_base"] if s else None),
                "base_acc": g["base_acc"], "base_run": g["base_run"],
                "floor": g["floor"], "bar90": g["bar90"],
                "reading_a_label": f"READING_A_{tag}_{g['verdict_bar90']}",
                "verdict_floor": g["verdict_floor"],
            }
    return {"cells": cells, "summary": summary,
            "stack": "vllm==0.22.0 / int4_g128_lmhead + Gemma4-MTP K7 / BI=1 / gb6144 / min_tokens=8",
            "max_num_seqs_served": 16, "concurrency": 16,
            "pr": 644, "analysis_only": True, "official_tps": 0}


def log_wandb(result: dict[str, Any], group: str, name: str) -> str | None:
    sys.path.insert(0, str(ROOT))
    try:
        from scripts import wandb_logging as wl
    except Exception as exc:  # noqa: BLE001
        print(f"[pr644] wandb_logging import failed: {exc}", file=sys.stderr)
        return None
    cfg = {
        "pr": 644, "analysis_only": True, "official_tps": 0,
        "stack": result["stack"], "max_num_seqs": 16, "concurrency": 16,
        "max_tokens": 6144, "min_tokens": 8, "batch_invariant": 1,
        "engine": "vllm==0.22.0",
        "base_mmlu": BASE["mmlu_pro"]["acc"], "base_mmlu_run": BASE["mmlu_pro"]["run"],
        "base_gsm8k": BASE["gsm8k"]["acc"], "base_gsm8k_run": BASE["gsm8k"]["run"],
    }
    run = wl.init_wandb_run(
        job_type="optionb-quality-mmlu-gsm8k", agent="fern",
        name=name, group=group,
        notes="PR644 Reading-A panel: Option-B MMLU-Pro + GSM8K %-of-base vs ubel #628 gb6144 base.",
        tags=["pr644", "option-b", "reading-a", "mmlu-pro", "gsm8k", "pct-of-base",
              "specdec", "int4", "0p22", "gb6144", "consolidation"],
        config=cfg,
    )
    if run is None:
        print("[pr644] wandb not configured — skipping", flush=True)
        return None
    metrics: dict[str, Any] = {}
    for key, c in result["cells"].items():
        for mk in ("acc", "n", "n_correct", "wilson_lo", "wilson_hi",
                   "pct_of_base", "pct_of_base_lo", "pct_of_base_hi",
                   "extract_fail_rate", "truncation_rate", "length_stop_rate",
                   "strict_rate", "wall_s"):
            v = c.get(mk)
            if isinstance(v, (int, float)):
                metrics[f"{key}/{mk}"] = v
    for task, s in result["summary"].items():
        metrics[f"verdict/{task}_label"] = 1.0  # placeholder numeric; label in summary
        metrics[f"{task}/greedy_pct_of_base"] = s["greedy_pct_of_base"]
        if s["sampled_pct_of_base"] is not None:
            metrics[f"{task}/sampled_pct_of_base"] = s["sampled_pct_of_base"]
    metrics["analysis_only"] = 1
    metrics["official_tps"] = 0
    wl.log_event(run, "panel_complete", step=0, metrics=metrics)
    for k, v in metrics.items():
        run.summary[k] = v
    for task, s in result["summary"].items():
        run.summary[f"{task}_reading_a_label"] = s["reading_a_label"]
        run.summary[f"{task}_greedy_acc"] = s["greedy_acc"]
        run.summary[f"{task}_greedy_pct_of_base"] = s["greedy_pct_of_base"]
    run.summary["analysis_only"] = True
    run.summary["official_tps"] = 0
    wl.log_json_artifact(run, name="pr644_reading_a_panel",
                         artifact_type="quality-panel", data=result)
    rid = run.id
    wl.finish_wandb(run)
    print(f"[pr644] wandb logged run_id={rid}", flush=True)
    return rid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb_group", default="optionb-quality-mmlu-gsm8k-fern")
    ap.add_argument("--wandb_name", default="fern/optionb-quality-mmlu-gsm8k")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", default=str(RES / "pr644_consolidated.json"))
    args = ap.parse_args()

    result = build()
    missing = [f"{t}_{d}" for (t, d) in FILES if f"{t}_{d}" not in result["cells"]]
    result["missing_cells"] = missing

    rid = None
    if not args.no_wandb:
        rid = log_wandb(result, args.wandb_group, args.wandb_name)
    result["wandb_run_id"] = rid

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    # human-readable
    print("\n==================== PR #644 READING-A PANEL ====================")
    print(f"stack: {result['stack']}")
    print(f"missing cells: {missing or 'none'}")
    for task in ("mmlu_pro", "gsm8k"):
        s = result["summary"].get(task)
        if not s:
            print(f"\n[{task}] NO DATA")
            continue
        print(f"\n[{task}]  base={s['base_acc']} (run {s['base_run']})  "
              f"floor={s['floor']}  bar90={s['bar90']}")
        print(f"  greedy : acc={s['greedy_acc']:.4f} CI[{s['greedy_ci'][0]:.4f},{s['greedy_ci'][1]:.4f}]  "
              f"pct_of_base={s['greedy_pct_of_base']:.4f} "
              f"CI[{s['greedy_pct_of_base_ci'][0]:.4f},{s['greedy_pct_of_base_ci'][1]:.4f}]")
        if s["sampled_acc"] is not None:
            print(f"  sampled: acc={s['sampled_acc']:.4f} CI[{s['sampled_ci'][0]:.4f},{s['sampled_ci'][1]:.4f}]  "
                  f"pct_of_base={s['sampled_pct_of_base']:.4f}")
        print(f"  >>> {s['reading_a_label']}  (floor-gate: {s['verdict_floor']})")
    print(f"\nwandb_run_id: {rid}")
    print(f"out: {args.out}")
    print("================================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
