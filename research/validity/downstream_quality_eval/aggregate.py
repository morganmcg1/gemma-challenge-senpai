#!/usr/bin/env python3
"""Aggregate the base vs ship downstream-quality arms (PR #511).

Reads the four per-arm result JSONs produced by run_eval.py, verifies base and
ship saw BYTE-IDENTICAL prompts (per-question prompt_sha), computes the A/B delta
and per-question answer agreement (the downstream-task identity metric), checks
the Morgan #483 gate and whether base reproduces dixie's #483 anchors, logs a W&B
summary (group downstream-quality-eval, analysis_only), and prints the single-line
SENPAI-RESULT plus a markdown Results block.

Usage:
  aggregate.py --base-mmlu base_mmlu_pro.json --ship-mmlu ship_mmlu_pro.json \
               --base-gpqa base_gpqa.json     --ship-gpqa ship_gpqa.json \
               --out aggregate.json [--no-wandb]
"""
import argparse
import json
import math
import os
import sys

# dixie-flatline #483 anchors (inspect_evals / greedy / pinned vLLM wheel)
DIXIE_BASE_MMLU = 0.668
DIXIE_BASE_GPQA = 0.470
DIXIE_SUBSTRATE_MMLU = 0.330
DIXIE_SUBSTRATE_GPQA = 0.283
# Morgan #483 proposed gate thresholds
GATE_MMLU = 0.60
GATE_GPQA = 0.42


def _load(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _ci95(p, n):
    if not n:
        return (float("nan"), float("nan"))
    h = 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n)
    return (max(0.0, p - h), min(1.0, p + h))


def _align(base, ship):
    """Pair base/ship per-sample rows by id; check identical prompts."""
    if base is None:
        return None
    b = {r["id"]: r for r in base["per_sample"]}
    pair = {
        "n_base": len(b),
        "base_acc": base["accuracy"],
        "base_scored": base["n_scored"],
        "base_correct": base["n_correct"],
        "base_err": base.get("n_error", 0),
    }
    if ship is None:
        pair.update(ship_acc=None)
        return pair
    s = {r["id"]: r for r in ship["per_sample"]}
    common = sorted(set(b) & set(s))
    prompt_mismatch = [i for i in common if b[i].get("prompt_sha") != s[i].get("prompt_sha")]
    agree = sum(1 for i in common if b[i]["answer"] == s[i]["answer"])
    both_c = sum(1 for i in common if b[i]["correct"] and s[i]["correct"])
    both_w = sum(1 for i in common if not b[i]["correct"] and not s[i]["correct"])
    base_only = sum(1 for i in common if b[i]["correct"] and not s[i]["correct"])
    ship_only = sum(1 for i in common if not b[i]["correct"] and s[i]["correct"])
    pair.update(
        n_ship=len(s),
        n_common=len(common),
        prompt_identical=(len(prompt_mismatch) == 0),
        n_prompt_mismatch=len(prompt_mismatch),
        ship_acc=ship["accuracy"],
        ship_scored=ship["n_scored"],
        ship_correct=ship["n_correct"],
        ship_err=ship.get("n_error", 0),
        answer_agreement=(agree / len(common)) if common else float("nan"),
        n_answer_agree=agree,
        both_correct=both_c,
        both_wrong=both_w,
        base_only_correct=base_only,
        ship_only_correct=ship_only,
        delta_ship_minus_base=(ship["accuracy"] - base["accuracy"]),
    )
    return pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-mmlu", required=True)
    ap.add_argument("--ship-mmlu", default=None)
    ap.add_argument("--base-gpqa", required=True)
    ap.add_argument("--ship-gpqa", default=None)
    ap.add_argument("--out", default="aggregate.json")
    ap.add_argument("--wandb_name", default="ubel/downstream-quality-eval")
    ap.add_argument("--wandb_group", default="downstream-quality-eval")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    bm, sm = _load(a.base_mmlu), _load(a.ship_mmlu)
    bg, sg = _load(a.base_gpqa), _load(a.ship_gpqa)
    mmlu = _align(bm, sm)
    gpqa = _align(bg, sg)

    base_mmlu = mmlu["base_acc"]
    base_gpqa = gpqa["base_acc"]
    ship_mmlu = mmlu.get("ship_acc")
    ship_gpqa = gpqa.get("ship_acc")
    mmlu_n = mmlu["base_scored"]
    gpqa_n = gpqa["base_scored"]

    base_mmlu_ci = _ci95(base_mmlu, mmlu_n)
    base_gpqa_ci = _ci95(base_gpqa, gpqa_n)
    reproduces = (
        base_mmlu_ci[0] <= DIXIE_BASE_MMLU <= base_mmlu_ci[1]
        and base_gpqa_ci[0] <= DIXIE_BASE_GPQA <= base_gpqa_ci[1]
    )

    have_ship = ship_mmlu is not None and ship_gpqa is not None
    gate_pass = None
    mmlu_delta = gpqa_delta = None
    if have_ship:
        gate_pass = bool(ship_mmlu >= GATE_MMLU and ship_gpqa >= GATE_GPQA)
        mmlu_delta = ship_mmlu - base_mmlu
        gpqa_delta = ship_gpqa - base_gpqa

    # Verdict
    if not have_ship:
        verdict = (
            f"BASE-ONLY: base MMLU-Pro={base_mmlu:.4f} GPQA-Diamond={base_gpqa:.4f}; "
            f"reproduces_dixie_anchors={reproduces}. Ship arm pending."
        )
    else:
        near0 = abs(mmlu_delta) <= 0.02 and abs(gpqa_delta) <= 0.03
        if near0 and gate_pass:
            verdict = (
                f"MOAT CONFIRMED: served ship reproduces base downstream quality "
                f"(MMLU delta={mmlu_delta:+.4f}, GPQA delta={gpqa_delta:+.4f}; "
                f"answer-agreement mmlu={mmlu['answer_agreement']:.3f} "
                f"gpqa={gpqa['answer_agreement']:.3f}); gate PASS."
            )
        else:
            verdict = (
                f"MOAT REFUTED: served ship degrades vs base "
                f"(MMLU {base_mmlu:.4f}->{ship_mmlu:.4f} delta={mmlu_delta:+.4f}; "
                f"GPQA {base_gpqa:.4f}->{ship_gpqa:.4f} delta={gpqa_delta:+.4f}); "
                f"gate {'PASS' if gate_pass else 'FAIL'}."
            )

    report = {
        "base_mmlu_pro": base_mmlu,
        "base_gpqa_diamond": base_gpqa,
        "ship_mmlu_pro": ship_mmlu,
        "ship_gpqa_diamond": ship_gpqa,
        "mmlu_pro_delta_ship_minus_base": mmlu_delta,
        "gpqa_delta_ship_minus_base": gpqa_delta,
        "ship_passes_quality_gate": gate_pass,
        "reproduces_dixie_base_anchors": reproduces,
        "eval_subset_n": mmlu_n,
        "gpqa_n": gpqa_n,
        "base_mmlu_ci95": base_mmlu_ci,
        "base_gpqa_ci95": base_gpqa_ci,
        "dixie_anchor_mmlu": DIXIE_BASE_MMLU,
        "dixie_anchor_gpqa": DIXIE_BASE_GPQA,
        "dixie_substrate_mmlu": DIXIE_SUBSTRATE_MMLU,
        "dixie_substrate_gpqa": DIXIE_SUBSTRATE_GPQA,
        "gate_mmlu_threshold": GATE_MMLU,
        "gate_gpqa_threshold": GATE_GPQA,
        "mmlu_detail": mmlu,
        "gpqa_detail": gpqa,
        "verdict": verdict,
        "analysis_only": True,
        "no_hf_job": True,
        "no_served_file_change": True,
        "official_tps": 0,
    }
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)

    senpai_result = {
        "terminal": have_ship,
        "status": "complete" if have_ship else "partial",
        "pending_arms": not have_ship,
        "wandb_run_ids": [],
        "primary_metric": {"name": "mmlu_pro_delta_ship_minus_base", "value": mmlu_delta},
        "test_metric": {"name": "ship_gpqa_diamond", "value": ship_gpqa},
    }

    print("\n==== DOWNSTREAM QUALITY A/B ====")
    print(f"base   : MMLU-Pro={base_mmlu:.4f} (n={mmlu_n}, ci95={base_mmlu_ci[0]:.3f}-{base_mmlu_ci[1]:.3f})  "
          f"GPQA-Diamond={base_gpqa:.4f} (n={gpqa_n}, ci95={base_gpqa_ci[0]:.3f}-{base_gpqa_ci[1]:.3f})")
    if have_ship:
        print(f"ship   : MMLU-Pro={ship_mmlu:.4f}  GPQA-Diamond={ship_gpqa:.4f}")
        print(f"delta  : MMLU={mmlu_delta:+.4f}  GPQA={gpqa_delta:+.4f}")
        print(f"agree  : MMLU answer-agreement={mmlu['answer_agreement']:.4f} ({mmlu['n_answer_agree']}/{mmlu['n_common']})  "
              f"GPQA={gpqa['answer_agreement']:.4f} ({gpqa['n_answer_agree']}/{gpqa['n_common']})")
        print(f"prompts identical: mmlu={mmlu['prompt_identical']} gpqa={gpqa['prompt_identical']}")
        print(f"gate   : ship_passes_quality_gate={gate_pass} (MMLU>={GATE_MMLU} AND GPQA>={GATE_GPQA})")
    print(f"reproduces_dixie_base_anchors: {reproduces}")
    print(f"VERDICT: {verdict}")
    print("SENPAI-RESULT:", json.dumps(senpai_result))

    if not a.no_wandb:
        _log_wandb(report, a)
    return 0


def _log_wandb(report, a):
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; JSON saved, skipping wandb", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="ubel",
        name=a.wandb_name, group=a.wandb_group,
        notes="PR#511 downstream-quality gate: does the surgical-357 served ship "
              "(37L osoi5 + 12k pruned head) reproduce the 42L stock int4 base on "
              "MMLU-Pro + GPQA-Diamond (greedy, inspect_evals), or collapse like "
              "dixie #483's substrate?",
        config={
            "pr": 511, "analysis_only": True, "no_hf_job": True,
            "no_served_file_change": True, "official_tps": 0,
            "eval_subset_n": report["eval_subset_n"], "gpqa_n": report["gpqa_n"],
            "gate_mmlu_threshold": GATE_MMLU, "gate_gpqa_threshold": GATE_GPQA,
            "dixie_anchor_mmlu": DIXIE_BASE_MMLU, "dixie_anchor_gpqa": DIXIE_BASE_GPQA,
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return
    keys = [
        "base_mmlu_pro", "base_gpqa_diamond", "ship_mmlu_pro", "ship_gpqa_diamond",
        "mmlu_pro_delta_ship_minus_base", "gpqa_delta_ship_minus_base",
        "ship_passes_quality_gate", "reproduces_dixie_base_anchors", "eval_subset_n",
        "gpqa_n", "official_tps", "analysis_only",
    ]
    for k in keys:
        run.summary[k] = report[k]
    for task, d in (("mmlu", report["mmlu_detail"]), ("gpqa", report["gpqa_detail"])):
        for kk in ("answer_agreement", "n_answer_agree", "n_common", "prompt_identical",
                   "both_correct", "both_wrong", "base_only_correct", "ship_only_correct",
                   "delta_ship_minus_base", "base_err", "ship_err"):
            if d is not None and kk in d:
                run.summary[f"{task}/{kk}"] = d[kk]
    run.summary["verdict_text"] = report["verdict"]
    run.summary["base_mmlu_ci95_lo"], run.summary["base_mmlu_ci95_hi"] = report["base_mmlu_ci95"]
    run.summary["base_gpqa_ci95_lo"], run.summary["base_gpqa_ci95_hi"] = report["base_gpqa_ci95"]
    print(f"[wandb] logged run id={getattr(run,'id',None)}", flush=True)
    try:
        finish_wandb(run)
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
