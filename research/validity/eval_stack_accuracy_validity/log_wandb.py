#!/usr/bin/env python3
"""PR #615 -- log the eval-stack accuracy-validity head-to-head to W&B.

Reads the two per-stack summaries (_summary_v0220.json, _summary_dev307.json) produced
by summarize_stack.py, computes per-eval stack deltas (0220 - dev307), and logs a single
analysis-only run (group eval-stack-accuracy-validity, official_tps=0, analysis_only=True).

Run under the repo .venv (has wandb): .venv/bin/python log_wandb.py --s547 <harness_bug|model_collapse>
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

HERE = Path("research/validity/eval_stack_accuracy_validity/runs")


def load(stack):
    p = HERE / f"_summary_{stack}.json"
    return json.load(open(p)) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="eval-stack-accuracy-validity")
    ap.add_argument("--name", default="lawine/eval-stack-accuracy-validity")
    ap.add_argument("--s547", default="harness_bug", choices=["harness_bug", "model_collapse"])
    ap.add_argument("--verdict", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    v0220 = load("v0220")
    dev307 = load("dev307")

    metrics = {}
    deltas = {}
    for task in ("gpqa", "mmlu", "gsm8k"):
        a = v0220.get(task, {})
        b = dev307.get(task, {})
        for k in ("mean_accuracy", "ci_lb_95", "ci_ub_95", "std_accuracy",
                  "finish_finish_length_rate", "mean_clears_bar", "ci_lb_clears_bar",
                  "n_questions", "n_seeds", "bar"):
            if k in a:
                metrics[f"{task}/0220/{k}"] = a[k]
            if k in b:
                metrics[f"{task}/dev307/{k}"] = b[k]
        if "mean_accuracy" in a and "mean_accuracy" in b:
            d = a["mean_accuracy"] - b["mean_accuracy"]
            metrics[f"{task}/stack_delta_0220_minus_dev307"] = d
            deltas[task] = d
        if "per_seed_accuracy" in a:
            metrics[f"{task}/0220/per_seed"] = a["per_seed_accuracy"]
        if "per_seed_accuracy" in b:
            metrics[f"{task}/dev307/per_seed"] = b["per_seed_accuracy"]

    print("[log_wandb] metrics to log:")
    for k in sorted(metrics):
        print(f"   {k} = {metrics[k]}")
    print(f"[log_wandb] deltas (0220-dev307): {deltas}")

    if args.dry_run:
        return 0

    os.environ.setdefault("WANDB_SILENT", "true")
    import wandb
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        group=args.group,
        name=args.name,
        config={
            "config_under_test": "int4_g128_lmhead (shipped submission)",
            "stacks": "vLLM 0.22.0 (submission pin) vs 0.22.1rc1.dev307",
            "serve_recipe": "canonical serve.py + --max-model-len 6144 (#598), identical flags both stacks",
            "decode_protocol": "lewtun#31 sampling T=1.0 top_p=0.95 top_k=64, min_tokens=8 (#541), max_tokens=4096 CoT / 1024 gsm8k",
            "gpqa_bar": 0.471, "mmlu_bar": 0.605, "gsm8k_bar": 0.807,
            "s547_failure_mode": args.s547,
            "eval_stack_verdict": args.verdict,
            "analysis_only": True,
            "official_tps": 0,
            "ppl_0220": 2.0188, "ppl_dev307": 2.6264, "ppl_dev307_inflation_pct": 30.1,
        },
    )
    wandb.log(metrics)
    wandb.summary.update({k: v for k, v in metrics.items() if not isinstance(v, list)})
    print(f"\n[wandb] logged run {run.id} (group={args.group})")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
