#!/usr/bin/env python
"""Log the int4_g128_lmhead + MTP-K7 spec-config 4-eval quality panel to W&B.

PR #605 (stark). ANALYSIS-ONLY decision evidence for the human's #481 A/B steer:
does the option-B spec config (NOT #319-identical, ~427.7 official-proxy TPS) keep
>=90% of vanilla base on MMLU-Pro / GPQA-Diamond / AIME / GSM8K?

Logs one W&B run per eval (job_type=quality-eval) plus a panel-summary run
(job_type=panel-verdict) in group `int4-mtp-spec-quality-panel`. LOCAL ONLY,
analysis_only=True, official_tps=0 -- this card does not fire.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import wandb

GROUP = "int4-mtp-spec-quality-panel"
HERE = Path(__file__).resolve().parent
ENTITY = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
PROJECT = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")

# Morgan #579 re-anchored 90%-of-vanilla-base bars.
BARS = {"mmlu_pro": 0.605, "gpqa_diamond": 0.471, "aime": 0.090, "gsm8k": 0.807}
IMPLIED_BASE = {"mmlu_pro": 0.672, "gpqa_diamond": 0.523, "aime": 0.10, "gsm8k": 0.897}

SPEC_CONFIG = {
    "config_name": "int4_g128_lmhead + MTP-K7 spec (fern #597)",
    "model_id": "/workspace/gemma_build/int4_g128_lmhead",
    "drafter": "/tmp/qat-assistant",
    "num_speculative_tokens": 7,
    "vllm_batch_invariant": 1,
    "vllm_version": "0.22.0",
    "max_model_len": 6144,
    "min_tokens": 8,
    "sampling": "generation_config.json (T=1.0, top_p=0.95, top_k=64)",
    "freerun_seq_exact": 0.3125,
    "is_319_identical": False,
    "official_proxy_tps": 427.7,
    "analysis_only": True,
    "official_tps": 0,
}


def _common_config(extra: dict | None = None) -> dict:
    cfg = dict(SPEC_CONFIG)
    if extra:
        cfg.update(extra)
    return cfg


def log_eval(name: str, eval_key: str, accuracy: float, extra: dict) -> bool:
    bar = BARS[eval_key]
    passed = accuracy >= bar
    run = wandb.init(
        project=PROJECT, entity=ENTITY, name=name, group=GROUP,
        job_type="quality-eval", reinit=True,
        config=_common_config({"eval": eval_key, "bar_90pct": bar,
                               "implied_base": IMPLIED_BASE[eval_key], **extra}),
    )
    wandb.log({
        "accuracy": accuracy,
        "bar_90pct": bar,
        "margin_over_bar": accuracy - bar,
        "pct_of_base": (accuracy / IMPLIED_BASE[eval_key]) if IMPLIED_BASE[eval_key] else None,
        "pass": int(passed),
        **{k: v for k, v in extra.items() if isinstance(v, (int, float))},
    })
    run.summary["pass"] = passed
    run.summary["accuracy"] = accuracy
    run.finish()
    print(f"[wandb] {eval_key}: acc={accuracy:.4f} bar={bar} -> {'PASS' if passed else 'FAIL'}")
    return passed


def log_verdict(results: dict) -> None:
    passes = {k: results[k]["accuracy"] >= BARS[k] for k in BARS if k in results}
    all_pass = all(passes.values()) and len(passes) == 4
    verdict = "OPTION-B-FIREABLE" if all_pass else "OPTION-B-DEAD"
    run = wandb.init(
        project=PROJECT, entity=ENTITY, name="stark/spec-quality-panel-VERDICT",
        group=GROUP, job_type="panel-verdict", reinit=True,
        config=_common_config({"bars": BARS, "implied_base": IMPLIED_BASE}),
    )
    log = {"verdict_fireable": int(all_pass), "n_pass": sum(passes.values()),
           "n_eval": len(passes)}
    for k in BARS:
        if k in results:
            log[f"{k}_accuracy"] = results[k]["accuracy"]
            log[f"{k}_pass"] = int(passes[k])
    wandb.log(log)
    run.summary["verdict"] = verdict
    run.finish()
    print(f"\n[VERDICT] {verdict}  ({sum(passes.values())}/{len(passes)} pass)")


def liveness() -> None:
    run = wandb.init(
        project=PROJECT, entity=ENTITY, name="stark/spec-quality-panel-LIVENESS",
        group=GROUP, job_type="liveness", reinit=True,
        config=_common_config({"phase": "0-preflight",
                               "drafter_persists": True,
                               "body_rebuild_required": True,
                               "disk_free_gb_at_pickup": 235}),
    )
    wandb.log({"alive": 1})
    run.finish()
    print("[wandb] liveness run logged")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["liveness", "results"], default="results")
    ap.add_argument("--results-json", type=Path, default=HERE / "panel_results.json")
    args = ap.parse_args()
    if args.mode == "liveness":
        liveness()
        return 0
    data = json.loads(args.results_json.read_text())
    for key, name in [("mmlu_pro", "stark/spec-mmlu_pro"),
                      ("gpqa_diamond", "stark/spec-gpqa_diamond"),
                      ("aime", "stark/spec-aime"), ("gsm8k", "stark/spec-gsm8k")]:
        if key in data:
            r = data[key]
            log_eval(name, key, r["accuracy"], {k: v for k, v in r.items() if k != "accuracy"})
    log_verdict(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
