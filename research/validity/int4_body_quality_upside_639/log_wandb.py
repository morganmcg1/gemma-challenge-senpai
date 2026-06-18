#!/usr/bin/env python
"""PR #639 (lawine) -- log an int4-body-quality-upside GPQA arm to W&B.

Logs one run per arm (official_g32 / ours_g32 / ours_g128_mse) under group
`awq-int4-body-quality-upside`. LOCAL ONLY: analysis_only=True, official_tps=0,
NO HF Job, NO submission. The binding metric is GPQA-D sampled (T=1/top_p0.95/
top_k64) pooled n=1980 @ gb6144, apples-to-apples with bf16 base 0.5404 and
Option-B int4+spec 0.4652.

Run under the repo .venv (has wandb), NOT the serve venv:
  ./.venv/bin/python research/validity/int4_body_quality_upside_639/log_wandb.py <arm> <recipe_desc>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import wandb

GROUP = "awq-int4-body-quality-upside"
HERE = Path(__file__).resolve().parent
ENTITY = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
PROJECT = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")

COMMON = {
    "vllm_version": "0.22.0",
    "engine": "manifest-pinned 0.22.0 (/tmp/vllm0220-srv)",
    "vllm_batch_invariant": 1,
    "max_model_len": 8192,
    "max_num_seqs": 16,
    "max_num_batched_tokens": 2048,
    "gpu_memory_utilization": 0.90,
    "min_tokens": 8,
    "max_tokens": 6144,
    "gpqa_sampling": "T=1.0, top_p=0.95, top_k=64",
    "gpqa_seeds": "10 (12345..13579), matched to fern #629 / ubel #628",
    "use_flashinfer_sampler": 0,
    "serve_path": "submissions/bf16_base_aime/serve.py (MODEL_ID repoint, --dtype bfloat16)",
    "analysis_only": True,
    "official_tps": 0,
    "anchor_bf16_base": 0.5404,
    "anchor_optionb_int4_spec": 0.4652,
    "anchor_recalibrated_0p9_bar": 0.4864,
}


def main() -> int:
    arm = sys.argv[1]
    recipe = sys.argv[2] if len(sys.argv) > 2 else arm
    pooled = json.loads((HERE / "results" / arm / "pooled.json").read_text())

    run = wandb.init(
        project=PROJECT, entity=ENTITY, name=f"lawine/{arm}",
        group=GROUP, job_type="quality-int4-recipe", reinit=True,
        config={**COMMON, "arm": arm, "recipe": recipe,
                "model_id": pooled.get("model_id", recipe),
                "n_seeds": len(pooled.get("seeds", []))},
    )

    acc = pooled["pooled_accuracy"]
    log = {
        "gpqa_sampled/accuracy": acc,
        "gpqa_sampled/n_correct": pooled["n_correct"],
        "gpqa_sampled/n_scored": pooled["n_scored"],
        "gpqa_sampled/stderr": pooled["stderr"],
        "gpqa_sampled/ci95_lo_wilson": pooled["ci95_wilson"][0],
        "gpqa_sampled/ci95_hi_wilson": pooled["ci95_wilson"][1],
        "gpqa_sampled/finish_length_rate": pooled["pooled_finish_length_rate"],
        "gpqa_sampled/n_request_error": pooled["n_request_error"],
        "gpqa_sampled/accuracy_deconf": pooled["accuracy_excl_request_error"],
        "gpqa_sampled/n_deconf": pooled["n_scored_excl_request_error"],
        "gpqa_sampled/ci95_lo_wilson_deconf": pooled["ci95_wilson_excl_request_error"][0],
        "gpqa_sampled/ci95_hi_wilson_deconf": pooled["ci95_wilson_excl_request_error"][1],
        "vs/optionb_int4_spec": pooled["vs_optionb"],
        "vs/bf16_base": pooled["vs_bf16_base"],
        "vs/pct_of_bf16_base": pooled["pct_of_bf16_base"],
        "vs/clears_recalibrated_bar": int(pooled["clears_recalibrated_bar"]),
    }
    tbl = wandb.Table(columns=["seed", "accuracy", "n_correct", "n_scored",
                               "finish_length_rate", "ctok_mean"])
    for r in pooled["per_seed"]:
        tbl.add_data(str(r["seed"]), r["accuracy"], r["n_correct"], r["n_scored"],
                     r.get("finish_length_rate"), r.get("completion_tokens_mean"))
    log["gpqa_sampled/per_seed"] = tbl
    wandb.log(log)

    run.summary["arm"] = arm
    run.summary["recipe"] = recipe
    run.summary["gpqa_sampled_accuracy"] = acc
    run.summary["gpqa_sampled_finish_length_rate"] = pooled["pooled_finish_length_rate"]
    run.summary["gpqa_sampled_accuracy_deconf"] = pooled["accuracy_excl_request_error"]
    run.summary["pct_of_bf16_base"] = pooled["pct_of_bf16_base"]
    run.summary["vs_optionb"] = pooled["vs_optionb"]
    run.summary["clears_recalibrated_bar"] = pooled["clears_recalibrated_bar"]
    run.summary["surface_to_human"] = True

    rid = run.id
    run.finish()
    print(f"[wandb] logged arm={arm} acc={acc:.4f} fl={pooled['pooled_finish_length_rate']:.4f} id={rid}")
    print(f"[wandb] group={GROUP} project={ENTITY}/{PROJECT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
