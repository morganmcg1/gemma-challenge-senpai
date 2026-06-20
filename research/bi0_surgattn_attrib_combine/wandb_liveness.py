"""Liveness W&B run for PR #794 (surgattn split-KV combine attribution).

Inits a run in the challenge project/group, logs the mechanistic findings as
config, and finishes. Proves the pod is live and W&B logging works before deep
work. Subsequent experiment runs (microbench, greedy-compare) are separate runs
in the same group.
"""
import os

import wandb

run = wandb.init(
    entity="wandb-applied-ai-team",
    project="gemma-challenge-senpai",
    group="bi0-surgattn-attrib",
    name="stark/surgattn-combine-liveness",
    job_type="liveness",
    config={
        "pr": 794,
        "base_submission": "int4_mtp_bi0_surgattn",
        "vllm": "0.22.0",
        "finding_1_partials_dtype": "float32 (config-a no-op)",
        "finding_2_divergence_source": "reduction reassociation (cross-segment combine + TILE 32-vs-16)",
        "bi0_control_tps": 218.02,
        "wirbel_785_3d_tps": 224.55,
        "wirbel_785_tok_divergence": 0.0176,
    },
)
print(f"[liveness] W&B run: {run.url}  id={run.id}")
wandb.log({"liveness/ok": 1})
wandb.finish()
print("[liveness] done")
