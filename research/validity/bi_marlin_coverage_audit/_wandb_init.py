#!/usr/bin/env python
"""Liveness init for PR #617 BI-coverage audit. Logs config + in-progress marker,
persists run id to run_id.txt so the full-findings pass can resume the same run."""
import os, json, pathlib

import wandb

HERE = pathlib.Path(__file__).resolve().parent

config = {
    "analysis_only": True,
    "official_tps": 0,
    "no_hf_job": True,
    "no_build": True,
    "no_served_file_change": True,
    "vllm_version": "0.22.0",
    "submission_under_audit": "int4_mtp_batchinv",
    "pr": 617,
    "question": "Does VLLM_BATCH_INVARIANT=1 cover the int4-Marlin verify GEMM on 0.22.0?",
    "anchor_wirbel_607_run": "yuvztndu",
    "anchor_stark_613_run": "eqvdyntw",
}

run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
    entity=os.environ.get("WANDB_ENTITY"),
    group="bi-marlin-coverage-audit",
    name="stark/bi-marlin-coverage-audit",
    job_type="analysis",
    config=config,
)
wandb.log({"audit_phase": 0, "audit_started": 1})
(HERE / "run_id.txt").write_text(run.id + "\n")
print(f"[bi-audit] wandb run id = {run.id}")
print(f"[bi-audit] url = {run.url}")
wandb.finish()
