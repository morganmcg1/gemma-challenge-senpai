#!/usr/bin/env python
"""Liveness init for PR #621 spec-#319 trigger localization. Logs config +
in-progress marker, persists run id to run_id.txt so the findings pass can
resume the same run.

analysis_only / official_tps=0 / NO HF Job / NO build — CPU/GPU introspection on
vLLM 0.22.0, single A10G, mirroring the no-build mode of #617 / #613.
"""
import os
import pathlib

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
    "pr": 621,
    "question": (
        "Localize the real spec-#319 trigger: which NON-Marlin op in the "
        "spec-verify path is batch-variant (M=8 vs M=1, row-0 probe)? Is the "
        "VLLM_BATCH_INVARIANT attention guard taken for this submission's "
        "backend at verify width, or bypassed by a TRITON_ATTN / 3D split-KV "
        "redirect?"
    ),
    "prime_suspect": "attention_varlen_paged_kv_combine",
    "probe_method": "row0_bitexact_M8_vs_M1",
    "anchor_stark_617_run": "fa1f9vm1",
    "anchor_wirbel_607_run": "yuvztndu",
    "shipped_strict_319_anchor": "int4_g128_lmhead@126.378_official_tps",
}

run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
    entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
    group="spec-identity-trigger-localize",
    name="stark/spec-identity-trigger-localize",
    job_type="analysis",
    config=config,
)
wandb.log({"audit_phase": 0, "audit_started": 1})
(HERE / "run_id.txt").write_text(run.id + "\n")
print(f"[spec-trigger] wandb run id = {run.id}")
print(f"[spec-trigger] url = {run.url}")
wandb.finish()
