"""PR #789 — log the drafter CUDA-graph capture-state verdict to W&B.

LOCAL-ONLY null finding (code-inspection + behavioral dispatch/timing probe).
No training, no HF Job. One summary run in group `bi0-drafter-cudagraph`.

Run with an ephemeral wandb env:
    uv run --with wandb python research/bi0_drafter_cudagraph/log_wandb.py
"""
from __future__ import annotations

import os

import wandb

run = wandb.init(
    entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
    project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
    group="bi0-drafter-cudagraph",
    name="stark/cgprobe-control-verdict",
    job_type="diagnostic-probe",
    config={
        "pr": 789,
        "base_submission": "int4_mtp_bi0_surgattn",
        "base_wandb_run": "s63tb03x",
        "base_official_tps": 218.02,
        "base_ppl": 2.0058,
        "vllm_version": "0.22.0",
        "num_speculative_tokens": 6,
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes_effective": [7],
        "max_cudagraph_capture_size": 7,
        "uniform_decode_query_len": 7,
        "probe": "CGPROBE=1 dispatch-mode + cpu/gpu-event timing",
        "decode_tokens": 1536,
        "decode_prompts": 8,
        "axis": "DRAFTER proposer M=1 passes (vLLM-native capture)",
        "not_duplicated": "verifier=lawine#787; loopgraph=kanna#771",
    },
)

# Decisive dispatch-mode census (steady-state, n=1000 dispatches).
run.summary["dispatch_n"] = 1000
run.summary["dispatch_piecewise"] = 993
run.summary["dispatch_none"] = 7
run.summary["dispatch_piecewise_frac"] = 993 / 1000
run.summary["decode_path_piecewise_frac"] = 1.0  # all 7 NONE are >>7 prefill shapes
run.summary["first_pass_7to7_piecewise"] = 493
run.summary["loop_pass_1to7_piecewise"] = 500

# Timing (steady-state p50, ms).
run.summary["draft_cpu_ms_p50"] = 5.596
run.summary["draft_gpu_ms_p50"] = 2.482
run.summary["draft_cpu_over_gpu"] = 5.596 / 2.482
run.summary["exec_cpu_ms_p50"] = 7.631
run.summary["exec_gpu_ms_p50"] = 11.874

# Verdict.
run.summary["capture_verdict"] = "PIECEWISE_CAPTURED_NOT_EAGER"
run.summary["step2_config_capture_premise"] = "FALSE (no eager GEMM slice)"
run.summary["full_loop_capture_exposed"] = False
run.summary["null_kind"] = "config-capture-not-exposed (already PIECEWISE)"
run.summary["remaining_lever"] = "FULL/whole-loop capture = kanna#771 LOOPGRAPH"
run.summary["fire_worthy"] = False

print(f"[wandb] logged run {run.id} ({run.name}) group=bi0-drafter-cudagraph")
run.finish()
