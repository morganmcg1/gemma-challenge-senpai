#!/usr/bin/env python
"""Log the QuantSpec drafter-KV premise-check verdict to W&B (PR #121).

Pure CPU code-inspection of the deployed MTP serving stack
(submissions/int4_mtp_batchinv + pinned vllm==0.22.0). No model load, no GPU,
no HF launch. The premise-check answers a single decisive question: does the
Gemma4 MTP drafter keep a SEPARATE KV cache (QuantSpec INT4 drafter-KV lever
live) or SHARE the verify/target model's KV (lever moot)?

Verdict: SHARED. The drafter allocates ZERO KV cache of its own -> the lever
does not apply to our architecture (RED / CLOSE). See FINDINGS.md for the full
code trace.
"""
from __future__ import annotations

import wandb

config = {
    "method": "quantspec_drafter_kv_premise_check",
    "lane": "quantspec-drafter-kv",
    "analysis_type": "cpu-code-inspection",
    "gpu_used": False,
    "hf_launch": False,
    "deployed_submission": "submissions/int4_mtp_batchinv",
    "target_model": "google/gemma-4-E4B-it-qat-w4a16-ct",
    "drafter_model": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
    "drafter_vllm_class": "Gemma4MTP (gemma4_mtp)",
    "vllm": "0.22.0",
    "transformers": "5.9.0",
    "num_speculative_tokens": 6,
    # Frontier context (PR #121 baseline)
    "baseline_official_tps": 481.53,
    "baseline_local_k7_tps": 454.338,
    "drafter_decode_slice_pct": 7.0,  # denken #75/#77, BW-bound
}

# Primary metric + premise test + the code evidence pinning the verdict.
summary = {
    # TEST (the premise): is there a separate, quantizable drafter KV cache?
    "drafter_kv_separate_bool": 0,  # False -> SHARED with target
    # PRIMARY: net wall_tps % from the QuantSpec INT4 drafter-KV lever.
    # 0.0 -> lever does not apply (no separate drafter-KV to quantize).
    "quantspec_drafter_kv_net_wall_tps_pct": 0.0,
    # Sizing of the (non-existent) separate drafter KV.
    "drafter_kv_bytes_per_step_bf16": 0,
    "drafter_kv_int4_saving_pct_of_drafter_slice": 0.0,
    # Verdict.
    "gate": "RED",  # RED / moot -> CLOSE the QuantSpec drafter-KV lane
    "verdict": "SHARED_KV_LEVER_MOOT",
    # Code evidence (file:line) that pins SHARED.
    "ev_mtp_docstring": "gemma4_mtp.py:5-8 (shares KV with target; Q-only, no K/V proj)",
    "ev_no_kv_proj": "gemma4_mtp.py:148-232 (Gemma4MTPAttention builds q_proj/o_proj/q_norm only)",
    "ev_dummy_kv": "gemma4_mtp.py:248-257 (kv_dummy zeros; reads target cache via KV sharing)",
    "ev_cross_model_share": "v1/spec_decode/gemma4.py:329 (sets kv_sharing_target_layer_name -> target layer)",
    "ev_zero_alloc": "v1/worker/gpu_model_runner.py:7304-7316 (kv_sharing_target_layer_name -> no KVCacheSpec, 0 bytes)",
}

run = wandb.init(
    project="gemma-challenge-senpai",
    entity="wandb-applied-ai-team",
    name="stark/quantspec-drafter-kv-premise",
    group="quantspec-drafter-kv-premise",
    job_type="cpu-code-inspection",
    config=config,
    notes=(
        "PR #121 premise-check (CPU code inspection, no GPU/HF). Gemma4 MTP "
        "drafter SHARES the target model's KV cache (cross-model KV sharing); "
        "it allocates ZERO KV of its own (Q-only attention, dummy K/V, "
        "kv_sharing_target_layer_name -> no KVCacheSpec). drafter_kv_separate=False "
        "-> QuantSpec INT4 drafter-KV lever is MOOT -> RED / CLOSE. Full trace in "
        "research/quantspec_drafter_kv/FINDINGS.md."
    ),
)
wandb.log(summary)
for k, v in summary.items():
    run.summary[k] = v
print("WANDB_RUN_ID=" + run.id)
print("WANDB_RUN_URL=" + run.url)
run.finish()
