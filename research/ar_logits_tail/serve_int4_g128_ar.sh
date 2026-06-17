#!/usr/bin/env bash
# PR #604 — AR logits→token tail profiling serve (LOCAL, no submission change).
# Plain stock vllm.entrypoints.openai.api_server replicating int4_g128_lmhead
# operative flags + MAX_NUM_SEQS=1 (card predicate). spec-OFF, temp set per-request.
set -euo pipefail

V=/tmp/senpai-venvs/20f658587e8a6643/bin/python
MODEL_ID=${MODEL_ID:-/workspace/gemma_build/int4_g128_lmhead}
PORT=${PORT:-8000}
LOG=${LOG:-research/ar_logits_tail/server_ar.log}

export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# Profiler dir is opt-in (set VLLM_TORCH_PROFILER_DIR before calling).

exec "$V" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --served-model-name gemma-4-e4b-it \
  --host 0.0.0.0 --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 1 \
  --trust-remote-code \
  --no-enable-log-requests \
  >"$LOG" 2>&1
