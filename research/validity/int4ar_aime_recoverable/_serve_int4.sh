#!/usr/bin/env bash
# int4-AR body server matching the #650 config (PR #668 reproduce block).
# Batch-invariant + flashinfer-sampler-off so greedy argmax is cross-session
# bit-exact (Marlin int4). Same venv/flags as the live bf16 server, model swapped.
set -euo pipefail
LOG="${1:-research/validity/int4ar_aime_recoverable/_serve_int4.log}"
# The container exposes the assigned A10G at index 0 (UUID GPU-51106f2c...); the
# shell inherits a stale CUDA_VISIBLE_DEVICES=4 that points at nothing. Pin 0.
export CUDA_VISIBLE_DEVICES=0
export VLLM_BATCH_INVARIANT=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_SEED=0
export MAX_NUM_BATCHED_TOKENS=2048
exec /tmp/vllm0220-srv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /workspace/gemma_build/int4_g128_lmhead \
  --served-model-name gemma-4-e4b-it \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 16 \
  --seed 0 \
  --trust-remote-code \
  --no-enable-log-requests \
  > "$LOG" 2>&1
