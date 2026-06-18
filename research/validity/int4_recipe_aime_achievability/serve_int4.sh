#!/usr/bin/env bash
# int4-AR body server, EXACT #672 / PR #668 config (batch-invariant, flashinfer
# sampler off, seed 0) so greedy argmax matches the #672 band protocol. Only the
# --model path changes across group-size arms.
set -euo pipefail
MODEL="${1:?model dir}"
LOG="${2:?log path}"
PORT="${3:-8000}"
# The container exposes the assigned A10G at index 0; the shell inherits a stale
# CUDA_VISIBLE_DEVICES that points at nothing. Pin 0.
export CUDA_VISIBLE_DEVICES=0
export VLLM_BATCH_INVARIANT=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_SEED=0
export MAX_NUM_BATCHED_TOKENS=2048
exec /tmp/vllm0220-srv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name gemma-4-e4b-it \
  --host 127.0.0.1 --port "$PORT" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 16 \
  --seed 0 \
  --trust-remote-code \
  --no-enable-log-requests \
  > "$LOG" 2>&1
