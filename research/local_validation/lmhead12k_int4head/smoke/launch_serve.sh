#!/usr/bin/env bash
set -uo pipefail
SMOKE=/workspace/senpai/target/research/local_validation/lmhead12k_int4head/smoke
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export MODEL_ID=/workspace/gemma_build/lmhead12k_int4head
export MAX_MODEL_LEN=4096
export GPU_MEMORY_UTILIZATION=0.90
export MAX_NUM_BATCHED_TOKENS=512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SENPAI_HEAD_BYTES_LOG="$SMOKE/served_head_bytes.json"
export HOST=127.0.0.1
export PORT=8000
export SERVED_MODEL_NAME=gemma-4-e4b-it
cd /workspace/senpai/target/submissions/lmhead12k_int4head
exec /tmp/server-venv/bin/python serve.py
