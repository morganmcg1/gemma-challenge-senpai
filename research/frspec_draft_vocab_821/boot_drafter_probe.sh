#!/usr/bin/env bash
# Runtime confirmation boot for PR #821: serve int4_mtp_bi0_int4head's serve.py on
# the cached base w4a16-ct + the gemma4_assistant MTP drafter, to capture the
# drafter's centroid-masking log lines. The verifier-head quant is irrelevant to
# the drafter mechanism, so we serve the plain cached base (no private 401).
set -u
cd /workspace/senpai/target

SERVE_PY=/tmp/senpai-venvs/20f658587e8a6643/bin/python
SUB=submissions/int4_mtp_bi0_int4head
BASE=/senpai-run/home/student-lawine/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0

export MODEL_ID="$BASE"
export DRAFTER_MODEL="google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant"
export NUM_SPECULATIVE_TOKENS=6
export MAX_MODEL_LEN=4096
export GPU_MEMORY_UTILIZATION=0.90
export MAX_NUM_BATCHED_TOKENS=512
export MAX_NUM_SEQS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_BATCH_INVARIANT=0
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1
export HOST=127.0.0.1
export PORT=8000
export SERVED_MODEL_NAME=gemma-4-e4b-it

exec "$SERVE_PY" "$SUB/serve.py"
