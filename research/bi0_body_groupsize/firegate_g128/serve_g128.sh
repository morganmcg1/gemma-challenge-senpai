#!/usr/bin/env bash
# PR #814 Step-3: serve the g128 body-group-size arm through the int4head serve
# path (submissions/int4_mtp_bi0_int4head) for the fire-prep quality panel.
#
# g128 = int4 W4A16 body re-quantized at group_size=128 (from the QAT bf16
# weights) + int4 g32 lm_head + MTP K=6 drafter + surgattn force-2D, BI=0.
# The ONLY delta vs the int4head fire candidate (#795) is the body group_size
# (g32 -> g128). So this panel isolates whether body group-size coarsening (the
# +2.79% HBM-traffic TPS lever) holds quality within band.
#
# Mirrors ubel #795 serve_int4head.sh EXACTLY except MODEL_ID + PORT:
#   MAX_MODEL_LEN=12288 (GPQA runs max_tokens=6144; longest item ~2418 prompt +
#     6144 gen = 8562 < 12288 -> truncation-equivalent to the #795 reference),
#   MAX_NUM_SEQS=16 (eval tractability), VLLM_USE_FLASHINFER_SAMPLER=0 (no
#     curand.h for the flashinfer JIT sampler on this box),
#   CUDA_VISIBLE_DEVICES=0 (container-local A10G).
# LOCAL ONLY -- no HF Job.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SUB=/workspace/senpai/target/submissions/int4_mtp_bi0_int4head
SERVE_PY=/tmp/senpai-venvs/20f658587e8a6643/bin/python   # vllm==0.22.0 (matches manifest)
LOG="$HERE/server_g128.log"

cd "$SUB"
MODEL_ID=/workspace/gemma_build/wirbel_body_gs/g128 \
SERVED_MODEL_NAME=gemma-4-e4b-it \
PORT=8021 \
HOST=127.0.0.1 \
MAX_MODEL_LEN=12288 \
MAX_NUM_SEQS=16 \
MAX_NUM_BATCHED_TOKENS=512 \
GPU_MEMORY_UTILIZATION=0.90 \
VLLM_BATCH_INVARIANT=0 \
DRAFTER_MODEL=google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant \
NUM_SPECULATIVE_TOKENS=6 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_USE_FLASHINFER_SAMPLER=0 \
CUDA_VISIBLE_DEVICES=0 \
"$SERVE_PY" serve.py > "$LOG" 2>&1 &
echo "$!" > "$HERE/_server.pid"
echo "[serve] launched pid=$(cat "$HERE/_server.pid") log=$LOG port=8021"
