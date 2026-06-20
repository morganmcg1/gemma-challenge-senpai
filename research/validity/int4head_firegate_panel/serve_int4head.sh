#!/usr/bin/env bash
# Serve submissions/int4_mtp_bi0_int4head (int4 W4A16 g32 body + int4 g32 lm_head,
# MTP K=6, VLLM_BATCH_INVARIANT=0, surgattn force-2D) for the #795 fire-prep
# quality panel. Mirrors the bi0 reference serving (#762 run_panel.py /
# #773 GPQA panel): MAX_NUM_SEQS=16 for eval tractability,
# VLLM_USE_FLASHINFER_SAMPLER=0 (this box ships no curand.h for the flashinfer
# JIT sampler), CUDA_VISIBLE_DEVICES=0 (container-local A10G; host inherits 4).
#
# MAX_MODEL_LEN=12288 (NOT the deployed 4096): the GPQA axis runs max_tokens=6144,
# and the bi0 reference reached completion_tokens_max=6144 with n_stop_model_length=0
# (zero context truncation). 12288 admits the longest GPQA item (~2418 prompt) +
# full 6144 generation = 8562 < 12288 with margin, so int4head is truncation-equivalent
# to the bi0 reference. MMLU (<=4096 gen) and AIME (3072 gen) fit trivially.
# LOCAL ONLY -- no HF Job.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SUB=/workspace/senpai/target/submissions/int4_mtp_bi0_int4head
SERVE_PY=/tmp/senpai-venvs/20f658587e8a6643/bin/python   # vllm==0.22.0 (matches manifest)
LOG="$HERE/server_int4head.log"

cd "$SUB"
MODEL_ID=/workspace/gemma_build/bi0_int4head_g32 \
SERVED_MODEL_NAME=gemma-4-e4b-it \
PORT=8020 \
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
echo "[serve] launched pid=$(cat "$HERE/_server.pid") log=$LOG port=8020"
