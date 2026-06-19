#!/usr/bin/env bash
# Launch the int4_qat SUBMISSION (plain int4 W4A16, NO spec, NO batch-invariant)
# as the official-anchor reference. int4_qat serves the SAME checkpoint as the
# fire config (google/gemma-4-E4B-it-qat-w4a16-ct) and has a known official
# a10g-small number: summary.json:tps = 95.463 (BASELINE.md, PR #4 / int4_qat PR #3).
# So R_int4 = 95.463 / local_int4_qat gives the pod->official ratio on the EXACT
# checkpoint the fire config uses (same Marlin int4 body GEMM = dominant decode
# cost), with which we anchor tps_BI1/tps_BI0 -> official.
#
# Usage: launch_int4qat.sh <PORT>
set -euo pipefail
PORT="${1:-8000}"
VENV=/tmp/senpai-venvs/20f658587e8a6643
SUB=/workspace/senpai/target/submissions/int4_qat

# int4_qat manifest env, verbatim (NO VLLM_BATCH_INVARIANT, NO spec, NO drafter)
export MAX_MODEL_LEN=4096
export GPU_MEMORY_UTILIZATION=0.90
export MAX_NUM_BATCHED_TOKENS=512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Harness participant_env equivalents (match launch_server.sh / official path)
export MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct
export SERVED_MODEL_NAME=gemma-4-e4b-it
export HOST=0.0.0.0
export PORT="$PORT"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0

# Same flashinfer/CUDA self-consistency fix as launch_server.sh (idempotent,
# harmless if no JIT rebuild is forced; protects against a sampling-kernel rebuild).
export CUDA_HOME="${CUDA_HOME:-$VENV/lib/python3.12/site-packages/nvidia/cu13}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
_CU13_LIB="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
mkdir -p /tmp/fern_cudalibs
[ -e /tmp/fern_cudalibs/libcudart.so ] || ln -sf "$_CU13_LIB/libcudart.so.13" /tmp/fern_cudalibs/libcudart.so
export LIBRARY_PATH="/tmp/fern_cudalibs:$_CU13_LIB:${LIBRARY_PATH:-}"

cd "$SUB"
exec "$VENV/bin/python" serve.py
