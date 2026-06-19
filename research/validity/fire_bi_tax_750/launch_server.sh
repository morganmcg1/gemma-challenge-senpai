#!/usr/bin/env bash
# Launch the int4_mtp_batchinv fire-config served api_server locally on the A10G.
# Mirrors official harness participant_env + serve.py launch path exactly, with
# only VLLM_BATCH_INVARIANT and SENPAI_REFERENCE_MODE varied per arm.
#
# Usage: launch_server.sh <BI:0|1> <REFMODE:0|1> <PORT>
#   BI=1            -> VLLM_BATCH_INVARIANT=1 (as-fired, batch-invariant kernels)
#   BI=0            -> VLLM_BATCH_INVARIANT=0 (naive)
#   REFMODE=1       -> SENPAI_REFERENCE_MODE=1 -> drafter OFF (M=1 AR greedy reference)
#   REFMODE=0       -> speculative decode ON (K=6, as deployed)
set -euo pipefail
BI="${1:?need BI}"; REF="${2:?need refmode}"; PORT="${3:-8000}"
VENV=/tmp/senpai-venvs/20f658587e8a6643
SUB=/workspace/senpai/target/submissions/int4_mtp_batchinv

# Fire config (manifest env), verbatim
export VLLM_BATCH_INVARIANT="$BI"
export SENPAI_REFERENCE_MODE="$REF"
export DRAFTER_MODEL=google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant
export NUM_SPECULATIVE_TOKENS=6
export MAX_MODEL_LEN=4096
export GPU_MEMORY_UTILIZATION=0.90
export MAX_NUM_BATCHED_TOKENS=512
export MAX_NUM_SEQS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Harness participant_env equivalents
export MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct
export SERVED_MODEL_NAME=gemma-4-e4b-it
export HOST=0.0.0.0
export PORT="$PORT"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0

# flashinfer JIT (sampling kernels) needs a self-consistent CUDA toolkit with
# curand.h. The bundled pip cu13 toolkit (nvidia/cu13) has BOTH bin/nvcc and the
# full include set (curand.h); /usr/local/cuda-13.2's include dir is missing
# curand.h. Point flashinfer at the pip toolkit so VLLM_BATCH_INVARIANT's kernel
# rebuild compiles (the .cu sources are newer than the prebuilt .o, so a rebuild
# is forced regardless). This matches the official vllm/vllm-openai image's
# bundled-toolkit layout.
export CUDA_HOME="${CUDA_HOME:-$VENV/lib/python3.12/site-packages/nvidia/cu13}"
# That pip toolkit ships nvcc 13.3 but cudart headers tagged 13.0; flashinfer's
# bundled CCCL hard-errors on the minor mismatch. CCCL exposes the documented
# escape hatch for exactly this (newer compiler than CTK headers); nvcc 13.3 is
# backward-compatible with the cudart-13.0 API the kernels call.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
# flashinfer's link step uses -lcudart and -L$CUDA_HOME/lib64, but the pip cu13
# layout ships only libcudart.so.13 (no unversioned symlink) under lib/. Expose an
# unversioned libcudart.so and put the real lib dir on LIBRARY_PATH so the c++
# driver link resolves -lcudart (-lcuda comes from the system driver lib).
_CU13_LIB="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
mkdir -p /tmp/fern_cudalibs
[ -e /tmp/fern_cudalibs/libcudart.so ] || ln -sf "$_CU13_LIB/libcudart.so.13" /tmp/fern_cudalibs/libcudart.so
export LIBRARY_PATH="/tmp/fern_cudalibs:$_CU13_LIB:${LIBRARY_PATH:-}"

cd "$SUB"
exec "$VENV/bin/python" serve.py
