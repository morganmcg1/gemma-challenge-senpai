#!/usr/bin/env bash
# PR #759: launch the int4_mtp_batchinv FIRE served path locally on the A10G with
# the vLLM torch profiler enabled, so /start_profile + /stop_profile dump a
# per-kernel Kineto device trace of a steady-state decode window.
#
# Faithfulness: this is NOT a submission edit. It reproduces submissions/
# int4_mtp_batchinv/serve.py's exact api_server argv + env + PYTHONPATH (so the
# submission's sitecustomize.py attention-group backport and BI kernels load
# byte-for-byte), and ONLY appends --profiler-config. The profiler hooks do not
# change kernels, quant, cudagraph capture, or the BI swaps; they only attach
# CUPTI tracing between /start_profile and /stop_profile. The served numerics are
# identical to the fire config.
#
# Usage: launch_prof_server.sh <BI:0|1> <PORT> <TRACE_DIR>
#   BI=1 -> VLLM_BATCH_INVARIANT=1 (as-fired, deterministic kernels)
#   BI=0 -> VLLM_BATCH_INVARIANT=0 (naive/fast)
# Spec is ALWAYS ON (REFMODE=0, NUM_SPECULATIVE_TOKENS=6): this decomposes the
# 229.85 (BI0) vs 156.95 (BI1) spec-ON gap from #750.
set -euo pipefail
BI="${1:?need BI}"; PORT="${2:-8000}"; TRACE_DIR="${3:?need absolute trace dir}"
mkdir -p "$TRACE_DIR"
TRACE_DIR="$(cd "$TRACE_DIR" && pwd)"   # absolutize (ProfilerConfig requires abs)

VENV=/tmp/senpai-venvs/20f658587e8a6643
SUB=/workspace/senpai/target/submissions/int4_mtp_batchinv

# ---- Fire config (manifest env), verbatim -----------------------------------
export VLLM_BATCH_INVARIANT="$BI"
export SENPAI_REFERENCE_MODE=0          # spec ON (the leaderboard serving path)
export DRAFTER_MODEL=google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant
export NUM_SPECULATIVE_TOKENS=6
export MAX_MODEL_LEN=4096
export GPU_MEMORY_UTILIZATION=0.90
export MAX_NUM_BATCHED_TOKENS=512
export MAX_NUM_SEQS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct
export SERVED_MODEL_NAME=gemma-4-e4b-it
export HOST=0.0.0.0
export PORT="$PORT"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0

# Submission dir on PYTHONPATH so its sitecustomize.py (attention-group num_heads
# backport, required for the {8,4} draft/target group assertion on 0.22.0) loads
# in every server/EngineCore/worker process, exactly as serve.py arranges. We do
# NOT set SENPAI_PR755_* so the PR #755 hooks in sitecustomize stay inert no-ops.
export PYTHONPATH="$SUB${PYTHONPATH:+:$PYTHONPATH}"

# ---- flashinfer JIT toolkit fixups (verbatim from #750 launch_server.sh) -----
# BI=1 forces a flashinfer sampling-kernel rebuild; point it at the self-consistent
# pip cu13 toolkit (has nvcc + curand.h) and supply an unversioned libcudart.so.
export CUDA_HOME="${CUDA_HOME:-$VENV/lib/python3.12/site-packages/nvidia/cu13}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
_CU13_LIB="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
mkdir -p /tmp/fern_cudalibs
[ -e /tmp/fern_cudalibs/libcudart.so ] || ln -sf "$_CU13_LIB/libcudart.so.13" /tmp/fern_cudalibs/libcudart.so
export LIBRARY_PATH="/tmp/fern_cudalibs:$_CU13_LIB:${LIBRARY_PATH:-}"

# ---- profiler config ---------------------------------------------------------
# torch profiler, device traces to TRACE_DIR. record_shapes=true so matmul events
# carry [M,N,K] (lets us split lm_head's N=vocab GEMM from body GEMMs). with_stack
# off (we attribute by kernel name on the device timeline, not CPU stacks) to keep
# traces small + overhead low. ignore_frontend so AsyncLLM frontend tracing is off.
# Default schedule (warmup=0) records ALL engine iterations between /start_profile
# and /stop_profile, so we bound the window via the driver instead.
PROF_JSON=$(cat <<JSON
{"profiler":"torch","torch_profiler_dir":"${TRACE_DIR}","torch_profiler_with_stack":false,"torch_profiler_record_shapes":true,"torch_profiler_use_gzip":true,"ignore_frontend":true}
JSON
)

# ---- api_server argv (serve.py's exact list) + profiler ----------------------
exec "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --trust-remote-code \
  --no-enable-log-requests \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --speculative-config "{\"model\":\"${DRAFTER_MODEL}\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS}}" \
  --profiler-config "$PROF_JSON"
