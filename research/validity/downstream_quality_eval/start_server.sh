#!/usr/bin/env bash
# Launch a vLLM OpenAI server for one downstream-quality eval arm and wait until ready.
#
# Usage: start_server.sh <arm:base|ship> <model_dir> [keepset_path]
#   base arm:  no keepset -> vanilla vLLM, full-vocab head.
#   ship arm:  keepset path -> pck04 sitecustomize inject rebuilds lm_head to K
#              rows + scatters to full vocab (-inf at non-kept). REQUIRED to load
#              the pruned osoi5-12k checkpoint (vanilla vLLM asserts otherwise).
#
# Canonical `python -m vllm.entrypoints.openai.api_server` launch (NOT a runpy
# wrapper): runpy with run_name="__main__" breaks multiprocessing-spawn's
# main-module detection in the EngineCore child. -m is spawn-safe.
set -euo pipefail

ARM="$1"
MODEL_DIR="$2"
KEEPSET="${3:-}"

HERE="$(cd "$(dirname "$0")" && pwd)"
SUB_DIR="/workspace/senpai/target/submissions/fa2sw_strict_surgical357"
VENV=/tmp/eval-serve-venv
PORT="${PORT:-8000}"
LOG="$HERE/_server_${ARM}.log"
PIDFILE="$HERE/_server_${ARM}.pid"

# Container-local GPU index is 0 (nvidia-smi/NVML see exactly one mapped A10G as
# index 0). The inherited host-level CUDA_VISIBLE_DEVICES=4 must be OVERRIDDEN —
# asking NVML for device 4 raises NVMLError_InvalidArgument.
export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true

# Disable the flashinfer top-k/top-p sampler. Its JIT kernel #includes <curand.h>,
# which is absent from this box's /usr/local/cuda/include (CUDA 13.2 toolkit ships
# no curand headers; the only curand.h is the CUDA-13.0 venv copy) -> EngineCore
# dies compiling sampling.cu. We decode greedily (temp=0) where the "sampler" is a
# pure argmax, so forward_native (torch) is bit-identical to the flashinfer path
# for token selection. Attention stays TRITON_ATTN (loads fine, no curand). This
# is set for BOTH arms, so the base-vs-ship A/B uses one identical decode path.
export VLLM_USE_FLASHINFER_SAMPLER=0

# The inject dir carries sitecustomize.py and goes on PYTHONPATH for BOTH arms:
# it ALWAYS installs the prometheus route-name compat shim (orthogonal to model
# numerics) and ONLY activates the pck04 lm_head patch when PCK04_KEEPSET is set.
if [[ -n "$KEEPSET" ]]; then
  export PCK04_KEEPSET="$KEEPSET"
  export PCK04_PATCH_DIR="$SUB_DIR"
  # inject dir FIRST so our sitecustomize wins; submission dir so the import resolves.
  export PYTHONPATH="$HERE/pck04_inject:$SUB_DIR${PYTHONPATH:+:$PYTHONPATH}"
else
  unset PCK04_KEEPSET || true
  # base arm: inject dir only (prometheus shim); pck04 stays inert because
  # PCK04_KEEPSET is unset, so the model is pure-vanilla vLLM.
  export PYTHONPATH="$HERE/pck04_inject${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "[start_server] arm=$ARM model=$MODEL_DIR keepset=${KEEPSET:-<none>} port=$PORT max_model_len=${MAX_MODEL_LEN:-4096} max_num_seqs=${MAX_NUM_SEQS:-16} $(date -u +%H:%M:%SZ)" | tee "$LOG"

setsid "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name gemma-4-e4b-it \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_MEM:-0.90}" \
  --max-num-seqs "${MAX_NUM_SEQS:-16}" \
  --trust-remote-code \
  --disable-log-stats \
  --override-generation-config '{"temperature":0.0,"top_p":1.0,"top_k":0}' \
  >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[start_server] server pid=$SRV_PID logging to $LOG"

# Wait for readiness (up to ~20 min for cold load of an int4 multimodal model).
for i in $(seq 1 240); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[start_server] FAILED: server process exited early; tail:"; tail -40 "$LOG"; exit 1
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
    echo "[start_server] READY after ${i}x5s ($(date -u +%H:%M:%SZ))"; exit 0
  fi
  sleep 5
done
echo "[start_server] TIMEOUT waiting for readiness; tail:"; tail -50 "$LOG"; exit 1
