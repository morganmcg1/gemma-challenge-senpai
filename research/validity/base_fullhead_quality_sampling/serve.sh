#!/usr/bin/env bash
# PR #563: serve ONE arm of the base_fullhead-quality-under-sampling A/B on the
# clean vanilla vLLM v0.22.1rc1.dev307 engine (the engine kanna #547 documented as
# correct; v0.22.0 craters this int4 model's MMLU). Both arms use this IDENTICAL
# plain-vLLM serve path (no head inject, no surgical attention) so the ONLY
# difference between arms is the checkpoint (int4 base_fullhead vs fp16 vanilla
# base) and the ONLY difference vs #547 is the decode protocol (client-side).
#
# Usage: serve.sh <arm:int4|fp16> [port]
#   int4  -- google/gemma-4-E4B-it-qat-w4a16-ct  (base_fullhead, full 262k head)
#   fp16  -- google/gemma-4-E4B-it                (vanilla-base denominator)
set -euo pipefail

ARM="$1"
PORT="${2:-8000}"

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-/workspace/senpai/target/.venvs/vllm0221}"

HUB="$HOME/.cache/huggingface/hub"
if [[ "$ARM" == "int4" ]]; then
  MODEL_DIR="$(ls -d "$HUB"/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/*/ 2>/dev/null | head -1)"
elif [[ "$ARM" == "fp16" ]]; then
  MODEL_DIR="$(ls -d "$HUB"/models--google--gemma-4-E4B-it/snapshots/*/ 2>/dev/null | head -1)"
else
  echo "[serve] FATAL: arm must be int4|fp16, got '$ARM'"; exit 1
fi
if [[ -z "$MODEL_DIR" ]]; then echo "[serve] FATAL: no local snapshot for arm=$ARM"; exit 1; fi

LOG="$HERE/_server_${ARM}.log"
PIDFILE="$HERE/_server_${ARM}.pid"

export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true
# flashinfer SAMPLER JIT #includes <curand.h> (absent on this box) -> EngineCore
# dies. Disabling it forces vLLM's native torch sampler, which DOES implement
# temperature/top_p/top_k multinomial sampling (verified by smoke_sampling.py) --
# the curand gap only blocks the flashinfer fast path, not sampling itself.
export VLLM_USE_FLASHINFER_SAMPLER=0

# Install the prometheus route-name compat shim (vLLM 0.22.1rc1 500s every request
# without it -- _IncludedRouter has no .path). Inherited by the EngineCore child.
export PYTHONPATH="$HERE/_inject${PYTHONPATH:+:$PYTHONPATH}"

# generation-config override = the lewtun #31 mandated gemma-4-E4B-it protocol.
# Per-request sampling params (sent by run_eval.py) take precedence anyway; this is
# a backstop so a request that omits a field still samples (never silently greedy).
OVERRIDE_GEN_CONFIG="${OVERRIDE_GEN_CONFIG:-{\"temperature\":1.0,\"top_p\":0.95,\"top_k\":64}}"

EAGER_ARGS=()
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then EAGER_ARGS+=(--enforce-eager); fi

echo "[serve] arm=$ARM model=$MODEL_DIR port=$PORT max_num_seqs=${MAX_NUM_SEQS:-16} gpu_mem=${GPU_MEM:-0.90} override='$OVERRIDE_GEN_CONFIG' $(date -u +%H:%M:%SZ)" | tee "$LOG"

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
  "${EAGER_ARGS[@]}" \
  --override-generation-config "$OVERRIDE_GEN_CONFIG" \
  >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[serve] server pid=$SRV_PID logging to $LOG"

for i in $(seq 1 300); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[serve] FAILED: server exited early; tail:"; tail -60 "$LOG"; exit 1
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
    echo "[serve] READY after ${i}x5s ($(date -u +%H:%M:%SZ))"; exit 0
  fi
  sleep 5
done
echo "[serve] TIMEOUT; tail:"; tail -60 "$LOG"; exit 1
