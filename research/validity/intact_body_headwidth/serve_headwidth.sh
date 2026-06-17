#!/usr/bin/env bash
# Serve the INTACT base-int4 body (full multimodal gemma-4-E4B-it-qat-w4a16-ct,
# 42 layers, no drop / no bake) at a chosen lm_head WIDTH for the PR #547 sweep.
#
# Usage: serve_headwidth.sh <mode:off|mask|slice> <keepset.json|-> [port]
#   off    -- full 262144-row head (the ubel #538 control substrate, no patch).
#   mask   -- full head + additive keepset mask (bit-faithful pruned-head QUALITY).
#   slice  -- runtime VRAM row-prune to K rows (genuine pruned-head TPS).
#
# Mirrors downstream_quality_eval/start_server.sh (vanilla vLLM + a compute_logits
# inject) so QUALITY is apples-to-apples with the 0.668/0.444 control, but points at
# .venvs/vllm022 and the full int4 model, and uses headwidth_inject instead of pck04.
set -euo pipefail

MODE="$1"
KEEPSET="${2:-}"
PORT="${3:-8000}"

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-/workspace/senpai/target/.venvs/vllm0221}"
# Full int4 model snapshot (intact body + full BF16 tied head).
MODEL_DIR="$(ls -d "$HOME"/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/*/ 2>/dev/null | head -1)"
if [[ -z "$MODEL_DIR" ]]; then echo "[serve-hw] FATAL: no local int4 snapshot"; exit 1; fi
LOG="$HERE/_server_${MODE}.log"
PIDFILE="$HERE/_server_${MODE}.pid"

export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true
# flashinfer sampler JIT #includes <curand.h> absent on this box -> EngineCore dies.
# Greedy (temp=0) argmax is bit-identical on forward_native; attention stays TRITON.
export VLLM_USE_FLASHINFER_SAMPLER=0

export PYTHONPATH="$HERE/headwidth_inject${PYTHONPATH:+:$PYTHONPATH}"
export HEADWIDTH_PATCH_DIR="$HERE"
if [[ "$MODE" == "off" ]]; then
  unset HEADWIDTH_KEEPSET || true
else
  if [[ -z "$KEEPSET" || "$KEEPSET" == "-" ]]; then echo "[serve-hw] FATAL: mode=$MODE needs a keepset"; exit 1; fi
  export HEADWIDTH_KEEPSET="$KEEPSET"
  export HEADWIDTH_MODE="$MODE"
  export HEADWIDTH_REQUIRE=1
fi

EAGER_ARGS=()
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then EAGER_ARGS+=(--enforce-eager); fi

echo "[serve-hw] mode=$MODE keepset=${KEEPSET:-<none>} model=$MODEL_DIR port=$PORT max_num_seqs=${MAX_NUM_SEQS:-16} enforce_eager=${ENFORCE_EAGER:-0} $(date -u +%H:%M:%SZ)" | tee "$LOG"

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
  --override-generation-config '{"temperature":0.0,"top_p":1.0,"top_k":0}' \
  >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[serve-hw] server pid=$SRV_PID logging to $LOG"

for i in $(seq 1 300); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[serve-hw] FAILED: server exited early; tail:"; tail -50 "$LOG"; exit 1
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
    echo "[serve-hw] READY after ${i}x5s ($(date -u +%H:%M:%SZ))"; exit 0
  fi
  sleep 5
done
echo "[serve-hw] TIMEOUT; tail:"; tail -60 "$LOG"; exit 1
