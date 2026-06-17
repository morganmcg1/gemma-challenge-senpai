#!/usr/bin/env bash
# Serve the genuine int4 drop=2 [37,38] carve (/tmp/gemma40L-int4) with the SAME
# vanilla vLLM config as the bf16 drop2/drop3 arms, so the int4-vs-bf16 gate
# comparison is on a byte-identical decode path:
#   - VLLM_USE_FLASHINFER_SAMPLER=0  (greedy argmax via torch; curand-free)
#   - CUDA_VISIBLE_DEVICES=0         (the one mapped A10G)
#   - dtype bfloat16 activations; w4a16 weights auto-detected from quantization_config
#   - full 262k tied head (no keepset injection; this is a base-style arm)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV=/tmp/eval-serve-venv
MODEL=/tmp/gemma40L-int4
PORT=8000
LOG="$HERE/_server_int4_drop2.log"
PIDFILE="$HERE/_server_int4_drop2.pid"

export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true
export VLLM_USE_FLASHINFER_SAMPLER=0

# Prometheus route-name compat shim (same as the bf16 arms' start_server.sh): vLLM
# 0.22.1rc1 unconditionally mounts prometheus_fastapi_instrumentator, whose
# _get_route_name does route.path on every app route; under this box's FastAPI some
# routes are _IncludedRouter (no .path) -> EVERY request (incl. /v1/models readiness
# and the eval completions) 500s. pck04_inject/sitecustomize.py installs a guarded
# shim. PCK04_KEEPSET is UNSET here, so pck04 stays inert and the int4 head stays
# vanilla (full 262k tied) -- identical decode path to the bf16 base-style arms.
INJECT_DIR="$(cd "$HERE/../downstream_quality_eval/pck04_inject" && pwd)"
export PYTHONPATH="$INJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
unset PCK04_KEEPSET || true

echo "[serve-int4] model=$MODEL port=$PORT inject=$INJECT_DIR $(date -u +%H:%M:%SZ)" | tee "$LOG"
setsid "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name gemma-4-e4b-it \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 \
  --trust-remote-code \
  --disable-log-stats \
  --override-generation-config '{"temperature":0.0,"top_p":1.0,"top_k":0}' \
  >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[serve-int4] pid=$SRV_PID logging to $LOG"

for i in $(seq 1 240); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[serve-int4] FAILED: server exited early; tail:"; tail -40 "$LOG"; exit 1
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
    echo "[serve-int4] READY after ${i}x5s ($(date -u +%H:%M:%SZ))"; exit 0
  fi
  sleep 5
done
echo "[serve-int4] TIMEOUT waiting for readiness"; tail -40 "$LOG"; exit 1
