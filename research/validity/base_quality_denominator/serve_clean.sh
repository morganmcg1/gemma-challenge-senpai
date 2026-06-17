#!/usr/bin/env bash
# PR #581: serve the unquantized bf16 base google/gemma-4-E4B-it using the EXACT
# int4 AIME-reference recipe (submissions/int4_base_aime/serve.py): NO
# --override-generation-config, --dtype auto, --seed 0, --max-num-batched-tokens
# 2048, --max-num-seqs 32, --no-enable-log-requests. The ONLY differences vs that
# int4 reference serve are (1) the checkpoint (bf16 vanilla base vs int4 QAT) and
# (2) two box-level env requirements that do NOT change generation semantics:
#   - VLLM_USE_FLASHINFER_SAMPLER=0  (curand.h absent on this box -> native torch sampler)
#   - PYTHONPATH=_inject prometheus route-name shim (vLLM 0.22.1rc1 500s without it)
# Purpose: isolate whether bf16<int4 on GSM8K is a serve confound (serve.sh's
# --override-generation-config) or a real model/engine effect, by serving bf16 on
# the identical recipe that produced the int4 base GSM8K denominator (0.896/0.878).
#
# Usage: serve_clean.sh [port]
set -euo pipefail
PORT="${1:-8000}"

HERE="$(cd "$(dirname "$0")" && pwd)"
SHIM="$HERE/../base_fullhead_quality_sampling/_inject"
VENV="${VENV:-/tmp/senpai-venvs/5f4c623f772358a2}"

HUB="$HOME/.cache/huggingface/hub"
MODEL_DIR="$(ls -d "$HUB"/models--google--gemma-4-E4B-it/snapshots/*/ 2>/dev/null | head -1)"
if [[ -z "$MODEL_DIR" ]]; then echo "[serve] FATAL: no bf16 snapshot under $HUB"; exit 1; fi

mkdir -p "$HERE/results"
LOG="$HERE/results/_server_bf16_clean.log"
PIDFILE="$HERE/results/_server_bf16_clean.pid"

export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH="$SHIM${PYTHONPATH:+:$PYTHONPATH}"

echo "[serve] bf16 CLEAN (int4-recipe, NO override) model=$MODEL_DIR port=$PORT venv=$VENV $(date -u +%H:%M:%SZ)" | tee "$LOG"

setsid "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name gemma-4-e4b-it \
  --host 127.0.0.1 --port "$PORT" \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 32 \
  --seed 0 \
  --trust-remote-code \
  --no-enable-log-requests \
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
