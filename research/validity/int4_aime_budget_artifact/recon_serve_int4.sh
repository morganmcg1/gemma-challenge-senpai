#!/usr/bin/env bash
# PR #699 -- DECISIVE greedy reconciliation serve (base OR int4), ENFORCE-EAGER.
# Engine finding (this session): on .venvs/vllm022 (torch2.11/cu130) the inductor/
# CUDA-graph COMPILE path corrupts GREEDY AIME decode for BOTH bodies -- base collapses
# 0.4667->0.1333 (trunc 8->22/60, 10/60 gibberish), int4 0.350->~0.05. The banked
# /tmp/vllm0220-srv engine (compile-ON, clean 0.350/0.4667) is GONE. This serve disables
# compile via --enforce-eager (the trustworthy numeric substrate) and otherwise mirrors
# the banked recipe (BI=1, mml configurable, seqs=16, mnbt=2048, seed=0,
# flashinfer-sampler=0, expandable_segments, HF_HUB_OFFLINE=1, pck04_inject INERT).
# Test: does enforce-eager reconcile greedy to the banked anchors? base->0.4667 / int4->0.350.
# analysis_only: local serve, NO HF Job.
set -euo pipefail

ROOT=/workspace/senpai/target
VENV="${VENV:-$ROOT/.venvs/vllm022}"
CKPT="${1:?model dir or HF id (e.g. /workspace/gemma_build/int4_g128_lmhead OR the bf16 base snapshot)}"
PORT="${2:-8000}"
MML="${3:-8192}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/_recon_serve_${PORT}.log"
PIDFILE="$HERE/_recon_serve_${PORT}.pid"

if [[ "$CKPT" == /* && ! -f "$CKPT/model.safetensors" ]]; then
  echo "[serve] FATAL: no checkpoint at $CKPT"; exit 2
fi
EAGER_FLAG=(); [[ "$ENFORCE_EAGER" == "1" ]] && EAGER_FLAG=(--enforce-eager)

export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true

INJECT_DIR="$ROOT/research/validity/downstream_quality_eval/pck04_inject"
export PYTHONPATH="$INJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
unset PCK04_KEEPSET || true   # inert -> int4 body numerically pure-vanilla

export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "[serve] model=$CKPT venv=$VENV mml=$MML BI=$VLLM_BATCH_INVARIANT eager=$ENFORCE_EAGER $(date -u +%FT%TZ)" | tee "$LOG"
"$VENV/bin/python" -c "import vllm,torch; print('[serve] vllm',vllm.__version__,'torch',torch.__version__)" | tee -a "$LOG"

# serve.py-equivalent raw api_server, --enforce-eager by default (bypass buggy compile).
setsid "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" \
  --served-model-name gemma-4-e4b-it \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MML" \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 16 \
  --seed 0 \
  --trust-remote-code \
  --no-enable-log-requests \
  "${EAGER_FLAG[@]}" \
  >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[serve] pid=$SRV_PID log=$LOG"

for i in $(seq 1 360); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[serve] FAILED: server exited early; tail:"; tail -50 "$LOG"; exit 1
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
    echo "[serve] READY after ${i}x5s ($(date -u +%FT%TZ))"; exit 0
  fi
  sleep 5
done
echo "[serve] TIMEOUT; tail:"; tail -60 "$LOG"; exit 1
