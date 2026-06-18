#!/usr/bin/env bash
# PR #639 Arm 3 -- serve OUR re-quantized pipeline at the SHIPPED g128/head-g128
# group sizes but with the GENUINE MSE observer (scales chosen to minimize int4
# round-trip MSE instead of raw amin/amax). On-disk format is byte-identical in
# size to the live int4_g128_lmhead submission (same packing/scale count) -- only
# the stored scale VALUES differ -- so the Marlin path and TPS are the same; this
# is the zero-speed-cost "free quality" probe. Built by build_ours_g128_mse.sh.
# Served through the SAME 0.22.0 harness as the ubel #628 bf16 denominator / Arm 1
# official-g32 / Arm 2 ours-g32 -- byte-for-byte serve_bf16_0p22.sh EXCEPT MODEL_ID.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/workspace/senpai/target
VENV="${VENV:-/tmp/vllm0220-srv}"
SERVE_PY="$ROOT/submissions/bf16_base_aime/serve.py"
MODEL_DIR="${MODEL_DIR:-/workspace/gemma_build/int4_g128_mse_lmhead}"
PORT="${PORT:-8000}"
LOG="$HERE/_server_ours_g128_mse.log"
PIDFILE="$HERE/_server_ours_g128_mse.pid"

if [[ ! -f "$MODEL_DIR/model.safetensors" || ! -f "$MODEL_DIR/config.json" ]]; then
  echo "[serve] FATAL: ours-g128-mse build incomplete at '$MODEL_DIR'"; exit 2
fi

export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true

INJECT_DIR="$ROOT/research/validity/downstream_quality_eval/pck04_inject"
export PYTHONPATH="$INJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
unset PCK04_KEEPSET || true

export MODEL_ID="$MODEL_DIR"
export SERVED_MODEL_NAME=gemma-4-e4b-it
export HOST=127.0.0.1
export PORT
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
export VLLM_SEED=0
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "[serve] ours-g128-mse@0.22.0 model=$MODEL_DIR mml=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS BI=$VLLM_BATCH_INVARIANT $(date -u +%FT%TZ)" | tee "$LOG"
"$VENV/bin/python" -c "import vllm; print('[serve] vllm', vllm.__version__)" | tee -a "$LOG"

setsid "$VENV/bin/python" "$SERVE_PY" >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[serve] pid=$SRV_PID log=$LOG"

for i in $(seq 1 300); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[serve] FAILED: server exited early; tail:"; tail -60 "$LOG"; exit 1
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
    echo "[serve] READY after ${i}x5s ($(date -u +%FT%TZ))"; exit 0
  fi
  sleep 5
done
echo "[serve] TIMEOUT; tail:"; tail -60 "$LOG"; exit 1
