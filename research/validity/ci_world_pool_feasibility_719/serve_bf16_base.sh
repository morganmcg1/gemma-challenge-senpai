#!/usr/bin/env bash
# PR #719 -- serve the canonical UNQUANTIZED bf16 base (google/gemma-4-E4B-it
# @fee6332c) for past-AIME base-greedy comparability measurement. Analysis-only,
# local A10G, NO HF job. Reuses the #580 bf16-base serve.py verbatim; only the
# venv (project .venv, vLLM 0.22.0) and my own HF cache snapshot differ from the
# #628 recipe, keeping the harness apples-to-apples with the gate's base anchor.
#
# Greedy argmax measurement: BI=1 (batch-invariant -> client concurrency leaves
# per-request greedy tokens unchanged, so the per-band rate is identical to
# sequential), flashinfer sampler OFF (curand.h absent in-container; greedy
# unaffected), CUDA_VISIBLE_DEVICES=0 (the one mapped A10G; inherited =5 breaks NVML).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/workspace/senpai/target
VENV="$ROOT/.venv"
SERVE_PY="$ROOT/submissions/bf16_base_aime/serve.py"
SNAP=/senpai-run/home/student-denken/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/fee6332c1abaafb77f6f9624236c63aa2f1d0187
PORT="${PORT:-8000}"
LOG="$HERE/_server_bf16.log"
PIDFILE="$HERE/_server_bf16.pid"

export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true

# prometheus_fastapi_instrumentator route-name shim (else every request 500s).
INJECT_DIR="$ROOT/research/validity/downstream_quality_eval/pck04_inject"
export PYTHONPATH="$INJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
unset PCK04_KEEPSET || true

export MODEL_ID="$SNAP"
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

echo "[serve] bf16-base model=$SNAP mml=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS BI=$VLLM_BATCH_INVARIANT $(date -u +%FT%TZ)" | tee "$LOG"
"$VENV/bin/python" -c "import vllm; print('[serve] vllm', vllm.__version__)" | tee -a "$LOG"

setsid "$VENV/bin/python" "$SERVE_PY" >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[serve] pid=$SRV_PID log=$LOG"

for i in $(seq 1 300); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[serve] FAILED: server exited early; tail:"; tail -50 "$LOG"; exit 1
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
    echo "[serve] READY after ${i}x5s ($(date -u +%FT%TZ))"; exit 0
  fi
  sleep 5
done
echo "[serve] TIMEOUT; tail:"; tail -60 "$LOG"; exit 1
