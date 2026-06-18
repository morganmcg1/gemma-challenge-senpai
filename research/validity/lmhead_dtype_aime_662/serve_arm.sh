#!/usr/bin/env bash
# PR #662 -- serve one head-dtype arm on vLLM 0.22.0, single-stream (MAX_NUM_SEQS=1),
# gb6144 greedy config. Byte-for-byte the #653/#639 AIME serve recipe (same 0.22.0
# Marlin path, BI=1, FLASHINFER_SAMPLER=0, pck04 prometheus shim) EXCEPT MODEL_ID.
# So the only thing varying across arms is the lm_head dtype baked into the checkpoint.
#
#   ARM=<name> MODEL_DIR=<dir> bash serve_arm.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/workspace/senpai/target
VENV="${VENV:-/tmp/vllm0220-srv}"
SERVE_PY="$ROOT/submissions/bf16_base_aime/serve.py"
ARM="${ARM:?set ARM=<name>}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR=<checkpoint dir>}"
PORT="${PORT:-8000}"
LOG="$HERE/_server_${ARM}.log"
PIDFILE="$HERE/_server_${ARM}.pid"

if [[ ! -f "$MODEL_DIR/model.safetensors" || ! -f "$MODEL_DIR/config.json" ]]; then
  echo "[serve] FATAL: arm '$ARM' build incomplete at '$MODEL_DIR'"; exit 2
fi

# One mapped A10G is NVML index 0; inherited host CUDA_VISIBLE_DEVICES=2 and
# NVIDIA_VISIBLE_DEVICES=void both break NVML -> override/unset.
export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true

# #614 harness prometheus route-name shim (auto-imported via PYTHONPATH in BOTH the
# APIServer and the spawned EngineCore). PCK04_KEEPSET UNSET -> only the prometheus
# shim fires; numerically pure-vanilla int4/int8 Marlin.
INJECT_DIR="$ROOT/research/validity/downstream_quality_eval/pck04_inject"
export PYTHONPATH="$INJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
unset PCK04_KEEPSET || true

export MODEL_ID="$MODEL_DIR"
export SERVED_MODEL_NAME=gemma-4-e4b-it
export HOST=127.0.0.1
export PORT
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
export VLLM_SEED=0
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "[serve] arm=$ARM model=$MODEL_DIR mml=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS BI=$VLLM_BATCH_INVARIANT $(date -u +%FT%TZ)" | tee "$LOG"
"$VENV/bin/python" -c "import vllm; print('[serve] vllm', vllm.__version__)" | tee -a "$LOG"

setsid "$VENV/bin/python" "$SERVE_PY" >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[serve] pid=$SRV_PID log=$LOG"

# Cold load (~11 GB int4/int8/bf16 multimodal) + Marlin repack + compile: allow ~25 min.
for i in $(seq 1 300); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[serve] FAILED: server exited early; tail:"; tail -80 "$LOG"; exit 1
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
    echo "[serve] READY after ${i}x5s ($(date -u +%FT%TZ))"; exit 0
  fi
  sleep 5
done
echo "[serve] TIMEOUT; tail:"; tail -80 "$LOG"; exit 1
