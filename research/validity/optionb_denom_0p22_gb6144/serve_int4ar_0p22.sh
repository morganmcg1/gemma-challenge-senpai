#!/usr/bin/env bash
# PR #638 -- serve the LIVE-RUNG int4 body (int4_g128_lmhead: W4A16 g128 + untied
# int4 g128 lm_head, embed bf16, all modalities) on vLLM *0.22.0*, autoregressive,
# NO spec/draft. This is leg 3 of the Option-B quality denominator: the EXACT body
# the locked submission ships, measured on the SAME engine + SAME budget as ubel
# #628's bf16 base so the bf16-vs-int4 contrast is apples-to-apples.
#
# IDENTICAL to serve_bf16_0p22.sh in EVERY engine flag -- same serve.py wrapper
# (submissions/bf16_base_aime/serve.py), same vLLM 0.22.0 venv (/tmp/vllm0220-srv),
# same VLLM_BATCH_INVARIANT=1, MAX_MODEL_LEN=8192, MAX_NUM_SEQS=16,
# MAX_NUM_BATCHED_TOKENS=2048, GPU_MEMORY_UTILIZATION=0.90, VLLM_SEED=0,
# VLLM_USE_FLASHINFER_SAMPLER=0, prometheus _get_route_name sitecustomize shim,
# CUDA_VISIBLE_DEVICES=0. The ONLY change is the model:
#   bf16 snapshot fee6332c  ->  the built int4_g128_lmhead checkpoint.
# The int4 checkpoint's config.json carries the W4A16 pack-quantized
# quantization_config; vLLM 0.22.0 auto-detects compressed-tensors and repacks to
# Marlin at load (no extra flags). --dtype bfloat16 sets the activation/compute
# dtype + the bf16 embed/vision/audio params, exactly as the int4 submission serves.
#
# We deliberately do NOT use the submission manifest's 4096/512 budget: this card
# budget-matches ubel #628 (gb6144, mml 8192, mnbt 2048), not the production serve.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/workspace/senpai/target
VENV=/tmp/vllm0220-srv
SERVE_PY="$ROOT/submissions/bf16_base_aime/serve.py"
CKPT=/workspace/gemma_build/int4_g128_lmhead       # built by build_int4ar.sh (PR #4 repro)
PORT="${PORT:-8000}"
LOG="$HERE/_server_int4ar_0p22.log"
PIDFILE="$HERE/_server_int4ar_0p22.pid"

if [[ ! -f "$CKPT/model.safetensors" || ! -f "$CKPT/config.json" ]]; then
  echo "[serve] FATAL: int4 checkpoint missing at $CKPT (run build_int4ar.sh first)"; exit 2
fi

# One mapped A10G is NVML index 0; the inherited host CUDA_VISIBLE_DEVICES=4 and
# NVIDIA_VISIBLE_DEVICES=void both break NVML -> override/unset.
export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true

# Same prometheus _get_route_name shim as the bf16 server (PYTHONPATH-injected into
# BOTH the APIServer and the spawned EngineCore). PCK04_KEEPSET stays UNSET so the
# pck04 lm_head patch is inert -- the int4 body is numerically pure-vanilla.
INJECT_DIR="$ROOT/research/validity/downstream_quality_eval/pck04_inject"
export PYTHONPATH="$INJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
unset PCK04_KEEPSET || true

export MODEL_ID="$CKPT"
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

echo "[serve] int4ar@0.22.0 model=$CKPT mml=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS BI=$VLLM_BATCH_INVARIANT $(date -u +%FT%TZ)" | tee "$LOG"
"$VENV/bin/python" -c "import vllm; print('[serve] vllm', vllm.__version__)" | tee -a "$LOG"

setsid "$VENV/bin/python" "$SERVE_PY" >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[serve] pid=$SRV_PID log=$LOG"

# int4 (~9 GB) + Marlin repack + compile: allow up to ~25 min.
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
