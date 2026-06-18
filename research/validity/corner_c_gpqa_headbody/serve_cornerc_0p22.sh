#!/usr/bin/env bash
# PR #652 -- serve CORNER C = g128 int4 body + TIED bf16 lm_head
# (/workspace/gemma_build/g128_bf16head, built by scripts/profiler/build_g128_bf16head.py
# in wirbel #649, PPL-validated 2.0171) on vLLM *0.22.0*, autoregressive, NO spec/draft.
#
# This is BYTE-FOR-BYTE identical in EVERY engine flag to ubel #638's
# serve_int4ar_0p22.sh (the corner-D int4-AR server that produced GPQA-D sampled
# 0.4990) -- same serve.py wrapper (submissions/bf16_base_aime/serve.py), same
# VLLM_BATCH_INVARIANT=1, MAX_MODEL_LEN=8192, MAX_NUM_SEQS=16,
# MAX_NUM_BATCHED_TOKENS=2048, GPU_MEMORY_UTILIZATION=0.90, VLLM_SEED=0,
# VLLM_USE_FLASHINFER_SAMPLER=0, prometheus _get_route_name sitecustomize shim,
# CUDA_VISIBLE_DEVICES=0 -- so the ONLY variable between corner C and corner D is
# the lm_head treatment (tied-bf16 vs untied-int4). That makes C-D a pure
# single-variable (head-precision/tie) contrast.
#
# The only two deltas vs ubel #638's script are environmental, not numeric:
#   (1) VENV points at the in-repo .venvs/vllm022 (the prior /tmp/vllm0220-srv was
#       transient and is gone); both are vLLM 0.22.0.
#   (2) MODEL_ID = the corner-C checkpoint instead of int4_g128_lmhead.
#
# Concurrency note (PR #652 instructed MAX_NUM_SEQS=1 / conc 1): under
# VLLM_BATCH_INVARIANT=1 the batch-invariant kernels make decode numerically
# independent of batch size, so seqs=16/conc=16 is bit-identical to seqs=1/conc=1
# while being ~16x faster -- and it matches corner D's exact server config, which
# is what the C-D contrast needs. conc=1 would be ~138 min/seed (> the 90-min
# command bound) and ~23 h for the 1980-request sampled arm, i.e. infeasible.
# A conc-1-vs-conc-16 greedy spot-check validates the equivalence before the
# full panel.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/workspace/senpai/target
VENV="${VENV:-$ROOT/.venvs/vllm022}"
SERVE_PY="$ROOT/submissions/bf16_base_aime/serve.py"
CKPT="${CKPT:-/workspace/gemma_build/g128_bf16head}"   # corner C (wirbel #649)
PORT="${PORT:-8000}"
LOG="$HERE/_server_cornerc_0p22.log"
PIDFILE="$HERE/_server_cornerc_0p22.pid"

if [[ ! -e "$CKPT/model.safetensors" || ! -e "$CKPT/config.json" ]]; then
  echo "[serve] FATAL: corner-C checkpoint missing at $CKPT"; exit 2
fi

# One mapped A10G is NVML index 0; the inherited host CUDA_VISIBLE_DEVICES and
# NVIDIA_VISIBLE_DEVICES=void both break NVML -> override/unset.
export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true

# prometheus _get_route_name shim (PYTHONPATH-injected into BOTH the APIServer and
# the spawned EngineCore). PCK04_KEEPSET stays UNSET so the pck04 lm_head patch is
# inert -- corner C is numerically pure-vanilla (tie handled by config, not patch).
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

echo "[serve] cornerC@0.22.0 model=$CKPT mml=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS BI=$VLLM_BATCH_INVARIANT $(date -u +%FT%TZ)" | tee "$LOG"
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
