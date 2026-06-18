#!/usr/bin/env bash
# PR #628 -- serve the UNQUANTIZED bf16 base (google/gemma-4-E4B-it) on vLLM
# *0.22.0* (the engine the locked-rung submission int4_g128_lmhead actually pins),
# NOT dev307. This is the Option-B *denominator* server: the same canonical bf16
# base that produced ubel #614's GPQA 0.5313 (snapshot fee6332c, full native 262k
# head, full multimodal tower, no patch/spec/prune/quant), re-served on 0.22.0 at
# fern #624's gb6144 budget so her int4+spec Option-B numerator panel has a clean,
# same-stack, same-budget base to clear.
#
# Config matched to fern #624 serve_spec EXCEPT (a) bf16 base instead of int4+spec,
# (b) engine 0.22.0 (/tmp/vllm0220-srv) instead of dev307:
#   max-model-len 8192, max-num-seqs 16, BI=1 (VLLM_BATCH_INVARIANT=1),
#   gpu-mem-util 0.90, max-num-batched-tokens 2048.
# Reuses the canonical bf16 base serve path submissions/bf16_base_aime/serve.py
# (#580) verbatim -- only the venv + env differ -- so this is apples-to-apples
# with the dev307 #580/#614 bf16 base anchors.
#
# VLLM_USE_FLASHINFER_SAMPLER=0: the flashinfer top-k/top-p sampler JIT #includes
# <curand.h> which is absent on this box -> EngineCore dies compiling sampling.cu.
# REQUIRED for the GPQA *sampled* arm (T=1/top_k=64). Greedy argmax is unaffected.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/workspace/senpai/target
VENV=/tmp/vllm0220-srv
SERVE_PY="$ROOT/submissions/bf16_base_aime/serve.py"
SNAP=/senpai-run/home/student-ubel/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/fee6332c1abaafb77f6f9624236c63aa2f1d0187
PORT="${PORT:-8000}"
LOG="$HERE/_server_bf16_0p22.log"
PIDFILE="$HERE/_server_bf16_0p22.pid"

# One mapped A10G is NVML index 0; the inherited host CUDA_VISIBLE_DEVICES=4 and
# NVIDIA_VISIBLE_DEVICES=void both break NVML -> override/unset.
export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true

# vLLM unconditionally mounts prometheus_fastapi_instrumentator, whose
# _get_route_name does route.path on every app.route; under this box's FastAPI some
# entries are _IncludedRouter objects with no .path -> EVERY request (incl.
# /v1/models readiness + eval completions) 500s. The #614 harness sitecustomize
# shim (auto-imported via PYTHONPATH in BOTH the APIServer and the spawned
# EngineCore) installs a .path-guarded _get_route_name. PCK04_KEEPSET is left
# UNSET so only the prometheus shim fires -- the pck04 lm_head patch stays inert
# and the base arm is numerically pure-vanilla.
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

echo "[serve] bf16-base@0.22.0 model=$SNAP mml=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS BI=$VLLM_BATCH_INVARIANT $(date -u +%FT%TZ)" | tee "$LOG"
"$VENV/bin/python" -c "import vllm; print('[serve] vllm', vllm.__version__)" | tee -a "$LOG"

setsid "$VENV/bin/python" "$SERVE_PY" >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[serve] pid=$SRV_PID log=$LOG"

# Cold load of a 16 GB bf16 multimodal model + compile: allow up to ~25 min.
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
