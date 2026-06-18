#!/usr/bin/env bash
# PR #631: serve int4_g128_lmhead on vLLM 0.22.1rc1.dev307 for the
# concurrency-determinism CROSS-CHECK. The ONLY change vs the #618 serve is the
# engine pin: 0.22.0 -> dev307 (the engine the LIVE submission serves on). #618
# showed 0.22.0+conc=1 is byte-deterministic but generation-degenerate (int4
# craters to acc 0.2121, ~45% loop-to-cap); lawine #606/#610 showed dev307 is
# determinism-bimodal at conc=16 (64/198 flips) but generation-HEALTHY (#615
# finish-length 3.1%, acc ~0.486). This run tests whether dev307+conc=1
# collapses to a single DETERMINISTIC mode the way 0.22.0 did -- i.e. whether
# dev307+conc=1 is the clean gate operating point (deterministic AND healthy).
# Everything else (6144 model-len, 16-way batch, gpu_mem 0.90, bf16,
# sampling-override backstop, native torch sampler) is held identical to #618 so
# the greedy conc=16 flip count is comparable to #618's 120/198 and #610's 64/198.
#
# int4_g128_lmhead is the OFFICIAL submission (BASELINE.md: 126.378 TPS / PPL
# 2.019 / 128/128 VALID) and the live submission serves on dev307; serve.py
# confirms the engine auto-detects the bundled compressed-tensors config ->
# Marlin int4 at load, no MODEL inject and no PLE-fold needed (the fold is baked
# into the shipped checkpoint). dev307 DOES require the prometheus route-name
# compat shim below (a serve-side metrics fix, not a model patch). We smoke-gate
# coherence before the full sweep regardless.
set -euo pipefail

PORT="${1:-8000}"
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-/workspace/senpai/target/.venvs/vllm0221}"         # vLLM 0.22.1rc1.dev307
MODEL_DIR="${MODEL_DIR:-/workspace/gemma_build/int4_g128_lmhead}" # shipped build
# REQUIRED on dev307 (not on 0.22.0): the #563 prometheus route-name compat shim.
# dev307's app.routes carry FastAPI `_IncludedRouter` entries with no `.path`, and
# prometheus_fastapi_instrumentator's `_get_route_name` does `route.path` on every
# route -> 500s EVERY request (incl. /v1/models, /v1/completions). The shim only
# patches metrics route-name resolution; it is orthogonal to model numerics /
# sampling / greedy argmax, so generation + determinism are unaffected. Same shim
# the known-good dev307 serve (#579/#610 int4g128_quality_gate) uses.
INJECT="/workspace/senpai/target/research/validity/base_fullhead_quality_sampling/_inject"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEM="${GPU_MEM:-0.90}"

if [[ ! -f "$MODEL_DIR/model.safetensors" ]]; then
  echo "[serve] FATAL: no checkpoint at $MODEL_DIR"; exit 1
fi

LOG="$HERE/_server.log"
PIDFILE="$HERE/_server.pid"

export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true
# curand-less box: flashinfer SAMPLER JIT #includes <curand.h> -> EngineCore dies.
# Disable -> vLLM native torch sampler (does temperature/top_p/top_k). (#610 parity)
export VLLM_USE_FLASHINFER_SAMPLER=0
# prometheus route-name compat shim (auto-imported via sitecustomize.py in EVERY
# process incl. the v1 EngineCore worker) -> dev307 stops 500ing every request.
export PYTHONPATH="$INJECT${PYTHONPATH:+:$PYTHONPATH}"

# generation-config backstop = lewtun #31 gemma-4-E4B-it protocol (matches #610).
# Per-request params win; greedy requests send temperature=0 explicitly so this
# only seeds defaults for omitted fields (irrelevant to greedy argmax).
OVERRIDE_GEN_CONFIG="${OVERRIDE_GEN_CONFIG:-{\"temperature\":1.0,\"top_p\":0.95,\"top_k\":64}}"

echo "[serve] engine=dev307 venv=$VENV model=$MODEL_DIR port=$PORT len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS gpu_mem=$GPU_MEM $(date -u +%H:%M:%SZ)" | tee "$LOG"
"$VENV/bin/python" -c "import vllm; print('[serve] vllm', vllm.__version__)" | tee -a "$LOG"

setsid "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name gemma-4-e4b-it \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --trust-remote-code \
  --disable-log-stats \
  --override-generation-config "$OVERRIDE_GEN_CONFIG" \
  >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[serve] server pid=$SRV_PID logging to $LOG"

for i in $(seq 1 360); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[serve] FAILED: server exited early; tail:"; tail -80 "$LOG"; exit 1
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
    echo "[serve] READY after ${i}x5s ($(date -u +%H:%M:%SZ))"; exit 0
  fi
  sleep 5
done
echo "[serve] TIMEOUT; tail:"; tail -80 "$LOG"; exit 1
