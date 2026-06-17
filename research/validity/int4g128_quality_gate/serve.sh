#!/usr/bin/env bash
# PR #579: serve int4_g128_lmhead locally on the clean vanilla vLLM
# v0.22.1rc1.dev307 engine (the engine kanna #547 documented as correct for this
# int4 model; v0.22.0 craters its MMLU). IDENTICAL plain-vLLM serve path to #563's
# base_fullhead-quality-under-sampling A/B -- the ONLY difference is the checkpoint
# (int4_g128_lmhead: g128 body + untied int4 g128 lm_head, vs base_fullhead's
# official w4a16-ct g32 body + full bf16 head). One server, all 4 evals via
# --base-url. analysis_only: local serve only, no HF Job.
set -euo pipefail

PORT="${1:-8000}"

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-/workspace/senpai/target/.venvs/vllm0221}"
MODEL_DIR="${MODEL_DIR:-/workspace/gemma_build/int4_g128_lmhead}"
# reuse #563's prometheus route-name compat shim (vLLM 0.22.1rc1 500s without it)
INJECT="/workspace/senpai/target/research/validity/base_fullhead_quality_sampling/_inject"

if [[ ! -f "$MODEL_DIR/model.safetensors" ]]; then
  echo "[serve] FATAL: no checkpoint at $MODEL_DIR"; exit 1
fi

LOG="$HERE/_server.log"
PIDFILE="$HERE/_server.pid"

export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true
# flashinfer SAMPLER JIT #includes <curand.h> (absent here) -> EngineCore dies.
# Disable it -> vLLM's native torch sampler (DOES temperature/top_p/top_k).
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH="$INJECT${PYTHONPATH:+:$PYTHONPATH}"

# generation-config backstop = the lewtun #31 mandated gemma-4-E4B-it protocol.
# Per-request params (sent by each harness) take precedence; this only ensures a
# request that omits a field still samples (and never silently greedy for the
# sampled evals). AIME sends temperature=0 explicitly -> greedy honoured.
OVERRIDE_GEN_CONFIG="${OVERRIDE_GEN_CONFIG:-{\"temperature\":1.0,\"top_p\":0.95,\"top_k\":64}}"

echo "[serve] model=$MODEL_DIR port=$PORT max_num_seqs=${MAX_NUM_SEQS:-32} gpu_mem=${GPU_MEM:-0.90} override='$OVERRIDE_GEN_CONFIG' $(date -u +%H:%M:%SZ)" | tee "$LOG"

setsid "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name gemma-4-e4b-it \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_MEM:-0.90}" \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" \
  --trust-remote-code \
  --disable-log-stats \
  --override-generation-config "$OVERRIDE_GEN_CONFIG" \
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
