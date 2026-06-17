#!/usr/bin/env bash
# PR #618: serve int4_g128_lmhead on vLLM 0.22.0 (NOT dev307) for the
# concurrency-determinism study. The ONLY change vs the #610 serve (serve.sh in
# int4g128_quality_gate, which ran dev307) is the engine: 0.22.0 is the faithful
# determinism engine per lawine #606 (dev307 is bimodal/non-faithful for
# determinism). Everything else (6144 model-len, 16-way batch, gpu_mem 0.90,
# bf16, sampling-override backstop, native torch sampler) is held identical to
# #610 so the greedy conc=16 flip count is comparable to #610's 64/198.
#
# int4_g128_lmhead is the OFFICIAL 0.22.0 submission (BASELINE.md: 126.378 TPS /
# PPL 2.019 / 128/128 VALID), whose serve.py confirms vLLM 0.22.0 auto-detects
# the bundled compressed-tensors config -> Marlin int4 at load, no inject and no
# PLE-fold needed (the fold is baked into the shipped checkpoint). The stale
# "0.22.0 craters MMLU" note in the #579 serve.sh predates #606; we smoke-gate
# coherence before the full sweep regardless.
set -euo pipefail

PORT="${1:-8000}"
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-/workspace/senpai/target/.venvs/vllm022}"          # vLLM 0.22.0
MODEL_DIR="${MODEL_DIR:-/workspace/gemma_build/int4_g128_lmhead}" # shipped build
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

# generation-config backstop = lewtun #31 gemma-4-E4B-it protocol (matches #610).
# Per-request params win; greedy requests send temperature=0 explicitly so this
# only seeds defaults for omitted fields (irrelevant to greedy argmax).
OVERRIDE_GEN_CONFIG="${OVERRIDE_GEN_CONFIG:-{\"temperature\":1.0,\"top_p\":0.95,\"top_k\":64}}"

echo "[serve] engine=0.22.0 venv=$VENV model=$MODEL_DIR port=$PORT len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS gpu_mem=$GPU_MEM $(date -u +%H:%M:%SZ)" | tee "$LOG"
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
