#!/usr/bin/env bash
# PR #590 -- serve base_fullhead at --max-model-len 6144 for the MMLU-Pro TRUNCATION
# DE-BIAS pass. IDENTICAL recipe to serve_base_fullhead.sh (same stock int4_g32 QAT
# checkpoint, full native 262k bf16 tied head, spec-OFF, VLLM_USE_FLASHINFER_SAMPLER=0,
# pck04_inject metrics shim with PCK04_KEEPSET unset -> numerically vanilla base) -- the
# ONLY change is --max-model-len 4096 -> 6144 so a truncated MMLU-Pro sample (input <=1604
# tok) can be regenerated with --max-tokens 4096 (1604+4096=5700 < 6144) without the
# output cap biting. Advisor-suggested model-len bump (#590 thread, 16:53Z).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV=/tmp/eval-serve-venv
PORT="${PORT:-8000}"
MODEL_DIR=/senpai-run/home/student-ubel/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0
LOG="$HERE/_server_base_fullhead_6144.log"
PIDFILE="$HERE/_server_base_fullhead_6144.pid"

export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true
export VLLM_USE_FLASHINFER_SAMPLER=0
unset PCK04_KEEPSET || true
export PYTHONPATH="/workspace/senpai/target/research/validity/downstream_quality_eval/pck04_inject${PYTHONPATH:+:$PYTHONPATH}"

echo "[serve6144] base_fullhead model=$MODEL_DIR port=$PORT max_model_len=6144 max_num_seqs=32 spec=OFF flashinfer_sampler=OFF $(date -u +%H:%M:%SZ)" | tee "$LOG"

setsid "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name gemma-4-e4b-it \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 6144 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --trust-remote-code \
  --disable-log-stats \
  >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[serve6144] pid=$SRV_PID -> $LOG"

for i in $(seq 1 300); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[serve6144] FAILED: process exited early; tail:"; tail -40 "$LOG"; exit 1
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
    echo "[serve6144] READY after ${i}x5s ($(date -u +%H:%M:%SZ))"; exit 0
  fi
  sleep 5
done
echo "[serve6144] TIMEOUT; tail:"; tail -60 "$LOG"; exit 1
