#!/usr/bin/env bash
# PR #627 -- DECISIVE serve: the EXACT submission serving config. Runs the
# submission's OWN serve.py under its manifest.json env (MAX_MODEL_LEN=4096,
# MAX_NUM_BATCHED_TOKENS=512 prefill chunk, GPU_MEMORY_UTILIZATION=0.90, default
# max_num_seqs) -- i.e. the flags the real benchmark/train.py serve path uses to
# serve submissions/int4_g128_lmhead. The submission is single-stream (conc=1) at
# the client. Served on vLLM 0.22.0 (the manifest pin).
#
# Local-only deltas vs the HF runner, both greedy-neutral:
#   CUDA_VISIBLE_DEVICES=0           (this pod's index quirk)
#   VLLM_USE_FLASHINFER_SAMPLER=0    (local flashinfer JIT crash guard; greedy uses
#                                     argmax, not the sampler, so this does not change
#                                     greedy outputs -- confirmed non-axis for temp=0)
# Usage: serve_submission_config.sh <vllm_python> [port]
set -u
VLLM_PY="${1:?vllm python path (must be 0.22.0)}"
PORT="${2:-8000}"
cd /workspace/senpai/target
SUB=submissions/int4_g128_lmhead
HERE=research/validity/optionb_crater_config_axis
LOG="$HERE/logs/serve_submission.log"
PIDFILE="$HERE/logs/serve_submission.pid"
mkdir -p "$HERE/logs"

# --- manifest.json env (the exact submission serve config) ---
export MODEL_ID=model
export MAX_MODEL_LEN=4096
export GPU_MEMORY_UTILIZATION=0.90
export MAX_NUM_BATCHED_TOKENS=512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SERVED_MODEL_NAME=gemma-4-e4b-it
export HOST=127.0.0.1
export PORT="$PORT"
# --- local-only, greedy-neutral ---
export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true
export VLLM_USE_FLASHINFER_SAMPLER=0

VER=$("$VLLM_PY" -c "import vllm;print(vllm.__version__)" 2>/dev/null)
echo "[serve:submission] vllm=$VER serve.py MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS gpu_util=$GPU_MEMORY_UTILIZATION (default max_num_seqs) fi_sampler=OFF $(date -u +%FT%TZ)" | tee "$LOG"

setsid "$VLLM_PY" "$SUB/serve.py" >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[serve:submission] pid=$SRV_PID -> $LOG"

for i in $(seq 1 360); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[serve:submission] FAILED: process exited early; tail:"; tail -40 "$LOG"; exit 1
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
    echo "[serve:submission] READY after ${i}x5s ($(date -u +%FT%TZ))"
    curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" | /usr/bin/python3 -c "import sys,json;d=json.load(sys.stdin)['data'][0];print('[serve:submission] served max_model_len=',d.get('max_model_len'))"
    exit 0
  fi
  sleep 5
done
echo "[serve:submission] TIMEOUT; tail:"; tail -60 "$LOG"; exit 1
