#!/usr/bin/env bash
# PR #615 -- serve the SHIPPED submissions/int4_g128_lmhead/model on a chosen vLLM
# build (0.22.0 vs dev307) with the CANONICAL serve.py flags (the #606 recipe that
# reproduces official PPL 2.019 on 0.22.0), bumped to --max-model-len 6144 (#598) so a
# CoT eval at --max-tokens 4096 cannot truncate (input<=~1600 -> 1600+4096<6144).
#
# IDENTICAL flags/env on both stacks -> the ONLY variable is the vLLM version. No
# PLE_FOLD_EMBED_SCALE (canonical serve omits it; the bundled compressed-tensors
# checkpoint bakes embed_scale -> #606 got correct 0.22.0 PPL without it; held
# constant across stacks regardless).
#
# Usage: serve_ship.sh <vllm_python> <stack_label> [port] [max_model_len]
set -u
VLLM_PY="${1:?vllm python path}"
STACK="${2:?stack label e.g. v0220 or dev307}"
PORT="${3:-8000}"
MML="${4:-6144}"

cd /workspace/senpai/target
MODEL_DIR=submissions/int4_g128_lmhead/model
HERE=research/validity/eval_stack_accuracy_validity
LOG="$HERE/logs/serve_${STACK}.log"
PIDFILE="$HERE/logs/serve_${STACK}.pid"
mkdir -p "$HERE/logs"

export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VER=$("$VLLM_PY" -c "import vllm;print(vllm.__version__)" 2>/dev/null)
echo "[serve:$STACK] vllm=$VER model=$MODEL_DIR port=$PORT max_model_len=$MML max_num_seqs=32 flashinfer_sampler=OFF $(date -u +%FT%TZ)" | tee "$LOG"

setsid "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name gemma-4-e4b-it \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MML" \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --trust-remote-code \
  --no-enable-log-requests \
  >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[serve:$STACK] pid=$SRV_PID -> $LOG"

for i in $(seq 1 360); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[serve:$STACK] FAILED: process exited early; tail:"; tail -40 "$LOG"; exit 1
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
    echo "[serve:$STACK] READY after ${i}x5s ($(date -u +%FT%TZ))"; exit 0
  fi
  sleep 5
done
echo "[serve:$STACK] TIMEOUT; tail:"; tail -60 "$LOG"; exit 1
