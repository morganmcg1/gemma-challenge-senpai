#!/usr/bin/env bash
# Fresh-process serve + eval for one AIME session (PR #672).
# Usage: serve_and_eval.sh <int4|bf16> <max_num_seqs> <logtag> <command...>
# Starts a fresh vLLM server (the "fresh process epoch"), waits for readiness,
# runs <command...> against it, then kills the server and waits for the GPU to
# free. Exit code is the command's exit code (server-start failures -> 3/4).
set -uo pipefail

ARM="$1"; MAXSEQS="$2"; LOGTAG="$3"; shift 3
DIR=/workspace/senpai/target/research/validity/int4ar_aime_band_neartie
PY=/tmp/vllm0220-srv/bin/python
SRVLOG="$DIR/_serve_${LOGTAG}.log"

if [ "$ARM" = int4 ]; then
  MODEL=/workspace/gemma_build/int4_g128_lmhead
else
  MODEL=google/gemma-4-E4B-it
fi

echo "[drv:$LOGTAG] starting fresh $ARM server (model=$MODEL maxseqs=$MAXSEQS) $(date -u +%H:%M:%S)"
# The container exposes the assigned A10G as device index 0; the inherited
# CUDA_VISIBLE_DEVICES=4 (host physical id) is invalid in this namespace and makes
# vLLM's NVML handle lookup fail at startup. Pin to the container-local index 0.
export CUDA_VISIBLE_DEVICES=0
VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name gemma-4-e4b-it --host 127.0.0.1 --port 8000 \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --max-num-batched-tokens 2048 \
  --max-num-seqs "$MAXSEQS" --seed 0 --trust-remote-code --no-enable-log-requests \
  > "$SRVLOG" 2>&1 &
SRVPID=$!

cleanup() {
  echo "[drv:$LOGTAG] killing server pid=$SRVPID $(date -u +%H:%M:%S)"
  kill "$SRVPID" 2>/dev/null
  pkill -f "VLLM::EngineCore" 2>/dev/null
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "$used" ] && break
    [ "$used" -lt 2000 ] && break
    sleep 2
  done
  echo "[drv:$LOGTAG] gpu freed (used=${used:-?}MiB)"
}
trap cleanup EXIT

# wait for readiness (<=10 min)
READY=0
for i in $(seq 1 300); do
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then READY=1; break; fi
  if ! kill -0 "$SRVPID" 2>/dev/null; then
    echo "[drv:$LOGTAG] SERVER DIED during startup; tail log:"; tail -20 "$SRVLOG"; exit 3
  fi
  sleep 2
done
if [ "$READY" != 1 ]; then echo "[drv:$LOGTAG] server NOT ready after 600s; tail log:"; tail -20 "$SRVLOG"; exit 4; fi
echo "[drv:$LOGTAG] server READY $(date -u +%H:%M:%S); running eval"

"$@"
RC=$?
echo "[drv:$LOGTAG] eval rc=$RC $(date -u +%H:%M:%S)"
exit $RC
