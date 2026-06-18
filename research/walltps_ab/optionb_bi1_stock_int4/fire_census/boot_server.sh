#!/usr/bin/env bash
# Boot the #632 Option-B BI=1 int4 spec lane for a given K via the submission's OWN
# launcher (submissions/int4_mtp_batchinv/serve.py). serve.py prepends the submission
# dir to PYTHONPATH so sitecustomize.py auto-applies vllm_attn_group_patch (the {8,4}
# draft/target attention-group num_heads backport) in the worker process -- a raw
# `python -m vllm...` boot misses it and dies on the heads assertion. Faithful to the
# exact #632 server config; only NUM_SPECULATIVE_TOKENS varies per K.
# Usage: ./boot_server.sh <K>   (logs -> server_k<K>.log, pid -> server_k<K>.pid)
set -euo pipefail
K="${1:?usage: boot_server.sh <K>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/workspace/senpai/target
VENV=/tmp/senpai-venvs/20f658587e8a6643/bin/python
SERVE="$ROOT/submissions/int4_mtp_batchinv/serve.py"
LOG="$HERE/server_k${K}.log"
PIDF="$HERE/server_k${K}.pid"

# stale inherited CUDA_VISIBLE_DEVICES=7 -> physical GPU is index 0 (memory: gpu_env.md)
export CUDA_VISIBLE_DEVICES=0
export VLLM_BATCH_INVARIANT=1
export VLLM_USE_FLASHINFER_SAMPLER=0
# serve.py reads these (faithful to #632 run_k456.sh + #645 launch)
export MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct
export DRAFTER_MODEL=/tmp/qat-assistant
export NUM_SPECULATIVE_TOKENS="$K"
export SERVED_MODEL_NAME=gemma-4-e4b-it
export HOST=127.0.0.1
export PORT=8000
export MAX_MODEL_LEN=4096
export GPU_MEMORY_UTILIZATION=0.90
export MAX_NUM_SEQS=1
export MAX_NUM_BATCHED_TOKENS=512
unset SENPAI_REFERENCE_MODE   # spec ON (we census the spec lane, not the M=1 reference)

echo "[boot] K=$K via serve.py -> $LOG"
nohup "$VENV" "$SERVE" >"$LOG" 2>&1 &
echo $! > "$PIDF"
echo "[boot] pid $(cat "$PIDF"); polling /health ..."

for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "[boot] healthy after ${i}0s"
    exit 0
  fi
  if ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    echo "[boot] server process died; tail log:"; tail -30 "$LOG"; exit 1
  fi
  sleep 10
done
echo "[boot] timeout waiting for health; tail log:"; tail -30 "$LOG"; exit 1
