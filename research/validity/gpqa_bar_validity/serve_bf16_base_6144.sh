#!/usr/bin/env bash
# PR #614 -- serve the UNQUANTIZED vanilla bf16 base google/gemma-4-E4B-it (the GPQA
# bar DENOMINATOR model, same one #580 used for AIME, run yokbmy9i) for the GPQA-bar
# truncation/regime audit. Reuses the validated #580 submissions/bf16_base_aime/serve.py
# (stock vLLM 0.22.1rc1.dev307, full native 262k bf16 tied head, full multimodal tower,
# NO patches/spec/prune/quant) driven via env. The ONLY deltas vs #580 are:
#   * MAX_MODEL_LEN 4096 -> 6144 (#598) so a long-CoT GPQA item (input <=~1600 tok) can
#     regenerate at --max-tokens 4096 (1600+4096=5696 < 6144) without the cap biting.
#   * VLLM_USE_FLASHINFER_SAMPLER=0 (mandated by the card / #547 dev307 recipe).
#   * MAX_NUM_SEQS 16 to fit bf16 (15GB weights) + KV at len 6144 on one A10G; the eval
#     client uses --max-connections 16 so no KV reservation is wasted.
# min_tokens=8 (#541) is a per-REQUEST guard forwarded by run_eval.py (--min-tokens 8),
# not a server flag -> no served-file change. dev307. analysis_only; NO FIRE.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=/workspace/senpai/target
VENV=/tmp/senpai-venvs/5f4c623f772358a2          # pinned dev307 venv (#580)
PYV="$VENV/bin/python"
PORT="${PORT:-8000}"
LOG="$HERE/_server_bf16_base_6144.log"
PIDFILE="$HERE/_server_bf16_base_6144.pid"

export CUDA_VISIBLE_DEVICES=0
unset NVIDIA_VISIBLE_DEVICES || true
export VLLM_USE_FLASHINFER_SAMPLER=0
# serve.py env knobs (submissions/bf16_base_aime/serve.py)
export MODEL_ID="google/gemma-4-E4B-it"
export SERVED_MODEL_NAME="gemma-4-e4b-it"
export HOST="127.0.0.1"
export PORT
export MAX_MODEL_LEN="6144"
export GPU_MEMORY_UTILIZATION="0.92"
export MAX_NUM_BATCHED_TOKENS="2048"
export MAX_NUM_SEQS="16"
export VLLM_SEED="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[serve614] UNQUANTIZED bf16 base model=$MODEL_ID port=$PORT max_model_len=$MAX_MODEL_LEN gpu_util=$GPU_MEMORY_UTILIZATION max_num_seqs=$MAX_NUM_SEQS flashinfer_sampler=OFF dev307 $(date -u +%H:%M:%SZ)" | tee "$LOG"

setsid "$PYV" "$ROOT/submissions/bf16_base_aime/serve.py" >>"$LOG" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[serve614] pid=$SRV_PID -> $LOG"

for i in $(seq 1 360); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[serve614] FAILED: process exited early; tail:"; tail -40 "$LOG"; exit 1
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
    echo "[serve614] READY after ${i}x5s ($(date -u +%H:%M:%SZ))"; exit 0
  fi
  sleep 5
done
echo "[serve614] TIMEOUT; tail:"; tail -60 "$LOG"; exit 1
