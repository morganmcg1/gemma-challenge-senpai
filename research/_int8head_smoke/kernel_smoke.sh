#!/usr/bin/env bash
# Spec-OFF kernel-dispatch smoke for an lm_head-bytefloor arm (PR #796).
# Isolates the lm_head load/kernel pick (drafter OFF) so we can confirm the
# checkpoint serves via a real Marlin W4A16 kernel and NOT a dense bf16 fallback.
# Native sampler (VLLM_USE_FLASHINFER_SAMPLER=0) is the canonical local fix; it
# does not touch logits. Usage: kernel_smoke.sh <model_dir> <port> <tag>
set -uo pipefail
MODEL=$1; PORT=$2; TAG=$3
ROOT=/workspace/senpai/target
SUB=$ROOT/submissions/int4_mtp_bi0_lmhead_bytefloor
VENV=/tmp/senpai-venvs/20f658587e8a6643/bin/python
LOG=$ROOT/logs/pr796/smoke_${TAG}_serve.log

cd "$SUB"
echo "[smoke:$TAG] starting server (spec OFF) MODEL=$MODEL $(date -u +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  MODEL_ID="$MODEL" SERVED_MODEL_NAME=gemma-4-e4b-it HOST=127.0.0.1 PORT=$PORT \
  MAX_MODEL_LEN=4096 GPU_MEMORY_UTILIZATION=0.90 MAX_NUM_SEQS=1 \
  MAX_NUM_BATCHED_TOKENS=512 VLLM_BATCH_INVARIANT=0 NUM_SPECULATIVE_TOKENS=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid "$VENV" serve.py > "$LOG" 2>&1 &
SPID=$!
echo "[smoke:$TAG] server pgid $SPID"

ready=0
for i in $(seq 1 120); do
  if ! kill -0 $SPID 2>/dev/null; then echo "[smoke:$TAG] server EXITED early"; break; fi
  if curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  sleep 5
done

if [ "$ready" = "1" ]; then
  echo "[smoke:$TAG] endpoint ready; greedy completions:"
  for p in "The capital of France is" "Q: What is 17 times 23? A:" "def fibonacci(n):"; do
    echo "--- PROMPT: $p"
    curl -s "http://127.0.0.1:$PORT/v1/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"gemma-4-e4b-it\",\"prompt\":\"$p\",\"max_tokens\":24,\"temperature\":0,\"return_token_ids\":true}" \
      | $VENV -c "import sys,json; r=json.load(sys.stdin); c=r['choices'][0]; print(repr(c.get('text')))"
  done
else
  echo "[smoke:$TAG] NOT READY"
fi

echo "=== kernel dispatch lines ==="
grep -iE "Using .*Kernel for CompressedTensorsWNA16|MarlinLinearKernel|AllSpark|no kernel|fall.?back|UnquantizedLinearMethod|lm_head" "$LOG" | head -20
echo "=== errors/tracebacks ==="
grep -iE "error|traceback|assert|exception|failed|no kernel|not support" "$LOG" | head -20

echo "[smoke:$TAG] shutting down $SPID $(date -u +%H:%M:%S)"
kill -TERM -$SPID 2>/dev/null || kill -TERM $SPID 2>/dev/null
sleep 3
kill -KILL -$SPID 2>/dev/null
echo "[smoke:$TAG] done"
