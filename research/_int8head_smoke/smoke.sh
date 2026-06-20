#!/usr/bin/env bash
# Cheap load + dispatch + coherence smoke for the int8-channelwise lm_head build.
# Spec OFF (NUM_SPECULATIVE_TOKENS=0) to isolate the head load/kernel dispatch.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
# FlashInfer JITs sampling.cu at startup and needs curand.h, which is absent from
# nvcc's system include but present in the serving venv's nvidia/cu13/include.
# Expose ONLY the curand* headers (symlink shim) so the build matches how the bi0
# control was validated, without shadowing the rest of the system CUDA toolchain.
export CPATH=/workspace/senpai/target/research/_int8head_smoke/_curand_shim${CPATH:+:$CPATH}
VENV=/tmp/senpai-venvs/20f658587e8a6643/bin/python
SUB=/workspace/senpai/target/submissions/int4_mtp_bi0_int8head
MODEL=${MODEL_ID:-/workspace/gemma_build/bi0_int8head_ch}
LOG=/workspace/senpai/target/research/_int8head_smoke/server.log
PORT=8009

cd "$SUB"
echo "[smoke] starting server (spec OFF) MODEL=$MODEL"
MODEL_ID="$MODEL" SERVED_MODEL_NAME=gemma-4-e4b-it HOST=127.0.0.1 PORT=$PORT \
  MAX_MODEL_LEN=4096 GPU_MEMORY_UTILIZATION=0.90 MAX_NUM_SEQS=1 \
  MAX_NUM_BATCHED_TOKENS=512 VLLM_BATCH_INVARIANT=0 NUM_SPECULATIVE_TOKENS=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid "$VENV" serve.py > "$LOG" 2>&1 &
SPID=$!
echo "[smoke] server pgid $SPID"

# poll for readiness
ready=0
for i in $(seq 1 180); do
  if ! kill -0 $SPID 2>/dev/null; then echo "[smoke] server EXITED early"; break; fi
  if curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  sleep 5
done

if [ "$ready" = "1" ]; then
  echo "[smoke] endpoint ready; sending greedy completions"
  for p in "The capital of France is" "Q: What is 17 times 23? A:" "def fibonacci(n):"; do
    echo "--- PROMPT: $p"
    curl -s "http://127.0.0.1:$PORT/v1/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"gemma-4-e4b-it\",\"prompt\":\"$p\",\"max_tokens\":32,\"temperature\":0,\"return_token_ids\":true}" \
      | $VENV -c "import sys,json; r=json.load(sys.stdin); c=r['choices'][0]; print(repr(c.get('text'))); print('token_ids:', c.get('token_ids'))"
  done
else
  echo "[smoke] NOT READY — dumping server log tail"
fi

echo "=== kernel dispatch lines ==="
grep -iE "Using .*Kernel for CompressedTensorsWNA16|AllSpark|Marlin|lm_head" "$LOG" | head -20
echo "=== errors/tracebacks ==="
grep -iE "error|traceback|assert|exception|failed|CUDA" "$LOG" | head -20

echo "[smoke] shutting down server $SPID"
kill -TERM -$SPID 2>/dev/null || kill -TERM $SPID 2>/dev/null
sleep 3
kill -KILL $SPID 2>/dev/null
echo "[smoke] done"
