#!/usr/bin/env bash
# Minimal boot smoke for fp8_e4m3 KV on the int4-ct bi0 target.
# Boots serve.py with the manifest env, waits for /v1/models, does one 1-token
# completion, then kills the server. Confirms: e4m3 boots (no e5m2 ValueError),
# TRITON_ATTN backend, force-2D patch fires, fp8 KV active.
set -uo pipefail
ROOT=/workspace/senpai/target
VENV=/tmp/senpai-venvs/20f658587e8a6643/bin/python
OUT=$ROOT/research/validity/bi0_fp8kv
LOG=$OUT/_smoke_e4m3_server.log
cd "$ROOT" || exit 1

export VLLM_BATCH_INVARIANT=0
export DRAFTER_MODEL=google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant
export NUM_SPECULATIVE_TOKENS=6
export KV_CACHE_DTYPE=fp8_e4m3
export MAX_MODEL_LEN=4096
export GPU_MEMORY_UTILIZATION=0.90
export MAX_NUM_BATCHED_TOKENS=512
export MAX_NUM_SEQS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PORT=8000

echo "[smoke] booting fp8_e4m3 server -> $LOG"
"$VENV" submissions/int4_mtp_bi0_fp8kv/serve.py > "$LOG" 2>&1 &
SRV=$!
echo "[smoke] server pid=$SRV"

ready=0
for i in $(seq 1 80); do
  if ! kill -0 "$SRV" 2>/dev/null; then echo "[smoke] SERVER EXITED EARLY (rc check)"; break; fi
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then ready=1; echo "[smoke] /v1/models ready after ~$((i*3))s"; break; fi
  sleep 3
done

if [ "$ready" = "1" ]; then
  echo "[smoke] === 1-token completion ==="
  curl -sf http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"gemma-4-e4b-it","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
    | "$VENV" -c "import json,sys; d=json.load(sys.stdin); print('[smoke] completion:', repr(d['choices'][0]['text']))" 2>&1
else
  echo "[smoke] NOT READY -- dumping last 40 log lines:"
  tail -n 40 "$LOG"
fi

echo "[smoke] killing server pid=$SRV"
kill "$SRV" 2>/dev/null
for i in $(seq 1 10); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
kill -9 "$SRV" 2>/dev/null
echo "[smoke] === signal grep ==="
grep -iE "KV cache dtype|force2d|unified_attention wrapped|TRITON|kv_cache_dtype|fp8_e4m3|fp8_e5m2|ValueError|Traceback|EngineCore" "$LOG" | tail -n 30
echo "[smoke] DONE rc=$ready"
