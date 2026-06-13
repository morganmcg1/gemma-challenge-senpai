#!/usr/bin/env bash
# Official-methodology greedy-identity check for the int4 g128 + int4 lm_head submission.
#
# The official gate (shared_resources/gemma_greedy_identity_verifier_flowian-powers)
# compares two HARNESS decode_outputs.jsonl files -- both produced by hitting a
# SERVED endpoint via the official decode_outputs.py (HTTP only, same Marlin int4
# kernel on both sides) -- and requires byte-for-byte token identity.
#
# We build the strongest possible test:
#   REFERENCE = plain vLLM, NO --max-num-batched-tokens  (default chunked prefill)
#   CANDIDATE = the submission serve.py, cap=512          (the actual served path)
# If these two are GREEDY_IDENTICAL, the prefill chunk cap provably does not flip
# any greedy token, so the submission passes regardless of which config the
# organizer uses to generate the reference.
#
# LOCAL A10G only: flashinfer sampler JIT is broken on this pod, so both servers
# run with VLLM_USE_FLASHINFER_SAMPLER=0 + VLLM_ATTENTION_BACKEND=FLASH_ATTN
# (greedy/argmax is sampler-independent; the backend is held identical on both
# sides, so it is a controlled variable). Not baked into serve.py.
set -uo pipefail
cd /workspace/senpai/target
ROOT=/workspace/senpai/target
PY="$ROOT/.venv/bin/python"
CKPT=/workspace/gemma_build/int4_g128_lmhead
DS="$ROOT/official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json"
DEC="$ROOT/official/main_bucket/shared_resources/speed_benchmark/scripts/decode_outputs.py"
VER="$ROOT/official/main_bucket/shared_resources/gemma_greedy_identity_verifier_flowian-powers"
OUT="$ROOT/research/_probe"
LOG="$OUT/logs"
mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM=128
OLEN=128

wait_ready() {
  local t=0 lim=${1:-600}
  until [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/v1/models 2>/dev/null)" = "200" ]; do
    sleep 5; t=$((t+5))
    if [ $t -ge $lim ]; then echo "TIMEOUT after ${t}s waiting for server"; return 1; fi
  done
  echo "server ready after ${t}s"
}

free_gpu() {
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  local t=0
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 500 ]; do
    sleep 3; t=$((t+3)); [ $t -ge 120 ] && break
  done
  echo "gpu freed (used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MiB)"
}

echo "=== REFERENCE: plain vLLM, NO cap (default chunked prefill) ==="
setsid "$PY" -m vllm.entrypoints.openai.api_server --model "$CKPT" --served-model-name gemma-4-e4b-it \
  --host 0.0.0.0 --port 8000 --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --trust-remote-code --no-enable-log-requests >"$LOG/greedy_ref_server.log" 2>&1 &
if ! wait_ready 600; then echo "REF server failed"; tail -50 "$LOG/greedy_ref_server.log"; free_gpu; exit 1; fi
"$PY" "$DEC" --base-url http://localhost:8000 --model gemma-4-e4b-it --dataset-path "$DS" \
  --tokenizer "$CKPT" --num-prompts $NUM --output-len $OLEN --seed 1 \
  --output-file "$OUT/greedy_ref_decode.jsonl" --summary-file "$OUT/greedy_ref_summary.json" 2>&1 | tail -6
free_gpu

echo "=== CANDIDATE: submission serve.py, cap=512 ==="
MODEL_ID=$CKPT SERVED_MODEL_NAME=gemma-4-e4b-it MAX_MODEL_LEN=4096 GPU_MEMORY_UTILIZATION=0.90 MAX_NUM_BATCHED_TOKENS=512 \
  setsid "$PY" "$ROOT/submissions/int4_g128_lmhead/serve.py" >"$LOG/greedy_cand_server.log" 2>&1 &
if ! wait_ready 600; then echo "CAND server failed"; tail -50 "$LOG/greedy_cand_server.log"; free_gpu; exit 1; fi
"$PY" "$DEC" --base-url http://localhost:8000 --model gemma-4-e4b-it --dataset-path "$DS" \
  --tokenizer "$CKPT" --num-prompts $NUM --output-len $OLEN --seed 1 \
  --output-file "$OUT/greedy_cand_decode.jsonl" --summary-file "$OUT/greedy_cand_summary.json" 2>&1 | tail -6
free_gpu

echo "=== VERIFY: official check_greedy_identity.py (reference vs candidate) ==="
PYTHONPATH="$VER" "$PY" "$VER/check_greedy_identity.py" \
  --reference "$OUT/greedy_ref_decode.jsonl" \
  --candidate "$OUT/greedy_cand_decode.jsonl" --json | tee "$OUT/greedy_official_verdict.json"
echo "exit_code=${PIPESTATUS[0]}"
echo "=== GREEDY OFFICIAL CHECK DONE ==="
