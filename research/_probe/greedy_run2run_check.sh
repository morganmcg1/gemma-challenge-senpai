#!/usr/bin/env bash
# Run-to-run determinism probe (the direct analog of stark's run#1-vs-run#2 test).
#
# Two IDENTICAL invocations of the submission serve.py at the official standard
# config (MAX_NUM_BATCHED_TOKENS=512), in two SEPARATE back-to-back vLLM
# processes, each decoded over HTTP via the official decode_outputs.py, then
# byte-compared with the official check_greedy_identity.py.
#
#   RUN#1 = serve.py cap=512  (fresh process) -> greedy_run1_decode.jsonl
#   RUN#2 = serve.py cap=512  (fresh process) -> greedy_run2_decode.jsonl
#
# Everything is held identical: same int4 checkpoint (same Marlin kernel on both
# sides), same seed, same prompts, same output_len, same env. The ONLY variable
# is "a second process run." If GREEDY_IDENTICAL, this int4 stack is run-to-run
# deterministic on this pod (stark's tied-bf16-head int4 base was 1/8 divergent).
#
# We ALSO compare RUN#1 against the already-committed greedy_cand_decode.jsonl
# (a serve.py cap=512 run captured hours earlier) as a longevity datapoint.
#
# LOCAL A10G only: flashinfer sampler JIT is broken on this pod, so both runs use
# VLLM_USE_FLASHINFER_SAMPLER=0 + VLLM_ATTENTION_BACKEND=FLASH_ATTN (greedy/argmax
# is sampler-independent; the backend is held identical, a controlled variable).
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
    sleep 5; t=$((t+5)); [ $t -ge $lim ] && { echo "TIMEOUT after ${t}s"; return 1; }
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

run_once() {
  local tag="$1"
  echo "=== RUN#$tag: submission serve.py, cap=512 (fresh process) ==="
  MODEL_ID=$CKPT SERVED_MODEL_NAME=gemma-4-e4b-it MAX_MODEL_LEN=4096 GPU_MEMORY_UTILIZATION=0.90 MAX_NUM_BATCHED_TOKENS=512 \
    setsid "$PY" "$ROOT/submissions/int4_g128_lmhead/serve.py" >"$LOG/greedy_run${tag}_server.log" 2>&1 &
  if ! wait_ready 600; then echo "RUN#$tag server failed"; tail -50 "$LOG/greedy_run${tag}_server.log"; free_gpu; exit 1; fi
  "$PY" "$DEC" --base-url http://localhost:8000 --model gemma-4-e4b-it --dataset-path "$DS" \
    --tokenizer "$CKPT" --num-prompts $NUM --output-len $OLEN --seed 1 \
    --output-file "$OUT/greedy_run${tag}_decode.jsonl" --summary-file "$OUT/greedy_run${tag}_summary.json" 2>&1 | tail -6
  free_gpu
}

run_once 1
run_once 2

echo "=== VERIFY A: run#1 vs run#2 (back-to-back, identical config) ==="
PYTHONPATH="$VER" "$PY" "$VER/check_greedy_identity.py" \
  --reference "$OUT/greedy_run1_decode.jsonl" \
  --candidate "$OUT/greedy_run2_decode.jsonl" --json | tee "$OUT/greedy_run2run_verdict.json"
echo "exit_code_A=${PIPESTATUS[0]}"

echo "=== VERIFY B: run#1 vs committed greedy_cand_decode.jsonl (hours-apart longevity) ==="
PYTHONPATH="$VER" "$PY" "$VER/check_greedy_identity.py" \
  --reference "$OUT/greedy_cand_decode.jsonl" \
  --candidate "$OUT/greedy_run1_decode.jsonl" --json | tee "$OUT/greedy_run1_vs_cand_verdict.json"
echo "exit_code_B=${PIPESTATUS[0]}"
echo "=== GREEDY RUN-TO-RUN CHECK DONE ==="
