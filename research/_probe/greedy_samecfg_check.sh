#!/usr/bin/env bash
# Actual-gate-config greedy-identity check: reference and candidate BOTH use the
# official standard serving config (MAX_NUM_BATCHED_TOKENS=512, the baseline
# manifest / README "default env"). The organizer generates the exact-greedy
# reference by honestly decoding the submitted checkpoint with this standard
# config; the candidate is the participant serve.py with the same config. If they
# match, the submission passes the official greedy-identity gate.
#
#   REFERENCE = plain vLLM, cap=512        (organizer's honest standard-config decode)
#   CANDIDATE = submission serve.py, cap=512 (already captured: greedy_cand_decode.jsonl)
#
# The no-cap reference diverged on 53/128 prompts purely because changing the
# prefill chunk size perturbs near-tie argmax in this int4 model -- that is NOT
# the gate config. This run isolates the standard config on both sides.
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

echo "=== REFERENCE: plain vLLM, cap=512 (standard config) ==="
setsid "$PY" -m vllm.entrypoints.openai.api_server --model "$CKPT" --served-model-name gemma-4-e4b-it \
  --host 0.0.0.0 --port 8000 --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 512 --trust-remote-code --no-enable-log-requests >"$LOG/greedy_refcap_server.log" 2>&1 &
if ! wait_ready 600; then echo "REF server failed"; tail -50 "$LOG/greedy_refcap_server.log"; free_gpu; exit 1; fi
"$PY" "$DEC" --base-url http://localhost:8000 --model gemma-4-e4b-it --dataset-path "$DS" \
  --tokenizer "$CKPT" --num-prompts $NUM --output-len $OLEN --seed 1 \
  --output-file "$OUT/greedy_ref_cap512_decode.jsonl" --summary-file "$OUT/greedy_ref_cap512_summary.json" 2>&1 | tail -6
free_gpu

echo "=== VERIFY (same-config): reference cap512 vs candidate cap512 ==="
PYTHONPATH="$VER" "$PY" "$VER/check_greedy_identity.py" \
  --reference "$OUT/greedy_ref_cap512_decode.jsonl" \
  --candidate "$OUT/greedy_cand_decode.jsonl" --json | tee "$OUT/greedy_samecfg_verdict.json"
echo "exit_code=${PIPESTATUS[0]}"
echo "=== GREEDY SAMECFG CHECK DONE ==="
