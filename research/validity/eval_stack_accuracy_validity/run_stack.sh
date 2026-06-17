#!/usr/bin/env bash
# PR #615 -- run the 3-eval head-to-head accuracy panel against an ALREADY-RUNNING
# server on :8000 (the shipped int4_g128_lmhead, served by serve_ship.sh on a chosen
# vLLM build). The server determines the stack; this driver is stack-agnostic.
#
# Identical decode protocol on every stack (lewtun #31 + #541 + ubel #590 guards):
#   T=1.0 top_p=0.95 top_k=64 min_tokens=8 ; CoT evals max_tokens=4096 (no truncation
#   at max_model_len=6144); GSM8K max_tokens=1024 (short CoT, logged finish_reason).
#
# Usage: run_stack.sh <stack_label> <mode: smoke|full>
set -u
STACK="${1:?stack label e.g. v0220 or dev307}"
MODE="${2:-full}"
TASKS="${3:-gpqa,mmlu,gsm8k}"     # comma list subset of {gpqa,mmlu,gsm8k}
want() { [[ ",$TASKS," == *",$1,"* ]]; }
cd /workspace/senpai/target

VENV=/senpai-run/home/student-lawine/eval-client-venv
PYI="$VENV/bin/python"            # inspect_ai client for mmlu_pro / gpqa
PYS=/usr/bin/python3              # stdlib urllib client for gsm8k
RUN_EVAL=research/validity/downstream_quality_eval/run_eval.py
GSM8K=research/downstream_quality_gsm8k/gsm8k_eval.py
URL=http://127.0.0.1:8000/v1
URL0=http://127.0.0.1:8000
OUT=research/validity/eval_stack_accuracy_validity/runs
LOGD=research/validity/eval_stack_accuracy_validity/logs
mkdir -p "$OUT" "$LOGD"

export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME=/senpai-run/home/student-lawine/.cache/huggingface

if [[ "$MODE" == "smoke" ]]; then
  GPQA_SEEDS=(1); MMLU_SEEDS=(1); GSM_SEEDS=(1); LIM="--limit 8"; MMLU_N=64; GSM_N=64
else
  GPQA_SEEDS=(1 2 3 4 5); MMLU_SEEDS=(1 2 3); GSM_SEEDS=(1 2 3); LIM=""; MMLU_N=200; GSM_N=200
fi

echo "[panel:$STACK] START mode=$MODE $(date -u +%FT%TZ) gpqa_seeds=${GPQA_SEEDS[*]} mmlu_seeds=${MMLU_SEEDS[*]} gsm_seeds=${GSM_SEEDS[*]}"
t0=$(date +%s); fail=0

# ---- GPQA-Diamond (priority): full n=198, fixed choice layout seed=12345 ----
if want gpqa; then
for s in "${GPQA_SEEDS[@]}"; do
  o="$OUT/gpqa_${STACK}_s${s}.json"; ld="$LOGD/gpqa_${STACK}_s${s}"
  if [[ -f "$o" ]] && "$PYS" -c "import json,sys;d=json.load(open('$o'));sys.exit(0 if d.get('n_scored',0)>=190 else 1)" 2>/dev/null; then
    echo "[panel:$STACK] gpqa s$s SKIP (complete)"; continue; fi
  echo "[panel:$STACK] gpqa s$s $(date -u +%H:%M:%SZ)"; ts=$(date +%s)
  if "$PYI" "$RUN_EVAL" --task gpqa_diamond --arm "ship_${STACK}" $LIM \
      --seed 12345 --sampling-seed "$s" \
      --temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 8 --max-tokens 4096 \
      --max-connections 32 --base-url "$URL" --out "$o" --log-dir "$ld" \
      >"$LOGD/gpqa_${STACK}_s${s}.log" 2>&1; then
    echo "[panel:$STACK]   gpqa s$s OK wall=$(( $(date +%s)-ts ))s $(grep -h 'run_eval] task=' "$LOGD/gpqa_${STACK}_s${s}.log" | tail -1)"
  else echo "[panel:$STACK]   gpqa s$s FAIL"; tail -6 "$LOGD/gpqa_${STACK}_s${s}.log"; fail=1; fi
done
fi

# ---- MMLU-Pro (confirm): n=MMLU_N, fixed subset seed=12345 ----
if want mmlu; then
for s in "${MMLU_SEEDS[@]}"; do
  o="$OUT/mmlu_${STACK}_s${s}.json"; ld="$LOGD/mmlu_${STACK}_s${s}"
  if [[ -f "$o" ]] && "$PYS" -c "import json,sys;d=json.load(open('$o'));sys.exit(0 if d.get('n_scored',0)>=$((MMLU_N-10)) else 1)" 2>/dev/null; then
    echo "[panel:$STACK] mmlu s$s SKIP (complete)"; continue; fi
  echo "[panel:$STACK] mmlu s$s $(date -u +%H:%M:%SZ)"; ts=$(date +%s)
  if "$PYI" "$RUN_EVAL" --task mmlu_pro --arm "ship_${STACK}" $LIM \
      --n "$MMLU_N" --seed 12345 --sampling-seed "$s" \
      --temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 8 --max-tokens 4096 \
      --max-connections 32 --base-url "$URL" --out "$o" --log-dir "$ld" \
      >"$LOGD/mmlu_${STACK}_s${s}.log" 2>&1; then
    echo "[panel:$STACK]   mmlu s$s OK wall=$(( $(date +%s)-ts ))s $(grep -h 'run_eval] task=' "$LOGD/mmlu_${STACK}_s${s}.log" | tail -1)"
  else echo "[panel:$STACK]   mmlu s$s FAIL"; tail -6 "$LOGD/mmlu_${STACK}_s${s}.log"; fail=1; fi
done
fi

# ---- GSM8K (confirm): n=GSM_N 8-shot CoT, fixed subset+fewshot seed=1234 ----
if want gsm8k; then
for s in "${GSM_SEEDS[@]}"; do
  lab="gsm8k_${STACK}"; o="$OUT/${lab}_sampled_s${s}.json"
  if [[ -f "$o" ]] && "$PYS" -c "import json,sys;d=json.load(open('$o'));sys.exit(0 if len(d.get('per_problem',[]))>=$((GSM_N-10)) else 1)" 2>/dev/null; then
    echo "[panel:$STACK] gsm8k s$s SKIP (complete)"; continue; fi
  echo "[panel:$STACK] gsm8k s$s $(date -u +%H:%M:%SZ)"; ts=$(date +%s)
  if "$PYS" "$GSM8K" --base-url "$URL0" --label "$lab" --regimes sampled \
      --n "$GSM_N" --seed 1234 --sampling-seed "$s" --n-shot 8 \
      --top-p 0.95 --top-k 64 --max-tokens 1024 --min-tokens 8 \
      --concurrency 32 --out-dir "$OUT" \
      >"$LOGD/${lab}_s${s}.log" 2>&1; then
    echo "[panel:$STACK]   gsm8k s$s OK wall=$(( $(date +%s)-ts ))s $(grep -h 'DONE' "$LOGD/${lab}_s${s}.log" | tail -1)"
  else echo "[panel:$STACK]   gsm8k s$s FAIL"; tail -6 "$LOGD/${lab}_s${s}.log"; fail=1; fi
done
fi

echo "[panel:$STACK] DONE wall=$(( $(date +%s)-t0 ))s $(date -u +%FT%TZ) fail=$fail"
exit $fail
