#!/usr/bin/env bash
# PR #814 Step-3 g128 quality panel, multi-stream axes against the already-running
# g128 server on :8021. Byte-for-byte the #795 int4head protocol (only the served
# body group_size differs: g128 vs g32), so g128-vs-int4head is a clean isolation
# of body group-size coarsening on downstream quality.
#   GPQA-Diamond: full 198 x 5 choice-shuffle seeds (SAME seeds as #795 -> seed-
#       paired), T=1 top_p=0.95 top_k=64, max_tokens=6144, sampling_seed=0.
#       Gate: pooled >= 0.471.  #795 int4head ref 0.5030.
#   MMLU-Pro: n=250 seed=12345 T=1 top_p=0.95 top_k=64 min_tokens=8 sampling_seed=0,
#       max_tokens=4096 (clean gate) AND 2048 (like-for-like vs #795 0.6040).
#       Gate: >= 0.572.  #795 int4head ref 0.6920 (@4096).
#   GSM8K: n=300 8-shot CoT, sampled (PRIMARY, lewtun #31) + greedy (diagnostic),
#       T=1/top_p=0.95/top_k=64, max_tokens=512, min_tokens=8, sampling_seed=0,
#       concurrency=16.  Gate: >= 0.807.  #795/#788 int4head greedy ref 0.915.
# AIME is run separately (run_aime_g128.sh) AFTER this -- its greedy maj@1 arm is
# single-stream (concurrency=1) and must not overlap this 16-way load.
# LOCAL ONLY -- no HF Job.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EVAL_PY=/tmp/eval-serve-venv/bin/python
RUN_EVAL="$HERE/../../validity/downstream_quality_eval/run_eval.py"
GSM8K_EVAL="$HERE/../../downstream_quality_gsm8k/gsm8k_eval.py"
RES="$HERE/results"
STATUS="$HERE/_panel.status"
BASE_URL="http://127.0.0.1:8021/v1"
GSM_BASE_URL="http://127.0.0.1:8021"
ARM=g128_body
mkdir -p "$RES"

GPQA_SEEDS=(12345 13579 23456 34567 45678)

echo "PANEL START $(date -u +%FT%TZ)" | tee "$STATUS"

# ---- GPQA-Diamond, full 198/seed ----
for s in "${GPQA_SEEDS[@]}"; do
  out="$RES/${ARM}_gpqa_s${s}.json"
  echo "===== gpqa_diamond seed=$s $(date -u +%H:%M:%SZ) =====" | tee -a "$STATUS"
  "$EVAL_PY" "$RUN_EVAL" \
    --task gpqa_diamond --arm "$ARM" \
    --out "$out" \
    --base-url "$BASE_URL" --model gemma-4-e4b-it \
    --temperature 1.0 --top-p 0.95 --top-k 64 \
    --max-tokens 6144 --sampling-seed 0 --seed "$s" \
    --max-connections 16 \
    >>"$HERE/_gpqa_s${s}.out" 2>&1
  rc=$?
  if [ -f "$out" ]; then
    acc=$("$EVAL_PY" -c "import json;d=json.load(open('$out'));print(f\"acc={d['accuracy']:.4f} ({d['n_correct']}/{d['n_scored']}) err={d['n_error']} trunc={d['n_length_truncated']} modlen={d['n_stop_model_length']} ctok_mean={d['completion_tokens_mean']:.0f} ctok_p95={d['completion_tokens_p95']}\")" 2>/dev/null)
    echo "  gpqa seed=$s rc=$rc $acc" | tee -a "$STATUS"
  else
    echo "  gpqa seed=$s rc=$rc NO_OUTPUT (see _gpqa_s${s}.out)" | tee -a "$STATUS"
  fi
done

# ---- MMLU-Pro n=250: 4096 (clean gate) then 2048 (like-for-like vs #795) ----
for mt in 4096 2048; do
  mout="$RES/${ARM}_mmlu_n250_mt${mt}_s12345.json"
  echo "===== mmlu_pro n=250 max_tokens=$mt seed=12345 $(date -u +%H:%M:%SZ) =====" | tee -a "$STATUS"
  "$EVAL_PY" "$RUN_EVAL" \
    --task mmlu_pro --arm "$ARM" \
    --out "$mout" \
    --base-url "$BASE_URL" --model gemma-4-e4b-it \
    --n 250 --temperature 1.0 --top-p 0.95 --top-k 64 \
    --max-tokens "$mt" --min-tokens 8 --sampling-seed 0 --seed 12345 \
    --max-connections 16 \
    >>"$HERE/_mmlu_mt${mt}.out" 2>&1
  rc=$?
  if [ -f "$mout" ]; then
    acc=$("$EVAL_PY" -c "import json;d=json.load(open('$mout'));print(f\"acc={d['accuracy']:.4f} ({d['n_correct']}/{d['n_scored']}) err={d['n_error']} trunc={d['n_length_truncated']} len@2048={d.get('finish_length_rate_at_2048')} ctok_mean={d['completion_tokens_mean']:.0f}\")" 2>/dev/null)
    echo "  mmlu n=250 mt=$mt rc=$rc $acc" | tee -a "$STATUS"
  else
    echo "  mmlu n=250 mt=$mt rc=$rc NO_OUTPUT (see _mmlu_mt${mt}.out)" | tee -a "$STATUS"
  fi
done

# ---- GSM8K n=300, sampled (PRIMARY) + greedy (diagnostic) ----
echo "===== gsm8k n=300 sampled+greedy $(date -u +%H:%M:%SZ) =====" | tee -a "$STATUS"
"$EVAL_PY" "$GSM8K_EVAL" \
  --base-url "$GSM_BASE_URL" --model gemma-4-e4b-it \
  --label "$ARM" --regimes sampled,greedy --n 300 \
  --temperature 1.0 --top-p 0.95 --top-k 64 \
  --max-tokens 512 --min-tokens 8 --sampling-seed 0 --seed 1234 \
  --concurrency 16 --max-num-seqs 16 \
  --out-dir "$RES" \
  >>"$HERE/_gsm8k.out" 2>&1
rc=$?
for reg in sampled greedy; do
  gout="$RES/${ARM}_${reg}_s0.json"
  if [ -f "$gout" ]; then
    v=$("$EVAL_PY" -c "import json;d=json.load(open('$gout'));print(f\"acc={d['accuracy']:.4f} ({d['n_correct']}/{d['n_scored']}) trunc={d.get('n_length_truncated','?')}\")" 2>/dev/null)
    echo "  gsm8k $reg rc=$rc $v" | tee -a "$STATUS"
  else
    echo "  gsm8k $reg rc=$rc NO_OUTPUT (see _gsm8k.out)" | tee -a "$STATUS"
  fi
done

echo "PANEL DONE $(date -u +%FT%TZ)" | tee -a "$STATUS"
