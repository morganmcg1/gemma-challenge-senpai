#!/usr/bin/env bash
# PR #814 Step-3 g128 panel RESUME orchestrator. The original run_all_g128.sh was
# interrupted mid-GPQA-seed-34567 (seeds 12345/13579/23456 already saved). This
# resumes the remaining axes against the still-running g128 server on :8021 using
# the IDENTICAL #795 protocol/flags, so g128-vs-int4head stays a clean body
# group-size isolation. Remaining: GPQA seeds 34567+45678, MMLU-Pro @4096&2048,
# GSM8K sampled+greedy, then AIME (greedy maj@1 + sampled maj@8). LOCAL ONLY.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EVAL_PY=/tmp/eval-serve-venv/bin/python
RUN_EVAL="$HERE/../../validity/downstream_quality_eval/run_eval.py"
GSM8K_EVAL="$HERE/../../downstream_quality_gsm8k/gsm8k_eval.py"
RES="$HERE/results"
STATUS="$HERE/_resume.status"
BASE_URL="http://127.0.0.1:8021/v1"
GSM_BASE_URL="http://127.0.0.1:8021"
ARM=g128_body
mkdir -p "$RES"

echo "RESUME START $(date -u +%FT%TZ)" | tee "$STATUS"

# ---- wait for server ready (cap ~5 min) ----
ready=0
for _ in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:8021/v1/models" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then ready=1; break; fi
  sleep 5
done
if [ "$ready" -ne 1 ]; then
  echo "RESUME ABORT: server not ready (last code=$code) $(date -u +%FT%TZ)" | tee -a "$STATUS"
  exit 1
fi
echo "RESUME: server ready $(date -u +%FT%TZ)" | tee -a "$STATUS"

# ---- GPQA-Diamond, remaining seeds 34567 + 45678 (full 198/seed) ----
for s in 34567 45678; do
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
    acc=$("$EVAL_PY" -c "import json;d=json.load(open('$out'));print(f\"acc={d['accuracy']:.4f} ({d['n_correct']}/{d['n_scored']}) err={d['n_error']} trunc={d['n_length_truncated']} ctok_mean={d['completion_tokens_mean']:.0f}\")" 2>/dev/null)
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
    acc=$("$EVAL_PY" -c "import json;d=json.load(open('$mout'));print(f\"acc={d['accuracy']:.4f} ({d['n_correct']}/{d['n_scored']}) err={d['n_error']} trunc={d['n_length_truncated']} ctok_mean={d['completion_tokens_mean']:.0f}\")" 2>/dev/null)
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
echo "RESUME: panel finished $(date -u +%FT%TZ)" | tee -a "$STATUS"

# ---- AIME (greedy maj@1 single-stream + sampled maj@8) ----
bash "$HERE/run_aime_g128.sh"
echo "RESUME ALL DONE $(date -u +%FT%TZ)" | tee -a "$STATUS"
