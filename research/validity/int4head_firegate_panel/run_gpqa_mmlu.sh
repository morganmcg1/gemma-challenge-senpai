#!/usr/bin/env bash
# int4head (#795) GPQA-Diamond 5-seed + MMLU-Pro panel against the already-running
# local int4head server on :8020. Byte-for-byte the bi0 reference protocol
# (#773 GPQA + #762 MMLU), only the served model differs (int4 g32 lm_head vs bf16
# lm_head; body byte-identical):
#   GPQA: full 198 x 5 choice-shuffle seeds, T=1 top_p=0.95 top_k=64,
#         max_tokens=6144, sampling_seed=0.  Gate: pooled >= 0.471 (bi0 0.4970).
#   MMLU-Pro: n=250 seed=12345 T=1 top_p=0.95 top_k=64 min_tokens=8 sampling_seed=0.
#         max_tokens=4096 (clean, PR-mandated; bi0 0.644 was at 2048 w/ 11.2% trunc)
#         AND 2048 (exact like-for-like vs bi0 0.644).  Gate: >= 0.572.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EVAL_PY=/tmp/eval-serve-venv/bin/python
RUN_EVAL="$HERE/../downstream_quality_eval/run_eval.py"
RES="$HERE/results"
STATUS="$HERE/_panel.status"
BASE_URL="http://127.0.0.1:8020/v1"
ARM=int4head
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

# ---- MMLU-Pro n=250: 4096 (clean) then 2048 (like-for-like vs bi0 0.644) ----
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

echo "PANEL DONE $(date -u +%FT%TZ)" | tee -a "$STATUS"
