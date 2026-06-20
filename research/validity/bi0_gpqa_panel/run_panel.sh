#!/usr/bin/env bash
# bi0 GPQA-Diamond multi-seed + MMLU-Pro sanity, against an already-running
# local bi0 server on :8000. Mirrors the specdec_gpqa_10seed (#652) protocol so
# numbers are directly comparable: Convention A = vary dataset --seed
# (choice-shuffle), fix --sampling-seed 0; GPQA full 198, T=1/top_p=0.95/top_k=64,
# max_tokens 6144 (#619 anti-truncation). MMLU-Pro n=100 same sampling, max_tokens 2048.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EVAL_PY=/tmp/eval-serve-venv/bin/python
RUN_EVAL="$HERE/../downstream_quality_eval/run_eval.py"
RES="$HERE/results"
STATUS="$HERE/_panel.status"
BASE_URL="http://127.0.0.1:8000/v1"
mkdir -p "$RES"

GPQA_SEEDS=(12345 13579 23456 34567 45678)

echo "PANEL START $(date -u +%FT%TZ)" | tee "$STATUS"

# ---- GPQA-Diamond, full 198/seed ----
for s in "${GPQA_SEEDS[@]}"; do
  out="$RES/bi0_gpqa_s${s}.json"
  echo "===== gpqa_diamond seed=$s $(date -u +%H:%M:%SZ) =====" | tee -a "$STATUS"
  "$EVAL_PY" "$RUN_EVAL" \
    --task gpqa_diamond --arm int4_mtp_bi0_surgattn \
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

# ---- MMLU-Pro n=100 sanity ----
echo "===== mmlu_pro n=100 seed=12345 $(date -u +%H:%M:%SZ) =====" | tee -a "$STATUS"
"$EVAL_PY" "$RUN_EVAL" \
  --task mmlu_pro --arm int4_mtp_bi0_surgattn \
  --out "$RES/bi0_mmlu_n100_s12345.json" \
  --base-url "$BASE_URL" --model gemma-4-e4b-it \
  --n 100 --temperature 1.0 --top-p 0.95 --top-k 64 \
  --max-tokens 2048 --sampling-seed 0 --seed 12345 \
  --max-connections 16 \
  >>"$HERE/_mmlu_n100.out" 2>&1
rc=$?
mout="$RES/bi0_mmlu_n100_s12345.json"
if [ -f "$mout" ]; then
  acc=$("$EVAL_PY" -c "import json;d=json.load(open('$mout'));print(f\"acc={d['accuracy']:.4f} ({d['n_correct']}/{d['n_scored']}) err={d['n_error']} trunc={d['n_length_truncated']}\")" 2>/dev/null)
  echo "  mmlu n=100 rc=$rc $acc" | tee -a "$STATUS"
else
  echo "  mmlu n=100 rc=$rc NO_OUTPUT (see _mmlu_n100.out)" | tee -a "$STATUS"
fi

echo "PANEL DONE $(date -u +%FT%TZ)" | tee -a "$STATUS"
