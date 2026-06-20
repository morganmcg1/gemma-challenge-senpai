#!/usr/bin/env bash
# PR #814 Step-3 g128 AIME panel against the already-running g128 server on :8021.
# Run AFTER run_panel_g128.sh (greedy maj@1 is single-stream, must not overlap the
# 16-way panel load). Byte-for-byte the #795 int4head AIME protocol.
#   PRIMARY (#580): greedy maj@1, n=30 (years=2024), temp=0 top_p=1 top_k=-1,
#     max_tokens=3072, min_tokens=8, no-thinking, seed 1234, concurrency 1.
#     Gate: >= 0.090 (3/30).  #795 int4head greedy ref 9/30=0.300.
#   SUPPLEMENT (like-for-like vs #795 12/30=0.400): sampled maj@8, T=1 top_p=0.95
#     top_k=64, max_tokens=3072, min_tokens=8, no-thinking, seed 1234, concurrency 16.
# LOCAL ONLY -- no HF Job.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EVAL_PY=/tmp/eval-serve-venv/bin/python
AIME="$HERE/../../downstream_quality_aime/aime_eval.py"
RES="$HERE/results"
STATUS="$HERE/_aime.status"
BASE_URL="http://127.0.0.1:8021"
ARM=g128_body
mkdir -p "$RES"

echo "AIME START $(date -u +%FT%TZ)" | tee "$STATUS"

# ---- PRIMARY: greedy maj@1, n=30 ----
echo "===== aime greedy maj@1 n=30 $(date -u +%H:%M:%SZ) =====" | tee -a "$STATUS"
"$EVAL_PY" "$AIME" --base-url "$BASE_URL" --model gemma-4-e4b-it \
  --label ${ARM}_aime_greedy --years 2024 --k 1 \
  --temperature 0.0 --top-p 1.0 --top-k -1 \
  --max-tokens 3072 --min-tokens 8 --seed 1234 --no-thinking \
  --client-concurrency 1 --save-text \
  --out "$RES/${ARM}_aime_greedy_n30.json" \
  >>"$HERE/_aime_greedy.out" 2>&1
rc=$?
g="$RES/${ARM}_aime_greedy_n30.json"
if [ -f "$g" ]; then
  v=$("$EVAL_PY" -c "import json;d=json.load(open('$g'));tr=sum(1 for p in d['per_problem'] for f in p['finish_reasons'] if f=='length');print(f\"maj@1={d['maj_k_accuracy']:.4f} ({d['n_correct_maj']}/{d['n_problems']}) extract_fail={d['extract_fail_rate']:.3f} length_trunc={tr}/{d['total_samples']}\")" 2>/dev/null)
  echo "  aime greedy rc=$rc $v" | tee -a "$STATUS"
else
  echo "  aime greedy rc=$rc NO_OUTPUT" | tee -a "$STATUS"
fi

# ---- SUPPLEMENT: sampled maj@8, n=30 ----
echo "===== aime sampled maj@8 n=30 $(date -u +%H:%M:%SZ) =====" | tee -a "$STATUS"
"$EVAL_PY" "$AIME" --base-url "$BASE_URL" --model gemma-4-e4b-it \
  --label ${ARM}_aime_sampled --years 2024 --k 8 \
  --temperature 1.0 --top-p 0.95 --top-k 64 \
  --max-tokens 3072 --min-tokens 8 --seed 1234 --no-thinking \
  --client-concurrency 16 --save-text \
  --out "$RES/${ARM}_aime_sampled_maj8_n30.json" \
  >>"$HERE/_aime_sampled.out" 2>&1
rc=$?
sout="$RES/${ARM}_aime_sampled_maj8_n30.json"
if [ -f "$sout" ]; then
  v=$("$EVAL_PY" -c "import json;d=json.load(open('$sout'));print(f\"maj@8={d['maj_k_accuracy']:.4f} ({d['n_correct_maj']}/{d['n_problems']}) mean_pass={d['mean_pass_rate']:.4f} extract_fail={d['extract_fail_rate']:.3f}\")" 2>/dev/null)
  echo "  aime sampled rc=$rc $v" | tee -a "$STATUS"
else
  echo "  aime sampled rc=$rc NO_OUTPUT" | tee -a "$STATUS"
fi

echo "AIME DONE $(date -u +%FT%TZ)" | tee -a "$STATUS"
