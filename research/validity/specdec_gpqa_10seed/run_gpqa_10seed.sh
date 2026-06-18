#!/usr/bin/env bash
# PR #656 — GPQA-Diamond sampled, one arm, N seeds, against the already-running
# server on :8000 (serve_arm.py). Single-variable AR-vs-spec contrast: identical
# sampling (T=1.0 top_p=0.95 top_k=64, lewtun #31), max-tokens 6144 (gb6144 clean
# budget, matches #634: zero length-truncation), min-tokens 8 (#541 EOS-guard),
# max-connections 16. Each seed = full GPQA-Diamond n=198; 10 seeds pool to 1980.
#
# Usage: run_gpqa_10seed.sh <arm-label> <seed1> [seed2 ...]
#   arm-label: "spec_k6" or "ar_m1" (only labels result files; server is whatever
#   is on :8000 — caller must start the matching serve_arm.py first).
set -uo pipefail
cd /workspace/senpai/target
PY=/tmp/eval-serve-venv/bin/python
DIR=research/validity/specdec_gpqa_10seed
RES=$DIR/results
mkdir -p "$RES"
BASE=http://127.0.0.1:8000
MODEL=gemma-4-e4b-it
MT=6144
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"  # local vLLM ignores the value

ARM="$1"; shift
SEEDS=("$@")
STATUS=$DIR/_run_${ARM}.status
: > "$STATUS"
echo "ARM=$ARM START $(date -u +%FT%TZ) seeds=${SEEDS[*]}" | tee -a "$STATUS"

for SEED in "${SEEDS[@]}"; do
  OUT="$RES/gpqa_${ARM}_s${SEED}.json"
  if [ -f "$OUT" ]; then
    echo "  [skip] $ARM seed=$SEED already done ($OUT)" | tee -a "$STATUS"
    continue
  fi
  echo "===== $ARM GPQA-Diamond seed=$SEED $(date -u +%H:%M:%S) =====" | tee -a "$STATUS"
  $PY research/validity/downstream_quality_eval/run_eval.py \
    --task gpqa_diamond --arm "$ARM" --out "$OUT" \
    --seed "$SEED" --max-tokens "$MT" --max-connections 16 \
    --temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 8 \
    --base-url "$BASE/v1" --model "$MODEL" \
    > "$DIR/_gpqa_${ARM}_s${SEED}.out" 2>&1
  rc=$?
  echo "  $ARM seed=$SEED rc=$rc $(date -u +%H:%M:%S)" | tee -a "$STATUS"
  if [ -f "$OUT" ]; then
    $PY -c "import json;d=json.load(open('$OUT'));print(f\"    acc={d['accuracy']:.4f} ({d['n_correct']}/{d['n_scored']}) err={d.get('n_error',0)} trunc={d.get('n_length',0)}\")" | tee -a "$STATUS"
  fi
done
echo "ARM=$ARM DONE $(date -u +%FT%TZ)" | tee -a "$STATUS"
