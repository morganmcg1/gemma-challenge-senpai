#!/usr/bin/env bash
# run_arm.sh <arm-label> [min_tokens] : MMLU-Pro(500) then GPQA-Diamond(198),
#   greedy, against localhost:8000. min_tokens=0 (default) = as-served; >0 applies
#   the EOS-guard floor (wirbel #541) via run_eval's --min-tokens.
set -euo pipefail
ARM="$1"
MINTOK="${2:-0}"
PY=/tmp/eval-serve-venv/bin/python
RE=../downstream_quality_eval/run_eval.py
echo "[run_arm] $ARM MMLU-Pro n=500 min_tokens=$MINTOK start $(date -u +%H:%M:%SZ)"
$PY "$RE" --task mmlu_pro --arm "$ARM" --out "${ARM}_mmlu_pro.json" \
  --n 500 --seed 12345 --max-tokens 2048 --min-tokens "$MINTOK" \
  --base-url http://127.0.0.1:8000/v1 --model gemma-4-e4b-it
echo "[run_arm] $ARM GPQA-Diamond n=198 min_tokens=$MINTOK start $(date -u +%H:%M:%SZ)"
$PY "$RE" --task gpqa_diamond --arm "$ARM" --out "${ARM}_gpqa.json" \
  --seed 12345 --max-tokens 3072 --min-tokens "$MINTOK" \
  --base-url http://127.0.0.1:8000/v1 --model gemma-4-e4b-it
echo "[run_arm] $ARM DONE $(date -u +%H:%M:%SZ)"
