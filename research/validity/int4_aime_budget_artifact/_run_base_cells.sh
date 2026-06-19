#!/usr/bin/env bash
set -uo pipefail
PY=/workspace/senpai/target/.venv/bin/python
AIME=/workspace/senpai/target/research/downstream_quality_aime/aime_eval.py
cd /workspace/senpai/target/research/validity/int4_aime_budget_artifact
for B in 6144 12288; do
  echo "[base2x] $(date -u +%H:%M:%SZ) base budget=$B START"
  $PY $AIME \
    --base-url http://127.0.0.1:8000 --model gemma-4-e4b-it \
    --years 2024,2025-I,2025-II --k 1 --seed 1234 \
    --temperature 0.0 --top-p 1.0 --top-k -1 \
    --max-tokens $B --min-tokens 8 --no-thinking \
    --client-concurrency 16 --save-text \
    --label base_greedy_$B --out results/base_greedy_$B.json 2>&1 | tail -3
  echo "[base2x] $(date -u +%H:%M:%SZ) base budget=$B DONE acc=$($PY -c "import json;print(json.load(open('results/base_greedy_$B.json'))['maj_k_accuracy'])" 2>/dev/null)"
done
echo "[base2x] ALL DONE $(date -u +%H:%M:%SZ)"
