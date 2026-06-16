#!/usr/bin/env bash
set -uo pipefail
cd /workspace/senpai/target/research/validity/downstream_quality_eval
VENV=/tmp/eval-serve-venv
echo "[ship-full] START $(date -u +%H:%M:%SZ)"
echo "[ship-full] === MMLU-Pro n=500 seed=12345 max_tokens=2048 ==="
"$VENV/bin/python" run_eval.py --task mmlu_pro --arm ship --out ship_mmlu_pro.json \
  --n 500 --seed 12345 --max-tokens 2048 --max-connections 16 \
  --base-url http://127.0.0.1:8000/v1 --model gemma-4-e4b-it
echo "[ship-full] MMLU done rc=$? $(date -u +%H:%M:%SZ)"
echo "[ship-full] === GPQA-Diamond full=198 seed=12345 max_tokens=3072 ==="
"$VENV/bin/python" run_eval.py --task gpqa_diamond --arm ship --out ship_gpqa.json \
  --seed 12345 --max-tokens 3072 --max-connections 16 \
  --base-url http://127.0.0.1:8000/v1 --model gemma-4-e4b-it
echo "[ship-full] GPQA done rc=$? $(date -u +%H:%M:%SZ)"
echo "[ship-full] ALL DONE $(date -u +%H:%M:%SZ)"
