#!/usr/bin/env bash
set -euo pipefail
HERE=/workspace/senpai/target/research/validity/int4_aime_budget_artifact
PORT=8001
bash "$HERE/serve_body.sh" /workspace/gemma_build/int4_g128_lmhead "$PORT" 13312 16
SRV=$(cat "$HERE/_server_${PORT}.pid")
trap 'kill $SRV 2>/dev/null||true; sleep 2; kill -9 $SRV 2>/dev/null||true' EXIT
nvidia-smi --query-gpu=memory.used --format=csv,nounits,nounits | tail -1
/workspace/senpai/target/.venv/bin/python /workspace/senpai/target/research/downstream_quality_aime/aime_eval.py \
  --base-url "http://127.0.0.1:${PORT}" --model gemma-4-e4b-it \
  --years 2024 --limit 2 --k 16 --seed 1234 \
  --temperature 1.0 --top-p 0.95 --top-k 64 --max-tokens 12288 --min-tokens 8 --no-thinking \
  --client-concurrency 1 --label smoke --save-text \
  --out "$HERE/_smoke/int4_smoke.json"
echo "PEAK_GPU_MIB:"; nvidia-smi --query-gpu=memory.used --format=csv,nounits | tail -1
