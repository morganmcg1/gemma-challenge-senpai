#!/usr/bin/env bash
# Capture decode_outputs.jsonl from a running endpoint using the official harness.
# Usage: capture_decode.sh <base-url> <model> <out-prefix> <num-prompts>
set -euo pipefail

BASE_URL="${1:?base-url}"
MODEL="${2:?model}"
OUT_PREFIX="${3:?out-prefix}"
NUM_PROMPTS="${4:-16}"

ROOT="/workspace/senpai/target"
SB="$ROOT/official/main_bucket/shared_resources/speed_benchmark"
PY="$ROOT/.venvs/vllm022/bin/python"

"$PY" "$SB/scripts/decode_outputs.py" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --dataset-path "$SB/data/eval_prompts_sharegpt.json" \
  --output-file "${OUT_PREFIX}.jsonl" \
  --summary-file "${OUT_PREFIX}.summary.json" \
  --num-prompts "$NUM_PROMPTS" \
  --output-len 512 \
  --seed 1
echo "wrote ${OUT_PREFIX}.jsonl"
