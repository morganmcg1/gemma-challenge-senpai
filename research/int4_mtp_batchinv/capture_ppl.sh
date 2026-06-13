#!/usr/bin/env bash
# Measure corpus PPL against the official ground-truth tokens via the endpoint.
# Usage: capture_ppl.sh <base-url> <model> <out-prefix>
set -euo pipefail
BASE_URL="${1:?base-url}"
MODEL="${2:?model}"
OUT_PREFIX="${3:?out-prefix}"

ROOT="/workspace/senpai/target"
SB="$ROOT/official/main_bucket/shared_resources/speed_benchmark"
PY="$ROOT/.venvs/vllm022/bin/python"

"$PY" "$SB/scripts/ppl_endpoint.py" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --dataset-path "$SB/data/ppl_ground_truth_tokens.jsonl" \
  --output-file "${OUT_PREFIX}.jsonl" \
  --summary-file "${OUT_PREFIX}.summary.json"
