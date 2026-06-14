#!/usr/bin/env bash
# PR #180 cheap smoke: control vs ARGMAX_ONLY_DECODE=1 token-identity + wall_tps.
# Both arms use the SAME throwaway submission (fa2sw_argmax_decode); only the
# candidate sets ARGMAX_ONLY_DECODE=1. Default workload 4 prompts x 64 tokens, n=1.
#
# Usage: run_smoke.sh [OUT_SUBDIR] [NUM_PROMPTS] [OUTPUT_LEN] [N]
set -euo pipefail
cd "$(dirname "$0")/../../.."   # -> target/

OUT_SUBDIR="${1:-smoke4}"
NUM_PROMPTS="${2:-4}"
OUTPUT_LEN="${3:-64}"
N="${4:-1}"

OUT_DIR="research/validity/argmax_decode_step/${OUT_SUBDIR}"
PY=.venv/bin/python

echo "[smoke] out=${OUT_DIR} workload=${NUM_PROMPTS}x${OUTPUT_LEN} n=${N}"

"$PY" scripts/profiler/paired_tps_ab.py \
  --baseline fa2sw_argmax_decode \
  --candidate fa2sw_argmax_decode \
  --candidate-env ARGMAX_ONLY_DECODE=1 \
  --baseline-label argmax_off \
  --candidate-label argmax_on \
  --n "$N" \
  --num-prompts "$NUM_PROMPTS" \
  --output-len "$OUTPUT_LEN" \
  --seed 1 \
  --out-dir "$OUT_DIR" \
  --no-wandb \
  --no-project

echo ""
echo "=== token identity diff (control vs patched) ==="
"$PY" research/validity/argmax_decode_step/token_identity_diff.py \
  "${OUT_DIR}/argmax_off/decode/run00.jsonl" \
  "${OUT_DIR}/argmax_on/decode/run00.jsonl" \
  --json-out "${OUT_DIR}/token_identity.json" || true
