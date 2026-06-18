#!/usr/bin/env bash
# Spec-break quality-materiality census (PR #682, wirbel) -- run the FULL panel
# for ONE arm against the already-running arm server at http://127.0.0.1:8000.
#
# Usage: run_arm.sh <arm: ar|spec>
#
# Runs the 4 quality legs SEQUENTIALLY (each internally conc=16) so the batching
# regime is identical between arms, then captures the canonical strict-#319 token
# stream (128 sharegpt x512 greedy, decode_outputs.py) for the break-rate diff.
# Background this (it runs many minutes); each leg stays well under the cap.
set -uo pipefail
cd /workspace/senpai/target
ARM="${1:?arm: ar|spec}"
DIR=research/validity/specbreak_quality_materiality
RES="$DIR/results"
CAP_VENV=/tmp/senpai-venvs/a341b8bdf5ec1fe0/bin/python
BASE=http://127.0.0.1:8000
MODEL=gemma-4-e4b-it
TS() { date -u +%H:%M:%S; }

echo "########## ARM=$ARM full panel start $(TS) ##########"
for LEG in gsm8k mmlu gpqa aime; do
  echo "===== [$ARM] leg=$LEG $(TS) ====="
  bash "$DIR/run_leg.sh" "$ARM" "$LEG"
  echo "===== [$ARM] leg=$LEG done $(TS) ====="
done

echo "===== [$ARM] token-break capture (canonical strict-#319 config) $(TS) ====="
"$CAP_VENV" "$DIR/token_break_probe.py" capture --arm "$ARM" \
  --base-url "$BASE" --model "$MODEL" --out "$RES/token_${ARM}.jsonl"
echo "########## ARM=$ARM full panel DONE $(TS) ##########"
