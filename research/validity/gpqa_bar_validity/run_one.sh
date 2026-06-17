#!/usr/bin/env bash
# PR #614 -- run ONE GPQA-Diamond decode realization against the live bf16 base server
# and write its result json. Thin wrapper over the merged downstream_quality_eval
# run_eval.py (now finish_reason-instrumented). Each realization is a SEPARATE process
# so no single invocation risks the 90-min SENPAI_TIMEOUT_MINUTES wall.
#
# Usage: run_one.sh <label> <temperature> <top_p> <top_k> <max_tokens> <sampling_seed> <out.json>
# Dataset choice-shuffle --seed is FIXED at 12345 for ALL realizations (byte-identical
# prompts; aggregate_ci.py asserts the id set matches). Only --sampling-seed varies.
set -u
ROOT=/workspace/senpai/target
HERE="$ROOT/research/validity/gpqa_bar_validity"
EVALPY=/tmp/eval-serve-venv/bin/python          # has inspect_ai/inspect_evals/openai
RUN_EVAL="$ROOT/research/validity/downstream_quality_eval/run_eval.py"

LABEL="$1"; TEMP="$2"; TOPP="$3"; TOPK="$4"; MAXTOK="$5"; SSEED="$6"; OUT="$7"
DATASET_SEED=12345
LOGDIR="$HERE/_inspect_logs/$LABEL"
mkdir -p "$LOGDIR"

echo "[$LABEL] START $(date -u +%FT%TZ) temp=$TEMP top_p=$TOPP top_k=$TOPK max_tokens=$MAXTOK min_tokens=8 sampling_seed=$SSEED dataset_seed=$DATASET_SEED"
t0=$(date +%s)
"$EVALPY" "$RUN_EVAL" \
  --task gpqa_diamond --arm "$LABEL" \
  --seed "$DATASET_SEED" \
  --temperature "$TEMP" --top-p "$TOPP" --top-k "$TOPK" \
  --max-tokens "$MAXTOK" --min-tokens 8 \
  --max-connections 16 \
  --sampling-seed "$SSEED" \
  --out "$OUT" \
  --log-dir "$LOGDIR"
rc=$?
echo "[$LABEL] DONE rc=$rc elapsed=$(( $(date +%s)-t0 ))s $(date -u +%FT%TZ)"
exit $rc
