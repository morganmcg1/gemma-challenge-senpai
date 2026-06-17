#!/usr/bin/env bash
# PR #612 — Option-B GPQA generation-budget retry.
# Same int4_g128_lmhead + MTP-K7 spec config #605 measured, served on vLLM dev307
# (PR #612 directive) @ max_model_len=8192, conc=16, sampling T=1.0/top_p=0.95/
# top_k=64 + min_tokens=8 (matches #605's protocol; only max_tokens + context vary).
#
# Adds the truncation diagnostics #605 lacked (instrumented run_eval.py):
# finish_reason / length_stop_rate / completion-token p50/p95.
#
# Usage: run_gpqa_genbudget.sh <max_tokens> <tag> <seed> [seed ...]
#   Arm B (generous): run_gpqa_genbudget.sh 4096 gb4096 12345 23456 34567  -> the VERDICT (pooled n=594)
#   Arm A (replicate): run_gpqa_genbudget.sh 3072 gb3072 12345             -> reproduce #605 0.4141 + old-budget trunc
set -uo pipefail
cd /workspace/senpai/target
PY=/tmp/eval-serve-venv/bin/python
DIR=research/validity/int4_mtp_spec_quality_panel
RES=$DIR/results
BASE=http://127.0.0.1:8000
MODEL=gemma-4-e4b-it
RUN_EVAL=research/validity/downstream_quality_eval/run_eval.py

MT=$1; TAG=$2; shift 2
mkdir -p "$RES"
for SEED in "$@"; do
  echo "===== GPQA $TAG budget=$MT seed=$SEED $(date -u +%H:%M:%S) ====="
  $PY "$RUN_EVAL" \
    --task gpqa_diamond --arm spec --out "$RES/spec_gpqa_${TAG}_s${SEED}.json" \
    --seed "$SEED" --max-tokens "$MT" --max-connections 16 \
    --temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 8 \
    --base-url "$BASE/v1" --model "$MODEL" \
    > "$DIR/_gpqa_${TAG}_s${SEED}.out" 2>&1
  rc=$?
  echo "  rc=$rc $(date -u +%H:%M:%S)"
  tail -2 "$DIR/_gpqa_${TAG}_s${SEED}.out"
done
echo "===== $TAG DONE $(date -u +%H:%M:%S) ====="
