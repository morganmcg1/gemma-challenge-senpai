#!/usr/bin/env bash
# PR #639 -- GPQA-Diamond SAMPLED (T=1/top_p=0.95/top_k=64) 10-seed pool, n=1980,
# against a running 0.22.0 server @ gb6144. Byte-for-byte the ubel #628 / fern #629
# protocol (seeds, budget, conc) so the arm pools apples-to-apples with bf16 base
# 0.5404 and Option-B int4+spec 0.4652. ARM name picks the results subdir.
#
#   ARM=<name> bash run_gpqa_sampled.sh         # full 10-seed pool
#   LIMIT=8 ARM=<name>_smoke bash run_gpqa_sampled.sh   # 1-seed smoke (load check)
set -uo pipefail
cd /workspace/senpai/target
PY=/tmp/eval-serve-venv/bin/python
DIR=research/validity/int4_body_quality_upside_639
ARM="${ARM:?set ARM=<name>}"
RES="$DIR/results/$ARM"
BASE=http://127.0.0.1:8000
MODEL=gemma-4-e4b-it
RUN_EVAL=research/validity/downstream_quality_eval/run_eval.py
MT=6144
LIMIT="${LIMIT:-0}"
if [[ "$LIMIT" -gt 0 ]]; then
  SEEDS=(12345); LIM_EVAL="--limit $LIMIT"
else
  SEEDS=(12345 23456 34567 45678 56789 67890 78901 89012 90123 13579); LIM_EVAL=""
fi
mkdir -p "$RES"
STATUS="$DIR/_gpqa_${ARM}.status"
: > "$STATUS"
echo "GPQA-SAMPLED-START arm=$ARM limit=$LIMIT $(date -u +%FT%TZ)" | tee -a "$STATUS"

for SEED in "${SEEDS[@]}"; do
  echo "===== GPQA SAMPLED seed=$SEED $(date -u +%H:%M:%S) =====" | tee -a "$STATUS"
  $PY "$RUN_EVAL" \
    --task gpqa_diamond --arm "$ARM" --out "$RES/gpqa_gb6144_s${SEED}.json" \
    --seed "$SEED" --max-tokens "$MT" --max-connections 16 \
    --temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 8 $LIM_EVAL \
    --base-url "$BASE/v1" --model "$MODEL" \
    > "$DIR/_gpqa_${ARM}_s${SEED}.out" 2>&1
  echo "  s$SEED rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"
  tail -2 "$DIR/_gpqa_${ARM}_s${SEED}.out" | tee -a "$STATUS"
done
echo "GPQA-SAMPLED-DONE arm=$ARM $(date -u +%FT%TZ)" | tee -a "$STATUS"
