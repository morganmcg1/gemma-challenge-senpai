#!/usr/bin/env bash
# PR #650 Arm B -- bf16 base GPQA-Diamond *sampled* 10-seed denominator (n=10x198=1980).
# BYTE-IDENTICAL protocol to the #638 int4-AR 10-seed GPQA-sampled leg
# (run_panel_int4ar.sh leg 1): dataset seed 12345 fixed, sampling-seeds 0..9, T=1,
# top_p=0.95, top_k=64, max_tokens 6144, min_tokens 8, conc 16. The ONLY difference is
# the served model (bf16 base instead of int4-AR), so this 10-seed bf16 MEAN replaces
# the single-seed 0.5404 denominator and recalibrates the 0.9x bar.
# Each seed writes its own JSON (resumable: a non-empty seed file is skipped).
set -uo pipefail
cd /workspace/senpai/target
HERE=research/validity/int4ar_denom_harden
CLIENT=/tmp/eval-serve-venv/bin/python
DSEED=12345
MT="${MT:-6144}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
RES="$HERE/results/gpqa_bf16_sampled"; mkdir -p "$RES"
STATUS="$HERE/_gpqa_bf16_sampled.status"
echo "GPQA-bf16-sampled START dseed=$DSEED seeds=[$SEEDS] mt=$MT $(date -u +%FT%TZ)" | tee -a "$STATUS"
for s in $SEEDS; do
  OUT="$RES/bf16_gpqa_sampled_s${s}.json"
  if [[ -s "$OUT" ]]; then echo "  s=$s exists, skip $(date -u +%H:%M:%S)" | tee -a "$STATUS"; continue; fi
  $CLIENT research/validity/downstream_quality_eval/run_eval.py \
    --task gpqa_diamond --arm bf16_sampled --out "$OUT" \
    --seed "$DSEED" --temperature 1.0 --top-p 0.95 --top-k 64 --sampling-seed "$s" \
    --max-tokens "$MT" --min-tokens 8 --max-connections 16 \
    --base-url http://127.0.0.1:8000/v1 --model gemma-4-e4b-it \
    > "$HERE/_gpqa_bf16_sampled_s${s}.out" 2>&1
  echo "  s=$s rc=$? $(date -u +%H:%M:%S): $(grep -oE 'acc=[0-9.]+' "$HERE/_gpqa_bf16_sampled_s${s}.out" | head -1)" | tee -a "$STATUS"
done
echo "GPQA-bf16-sampled DONE $(date -u +%FT%TZ)" | tee -a "$STATUS"
