#!/usr/bin/env bash
# PR #661 -- run the two MISSING #515 gate axes (MMLU-Pro + GSM8K) against the
# server stood up by serve_panel.sh, at the gb6144 M=1-AR panel. Byte-for-byte
# the #638 eval-client invocations (same run_eval.py / gsm8k_eval.py, MT=6144,
# min_tokens=8, same dataset seeds) so the seqs=1 cells pair item-for-item
# against #638's banked seqs=16 cells AND against the paired bf16 base arm.
#
# Usage: run_panel.sh <arm_label> [N_MMLU] [N_GSM8K] [LIMIT]
#   arm_label : int4ar | bf16   (folded into output filenames)
#   N_MMLU    : MMLU-Pro subset size (default 500; --seed 12345)
#   N_GSM8K   : GSM8K subset size   (default 500; --seed 1234)
#   LIMIT     : >0 -> smoke cap each leg to LIMIT items
set -uo pipefail
cd /workspace/senpai/target

ARM="${1:?usage: run_panel.sh <arm_label> [N_MMLU] [N_GSM8K] [LIMIT]}"
N_MMLU="${2:-500}"
N_GSM8K="${3:-500}"
LIMIT="${4:-0}"

HERE=research/validity/quality_gate_4axis
CLIENT=/tmp/eval-serve-venv/bin/python
BASE=http://127.0.0.1:8000
MODEL=gemma-4-e4b-it
MT=6144
CONC=16
RES="$HERE/results"
mkdir -p "$RES"

LIM_E=""; LIM_G=""; TAG="full"
if [[ "$LIMIT" -gt 0 ]]; then LIM_E="--limit $LIMIT"; LIM_G="--limit $LIMIT"; TAG="smoke"; fi

STATUS="$HERE/_panel_${ARM}_${TAG}.status"
: > "$STATUS"
echo "PANEL-$ARM-$TAG-START $(date -u +%FT%TZ) n_mmlu=$N_MMLU n_gsm8k=$N_GSM8K limit=$LIMIT seqs=$(curl -s $BASE/v1/models >/dev/null 2>&1 && echo up || echo DOWN)" | tee -a "$STATUS"

# ---- [1/2] MMLU-Pro greedy (run_eval.py / inspect_evals) --------------------
echo "===== [1/2] MMLU-Pro greedy n=$N_MMLU $(date -u +%H:%M:%S) =====" | tee -a "$STATUS"
$CLIENT research/validity/downstream_quality_eval/run_eval.py \
  --task mmlu_pro --arm "${ARM}_mmlu" --out "$RES/${ARM}_mmlu_pro_greedy.json" \
  --n "$N_MMLU" --seed 12345 --max-tokens "$MT" --min-tokens 8 \
  --max-connections "$CONC" $LIM_E \
  --base-url "$BASE/v1" --model "$MODEL" \
  > "$HERE/_mmlu_${ARM}_${TAG}.out" 2>&1
echo "  mmlu rc=$? $(date -u +%H:%M:%S): $(grep -oiE 'accuracy[^,}]*' "$HERE/_mmlu_${ARM}_${TAG}.out" | tail -1)" | tee -a "$STATUS"
tail -3 "$HERE/_mmlu_${ARM}_${TAG}.out" | tee -a "$STATUS"

# ---- [2/2] GSM8K greedy 8-shot (gsm8k_eval.py) ------------------------------
echo "===== [2/2] GSM8K greedy n=$N_GSM8K $(date -u +%H:%M:%S) =====" | tee -a "$STATUS"
$CLIENT research/downstream_quality_gsm8k/gsm8k_eval.py \
  --base-url "$BASE" --model "$MODEL" --label "${ARM}_gsm8k" \
  --regimes greedy --n "$N_GSM8K" --n-shot 8 --seed 1234 \
  --max-tokens "$MT" --min-tokens 8 --concurrency "$CONC" $LIM_G \
  --out-dir "$RES" \
  > "$HERE/_gsm8k_${ARM}_${TAG}.out" 2>&1
echo "  gsm8k rc=$? $(date -u +%H:%M:%S): $(grep -oE 'acc=[0-9.]*' "$HERE/_gsm8k_${ARM}_${TAG}.out" | tail -1)" | tee -a "$STATUS"
tail -3 "$HERE/_gsm8k_${ARM}_${TAG}.out" | tee -a "$STATUS"

echo "PANEL-$ARM-$TAG-DONE $(date -u +%FT%TZ)" | tee -a "$STATUS"
