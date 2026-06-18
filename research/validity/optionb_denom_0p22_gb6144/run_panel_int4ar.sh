#!/usr/bin/env bash
# PR #638 -- Option-B denominator LEG 3: int4-AR (live-rung body) 5-gate panel on
# vLLM 0.22.0 at gb6144, against serve_int4ar_0p22.sh. Byte-for-byte the ubel #628
# bf16 panel invocations (same eval client, same flags, MT=6144, min_tokens=8,
# conc=16) EXCEPT the binding GPQA-sampled leg is the 10-seed n=1980 protocol fern
# #629 used (dataset_seed 12345 fixed, sampling_seed 0..9), so int4-AR vs int4+spec
# #629 (0.4652) is an apples-to-apples 10-seed head-to-head.
#
# Leg order = PR #638 priority (highest-value first under time pressure):
#   [1] GPQA-D sampled 10-seed n=1980  (BINDING / primary_metric)
#   [2] AIME n=60 greedy maj@1 no-think
#   [3] GPQA-D greedy n=198            (cross-check vs lawine #627 0.4444)
#   [4] MMLU-Pro greedy n=500
#   [5] GSM8K greedy n=500 8-shot
# Each leg writes its own JSON so a crash mid-panel keeps prior legs.
#
# LIMIT>0 -> smoke (cap each leg to LIMIT items; sampled uses only SEEDS_SMOKE).
set -uo pipefail
cd /workspace/senpai/target
HERE=research/validity/optionb_denom_0p22_gb6144
CLIENT=/tmp/eval-serve-venv/bin/python
BASE=http://127.0.0.1:8000
MODEL=gemma-4-e4b-it
MT=6144
DSEED=12345
LIMIT="${LIMIT:-0}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
if [[ "$LIMIT" -gt 0 ]]; then
  RES="$HERE/results_int4ar/_smoke"; LIM_EVAL="--limit $LIMIT"; TAG="smoke"; SEEDS="${SEEDS_SMOKE:-0 1}"
else
  RES="$HERE/results_int4ar"; LIM_EVAL=""; TAG="full"
fi
mkdir -p "$RES"
STATUS="$HERE/_panel_int4ar_${TAG}.status"
: > "$STATUS"
echo "PANEL-int4ar-$TAG-START $(date -u +%FT%TZ) limit=$LIMIT seeds=[$SEEDS] server_pid=$(cat "$HERE/_server_int4ar_0p22.pid" 2>/dev/null)" | tee -a "$STATUS"
run_leg () { echo "===== $1 $(date -u +%H:%M:%S) =====" | tee -a "$STATUS"; }

# ---- [1/5] GPQA-D sampled 10-seed n=1980 (BINDING) --------------------------
run_leg "[1/5] GPQA-D sampled 10-seed (dseed=$DSEED, sseeds=[$SEEDS])"
for s in $SEEDS; do
  $CLIENT research/validity/downstream_quality_eval/run_eval.py \
    --task gpqa_diamond --arm int4ar_sampled --out "$RES/int4ar_gpqa_sampled_s${s}.json" \
    --seed "$DSEED" --temperature 1.0 --top-p 0.95 --top-k 64 --sampling-seed "$s" \
    --max-tokens "$MT" --min-tokens 8 --max-connections 16 $LIM_EVAL \
    --base-url "$BASE/v1" --model "$MODEL" \
    > "$HERE/_gpqa_sampled_int4ar_s${s}_${TAG}.out" 2>&1
  echo "  sampled s=$s rc=$? $(date -u +%H:%M:%S): $(grep -o 'acc=[0-9.]*' "$HERE/_gpqa_sampled_int4ar_s${s}_${TAG}.out" | head -1)" | tee -a "$STATUS"
done

# ---- [2/5] AIME greedy maj@1 no-think n=60 ----------------------------------
run_leg "[2/5] AIME greedy"
$CLIENT research/downstream_quality_aime/aime_eval.py \
  --base-url "$BASE" --model "$MODEL" --years 2024,2025-I,2025-II --k 1 \
  --temperature 0.0 --top-p 1.0 --top-k -1 --max-tokens "$MT" --min-tokens 8 \
  --no-thinking --seed 1234 --save-text $LIM_EVAL \
  --label int4ar_aime_greedy_gb6144 --out "$RES/int4ar_aime_greedy_gb6144.json" \
  > "$HERE/_aime_int4ar_${TAG}.out" 2>&1
echo "  aime rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"; tail -4 "$HERE/_aime_int4ar_${TAG}.out" | tee -a "$STATUS"

# ---- [3/5] GPQA-D greedy n=198 (cross-check vs lawine #627 0.4444) -----------
run_leg "[3/5] GPQA-D greedy"
$CLIENT research/validity/downstream_quality_eval/run_eval.py \
  --task gpqa_diamond --arm int4ar_greedy --out "$RES/int4ar_gpqa_greedy_gb6144.json" \
  --seed 12345 --max-tokens "$MT" --min-tokens 8 --max-connections 16 $LIM_EVAL \
  --base-url "$BASE/v1" --model "$MODEL" \
  > "$HERE/_gpqa_greedy_int4ar_${TAG}.out" 2>&1
echo "  gpqa_greedy rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"; tail -3 "$HERE/_gpqa_greedy_int4ar_${TAG}.out" | tee -a "$STATUS"

# ---- [4/5] MMLU-Pro greedy n=500 --------------------------------------------
run_leg "[4/5] MMLU-Pro greedy"
$CLIENT research/validity/downstream_quality_eval/run_eval.py \
  --task mmlu_pro --arm int4ar_greedy --out "$RES/int4ar_mmlu_pro_greedy_gb6144.json" \
  --n 500 --seed 12345 --max-tokens "$MT" --min-tokens 8 --max-connections 16 $LIM_EVAL \
  --base-url "$BASE/v1" --model "$MODEL" \
  > "$HERE/_mmlu_int4ar_${TAG}.out" 2>&1
echo "  mmlu rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"; tail -3 "$HERE/_mmlu_int4ar_${TAG}.out" | tee -a "$STATUS"

# ---- [5/5] GSM8K greedy n=500 8-shot ----------------------------------------
run_leg "[5/5] GSM8K greedy"
$CLIENT research/downstream_quality_gsm8k/gsm8k_eval.py \
  --base-url "$BASE" --model "$MODEL" --label int4ar_greedy_gb6144 \
  --regimes greedy --n 500 --n-shot 8 --seed 1234 \
  --max-tokens "$MT" --min-tokens 8 --concurrency 16 $LIM_EVAL --out-dir "$RES" \
  > "$HERE/_gsm8k_int4ar_${TAG}.out" 2>&1
echo "  gsm8k rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"; tail -3 "$HERE/_gsm8k_int4ar_${TAG}.out" | tee -a "$STATUS"

echo "PANEL-int4ar-$TAG-DONE $(date -u +%FT%TZ)" | tee -a "$STATUS"
