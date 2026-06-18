#!/usr/bin/env bash
# PR #628 -- Option-B *denominator* 4-gate panel: bf16 unquantized base on vLLM
# 0.22.0 at fern #624's gb6144 budget. Five legs, SEQUENTIAL on one server so each
# gets the full 16-seq KV cache. Invocations are byte-for-byte fern #624's
# (run_greedy_panel_gb6144.sh) EXCEPT they point at the bf16-base@0.22.0 server,
# plus the two GPQA decode reads (greedy AND sampled) the bar needs.
#
#   MMLU-Pro n=500 greedy seed12345 ; GSM8K n=500 greedy 8-shot seed1234 ;
#   AIME n=60 (2024,2025-I,2025-II) greedy maj@1 no-think seed1234 ;
#   GPQA-Diamond n=198 greedy seed12345 ; GPQA-Diamond n=198 sampled (T=1/
#   top_p=0.95/top_k=64) seed12345 sampling-seed12345.
#   All max_tokens=6144 (gb6144), min_tokens=8 (#541 EOS-guard), conc=16.
#
# LIMIT>0 -> smoke (cap each leg to LIMIT items, write under results/_smoke).
set -uo pipefail
cd /workspace/senpai/target
HERE=research/validity/optionb_denom_0p22_gb6144
CLIENT=/tmp/eval-serve-venv/bin/python      # version-agnostic eval client (inspect/openai/requests)
BASE=http://127.0.0.1:8000
MODEL=gemma-4-e4b-it
MT=6144
LIMIT="${LIMIT:-0}"                          # 0 = full panel
if [[ "$LIMIT" -gt 0 ]]; then
  RES="$HERE/results/_smoke"; LIM_EVAL="--limit $LIMIT"; TAG="smoke"
else
  RES="$HERE/results"; LIM_EVAL=""; TAG="full"
fi
mkdir -p "$RES"
STATUS="$HERE/_panel_${TAG}.status"
: > "$STATUS"
echo "PANEL-$TAG-START $(date -u +%FT%TZ) limit=$LIMIT" | tee -a "$STATUS"

run_leg () {  # $1 = human label
  echo "===== $1 $(date -u +%H:%M:%S) =====" | tee -a "$STATUS"
}

# ---- [1/5] GSM8K greedy n=500 8-shot ----------------------------------------
run_leg "[1/5] GSM8K greedy"
$CLIENT research/downstream_quality_gsm8k/gsm8k_eval.py \
  --base-url "$BASE" --model "$MODEL" --label base_greedy_gb6144 \
  --regimes greedy --n 500 --n-shot 8 --seed 1234 \
  --max-tokens "$MT" --min-tokens 8 --concurrency 16 $LIM_EVAL --out-dir "$RES" \
  > "$HERE/_gsm8k_${TAG}.out" 2>&1
echo "  gsm8k rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"; tail -3 "$HERE/_gsm8k_${TAG}.out" | tee -a "$STATUS"

# ---- [2/5] MMLU-Pro greedy n=500 (the #547 0.22.0 crater canary) ------------
run_leg "[2/5] MMLU-Pro greedy"
$CLIENT research/validity/downstream_quality_eval/run_eval.py \
  --task mmlu_pro --arm base_greedy --out "$RES/base_mmlu_pro_greedy_gb6144.json" \
  --n 500 --seed 12345 --max-tokens "$MT" --min-tokens 8 --max-connections 16 $LIM_EVAL \
  --base-url "$BASE/v1" --model "$MODEL" \
  > "$HERE/_mmlu_${TAG}.out" 2>&1
echo "  mmlu rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"; tail -3 "$HERE/_mmlu_${TAG}.out" | tee -a "$STATUS"

# ---- [3/5] GPQA-Diamond greedy n=198 ----------------------------------------
run_leg "[3/5] GPQA-D greedy"
$CLIENT research/validity/downstream_quality_eval/run_eval.py \
  --task gpqa_diamond --arm base_greedy --out "$RES/base_gpqa_greedy_gb6144.json" \
  --seed 12345 --max-tokens "$MT" --min-tokens 8 --max-connections 16 $LIM_EVAL \
  --base-url "$BASE/v1" --model "$MODEL" \
  > "$HERE/_gpqa_greedy_${TAG}.out" 2>&1
echo "  gpqa_greedy rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"; tail -3 "$HERE/_gpqa_greedy_${TAG}.out" | tee -a "$STATUS"

# ---- [4/5] GPQA-Diamond sampled n=198 (T=1/top_p=0.95/top_k=64) --------------
run_leg "[4/5] GPQA-D sampled"
$CLIENT research/validity/downstream_quality_eval/run_eval.py \
  --task gpqa_diamond --arm base_sampled --out "$RES/base_gpqa_sampled_gb6144.json" \
  --seed 12345 --temperature 1.0 --top-p 0.95 --top-k 64 --sampling-seed 12345 \
  --max-tokens "$MT" --min-tokens 8 --max-connections 16 $LIM_EVAL \
  --base-url "$BASE/v1" --model "$MODEL" \
  > "$HERE/_gpqa_sampled_${TAG}.out" 2>&1
echo "  gpqa_sampled rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"; tail -3 "$HERE/_gpqa_sampled_${TAG}.out" | tee -a "$STATUS"

# ---- [5/5] AIME greedy maj@1 no-think n=60 (risk leg; sequential) ------------
run_leg "[5/5] AIME greedy"
$CLIENT research/downstream_quality_aime/aime_eval.py \
  --base-url "$BASE" --model "$MODEL" --years 2024,2025-I,2025-II --k 1 \
  --temperature 0.0 --top-p 1.0 --top-k -1 --max-tokens "$MT" --min-tokens 8 \
  --no-thinking --seed 1234 --save-text $LIM_EVAL \
  --label base_aime_greedy_gb6144 --out "$RES/base_aime_greedy_gb6144.json" \
  > "$HERE/_aime_${TAG}.out" 2>&1
echo "  aime rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"; tail -4 "$HERE/_aime_${TAG}.out" | tee -a "$STATUS"

echo "PANEL-$TAG-DONE $(date -u +%FT%TZ)" | tee -a "$STATUS"
