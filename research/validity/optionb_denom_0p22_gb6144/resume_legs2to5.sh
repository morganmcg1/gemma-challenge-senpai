#!/usr/bin/env bash
# PR #628 -- RESUME of the full denominator panel: legs [2/5]..[5/5] only.
#
# Leg [1/5] GSM8K greedy n=500 already completed on THIS SAME warm vLLM-0.22.0
# server process (PID in _server_bf16_0p22.pid): acc=0.9280 (464/500), trunc=0,
# results/base_greedy_gb6144_greedy.json. The prior session's run_panel.sh driver
# was killed at session end mid-MMLU (~112/500), but the detached server survived,
# so we resume the remaining four legs on the identical process -- this keeps every
# gate on one continuous bf16-head argmax (cross-process bf16 GEMV argmax drift is a
# known hazard; staying on one process removes it). Legs are byte-for-byte
# run_panel.sh full-mode (LIMIT=0); only GSM8K is skipped because it is already on disk.
set -uo pipefail
cd /workspace/senpai/target
HERE=research/validity/optionb_denom_0p22_gb6144
CLIENT=/tmp/eval-serve-venv/bin/python
BASE=http://127.0.0.1:8000
MODEL=gemma-4-e4b-it
MT=6144
RES="$HERE/results"
TAG=full
mkdir -p "$RES"
STATUS="$HERE/_panel_resume.status"
: > "$STATUS"
echo "PANEL-resume(legs2-5)-START $(date -u +%FT%TZ) server_pid=$(cat "$HERE/_server_bf16_0p22.pid" 2>/dev/null)" | tee -a "$STATUS"

run_leg () { echo "===== $1 $(date -u +%H:%M:%S) =====" | tee -a "$STATUS"; }

# ---- [2/5] MMLU-Pro greedy n=500 (the #547 0.22.0 crater canary) ------------
run_leg "[2/5] MMLU-Pro greedy"
$CLIENT research/validity/downstream_quality_eval/run_eval.py \
  --task mmlu_pro --arm base_greedy --out "$RES/base_mmlu_pro_greedy_gb6144.json" \
  --n 500 --seed 12345 --max-tokens "$MT" --min-tokens 8 --max-connections 16 \
  --base-url "$BASE/v1" --model "$MODEL" \
  > "$HERE/_mmlu_${TAG}.out" 2>&1
echo "  mmlu rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"; tail -3 "$HERE/_mmlu_${TAG}.out" | tee -a "$STATUS"

# ---- [3/5] GPQA-Diamond greedy n=198 ----------------------------------------
run_leg "[3/5] GPQA-D greedy"
$CLIENT research/validity/downstream_quality_eval/run_eval.py \
  --task gpqa_diamond --arm base_greedy --out "$RES/base_gpqa_greedy_gb6144.json" \
  --seed 12345 --max-tokens "$MT" --min-tokens 8 --max-connections 16 \
  --base-url "$BASE/v1" --model "$MODEL" \
  > "$HERE/_gpqa_greedy_${TAG}.out" 2>&1
echo "  gpqa_greedy rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"; tail -3 "$HERE/_gpqa_greedy_${TAG}.out" | tee -a "$STATUS"

# ---- [4/5] GPQA-Diamond sampled n=198 (T=1/top_p=0.95/top_k=64) --------------
run_leg "[4/5] GPQA-D sampled"
$CLIENT research/validity/downstream_quality_eval/run_eval.py \
  --task gpqa_diamond --arm base_sampled --out "$RES/base_gpqa_sampled_gb6144.json" \
  --seed 12345 --temperature 1.0 --top-p 0.95 --top-k 64 --sampling-seed 12345 \
  --max-tokens "$MT" --min-tokens 8 --max-connections 16 \
  --base-url "$BASE/v1" --model "$MODEL" \
  > "$HERE/_gpqa_sampled_${TAG}.out" 2>&1
echo "  gpqa_sampled rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"; tail -3 "$HERE/_gpqa_sampled_${TAG}.out" | tee -a "$STATUS"

# ---- [5/5] AIME greedy maj@1 no-think n=60 ----------------------------------
run_leg "[5/5] AIME greedy"
$CLIENT research/downstream_quality_aime/aime_eval.py \
  --base-url "$BASE" --model "$MODEL" --years 2024,2025-I,2025-II --k 1 \
  --temperature 0.0 --top-p 1.0 --top-k -1 --max-tokens "$MT" --min-tokens 8 \
  --no-thinking --seed 1234 --save-text \
  --label base_aime_greedy_gb6144 --out "$RES/base_aime_greedy_gb6144.json" \
  > "$HERE/_aime_${TAG}.out" 2>&1
echo "  aime rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"; tail -4 "$HERE/_aime_${TAG}.out" | tee -a "$STATUS"

echo "PANEL-resume(legs2-5)-DONE $(date -u +%FT%TZ)" | tee -a "$STATUS"
