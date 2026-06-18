#!/usr/bin/env bash
# PR #629 — Option-B quality panel on vLLM 0.22.0 (engine=manifest), GREEDY gates @ gb6144.
# IDENTICAL to #624's run_greedy_panel_gb6144.sh EXCEPT the server engine is now 0.22.0
# (manifest pin) instead of dev307. Same spec stack (int4_g128_lmhead body + MTP-K7 drafter,
# BI=1, max-model-len 8192, max-num-seqs 16), same decode (GREEDY: temp=0/top_p=1/top_k off),
# same protocol (MMLU-Pro n=500 seed12345; GSM8K n=500 seed1234 8-shot; AIME n=60 k=1
# no-thinking seed1234), same budget (max_tokens=6144, min_tokens=8).
#
# Output -> results-greedy-0p22/ (kept SEPARATE from the dev307 baseline results-greedy/).
# Order: GSM8K -> MMLU first (the fast finish_length_rate crater detectors, kanna #618
# signature ~50% finish-length), AIME last (slow sequential leg).
set -uo pipefail
cd /workspace/senpai/target
PY=/tmp/eval-serve-venv/bin/python
DIR=research/validity/int4_mtp_spec_quality_panel
RES=$DIR/results-greedy-0p22
mkdir -p "$RES"
BASE=http://127.0.0.1:8000
MODEL=gemma-4-e4b-it
MT=6144
STATUS=$DIR/_greedy_panel_0p22.status
: > "$STATUS"
echo "PANEL-START-0p22 $(date -u +%FT%TZ)" | tee -a "$STATUS"

# ---- [1/3] GSM8K greedy (fastest crater detector first) ---------------------
echo "===== [1/3] GSM8K greedy n=500 8-shot $(date -u +%H:%M:%S) =====" | tee -a "$STATUS"
$PY research/downstream_quality_gsm8k/gsm8k_eval.py \
  --base-url "$BASE" --model "$MODEL" --label spec_greedy_0p22_gb6144 \
  --regimes greedy --n 500 --n-shot 8 --seed 1234 \
  --max-tokens "$MT" --min-tokens 8 --concurrency 16 --out-dir "$RES" \
  > "$DIR/_gsm8k_greedy_0p22.out" 2>&1
echo "  gsm8k rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"
tail -2 "$DIR/_gsm8k_greedy_0p22.out" | tee -a "$STATUS"

# ---- [2/3] MMLU-Pro greedy (the #547 crater leg) ----------------------------
echo "===== [2/3] MMLU-Pro greedy n=500 $(date -u +%H:%M:%S) =====" | tee -a "$STATUS"
$PY research/validity/downstream_quality_eval/run_eval.py \
  --task mmlu_pro --arm spec_greedy_0p22 --out "$RES/spec_mmlu_pro_greedy_0p22_gb6144.json" \
  --n 500 --seed 12345 --max-tokens "$MT" --min-tokens 8 --max-connections 16 \
  --base-url "$BASE/v1" --model "$MODEL" \
  > "$DIR/_mmlu_greedy_0p22.out" 2>&1
echo "  mmlu rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"
tail -2 "$DIR/_mmlu_greedy_0p22.out" | tee -a "$STATUS"

# ---- [3/3] AIME greedy maj@1 no-thinking (slow sequential 60) ----------------
echo "===== [3/3] AIME greedy k=1 no-thinking n=60 $(date -u +%H:%M:%S) =====" | tee -a "$STATUS"
/usr/bin/python3 research/downstream_quality_aime/aime_eval.py \
  --base-url "$BASE" --model "$MODEL" --years 2024,2025-I,2025-II --k 1 \
  --temperature 0.0 --top-p 1.0 --top-k -1 --max-tokens "$MT" --min-tokens 8 \
  --no-thinking --seed 1234 --save-text \
  --label spec_aime_greedy_0p22_gb6144 --out "$RES/spec_aime_greedy_0p22_gb6144.json" \
  > "$DIR/_aime_greedy_0p22.out" 2>&1
echo "  aime rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"
tail -3 "$DIR/_aime_greedy_0p22.out" | tee -a "$STATUS"

echo "PANEL-DONE-0p22 $(date -u +%FT%TZ)" | tee -a "$STATUS"
