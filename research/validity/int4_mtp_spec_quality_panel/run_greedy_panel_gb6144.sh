#!/usr/bin/env bash
# PR #624 — Option-B quality panel, GREEDY gate reads at the gb6144 clean budget.
# Runs the 3 remaining legs (MMLU-Pro / GSM8K / AIME) on the ALREADY-RUNNING spec
# server (serve_spec.py: dev307, int4_g128_lmhead body + MTP-K7 drafter, BI=1,
# max-model-len 8192, max-num-seqs 16) — the exact #612 serve config.
#
# Decode = GREEDY (temp=0, top_p=1, top_k off) per PR #624 ("the gates are defined
# on greedy") + the base denominators: MMLU/GSM8K greedy-anchors (0.678/0.904) and
# the AIME greedy maj@1 base (0.100 = 6/60, base_fullhead 0.1167 = 7/60). min_tokens=8
# (#541 EOS-guard) on all three; max_tokens=6144 (gb6144 clean budget). Legs run
# SEQUENTIALLY so each gets the full 16-seq KV cache (clean per-leg timing).
#
# Protocol matched to the gate base denominators (research/validity/base_quality_
# denominator + base_fullhead_aime_n60): MMLU-Pro n=500 seed12345; GSM8K n=500
# seed1234 8-shot; AIME n=60 (2024,2025-I,2025-II) seed1234 no-thinking k=1.
set -uo pipefail
cd /workspace/senpai/target
PY=/tmp/eval-serve-venv/bin/python
DIR=research/validity/int4_mtp_spec_quality_panel
RES=$DIR/results-greedy
mkdir -p "$RES"
BASE=http://127.0.0.1:8000
MODEL=gemma-4-e4b-it
MT=6144
STATUS=$DIR/_greedy_panel.status
: > "$STATUS"
echo "PANEL-START $(date -u +%FT%TZ)" | tee -a "$STATUS"

# ---- [1/3] GSM8K greedy (fastest first) -------------------------------------
echo "===== [1/3] GSM8K greedy n=500 8-shot $(date -u +%H:%M:%S) =====" | tee -a "$STATUS"
$PY research/downstream_quality_gsm8k/gsm8k_eval.py \
  --base-url "$BASE" --model "$MODEL" --label spec_greedy_gb6144 \
  --regimes greedy --n 500 --n-shot 8 --seed 1234 \
  --max-tokens "$MT" --min-tokens 8 --concurrency 16 --out-dir "$RES" \
  > "$DIR/_gsm8k_greedy.out" 2>&1
echo "  gsm8k rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"
tail -2 "$DIR/_gsm8k_greedy.out" | tee -a "$STATUS"

# ---- [2/3] MMLU-Pro greedy --------------------------------------------------
echo "===== [2/3] MMLU-Pro greedy n=500 $(date -u +%H:%M:%S) =====" | tee -a "$STATUS"
$PY research/validity/downstream_quality_eval/run_eval.py \
  --task mmlu_pro --arm spec_greedy --out "$RES/spec_mmlu_pro_greedy_gb6144.json" \
  --n 500 --seed 12345 --max-tokens "$MT" --min-tokens 8 --max-connections 16 \
  --base-url "$BASE/v1" --model "$MODEL" \
  > "$DIR/_mmlu_greedy.out" 2>&1
echo "  mmlu rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"
tail -2 "$DIR/_mmlu_greedy.out" | tee -a "$STATUS"

# ---- [3/3] AIME greedy maj@1 no-thinking (risk leg; sequential 60) ----------
echo "===== [3/3] AIME greedy k=1 no-thinking n=60 $(date -u +%H:%M:%S) =====" | tee -a "$STATUS"
/usr/bin/python3 research/downstream_quality_aime/aime_eval.py \
  --base-url "$BASE" --model "$MODEL" --years 2024,2025-I,2025-II --k 1 \
  --temperature 0.0 --top-p 1.0 --top-k -1 --max-tokens "$MT" --min-tokens 8 \
  --no-thinking --seed 1234 --save-text \
  --label spec_aime_greedy_gb6144 --out "$RES/spec_aime_greedy_gb6144.json" \
  > "$DIR/_aime_greedy.out" 2>&1
echo "  aime rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"
tail -3 "$DIR/_aime_greedy.out" | tee -a "$STATUS"

echo "PANEL-DONE $(date -u +%FT%TZ)" | tee -a "$STATUS"
