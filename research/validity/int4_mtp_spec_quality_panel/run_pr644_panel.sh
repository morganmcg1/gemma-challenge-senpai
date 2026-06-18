#!/usr/bin/env bash
# PR #644 — Complete the Reading-A panel: Option-B MMLU-Pro + GSM8K %-of-base.
# Option-B = int4_g128_lmhead body + Gemma4-MTP K=7 drafter, BI=1, vLLM 0.22.0 (manifest).
# Served by serve_spec.py --engine manifest --max-model-len 8192 --max-num-seqs 16.
#
# gb6144 base-panel config matching ubel #628 denominator EXACTLY:
#   max_tokens=6144, min_tokens=8 (wirbel #541 EOS-guard), n=500, concurrency 16,
#   MMLU-Pro seed 12345, GSM8K seed 1234 8-shot  -> SAME seeded subsets as #628 base.
# Both decode modes per bench (lewtun #31): greedy (T=0) and sampled (T=1/0.95/64).
#
# Base denominators (ubel #628, GREEDY): MMLU-Pro 0.7180 (367i9s0t), GSM8K 0.9280 (4cxd1gfx).
set -uo pipefail
cd /workspace/senpai/target
PY=/tmp/eval-serve-venv/bin/python
DIR=research/validity/int4_mtp_spec_quality_panel
RES=$DIR/results-pr644
mkdir -p "$RES"
BASE=http://127.0.0.1:8000
MODEL=gemma-4-e4b-it
MT=6144
STATUS=$DIR/_pr644_panel.status
: > "$STATUS"
echo "PR644-PANEL-START $(date -u +%FT%TZ)" | tee -a "$STATUS"

# ---- [1/3] GSM8K both regimes (fast: ~105s/regime) --------------------------
echo "===== [1/3] GSM8K sampled+greedy n=500 8-shot $(date -u +%H:%M:%S) =====" | tee -a "$STATUS"
$PY research/downstream_quality_gsm8k/gsm8k_eval.py \
  --base-url "$BASE" --model "$MODEL" --label optionb_pr644_gb6144 \
  --regimes sampled,greedy --n 500 --n-shot 8 --seed 1234 --sampling-seed 1234 \
  --max-tokens "$MT" --min-tokens 8 --concurrency 16 --out-dir "$RES" \
  > "$DIR/_pr644_gsm8k.out" 2>&1
echo "  gsm8k rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"
grep -E "\[gsm8k\] DONE" "$DIR/_pr644_gsm8k.out" | tee -a "$STATUS"

# ---- [2/3] MMLU-Pro greedy (T=0) --------------------------------------------
echo "===== [2/3] MMLU-Pro greedy n=500 $(date -u +%H:%M:%S) =====" | tee -a "$STATUS"
$PY research/validity/downstream_quality_eval/run_eval.py \
  --task mmlu_pro --arm optionb_mmlu_greedy_pr644 --out "$RES/mmlu_pro_greedy.json" \
  --n 500 --seed 12345 --max-tokens "$MT" --min-tokens 8 --max-connections 16 \
  --temperature 0.0 --top-p 1.0 --top-k 0 \
  --base-url "$BASE/v1" --model "$MODEL" \
  > "$DIR/_pr644_mmlu_greedy.out" 2>&1
echo "  mmlu greedy rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"
grep -E "\[run_eval\]" "$DIR/_pr644_mmlu_greedy.out" | tee -a "$STATUS"

# ---- [3/3] MMLU-Pro sampled (T=1/top_p=0.95/top_k=64, per generation_config) -
echo "===== [3/3] MMLU-Pro sampled n=500 $(date -u +%H:%M:%S) =====" | tee -a "$STATUS"
$PY research/validity/downstream_quality_eval/run_eval.py \
  --task mmlu_pro --arm optionb_mmlu_sampled_pr644 --out "$RES/mmlu_pro_sampled.json" \
  --n 500 --seed 12345 --max-tokens "$MT" --min-tokens 8 --max-connections 16 \
  --temperature 1.0 --top-p 0.95 --top-k 64 --sampling-seed 1234 \
  --base-url "$BASE/v1" --model "$MODEL" \
  > "$DIR/_pr644_mmlu_sampled.out" 2>&1
echo "  mmlu sampled rc=$? $(date -u +%H:%M:%S)" | tee -a "$STATUS"
grep -E "\[run_eval\]" "$DIR/_pr644_mmlu_sampled.out" | tee -a "$STATUS"

echo "PR644-PANEL-DONE $(date -u +%FT%TZ)" | tee -a "$STATUS"
