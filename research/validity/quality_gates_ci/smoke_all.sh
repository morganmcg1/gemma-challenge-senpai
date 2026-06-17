#!/usr/bin/env bash
# PR #590 -- tiny end-to-end smoke of all three eval harnesses against the live
# base_fullhead server under the lewtun #31 sampling protocol + min_tokens=8.
# Validates: request plumbing, sampling params reach vLLM, answer extraction,
# min_tokens EOS-guard, and -- critically -- AIME per-problem WALL so we can size k
# to stay under the 90-min/run bound. NOT a measurement; --limit is tiny.
set -u
cd /workspace/senpai/target
PY=/usr/bin/python3                      # aime/gsm8k: stdlib urllib client
PYI=/tmp/eval-serve-venv/bin/python      # mmlu_pro: needs inspect_ai/inspect_evals
URL0=http://127.0.0.1:8000               # aime/gsm8k append /v1/chat/completions themselves
URL=http://127.0.0.1:8000/v1             # inspect openai-api provider wants the /v1 base
OUT=research/validity/quality_gates_ci
SAMP="--temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 8"

echo "===== AIME smoke: k=5, 3 problems (time -> extrapolate to n=60) ====="
t0=$(date +%s)
$PY research/downstream_quality_aime/aime_eval.py --base-url $URL0 \
  --years 2024 --k 5 $SAMP --max-tokens 3072 --seed 1234 --no-thinking --save-text \
  --limit 3 --label smoke_aime --out $OUT/_smoke_aime.json 2>&1 | tail -5
aime_dt=$(( $(date +%s)-t0 ))
echo "[smoke] AIME 3-problem k=5 wall=${aime_dt}s -> n=60 extrapolation ~$(( aime_dt*20 ))s (~$(( aime_dt*20/60 ))min)"

echo "===== MMLU-Pro smoke: 8 questions, sampling ====="
t0=$(date +%s)
$PYI research/validity/downstream_quality_eval/run_eval.py --task mmlu_pro --arm smoke \
  --n 500 --seed 12345 --limit 8 $SAMP --sampling-seed 1 --max-tokens 2048 \
  --max-connections 32 --base-url $URL \
  --out $OUT/_smoke_mmlu.json --log-dir $OUT/_smoke_mmlu_logs 2>&1 | tail -4
echo "[smoke] MMLU-Pro 8q wall=$(( $(date +%s)-t0 ))s"

echo "===== GSM8K smoke: 16 items, sampled ====="
t0=$(date +%s)
$PY research/downstream_quality_gsm8k/gsm8k_eval.py --base-url $URL0 --label smoke_gsm8k \
  --regimes sampled --n 500 --seed 1234 --sampling-seed 1 --limit 16 \
  --n-shot 8 --top-p 0.95 --top-k 64 --max-tokens 512 --min-tokens 8 \
  --concurrency 16 --out-dir $OUT 2>&1 | tail -4
echo "[smoke] GSM8K 16item wall=$(( $(date +%s)-t0 ))s"
echo "===== SMOKE DONE ====="
