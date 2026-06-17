#!/usr/bin/env bash
# PR #609 deliverable #4: GSM8K quality proxy for the ngram option-B lane.
# Two MATCHED arms on the SAME int4_g128_lmhead substrate (within-substrate control
# the existing int4_base arm cannot give): AR floor vs ng_max4_k5 ngram spec.
# sampled T=1.0/top_p=0.95/top_k=64, min_tokens=8 (#541 EOS-guard), n=500, seed=1234.
# LOCAL/analysis-only. No HF Job, no submission.
set -uo pipefail

V=/tmp/senpai-venvs/20f658587e8a6643/bin/python
ROOT=/workspace/senpai/target
EVAL="$ROOT/research/downstream_quality_gsm8k/gsm8k_eval.py"
SUB="$ROOT/research/validity/ngram_spec_dec_int4/serve_scratch"
CKPT=/workspace/gemma_build/int4_g128_lmhead
OUT="$ROOT/research/validity/ngram_spec_dec_int4/_sweep/gsm8k"
cd "$ROOT"
mkdir -p "$OUT"

echo "[gsm8k-driver $(date -u +%H:%M:%S)] === ARM 1/2: AR floor (SPECULATIVE_CONFIG empty) ==="
"$V" "$EVAL" --submission "$SUB" --server-python "$V" \
  --label ng609_ar_int4g128 --regimes sampled --n 500 --n-shot 8 \
  --min-tokens 8 --seed 1234 --concurrency 32 \
  --serve-env "MODEL_ID=$CKPT" --serve-env "SPECULATIVE_CONFIG=" \
  --out-dir "$OUT" --save-text
echo "[gsm8k-driver $(date -u +%H:%M:%S)] ARM 1 exit=$?"

echo "[gsm8k-driver $(date -u +%H:%M:%S)] === ARM 2/2: ngram ng_max4_k5 ==="
"$V" "$EVAL" --submission "$SUB" --server-python "$V" \
  --label ng609_ngram_max4k5 --regimes sampled --n 500 --n-shot 8 \
  --min-tokens 8 --seed 1234 --concurrency 32 \
  --serve-env "MODEL_ID=$CKPT" \
  --serve-env 'SPECULATIVE_CONFIG={"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
  --out-dir "$OUT" --save-text
echo "[gsm8k-driver $(date -u +%H:%M:%S)] ARM 2 exit=$?"
echo "[gsm8k-driver $(date -u +%H:%M:%S)] DONE"
