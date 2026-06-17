#!/usr/bin/env bash
# PR #590 -- GSM8K multi-seed decode realizations on the live base_fullhead server.
#
# n=500 fixed seeded subset (--seed 1234 builds a byte-identical 500-item subset +
# 8-shot fewshot block). FIVE sampling-seeds vary ONLY the decode RNG (--sampling-seed
# 1..5), so the question set and few-shot prompt are identical across realizations ->
# aggregate_ci.py merges 5 columns/question. lewtun #31 sampled regime hardcodes T=1.0;
# top_p=0.95 top_k=64, min_tokens=8 EOS-guard, max_tokens=512 (GSM8K answers are short).
# Each pass ~4 min (<< 90-min/run bound); run sequentially for clean per-seed logs.
set -u
cd /workspace/senpai/target
PY=/usr/bin/python3                       # gsm8k_eval uses stdlib urllib (no inspect_ai)
URL0=http://127.0.0.1:8000                # gsm8k_eval appends /v1/chat/completions itself
OUT=research/validity/quality_gates_ci/runs
LOGD=research/validity/quality_gates_ci/logs
SEEDS=(1 2 3 4 5)

mkdir -p "$OUT" "$LOGD"
echo "[gsm8k] multiseed START $(date -u +%H:%M:%SZ) sampling-seeds=${SEEDS[*]} (subset seed=1234)"
t0=$(date +%s)
fail=0
for s in "${SEEDS[@]}"; do
  echo "[gsm8k]   === sampling-seed=$s $(date -u +%H:%M:%SZ) ==="
  if $PY research/downstream_quality_gsm8k/gsm8k_eval.py --base-url $URL0 \
      --label base_fullhead --regimes sampled --n 500 --seed 1234 --sampling-seed "$s" \
      --n-shot 8 --top-p 0.95 --top-k 64 --max-tokens 512 --min-tokens 8 \
      --concurrency 16 --out-dir "$OUT" >"$LOGD/gsm8k_s${s}.log" 2>&1; then
    grep -h 'DONE' "$LOGD/gsm8k_s${s}.log" | tail -1
  else
    echo "[gsm8k]   sampling-seed=$s FAILED (tail below)"; tail -8 "$LOGD/gsm8k_s${s}.log"; fail=1
  fi
done
echo "[gsm8k] multiseed DONE wall=$(( $(date +%s)-t0 ))s $(date -u +%H:%M:%SZ) fail=$fail"
echo "[gsm8k] outputs: $OUT/base_fullhead_sampled_s{1..5}.json"
exit $fail
