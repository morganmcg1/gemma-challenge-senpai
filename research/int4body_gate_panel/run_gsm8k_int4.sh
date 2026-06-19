#!/usr/bin/env bash
# PR #703 -- GSM8K on the int4_g128_lmhead checkpoint at the #31 gate basis.
# Reuses #693's served int4 endpoint (max_model_len=6144) + the canonical #590
# gsm8k_eval harness. TWO arms over the SAME byte-identical 500-item seeded subset:
#   guarded  = min_tokens=8 EOS-guard (the #541 fix; the fair body-quality read)
#   noguard  = as-served (no min_tokens; exposes the first-token-EOS serving artifact)
# 5 sampling-seeds vary only the decode RNG. lewtun #31 sampling T=1.0 top_p=0.95 top_k=64.
set -u
cd /workspace/senpai/target
PY=/usr/bin/python3
URL=http://127.0.0.1:8000
OUT=research/int4body_gate_panel/runs
LOGD=research/int4body_gate_panel/logs
SEEDS=(1 2 3 4 5)
mkdir -p "$OUT" "$LOGD"

run_arm() {  # $1=label  $2=extra flags
  local label="$1"; shift
  for s in "${SEEDS[@]}"; do
    echo "[gsm8k:$label] === sampling-seed=$s $(date -u +%H:%M:%SZ) ==="
    if $PY research/downstream_quality_gsm8k/gsm8k_eval.py --base-url $URL \
        --label "$label" --regimes sampled --n 500 --seed 1234 --sampling-seed "$s" \
        --n-shot 8 --top-p 0.95 --top-k 64 --max-tokens 512 --concurrency 16 \
        --save-text "$@" --out-dir "$OUT" >"$LOGD/gsm8k_${label}_s${s}.log" 2>&1; then
      grep -h 'DONE' "$LOGD/gsm8k_${label}_s${s}.log" | tail -1
    else
      echo "[gsm8k:$label] seed=$s FAILED"; tail -8 "$LOGD/gsm8k_${label}_s${s}.log"
    fi
  done
}

t0=$(date +%s)
echo "[gsm8k] START $(date -u +%H:%M:%SZ)"
run_arm int4g128_guard   --min-tokens 8
run_arm int4g128_noguard
echo "[gsm8k] ALL DONE wall=$(( $(date +%s)-t0 ))s $(date -u +%H:%M:%SZ)"
echo "[gsm8k] outputs: $OUT/int4g128_{guard,noguard}_sampled_s{1..5}.json"
