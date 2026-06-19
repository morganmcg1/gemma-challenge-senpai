#!/usr/bin/env bash
# PR #703 -- MMLU-Pro on int4_g128_lmhead at the #31 gate basis, 5-seed + debias.
# Identical construction to the #590 base reference (n=2000 subset from --seed 12345,
# lewtun #31 sampling T=1.0 top_p=0.95 top_k=64, min_tokens=8, max_tokens=2048, then
# the 4096-token truncation de-bias) so int4 is directly comparable to base 0.6695.
# NOTE: output files keep the canonical `mmlu_base_fullhead_n{N}_s{S}` name ONLY
# because debias_mmlu.py hard-codes that pattern; the SERVED MODEL is int4 (the
# vLLM server on --base-url), so these JSONs hold int4 results, not base.
set -u
cd /workspace/senpai/target
PYI=/tmp/eval-serve-venv/bin/python
URL=http://127.0.0.1:8000/v1
OUT=research/int4body_gate_panel/runs
LOGD=research/int4body_gate_panel/logs
N="${N:-2000}"
SUBSET_SEED=12345
SEEDS=(1 2 3 4 5)
mkdir -p "$OUT" "$LOGD"

echo "[mmlu] START $(date -u +%H:%M:%SZ) N=$N subset_seed=$SUBSET_SEED seeds=${SEEDS[*]}"
t0=$(date +%s)
for s in "${SEEDS[@]}"; do
  OUTJSON="$OUT/mmlu_base_fullhead_n${N}_s${s}.json"
  if [[ -f "$OUTJSON" ]] && $PYI -c "import json,sys; d=json.load(open('$OUTJSON')); sys.exit(0 if d.get('n_samples',0)>=$N else 1)" 2>/dev/null; then
    echo "[mmlu]   seed=$s SKIP (complete)"; continue
  fi
  echo "[mmlu]   === seed=$s $(date -u +%H:%M:%SZ) ==="
  ts=$(date +%s)
  if $PYI research/validity/downstream_quality_eval/run_eval.py --task mmlu_pro --arm int4g128 \
      --n "$N" --seed "$SUBSET_SEED" --sampling-seed "$s" \
      --temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 8 \
      --max-tokens 2048 --max-connections 32 --base-url $URL \
      --out "$OUTJSON" --log-dir "$LOGD/mmlu_n${N}_s${s}" >"$LOGD/mmlu_n${N}_s${s}.log" 2>&1; then
    echo "[mmlu]   seed=$s OK wall=$(( $(date +%s)-ts ))s $(grep -h 'run_eval] task=' "$LOGD/mmlu_n${N}_s${s}.log" | tail -1)"
  else
    echo "[mmlu]   seed=$s FAILED"; tail -8 "$LOGD/mmlu_n${N}_s${s}.log"
  fi
done
echo "[mmlu] RAW SWEEP DONE wall=$(( $(date +%s)-t0 ))s $(date -u +%H:%M:%SZ)"

echo "[mmlu] DEBIAS (re-run truncated ids @4096) $(date -u +%H:%M:%SZ)"
$PYI research/validity/quality_gates_ci/debias_mmlu.py \
  --seeds "${SEEDS[@]}" --n "$N" --subset-seed "$SUBSET_SEED" --max-tokens 4096 \
  --base-url $URL --runs "$OUT" --logs "$LOGD" 2>&1 | tail -30
echo "[mmlu] ALL DONE wall=$(( $(date +%s)-t0 ))s $(date -u +%H:%M:%SZ)"
echo "[mmlu] debiased outputs: $OUT/mmlu_debias_n${N}_s{1..5}.json"
