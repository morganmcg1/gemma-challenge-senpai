#!/usr/bin/env bash
# PR #590 -- MMLU-Pro multi-seed decode realizations on the live base_fullhead server.
#
# N (env, default 500) fixed seeded subset (--seed 12345 -> byte-identical N-question
# subset). FIVE sampling-seeds vary ONLY the decode RNG (--sampling-seed 1..5) so the
# question set is identical across realizations -> aggregate_ci.py merges 5 cols/question.
# lewtun #31 sampling (T=1.0 top_p=0.95 top_k=64) + min_tokens=8, max_tokens=2048.
#
# N is sized from a timed calibration pass: MMLU-Pro is the swing gate (point ~0.63 vs
# bar 0.605, ~0.026 slack) so the CI half-width ~0.95/sqrt(N) must be < slack -> N in the
# low thousands to clear. Each pass must stay < 90 min, so N is capped by measured
# per-question wall. Passes run SEQUENTIALLY (each fills the 32-way server alone).
set -u
cd /workspace/senpai/target
PYI=/tmp/eval-serve-venv/bin/python        # mmlu_pro needs inspect_ai/inspect_evals
URL=http://127.0.0.1:8000/v1               # inspect openai-api provider wants /v1 base
OUT=research/validity/quality_gates_ci/runs
LOGD=research/validity/quality_gates_ci/logs
N="${N:-500}"
SUBSET_SEED="${SUBSET_SEED:-12345}"
SEEDS=(1 2 3 4 5)

mkdir -p "$OUT" "$LOGD"
echo "[mmlu] multiseed START $(date -u +%H:%M:%SZ) N=$N subset_seed=$SUBSET_SEED sampling-seeds=${SEEDS[*]}"
t0=$(date +%s)
fail=0
for s in "${SEEDS[@]}"; do
  OUTJSON="$OUT/mmlu_base_fullhead_n${N}_s${s}.json"
  # Resumable: skip a seed whose output already has all N graded samples (a mid-sweep
  # restart must not redo ~37-min passes already on disk).
  if [[ -f "$OUTJSON" ]] && $PYI -c "import json,sys; d=json.load(open('$OUTJSON')); sys.exit(0 if d.get('n_samples',0)>=$N else 1)" 2>/dev/null; then
    echo "[mmlu]   === sampling-seed=$s N=$N SKIP (complete output exists) ==="; continue
  fi
  echo "[mmlu]   === sampling-seed=$s N=$N $(date -u +%H:%M:%SZ) ==="
  ts=$(date +%s)
  if $PYI research/validity/downstream_quality_eval/run_eval.py --task mmlu_pro --arm base_fullhead \
      --n "$N" --seed "$SUBSET_SEED" --sampling-seed "$s" \
      --temperature 1.0 --top-p 0.95 --top-k 64 --min-tokens 8 \
      --max-tokens 2048 --max-connections 32 --base-url $URL \
      --out "$OUT/mmlu_base_fullhead_n${N}_s${s}.json" \
      --log-dir "$LOGD/mmlu_n${N}_s${s}" >"$LOGD/mmlu_n${N}_s${s}.log" 2>&1; then
    echo "[mmlu]   ss=$s OK wall=$(( $(date +%s)-ts ))s $(grep -h 'run_eval] task=' "$LOGD/mmlu_n${N}_s${s}.log" | tail -1)"
  else
    echo "[mmlu]   ss=$s FAILED (tail below)"; tail -8 "$LOGD/mmlu_n${N}_s${s}.log"; fail=1
  fi
done
echo "[mmlu] multiseed DONE wall=$(( $(date +%s)-t0 ))s $(date -u +%H:%M:%SZ) fail=$fail"
echo "[mmlu] outputs: $OUT/mmlu_base_fullhead_n${N}_s{1..5}.json"
exit $fail
