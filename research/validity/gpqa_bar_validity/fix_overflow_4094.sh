#!/usr/bin/env bash
# PR #614 -- fix the off-by-one in the overflow-item recovery. The single GPQA-Diamond
# item recnTTKdBzfuoZ7w7 has input_tokens=2050 (vLLM reported it exactly), so the prior
# recovery at max_tokens=4095 re-triggered a context-overflow (2050+4095=6145>6144) and
# the item stayed force-scored WRONG in every merged file. The largest output budget the
# 6144 context allows for a 2050-tok input is 4094 (2050+4094=6144). Re-run JUST that one
# item at 4094 for greedy + 5 sampled arms on the SAME live validated-clean bf16 serve,
# splice over the ORIGINAL base runs, then recompute the sampled CI + bars verdict.
# analysis_only; no served-file change.
set -u
ROOT=/workspace/senpai/target
HERE="$ROOT/research/validity/gpqa_bar_validity"
RUNS="$HERE/runs"
SUMM="$HERE/summaries"
EVALPY=/tmp/eval-serve-venv/bin/python
RUN_EVAL="$ROOT/research/validity/downstream_quality_eval/run_eval.py"
AGG="$ROOT/research/validity/quality_gates_ci/aggregate_ci.py"
IDS="$HERE/_overflow_ids.json"
# vLLM's "input_tokens=N" in the overflow error is a MOVING LOWER BOUND (= max_model_len+1
# - max_tokens: 4095->2050, 4094->2051, 4090->2055), NOT the real prompt length. /tokenize
# reports the item's true templated prompt = 2427 tokens, so under the 6144 context the max
# feasible output budget is 6144-2427 = 3717. The standard 4096 budget is physically
# infeasible for this one long-prompt item; 3700 gives ~the full feasible budget with margin
# (2427+3700=6127<6144). All 197 other items get 4096; this one gets 3700 (documented).
MAXTOK=3700
GPQA_BAR=0.471
echo "[fix4094] START $(date -u +%FT%TZ) max_tokens=$MAXTOK"

if ! curl -s --max-time 5 "http://127.0.0.1:8000/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
  echo "[fix4094] FATAL: base server not responding on :8000"; exit 1
fi

recover4094_and_merge() {
  local label="$1" temp="$2" topp="$3" topk="$4" sseed="$5"
  local base="$RUNS/${label}.json"
  local rec="$RUNS/_rec3700_${label}.json"
  local merged="$RUNS/${label}.merged.json"
  [ -s "$base" ] || { echo "[fix4094] MISSING base $base"; return 1; }
  "$EVALPY" "$RUN_EVAL" --task gpqa_diamond --arm "rec3700_${label}" --seed 12345 \
    --temperature "$temp" --top-p "$topp" --top-k "$topk" --max-tokens "$MAXTOK" --min-tokens 8 \
    --max-connections 4 --sampling-seed "$sseed" --ids-file "$IDS" \
    --out "$rec" --log-dir "$HERE/_inspect_logs/rec3700_${label}" || { echo "[fix4094] ABORT eval $label"; return 1; }
  "$EVALPY" "$HERE/merge_recover.py" --base "$base" --recover "$rec" --out "$merged" || { echo "[fix4094] ABORT merge $label"; return 1; }
}

recover4094_and_merge greedy_4096 0.0 1.0 0 0 || exit 1
for s in 1 2 3 4 5; do
  recover4094_and_merge "sampled_4096_s${s}" 1.0 0.95 64 "$s" || exit 1
done

"$EVALPY" "$AGG" --task gpqa_diamond --label GPQA-D-sampled --bar "$GPQA_BAR" \
  --out "$SUMM/sampled_ci.json" \
  --inputs "$RUNS"/sampled_4096_s1.merged.json "$RUNS"/sampled_4096_s2.merged.json \
           "$RUNS"/sampled_4096_s3.merged.json "$RUNS"/sampled_4096_s4.merged.json \
           "$RUNS"/sampled_4096_s5.merged.json || exit 1

"$EVALPY" "$HERE/bars_verdict.py" \
  --greedy-4096 "$RUNS/greedy_4096.merged.json" \
  --greedy-2048 "$RUNS/greedy_2048.json" \
  --sampled-4096 "$RUNS"/sampled_4096_s1.merged.json "$RUNS"/sampled_4096_s2.merged.json \
                 "$RUNS"/sampled_4096_s3.merged.json "$RUNS"/sampled_4096_s4.merged.json \
                 "$RUNS"/sampled_4096_s5.merged.json \
  --sampled-agg "$SUMM/sampled_ci.json" \
  --out "$SUMM/bars_verdict.json" || exit 1

echo "[fix4094] ALL DONE $(date -u +%FT%TZ)"
