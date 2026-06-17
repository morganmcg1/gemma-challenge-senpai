#!/usr/bin/env bash
# PR #614 -- finalize the GPQA-bar audit AFTER the clean-audit orchestrator has produced
# greedy_4096 + sampled_4096_s1..s5. Steps (idempotent; each skips if its output exists):
#   1. greedy_2048 -- the exact deterministic accuracy@2048 (item recnTTKdBzfuoZ7w7 FITS
#      at 2048: 2049+2048=4097<6144, so NO recovery needed for the 2048 run).
#   2. recover the 1 context-overflow item (recnTTKdBzfuoZ7w7) at max_tokens=4095 for the
#      six 4096-runs (greedy_4096 + 5 sampled) and splice it in (merge_recover.py).
#   3. cluster-bootstrap CI over the 5 merged sampled seeds (aggregate_ci.py).
#   4. bars + truncation/regime verdict (bars_verdict.py).
# Runs against the live validated-clean serve; touches the server, so run it only when the
# orchestrator is DONE (server otherwise dedicated to the sampled sweep). analysis_only.
set -u
ROOT=/workspace/senpai/target
HERE="$ROOT/research/validity/gpqa_bar_validity"
RUNS="$HERE/runs"
SUMM="$HERE/summaries"
EVALPY=/tmp/eval-serve-venv/bin/python
RUN_EVAL="$ROOT/research/validity/downstream_quality_eval/run_eval.py"
RUN_ONE="$HERE/run_one.sh"
AGG="$ROOT/research/validity/quality_gates_ci/aggregate_ci.py"
OVERFLOW_ID=recnTTKdBzfuoZ7w7
IDS="$HERE/_overflow_ids.json"
GPQA_BAR=0.471
mkdir -p "$RUNS" "$SUMM"
echo "[\"$OVERFLOW_ID\"]" > "$IDS"
echo "[finalize] START $(date -u +%FT%TZ)"

# server liveness guard
if ! curl -s --max-time 5 "http://127.0.0.1:8000/v1/models" 2>/dev/null | grep -q gemma-4-e4b-it; then
  echo "[finalize] FATAL: base server not responding on :8000"; exit 1
fi

# --- 1) greedy_2048 (no recovery; item fits at 2048) ---
g2="$RUNS/greedy_2048.json"
if [ -s "$g2" ]; then echo "[finalize] greedy_2048 exists, skip"; else
  bash "$RUN_ONE" greedy_2048 0.0 1.0 0 2048 0 "$g2" || { echo "[finalize] ABORT greedy_2048"; exit 1; }
fi

# --- 2) recover overflow item @4095 for the six 4096-runs + merge ---
recover_and_merge() {
  local label="$1" temp="$2" topp="$3" topk="$4" sseed="$5" base="$6"
  local rec="$RUNS/_rec_${label}.json"
  local merged="$RUNS/${label}.merged.json"
  if [ -s "$merged" ]; then echo "[finalize] $label merged exists, skip"; return 0; fi
  if [ ! -s "$base" ]; then echo "[finalize] WAIT: $base not present yet"; return 2; fi
  "$EVALPY" "$RUN_EVAL" --task gpqa_diamond --arm "rec_${label}" --seed 12345 \
    --temperature "$temp" --top-p "$topp" --top-k "$topk" --max-tokens 4095 --min-tokens 8 \
    --max-connections 4 --sampling-seed "$sseed" --ids-file "$IDS" \
    --out "$rec" --log-dir "$HERE/_inspect_logs/rec_${label}" || { echo "[finalize] ABORT recover $label"; return 1; }
  "$EVALPY" "$HERE/merge_recover.py" --base "$base" --recover "$rec" --out "$merged" || return 1
}

recover_and_merge greedy_4096 0.0 1.0 0 0 "$RUNS/greedy_4096.json" || exit 1
for s in 1 2 3 4 5; do
  recover_and_merge "sampled_4096_s${s}" 1.0 0.95 64 "$s" "$RUNS/sampled_4096_s${s}.json" || exit 1
done

# --- 3) cluster-bootstrap CI over the 5 merged sampled seeds ---
"$EVALPY" "$AGG" --task gpqa_diamond --label GPQA-D-sampled --bar "$GPQA_BAR" \
  --out "$SUMM/sampled_ci.json" \
  --inputs "$RUNS"/sampled_4096_s1.merged.json "$RUNS"/sampled_4096_s2.merged.json \
           "$RUNS"/sampled_4096_s3.merged.json "$RUNS"/sampled_4096_s4.merged.json \
           "$RUNS"/sampled_4096_s5.merged.json || exit 1

# --- 4) bars + verdict ---
"$EVALPY" "$HERE/bars_verdict.py" \
  --greedy-4096 "$RUNS/greedy_4096.merged.json" \
  --greedy-2048 "$g2" \
  --sampled-4096 "$RUNS"/sampled_4096_s1.merged.json "$RUNS"/sampled_4096_s2.merged.json \
                 "$RUNS"/sampled_4096_s3.merged.json "$RUNS"/sampled_4096_s4.merged.json \
                 "$RUNS"/sampled_4096_s5.merged.json \
  --sampled-agg "$SUMM/sampled_ci.json" \
  --out "$SUMM/bars_verdict.json" || exit 1

echo "[finalize] ALL DONE $(date -u +%FT%TZ)"
