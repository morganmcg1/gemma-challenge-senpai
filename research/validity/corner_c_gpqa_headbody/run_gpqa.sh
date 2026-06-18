#!/usr/bin/env bash
# PR #652 -- corner-C GPQA-Diamond panel against an already-running corner-C server
# on :8000 (serve_cornerc_0p22.sh). Mirrors ubel #638's run_panel_int4ar.sh GPQA
# legs EXACTLY (same run_eval.py client, dseed=12345, T=1/top_p=0.95/top_k=64,
# sampling-seeds 0..9, MT=6144, min_tokens=8, conc=16) so corner C and corner D
# share one harness/protocol and C-D is a pure single-variable (lm_head) contrast.
#
#   [1] GPQA-D sampled 10-seed n=1980  (dseed 12345 fixed; sseed 0..9)  -> primary
#   [2] GPQA-D greedy   n=198          (temp 0, single pass)            -> cross-check
#
# Resumable: a seed/greedy leg whose JSON already has n_scored>=190 is skipped.
# Each per-seed leg writes its own JSON so a crash keeps prior legs.
set -uo pipefail
cd /workspace/senpai/target
HERE=research/validity/corner_c_gpqa_headbody
CLIENT="${CLIENT:-/tmp/eval-serve-venv/bin/python}"
RUN_EVAL=research/validity/downstream_quality_eval/run_eval.py
BASE=http://127.0.0.1:8000
MODEL=gemma-4-e4b-it
MT=6144
DSEED=12345
CONC="${CONC:-16}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
RES="$HERE/results"
mkdir -p "$RES"
STATUS="$HERE/_panel.status"
complete () { "$CLIENT" -c "import json,sys;d=json.load(open('$1'));sys.exit(0 if d.get('n_scored',0)>=190 else 1)" 2>/dev/null; }

echo "PANEL-cornerC-START $(date -u +%FT%TZ) seeds=[$SEEDS] conc=$CONC server_pid=$(cat "$HERE/_server_cornerc_0p22.pid" 2>/dev/null)" | tee -a "$STATUS"

# ---- [1] GPQA-D sampled 10-seed n=1980 (primary) ----------------------------
for s in $SEEDS; do
  o="$RES/cornerc_gpqa_sampled_s${s}.json"
  if [[ -f "$o" ]] && complete "$o"; then echo "  sampled s=$s SKIP (complete) $(date -u +%H:%M:%S)" | tee -a "$STATUS"; continue; fi
  echo "  sampled s=$s START $(date -u +%H:%M:%S)" | tee -a "$STATUS"; ts=$(date +%s)
  "$CLIENT" "$RUN_EVAL" \
    --task gpqa_diamond --arm cornerc_sampled --out "$o" \
    --seed "$DSEED" --temperature 1.0 --top-p 0.95 --top-k 64 --sampling-seed "$s" \
    --max-tokens "$MT" --min-tokens 8 --max-connections "$CONC" \
    --base-url "$BASE/v1" --model "$MODEL" \
    > "$HERE/_gpqa_sampled_s${s}.out" 2>&1
  rc=$?
  echo "  sampled s=$s rc=$rc wall=$(( $(date +%s)-ts ))s $(date -u +%H:%M:%S): $(grep -o 'acc=[0-9.]*' "$HERE/_gpqa_sampled_s${s}.out" | head -1)" | tee -a "$STATUS"
done

# ---- [2] GPQA-D greedy n=198 (cross-check) ----------------------------------
og="$RES/cornerc_gpqa_greedy.json"
if [[ -f "$og" ]] && complete "$og"; then
  echo "  greedy SKIP (complete) $(date -u +%H:%M:%S)" | tee -a "$STATUS"
else
  echo "  greedy START $(date -u +%H:%M:%S)" | tee -a "$STATUS"; ts=$(date +%s)
  "$CLIENT" "$RUN_EVAL" \
    --task gpqa_diamond --arm cornerc_greedy --out "$og" \
    --seed "$DSEED" --max-tokens "$MT" --min-tokens 8 --max-connections "$CONC" \
    --base-url "$BASE/v1" --model "$MODEL" \
    > "$HERE/_gpqa_greedy.out" 2>&1
  rc=$?
  echo "  greedy rc=$rc wall=$(( $(date +%s)-ts ))s $(date -u +%H:%M:%S): $(grep -o 'acc=[0-9.]*' "$HERE/_gpqa_greedy.out" | head -1)" | tee -a "$STATUS"
fi

echo "PANEL-cornerC-DONE $(date -u +%FT%TZ)" | tee -a "$STATUS"
