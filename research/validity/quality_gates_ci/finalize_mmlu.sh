#!/usr/bin/env bash
# PR #590 -- autonomous MMLU-Pro finalize.
# 1. wait for the master sweep orchestrator (run_all_multiseed.sh) to exit (it owns
#    seeds 3,4,5); 2. re-run the MMLU sweep (per-seed resumable -> redoes ONLY the
#    seed that crashed on the transient disk-full, skips complete seeds); 3. verify
#    all 5 seeds are complete (n_samples>=N); 4. aggregate the MMLU-Pro CI summary.
# Stops at aggregation -- W&B logging + the results comment are done by hand after a
# human/agent review of the CI-lb vs the 0.605 bar (escalation-N decision point).
set -u
cd /workspace/senpai/target
HERE=research/validity/quality_gates_ci
OUT=$HERE/runs
N=2000
PYI=/tmp/eval-serve-venv/bin/python
log(){ echo "[finalize $(date -u +%H:%M:%SZ)] $*"; }

MASTER_PID="${MASTER_PID:-1471716}"   # run_all_multiseed.sh (seeds 3,4,5)
log "waiting for master orchestrator PID $MASTER_PID to exit (cap 7200s)..."
waited=0
while kill -0 "$MASTER_PID" 2>/dev/null; do
  sleep 30; waited=$((waited+30))
  if [[ $waited -ge 7200 ]]; then log "WARN: master still alive after 7200s; proceeding anyway"; break; fi
done
log "master no longer running (waited ${waited}s)."

log "running MMLU sweep to fill the missing seed (resumable: skips complete seeds)..."
N=$N bash "$HERE/run_mmlu_multiseed.sh" >"$HERE/_mmlu_redo.out" 2>&1
log "mmlu sweep redo rc=$?"

miss=0
for s in 1 2 3 4 5; do
  f="$OUT/mmlu_base_fullhead_n${N}_s${s}.json"
  if [[ -f "$f" ]] && $PYI -c "import json,sys;d=json.load(open('$f'));sys.exit(0 if d.get('n_samples',0)>=$N else 1)" 2>/dev/null; then
    acc=$($PYI -c "import json;print(json.load(open('$f'))['accuracy'])" 2>/dev/null)
    log "seed $s OK (acc=$acc)"
  else
    log "seed $s MISSING/INCOMPLETE"; miss=1
  fi
done
if [[ $miss -ne 0 ]]; then
  log "FATAL: not all 5 seeds complete; NOT aggregating."
  echo "FINALIZE_FAIL $(date -u +%FT%TZ)" > "$HERE/_finalize.status"; exit 1
fi

log "aggregating MMLU-Pro CI (bar=0.605)..."
$PYI "$HERE/aggregate_ci.py" --task mmlu_pro --label MMLU_Pro --bar 0.605 \
  --out "$HERE/summaries/mmlu_sampled.json" \
  --inputs "$OUT"/mmlu_base_fullhead_n${N}_s{1,2,3,4,5}.json
arc=$?
log "aggregate rc=$arc"
if [[ $arc -ne 0 ]]; then echo "FINALIZE_FAIL_AGG $(date -u +%FT%TZ)" > "$HERE/_finalize.status"; exit 1; fi
echo "FINALIZE_DONE $(date -u +%FT%TZ)" > "$HERE/_finalize.status"
log "DONE -> $HERE/summaries/mmlu_sampled.json"
