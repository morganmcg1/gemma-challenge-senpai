#!/usr/bin/env bash
# PR #699: one SENPAI window's worth of work. Serve ONE body once, then eval a list of
# token budgets against a YEAR-SUBSET of AIME (the natural 30/30 checkpoint split), all
# on the #31 sampled basis. Writes results/<body>_<budget>_<yeartag>.json per sub-cell
# (SKIP-existing -> resumable across windows). Merge the year halves later with
# merge_year_splits.py, then feed the 4 merged cells to aggregate_aime_budget.py.
#
# The ONLY axis that varies between a body's two budget JSONs is request max_tokens
# (max_num_seqs / model-len / engine held fixed across ALL cells). analysis_only: local
# serve only, NO HF Job.
set -euo pipefail

BODY="${1:?body label: int4 | base}"
MODEL="${2:?model dir or HF id}"
PORT="${3:-8000}"
YEARTAG="${4:?year tag for output filename, e.g. y2024 | y2025}"
YEARS="${5:?comma years, e.g. 2024 | 2025-I,2025-II}"
BUDGETS="${6:-6144 12288}"        # space-separated list

K="${K:-10}"                      # maj@k samples/problem (#31 sampled draws); PR compute-bound floor
SEED="${SEED:-1234}"
MML="${MML:-13312}"
MNS="${MNS:-16}"                  # served-anchor batch width; HOLD FIXED across ALL cells
CC="${CC:-2}"                     # client concurrency: continuous-batching overlap of problems
LIMIT_ARG=""; [[ -n "${LIMIT:-}" ]] && LIMIT_ARG="--limit ${LIMIT}"

HERE="$(cd "$(dirname "$0")" && pwd)"
RES="$HERE/results"; mkdir -p "$RES"
PY="/workspace/senpai/target/.venv/bin/python"      # eval CLIENT venv (NOT the serve venv)
AIME="/workspace/senpai/target/research/downstream_quality_aime/aime_eval.py"

echo "[win] BODY=$BODY MODEL=$MODEL tag=$YEARTAG years=$YEARS budgets={$BUDGETS} K=$K mns=$MNS cc=$CC seed=$SEED $(date -u +%H:%M:%SZ)"

bash "$HERE/serve_body.sh" "$MODEL" "$PORT" "$MML" "$MNS"
SRV_PID="$(cat "$HERE/_server_${PORT}.pid")"
cleanup() { echo "[win] stopping server pid=$SRV_PID"; kill "$SRV_PID" 2>/dev/null || true; sleep 3; kill -9 "$SRV_PID" 2>/dev/null || true; }
trap cleanup EXIT

run_cell () {
  local budget="$1"
  local out="$RES/${BODY}_${budget}_${YEARTAG}.json"
  if [[ -s "$out" ]]; then echo "[win] SKIP existing $out"; return 0; fi
  echo "[win] $(date -u +%H:%M:%SZ) body=$BODY budget=$budget years=$YEARS -> $out"
  local t0=$(date +%s)
  "$PY" "$AIME" \
    --base-url "http://127.0.0.1:${PORT}" \
    --model gemma-4-e4b-it \
    --years "$YEARS" $LIMIT_ARG \
    --k "$K" --seed "$SEED" \
    --temperature 1.0 --top-p 0.95 --top-k 64 \
    --max-tokens "$budget" --min-tokens 8 \
    --no-thinking \
    --client-concurrency "$CC" \
    --label "${BODY}_${budget}_${YEARTAG}" \
    --out "$out" 2>&1 | tail -8
  echo "[win] cell done in $(( $(date +%s) - t0 ))s"
}

for b in $BUDGETS; do run_cell "$b"; done
echo "[win] DONE body=$BODY tag=$YEARTAG $(date -u +%H:%M:%SZ)"
