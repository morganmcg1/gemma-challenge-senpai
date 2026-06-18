#!/usr/bin/env bash
# PR #662 -- chain the GPQA-D guard phase after the AIME phase completes.
# Waits (bounded) for the AIME orchestrator's _all_aime_done.marker, then runs
# drive_gpqa_phase.sh (3 arms, serve->GPQA n=198 greedy->teardown). Pure local
# analysis_only; no HF Job, no served-file change. Idempotent: GPQA driver skips
# any arm whose results/gpqa_<arm>.json already exists.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
AIME_MARKER="$HERE/_all_aime_done.marker"
log(){ echo "[chain $(date -u +%T)Z] $*"; }

# Bound the wait: AIME phase is shipped(~55m)+int8head(~55m) from ~12:31Z, so
# ~150 min is a generous ceiling. If it never lands, exit non-zero (no GPQA).
log "waiting for AIME marker $AIME_MARKER (<=180 polls x 60s) ..."
for i in $(seq 1 180); do
  [[ -f "$AIME_MARKER" ]] && { log "AIME done: $(cat "$AIME_MARKER")"; break; }
  sleep 60
done
if [[ ! -f "$AIME_MARKER" ]]; then
  log "FATAL: AIME marker never appeared after 180 min; NOT launching GPQA"; exit 1
fi

# Make sure the AIME arm's server is fully torn down before GPQA serves on :8000.
sleep 10
log "launching GPQA phase ..."
bash "$HERE/drive_gpqa_phase.sh" > "$HERE/_drive_gpqa_phase.log" 2>&1
rc=$?
log "GPQA phase exited rc=$rc; marker: $(cat "$HERE/_all_gpqa_done.marker" 2>&1)"
exit $rc
