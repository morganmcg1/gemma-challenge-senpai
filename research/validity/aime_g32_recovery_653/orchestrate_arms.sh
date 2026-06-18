#!/usr/bin/env bash
# PR #653 (lawine) -- run arms 3 (ours_g32, HEADLINE) then 2 (official_g32, cross-check)
# back-to-back on the single A10G AFTER arm 1 (shipped_g128) finishes. One server on
# :8000 at a time; each arm serves byte-for-byte the same gb6144/BI=1/seqs=1 config as
# the already-running shipped_g128 server -- only MODEL_ID changes. ours_g32 first so the
# must-have attribution (ours_g32 - shipped_g128) lands even if budget cuts official_g32.
set -uo pipefail

ROOT=/workspace/senpai/target
HERE="$ROOT/research/validity/aime_g32_recovery_653"
RES="$HERE/results"
LOG="$HERE/_orchestrate.log"
SERVE_OURS="$ROOT/research/validity/int4_body_quality_upside_639/serve_ours_g32.sh"
SERVE_OFFICIAL="$ROOT/research/validity/int4_body_quality_upside_639/serve_official_g32.sh"
SHIPPED_PIDFILE="$ROOT/research/validity/optionb_denom_0p22_gb6144/_server_int4ar_0p22.pid"

log() { echo "[orch $(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

kill_server() {  # $1 = pidfile
  local pf="$1" pid
  [[ -f "$pf" ]] || { log "no pidfile $pf, nothing to kill"; return 0; }
  pid="$(cat "$pf" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "killing server pid=$pid (group)"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && { log "SIGKILL pid=$pid"; kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true; }
  fi
  # wait for GPU memory to drain so the next load has headroom
  for _ in $(seq 1 30); do
    local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    log "gpu mem used=${used}MiB (waiting for drain)"
    [[ "${used:-9999}" -lt 3000 ]] && break
    sleep 3
  done
}

run_arm() {  # $1 = arm label, $2 = serve script
  local arm="$1"
  local serve="$2"
  local out="$RES/${arm}_aime_gb6144.json"
  if [[ -f "$out" ]]; then log "arm=$arm already has $out -- skipping"; return 0; fi
  log "=== ARM $arm: launching server ($serve) ==="
  if ! MAX_NUM_SEQS=1 bash "$serve" >>"$LOG" 2>&1; then
    log "FATAL: serve script for $arm failed; aborting (see $LOG and the arm _server log)"; return 1
  fi
  log "=== ARM $arm: server READY, running aime_eval ==="
  if ! bash "$HERE/run_arm.sh" "$arm" >>"$LOG" 2>&1; then
    log "FATAL: run_arm.sh $arm failed"; return 1
  fi
  [[ -f "$out" ]] || { log "FATAL: $arm produced no $out"; return 1; }
  log "=== ARM $arm DONE: $(grep -o 'maj@1=[0-9.]*' "$HERE/_${arm}_aime.out" | tail -1) -> $out ==="
}

log "orchestrator start; waiting for arm1 (shipped_g128) result json"
for _ in $(seq 1 240); do  # up to ~120 min
  [[ -f "$RES/shipped_g128_aime_gb6144.json" ]] && break
  sleep 30
done
if [[ ! -f "$RES/shipped_g128_aime_gb6144.json" ]]; then
  log "FATAL: arm1 json never appeared after wait window; aborting"; exit 1
fi
log "arm1 done; tearing down shipped_g128 server"
kill_server "$SHIPPED_PIDFILE"

run_arm ours_g32 "$SERVE_OURS" || exit 1
kill_server "$ROOT/research/validity/int4_body_quality_upside_639/_server_ours_g32.pid"

run_arm official_g32 "$SERVE_OFFICIAL" || { log "official_g32 failed/cut -- headline (ours_g32) already banked"; exit 0; }
kill_server "$ROOT/research/validity/int4_body_quality_upside_639/_server_official_g32.pid"

log "orchestrator COMPLETE -- all available arms done"
