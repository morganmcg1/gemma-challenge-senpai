#!/usr/bin/env bash
# PR #765 orchestrator: profile the FIRE spec-ON served path (int4_mtp_batchinv)
# under BI=0 then BI=1 on a single A10G, serially. Each arm boots ONE fresh
# profiler-enabled server (reusing #759's launch_prof_server.sh verbatim -> same
# fire config, only --profiler-config appended), then drives a generation-length
# sweep of pure-decode windows + a cold-prefill window in that single boot, then
# tears down (wait for GPU mem release). Finally parses both arms into the
# operating-point ledger.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
LAUNCH="$ROOT/../bi_tax_op_ledger_759/launch_prof_server.sh"   # reuse #759 fire config
D="$ROOT/runs"
mkdir -p "$D"
DRIVER=/tmp/bench-venv/bin/python      # stdlib-only driver; any interpreter works
PROBE=/tmp/bench-venv/bin/python
GEN_SWEEP="${GEN_SWEEP:-128,256,512,1024}"
M="$D/run_sweep.master.log"
: > "$M"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$M"; }

[ -f "$LAUNCH" ] || { log "FATAL: missing $LAUNCH"; exit 1; }

SRV_PID=""
boot() { setsid bash -c "$1" > "$2" 2>&1 & SRV_PID=$!; log "  booted pid=$SRV_PID"; }
ready() {
  local deadline=$(( $(date +%s) + $1 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if "$PROBE" - <<'PY' 2>/dev/null
import sys,urllib.request
try:
    r=urllib.request.urlopen("http://127.0.0.1:8000/v1/models",timeout=5)
    sys.exit(0 if r.status==200 else 1)
except Exception:
    sys.exit(1)
PY
    then return 0; fi
    if [ -n "$SRV_PID" ] && ! kill -0 "$SRV_PID" 2>/dev/null; then log "  !! server pid $SRV_PID exited before ready"; return 1; fi
    sleep 5
  done
  log "  !! readiness timeout after $1 s"; return 1
}
teardown() {
  [ -z "$SRV_PID" ] && return 0
  local pgid; pgid=$(ps -o pgid= -p "$SRV_PID" 2>/dev/null | tr -d ' ')
  [ -n "$pgid" ] && kill -TERM -- "-$pgid" 2>/dev/null
  sleep 8
  [ -n "$pgid" ] && kill -KILL -- "-$pgid" 2>/dev/null
  SRV_PID=""
  local deadline=$(( $(date +%s) + 120 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$used" ] && [ "$used" -lt 3000 ] && break
    sleep 4
  done
  log "  torn down (gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1))"
}

# Stage 0: clear leftovers on :8000
log "STAGE 0: clearing leftover api_server on :8000"
for p in $(pgrep -f "vllm.entrypoints.openai.api_server" 2>/dev/null); do
  pg=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' '); [ -n "$pg" ] && kill -TERM -- "-$pg" 2>/dev/null
done
sleep 8
for p in $(pgrep -f "vllm.entrypoints.openai.api_server" 2>/dev/null); do
  pg=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' '); [ -n "$pg" ] && kill -KILL -- "-$pg" 2>/dev/null
done
sleep 5
log "STAGE 0 done (gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1))"

run_arm() {  # $1 = BI (0|1)
  local bi="$1"
  local tdir="$D/trace_bi${bi}"
  rm -rf "$tdir"; mkdir -p "$tdir"
  log "ARM BI=${bi}: boot prof server"
  boot "bash '${LAUNCH}' ${bi} 8000 '${tdir}'" "$D/server_bi${bi}.log"
  if ready 1200; then
    log "  ready; gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)"
    log "  driving sweep (warmup=4, prompt~200, gen_sweep=${GEN_SWEEP}, +prefill) ..."
    timeout 2400 "$DRIVER" profile_sweep.py \
      --port 8000 --model gemma-4-e4b-it --trace-dir "$tdir" \
      --prompt-tokens 200 --warmup 4 --warmup-tokens 96 \
      --gen-sweep "${GEN_SWEEP}" --throwaway-tokens 32 \
      --summary "$D/arm_bi${bi}_summary.json" \
      > "$D/driver_bi${bi}.log" 2>&1 && log "  sweep done" || log "  !! sweep failed/timeout"
  else
    log "  !! ARM BI=${bi} boot failed"
  fi
  teardown
  log "ARM_DONE: BI=${bi}"
}

# Arms come from CLI ("0", "1", or "0 1"); default both.
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(0 1)
for a in "${ARMS[@]}"; do run_arm "$a"; done

if [ -f "$D/arm_bi0_summary.json" ] && [ -f "$D/arm_bi1_summary.json" ]; then
  log "ANALYZE: building operating-point ledger"
  "$DRIVER" analyze_sweep.py \
    --bi0-summary "$D/arm_bi0_summary.json" \
    --bi1-summary "$D/arm_bi1_summary.json" \
    --out "$D/sweep_ledger.json" 2>&1 | tee -a "$M" || log "  !! analyze failed"
fi

log "ALL DONE"
