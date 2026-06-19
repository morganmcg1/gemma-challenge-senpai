#!/usr/bin/env bash
# Master orchestration for PR #750 remaining GPU work, single A10G, serial.
# Order: headline TPS first (banked even if a later stage fails), then identity.
#   Stage 1  BI=0 spec-ON   -> bench tps_BI0 + decode cand_BI0 (128)
#   Stage 2  int4_qat anchor-> bench local_int4_qat (HEADLINE anchor)
#   Stage 3  BI=0 spec-OFF  -> decode ref_BI0 (128) -> identity_BI0
#   Stage 4  BI=1 spec-OFF  -> decode 48 COLD -> cold AR determinism floor
# Each stage: kill prior server, boot fresh, wait-ready, run, teardown.
set -uo pipefail
cd "$(dirname "$0")"
D=runs
PROBE=/tmp/bench-venv/bin/python
M="$D/run_all.master.log"
: > "$M"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$M"; }

SRV_PID=""
boot() {  # $1 = launch command string, $2 = server logfile
  setsid bash -c "$1" > "$2" 2>&1 &
  SRV_PID=$!
  log "  booted pid=$SRV_PID  ($1)"
}
ready() {  # $1 = timeout_s ; returns 0 when /v1/models answers 200
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
  if [ -n "$pgid" ]; then kill -TERM -- "-$pgid" 2>/dev/null; fi
  sleep 8
  if [ -n "$pgid" ]; then kill -KILL -- "-$pgid" 2>/dev/null; fi
  SRV_PID=""
  # Wait for GPU memory to actually release so the next boot won't OOM.
  local deadline=$(( $(date +%s) + 90 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$used" ] && [ "$used" -lt 3000 ] && break
    sleep 4
  done
  log "  torn down (gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1))"
}

# Stage 0: clear any server already holding port 8000 / the GPU.
log "STAGE 0: clearing leftover servers on :8000"
for p in $(pgrep -f "vllm.entrypoints.openai.api_server" 2>/dev/null); do
  pg=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' '); [ -n "$pg" ] && kill -TERM -- "-$pg" 2>/dev/null
done
sleep 10
for p in $(pgrep -f "vllm.entrypoints.openai.api_server" 2>/dev/null); do
  pg=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' '); [ -n "$pg" ] && kill -KILL -- "-$pg" 2>/dev/null
done
sleep 6
log "STAGE 0 done"

# ---------------------------------------------------------------- Stage 1
log "STAGE 1: BI=0 spec-ON  (tps_BI0 + cand_BI0)"
boot "bash launch_server.sh 0 0 8000" "$D/server_bi0_specon.log"
if ready 1200; then
  log "  ready; sampling gpu mem"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$D/mem_bi0.txt" 2>/dev/null || true
  log "  bench (sglang 128) ..."
  timeout 2400 bash run_bench.sh 8000 "$D/bench_bi0.jsonl" > "$D/bench_bi0.log" 2>&1 && log "  bench done" || log "  !! bench failed/timeout"
  log "  decode cand_BI0 (128) ..."
  timeout 2400 bash run_decode.sh 8000 "$D/decode_cand_bi0.jsonl" "$D/decode_cand_bi0_summary.json" 128 > "$D/decode_cand_bi0.log" 2>&1 && log "  decode done" || log "  !! decode failed/timeout"
else
  log "  !! STAGE 1 boot failed"
fi
teardown
log "STAGE_DONE: 1"

# ---------------------------------------------------------------- Stage 2
log "STAGE 2: int4_qat anchor (local_int4_qat)"
boot "bash launch_int4qat.sh 8000" "$D/server_int4qat.log"
if ready 1200; then
  log "  ready; bench (sglang 128) ..."
  timeout 2400 bash run_bench.sh 8000 "$D/bench_int4qat.jsonl" > "$D/bench_int4qat.log" 2>&1 && log "  bench done" || log "  !! bench failed/timeout"
else
  log "  !! STAGE 2 boot failed"
fi
teardown
log "STAGE_DONE: 2"

# ---------------------------------------------------------------- Stage 3
log "STAGE 3: BI=0 spec-OFF (ref_BI0 -> identity_BI0)"
boot "bash launch_server.sh 0 1 8000" "$D/server_bi0_specoff.log"
if ready 1200; then
  log "  ready; decode ref_BI0 (128) ..."
  timeout 2400 bash run_decode.sh 8000 "$D/decode_ref_bi0.jsonl" "$D/decode_ref_bi0_summary.json" 128 > "$D/decode_ref_bi0.log" 2>&1 && log "  decode done" || log "  !! decode failed/timeout"
else
  log "  !! STAGE 3 boot failed"
fi
teardown
log "STAGE_DONE: 3"

# ---------------------------------------------------------------- Stage 4
log "STAGE 4: BI=1 spec-OFF COLD (48-prompt AR determinism floor)"
boot "bash launch_server.sh 1 1 8000" "$D/server_bi1_specoff_cold.log"
if ready 1200; then
  log "  ready; decode cold 48 ..."
  timeout 1200 bash run_decode.sh 8000 "$D/decode_ref_bi1_cold48.jsonl" "$D/decode_ref_bi1_cold48_summary.json" 48 > "$D/decode_ref_bi1_cold48.log" 2>&1 && log "  decode done" || log "  !! decode failed/timeout"
else
  log "  !! STAGE 4 boot failed"
fi
teardown
log "STAGE_DONE: 4"

log "ALL STAGES COMPLETE"
