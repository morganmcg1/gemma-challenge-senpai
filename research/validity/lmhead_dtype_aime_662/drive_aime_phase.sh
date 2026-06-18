#!/usr/bin/env bash
# PR #662 -- drive the remaining AIME phase end-to-end on one A10G, sequentially:
#   (0) wait for the live bf16head AIME (pid $BF16_AIME_PID) to finish + tear its server
#   (1) shipped_g128 arm  (anchor)      serve -> AIME -> TPS -> teardown
#   (2) our_g128_int8head arm (cheaper) serve -> AIME -> TPS -> teardown
# Writes per-stage markers; a single _all_aime_done.marker at the end.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BF16_AIME_PID="${BF16_AIME_PID:?set BF16_AIME_PID}"
BF16_SRV_PID="${BF16_SRV_PID:?set BF16_SRV_PID}"
ALL_MARKER="$HERE/_all_aime_done.marker"
rm -f "$ALL_MARKER"
log(){ echo "[drive $(date -u +%T)Z] $*"; }

# (0) wait for the live bf16head AIME to finish, then free the GPU
log "waiting for bf16head AIME pid=$BF16_AIME_PID ..."
while kill -0 "$BF16_AIME_PID" 2>/dev/null; do sleep 15; done
log "bf16head AIME exited; result: $(ls -la "$HERE/results/aime_bf16head.json" 2>&1)"
log "tearing down bf16head server pid=$BF16_SRV_PID"
kill "$BF16_SRV_PID" 2>/dev/null || true
for i in $(seq 1 30); do kill -0 "$BF16_SRV_PID" 2>/dev/null || break; sleep 1; done
kill -9 "$BF16_SRV_PID" 2>/dev/null || true
sleep 5

# (1) shipped_g128 anchor
log "=== shipped_g128 arm ==="
ARM=shipped_g128 MODEL_DIR=/workspace/gemma_build/int4_g128_lmhead bash "$HERE/run_aime_arm.sh" \
  > "$HERE/_drive_shipped.log" 2>&1
log "shipped marker: $(cat "$HERE/_done_shipped_g128.marker" 2>&1)"
sleep 5

# (2) our_g128_int8head cheaper intermediate
log "=== our_g128_int8head arm ==="
ARM=our_g128_int8head MODEL_DIR=/workspace/gemma_build/our_g128_int8head bash "$HERE/run_aime_arm.sh" \
  > "$HERE/_drive_int8head.log" 2>&1
log "int8head marker: $(cat "$HERE/_done_our_g128_int8head.marker" 2>&1)"

echo "all_done $(date -u +%FT%TZ)" > "$ALL_MARKER"
log "ALL AIME ARMS DONE"
