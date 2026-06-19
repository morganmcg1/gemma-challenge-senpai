#!/usr/bin/env bash
# PR #699 continuation driver. Waits for the in-flight int4/y2024 window to exit
# (it kills its own server on EXIT), then runs the remaining 3 windows in sequence
# against ONE warmed server each. SKIP-existing in run_window.sh protects completed
# cells, so this is idempotent / safe to re-launch. All cells share IDENTICAL serve
# params (mns=16, mml=13312, gpu_mem=0.90, BATCH_INVARIANT=1); the ONLY varied axis
# is request max_tokens and the body. base bf16 (15GB weights) fits because the eval
# client only drives cc=2 concurrent requests. analysis_only, NO HF Job.
set -uo pipefail   # NOT -e: one window failing must not abort the rest.

HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/_rest_driver.log"
INT4=/workspace/gemma_build/int4_g128_lmhead
BASE=google/gemma-4-E4B-it
PORT=8000
WAIT_PID="${1:-672122}"   # the in-flight int4/y2024 window

say(){ echo "[rest] $(date -u +%H:%M:%SZ) $*" | tee -a "$LOG"; }
disk(){ df -BG --output=avail / | tail -1 | tr -d ' G'; }

say "driver up; waiting for in-flight window pid=$WAIT_PID to exit"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
say "in-flight window exited; free_disk=$(disk)G"

run_win(){
  local body="$1" model="$2" tag="$3" years="$4"
  sleep 8   # let the previous window's server fully release port $PORT
  say "START window body=$body tag=$tag years=$years free_disk=$(disk)G"
  bash "$HERE/run_window.sh" "$body" "$model" "$PORT" "$tag" "$years" "6144 12288" \
      >>"$HERE/_win_${body}_${tag}.log" 2>&1
  say "END   window body=$body tag=$tag rc=$? free_disk=$(disk)G"
}

run_win int4 "$INT4" y2025 "2025-I,2025-II"
run_win base "$BASE" y2024 "2024"
run_win base "$BASE" y2025 "2025-I,2025-II"

say "driver DONE; results:"
ls -1 "$HERE/results/" | tee -a "$LOG"
