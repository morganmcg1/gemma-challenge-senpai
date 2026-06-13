#!/usr/bin/env bash
# Poll a vLLM endpoint until /v1/models responds or timeout. Also fails fast if
# the server log shows a fatal error. Usage: wait_ready.sh <log-file> [timeout-s]
set -uo pipefail
LOG="${1:?log-file}"
TIMEOUT="${2:-900}"
BASE="http://127.0.0.1:8000"
start=$(date +%s)
while true; do
  if curl -fsS "$BASE/v1/models" >/dev/null 2>&1; then
    echo "READY after $(( $(date +%s) - start ))s"
    exit 0
  fi
  if grep -qE "AssertionError|Traceback \(most recent|RuntimeError|CUDA out of memory|Error:|raise |ValueError" "$LOG" 2>/dev/null; then
    echo "FATAL in $LOG:"; tail -25 "$LOG"; exit 1
  fi
  if [ $(( $(date +%s) - start )) -ge "$TIMEOUT" ]; then
    echo "TIMEOUT after ${TIMEOUT}s"; tail -25 "$LOG"; exit 2
  fi
  sleep 3
done
