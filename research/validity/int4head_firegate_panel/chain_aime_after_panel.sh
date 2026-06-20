#!/usr/bin/env bash
# Wait for the GPQA+MMLU panel to finish (PANEL DONE sentinel in _panel.status),
# then run the AIME panel. AIME's PRIMARY arm is greedy maj@1 single-stream
# (client-concurrency=1) which must own the server -- it cannot overlap the
# panel's 16-way load without breaking the effective batch=1 protocol. Gate the
# launch on the sentinel (not a PID) so PID reuse can't fool us and the waiter
# never matches itself.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STATUS="$HERE/_panel.status"
echo "CHAIN: waiting for PANEL DONE in $STATUS $(date -u +%FT%TZ)"
# Cap the wait at ~100 min (300 * 20s) so this never hangs forever.
done_ok=0
for _ in $(seq 1 300); do
  if grep -q "PANEL DONE" "$STATUS" 2>/dev/null; then done_ok=1; break; fi
  sleep 20
done
if [ "$done_ok" -ne 1 ]; then
  echo "CHAIN: PANEL DONE not seen within wait cap; NOT starting AIME (avoid concurrent load). $(date -u +%FT%TZ)"
  exit 1
fi
echo "CHAIN: panel done, starting AIME $(date -u +%FT%TZ)"
bash "$HERE/run_aime.sh"
echo "CHAIN: AIME finished $(date -u +%FT%TZ)"
