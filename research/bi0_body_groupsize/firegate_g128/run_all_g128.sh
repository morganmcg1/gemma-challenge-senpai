#!/usr/bin/env bash
# PR #814 Step-3 g128 panel orchestrator: wait for the g128 server (:8021) to be
# ready, then run the multi-stream panel (GPQA+MMLU+GSM8K) and finally the AIME
# panel (greedy maj@1 single-stream + sampled maj@8). One background process to
# manage; ALL-DONE sentinel in _all.status. LOCAL ONLY -- no HF Job.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EVAL_PY=/tmp/eval-serve-venv/bin/python
ALLSTATUS="$HERE/_all.status"
BASE_URL="http://127.0.0.1:8021/v1/models"

echo "ALL START $(date -u +%FT%TZ)" | tee "$ALLSTATUS"

# ---- wait for server ready (cap ~12 min: 144 * 5s) ----
ready=0
for _ in $(seq 1 144); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then ready=1; break; fi
  sleep 5
done
if [ "$ready" -ne 1 ]; then
  echo "ALL ABORT: server not ready within cap (last code=$code) $(date -u +%FT%TZ)" | tee -a "$ALLSTATUS"
  exit 1
fi
echo "ALL: server ready $(date -u +%FT%TZ)" | tee -a "$ALLSTATUS"

# ---- multi-stream panel: GPQA + MMLU + GSM8K ----
bash "$HERE/run_panel_g128.sh"
echo "ALL: panel finished $(date -u +%FT%TZ)" | tee -a "$ALLSTATUS"

# ---- single-stream-sensitive AIME (greedy maj@1 + sampled maj@8) ----
bash "$HERE/run_aime_g128.sh"
echo "ALL DONE $(date -u +%FT%TZ)" | tee -a "$ALLSTATUS"
