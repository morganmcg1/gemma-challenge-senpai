#!/usr/bin/env bash
# PR #679 group-size sweep: run g128-control + g64 fresh 4-session bands, and
# extend the existing g32 band to a 4th session, all in THIS harness (run_band.sh
# + serve_int4.sh + aime_eval.py) so g128/g64/g32 are a clean controlled sweep.
# Single GPU -> strictly sequential. Backgrounded; progress in _chain_bands.out.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd /workspace/senpai/target
OUT="$HERE/_chain_bands.out"
log(){ echo "[chain] $* $(date -u +%H:%M:%S)" | tee -a "$OUT"; }

# safety: clear any stale server on the GPU before we start
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null && sleep 8 || true

log "=== ARM 1/3: g128 control (4 sessions, fresh) ==="
bash "$HERE/run_band.sh" g128 /workspace/gemma_build/int4_g128_lmhead 4 0 0

log "=== ARM 2/3: g64 (4 sessions, fresh) ==="
bash "$HERE/run_band.sh" g64 /workspace/gemma_build/int4_g64body_lmhead 4 0 0

log "=== ARM 3/3: g32 extend (session 3 only; 0-2 already done) ==="
bash "$HERE/run_band.sh" g32 /workspace/gemma_build/int4_g32body_lmhead 4 0 3

log "=== aggregate ==="
/tmp/vllm0220-srv/bin/python "$HERE/aggregate.py" >> "$OUT" 2>&1 || log "aggregate failed"
log "=== CHAIN DONE ==="
touch "$HERE/_chain_bands.DONE"
