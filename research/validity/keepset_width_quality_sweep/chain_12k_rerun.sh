#!/usr/bin/env bash
# Wait for the in-flight speed sweep (full width) to exit, then rerun the 12k
# speed leg (failed earlier on a cold-start decode-client crash, 124/128).
# speed_sweep.py reads+updates results/_speed.json in place, so 16k/32k/full are
# preserved and only the 12k key is rewritten.
set -uo pipefail
HERE="/workspace/senpai/target/research/validity/keepset_width_quality_sweep"
SWEEP_PID="${1:?need sweep pid}"
PY=/tmp/eval-serve-venv/bin/python
echo "[chain $(date -u +%H:%M:%SZ)] waiting for sweep PID $SWEEP_PID (full width) to finish ..."
while kill -0 "$SWEEP_PID" 2>/dev/null; do sleep 15; done
echo "[chain $(date -u +%H:%M:%SZ)] sweep done; current _speed.json:"
$PY -c "import json;d=json.load(open('$HERE/results/_speed.json'));print({k:(v.get('warm_median_tps') if isinstance(v,dict) else v) for k,v in d.items()})" || true
sleep 5
echo "[chain $(date -u +%H:%M:%SZ)] launching 12k speed rerun ..."
cd "$HERE"
$PY speed_sweep.py --widths 12k > results/_speed_12k_rerun.out 2>&1
rc=$?
echo "[chain $(date -u +%H:%M:%SZ)] 12k rerun exit=$rc"
$PY -c "import json;d=json.load(open('$HERE/results/_speed.json'));print('12k ->', d.get('12k'))" || true
echo "[chain $(date -u +%H:%M:%SZ)] CHAIN COMPLETE"
