#!/usr/bin/env bash
# PR #643 resumable seed launcher. Launches ONE run_gate.py invocation detached, keeping it
# under the 90-min per-run bound (<=2 cells: one seed both modes ~76min, OR two sampled-only
# seeds ~76min). The driver is idempotent: it SKIPs result JSONs already on disk and resumes
# the same W&B run (cr3c4y3q), so passing already-done seeds is a fast no-op.
#   usage: _launch_seed.sh "<seeds-csv>" ["<modes-csv>"]
#   e.g.   _launch_seed.sh 23456                 # seed 23456, both greedy+sampled
#          _launch_seed.sh 45678,56789 sampled   # two sampled-only seeds in one invocation
set -euo pipefail
cd /workspace/senpai/target
HERE=research/validity/gpqa_gate_readingA_4096
SEEDS="${1:?need seeds csv}"
MODES="${2:-}"

# Refuse to launch if a gate driver is already running (avoid conc=1 server contention).
if [[ -f "$HERE/_gate.pid" ]] && kill -0 "$(cat "$HERE/_gate.pid")" 2>/dev/null; then
  echo "REFUSE: gate driver $(cat "$HERE/_gate.pid") still running"; exit 2
fi
# Belt-and-suspenders: refuse if ANY gate driver OR eval client is alive, even one that
# bypassed _gate.pid by launching run_eval.py directly. This is the exact failure that caused
# the seed-12345 double-launch: a stray run_eval.py raced run_gate.py at conc=1, putting two
# clients on the server and contaminating the seed. pgrep self-excludes, so it won't match here.
STRAY=$(pgrep -af 'gpqa_gate_readingA_4096/run_gate\.py|downstream_quality_eval/run_eval\.py' || true)
if [[ -n "$STRAY" ]]; then
  echo "REFUSE: a gate driver/eval client is already running (conc=1 contract):"; echo "$STRAY"; exit 2
fi
# Server must be live (Option-B: int4_g128_lmhead + K=7 drafter + BI=1 + dev307 @ 8192).
if ! curl -s --max-time 5 http://127.0.0.1:8000/v1/models | grep -q gemma-4-e4b-it; then
  echo "REFUSE: no live gemma-4-e4b-it server at :8000"; exit 3
fi

ARGS=(--seeds "$SEEDS")
[[ -n "$MODES" ]] && ARGS+=(--modes "$MODES")
L="$HERE/_gate_run_${SEEDS//,/_}${MODES:+_$MODES}.log"
nohup .venv/bin/python "$HERE/run_gate.py" "${ARGS[@]}" > "$L" 2>&1 &
P=$!; disown "$P" 2>/dev/null || true
echo "$P" > "$HERE/_gate.pid"
echo "launched pid=$P seeds=$SEEDS modes=${MODES:-both} -> $L"
