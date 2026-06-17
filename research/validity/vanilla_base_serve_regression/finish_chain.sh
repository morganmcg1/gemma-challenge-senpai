#!/usr/bin/env bash
# PR #557 finish chain: after the ple_fold arm exits, run the remaining single-GPU
# work sequentially — global_fa load-failure probe, selfdet on the recovered ple_fold
# serve, then aggregate (resumes W&B run yw6vwk1w). LOCAL, NO FIRE.
set -u
cd /workspace/senpai/target/research/validity/vanilla_base_serve_regression
PLE_PID="${1:-1145512}"

echo "[chain] start $(date -u +%H:%M:%S) waiting on ple_fold driver pid=$PLE_PID"
i=0
while kill -0 "$PLE_PID" 2>/dev/null; do
  sleep 10; i=$((i+1))
  if [ "$i" -gt 180 ]; then echo "[chain] TIMEOUT waiting ple_fold (30m)"; break; fi
done
echo "[chain] ple_fold driver gone $(date -u +%H:%M:%S)"
ls -la ple_fold_gpqa_diamond.json 2>&1

echo "[chain] === global_fa load-failure probe ==="
python3 global_fa_loadtest.py > global_fa.out 2>&1; echo "[chain] global_fa rc=$?"

echo "[chain] === selfdet (ple_fold serve) ==="
python3 selfdet_probe.py > selfdet.out 2>&1; echo "[chain] selfdet rc=$?"

echo "[chain] === aggregate (wandb resume yw6vwk1w) ==="
python3 aggregate_557.py > aggregate.out 2>&1; echo "[chain] aggregate rc=$?"

echo "[chain] DONE $(date -u +%H:%M:%S)"
