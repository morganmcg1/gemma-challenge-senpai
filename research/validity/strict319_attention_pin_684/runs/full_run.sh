#!/usr/bin/env bash
# PR #684 full 3-config measurement. Each config in a FRESH process (BI / monkeypatch are
# process-global, snapshotted at import). Writes runs/<config>.json for decide_and_log.py.
set -u
cd "$(dirname "$0")/.."
PY=/tmp/senpai-venvs/20f658587e8a6643/bin/python
export CUDA_VISIBLE_DEVICES=0

COMMON="--verify-width 6 --n-prompts 24 --n-new 32 --ctx-cap 512 --det-prompts 10 \
  --tps-warmup 24 --tps-long 80 --tps-short 16 --tps-reps 4 --tps-ctx-prompts 3 \
  --attn-tokens 64 --attn-warmup 16 --attn-reps 12"

for C in baseline bi1 fixed2d; do
  echo "================= CONFIG $C ================="
  ts=$(date -u +%H:%M:%S)
  echo "[full_run] $C start $ts"
  $PY attn_pin_cost.py --config "$C" $COMMON --out "runs/${C}.json"
  rc=$?
  echo "[full_run] $C exit=$rc $(date -u +%H:%M:%S)"
  if [ $rc -ne 0 ]; then echo "[full_run] ABORT: $C failed"; exit $rc; fi
done
echo "ALL_CONFIGS_DONE"
