#!/usr/bin/env bash
# PR #479 tail-KV tax L-sweep driver. Three FRESH processes (= 3 sessions, fresh CUDA
# context + clock warmup each) so the deepest tail anchor L=2048 carries its own between-
# session sigma. Sequential (single GPU) to avoid contention corrupting the timing. The
# headline stays --L 640 so the iso_delta_reproduces_466 calibration guard (anchored at
# L=640) keeps passing; the extended grid only ADDS tail points to per_L.
set -euo pipefail
cd /workspace/senpai/target
PY=/tmp/senpai-venvs/5f4c623f772358a2/bin/python
OUT=research/speed/tail_kv_tax_lsweep
export CUDA_VISIBLE_DEVICES=0

echo "=== RUN A: full 6-point sweep (128,384,640,896,1280,2048) @ $(date -u +%H:%M:%S)Z ==="
$PY research/speed/strict_wholecycle_ab/strict_wholecycle_ab.py \
  --Ls 128,384,640,896,1280,2048 --L 640 --no-wandb \
  --output "$OUT/strict_wholecycle_ab_tail.json" \
  --selftest-output "$OUT/selftest_run_a.json" 2>&1 | tail -12

echo "=== RUN B: L=2048 session repeat 2 (640,2048) @ $(date -u +%H:%M:%S)Z ==="
$PY research/speed/strict_wholecycle_ab/strict_wholecycle_ab.py \
  --Ls 640,2048 --L 640 --no-wandb \
  --output "$OUT/tail_run_b.json" \
  --selftest-output "$OUT/selftest_run_b.json" 2>&1 | tail -6

echo "=== RUN C: L=2048 session repeat 3 (640,2048) @ $(date -u +%H:%M:%S)Z ==="
$PY research/speed/strict_wholecycle_ab/strict_wholecycle_ab.py \
  --Ls 640,2048 --L 640 --no-wandb \
  --output "$OUT/tail_run_c.json" \
  --selftest-output "$OUT/selftest_run_c.json" 2>&1 | tail -6

echo "=== ALL RUNS DONE @ $(date -u +%H:%M:%S)Z ==="
