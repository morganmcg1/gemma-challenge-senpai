#!/bin/bash
set -e
D=research/speed/fp8_kv_decode_speed
COMMON="--tps-lengths 512,2048,8192 --tps-prompts 12,4,2 --greedy-prompts 32 --ppl-records 128 --max-model-len 9216"
for ARM in auto fp8 fp8_e5m2; do
  echo "=== ARM $ARM $(date -u +%H:%M:%S) ==="
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python $D/run_kv_arm.py --kv-dtype $ARM --out $D/arm_$ARM.json $COMMON
done
echo "=== ALL ARMS DONE $(date -u +%H:%M:%S) ==="
