#!/bin/bash
set -x
cd /workspace/senpai/target
export CUDA_VISIBLE_DEVICES=0
PY=.venvs/vllm022/bin/python
echo "=== BI=1 run ==="
VLLM_BATCH_INVARIANT=1 $PY research/strict_clean_verify_attn_kernel/verify_attn_locus.py \
  --n-prompts 4 --max-tokens 48 --stride 3 \
  --out research/strict_clean_verify_attn_kernel/verify_attn_report.json \
  > research/strict_clean_verify_attn_kernel/logs/full_bi1.log 2>&1
echo "BI1_EXIT=$?"
echo "=== BI=0 run ==="
VLLM_BATCH_INVARIANT=0 $PY research/strict_clean_verify_attn_kernel/verify_attn_locus.py \
  --n-prompts 4 --max-tokens 48 --stride 3 \
  --out research/strict_clean_verify_attn_kernel/verify_attn_report_bi0.json \
  > research/strict_clean_verify_attn_kernel/logs/full_bi0.log 2>&1
echo "BI0_EXIT=$?"
echo "=== ALL DONE ==="
