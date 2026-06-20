#!/bin/bash
set -euo pipefail
cd /workspace/senpai/target
export CUDA_VISIBLE_DEVICES=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_BATCH_INVARIANT=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export MODEL_ID=/workspace/gemma_build/int4_g32_lmhead
export DRAFTER_MODEL=google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant
export NUM_SPECULATIVE_TOKENS=6
export PYTHONPATH="/workspace/senpai/target/submissions/int4_mtp_bi0_int4head:/workspace/senpai/target"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VENV=/tmp/senpai-venvs/20f658587e8a6643/bin/python
exec "$VENV" research/bf16_gemm_attribution/attribute_modules.py
