#!/usr/bin/env bash
# Local-only batch-invariant spec-decode server launcher (AWS A10G). NOT an HF Job.
# Serves submissions/int4_mtp_batchinv (int4 W4A16 target + gemma4_assistant MTP
# drafter). The ONLY engine delta vs research/int4_mtp_drafter is the
# VLLM_BATCH_INVARIANT env var, which serve.py passes straight through to the
# vLLM worker (gpu_worker.init_batch_invariance() reads vllm.envs.VLLM_BATCH_INVARIANT
# and installs the aten batch-invariant overrides + TF32-off + num_splits=1 attn).
set -u
cd "$(dirname "$0")/../../submissions/int4_mtp_batchinv"
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Batch-invariance is the whole point of this submission. Default ON; the
# greedy ON/OFF control arm sets VLLM_BATCH_INVARIANT=0 explicitly.
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
# Local toolchain: the venv's flashinfer JIT-compiles sampling kernels at startup
# and fails on the local CUDA toolchain. Use the PyTorch-native sampler; at
# temperature=0 it is argmax either way, so greedy identity is unaffected. The
# official vllm/vllm-openai image ships prebuilt flashinfer, so this stays out of
# serve.py (local workaround only).
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-6}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# serve.py uses sys.executable for the api_server subprocess, so the interpreter
# here must be the pinned vLLM 0.22.0 venv.
exec /workspace/senpai/target/.venvs/vllm022/bin/python serve.py
