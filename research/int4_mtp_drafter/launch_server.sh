#!/usr/bin/env bash
# Local-only spec-decode server launcher (AWS A10G). NOT an HF Job.
set -u
cd "$(dirname "$0")/../../submissions/int4_mtp_drafter"
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN="${HF_TOKEN:-}"
# Use local HF cache only: the in-process hub re-check stalled startup before.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Local-only: this venv's pip flashinfer has no prebuilt sampling kernels, so it
# JIT-compiles at startup and fails on the local CUDA toolchain (missing
# curand.h; cu13 cooperative_groups vs flashinfer's bundled libcudacxx). Use the
# PyTorch-native sampler instead. At temperature=0 the sampler is argmax either
# way, so greedy identity and spec-decode acceptance are unaffected. The official
# vllm/vllm-openai image ships prebuilt flashinfer kernels, so this is a local
# toolchain workaround only and stays out of the submission serve.py.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-6}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Run serve.py with the pinned vLLM 0.22.0 venv (serve.py uses sys.executable
# for the api_server subprocess, so the interpreter here must be the venv).
exec /workspace/senpai/target/.venvs/vllm022/bin/python serve.py
