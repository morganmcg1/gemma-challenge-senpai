#!/usr/bin/env python
"""Serve the lmhead12k empirical pruned checkpoint on vLLM.

Mirrors submissions/vllm_baseline/serve.py, with two additions:
  1. Register the custom Gemma3ForCausalLMLMHead12k class (pruned lm_head +
     scatter-to-full-vocab logits) BEFORE vLLM resolves the architecture.
  2. Point --model at the pruned checkpoint (MODEL_ID), which ships kept_ids.json.

Registration uses the string form so CUDA is not initialized in this launcher
process. The server is started in-process via runpy so the registration is live
when vLLM builds the model.

!!! NEEDS GPU VALIDATION: for tensor-parallel/multi-process workers, vLLM may
require the model to be registered through a vllm general-plugins entrypoint so
worker processes see it. On a single A10G (TP=1) the in-process registration
below is the first thing to try; if workers don't pick it up, switch to a plugin
entrypoint. Validate against vLLM 0.22.0 on a live GPU before any benchmark.
"""
from __future__ import annotations

import os
import runpy
import sys


def main() -> None:
    model_id = os.environ.get("MODEL_ID", "/workspace/gemma_build/lmhead12k_empirical")
    served_model_name = os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it")
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "8000")
    max_model_len = os.environ.get("MAX_MODEL_LEN", "4096")
    gpu_memory_utilization = os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")
    max_num_batched_tokens = os.environ.get("MAX_NUM_BATCHED_TOKENS")
    os.environ.setdefault("MODEL_ID", model_id)  # model __init__ reads kept_ids here

    from vllm import ModelRegistry

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ModelRegistry.register_model(
        "Gemma3ForCausalLM", "lmhead12k_model:Gemma3ForCausalLMLMHead12k"
    )

    argv = [
        "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--served-model-name", served_model_name,
        "--host", host,
        "--port", port,
        "--max-model-len", max_model_len,
        "--gpu-memory-utilization", gpu_memory_utilization,
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]
    if max_num_batched_tokens:
        argv += ["--max-num-batched-tokens", max_num_batched_tokens]
    sys.argv = argv
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
