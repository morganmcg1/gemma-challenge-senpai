#!/usr/bin/env python
"""Serve the bundled int4 g128 + untied int4 lm_head Gemma-4-E4B checkpoint.

Identical serving path to the vLLM baseline -- same flags, all modalities on --
the only difference is MODEL_ID points at the bundled compressed-tensors
checkpoint (model/), whose config.json carries the W4A16 pack-quantized
quantization_config. vLLM 0.22.0 auto-detects compressed-tensors from that
config and repacks the int4 weights to Marlin at load; no extra flags needed.

MODEL_ID is resolved as a path: absolute is used as-is; relative is taken
against this file's directory (so "model" -> <submission>/model on the harness,
while local runs override MODEL_ID with the build output path).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def resolve_model_path() -> str:
    here = Path(__file__).resolve().parent
    raw = os.environ.get("MODEL_ID", "model")
    p = Path(raw)
    return str(p if p.is_absolute() else (here / p))


def main() -> None:
    model_id = resolve_model_path()
    served_model_name = os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it")
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "8000")
    max_model_len = os.environ.get("MAX_MODEL_LEN", "4096")
    gpu_memory_utilization = os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")
    # Bound the prefill chunk so the prompt-logprobs (PPL stage) log_softmax peak
    # is set by chunk size, not full prompt length. Chunked prefill is on by
    # default in vLLM V1, so this alone bounds the per-step allocation.
    max_num_batched_tokens = os.environ.get("MAX_NUM_BATCHED_TOKENS")

    args = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_id,
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        port,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        max_model_len,
        "--gpu-memory-utilization",
        gpu_memory_utilization,
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]
    if max_num_batched_tokens:
        args += ["--max-num-batched-tokens", max_num_batched_tokens]
    os.execvpe(args[0], args, os.environ)


if __name__ == "__main__":
    main()
