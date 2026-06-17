#!/usr/bin/env python
"""Plain vLLM server for the stock UNQUANTIZED bf16 base, the AIME denominator base.

PR #580: the ">=90% of unquantized-base AIME" gate is only as trustworthy as its
denominator. This serves the canonical unquantized ``google/gemma-4-E4B-it``
(bf16, full native 262k head, full multimodal tower) with NO surgical patches, NO
speculation, NO lm_head prune, NO quantization -- only the pinned challenge vLLM
wheel (0.22.1rc1). It is the apples-to-apples bf16 counterpart of
``int4_base_aime``: identical serve path and AIME harness, so the only variable
is bf16-vs-int4, which is exactly the int4-quant tax we want to measure.
LOCAL eval base only; not a leaderboard submission, never launched.
"""
from __future__ import annotations

import os
import sys


def main() -> None:
    model_id = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it")
    served_model_name = os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it")
    host = os.environ.get("HOST", "127.0.0.1")
    port = os.environ.get("PORT", "8000")
    max_model_len = os.environ.get("MAX_MODEL_LEN", "4096")
    gpu_memory_utilization = os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")
    max_num_batched_tokens = os.environ.get("MAX_NUM_BATCHED_TOKENS", "2048")
    max_num_seqs = os.environ.get("MAX_NUM_SEQS", "32")
    seed = os.environ.get("VLLM_SEED", "0")

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
        # Unquantized checkpoint: its native config dtype is bfloat16 and there is
        # no quantization_config, so we pin bfloat16 explicitly. This is the exact
        # full-precision baseline the int4 QAT checkpoint is derived from.
        "--dtype",
        "bfloat16",
        "--max-model-len",
        max_model_len,
        "--gpu-memory-utilization",
        gpu_memory_utilization,
        "--max-num-batched-tokens",
        max_num_batched_tokens,
        "--max-num-seqs",
        max_num_seqs,
        "--seed",
        seed,
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]
    os.execvpe(args[0], args, os.environ)


if __name__ == "__main__":
    main()
