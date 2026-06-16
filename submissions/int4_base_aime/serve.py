#!/usr/bin/env python
"""Plain vLLM server for the stock int4 QAT base, used as the AIME A/B reference.

This is the PR #514 "(a) stock int4 base gemma-4-E4B-it" row: the canonical
public QAT w4a16 checkpoint served with NO surgical patches, NO speculation, NO
lm_head prune — only the pinned challenge vLLM wheel (0.22.1rc1), so the only
variables between this and the surgical-357 ship are the ship's optimizations.
LOCAL eval base only; not a leaderboard submission, never launched.
"""
from __future__ import annotations

import os
import sys


def main() -> None:
    model_id = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it-qat-w4a16-ct")
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
        # 'auto' reads the checkpoint's own quant + torch_dtype (w4a16 compressed
        # tensors); do not force a dtype that could fight the quant config.
        "--dtype",
        "auto",
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
