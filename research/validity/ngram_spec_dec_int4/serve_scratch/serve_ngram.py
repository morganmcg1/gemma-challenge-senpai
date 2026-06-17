#!/usr/bin/env python
"""PR #609 scratch serve — byte-identical to submissions/int4_g128_lmhead/serve.py
EXCEPT it appends ``--speculative-config <JSON>`` when the SPECULATIVE_CONFIG env
var is set & non-empty.

Rationale: the shipped int4_g128_lmhead serve.py does NOT read SPECULATIVE_CONFIG,
and PR #609 is analysis-only (no served-file change). This local wrapper enables
vLLM's draft-free ngram/prompt-lookup spec-decode on the SAME flags + SAME
checkpoint. With SPECULATIVE_CONFIG unset/empty the launched argv is identical to
the shipped serve, so the AR floor here is provably the shipped serve path.
LOCAL only — does not launch any HF Job, not a submission.
"""
from __future__ import annotations

import json
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
    # PR #609 ONLY delta: opt-in ngram/prompt-lookup speculative decoding. Empty/
    # unset -> byte-identical to the shipped serve (the AR floor & identity ref).
    spec = os.environ.get("SPECULATIVE_CONFIG", "").strip()
    if spec:
        json.loads(spec)  # fail fast on malformed JSON before exec
        args += ["--speculative-config", spec]
    os.execvpe(args[0], args, os.environ)


if __name__ == "__main__":
    main()
