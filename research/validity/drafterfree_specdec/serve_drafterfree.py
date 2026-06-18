#!/usr/bin/env python
"""Drafter-FREE speculative serve for PR #673 (analysis_only screen).

Serves the SHIPPED int4_g128_lmhead body UNCHANGED -- identical engine flags to
``submissions/int4_g128_lmhead/serve.py`` (dtype bf16, max-model-len,
gpu-mem-util, trust-remote-code, no-enable-log-requests) -- plus an optional
DRAFTER-FREE speculative config injected verbatim through ``SPEC_CONFIG_JSON``:

  * ngram / prompt-lookup : {"method":"ngram","num_speculative_tokens":N,
                             "prompt_lookup_min":2,"prompt_lookup_max":6}
  * Arctic suffix decoding: {"method":"suffix","num_speculative_tokens":N}

Both proposers run NO model forward (pure string match on the running context), so
unlike the MTP submission there is NO neural draft head, NO {8,4} attention-group
mismatch, and the int4_mtp_batchinv sitecustomize attn-group backport is NOT
needed. vLLM verifies every proposed token against the base argmax => decode stays
greedy byte-identical to plain int4 AR (the #319 launch contract).

This file lives under research/ and points MODEL_ID at the build output directly;
it does NOT modify the shipped submission. Set ``MAX_NUM_SEQS`` (default 16) and
``VLLM_BATCH_INVARIANT=1`` via the harness env. ``SPEC_CONFIG_JSON`` empty/unset =>
plain M=1 AR (the matched greedy-reference anchor).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def resolve_model_path() -> str:
    raw = os.environ.get("MODEL_ID", "/workspace/gemma_build/int4_g128_lmhead")
    p = Path(raw)
    return str(p if p.is_absolute() else (Path(__file__).resolve().parent / p))


def main() -> None:
    model_id = resolve_model_path()
    served_model_name = os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it")
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "8000")
    max_model_len = os.environ.get("MAX_MODEL_LEN", "4096")
    gpu_memory_utilization = os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")
    max_num_seqs = os.environ.get("MAX_NUM_SEQS", "16")
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
        "--max-num-seqs",
        max_num_seqs,
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]
    if max_num_batched_tokens:
        args += ["--max-num-batched-tokens", max_num_batched_tokens]

    # Drafter-free speculative config, passed through verbatim. Empty/unset => AR.
    spec_json = os.environ.get("SPEC_CONFIG_JSON", "").strip()
    if spec_json:
        json.loads(spec_json)  # fail fast on malformed JSON
        args += ["--speculative-config", spec_json]
        print(f"[serve] drafter-free speculative-config={spec_json}", flush=True)
    else:
        print("[serve] no speculative-config (plain M=1 AR anchor)", flush=True)

    if os.environ.get("ENFORCE_EAGER", "0") == "1":
        args += ["--enforce-eager"]

    print(f"[serve] MODEL_ID={model_id} max_num_seqs={max_num_seqs} "
          f"VLLM_BATCH_INVARIANT={os.environ.get('VLLM_BATCH_INVARIANT')}", flush=True)
    os.execvpe(args[0], args, os.environ)


if __name__ == "__main__":
    main()
