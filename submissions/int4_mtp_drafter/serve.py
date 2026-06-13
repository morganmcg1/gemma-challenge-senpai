#!/usr/bin/env python
"""OpenAI-compatible vLLM server: int4 W4A16 Gemma-4-E4B target + MTP drafter.

Target: google/gemma-4-E4B-it-qat-w4a16-ct (official QAT W4A16 compressed-tensors,
loaded via Marlin). Drafter: the QAT-matched gemma4_assistant
(google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant), a lightweight Q-only
KV-shared decoder that vLLM resolves to Gemma4MTPModel and serves as a
speculative proposer. At temperature=0 vLLM's rejection sampler short-circuits to
target-argmax, so decode stays token-identical to plain greedy AR of the int4
target while amortizing the int4 weight read over the accepted draft tokens.

Set NUM_SPECULATIVE_TOKENS=0 (or empty) to disable speculation and serve the
plain int4 target. That mode is the exact-greedy reference for the
greedy-identity gate: reference and candidate then differ only in the drafter.

All modalities (text/image/audio) stay enabled: no --limit-mm-per-prompt, no
text-only shortcut. The draft head is left in its native bf16/centroid path
(never force-quantized): the assistant's masked-embedding centroid logits have no
packed-weight branch, so quantizing it would force the ~11x-slower dense path.
"""
from __future__ import annotations

import json
import os
import sys


def main() -> None:
    model_id = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it-qat-w4a16-ct")
    served_model_name = os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it")
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "8000")
    max_model_len = os.environ.get("MAX_MODEL_LEN", "4096")
    gpu_memory_utilization = os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")
    max_num_batched_tokens = os.environ.get("MAX_NUM_BATCHED_TOKENS", "512")
    max_num_seqs = os.environ.get("MAX_NUM_SEQS", "1")

    drafter_model = os.environ.get(
        "DRAFTER_MODEL", "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant"
    )
    num_spec = int(os.environ.get("NUM_SPECULATIVE_TOKENS", "6") or "0")
    spec_method = os.environ.get("SPECULATIVE_METHOD")  # optional override

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

    # Speculative decoding with the gemma4_assistant MTP drafter. vLLM's
    # speculative config rewrites model_type gemma4_assistant -> gemma4_mtp
    # (Gemma4MTPModel) and sets num_kv_shared_layers=0 so the draft attention
    # layers form their own group and read the target KV cache via the proposer
    # after construction. Leave the drafter unquantized (no "quantization" key).
    if num_spec > 0 and drafter_model:
        spec_config: dict[str, object] = {
            "model": drafter_model,
            "num_speculative_tokens": num_spec,
        }
        if spec_method:
            spec_config["method"] = spec_method
        args += ["--speculative-config", json.dumps(spec_config)]

    if os.environ.get("ENFORCE_EAGER", "0") == "1":
        args += ["--enforce-eager"]

    # Ship the attention-group num_heads backport (see sitecustomize.py /
    # vllm_attn_group_patch.py) into every server process. We pin vllm==0.22.0
    # to match the official vllm/vllm-openai image, which predates the upstream
    # fix for the {8,4} draft/target attention-group assertion; putting this
    # directory on PYTHONPATH makes Python auto-import our sitecustomize.py at
    # startup in the api_server, EngineCore, and worker processes (works under
    # both fork and spawn). The patch is a no-op when speculation is disabled.
    here = os.path.dirname(os.path.abspath(__file__))
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        here + os.pathsep + existing_pythonpath if existing_pythonpath else here
    )

    os.execvpe(args[0], args, os.environ)


if __name__ == "__main__":
    main()
