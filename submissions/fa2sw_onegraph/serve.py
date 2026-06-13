#!/usr/bin/env python
"""Serve the int4 QAT Gemma4 endpoint with the fa2sw + onegraph levers.

The fa2sw lever monkeypatches a vLLM config hook, so the engine must run in the
same process as this script (no V1 multiprocessing). We therefore drive the
OpenAI API server programmatically instead of exec'ing the CLI: set env, apply
the in-process levers, build the arg list, then run_server in-process.

Levers are env-gated (FA2SW, ONEGRAPH) so this one image can serve base /
+fa2sw / +onegraph / both.
"""
from __future__ import annotations

import os
import sys

# Must be set before importing vllm: in-process engine (so the fa2sw patch
# reaches the model runner) + native sampler (the flashinfer sampler JITs a
# kernel needing a curand.h absent from this CUDA toolkit; greedy is argmax so
# the sampler backend cannot change tokens).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import levers  # noqa: E402


def build_args() -> list[str]:
    model_id = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it-qat-w4a16-ct")
    served_model_name = os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it")
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "8000")
    max_model_len = os.environ.get("MAX_MODEL_LEN", "4096")
    gpu_memory_utilization = os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")
    max_num_batched_tokens = os.environ.get("MAX_NUM_BATCHED_TOKENS")
    quantization = os.environ.get("QUANTIZATION", "compressed-tensors")

    args = [
        "--model", model_id,
        "--served-model-name", served_model_name,
        "--host", host,
        "--port", port,
        "--dtype", "bfloat16",
        "--quantization", quantization,
        "--max-model-len", max_model_len,
        "--gpu-memory-utilization", gpu_memory_utilization,
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]
    if max_num_batched_tokens:
        args += ["--max-num-batched-tokens", max_num_batched_tokens]
    if levers.onegraph_enabled():
        args += ["--compilation-config", levers.onegraph_compilation_config()]
    return args


def main() -> None:
    backend_map_out = os.environ.get(
        "BACKEND_MAP_OUT",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend_map.json"),
    )
    active = levers.apply_levers(backend_map_out=backend_map_out)
    print(f"[fa2sw_onegraph] active levers: {active or ['none']}; "
          f"backend map -> {backend_map_out}", flush=True)

    import uvloop
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.entrypoints.utils import cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    cli_env_setup()
    parser = FlexibleArgumentParser(description="fa2sw_onegraph Gemma endpoint")
    parser = make_arg_parser(parser)
    args = parser.parse_args(build_args())
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
