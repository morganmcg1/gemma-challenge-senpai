#!/usr/bin/env python
"""Serve the lmhead12k_int4head pruned + int4-head checkpoint on vLLM.

Identical to submissions/vllm_baseline/serve.py except:
  * ``--model`` points at the pruned checkpoint (``MODEL_ID``), which ships
    ``kept_ids.json`` alongside the weights, and
  * ``MODEL_ID`` is exported so the custom model class can locate ``kept_ids.json``.

The pruned ``lm_head`` here is int4 W4A16 (its quant group is in the checkpoint's
``quantization_config``); the custom model class passes the body quant_config to
the rebuilt head so it loads through the same compressed-tensors Marlin int4 GEMV.

The custom architecture (``Gemma4ForCausalLMLMHead12k``) is registered through the
``vllm_lmhead12k`` general-plugin entry point, NOT in-process here: vLLM's V1
async server builds the model in a separate ``EngineCore`` process, which an
in-process ``ModelRegistry.register_model()`` in this launcher would not reach.
The plugin's ``register()`` runs in every vLLM process via
``load_general_plugins()``. The package must be importable in the serving venv
(``pip install -e submissions/lmhead12k_int4head/vllm_plugin``). The project dir
is named ``vllm_plugin`` (not ``vllm_lmhead12k``) on purpose: this submission dir
goes on ``sys.path`` when ``serve.py`` launches, and a project dir sharing the
import name would shadow the installed package as an empty namespace package.
"""
from __future__ import annotations

import os
import sys


def main() -> None:
    model_id = os.environ.get("MODEL_ID", "/workspace/gemma_build/lmhead12k_int4head")
    served_model_name = os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it")
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "8000")
    max_model_len = os.environ.get("MAX_MODEL_LEN", "4096")
    gpu_memory_utilization = os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")
    max_num_batched_tokens = os.environ.get("MAX_NUM_BATCHED_TOKENS")
    # The custom model class reads kept_ids.json from MODEL_ID; make sure it is set
    # in the environment the EngineCore subprocess inherits.
    os.environ["MODEL_ID"] = model_id

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
