#!/usr/bin/env python
"""PR #713 serve: bf16 qat-unquantized master + IN-MEMORY int4 fake-quant.

LocalServer launches this as `server_python serve.py` (cwd = the out-of-tree serve
dir). It loads the read-only bf16 master through vLLM, then — AFTER the weights are
on GPU — fake-quantizes the selected text-decoder Linear weights in place. NO
checkpoint build, NO disk write.

TOPOLOGY (why the patch lives in sitecustomize.py, not here):
The V1 OpenAI *async* server ALWAYS runs the engine core in a child process
(AsyncMPClient). CUDA is initialized in THIS (api_server parent) process before the
child launches, so the child cannot fork ("Cannot re-initialize CUDA in forked
subprocess") — it must SPAWN. A spawned child is a fresh interpreter, so a runtime
monkeypatch on GPUModelRunner set here would be LOST. Instead, this parent:

  * puts the serve dir on PYTHONPATH so the spawned child auto-imports our
    sitecustomize.py at interpreter startup,
  * sets FQ_APPLY=1 so that sitecustomize registers a MetaPathFinder which wraps
    GPUModelRunner.load_model AFTER vLLM imports it IN THE CHILD,
  * forces VLLM_WORKER_MULTIPROC_METHOD=spawn for a deterministic CUDA-safe child.

--enforce-eager removes CUDA-graph capture so the in-place weight mutation can never
be shadowed by a captured pre-fakequant pointer. The smoke test verifies the patch
fired in the child (the "[fq-serve] APPLIED" line, printed from the engine-core pid)
and that N=0 (all-g128) AIME reproduces the int4 body.

The fake-quant spec is env-driven (one server == one cell):
  FQ_G32_LAYERS   decoder layers on the finer g32 grid (e.g. "14-27", "" = N=0 all-g128)
  FQ_G32_GROUP    finer group size (default 32)
  FQ_G128_GROUP   operative body group size (default 128)
  FQ_QUANT_HEAD   1 => also fake-quant lm_head/embed at g128 (absolute anchor); 0 => skip
"""
from __future__ import annotations

import os
import runpy
import sys


def _b(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _truthy(v: str) -> bool:
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    # 1) Spawned engine-core child must auto-import our sitecustomize.py: put the
    #    serve dir on PYTHONPATH (inherited by the child's interpreter startup).
    existing = os.environ.get("PYTHONPATH", "")
    parts = [here] + ([existing] if existing else [])
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    # 2) Arm the child's MetaPathFinder (sitecustomize is a no-op without this).
    os.environ["FQ_APPLY"] = "1"
    # 3) Deterministic CUDA-safe child: CUDA is initialized in this parent before the
    #    engine-core launch, so the child MUST spawn (fork would re-init CUDA).
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    print(
        f"[fq-serve] parent pid={os.getpid()} "
        f"FQ_G32_LAYERS={_b('FQ_G32_LAYERS', '')!r} "
        f"FQ_G32_GROUP={_b('FQ_G32_GROUP', '32')} FQ_G128_GROUP={_b('FQ_G128_GROUP', '128')} "
        f"FQ_QUANT_HEAD={_b('FQ_QUANT_HEAD', '1')} "
        f"PYTHONPATH[0]={here} method=spawn",
        flush=True,
    )

    model_id = _b("MODEL_ID", "")
    if not model_id:
        raise SystemExit("[fq-serve] MODEL_ID (bf16 master path) is required")
    argv = [
        "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--served-model-name", _b("SERVED_MODEL_NAME", "gemma-4-e4b-it"),
        "--host", _b("HOST", "127.0.0.1"),
        "--port", str(_b("PORT", "8000")),
        "--dtype", _b("SERVE_DTYPE", "bfloat16"),
        "--max-model-len", str(_b("MAX_MODEL_LEN", "8192")),
        "--gpu-memory-utilization", str(_b("GPU_MEMORY_UTILIZATION", "0.90")),
        "--max-num-batched-tokens", str(_b("MAX_NUM_BATCHED_TOKENS", "2048")),
        "--max-num-seqs", str(_b("MAX_NUM_SEQS", "1")),
        "--seed", str(_b("VLLM_SEED", "0")),
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]
    # --enforce-eager is the proven default. fakequant runs at the END of
    # load_model (BEFORE any forward, so BEFORE CUDA-graph capture) and mutates
    # weights IN PLACE (storage pointer unchanged), so a captured graph reads the
    # already-fakequanted storage at replay — cudagraph is therefore output-neutral
    # in principle. FQ_ENFORCE_EAGER=0 enables CUDA graphs for a ~2x decode speedup
    # on the slow bf16 substrate; ONLY adopt it after proving byte-identical greedy
    # token ids vs the enforce-eager reference (validity > speed).
    if _truthy(_b("FQ_ENFORCE_EAGER", "1")):
        argv.append("--enforce-eager")
    else:
        print("[fq-serve] FQ_ENFORCE_EAGER=0 -> CUDA graphs ENABLED (speedup A/B; "
              "must prove token-id parity vs enforce-eager)", flush=True)
    print(f"[fq-serve] launching: {' '.join(argv)}", flush=True)
    sys.argv = argv
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
