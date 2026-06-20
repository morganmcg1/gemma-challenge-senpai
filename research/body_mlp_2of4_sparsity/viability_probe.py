#!/usr/bin/env python3
"""PR #808 Step-1 viability kill-gate probe.

Question: can vLLM (the pinned serving stack on this pod) load + serve a
compressed-tensors checkpoint that stacks 2:4 structured sparsity on top of
int4 W4A16 for the body MLP, routed to a sparse-aware kernel on sm_86 (A10G)?

This probe does NOT touch the GPU. It (1) enumerates the installed
compressed-tensors quant schemes and the compiled torch.ops._C kernels, then
(2) drives the REAL vLLM load-time config parser
(CompressedTensorsConfig.from_config) on a faithful reproduction of what
llm-compressor writes for a "2:4 + W4A16" export. If the serving stack has no
sparse path, this is the cheapest possible kill.

Run with the venv that backs the running int4head server, e.g.:
    /tmp/senpai-venvs/<hash>/bin/python research/body_mlp_2of4_sparsity/viability_probe.py
"""
import importlib.metadata as md
import os
import traceback

import torch
import vllm  # noqa: F401
import vllm._C  # noqa: F401  (registers compiled ops into torch.ops._C)
from vllm.model_executor.layers.quantization.compressed_tensors import schemes
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)


def section(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * 4}")


def main() -> int:
    section("PINNED STACK")
    for p in ("vllm", "transformers", "compressed-tensors", "torch"):
        try:
            print(f"  {p}=={md.version(p)}")
        except Exception as e:  # noqa: BLE001
            print(f"  {p}: NOT INSTALLED ({type(e).__name__})")
    print(f"  device capability: sm_{''.join(map(str, torch.cuda.get_device_capability()))}"
          if torch.cuda.is_available() else "  CUDA not available")

    section("INSTALLED compressed_tensors SCHEMES")
    scheme_dir = os.path.dirname(schemes.__file__)
    files = sorted(f for f in os.listdir(scheme_dir) if f.endswith(".py"))
    print("  scheme files:", files)
    has_24_scheme = any("24" in f for f in files)
    print(f"  -> 2:4 sparse scheme file present? {has_24_scheme}")

    section("COMPILED SPARSE/MARLIN-24 KERNELS in torch.ops._C")
    ops = [x for x in dir(torch.ops._C)
           if "marlin" in x.lower() or "sparse" in x.lower() or "24" in x]
    print("  marlin/sparse/24 ops:", ops)
    print(f"  -> gptq_marlin_24_gemm compiled? {'gptq_marlin_24_gemm' in ops}")

    section("DRIVE THE REAL vLLM LOAD-TIME PARSER on a 2:4 + W4A16 config")
    # Faithful to an llm-compressor 2:4 + int4-W4A16 body-MLP export
    # (gate/up/down only; attention, lm_head, embeddings left dense/ignored).
    cfg = {
        "format": "marlin-24",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {"num_bits": 4, "type": "int", "strategy": "group",
                            "group_size": 128, "symmetric": True},
                "input_activations": None,
            }
        },
        "ignore": ["lm_head", "re:.*self_attn.*", "re:.*embed.*"],
        "sparsity_config": {
            "format": "sparse-24-bitmask",
            "sparsity_structure": "2:4",
            "targets": ["Linear"],
            "ignore": ["lm_head", "re:.*self_attn.*"],
        },
    }
    try:
        CompressedTensorsConfig.from_config(cfg)
        print("  LOADED OK (UNEXPECTED) -- sparse config accepted")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"  RAISED {type(e).__name__}: {e}")
        print("  --- traceback ---")
        traceback.print_exc()
        print("\n  VERDICT: serving stack rejects W4 + 2:4 at load -> lane DEAD at viability.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
