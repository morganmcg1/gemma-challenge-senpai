#!/usr/bin/env python
"""PR #708 step 3 — cheap greedy-identity / determinism spot-check.

The source read says mixed-grid IS servable (per-layer group_size). This confirms
the mixed g32/g128 Marlin kernel SELECTION introduces no nondeterminism: the exact
served kernel (apply_gptq_marlin_linear) must be byte-deterministic across repeated
calls for BOTH group sizes and for an interleaved g32/g128 sequence. (Marlin's
optional use_atomic_add path is non-deterministic; default off — we verify.)

This is NOT the full 128/128 launch gate (out of scope, no served-file change); it
is the kernel-determinism precondition for strict-#319 greedy identity under
mixed-grid kernel selection.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
_here = os.path.dirname(os.path.abspath(__file__))

HIDDEN, INTERMEDIATE = 2560, 10240
SHAPES = {
    "qkv_full": (2560, 3072), "o_proj": (2048, 2560),
    "gate_up_proj": (2560, 20480), "down_proj": (10240, 2560),
    "per_layer_input_gate": (2560, 256),
}
G128, G32 = 128, 32
SEED = 707


def main():
    import torch
    from vllm.model_executor.layers.quantization.utils import marlin_utils_test as mt
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear as apply_marlin, marlin_make_workspace_new as mk_ws)
    from vllm.scalar_type import scalar_types
    QT = scalar_types.uint4b8
    dev = torch.device("cuda:0")
    ws = mk_ws(dev); zp = torch.zeros(0, dtype=torch.int, device=dev)
    atomic = os.environ.get("VLLM_MARLIN_USE_ATOMIC_ADD", "0")

    def build(K, N, gs, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        w = torch.randn(K, N, dtype=torch.float16, device=dev, generator=g) * 0.02
        _, q_w, s, g_idx, sort_idx, _ = mt.marlin_quantize(w, QT, gs, False)
        return q_w, s, g_idx, sort_idx

    def fwd(x, packed, K, N):
        q_w, s, g_idx, sort_idx = packed
        return apply_marlin(x, q_w, s, zp, g_idx, sort_idx, ws, QT, N, K,
                            is_k_full=True, bias=None)

    results = {"atomic_add_env": atomic, "per_shape": {}, "interleave": {}}
    all_ok = True
    for name, (K, N) in SHAPES.items():
        x = torch.randn(1, K, dtype=torch.float16, device=dev,
                        generator=torch.Generator(device=dev).manual_seed(SEED + 9))
        w128 = build(K, N, G128, SEED)
        w32 = build(K, N, G32, SEED)
        # repeated-call determinism for each group size
        y128a = fwd(x, w128, K, N).clone(); y128b = fwd(x, w128, K, N).clone()
        y32a = fwd(x, w32, K, N).clone(); y32b = fwd(x, w32, K, N).clone()
        det128 = bool(torch.equal(y128a, y128b))
        det32 = bool(torch.equal(y32a, y32b))
        results["per_shape"][name] = {
            "det_g128": det128, "det_g32": det32,
            "g128_vs_g32_differ": bool(not torch.equal(y128a, y32a)),  # sanity: gs changes numbers
            "max_abs_g128_minus_g32": float((y128a - y32a).abs().max()),
        }
        all_ok = all_ok and det128 and det32

    # interleaved mixed-grid sequence determinism (g128 module then g32 module),
    # run twice -> byte identical concatenation.
    seq = [("gate_up_proj", G128), ("per_layer_input_gate", G32),
           ("qkv_full", G128), ("o_proj", G32), ("down_proj", G128)]
    def run_seq():
        outs = []
        for name, gs in seq:
            K, N = SHAPES[name]
            x = torch.randn(1, K, dtype=torch.float16, device=dev,
                            generator=torch.Generator(device=dev).manual_seed(SEED + hash(name) % 1000))
            packed = build(K, N, gs, SEED)
            outs.append(fwd(x, packed, K, N).flatten())
        return torch.cat(outs)
    s1 = run_seq(); s2 = run_seq()
    inter_det = bool(torch.equal(s1, s2))
    results["interleave"] = {"mixed_grid_sequence_deterministic": inter_det,
                             "seq": [[n, g] for n, g in seq]}
    all_ok = all_ok and inter_det

    results["all_deterministic"] = all_ok
    results["verdict"] = ("MIXED_GRID_DETERMINISTIC" if all_ok
                          else "MIXED_GRID_NONDETERMINISTIC")
    json.dump(results, open(os.path.join(_here, "determinism_results.json"), "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
