#!/usr/bin/env python
"""Step-1 viability probe (PR #807): is there ANY servable W4A8 body GEMM kernel
on this A10G (sm_86) in vLLM 0.22.0?

We answer it at the kernel-selector level (the layer the compressed-tensors
W4A8-int / QQQ / Cutlass-W4A8 schemes all funnel through): build the exact
``MPLinearLayerConfig`` each W4A8 path would produce and ask
``choose_mp_linear_kernel`` what it dispatches on the live device. A W4A8 lever
is only real if a kernel that consumes *reduced-precision activations* (int8 or
fp8) for an int4 weight can be selected on sm_86. A fall-back to a bf16-activation
W4A16 kernel (Marlin) is NOT W4A8 -- it gives zero activation-byte reduction.

LOCAL ONLY -- no HF Job, no model load. Pure kernel-selector introspection.
Run: CUDA_VISIBLE_DEVICES=0 /tmp/server-venv/bin/python probe_w4a8_kernels.py
"""
from __future__ import annotations

import torch

from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.model_executor.kernels.linear import (
    choose_mp_linear_kernel,
    _POSSIBLE_KERNELS,
)
from vllm.model_executor.kernels.linear.mixed_precision import MPLinearLayerConfig


def cc_int() -> int:
    cc = current_platform.get_device_capability()
    return cc[0] * 10 + cc[1]


# Gemma-4-E4B body MLP-ish shape (in=2560 hidden, big out); divisible by 128.
IN, OUT = 2560, 8192


def make_cfg(act: torch.dtype, gs: int, wt=scalar_types.int4) -> MPLinearLayerConfig:
    return MPLinearLayerConfig(
        full_weight_shape=(IN, OUT),
        partition_weight_shape=(IN, OUT),
        weight_type=wt,
        act_type=act,
        group_size=gs,
        zero_points=False,
        has_g_idx=False,
        out_type=torch.bfloat16,
    )


def main() -> None:
    assert current_platform.is_cuda(), "must run on the A10G with CUDA_VISIBLE_DEVICES=0"
    cc = cc_int()
    print(f"device = {torch.cuda.get_device_name(0)}  sm_{cc}")
    print(f"platform enum = {current_platform._enum}")
    print()

    order = _POSSIBLE_KERNELS[current_platform._enum]
    print("MPLinear kernel search order + min-capability (this platform):")
    for k in order:
        print(f"  {k.__name__:30s} min_cap={k.get_min_capability()}  "
              f"{'OK' if k.get_min_capability() <= cc else 'BLOCKED@sm_%d' % cc}")
    print()

    # Two weight encodings matter on sm_86:
    #  * scalar_types.int4   -> what compressed_tensors' CompressedTensorsW4A8Int
    #                            scheme actually uses (signed int4).
    #  * scalar_types.uint4b8-> the GPTQ-style packing the DEPLOYED int4 W4A16
    #                            body uses and that Marlin actually accepts.
    # A W4A8 lever needs a kernel taking int8/fp8 *activations*; the bf16 rows are
    # the W4A16 controls that prove what truly serves on this device.
    i4 = scalar_types.int4
    u4 = scalar_types.uint4b8
    cases = [
        # (label, act_dtype, group_size, weight_type)
        ("W4A8-int  signed-int4 (int8 act)", torch.int8,          128, i4),
        ("W4A8-int  signed-int4 (int8 act) g32", torch.int8,       32, i4),
        ("W4A8-fp8  signed-int4 (e4m3 act)", torch.float8_e4m3fn,  128, i4),
        ("W4A16 ctl signed-int4 (bf16)",     torch.bfloat16,       128, i4),
        ("--- GPTQ-packed uint4b8 (deployed body encoding) ---", torch.bfloat16, 32, u4),
        ("W4A16 ctl uint4b8 (bf16) g32 [DEPLOYED]", torch.bfloat16, 32, u4),
        ("W4A16 ctl uint4b8 (bf16) g128",    torch.bfloat16,      128, u4),
        ("W4A8-int  uint4b8 (int8 act) g128", torch.int8,         128, u4),
        ("W4A8-int  uint4b8 (int8 act) g32",  torch.int8,          32, u4),
        ("W4A8-fp8  uint4b8 (e4m3 act) g128", torch.float8_e4m3fn, 128, u4),
    ]

    for label, act, gs, wt in cases:
        if label.startswith("---"):
            print(label + "\n")
            continue
        cfg = make_cfg(act, gs, wt)
        print(f"=== {label}: act={act}, group_size={gs}, wtype={wt} ===")
        try:
            chosen = choose_mp_linear_kernel(cfg, compute_capability=cc)
            print(f"  -> SELECTED: {chosen.__name__}")
        except Exception as e:
            print(f"  -> NO KERNEL. {type(e).__name__}")
            # Per-kernel reasons (so we see WHY each W4A8-capable kernel refuses).
            for k in order:
                if k.get_min_capability() > cc:
                    print(f"     {k.__name__}: BLOCKED min_cap={k.get_min_capability()} > sm_{cc}")
                    continue
                ok, why = k.can_implement(cfg)
                print(f"     {k.__name__}: {'CAN' if ok else 'cannot'} -- {why}")
        print()


if __name__ == "__main__":
    main()
