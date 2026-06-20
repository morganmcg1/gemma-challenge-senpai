#!/usr/bin/env python
"""Step-1 viability KILL-GATE (PR #807), part 2: does the Marlin W4A8-INT8 CUDA
kernel actually DISPATCH and produce sane numerics on this A10G (sm_86)?

The kernel-selector probe (probe_w4a8_kernels.py) showed the *Python* selector
picks MarlinLinearKernel for (uint4b8 weight + int8 act), min_cap=75, and that
``VLLM_MARLIN_INPUT_DTYPE=int8`` -> get_marlin_input_dtype()=torch.int8 with NO
sm_86 capability block (only the fp8 path requires sm_89). But ``can_implement``
passing does NOT prove the compiled ``ops.marlin_gemm`` has an sm_86 binary for
the int8 path. A web-research pass claimed W4A8-int8 needs sm_89 minimum, citing
silent-W4A16-fallback issues. The only way to resolve the conflict is to RUN the
kernel on the device.

This mirrors the production path (MarlinLinearKernel.process_weights_after_loading
+ apply_gptq_marlin_linear) on a small int4 layer, for BOTH:
  * bf16  (W4A16, the deployed body kernel) -- the control that proves our setup,
  * int8  (W4A8, the lever under test).
If the int8 path RUNS and out8 ~= x @ w_ref (within int8 quant error) the kernel
exists and is numerically real on sm_86 -> Step-1 PASSES. If it raises a CUDA
"no kernel image" / dispatch error -> KILL (clean sm_86 wall, like #781/#779).

LOCAL ONLY -- no HF Job, no model load. Run:
  CUDA_VISIBLE_DEVICES=0 /tmp/server-venv/bin/python microkernel_w4a8_int8.py
"""
from __future__ import annotations

import torch

from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear,
    marlin_act_int8_process_scales,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)

DEV = torch.device("cuda:0")
U4 = scalar_types.uint4b8


def run_case(size_k: int, size_n: int, group_size: int, M: int, act_int8: bool):
    torch.manual_seed(0)
    w = (torch.randn(size_k, size_n, device=DEV, dtype=torch.bfloat16) * 0.08)
    x = (torch.randn(M, size_k, device=DEV, dtype=torch.bfloat16) * 1.0)

    input_dtype = torch.int8 if act_int8 else None
    w_ref, marlin_q_w, marlin_s, g_idx, sort_idx, _ = marlin_quantize(
        w, U4, group_size, act_order=False, input_dtype=input_dtype
    )

    workspace = marlin_make_workspace_new(DEV)
    empty = marlin_make_empty_g_idx(DEV)

    input_global_scale = None
    weight_scale = marlin_s
    num_groups = size_k // group_size if group_size != -1 else 1
    if act_int8 and num_groups > 1:
        weight_scale, input_global_scale = marlin_act_int8_process_scales(marlin_s)

    out = apply_gptq_marlin_linear(
        input=x,
        weight=marlin_q_w,
        weight_scale=weight_scale,
        weight_zp=empty,
        g_idx=empty,
        g_idx_sort_indices=empty,
        workspace=workspace,
        wtype=U4,
        output_size_per_partition=size_n,
        input_size_per_partition=size_k,
        is_k_full=True,
        input_global_scale=input_global_scale,
        bias=None,
        input_dtype=input_dtype,
    )
    torch.cuda.synchronize()

    ref = x.float() @ w_ref.float()
    rel = (out.float() - ref).norm() / ref.norm().clamp_min(1e-9)
    finite = bool(torch.isfinite(out).all().item())
    return out.shape, float(rel), finite


def main() -> None:
    print(f"device = {torch.cuda.get_device_name(0)} "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    print()
    # (size_k, size_n, group_size); body-like shapes, tile-divisible.
    shapes = [
        (2560, 5120, 128),
        (2560, 5120, 32),   # g32 == the DEPLOYED int4 body group size
    ]
    for (k, n, gs) in shapes:
        for M in (1, 8):     # M=1 decode, M=8 spec-verify
            print(f"--- size_k={k} size_n={n} group_size={gs} M={M} ---")
            # bf16 W4A16 control first (proves the harness is wired correctly).
            try:
                sh, rel, fin = run_case(k, n, gs, M, act_int8=False)
                print(f"  W4A16 bf16 : out={tuple(sh)} rel_err={rel:.4f} finite={fin}  OK")
            except Exception as e:
                print(f"  W4A16 bf16 : FAILED {type(e).__name__}: {e}")
            # int8 W4A8 lever under test.
            try:
                sh, rel, fin = run_case(k, n, gs, M, act_int8=True)
                verdict = "DISPATCHED+SANE" if (fin and rel < 0.2) else (
                    "DISPATCHED-BUT-NUMERICALLY-OFF" if fin else "NONFINITE")
                print(f"  W4A8  int8 : out={tuple(sh)} rel_err={rel:.4f} finite={fin}  {verdict}")
            except Exception as e:
                print(f"  W4A8  int8 : KILL -> {type(e).__name__}: {e}")
            print()


if __name__ == "__main__":
    main()
