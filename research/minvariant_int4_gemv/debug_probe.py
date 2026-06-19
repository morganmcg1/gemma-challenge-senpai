#!/usr/bin/env python
"""Debug: isolate NaN source + GENUINE finite M1-vs-M7 divergence."""
from __future__ import annotations
import torch
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT, apply_gptq_marlin_linear, marlin_make_empty_g_idx,
    marlin_make_workspace_new, should_use_atomic_add_reduce,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
from vllm.scalar_type import scalar_types

dev = torch.device("cuda:0")
WTYPE = scalar_types.uint4b8
sk, sn = 2560, 3072

def make(scale_w, seed=7):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(sk, sn, generator=g) * scale_w).to(dev, torch.float32)
    w_ref, q, s, g_idx, sort, _ = marlin_quantize(w, WTYPE, 128, act_order=False)
    return w_ref, q, s

empty = marlin_make_empty_g_idx(dev)
gx = torch.Generator(device="cpu").manual_seed(123)
x = (torch.randn(sk, generator=gx) * 0.1).to(dev, torch.bfloat16)  # milder activation

def call(M, q, s, ws):
    a = x.view(1, sk).expand(M, sk).contiguous()
    return apply_gptq_marlin_linear(
        input=a, weight=q, weight_scale=s, weight_zp=empty, g_idx=empty,
        g_idx_sort_indices=empty, workspace=ws, wtype=WTYPE,
        output_size_per_partition=sn, input_size_per_partition=sk,
        is_k_full=True, bias=None, use_fp32_reduce=USE_FP32_REDUCE_DEFAULT,
    )

for scale_w in (0.05, 0.2):
    w_ref, q, s = make(scale_w)
    o1 = call(1, q, s, marlin_make_workspace_new(dev))[0].float()
    o7 = call(7, q, s, marlin_make_workspace_new(dev))[0].float()
    nan1, nan7 = o1.isnan(), o7.isnan()
    both_fin = (~nan1) & (~nan7)
    nan_same = bool(torch.equal(nan1, nan7))
    d = (o7 - o1).abs()
    df = d[both_fin]
    nbit = int((o7[both_fin] != o1[both_fin]).sum())
    print(f"\n--- scale_w={scale_w} ---")
    print(f"nan1={int(nan1.sum())} nan7={int(nan7.sum())} nan_positions_identical={nan_same}")
    print(f"finite_both={int(both_fin.sum())}/{sn}  bitdiff(finite)={nbit} "
          f"({nbit/max(1,int(both_fin.sum()))*100:.2f}%)  max|d|={df.max().item():.3e} "
          f"mean|d|={df.mean().item():.3e}")
    # vs fp32 dequant reference: which is closer? (neither is ground truth for argmax)
    ref = (x.float() @ w_ref.float())[None].float()[0] if False else (x.float() @ w_ref.float())
    e1 = (o1 - ref).abs()[both_fin].mean().item()
    e7 = (o7 - ref).abs()[both_fin].mean().item()
    print(f"mean|o1-ref|={e1:.3e}  mean|o7-ref|={e7:.3e}  (ref=fp32 dequant matmul)")
