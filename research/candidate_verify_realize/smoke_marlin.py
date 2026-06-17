#!/usr/bin/env python3
"""PR #560 smoke: can we Marlin-quantize the real 262k bf16 lm_head to int4 g128
and run the int4 Marlin GEMV at M=1, getting logits that match the dequantized
reference? Validates the kernel path before the full microbench."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
import vllm.model_executor.layers.quantization.utils.marlin_utils_test as mt

MODEL_DIR = (
    "/senpai-run/home/student-fern/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)
HIDDEN = 2560
VOCAB = 262144
HEAD_KEY = "lm_head.weight"
GROUP = 128
DTYPE = torch.bfloat16


def main() -> int:
    dev = "cuda"
    torch.cuda.init()
    print(f"[smoke] device={torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)

    path = Path(MODEL_DIR) / "model.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as f:
        W_head = f.get_tensor(HEAD_KEY)  # [VOCAB, HIDDEN], logits = x @ W_head.T
    assert tuple(W_head.shape) == (VOCAB, HIDDEN), W_head.shape
    print(f"[smoke] loaded {HEAD_KEY} {tuple(W_head.shape)} {W_head.dtype}", flush=True)

    # Marlin wants w = [K=HIDDEN, N=VOCAB]: output = x[M,K] @ w[K,N] = logits[M,VOCAB]
    w = W_head.t().contiguous().to(device=dev, dtype=DTYPE)  # [2560, 262144]
    del W_head

    t0 = time.time()
    w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, rand_perm = mt.marlin_quantize(
        w, scalar_types.uint4b8, GROUP, act_order=False
    )
    torch.cuda.synchronize()
    print(f"[smoke] marlin_quantize done in {time.time()-t0:.1f}s", flush=True)
    print(f"[smoke]   marlin_q_w {tuple(marlin_q_w.shape)} {marlin_q_w.dtype} "
          f"{marlin_q_w.numel()*marlin_q_w.element_size()/1e6:.1f}MB", flush=True)
    print(f"[smoke]   marlin_s   {tuple(marlin_s.shape)} {marlin_s.dtype} "
          f"{marlin_s.numel()*marlin_s.element_size()/1e6:.1f}MB", flush=True)
    print(f"[smoke]   g_idx {tuple(g_idx.shape)} sort_indices {tuple(sort_indices.shape)}", flush=True)

    zp = mu.marlin_make_empty_g_idx(dev)
    workspace = mu.marlin_make_workspace_new(torch.device(dev))
    print(f"[smoke]   workspace {tuple(workspace.shape)}  zp {tuple(zp.shape)}", flush=True)

    x = torch.randn(1, HIDDEN, dtype=DTYPE, device=dev)

    def marlin_gemm(xin):
        return ops.marlin_gemm(
            xin, None, marlin_q_w, None, marlin_s, None, None, zp,
            g_idx, sort_indices, workspace, scalar_types.uint4b8,
            size_m=xin.shape[0], size_n=VOCAB, size_k=HIDDEN,
            is_k_full=True, use_atomic_add=False, use_fp32_reduce=True,
            is_zp_float=False,
        )

    out = marlin_gemm(x)
    torch.cuda.synchronize()
    print(f"[smoke] marlin out {tuple(out.shape)} {out.dtype}", flush=True)

    ref = (x.float() @ w_ref.float())  # [1, VOCAB]
    out_f = out.float()
    max_abs = (out_f - ref).abs().max().item()
    rel = max_abs / (ref.abs().max().item() + 1e-9)
    # argmax agreement: does the int4 Marlin argmax match the dequant-ref argmax?
    am_marlin = out_f.argmax(dim=1)
    am_ref = ref.argmax(dim=1)
    print(f"[smoke] max_abs(out-ref)={max_abs:.4e} rel={rel:.4e} "
          f"argmax_marlin={am_marlin.item()} argmax_ref={am_ref.item()} "
          f"match={bool((am_marlin==am_ref).all())}", flush=True)

    # also vs the TRUE bf16 head argmax (the served token)
    wt = w  # [K,N] bf16 full precision (== W_head.T)
    ref_bf16 = (x @ wt).float()
    am_bf16 = ref_bf16.argmax(dim=1)
    print(f"[smoke] bf16-head argmax={am_bf16.item()} (int4 may differ -> that's why verify exists)", flush=True)
    print("[smoke] OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
