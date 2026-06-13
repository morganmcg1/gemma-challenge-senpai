#!/usr/bin/env python
"""Decisive kernel-level confirmation that vLLM batch-invariant mode is active
AND functional on this GPU. LOCAL A10G diagnostic; NOT an HF Job.

The PR rests on one mechanism: with VLLM_BATCH_INVARIANT=1, a GEMM y = x @ W
returns the SAME output row regardless of how many rows are in the batch. That is
exactly the M=1 (autoregressive decode) vs M=K+1 (speculative verify) equivalence
we need for the argmax-determining bf16 lm_head matmul.

We test it directly: build one input batch, run the full M=8 GEMM, then re-run the
M=1 and M=7(=K+1) sub-batches, and check the overlapping rows are BIT-IDENTICAL.
- VLLM_BATCH_INVARIANT=1  -> rows must match bit-exactly (max|diff|==0).
- VLLM_BATCH_INVARIANT=0  -> on cuBLAS the M=1 vs M=7 kernels may differ -> nonzero
  (this is the control: the source of the spec-decode greedy flip).

Also reports the module mode flag and that the invariant Triton matmul ran without
error (a failure would mean Triton is unavailable on this box -> invariance broken).
"""
from __future__ import annotations

import json
import os
import sys


def main() -> int:
    want = os.environ.get("VLLM_BATCH_INVARIANT", "<unset>")
    print(f"[confirm] VLLM_BATCH_INVARIANT(env) = {want}")

    import torch

    if not torch.cuda.is_available():
        print("[confirm] FAIL: CUDA not available")
        return 2
    print(f"[confirm] device = {torch.cuda.get_device_name(0)}  torch={torch.__version__}")

    import vllm.envs as envs
    import vllm.model_executor.layers.batch_invariant as bi

    print(f"[confirm] vllm.envs.VLLM_BATCH_INVARIANT = {envs.VLLM_BATCH_INVARIANT}")

    # The GPU worker calls this during init; call it here so the standalone probe
    # mirrors the server. It is gated internally on envs.VLLM_BATCH_INVARIANT, so
    # with the env unset/0 it is a no-op (giving the OFF control).
    bi.init_batch_invariance()
    mode = getattr(bi, "_batch_invariant_MODE", None)
    print(f"[confirm] batch_invariant._batch_invariant_MODE = {mode}")

    # Triton availability: the invariant matmul is a Triton persistent kernel. If
    # Triton is disabled on this box, enabling invariance + running aten::mm would
    # raise. We surface it explicitly.
    try:
        import triton  # noqa: F401
        triton_ok = True
    except Exception as exc:  # pragma: no cover
        triton_ok = False
        print(f"[confirm] WARN: triton import failed: {exc}")
    print(f"[confirm] triton import ok = {triton_ok}")

    torch.manual_seed(0)
    dev = "cuda"
    dt = torch.bfloat16
    Kdim, Ndim, M = 2560, 8192, 8  # M>=K+1=7; bf16 like the tied lm_head matmul
    x = torch.randn(M, Kdim, device=dev, dtype=dt)
    W = torch.randn(Kdim, Ndim, device=dev, dtype=dt)

    y_full = x @ W  # the M=8 "batched verify"-shape forward
    results = {}
    ok = True
    for m in (1, 7):  # M=1 decode, M=7 = K+1 verify
        y_sub = x[:m] @ W
        diff = (y_sub - y_full[:m]).abs().max().item()
        exact = diff == 0.0
        results[f"M{m}_vs_full_maxabsdiff"] = diff
        results[f"M{m}_vs_full_bitexact"] = exact
        print(f"[confirm] M={m} vs full[:{m}]  max|diff|={diff:.3e}  bitexact={exact}")
        ok = ok and exact

    enabled = bool(envs.VLLM_BATCH_INVARIANT)
    verdict = (
        "INVARIANT_ACTIVE_AND_FUNCTIONAL" if (enabled and ok)
        else "ENABLED_BUT_NOT_BITEXACT" if (enabled and not ok)
        else "OFF_CONTROL_NONZERO" if (not enabled and not ok)
        else "OFF_BUT_BITEXACT"
    )
    out = {
        "env_VLLM_BATCH_INVARIANT": want,
        "vllm_envs_enabled": enabled,
        "mode_flag": bool(mode),
        "triton_ok": triton_ok,
        "rows_bitexact": ok,
        "verdict": verdict,
        **results,
    }
    print("CONFIRM_JSON " + json.dumps(out))
    # Exit 0 only when the env+behavior agree (ON->bitexact, OFF->control).
    if enabled and not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
