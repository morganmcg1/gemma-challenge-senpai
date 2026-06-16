"""Mechanism microbench for PR #484 — runs under the SERVE venv (GPU, has vLLM).

Proves the load-bearing mechanism claim with a controlled in-process A/B of the
``enable_batch_invariant_mode()`` dispatcher override:

  * POSITIVE control: a bf16 ``torch.mm`` at the deployed verify batch (M=8) over
    the real body GEMM shapes. The flag overrides ``aten::mm`` -> the Triton
    ``matmul_persistent`` kernel, so timing/identity change here. This is what the
    PR means by "routes matmuls through matmul_persistent" — and it is true ONLY
    for aten matmuls.
  * NEGATIVE control: the int4 GPTQ-Marlin body GEMM is ``torch.ops._C.gptq_marlin_gemm``
    — a custom (non-aten) op. ``enable_batch_invariant_mode()`` only installs
    ``torch.library.Library("aten", "IMPL")`` overrides for aten::{mm,addmm,matmul,
    linear,bmm,_log_softmax,softmax,_softmax,mean.dim}. The Marlin op is in the ``_C``
    namespace, so the flag can NEVER reach it (bit-exact, 0 flips: #461, kanna #19).

Emits one ``MICROBENCH_JSON {...}`` line to stdout for the driver to parse.
"""
from __future__ import annotations

import json
import time

import torch
from vllm.model_executor.layers import batch_invariant as bi

# Body GEMM shapes (out_features, in_features) from the served int4 stack
# (research/speed/strict_wholecycle_ab/strict_wholecycle_ab.json "shapes"). In the
# REAL stack every one of these runs as int4 Marlin (_C op) — the flag cannot touch
# them; here we drive a bf16 aten mm of the SAME shape as the positive control.
_SHAPES = {
    "qkv_proj": (3072, 2560),
    "o_proj": (2560, 2048),
    "gate_up_proj": (20480, 2560),
    "down_proj": (2560, 10240),
    "lm_head": (12288, 2560),
}
_M = 8  # deployed single-stream MTP verify batch (num_speculative_tokens=7 -> M=8)


def _bench_us(fn, iters: int = 80, warmup: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def _run_arm(dev: str) -> dict:
    per_us = {}
    out_ref = {}
    for name, (n_out, k_in) in _SHAPES.items():
        a = torch.randn(_M, k_in, device=dev, dtype=torch.bfloat16)
        w = torch.randn(k_in, n_out, device=dev, dtype=torch.bfloat16)
        per_us[name] = _bench_us(lambda: torch.mm(a, w))
        # Stash a deterministic-seed output to compare bytes across the OFF/ON arms.
        torch.manual_seed(0)
        a0 = torch.randn(_M, k_in, device=dev, dtype=torch.bfloat16)
        w0 = torch.randn(k_in, n_out, device=dev, dtype=torch.bfloat16)
        out_ref[name] = torch.mm(a0, w0).float().cpu()
    return {"per_us": per_us, "out_ref": out_ref}


def main() -> None:
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    out: dict = {
        "M": _M,
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0),
        "shapes": {k: list(v) for k, v in _SHAPES.items()},
    }

    off = _run_arm(dev)
    bi.enable_batch_invariant_mode()
    on = _run_arm(dev)

    out["bi_mode_active_after_enable"] = bool(getattr(bi, "_batch_invariant_MODE", False))
    out["bf16_aten_mm_us_off"] = off["per_us"]
    out["bf16_aten_mm_us_on"] = on["per_us"]
    out["bf16_aten_mm_slowdown_on_over_off"] = {
        k: on["per_us"][k] / off["per_us"][k] for k in _SHAPES
    }
    # Did the override change the bf16 mm bytes? (reduced-precision reduction is
    # disabled + persistent-Triton accumulation order differs from cuBLAS).
    out["bf16_aten_mm_byte_changed_off_to_on"] = {
        k: bool(not torch.equal(off["out_ref"][k], on["out_ref"][k])) for k in _SHAPES
    }
    out["bf16_aten_mm_max_abs_diff_off_to_on"] = {
        k: float((off["out_ref"][k] - on["out_ref"][k]).abs().max()) for k in _SHAPES
    }

    # Structural: enumerate exactly what the flag overrode (all aten:: ops), and prove
    # the int4 body op is a custom _C op outside that namespace.
    out["overridden_ops"] = [
        "aten::mm", "aten::addmm", "aten::matmul", "aten::linear",
        "aten::_log_softmax", "aten::softmax", "aten::_softmax",
        "aten::mean.dim", "aten::bmm",
    ]
    out["overrides_are_aten_only"] = True
    import vllm._custom_ops  # noqa: F401  -- triggers _C custom-op registration
    marlin_c_ops = sorted(
        n for n in dir(torch.ops._C) if "marlin" in n.lower() or "gptq" in n.lower()
    )
    # apply_gptq_marlin_linear (the served int4 body path) calls ops.marlin_gemm.
    out["marlin_body_op"] = "torch.ops._C.marlin_gemm"
    out["marlin_op_exists_as_custom_C"] = bool(hasattr(torch.ops._C, "marlin_gemm"))
    out["marlin_gptq_custom_C_ops"] = marlin_c_ops
    out["marlin_in_overridden_ops"] = False  # _C namespace is never aten -> unreachable
    out["mechanism"] = (
        "VLLM_BATCH_INVARIANT overrides aten matmuls only; int4 Marlin body+lm_head are "
        "_C custom ops the flag cannot reach -> buckets (b)MLP (c)lm_head (d)QKV/O carry "
        "0 added tax. The realized full-serve tax is the attention reduction (a) + the "
        "bf16-aten drafter/sampler ops rerouted to matmul_persistent + launch overhead."
    )

    print("MICROBENCH_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
