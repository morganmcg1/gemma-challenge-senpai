#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""FlashInfer decode-attention M=1-vs-M=8 byte-exact census + TPS micro-bench (PR #507).

Runs INSIDE the isolated flashinfer venv (default /tmp/fi_probe/bin/python), because
flashinfer-python pins torch 2.10+cu128 while the pod's serving-adjacent .venv carries
torch 2.11+cu130 (version skew; see the parent probe's ledger). Emits a single
`JSON:{...}` line consumed by probe_flashinfer_feasibility.py.

The decisive test: with all M batch members sharing the SAME physical paged-KV pages and
an identical decode query, is request-0's attention output byte-identical between an M=1
plan and an M=8 plan? Different batch -> the split-KV scheduler may pick a different number
of KV chunks -> a different merge_states reduction tree -> low-mantissa-bit divergence.

Gemma-4-E4B sliding-layer attention shape is baked in: num_qo_heads=8, num_kv_heads=2
(GQA group 4 -> vLLM uses the tensor-core decode path), head_dim=256, page_size=16,
bf16 KV. head_dim=512 (the 7 full-attention layers) is probed separately for support.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import torch

try:
    import flashinfer
except Exception as exc:  # noqa: BLE001
    print("JSON:" + json.dumps({"flashinfer_import": False,
                                "error": f"{type(exc).__name__}: {exc}"}))
    raise SystemExit(0)

DEV = "cuda"
PAGE = 16
H_Q, H_KV, D = 8, 2, 256  # Gemma-4-E4B sliding-layer attention (GQA group 4)
SEED = 1234
WS_BYTES = 512 * 1024 * 1024
_WS = torch.empty(WS_BYTES, dtype=torch.uint8, device=DEV)


def _inputs(L: int, D_: int, dtype: torch.dtype):
    """Identical q/kv for every batch member: one physical KV pool, one decode query."""
    npages = (L + PAGE - 1) // PAGE
    g = torch.Generator(device=DEV).manual_seed(SEED)
    kv_pool = torch.randn(npages, 2, PAGE, H_KV, D_, device=DEV, dtype=dtype, generator=g) * 0.5
    q1 = torch.randn(H_Q, D_, device=DEV, dtype=dtype, generator=g) * 0.5
    return kv_pool, q1, npages


def _run(M: int, L: int, D_: int, dtype: torch.dtype, use_tc: bool, **plan_kw):
    kv_pool, q1, npages = _inputs(L, D_, dtype)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(_WS, kv_layout="NHD", use_tensor_cores=use_tc)
    # every request points at the SAME npages physical pages -> identical KV across the batch
    indptr = torch.arange(0, (M + 1) * npages, npages, device=DEV, dtype=torch.int32)
    indices = torch.arange(0, npages, device=DEV, dtype=torch.int32).repeat(M)
    last_page_len = torch.full((M,), L - (npages - 1) * PAGE, device=DEV, dtype=torch.int32)
    w.plan(indptr, indices, last_page_len, H_Q, H_KV, D_, PAGE,
           pos_encoding_mode="NONE", q_data_type=dtype, kv_data_type=dtype, **plan_kw)
    q = q1.unsqueeze(0).expand(M, H_Q, D_).contiguous()
    out = w.run(q, kv_pool)
    torch.cuda.synchronize()
    return out


def _census_cell(L: int, dtype: torch.dtype, use_tc: bool, **plan_kw) -> dict[str, Any]:
    o1 = _run(1, L, D, dtype, use_tc, **plan_kw)[0].float()
    o8 = _run(8, L, D, dtype, use_tc, **plan_kw)[0].float()
    flips = int((o1 != o8).sum().item())
    maxabs = float((o1 - o8).abs().max().item())
    nan = bool(torch.isnan(o1).any() or torch.isnan(o8).any())
    return {"flips": flips, "numel": int(o1.numel()), "max_abs": maxabs,
            "nan": nan, "invariant": int(flips == 0 and not nan)}


def headdim_support() -> dict[str, Any]:
    """Does flashinfer decode dispatch at head_dim 256 (sliding) and 512 (full) on sm_86?"""
    out: dict[str, Any] = {}
    for use_tc in (False, True):
        for D_ in (256, 512):
            key = f"{'tensor_core' if use_tc else 'cuda_core'}_d{D_}"
            try:
                o = _run(1, 4096, D_, torch.bfloat16, use_tc)
                out[key] = {"ok": True, "shape": list(o.shape),
                            "nan": bool(torch.isnan(o).any())}
            except Exception as exc:  # noqa: BLE001
                out[key] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:140]}"}
    return out


def census(L_set: list[int]) -> dict[str, Any]:
    dtype = torch.bfloat16
    res: dict[str, Any] = {}
    for use_tc in (False, True):
        path = "tensor_core" if use_tc else "cuda_core"
        res[path] = {}
        modes = [("auto", {})]
        if use_tc:
            modes += [("fixed_split_512", dict(fixed_split_size=512)),
                      ("fixed_split_256", dict(fixed_split_size=256))]
        modes += [("disable_split_kv", dict(disable_split_kv=True))]
        for name, kw in modes:
            res[path][name] = {}
            for L in L_set:
                try:
                    res[path][name][str(L)] = _census_cell(L, dtype, use_tc, **kw)
                except Exception as exc:  # noqa: BLE001
                    res[path][name][str(L)] = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
    return res


def _bench(w, q, kv, iters=200) -> float:
    for _ in range(20):
        w.run(q, kv)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        w.run(q, kv)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms/step


def tps_bench(L_set: list[int]) -> dict[str, Any]:
    """M=1 decode-attn micro-throughput (tensor-core, D=256, bf16): the cost of invariance."""
    dtype = torch.bfloat16
    out: dict[str, Any] = {}
    for L in L_set:
        out[str(L)] = {}
        for name, kw in [("auto", {}), ("fixed_split_512", dict(fixed_split_size=512)),
                         ("disable_split_kv", dict(disable_split_kv=True))]:
            try:
                kv_pool, q1, npages = _inputs(L, D, dtype)
                w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(_WS, kv_layout="NHD",
                                                                  use_tensor_cores=True)
                indptr = torch.tensor([0, npages], device=DEV, dtype=torch.int32)
                indices = torch.arange(0, npages, device=DEV, dtype=torch.int32)
                lpl = torch.tensor([L - (npages - 1) * PAGE], device=DEV, dtype=torch.int32)
                w.plan(indptr, indices, lpl, H_Q, H_KV, D, PAGE, pos_encoding_mode="NONE",
                       q_data_type=dtype, kv_data_type=dtype, **kw)
                q = q1.unsqueeze(0).contiguous()
                ms = _bench(w, q, kv_pool)
                out[str(L)][name] = {"us_per_step": ms * 1000.0, "steps_per_s": 1000.0 / ms}
            except Exception as exc:  # noqa: BLE001
                out[str(L)][name] = {"error": f"{type(exc).__name__}: {str(exc)[:80]}"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census-l", default="2048,8192,32768")
    ap.add_argument("--tps-l", default="2048,4096,8192")
    args = ap.parse_args()
    census_l = [int(x) for x in args.census_l.split(",")]
    tps_l = [int(x) for x in args.tps_l.split(",")]

    payload: dict[str, Any] = {
        "flashinfer_import": True,
        "flashinfer_version": getattr(flashinfer, "__version__", "?"),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "config": {"num_qo_heads": H_Q, "num_kv_heads": H_KV, "head_dim": D,
                   "page_size": PAGE, "dtype": "bfloat16", "seed": SEED,
                   "census_l": census_l, "tps_l": tps_l},
        "headdim_support": headdim_support(),
        "census": census(census_l),
        "tps": tps_bench(tps_l),
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 1e6,
    }
    print("JSON:" + json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
