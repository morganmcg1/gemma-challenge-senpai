#!/usr/bin/env python
"""PR #736 wirbel — definitive M-sweep of the int4 Marlin GEMM on the REAL served
shapes, to pin whether the GEMM is the M-dependence locus route (a) assumes.

locus_real.py showed M=1 row-0 == M=7 row-0 BIT-EXACT for down_proj/q_proj/lm_head
(all N>=2048). This script closes the remaining gaps:

  * FUSED served shapes: vLLM fuses q/k/v -> ONE qkv GEMM (N=3072) and gate/up ->
    gate_up (N=20480). We reconstruct those fused weights (concat output channels
    + repack) so we time the EXACT GEMMs the served verify forward runs. All
    served Marlin GEMMs have N>=2048.
  * SEPARATE small-N k_proj/v_proj (N=512): NOT served (they're fused), but probe
    the small-N regime where Marlin's split-K heuristic is most M-sensitive.
  * Full M sweep [1..16] incl the real verify M=K+1 in {6,7,8}: catch any tile/
    split-K threshold where row-0 would start to diverge from the M=1 GEMV.
  * DISTINCT-rows verify at M=8: the truly faithful scenario (8 different candidate
    activations in one batch); compare each row i to its own M=1 GEMV.

If every SERVED shape is bit-exact across all M, the int4 Marlin GEMM is NOT the
M-dependence locus and route (a)'s premise ("make the verify GEMM reproduce the
M=1 reduction order") is already satisfied by the stock kernel.

ANALYSIS ONLY. No HF Job, no submission change, no fire.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from locus_real import (  # noqa: E402
    GROUP, MODEL, PACK, WTYPE, call, dequant_fp32, marlin_make_workspace_new,
)
from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # noqa: E402
    marlin_make_empty_g_idx, marlin_permute_scales, should_use_atomic_add_reduce,
)

MS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16]


def repack(wp, ws, dev):
    N, Kp = wp.shape
    K = Kp * PACK
    wp = wp.to(dev); ws = ws.to(dev)
    empty = marlin_make_empty_g_idx(dev)
    q = ops.gptq_marlin_repack(wp.t().contiguous(), perm=empty, size_k=K, size_n=N,
                               num_bits=4, is_a_8bit=False)
    s = marlin_permute_scales(ws.t().contiguous(), size_k=K, size_n=N, group_size=GROUP,
                              is_a_8bit=False)
    return {"q": q, "s": s, "K": K, "N": N, "wp": wp, "ws": ws, "empty": empty}


def load_fused(prefixes, dev):
    wps, wss = [], []
    with safe_open(MODEL, framework="pt", device="cpu") as f:
        for p in prefixes:
            wps.append(f.get_tensor(f"{p}.weight_packed"))
            wss.append(f.get_tensor(f"{p}.weight_scale"))
    wp = torch.cat(wps, dim=0)  # concat output channels (N)
    ws = torch.cat(wss, dim=0)
    return repack(wp, ws, dev)


def load_single(prefix, dev):
    with safe_open(MODEL, framework="pt", device="cpu") as f:
        wp = f.get_tensor(f"{prefix}.weight_packed")
        ws = f.get_tensor(f"{prefix}.weight_scale")
    return repack(wp, ws, dev)


A = "model.language_model.layers.0.self_attn"
M_ = "model.language_model.layers.0.mlp"
SHAPES = [
    ("qkv_proj.FUSED(served)", "fused", [f"{A}.q_proj", f"{A}.k_proj", f"{A}.v_proj"]),
    ("gate_up_proj.FUSED(served)", "fused", [f"{M_}.gate_proj", f"{M_}.up_proj"]),
    ("o_proj(served)", "single", f"{A}.o_proj"),
    ("down_proj(served)", "single", f"{M_}.down_proj"),
    ("lm_head(served)", "single", "lm_head"),
    ("k_proj.SEP(N=512,unserved)", "single", f"{A}.k_proj"),
    ("v_proj.SEP(N=512,unserved)", "single", f"{A}.v_proj"),
]


def sweep_layer(name, kind, spec, dev, seed):
    layer = load_fused(spec, dev) if kind == "fused" else load_single(spec, dev)
    K, N = layer["K"], layer["N"]
    wsb = marlin_make_workspace_new(dev)
    atomic = should_use_atomic_add_reduce(m=16, n=N, k=K, device=dev, dtype=torch.bfloat16)
    gx = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(K, generator=gx, dtype=torch.float32)).to(dev, torch.bfloat16)
    out1 = call(layer, x, 1, wsb)[0].float().clone()

    per_m = {}
    any_div = False
    for M in MS:
        if M == 1:
            continue
        r0 = call(layer, x, M, wsb)[0].float()
        nb = int((r0 != out1).sum().item())
        any_div = any_div or nb > 0
        per_m[M] = {"n_bitdiff": nb, "frac": nb / N, "max_abs": (r0 - out1).abs().max().item()}

    # distinct-rows verify at M=8 (faithful spec batch: 8 different candidates)
    gx2 = torch.Generator(device="cpu").manual_seed(seed + 1)
    Xd = (torch.randn(8, K, generator=gx2, dtype=torch.float32)).to(dev, torch.bfloat16)
    a8 = Xd.contiguous()
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        USE_FP32_REDUCE_DEFAULT, apply_gptq_marlin_linear,
    )
    out_batched = apply_gptq_marlin_linear(
        input=a8, weight=layer["q"], weight_scale=layer["s"], weight_zp=layer["empty"],
        g_idx=layer["empty"], g_idx_sort_indices=layer["empty"], workspace=wsb, wtype=WTYPE,
        output_size_per_partition=N, input_size_per_partition=K, is_k_full=True, bias=None,
        use_fp32_reduce=USE_FP32_REDUCE_DEFAULT,
    )
    distinct_bitdiff = 0
    for i in range(8):
        oi1 = call(layer, Xd[i], 1, wsb)[0].float()
        distinct_bitdiff += int((out_batched[i].float() != oi1).sum().item())

    return {
        "name": name, "size_k": K, "size_n": N, "use_atomic_add": bool(atomic),
        "any_divergence_vs_m1": any_div, "per_m": per_m,
        "distinct8_total_bitdiff_vs_own_m1": distinct_bitdiff,
        "distinct8_n_elem": 8 * N,
    }


def main() -> int:
    assert torch.cuda.is_available(), "CUDA required (CUDA_VISIBLE_DEVICES=0)"
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"[msweep] {torch.cuda.get_device_name(dev)} cap={torch.cuda.get_device_capability(dev)}", flush=True)
    print(f"[msweep] M sweep = {MS}", flush=True)

    out = []
    for i, (name, kind, spec) in enumerate(SHAPES):
        r = sweep_layer(name, kind, spec, dev, seed=4000 + i)
        out.append(r)
        ms_str = " ".join(f"M{M}:{r['per_m'][M]['n_bitdiff']}" for M in MS if M != 1)
        print(f"[msweep] {name:32s} K={r['size_k']:5d} N={r['size_n']:6d} atomic={r['use_atomic_add']} "
              f"ANY_DIV={r['any_divergence_vs_m1']} distinct8_bitdiff={r['distinct8_total_bitdiff_vs_own_m1']}",
              flush=True)
        print(f"[msweep]     bitdiff_vs_M1 per M: {ms_str}  "
              f"max|d|(M8)={r['per_m'][8]['max_abs']:.3e}", flush=True)

    report = {
        "pr": 736, "analysis_only": True, "official_tps": 0,
        "device": torch.cuda.get_device_name(dev), "M_sweep": MS,
        "note": "FUSED shapes are the real served GEMMs (vLLM fuses qkv & gate_up). "
                "SEP k/v (N=512) are checkpoint-only, never served.",
        "results": out,
    }
    (HERE / "locus_msweep_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[msweep] report -> {HERE / 'locus_msweep_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
