#!/usr/bin/env python
"""PR #736 wirbel — FAITHFUL real-weight int4 Marlin GEMV M-locus (route (a)).

Replaces the synthetic marlin_quantize microbench. The test-helper packing
(marlin_utils_test.marlin_quantize) is INCOMPATIBLE with the deployed vLLM 0.22.0
unified ops.marlin_gemm in this venv: its outputs are garbage (~1e37, median
relerr 0.93 vs an fp32 dequant reference), so the synthetic locus could not be
trusted. Here we instead load the REAL compressed-tensors int4 weights from the
served build and repack them through the EXACT production path used by
MarlinLinearKernel.process_weights_after_loading:

    transform_w_q: permute_param_layout_(input_dim=0,output_dim=1,packed_dim=0)  # plain .t()
                 + ops.gptq_marlin_repack(size_k,size_n,num_bits=4,is_a_8bit=False)
    transform_w_s: permute_param_layout_(input_dim=0,output_dim=1)               # plain .t()
                 + marlin_permute_scales(size_k,size_n,group_size=128)

so q / s are byte-identical to what the served Marlin kernel consumes. Then we
call the production apply_gptq_marlin_linear at M=1 (the AR / GEMV path that
#728's AR-vs-AR control proved run-to-run deterministic) vs M=K+1 (the spec
verify batch, K in {5,6}) on identical activation rows and report:

  (1) LOCUS         row-0 of the M=K+1 batched GEMM != the M=1 GEMV bit-for-bit
                    -> the reduction order is M-dependent. This is the
                    byte-divergence source #607 census + #616 flip-rate localized
                    end-to-end; #728's AR-vs-AR M=1 determinism control bounds it
                    100% to this kernel.
  (2) DETERMINISM   same M -> identical bits run-to-run (fixed schedule,
                    repairable in principle; NOT atomicAdd noise). Corroborates
                    #728 from the KERNEL side. should_use_atomic_add_reduce==False
                    on this A10G (sm86<90 + bf16; n>=2048 every verify shape).
  (3) MAGNITUDE     max|d| is a tiny FP reduction-order sliver, NOT a lossy bug.
  (4) TIE-FLIPS     on lm_head, M-dependence flips argmax only at near-tie
                    positions: every flip has top1-top2 gap <= ~2*max|d| << 0.3
                    nat (logit gap == nat gap since log p_i - log p_j = z_i - z_j).
                    Corroborates #616 (100% of flips are int4-grid ties).
  (5) FAITHFULNESS  kernel(M=1) ~= x @ dequant(real int4).T at the int4 quant-noise
                    floor (relerr ~1e-2, NOT 1e37) -> the real repack is correct.

ANALYSIS ONLY. analysis_only=1, official_tps=0. No HF Job, no submission change,
no fire. Assigned local A10G only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    apply_gptq_marlin_linear,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_permute_scales,
    should_use_atomic_add_reduce,
)
from vllm.scalar_type import scalar_types

HERE = Path(__file__).resolve().parent
MODEL = "/workspace/gemma_build/int4_g128_lmhead/model.safetensors"
WTYPE = scalar_types.uint4b8  # GPTQ-style int4, symmetric (bias 8, no zero-point)
GROUP = 128
PACK = 8  # 32 // 4

# (report-name, safetensors key prefix). K/N are read from the real tensors.
LAYERS = [
    ("down_proj.L0", "model.language_model.layers.0.mlp.down_proj"),  # biggest K=10240
    ("q_proj.L0", "model.language_model.layers.0.self_attn.q_proj"),  # K=2560 N=2048
    ("lm_head", "lm_head"),  # K=2560 N=262144, the token-argmax producer
]


def load_real_marlin(prefix: str, dev: torch.device):
    """Load real compressed-tensors int4 weight + scale and repack EXACTLY as
    MarlinLinearKernel.process_weights_after_loading does."""
    with safe_open(MODEL, framework="pt", device="cpu") as f:
        wp = f.get_tensor(f"{prefix}.weight_packed")  # [N, K//PACK] int32
        ws = f.get_tensor(f"{prefix}.weight_scale")  # [N, K//GROUP] bf16
    N, Kp = wp.shape
    K = Kp * PACK
    wp = wp.to(dev)
    ws = ws.to(dev)
    empty = marlin_make_empty_g_idx(dev)
    # transform_w_q: permute_param_layout_(input_dim=0,output_dim=1,packed_dim=0)
    # stored layout is (output_dim=0,input_dim=1,packed_dim=1) -> plain transpose.
    wp_t = wp.t().contiguous()  # [K//PACK, N], packed along dim0
    q = ops.gptq_marlin_repack(
        wp_t, perm=empty, size_k=K, size_n=N, num_bits=4, is_a_8bit=False
    )
    # transform_w_s: permute_param_layout_(input_dim=0,output_dim=1) -> transpose.
    ws_t = ws.t().contiguous()  # [K//GROUP, N]
    s = marlin_permute_scales(ws_t, size_k=K, size_n=N, group_size=GROUP, is_a_8bit=False)
    return {"q": q, "s": s, "K": K, "N": N, "wp": wp, "ws": ws, "empty": empty}


def dequant_fp32(wp: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """Dequantize the ORIGINAL compressed-tensors int4 packing to fp32 [N, K].
    uint4b8: nibble v in [0,15], real level = v - 8. Little-endian 8 nibbles/int32."""
    N, Kp = wp.shape
    K = Kp * PACK
    wu = wp.to(torch.int64) & 0xFFFFFFFF  # unsigned 32-bit, kill sign-extension
    shifts = (torch.arange(PACK, device=wp.device) * 4).view(1, 1, PACK)
    nib = (wu.unsqueeze(-1) >> shifts) & 0xF  # [N, Kp, 8]
    q = nib.reshape(N, K).to(torch.float32) - 8.0
    s = ws.to(torch.float32).repeat_interleave(GROUP, dim=1)  # [N, K]
    return q * s  # [N, K] fp32 dequantized weight (row=output n)


def call(layer, x_row: torch.Tensor, M: int, ws_buf) -> torch.Tensor:
    a = x_row.view(1, layer["K"]).expand(M, layer["K"]).contiguous()
    out = apply_gptq_marlin_linear(
        input=a, weight=layer["q"], weight_scale=layer["s"],
        weight_zp=layer["empty"], g_idx=layer["empty"], g_idx_sort_indices=layer["empty"],
        workspace=ws_buf, wtype=WTYPE,
        output_size_per_partition=layer["N"], input_size_per_partition=layer["K"],
        is_k_full=True, bias=None, use_fp32_reduce=USE_FP32_REDUCE_DEFAULT,
    )
    return out


def locus_for_layer(name, prefix, dev, ms, act_rms, seed):
    t0 = time.time()
    layer = load_real_marlin(prefix, dev)
    K, N = layer["K"], layer["N"]
    ws_buf = marlin_make_workspace_new(dev)
    atomic = should_use_atomic_add_reduce(m=max(ms), n=N, k=K, device=dev, dtype=torch.bfloat16)

    gx = torch.Generator(device="cpu").manual_seed(seed)
    x_row = (torch.randn(K, generator=gx, dtype=torch.float32) * act_rms).to(dev, torch.bfloat16)

    # M=1 baseline (the proven-deterministic AR/GEMV path) + its run-to-run repeat
    out1 = call(layer, x_row, 1, ws_buf)[0].float().clone()
    out1b = call(layer, x_row, 1, ws_buf)[0].float().clone()
    det_m1 = bool(torch.equal(out1, out1b))

    # faithfulness: kernel(M=1) vs fp32 dequant of the REAL int4 weights
    faith = None
    if N <= 70000:  # skip lm_head full dequant (2.7GB); proven on the proj layers
        w = dequant_fp32(layer["wp"], layer["ws"])  # [N, K]
        ref = (x_row.float() @ w.t())  # [N]
        denom = ref.abs().mean().clamp_min(1e-9)
        faith = {
            "mean_abs_kernel": out1.abs().mean().item(),
            "mean_abs_ref": ref.abs().mean().item(),
            "median_relerr_vs_fp32dequant": ((out1 - ref).abs() / (ref.abs() + 1e-6)).median().item(),
            "mean_abs_err_vs_fp32dequant": (out1 - ref).abs().mean().item(),
            "rel_l2_vs_fp32dequant": ((out1 - ref).norm() / ref.norm()).item(),
        }
        del w, ref

    per_m = {}
    for M in ms:
        if M == 1:
            continue
        outM = call(layer, x_row, M, ws_buf)
        row0 = outM[0].float()
        row0b = call(layer, x_row, M, ws_buf)[0].float()
        det_mM = bool(torch.equal(row0, row0b))
        cross = (outM[0].float() - outM[-1].float()).abs().max().item()
        d = (row0 - out1).abs()
        nbit = int((row0 != out1).sum().item())
        per_m[M] = {
            "n_elem": N,
            "n_bitdiff_vs_m1": nbit,
            "frac_bitdiff_vs_m1": nbit / N,
            "max_abs_diff_vs_m1": d.max().item(),
            "mean_abs_diff_vs_m1": d.mean().item(),
            "rel_max_diff_vs_m1": (d.max() / (out1.abs().max() + 1e-9)).item(),
            "row0_run2run_bitexact": det_mM,
            "cross_row_max_abs": cross,
        }
        del outM, row0, row0b

    return {
        "name": name, "size_k": K, "size_n": N, "act_rms": act_rms,
        "use_atomic_add": bool(atomic),
        "m1_run2run_bitexact": det_m1,
        "faithfulness": faith,
        "per_m": per_m,
        "load_s": time.time() - t0,
    }


def lmhead_tie_flips(dev, ms, n_act, act_rms, seed):
    """Kernel-level reproduction of #616: for many distinct activations, compare
    argmax(M=1) vs argmax(row-0 of an M=K+1 batch). Every flip must sit at a
    top1-top2 gap <= the FP reduction-noise envelope (~2*max|d|) << 0.3 nat."""
    layer = load_real_marlin("lm_head", dev)
    K, N = layer["K"], layer["N"]
    ws_buf = marlin_make_workspace_new(dev)
    gx = torch.Generator(device="cpu").manual_seed(seed)
    X = (torch.randn(n_act, K, generator=gx, dtype=torch.float32) * act_rms).to(dev, torch.bfloat16)

    out = {}
    for M in ms:
        if M == 1:
            continue
        flips = []
        max_d = 0.0
        gaps_all = []
        for b in range(n_act):
            xr = X[b]
            o1 = call(layer, xr, 1, ws_buf)[0].float()
            rM = call(layer, xr, M, ws_buf)[0].float()
            max_d = max(max_d, (rM - o1).abs().max().item())
            top2 = torch.topk(o1, 2).values
            gap = (top2[0] - top2[1]).item()  # nat gap (logit == log-prob diff)
            gaps_all.append(gap)
            t1 = int(o1.argmax()); tM = int(rM.argmax())
            if t1 != tM:
                gM = torch.topk(rM, 2).values
                flips.append({
                    "act": b, "gap_m1_nat": gap,
                    "gap_mM_nat": (gM[0] - gM[1]).item(),
                    "tok_m1": t1, "tok_mM": tM,
                    "delta_logit_at_flip": (rM[t1] - o1[t1]).abs().item(),
                })
        gaps_t = torch.tensor(gaps_all)
        out[M] = {
            "n_act": n_act,
            "n_flips": len(flips),
            "flip_rate": len(flips) / n_act,
            "max_abs_diff_vs_m1_nat": max_d,
            "max_gap_among_flips_nat": max((f["gap_m1_nat"] for f in flips), default=None),
            "min_gap_overall_nat": gaps_t.min().item(),
            "median_gap_overall_nat": gaps_t.median().item(),
            "all_flips_within_2maxd": all(f["gap_m1_nat"] <= 2 * max_d for f in flips),
            "flips": flips[:20],
        }
    return {"name": "lm_head.tie_flips", "size_k": K, "size_n": N, "act_rms": act_rms, "per_m": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ks", default="5,6", help="comma list of K (num_spec); verify M=K+1")
    ap.add_argument("--n-act", type=int, default=256, help="distinct activations for lm_head tie-flip stats")
    ap.add_argument("--act-rms", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=HERE / "locus_real_report.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required (set CUDA_VISIBLE_DEVICES=0)"
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    verify_ms = sorted({k + 1 for k in ks})
    ms = [1] + verify_ms

    print(f"[locus] device={torch.cuda.get_device_name(dev)} cap={torch.cuda.get_device_capability(dev)}", flush=True)
    print(f"[locus] Ks={ks} -> verify M=K+1 in {verify_ms}", flush=True)

    results = []
    for name, prefix in LAYERS:
        print(f"[locus] === {name} ({prefix}) ===", flush=True)
        r = locus_for_layer(name, prefix, dev, ms, args.act_rms, seed=1234 + len(results))
        results.append(r)
        f = r["faithfulness"]
        if f:
            print(f"[locus]   FAITHFUL: median_relerr_vs_fp32dequant={f['median_relerr_vs_fp32dequant']:.3e} "
                  f"rel_l2={f['rel_l2_vs_fp32dequant']:.3e} (expect ~1e-2 int4 floor, NOT 1e37)", flush=True)
        print(f"[locus]   m1_run2run_bitexact={r['m1_run2run_bitexact']} use_atomic_add={r['use_atomic_add']}", flush=True)
        for M, lm in r["per_m"].items():
            print(f"[locus]   M={M}: bitdiff_vs_M1={lm['n_bitdiff_vs_m1']}/{lm['n_elem']} "
                  f"({lm['frac_bitdiff_vs_m1']*100:.2f}%) max|d|={lm['max_abs_diff_vs_m1']:.3e} "
                  f"rel={lm['rel_max_diff_vs_m1']:.2e} row0_det={lm['row0_run2run_bitexact']} "
                  f"cross_row_max={lm['cross_row_max_abs']:.3e}", flush=True)

    print(f"[locus] === lm_head tie-flip stats (n_act={args.n_act}) ===", flush=True)
    tf = lmhead_tie_flips(dev, ms, args.n_act, args.act_rms, seed=9999)
    results.append(tf)
    for M, s in tf["per_m"].items():
        print(f"[locus]   M={M}: flips={s['n_flips']}/{s['n_act']} ({s['flip_rate']*100:.2f}%) "
              f"max|d|={s['max_abs_diff_vs_m1_nat']:.3e}nat max_gap_among_flips="
              f"{s['max_gap_among_flips_nat']} all_within_2maxd={s['all_flips_within_2maxd']}", flush=True)

    report = {
        "pr": 736, "analysis_only": True, "official_tps": 0,
        "device": torch.cuda.get_device_name(dev),
        "capability": list(torch.cuda.get_device_capability(dev)),
        "model": MODEL, "wtype": "uint4b8", "group_size": GROUP,
        "ks": ks, "verify_ms": verify_ms,
        "use_fp32_reduce": USE_FP32_REDUCE_DEFAULT,
        "faithful_repack": "ops.gptq_marlin_repack + marlin_permute_scales (production path)",
        "results": results,
    }
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"[locus] report -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
