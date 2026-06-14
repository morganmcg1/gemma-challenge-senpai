#!/usr/bin/env python
"""SplitK vs single-K W4A16 verify-GEMM micro-bench (PR #108, Phase-1).

WHAT THIS TESTS
---------------
denken #117 (roofline, RED) claims the int4-Marlin verify GEMM has almost no
SplitK headroom at the served verify width M=8 because the dominant GEMM
`mlp.gate_up_proj` (54% of verify time) is *CTA-saturated*: N=20480 with a 128-wide
N-tile launches 160 CTAs = 2 full waves on the A10G's 80 SMs, so splitting the
K-reduction across more CTAs adds reduction overhead with ~0 extra bandwidth.
The occupancy-limited laggards (`down_proj` N=2560 -> 20 CTAs; attention
o/qkv -> 20/24/48 CTAs) should instead be SplitK-liftable, but only up to the
in-situ GDDR6 DRAM-efficiency wall (~79-88%).

This is an isolated Triton A/B: the SAME int4 W4A16 GEMM run single-K (SPLIT_K=1)
vs SplitK (SPLIT_K in {2,4,8}) on the 6 architecture-determined verify shapes from
denken #68's roofline (research/spec_cost_model/verify_gemm_roofline.json), swept
at M in {8,16,32}. It isolates the SplitK *mechanism's* headroom so we can
confirm/refute the #117 ceiling BEFORE any served kernel swap.

It is NOT a Marlin-replacement speedup: the single-K baseline is a stock Triton
W4A16 GEMM, not the hand-tuned Marlin kernel (Marlin is faster than a stock Triton
baseline). The decision-relevant quantity is the RELATIVE single-K -> SplitK lift
per shape and whether gate_up's saturation kills it, which is a property of the
A10G wave scheduler and therefore transfers across kernels.

NUMERICS (kanna #114: "greedy-safe by construction" is dead): SplitK reorders the
K-reduction across CTAs vs the single-K sequential sum. We keep **fp32 partials**
(atomic_add into a zeroed fp32 buffer; wirbel #98 says fp32 partials are ~free via
A10G L2 residency) and report max-abs-err + greedy argmax-flip rate of SplitK vs
single-K on a fixed input (`splitk_bit_identical` / `splitk_greedy_identical`).

Bandwidth accounting reuses denken #68's exact byte model (4-bit weight + fp16
group scales + bf16 act + bf16 out) so achieved GB/s and %HBM peak are directly
comparable to the roofline JSON.

LOCAL micro-bench only: no vLLM, no served-file change, no HF Job. Lossless by
construction (never touches the emitted token stream).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time

# One A10G is exposed as device 0, but the container inherits CUDA_VISIBLE_DEVICES
# pointing at a non-existent host index -> torch sees 0 devices. Pin to 0.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import triton
import triton.language as tl

A10G_HBM_GBS = 600.0
A10G_SMS = 80
WEIGHT_BITS = 4
SCALE_BYTES = 2  # fp16/bf16 group scales

# 6 unique verify-GEMM shapes (role, K=in, N=out, per-step count) from denken #68.
SHAPES = [
    ("attn.qkv_proj.main", 2560, 3072, 35),
    ("attn.o_proj.main", 2048, 2560, 35),
    ("mlp.gate_up_proj", 2560, 20480, 42),
    ("mlp.down_proj", 10240, 2560, 42),
    ("attn.qkv_proj.kv", 2560, 6144, 7),
    ("attn.o_proj.kv", 4096, 2560, 7),
]


@triton.jit
def _w4a16_splitk_kernel(
    x_ptr, wq_ptr, s_ptr, o_ptr,
    M, N, K, G,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_sk, stride_sn,
    stride_om, stride_on,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    SPLIT_K: tl.constexpr, ATOMIC: tl.constexpr,
):
    # BK assumed a multiple of 8 and <= group_size G; each K-tile lies in ONE group.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    BKP: tl.constexpr = BK // 8
    shifts = (tl.arange(0, 8) * 4).to(tl.int32)           # [8] nibble shifts
    k_per_split = K // SPLIT_K
    k_start = pid_k * k_per_split
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, k_per_split, BK):
        k_base = k_start + k0
        offk = k_base + tl.arange(0, BK)                  # logical K indices [BK]
        offkp = (k_base // 8) + tl.arange(0, BKP)         # compact packed rows [BKP]
        x = tl.load(x_ptr + offm[:, None] * stride_xm + offk[None, :] * stride_xk,
                    mask=(offm[:, None] < M) & (offk[None, :] < K), other=0.0)
        wq = tl.load(wq_ptr + offkp[:, None] * stride_wk + offn[None, :] * stride_wn,
                     mask=offn[None, :] < N, other=0)     # [BKP, BN] int32 (compact)
        # unpack 8 nibbles along K: [BKP,1,BN] >> [1,8,1] -> [BKP,8,BN] -> [BK,BN]
        w3 = (wq[:, None, :] >> shifts[None, :, None]) & 0xF
        w = tl.reshape(w3, (BK, BN)).to(tl.float32) - 8.0
        s_row = tl.load(s_ptr + (k_base // G) * stride_sk + offn * stride_sn,
                        mask=offn < N, other=0.0)         # [BN] one group row
        w = (w * s_row[None, :]).to(tl.bfloat16)
        acc += tl.dot(x.to(tl.bfloat16), w)
    out_off = o_ptr + offm[:, None] * stride_om + offn[None, :] * stride_on
    mask = (offm[:, None] < M) & (offn[None, :] < N)
    if ATOMIC:
        tl.atomic_add(out_off, acc, mask=mask)
    else:
        tl.store(out_off, acc, mask=mask)


def make_weights(K, N, group, device, seed=0):
    """Random symmetric uint4b8 weights packed 8-along-K into int32, + fp16 scales."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randint(0, 16, (K, N), generator=g, dtype=torch.int32)  # uint4 [0,15]
    # pack 8 consecutive-K nibbles -> int32 [K//8, N]
    qp = q.view(K // 8, 8, N)
    shifts = (torch.arange(8, dtype=torch.int32) * 4).view(1, 8, 1)
    packed = (qp << shifts).sum(dim=1).to(torch.int32).to(device)         # [K//8, N]
    scales = (torch.rand(K // group, N, generator=g) * 0.02 + 0.005).to(torch.bfloat16).to(device)
    return q.to(device), packed, scales


def dequant_ref(q, scales, group):
    K, N = q.shape
    s = scales.to(torch.float32).repeat_interleave(group, dim=0)[:K]      # [K,N]
    return (q.to(torch.float32) - 8.0) * s


def launch(x, packed, scales, K, N, G, BM, BN, BK, split_k, out=None, atomic=None,
           num_warps=4, num_stages=4):
    M = x.shape[0]
    if atomic is None:
        atomic = split_k > 1
    if out is None:
        out = torch.zeros(M, N, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), split_k)
    _w4a16_splitk_kernel[grid](
        x, packed, scales, out,
        M, N, K, G,
        x.stride(0), x.stride(1),
        packed.stride(0), packed.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        BM=BM, BN=BN, BK=BK, SPLIT_K=split_k, ATOMIC=atomic,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def time_graph(fn, iters, warmup):
    """Launch-free CUDA-graph replay timing (matches denken #68 methodology)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(max(10, warmup)):
        g.replay()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    # median over several timed windows of `iters` replays each
    samples = []
    for _ in range(7):
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        samples.append(e0.elapsed_time(e1) / iters)
    del g
    return statistics.median(samples)


def bytes_model(M, K, N, group):
    w_bytes = (WEIGHT_BITS / 8.0) * N * K + SCALE_BYTES * N * math.ceil(K / group)
    act_bytes = 2.0 * M * K
    out_bytes = 2.0 * M * N
    return w_bytes + act_bytes + out_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-list", default="8,16,32")
    ap.add_argument("--split-k", default="1,2,4,8")
    ap.add_argument("--group-size", type=int, default=128,
                    help="deployed osoi5-v0-baked is g128 (wirbel #104); denken #68 used g32 for the HF ckpt")
    ap.add_argument("--bm", type=int, default=16)
    ap.add_argument("--bn", type=int, default=128)
    ap.add_argument("--bk", type=int, default=64)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--shapes", default="all", help="comma idx subset or 'all'")
    ap.add_argument("--output", default="research/spec_cost_model/splitk_verify_gemm_microbench.json")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="splitk-verify-gemm")
    ap.add_argument("--wandb_name", default="splitk-verify-gemm-microbench")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    m_list = [int(x) for x in args.m_list.split(",") if x.strip()]
    split_ks = [int(x) for x in args.split_k.split(",") if x.strip()]
    G = args.group_size
    BM, BN, BK = args.bm, args.bn, args.bk
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"[splitk] device={props.name} SMs={props.multi_processor_count} "
          f"torch={torch.__version__} triton={triton.__version__}", flush=True)
    print(f"[splitk] BM={BM} BN={BN} BK={BK} group={G} M={m_list} split_k={split_ks}", flush=True)

    shape_idx = range(len(SHAPES)) if args.shapes == "all" else [int(i) for i in args.shapes.split(",")]

    rows = []
    numerics = []
    for si in shape_idx:
        role, K, N, count = SHAPES[si]
        q, packed, scales = make_weights(K, N, G, dev)
        ref_dq = dequant_ref(q, scales, G)  # [K,N] fp32, for correctness
        for M in m_list:
            torch.manual_seed(100 + M)
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            ref = (x.to(torch.float32) @ ref_dq)  # fp32 reference (bf16 inputs)
            base_t = None
            base_out = None
            for sk in split_ks:
                # correctness: clean zeroed run
                out = torch.zeros(M, N, device=dev, dtype=torch.float32)
                launch(x, packed, scales, K, N, G, BM, BN, BK, sk, out=out, atomic=(sk > 1))
                torch.cuda.synchronize()
                err_vs_ref = (out - ref).abs().max().item()
                relerr_vs_ref = err_vs_ref / (ref.abs().max().item() + 1e-9)
                if sk == split_ks[0]:
                    base_out = out.clone()
                    splitk_max_abs_err = 0.0
                    argmax_flip = 0
                else:
                    splitk_max_abs_err = (out - base_out).abs().max().item()
                    argmax_flip = int((out.argmax(-1) != base_out.argmax(-1)).sum().item())
                # timing
                persistent = torch.zeros(M, N, device=dev, dtype=torch.float32)
                t_ms = time_graph(
                    lambda sk=sk, persistent=persistent: launch(
                        x, packed, scales, K, N, G, BM, BN, BK, sk,
                        out=persistent, atomic=(sk > 1)),
                    args.iters, args.warmup)
                t_us = t_ms * 1000.0
                tot_bytes = bytes_model(M, K, N, G)
                gbytes_s = tot_bytes / (t_ms / 1000.0) / 1e9
                ctas = triton.cdiv(M, BM) * triton.cdiv(N, BN) * sk
                if sk == split_ks[0]:
                    base_t = t_us
                speedup = 100.0 * (base_t - t_us) / base_t if base_t else 0.0
                row = {
                    "role": role, "K": K, "N": N, "count": count, "M": M, "split_k": sk,
                    "t_us": t_us, "gbytes_s": gbytes_s,
                    "pct_hbm_peak": 100.0 * gbytes_s / A10G_HBM_GBS,
                    "ctas": ctas, "waves": ctas / A10G_SMS,
                    "speedup_vs_split1_pct": speedup,
                    "relerr_vs_fp32ref": relerr_vs_ref,
                    "splitk_max_abs_err_vs_split1": splitk_max_abs_err,
                    "splitk_argmax_flips_vs_split1": argmax_flip,
                }
                rows.append(row)
                print(f"[splitk] {role:>20s} K={K:5d} N={N:6d} M={M:2d} sk={sk} | "
                      f"{t_us:7.1f}us {gbytes_s:5.0f}GB/s ({row['pct_hbm_peak']:4.1f}%BW) "
                      f"CTAs={ctas:4d}({row['waves']:.2f}w) "
                      f"spd={speedup:+5.1f}% relerr={relerr_vs_ref:.1e} "
                      f"flips={argmax_flip}", flush=True)

    # ---- aggregate verify-GEMM speedup, weighted by per-step time (count*t_us @ split1) ----
    agg = {}
    for M in m_list:
        base1 = {(r["role"], r["K"], r["N"]): r for r in rows if r["M"] == M and r["split_k"] == 1}
        total_base = sum(r["t_us"] * r["count"] for r in base1.values())
        best_total = None
        best_by_shape_total = 0.0
        for r in base1.values():
            shape_rows = [x for x in rows if x["M"] == M and x["role"] == r["role"]
                          and x["K"] == r["K"] and x["N"] == r["N"]]
            best = min(shape_rows, key=lambda z: z["t_us"])  # best split per shape
            best_by_shape_total += best["t_us"] * r["count"]
        # also: single global best split_k (same split applied to all shapes)
        per_sk_total = {}
        for sk in split_ks:
            per_sk_total[sk] = sum(
                next(x for x in rows if x["M"] == M and x["role"] == r["role"]
                     and x["K"] == r["K"] and x["N"] == r["N"] and x["split_k"] == sk)["t_us"] * r["count"]
                for r in base1.values())
        best_global_sk = min(per_sk_total, key=per_sk_total.get)
        agg[M] = {
            "M": M,
            "total_us_split1": total_base,
            "total_us_best_per_shape": best_by_shape_total,
            "speedup_best_per_shape_pct": 100.0 * (total_base - best_by_shape_total) / total_base,
            "per_sk_total_us": per_sk_total,
            "best_global_split_k": best_global_sk,
            "speedup_best_global_pct": 100.0 * (total_base - per_sk_total[best_global_sk]) / total_base,
        }

    m8 = agg.get(8, {})
    primary = m8.get("speedup_best_per_shape_pct", 0.0)
    # gate detail at M=8
    gate_up_m8 = [r for r in rows if r["M"] == 8 and r["role"] == "mlp.gate_up_proj"]
    down_m8 = [r for r in rows if r["M"] == 8 and r["role"] == "mlp.down_proj"]
    gate_up_best = max((r["speedup_vs_split1_pct"] for r in gate_up_m8), default=0.0)
    down_best = max((r["speedup_vs_split1_pct"] for r in down_m8), default=0.0)
    max_flip = max((r["splitk_argmax_flips_vs_split1"] for r in rows), default=0)
    splitk_bit_identical = 1.0 if all(r["splitk_max_abs_err_vs_split1"] == 0.0 for r in rows) else 0.0
    splitk_greedy_identical = 1.0 if max_flip == 0 else 0.0

    verdict = {
        "splitk_verify_gemm_m8_speedup_pct": primary,
        "m8_best_global_split_k": m8.get("best_global_split_k"),
        "m8_speedup_best_global_pct": m8.get("speedup_best_global_pct"),
        "gate_up_m8_best_speedup_pct": gate_up_best,
        "down_proj_m8_best_speedup_pct": down_best,
        "splitk_bit_identical": splitk_bit_identical,
        "splitk_greedy_identical": splitk_greedy_identical,
        "max_argmax_flips": max_flip,
        "gate_pass_10pct": bool(primary >= 10.0),
        "ship_target_5_84pct": bool(primary >= 5.84),
    }
    print("\n[splitk] ===== AGGREGATE (per-step time-weighted) =====", flush=True)
    for M in m_list:
        a = agg[M]
        print(f"  M={M:2d}: split1={a['total_us_split1']:8.1f}us  "
              f"best-per-shape={a['total_us_best_per_shape']:8.1f}us "
              f"({a['speedup_best_per_shape_pct']:+.2f}%)  "
              f"best-global-sk={a['best_global_split_k']} "
              f"({a['speedup_best_global_pct']:+.2f}%)", flush=True)
    print(f"\n[splitk] VERDICT @ M=8: primary speedup = {primary:.2f}% "
          f"(gate>=10%? {verdict['gate_pass_10pct']}; ship>=5.84%? {verdict['ship_target_5_84pct']})", flush=True)
    print(f"[splitk]   gate_up M=8 best SplitK = {gate_up_best:+.2f}% (saturation control)", flush=True)
    print(f"[splitk]   down_proj M=8 best SplitK = {down_best:+.2f}% (occupancy-limited laggard)", flush=True)
    print(f"[splitk]   bit-identical={splitk_bit_identical} greedy-identical={splitk_greedy_identical} "
          f"max_flips={max_flip}", flush=True)

    payload = {
        "config": {
            "device": props.name, "sms": props.multi_processor_count,
            "torch": torch.__version__, "triton": triton.__version__,
            "BM": BM, "BN": BN, "BK": BK, "group_size": G,
            "m_list": m_list, "split_ks": split_ks,
            "iters": args.iters, "warmup": args.warmup,
            "A10G_HBM_GBS": A10G_HBM_GBS,
            "note": "isolated Triton single-K vs SplitK W4A16 A/B; fp32 atomic partials; "
                    "baseline is stock Triton (NOT Marlin) -> relative mechanism headroom, "
                    "not a Marlin-replacement speedup. Shapes from denken #68.",
        },
        "shapes": [{"role": r, "K": k, "N": n, "count": c} for (r, k, n, c) in SHAPES],
        "rows": rows,
        "aggregate_by_M": agg,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[splitk] wrote {args.output}", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[splitk] W&B logging failed: {exc!r}", flush=True)


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="profiling", config=payload["config"])
    cols = ["role", "K", "N", "count", "M", "split_k", "t_us", "gbytes_s",
            "pct_hbm_peak", "ctas", "waves", "speedup_vs_split1_pct",
            "relerr_vs_fp32ref", "splitk_max_abs_err_vs_split1", "splitk_argmax_flips_vs_split1"]
    tbl = wandb.Table(columns=cols)
    for r in payload["rows"]:
        tbl.add_data(*[r[c] for c in cols])
    run.log({"splitk_microbench_table": tbl})
    v = payload["verdict"]
    run.summary.update({k: val for k, val in v.items() if val is not None})
    for M, a in payload["aggregate_by_M"].items():
        run.summary[f"m{M}_speedup_best_per_shape_pct"] = a["speedup_best_per_shape_pct"]
        run.summary[f"m{M}_speedup_best_global_pct"] = a["speedup_best_global_pct"]
    run.finish()
    print(f"[splitk] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
