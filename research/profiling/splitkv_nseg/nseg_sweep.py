"""Split-KV segment-count (num_par_softmax_segments) sweep at the deployed M=8
spec-verify shape — the one un-audited lever PR #69 names ("low occupancy at
M=8 / context-suboptimal split heuristic a la #53").

Post-#43 the M=8 verify is redirected to vLLM's 3D split-KV (FlashDecoding)
kernel.  The deployed split count is fixed at NUM_PAR_SOFTMAX_SEGMENTS=16
(vllm/v1/attention/backends/triton_attn.py).  The 3D launch grid is
(total_num_q_blocks, num_kv_heads, num_par_softmax_segments); at M=8 that is
(3, 2, n_seg) = 6*n_seg CTAs on the A10G's 80 SMs.

Question: at the *deployed* M=8 + decode ctx range, does a different n_seg lower
total attention device time (attention kernel + reduce_segments merge)?  If the
optimum is ~16 (or more segments only add reduce_segments cost), the residual
sub-peak bandwidth is the irreducible conc=1 small-read latency floor and the
split heuristic carries no lossless headroom -> terminal NEGATIVE.  If a
materially faster n_seg exists at the served shapes, that is a candidate
lossless fix to prototype.

device time is read with torch.profiler self_device_time (the same instrument as
the served 19.6%/7.6% figures); reduce_segments is broken out separately so the
merge/attention tradeoff is explicit.  KV cache is rotated to defeat the 6 MB L2.

Local A10G op-microbench — no server, no leaderboard number.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.local_validation.profile_attention import (  # noqa: E402
    BLOCK_SIZE,
    DTYPE_BYTES,
    HEAD_DIM,
    N_KV_HEADS,
    N_Q_HEADS,
    SLIDING_WINDOW,
    _build_paged_kv,
    _measure_peak_bw,
    _op_bytes,
    _rot_for,
    _sdpa_reference,
)

A10G_PEAK_GBPS = 600.0
SEQ_THRESHOLD_3D = 128 // N_KV_HEADS  # = 64, matches the deployed backend
DEPLOYED_NSEG = 16


def _per_kernel_device_us(torch, fn, n_iter: int, warmup: int = 20) -> dict:
    """Return per-kernel mean self-device-us over n_iter calls, keyed by a
    coarse kernel class (attention / reduce_segments / other), plus the total."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
    attn_us = redseg_us = other_us = 0.0
    for e in prof.key_averages():
        us = e.self_device_time_total / n_iter
        if us <= 0:
            continue
        name = e.key.lower()
        if "reduce_segment" in name:
            redseg_us += us
        elif "unified_attention" in name or "attention" in name:
            attn_us += us
        else:
            other_us += us
    return {
        "total_us": attn_us + redseg_us + other_us,
        "attn_kernel_us": attn_us,
        "reduce_segments_us": redseg_us,
        "other_us": other_us,
    }


def bench_nseg(torch, layer_type: str, M: int, ctx: int, n_seg: int,
               *, n_iter: int = 200, validate: bool = False) -> dict:
    """Drive the real vLLM 3D split-KV kernel at a chosen num_par_softmax_segments.

    Forces the 3D path the way the deployed #43 patch does: pass max_seqlen_q=1
    while q carries M real rows (q.shape[0]=M)."""
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    device = torch.device("cuda")
    hd = HEAD_DIM[layer_type]
    scale = 1.0 / math.sqrt(hd)
    window = (SLIDING_WINDOW - 1, 0) if layer_type == "sliding" else (-1, -1)

    rot = _rot_for(layer_type, ctx)
    key_cache, value_cache, block_tables, nb = _build_paged_kv(
        torch, device, layer_type, ctx, rot)
    q = torch.randn(M, N_Q_HEADS, hd, dtype=torch.bfloat16, device=device) * 0.1
    out = torch.empty(M, N_Q_HEADS, hd, dtype=torch.bfloat16, device=device)
    cu_seqlens_q = torch.tensor([0, M], dtype=torch.int32, device=device)
    seqused_k = torch.tensor([ctx], dtype=torch.int32, device=device)

    hdp = hd  # head dims here are already pow2 (256 / 512)
    segm_out = torch.empty(SEQ_THRESHOLD_3D, N_Q_HEADS, n_seg, hdp,
                           dtype=torch.float32, device=device)
    segm_max = torch.empty(SEQ_THRESHOLD_3D, N_Q_HEADS, n_seg,
                           dtype=torch.float32, device=device)
    segm_exp = torch.empty(SEQ_THRESHOLD_3D, N_Q_HEADS, n_seg,
                           dtype=torch.float32, device=device)

    state = {"i": 0}

    def call():
        bt = block_tables[state["i"] % rot]
        state["i"] += 1
        unified_attention(
            q=q, k=key_cache, v=value_cache, out=out,
            cu_seqlens_q=cu_seqlens_q, max_seqlen_q=1,  # <- forces 3D, #43-style
            seqused_k=seqused_k, max_seqlen_k=ctx,
            softmax_scale=scale, causal=True, window_size=window,
            block_table=bt, softcap=0.0,
            q_descale=None, k_descale=None, v_descale=None,
            seq_threshold_3D=SEQ_THRESHOLD_3D, num_par_softmax_segments=n_seg,
            softmax_segm_output=segm_out, softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_exp,
        )

    err = None
    if validate:
        state["i"] = 0
        call()
        torch.cuda.synchronize()
        ref = _sdpa_reference(torch, q, key_cache, value_cache,
                              block_tables[0], ctx, M, layer_type, scale)
        err = {
            "max_abs_err": (out - ref).abs().max().item(),
            "mean_abs_err": (out - ref).abs().mean().item(),
            "ref_abs_mean": ref.abs().mean().item(),
        }

    state["i"] = 0
    kt = _per_kernel_device_us(torch, call, n_iter)

    # launch grid (matches the kernel: (total_num_q_blocks, num_kv_heads, n_seg))
    num_queries_per_kv = N_Q_HEADS // N_KV_HEADS
    block_m = 16 if num_queries_per_kv <= 16 else 1 << (num_queries_per_kv - 1).bit_length()
    block_q = max(1, block_m // num_queries_per_kv)
    total_num_q_blocks = q.shape[0] // block_q + 1  # num_seqs = 1
    nominal_ctas = total_num_q_blocks * N_KV_HEADS * n_seg

    b = _op_bytes(layer_type, ctx, M)
    tot_us = kt["total_us"]
    gbps_total = b["total_raw_bytes"] / (tot_us / 1e6) / 1e9
    gbps_floor = b["kv_floor_bytes"] / (tot_us / 1e6) / 1e9
    del key_cache, value_cache
    torch.cuda.empty_cache()
    return {
        "layer_type": layer_type, "M": M, "ctx": ctx, "n_seg": n_seg,
        "rot_buffers": rot, "nominal_ctas": nominal_ctas,
        "total_us": tot_us, "attn_kernel_us": kt["attn_kernel_us"],
        "reduce_segments_us": kt["reduce_segments_us"], "other_us": kt["other_us"],
        "achieved_gbps_total": gbps_total, "achieved_gbps_kv_floor": gbps_floor,
        "validation": err,
    }


def run(out_path: Path, *, ctx_values, n_seg_values, n_iter: int = 200) -> dict:
    import torch
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    t0 = time.time()
    props = torch.cuda.get_device_properties(0)
    result = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "sm_count": props.multi_processor_count,
        "l2_bytes": props.L2_cache_size,
        "deployed_nseg": DEPLOYED_NSEG,
        "seq_threshold_3D": SEQ_THRESHOLD_3D,
        "M": 8, "ctx_values": list(ctx_values), "n_seg_values": list(n_seg_values),
    }
    pk = _measure_peak_bw(torch, device)
    peak = pk["measured_peak_gbps_copy"]
    result["peak_bw"] = pk
    print(f"[nseg] GPU {result['gpu']} SMs={result['sm_count']} "
          f"measured_peak={peak:.1f} GB/s spec_peak={A10G_PEAK_GBPS}", flush=True)

    sweeps = {}
    for lt in ("sliding", "full"):
        for ctx in ctx_values:
            rows = []
            for ns in n_seg_values:
                r = bench_nseg(torch, lt, 8, ctx, ns, n_iter=n_iter,
                               validate=(ns in (DEPLOYED_NSEG, n_seg_values[0],
                                                n_seg_values[-1])))
                r["peak_eff_total"] = r["achieved_gbps_total"] / peak
                rows.append(r)
                v = r["validation"]
                print(f"   {lt:8s} ctx={ctx:<5d} nseg={ns:<3d} "
                      f"CTAs={r['nominal_ctas']:<4d} tot={r['total_us']:6.1f}us "
                      f"(attn={r['attn_kernel_us']:5.1f} red={r['reduce_segments_us']:5.1f}) "
                      f"{r['achieved_gbps_total']:6.1f} GB/s "
                      f"({r['peak_eff_total']*100:4.1f}% peak)"
                      + (f" err={v['max_abs_err']:.2e}" if v else ""), flush=True)
            best = min(rows, key=lambda x: x["total_us"])
            dep = next(x for x in rows if x["n_seg"] == DEPLOYED_NSEG)
            sweeps[f"{lt}_ctx{ctx}"] = {
                "rows": rows,
                "deployed_nseg_us": dep["total_us"],
                "best_nseg": best["n_seg"], "best_us": best["total_us"],
                "speedup_best_vs_deployed": dep["total_us"] / best["total_us"],
            }
            print(f"   -> {lt} ctx={ctx}: deployed n=16 {dep['total_us']:.1f}us | "
                  f"best n={best['n_seg']} {best['total_us']:.1f}us | "
                  f"speedup {dep['total_us']/best['total_us']:.3f}x", flush=True)
    result["sweeps"] = sweeps

    # Aggregate verdict: cycle-weighted best-vs-deployed over the served layer
    # mix (30 sliding + 7 full) at the mean decode ctx (~528). Conservative: use
    # ctx=512 row. Aggregate attention us/cycle at deployed vs best-per-shape.
    def agg_at(ctx, key):
        s = sweeps[f"sliding_ctx{ctx}"]
        f = sweeps[f"full_ctx{ctx}"]
        sdep = next(x for x in s["rows"] if x["n_seg"] == DEPLOYED_NSEG)["total_us"]
        fdep = next(x for x in f["rows"] if x["n_seg"] == DEPLOYED_NSEG)["total_us"]
        sbest = s["best_us"]; fbest = f["best_us"]
        dep_cycle = 30 * sdep + 7 * fdep
        best_cycle = 30 * sbest + 7 * fbest
        return {"ctx": ctx, "deployed_us_per_cycle": dep_cycle,
                "best_us_per_cycle": best_cycle,
                "speedup": dep_cycle / best_cycle,
                "saving_frac": 1.0 - best_cycle / dep_cycle}
    if 512 in ctx_values:
        result["aggregate_ctx512"] = agg_at(512, "agg")
        a = result["aggregate_ctx512"]
        print(f"[nseg] AGG ctx512: deployed {a['deployed_us_per_cycle']:.1f}us/cyc "
              f"best {a['best_us_per_cycle']:.1f}us/cyc "
              f"saving {a['saving_frac']*100:.1f}%", flush=True)

    result["elapsed_s"] = time.time() - t0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"[nseg] wrote {out_path} ({result['elapsed_s']:.0f}s)", flush=True)
    return result


if __name__ == "__main__":
    run(
        Path(__file__).with_name("nseg_sweep.json"),
        ctx_values=(128, 256, 512, 1024),
        n_seg_values=(1, 2, 4, 8, 16, 32, 64),  # reduce_segments requires pow2
        n_iter=200,
    )
