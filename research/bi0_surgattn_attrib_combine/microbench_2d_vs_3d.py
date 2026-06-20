"""PR #794 microbench: 2D vs 3D split-KV M=1 attention — numerics + timing.

Controlled kernel isolation. Builds ONE fixed set of M=1 decode inputs per
(layer_type, ctx) and runs the REAL vLLM 0.22.0 `unified_attention` three ways:

  - mode '2d'       : softmax_segm_* = None  -> use_3d=False (bi0's force-2D path,
                      the byte-identical control).
  - mode '3d_fp32'  : fp32 split-KV scratch buffers (the SHIPPED vLLM allocation
                      from triton_attn.py:185-204) -> the surgattn-OFF +6.69% path.
  - mode '3d_bf16'  : same but segm_output buffer is bf16 -> simulates the
                      "bf16 partial accumulation" the PR's config (a) assumed was
                      the status quo. If this diverges MORE than 3d_fp32, fp32 is
                      already the best accumulator => config (a) cannot help.

Outputs captured at both out-dtype bf16 (what the model actually consumes) and
out-dtype fp32 (diagnostic: the underlying reassociation magnitude pre-cast).
Compared against a dense fp32 SDPA reference (ground truth) to show neither 2D
nor 3D is "more correct" — the 3D-vs-2D gap is pure reassociation noise.

LOCAL ONLY. Run with the serve venv (vllm022); logs JSON to be picked up by a
separate uv-env W&B step.

    CUDA_VISIBLE_DEVICES=0 .venvs/vllm022/bin/python \
        research/bi0_surgattn_attrib_combine/microbench_2d_vs_3d.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import vllm.envs as envs  # noqa: E402
from scripts.local_validation.profile_attention import (  # noqa: E402
    BLOCK_SIZE,
    HEAD_DIM,
    LAYER_TYPES,
    N_KV_HEADS,
    N_Q_HEADS,
    SLIDING_WINDOW,
    _build_paged_kv,
    _sdpa_reference,
    bench_op,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention  # noqa: E402

SEQ_THRESHOLD_3D = 128 // N_KV_HEADS  # 64, matches the backend allocation
N_SEG = 16  # NUM_PAR_SOFTMAX_SEGMENTS in the backend
DEVICE = torch.device("cuda")


def _segm_buffers(hd: int, out_dtype: torch.dtype):
    """Allocate the three split-KV scratch buffers. segm_output dtype = out_dtype
    knob (fp32 = shipped; bf16 = config-(a) simulation)."""
    segm_out = torch.empty(
        SEQ_THRESHOLD_3D, N_Q_HEADS, N_SEG, hd, dtype=out_dtype, device=DEVICE
    )
    segm_max = torch.empty(
        SEQ_THRESHOLD_3D, N_Q_HEADS, N_SEG, dtype=torch.float32, device=DEVICE
    )
    segm_exp = torch.empty(
        SEQ_THRESHOLD_3D, N_Q_HEADS, N_SEG, dtype=torch.float32, device=DEVICE
    )
    return segm_out, segm_max, segm_exp


def run_kernel(q, kc, vc, bt, ctx, layer_type, scale, window, *, mode, out_dtype):
    hd = HEAD_DIM[layer_type]
    out = torch.empty(1, N_Q_HEADS, hd, dtype=out_dtype, device=DEVICE)
    cu = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    seqused = torch.tensor([ctx], dtype=torch.int32, device=DEVICE)

    if mode == "2d":
        thr = None
        segm_out = segm_max = segm_exp = None
    else:
        thr = SEQ_THRESHOLD_3D
        segm_dtype = torch.bfloat16 if mode == "3d_bf16" else torch.float32
        segm_out, segm_max, segm_exp = _segm_buffers(hd, segm_dtype)

    unified_attention(
        q=q, k=kc, v=vc, out=out,
        cu_seqlens_q=cu, max_seqlen_q=1,
        seqused_k=seqused, max_seqlen_k=ctx,
        softmax_scale=scale, causal=True, window_size=window,
        block_table=bt, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
        seq_threshold_3D=thr, num_par_softmax_segments=N_SEG,
        softmax_segm_output=segm_out, softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_exp,
    )
    torch.cuda.synchronize()
    return out


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a32, b32 = a.float(), b.float()
    d = (a32 - b32).abs()
    denom = b32.abs().clamp_min(1e-12)
    rel = (d / denom)
    n = a.numel()
    n_diff = int((a32 != b32).sum().item())
    return {
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "max_rel": rel.max().item(),
        "n_elems": n,
        "n_differ": n_diff,
        "frac_differ": n_diff / n,
        "ref_abs_mean": b32.abs().mean().item(),
    }


def numerics_for(layer_type: str, ctx: int) -> dict:
    """Build identical inputs once; run 2D / 3D-fp32 / 3D-bf16 and compare."""
    torch.manual_seed(0)
    hd = HEAD_DIM[layer_type]
    scale = 1.0 / math.sqrt(hd)
    window = (SLIDING_WINDOW - 1, 0) if layer_type == "sliding" else (-1, -1)
    kc, vc, block_tables, nb = _build_paged_kv(torch, DEVICE, layer_type, ctx, rot=1)
    bt = block_tables[0]
    q = torch.randn(1, N_Q_HEADS, hd, dtype=torch.bfloat16, device=DEVICE) * 0.1

    res = {"layer_type": layer_type, "ctx": ctx, "head_dim": hd}
    for od_name, od in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        out2d = run_kernel(q, kc, vc, bt, ctx, layer_type, scale, window,
                           mode="2d", out_dtype=od)
        out3d = run_kernel(q, kc, vc, bt, ctx, layer_type, scale, window,
                           mode="3d_fp32", out_dtype=od)
        res[f"d_2d_vs_3dfp32_{od_name}"] = diff_stats(out3d, out2d)
        if od_name == "bf16":
            out3d_bf16 = run_kernel(q, kc, vc, bt, ctx, layer_type, scale, window,
                                    mode="3d_bf16", out_dtype=od)
            res["d_2d_vs_3dbf16_bf16"] = diff_stats(out3d_bf16, out2d)
            # ground-truth fp32 SDPA: which arm is closer?
            ref = _sdpa_reference(torch, q, kc, vc, bt, ctx, 1, layer_type, scale)
            res["err_2d_vs_sdpa"] = diff_stats(out2d, ref)
            res["err_3dfp32_vs_sdpa"] = diff_stats(out3d, ref)
    del kc, vc
    torch.cuda.empty_cache()
    return res


def timing_for(layer_type: str, ctx: int, n_iter: int = 200) -> dict:
    """Device-time 2D vs 3D at M=1 via the validated bench_op (L2-defeating)."""
    b2d = bench_op(torch, layer_type, M=1, ctx=ctx, dispatch="force2d", n_iter=n_iter)
    b3d = bench_op(torch, layer_type, M=1, ctx=ctx, dispatch="served", n_iter=n_iter)
    speedup = b2d["device_us"] / b3d["device_us"] if b3d["device_us"] else float("nan")
    return {
        "layer_type": layer_type, "ctx": ctx,
        "us_2d": b2d["device_us"], "us_3d": b3d["device_us"],
        "used_3d": b3d["used_3d_split_kv"],
        "speedup_2d_over_3d": speedup,
        "gbps_2d": b2d["achieved_gbps_kv_floor"],
        "gbps_3d": b3d["achieved_gbps_kv_floor"],
    }


def main():
    cc = torch.cuda.get_device_capability()
    print(f"[microbench] device={torch.cuda.get_device_name()} cc={cc} "
          f"VLLM_BATCH_INVARIANT={envs.VLLM_BATCH_INVARIANT}", flush=True)
    ctxs = [512, 1024, 2048, 4096]
    layer_types = ["sliding", "full"]

    numerics, timing = [], []
    for lt in layer_types:
        for ctx in ctxs:
            n = numerics_for(lt, ctx)
            numerics.append(n)
            t = timing_for(lt, ctx)
            timing.append(t)
            d_fp32 = n["d_2d_vs_3dfp32_bf16"]
            d_bf16 = n["d_2d_vs_3dbf16_bf16"]
            print(
                f"[{lt:7s} ctx={ctx:5d}] "
                f"3Dfp32-vs-2D: max_abs={d_fp32['max_abs']:.2e} "
                f"frac_differ={d_fp32['frac_differ']:.4f} | "
                f"3Dbf16-vs-2D: max_abs={d_bf16['max_abs']:.2e} "
                f"frac_differ={d_bf16['frac_differ']:.4f} | "
                f"t2d={t['us_2d']:.1f}us t3d={t['us_3d']:.1f}us "
                f"sp={t['speedup_2d_over_3d']:.3f}x",
                flush=True,
            )

    # layer-weighted projected attention-time speedup (30 sliding + 7 full)
    n_sliding = sum(1 for x in LAYER_TYPES if x == "sliding")
    n_full = sum(1 for x in LAYER_TYPES if x == "full")
    out = {
        "device": torch.cuda.get_device_name(),
        "vllm_batch_invariant": envs.VLLM_BATCH_INVARIANT,
        "seq_threshold_3D": SEQ_THRESHOLD_3D,
        "num_par_softmax_segments": N_SEG,
        "n_sliding_layers": n_sliding, "n_full_layers": n_full,
        "numerics": numerics,
        "timing": timing,
    }
    outpath = Path(__file__).resolve().parent / "microbench_results.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"[microbench] wrote {outpath}", flush=True)


if __name__ == "__main__":
    main()
