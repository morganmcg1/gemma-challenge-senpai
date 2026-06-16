"""PR #496 microbench: decompose the -107 TPS byte-exact attention tax and
numerically prove M-invariance of the fixed-chunk split-KV scheme.

Runs IN-PROCESS against the serve-venv vLLM (the gated
``triton_unified_attention.py`` with ``FIXED_TILES_PER_SEGMENT`` honoring env
``BYTEEXACT_FIXED_TPS``). Targets the Gemma-4-E4B GLOBAL full-attention layer
shape (the 7 layers that pay the byte-exact attention tax):
    num_query_heads=8, num_kv_heads=2, head_dim=512, block_size=16, bf16,
    causal, NO sliding window, softcap=0.

Two products:
  (A) attn_tax_decomposition -- kernel-time arms (eager = with launch; cudagraph
      = compute-only, the ONEGRAPH-served regime) for:
        2D                 byte-exact, no KV split          (the surgical 357.6 path)
        3D_seg1            3D codepath, 1 segment           (no split parallelism)
        3D_seg16_adaptive  stock fast deployed path         (NOT byte-exact)
        3D_fixed_*         candidate byte-exact split-KV    (fixed tiles_per_segment)
      Additive split: (t2D - t3D_seg16) = (t2D - t3D_seg1)            [codepath/memory]
                                        + (t3D_seg1 - t3D_seg16)      [split parallelism]
  (B) m_invariance -- at straddle positions, compare M=8 verify row-i bytes
      vs M=1 AR at the same absolute position. Adaptive flips; fixed is 0-flip.

NOT analysis of any deployed file: the kernel edit is gated (default-off =
stock) and restored after. analysis_only / official_tps=0.

Run:
  CUDA_VISIBLE_DEVICES=0 /tmp/senpai-venvs/<hash>/bin/python \
      research/speed/byteexact_attn/microbench_attn_tax.py --out <json>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch

# The serve-venv vLLM unified_attention (gated FIXED_TILES_PER_SEGMENT build).
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

# ---- Gemma-4-E4B GLOBAL full-attention layer shape ----
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 512
BLOCK_SIZE = 16
DTYPE = torch.bfloat16
SEQ_THRESHOLD_3D = 64  # serve value: MIN_LAUNCH_GRID_SIZE_2D(128)//num_kv_heads(2)
SCALE = 1.0 / math.sqrt(HEAD_DIM)
WINDOW = (-1, -1)  # global -> no sliding window
SOFTCAP = 0.0

DEVICE = "cuda"
NUM_BLOCKS = 256  # identity block table; covers 256*16 = 4096 key positions
MAXPOS = NUM_BLOCKS * BLOCK_SIZE


def _env_fixed(T: int):
    if T and T > 0:
        os.environ["BYTEEXACT_FIXED_TPS"] = str(int(T))
    else:
        os.environ.pop("BYTEEXACT_FIXED_TPS", None)


def build_static():
    """Deterministic paged KV cache + query bank + identity block table."""
    g = torch.Generator(device=DEVICE).manual_seed(1234)
    # cache layout: (num_blocks, block_size, num_kv_heads, head_dim)
    kcache = (
        torch.randn(
            NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
            device=DEVICE, dtype=DTYPE, generator=g,
        )
        * 0.1
    )
    vcache = (
        torch.randn(
            NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
            device=DEVICE, dtype=DTYPE, generator=g,
        )
        * 0.1
    )
    # query bank: one query row per absolute position
    qbank = (
        torch.randn(MAXPOS, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE, generator=g)
        * 0.1
    )
    block_table = torch.arange(NUM_BLOCKS, device=DEVICE, dtype=torch.int32).view(1, NUM_BLOCKS)
    return kcache, vcache, qbank, block_table


def make_segm_buffers(nseg: int):
    """Per-segment scratch, sized like the serve backend (first dim = q rows)."""
    segm_out = torch.empty(
        SEQ_THRESHOLD_3D, NUM_Q_HEADS, nseg, HEAD_DIM, device=DEVICE, dtype=torch.float32
    )
    segm_max = torch.empty(
        SEQ_THRESHOLD_3D, NUM_Q_HEADS, nseg, device=DEVICE, dtype=torch.float32
    )
    segm_exp = torch.empty(
        SEQ_THRESHOLD_3D, NUM_Q_HEADS, nseg, device=DEVICE, dtype=torch.float32
    )
    return segm_out, segm_max, segm_exp


def run_attn(static, *, M, base_pos, mode, nseg, fixed_T, seqused_override=None):
    """One unified_attention call.

    mode: "2D" (buffers=None -> 2D path) | "3D" (split-KV path, max_seqlen_q
    forced to 1 to mimic splitkv_verify_patch for M>1).
    The M query rows occupy absolute positions [base_pos .. base_pos+M-1];
    seq_len (seqused_k) = base_pos + M unless overridden.
    """
    kcache, vcache, qbank, block_table = static
    q = qbank[base_pos : base_pos + M].contiguous()
    out = torch.empty(M, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    S = int(base_pos + M) if seqused_override is None else int(seqused_override)
    cu_q = torch.tensor([0, M], device=DEVICE, dtype=torch.int32)
    seqused = torch.tensor([S], device=DEVICE, dtype=torch.int32)

    _env_fixed(fixed_T if mode == "3D" else 0)
    if mode == "2D":
        unified_attention(
            q=q, k=kcache, v=vcache, out=out,
            cu_seqlens_q=cu_q, max_seqlen_q=M, seqused_k=seqused, max_seqlen_k=S,
            softmax_scale=SCALE, causal=True, window_size=WINDOW,
            block_table=block_table, softcap=SOFTCAP,
            q_descale=None, k_descale=None, v_descale=None,
            seq_threshold_3D=SEQ_THRESHOLD_3D, num_par_softmax_segments=None,
            softmax_segm_output=None, softmax_segm_max=None, softmax_segm_expsum=None,
        )
    else:
        segm_out, segm_max, segm_exp = make_segm_buffers(nseg)
        unified_attention(
            q=q, k=kcache, v=vcache, out=out,
            cu_seqlens_q=cu_q, max_seqlen_q=1, seqused_k=seqused, max_seqlen_k=S,
            softmax_scale=SCALE, causal=True, window_size=WINDOW,
            block_table=block_table, softcap=SOFTCAP,
            q_descale=None, k_descale=None, v_descale=None,
            seq_threshold_3D=SEQ_THRESHOLD_3D, num_par_softmax_segments=nseg,
            softmax_segm_output=segm_out, softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_exp,
        )
    _env_fixed(0)
    return out


# ---------------------------------------------------------------- timing
def _time_eager(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)  # us
    return statistics.median(ts), (statistics.pstdev(ts) if len(ts) > 1 else 0.0)


def _time_graph(fn, iters, warmup):
    """Compute-only (CUDA-graph replay) -- the ONEGRAPH-served regime."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(max(5, warmup)):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e6)
        return statistics.median(ts), (statistics.pstdev(ts) if len(ts) > 1 else 0.0), True
    except Exception as exc:  # noqa: BLE001
        print(f"    [graph capture failed: {exc!r}]", flush=True)
        return None, None, False


def time_arm(static, *, M, S, mode, nseg, fixed_T, iters, warmup):
    base = S - M
    # preallocate the call closure (fresh out each call to be realistic, but
    # segm buffers are reallocated inside run_attn -> for graph stability we
    # bind a single set of tensors here instead).
    kcache, vcache, qbank, block_table = static
    q = qbank[base : base + M].contiguous()
    out = torch.empty(M, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    cu_q = torch.tensor([0, M], device=DEVICE, dtype=torch.int32)
    seqused = torch.tensor([S], device=DEVICE, dtype=torch.int32)
    if mode == "3D":
        segm_out, segm_max, segm_exp = make_segm_buffers(nseg)

    def call():
        _env_fixed(fixed_T if mode == "3D" else 0)
        if mode == "2D":
            unified_attention(
                q=q, k=kcache, v=vcache, out=out,
                cu_seqlens_q=cu_q, max_seqlen_q=M, seqused_k=seqused, max_seqlen_k=S,
                softmax_scale=SCALE, causal=True, window_size=WINDOW,
                block_table=block_table, softcap=SOFTCAP,
                q_descale=None, k_descale=None, v_descale=None,
                seq_threshold_3D=SEQ_THRESHOLD_3D, num_par_softmax_segments=None,
                softmax_segm_output=None, softmax_segm_max=None, softmax_segm_expsum=None,
            )
        else:
            unified_attention(
                q=q, k=kcache, v=vcache, out=out,
                cu_seqlens_q=cu_q, max_seqlen_q=1, seqused_k=seqused, max_seqlen_k=S,
                softmax_scale=SCALE, causal=True, window_size=WINDOW,
                block_table=block_table, softcap=SOFTCAP,
                q_descale=None, k_descale=None, v_descale=None,
                seq_threshold_3D=SEQ_THRESHOLD_3D, num_par_softmax_segments=nseg,
                softmax_segm_output=segm_out, softmax_segm_max=segm_max,
                softmax_segm_expsum=segm_exp,
            )

    eager_med, eager_std = _time_eager(call, iters, warmup)
    # env must be set during graph capture too; _env_fixed is inside call().
    graph_med, graph_std, graph_ok = _time_graph(call, iters, warmup)
    _env_fixed(0)
    return {
        "eager_us": round(eager_med, 3),
        "eager_std_us": round(eager_std, 3),
        "graph_us": round(graph_med, 3) if graph_ok else None,
        "graph_std_us": round(graph_std, 3) if graph_ok else None,
        "launch_overhead_us": round(eager_med - graph_med, 3) if graph_ok else None,
    }


# ------------------------------------------------------- M-invariance proof
def m_invariance(static, *, base_pos, M, mode, nseg, fixed_T):
    """Compare M-row verify output row-i bytes vs M=1 AR at the same abs pos.

    Returns per-row flip flags + max abs err. Byte-exact <=> 0 flips.
    """
    # verify batch: rows at positions base_pos..base_pos+M-1, seq_len=base_pos+M
    out_verify = run_attn(static, M=M, base_pos=base_pos, mode=mode, nseg=nseg, fixed_T=fixed_T)
    rows = []
    for i in range(M):
        pos = base_pos + i
        # M=1 AR at this absolute position: single row, seqused=pos+1
        out_ar = run_attn(
            static, M=1, base_pos=pos, mode=mode, nseg=nseg, fixed_T=fixed_T,
            seqused_override=pos + 1,
        )
        a = out_verify[i].float()
        b = out_ar[0].float()
        # byte equality on the stored bf16
        eq_bytes = torch.equal(
            out_verify[i].view(torch.int16), out_ar[0].view(torch.int16)
        )
        max_abs = float((a - b).abs().max().item())
        rows.append(
            {
                "row": i,
                "abs_pos": pos,
                "ar_seqused": pos + 1,
                "verify_seqused": base_pos + M,
                "byte_equal": bool(eq_bytes),
                "max_abs_err": max_abs,
            }
        )
    flips = sum(0 if r["byte_equal"] else 1 for r in rows)
    return {"flips": flips, "n_rows": M, "rows": rows}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/speed/byteexact_attn/microbench_results.json")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seqlens", type=int, nargs="+", default=[128, 256, 384, 512, 768, 1024])
    args = ap.parse_args()

    assert torch.cuda.is_available(), "need GPU (set CUDA_VISIBLE_DEVICES=0)"
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    static = build_static()

    # candidate fixed-chunk configs: (label, nseg, fixed_T) -> CHUNK = T*TILE(16)
    #   covers nseg*T*16 key positions; active segments at seq L = cdiv(cdiv(L,16),T)
    fixed_candidates = [
        ("3D_fixed_T2_seg64", 64, 2),    # CHUNK=32, covers 2048, hi occupancy
        ("3D_fixed_T4_seg32", 32, 4),    # CHUNK=64, covers 2048, med occupancy
        ("3D_fixed_T8_seg32", 32, 8),    # CHUNK=128, covers 4096
        ("3D_fixed_T16_seg16", 16, 16),  # CHUNK=256, stock-buffer-compatible, lo occ
    ]
    base_arms = [
        ("2D", "2D", None, 0),
        ("3D_seg1", "3D", 1, 0),
        ("3D_seg16_adaptive", "3D", 16, 0),
    ]
    arms = base_arms + [(lbl, "3D", ns, ft) for (lbl, ns, ft) in fixed_candidates]

    results = {
        "shape": {
            "num_q_heads": NUM_Q_HEADS, "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM, "block_size": BLOCK_SIZE, "dtype": "bf16",
            "tile_size_decode": 16, "segment_boundary_period": 256,
        },
        "timing": {},  # timing[M][seqlen][arm] = {eager_us, graph_us, ...}
    }

    for M in (1, 8):
        results["timing"][str(M)] = {}
        for S in args.seqlens:
            print(f"\n== M={M} seq_len={S} ==", flush=True)
            results["timing"][str(M)][str(S)] = {}
            for (lbl, mode, nseg, ft) in arms:
                # skip fixed configs that cannot cover this seq_len (byte-wrong)
                if ft and nseg * ft * 16 < S:
                    print(f"  {lbl:20s} SKIP (coverage {nseg*ft*16} < {S})", flush=True)
                    continue
                r = time_arm(
                    static, M=M, S=S, mode=mode, nseg=nseg, fixed_T=ft,
                    iters=args.iters, warmup=args.warmup,
                )
                results["timing"][str(M)][str(S)][lbl] = r
                go = r["graph_us"]
                print(
                    f"  {lbl:20s} eager={r['eager_us']:8.2f}us "
                    f"graph={go if go is None else round(go,2):>8}us "
                    f"launch={r['launch_overhead_us']}us",
                    flush=True,
                )

    # ---- decomposition (compute-only / graph us preferred; fall back eager) ----
    def pick(metric_arm, M, S):
        d = results["timing"][str(M)][str(S)].get(metric_arm)
        if not d:
            return None, None
        if d["graph_us"] is not None:
            return d["graph_us"], "graph"
        return d["eager_us"], "eager"

    decomp = {}
    for M in (1, 8):
        decomp[str(M)] = {}
        for S in args.seqlens:
            t2d, src = pick("2D", M, S)
            t_s1, _ = pick("3D_seg1", M, S)
            t_s16, _ = pick("3D_seg16_adaptive", M, S)
            best_fixed_lbl, best_fixed = None, None
            for (lbl, _ns, _ft) in fixed_candidates:
                v, _ = pick(lbl, M, S)
                if v is not None and (best_fixed is None or v < best_fixed):
                    best_fixed, best_fixed_lbl = v, lbl
            if None in (t2d, t_s1, t_s16):
                continue
            decomp[str(M)][str(S)] = {
                "source": src,
                "t_2D_byteexact_us": t2d,
                "t_3D_seg1_us": t_s1,
                "t_3D_seg16_adaptive_us": t_s16,
                "best_fixed_arm": best_fixed_lbl,
                "t_3D_fixed_best_us": best_fixed,
                # additive split of the 2D->fast-3D gap:
                "gap_2D_minus_fast3D_us": round(t2d - t_s16, 3),
                "codepath_memory_us": round(t2d - t_s1, 3),       # 2D vs 3D@no-split
                "split_parallelism_us": round(t_s1 - t_s16, 3),   # 1->16 segments
                # candidate cost vs the fast adaptive path (byte-exact penalty):
                "fixed_vs_adaptive_us": round(best_fixed - t_s16, 3) if best_fixed else None,
                "fixed_recovers_vs_2D_us": round(t2d - best_fixed, 3) if best_fixed else None,
            }
    results["attn_tax_decomposition"] = decomp

    # ---- M-invariance proof ----
    print("\n== M-invariance proof ==", flush=True)
    # straddle positions: base s.t. AR seqused (base+1..) and verify seqused
    # (base+8) straddle a 256 multiple -> adaptive flips, fixed is exact.
    proof = {}
    proof_cfgs = [
        ("2D_byteexact_ref", "2D", None, 0),
        ("3D_seg16_adaptive", "3D", 16, 0),
        ("3D_fixed_T2_seg64", "3D", 64, 2),
    ]
    straddle_bases = [250, 506]   # 251..258 straddles 256; 507..514 straddles 512
    control_bases = [100]         # no 256-boundary crossing -> all should match
    for tag, bases in (("straddle", straddle_bases), ("control", control_bases)):
        proof[tag] = {}
        for base in bases:
            proof[tag][str(base)] = {}
            for (lbl, mode, nseg, ft) in proof_cfgs:
                res = m_invariance(static, base_pos=base, M=8, mode=mode, nseg=nseg, fixed_T=ft)
                proof[tag][str(base)][lbl] = res
                print(
                    f"  {tag:8s} base={base:4d} {lbl:20s} flips={res['flips']}/{res['n_rows']}",
                    flush=True,
                )
    results["m_invariance"] = proof

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)

    # restore stock env
    _env_fixed(0)


if __name__ == "__main__":
    sys.exit(main())
