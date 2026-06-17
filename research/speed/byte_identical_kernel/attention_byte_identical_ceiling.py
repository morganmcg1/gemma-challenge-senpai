#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #562 -- Attention byte-identical ceiling (denken).

Completes the per-component decode-step byte-identical census. #555 measured the
HEAD GEMV wall (served ~483 GB/s, no byte-identical sliver -- every faster read
reorders the K=2560 reduction, land #506). #550 pinned the BODY int4 GEMV
(Marlin is the ONLY w4a16 kernel on sm_86 -> byte-identical by construction, no
alternate kernel to even test). The one decode component NOT yet measured my way
is the ATTENTION: vLLM is forced onto a Triton SDPA kernel for this checkpoint's
heterogeneous head_dim (FlashAttention can't serve head_dim 512 on sm_86, #550 /
fern #507). lawine #554 priced the attention LAUNCH-overhead floor (0.573 ms / 42
launches) -- the fixed dispatch tax. This card measures the orthogonal quantity:
the per-kernel COMPUTE/BW ceiling and BYTE-IDENTITY of the served Triton SDPA.

Binding question: is vLLM's served decode-attention kernel already at its
byte-identical wall (every faster attention config reorders the softmax/PV
reduction -> not bit-identical, exactly like the head GEMV), or is there a faster
bit-identical attention sliver = a fresh head_dim-robust lever?

Served decode op (read straight from vllm 0.22.1rc1):
  TritonAttentionImpl.forward -> unified_attention(...). At M=1 single-stream
  decode (max_seqlen_q==1, num_seqs=1 <= seq_threshold_3D=128//nkv) the wrapper
  selects the 3D split-softmax path: NUM_PAR_SOFTMAX_SEGMENTS=16 parallel
  FlashDecoding-style segments + a reduce_segments epilogue. THAT specific
  reduction order is the #319 byte-identical reference.

Method = my #555 template, applied to attention outputs:
  Stage 1 (PRIMARY): measure served Triton SDPA achievable BW/latency at M=1,
    per head_dim class (KV-read bound: bytes = 2*L_eff*nkv*hs*elt; Q negligible).
  Stage 2 (PRIMARY, load-bearing): enumerate faster-or-different attention
    configs (2D single-pass, seg8/seg32 splits, alt tile, torch-SDPA math/
    mem-efficient/flash, fp32 math) and probe attn_bitwise_rate (torch.equal vs
    served), attn_max_abs_diff, attn_argmax_identity_rate. A config is a GREEN
    lever ONLY at bitwise_rate==1.0 AND faster than served (land #506 / #319:
    argmax-rate on a finite sample is necessary-not-sufficient -- proved in #555).
  Stage 3 (PRIMARY): compose head(#555)+body(#550)+attn into
    decode_step_byte_identical_lever_exists (bool over all 3) + verdict.

LOCAL-ONLY. analysis_only=true, official_tps=0. NO HF Job, NO --launch, NO
submission, NO served-file change. One pod A10G sm_86. MAX_NUM_SEQS=1.

Cite: #555 (aiu5pkdw, head-GEMV byte-identical method template), #550 (5aobahij,
Marlin-lock + FA-can't-serve-het-head_dim -> Triton forced), #554 (launch-floor
complement), land #506 (reduction-order breaks byte-identity), fern #507
(head_dim-512 no invariant attention path, Ampere SMEM cap, cross-framework prior).

Reproduce: cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  research/speed/byte_identical_kernel/attention_byte_identical_ceiling.py \
  --wandb_group attention-byte-identical-ceiling \
  --wandb_name denken/attention-byte-identical-ceiling
"""
from __future__ import annotations

import os

# A10G GPU 0 is the only real device; the pod exports a non-zero
# CUDA_VISIBLE_DEVICES (env quirk, #551/#555). Force 0 BEFORE torch.
if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

# --------------------------------------------------------------------------- #
# anchors (cite, do not re-derive)                                            #
# --------------------------------------------------------------------------- #
A10G_HBM_GBS = 600.0                       # GA102 datasheet HBM peak
NUM_PAR_SOFTMAX_SEGMENTS = 16              # vLLM served 3D-decode segment count
MIN_LAUNCH_GRID_SIZE_2D = 128             # vLLM 2D->3D threshold numerator

# head component (#555 aiu5pkdw): byte-identical wall = served ~483 GB/s, no sliver
HEAD_BYTE_IDENTICAL_WALL_GBS_555 = 483.41
HEAD_BW_LEVER_GREEN_555 = False
# body component (#550 5aobahij): Marlin is the ONLY w4a16 kernel on sm_86
BODY_BYTE_IDENTICAL_LEVER_GREEN_550 = False  # no alternate kernel exists to test

BFH_TPS = 252.31                           # base_fullhead local served anchor (wirbel #553)
FREE_CEILING_TPS_554 = 311.25              # lawine #554 corrected magically-free ceiling
STRICT_CEILING_TPS_549 = 292.0             # fern #549 strict-via-candidate-verify
OFFICIAL_1_TPS = 481.53                     # official public #1 (NOT touched; official_tps=0)
SIGMA_HW = 4.864                           # absolute hardware TPS noise (1 sigma)
LAUNCH_FLOOR_MS_554 = 0.573                # lawine #554 attention launch-overhead floor

# materially-faster margin: same Triton-kernel jitter is <~1%; a genuine reduction
# re-tile / different kernel is several %. 1% cleanly separates a real lever from
# same-kernel noise. (Structurally: bit-identical => same reduction => same kernel
# => cannot be materially faster.)  Mirrors #555 FASTER_MARGIN_FRAC.
FASTER_MARGIN_FRAC = 0.01

LOCAL_CKPT = (
    "/senpai-run/home/student-denken/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)


def roofline_us(num_bytes: float, gbs: float = A10G_HBM_GBS) -> float:
    return num_bytes / (gbs * 1e9) * 1e6


def kv_bytes(L_eff: int, nkv: int, hs: int, elt: int) -> float:
    """Dominant decode-attention byte read: K + V cache for L_eff context tokens.
    Q (one token) is negligible at any realistic L (researcher accounting)."""
    return 2.0 * L_eff * nkv * hs * elt


# --------------------------------------------------------------------------- #
# self-test: roofline + census arithmetic (no GPU)                            #
# --------------------------------------------------------------------------- #
def self_test() -> bool:
    ok = True
    # KV byte accounting: L=1024, nkv=4, hs=256, bf16 -> 4.0 MiB read
    b = kv_bytes(1024, 4, 256, 2)
    ok &= abs(b - 2 * 1024 * 4 * 256 * 2) < 1
    ok &= abs(b / 1e6 - 4.194304) < 0.001
    # roofline: that read at 600 GB/s = 6.99 us
    ok &= abs(roofline_us(b) - 6.99) < 0.05
    # 2D->3D threshold for nkv=4 -> 32; num_seqs=1 <= 32 -> 3D served at M=1
    thr = MIN_LAUNCH_GRID_SIZE_2D // 4
    ok &= (thr == 32) and (1 <= thr)
    # census composition: head False + body False + attn False -> no lever
    ok &= (HEAD_BW_LEVER_GREEN_555 is False)
    ok &= (BODY_BYTE_IDENTICAL_LEVER_GREEN_550 is False)
    composed = bool(HEAD_BW_LEVER_GREEN_555 or BODY_BYTE_IDENTICAL_LEVER_GREEN_550 or False)
    ok &= (composed is False)
    print(f"[self-test] kv_read(L1024,nkv4,hs256,bf16)={b/1e6:.3f}MB "
          f"floor@600={roofline_us(b):.2f}us 2d3d_thr(nkv4)={thr} "
          f"census_no_lever={not composed} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


# --------------------------------------------------------------------------- #
# timing primitives (mirror #555)                                             #
# --------------------------------------------------------------------------- #
def eager_time(fn, iters: int, warmup: int):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
           for _ in range(iters)]
    for s, e in evs:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in evs)
    return {"min_ms": ms[0], "median_ms": statistics.median(ms)}


def graph_time(fn, reps: int, warmup: int, repeats: int):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    try:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                for _ in range(reps):
                    fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                fn()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        return {"unsupported": True, "error": repr(exc)[:160]}
    ms = []
    for _ in range(repeats):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); g.replay(); en.record(); torch.cuda.synchronize()
        ms.append(st.elapsed_time(en) / reps)
    ms.sort(); del g
    return {"min_ms": ms[0], "median_ms": statistics.median(ms)}


def measure(fn, nbytes, reps_graph, warmup, iters, repeats):
    eg = eager_time(fn, iters, warmup)
    gr = graph_time(fn, reps_graph, warmup, repeats)
    out = {
        "eager_min_ms": eg["min_ms"], "eager_median_ms": eg["median_ms"],
        "eager_bw_gbps": (nbytes / (eg["min_ms"] * 1e-3) / 1e9) if eg["min_ms"] > 0 else float("nan"),
    }
    if gr.get("unsupported"):
        out.update({"graph_unsupported": True, "graph_error": gr.get("error"),
                    "graph_bw_gbps": float("nan"), "graph_min_ms": float("nan")})
    else:
        out.update({"graph_min_ms": gr["min_ms"], "graph_median_ms": gr["median_ms"],
                    "graph_bw_gbps": (nbytes / (gr["min_ms"] * 1e-3) / 1e9) if gr["min_ms"] > 0 else float("nan")})
    out["min_ms"] = out.get("graph_min_ms") if math.isfinite(out.get("graph_min_ms", float("nan"))) \
        else out["eager_min_ms"]
    out["bw_gbps"] = out.get("graph_bw_gbps") if math.isfinite(out.get("graph_bw_gbps", float("nan"))) \
        else out["eager_bw_gbps"]
    return out


# --------------------------------------------------------------------------- #
# engine introspection (reuse #550 model-runner reach)                        #
# --------------------------------------------------------------------------- #
def get_model_runner(llm):
    cands = []
    try:
        cands.append(llm.llm_engine.engine_core.engine_core
                     .model_executor.driver_worker.worker.model_runner)
    except Exception:  # noqa: BLE001
        pass
    try:
        ec = llm.llm_engine.engine_core
        mexec = getattr(ec, "model_executor", None)
        if mexec is not None:
            dw = getattr(mexec, "driver_worker", None)
            w = getattr(dw, "worker", dw)
            mr = getattr(w, "model_runner", None)
            if mr is not None:
                cands.append(mr)
    except Exception:  # noqa: BLE001
        pass
    for mr in cands:
        if mr is not None and hasattr(mr, "model"):
            return mr
    raise RuntimeError("could not reach model_runner")


def enumerate_attention(model_runner) -> dict:
    """Walk every self_attn.attn module; record per-layer impl class + the
    attention shape knobs (head_size, num_heads, num_kv_heads, sliding window,
    softcap, scale). Confirms the served impl is TritonAttentionImpl."""
    model = model_runner.model
    layers = []
    impl_classes = set()
    for name, mod in model.named_modules():
        if name.endswith("self_attn.attn") and hasattr(mod, "impl"):
            impl = mod.impl
            ic = type(impl).__name__
            impl_classes.add(ic)
            ws = getattr(impl, "sliding_window", None)
            layers.append({
                "name": name,
                "impl_class": ic,
                "num_heads": int(getattr(impl, "num_heads", -1)),
                "head_size": int(getattr(impl, "head_size", -1)),
                "num_kv_heads": int(getattr(impl, "num_kv_heads", -1)),
                "scale": float(getattr(impl, "scale", float("nan"))),
                "logits_soft_cap": float(getattr(impl, "logits_soft_cap", 0.0) or 0.0),
                "sliding_window": list(ws) if isinstance(ws, (tuple, list)) else ws,
                "kv_cache_dtype": str(getattr(impl, "kv_cache_dtype", "")),
            })
    head_sizes = sorted({l["head_size"] for l in layers})
    by_hs = {}
    for hs in head_sizes:
        cls = [l for l in layers if l["head_size"] == hs]
        by_hs[hs] = {"n_layers": len(cls), "example": cls[0]}
    return {"n_attn_layers": len(layers), "impl_classes": sorted(impl_classes),
            "head_size_classes": head_sizes, "by_head_size": by_hs, "layers": layers}


# --------------------------------------------------------------------------- #
# capture: intercept the real served unified_attention call per head_size      #
# --------------------------------------------------------------------------- #
def install_capture(captures: dict, want_iters: int, want_warmup: int):
    """Monkeypatch vllm ...triton_attn.unified_attention so the FIRST decode
    call (max_seqlen_q==1) per distinct head_size:
      (1) times the real served kernel on the LIVE paged KV (eager CUDA events),
      (2) gathers+clones the real contiguous K/V for that sequence, clones Q,
      (3) records the exact served scalars (scale, softcap, window, seq_threshold,
          segments, kv layout strides + dtype, block_size, head counts).
    Returns (module, original_fn) so the caller can restore."""
    import torch
    import vllm.v1.attention.backends.triton_attn as ta_mod
    orig = ta_mod.unified_attention

    def wrapped(*args, **kw):
        try:
            q = kw.get("q"); k = kw.get("k"); v = kw.get("v")
            max_q = kw.get("max_seqlen_q"); seqk = kw.get("seqused_k")
            bt = kw.get("block_table")
            hs = int(q.shape[2]) if q is not None else -1
            decode = (max_q == 1) and (seqk is not None) and (int(seqk.numel()) == 1)
            if decode and hs not in captures:
                L = int(seqk[0].item())
                block_size = int(v.shape[1])
                nkv = int(k.shape[2])
                nblk = (L + block_size - 1) // block_size
                phys = bt[0, :nblk].to(torch.long)
                Kc = k[phys].reshape(-1, nkv, hs)[:L].clone()      # [L, nkv, hs]
                Vc = v[phys].reshape(-1, nkv, hs)[:L].clone()
                # time the REAL served kernel on the live args (eager, min ms)
                out_buf = kw.get("out")
                def _call():
                    return orig(*args, **kw)
                for _ in range(want_warmup):
                    _call()
                torch.cuda.synchronize()
                evs = [(torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True)) for _ in range(want_iters)]
                for s, e in evs:
                    s.record(); _call(); e.record()
                torch.cuda.synchronize()
                live_ms = sorted(s.elapsed_time(e) for s, e in evs)
                ws = kw.get("window_size")
                # replicate vllm's use_3d gate on the LIVE kwargs so we know the
                # served path (3D split-softmax vs 2D single-pass) per head_size.
                _seg_out = kw.get("softmax_segm_output")
                _thr = kw.get("seq_threshold_3D")
                _seg = kw.get("num_par_softmax_segments")
                live_use_3d = bool(
                    _thr is not None and _seg is not None and _seg_out is not None
                    and kw.get("softmax_segm_max") is not None
                    and kw.get("softmax_segm_expsum") is not None
                    and int(max_q) <= 1 and int(seqk.numel()) <= int(_thr))
                captures[hs] = {
                    "head_size": hs,
                    "num_query_heads": int(q.shape[1]),
                    "num_kv_heads": nkv,
                    "block_size": block_size,
                    "real_L": L,
                    "live_use_3d": live_use_3d,
                    "live_segm_out_is_none": _seg_out is None,
                    "live_max_seqlen_q": int(max_q),
                    "live_num_seqs": int(seqk.numel()),
                    "q_shape": list(q.shape),
                    "out_shape": list(out_buf.shape) if out_buf is not None else None,
                    "scale": float(kw.get("softmax_scale")),
                    "softcap": float(kw.get("softcap") or 0.0),
                    "window_size": [int(ws[0]), int(ws[1])] if ws is not None else [-1, -1],
                    "seq_threshold_3D": int(kw.get("seq_threshold_3D")) if kw.get("seq_threshold_3D") is not None else None,
                    "num_par_softmax_segments": int(kw.get("num_par_softmax_segments")) if kw.get("num_par_softmax_segments") is not None else None,
                    "k_dtype": str(k.dtype),
                    "k_elt": int(k.element_size()),
                    "k_strides": [int(s) for s in k.stride()],
                    "k_num_blocks": int(k.shape[0]),
                    "live_min_ms": live_ms[0],
                    "live_median_ms": statistics.median(live_ms),
                    "Kc": Kc, "Vc": Vc,
                    "q": q[:1].clone(),
                    # genuine served-kernel output on the LIVE paged cache; used in
                    # Stage 2 to prove the compact-cache rerun is bit-faithful.
                    "live_out": out_buf[:1].clone() if out_buf is not None else None,
                }
        except Exception as exc:  # noqa: BLE001
            captures.setdefault("_errors", []).append(repr(exc)[:200])
        return orig(*args, **kw)

    ta_mod.unified_attention = wrapped
    return ta_mod, orig


# --------------------------------------------------------------------------- #
# compact paged-KV reconstruction (layout-faithful) + metadata                #
# --------------------------------------------------------------------------- #
def get_layout():
    try:
        from vllm.v1.attention.backends.utils import get_kv_cache_layout
        return get_kv_cache_layout()
    except Exception:  # noqa: BLE001
        return "NHD"


def build_compact_cache(Kc, Vc, block_size, layout, device):
    """Scatter contiguous [L,nkv,hs] K/V into a fresh compact paged cache whose
    physical stride order matches the served layout (NHD or HND), so Stage-1 BW
    is layout-faithful. Returns the unified_attention kwargs subset for one M=1
    decode of a single sequence covering all L tokens."""
    import torch
    L, nkv, hs = Kc.shape
    nblk = (L + block_size - 1) // block_size
    Lpad = nblk * block_size
    dtype = Kc.dtype
    # contiguous padded logical buffers [Lpad, nkv, hs]; assign into the
    # interleaved (factor-2 block stride) kv tensor via the VIEW so the data
    # actually lands in the cache the kernel reads. (A reshape of the
    # non-contiguous kv[:,0] slice would silently COPY -> zero cache.)
    Kp = torch.zeros(Lpad, nkv, hs, dtype=dtype, device=device); Kp[:L] = Kc
    Vp = torch.zeros(Lpad, nkv, hs, dtype=dtype, device=device); Vp[:L] = Vc
    if layout == "HND":
        # served physical key_cache shape [nblk, nkv, block_size, hs]
        kv = torch.zeros(nblk, 2, nkv, block_size, hs, dtype=dtype, device=device)
        kv[:, 0] = Kp.reshape(nblk, block_size, nkv, hs).permute(0, 2, 1, 3)
        kv[:, 1] = Vp.reshape(nblk, block_size, nkv, hs).permute(0, 2, 1, 3)
        key_cache, value_cache = kv.unbind(1)            # [nblk, nkv, bs, hs]
    else:  # NHD: served physical key_cache shape [nblk, block_size, nkv, hs]
        kv = torch.zeros(nblk, 2, block_size, nkv, hs, dtype=dtype, device=device)
        kv[:, 0] = Kp.reshape(nblk, block_size, nkv, hs)
        kv[:, 1] = Vp.reshape(nblk, block_size, nkv, hs)
        key_cache, value_cache = kv.unbind(1)            # [nblk, bs, nkv, hs]
    block_table = torch.arange(nblk, device=device, dtype=torch.int32).reshape(1, nblk)
    seqused_k = torch.tensor([L], device=device, dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
    return {"k": key_cache, "v": value_cache, "block_table": block_table,
            "seqused_k": seqused_k, "cu_seqlens_q": cu_seqlens_q,
            "max_seqlen_q": 1, "max_seqlen_k": L, "nblk": nblk}


def make_segm_buffers(num_query_heads, head_size, segments, device):
    import torch
    from vllm.utils.math_utils import next_power_of_2
    pad = next_power_of_2(head_size)
    rows = 1  # M=1 decode: one query token
    return {
        "softmax_segm_output": torch.empty((rows, num_query_heads, segments, pad),
                                            dtype=torch.float32, device=device),
        "softmax_segm_max": torch.empty((rows, num_query_heads, segments),
                                         dtype=torch.float32, device=device),
        "softmax_segm_expsum": torch.empty((rows, num_query_heads, segments),
                                            dtype=torch.float32, device=device),
    }


# --------------------------------------------------------------------------- #
# candidate attention runners (all produce out [1, H, hs] bf16)               #
# --------------------------------------------------------------------------- #
def _uatt():
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    return unified_attention


def make_unified_fn(cfg, compact, out, segments, seq_threshold, segm):
    uatt = _uatt()
    q = cfg["q"]; ws = tuple(cfg["window_size"])
    scale = cfg["scale"]; softcap = cfg["softcap"]
    k = compact["k"]; v = compact["v"]; bt = compact["block_table"]
    seqk = compact["seqused_k"]; cuq = compact["cu_seqlens_q"]; mk = compact["max_seqlen_k"]
    so = segm["softmax_segm_output"]; sm = segm["softmax_segm_max"]; se = segm["softmax_segm_expsum"]

    def fn():
        uatt(q=q, k=k, v=v, out=out, cu_seqlens_q=cuq, max_seqlen_q=1,
             seqused_k=seqk, max_seqlen_k=mk, softmax_scale=scale, causal=True,
             window_size=ws, block_table=bt, softcap=softcap,
             q_descale=None, k_descale=None, v_descale=None,
             seq_threshold_3D=seq_threshold, num_par_softmax_segments=segments,
             softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se)
        return out
    return fn


def _expand_kv(Kc, Vc, nqpkv):
    # [L,nkv,hs] -> [1, H=nkv*nqpkv, L, hs]
    L, nkv, hs = Kc.shape
    K = Kc.permute(1, 0, 2).repeat_interleave(nqpkv, dim=0).unsqueeze(0).contiguous()
    V = Vc.permute(1, 0, 2).repeat_interleave(nqpkv, dim=0).unsqueeze(0).contiguous()
    return K, V


def _sliding_addmask(L, window_size, device):
    import torch
    w0 = window_size[0]
    if w0 < 0:
        return None  # full causal; decode token sees all 0..L-1
    W = w0 + 1
    allowed = torch.arange(L, device=device) > (L - 1 - W)
    m = torch.zeros(L, dtype=torch.float32, device=device)
    m[~allowed] = float("-inf")
    return m.view(1, 1, 1, L)


def make_sdpa_fn(cfg, Kc, Vc, backend):
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    H = cfg["num_query_heads"]; hs = cfg["head_size"]; nkv = cfg["num_kv_heads"]
    nqpkv = H // nkv
    q = cfg["q"][0].unsqueeze(0).unsqueeze(2)         # [1,H,1,hs]
    K4, V4 = _expand_kv(Kc, Vc, nqpkv)                # [1,H,L,hs]
    L = Kc.shape[0]
    am = _sliding_addmask(L, cfg["window_size"], q.device)
    if am is not None:
        am = am.to(q.dtype)   # EFFICIENT_ATTENTION requires bias dtype == query dtype
    scale = cfg["scale"]
    bk = {"math": SDPBackend.MATH, "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
          "flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION}[backend]
    out_holder = {}

    def fn():
        with sdpa_kernel([bk]):
            o = F.scaled_dot_product_attention(q, K4, V4, attn_mask=am,
                                               is_causal=False, scale=scale)
        out_holder["o"] = o.reshape(1, H, hs).to(torch.bfloat16)
        return out_holder["o"]
    return fn, out_holder


def make_manual_fp32_fn(cfg, Kc, Vc):
    import torch
    H = cfg["num_query_heads"]; hs = cfg["head_size"]; nkv = cfg["num_kv_heads"]
    nqpkv = H // nkv
    q = cfg["q"][0].unsqueeze(1).float()              # [H,1,hs]
    K4, V4 = _expand_kv(Kc, Vc, nqpkv)
    Kh = K4[0].float(); Vh = V4[0].float()            # [H,L,hs]
    L = Kc.shape[0]
    am = _sliding_addmask(L, cfg["window_size"], q.device)
    scale = cfg["scale"]; softcap = cfg["softcap"]
    out_holder = {}

    def fn():
        S = torch.matmul(q, Kh.transpose(-1, -2)) * scale     # [H,1,L]
        if softcap and softcap > 0:
            S = softcap * torch.tanh(S / softcap)
        if am is not None:
            S = S + am.reshape(1, 1, L)
        P = torch.softmax(S, dim=-1)
        o = torch.matmul(P, Vh)                                # [H,1,hs]
        out_holder["o"] = o.transpose(0, 1).reshape(1, H, hs).to(torch.bfloat16)
        return out_holder["o"]
    return fn, out_holder


def _set_tile_override(value):
    """Force unified_attention's decode/prefill tile size; returns a restore fn."""
    import vllm.v1.attention.ops.triton_unified_attention as u
    orig = u._get_tile_size
    u._get_tile_size = lambda *a, **k: value
    def restore():
        u._get_tile_size = orig
    return restore


# --------------------------------------------------------------------------- #
def synth_kv(cfg, L, rms_k, rms_v, device, seed):
    import torch
    g = torch.Generator(device=device).manual_seed(seed)
    nkv = cfg["num_kv_heads"]; hs = cfg["head_size"]
    Kc = torch.randn(L, nkv, hs, generator=g, dtype=torch.float32, device=device).to(torch.bfloat16) * rms_k
    Vc = torch.randn(L, nkv, hs, generator=g, dtype=torch.float32, device=device).to(torch.bfloat16) * rms_v
    return Kc.contiguous(), Vc.contiguous()


def served_out(cfg, compact, segm, seq_threshold, segments, device):
    import torch
    H = cfg["num_query_heads"]; hs = cfg["head_size"]
    out = torch.empty(1, H, hs, dtype=torch.bfloat16, device=device)
    fn = make_unified_fn(cfg, compact, out, segments, seq_threshold, segm)
    fn()
    torch.cuda.synchronize()
    return out.clone()


# --------------------------------------------------------------------------- #
# Stage 1: served Triton SDPA achievable BW / latency at M=1                   #
# --------------------------------------------------------------------------- #
def stage1(cfg, layout, args, device):
    import torch
    hs = cfg["head_size"]; nkv = cfg["num_kv_heads"]; elt = cfg["k_elt"]
    seg = cfg["num_par_softmax_segments"] or NUM_PAR_SOFTMAX_SEGMENTS
    seq_thr = cfg["seq_threshold_3D"] if cfg["seq_threshold_3D"] is not None else (MIN_LAUNCH_GRID_SIZE_2D // nkv)
    rms_k = float(cfg["Kc"].float().pow(2).mean().sqrt())
    rms_v = float(cfg["Vc"].float().pow(2).mean().sqrt())
    w0 = cfg["window_size"][0]
    sweep = sorted({L for L in args.l_sweep if L >= 16})
    rows = []
    segm = make_segm_buffers(cfg["num_query_heads"], hs, seg, device)
    for L in sweep:
        Kc, Vc = synth_kv(cfg, L, rms_k, rms_v, device, seed=1000 + L)
        compact = build_compact_cache(Kc, Vc, cfg["block_size"], layout, device)
        H = cfg["num_query_heads"]
        out = torch.empty(1, H, hs, dtype=torch.bfloat16, device=device)
        fn = make_unified_fn(cfg, compact, out, seg, seq_thr, segm)
        L_eff = L if w0 < 0 else min(L, w0 + 1)
        nb = kv_bytes(L_eff, nkv, hs, elt)
        m = measure(fn, nb, args.reps_graph, args.warmup, args.iters, args.repeats)
        rows.append({"L": L, "L_eff": L_eff, "kv_read_bytes": nb,
                     "kv_read_MB": nb / 1e6, "latency_ms": m["min_ms"],
                     "achievable_bw_GBs": m["bw_gbps"], "pct_of_peak": m["bw_gbps"] / A10G_HBM_GBS * 100.0,
                     "eager_min_ms": m["eager_min_ms"], "graph_min_ms": m.get("graph_min_ms")})
        del compact, out
    # headline = the largest swept L (best amortization of fixed overhead)
    head = max(rows, key=lambda r: r["L"]) if rows else {}
    # live cross-check (real served kernel on live KV at real_L, eager)
    live_L_eff = cfg["real_L"] if w0 < 0 else min(cfg["real_L"], w0 + 1)
    live_nb = kv_bytes(live_L_eff, nkv, hs, elt)
    live_bw = live_nb / (cfg["live_min_ms"] * 1e-3) / 1e9 if cfg["live_min_ms"] > 0 else float("nan")
    return {"sweep": rows, "headline_L": head.get("L"),
            "attn_sdpa_latency_ms": head.get("latency_ms"),
            "attn_sdpa_achievable_bw_GBs": head.get("achievable_bw_GBs"),
            "attn_sdpa_pct_of_peak": head.get("pct_of_peak"),
            "live_real_L": cfg["real_L"], "live_min_ms": cfg["live_min_ms"],
            "live_achievable_bw_GBs": live_bw,
            "seg": seg, "seq_threshold_3D": seq_thr,
            "rms_k": rms_k, "rms_v": rms_v}


# --------------------------------------------------------------------------- #
# Stage 2: the bitwise-identity probe (the load-bearing leg)                   #
# --------------------------------------------------------------------------- #
def stage2(cfg, layout, args, device, stage1_info):
    import torch
    hs = cfg["head_size"]; nkv = cfg["num_kv_heads"]; elt = cfg["k_elt"]
    H = cfg["num_query_heads"]
    served_seg = stage1_info["seg"]; seq_thr = stage1_info["seq_threshold_3D"]
    rms_k = stage1_info["rms_k"]; rms_v = stage1_info["rms_v"]
    w0 = cfg["window_size"][0]
    L = args.probe_L
    L_eff = L if w0 < 0 else min(L, w0 + 1)
    served_bytes = kv_bytes(L_eff, nkv, hs, elt)
    kv_is_float = cfg["k_dtype"] in ("torch.bfloat16", "torch.float16")

    # ---- faithfulness anchor: compact-cache served rerun vs LIVE served out ----
    # Rebuild the compact paged cache from the FULL captured KV at real_L and run
    # the served config; bit-compare against the genuine served-kernel output that
    # ran on the live paged cache. bitwise==1 proves the reconstruction + rerun is
    # byte-faithful to the served path, so the Stage-2 ref is the served truth.
    faithful = {"checked": False, "bitwise": None, "max_abs_diff": None,
                "real_L": cfg["real_L"]}
    live_out = cfg.get("live_out")
    if live_out is not None:
        try:
            Lr = int(cfg["real_L"])
            Kf = cfg["Kc"][:Lr].contiguous(); Vf = cfg["Vc"][:Lr].contiguous()
            compf = build_compact_cache(Kf, Vf, cfg["block_size"], layout, device)
            segm_f = make_segm_buffers(H, hs, served_seg, device)
            ref_live = served_out(cfg, compf, segm_f, seq_thr, served_seg, device)
            lo = live_out.to(ref_live.dtype).reshape(ref_live.shape)
            faithful = {"checked": True,
                        "bitwise": int(torch.equal(ref_live, lo)),
                        "max_abs_diff": float((ref_live.float() - lo.float()).abs().max()),
                        "real_L": Lr}
            del compf
        except Exception as exc:  # noqa: BLE001
            faithful = {"checked": False, "error": repr(exc)[:200],
                        "real_L": int(cfg["real_L"])}

    # candidate registry: name -> ("unified"|"sdpa"|"manual", params)
    cand_specs = [
        ("served_3d_rerun", "unified", {"seg": served_seg, "thr": seq_thr}),
        ("unified_2d", "unified", {"seg": served_seg, "thr": 0}),       # single-pass reduction
        ("unified_seg8", "unified", {"seg": 8, "thr": seq_thr}),
        ("unified_seg32", "unified", {"seg": 32, "thr": seq_thr}),
        ("unified_tile_alt", "unified_tile", {"seg": served_seg, "thr": seq_thr}),
    ]
    if kv_is_float:
        for be in ("math", "mem_efficient", "flash", "cudnn"):
            cand_specs.append((f"sdpa_{be}", "sdpa", {"backend": be}))
        cand_specs.append(("manual_fp32", "manual", {}))

    # accumulate per-candidate bitwise / argmax over N draws
    agg = {name: {"kind": kind, "bitwise_hits": 0, "argmax_hits": 0, "n": 0,
                  "max_abs_diff": 0.0, "error": None, "latency_ms": None,
                  "bw_GBs": None, "faster_than_served": None}
           for (name, kind, _) in cand_specs}

    seeds = [424242] + [7000 + i for i in range(args.n_draws - 1)]
    served_lat = None
    served_ref_error = None
    for di, seed in enumerate(seeds):
        if di == 0:
            Kc, Vc = cfg["Kc"][:L].contiguous(), cfg["Vc"][:L].contiguous()
            if Kc.shape[0] < L:  # real capture shorter than probe_L -> pad with synth
                Ks, Vs = synth_kv(cfg, L - Kc.shape[0], rms_k, rms_v, device, seed=99)
                Kc = torch.cat([Kc, Ks], 0); Vc = torch.cat([Vc, Vs], 0)
        else:
            Kc, Vc = synth_kv(cfg, L, rms_k, rms_v, device, seed=seed)
        compact = build_compact_cache(Kc, Vc, cfg["block_size"], layout, device)
        segm_served = make_segm_buffers(H, hs, served_seg, device)
        try:
            ref = served_out(cfg, compact, segm_served, seq_thr, served_seg, device)
        except Exception as exc:  # noqa: BLE001
            served_ref_error = repr(exc)[:200]
            del compact
            continue
        ref_arg = int(ref.reshape(-1).argmax())
        for (name, kind, p) in cand_specs:
            a = agg[name]
            restore = None
            try:
                if kind in ("unified", "unified_tile"):
                    seg = p["seg"]; thr = p["thr"]
                    segm = make_segm_buffers(H, hs, seg, device)
                    out = torch.empty(1, H, hs, dtype=torch.bfloat16, device=device)
                    if kind == "unified_tile":
                        # served decode tile is 16 (bf16, non-gemma3 -> default).
                        # 32 tests a re-tile of the intra-segment reduction; for
                        # head_dim 512 the bigger tile overflows A10G SMEM (recorded
                        # as error -- itself the Ampere-SMEM-cap finding, fern #507).
                        alt = 32
                        a["tile_alt"] = alt
                        restore = _set_tile_override(alt)
                    fn = make_unified_fn(cfg, compact, out, seg, thr, segm)
                    fn(); torch.cuda.synchronize()
                    cand = out.clone()
                elif kind == "sdpa":
                    fn, hold = make_sdpa_fn(cfg, Kc, Vc, p["backend"])
                    cand = fn().clone(); torch.cuda.synchronize()
                elif kind == "manual":
                    fn, hold = make_manual_fp32_fn(cfg, Kc, Vc)
                    cand = fn().clone(); torch.cuda.synchronize()
                else:
                    continue
                bit = int(torch.equal(cand, ref))
                mad = float((cand.float() - ref.float()).abs().max())
                am = int(int(cand.reshape(-1).argmax()) == ref_arg)
                a["bitwise_hits"] += bit; a["argmax_hits"] += am; a["n"] += 1
                a["max_abs_diff"] = max(a["max_abs_diff"], mad)
                # latency: measure once on the first draw (data-independent timing).
                # Still inside the tile override (if any) so unified_tile is timed
                # with its alt tile; the finally below always restores it.
                if di == 0:
                    m = measure(fn, served_bytes, args.reps_graph, args.warmup,
                                args.iters, args.repeats)
                    a["latency_ms"] = m["min_ms"]; a["bw_GBs"] = m["bw_gbps"]
                    if name == "served_3d_rerun":
                        served_lat = m["min_ms"]
            except Exception as exc:  # noqa: BLE001
                a["error"] = repr(exc)[:200]
            finally:
                if restore is not None:
                    restore()   # never leak the tile override into the next draw
        del compact

    # served latency reference (the 3d rerun == served kernel)
    if served_lat is None:
        served_lat = stage1_info["attn_sdpa_latency_ms"]
    faster_thr = served_lat * (1.0 - FASTER_MARGIN_FRAC)
    for name, a in agg.items():
        if a["n"]:
            a["bitwise_rate"] = a["bitwise_hits"] / a["n"]
            a["argmax_rate"] = a["argmax_hits"] / a["n"]
        else:
            a["bitwise_rate"] = float("nan"); a["argmax_rate"] = float("nan")
        a["faster_than_served"] = bool(a["latency_ms"] is not None and a["latency_ms"] < faster_thr)

    # GREEN lever = bit-identical (rate 1.0) AND materially faster than served,
    # excluding the served rerun itself.
    green = {n: a for n, a in agg.items()
             if n not in ("served_3d_rerun",) and a.get("bitwise_rate") == 1.0
             and a["faster_than_served"]}
    # self-determinism: served rerun must be bit-identical to served reference
    self_det = agg["served_3d_rerun"].get("bitwise_rate", float("nan"))
    # byte-identical wall = fastest bit-identical config's BW. Only the served
    # reduction order is bit-identical; the wall is the served BW.
    bitwise_safe = {n: a for n, a in agg.items()
                    if a.get("bitwise_rate") == 1.0 and a.get("bw_GBs") is not None}
    wall_bw = max((a["bw_GBs"] for a in bitwise_safe.values()),
                  default=stage1_info["attn_sdpa_achievable_bw_GBs"])
    n_faster = sum(1 for a in agg.values() if a["faster_than_served"])
    max_abs_overall = max((a["max_abs_diff"] for a in agg.values()
                           if a.get("n", 0)), default=0.0)
    return {
        "candidates": agg, "served_latency_ms": served_lat,
        "probe_L": L, "probe_L_eff": L_eff, "served_bytes": served_bytes,
        "n_draws_per_candidate": args.n_draws,
        "self_det_bitwise": self_det,
        "n_faster_attn_configs": n_faster,
        "attn_bitwise_green_configs": sorted(green.keys()),
        "attn_byte_identical_wall_GBs": wall_bw,
        "attn_byte_identical_lever_is_green": bool(green),
        "attn_max_abs_diff": max_abs_overall,
        "faithfulness": faithful,
        "served_ref_error": served_ref_error,
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--inspect", action="store_true",
                    help="load + capture + dump architecture, then exit (no stages)")
    ap.add_argument("--smoke", action="store_true", help="tiny N/L for a fast validity pass")
    ap.add_argument("--model-dir", default=LOCAL_CKPT)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--prompt-tokens", type=int, default=1024,
                    help="realistic decode KV length to capture at")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--reps-graph", type=int, default=20)
    ap.add_argument("--n-draws", type=int, default=24)
    ap.add_argument("--probe-L", type=int, default=1024)
    ap.add_argument("--l-sweep", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--out", default=str(HERE / "attention_byte_identical_ceiling.json"))
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="denken/attention-byte-identical-ceiling")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="attention-byte-identical-ceiling")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if args.smoke:
        args.prompt_tokens = 256; args.iters = 12; args.warmup = 5; args.repeats = 4
        args.reps_graph = 6; args.n_draws = 3; args.probe_L = 256
        args.l_sweep = [128, 256]

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import torch
    from vllm import LLM, SamplingParams

    dev = torch.device("cuda:0")
    print(f"[load] base_fullhead {args.model_dir}", flush=True)
    t0 = time.time()
    llm = LLM(model=args.model_dir, quantization="compressed-tensors", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=args.gpu_mem_util, max_num_seqs=1,
              enable_prefix_caching=False, enforce_eager=True, trust_remote_code=True)
    print(f"[load] done {time.time()-t0:.1f}s", flush=True)

    mr = get_model_runner(llm)
    arch = enumerate_attention(mr)
    print(f"[arch] {arch['n_attn_layers']} attn layers; impl={arch['impl_classes']}; "
          f"head_size classes={arch['head_size_classes']}", flush=True)
    for hs, info in arch["by_head_size"].items():
        ex = info["example"]
        print(f"   hs={hs}: {info['n_layers']} layers, nheads={ex['num_heads']} "
              f"nkv={ex['num_kv_heads']} scale={ex['scale']:.5f} softcap={ex['logits_soft_cap']} "
              f"window={ex['sliding_window']} kv_dtype={ex['kv_cache_dtype']}", flush=True)
    triton_forced = arch["impl_classes"] == ["TritonAttentionImpl"]

    # ---- capture the real served unified_attention call per head_size ----
    captures: dict = {}
    cap_iters = 8 if not args.smoke else 4
    ta_mod, orig_uatt = install_capture(captures, want_iters=cap_iters * 4, want_warmup=cap_iters)
    tok = llm.get_tokenizer()
    base = ("The history of computing spans many centuries of human ingenuity, from the "
            "earliest counting tools through mechanical calculators to modern processors. ")
    ids = tok.encode(base)
    while len(ids) < args.prompt_tokens:
        ids = ids + ids
    ids = ids[:args.prompt_tokens]
    print(f"[capture] decode at realistic L~{len(ids)} (single stream) ...", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=4, min_tokens=4)
    try:
        llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)
    finally:
        ta_mod.unified_attention = orig_uatt
    cap_keys = [k for k in captures.keys() if isinstance(k, int)]
    print(f"[capture] captured head_size classes: {sorted(cap_keys)} "
          f"errors={captures.get('_errors')}", flush=True)
    for hs in sorted(cap_keys):
        c = captures[hs]
        print(f"   hs={hs}: real_L={c['real_L']} nkv={c['num_kv_heads']} H={c['num_query_heads']} "
              f"scale={c['scale']:.5f} softcap={c['softcap']} window={c['window_size']} "
              f"seg={c['num_par_softmax_segments']} thr={c['seq_threshold_3D']} "
              f"kv_dtype={c['k_dtype']} live_min_ms={c['live_min_ms']:.4f}", flush=True)

    layout = get_layout()
    print(f"[layout] kv_cache_layout={layout}", flush=True)

    if args.inspect:
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        dump = {"arch": arch, "layout": layout, "triton_forced": triton_forced,
                "captured": {hs: {k: v for k, v in captures[hs].items()
                                  if k not in ("Kc", "Vc", "q", "live_out")} for hs in cap_keys}}
        Path(args.out + ".inspect.json").write_text(json.dumps(dump, indent=2, default=str))
        print(f"[inspect] wrote {args.out}.inspect.json peak_vram={peak:.2f}GiB", flush=True)
        return 0

    # =================== run stages per captured head_size class ============ #
    stage_results = {}
    for hs in sorted(cap_keys):
        cfg = captures[hs]
        print(f"\n[stage1] head_size={hs} served Triton SDPA BW/latency ...", flush=True)
        s1 = stage1(cfg, layout, args, dev)
        print(f"   served BW @L={s1['headline_L']}: {s1['attn_sdpa_achievable_bw_GBs']:.1f} GB/s "
              f"({s1['attn_sdpa_pct_of_peak']:.1f}% peak), latency={s1['attn_sdpa_latency_ms']*1e3:.1f}us; "
              f"live@realL {s1['live_achievable_bw_GBs']:.1f} GB/s", flush=True)
        print(f"[stage2] head_size={hs} bitwise-identity probe ...", flush=True)
        s2 = stage2(cfg, layout, args, dev, s1)
        for n, a in s2["candidates"].items():
            print(f"   {n:18s} bitwise={a.get('bitwise_rate')} argmax={a.get('argmax_rate')} "
                  f"maxΔ={a['max_abs_diff']:.4g} lat={a['latency_ms']} faster={a['faster_than_served']} "
                  f"err={a['error']}", flush=True)
        fa = s2.get("faithfulness", {})
        print(f"   -> wall={s2['attn_byte_identical_wall_GBs']:.1f} GB/s "
              f"green={s2['attn_byte_identical_lever_is_green']} "
              f"green_cfgs={s2['attn_bitwise_green_configs']} self_det={s2['self_det_bitwise']} "
              f"faithful(rerun==live)={fa.get('bitwise')} maxΔ={fa.get('max_abs_diff')}",
              flush=True)
        stage_results[hs] = {"stage1": s1, "stage2": s2}

    # =================== STAGE 3: per-component census + verdict ============ #
    attn_green = any(stage_results[hs]["stage2"]["attn_byte_identical_lever_is_green"]
                     for hs in cap_keys)
    decode_step_lever = bool(HEAD_BW_LEVER_GREEN_555 or BODY_BYTE_IDENTICAL_LEVER_GREEN_550 or attn_green)
    # representative attention wall = min BW across classes (the binding ceiling),
    # reported per-class in stage_results.
    walls = {hs: stage_results[hs]["stage2"]["attn_byte_identical_wall_GBs"] for hs in cap_keys}
    attn_wall_min = min(walls.values()) if walls else float("nan")
    self_dets = [stage_results[hs]["stage2"]["self_det_bitwise"] for hs in cap_keys]
    self_det = min([d for d in self_dets if isinstance(d, (int, float)) and math.isfinite(d)] or [0.0])

    if attn_green:
        verdict_line = ("FRESH ATTENTION LEVER FOUND: a faster bit-identical attention config "
                        "exists -> portability call required (vLLM-submittable?). Census head/body "
                        "closed, attention OPEN.")
        attn_portable = None  # filled per finding
    else:
        verdict_line = (
            "byte-identical decode ceiling is fully grounded, NO sliver in any component: "
            f"HEAD wall=served {HEAD_BYTE_IDENTICAL_WALL_GBS_555:.1f} GB/s (no sliver, #555) + "
            "BODY Marlin-locked (only w4a16 kernel on sm_86, no alternate to test, #550) + "
            f"ATTENTION served Triton SDPA at its byte-identical wall (every faster config -- 2D "
            "single-pass, seg8/seg32 splits, alt tile, torch-SDPA, fp32-math -- reorders the "
            "softmax/PV reduction -> bitwise_rate<1, land #506; FA2/flash blocked at head_dim 512 "
            "on sm_86, fern #507). decode_step_byte_identical_lever_exists=False across ALL 3 "
            "components -> the capstone can assert 'no byte-identical speed lever anywhere in the "
            "decode step' with every component measured my way.")
        attn_portable = None

    peak = torch.cuda.max_memory_allocated() / (1024**3)
    st_pass = self_test()

    # primary metric = the attention byte-identical wall (min across classes)
    verdict = {
        "pr": 562, "analysis_only": True, "official_tps": 0, "no_hf_job": True,
        "no_served_file_change": True,
        "device": torch.cuda.get_device_name(0),
        "sm": "".join(map(str, torch.cuda.get_device_capability(0))),
        "torch": torch.__version__, "vllm": __import__("vllm").__version__,
        "A10G_HBM_GBS": A10G_HBM_GBS, "kv_cache_layout": layout,
        "triton_attn_forced": triton_forced,
        "n_attn_layers": arch["n_attn_layers"], "impl_classes": arch["impl_classes"],
        "head_size_classes": arch["head_size_classes"],
        "captured_head_sizes": sorted(cap_keys),
        # ---- per-class headline (flatten primary numbers) ----
        **{f"hs{hs}_attn_sdpa_achievable_bw_GBs": stage_results[hs]["stage1"]["attn_sdpa_achievable_bw_GBs"] for hs in cap_keys},
        **{f"hs{hs}_attn_sdpa_pct_of_peak": stage_results[hs]["stage1"]["attn_sdpa_pct_of_peak"] for hs in cap_keys},
        **{f"hs{hs}_attn_sdpa_latency_ms": stage_results[hs]["stage1"]["attn_sdpa_latency_ms"] for hs in cap_keys},
        **{f"hs{hs}_attn_byte_identical_wall_GBs": stage_results[hs]["stage2"]["attn_byte_identical_wall_GBs"] for hs in cap_keys},
        **{f"hs{hs}_attn_lever_is_green": stage_results[hs]["stage2"]["attn_byte_identical_lever_is_green"] for hs in cap_keys},
        **{f"hs{hs}_n_faster_attn_configs": stage_results[hs]["stage2"]["n_faster_attn_configs"] for hs in cap_keys},
        **{f"hs{hs}_attn_max_abs_diff": stage_results[hs]["stage2"]["attn_max_abs_diff"] for hs in cap_keys},
        **{f"hs{hs}_rerun_equals_live": stage_results[hs]["stage2"]["faithfulness"].get("bitwise") for hs in cap_keys},
        # ---- census composition ----
        "head_byte_identical_wall_GBs_555": HEAD_BYTE_IDENTICAL_WALL_GBS_555,
        "head_bw_lever_is_green_555": HEAD_BW_LEVER_GREEN_555,
        "body_byte_identical_lever_is_green_550": BODY_BYTE_IDENTICAL_LEVER_GREEN_550,
        "attn_byte_identical_lever_is_green": attn_green,
        "attn_byte_identical_wall_GBs": attn_wall_min,
        "decode_step_byte_identical_lever_exists": decode_step_lever,
        "attn_lever_portable_to_vllm": attn_portable,
        "n_faster_attn_configs": sum(stage_results[hs]["stage2"]["n_faster_attn_configs"] for hs in cap_keys),
        "attn_bitwise_green_configs": sorted({c for hs in cap_keys
                                              for c in stage_results[hs]["stage2"]["attn_bitwise_green_configs"]}),
        "attn_max_abs_diff": max((stage_results[hs]["stage2"]["attn_max_abs_diff"] for hs in cap_keys), default=0.0),
        "self_det": self_det,
        "rerun_equals_live_all": all(
            stage_results[hs]["stage2"]["faithfulness"].get("bitwise") == 1 for hs in cap_keys),
        "faithfulness_by_hs": {hs: stage_results[hs]["stage2"]["faithfulness"] for hs in cap_keys},
        # ---- anchors ----
        "anchor_bfh_tps": BFH_TPS, "anchor_free_ceiling_554_tps": FREE_CEILING_TPS_554,
        "anchor_strict_ceiling_549_tps": STRICT_CEILING_TPS_549,
        "anchor_official_1_tps": OFFICIAL_1_TPS, "anchor_launch_floor_554_ms": LAUNCH_FLOOR_MS_554,
        "peak_vram_gib": peak, "self_test_passes": st_pass,
        "verdict_line": verdict_line,
        "primary_metric_name": "attn_byte_identical_wall_GBs",
        "primary_metric_value": attn_wall_min,
    }
    payload = {"verdict": verdict, "stages": stage_results, "arch": arch,
               "captured_meta": {hs: {k: v for k, v in captures[hs].items()
                                      if k not in ("Kc", "Vc", "q", "live_out")} for hs in cap_keys},
               "config": vars(args)}
    Path(args.out).write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n[done] wrote {args.out} peak_vram={peak:.2f}GiB", flush=True)
    print(f"[verdict] {verdict_line}", flush=True)

    rid = None
    if not args.no_wandb:
        rid = log_wandb(args, payload)
        verdict["wandb_run_id"] = rid

    print(
        "SENPAI-RESULT analysis_only=true official_tps=0 "
        f"attn_byte_identical_wall_GBs={attn_wall_min:.2f} "
        f"attn_byte_identical_lever_is_green={int(attn_green)} "
        f"decode_step_byte_identical_lever_exists={int(decode_step_lever)} "
        f"n_faster_attn_configs={verdict['n_faster_attn_configs']} "
        f"attn_max_abs_diff={verdict['attn_max_abs_diff']:.4g} "
        f"head_bw_lever_is_green_555={int(HEAD_BW_LEVER_GREEN_555)} "
        f"body_lever_is_green_550={int(BODY_BYTE_IDENTICAL_LEVER_GREEN_550)} "
        f"triton_attn_forced={int(triton_forced)} "
        f"self_det={self_det:.4f} self_test_passes={int(st_pass)} "
        f"peak_gib={peak:.2f} primary_metric={attn_wall_min:.2f} "
        f"wandb_run_id={rid}", flush=True)
    return 0 if st_pass else 1


def log_wandb(args, payload):
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_json_artifact, log_summary)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] unavailable: {exc}", flush=True)
        return None
    v = payload["verdict"]
    run = init_wandb_run(
        job_type="analysis", agent="denken", name=args.wandb_name, group=args.wandb_group,
        notes="PR #562: MEASURED vLLM forced-Triton SDPA byte-identical attention ceiling at M=1, "
              "completing the per-component decode census (head #555 + body #550 + attention here). "
              "Stage1 served Triton SDPA achievable BW/latency per head_dim class; Stage2 BITWISE "
              "probe of every faster reduction-order variant (2D, seg8/32, alt tile, torch-SDPA, "
              "fp32-math) vs the served 3D split-softmax output -- the load-bearing leg. LOCAL "
              "analysis_only, no HF job.",
        tags=["byte-exact", "attention-roofline", "triton-sdpa", "hbm-bandwidth",
              "kernel-selection", "pr-562", "attention-byte-identical-ceiling", "analysis-only",
              "local-only", "bitwise-identity", "land-506", "per-component-census", "het-head-dim"],
        config={"pr": 562, "wandb_group": args.wandb_group, "analysis_only": True, "official_tps": 0,
                "model_id": "google/gemma-4-E4B-it (int4 w4a16 base_fullhead, forced Triton SDPA)",
                "hardware": f"{v['device']} sm_{v['sm']}",
                "baseline_base_fullhead_tps": BFH_TPS,
                "free_ceiling_554_tps": FREE_CEILING_TPS_554},
    )
    if run is None:
        print("[wandb] no run (no key / disabled)", flush=True)
        return None
    flat = {k: val for k, val in v.items() if isinstance(val, (int, float, bool, str))}
    log_summary(run, flat, step=0)
    log_json_artifact(run, name="attention_byte_identical_ceiling", artifact_type="analysis", data=payload)
    rid = getattr(run, "id", None)
    print(f"[wandb] run id: {rid}", flush=True)
    finish_wandb(run)
    return rid


if __name__ == "__main__":
    raise SystemExit(main())
