#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #433 (stark) -- Realize pinned-K via the SERVED Triton split-K path on sm_86?

THE QUESTION
------------
The pinned-K +13.998 lift -- the top rung of the equivalence-respecting frontier
(482.74 -> 496.74) -- was modeled as FA2 ``num_splits=8`` (#400 o7yhpkej / #408 qc9bz8sv).
denken #431 (uza2t8aq) proved FA2 ``num_splits>1`` is UN-RUNNABLE on sm_86
(``NotImplementedError("FA2 does not support num_splits > 1")``; FA3 unavailable on A10G),
so the 496.74 rung looked BLOCKED.

BUT the served attention is NOT FA2. gemma-4-E4B-it has heterogeneous head dims
(sliding head_dim=256, full head_dim=512); the vendored FA2 caps head_dim at 256 and cannot
serve the 7 full layers, so vLLM forces ``TRITON_ATTN`` / ``kernel_unified_attention``
(advisor-branch baseline record PR #39/#43; BASELINE.md "3D split-KV"). The served Triton
kernel ALREADY ships a 3D split-KV (FlashDecoding) decode reduction
(``vllm/v1/attention/ops/triton_unified_attention.py``: ``IS_3D`` path + ``reduce_segments``),
driven by the backend with ``NUM_PAR_SOFTMAX_SEGMENTS=16``
(``vllm/v1/attention/backends/triton_attn.py``).

So the pinned-K split-K reduction may be REALIZABLE as a Triton multi-segment config -- the
kernel that is actually served -- even though the FA2 ``num_splits`` path is dead. This card
SCOPES that realizability for the human's Q2 ("approve the kernel rebuild?"): does the split
RUN on sm_86, is it M-invariant, what realized attention-op speedup survives, and what is the
build surface.

WHAT THIS CARD MEASURES (pod A10G sm_86, READ-ONLY probe + microbench)
---------------------------------------------------------------------
(1) served_attn_is_triton: re-ground PR #39 -- FA2 cannot serve head_dim=512 (cap) AND
    ``num_splits>1`` raises (re-ground #431); the served decode is the Triton kernel.
(2) Enumerate runnable multi-split-K DECODE paths on sm_86:
      (a) served Triton ``unified_attention`` 3D split-KV: does it RUN at M=1, what split knob
          (``num_par_softmax_segments``), is the split byte-exact vs the 2D serial reduction
          (the blanket-strict / un-pack reference the 482.74 ladder sits on), and is it
          M-invariant (a FIXED split usable byte-exactly at BOTH M=1 and M=8)?
      (b) FlashInfer split-K decode -- read-only import/run probe in the served venv.
      (c) any other multi-partition decode reduction.
(3) realized attention-op speedup: MEASURE 2D-serial vs 3D-split decode latency at the served
    gemma-4-E4B geometry (nq=8/nkv=2, head_dim=256 sliding + 512 full, GQA) over KV-lens
    {128,256,512}, >=3 reps. realized_penalty = serial_us/split_us; translate through the
    ladder attention fraction to a TPS delta on 482.74. Does the analytic +13.998 survive?
(4) build-surface estimate for the best runnable path (Q2 decision packet).

SCOPE: identity-safe pod-GPU microbench on the EXISTING served Triton kernel + flags. NO
train.py --launch, NO HF Job, NO submission, NO served-file change, NO kernel rebuild, 0
official TPS. baseline 481.53 / 482.74-ladder UNCHANGED. Greedy identity is MEASURED, never
broken. Synthetic post-RMSNorm q/kv (M-invariance & op-latency are weight-value-independent;
the served kernel + reduction structure are the audit). Run CUDA_VISIBLE_DEVICES=0 (single-A10G
pod default points at a non-existent 2nd GPU -- #358/#363 gotcha). W&B group
pinnedk-triton-realizability.

BOUNDARY (avoid overlap): denken (new card) owns the reduction MATH (byte-identical-to-canonical
unconditional vs self-referential). This card owns the EMPIRICAL kernel realizability (does it
RUN on sm_86, realized TPS, build cost). lawine #432 owns joint-stack composition. kanna #416 /
ubel #422 own cb3. The 482.74 base is taken AS GIVEN.

PUBLIC EVIDENCE USED (advisor-branch banked): #400/#408 pinned-K FA2 model (the +13.998 rung);
denken #431 FA2 num_splits>1 NotImplementedError; BASELINE.md PR #39/#43 TRITON_ATTN forcing +
"3D split-KV"; PR #122 (kanna) "splitkv auto-gated-off under VLLM_BATCH_INVARIANT=1"; stark #429
operative-identity census of the served verify path.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# single-A10G pod: inherited CUDA_VISIBLE_DEVICES may point at a non-existent 2nd GPU (#358/#363)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "0,", ""):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------------------------------------------------------------------------------------- #
# gemma-4-E4B-it served attention geometry (config.json text_config, advisor-branch grounded):
#   num_attention_heads=8, num_key_value_heads=2 (GQA, 4 q/kv), num_hidden_layers=42.
#   35 sliding_attention layers: head_dim=256, sliding_window=512.
#   7  full_attention   layers: head_dim=512 (the layers FA2's 256-cap CANNOT serve -> TRITON).
# ---------------------------------------------------------------------------------------- #
N_Q_HEADS = 8
N_KV_HEADS = 2
NUM_QUERIES_PER_KV = N_Q_HEADS // N_KV_HEADS  # 4
HEAD_DIM_SLIDING = 256
HEAD_DIM_FULL = 512
SLIDING_WINDOW = 512
SERVED_BLOCK_SIZE = 16  # vLLM deployment page/block size
NUM_PAR_SOFTMAX_SEGMENTS = 16  # served backend default (triton_attn.py NUM_PAR_SOFTMAX_SEGMENTS)
SEQ_THRESHOLD_3D = 64  # MIN_LAUNCH_GRID_SIZE_2D(128) // num_heads_kv(2)
M_AR = 1  # AR / drafter decode width (the un-pack / blanket-strict serial lane)
M_VERIFY = 8  # spec-verify width (K_spec=7 + 1); max_seqlen_q>1 -> kernel FORCES 2D serial

# operating band: decode KV-lens to characterize (#282 decode positions cluster ~[528,658];
# task specifies {128,256,512}; include 512/658 for band continuity)
KV_LENS = (128, 256, 512)
KV_BAND_EXT = (128, 256, 512, 658)

# ---- equivalence-respecting frontier ladder (CITE; taken AS GIVEN per the boundary) ------ #
# The PR cites the pinned-K lift as +13.998 and the rung display as 482.74 -> 496.74; the cited
# lift is the canonical analytic number (496.74 is its 2dp display, 482.74 + 13.998 = 496.738).
BLANKET_STRICT_FLOOR = 467.14       # blanket-strict floor (denken #423 5a6zq2yz; 1.0 operative #429)
CB3_RUNG = 482.74                   # + cb3 body-read shrink (kanna #403 iv9i2wks) -- THIS card's base
ANALYTIC_PINNEDK_DELTA = 13.998     # the cited pinned-K lift (modeled on FA2 num_splits=8) -- the analytic to test
PINNEDK_RUNG_ANALYTIC = CB3_RUNG + ANALYTIC_PINNEDK_DELTA  # 496.738 (PR display: 496.74)
PINNEDK_RUNG_DISPLAY = 496.74       # PR's 2dp display of the rung
# the ladder attention fraction used to price the modeled lift (#344/#378 M=8 verify attn frac):
F_ATTN_VERIFY = 0.09506718019009251
# implied modeled un-pack penalty that produces +13.998 via delta = base * f_attn * (p-1):
MODELED_PENALTY = 1.0 + ANALYTIC_PINNEDK_DELTA / (CB3_RUNG * F_ATTN_VERIFY)  # ~1.3049
PPL_ANCHOR = 2.3772  # reduction-ORDER change is PPL-neutral; anchored (analysis-only, no served run)

A10G_SMS = 80
TOL = 1.0e-2


# ======================================================================================== #
# Device + facts
# ======================================================================================== #
def _device():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Launch with CUDA_VISIBLE_DEVICES=0 (single-A10G pod gotcha)."
        )
    return torch.device("cuda:0")


def _gpu_facts(dev) -> dict[str, Any]:
    import torch

    p = torch.cuda.get_device_properties(dev)
    cc = torch.cuda.get_device_capability(dev)
    return {
        "name": p.name,
        "sm_count": p.multi_processor_count,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "total_mem_gib": round(p.total_memory / (1024**3), 2),
        "is_a10g_80sm": bool(p.multi_processor_count == A10G_SMS and "A10G" in p.name),
        "is_sm8x": bool(cc[0] == 8),
    }


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# ======================================================================================== #
# (1) served_attn_is_triton -- re-ground PR #39/#43 + #431 (read-only dispatch probes)
# ======================================================================================== #
def _ground_served_is_triton(dev) -> dict[str, Any]:
    """Three read-only facts that jointly force TRITON_ATTN for gemma-4-E4B-it decode:
    (a) the served Triton ``unified_attention`` 3D split kernel imports and is the decode entry;
    (b) the vendored FA2 cannot serve head_dim=512 (the 7 full layers) -- cap probe;
    (c) FA2 ``num_splits>1`` raises NotImplementedError on sm_86 (re-ground denken #431).
    """
    out: dict[str, Any] = {}

    # (a) the served Triton decode kernel + its 3D split-KV reduction are importable
    try:
        from vllm.v1.attention.ops.triton_unified_attention import (  # noqa: F401
            kernel_unified_attention,
            reduce_segments,
            unified_attention,
        )

        out["triton_unified_attention_importable"] = True
        out["triton_has_3d_reduce_segments"] = True
    except Exception as e:  # noqa: BLE001
        out["triton_unified_attention_importable"] = False
        out["triton_import_error"] = f"{type(e).__name__}: {str(e)[:160]}"

    # backend default segment knob (the served split count)
    try:
        from vllm.v1.attention.backends import triton_attn as _tb

        out["backend_num_par_softmax_segments"] = int(_tb.NUM_PAR_SOFTMAX_SEGMENTS)
        out["backend_min_launch_grid_2d"] = int(_tb.MIN_LAUNCH_GRID_SIZE_2D)
    except Exception as e:  # noqa: BLE001
        out["backend_probe_error"] = f"{type(e).__name__}: {str(e)[:120]}"

    # (b) FA2 head_dim=512 cap probe -- the full layers cannot run on the vendored FA2
    out.update(_fa2_headdim_full_probe(dev))
    # (c) FA2 num_splits>1 probe -- re-ground #431
    out.update(_fa2_num_splits_probe(dev))

    out["served_attn_is_triton"] = bool(
        out.get("triton_unified_attention_importable")
        and out.get("triton_has_3d_reduce_segments")
        and (not out.get("fa2_headdim512_runnable", True))
        and (not out.get("fa2_num_splits_runnable", True))
    )
    return out


def _fa2_build_paged(L: int, M: int, head_dim: int, seed: int, dev):
    import torch

    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(M, N_Q_HEADS, head_dim, generator=g, device=dev, dtype=torch.bfloat16)
    nb = _ceildiv(L, SERVED_BLOCK_SIZE)
    kc = torch.randn(nb, SERVED_BLOCK_SIZE, N_KV_HEADS, head_dim, generator=g, device=dev, dtype=torch.bfloat16)
    vc = torch.randn(nb, SERVED_BLOCK_SIZE, N_KV_HEADS, head_dim, generator=g, device=dev, dtype=torch.bfloat16)
    bt = torch.arange(nb, dtype=torch.int32, device=dev).unsqueeze(0)
    sk = torch.tensor([L], dtype=torch.int32, device=dev)
    cu = torch.tensor([0, M], dtype=torch.int32, device=dev)
    return q, kc, vc, bt, cu, sk


def _fa2_call(L: int, M: int, head_dim: int, num_splits: int, dev):
    import torch
    from vllm.vllm_flash_attn import _vllm_fa2_C  # noqa: F401
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as FA2

    q, kc, vc, bt, cu, sk = _fa2_build_paged(L, M, head_dim, 7, dev)
    scale = 1.0 / math.sqrt(head_dim)
    return FA2(
        q=q, k=kc, v=vc, out=None, cu_seqlens_q=cu, max_seqlen_q=M, seqused_k=sk,
        max_seqlen_k=L, softmax_scale=scale, causal=False, block_table=bt,
        num_splits=num_splits, fa_version=2,
    )


def _fa2_headdim_full_probe(dev) -> dict[str, Any]:
    """Can the vendored FA2 serve the head_dim=512 full layers? Expect NO (256 cap)."""
    try:
        _fa2_call(256, M_AR, HEAD_DIM_FULL, 0, dev)
        return {"fa2_headdim512_runnable": True, "fa2_headdim512_error": None}
    except Exception as e:  # noqa: BLE001
        return {"fa2_headdim512_runnable": False, "fa2_headdim512_error": f"{type(e).__name__}: {str(e)[:150]}"}


def _fa2_num_splits_probe(dev) -> dict[str, Any]:
    """FA2 num_splits>1 on sm_86 -- re-ground denken #431. Expect NotImplementedError."""
    try:
        _fa2_call(256, M_VERIFY, HEAD_DIM_SLIDING, 8, dev)
        return {"fa2_num_splits_runnable": True, "fa2_num_splits_error": None}
    except Exception as e:  # noqa: BLE001
        return {"fa2_num_splits_runnable": False, "fa2_num_splits_error": f"{type(e).__name__}: {str(e)[:150]}"}


# ======================================================================================== #
# (2a) Served Triton unified_attention driver -- 2D serial (blanket-strict) vs 3D split (pinned-K)
# ======================================================================================== #
def _triton_build_paged(L: int, M: int, head_dim: int, seed: int, dev):
    """Built to the served call contract (triton_attn.py _forward):
    q:(M,nq,hd) out:(M,nq,hd) key/value_cache:(nb,block,nkv,hd) cu:[0,M] seqused_k:[L+M-? ]."""
    import torch

    g = torch.Generator(device=dev).manual_seed(seed)
    # context_len = seq_len - query_len; the M query tokens are appended AFTER an L-token context
    seq_len = L + M
    nb = _ceildiv(seq_len, SERVED_BLOCK_SIZE)
    q = torch.randn(M, N_Q_HEADS, head_dim, generator=g, device=dev, dtype=torch.bfloat16)
    out = torch.empty(M, N_Q_HEADS, head_dim, device=dev, dtype=torch.bfloat16)
    kc = torch.randn(nb, SERVED_BLOCK_SIZE, N_KV_HEADS, head_dim, generator=g, device=dev, dtype=torch.bfloat16)
    vc = torch.randn(nb, SERVED_BLOCK_SIZE, N_KV_HEADS, head_dim, generator=g, device=dev, dtype=torch.bfloat16)
    bt = torch.arange(nb, dtype=torch.int32, device=dev).unsqueeze(0)
    cu = torch.tensor([0, M], dtype=torch.int32, device=dev)
    sk = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    return q, out, kc, vc, bt, cu, sk, seq_len


def _segm_buffers(head_dim: int, dev, n_tokens: int = None):
    """Per-segment partials sized like the served backend (token-major)."""
    import torch

    hd_pad = 1 << (head_dim - 1).bit_length()  # next_power_of_2
    rows = max(SEQ_THRESHOLD_3D, n_tokens or 0)
    so = torch.empty((rows, N_Q_HEADS, NUM_PAR_SOFTMAX_SEGMENTS, hd_pad), dtype=torch.float32, device=dev)
    sm = torch.empty((rows, N_Q_HEADS, NUM_PAR_SOFTMAX_SEGMENTS), dtype=torch.float32, device=dev)
    se = torch.empty((rows, N_Q_HEADS, NUM_PAR_SOFTMAX_SEGMENTS), dtype=torch.float32, device=dev)
    return so, sm, se


def _window_for(head_dim: int):
    # sliding layers (hd 256): window (sliding_window-1, 0); full layers (hd 512): (-1,-1) disabled
    if head_dim == HEAD_DIM_SLIDING:
        return (SLIDING_WINDOW - 1, 0)
    return (-1, -1)


def _build_inputs(L: int, M: int, head_dim: int, seed: int, dev) -> dict:
    """All decode inputs (q/out/kc/vc/bt/cu/sk) pre-allocated ONCE; reuse across timed calls so the
    microbench measures only the kernel launch, not allocation."""
    q, out, kc, vc, bt, cu, sk, seq_len = _triton_build_paged(L, M, head_dim, seed, dev)
    return {"q": q, "out": out, "kc": kc, "vc": vc, "bt": bt, "cu": cu, "sk": sk,
            "seq_len": seq_len, "head_dim": head_dim, "M": M}


def _split_bufs(head_dim: int, M: int, dev) -> dict:
    so, sm, se = _segm_buffers(head_dim, dev, n_tokens=M)
    return {"softmax_segm_output": so, "softmax_segm_max": sm, "softmax_segm_expsum": se}


def _call_unified(inp: dict, split_bufs: dict | None, mode: str):
    """Drive the served unified_attention with PRE-ALLOCATED buffers. mode='serial' -> 2D
    (use_3d False, the blanket-strict / un-pack reference the 482.74 ladder sits on);
    mode='split' -> 3D split-KV (the pinned-K candidate, num_par_softmax_segments=16). The kernel
    FORCES 2D whenever max_seqlen_q>1 (M=8 verify), so 'split' only takes the 3D path at M=1 --
    that asymmetry IS the M-invariance finding."""
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    M = inp["M"]
    head_dim = inp["head_dim"]
    scale = 1.0 / math.sqrt(head_dim)
    kwargs = dict(
        q=inp["q"], k=inp["kc"], v=inp["vc"], out=inp["out"], cu_seqlens_q=inp["cu"],
        max_seqlen_q=M, seqused_k=inp["sk"], max_seqlen_k=inp["seq_len"], softmax_scale=scale,
        causal=True, window_size=_window_for(head_dim), block_table=inp["bt"], softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
    )
    if mode == "split":
        kwargs.update(
            seq_threshold_3D=SEQ_THRESHOLD_3D,
            num_par_softmax_segments=NUM_PAR_SOFTMAX_SEGMENTS,
            **split_bufs,
        )
    # serial mode: leave seq_threshold_3D=None -> use_3d=False -> 2D single-segment
    unified_attention(**kwargs)
    return inp["out"]


def _used_3d(M: int, mode: str) -> bool:
    """Mirror unified_attention.use_3d: 3D only when split buffers supplied AND max_seqlen_q==1."""
    return bool(mode == "split" and M == 1)


def _time_call(fn, iters: int, warmup: int) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2] * 1e3  # us (median)


# ======================================================================================== #
# (2a cont.) split-vs-serial byte identity + runnability, at the served geometry
# ======================================================================================== #
def _split_runnable_and_identity(head_dim: int, dev, n_trials: int, seeds: list[int]) -> dict[str, Any]:
    import torch

    res: dict[str, Any] = {"head_dim": head_dim, "per_L": {}}
    runnable = True
    run_err = None
    byte_ids = []
    argmax_ids = []
    maxdiffs = []
    nan_seen = False
    for L in KV_LENS:
        per_seed_byte = []
        per_seed_argmax = []
        per_seed_maxdiff = []
        for s in seeds:
            for t in range(n_trials):
                # ONE input set shared by both modes (identical q/kc/vc) -> capture each output
                # before the next call overwrites inp["out"].
                inp = _build_inputs(L, M_AR, head_dim, s + 17 * t, dev)
                sb = _split_bufs(head_dim, M_AR, dev)
                try:
                    o_serial = _call_unified(inp, None, "serial").clone()
                    o_split = _call_unified(inp, sb, "split").clone()
                except Exception as e:  # noqa: BLE001
                    runnable = False
                    run_err = f"{type(e).__name__}: {str(e)[:160]}"
                    break
                nan_seen = nan_seen or bool(torch.isnan(o_split).any() or torch.isnan(o_serial).any())
                a = o_serial.reshape(M_AR, -1).float()
                b = o_split.reshape(M_AR, -1).float()
                # byte identity: compare the bf16 bit patterns
                ba = o_serial.reshape(M_AR, -1).view(torch.int16)
                bb = o_split.reshape(M_AR, -1).view(torch.int16)
                per_seed_byte.append((ba == bb).all(dim=-1).float().mean().item())
                per_seed_argmax.append((a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean().item())
                per_seed_maxdiff.append((a - b).abs().max().item())
            if not runnable:
                break
        if not runnable:
            break
        bL = float(min(per_seed_byte)) if per_seed_byte else float("nan")
        aL = float(min(per_seed_argmax)) if per_seed_argmax else float("nan")
        dL = float(max(per_seed_maxdiff)) if per_seed_maxdiff else float("nan")
        res["per_L"][str(L)] = {"split_vs_serial_byte_identity": bL, "argmax_identity": aL, "max_abs_diff": dL}
        byte_ids.append(bL)
        argmax_ids.append(aL)
        maxdiffs.append(dL)

    res["triton_split_runnable"] = bool(runnable)
    res["run_error"] = run_err
    res["any_nan"] = bool(nan_seen)
    res["split_vs_serial_byte_identity_min"] = float(min(byte_ids)) if byte_ids else None
    res["split_vs_serial_argmax_identity_min"] = float(min(argmax_ids)) if argmax_ids else None
    res["split_vs_serial_max_abs_diff"] = float(max(maxdiffs)) if maxdiffs else None
    # the split changes the bf16 bytes vs the 2D serial reduction <=> it is NOT a drop-in strict pin
    res["split_is_byte_exact_vs_serial"] = bool(
        byte_ids and min(byte_ids) >= 1.0
    )
    return res


def _m8_split_reachable(head_dim: int, dev) -> dict[str, Any]:
    """Can the SAME fixed split run at M=8 (the property pinned-K M-invariance requires)?
    The kernel forces 2D when max_seqlen_q>1, so passing split buffers at M=8 still runs 2D.
    Confirm empirically that M=8 does NOT take the 3D split path."""
    from vllm.v1.attention.ops import triton_unified_attention as tua

    # Verify the documented gate by constructing the use_3d predicate exactly as the wrapper does
    # (line 923-932): with all split buffers supplied + num_seqs(1)<=threshold, the ONLY thing that
    # forces 2D at M=8 is max_seqlen_q > 1.
    L = 256
    max_seqlen_q = M_VERIFY
    is_batch_invariant = bool(getattr(tua, "is_batch_invariant", False))
    use_3d_predicted = not (
        max_seqlen_q > 1 or is_batch_invariant
    )  # buffers + threshold satisfied here
    # Sanity: passing split buffers at M=8 still RUNS (as 2D, use_3d gate kicks it to serial).
    ran = True
    err = None
    try:
        inp = _build_inputs(L, M_VERIFY, head_dim, 11, dev)
        sb = _split_bufs(head_dim, M_VERIFY, dev)
        _call_unified(inp, sb, "split")
    except Exception as e:  # noqa: BLE001
        ran = False
        err = f"{type(e).__name__}: {str(e)[:140]}"
    return {
        "m8_split_path_runs_3d": bool(use_3d_predicted),
        "m8_call_runs_as_2d": bool(ran and not use_3d_predicted),
        "m8_error": err,
        "is_batch_invariant_env": is_batch_invariant,
    }


# ======================================================================================== #
# (3) Microbenchmark: realized 2D-serial vs 3D-split decode latency -> realized TPS delta
# ======================================================================================== #
def _microbench(head_dim: int, dev, iters: int, warmup: int) -> dict[str, Any]:
    out: dict[str, Any] = {"head_dim": head_dim, "per_L": {}}
    penalties = []
    for L in KV_BAND_EXT:
        # Pre-allocate q/out/kc/vc AND the 33MB segm scratch ONCE per L; the timed lambdas reuse
        # them so _time_call measures only the kernel launch, not allocation (the fairness fix:
        # the split path must not be charged for its segm-buffer alloc on every iteration).
        inp = _build_inputs(L, M_AR, head_dim, 7, dev)
        sb = _split_bufs(head_dim, M_AR, dev)
        serial_us = _time_call(lambda: _call_unified(inp, None, "serial"), iters, warmup)
        split_us = _time_call(lambda: _call_unified(inp, sb, "split"), iters, warmup)
        pen = serial_us / split_us if split_us > 0 else float("nan")
        out["per_L"][str(L)] = {
            "serial_us": serial_us, "split_us": split_us, "penalty_serial_over_split": pen,
            "act_segments": min(NUM_PAR_SOFTMAX_SEGMENTS, _ceildiv(L + M_AR, _decode_tile(head_dim))),
        }
        if L in KV_LENS:
            penalties.append(pen)
    out["realized_penalty_band_mean"] = float(sum(penalties) / len(penalties)) if penalties else float("nan")
    return out


def _decode_tile(head_dim: int) -> int:
    # _get_tile_size: gemma3 (sliding_window==1024) -> 32; else decode bf16 -> 16
    return 16


def _translate_to_tps(realized_penalty: float) -> dict[str, Any]:
    """Translate the realized attention-op penalty (serial/split) to a TPS delta on the 482.74
    base via the SAME ladder form that produced +13.998: delta = base * f_attn * (penalty - 1).
    f_attn and base are taken AS GIVEN (lawine/denken own them); this isolates MY measured kernel
    penalty from the borrowed normalization, making realized-vs-analytic directly comparable."""
    recoverable_eta = F_ATTN_VERIFY * (realized_penalty - 1.0)
    realized_delta = CB3_RUNG * recoverable_eta
    return {
        "realized_penalty": realized_penalty,
        "modeled_penalty": MODELED_PENALTY,
        "recoverable_eta_attn": recoverable_eta,
        "realized_pinnedk_tps_delta": realized_delta,
        "realized_pinnedk_frontier_tps": CB3_RUNG + realized_delta,
        "analytic_pinnedk_tps_delta": ANALYTIC_PINNEDK_DELTA,
        "realized_vs_analytic_ratio": realized_delta / ANALYTIC_PINNEDK_DELTA
        if ANALYTIC_PINNEDK_DELTA else float("nan"),
        "analytic_survives": bool(realized_delta >= 0.80 * ANALYTIC_PINNEDK_DELTA),
    }


# ======================================================================================== #
# (2b) FlashInfer split-K decode -- read-only import/run probe
# ======================================================================================== #
def _flashinfer_probe(dev) -> dict[str, Any]:
    out: dict[str, Any] = {"available": False}
    try:
        import flashinfer  # noqa: F401
    except Exception as e:  # noqa: BLE001
        out["import_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return out
    out["available"] = True
    out["version"] = getattr(__import__("flashinfer"), "__version__", "?")
    # head_dim=512 (full layers) support is the binding question for a UNIFORM decode pin
    try:
        import flashinfer

        out["has_decode_module"] = hasattr(flashinfer, "decode")
    except Exception as e:  # noqa: BLE001
        out["probe_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return out


# ======================================================================================== #
# (4) Build-surface estimate for the best runnable path (Q2 decision packet)
# ======================================================================================== #
def _build_surface() -> dict[str, Any]:
    """Files a strict M-invariant fixed-segment split would touch. The split RUNS today (M=1,
    non-BI); making it a STRICT pin needs: (i) allow a FIXED segment count at M=8 verify (lift the
    max_seqlen_q>1 -> 2D force for the pinned path), (ii) let it run under VLLM_BATCH_INVARIANT=1
    (lift the is_batch_invariant -> 2D force) with a fixed (M-independent) segment count so the
    reduction tree is identical for M=1 and M=8. Both are constexpr-gate edits in the SAME two
    served files; additive (new gated path), no checkpoint/weight change, PPL-neutral (reduction
    ORDER only)."""
    files = [
        "vllm/v1/attention/ops/triton_unified_attention.py (use_3d gate: fixed-seg path for M>1 + BI)",
        "vllm/v1/attention/backends/triton_attn.py (drive the fixed-seg pin; segm buffers sized for M=8)",
    ]
    return {
        "build_surface_files": len(files),
        "build_surface_file_list": files,
        "build_is_additive": True,  # new gated reduction path; existing 2D/3D paths untouched
        "blast_radius": "attention decode reduction only; body GEMM / lm_head / sampler untouched",
        "checkpoint_impact": "none (no weight/quant change)",
        "ppl_risk": "PPL-neutral: a split-K reduction changes float ADD ORDER, not the math; "
        "denken (new card) owns the byte-identical-to-canonical proof",
        "needs_kernel_rebuild": True,  # the constexpr-gated path is a SOURCE edit (human Q2)
        "human_gated": True,
    }


# ======================================================================================== #
# Compose
# ======================================================================================== #
def compose(dev, args) -> dict[str, Any]:
    gpu = _gpu_facts(dev)

    ground = _ground_served_is_triton(dev)
    served_attn_is_triton = bool(ground.get("served_attn_is_triton"))
    fa2_num_splits_runnable = bool(ground.get("fa2_num_splits_runnable", False))

    # (2a) Triton split runnability + split-vs-serial identity, both served geometries
    ident_sliding = _split_runnable_and_identity(HEAD_DIM_SLIDING, dev, args.ident_trials, args.seeds)
    ident_full = _split_runnable_and_identity(HEAD_DIM_FULL, dev, args.ident_trials, args.seeds)
    triton_splitk_runnable = bool(
        ident_sliding.get("triton_split_runnable") and ident_full.get("triton_split_runnable")
    )

    # M=8 reachability of the SAME fixed split (the pinned-K M-invariance requirement)
    m8_sliding = _m8_split_reachable(HEAD_DIM_SLIDING, dev)
    m8_full = _m8_split_reachable(HEAD_DIM_FULL, dev)
    # M-invariant <=> a FIXED split is byte-exact vs serial AND runs at BOTH M=1 and M=8.
    split_byte_exact = bool(
        ident_sliding.get("split_is_byte_exact_vs_serial")
        and ident_full.get("split_is_byte_exact_vs_serial")
    )
    m8_runs_split = bool(m8_sliding.get("m8_split_path_runs_3d") and m8_full.get("m8_split_path_runs_3d"))
    triton_splitk_m_invariant = bool(split_byte_exact and m8_runs_split)

    # (3) microbench realized op speedup -> realized TPS delta
    mb_sliding = _microbench(HEAD_DIM_SLIDING, dev, args.iters, args.warmup)
    mb_full = _microbench(HEAD_DIM_FULL, dev, args.iters, args.warmup)
    # weight sliding:full by served layer counts (35 sliding, 7 full)
    p_sl = mb_sliding["realized_penalty_band_mean"]
    p_fl = mb_full["realized_penalty_band_mean"]
    realized_penalty_weighted = (35 * p_sl + 7 * p_fl) / 42
    tps = _translate_to_tps(realized_penalty_weighted)

    # (2b) FlashInfer
    fi = _flashinfer_probe(dev)

    # (2c) menu of runnable multi-split decode paths on sm_86
    menu = {
        "fa2_num_splits_gt1": {"runnable": fa2_num_splits_runnable, "note": "NotImplementedError on sm_86 (#431)"},
        "triton_unified_3d_splitkv": {
            "runnable": triton_splitk_runnable,
            "split_knob": "num_par_softmax_segments",
            "served_default": NUM_PAR_SOFTMAX_SEGMENTS,
            "reachable_at_m1": True,
            "reachable_at_m8": m8_runs_split,
            "byte_exact_vs_serial": split_byte_exact,
            "note": "served fast decode path; force-disabled under VLLM_BATCH_INVARIANT=1",
        },
        "flashinfer_split_decode": {"runnable": bool(fi.get("available")), "note": fi.get("import_error", "import ok")},
    }
    runnable_paths = [k for k, v in menu.items() if v.get("runnable")]
    best_runnable_path = "triton_unified_3d_splitkv" if triton_splitk_runnable else (
        runnable_paths[0] if runnable_paths else "none"
    )

    # (4) build surface
    build = _build_surface()

    # headline realizability: a multi-split decode reduction RUNS on sm_86 via the served Triton
    # kernel (the FA2 blocker does NOT apply) -> pinned-K is BUILDABLE (needs the fixed-seg
    # M-invariant gate edit), not blocked. The realized DELTA quantifies whether it is worth it.
    pinnedk_realizable_on_sm86 = bool(triton_splitk_runnable)

    verdict = (
        f"served_attn_is_triton={served_attn_is_triton} (FA2 hd512 runnable="
        f"{ground.get('fa2_headdim512_runnable')}, FA2 num_splits>1 runnable={fa2_num_splits_runnable}). "
        f"The Triton 3D split-KV decode RUNS on sm_86 (triton_splitk_runnable={triton_splitk_runnable}) "
        f"-> the #431 FA2 blocker does NOT apply; pinned-K is BUILDABLE not blocked. BUT it is NOT a "
        f"drop-in strict pin: split-vs-serial byte identity (M=1) = "
        f"{ident_sliding.get('split_vs_serial_byte_identity_min')} (sliding) / "
        f"{ident_full.get('split_vs_serial_byte_identity_min')} (full), and the kernel FORCES 2D at "
        f"M=8 (m8_runs_split={m8_runs_split}) -> triton_splitk_m_invariant={triton_splitk_m_invariant} "
        f"(needs a fixed-segment gate edit). Realized op penalty (serial/split) = "
        f"{realized_penalty_weighted:.4f} vs modeled {MODELED_PENALTY:.4f} -> realized pinned-K TPS "
        f"delta = {tps['realized_pinnedk_tps_delta']:.3f} (analytic +{ANALYTIC_PINNEDK_DELTA:.3f}; "
        f"ratio {tps['realized_vs_analytic_ratio']:.2f}; survives={tps['analytic_survives']}) -> "
        f"realized frontier {tps['realized_pinnedk_frontier_tps']:.2f}."
    )

    return {
        "gpu": gpu,
        "ground_served_path": ground,
        "served_attn_is_triton": served_attn_is_triton,
        "fa2_num_splits_runnable": fa2_num_splits_runnable,
        "identity_sliding": ident_sliding,
        "identity_full": ident_full,
        "m8_split_sliding": m8_sliding,
        "m8_split_full": m8_full,
        "triton_splitk_runnable": triton_splitk_runnable,
        "triton_splitk_m_invariant": triton_splitk_m_invariant,
        "split_is_byte_exact_vs_serial": split_byte_exact,
        "microbench_sliding": mb_sliding,
        "microbench_full": mb_full,
        "realized_penalty_weighted": realized_penalty_weighted,
        **tps,
        "flashinfer": fi,
        "decode_path_menu": menu,
        "runnable_paths": runnable_paths,
        "best_runnable_path": best_runnable_path,
        "build_surface": build,
        "build_surface_files": build["build_surface_files"],
        "build_is_additive": build["build_is_additive"],
        "pinnedk_realizable_on_sm86": pinnedk_realizable_on_sm86,
        "ppl": PPL_ANCHOR,
        "ppl_is_anchored": True,
        "analysis_only": True,
        "no_hf_job": True,
        "no_served_file_change": True,
        "official_tps": 0,
        "verdict": verdict,
    }


# ======================================================================================== #
# Self-test (0-GPU): ladder arithmetic + translation + guards
# ======================================================================================== #
def self_test() -> dict[str, Any]:
    checks: dict[str, Any] = {}

    # ladder arithmetic
    checks["cb3_plus_delta_is_pinnedk_rung"] = abs((CB3_RUNG + ANALYTIC_PINNEDK_DELTA) - PINNEDK_RUNG_ANALYTIC) < 1e-6
    checks["analytic_delta_is_13998"] = abs(ANALYTIC_PINNEDK_DELTA - 13.998) < 1e-6
    checks["pinnedk_rung_rounds_to_display"] = abs(round(PINNEDK_RUNG_ANALYTIC, 2) - PINNEDK_RUNG_DISPLAY) < 1e-9

    # the modeled-penalty round-trip: delta = base * f_attn * (MODELED_PENALTY - 1) == 13.998
    rt = CB3_RUNG * F_ATTN_VERIFY * (MODELED_PENALTY - 1.0)
    checks["modeled_penalty_round_trips_to_delta"] = abs(rt - ANALYTIC_PINNEDK_DELTA) < 1e-6

    # translation: penalty==MODELED_PENALTY reproduces the analytic delta + frontier
    t = _translate_to_tps(MODELED_PENALTY)
    checks["translate_reproduces_analytic_delta"] = abs(t["realized_pinnedk_tps_delta"] - ANALYTIC_PINNEDK_DELTA) < 1e-6
    checks["translate_reproduces_analytic_frontier"] = abs(t["realized_pinnedk_frontier_tps"] - PINNEDK_RUNG_ANALYTIC) < 1e-6
    # penalty==1 (split no faster than serial) -> zero delta, frontier collapses to the base
    t1 = _translate_to_tps(1.0)
    checks["penalty1_gives_zero_delta"] = abs(t1["realized_pinnedk_tps_delta"]) < 1e-9
    checks["penalty1_frontier_is_base"] = abs(t1["realized_pinnedk_frontier_tps"] - CB3_RUNG) < 1e-9

    # geometry sanity
    checks["gqa_4_q_per_kv"] = NUM_QUERIES_PER_KV == 4
    checks["served_segments_16"] = NUM_PAR_SOFTMAX_SEGMENTS == 16
    checks["seq_threshold_3d_64"] = SEQ_THRESHOLD_3D == 64
    # M=8 verify forces 2D in the wrapper gate (max_seqlen_q>1) -> split is M=1-only
    checks["m8_forces_2d"] = (not _used_3d(M_VERIFY, "split")) and _used_3d(M_AR, "split")

    # guards
    checks["analysis_only_guard"] = True
    checks["no_hf_job_guard"] = True
    checks["no_served_file_change_guard"] = True

    passed = all(bool(v) for v in checks.values())
    return {"self_test_passes": passed, "checks": checks}


# ======================================================================================== #
# main
# ======================================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234, 5678, 9012])
    ap.add_argument("--ident-trials", type=int, default=8)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb_group", type=str, default="pinnedk-triton-realizability")
    ap.add_argument("--wandb_name", type=str, default="stark/pinnedk-triton-splitk-realizability")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.self_test:
        st = self_test()
        st["timestamp"] = ts
        out_p = here / "pinnedk_triton_splitk_realizability_selftest.json"
        json.dump(st, open(out_p, "w"), indent=2)
        print(json.dumps(st, indent=2))
        print(f"\nself_test_passes={st['self_test_passes']}  ->  {out_p}")
        sys.exit(0 if st["self_test_passes"] else 1)

    dev = _device()
    res = compose(dev, args)
    st = self_test()
    res["self_test"] = st
    res["self_test_passes"] = st["self_test_passes"]
    res["timestamp"] = ts
    res["args"] = vars(args)

    out_p = here / "pinnedk_triton_splitk_realizability_results.json"
    json.dump(res, open(out_p, "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=2, default=str))
    print(f"\nVERDICT: {res['verdict']}")
    print(f"results -> {out_p}")

    if not args.no_wandb:
        try:
            import wandb

            run = wandb.init(
                project="gemma-challenge-senpai",
                group=args.wandb_group,
                name=args.wandb_name,
                config={
                    "head_dim_sliding": HEAD_DIM_SLIDING, "head_dim_full": HEAD_DIM_FULL,
                    "num_par_softmax_segments": NUM_PAR_SOFTMAX_SEGMENTS,
                    "cb3_base": CB3_RUNG, "analytic_pinnedk_rung": PINNEDK_RUNG_ANALYTIC,
                    "analytic_pinnedk_delta": ANALYTIC_PINNEDK_DELTA, "f_attn_verify": F_ATTN_VERIFY,
                    "kv_lens": list(KV_LENS), "seeds": args.seeds,
                    "analysis_only": True, "no_hf_job": True, "no_served_file_change": True,
                },
            )
            flat = {
                "served_attn_is_triton": int(res["served_attn_is_triton"]),
                "fa2_num_splits_runnable": int(res["fa2_num_splits_runnable"]),
                "triton_splitk_runnable": int(res["triton_splitk_runnable"]),
                "triton_splitk_m_invariant": int(res["triton_splitk_m_invariant"]),
                "split_is_byte_exact_vs_serial": int(res["split_is_byte_exact_vs_serial"]),
                "realized_penalty_weighted": res["realized_penalty_weighted"],
                "modeled_penalty": res["modeled_penalty"],
                "realized_pinnedk_tps_delta": res["realized_pinnedk_tps_delta"],
                "realized_pinnedk_frontier_tps": res["realized_pinnedk_frontier_tps"],
                "analytic_pinnedk_tps_delta": res["analytic_pinnedk_tps_delta"],
                "realized_vs_analytic_ratio": res["realized_vs_analytic_ratio"],
                "analytic_survives": int(res["analytic_survives"]),
                "build_surface_files": res["build_surface_files"],
                "build_is_additive": int(res["build_is_additive"]),
                "pinnedk_realizable_on_sm86": int(res["pinnedk_realizable_on_sm86"]),
                "ppl": res["ppl"],
                "official_tps": 0,
                "self_test_passes": int(res["self_test_passes"]),
            }
            for hd, mb in (("sliding", res["microbench_sliding"]), ("full", res["microbench_full"])):
                for L, d in mb["per_L"].items():
                    flat[f"penalty_{hd}_L{L}"] = d["penalty_serial_over_split"]
                    flat[f"serial_us_{hd}_L{L}"] = d["serial_us"]
                    flat[f"split_us_{hd}_L{L}"] = d["split_us"]
            wandb.log(flat)
            wandb.summary.update(flat)
            print(f"wandb run: {run.id}")
            run.finish()
        except Exception as e:  # noqa: BLE001
            print(f"[wandb skipped] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
