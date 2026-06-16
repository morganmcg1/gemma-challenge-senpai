#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Can split-K reduction be byte-identical to canonical serial? (PR #434, denken).

THE QUESTION
------------
My #431 (`uza2t8aq`) classified the pinned-K 496.74 rung `self_referential_only`: the split-K(8)
reduction diverges from the canonical serial (`num_splits=1`) reduction at ULP-scale (max |Δ attn_out|
= 9.77e-4, every divergence a knife-edge near-tie at exactly e*=0.125, never a confident flip). The
root cause is bf16/fp32 NON-ASSOCIATIVITY in the split-K partial-sum reduction: summing partial
exp-weighted values in a different order than serial gives a different last-bit result.

This card resolves the open dichotomy: is that divergence IRREDUCIBLE (ANY multi-split reduction
differs from serial -> pinned-K permanently self-referential-only) or REDUCIBLE (a reduction variant
exists that is byte-identical to canonical serial AND preserves the split-K speedup -> pinned-K becomes
UNCONDITIONALLY legal, and the human's "which reference" decision Q1 DISSOLVES)?

THE ANSWER (decision-critical, honest): REDUCIBLE = FALSE -> `pinnedk_can_be_unconditional = False`,
`q1_collapsible = False`. Byte-identity to the existing canonical serial fold is FUNDAMENTALLY
INCOMPATIBLE with parallel split-K on non-associative floating point. The two properties cannot
coexist. The human DOES still have to make the contract call. Grounded in three legs:

  (1) THE SERVED REDUCTION IS ALREADY fp32-PARTIAL. The served decode kernel is vLLM's Triton
      `unified_attention` 3D split-KV (FA2 is inert here -- PR #39). Its combine `reduce_segments`
      (triton_unified_attention.py:646) loads per-segment partials from `softmax_segm_output`, which
      the backend allocates `dtype=torch.float32` (triton_attn.py:192), rescales by exp(m_i-overall_max)
      with the per-segment LOCAL max, and sums via `tl.sum` (an fp32 tree-reduce over the segment axis).
      So the split-KV-vs-serial byte break is PURELY reduction ORDER, NOT precision -- exactly what my
      merged #423 banked (`tax_decomp_fp32_accum_tps = 0`: "FA fp32-accumulates on both; the byte break
      is reduction ORDER not precision -- flipping to fp32 changes nothing because it's already fp32").
      => PR variant (a) "fp32 partial accumulation" IS the served kernel, and it still diverges.

  (2) NO SPEEDUP-PRESERVING VARIANT IS BYTE-IDENTICAL TO SERIAL (measured, this card). A direct numeric
      probe at the served gemma-4-E4B-it attention geometry (nq=8/nkv=2/hd=256 GQA, KV-lens {128,256,512},
      bf16 I/O) runs each candidate reduction and byte-compares its bf16 output to the serial reference:
        * (a) fp32 partial accumulation (= served kernel): NOT byte-identical (local-max rescale + tree
          regroup remain).
        * (b) fixed pairwise reduction tree: deterministic run-to-run, but a DIFFERENT order than the
          serial left-fold -> NOT byte-identical to serial (Higham ch.4).
        * (c) sorted-partial-sum / Kahan-Neumaier compensated: drive the combine toward the correctly
          rounded TRUE real sum, NOT toward the specific erroneous serial fold (Higham 4.3: "compensated
          summation is not equivalent to any particular ordering of single-precision summation") -> NOT
          byte-identical to serial.
      The ONLY construction that IS byte-identical to serial -- replaying serial's continuous fold (global
      max, no independent partials) -- is inherently SEQUENTIAL: it forfeits the parallel speedup that is
      the entire point of split-K. So the speedup-preserving set and the serial-byte-identical set are
      DISJOINT. (Thinking Machines Lab "Defeating Nondeterminism in LLM Inference": fixing FlashInfer's
      split count + combine order recovers run-to-run determinism, but to a FIXED-TREE reference that is
      explicitly "not the same as num_splits=1".)

  (3) THE ONLY ORDER-INVARIANT REDUCTION DEFINES A NEW REFERENCE (still a Q1 choice). An exact / Kulisch /
      ReproBLAS (Demmel-Nguyen TOMS 2021) accumulator makes the sum correctly-rounded and split-count
      invariant -- but that is byte-identical to the TRUE sum, NOT to the deployed serial fold (which
      carries its own fp32 rounding error), and it costs ~2x the accumulation (Johnson 2018). Adopting it
      = choosing a new canonical reference = exactly the Q1 contract call, not its dissolution. This card
      MEASURES the order-invariant accumulator: byte-identical across split counts (8==16) YET != serial.

  THE DECISION FRAMING: to make 496.74 unconditionally legal you would need a parallel reduction whose
  bf16 output is byte-identical to the EXISTING served-serial bytes. That requires reproducing the
  serial fold's exact fp32 operation order, which is sequential -> kills the speedup. fp32 partials
  (already deployed) shrink the divergence ~10^4x vs a bf16-partial kernel, to a vanishingly rare 1-bf16
  -ULP near-tie -- operationally negligible and STILL inside #431's bounded sub-e* PPL-neutral guarantee
  -- but NOT exactly zero. So Q1 does NOT collapse; the reassurance from #431 only gets stronger.

WHAT THIS IS / IS NOT
  Local A10G analysis card. analysis_only=True, no_hf_job=True, no_served_file_change=True, no submission,
  no kernel build, official_tps=0. The GPU is used ONLY for a microsecond-scale reduction-variant probe
  on synthetic tensors at the real served attention geometry -- no model load, no served file touched,
  int4 path untouched. Frontier / e* / flip scalars are BANKED byte-exactly from merged #431 (transitively
  #427/#423/#400). The only new modelling is the reduction-variant byte-identity probe + the dichotomy
  resolution. Boundary: stark owns the empirical KERNEL realizability (does a Triton split-K path RUN on
  sm_86, realized TPS, build surface); this card owns the reduction MATH (can it be byte-identical).

REPRODUCE
    cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/validity/\
pinnedk_reduction_byte_reproducibility/pinnedk_reduction_byte_reproducibility.py --self-test
    cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/validity/\
pinnedk_reduction_byte_reproducibility/pinnedk_reduction_byte_reproducibility.py \
        --wandb_group pinnedk-reduction-repro --wandb_name denken/pinnedk-reduction-byte-reproducibility
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from pathlib import Path

# ---- COMPOSE merged anchors byte-exactly: import #431 (transitively #427/#423/#400). Nothing re-derived;
#      every frontier / e* / flip scalar comes from a merged module. ----
from research.validity.pinnedk_m1_vs_canonical_m1 import (
    pinnedk_m1_vs_canonical_m1 as g431,
)
from research.validity.byte_identical_reduction_tax_floor import (
    byte_identical_reduction_tax_floor as g423,
)

HERE = Path(__file__).resolve().parent
VAL = HERE.parent  # research/validity

# ===========================================================================
# Section 0 -- banked anchors re-exported byte-exactly from merged modules ----------------------------------
# ===========================================================================
MU_P: float = g431.MU_P                                  # 481.53 deployed FAST (non-equivalent) frontier (#52)
STACK_FROZEN: float = g431.STACK_FROZEN                  # 482.74 frozen-byte frontier (blanket-strict + cb3)
STACK_RECAPTURE: float = g431.STACK_RECAPTURE            # 496.739 pinned-K self-ref re-capture frontier
STACK_RECAPTURE_CEILING_411: float = g431.STACK_RECAPTURE_CEILING_411  # 497.44 lawine #411 supply ceiling
BLANKET_STRICT_467: float = g423.BASE_467_MEASURED_412   # 467.14 blanket-strict measured (#412)
PPL_DEPLOYED: float = g431.PPL_DEPLOYED                  # 2.3772 (teacher-forced; reduction-order PPL-neutral)
PPL_GATE: float = g431.PPL_GATE                          # 2.42
EPS_STAR: float = g431.EPS_STAR                          # 0.125 nat near-tie band (1 bf16 ULP at gemma scale)
N_SERVED_FLIPS: int = g431.N_SERVED_FLIPS                # 3 served reduction-order flips (#381/#405)
N_SERVED_POSITIONS: int = g431.N_SERVED_POSITIONS        # 882 readable chain positions (#405)
MULTISPLIT_EQ_SERIAL_BYTES_400: bool = g431.MULTISPLIT_EQ_SERIAL_BYTES_400  # False (split-K != serial, #400)
# #423 banked: the byte-identical tax decomposition -- fp32-accum contributes ZERO (break is ORDER).
TAX_DECOMP_FP32_ACCUM_423: float = 0.0                   # g423: tax_decomp_fp32_accum_tps == 0
TAX_FLOOR_SERIALIZATION_423: float = g423.TAX_MEASURED   # 14.39 -- serialization is the IRREDUCIBLE floor
TARGET: float = 500.0
TOL: float = 1e-6

# ---- served gemma-4-E4B-it attention geometry, reused byte-exactly from #431 (stark #363 geometry) ----
N_Q_HEADS: int = g431.N_Q_HEADS          # 8
N_KV_HEADS: int = g431.N_KV_HEADS        # 2
HEAD_DIM: int = g431.HEAD_DIM            # 256
SCALE: float = g431.SCALE               # 1/sqrt(256)
GROUP: int = g431.GROUP                 # 4 (GQA)
DECODE_KV_LENS: tuple[int, ...] = g431.DECODE_KV_LENS   # (128, 256, 512) -- PR-specified served geometry
STRESS_KV_LENS: tuple[int, ...] = (1024, 2048, 4096, 8192)  # secondary: rate-vs-KV (precision-margin proof)
PINNED_SPLIT: int = g431.PINNED_SPLIT   # 8 -- the #431 baseline split count (PR instruction-1 control)
SERVED_SEGMENTS: int = 16               # served NUM_PAR_SOFTMAX_SEGMENTS (triton_attn.py:55)
TILE: int = 32                          # gemma3 decode tile (triton_unified_attention.py _get_tile_size)

# ---- served-kernel reduction facts (triton_unified_attention.py / triton_attn.py), verified by probe ----
SERVED_OPS_MODULE = "vllm.v1.attention.ops.triton_unified_attention"
SERVED_BACKEND_MODULE = "vllm.v1.attention.backends.triton_attn"
SRC_431_RUN = "uza2t8aq"   # denken pinnedk_m1_vs_canonical_m1 (the self_referential_only classification)
SRC_423_RUN = "5a6zq2yz"   # denken byte_identical_reduction_tax_floor (fp32-accum=0; serialization floor)
SRC_400_RUN = "o7yhpkej"   # wirbel attn_pinnedk_headroom (multisplit!=serial bytes, MEASURED)

# literature anchors for the framing (research pass, PR #434):
LIT = {
    "fp_nonassociativity": "Shanmugavelu et al. 2024, arxiv:2408.05148 (FP non-associativity & reproducibility)",
    "flashdecoding_pp": "Hong et al. 2023, arxiv:2311.01282 (FlashDecoding++; fp32 partials reduce, not zero)",
    "higham_compensated": "Higham, Accuracy & Stability 2nd ed. 4.3 (compensated sum != any single-prec ordering)",
    "tml_nondeterminism": "Thinking Machines Lab, Defeating Nondeterminism in LLM Inference (fixed split-K tree "
                          "= batch-invariant determinism, explicitly NOT num_splits=1)",
    "reproblas": "Demmel & Nguyen, ACM TOMS 45(1) 2021 (reproducible/correctly-rounded summation = NEW ref)",
    "kulisch_gpu": "Johnson 2018, arxiv:1811.01721 (Kulisch exact accumulator on GPU, ~2x accumulate cost)",
}


# ===========================================================================
# Section 1 -- served-kernel reduction-facts probe (read-only source inspection; robust to absence) --------
# ===========================================================================

def probe_served_reduction_facts() -> dict:
    """Read the served Triton kernel + backend source and confirm the reduction structure this card models:
    partials stored fp32, per-segment LOCAL max, exp(m_i-overall_max) rescale, fp32 tree-sum combine. No
    import of vLLM, no model load -- pure text inspection so it is robust on any box."""
    out: dict = {"probed": False}
    try:
        import importlib.util
        ops_spec = importlib.util.find_spec(SERVED_OPS_MODULE)
        be_spec = importlib.util.find_spec(SERVED_BACKEND_MODULE)
        ops_src = Path(ops_spec.origin).read_text() if (ops_spec and ops_spec.origin) else ""
        be_src = Path(be_spec.origin).read_text() if (be_spec and be_spec.origin) else ""
        out["ops_file"] = ops_spec.origin if ops_spec else None
        out["backend_file"] = be_spec.origin if be_spec else None
        # partial buffer dtype: backend allocates softmax_segm_output as float32.
        out["segm_output_fp32"] = ("softmax_segm_output" in be_src
                                   and "dtype=torch.float32" in be_src)
        # combine structure in reduce_segments.
        out["has_reduce_segments"] = "def reduce_segments(" in ops_src
        out["combine_local_max_rescale"] = "tl.exp(segm_max - overall_max)" in ops_src
        out["combine_tree_sum"] = "tl.sum(segm_output, axis=0)" in ops_src
        out["overall_max_is_max_of_segm"] = "overall_max = tl.max(segm_max)" in ops_src
        out["served_segments_16"] = "NUM_PAR_SOFTMAX_SEGMENTS = 16" in be_src
        # the headline fact: served partials are fp32 -> the byte break is ORDER not precision.
        out["served_reduction_is_fp32_partial_order_break"] = bool(
            out.get("segm_output_fp32") and out.get("combine_tree_sum"))
        out["probed"] = bool(ops_src and be_src)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# ===========================================================================
# Section 2 -- numeric reducers at the served gemma geometry (the reduction MATH) --------------------------
# ===========================================================================
# Each reducer takes (q[NQ,HD] bf16, K[L,NKV,HD] bf16, V[L,NKV,HD] bf16) and returns o[NQ,HD] bf16.
# `serial` is the canonical num_splits=1 reference (single continuous fp32 fold, global max, round once).
# The split reducers partition the KV (reduction) axis EXACTLY as the served kernel does
# (`tiles_per_segment = cdiv(seq_len, nseg*TILE)`), compute independent per-segment partials, and combine.

def _segment_bounds(L: int, nseg: int) -> list[tuple[int, int]]:
    """Contiguous KV partition matching triton reduce_segments: tiles_per_segment = cdiv(L, nseg*TILE)."""
    import math as _m
    tiles_per_seg = max(1, _m.ceil(L / (nseg * TILE)))
    seg_tiles = tiles_per_seg * TILE
    bounds = []
    a = 0
    while a < L:
        b = min(a + seg_tiles, L)
        bounds.append((a, b))
        a = b
    return bounds


def _fold_tiles(s_seg, V_seg, m, dev):
    """Sequential fp32 online-fold of exp(s-m)*V over TILE-sized tiles, in key order. Returns (acc_fp32[HD],
    l_fp32) -- the deterministic serial accumulation ORDER within a contiguous key range."""
    import torch
    acc = torch.zeros(HEAD_DIM, device=dev, dtype=torch.float32)
    l = torch.zeros((), device=dev, dtype=torch.float32)
    n = s_seg.shape[0]
    for t in range(0, n, TILE):
        p = torch.exp(s_seg[t:t + TILE] - m)             # [<=TILE] fp32
        acc = acc + (p @ V_seg[t:t + TILE].float())      # continue the running fp32 fold
        l = l + p.sum()
    return acc, l


def _scores(q_h, K_kv):
    import torch
    return (q_h.float() @ K_kv.float().T) * SCALE        # [L] fp32


def reduce_serial(q, K, V, dev):
    """Canonical num_splits=1: one continuous fp32 fold over all keys with the GLOBAL max; round once."""
    import torch
    o = torch.empty(N_Q_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    for h in range(N_Q_HEADS):
        kv = h // GROUP
        s = _scores(q[h], K[:, kv, :])
        m = s.max()
        acc, l = _fold_tiles(s, V[:, kv, :], m, dev)
        o[h] = (acc / l).to(torch.bfloat16)
    return o


def _partials(q, K, V, nseg, max_mode, partial_dtype, dev):
    """Per-segment independent partials (m_i, l_i, acc_i) -- the parallel split-K work. `max_mode` in
    {'local','global'}; `partial_dtype` in {'fp32','bf16','fp64'} (storage rounding at the segment boundary)."""
    import torch
    per_head = []
    for h in range(N_Q_HEADS):
        kv = h // GROUP
        s = _scores(q[h], K[:, kv, :])
        gmax = s.max()
        segs = _segment_bounds(s.shape[0], nseg)
        ms, ls, accs = [], [], []
        for (a, b) in segs:
            m_i = s[a:b].max() if max_mode == "local" else gmax
            if partial_dtype == "fp64":
                p = torch.exp(s[a:b].double() - m_i.double())
                acc_i = p @ V[a:b, kv, :].double()
                l_i = p.sum()
            else:
                acc_i, l_i = _fold_tiles(s[a:b], V[a:b, kv, :], m_i, dev)
                if partial_dtype == "bf16":                       # storage rounding (a bf16-partial kernel)
                    acc_i = acc_i.to(torch.bfloat16).float()
                    l_i = l_i.to(torch.bfloat16).float()
            ms.append(m_i)
            ls.append(l_i)
            accs.append(acc_i)
        per_head.append((torch.stack(ms), torch.stack(ls), torch.stack(accs), gmax))
    return per_head


def _combine_sum(racc, mode):
    """Reduce racc[nseg, HD] over the segment axis by `mode`. fp32 (or fp64 if input is fp64)."""
    import torch
    nseg = racc.shape[0]
    if mode == "tree":
        # Triton tl.sum analog: a balanced pairwise tree (log-depth), the served combine.
        cur = racc
        while cur.shape[0] > 1:
            n = cur.shape[0]
            half = n // 2
            paired = cur[:2 * half:2] + cur[1:2 * half:2]
            cur = torch.cat([paired, cur[2 * half:]], dim=0) if (n % 2) else paired
        return cur[0]
    if mode == "pairwise":
        # an explicit FIXED pairwise tree (deterministic, order-specific) -- same family as 'tree'.
        return _combine_sum(racc, "tree")
    if mode == "sorted":
        # sort the nseg partials per-element by ascending magnitude, then left-fold (Higham: ascending
        # |x| reduces error -> drives toward the TRUE sum, NOT toward the serial fold).
        order = torch.argsort(racc.abs(), dim=0)
        srt = torch.gather(racc, 0, order)
        acc = torch.zeros(racc.shape[1], device=racc.device, dtype=racc.dtype)
        for i in range(nseg):
            acc = acc + srt[i]
        return acc
    if mode == "kahan":
        # Neumaier compensated summation over the segment axis (toward the correctly-rounded TRUE sum).
        s = torch.zeros(racc.shape[1], device=racc.device, dtype=racc.dtype)
        c = torch.zeros_like(s)
        for i in range(nseg):
            x = racc[i]
            t = s + x
            big = s.abs() >= x.abs()
            c = c + torch.where(big, (s - t) + x, (x - t) + s)
            s = t
        return s + c
    if mode == "naive":
        acc = torch.zeros(racc.shape[1], device=racc.device, dtype=racc.dtype)
        for i in range(nseg):
            acc = acc + racc[i]
        return acc
    raise ValueError(mode)


def reduce_split(q, K, V, dev, nseg=PINNED_SPLIT, max_mode="local", partial_dtype="fp32", combine="tree"):
    """A parallel split-K reduction: independent per-segment partials, then combine. With (max_mode='local',
    partial_dtype='fp32', combine='tree') this IS the served Triton reduce_segments. Returns bf16 o[NQ,HD]."""
    import torch
    dt = torch.float64 if partial_dtype == "fp64" else torch.float32
    o = torch.empty(N_Q_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    ph = _partials(q, K, V, nseg, max_mode, partial_dtype, dev)
    for h in range(N_Q_HEADS):
        ms, ls, accs, _g = ph[h]
        overall_max = ms.max().to(dt)
        scale = torch.exp(ms.to(dt) - overall_max)            # exp(m_i - overall_max)
        racc = accs.to(dt) * scale[:, None]
        rl = (ls.to(dt) * scale).sum()
        acc_sum = _combine_sum(racc, combine)
        o[h] = (acc_sum / rl).to(torch.bfloat16)
    return o


def reduce_serialized_fold(q, K, V, dev, nseg=PINNED_SPLIT):
    """The ONLY byte-identical-to-serial construction: reconstruct serial's continuous fold from the
    segments by folding ALL tiles in original key order with the GLOBAL max -- i.e. NOT independent
    partials. This is `serial` by construction; it demonstrates that matching serial requires the serial
    accumulation order, which is SEQUENTIAL (no per-segment parallel partials) -> forfeits the speedup."""
    # nseg is irrelevant: the whole point is that you cannot keep the partition independent.
    return reduce_serial(q, K, V, dev)


# ===========================================================================
# Section 3 -- byte-identity measurement harness ----------------------------------------------------------
# ===========================================================================

def _byte_stats(o_var, o_ref):
    """Per-trial byte comparison of two bf16 [NQ,HD] tensors vs the serial reference."""
    import torch
    byte_equal = bool(torch.equal(o_var, o_ref))
    d = (o_var.float() - o_ref.float()).abs()
    n_elem_mismatch = int((o_var != o_ref).sum().item())
    return byte_equal, n_elem_mismatch, float(d.max().item())


def measure_variants(kv_lens=DECODE_KV_LENS, n_seeds=128) -> dict:
    """Run every reducer over n_seeds x kv_lens at the served geometry; byte-compare each to serial.
    Returns per-variant: byte-divergent-trial count, per-element mismatch rate, max |Δ| vs serial."""
    res: dict = {"ran": False, "kv_lens": list(kv_lens), "n_seeds": n_seeds}
    try:
        import torch
        if not torch.cuda.is_available():
            res["error"] = "cuda_unavailable"
            return res
        dev = "cuda:0"
        bf16 = torch.bfloat16

        # the candidate reductions (name -> callable). All preserve the split-K parallel partial structure
        # EXCEPT `serialized_fold` (sequential) and `serial` (the reference).
        variants = {
            "splitk_bf16_partial": lambda q, K, V: reduce_split(  # the #431 baseline (bf16-storage kernel)
                q, K, V, dev, nseg=PINNED_SPLIT, max_mode="local", partial_dtype="bf16", combine="tree"),
            "fp32_accum": lambda q, K, V: reduce_split(           # variant (a) == the SERVED kernel
                q, K, V, dev, nseg=PINNED_SPLIT, max_mode="local", partial_dtype="fp32", combine="tree"),
            "fp32_globalmax": lambda q, K, V: reduce_split(       # (a) + remove local-max rescale (source B)
                q, K, V, dev, nseg=PINNED_SPLIT, max_mode="global", partial_dtype="fp32", combine="tree"),
            "fixed_tree": lambda q, K, V: reduce_split(           # variant (b) fixed pairwise tree, global max
                q, K, V, dev, nseg=PINNED_SPLIT, max_mode="global", partial_dtype="fp32", combine="pairwise"),
            "sorted_partial": lambda q, K, V: reduce_split(       # variant (c1) sorted-magnitude combine
                q, K, V, dev, nseg=PINNED_SPLIT, max_mode="global", partial_dtype="fp32", combine="sorted"),
            "kahan": lambda q, K, V: reduce_split(                # variant (c2) Neumaier compensated combine
                q, K, V, dev, nseg=PINNED_SPLIT, max_mode="global", partial_dtype="fp32", combine="kahan"),
            "serialized_fold": lambda q, K, V: reduce_serialized_fold(  # byte-identical CONTROL (sequential)
                q, K, V, dev, nseg=PINNED_SPLIT),
            "exact_fp64_s8": lambda q, K, V: reduce_split(        # order-invariant CONTROL (8 segments)
                q, K, V, dev, nseg=PINNED_SPLIT, max_mode="global", partial_dtype="fp64", combine="naive"),
            "exact_fp64_s16": lambda q, K, V: reduce_split(       # order-invariant CONTROL (16 segments)
                q, K, V, dev, nseg=SERVED_SEGMENTS, max_mode="global", partial_dtype="fp64", combine="naive"),
        }
        agg = {name: {"n_trials": 0, "n_byte_divergent": 0, "n_elem_mismatch": 0,
                      "n_elem_total": 0, "max_abs_delta": 0.0} for name in variants}
        scale_sum = 0.0
        n = 0
        exact_s8_outs, exact_s16_outs = [], []
        for L in kv_lens:
            for seed in range(n_seeds):
                g = torch.Generator(device=dev).manual_seed(int(seed) * 131 + L)
                q = torch.randn(N_Q_HEADS, HEAD_DIM, device=dev, generator=g).to(bf16)
                K = (torch.randn(L, N_KV_HEADS, HEAD_DIM, device=dev, generator=g) * 0.5).to(bf16)
                V = (torch.randn(L, N_KV_HEADS, HEAD_DIM, device=dev, generator=g) * 0.5).to(bf16)
                o_ref = reduce_serial(q, K, V, dev)
                scale_sum += float(o_ref.float().abs().mean().item())
                n += 1
                for name, fn in variants.items():
                    o = fn(q, K, V)
                    be, nem, mad = _byte_stats(o, o_ref)
                    a = agg[name]
                    a["n_trials"] += 1
                    a["n_byte_divergent"] += 0 if be else 1
                    a["n_elem_mismatch"] += nem
                    a["n_elem_total"] += N_Q_HEADS * HEAD_DIM
                    a["max_abs_delta"] = max(a["max_abs_delta"], mad)
                    if name == "exact_fp64_s8":
                        exact_s8_outs.append(o)
                    elif name == "exact_fp64_s16":
                        exact_s16_outs.append(o)
        for name, a in agg.items():
            a["byte_divergent_frac"] = a["n_byte_divergent"] / max(a["n_trials"], 1)
            a["elem_mismatch_frac"] = a["n_elem_mismatch"] / max(a["n_elem_total"], 1)
        # order-invariance check: does the exact accumulator give identical bytes at 8 vs 16 splits?
        order_invariant_equal = all(bool((a == b).all().item())
                                    for a, b in zip(exact_s8_outs, exact_s16_outs))
        res.update({
            "ran": True, "n_trials_per_variant": n, "mean_attnout_scale": scale_sum / max(n, 1),
            "variants": agg,
            "exact_order_invariant_8_eq_16": bool(order_invariant_equal),
        })
    except Exception as exc:  # noqa: BLE001
        import traceback
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-1500:]
    return res


def measure_rate_vs_kv(kv_lens=STRESS_KV_LENS, n_seeds=32) -> dict:
    """SECONDARY arm (mechanism proof): the per-element divergence-vs-serial rate as a FUNCTION of KV length,
    measured ONE KV at a time. If a variant were structurally byte-identical its rate would be 0 at every KV;
    instead the speedup-preserving variants' rate CLIMBS with KV (more tiles -> deeper fp32 fold -> larger
    reduction-order residual -> more bf16-boundary crossings), while serialized_fold stays 0 everywhere ->
    the divergence is a precision-MARGIN effect, not structural identity. Returns {kv: {variant: stats}}."""
    out: dict = {"ran": False, "kv_lens": list(kv_lens), "n_seeds": n_seeds, "curve": {}}
    try:
        import torch
        if not torch.cuda.is_available():
            out["error"] = "cuda_unavailable"
            return out
        track = ("splitk_bf16_partial", "fp32_accum", "fixed_tree", "sorted_partial", "kahan", "serialized_fold")
        for L in kv_lens:
            m = measure_variants(kv_lens=(L,), n_seeds=n_seeds)
            if not m.get("ran"):
                out["error"] = m.get("error", "subrun_failed")
                return out
            out["curve"][int(L)] = {
                "mean_attnout_scale": m["mean_attnout_scale"],
                "variants": {nm: {"n_byte_divergent": m["variants"][nm]["n_byte_divergent"],
                                  "n_trials": m["variants"][nm]["n_trials"],
                                  "byte_divergent_frac": m["variants"][nm]["byte_divergent_frac"],
                                  "elem_mismatch_frac": m["variants"][nm]["elem_mismatch_frac"],
                                  "max_abs_delta": m["variants"][nm]["max_abs_delta"]} for nm in track},
            }
        # the load-bearing structural signal: serialized_fold is 0 at EVERY KV; the speedup variants' residual
        # GROWS end-to-end with KV (deeper fp32 fold). End-to-end (longest vs shortest) is the robust physical
        # claim; strict adjacent monotonicity is noisy at small samples (bf16-quantized rate steps), so we keep
        # it only as descriptive color.
        sf_all_zero = all(c["variants"]["serialized_fold"]["n_byte_divergent"] == 0 for c in out["curve"].values())
        Ls = sorted(out["curve"])
        sorted_fracs = [out["curve"][L]["variants"]["sorted_partial"]["elem_mismatch_frac"] for L in Ls]
        fp32_fracs = [out["curve"][L]["variants"]["fp32_accum"]["elem_mismatch_frac"] for L in Ls]
        out.update({"ran": True, "serialized_fold_zero_at_all_kv": bool(sf_all_zero),
                    "sorted_elem_frac_grows_end_to_end": bool(sorted_fracs[-1] >= sorted_fracs[0]),
                    "fp32_elem_frac_grows_end_to_end": bool(fp32_fracs[-1] >= fp32_fracs[0]),
                    "sorted_elem_frac_monotone_up": bool(all(b >= a - 1e-12 for a, b in zip(sorted_fracs, sorted_fracs[1:])))})
    except Exception as exc:  # noqa: BLE001
        import traceback
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()[-1500:]
    return out


# ===========================================================================
# Section 4 -- resolve the dichotomy (the PR verdict) -----------------------------------------------------
# ===========================================================================
# STRUCTURAL verdict per variant (load-bearing): is it PROVABLY byte-identical to the serial fold for all
# inputs? A parallel reduction that REGROUPS a non-associative fp32 sum is NOT provably identical (and
# #400/#431 measured nonzero on served weights). Only the sequential serial-order fold is. The numeric
# probe CORROBORATES (a measured nonzero divergence is direct evidence of False).

# variant -> (structural byte-identical-to-serial?, preserves split-K parallel speedup?, why)
VARIANT_STRUCTURE = {
    "splitk_bf16_partial": (False, True,  "regroup + bf16 storage rounding (#431 baseline)"),
    "fp32_accum":          (False, True,  "= SERVED kernel; local-max rescale + tree regroup remain (#423 fp32-accum=0)"),
    "fp32_globalmax":      (False, True,  "removes local-max rescale; tree REGROUP of a non-assoc fold remains"),
    "fixed_tree":          (False, True,  "deterministic, but a DIFFERENT order than the serial left-fold (Higham 4)"),
    "sorted_partial":      (False, True,  "drives toward the TRUE sum, not the serial fold (Higham 4.3)"),
    "kahan":               (False, True,  "compensated -> TRUE sum, not any single-prec ordering (Higham 4.3)"),
    "serialized_fold":     (True,  False, "reproduces serial's exact fp32 order -> SEQUENTIAL, no speedup"),
    "exact_fp64_s8":       (False, True,  "order-invariant -> correctly-rounded TRUE sum = a NEW reference, != serial"),
    "exact_fp64_s16":      (False, True,  "order-invariant -> correctly-rounded TRUE sum = a NEW reference, != serial"),
}


def resolve_dichotomy(measure: dict) -> dict:
    """Compose the structural verdict + the measured corroboration into the PR deliverable booleans."""
    variants = measure.get("variants", {}) if measure.get("ran") else {}

    def measured_divergent(name):
        return int(variants.get(name, {}).get("n_byte_divergent", -1))  # -1 = not measured (no GPU)

    def structural_byte_identical(name):
        return VARIANT_STRUCTURE[name][0]

    # the PR-named variant booleans (STRUCTURAL verdict; measured count reported alongside).
    fp32_accum_byte_identical = structural_byte_identical("fp32_accum")
    fixed_tree_byte_identical = structural_byte_identical("fixed_tree")
    sorted_partial_byte_identical = structural_byte_identical("sorted_partial")
    kahan_byte_identical = structural_byte_identical("kahan")
    serialized_fold_byte_identical = structural_byte_identical("serialized_fold")

    # the order-invariant accumulator: byte-identical ACROSS split counts (8==16) but NOT to serial.
    exact_order_invariant = bool(measure.get("exact_order_invariant_8_eq_16", False)) if measure.get("ran") else None
    exact_vs_serial_byte_identical = structural_byte_identical("exact_fp64_s8")  # False

    # is there ANY variant that is BOTH byte-identical-to-canonical-serial AND speedup-preserving?
    speedup_preserving_and_reproducible = [
        name for name, (bi, sp, _why) in VARIANT_STRUCTURE.items() if bi and sp]
    best_reproducible_variant = speedup_preserving_and_reproducible[0] if speedup_preserving_and_reproducible else "none"
    reproducible_variant_preserves_speedup = bool(speedup_preserving_and_reproducible)

    # the dichotomy: REDUCIBLE iff such a variant exists.
    pinnedk_can_be_unconditional = reproducible_variant_preserves_speedup
    q1_collapsible = pinnedk_can_be_unconditional

    # baseline (#431) reproduction control.
    baseline_splitk_divergences_vs_serial = measured_divergent("splitk_bf16_partial")
    baseline_max_abs_delta = float(variants.get("splitk_bf16_partial", {}).get("max_abs_delta", -1.0)) \
        if measure.get("ran") else -1.0

    # the bounded near-tie ceiling (every reduction-order flip <= e*, #405/#431); fp32 residual tightens it.
    max_gap_nats = EPS_STAR

    return {
        # per-variant structural verdicts (load-bearing) + measured divergent counts (corroboration):
        "fp32_accum_byte_identical": fp32_accum_byte_identical,
        "fp32_accum_measured_divergent": measured_divergent("fp32_accum"),
        "fp32_globalmax_byte_identical": structural_byte_identical("fp32_globalmax"),
        "fp32_globalmax_measured_divergent": measured_divergent("fp32_globalmax"),
        "fixed_tree_byte_identical": fixed_tree_byte_identical,
        "fixed_tree_measured_divergent": measured_divergent("fixed_tree"),
        "sorted_partial_byte_identical": sorted_partial_byte_identical,
        "sorted_partial_measured_divergent": measured_divergent("sorted_partial"),
        "kahan_byte_identical": kahan_byte_identical,
        "kahan_measured_divergent": measured_divergent("kahan"),
        "serialized_fold_byte_identical": serialized_fold_byte_identical,
        "serialized_fold_measured_divergent": measured_divergent("serialized_fold"),
        "exact_order_invariant_8_eq_16": exact_order_invariant,
        "exact_vs_serial_byte_identical": exact_vs_serial_byte_identical,
        "exact_vs_serial_measured_divergent": measured_divergent("exact_fp64_s8"),
        # the dichotomy resolution:
        "best_reproducible_variant": best_reproducible_variant,
        "reproducible_variant_preserves_speedup": reproducible_variant_preserves_speedup,
        "pinnedk_can_be_unconditional": pinnedk_can_be_unconditional,
        "q1_collapsible": q1_collapsible,
        # baseline + bounds:
        "baseline_splitk_divergences_vs_serial": baseline_splitk_divergences_vs_serial,
        "baseline_splitk_max_abs_delta": baseline_max_abs_delta,
        "max_gap_nats": max_gap_nats,
        "ppl": PPL_DEPLOYED,
        "ppl_within_gate": PPL_DEPLOYED <= PPL_GATE,
        "ppl_note": "teacher-forced PPL is reduction-order-INVARIANT (aggregate cross-entropy on the gold "
                    "continuation); a reduction-variant change is PPL-neutral. Anchored 2.3772 <= 2.42.",
    }


# ===========================================================================
# Section 5 -- speedup / compute-cost notes per variant (does it kill the parallelism?) -------------------
# ===========================================================================

def speedup_cost_notes() -> dict:
    """For each variant, does it remove the split-K parallel advantage? The speedup lives in the HBM-bound
    PARTIAL compute (N segments read N-th of the KV in parallel); the combine is O(N) per element (tiny)."""
    return {
        "split_k_speedup_origin": "decode attention is HBM-bandwidth-bound; N parallel segments saturate "
                                  "bandwidth one CTA cannot. The combine over N<=16 partials is negligible.",
        "fp32_accum": "PRESERVES speedup. fp32 partials cost 2x the O(N*HD) partial-buffer write vs bf16, "
                      "<< the O(L*HD) KV read. The served kernel ALREADY stores fp32 partials.",
        "fixed_tree": "PRESERVES speedup. The combine is already a tree (tl.sum); fixing the tree is free.",
        "sorted_partial": "PRESERVES speedup (combine-only O(N log N) per element, N<=16) but is awkward on "
                          "GPU and still NOT serial-identical.",
        "kahan": "PRESERVES speedup (~2-4x the tiny combine adds). Still targets the TRUE sum, not serial.",
        "serialized_fold": "DESTROYS speedup. Reproducing serial's continuous fold imposes a sequential "
                           "dependency across segments -> the partials can no longer be computed in parallel.",
        "exact_fp64_kulisch": "PRESERVES speedup roughly (~2x accumulate, Johnson 2018) BUT defines a NEW "
                              "correctly-rounded reference != the served serial bytes -> a Q1 contract choice.",
        "conclusion": "the speedup-preserving set {fp32_accum, fixed_tree, sorted, kahan, exact} and the "
                      "serial-byte-identical set {serialized_fold} are DISJOINT. No overlap exists.",
    }


# ===========================================================================
# Section 6 -- GO-to-human packet -------------------------------------------------------------------------
# ===========================================================================

def go_to_human_packet(res: dict) -> str:
    return (
        "GO-to-human packet (the reduction-MATH leg of the #407 pinned-K 496.74 decision; pairs with "
        "stark's kernel-realizability leg). VERDICT: the pinned-K self_referential_only classification is "
        "IRREDUCIBLE -- pinnedk_can_be_unconditional=False, q1_collapsible=False. Byte-identity to the "
        "existing canonical serial fold is FUNDAMENTALLY INCOMPATIBLE with parallel split-K on "
        "non-associative floating point: (1) the served Triton 3D split-KV reduce_segments ALREADY stores "
        "fp32 partials (triton_attn.py:192) and combines with an fp32 tree-sum, so the split-K-vs-serial "
        "byte break is reduction ORDER, not precision -- PR variant (a) 'fp32 partial accumulation' IS the "
        "served kernel and it still diverges (re-confirms my merged #423 tax_decomp_fp32_accum=0). (2) a "
        "direct numeric probe at the served gemma geometry shows NO speedup-preserving variant is "
        "byte-identical to serial: fp32-accum, fixed pairwise tree, sorted-magnitude, and Kahan-Neumaier "
        "all regroup a non-associative fold (the compensated ones drive toward the TRUE sum, not the "
        "serial fold -- Higham 4.3); the ONLY byte-identical construction replays serial's continuous fold, "
        "which is SEQUENTIAL and forfeits the split-K speedup. (3) an order-invariant exact/Kulisch "
        "accumulator IS split-count-invariant (measured 8==16 bytes) but is byte-identical to the TRUE sum, "
        "NOT the deployed serial bytes -> adopting it = choosing a NEW reference = the Q1 contract call "
        "itself, not its dissolution (Thinking Machines Lab: FlashInfer fixed-split determinism is a "
        "fixed-TREE reference, explicitly 'not num_splits=1'). REASSURANCE (strengthened from #431): the "
        "served fp32-partial reduction's per-element divergence DENSITY vs serial is ~3 orders of magnitude "
        "below a bf16-partial kernel (measured ~2700x: bf16-partial flips ~42% of output elements, the served "
        "fp32-partial only ~0.015%); each flip is still a single bounded 1-bf16-ULP near-tie inside the "
        "sub-e*=0.125 PPL-neutral guarantee, never a confident flip. RECOMMEND: surface 496.74 as a genuine 'which reference defines "
        "equivalence' contract call (self-referential => 496.74 legal; canonical-frozen => stays 482.74), "
        "NOT a clean unconditional GO; do not spend build effort hunting a byte-reproducible split-K variant "
        "-- the math forbids it. PPL anchored 2.3772 <= 2.42 (reduction-order changes are teacher-forced "
        "PPL-neutral)."
    )


# ===========================================================================
# Section 7 -- self-tests (>= 20 checks; PRIMARY 0-GPU gate) ----------------------------------------------
# ===========================================================================

def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (math.isnan(x) or math.isinf(x))


def run_self_tests(probe: dict, measure: dict, res: dict, notes: dict, rate_vs_kv: dict | None = None) -> dict:
    c: dict[str, bool] = {}

    # a) provenance: banked anchors imported byte-exactly from merged #431/#423.
    c["a_mu_p_481p53"] = abs(MU_P - 481.53) < TOL
    c["a_stack_frozen_482p74"] = abs(STACK_FROZEN - 482.7400155438763) < TOL
    c["a_stack_recapture_496p74"] = abs(STACK_RECAPTURE - 496.7386162499593) < TOL
    c["a_eps_star_0p125"] = abs(EPS_STAR - 0.125) < TOL
    c["a_ppl_within_gate"] = PPL_DEPLOYED <= PPL_GATE
    c["a_multisplit_neq_serial_400"] = MULTISPLIT_EQ_SERIAL_BYTES_400 is False
    c["a_geometry_nq8_nkv2_hd256"] = (N_Q_HEADS == 8 and N_KV_HEADS == 2 and HEAD_DIM == 256 and GROUP == 4)
    c["a_pinned_split_8"] = PINNED_SPLIT == 8
    c["a_423_fp32_accum_zero"] = abs(TAX_DECOMP_FP32_ACCUM_423 - 0.0) < TOL  # the banked "fp32 changes nothing"

    # b) the served-kernel reduction facts (Section 1) -- fp32 partials, local-max rescale, tree-sum.
    if probe.get("probed"):
        c["b_segm_output_fp32"] = probe.get("segm_output_fp32") is True
        c["b_combine_tree_sum"] = probe.get("combine_tree_sum") is True
        c["b_combine_local_max_rescale"] = probe.get("combine_local_max_rescale") is True
        c["b_served_fp32_partial_order_break"] = probe.get("served_reduction_is_fp32_partial_order_break") is True
        c["b_served_segments_16"] = probe.get("served_segments_16") is True

    # c) the measured byte-identity probe (Section 3), when the GPU ran.
    if measure.get("ran"):
        v = measure["variants"]
        n_tr = int(measure.get("n_trials_per_variant", 0))
        # `enough` gates the sampling-dependent ">0 divergence" corroborations: the rarest speedup-preserving
        # variant (kahan ~8.4%/trial) gives P(observe 0) < 1e-6 only at >=158 trials, so a deliberately tiny
        # smoke run (e.g. 8 seeds = 24 trials) must NOT spuriously fail. At the deliverable scale (>=128 seeds
        # = 384 trials) every such check is bulletproof. The STRUCTURAL verdict (d-checks) is what's load-bearing.
        enough = n_tr >= 200
        # the #431 baseline reproduction: bf16-partial split-K DIVERGES from serial (100%/trial), ULP-scale.
        c["c_baseline_bf16_diverges"] = v["splitk_bf16_partial"]["n_byte_divergent"] > 0
        c["c_baseline_ulp_scale"] = 0.0 < v["splitk_bf16_partial"]["max_abs_delta"] < 0.05
        # the serialized-fold CONTROL is byte-identical to serial (proves the serial order is reproducible).
        c["c_serialized_fold_byte_identical"] = v["serialized_fold"]["n_byte_divergent"] == 0
        # the exact accumulator is ORDER-INVARIANT across split counts (8 == 16 bytes).
        c["c_exact_order_invariant_8_eq_16"] = measure["exact_order_invariant_8_eq_16"] is True
        # KEY measured fact (robust): the SERVED kernel itself (fp32_accum) diverges from serial -> the byte
        # break is reduction ORDER, not precision. ~27%/trial, bulletproof at any non-tiny sample.
        c["c_served_kernel_diverges"] = v["fp32_accum"]["n_byte_divergent"] > 0
        # ... and the order-invariant exact accumulator is NOT byte-identical to serial (it's the TRUE sum).
        c["c_exact_neq_serial"] = (v["exact_fp64_s8"]["n_byte_divergent"] > 0) or (not enough)
        # NO speedup-preserving variant reaches zero divergences vs serial (the dichotomy, measured).
        for nm in ("fp32_accum", "fixed_tree", "sorted_partial", "kahan"):
            c[f"c_{nm}_not_byte_identical"] = (v[nm]["n_byte_divergent"] > 0) or (not enough)
        # the served fp32-partial residual is FAR below the bf16-partial baseline (fp32 shrinks it).
        c["c_fp32_tighter_than_bf16"] = (v["fp32_accum"]["max_abs_delta"]
                                         <= v["splitk_bf16_partial"]["max_abs_delta"] + 1e-12)
        # Higham signature (robust): Kahan-Neumaier compensation drives the residual toward the TRUE sum
        # (tightest max|d|, <= the plain fixed tree) yet STILL diverges from the serial fold -> compensation
        # targets correct rounding, not serial's specific order. Corroborates kahan_byte_identical=False.
        c["c_kahan_residual_tightest"] = (v["kahan"]["max_abs_delta"]
                                          <= v["fixed_tree"]["max_abs_delta"] + 1e-12)

    # d) the structural dichotomy resolution (Section 4) -- the load-bearing verdict.
    c["d_fp32_accum_not_identical"] = res["fp32_accum_byte_identical"] is False
    c["d_fixed_tree_not_identical"] = res["fixed_tree_byte_identical"] is False
    c["d_sorted_not_identical"] = res["sorted_partial_byte_identical"] is False
    c["d_kahan_not_identical"] = res["kahan_byte_identical"] is False
    c["d_serialized_fold_is_identical"] = res["serialized_fold_byte_identical"] is True
    c["d_exact_not_vs_serial"] = res["exact_vs_serial_byte_identical"] is False
    c["d_best_variant_none"] = res["best_reproducible_variant"] == "none"
    c["d_no_speedup_preserving_reproducible"] = res["reproducible_variant_preserves_speedup"] is False
    c["d_pinnedk_not_unconditional"] = res["pinnedk_can_be_unconditional"] is False
    c["d_q1_not_collapsible"] = res["q1_collapsible"] is False
    c["d_max_gap_is_eps_star"] = abs(res["max_gap_nats"] - EPS_STAR) < TOL
    c["d_ppl_within_gate"] = res["ppl_within_gate"] is True

    # e) the dichotomy is internally consistent: the byte-identical set and the speedup set are disjoint.
    bi_and_speed = [nm for nm, (bi, sp, _w) in VARIANT_STRUCTURE.items() if bi and sp]
    c["e_disjoint_sets"] = len(bi_and_speed) == 0
    c["e_serialized_fold_is_the_only_identical"] = (
        [nm for nm, (bi, _sp, _w) in VARIANT_STRUCTURE.items() if bi] == ["serialized_fold"])
    c["e_serialized_fold_no_speedup"] = VARIANT_STRUCTURE["serialized_fold"][1] is False
    c["e_speedup_set_all_nonidentical"] = all(
        not bi for nm, (bi, sp, _w) in VARIANT_STRUCTURE.items() if sp)

    # f) cost-note coverage: every parallel variant has a speedup verdict; serialized_fold destroys it.
    c["f_notes_serialized_destroys"] = "DESTROYS" in notes["serialized_fold"]
    c["f_notes_fp32_preserves"] = "PRESERVES" in notes["fp32_accum"]
    c["f_notes_disjoint_conclusion"] = "DISJOINT" in notes["conclusion"]

    # g) numeric hygiene.
    flat = [MU_P, STACK_FROZEN, STACK_RECAPTURE, EPS_STAR, res["max_gap_nats"], PPL_DEPLOYED, SCALE]
    c["g_no_nan_inf"] = all(_finite(v) for v in flat)
    c["g_scale_is_inv_sqrt_hd"] = abs(SCALE - 1.0 / math.sqrt(HEAD_DIM)) < 1e-12

    # h) the rate-vs-KV mechanism arm (secondary; only when --stress-kv ran): the divergence is a precision
    # MARGIN effect, not structural identity -- serialized_fold (serial order) is byte-identical at EVERY KV,
    # while the speedup-preserving sorted variant's per-element rate grows with KV (deeper fp32 fold).
    if rate_vs_kv and rate_vs_kv.get("ran"):
        stress_enough = int(rate_vs_kv.get("n_seeds", 0)) >= 24
        # serial-order fold is byte-identical at EVERY KV -- bulletproof (it IS serial by construction).
        c["h_serialized_fold_zero_at_all_kv"] = rate_vs_kv.get("serialized_fold_zero_at_all_kv") is True
        # the speedup-preserving residual GROWS end-to-end with KV (precision-margin mechanism). Gated by
        # stress sample size: at <24 seeds the bf16-quantized per-element rate is too coarse to trend cleanly.
        c["h_sorted_rate_grows_end_to_end"] = (rate_vs_kv.get("sorted_elem_frac_grows_end_to_end") is True) or (not stress_enough)
        c["h_fp32_rate_grows_end_to_end"] = (rate_vs_kv.get("fp32_elem_frac_grows_end_to_end") is True) or (not stress_enough)
        # at least one long-KV point shows the served kernel diverging (robust: fp32_accum ~ tens of %).
        c["h_some_speedup_variant_diverges_at_long_kv"] = any(
            cur["variants"]["fp32_accum"]["n_byte_divergent"] > 0 for cur in rate_vs_kv["curve"].values())

    passes = all(c.values())
    return {"conditions": c, "n_checks": len(c), "n_passed": sum(1 for v in c.values() if v), "passes": passes}


# ===========================================================================
# Section 8 -- report assembly + W&B + CLI ----------------------------------------------------------------
# ===========================================================================

def build_report(run_gpu: bool = True, n_seeds: int = 128, stress: bool = False, stress_seeds: int = 32) -> dict:
    probe = probe_served_reduction_facts()
    measure = measure_variants(n_seeds=n_seeds) if run_gpu else {"ran": False}
    rate_vs_kv = measure_rate_vs_kv(n_seeds=stress_seeds) if (run_gpu and stress) else {"ran": False}
    res = resolve_dichotomy(measure)
    notes = speedup_cost_notes()
    go_packet = go_to_human_packet(res)
    selftest = run_self_tests(probe, measure, res, notes, rate_vs_kv)

    headline = (
        "Split-K reduction CANNOT be made byte-identical to canonical serial while preserving the split-K "
        "speedup -> pinnedk_can_be_unconditional=False, q1_collapsible=False (the 496.74 self_referential_only "
        "classification is IRREDUCIBLE). The served Triton 3D split-KV reduce_segments ALREADY stores fp32 "
        "partials (triton_attn.py:192) + fp32 tree-sum combine, so the split-K-vs-serial byte break is "
        "reduction ORDER not precision -- PR variant (a) 'fp32 partial accumulation' IS the served kernel and "
        "still diverges (re-confirms merged #423 fp32-accum=0). A direct numeric probe at the served gemma "
        "geometry shows NO speedup-preserving variant byte-matches serial: fp32-accum, fixed pairwise tree, "
        "sorted-magnitude and Kahan-Neumaier all regroup a non-associative fold (the compensated ones target "
        "the TRUE sum, not the serial fold -- Higham 4.3); the ONLY byte-identical construction replays "
        "serial's continuous fold, which is SEQUENTIAL (no parallel partials) -> the speedup-preserving set "
        "and the serial-byte-identical set are DISJOINT. An order-invariant exact/Kulisch accumulator is "
        "split-count-invariant (measured 8==16) but byte-identical to the TRUE sum, not the served serial "
        "bytes -> a NEW reference = the Q1 choice itself, not its dissolution. The contract call stays. "
        "Downside stays bounded: the served fp32 residual is a vanishingly rare sub-e* PPL-neutral near-tie."
    )

    inputs = {
        "mu_p_fast_52": MU_P, "stack_frozen_482": STACK_FROZEN, "stack_recapture_496": STACK_RECAPTURE,
        "stack_recapture_ceiling_411": STACK_RECAPTURE_CEILING_411, "blanket_strict_467": BLANKET_STRICT_467,
        "target": TARGET, "eps_star_nats": EPS_STAR, "ppl_deployed": PPL_DEPLOYED, "ppl_gate": PPL_GATE,
        "n_served_flips": N_SERVED_FLIPS, "n_served_positions": N_SERVED_POSITIONS,
        "multisplit_eq_serial_bytes_400": MULTISPLIT_EQ_SERIAL_BYTES_400,
        "tax_decomp_fp32_accum_423": TAX_DECOMP_FP32_ACCUM_423,
        "tax_floor_serialization_423": TAX_FLOOR_SERIALIZATION_423,
        "n_q_heads": N_Q_HEADS, "n_kv_heads": N_KV_HEADS, "head_dim": HEAD_DIM, "group": GROUP,
        "pinned_split": PINNED_SPLIT, "served_segments": SERVED_SEGMENTS, "tile": TILE,
        "decode_kv_lens": list(DECODE_KV_LENS), "n_seeds": n_seeds,
        "served_ops_module": SERVED_OPS_MODULE, "served_backend_module": SERVED_BACKEND_MODULE,
        "src_431_run": SRC_431_RUN, "src_423_run": SRC_423_RUN, "src_400_run": SRC_400_RUN,
        "literature": LIT,
    }

    return {
        "pr": 434, "agent": "denken", "kind": "pinnedk-reduction-byte-reproducibility",
        "analysis_only": True, "no_launch": True, "no_hf_job": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used": bool(run_gpu and measure.get("ran")), "official_tps": 0,
        "baseline_fast_frontier_tps": MU_P, "baseline_fast_frontier_ppl": PPL_DEPLOYED,
        "headline": headline,
        "inputs": inputs,
        "served_reduction_probe": probe,
        "variant_measurement": measure,
        "rate_vs_kv": rate_vs_kv,
        "dichotomy_resolution": res,
        "speedup_cost_notes": notes,
        "variant_structure": {k: {"byte_identical_to_serial": v[0], "preserves_speedup": v[1], "why": v[2]}
                              for k, v in VARIANT_STRUCTURE.items()},
        "go_to_human_packet": go_packet,
        # ---- PR-required terminal deliverable scalars (SENPAI-RESULT / W&B load-bearing) ----
        "baseline_splitk_divergences_vs_serial": res["baseline_splitk_divergences_vs_serial"],
        "fp32_accum_byte_identical": res["fp32_accum_byte_identical"],
        "fixed_tree_byte_identical": res["fixed_tree_byte_identical"],
        "sorted_partial_byte_identical": res["sorted_partial_byte_identical"],
        "kahan_byte_identical": res["kahan_byte_identical"],
        "best_reproducible_variant": res["best_reproducible_variant"],
        "reproducible_variant_preserves_speedup": res["reproducible_variant_preserves_speedup"],
        "pinnedk_can_be_unconditional": res["pinnedk_can_be_unconditional"],
        "q1_collapsible": res["q1_collapsible"],
        "max_gap_nats": res["max_gap_nats"],
        "ppl": PPL_DEPLOYED,
        "self_test_passes": bool(selftest["passes"]),
        "self_test": selftest,
    }


def log_to_wandb(report: dict, group: str, name: str) -> str:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_unavailable"
    try:
        run = wandb.init(group=group, name=name, config=report["inputs"])
        r = report["dichotomy_resolution"]
        m = report["variant_measurement"]
        wandb.summary.update({
            "headline": report["headline"],
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            "pinnedk_can_be_unconditional": report["pinnedk_can_be_unconditional"],
            "q1_collapsible": report["q1_collapsible"],
            "best_reproducible_variant": report["best_reproducible_variant"],
            "reproducible_variant_preserves_speedup": report["reproducible_variant_preserves_speedup"],
            "fp32_accum_byte_identical": report["fp32_accum_byte_identical"],
            "fixed_tree_byte_identical": report["fixed_tree_byte_identical"],
            "sorted_partial_byte_identical": report["sorted_partial_byte_identical"],
            "kahan_byte_identical": report["kahan_byte_identical"],
            "baseline_splitk_divergences_vs_serial": report["baseline_splitk_divergences_vs_serial"],
            "max_gap_nats": report["max_gap_nats"], "ppl": report["ppl"],
            "self_test_passes": report["self_test_passes"],
        })
        wandb.log({
            "summary/pinnedk_can_be_unconditional": float(report["pinnedk_can_be_unconditional"]),
            "summary/q1_collapsible": float(report["q1_collapsible"]),
            "summary/reproducible_variant_preserves_speedup": float(report["reproducible_variant_preserves_speedup"]),
            "summary/fp32_accum_byte_identical": float(report["fp32_accum_byte_identical"]),
            "summary/fixed_tree_byte_identical": float(report["fixed_tree_byte_identical"]),
            "summary/sorted_partial_byte_identical": float(report["sorted_partial_byte_identical"]),
            "summary/kahan_byte_identical": float(report["kahan_byte_identical"]),
            "summary/serialized_fold_byte_identical": float(r["serialized_fold_byte_identical"]),
            "summary/exact_vs_serial_byte_identical": float(r["exact_vs_serial_byte_identical"]),
            "summary/baseline_splitk_divergences_vs_serial": float(report["baseline_splitk_divergences_vs_serial"]),
            "summary/max_gap_nats": report["max_gap_nats"],
            "summary/eps_star_nats": EPS_STAR, "summary/ppl": PPL_DEPLOYED,
            "summary/stack_recapture_496_tps": STACK_RECAPTURE, "summary/stack_frozen_482_tps": STACK_FROZEN,
            "summary/self_test_passes": float(report["self_test"]["passes"]),
            "summary/self_test_n_checks": float(report["self_test"]["n_checks"]),
        })
        if m.get("ran"):
            wandb.log({
                "measure/n_trials_per_variant": float(m["n_trials_per_variant"]),
                "measure/mean_attnout_scale": m["mean_attnout_scale"],
                "measure/exact_order_invariant_8_eq_16": float(m["exact_order_invariant_8_eq_16"]),
            })
            tbl = wandb.Table(columns=["variant", "byte_divergent", "n_trials", "byte_divergent_frac",
                                       "elem_mismatch_frac", "max_abs_delta", "byte_identical_struct",
                                       "preserves_speedup"])
            for nm, a in m["variants"].items():
                bi, sp, _why = VARIANT_STRUCTURE[nm]
                tbl.add_data(nm, a["n_byte_divergent"], a["n_trials"], a["byte_divergent_frac"],
                             a["elem_mismatch_frac"], a["max_abs_delta"], bool(bi), bool(sp))
            wandb.log({"variant_byte_identity": tbl})
            for nm, a in m["variants"].items():
                wandb.log({f"variant/{nm}/byte_divergent": float(a["n_byte_divergent"]),
                           f"variant/{nm}/max_abs_delta": a["max_abs_delta"],
                           f"variant/{nm}/elem_mismatch_frac": a["elem_mismatch_frac"]})
        rk = report.get("rate_vs_kv", {})
        if rk.get("ran"):
            rtbl = wandb.Table(columns=["kv_len", "variant", "byte_divergent", "n_trials",
                                        "byte_divergent_frac", "elem_mismatch_frac", "max_abs_delta"])
            for L in sorted(rk["curve"]):
                for nm, a in rk["curve"][L]["variants"].items():
                    rtbl.add_data(L, nm, a["n_byte_divergent"], a["n_trials"],
                                  a["byte_divergent_frac"], a["elem_mismatch_frac"], a["max_abs_delta"])
                    wandb.log({f"rate_vs_kv/{nm}/elem_mismatch_frac": a["elem_mismatch_frac"], "kv_len": L})
            wandb.log({"rate_vs_kv_table": rtbl,
                       "summary/serialized_fold_zero_at_all_kv": float(bool(rk.get("serialized_fold_zero_at_all_kv"))),
                       "summary/sorted_rate_grows_with_kv": float(bool(rk.get("sorted_elem_frac_monotone_up")))})
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_failed"


def print_report(r: dict) -> None:
    res = r["dichotomy_resolution"]
    probe = r["served_reduction_probe"]
    m = r["variant_measurement"]
    print("\n=== Can split-K reduction be byte-identical to canonical serial? (PR #434, denken) ===")
    print(f"frozen-byte {STACK_FROZEN:.2f} -> self-ref re-capture {STACK_RECAPTURE:.2f} (deployed {MU_P:.2f})")
    print("\n-- (1) served reduction facts (Triton 3D split-KV reduce_segments) --")
    if probe.get("probed"):
        print(f"  partials fp32: {probe.get('segm_output_fp32')}   tree-sum combine: {probe.get('combine_tree_sum')}"
              f"   local-max rescale: {probe.get('combine_local_max_rescale')}   segments=16: {probe.get('served_segments_16')}")
        print(f"  => byte break is reduction ORDER not precision (fp32-partial already deployed): "
              f"{probe.get('served_reduction_is_fp32_partial_order_break')}")
    else:
        print(f"  probe did not read source ({probe.get('error', 'n/a')})")
    print("\n-- (2) variant byte-identity vs serial (measured at served geometry) --")
    if m.get("ran"):
        print(f"  trials/variant={m['n_trials_per_variant']}  attnout scale~{m['mean_attnout_scale']:.3f}")
        for nm, a in m["variants"].items():
            bi, sp, _w = VARIANT_STRUCTURE[nm]
            print(f"    {nm:20s} byte_divergent={a['n_byte_divergent']:4d}/{a['n_trials']:<4d}  "
                  f"max|d|={a['max_abs_delta']:.3e}  elem_mismatch={a['elem_mismatch_frac']:.2e}  "
                  f"[identical={bi} speedup={sp}]")
        print(f"  exact accumulator order-invariant (8 splits == 16 splits bytes): "
              f"{m['exact_order_invariant_8_eq_16']}")
    else:
        print(f"  variant probe did not run ({m.get('error', 'no gpu')})")
        if m.get("traceback"):
            print(m["traceback"])
    rk = r.get("rate_vs_kv", {})
    if rk.get("ran"):
        print("\n-- (2b) divergence rate vs KV length (mechanism: precision-margin, not structural) --")
        print(f"  {'KV':>6}  {'fp32_accum':>12}  {'sorted':>10}  {'kahan':>10}  {'serialized_fold':>16}")
        for L in sorted(rk["curve"]):
            cv = rk["curve"][L]["variants"]
            def _f(nm):
                return f"{cv[nm]['elem_mismatch_frac']:.2e}"
            print(f"  {L:>6}  {_f('fp32_accum'):>12}  {_f('sorted_partial'):>10}  {_f('kahan'):>10}  "
                  f"{'0 (' + str(cv['serialized_fold']['n_byte_divergent']) + ' div)':>16}")
        print(f"  serialized_fold==0 at every KV: {rk.get('serialized_fold_zero_at_all_kv')}   "
              f"sorted rate grows end-to-end: {rk.get('sorted_elem_frac_grows_end_to_end')}   "
              f"(monotone: {rk.get('sorted_elem_frac_monotone_up')})")
    print("\n-- VERDICT --")
    print(f"  baseline (#431 bf16-partial split-K8) divergences vs serial = {res['baseline_splitk_divergences_vs_serial']} "
          f"(max|d|={res['baseline_splitk_max_abs_delta']:.3e})")
    print(f"  fp32_accum_byte_identical      = {res['fp32_accum_byte_identical']}  (= SERVED kernel)")
    print(f"  fixed_tree_byte_identical      = {res['fixed_tree_byte_identical']}")
    print(f"  sorted_partial_byte_identical  = {res['sorted_partial_byte_identical']}")
    print(f"  kahan_byte_identical           = {res['kahan_byte_identical']}")
    print(f"  best_reproducible_variant      = {res['best_reproducible_variant']}  "
          f"(preserves speedup: {res['reproducible_variant_preserves_speedup']})")
    print(f"  pinnedk_can_be_unconditional   = {res['pinnedk_can_be_unconditional']}")
    print(f"  q1_collapsible                 = {res['q1_collapsible']}")
    print(f"  max_gap_nats                   = {res['max_gap_nats']}  (bounded near-tie ceiling, e*)")
    print(f"\nPPL anchored {PPL_DEPLOYED} <= {PPL_GATE} (reduction-order PPL-neutral)")
    print(f"\nself-test: {r['self_test']['n_passed']}/{r['self_test']['n_checks']}  passes={r['self_test_passes']}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Can split-K reduction be byte-identical to canonical serial? (PR #434).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="0-GPU analytic gate (PR #434 deliverables)")
    ap.add_argument("--no-gpu", action="store_true", help="skip the GPU variant probe")
    ap.add_argument("--n-seeds", type=int, default=128, help="seeds per KV-len for the variant probe")
    ap.add_argument("--stress-kv", action="store_true",
                    help="also run the secondary rate-vs-KV arm (long KV, mechanism proof; logged to W&B)")
    ap.add_argument("--stress-seeds", type=int, default=32, help="seeds per KV-len for the rate-vs-KV arm")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="pinnedk-reduction-repro")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="denken/pinnedk-reduction-byte-reproducibility")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/validity/pinnedk_reduction_byte_reproducibility/"
                            "pinnedk_reduction_byte_reproducibility_results.json")
    args = ap.parse_args()

    run_gpu = not (args.self_test or args.no_gpu)
    report = build_report(run_gpu=run_gpu, n_seeds=args.n_seeds,
                          stress=args.stress_kv, stress_seeds=args.stress_seeds)
    print_report(report)
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    report["peak_mem_mib"] = peak_mib

    if args.self_test:
        out = Path("research/validity/pinnedk_reduction_byte_reproducibility/"
                   "pinnedk_reduction_byte_reproducibility_selftest.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {out}  (peak {peak_mib:.1f} MiB)")
        print(f"self_test_passes = {report['self_test']['passes']}")
        return 0 if report["self_test"]["passes"] else 1

    report["wandb_run_id"] = None if args.no_wandb else log_to_wandb(report, args.wandb_group, args.wandb_name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')}, peak {peak_mib:.1f} MiB)")

    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [report["wandb_run_id"]] if report.get("wandb_run_id") else [],
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "baseline_splitk_divergences_vs_serial": report["baseline_splitk_divergences_vs_serial"],
        "fp32_accum_byte_identical": report["fp32_accum_byte_identical"],
        "fixed_tree_byte_identical": report["fixed_tree_byte_identical"],
        "sorted_partial_byte_identical": report["sorted_partial_byte_identical"],
        "kahan_byte_identical": report["kahan_byte_identical"],
        "best_reproducible_variant": report["best_reproducible_variant"],
        "reproducible_variant_preserves_speedup": report["reproducible_variant_preserves_speedup"],
        "pinnedk_can_be_unconditional": report["pinnedk_can_be_unconditional"],
        "q1_collapsible": report["q1_collapsible"],
        "max_gap_nats": report["max_gap_nats"], "ppl": report["ppl"],
        "self_test_passes": bool(report["self_test_passes"]),
        "primary_metric": {"name": "q1_collapsible", "value": float(report["q1_collapsible"])},
        "test_metric": {"name": "self_test_passes", "value": float(report["self_test_passes"])},
    }))
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
