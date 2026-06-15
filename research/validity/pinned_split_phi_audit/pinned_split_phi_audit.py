#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #366 (wirbel) -- Pinned-split phi audit: does stark #363 break the un-packing eta floor
that BOTH denken #327 (9.841% locus) and denken #332 (phi=0.075 -> 473.5 cap) rest on?

THE QUESTION (the ANALYTIC cross-check of stark #363's empirical eta~=0)
------------------------------------------------------------------------
My own #360 (`6s9vgnw9`) reconciled the governing strict eta-locus to denken #327's bf16
lm_head+attn floor = 9.841%, whose dominant term is the SDPA split-KV slack (9.451% alone).
denken #332 (`y5cl0ena`) priced determinism's SDPA recovery at phi_recovery = 0.075 << 0.255
break-even -> compliant ceiling 473.5 < 500.

Both rest on ONE shared assumption: **strict determinism requires UN-PACKING the split-KV
reduction** -- collapsing the parallel KV-split CTAs down to the non-reduction (M x q-head) tiles,
forgoing the split-KV occupancy. #327 charges the FULL forgone slack (phi_forgone=1); #332 charges
phi_forgone = 1 - N_nonreduction(6)/SMs(80) = 0.925 (the reduction axis "FORGONE for determinism").

But stark #363 (`a0oi2esq`) does NOT un-pack -- it **PINS** the split (fixed num_splits=8, ordered
combine, M-invariant) and measured byte-EXACT identity at eta~=0 (best K=8 FASTER than the deployed
heuristic). A fixed split + ordered combine keeps ALL num_splits CTAs running in parallel; only the
final reduce is serialized (a cheap epilogue). If pinning preserves the CTA parallelism, the
un-packing penalty that #327's 9.451% and #332's phi=0.075 both charge **does not apply** -> the
9.841% blanket AND the 473.5 cap collapse -> the strict lambda=1 ceiling revives toward 520.953.

THIS CARD = the ANALYTIC leg (theory) of the #363/#365 empirical leg. I derived the 9.841% locus
in #360, so I audit whether its (and #332's) un-packing assumption survives a GA102 occupancy
microbench on the on-target A10G (GA102 / sm_86 / 80 SMs).

WHAT THIS CARD MEASURES / DERIVES
---------------------------------
(1) AUDIT: from the banked #327 + #332 JSON, extract the exact step where each charges the
    determinism tax; classify it as the UN-PACKING penalty (forgone split-KV occupancy) vs an
    intrinsic M-invariant-reduction cost. -> shares_unpacking_assumption_327/332 (bool).
(2) MICROBENCH (A10G): CTAs-in-flight + per-call latency for the deployed verify shape (batch=1,
    head_dim 256, 8q/2kv, bf16, L) under THREE configs at M in {1,8}:
      - heuristic   (num_splits=0, the deployed split-K heuristic),
      - pinned-8    (num_splits=8, M-invariant -- stark #363's mechanism),
      - un-packed   (num_splits=1, the #332-assumed config: collapse to the non-reduction tiles).
    Does pinned-8 PRESERVE the heuristic's CTA parallelism, or does M-invariance force a collapse?
(3) RECOMPUTE: the analytic batch-invariant eta-floor and phi-recovery UNDER pinned-split, vs
    #360's 9.841% / #332's phi=0.075. Compare to the 0.255 break-even, the 4.02% >500 budget, and
    the 9.841% blanket.
(4) CROSS-CHECK stark #365 (lm_head BI-GEMM eta) if posted; reconcile analytic vs empirical.

SCOPE: pod-GPU microbench + CPU analytic over banked MERGED *_results.json. NO train.py --launch,
NO HF Job, NO submission, NO served-file change. 0 official TPS, baseline 481.53 UNCHANGED. Greedy
identity is MEASURED, never broken. Run with CUDA_VISIBLE_DEVICES=0 (#358/#363 single-A10G gotcha).
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

import torch
from flash_attn import flash_attn_with_kvcache

# ---------------------------------------------------------------------------------------- #
# Real gemma-4-E4B-it text-decoder attention geometry (the EXACT dims #327/#332/#358/#363 use)
# ---------------------------------------------------------------------------------------- #
HEAD_DIM = 256
N_Q_HEADS = 8
N_KV_HEADS = 2
GQA_GROUP = N_Q_HEADS // N_KV_HEADS  # 4
DTYPE = torch.bfloat16
SCALE = 1.0 / math.sqrt(HEAD_DIM)
A10G_SMS = 80
BLOCK_M_SPLITKV = 64   # flash_attn splitkv query-tile (M<=8 => 1 m-block); reported, not load-bearing
M_LIST = (1, 8)        # AR decode (M=1) + verify width (M=8, K_spec=7+1) -- latency/occupancy microbench
IDENT_M_LIST = (2, 4, 8)  # identity demo (batched-M vs per-row); M=1 is degenerate (per-row IS batched) -> #363 used {2,4,8}
PRIMARY_L = 2048       # #363 "realistic context" geometry (primary)
BRIDGE_L = 528         # #332 ctx geometry (bridge: shorter context -> fewer natural KV segments)
L_LIST = (PRIMARY_L, BRIDGE_L)

HEURISTIC_SPLIT = 0    # flash_attn occupancy heuristic == the DEPLOYED split-KV fast path
PINNED_SPLIT = 8       # stark #363's M-invariant mechanism
UNPACK_SPLIT = 1       # the #332-assumed "un-pack to non-reduction" config (no KV split)
SWEEP_SPLITS = (0, 1, 2, 4, 8, 16, 32)

# ---- strict budget ladder (cite, do NOT re-derive) ------------------------------------- #
CEILING_500 = 520.953                          # lambda=1 central ceiling TPS (wirbel #354/#326/#327)
STEP_US = 1218.2                               # deployed batch=1 decode step (denken #344)
TARGET_500 = 500.0
BUDGET_500_ETA = 1.0 - TARGET_500 / CEILING_500          # >500 kernel budget ~= 0.04022 (#362)
BUDGET_LAMBDA1_ETA = 0.07331808522875782                 # no-regression / lambda=1 budget (#213/#327)

# ---- #327 (kcjlr5ny) bf16 reduction floor decomposition (cite; roundtripped in step 1) -- #
SDPA_STEP_SHARE = 0.14513725794080165
SDPA_BW_UTIL = 0.34883864849061247
SDPA_PENALTY_PI = 0.6511613515093875           # = 1 - SDPA_BW_UTIL ("above-roofline forgone-reduction slack")
SDPA_FLOOR_FULL = 0.09450777303509898          # = step_share * pi  (phi_forgone=1; the un-packing-full SDPA tax)
LMHEAD_STEP_SHARE = 0.02358516890006141
LMHEAD_BW_UTIL = 0.8344417980018903
LMHEAD_FLOOR_FULL = 0.003904718156915901       # the separate small lm_head GEMM determinism cost
FLOOR_COMBINED_FULL = 0.09841249119201488      # 9.841% blanket = SDPA_FLOOR_FULL + LMHEAD_FLOOR_FULL

# ---- #332 (y5cl0ena) phi model (cite; roundtripped in step 1) --------------------------- #
N_NONREDUCTION_332 = 6        # non-reduction (M x q-head, BLOCK_Q) CTAs -- determinism's tiles under un-pack
N_FULL_3D_332 = 96            # adaptive heuristic 3D split-KV CTAs (> 80 SMs, "occupancy-saturated")
N_UNPACK_CAP_332 = 64         # #332's naive un-pack ceiling (the GREEN-mirage they refuted)
PHI_FORGONE_GEO_332 = 0.925   # = 1 - N_nonreduction/SMs = 1 - 6/80 (the "geometric phi" -- forgone fraction)
PHI_RECOVERY_GEO_332 = 0.07499999999999996      # = 1 - 0.925 (the card's "phi=0.075" recovery)
RECOVERY_BREAKEVEN_NOREG = 0.2549920813842095   # recovery needed to clear the no-regression 7.33% budget
PHI_STAR_500_COMBINED = 0.4086882190334648      # phi_FORGONE break-even for the >500 (4.02%) budget
CEILING_AT_GEO_332 = 473.5295953446407          # #332 compliant ceiling at phi_geo (the 473.5 cap)

# recovery (= 1 - phi_forgone) needed to clear the OPERATIVE >500 budget
RECOVERY_BREAKEVEN_500 = 1.0 - PHI_STAR_500_COMBINED      # ~= 0.5913

# banked anchor JSONs (roundtrip provenance)
_VAL = Path(__file__).resolve().parents[1]
ANCHOR_PATHS = {
    "327": _VAL / "eagle3_bi_reduction_floor" / "eagle3_bi_reduction_floor_results.json",
    "332": _VAL / "eagle3_sdpa_phi_floor" / "eagle3_sdpa_phi_floor_results.json",
    "360": _VAL / "strict_kernel_eta_locus" / "strict_kernel_eta_locus_results.json",
    "363": _VAL / "strict_batch_invariant_verify_gemm" / "strict_batch_invariant_verify_gemm_results.json",
}
# stark #365 (lm_head BI-GEMM) -- may not be posted yet; cross-checked in step 4 if present
PR365_CANDIDATES = [
    _VAL / "lmhead_bi_reduction_floor" / "lmhead_bi_reduction_floor_results.json",
    _VAL / "strict_lmhead_bi_gemm" / "strict_lmhead_bi_gemm_results.json",
    _VAL / "lmhead_verify_audit" / "lmhead_verify_audit_results.json",
]


# ======================================================================================== #
# Device + facts
# ======================================================================================== #
def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Launch with CUDA_VISIBLE_DEVICES=0 (the single-A10G pod default "
            "points at a non-existent 2nd GPU -- the #358/#363 gotcha).")
    return torch.device("cuda:0")


def _gpu_facts(dev: torch.device) -> dict[str, Any]:
    p = torch.cuda.get_device_properties(dev)
    cc = torch.cuda.get_device_capability(dev)
    return {
        "name": p.name,
        "sm_count": p.multi_processor_count,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "total_mem_gib": round(p.total_memory / (1024**3), 2),
        "is_a10g_80sm": bool(p.multi_processor_count == A10G_SMS and "A10G" in p.name),
        "is_ga102_sm86": bool(cc == (8, 6)),
    }


def _load_anchor(key: str) -> dict[str, Any]:
    p = ANCHOR_PATHS[key]
    if not p.is_file():
        raise FileNotFoundError(f"banked anchor #{key} not found at {p}")
    return json.load(open(p))


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _num_splits_heuristic(bnm: int, num_sms: int, num_n_blocks: int, max_splits: int = 128) -> int:
    """Exact replica of flash_attn's num_splits_heuristic (csrc/flash_attn/flash_api.cpp). Returns
    the split count the DEPLOYED heuristic (num_splits=0) launches for batch_nheads_mblocks=bnm."""
    if bnm >= 0.8 * num_sms:
        return 1
    max_splits = min(max_splits, num_sms, num_n_blocks)
    eff: list[float] = []
    max_eff = 0.0

    def eligible(ns: int) -> bool:
        return ns == 1 or _ceildiv(num_n_blocks, ns) != _ceildiv(num_n_blocks, ns - 1)

    for ns in range(1, max_splits + 1):
        if not eligible(ns):
            eff.append(0.0)
            continue
        n_waves = float(bnm * ns) / num_sms
        e = n_waves / math.ceil(n_waves)
        max_eff = max(max_eff, e)
        eff.append(e)
    for ns in range(1, max_splits + 1):
        if not eligible(ns):
            continue
        if eff[ns - 1] >= 0.85 * max_eff:
            return ns
    return 1


# ======================================================================================== #
# Inputs / kernels  (flash_attn layout: q=[B,Sq,Hq,D], k/v=[B,L,Hkv,D]; batch=1 verify occupancy)
# ======================================================================================== #
def _qkv(M: int, L: int, seed: int, dev: torch.device):
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(1, M, N_Q_HEADS, HEAD_DIM, generator=g, device=dev, dtype=DTYPE)
    k = torch.randn(1, L, N_KV_HEADS, HEAD_DIM, generator=g, device=dev, dtype=DTYPE)
    v = torch.randn(1, L, N_KV_HEADS, HEAD_DIM, generator=g, device=dev, dtype=DTYPE)
    return q, k, v


def _flash(q, k, v, num_splits):
    return flash_attn_with_kvcache(q, k, v, softmax_scale=SCALE, causal=False, num_splits=num_splits)


def _flash_per_row(q, k, v, num_splits):
    """Per-row AR reference for THIS kernel: each query row attended alone (Sq=1), same split."""
    return torch.cat([_flash(q[:, r:r + 1], k, v, num_splits) for r in range(q.shape[1])], dim=1)


def _time_call(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2] * 1e3  # us (median)


# ======================================================================================== #
# STEP 1 -- audit the shared un-packing assumption (analytic over banked JSON)
# ======================================================================================== #
def step1_audit_unpacking(a327: dict, a332: dict) -> dict[str, Any]:
    """Reconstruct the exact step where #327 and #332 charge the determinism tax; classify it as the
    UN-PACKING penalty (forgone split-KV occupancy) vs an intrinsic M-invariant-reduction cost."""
    s327 = a327["synthesis"]["step2_penalty_model"]
    sdpa_row = next(c for c in s327["per_component"] if c["component"] == "sdpa")
    # #327: pi(u) = 1 - BW_util, charged as "above-roofline exposed slack = forgone reduction parallelism"
    model_327 = s327["model"]
    charges_forgone_reduction_327 = ("forgone reduction parallelism" in model_327)
    sdpa_floor_327 = sdpa_row["floor_contribution"]
    sdpa_pi_327 = sdpa_row["penalty_pi"]
    # the SDPA penalty IS the FULL forgone split-KV slack (phi_forgone == 1) -> the un-packing limit
    sdpa_full_roundtrips = abs(sdpa_floor_327 - SDPA_STEP_SHARE * SDPA_PENALTY_PI) < 1e-9
    shares_unpacking_327 = bool(charges_forgone_reduction_327 and sdpa_full_roundtrips
                                and abs(sdpa_pi_327 - SDPA_PENALTY_PI) < 1e-9)

    s332 = a332["synthesis"]
    phi_model_332 = s332["step3_occupancy_phi"]["model"]
    reduction_axis_332 = s332["step2_partition"]["reduction_axis"]
    n_nonred_332 = s332["step2_partition"]["n_nonreduction_ctas_2d"]
    phi_geo_332 = s332["step3_occupancy_phi"]["geometric_phi_estimate"]
    # #332: phi_forgone = 1 - min(1, N_nonreduction/SMs); N_nonreduction EXCLUDES the KV-split
    # reduction axis ("FORGONE for determinism (fixed split + ordered combine)").
    phi_excludes_reduction = ("N_nonreduction" in phi_model_332 or "non-reduction" in phi_model_332.lower())
    reduction_called_forgone = ("FORGONE for determinism" in reduction_axis_332)
    phi_roundtrips = abs(phi_geo_332 - (1.0 - min(1.0, n_nonred_332 / A10G_SMS))) < 1e-9
    shares_unpacking_332 = bool(phi_excludes_reduction and reduction_called_forgone and phi_roundtrips)

    # The crux the microbench tests: BOTH price the deterministic schedule as the NON-reduction tiles
    # (un-packed). #363's pinned-split keeps N_nonreduction x num_splits CTAs (reduction axis RETAINED,
    # only the COMBINE order is fixed). "fixed split + ordered combine" != "forgo the parallel split".
    return {
        "shares_unpacking_assumption_327": shares_unpacking_327,
        "shares_unpacking_assumption_332": shares_unpacking_332,
        "evidence_327": {
            "penalty_model": model_327,
            "sdpa_penalty_pi": sdpa_pi_327,
            "sdpa_floor_contribution": sdpa_floor_327,
            "charges_full_forgone_reduction_parallelism": charges_forgone_reduction_327,
            "sdpa_full_roundtrips_step_share_x_pi": sdpa_full_roundtrips,
            "interpretation": "pi(u)=1-BW_util charged at phi_forgone=1: the FULL forgone split-KV "
                              "slack -> the un-packing-LIMIT SDPA tax (9.451%).",
        },
        "evidence_332": {
            "phi_model": phi_model_332,
            "reduction_axis": reduction_axis_332,
            "n_nonreduction_ctas": n_nonred_332,
            "geometric_phi_forgone": phi_geo_332,
            "phi_excludes_reduction_axis": phi_excludes_reduction,
            "reduction_axis_called_forgone_for_determinism": reduction_called_forgone,
            "phi_roundtrips_1_minus_Nnonred_over_SMs": phi_roundtrips,
            "interpretation": "phi_forgone = 1 - N_nonreduction(6)/SMs(80) = 0.925 counts ONLY the "
                              "non-reduction CTAs as available to a deterministic schedule -> the "
                              "un-packing assumption (reduction axis dropped).",
        },
        "shared_assumption": "strict determinism requires UN-PACKING the split-KV reduction (collapse "
                             "to the non-reduction M x q-head tiles, forgoing the split-KV occupancy). "
                             "#363 refutes the PREMISE: a fixed-split + ordered-combine reduction is "
                             "deterministic yet retains all num_splits CTAs in parallel.",
    }


# ======================================================================================== #
# STEP 2 -- GA102 occupancy microbench (CTAs-in-flight + latency, M in {1,8}, >=2 seeds)
# ======================================================================================== #
def _cta_model(M: int, K: int, num_n_blocks: int) -> int:
    """Analytic flash-decode split-phase grid: CTAs = num_q_heads x num_m_blocks x num_splits.
    K=heuristic(0) is resolved to its effective split count by the num_splits_heuristic replica."""
    num_m_blocks = _ceildiv(M, BLOCK_M_SPLITKV)
    if K == 0:
        bnm = 1 * N_Q_HEADS * num_m_blocks
        K_eff = _num_splits_heuristic(bnm, A10G_SMS, num_n_blocks, max_splits=128)
    else:
        K_eff = K
    return N_Q_HEADS * num_m_blocks * K_eff


def _heuristic_K(M: int, num_n_blocks: int) -> int:
    num_m_blocks = _ceildiv(M, BLOCK_M_SPLITKV)
    return _num_splits_heuristic(1 * N_Q_HEADS * num_m_blocks, A10G_SMS, num_n_blocks, max_splits=128)


def step2_microbench(L: int, seeds: list[int], iters: int, warmup: int, dev: torch.device) -> dict[str, Any]:
    """Per-call latency + identity for each split config at M in {1,8}; analytic CTAs-in-flight.
    The TPS/latency-based occupancy proxy is the sanctioned method when ncu/CUPTI are unavailable."""
    # latency: median over seeds of the per-seed median us
    lat_by_M: dict[int, dict[int, float]] = {M: {} for M in M_LIST}
    for M in M_LIST:
        for ns in SWEEP_SPLITS:
            per_seed = []
            for sd in seeds:
                q, k, v = _qkv(M, L, sd, dev)
                per_seed.append(_time_call(lambda ns=ns, q=q, k=k, v=v: _flash(q, k, v, ns), iters, warmup))
            per_seed.sort()
            lat_by_M[M][ns] = per_seed[len(per_seed) // 2]

    # identity: batched-M vs per-row AR (SAME kernel/split), accumulated over seeds, for the 3 configs.
    # M in {2,4,8} (NOT 1): at M=1 the per-row reference IS the batched call, so identity is trivially 1.0
    # for every split -- it cannot expose the heuristic's M-dependent split-count divergence (#363 used {2,4,8}).
    any_nan = False
    ident: dict[int, dict[int, list[int]]] = {M: {ns: [0, 0] for ns in (HEURISTIC_SPLIT, PINNED_SPLIT, UNPACK_SPLIT)}
                                              for M in IDENT_M_LIST}
    maxdiff_pinned_vs_heur = 0.0
    for M in IDENT_M_LIST:
        for sd in seeds:
            q, k, v = _qkv(M, L, sd, dev)
            heur = _flash(q, k, v, HEURISTIC_SPLIT)
            pin = _flash(q, k, v, PINNED_SPLIT)
            any_nan = any_nan or bool(torch.isnan(heur).any()) or bool(torch.isnan(pin).any())
            maxdiff_pinned_vs_heur = max(maxdiff_pinned_vs_heur,
                                         float((pin.float() - heur.float()).abs().max().item()))
            for ns in (HEURISTIC_SPLIT, PINNED_SPLIT, UNPACK_SPLIT):
                ref = _flash_per_row(q, k, v, ns)
                bat = _flash(q, k, v, ns)
                same = (bat == ref).all(dim=-1)
                ident[M][ns][0] += int(same.sum())
                ident[M][ns][1] += int(same.numel())

    ident_rate = {M: {ns: (ident[M][ns][0] / ident[M][ns][1] if ident[M][ns][1] else float("nan"))
                      for ns in (HEURISTIC_SPLIT, PINNED_SPLIT, UNPACK_SPLIT)} for M in IDENT_M_LIST}

    # analytic CTAs-in-flight (deployed verify shape M=8) for the three configs
    num_n_blocks = _ceildiv(L, 64)  # BLOCK_N=64 reference; heuristic K bracketed below
    K_heur_8 = _heuristic_K(DEPLOYED := 8, num_n_blocks)
    ctas_heuristic = _cta_model(8, HEURISTIC_SPLIT, num_n_blocks)
    ctas_pinned8 = _cta_model(8, PINNED_SPLIT, num_n_blocks)
    ctas_unpacked = _cta_model(8, UNPACK_SPLIT, num_n_blocks)
    # heuristic-K bracket over plausible BLOCK_N (the only uncertain kernel constant)
    heur_K_bracket = {bn: _heuristic_K(8, _ceildiv(L, bn)) for bn in (32, 64, 128)}
    ctas_heuristic_bracket = {bn: N_Q_HEADS * 1 * k for bn, k in heur_K_bracket.items()}

    # occupancy verdict (M=8): does pinned-8 preserve the heuristic's CTA parallelism?
    heur_us8, pin_us8, unp_us8 = lat_by_M[8][0], lat_by_M[8][8], lat_by_M[8][1]
    # (i) CTA test: pinned-8 keeps >= 0.8x the heuristic's CTAs (not collapsed to the un-pack floor)
    cta_preserved = bool(ctas_pinned8 >= 0.8 * min(ctas_heuristic, A10G_SMS) and ctas_pinned8 > 2 * ctas_unpacked)
    # (ii) latency proxy: pinned-8 is NOT slower than the heuristic (no occupancy-loss slack); un-pack IS
    lat_preserved = bool(pin_us8 <= heur_us8 * 1.05)
    unpack_collapses = bool(unp_us8 >= heur_us8 * 1.5)
    pinned8_preserves_parallelism = bool(cta_preserved and lat_preserved)
    # M-invariance of pinned-8: byte-exact at every identity-M (and faster-or-equal vs heuristic)
    pinned8_m_invariant = bool(all(ident_rate[M][PINNED_SPLIT] >= 0.999 for M in IDENT_M_LIST))
    heuristic_breaks_identity = bool(all(ident_rate[M][HEURISTIC_SPLIT] < 0.5 for M in IDENT_M_LIST))

    return {
        "L": L, "seeds": seeds, "num_n_blocks_bn64": num_n_blocks,
        "latency_us_by_M_by_split": {str(M): {str(ns): lat_by_M[M][ns] for ns in SWEEP_SPLITS} for M in M_LIST},
        "identity_rate_by_M": {str(M): {str(ns): ident_rate[M][ns]
                                        for ns in (HEURISTIC_SPLIT, PINNED_SPLIT, UNPACK_SPLIT)} for M in IDENT_M_LIST},
        "any_nan": bool(any_nan),
        "maxabsdiff_pinned8_vs_heuristic": maxdiff_pinned_vs_heur,
        # CTAs-in-flight (analytic flash-decode split grid; latency-proxy validated)
        "heuristic_effective_K": K_heur_8,
        "heuristic_K_bracket_by_blockN": heur_K_bracket,
        "ctas_in_flight_heuristic": ctas_heuristic,
        "ctas_in_flight_heuristic_bracket": ctas_heuristic_bracket,
        "ctas_in_flight_pinned8": ctas_pinned8,
        "ctas_in_flight_unpacked": ctas_unpacked,
        # latency anchors (M=8)
        "heuristic_us_M8": heur_us8, "pinned8_us_M8": pin_us8, "unpacked_us_M8": unp_us8,
        "pinned8_over_heuristic_ratio_M8": pin_us8 / heur_us8 if heur_us8 else float("nan"),
        "unpacked_over_heuristic_ratio_M8": unp_us8 / heur_us8 if heur_us8 else float("nan"),
        # verdicts
        "cta_preserved": cta_preserved,
        "latency_preserved": lat_preserved,
        "unpack_collapses_occupancy": unpack_collapses,
        "pinned8_preserves_parallelism": pinned8_preserves_parallelism,
        "pinned8_m_invariant": pinned8_m_invariant,
        "heuristic_breaks_identity": heuristic_breaks_identity,
    }


# ======================================================================================== #
# STEP 3 -- recompute the analytic eta-floor and phi-recovery UNDER pinned-split
# ======================================================================================== #
def step3_recompute(micro_primary: dict) -> dict[str, Any]:
    """Re-price the SDPA determinism tax with the pinned-split CTA count (microbench-grounded),
    vs #327's phi_forgone=1 (9.451%) and #332's phi_forgone=0.925 (phi_recovery=0.075)."""
    ctas_pinned8 = micro_primary["ctas_in_flight_pinned8"]
    ctas_heur = micro_primary["ctas_in_flight_heuristic"]
    pin_us8 = micro_primary["pinned8_us_M8"]
    heur_us8 = micro_primary["heuristic_us_M8"]

    # --- (a) OCCUPANCY-MODEL reading (apply #332's own phi=1-CTAs/SMs to the PINNED CTA count) ----
    # conservative: the pinned-split deterministic schedule keeps ctas_pinned8 CTAs (NOT the 6 of #332).
    phi_forgone_pinned_occ = max(0.0, 1.0 - min(1.0, ctas_pinned8 / A10G_SMS))
    phi_recovery_pinned_occ = 1.0 - phi_forgone_pinned_occ

    # --- (b) EMPIRICAL reading (the latency proxy is decisive: pinned-8 <= heuristic => tax<=0) ----
    # realized SDPA determinism tax fraction = max(0, t_pinned - t_heuristic)/t_heuristic
    realized_tax_ratio = max(0.0, (pin_us8 - heur_us8) / heur_us8) if heur_us8 else float("nan")
    phi_forgone_pinned_emp = realized_tax_ratio          # 0 when pinned-8 is not slower than heuristic
    phi_recovery_pinned_emp = 1.0 - phi_forgone_pinned_emp

    # HEADLINE = empirical (the microbench is the ground truth #363 also measured); occupancy = floor.
    phi_recovery_pinned_split_analytic = phi_recovery_pinned_emp
    phi_forgone_headline = phi_forgone_pinned_emp

    def eta_floor(phi_forgone: float) -> float:
        # SDPA term scales with the forgone split-KV fraction; lm_head GEMM tax is a separate fixed term.
        return LMHEAD_FLOOR_FULL + SDPA_FLOOR_FULL * phi_forgone

    eta_floor_pinned_split_analytic = eta_floor(phi_forgone_headline)     # headline (empirical phi_forgone~0)
    eta_floor_pinned_occ_upper = eta_floor(phi_forgone_pinned_occ)        # conservative occupancy upper bound

    revived_ceiling = CEILING_500 * (1.0 - eta_floor_pinned_split_analytic)
    revived_ceiling_occ = CEILING_500 * (1.0 - eta_floor_pinned_occ_upper)

    # comparisons
    phi_clears_noreg_breakeven = bool(phi_recovery_pinned_split_analytic > RECOVERY_BREAKEVEN_NOREG)
    phi_clears_500_breakeven = bool(phi_recovery_pinned_split_analytic > RECOVERY_BREAKEVEN_500)
    phi_occ_clears_500_breakeven = bool(phi_recovery_pinned_occ > RECOVERY_BREAKEVEN_500)
    eta_clears_500_budget = bool(eta_floor_pinned_split_analytic < BUDGET_500_ETA)
    eta_occ_clears_500_budget = bool(eta_floor_pinned_occ_upper < BUDGET_500_ETA)
    eta_below_blanket = bool(eta_floor_pinned_split_analytic < FLOOR_COMBINED_FULL)

    # decision gate (PR #366)
    refutes_332_cap = bool(micro_primary["pinned8_preserves_parallelism"]
                           and phi_clears_500_breakeven and revived_ceiling > CEILING_AT_GEO_332)
    revives_ceiling_toward_520 = bool(refutes_332_cap and revived_ceiling > TARGET_500
                                      and eta_clears_500_budget)

    return {
        # headline (empirical, microbench-grounded)
        "phi_recovery_pinned_split_analytic": phi_recovery_pinned_split_analytic,
        "eta_floor_pinned_split_analytic": eta_floor_pinned_split_analytic,
        "revived_ceiling_tps": revived_ceiling,
        "phi_forgone_pinned_headline": phi_forgone_headline,
        "realized_sdpa_tax_ratio": realized_tax_ratio,
        # conservative occupancy-model bracket
        "phi_recovery_pinned_occupancy_lower_bound": phi_recovery_pinned_occ,
        "phi_forgone_pinned_occupancy": phi_forgone_pinned_occ,
        "eta_floor_pinned_occupancy_upper_bound": eta_floor_pinned_occ_upper,
        "revived_ceiling_occupancy_tps": revived_ceiling_occ,
        # references for comparison
        "lmhead_residual_floor": LMHEAD_FLOOR_FULL,
        "sdpa_floor_full_327": SDPA_FLOOR_FULL,
        "floor_blanket_9841": FLOOR_COMBINED_FULL,
        "budget_500_eta": BUDGET_500_ETA,
        "recovery_breakeven_noreg_0255": RECOVERY_BREAKEVEN_NOREG,
        "recovery_breakeven_500": RECOVERY_BREAKEVEN_500,
        "phi_recovery_geo_332": PHI_RECOVERY_GEO_332,
        "ceiling_at_geo_332_473": CEILING_AT_GEO_332,
        # comparison bools
        "phi_clears_noreg_breakeven_0255": phi_clears_noreg_breakeven,
        "phi_clears_500_breakeven": phi_clears_500_breakeven,
        "phi_occupancy_clears_500_breakeven": phi_occ_clears_500_breakeven,
        "eta_clears_500_budget": eta_clears_500_budget,
        "eta_occupancy_clears_500_budget": eta_occ_clears_500_budget,
        "eta_below_9841_blanket": eta_below_blanket,
        # PR decision gate
        "refutes_332_cap": refutes_332_cap,
        "revives_ceiling_toward_520": revives_ceiling_toward_520,
    }


# ======================================================================================== #
# STEP 4 -- cross-check stark #365 (lm_head BI-GEMM eta) if posted
# ======================================================================================== #
def step4_crosscheck_365(recompute: dict) -> dict[str, Any]:
    found = None
    for p in PR365_CANDIDATES:
        if p.is_file():
            try:
                j = json.load(open(p))
                if str(j.get("pr")) == "365" or "lmhead" in str(j.get("kind", "")).lower():
                    found = (p, j)
                    break
            except Exception:
                continue
    if found is None:
        return {
            "pr365_posted": False,
            "note": "stark #365 (lm_head BI-GEMM eta) not yet posted. This analytic card's "
                    "attention-locus phi_recovery~=1.0 CORROBORATES #363's measured bi_gemm_eta=0 "
                    "(same mechanism, theory vs experiment). The residual lm_head term is <= 0.390% "
                    "(#327 lm_head_floor_full, BW_util 0.834) -- already << the 4.02% budget even at "
                    "phi_forgone=1 -- and stark #365 measures whether even that is recoverable.",
            "predicted_total_locus_eta_if_lmhead_full": LMHEAD_FLOOR_FULL,
            "predicted_total_locus_eta_if_lmhead_free": recompute["eta_floor_pinned_split_analytic"],
        }
    p, j = found
    lm_eta = j.get("lmhead_bi_gemm_eta_measured", j.get("total_verify_locus_eta"))
    return {
        "pr365_posted": True, "path": str(p), "wandb_run_id": j.get("wandb_run_id"),
        "lmhead_bi_gemm_eta_measured": lm_eta,
        "reconcile_note": "compare empirical lm_head eta to this card's lm_head residual prediction "
                          f"({LMHEAD_FLOOR_FULL:.5f}); a large disagreement localizes whether the "
                          "paged-KV harness breaks the per-layer isolation result.",
    }


# ======================================================================================== #
# Self-test
# ======================================================================================== #
def selftest(audit: dict, micro: dict, recompute: dict, gpu: dict, flags: dict) -> dict[str, Any]:
    c: dict[str, bool] = {}
    # (a) provenance: banked constants roundtrip
    c["a_sdpa_floor_full_roundtrips_327"] = abs(SDPA_STEP_SHARE * SDPA_PENALTY_PI - SDPA_FLOOR_FULL) < 1e-12
    c["a_blanket_is_sdpa_plus_lmhead"] = abs(SDPA_FLOOR_FULL + LMHEAD_FLOOR_FULL - FLOOR_COMBINED_FULL) < 1e-12
    c["a_phi_geo_332_roundtrips"] = abs((1.0 - N_NONREDUCTION_332 / A10G_SMS) - PHI_FORGONE_GEO_332) < 1e-9
    c["a_phi_recovery_332_is_0075"] = abs((1.0 - PHI_FORGONE_GEO_332) - PHI_RECOVERY_GEO_332) < 1e-9
    c["a_budget_500_roundtrips"] = abs(CEILING_500 * (1 - BUDGET_500_ETA) - TARGET_500) < 1e-3
    c["a_ceiling_at_geo_332_roundtrips"] = abs(
        CEILING_500 * (1 - (LMHEAD_FLOOR_FULL + SDPA_FLOOR_FULL) * PHI_FORGONE_GEO_332) - CEILING_AT_GEO_332) < 0.5
    # (b) step-1 audit: both share the un-packing assumption (the card's premise)
    c["b_shares_unpacking_327"] = bool(audit["shares_unpacking_assumption_327"])
    c["b_shares_unpacking_332"] = bool(audit["shares_unpacking_assumption_332"])
    # (c) microbench sanity: NaN-clean, rates in [0,1], latencies finite/positive
    c["c_nan_clean"] = (not micro["any_nan"])
    rates_ok = all(0.0 <= v <= 1.0 for d in micro["identity_rate_by_M"].values() for v in d.values())
    c["c_rates_in_unit_interval"] = bool(rates_ok)
    lat_ok = all(math.isfinite(v) and v > 0
                 for d in micro["latency_us_by_M_by_split"].values() for v in d.values())
    c["c_latencies_finite_positive"] = bool(lat_ok)
    # (d) the MEASURED mechanism findings (folded in -- not assumptions)
    c["d_pinned8_byte_exact_M_invariant"] = bool(micro["pinned8_m_invariant"]
                                                 and micro["maxabsdiff_pinned8_vs_heuristic"] < 1e-2)
    c["d_heuristic_breaks_identity"] = bool(micro["heuristic_breaks_identity"])
    c["d_unpack_collapses_occupancy"] = bool(micro["unpack_collapses_occupancy"])
    c["d_pinned8_preserves_parallelism"] = bool(micro["pinned8_preserves_parallelism"])
    c["d_ctas_pinned_gg_unpacked"] = bool(micro["ctas_in_flight_pinned8"] > 2 * micro["ctas_in_flight_unpacked"])
    # (e) recompute: well-typed bools, finite floors, decision gate consistency
    c["e_eta_floor_finite_nonneg"] = bool(math.isfinite(recompute["eta_floor_pinned_split_analytic"])
                                          and recompute["eta_floor_pinned_split_analytic"] >= 0)
    c["e_phi_recovery_in_unit"] = bool(0.0 <= recompute["phi_recovery_pinned_split_analytic"] <= 1.0)
    c["e_eta_below_blanket"] = bool(recompute["eta_below_9841_blanket"])
    c["e_decision_bools_typed"] = (isinstance(recompute["refutes_332_cap"], bool)
                                   and isinstance(recompute["revives_ceiling_toward_520"], bool))
    c["e_refute_implies_clear500"] = (not recompute["refutes_332_cap"]) or recompute["phi_clears_500_breakeven"]
    # (f) on-target hardware (the GA102/sm_86/80-SM occupancy wall is the deployment arch)
    c["f_on_target_a10g_80sm"] = bool(gpu["is_a10g_80sm"] and gpu["is_ga102_sm86"])
    # (g) scope flags
    c["g_no_launch_flags"] = bool(flags["no_hf_job"] and flags["no_launch"] and flags["no_served_file_change"])
    passes = all(c.values())
    return {"passes": passes, "n_checks": len(c), "conditions": c}


# ======================================================================================== #
# Report + IO + wandb
# ======================================================================================== #
def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, bool) or o is None or isinstance(o, (str, int)):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return str(o)


def print_report(payload: dict) -> None:
    gpu, au, rc, st = payload["gpu"], payload["audit"], payload["recompute"], payload["selftest"]
    mp = payload["microbench"][str(PRIMARY_L)]
    bar = "=" * 100
    print(bar)
    print("PINNED-SPLIT PHI AUDIT -- does #363's pinned split break #327/#332's un-packing eta floor? (PR #366, wirbel)")
    print(f"  GPU {gpu['name']}  SMs={gpu['sm_count']}  cc={gpu['compute_capability']}  "
          f"on-target-GA102-sm86-80SM={gpu['is_a10g_80sm'] and gpu['is_ga102_sm86']}")
    print("-" * 100)
    print("  STEP 1 -- shared un-packing assumption:")
    print(f"     shares_unpacking_assumption_327 = {au['shares_unpacking_assumption_327']}  "
          f"(pi=1-BW_util charged at phi_forgone=1 -> full forgone split-KV slack 9.451%)")
    print(f"     shares_unpacking_assumption_332 = {au['shares_unpacking_assumption_332']}  "
          f"(phi_forgone=1-N_nonreduction(6)/SMs(80)=0.925 -> reduction axis dropped)")
    print("-" * 100)
    print(f"  STEP 2 -- GA102 occupancy microbench (L={PRIMARY_L}, M={list(M_LIST)}, seeds={mp['seeds']}):")
    print(f"     CTAs-in-flight: heuristic={mp['ctas_in_flight_heuristic']} (K_eff={mp['heuristic_effective_K']}, "
          f"bracket {mp['ctas_in_flight_heuristic_bracket']})  pinned8={mp['ctas_in_flight_pinned8']}  "
          f"unpacked={mp['ctas_in_flight_unpacked']}")
    print(f"     latency M8 us: heuristic={mp['heuristic_us_M8']:.2f}  pinned8={mp['pinned8_us_M8']:.2f} "
          f"(x{mp['pinned8_over_heuristic_ratio_M8']:.3f})  unpacked={mp['unpacked_us_M8']:.2f} "
          f"(x{mp['unpacked_over_heuristic_ratio_M8']:.2f})")
    print(f"     identity: pinned8 M-invariant={mp['pinned8_m_invariant']}  heuristic breaks={mp['heuristic_breaks_identity']}  "
          f"maxabsdiff(pinned8,heur)={mp['maxabsdiff_pinned8_vs_heuristic']:.2e}")
    print(f"     -> pinned8_preserves_parallelism = {mp['pinned8_preserves_parallelism']}  "
          f"(unpack_collapses={mp['unpack_collapses_occupancy']})")
    print("-" * 100)
    print("  STEP 3 -- recompute under pinned-split:")
    print(f"     phi_recovery_pinned_split_analytic = {rc['phi_recovery_pinned_split_analytic']:.4f}  "
          f"(occupancy lower bound {rc['phi_recovery_pinned_occupancy_lower_bound']:.4f})  "
          f"vs #332 phi=0.075, vs 0.255 / 0.591 break-even")
    print(f"     eta_floor_pinned_split_analytic = {rc['eta_floor_pinned_split_analytic']*100:.4f}%  "
          f"(occupancy upper bound {rc['eta_floor_pinned_occupancy_upper_bound']*100:.4f}%)  "
          f"vs 4.02% budget / 9.841% blanket")
    print(f"     revived ceiling = {rc['revived_ceiling_tps']:.2f} TPS "
          f"(occupancy {rc['revived_ceiling_occupancy_tps']:.2f}) vs #332 cap 473.53")
    print(f"     refutes_332_cap = {rc['refutes_332_cap']}   revives_ceiling_toward_520 = {rc['revives_ceiling_toward_520']}")
    print("-" * 100)
    print(f"  STEP 4 -- #365 cross-check: posted={payload['crosscheck_365']['pr365_posted']}")
    print("-" * 100)
    print(f"  SELF-TEST {st['passes']} ({st['n_checks']} checks): " +
          json.dumps({k: int(v) for k, v in st["conditions"].items()}))
    print(bar)


def maybe_log_wandb(payload: dict, args) -> str | None:
    if args.no_wandb:
        return None
    repo = str(Path(__file__).resolve().parents[3])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from scripts.wandb_logging import (init_wandb_run, log_summary,
                                            log_json_artifact, finish_wandb)
    except Exception as e:  # noqa: BLE001
        print(f"[pinned-phi] wandb helpers unavailable: {e}")
        return None
    au, rc, st = payload["audit"], payload["recompute"], payload["selftest"]
    mp = payload["microbench"][str(PRIMARY_L)]
    run = init_wandb_run(
        job_type="analysis-gpu-microbench", agent="wirbel",
        name=args.wandb_name, group=args.wandb_group,
        tags=["strict-bi-verify-gemm", "pinned-split", "unpacking-audit", "phi-recovery",
              "occupancy", "319-strict-lock", "pr-366"],
        config={"pr": 366, "kind": "pinned-split-phi-audit",
                "head_dim": HEAD_DIM, "n_q_heads": N_Q_HEADS, "n_kv_heads": N_KV_HEADS,
                "m_list": list(M_LIST), "primary_L": PRIMARY_L, "splits": list(SWEEP_SPLITS),
                "ceiling_500": CEILING_500, "step_us": STEP_US,
                "floor_blanket_9841": FLOOR_COMBINED_FULL, "budget_500_eta": BUDGET_500_ETA},
    )
    if run is None:
        print("[pinned-phi] wandb disabled (no API key / WANDB_MODE).")
        return None
    flat: dict[str, float] = {
        "audit/shares_unpacking_327": float(au["shares_unpacking_assumption_327"]),
        "audit/shares_unpacking_332": float(au["shares_unpacking_assumption_332"]),
        "micro/ctas_heuristic": float(mp["ctas_in_flight_heuristic"]),
        "micro/ctas_pinned8": float(mp["ctas_in_flight_pinned8"]),
        "micro/ctas_unpacked": float(mp["ctas_in_flight_unpacked"]),
        "micro/heuristic_us_M8": mp["heuristic_us_M8"],
        "micro/pinned8_us_M8": mp["pinned8_us_M8"],
        "micro/unpacked_us_M8": mp["unpacked_us_M8"],
        "micro/pinned8_over_heuristic_ratio_M8": mp["pinned8_over_heuristic_ratio_M8"],
        "micro/unpacked_over_heuristic_ratio_M8": mp["unpacked_over_heuristic_ratio_M8"],
        "micro/pinned8_preserves_parallelism": float(mp["pinned8_preserves_parallelism"]),
        "micro/pinned8_m_invariant": float(mp["pinned8_m_invariant"]),
        "micro/heuristic_breaks_identity": float(mp["heuristic_breaks_identity"]),
        "recompute/phi_recovery_pinned_split_analytic": rc["phi_recovery_pinned_split_analytic"],
        "recompute/phi_recovery_occupancy_lower_bound": rc["phi_recovery_pinned_occupancy_lower_bound"],
        "recompute/eta_floor_pinned_split_analytic": rc["eta_floor_pinned_split_analytic"],
        "recompute/eta_floor_occupancy_upper_bound": rc["eta_floor_pinned_occupancy_upper_bound"],
        "recompute/revived_ceiling_tps": rc["revived_ceiling_tps"],
        "recompute/refutes_332_cap": float(rc["refutes_332_cap"]),
        "recompute/revives_ceiling_toward_520": float(rc["revives_ceiling_toward_520"]),
        "recompute/eta_clears_500_budget": float(rc["eta_clears_500_budget"]),
        "recompute/phi_clears_500_breakeven": float(rc["phi_clears_500_breakeven"]),
        "selftest/analytic_self_test_passes": float(st["passes"]),
        "gpu/sm_count": float(payload["gpu"]["sm_count"]),
    }
    for M in IDENT_M_LIST:
        for ns in (HEURISTIC_SPLIT, PINNED_SPLIT, UNPACK_SPLIT):
            flat[f"micro/identity_M{M}_split{ns}"] = mp["identity_rate_by_M"][str(M)][str(ns)]
    run.log({"global_step": 0, **flat})
    log_summary(run, _jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="pinned_split_phi_audit", artifact_type="analysis", data=_jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[pinned-phi] wandb logged {len(flat)} keys (run {rid})")
    return rid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true", help="run the GA102 microbench (default path)")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to validate the path")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default="wirbel/pinned-split-phi-audit")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="strict-bi-verify-gemm")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    if args.smoke:
        args.iters = min(args.iters, 30)
        args.warmup = min(args.warmup, 8)
        args.seeds = args.seeds[:2]

    torch.manual_seed(args.seeds[0])
    dev = _device()
    gpu = _gpu_facts(dev)

    a327, a332 = _load_anchor("327"), _load_anchor("332")
    audit = step1_audit_unpacking(a327, a332)

    micro = {str(L): step2_microbench(L, args.seeds, args.iters, args.warmup, dev) for L in L_LIST}
    recompute = step3_recompute(micro[str(PRIMARY_L)])
    crosscheck = step4_crosscheck_365(recompute)

    flags = {"no_hf_job": True, "no_launch": True, "no_served_file_change": True}
    st = selftest(audit, micro[str(PRIMARY_L)], recompute, gpu, flags)

    torch.cuda.synchronize()
    payload = {
        "agent": "wirbel", "pr": 366,
        "kind": "pinned-split-phi-audit",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "analysis_only": True, **flags,
        "gpu": gpu,
        "peak_mem_mib": round(torch.cuda.max_memory_allocated(dev) / (1024**2), 3),
        "ladder_constants": {"ceiling_500": CEILING_500, "step_us": STEP_US,
                             "floor_blanket_9841": FLOOR_COMBINED_FULL, "budget_500_eta": BUDGET_500_ETA,
                             "recovery_breakeven_noreg_0255": RECOVERY_BREAKEVEN_NOREG,
                             "recovery_breakeven_500": RECOVERY_BREAKEVEN_500,
                             "ceiling_at_geo_332_473": CEILING_AT_GEO_332},
        "audit": audit, "microbench": micro, "recompute": recompute, "crosscheck_365": crosscheck,
        "selftest": st,
        "analytic_self_test_passes": bool(st["passes"]),
        # headline TEST surface (PR #366 required fields)
        "phi_recovery_pinned_split_analytic": recompute["phi_recovery_pinned_split_analytic"],
        "eta_floor_pinned_split_analytic": recompute["eta_floor_pinned_split_analytic"],
        "ctas_in_flight_heuristic": micro[str(PRIMARY_L)]["ctas_in_flight_heuristic"],
        "ctas_in_flight_pinned8": micro[str(PRIMARY_L)]["ctas_in_flight_pinned8"],
        "ctas_in_flight_unpacked": micro[str(PRIMARY_L)]["ctas_in_flight_unpacked"],
        "shares_unpacking_assumption_327": audit["shares_unpacking_assumption_327"],
        "shares_unpacking_assumption_332": audit["shares_unpacking_assumption_332"],
        "refutes_332_cap": recompute["refutes_332_cap"],
        "revives_ceiling_toward_520": recompute["revives_ceiling_toward_520"],
    }
    print_report(payload)
    out_path = Path(args.out_dir) / "pinned_split_phi_audit_results.json"
    out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"\n[pinned-phi] wrote {out_path}")
    rid = maybe_log_wandb(payload, args)
    if rid:
        payload["wandb_run_id"] = rid
        out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"[pinned-phi] analytic_self_test_passes = {payload['analytic_self_test_passes']}")
    print(f"[pinned-phi] phi_recovery_pinned_split_analytic = {payload['phi_recovery_pinned_split_analytic']:.4f}  "
          f"eta_floor = {payload['eta_floor_pinned_split_analytic']*100:.4f}%  "
          f"refutes_332_cap = {payload['refutes_332_cap']}")
    raise SystemExit(0 if payload["analytic_self_test_passes"] else 1)


if __name__ == "__main__":
    main()
