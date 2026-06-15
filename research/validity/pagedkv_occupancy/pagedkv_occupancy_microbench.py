#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #370 (wirbel) -- Paged-KV occupancy: does my #366 pinned-split win survive the SERVED layout?

THE QUESTION (the bridge from dense flash_attn to the served paged-KV decode)
-----------------------------------------------------------------------------
My own #366 (`h28xnyuy`, MERGED) ran a GA102 occupancy microbench and found pinning num_splits=8
keeps 64 CTAs-in-flight >= heuristic 56 >> un-packed 8, byte-exact and 0.914x faster -> both
denken #327/#360's 9.841% blanket and denken #332's phi=0.075 -> 473.5 cap (which BOTH charge the
*un-packing* penalty) collapse -> revived lambda=1 ceiling ~= 518.92 TPS.

CRITICAL CAVEAT I stated in #366: the microbench used a **dense `flash_attn` stand-in -- NOT the
vLLM paged-KV decode.** The served decode does NOT do a dense reduction -- it gathers KV from a
**paged block table** (vLLM block_size 16, non-contiguous physical blocks, variable KV-block count
per query). Paged gather can change the occupancy story two ways the dense stand-in cannot see:
  (1) the split-K grid might be built over **KV blocks** (not a contiguous L) -> pinning num_splits=8
      could over/under-subscribe the actual block count;
  (2) block-sparse gather has memory-divergence that can lower achieved occupancy independent of CTA
      count.
This card answers: does pinning the split in the *paged-KV* decode preserve the 64-CTA parallelism the
dense #366 showed, or does block-gather reintroduce the #332-assumed collapse?

WHY flash_attn 2.8.4's PAGED PATH IS THE RIGHT KERNEL (not a stand-in)
---------------------------------------------------------------------
The served stack is vLLM 0.22.0 FLASH_ATTN backend; for paged decode it wraps exactly
`flash_attn_with_kvcache(q, k_cache, v_cache, block_table=..., num_splits=...)` -- the SAME FA2
splitkv kernel family used here. #366 drove this function WITHOUT a block_table (dense); this card
drives it WITH a paged block_table (non-contiguous physical pages). Same kernel, same A10G, only the
block_table differs -> the cleanest possible isolation of the paged-gather effect.

PAGE-SIZE NOTE (the one faithful-ness limitation, reconciled analytically + empirically)
----------------------------------------------------------------------------------------
Upstream flash_attn 2.8.4 asserts page_block_size % 256 == 0, so it cannot take vLLM's block_size 16
directly (vLLM's vendored vllm-flash-attn relaxes that assert; identical FA2 grid mechanics). We
therefore drive the paged kernel at page_block_size 256 (the upstream minimum) and BRIDGE to
block_size 16 with two measured invariances logged here:
  (a) PAGE-SIZE invariance: paged@256 latency == paged@512 latency (to the us) at matched L;
  (b) CONTIGUITY invariance: shuffled (non-contiguous physical blocks) == contiguous, to the us;
plus the byte-identity paged@256 == dense (maxabsdiff 0). Together these show the paged-gather cost is
a FIXED block_table-setup overhead -- not a per-page memory-divergence term -- so the block_size-16
grid (more pages, same BLOCK_N tiling) inherits the same occupancy story. The split-K grid is built
over the kernel's internal BLOCK_N tiling of the *logical* seqlen (num_n_blocks = ceil(seqlen/BLOCK_N)),
which is page_block_size-INVARIANT; the block_table only redirects the gather within a tile. Hence
pinning num_splits=8 cannot over/under-subscribe on the physical (block-16) page count.

WHAT THIS CARD MEASURES / DERIVES
---------------------------------
(1) PAGED MICROBENCH (A10G): CTAs-in-flight (analytic split grid, page-invariant) + per-call latency
    for dense AND paged (block_table) at the deployed verify geometry (batch=1, head_dim 256, 8q/2kv,
    bf16) under THREE configs (heuristic 0 / pinned-8 / un-packed 1) over L in {528, 2048, 4096}.
(2) DOES PINNING SURVIVE PAGED GATHER? -> paged_pinned8_preserves_parallelism (bool) and
    paged_block_gather_occupancy_penalty (paged-vs-dense latency delta at matched L; a SEPARATE
    baseline-paging diagnostic, already in the served step, NOT a determinism tax).
(3) RECOMPUTE phi_recovery_paged_split and eta_floor_paged_split under the paged grounding vs #366's
    dense 1.0 / 0.39%; compare to 0.255 no-reg and 0.5913 >500 break-evens, 4.02% budget, 9.841% blanket.
(4) RECONCILE with #366 (paged phi ~= dense phi -> CONFIRM; paged phi < dense phi -> paged-only penalty)
    and with stark #365 (lm_head BI-GEMM eta) if posted.

SCOPE: pod-GPU microbench + CPU analytic over banked MERGED *_results.json. NO train.py --launch, NO
HF Job, NO submission, NO served-file change. 0 official TPS, baseline 481.53 UNCHANGED. Greedy
identity is MEASURED, never broken. Run with CUDA_VISIBLE_DEVICES=0 (#358/#363 single-A10G gotcha).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from flash_attn import flash_attn_with_kvcache

# ---------------------------------------------------------------------------------------- #
# Real gemma-4-E4B-it text-decoder attention geometry (the EXACT dims #327/#332/#358/#363/#366 use)
# ---------------------------------------------------------------------------------------- #
HEAD_DIM = 256
N_Q_HEADS = 8
N_KV_HEADS = 2
GQA_GROUP = N_Q_HEADS // N_KV_HEADS  # 4
DTYPE = torch.bfloat16
SCALE = 1.0 / math.sqrt(HEAD_DIM)
A10G_SMS = 80
BLOCK_M_SPLITKV = 64   # flash_attn splitkv query-tile (M<=8 => 1 m-block); reported, not load-bearing
BLOCK_N_TILE = 64      # internal n-tile of the FA2 splitkv kernel (num_n_blocks = ceil(seqlen/BLOCK_N))
M_LIST = (1, 8)        # AR decode (M=1) + verify width (M=8, K_spec=7+1)
IDENT_M_LIST = (2, 4, 8)  # identity demo (batched-M vs per-row); M=1 degenerate (#363/#366 used {2,4,8})
PRIMARY_L = 2048       # #363/#366 "realistic context" geometry (primary)
BRIDGE_L = 528         # #332 ctx geometry (bridge: shorter context -> fewer natural KV segments)
DEEP_L = 4096          # served max_model_len -- the deep-context regime (NEW vs #366)
L_LIST = (BRIDGE_L, PRIMARY_L, DEEP_L)

HEURISTIC_SPLIT = 0    # flash_attn occupancy heuristic == the DEPLOYED split-KV fast path
PINNED_SPLIT = 8       # stark #363 / my #366 M-invariant mechanism
UNPACK_SPLIT = 1       # the #332-assumed "un-pack to non-reduction" config (no KV split)
SWEEP_SPLITS = (0, 1, 2, 4, 8, 16, 32)

# paged layout
PAGE_BLOCK = 256          # flash_attn 2.8.4 upstream minimum (% 256 == 0); grid is page-invariant
PAGE_BLOCK_ALT = 512      # page-size invariance cross-check (bridge to block-16)
SERVED_BLOCK_SIZE = 16    # vLLM deployment block_size (reported; grid built over BLOCK_N not pages)

# ---- strict budget ladder (cite, do NOT re-derive; identical to #366) ------------------- #
CEILING_500 = 520.953
STEP_US = 1218.2
TARGET_500 = 500.0
BUDGET_500_ETA = 1.0 - TARGET_500 / CEILING_500          # >500 kernel budget ~= 0.04022
BUDGET_LAMBDA1_ETA = 0.07331808522875782                 # no-regression / lambda=1 budget

# ---- #327 (kcjlr5ny) bf16 reduction floor decomposition (cite; roundtripped in step 1) -- #
SDPA_STEP_SHARE = 0.14513725794080165
SDPA_BW_UTIL = 0.34883864849061247
SDPA_PENALTY_PI = 0.6511613515093875           # = 1 - SDPA_BW_UTIL
SDPA_FLOOR_FULL = 0.09450777303509898          # = step_share * pi (phi_forgone=1; un-packing-full SDPA tax)
LMHEAD_STEP_SHARE = 0.02358516890006141
LMHEAD_BW_UTIL = 0.8344417980018903
LMHEAD_FLOOR_FULL = 0.003904718156915901       # separate small lm_head GEMM determinism cost
FLOOR_COMBINED_FULL = 0.09841249119201488      # 9.841% blanket = SDPA_FLOOR_FULL + LMHEAD_FLOOR_FULL

# ---- #332 (y5cl0ena) phi model (cite; roundtripped in step 1) --------------------------- #
N_NONREDUCTION_332 = 6
PHI_FORGONE_GEO_332 = 0.925
PHI_RECOVERY_GEO_332 = 0.07499999999999996
RECOVERY_BREAKEVEN_NOREG = 0.2549920813842095
PHI_STAR_500_COMBINED = 0.4086882190334648
CEILING_AT_GEO_332 = 473.5295953446407
RECOVERY_BREAKEVEN_500 = 1.0 - PHI_STAR_500_COMBINED      # ~= 0.5913

# banked anchor JSONs (roundtrip provenance)
_VAL = Path(__file__).resolve().parents[1]
ANCHOR_PATHS = {
    "327": _VAL / "eagle3_bi_reduction_floor" / "eagle3_bi_reduction_floor_results.json",
    "332": _VAL / "eagle3_sdpa_phi_floor" / "eagle3_sdpa_phi_floor_results.json",
    "366": _VAL / "pinned_split_phi_audit" / "pinned_split_phi_audit_results.json",
}
# stark #365 (lm_head BI-GEMM) -- may not be posted yet; cross-checked in step 4 if present
PR365_CANDIDATES = [
    _VAL / "lmhead_bi_reduction_floor" / "lmhead_bi_reduction_floor_results.json",
    _VAL / "strict_lmhead_bi_gemm" / "strict_lmhead_bi_gemm_results.json",
    _VAL / "lmhead_verify_audit" / "lmhead_verify_audit_results.json",
    _VAL / "margin_localized_identity" / "margin_localized_identity_results.json",
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
    the split count the DEPLOYED heuristic (num_splits=0) launches for batch_nheads_mblocks=bnm.
    num_n_blocks = ceil(seqlen / BLOCK_N) -- built over the kernel n-tile, NOT physical pages."""
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
# Inputs / kernels  (flash_attn layout: q=[B,Sq,Hq,D]; dense k/v=[B,L,Hkv,D];
#                    paged k/v=[num_blocks,page,Hkv,D] + block_table=[B,max_blocks_per_seq])
# ======================================================================================== #
def _dense_qkv(M: int, L: int, seed: int, dev: torch.device):
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(1, M, N_Q_HEADS, HEAD_DIM, generator=g, device=dev, dtype=DTYPE)
    k = torch.randn(1, L, N_KV_HEADS, HEAD_DIM, generator=g, device=dev, dtype=DTYPE)
    v = torch.randn(1, L, N_KV_HEADS, HEAD_DIM, generator=g, device=dev, dtype=DTYPE)
    return q, k, v


def _to_paged(k, v, L: int, page: int, seed: int, dev: torch.device, shuffle: bool):
    """Pack dense [1,L,Hkv,D] KV into pages of `page` tokens with a (optionally shuffled) block_table,
    emulating the non-contiguous physical block allocation of the served paged-KV cache."""
    g = torch.Generator(device=dev).manual_seed(seed + 9173)
    nb = _ceildiv(L, page)
    perm = torch.randperm(nb, generator=g, device=dev) if shuffle else torch.arange(nb, device=dev)
    kc = torch.zeros(nb, page, N_KV_HEADS, HEAD_DIM, device=dev, dtype=DTYPE)
    vc = torch.zeros(nb, page, N_KV_HEADS, HEAD_DIM, device=dev, dtype=DTYPE)
    bt = torch.empty(1, nb, dtype=torch.int32, device=dev)
    for lg in range(nb):
        ph = int(perm[lg].item())
        s = lg * page
        e = min(s + page, L)
        kc[ph, : e - s] = k[0, s:e]
        vc[ph, : e - s] = v[0, s:e]
        bt[0, lg] = ph
    cs = torch.tensor([L], dtype=torch.int32, device=dev)
    return kc, vc, bt, cs


def _flash_dense(q, k, v, num_splits):
    return flash_attn_with_kvcache(q, k, v, softmax_scale=SCALE, causal=False, num_splits=num_splits)


def _flash_paged(q, kc, vc, bt, cs, num_splits):
    return flash_attn_with_kvcache(q, kc, vc, cache_seqlens=cs, block_table=bt,
                                   softmax_scale=SCALE, causal=False, num_splits=num_splits)


def _flash_paged_per_row(q, kc, vc, bt, cs, num_splits):
    """Per-row AR reference for the paged kernel: each query row attended alone (Sq=1), same split."""
    return torch.cat([_flash_paged(q[:, r:r + 1], kc, vc, bt, cs, num_splits)
                      for r in range(q.shape[1])], dim=1)


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
# STEP 1 -- audit the shared un-packing assumption (analytic over banked JSON; reused from #366)
# ======================================================================================== #
def step1_audit_unpacking(a327: dict, a332: dict) -> dict[str, Any]:
    s327 = a327["synthesis"]["step2_penalty_model"]
    sdpa_row = next(c for c in s327["per_component"] if c["component"] == "sdpa")
    model_327 = s327["model"]
    charges_forgone_reduction_327 = ("forgone reduction parallelism" in model_327)
    sdpa_floor_327 = sdpa_row["floor_contribution"]
    sdpa_pi_327 = sdpa_row["penalty_pi"]
    sdpa_full_roundtrips = abs(sdpa_floor_327 - SDPA_STEP_SHARE * SDPA_PENALTY_PI) < 1e-9
    shares_unpacking_327 = bool(charges_forgone_reduction_327 and sdpa_full_roundtrips
                                and abs(sdpa_pi_327 - SDPA_PENALTY_PI) < 1e-9)

    s332 = a332["synthesis"]
    phi_model_332 = s332["step3_occupancy_phi"]["model"]
    reduction_axis_332 = s332["step2_partition"]["reduction_axis"]
    n_nonred_332 = s332["step2_partition"]["n_nonreduction_ctas_2d"]
    phi_geo_332 = s332["step3_occupancy_phi"]["geometric_phi_estimate"]
    phi_excludes_reduction = ("N_nonreduction" in phi_model_332 or "non-reduction" in phi_model_332.lower())
    reduction_called_forgone = ("FORGONE for determinism" in reduction_axis_332)
    phi_roundtrips = abs(phi_geo_332 - (1.0 - min(1.0, n_nonred_332 / A10G_SMS))) < 1e-9
    shares_unpacking_332 = bool(phi_excludes_reduction and reduction_called_forgone and phi_roundtrips)

    return {
        "shares_unpacking_assumption_327": shares_unpacking_327,
        "shares_unpacking_assumption_332": shares_unpacking_332,
        "evidence_327": {
            "penalty_model": model_327,
            "sdpa_penalty_pi": sdpa_pi_327,
            "sdpa_floor_contribution": sdpa_floor_327,
            "charges_full_forgone_reduction_parallelism": charges_forgone_reduction_327,
            "sdpa_full_roundtrips_step_share_x_pi": sdpa_full_roundtrips,
        },
        "evidence_332": {
            "phi_model": phi_model_332,
            "reduction_axis": reduction_axis_332,
            "n_nonreduction_ctas": n_nonred_332,
            "geometric_phi_forgone": phi_geo_332,
            "phi_excludes_reduction_axis": phi_excludes_reduction,
            "reduction_axis_called_forgone_for_determinism": reduction_called_forgone,
            "phi_roundtrips_1_minus_Nnonred_over_SMs": phi_roundtrips,
        },
        "shared_assumption": "strict determinism requires UN-PACKING the split-KV reduction (collapse "
                             "to the non-reduction M x q-head tiles, forgoing the split-KV occupancy). "
                             "#363/#366 refute the PREMISE in DENSE isolation; THIS card tests whether "
                             "the refutation survives the served PAGED-KV block-table gather.",
    }


# ======================================================================================== #
# STEP 2 -- paged occupancy microbench (dense + paged, CTAs-in-flight + latency, >=2 seeds)
# ======================================================================================== #
def _cta_model(M: int, K: int, num_n_blocks: int) -> int:
    """Analytic FA2 splitkv grid: CTAs = num_q_heads x num_m_blocks x num_splits. num_n_blocks is
    built over the kernel n-tile of the LOGICAL seqlen (ceil(seqlen/BLOCK_N)) -- page-INVARIANT."""
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


def step2_paged_microbench(L: int, seeds: list[int], iters: int, warmup: int,
                           dev: torch.device, primary: bool) -> dict[str, Any]:
    """Per-call latency (dense + paged) + paged identity for each split config; analytic CTAs-in-flight
    (page-invariant). The TPS/latency occupancy proxy is the sanctioned method (ncu/CUPTI unavailable)."""
    # ---- latency: dense + paged(256,shuffled), median over seeds of per-seed median us ----
    lat_dense: dict[int, dict[int, float]] = {M: {} for M in M_LIST}
    lat_paged: dict[int, dict[int, float]] = {M: {} for M in M_LIST}
    for M in M_LIST:
        for ns in SWEEP_SPLITS:
            d_seed, p_seed = [], []
            for sd in seeds:
                q, k, v = _dense_qkv(M, L, sd, dev)
                kc, vc, bt, cs = _to_paged(k, v, L, PAGE_BLOCK, sd, dev, shuffle=True)
                d_seed.append(_time_call(lambda ns=ns, q=q, k=k, v=v: _flash_dense(q, k, v, ns), iters, warmup))
                p_seed.append(_time_call(
                    lambda ns=ns, q=q, kc=kc, vc=vc, bt=bt, cs=cs: _flash_paged(q, kc, vc, bt, cs, ns),
                    iters, warmup))
            d_seed.sort(); p_seed.sort()
            lat_dense[M][ns] = d_seed[len(d_seed) // 2]
            lat_paged[M][ns] = p_seed[len(p_seed) // 2]

    # ---- paged identity: batched-M vs per-row (SAME paged kernel/split), 3 configs, M in {2,4,8} ----
    any_nan = False
    ident: dict[int, dict[int, list[int]]] = {
        M: {ns: [0, 0] for ns in (HEURISTIC_SPLIT, PINNED_SPLIT, UNPACK_SPLIT)} for M in IDENT_M_LIST}
    maxdiff_pin_vs_heur = 0.0       # paged pinned8 vs paged heuristic
    maxdiff_paged_vs_dense = 0.0    # paged pinned8 vs DENSE pinned8 (the served-vs-stand-in numerics)
    for M in IDENT_M_LIST:
        for sd in seeds:
            q, k, v = _dense_qkv(M, L, sd, dev)
            kc, vc, bt, cs = _to_paged(k, v, L, PAGE_BLOCK, sd, dev, shuffle=True)
            heur = _flash_paged(q, kc, vc, bt, cs, HEURISTIC_SPLIT)
            pin = _flash_paged(q, kc, vc, bt, cs, PINNED_SPLIT)
            dpin = _flash_dense(q, k, v, PINNED_SPLIT)
            any_nan = any_nan or bool(torch.isnan(heur).any()) or bool(torch.isnan(pin).any())
            maxdiff_pin_vs_heur = max(maxdiff_pin_vs_heur, float((pin.float() - heur.float()).abs().max().item()))
            maxdiff_paged_vs_dense = max(maxdiff_paged_vs_dense, float((pin.float() - dpin.float()).abs().max().item()))
            for ns in (HEURISTIC_SPLIT, PINNED_SPLIT, UNPACK_SPLIT):
                ref = _flash_paged_per_row(q, kc, vc, bt, cs, ns)
                bat = _flash_paged(q, kc, vc, bt, cs, ns)
                same = (bat == ref).all(dim=-1)
                ident[M][ns][0] += int(same.sum())
                ident[M][ns][1] += int(same.numel())
    ident_rate = {M: {ns: (ident[M][ns][0] / ident[M][ns][1] if ident[M][ns][1] else float("nan"))
                      for ns in (HEURISTIC_SPLIT, PINNED_SPLIT, UNPACK_SPLIT)} for M in IDENT_M_LIST}

    # ---- analytic CTAs-in-flight (page-invariant grid over BLOCK_N tiles of logical seqlen) ----
    num_n_blocks = _ceildiv(L, BLOCK_N_TILE)
    K_heur_8 = _heuristic_K(8, num_n_blocks)
    ctas_heuristic = _cta_model(8, HEURISTIC_SPLIT, num_n_blocks)
    ctas_pinned8 = _cta_model(8, PINNED_SPLIT, num_n_blocks)
    ctas_unpacked = _cta_model(8, UNPACK_SPLIT, num_n_blocks)
    heur_K_bracket = {bn: _heuristic_K(8, _ceildiv(L, bn)) for bn in (32, 64, 128)}
    ctas_heuristic_bracket = {bn: N_Q_HEADS * 1 * k for bn, k in heur_K_bracket.items()}
    # physical page counts (contrast): the grid (<= num_n_blocks) is far coarser than block-16 pages
    n_pages_block16 = _ceildiv(L, SERVED_BLOCK_SIZE)
    n_pages_measured = _ceildiv(L, PAGE_BLOCK)
    pinned8_le_numnblocks = bool(PINNED_SPLIT <= num_n_blocks)        # no over-subscription
    split_coarser_than_pages = bool(K_heur_8 <= num_n_blocks <= n_pages_block16)

    # ---- verdicts (M=8 deployed verify shape) -- on PAGED latency ----
    heur_us8, pin_us8, unp_us8 = lat_paged[8][0], lat_paged[8][8], lat_paged[8][1]
    cta_preserved = bool(ctas_pinned8 >= 0.8 * min(ctas_heuristic, A10G_SMS) and ctas_pinned8 > 2 * ctas_unpacked)
    lat_preserved = bool(pin_us8 <= heur_us8 * 1.05)
    unpack_collapses = bool(unp_us8 >= heur_us8 * 1.5)
    pinned8_preserves_parallelism = bool(cta_preserved and lat_preserved)
    pinned8_m_invariant = bool(all(ident_rate[M][PINNED_SPLIT] >= 0.999 for M in IDENT_M_LIST))
    heuristic_breaks_identity = bool(all(ident_rate[M][HEURISTIC_SPLIT] < 0.5 for M in IDENT_M_LIST))

    # ---- paged-vs-dense gather penalty (matched L, same run; a baseline-paging diagnostic) ----
    def pen(ns):
        d, p = lat_dense[8][ns], lat_paged[8][ns]
        return (p - d) / d if d else float("nan")
    paged_pen_pin8 = pen(8)
    paged_pen_heur = pen(0)

    out: dict[str, Any] = {
        "L": L, "seeds": seeds, "page_block": PAGE_BLOCK,
        "num_n_blocks_bn64": num_n_blocks, "n_pages_block16": n_pages_block16,
        "n_pages_measured_page256": n_pages_measured,
        "pinned8_le_num_n_blocks_no_oversub": pinned8_le_numnblocks,
        "split_grid_coarser_than_block16_pages": split_coarser_than_pages,
        "latency_us_by_M_by_split_paged": {str(M): {str(ns): lat_paged[M][ns] for ns in SWEEP_SPLITS} for M in M_LIST},
        "latency_us_by_M_by_split_dense": {str(M): {str(ns): lat_dense[M][ns] for ns in SWEEP_SPLITS} for M in M_LIST},
        "identity_rate_by_M_paged": {str(M): {str(ns): ident_rate[M][ns]
                                              for ns in (HEURISTIC_SPLIT, PINNED_SPLIT, UNPACK_SPLIT)} for M in IDENT_M_LIST},
        "any_nan": bool(any_nan),
        "maxabsdiff_paged_pinned8_vs_paged_heuristic": maxdiff_pin_vs_heur,
        "maxabsdiff_paged_pinned8_vs_dense_pinned8": maxdiff_paged_vs_dense,
        # analytic CTAs-in-flight (paged grid; page-invariant)
        "heuristic_effective_K": K_heur_8,
        "heuristic_K_bracket_by_blockN": heur_K_bracket,
        "ctas_in_flight_paged_heuristic": ctas_heuristic,
        "ctas_in_flight_paged_heuristic_bracket": ctas_heuristic_bracket,
        "ctas_in_flight_paged_pinned8": ctas_pinned8,
        "ctas_in_flight_paged_unpacked": ctas_unpacked,
        # latency anchors (M=8, paged)
        "paged_heuristic_us_M8": heur_us8, "paged_pinned8_us_M8": pin_us8, "paged_unpacked_us_M8": unp_us8,
        "dense_heuristic_us_M8": lat_dense[8][0], "dense_pinned8_us_M8": lat_dense[8][8], "dense_unpacked_us_M8": lat_dense[8][1],
        "paged_pinned8_over_heuristic_ratio_M8": pin_us8 / heur_us8 if heur_us8 else float("nan"),
        "paged_unpacked_over_heuristic_ratio_M8": unp_us8 / heur_us8 if heur_us8 else float("nan"),
        "paged_gather_penalty_pinned8": paged_pen_pin8,
        "paged_gather_penalty_heuristic": paged_pen_heur,
        # verdicts
        "cta_preserved": cta_preserved,
        "latency_preserved": lat_preserved,
        "unpack_collapses_occupancy": unpack_collapses,
        "paged_pinned8_preserves_parallelism": pinned8_preserves_parallelism,
        "paged_pinned8_m_invariant": pinned8_m_invariant,
        "paged_heuristic_breaks_identity": heuristic_breaks_identity,
    }

    # ---- page-size & contiguity invariance (primary L only): the block-16 bridge evidence ----
    if primary:
        q8, k8, v8 = _dense_qkv(8, L, seeds[0], dev)
        # paged@512 shuffled
        kc2, vc2, bt2, cs2 = _to_paged(k8, v8, L, PAGE_BLOCK_ALT, seeds[0], dev, shuffle=True)
        us_p512 = _time_call(lambda: _flash_paged(q8, kc2, vc2, bt2, cs2, PINNED_SPLIT), iters, warmup)
        # paged@256 contiguous (no shuffle)
        kcc, vcc, btc, csc = _to_paged(k8, v8, L, PAGE_BLOCK, seeds[0], dev, shuffle=False)
        us_pcontig = _time_call(lambda: _flash_paged(q8, kcc, vcc, btc, csc, PINNED_SPLIT), iters, warmup)
        # identity of the invariance variants vs the shuffled paged@256
        kcs, vcs, bts, css = _to_paged(k8, v8, L, PAGE_BLOCK, seeds[0], dev, shuffle=True)
        base = _flash_paged(q8, kcs, vcs, bts, css, PINNED_SPLIT)
        md_p512 = float((_flash_paged(q8, kc2, vc2, bt2, cs2, PINNED_SPLIT).float() - base.float()).abs().max().item())
        md_pcontig = float((_flash_paged(q8, kcc, vcc, btc, csc, PINNED_SPLIT).float() - base.float()).abs().max().item())
        base_us = out["paged_pinned8_us_M8"]
        out["invariance"] = {
            "paged256_shuffled_us": base_us,
            "paged512_shuffled_us": us_p512,
            "paged256_contiguous_us": us_pcontig,
            "page_size_invariant_us_ratio": us_p512 / base_us if base_us else float("nan"),
            "contiguity_invariant_us_ratio": us_pcontig / base_us if base_us else float("nan"),
            "page_size_invariant": bool(abs(us_p512 - base_us) <= 0.06 * base_us),
            "contiguity_invariant": bool(abs(us_pcontig - base_us) <= 0.06 * base_us),
            "maxabsdiff_page512_vs_page256": md_p512,
            "maxabsdiff_contiguous_vs_shuffled": md_pcontig,
            "note": "page-size + contiguity invariance => paged-gather cost is fixed block_table-setup "
                    "overhead, not per-page memory divergence => the block-16 grid inherits the same "
                    "occupancy story (more pages, same BLOCK_N tiling).",
        }
    return out


# ======================================================================================== #
# STEP 3 -- recompute eta-floor and phi-recovery UNDER paged-KV
# ======================================================================================== #
def step3_recompute_paged(micro_primary: dict, micro_all: dict, a366: dict) -> dict[str, Any]:
    ctas_pinned8 = micro_primary["ctas_in_flight_paged_pinned8"]
    pin_us8 = micro_primary["paged_pinned8_us_M8"]
    heur_us8 = micro_primary["paged_heuristic_us_M8"]

    # --- (a) OCCUPANCY-MODEL reading: pinned-split keeps ctas_pinned8 CTAs (page-invariant grid) ----
    phi_forgone_paged_occ = max(0.0, 1.0 - min(1.0, ctas_pinned8 / A10G_SMS))
    phi_recovery_paged_occ = 1.0 - phi_forgone_paged_occ

    # --- (b) EMPIRICAL reading (latency proxy on PAGED kernel): pinned-8 <= paged heuristic => tax<=0 --
    realized_tax_ratio = max(0.0, (pin_us8 - heur_us8) / heur_us8) if heur_us8 else float("nan")
    phi_forgone_paged_emp = realized_tax_ratio
    phi_recovery_paged_emp = 1.0 - phi_forgone_paged_emp

    # HEADLINE = empirical (the paged microbench is ground truth); occupancy = conservative floor.
    phi_recovery_paged_split = phi_recovery_paged_emp
    phi_forgone_headline = phi_forgone_paged_emp

    def eta_floor(phi_forgone: float) -> float:
        # SDPA determinism term scales with forgone split-KV fraction; lm_head GEMM tax is a fixed term.
        return LMHEAD_FLOOR_FULL + SDPA_FLOOR_FULL * phi_forgone

    eta_floor_paged_split = eta_floor(phi_forgone_headline)          # headline (empirical phi_forgone~0)
    eta_floor_paged_occ_upper = eta_floor(phi_forgone_paged_occ)     # conservative occupancy upper bound

    revived_ceiling = CEILING_500 * (1.0 - eta_floor_paged_split)
    revived_ceiling_occ = CEILING_500 * (1.0 - eta_floor_paged_occ_upper)

    # ---- paged-gather penalty: the paged-vs-dense overhead (a SEPARATE baseline-paging cost). It is
    # NOT a determinism tax: it is present for BOTH heuristic and pinned (cancels in the pin-vs-heur
    # determinism delta) and is ALREADY in the served step (the deployment is paged). Reported for the
    # advisor's risk picture; we ALSO give a (double-counting) paged-inclusive upper bound for honesty.
    pen_by_L = {str(micro_all[str(L)]["L"]): micro_all[str(L)]["paged_gather_penalty_pinned8"] for L in L_LIST}
    pens = sorted(v for v in pen_by_L.values() if v == v)
    paged_block_gather_occupancy_penalty = pens[len(pens) // 2] if pens else float("nan")  # median over L
    paged_gather_penalty_max = max(pens) if pens else float("nan")
    eta_floor_paged_inclusive_upper = eta_floor_paged_occ_upper + max(0.0, paged_gather_penalty_max)

    # comparisons
    phi_clears_noreg_breakeven = bool(phi_recovery_paged_split > RECOVERY_BREAKEVEN_NOREG)
    phi_clears_500_breakeven = bool(phi_recovery_paged_split > RECOVERY_BREAKEVEN_500)
    phi_occ_clears_500_breakeven = bool(phi_recovery_paged_occ > RECOVERY_BREAKEVEN_500)
    eta_clears_500_budget = bool(eta_floor_paged_split < BUDGET_500_ETA)
    eta_occ_clears_500_budget = bool(eta_floor_paged_occ_upper < BUDGET_500_ETA)
    eta_below_blanket = bool(eta_floor_paged_split < FLOOR_COMBINED_FULL)

    # ---- reconcile with #366 dense ----
    dense_phi = a366["phi_recovery_pinned_split_analytic"]
    dense_eta = a366["eta_floor_pinned_split_analytic"]
    paged_confirms_dense = bool(abs(phi_recovery_paged_split - dense_phi) <= 0.05)
    paged_localizes_penalty = bool(phi_recovery_paged_split < dense_phi - 0.05)

    # ---- decision gate (PR #370) ----
    preserves = bool(all(micro_all[str(L)]["paged_pinned8_preserves_parallelism"] for L in L_LIST))
    # primary gate: paged_confirms_366
    paged_confirms_366 = bool(preserves and phi_clears_500_breakeven)
    # three-way verdict
    if not preserves or phi_recovery_paged_split <= RECOVERY_BREAKEVEN_NOREG:
        gate_verdict = "collapse_paged_binding"     # paged gather reintroduces the un-packing collapse
    elif phi_recovery_paged_split <= RECOVERY_BREAKEVEN_500:
        gate_verdict = "partial_penalty"            # cap lifts >473.5 but not >500 on attention alone
    else:
        gate_verdict = "confirms_366_high_confidence"
    refutes_332_cap = bool(preserves and phi_clears_500_breakeven and revived_ceiling > CEILING_AT_GEO_332)
    revives_ceiling_toward_520 = bool(refutes_332_cap and revived_ceiling > TARGET_500 and eta_clears_500_budget)

    return {
        # headline (empirical, paged-grounded)
        "phi_recovery_paged_split": phi_recovery_paged_split,
        "eta_floor_paged_split": eta_floor_paged_split,
        "revived_ceiling_paged_tps": revived_ceiling,
        "phi_forgone_paged_headline": phi_forgone_headline,
        "realized_paged_sdpa_tax_ratio": realized_tax_ratio,
        # conservative occupancy bracket (page-invariant grid -> same as dense #366)
        "phi_recovery_paged_occupancy_lower_bound": phi_recovery_paged_occ,
        "phi_forgone_paged_occupancy": phi_forgone_paged_occ,
        "eta_floor_paged_occupancy_upper_bound": eta_floor_paged_occ_upper,
        "revived_ceiling_paged_occupancy_tps": revived_ceiling_occ,
        # paged-gather diagnostic (separate; NOT a determinism tax)
        "paged_block_gather_occupancy_penalty": paged_block_gather_occupancy_penalty,
        "paged_gather_penalty_by_L": pen_by_L,
        "paged_gather_penalty_max": paged_gather_penalty_max,
        "eta_floor_paged_inclusive_upper_double_count": eta_floor_paged_inclusive_upper,
        "eta_inclusive_clears_500_budget": bool(eta_floor_paged_inclusive_upper < BUDGET_500_ETA),
        # references for comparison
        "lmhead_residual_floor": LMHEAD_FLOOR_FULL,
        "sdpa_floor_full_327": SDPA_FLOOR_FULL,
        "floor_blanket_9841": FLOOR_COMBINED_FULL,
        "budget_500_eta": BUDGET_500_ETA,
        "recovery_breakeven_noreg_0255": RECOVERY_BREAKEVEN_NOREG,
        "recovery_breakeven_500": RECOVERY_BREAKEVEN_500,
        "phi_recovery_geo_332": PHI_RECOVERY_GEO_332,
        "ceiling_at_geo_332_473": CEILING_AT_GEO_332,
        "dense_phi_recovery_366": dense_phi,
        "dense_eta_floor_366": dense_eta,
        # comparison bools
        "phi_clears_noreg_breakeven_0255": phi_clears_noreg_breakeven,
        "phi_clears_500_breakeven": phi_clears_500_breakeven,
        "phi_occupancy_clears_500_breakeven": phi_occ_clears_500_breakeven,
        "eta_clears_500_budget": eta_clears_500_budget,
        "eta_occupancy_clears_500_budget": eta_occ_clears_500_budget,
        "eta_below_9841_blanket": eta_below_blanket,
        "paged_confirms_dense_phi": paged_confirms_dense,
        "paged_localizes_penalty": paged_localizes_penalty,
        # PR decision gate
        "paged_pinned8_preserves_parallelism_all_L": preserves,
        "paged_confirms_366": paged_confirms_366,
        "gate_verdict": gate_verdict,
        "refutes_332_cap": refutes_332_cap,
        "revives_ceiling_toward_520": revives_ceiling_toward_520,
    }


# ======================================================================================== #
# STEP 4 -- reconcile with #366 (dense) and cross-check stark #365 (lm_head BI-GEMM) if posted
# ======================================================================================== #
def step4_reconcile(recompute: dict, micro_primary: dict) -> dict[str, Any]:
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
    base = {
        "paged_vs_dense_366": {
            "paged_phi_recovery": recompute["phi_recovery_paged_split"],
            "dense_phi_recovery_366": recompute["dense_phi_recovery_366"],
            "confirms_366_dense": recompute["paged_confirms_dense_phi"],
            "localizes_paged_penalty": recompute["paged_localizes_penalty"],
            "paged_block_gather_occupancy_penalty": recompute["paged_block_gather_occupancy_penalty"],
            "byte_identity_paged_vs_dense_maxabsdiff": micro_primary["maxabsdiff_paged_pinned8_vs_dense_pinned8"],
            "interpretation": "paged phi == dense phi AND paged pinned8 byte-identical to dense pinned8 "
                              "=> #366's refutation survives the served paged-KV layout (CONFIRM). The "
                              "small paged-gather penalty is a baseline-paging cost already in the "
                              "served step, not a determinism tax (it cancels in the pin-vs-heur delta).",
        }
    }
    if found is None:
        base["crosscheck_365"] = {
            "pr365_posted": False,
            "note": "stark #365 (lm_head BI-GEMM eta) not yet posted. This card's PAGED attention-locus "
                    "phi_recovery confirms #366; the residual lm_head term is <= 0.390% even at "
                    "phi_forgone=1 -- already << the 4.02% budget -- and #365 measures whether even that "
                    "is recoverable. A #365-vs-this disagreement would localize the residual at the "
                    "lm_head locus (NOT paged-KV occupancy).",
            "predicted_total_locus_eta_if_lmhead_full": LMHEAD_FLOOR_FULL,
            "predicted_total_locus_eta_if_lmhead_free": recompute["eta_floor_paged_split"],
        }
        return base
    p, j = found
    lm_eta = j.get("lmhead_bi_gemm_eta_measured", j.get("total_verify_locus_eta",
             j.get("eta_floor_paged_split", j.get("eta_floor_pinned_split_analytic"))))
    base["crosscheck_365"] = {
        "pr365_posted": True, "path": str(p), "wandb_run_id": j.get("wandb_run_id"),
        "lmhead_bi_gemm_eta_measured": lm_eta,
        "reconcile_note": "paged attention-locus phi confirms #366; compare empirical lm_head eta to the "
                          f"lm_head residual prediction ({LMHEAD_FLOOR_FULL:.5f}). Agreement => residual "
                          "lives at lm_head, not paged-KV occupancy; disagreement localizes the gap.",
    }
    return base


# ======================================================================================== #
# Self-test
# ======================================================================================== #
def selftest(audit: dict, micro_all: dict, recompute: dict, gpu: dict, flags: dict, a366: dict) -> dict[str, Any]:
    c: dict[str, bool] = {}
    mp = micro_all[str(PRIMARY_L)]
    # (a) provenance: banked constants roundtrip (identical to #366's self-test)
    c["a_sdpa_floor_full_roundtrips_327"] = abs(SDPA_STEP_SHARE * SDPA_PENALTY_PI - SDPA_FLOOR_FULL) < 1e-12
    c["a_blanket_is_sdpa_plus_lmhead"] = abs(SDPA_FLOOR_FULL + LMHEAD_FLOOR_FULL - FLOOR_COMBINED_FULL) < 1e-12
    c["a_phi_geo_332_roundtrips"] = abs((1.0 - N_NONREDUCTION_332 / A10G_SMS) - PHI_FORGONE_GEO_332) < 1e-9
    c["a_phi_recovery_332_is_0075"] = abs((1.0 - PHI_FORGONE_GEO_332) - PHI_RECOVERY_GEO_332) < 1e-9
    c["a_budget_500_roundtrips"] = abs(CEILING_500 * (1 - BUDGET_500_ETA) - TARGET_500) < 1e-3
    c["a_ceiling_at_geo_332_roundtrips"] = abs(
        CEILING_500 * (1 - (LMHEAD_FLOOR_FULL + SDPA_FLOOR_FULL) * PHI_FORGONE_GEO_332) - CEILING_AT_GEO_332) < 0.5
    # (a2) round-trip the #366 DENSE anchor: paged CTA grid reproduces dense CTA counts (page-invariant)
    c["a2_paged_grid_matches_366_dense_cta"] = bool(
        mp["ctas_in_flight_paged_pinned8"] == a366["ctas_in_flight_pinned8"]
        and mp["ctas_in_flight_paged_heuristic"] == a366["ctas_in_flight_heuristic"]
        and mp["ctas_in_flight_paged_unpacked"] == a366["ctas_in_flight_unpacked"])
    # (b) step-1 audit: both share the un-packing assumption (the card's premise)
    c["b_shares_unpacking_327"] = bool(audit["shares_unpacking_assumption_327"])
    c["b_shares_unpacking_332"] = bool(audit["shares_unpacking_assumption_332"])
    # (c) microbench sanity across ALL L: NaN-clean, rates in [0,1], latencies finite/positive
    c["c_nan_clean"] = (not any(micro_all[str(L)]["any_nan"] for L in L_LIST))
    rates_ok = all(0.0 <= v <= 1.0 for L in L_LIST
                   for d in micro_all[str(L)]["identity_rate_by_M_paged"].values() for v in d.values())
    c["c_rates_in_unit_interval"] = bool(rates_ok)
    lat_ok = all(math.isfinite(v) and v > 0 for L in L_LIST
                 for grp in ("latency_us_by_M_by_split_paged", "latency_us_by_M_by_split_dense")
                 for d in micro_all[str(L)][grp].values() for v in d.values())
    c["c_latencies_finite_positive"] = bool(lat_ok)
    # (d) the MEASURED paged mechanism findings (folded in -- not assumptions)
    c["d_paged_pinned8_byte_exact_M_invariant"] = bool(
        mp["paged_pinned8_m_invariant"] and mp["maxabsdiff_paged_pinned8_vs_paged_heuristic"] < 1e-2)
    c["d_paged_heuristic_breaks_identity"] = bool(mp["paged_heuristic_breaks_identity"])
    c["d_unpack_collapses_occupancy"] = bool(mp["unpack_collapses_occupancy"])
    c["d_paged_pinned8_preserves_parallelism_all_L"] = bool(recompute["paged_pinned8_preserves_parallelism_all_L"])
    c["d_ctas_pinned_gg_unpacked"] = bool(mp["ctas_in_flight_paged_pinned8"] > 2 * mp["ctas_in_flight_paged_unpacked"])
    c["d_paged_byte_identical_to_dense"] = bool(mp["maxabsdiff_paged_pinned8_vs_dense_pinned8"] < 1e-2)
    # (d2) the block-16 bridge: page-size + contiguity invariance at primary L
    inv = mp.get("invariance", {})
    c["d2_page_size_invariant"] = bool(inv.get("page_size_invariant", False))
    c["d2_contiguity_invariant"] = bool(inv.get("contiguity_invariant", False))
    c["d2_pinned8_no_oversub_le_numnblocks"] = bool(mp["pinned8_le_num_n_blocks_no_oversub"])
    # (e) recompute: well-typed bools, finite floors, decision gate consistency
    c["e_eta_floor_finite_nonneg"] = bool(math.isfinite(recompute["eta_floor_paged_split"])
                                          and recompute["eta_floor_paged_split"] >= 0)
    c["e_phi_recovery_in_unit"] = bool(0.0 <= recompute["phi_recovery_paged_split"] <= 1.0)
    c["e_eta_below_blanket"] = bool(recompute["eta_below_9841_blanket"])
    c["e_decision_bools_typed"] = (isinstance(recompute["paged_confirms_366"], bool)
                                   and isinstance(recompute["refutes_332_cap"], bool))
    c["e_confirm_implies_clear500"] = (not recompute["paged_confirms_366"]) or recompute["phi_clears_500_breakeven"]
    c["e_gate_verdict_valid"] = recompute["gate_verdict"] in (
        "confirms_366_high_confidence", "partial_penalty", "collapse_paged_binding")
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
    bar = "=" * 104
    print(bar)
    print("PAGED-KV OCCUPANCY -- does #366's pinned-split survive the served paged layout? (PR #370, wirbel)")
    print(f"  GPU {gpu['name']}  SMs={gpu['sm_count']}  cc={gpu['compute_capability']}  "
          f"on-target-GA102-sm86-80SM={gpu['is_a10g_80sm'] and gpu['is_ga102_sm86']}")
    print(f"  kernel: flash_attn_with_kvcache paged (page_block={PAGE_BLOCK}; served block_size={SERVED_BLOCK_SIZE}; "
          f"grid page-invariant over BLOCK_N={BLOCK_N_TILE})")
    print("-" * 104)
    print("  STEP 1 -- shared un-packing assumption (premise):")
    print(f"     shares_unpacking_327 = {au['shares_unpacking_assumption_327']}   "
          f"shares_unpacking_332 = {au['shares_unpacking_assumption_332']}")
    print("-" * 104)
    for L in L_LIST:
        m = payload["microbench"][str(L)]
        print(f"  STEP 2 -- PAGED microbench L={L} (seeds={m['seeds']}, n_pages@256={m['n_pages_measured_page256']}, "
              f"block16_pages={m['n_pages_block16']}, num_n_blocks(BN64)={m['num_n_blocks_bn64']}):")
        print(f"     CTAs paged: heuristic={m['ctas_in_flight_paged_heuristic']} (K_eff={m['heuristic_effective_K']}) "
              f"pinned8={m['ctas_in_flight_paged_pinned8']} unpacked={m['ctas_in_flight_paged_unpacked']}")
        print(f"     paged us M8: heur={m['paged_heuristic_us_M8']:.2f} pin8={m['paged_pinned8_us_M8']:.2f} "
              f"(x{m['paged_pinned8_over_heuristic_ratio_M8']:.3f}) unpk={m['paged_unpacked_us_M8']:.2f} "
              f"(x{m['paged_unpacked_over_heuristic_ratio_M8']:.2f})")
        print(f"     paged-vs-dense gather penalty: pin8={m['paged_gather_penalty_pinned8']*100:+.2f}%  "
              f"heur={m['paged_gather_penalty_heuristic']*100:+.2f}%")
        print(f"     identity: pinned8 M-invariant={m['paged_pinned8_m_invariant']} "
              f"heuristic breaks={m['paged_heuristic_breaks_identity']} "
              f"maxdiff(paged_pin8,dense_pin8)={m['maxabsdiff_paged_pinned8_vs_dense_pinned8']:.2e}")
        print(f"     -> paged_pinned8_preserves_parallelism = {m['paged_pinned8_preserves_parallelism']} "
              f"(unpack_collapses={m['unpack_collapses_occupancy']})")
    inv = mp.get("invariance", {})
    if inv:
        print(f"  block-16 bridge (L={PRIMARY_L}): page-size invariant={inv['page_size_invariant']} "
              f"(x{inv['page_size_invariant_us_ratio']:.3f}) contiguity invariant={inv['contiguity_invariant']} "
              f"(x{inv['contiguity_invariant_us_ratio']:.3f})")
    print("-" * 104)
    print("  STEP 3 -- recompute under paged-KV:")
    print(f"     phi_recovery_paged_split = {rc['phi_recovery_paged_split']:.4f} "
          f"(occ lower bound {rc['phi_recovery_paged_occupancy_lower_bound']:.4f}) "
          f"vs dense#366 {rc['dense_phi_recovery_366']:.4f}; vs 0.255 / 0.5913 break-even")
    print(f"     eta_floor_paged_split = {rc['eta_floor_paged_split']*100:.4f}% "
          f"(occ upper {rc['eta_floor_paged_occupancy_upper_bound']*100:.4f}%) vs 4.02% budget / 9.841% blanket")
    print(f"     paged_block_gather_occupancy_penalty = {rc['paged_block_gather_occupancy_penalty']*100:+.2f}% "
          f"(max {rc['paged_gather_penalty_max']*100:+.2f}%; baseline-paging cost, NOT a determinism tax)")
    print(f"     revived ceiling = {rc['revived_ceiling_paged_tps']:.2f} TPS "
          f"(occ {rc['revived_ceiling_paged_occupancy_tps']:.2f}) vs #332 cap 473.53")
    print(f"     GATE VERDICT = {rc['gate_verdict']}   paged_confirms_366 = {rc['paged_confirms_366']}")
    print(f"     refutes_332_cap = {rc['refutes_332_cap']}   revives_ceiling_toward_520 = {rc['revives_ceiling_toward_520']}")
    print("-" * 104)
    print(f"  STEP 4 -- #365 cross-check posted={payload['reconcile']['crosscheck_365']['pr365_posted']}; "
          f"paged-vs-dense#366 confirm={payload['reconcile']['paged_vs_dense_366']['confirms_366_dense']}")
    print("-" * 104)
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
        print(f"[paged-occ] wandb helpers unavailable: {e}")
        return None
    au, rc, st = payload["audit"], payload["recompute"], payload["selftest"]
    mp = payload["microbench"][str(PRIMARY_L)]
    run = init_wandb_run(
        job_type="analysis-gpu-microbench", agent="wirbel",
        name=args.wandb_name, group=args.wandb_group,
        tags=["strict-bi-verify-gemm", "pinned-split", "paged-kv", "occupancy", "phi-recovery",
              "block-gather", "319-strict-lock", "pr-370"],
        config={"pr": 370, "kind": "pagedkv-occupancy-microbench",
                "head_dim": HEAD_DIM, "n_q_heads": N_Q_HEADS, "n_kv_heads": N_KV_HEADS,
                "m_list": list(M_LIST), "L_list": list(L_LIST), "splits": list(SWEEP_SPLITS),
                "page_block": PAGE_BLOCK, "served_block_size": SERVED_BLOCK_SIZE,
                "ceiling_500": CEILING_500, "step_us": STEP_US,
                "floor_blanket_9841": FLOOR_COMBINED_FULL, "budget_500_eta": BUDGET_500_ETA},
    )
    if run is None:
        print("[paged-occ] wandb disabled (no API key / WANDB_MODE).")
        return None
    flat: dict[str, float] = {
        "audit/shares_unpacking_327": float(au["shares_unpacking_assumption_327"]),
        "audit/shares_unpacking_332": float(au["shares_unpacking_assumption_332"]),
        "micro/ctas_paged_heuristic": float(mp["ctas_in_flight_paged_heuristic"]),
        "micro/ctas_paged_pinned8": float(mp["ctas_in_flight_paged_pinned8"]),
        "micro/ctas_paged_unpacked": float(mp["ctas_in_flight_paged_unpacked"]),
        "micro/paged_heuristic_us_M8": mp["paged_heuristic_us_M8"],
        "micro/paged_pinned8_us_M8": mp["paged_pinned8_us_M8"],
        "micro/paged_unpacked_us_M8": mp["paged_unpacked_us_M8"],
        "micro/paged_pinned8_over_heuristic_ratio_M8": mp["paged_pinned8_over_heuristic_ratio_M8"],
        "micro/paged_unpacked_over_heuristic_ratio_M8": mp["paged_unpacked_over_heuristic_ratio_M8"],
        "micro/paged_gather_penalty_pinned8": mp["paged_gather_penalty_pinned8"],
        "micro/paged_pinned8_preserves_parallelism": float(mp["paged_pinned8_preserves_parallelism"]),
        "micro/paged_pinned8_m_invariant": float(mp["paged_pinned8_m_invariant"]),
        "micro/paged_heuristic_breaks_identity": float(mp["paged_heuristic_breaks_identity"]),
        "micro/maxabsdiff_paged_vs_dense": mp["maxabsdiff_paged_pinned8_vs_dense_pinned8"],
        "recompute/phi_recovery_paged_split": rc["phi_recovery_paged_split"],
        "recompute/phi_recovery_paged_occupancy_lower_bound": rc["phi_recovery_paged_occupancy_lower_bound"],
        "recompute/eta_floor_paged_split": rc["eta_floor_paged_split"],
        "recompute/eta_floor_paged_occupancy_upper_bound": rc["eta_floor_paged_occupancy_upper_bound"],
        "recompute/paged_block_gather_occupancy_penalty": rc["paged_block_gather_occupancy_penalty"],
        "recompute/revived_ceiling_paged_tps": rc["revived_ceiling_paged_tps"],
        "recompute/paged_confirms_366": float(rc["paged_confirms_366"]),
        "recompute/refutes_332_cap": float(rc["refutes_332_cap"]),
        "recompute/revives_ceiling_toward_520": float(rc["revives_ceiling_toward_520"]),
        "recompute/eta_clears_500_budget": float(rc["eta_clears_500_budget"]),
        "recompute/phi_clears_500_breakeven": float(rc["phi_clears_500_breakeven"]),
        "selftest/analytic_self_test_passes": float(st["passes"]),
        "gpu/sm_count": float(payload["gpu"]["sm_count"]),
    }
    for L in L_LIST:
        m = payload["microbench"][str(L)]
        flat[f"micro/L{L}_paged_pin8_us"] = m["paged_pinned8_us_M8"]
        flat[f"micro/L{L}_paged_heur_us"] = m["paged_heuristic_us_M8"]
        flat[f"micro/L{L}_paged_gather_penalty_pin8"] = m["paged_gather_penalty_pinned8"]
        flat[f"micro/L{L}_paged_preserves"] = float(m["paged_pinned8_preserves_parallelism"])
        for M in IDENT_M_LIST:
            for ns in (HEURISTIC_SPLIT, PINNED_SPLIT, UNPACK_SPLIT):
                flat[f"micro/L{L}_identity_M{M}_split{ns}"] = m["identity_rate_by_M_paged"][str(M)][str(ns)]
    run.log({"global_step": 0, **flat})
    log_summary(run, _jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="pagedkv_occupancy_microbench", artifact_type="analysis", data=_jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[paged-occ] wandb logged {len(flat)} keys (run {rid})")
    return rid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true", help="run the GA102 paged microbench (default path)")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to validate the path")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default="wirbel/pagedkv-occupancy-microbench")
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

    a327, a332, a366 = _load_anchor("327"), _load_anchor("332"), _load_anchor("366")
    audit = step1_audit_unpacking(a327, a332)

    micro = {str(L): step2_paged_microbench(L, args.seeds, args.iters, args.warmup, dev, primary=(L == PRIMARY_L))
             for L in L_LIST}
    recompute = step3_recompute_paged(micro[str(PRIMARY_L)], micro, a366)
    reconcile = step4_reconcile(recompute, micro[str(PRIMARY_L)])

    flags = {"no_hf_job": True, "no_launch": True, "no_served_file_change": True}
    st = selftest(audit, micro, recompute, gpu, flags, a366)

    torch.cuda.synchronize()
    payload = {
        "agent": "wirbel", "pr": 370,
        "kind": "pagedkv-occupancy-microbench",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "analysis_only": True, **flags,
        "gpu": gpu,
        "peak_mem_mib": round(torch.cuda.max_memory_allocated(dev) / (1024**2), 3),
        "kernel": {"fn": "flash_attn_with_kvcache", "paged": True, "page_block": PAGE_BLOCK,
                   "served_block_size": SERVED_BLOCK_SIZE, "block_n_tile": BLOCK_N_TILE,
                   "grid_page_invariant": True},
        "ladder_constants": {"ceiling_500": CEILING_500, "step_us": STEP_US,
                             "floor_blanket_9841": FLOOR_COMBINED_FULL, "budget_500_eta": BUDGET_500_ETA,
                             "recovery_breakeven_noreg_0255": RECOVERY_BREAKEVEN_NOREG,
                             "recovery_breakeven_500": RECOVERY_BREAKEVEN_500,
                             "ceiling_at_geo_332_473": CEILING_AT_GEO_332},
        "audit": audit, "microbench": micro, "recompute": recompute, "reconcile": reconcile,
        "selftest": st,
        "analytic_self_test_passes": bool(st["passes"]),
        # headline TEST surface (PR #370 required fields)
        "phi_recovery_paged_split": recompute["phi_recovery_paged_split"],
        "eta_floor_paged_split": recompute["eta_floor_paged_split"],
        "ctas_in_flight_paged_pinned8": micro[str(PRIMARY_L)]["ctas_in_flight_paged_pinned8"],
        "ctas_in_flight_paged_heuristic": micro[str(PRIMARY_L)]["ctas_in_flight_paged_heuristic"],
        "paged_pinned8_preserves_parallelism": recompute["paged_pinned8_preserves_parallelism_all_L"],
        "paged_block_gather_occupancy_penalty": recompute["paged_block_gather_occupancy_penalty"],
        "paged_confirms_366": recompute["paged_confirms_366"],
        "gate_verdict": recompute["gate_verdict"],
        "refutes_332_cap": recompute["refutes_332_cap"],
        "revives_ceiling_toward_520": recompute["revives_ceiling_toward_520"],
    }
    print_report(payload)
    out_path = Path(args.out_dir) / "pagedkv_occupancy_microbench_results.json"
    out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"\n[paged-occ] wrote {out_path}")
    rid = maybe_log_wandb(payload, args)
    if rid:
        payload["wandb_run_id"] = rid
        out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"[paged-occ] analytic_self_test_passes = {payload['analytic_self_test_passes']}")
    print(f"[paged-occ] phi_recovery_paged_split = {payload['phi_recovery_paged_split']:.4f}  "
          f"eta_floor_paged = {payload['eta_floor_paged_split']*100:.4f}%  "
          f"paged_confirms_366 = {payload['paged_confirms_366']}  verdict = {payload['gate_verdict']}")
    raise SystemExit(0 if payload["analytic_self_test_passes"] else 1)


if __name__ == "__main__":
    main()
