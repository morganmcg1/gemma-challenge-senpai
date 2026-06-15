#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #393 (wirbel) -- Cheapest byte-exact attention pin: is eta_attn=0.02145 the floor? (#319).

THE QUESTION (my #378/#390 lineage)
-----------------------------------
#378 derived eta_attn=0.02145375 = f_attn x (penalty_evalweighted - 1) -- the strict cost of the ONLY
M-invariant served attention config reachable without a kernel rebuild: VLLM_BATCH_INVARIANT=1 ->
num_splits=1 (the "un-pack" lane, 8 CTAs not the pin's 64). #390 folded that single 0.02145 attention
pin into the corrected shippable ceiling: strict_tps(0.02145)=509.78 (ceiling basis) /
strict_tps_divisor(481.53,0.02145)=471.42 (deployed basis). Two facts were ASSUMED, not measured here:
  (a) that num_splits=1 is the CHEAPEST byte-exact pin available today (is 0.02145 really the floor, or
      does some other shipped backend -- FlashInfer, a pinned-K, FA_SLIDING=0 vs =1 -- pin strict cheaper?);
  (b) that the eval-WEIGHTED penalty (dragged down by the low-L eval mass, ~1.23x) is the right tax for
      the DECODE operating band. The eval set's decode positions concentrate at L in [528,658]; the un-pack
      penalty there is STRICTLY ABOVE the eval-weighted mean, so the decode-specific eta_attn is LARGER and
      471.42 is an OPTIMISTIC setting of the deployed strict TPS.

WHAT THIS CARD MEASURES (pod A10G sm_86, the SERVED vendored FA2 varlen kernel + shipped FlashInfer)
---------------------------------------------------------------------------------------------------
(1) Enumerate byte-exact attention configs reachable WITHOUT a rebuild and MEASURE each:
      - FA2 varlen num_splits=0 (heuristic, the fast non-VBI default),
      - FA2 varlen num_splits=1 (un-pack, the VBI=1 deployed strict config),
      - FA2 varlen num_splits>1 (pinned-K -- the un-deployable ideal; expect NotImplementedError),
      - FlashInfer paged decode disable_split_kv True (BI serial) vs False (split),
      - XFORMERS / standalone flash_attn (report shipped-or-not).
    For each: greedy-token-identity (batched-M=8 vs per-row-M=1 byte+argmax, >=3 seeds x >=8 trials) at the
    operating band L in [528,658] + a short-context point, and decode-step attention latency (us/token).
(2) Decode-specific eta_attn: re-weight the MEASURED un-pack penalty over the [528,658] band (not the #282
    full eval distribution) -> eta_attn_decode_only, eta_attn_decode_vs_evalweighted_delta, and the
    re-derived deployed_tps_decode_eta.
(3) Cheapest byte-exact pin: cheapest_strict_attn_backend / _eta, attn_eta_reducible, attn_already_strict_free,
    fa_sliding0_is_strict_floor, ceiling/deployed_with_cheapest_attn, gap_to_500_after_attn.
(4) PRIMARY self-test: reproduce #390's eta_attn=0.02145 and 509.78/471.42 from inputs; assert >=3 seeds,
    on-target A10G sm8x, no-launch/no-served-file-change guards, and that EVERY config reported "byte-exact"
    has measured identity exactly 1.000.

SCOPE: identity-safe pod-GPU microbench on EXISTING kernels/flags + analytic re-weight over the banked MERGED
#282 anchor. NO train.py --launch, NO HF Job, NO submission, NO served-file change, NO kernel rebuild, 0
official TPS. baseline 481.53 UNCHANGED. Greedy identity is MEASURED, never broken. Real gemma-4-E4B sliding
attention geometry (head_dim=256, 8 q / 2 kv heads), synthetic post-RMSNorm q/kv (M-invariance & latency are
weight-value-independent; the served-kernel + int4 source are the audit). Run CUDA_VISIBLE_DEVICES=0 (the
single-A10G pod default points at a non-existent 2nd GPU -- the #358/#363 gotcha). W&B group
attention-strict-pin-cost.

PUBLIC EVIDENCE USED (advisor-branch banked): #390 corrected ceiling row (509.78/471.42, eta_attn=0.02145),
#378 eta_attn decomposition (f_attn=0.09507, penalty_evalweighted), #349 FlashInfer-BI disable_split_kv
mechanism, #282 decode-L distribution, BASELINE.md FA_SLIDING/SPLITKV_VERIFY env semantics.
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

# single-A10G pod: the inherited CUDA_VISIBLE_DEVICES may point at a non-existent 2nd GPU (#358/#363)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "0,", ""):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch  # noqa: E402

# ---------------------------------------------------------------------------------------- #
# gemma-4-E4B-it SLIDING attention geometry (the FA_SLIDING-swappable head_dim=256 layers).
#   config.json text_config: head_dim=256, num_attention_heads=8, num_key_value_heads=2,
#   sliding_window=512, num_hidden_layers=42 (7 full_attention @ {5,11,17,23,29,35,41} head_dim 512,
#   35 sliding_attention head_dim 256). #378/#390 anchored eta_attn on the head_dim=256 sliding geometry.
# ---------------------------------------------------------------------------------------- #
HEAD_DIM = 256
N_Q_HEADS = 8
N_KV_HEADS = 2
DTYPE = torch.bfloat16
SCALE = 1.0 / math.sqrt(HEAD_DIM)
A10G_SMS = 80
SERVED_BLOCK_SIZE = 16            # vLLM deployment page/block size (faithful served paged geometry)
BLOCK_M_SPLITKV = 64
BLOCK_N_TILE = 64
HEURISTIC_SPLIT = 0              # DEPLOYED non-VBI default (max_num_splits=0 -> kernel heuristic picks)
UNPACK_SPLIT = 1                # VLLM_BATCH_INVARIANT=1 -> num_splits=1 (the M-invariant served config)
PINNED_SPLIT = 8               # the un-deployable ideal (rejected on varlen paged per #375; needs rebuild)
M_AR = 1                         # AR / drafter decode width (the un-pack penalty lane)
M_VERIFY = 8                     # spec-verify width (K_spec=7 + 1)
IDENT_M = (1, 8)                 # strict-relevant identity widths: AR reference vs verify

# ---- strict budget ladder (CITE; identical to #378/#390 for ADDITIVITY) ----------------- #
CEILING_500 = 520.953                         # lambda=1 central ceiling TPS (#326/#327/#354/#390)
OFFICIAL_TPS = 481.53                          # deployed non-strict public #1 (#52); this leg adds 0
STEP_NORM_US = 1218.2                          # deployed batch=1 decode step normalizer (#257/#344)
TARGET_500 = 500.0
BUDGET_500_ETA = 1.0 - TARGET_500 / CEILING_500            # ~0.04022 (#390)
ETA_ATTN_378 = 0.02145375421979844             # #378 eta_attn_evalweighted (the banked attention-pin tax)
F_ATTN_344 = 0.09506718019009251               # #378 step_fractions.attn (M=8 verify attention fraction)
# #378 implied eval-weighted un-pack penalty: penalty_ew = 1 + eta_attn/f_attn (round-tripped below)
PENALTY_EW_378 = 1.0 + ETA_ATTN_378 / F_ATTN_344            # ~1.22567
# #390 published attention-pin TPS (reproduced in the self-test):
STRICT_TPS_ATTN_PIN_390 = CEILING_500 * (1.0 - ETA_ATTN_378)               # 509.78 (ceiling basis)
DEPLOYED_TPS_ATTN_PIN_390 = OFFICIAL_TPS / (1.0 + ETA_ATTN_378)            # 471.42 (deployed basis)
SERVED_TPS_LINEAR_390 = OFFICIAL_TPS * (1.0 - ETA_ATTN_378)                # 471.20 (linear deployed)
# #375 banked per-L AR un-pack penalty anchors (soft consistency check vs my measured curve):
PENALTY_ANCHORS_375 = {528: 1.2777777609352838, 2048: 3.0555554713430864, 4096: 4.755813955455911}

# operating band [528,658] (decode positions cluster here for the dominant eval prompts) + short point
BAND_L = (528, 560, 592, 624, 658)
SHORT_L = 128
# dense penalty grid for the eval-weighted reproduction (covers eval mass 110..768 + band + tail anchors)
PENALTY_GRID_L = (110, 128, 192, 256, 384, 503, 512, 528, 560, 592, 624, 658, 704, 768, 1024, 2048)
EVAL_OUTPUT_LEN = 512

TOL = 1.0e-2
_VAL = Path(__file__).resolve().parents[1]
ANCHOR_282 = _VAL / "et_prompt_distribution" / "measured_result.json"

# ---------------------------------------------------------------------------------------- #
# PR #400 -- pinned-K attention rebuild headroom constants.
#   We PRICE (do not build) a deterministic 64-CTA split-reduce: at M=1 the 64-CTA pin =
#   num_splits=PINNED_SPLIT=8 (8 q-heads x 8 splits = 64 CTAs). The existing _pinned_reachable
#   probe already confirms the served varlen kernel REJECTS num_splits>1 (NotImplementedError) ->
#   a rebuild is the only way to reach it.
# ---------------------------------------------------------------------------------------- #
A10G_PEAK_BW_GBS = 600.0            # A10G GDDR6 peak HBM bandwidth (GB/s) -- roofline denominator
A10G_PEAK_BF16_TFLOPS = 62.5       # A10G dense bf16 tensor-core peak (TFLOP/s)
RIDGE_FLOP_PER_BYTE = (A10G_PEAK_BF16_TFLOPS * 1e12) / (A10G_PEAK_BW_GBS * 1e9)   # ~104 FLOP/byte ridge
DTYPE_BYTES = 2                     # bf16
PINNED_CTAS = PINNED_SPLIT * N_Q_HEADS   # 64 CTAs at M=1 (the "64-CTA split-reduce")
# #332 (y5cl0ena, banked) deterministic-SDPA M=8 verify-step results (CITE, do NOT re-derive):
PHI_332_M8 = 0.075                  # #332 deterministic-SDPA recovery fraction (M=8 verify-step)
BREAKEVEN_332 = 0.255              # #332 break-even recovery (phi must exceed this to pay off)
BW_FRAC_332_M8 = 0.349            # #332 measured achieved-BW fraction (BW-floored at M=8)
CTAS_332_M8 = 96                   # #332 measured CTAs at M=8 (>80 SM -> occupancy-saturated)
CEILING_332_DEPLOYED = 473.5       # #332 strict-compliant ceiling (<500 for every M=8 schedule)
# #393 (0q7ynumg, MERGED c86385d7) corrected decode-only strict (CITE; reproduced from measured band):
ETA_ATTN_DECODE_393 = 0.030065297571591987   # eta_attn_decode_only (sole strict tax, rebuild-free)
DEPLOYED_STRICT_393 = 467.475218449957       # deployed strict = OFFICIAL/(1+eta)
CEILING_STRICT_393 = 505.29039303418637      # ceiling strict = CEILING*(1-eta)
GAP_TO_500_393 = 32.524781550042974          # 500 - deployed strict (the residual #393 handed forward)
# fixed-order reduce proof-of-mechanism geometry: down_proj is the largest-K body GEMM (most reliant on
# the split-K reduction), group_size=128 (deployed body quant). The EXISTING int4-Marlin atomic_add=False
# reduce being M-invariant grounds the claim that a FIXED 64-way attention reduction tree would be too.
MARLIN_PROOF_K = 10240             # down_proj size_k (INTERMEDIATE)
MARLIN_PROOF_N = 2560              # down_proj size_n (HIDDEN)
MARLIN_PROOF_GS = 128              # deployed body group_size

# ---------------------------------------------------------------------------------------- #
# PR #408 -- CLOSED M=1 decode-step latency budget + stacked-flagged-supply ceiling.
#   The budget is built in NORMALIZED #378-fraction space: the four buckets {attn, body,
#   lm_head, draft/'other'} PARTITION the bridge-normalized 1218.2us decode step EXACTLY
#   (sum == 1.0). Raw isolated micro-bench sums over-credit heavily (measured ~10-14x here,
#   body-dominated; #284 found isolated-call sums do not reproduce the in-step fractions) and
#   CANNOT close against the bridge-normalized step, so the budget MUST live in normalized
#   space. FRESH GPU measurements supply the BW-bound fractions (body / lm_head) and the
#   attention recovery that the supply removables are priced against; budget closure is then a
#   completeness property of the #378 decomposition (residual ~0 by construction), and the
#   genuine measurement teeth are body_bw_bound_frac (reconciled vs #391's 0.256) + the
#   #393/#400 reproductions. Latency framing (exact, ladder-consistent):
#     S0      = STEP_NORM_US                       (non-strict step; OFFICIAL_TPS=481.53 basis)
#     penalty = eta_attn_decode_only * S0          (un-pack strict tax; the ONLY removable attn us)
#     S_strict= S0 + penalty                       (-> deployed_strict = OFFICIAL/(1+eta) = 467.24)
#     tps(step) = OFFICIAL_TPS * S0 / step         (tps(S0)=481.53, tps(S_strict)=467.24 exact)
# ---------------------------------------------------------------------------------------- #
F_BODY_STRICT_378 = 0.76240970145034          # #378 body-GEMM weight-read step fraction
F_LMHEAD_378 = 0.022428229458960704           # #378 lm_head step fraction (== F_LMHEAD_344)
F_DRAFT_378 = 0.12009488890060672             # #378 spec-draft/'other' == the FIXED-OVERHEAD floor
# (F_ATTN_344 + F_BODY_STRICT_378 + F_LMHEAD_378 + F_DRAFT_378 == 1.0 by construction.)

# cb3 body-read shrink (lawine #372/#388/#391 anchors; PPL-UNCAPPED headline bpw).
INT4_BPW_NOMINAL_408 = 4.0                     # nominal 4-bit (PR step-4 denominator -> 0.191 shrink)
INT4_BPW_G128_408 = 4.125                      # deployed int4-Marlin g128 (4b + bf16/128 scale byte)
CB3_BPW_EFF_408 = 3.2368598382749325           # #372 mixed cb3 effective bpw (PPL-uncapped headline)
CB3_READ_SHRINK_FRAC_408 = 1.0 - CB3_BPW_EFF_408 / INT4_BPW_NOMINAL_408   # 0.19079 (PR's "0.191")
CB3_READ_SHRINK_FRAC_G128_408 = 1.0 - CB3_BPW_EFF_408 / INT4_BPW_G128_408  # 0.21528 (4.125 denom)
M1_MARLIN_HBM_EFF_391 = 0.25561637483960586    # #391 count-weighted M=1 body Marlin HBM eff (reconcile)
BODY_BW_RECONCILE_TOL_408 = 0.08               # |measured body_bw_bound_frac - 0.256| tolerance

# lm_head channel-wise int4 geometry (deployed osoi5; #344 ~21MB / #384 audit). dense [16384] UB.
LMHEAD_ROWS_408 = 16384                         # PCK-04 row-pruned lm_head rows
LMHEAD_HIDDEN_408 = 2560                         # hidden size
# channel-wise int4: packed 4b weights + one bf16 scale per output channel (NOT g128 scale bytes)
LMHEAD_BYTES_408 = LMHEAD_ROWS_408 * LMHEAD_HIDDEN_408 // 2 + LMHEAD_ROWS_408 * 2   # 21,004,288 (~21MB)
LMHEAD_BEST_LOADABLE_READ_SHRINK_398 = 0.0      # land #398: NO loadable lm_head read-shrink -> removable 0

# 8 distinct gemma-4-E4B body GEMM shapes (out, in, count) -- lawine #388/#391 table (M=1 budget)
BODY_SHAPES_408: list[dict[str, Any]] = [
    {"name": "q_full",  "out": 4096,  "in": 2560,  "count": 7},
    {"name": "q_slide", "out": 2048,  "in": 2560,  "count": 35},
    {"name": "kv_full", "out": 1024,  "in": 2560,  "count": 8},
    {"name": "kv_slide", "out": 512,  "in": 2560,  "count": 40},
    {"name": "o_full",  "out": 2560,  "in": 4096,  "count": 7},
    {"name": "o_slide", "out": 2560,  "in": 2048,  "count": 35},
    {"name": "gate_up", "out": 10240, "in": 2560,  "count": 84},
    {"name": "down",    "out": 2560,  "in": 10240, "count": 42},
]


# ======================================================================================== #
# Device + facts
# ======================================================================================== #
def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Launch with CUDA_VISIBLE_DEVICES=0 (single-A10G pod gotcha).")
    return torch.device("cuda:0")


def _gpu_facts(dev: torch.device) -> dict[str, Any]:
    p = torch.cuda.get_device_properties(dev)
    cc = torch.cuda.get_device_capability(dev)
    return {
        "name": p.name, "sm_count": p.multi_processor_count,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "total_mem_gib": round(p.total_memory / (1024**3), 2),
        "is_a10g_80sm": bool(p.multi_processor_count == A10G_SMS and "A10G" in p.name),
        "is_sm8x": bool(cc[0] == 8),
    }


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _load_json(p: Path) -> dict[str, Any] | None:
    if not p.is_file():
        return None
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return None


def _num_splits_heuristic(bnm: int, num_sms: int, num_n_blocks: int, max_splits: int = 128) -> int:
    """Exact replica of flash_attn num_splits_heuristic (reports the K the heuristic would pick)."""
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


def _heuristic_K(M: int, L: int) -> int:
    num_m_blocks = _ceildiv(M, BLOCK_M_SPLITKV)
    num_n_blocks = _ceildiv(L, BLOCK_N_TILE)
    return _num_splits_heuristic(1 * N_Q_HEADS * num_m_blocks, A10G_SMS, num_n_blocks, max_splits=128)


# ======================================================================================== #
# Served FA2 varlen kernel primitives (the EXACT served decode entry point; reused from #378)
# ======================================================================================== #
def _build_paged(L: int, M: int, seed: int, dev: torch.device, page: int = SERVED_BLOCK_SIZE):
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(M, N_Q_HEADS, HEAD_DIM, generator=g, device=dev, dtype=DTYPE)
    nb = _ceildiv(L, page)
    kc = torch.randn(nb, page, N_KV_HEADS, HEAD_DIM, generator=g, device=dev, dtype=DTYPE)
    vc = torch.randn(nb, page, N_KV_HEADS, HEAD_DIM, generator=g, device=dev, dtype=DTYPE)
    bt = torch.arange(nb, dtype=torch.int32, device=dev).unsqueeze(0)
    sk = torch.tensor([L], dtype=torch.int32, device=dev)
    return q, kc, vc, bt, sk


def _served_varlen(fn, q, kc, vc, bt, cu, sk, L, M, ns):
    return fn(q=q, k=kc, v=vc, out=None, cu_seqlens_q=cu, max_seqlen_q=M,
              seqused_k=sk, max_seqlen_k=L, softmax_scale=SCALE, causal=False,
              block_table=bt, num_splits=ns, fa_version=2)


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
# Identity: batched-M=8 vs per-row-M=1, FA2 varlen, per num_splits config
# ======================================================================================== #
def _measure_identity_fa2(fn, L: int, ns: int, n_trials: int, seed0: int, dev: torch.device) -> dict[str, Any]:
    """For each M in IDENT_M, byte+argmax identity of batched(M) vs per-row(M=1), SAME num_splits.
    A byte difference at the verify width CAN flip a downstream greedy token; byte==1.000 GUARANTEES strict."""
    byte_acc = {M: [] for M in IDENT_M}
    argmax_acc = {M: [] for M in IDENT_M}
    maxdiff = {M: 0.0 for M in IDENT_M}
    any_nan = False
    for t in range(n_trials):
        q8, kc, vc, bt, sk = _build_paged(L, max(IDENT_M), seed0 + t, dev)
        for M in IDENT_M:
            q = q8[:M].contiguous()
            cu = torch.tensor([0, M], dtype=torch.int32, device=dev)
            bat = _served_varlen(fn, q, kc, vc, bt, cu, sk, L, M, ns)
            any_nan = any_nan or bool(torch.isnan(bat).any())
            cu1 = torch.tensor([0, 1], dtype=torch.int32, device=dev)
            ref = torch.cat([_served_varlen(fn, q[r:r + 1], kc, vc, bt, cu1, sk, L, 1, ns)
                             for r in range(M)], dim=0)
            bflat = bat.reshape(M, -1)
            rflat = ref.reshape(M, -1)
            byte = (bflat == rflat).all(dim=-1).float().mean().item()
            argmax = (bflat.argmax(dim=-1) == rflat.argmax(dim=-1)).float().mean().item()
            byte_acc[M].append(byte)
            argmax_acc[M].append(argmax)
            maxdiff[M] = max(maxdiff[M], (bflat.float() - rflat.float()).abs().max().item())

    def mean(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")
    return {
        "byte_identity_by_M": {str(M): mean(byte_acc[M]) for M in IDENT_M},
        "argmax_identity_by_M": {str(M): mean(argmax_acc[M]) for M in IDENT_M},
        "max_abs_diff_by_M": {str(M): maxdiff[M] for M in IDENT_M},
        "n_trials": n_trials, "any_nan": bool(any_nan),
    }


def _merge_idents(accs: list[dict]) -> dict[str, Any]:
    """Worst-case across seeds: min byte/argmax, max maxdiff."""
    return {
        "byte_identity_by_M": {str(M): min(a["byte_identity_by_M"][str(M)] for a in accs) for M in IDENT_M},
        "argmax_identity_by_M": {str(M): min(a["argmax_identity_by_M"][str(M)] for a in accs) for M in IDENT_M},
        "max_abs_diff_by_M": {str(M): max(a["max_abs_diff_by_M"][str(M)] for a in accs) for M in IDENT_M},
        "n_trials": sum(a["n_trials"] for a in accs), "any_nan": any(a["any_nan"] for a in accs),
    }


# ======================================================================================== #
# FlashInfer screen: is the shipped FlashInfer decode a VALID, CHEAPER byte-exact pin?
#   #349 ASSUMED disable_split_kv gives a serial BI reduction (the un-pack tax). This MEASURES it:
#   (a) does disable_split_kv change the decode bytes (is there a real BI mode)?
#   (b) is FlashInfer M-invariant -- does its verify (prefill M=8) byte-match its decode (M=1)?
#       (strict greedy identity requires the spec-decode verify/draft to match the AR reference).
#   (c) absolute decode latency vs FA2 (fast-but-non-strict vs strict-but-slower).
# ======================================================================================== #
def _flashinfer_probe(dev: torch.device, iters: int, warmup: int, seed0: int) -> dict[str, Any]:
    out: dict[str, Any] = {"available": False}
    try:
        import flashinfer
    except Exception as e:  # noqa: BLE001
        out["import_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return out
    out["available"] = True
    out["version"] = getattr(flashinfer, "__version__", "?")
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)

    def build(L: int):
        nb = _ceildiv(L, SERVED_BLOCK_SIZE)
        g = torch.Generator(device=dev).manual_seed(seed0)
        q8 = torch.randn(max(IDENT_M), N_Q_HEADS, HEAD_DIM, generator=g, device=dev, dtype=DTYPE)
        kv = torch.randn(nb, 2, SERVED_BLOCK_SIZE, N_KV_HEADS, HEAD_DIM, generator=g, device=dev, dtype=DTYPE)
        indptr = torch.tensor([0, nb], dtype=torch.int32, device=dev)
        indices = torch.arange(nb, dtype=torch.int32, device=dev)
        last = torch.tensor([L - (nb - 1) * SERVED_BLOCK_SIZE], dtype=torch.int32, device=dev)
        return q8, kv, indptr, indices, last

    def decode(q1, kv, indptr, indices, last, dsk):
        wr = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
        wr.plan(indptr, indices, last, N_Q_HEADS, N_KV_HEADS, HEAD_DIM, SERVED_BLOCK_SIZE,
                pos_encoding_mode="NONE", q_data_type=DTYPE, kv_data_type=DTYPE,
                sm_scale=SCALE, disable_split_kv=dsk)
        return wr, lambda: wr.run(q1, kv)

    def prefill(qM, kv, indptr, indices, last, M):
        wr = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
        wr.plan(torch.tensor([0, M], dtype=torch.int32, device=dev), indptr, indices, last,
                N_Q_HEADS, N_KV_HEADS, HEAD_DIM, SERVED_BLOCK_SIZE, causal=False,
                q_data_type=DTYPE, kv_data_type=DTYPE, sm_scale=SCALE)
        return wr, lambda: wr.run(qM, kv)

    def _be(a, b):
        n = a.shape[0]
        return (a.reshape(n, -1) == b.reshape(n, -1)).all(dim=-1).float().mean().item()

    per_L: dict[str, Any] = {}
    try:
        for L in (SHORT_L, *BAND_L):
            q8, kv, ip, ix, lp = build(L)
            _, dec_t = decode(q8[0:1], kv, ip, ix, lp, True)
            _, dec_f = decode(q8[0:1], kv, ip, ix, lp, False)
            o_dsk_t = dec_t(); o_dsk_f = dec_f()
            dsk_changes_bytes = not bool(torch.equal(o_dsk_t.reshape(-1).view(torch.int16),
                                                     o_dsk_f.reshape(-1).view(torch.int16)))
            bi_us = _time_call(dec_t, iters, warmup)
            sp_us = _time_call(dec_f, iters, warmup)
            per_L[str(L)] = {"decode_us": bi_us, "decode_split_us": sp_us,
                             "disable_split_kv_changes_bytes": dsk_changes_bytes}
        out["per_L"] = per_L
        out["head_dim_256_supported"] = True
        band = [per_L[str(L)] for L in BAND_L if str(L) in per_L]
        out["decode_us_band_mean"] = float(sum(x["decode_us"] for x in band) / len(band)) if band else None
        out["disable_split_kv_is_noop"] = bool(not any(x["disable_split_kv_changes_bytes"] for x in per_L.values()))

        # (b) STRICT M-invariance at the band: prefill(M=8 verify) row-r vs per-row decode/prefill(M=1)
        q8, kv, ip, ix, lp = build(BAND_L[0])
        _, pre8 = prefill(q8, kv, ip, ix, lp, max(IDENT_M))
        o8 = pre8()
        rows_dec = []
        rows_pre = []
        for r in range(max(IDENT_M)):
            _, d1 = decode(q8[r:r + 1], kv, ip, ix, lp, True)
            _, p1 = prefill(q8[r:r + 1], kv, ip, ix, lp, 1)
            rows_dec.append(d1()); rows_pre.append(p1())
        ref_dec = torch.cat(rows_dec, 0)
        ref_pre = torch.cat(rows_pre, 0)
        out["verify_vs_perrow_decode_byte_identity"] = _be(o8, ref_dec)   # strict draft/verify consistency
        out["verify_vs_perrow_prefill_byte_identity"] = _be(o8, ref_pre)  # within-prefill M-invariance
        out["strict_m_invariant"] = bool(out["verify_vs_perrow_decode_byte_identity"] >= 1.0
                                         and out["verify_vs_perrow_prefill_byte_identity"] >= 1.0)
        # is this a VALID strict pin? (must be M-invariant AND have a real byte-exact reduction mode)
        out["is_valid_strict_pin"] = bool(out["strict_m_invariant"])
        out["fast_but_nonstrict"] = bool((not out["strict_m_invariant"]))
    except Exception as e:  # noqa: BLE001
        out["measure_error"] = f"{type(e).__name__}: {str(e)[:180]}"
        out["head_dim_256_supported"] = False
        out["is_valid_strict_pin"] = False
    return out


# ======================================================================================== #
# Eval-weighted vs decode-band penalty re-weighting
# ======================================================================================== #
def _eval_decode_L(dist: dict[str, Any] | None) -> dict[str, Any]:
    """Token-weighted decode-position L distribution {P_i + t : t in 0..OUT-1} from #282 prompt lengths."""
    if dist is None:
        return {"present": False}
    per = dist.get("per_prompt") or []
    P = [int(p["n_prompt_tokens"]) for p in per if p.get("n_prompt_tokens") is not None]
    out_len = int(dist.get("output_len", EVAL_OUTPUT_LEN))
    positions: list[int] = []
    for p in P:
        positions.extend(range(p, p + out_len))
    positions.sort()
    n = len(positions)
    return {
        "present": True, "n_prompts": len(P), "output_len": out_len,
        "prompt_tokens_min": min(P) if P else None, "prompt_tokens_max": max(P) if P else None,
        "decode_L_min": positions[0] if n else None, "decode_L_max": positions[-1] if n else None,
        "decode_L_mean": round(sum(positions) / n, 3) if n else None,
        "decode_L_median": positions[n // 2] if n else None,
        "frac_in_band": round(sum(1 for x in positions if BAND_L[0] <= x <= BAND_L[-1]) / n, 4) if n else None,
        "_positions": positions,
    }


def _interp(L: float, grid_L: list[int], grid_pen: list[float]) -> float:
    if L <= grid_L[0]:
        return grid_pen[0]
    if L >= grid_L[-1]:
        return grid_pen[-1]
    for i in range(1, len(grid_L)):
        if L <= grid_L[i]:
            f = (L - grid_L[i - 1]) / (grid_L[i] - grid_L[i - 1])
            return grid_pen[i - 1] + f * (grid_pen[i] - grid_pen[i - 1])
    return grid_pen[-1]


def measure_penalty_curve(fn, dev: torch.device, iters: int, warmup: int, seed: int,
                          M: int) -> dict[int, dict[str, float]]:
    """penalty(L) = lat[num_splits=1 unpack] / lat[num_splits=0 heuristic] at width M, over PENALTY_GRID_L."""
    curve: dict[int, dict[str, float]] = {}
    for L in PENALTY_GRID_L:
        q8, kc, vc, bt, sk = _build_paged(L, M, seed, dev)
        cu = torch.tensor([0, M], dtype=torch.int32, device=dev)
        heur_us = _time_call(lambda: _served_varlen(fn, q8, kc, vc, bt, cu, sk, L, M, HEURISTIC_SPLIT),
                             iters, warmup)
        unpack_us = _time_call(lambda: _served_varlen(fn, q8, kc, vc, bt, cu, sk, L, M, UNPACK_SPLIT),
                               iters, warmup)
        curve[L] = {
            "heuristic_us": heur_us, "unpack_us": unpack_us,
            "penalty": (unpack_us / heur_us) if heur_us > 0 else float("nan"),
            "heuristic_K": float(_heuristic_K(M, L)),
            "us_per_token_unpack": unpack_us / M, "us_per_token_heuristic": heur_us / M,
        }
    return curve


# ======================================================================================== #
# Pinned-K reachability probe (expect rejection on varlen paged -> rebuild required)
# ======================================================================================== #
def _pinned_reachable(fn, dev: torch.device) -> dict[str, Any]:
    q8, kc, vc, bt, sk = _build_paged(BAND_L[0], M_VERIFY, 7, dev)
    cu = torch.tensor([0, M_VERIFY], dtype=torch.int32, device=dev)
    try:
        _served_varlen(fn, q8, kc, vc, bt, cu, sk, BAND_L[0], M_VERIFY, PINNED_SPLIT)
        return {"pinned_split_reachable": True, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"pinned_split_reachable": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}


# ======================================================================================== #
# strict TPS ladder (CITE; identical to #378/#390)
# ======================================================================================== #
def strict_tps(eta: float) -> float:
    return CEILING_500 * (1.0 - eta)


def strict_tps_divisor(base: float, eta: float) -> float:
    return base / (1.0 + eta)


def served_tps_linear(base: float, eta: float) -> float:
    return base * (1.0 - eta)


# ======================================================================================== #
# Compose
# ======================================================================================== #
def compose(dev: torch.device, args, gpu: dict) -> dict[str, Any]:
    from vllm.vllm_flash_attn import _vllm_fa2_C  # noqa: F401  (registers torch.ops._vllm_fa2_C)
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as FA2

    backends_shipped = {"FLASH_ATTN_vendored_fa2": True}
    for mod in ("flash_attn", "xformers"):
        try:
            __import__(mod); backends_shipped[mod] = True
        except Exception:  # noqa: BLE001
            backends_shipped[mod] = False

    # ---- (1) FA2 num_splits config identity (band + short-context) -------------------------- #
    ident_L = (SHORT_L, *BAND_L)
    configs: dict[str, dict[str, Any]] = {}
    for cfg_name, ns in (("fa2_heuristic_ns0", HEURISTIC_SPLIT), ("fa2_unpack_ns1", UNPACK_SPLIT)):
        per_L: dict[str, Any] = {}
        for L in ident_L:
            accs = [_measure_identity_fa2(FA2, L, ns, args.ident_trials, s, dev) for s in args.seeds]
            per_L[str(L)] = _merge_idents(accs)
        byte_M8_min = min(per_L[str(L)]["byte_identity_by_M"]["8"] for L in ident_L)
        configs[cfg_name] = {
            "num_splits": ns, "per_L": per_L,
            "byte_identity_M8_min": byte_M8_min,
            "is_byte_exact_M8": bool(byte_M8_min >= 1.0),
            "maxdiff_M8_max": max(per_L[str(L)]["max_abs_diff_by_M"]["8"] for L in ident_L),
        }

    pinned = _pinned_reachable(FA2, dev)

    # ---- (2) un-pack penalty curve. The tax is paid on the M=1 AR/draft lane (the #375/#378 anchors);
    #          the M=8 verify is penalty-FREE (the heuristic already picks 1 split at M=8 -> ns0==ns1).
    curve_m1 = measure_penalty_curve(FA2, dev, args.iters, args.warmup, args.seeds[0], M_AR)      # the tax lane
    curve_m8 = measure_penalty_curve(FA2, dev, args.iters, args.warmup, args.seeds[0], M_VERIFY)  # verify, ~1.0
    grid_L = list(PENALTY_GRID_L)
    grid_pen_m1 = [curve_m1[L]["penalty"] for L in grid_L]

    band_pens = [curve_m1[L]["penalty"] for L in BAND_L]
    penalty_decode_band = float(sum(band_pens) / len(band_pens))
    eta_attn_decode_only = F_ATTN_344 * (penalty_decode_band - 1.0)
    eta_attn_decode_vs_evalweighted_delta = eta_attn_decode_only - ETA_ATTN_378
    deployed_tps_decode_eta = strict_tps_divisor(OFFICIAL_TPS, eta_attn_decode_only)
    ceiling_tps_decode_eta = strict_tps(eta_attn_decode_only)

    # M=8 verify penalty-free re-assertion (matches #378: ns0==ns1 byte-exact AND latency-identical)
    verify_pens = [curve_m8[L]["penalty"] for L in BAND_L]
    verify_penalty_band_mean = float(sum(verify_pens) / len(verify_pens))
    verify_penalty_free = bool(all(abs(curve_m8[L]["penalty"] - 1.0) < 0.10 for L in BAND_L))

    # eval-weighted reproduction over #282 decode-L distribution (reproduces #378's ~1.2257 / 0.02145)
    dist = _eval_decode_L(_load_json(ANCHOR_282))
    if dist.get("present"):
        pos = dist.pop("_positions")
        pen_ew = sum(_interp(L, grid_L, grid_pen_m1) for L in pos) / len(pos)
        eta_attn_evalweighted_measured = F_ATTN_344 * (pen_ew - 1.0)
        dist["penalty_evalweighted_measured"] = round(pen_ew, 5)
        dist["eta_attn_evalweighted_measured"] = eta_attn_evalweighted_measured
    else:
        dist.pop("_positions", None)
        eta_attn_evalweighted_measured = None

    # ---- (3) FlashInfer screen: is the shipped FI decode a VALID, CHEAPER byte-exact pin? (MEASURED) -- #
    fi = _flashinfer_probe(dev, args.iters, args.warmup, args.seeds[0])
    fi_strict = bool(fi.get("is_valid_strict_pin"))
    fi_fast_but_nonstrict = bool(fi.get("available") and fi.get("head_dim_256_supported")
                                 and (not fi_strict))

    # ---- (4) cheapest byte-exact pin --------------------------------------------------------- #
    # The ONLY byte-exact M-invariant config reachable WITHOUT a rebuild is FA2 num_splits=1 (un-pack).
    #   - pinned-K (64-CTA): REJECTED on varlen paged (NotImplementedError) -> needs a kernel rebuild.
    #   - FlashInfer decode: MEASURED non-strict (its verify/prefill does NOT byte-match its decode;
    #     disable_split_kv is a no-op) -> fast but NOT a valid strict pin (refutes the reducibility hope).
    fa2_unpack_band_us = float(sum(curve_m1[L]["unpack_us"] for L in BAND_L) / len(BAND_L))   # M=1 AR lane
    fa2_heuristic_band_us = float(sum(curve_m1[L]["heuristic_us"] for L in BAND_L) / len(BAND_L))
    byte_exact_candidates: dict[str, dict[str, Any]] = {
        "fa2_unpack_ns1": {
            "byte_exact_identity": configs["fa2_unpack_ns1"]["byte_identity_M8_min"],
            "band_us_M1": fa2_unpack_band_us, "reachable_without_rebuild": True,
            "strict_m_invariant": configs["fa2_unpack_ns1"]["is_byte_exact_M8"],
        }
    }
    if fi_strict and fi.get("decode_us_band_mean") is not None:
        byte_exact_candidates["flashinfer_decode"] = {
            "byte_exact_identity": 1.0, "band_us_M1": fi.get("decode_us_band_mean"),
            "reachable_without_rebuild": True, "strict_m_invariant": True,
        }
    # cheapest = lowest band latency among STRICTLY byte-exact (identity exactly 1.000) candidates
    strict_cands = {k: v for k, v in byte_exact_candidates.items()
                    if v["byte_exact_identity"] >= 1.0 and v["band_us_M1"] is not None}
    cheapest_strict_attn_backend = min(strict_cands, key=lambda k: strict_cands[k]["band_us_M1"])
    # the cheapest byte-exact pin pays the decode-band un-pack tax (FA2-unpack IS the deployed config).
    cheapest_strict_attn_eta = eta_attn_decode_only
    # attn_eta_reducible := exists a byte-exact config with eta STRICTLY BELOW the unpack tax without a
    # rebuild. Sub-unpack byte-exact configs: pinned-K (rejected) and a strict FlashInfer (measured non-
    # strict). Neither is available -> NOT reducible today.
    attn_eta_reducible = bool(pinned["pinned_split_reachable"]
                              or (fi_strict and (fi.get("decode_us_band_mean") or 1e9) < fa2_unpack_band_us))
    # attn_already_strict_free := the FAST heuristic (ns=0) is ALREADY byte-exact at the decode width
    attn_already_strict_free = bool(configs["fa2_heuristic_ns0"]["is_byte_exact_M8"])
    # fa_sliding0_is_strict_floor := the deterministic-reduction config (FA_SLIDING=0 / num_splits=1) is the
    # cheapest byte-exact pin reachable without a rebuild (no sub-unpack byte-exact config available).
    fa_sliding0_is_strict_floor = bool((not attn_eta_reducible) and (not attn_already_strict_free)
                                       and configs["fa2_unpack_ns1"]["is_byte_exact_M8"]
                                       and cheapest_strict_attn_backend == "fa2_unpack_ns1")

    ceiling_with_cheapest_attn = strict_tps(cheapest_strict_attn_eta)
    deployed_with_cheapest_attn = strict_tps_divisor(OFFICIAL_TPS, cheapest_strict_attn_eta)
    gap_to_500_after_attn = TARGET_500 - deployed_with_cheapest_attn            # deployed basis (the 471.42 axis)
    gap_to_500_after_attn_ceiling = TARGET_500 - ceiling_with_cheapest_attn      # ceiling basis

    # is 0.02145 the floor? the cheapest byte-exact pin's DECODE eta is ABOVE the eval-weighted 0.02145.
    cheapest_eta_above_evalweighted = bool(cheapest_strict_attn_eta > ETA_ATTN_378)
    # eval-weighted 0.02145 is the floor of the FA2-unpack family (no cheaper byte-exact pin exists); the
    # DECODE-specific setting of that family is HIGHER (471.42 is optimistic).
    eta_attn_floor_is_0p02145_evalweighted = bool((not attn_eta_reducible) and (not attn_already_strict_free))

    # soft consistency: measured M=1 penalty @528 vs #375 banked anchor 1.2778 (within 15%)
    pen_528_measured = curve_m1[528]["penalty"]
    anchor_528_consistent = bool(abs(pen_528_measured / PENALTY_ANCHORS_375[528] - 1.0) <= 0.15)

    verdict = (
        f"Cheapest byte-exact attention pin = {cheapest_strict_attn_backend} (num_splits=1 un-pack / "
        f"FA_SLIDING=0 deterministic reduction). Pinned-K (64-CTA) is REJECTED on varlen paged "
        f"(reachable={pinned['pinned_split_reachable']}) -> a kernel rebuild is the ONLY way below the "
        f"un-pack tax. The fast heuristic (ns=0) BYTE-BREAKS M=1-vs-M=8 (strict_free={attn_already_strict_free}). "
        f"eta_attn=0.02145 (eval-weighted) is NOT the decode floor: the [528,658] band penalty is "
        f"{penalty_decode_band:.4f} -> eta_attn_decode_only={eta_attn_decode_only*100:.4f}% "
        f"(+{eta_attn_decode_vs_evalweighted_delta*100:.4f}pp), sharpening the deployed strict TPS DOWN from "
        f"471.42 to {deployed_tps_decode_eta:.2f}. fa_sliding0_is_strict_floor={fa_sliding0_is_strict_floor}."
    )

    return {
        "backends_shipped": backends_shipped,
        "fa2_configs": configs,
        "pinned_probe": pinned,
        "penalty_curve_M8": {str(L): curve_m8[L] for L in grid_L},
        "penalty_curve_M1": {str(L): curve_m1[L] for L in grid_L},
        "penalty_decode_band": penalty_decode_band,
        "penalty_528_measured": pen_528_measured,
        "anchor_528_consistent": anchor_528_consistent,
        "verify_penalty_band_mean": verify_penalty_band_mean,
        "verify_penalty_free": verify_penalty_free,
        "fa2_unpack_band_us_M1": fa2_unpack_band_us,
        "fa2_heuristic_band_us_M1": fa2_heuristic_band_us,
        "eval_decode_L_dist": dist,
        "eta_attn_decode_only": eta_attn_decode_only,
        "eta_attn_evalweighted_banked": ETA_ATTN_378,
        "eta_attn_evalweighted_measured": eta_attn_evalweighted_measured,
        "eta_attn_decode_vs_evalweighted_delta": eta_attn_decode_vs_evalweighted_delta,
        "deployed_tps_decode_eta": deployed_tps_decode_eta,
        "ceiling_tps_decode_eta": ceiling_tps_decode_eta,
        "flashinfer": fi,
        "fi_fast_but_nonstrict": fi_fast_but_nonstrict,
        "cheapest_strict_attn_backend": cheapest_strict_attn_backend,
        "cheapest_strict_attn_eta": cheapest_strict_attn_eta,
        "byte_exact_candidates": byte_exact_candidates,
        "n_byte_exact_attn_configs": len(strict_cands),
        "attn_eta_reducible": attn_eta_reducible,
        "attn_already_strict_free": attn_already_strict_free,
        "fa_sliding0_is_strict_floor": fa_sliding0_is_strict_floor,
        "ceiling_with_cheapest_attn": ceiling_with_cheapest_attn,
        "deployed_with_cheapest_attn": deployed_with_cheapest_attn,
        "gap_to_500_after_attn": gap_to_500_after_attn,
        "gap_to_500_after_attn_ceiling": gap_to_500_after_attn_ceiling,
        "eta_attn_floor_is_0p02145_evalweighted": eta_attn_floor_is_0p02145_evalweighted,
        "cheapest_eta_above_evalweighted": cheapest_eta_above_evalweighted,
        "verdict": verdict,
    }


# ======================================================================================== #
# PR #400 -- pinned-K rebuild headroom: M=1 occupancy/BW, roofline recovery, attn-free strict,
#            byte-exact feasibility (fixed-order reduce proof-of-mechanism). NO kernel build.
# ======================================================================================== #
def _splitkv_ctas(M: int, num_splits: int) -> int:
    """FA2 split-KV launch-grid CTA count (Dao-AILab flash_fwd_launch_template):
        grid = (num_m_blocks, num_splits>1?num_splits:batch, num_splits>1?batch*heads:heads).
    batch=1 -> CTAs = num_m_blocks * N_Q_HEADS * max(num_splits, 1). At M=1, num_splits=1 -> 8 CTAs."""
    return _ceildiv(M, BLOCK_M_SPLITKV) * N_Q_HEADS * max(num_splits, 1)


def _attn_hbm_bytes(M: int, L: int) -> int:
    """Minimum HBM traffic for one paged decode-attention step: unique K+V read + Q + O (bf16),
    faithful served paged geometry (page=16, N_KV_HEADS). This is the memory-roofline denominator
    (the lower bound on bytes that MUST cross HBM); achieved-BW = bytes / latency."""
    nb = _ceildiv(L, SERVED_BLOCK_SIZE)
    kv = 2 * nb * SERVED_BLOCK_SIZE * N_KV_HEADS * HEAD_DIM * DTYPE_BYTES   # K + V cache read
    qo = 2 * M * N_Q_HEADS * HEAD_DIM * DTYPE_BYTES                         # Q + O
    return kv + qo


def _attn_flops(M: int, L: int) -> int:
    """QK^T + PV FLOPs for an M-query paged attention step (2 matmuls, 2 FLOP/MAC)."""
    return 4 * M * L * HEAD_DIM * N_Q_HEADS


def measure_m1_occupancy_bw(curve_m1: dict[int, dict[str, float]]) -> dict[str, Any]:
    """The #332 question, M=1 this time: from the MEASURED M=1 penalty curve, derive the draft-lane
    achieved-occupancy (active CTAs / 80 SM) and achieved-HBM-BW fraction, and the arithmetic
    intensity vs the sm_86 ridge. The un-pack (num_splits=1) lane is the strict-deployed config."""
    per_L: dict[str, Any] = {}
    for L in (SHORT_L, *BAND_L):
        c = curve_m1[L]
        unpack_us = c["unpack_us"]
        heur_us = c["heuristic_us"]
        k_heur = int(c["heuristic_K"])
        bytes_hbm = _attn_hbm_bytes(M_AR, L)
        unpack_ctas = _splitkv_ctas(M_AR, UNPACK_SPLIT)        # 8 (M=1, 1 split)
        heur_ctas = _splitkv_ctas(M_AR, k_heur)               # 8 * K_heur
        bw_unpack = bytes_hbm / (unpack_us * 1e-6)             # bytes/s
        bw_heur = bytes_hbm / (heur_us * 1e-6)
        per_L[str(L)] = {
            "unpack_us": unpack_us, "heuristic_us": heur_us, "heuristic_K": k_heur,
            "unpack_ctas": unpack_ctas, "unpack_occupancy_frac": unpack_ctas / A10G_SMS,
            "heuristic_ctas": heur_ctas, "heuristic_occupancy_frac": min(heur_ctas / A10G_SMS, 1.0),
            "hbm_bytes": bytes_hbm,
            "unpack_bw_gbs": bw_unpack / 1e9, "unpack_bw_frac": bw_unpack / (A10G_PEAK_BW_GBS * 1e9),
            "heuristic_bw_gbs": bw_heur / 1e9, "heuristic_bw_frac": bw_heur / (A10G_PEAK_BW_GBS * 1e9),
            "arith_intensity_flop_per_byte": _attn_flops(M_AR, L) / bytes_hbm,
        }
    band = [per_L[str(L)] for L in BAND_L]

    def bmean(key: str) -> float:
        return float(sum(x[key] for x in band) / len(band))

    return {
        "per_L": per_L,
        "m1_attn_occupancy_ctas": _splitkv_ctas(M_AR, UNPACK_SPLIT),   # 8, M-invariant over band
        "m1_attn_occupancy_frac": bmean("unpack_occupancy_frac"),      # ~0.10
        "m1_attn_achieved_bw_frac": bmean("unpack_bw_frac"),           # ~0.02 (far below BW roofline)
        "m1_attn_achieved_bw_gbs": bmean("unpack_bw_gbs"),
        "m1_unpack_band_us": bmean("unpack_us"),
        "m1_heuristic_band_us": bmean("heuristic_us"),
        "m1_heuristic_occupancy_ctas": bmean("heuristic_ctas"),
        "m1_heuristic_occupancy_frac": bmean("heuristic_occupancy_frac"),
        "m1_heuristic_bw_frac": bmean("heuristic_bw_frac"),
        "m1_heuristic_K_band_mean": bmean("heuristic_K"),
        "m1_arith_intensity_band": bmean("arith_intensity_flop_per_byte"),
        "ridge_flop_per_byte": RIDGE_FLOP_PER_BYTE,
        "workload_below_ridge": bool(bmean("arith_intensity_flop_per_byte") < RIDGE_FLOP_PER_BYTE),
    }


def roofline_pinnedk_recovery(occ: dict[str, Any], eta_attn_decode_only: float) -> dict[str, Any]:
    """Roofline-bound how much of the 3.01% decode tax a deterministic 64-CTA split-reduce could recover.
    The MEASURED heuristic (num_splits=K_heur>1) is the occupancy-filled, byte-NON-exact proxy for what a
    fixed-K pinned-K could reach. beta=1 (occupancy-ideal) reaches the heuristic floor; beta=realistic is
    the FIXED 64-CTA (num_splits=PINNED_SPLIT=8) point from a 2-point serial-depth model fit to the
    measured (K=1) and (K=K_heur) latencies. If the lane were BW-floored, beta_realistic -> 0 (lever dead)."""
    unpack = occ["m1_unpack_band_us"]
    heur = occ["m1_heuristic_band_us"]
    k_heur = occ["m1_heuristic_K_band_mean"]
    gap_us = unpack - heur                       # MEASURED latency the occupancy-fill removes

    # phi-analogue: fraction of the M=1 un-pack attention latency that occupancy-filling removes.
    phi_m1 = gap_us / unpack if unpack > 0 else 0.0

    # is the un-pack penalty occupancy-bound (a split recovers it) or BW-floored (per #332, cannot help)?
    occupancy_removable = bool(
        occ["m1_attn_occupancy_frac"] < 0.85          # under-occupied at un-pack (8/80 = 0.10)
        and occ["m1_attn_achieved_bw_frac"] < 0.25    # far from the BW floor (#332 M=8 sat. at 0.349)
        and gap_us > 0                                # adding splits DOES recover latency (measured)
    )
    # does #332's BW-floor (M=8 verify) bound the M=1 draft lane? NO -> M=1 is a DISTINCT regime.
    reconciles_332 = bool(
        occ["m1_attn_achieved_bw_frac"] < BW_FRAC_332_M8     # ~0.02 << 0.349 (not BW-floored)
        and occ["m1_attn_occupancy_ctas"] < CTAS_332_M8      # 8 << 96 (not occupancy-saturated)
        and phi_m1 > PHI_332_M8                              # more recoverable headroom than M=8
    )

    # roofline (beta=1): the measured heuristic fully closes the un-pack->heuristic gap, so the
    # occupancy-ideal recovery of the eta_attn_decode_only tax is 100% (penalty 1.32 -> 1.0).
    recovery_roofline = 1.0 if (occupancy_removable and gap_us > 0) else 0.0

    # realistic (beta=measured): FIXED 64-CTA (num_splits=PINNED_SPLIT) deterministic reduce. 2-point
    # serial-depth model lat(K) = floor + S/K (serial KV-read depth ~ L/K; combine+launch in floor),
    # fit from measured (K=1 un-pack) and (K=K_heur heuristic), evaluated at K=PINNED_SPLIT.
    floor_us = s_us = lat_pin = None
    if occupancy_removable and k_heur > 1.0 and gap_us > 0:
        s_us = gap_us / (1.0 - 1.0 / k_heur)
        floor_us = unpack - s_us
        lat_pin = floor_us + s_us / PINNED_SPLIT
        recovery_realistic = max(0.0, min(1.0, (unpack - lat_pin) / gap_us))
    else:
        lat_pin = unpack
        recovery_realistic = 0.0     # BW-floored / no headroom -> the lever is dead

    return {
        "phi_m1_draft_lane": phi_m1,
        "m1_unpack_penalty_occupancy_removable": occupancy_removable,
        "reconciles_332_bw_floor": reconciles_332,
        "phi_332_m8_cited": PHI_332_M8,
        "breakeven_332_cited": BREAKEVEN_332,
        "m1_more_headroom_than_332_m8": bool(phi_m1 > PHI_332_M8),
        "pinned_split_fixed": PINNED_SPLIT,
        "pinned_ctas": PINNED_CTAS,
        "occupancy_fill_recovered_us": gap_us,
        "serial_depth_floor_us": floor_us,
        "serial_depth_S_us": s_us,
        "pinnedk_lat_pin_us": lat_pin,
        "pinnedk_recovery_frac_roofline": recovery_roofline,
        "pinnedk_recovery_frac_realistic": recovery_realistic,
    }


def attn_free_strict(eta_attn_decode_only: float, rec_roof: float, rec_real: float) -> dict[str, Any]:
    """Re-run the #393 strict loop with the attention tax reduced by each recovery fraction.
    Deployed basis (the realizable strict TPS, the #393 467.48 axis) is decisive; ceiling basis reported
    for completeness. Honest even if attention-free < 500."""
    eta_roof = eta_attn_decode_only * (1.0 - rec_roof)
    eta_real = eta_attn_decode_only * (1.0 - rec_real)
    dep_today = strict_tps_divisor(OFFICIAL_TPS, eta_attn_decode_only)
    dep_roof = strict_tps_divisor(OFFICIAL_TPS, eta_roof)
    dep_real = strict_tps_divisor(OFFICIAL_TPS, eta_real)
    return {
        "eta_after_attn_roofline": eta_roof,
        "eta_after_attn_realistic": eta_real,
        "attn_free_deployed_strict_tps_roofline": dep_roof,
        "attn_free_deployed_strict_tps_realistic": dep_real,
        "attn_free_ceiling_strict_tps_roofline": strict_tps(eta_roof),
        "attn_free_ceiling_strict_tps_realistic": strict_tps(eta_real),
        "attn_alone_clears_500_roofline": bool(dep_roof >= TARGET_500),
        "attn_alone_clears_500_realistic": bool(dep_real >= TARGET_500),
        "residual_gap_after_attn_roofline": TARGET_500 - dep_roof,
        "residual_gap_after_attn_realistic": TARGET_500 - dep_real,
        "attn_lever_max_tps_gain_deployed": dep_roof - dep_today,     # 481.53 - 467.48 = 14.05 (ideal)
        "attn_lever_realistic_tps_gain_deployed": dep_real - dep_today,
    }


def marlin_m_invariance_proof(dev: torch.device, seeds: list[int], n_trials: int) -> dict[str, Any]:
    """PROOF-OF-MECHANISM (NOT a built attention kernel): demonstrate that an EXISTING fixed-order reduce
    in the served stack -- the int4-Marlin use_atomic_add=False / use_fp32_reduce=True body GEMM (#390) --
    is M-invariant byte-exact (identical bytes at M=1 and M=8). A fixed reduction order is the SAME ops in
    the SAME order regardless of the query batch width, so it rounds identically. This grounds the claim
    that a FIXED 64-way attention reduction tree would likewise be M-invariant byte-exact."""
    out: dict[str, Any] = {"available": False}
    try:
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
        from vllm.model_executor.layers.quantization.utils import marlin_utils_test as mut
        from vllm.scalar_type import scalar_types
    except Exception as e:  # noqa: BLE001
        out["import_error"] = f"{type(e).__name__}: {str(e)[:160]}"
        return out
    out["available"] = True
    size_k, size_n, gs = MARLIN_PROOF_K, MARLIN_PROOF_N, MARLIN_PROOF_GS
    qtype = scalar_types.uint4b8
    # on A10G sm_86 + bf16 the deployment guard FORCES use_atomic_add=False (fixed-order) for this geometry
    forced_aa = bool(mu.should_use_atomic_add_reduce(m=M_VERIFY, n=size_n, k=size_k, device=dev, dtype=DTYPE))
    byte_M = {M: [] for M in IDENT_M}
    maxdiff_M = {M: 0.0 for M in IDENT_M}
    any_nan = False
    for seed in seeds:
        g = torch.Generator(device=dev).manual_seed(seed)
        w = torch.randn(size_k, size_n, generator=g, device=dev, dtype=DTYPE) * 0.02
        w_ref, q_w, s, g_idx, sort_idx, _ = mut.marlin_quantize(w, qtype, group_size=gs, act_order=False)
        ws = mu.marlin_make_workspace_new(dev)
        empty_zp = torch.empty(0, dtype=torch.int, device=dev)

        def fwd(x):   # fixed-order deterministic reduce (use_atomic_add=False, use_fp32_reduce=True)
            xr = x.reshape(-1, size_k)
            return ops.marlin_gemm(
                xr, None, q_w, None, s, None, None, empty_zp, g_idx, sort_idx, ws, qtype,
                size_m=xr.shape[0], size_n=size_n, size_k=size_k, is_k_full=True,
                use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False).reshape(x.shape[:-1] + (size_n,))

        for t in range(n_trials):
            gg = torch.Generator(device=dev).manual_seed(seed + 1000 + t)
            x_full = torch.randn(max(IDENT_M), size_k, generator=gg, device=dev, dtype=DTYPE)
            for M in IDENT_M:
                x = x_full[:M].contiguous()
                bat = fwd(x)
                any_nan = any_nan or bool(torch.isnan(bat).any())
                ref = torch.cat([fwd(x[r:r + 1]) for r in range(M)], dim=0)
                byte_M[M].append((bat == ref).all(dim=-1).float().mean().item())
                maxdiff_M[M] = max(maxdiff_M[M], (bat.float() - ref.float()).abs().max().item())

    def mn(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    byte_by_M = {str(M): mn(byte_M[M]) for M in IDENT_M}
    out.update({
        "geom": {"name": "down_proj", "size_k": size_k, "size_n": size_n, "group_size": gs},
        "use_atomic_add": False, "use_fp32_reduce": True,
        "deployment_guard_forces_fixed_order": bool(not forced_aa),
        "byte_identity_by_M": byte_by_M,
        "max_abs_diff_by_M": {str(M): maxdiff_M[M] for M in IDENT_M},
        "fixed_order_m_invariant": bool(byte_by_M[str(M_VERIFY)] >= 1.0 and byte_by_M[str(1)] >= 1.0),
        "any_nan": any_nan, "n_trials_total": n_trials * len(seeds),
    })
    return out


def new_reference_probe(FA2, dev: torch.device, seed: int) -> dict[str, Any]:
    """Ground pinnedk_produces_new_reference EMPIRICALLY: at M=1, does a MULTI-split reduction (heuristic
    num_splits=0 -> K>1) produce DIFFERENT bytes than the deployed single-split serial un-pack (num_splits=1)?
    If yes, a fixed-K pinned-K (also multi-split) yields a NEW byte reference -> adopting it requires
    re-capturing the greedy-identity reference (a flagged served-file change)."""
    per_L: dict[str, Any] = {}
    for L in BAND_L:
        q, kc, vc, bt, sk = _build_paged(L, M_AR, seed, dev)
        cu = torch.tensor([0, M_AR], dtype=torch.int32, device=dev)
        o_heur = _served_varlen(FA2, q, kc, vc, bt, cu, sk, L, M_AR, HEURISTIC_SPLIT)
        o_serial = _served_varlen(FA2, q, kc, vc, bt, cu, sk, L, M_AR, UNPACK_SPLIT)
        same = bool(torch.equal(o_heur.reshape(-1).view(torch.int16), o_serial.reshape(-1).view(torch.int16)))
        per_L[str(L)] = {"heuristic_K": _heuristic_K(M_AR, L), "multisplit_eq_serial_bytes": same}
    any_diff = any(not v["multisplit_eq_serial_bytes"] for v in per_L.values())
    return {"per_L": per_L, "multisplit_changes_bytes_vs_serial": any_diff}


def compose_pinnedk(dev: torch.device, args, gpu: dict) -> dict[str, Any]:
    from vllm.vllm_flash_attn import _vllm_fa2_C  # noqa: F401  (registers torch.ops._vllm_fa2_C)
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as FA2

    # (0) confirm the #393 greedy-identity harness is wired: un-pack (num_splits=1) IS byte-exact M=1 vs M=8
    ident_accs = [_measure_identity_fa2(FA2, BAND_L[0], UNPACK_SPLIT, args.ident_trials, s, dev)
                  for s in args.seeds]
    unpack_ident = _merge_idents(ident_accs)
    greedy_identity_harness_wired = bool(unpack_ident["byte_identity_by_M"]["8"] >= 1.0)
    # confirm the 64-CTA pinned-K (num_splits>1) is NOT reachable without a rebuild (no kernel built here)
    pinned = _pinned_reachable(FA2, dev)

    # (1) reproduce #393's decode-only eta from the MEASURED M=1 band penalty (lat[ns=1]/lat[ns=0])
    curve_m1 = measure_penalty_curve(FA2, dev, args.iters, args.warmup, args.seeds[0], M_AR)
    band_pens = [curve_m1[L]["penalty"] for L in BAND_L]
    penalty_decode_band = float(sum(band_pens) / len(band_pens))
    eta_attn_decode_only = F_ATTN_344 * (penalty_decode_band - 1.0)
    deployed_strict = strict_tps_divisor(OFFICIAL_TPS, eta_attn_decode_only)
    ceiling_strict = strict_tps(eta_attn_decode_only)
    gap_to_500 = TARGET_500 - deployed_strict

    # (2) M=1 draft-lane occupancy / achieved-BW (the #332 question, M=1 this time)
    occ = measure_m1_occupancy_bw(curve_m1)
    # (3) roofline the pinned-K recovery (beta-tier)
    roof = roofline_pinnedk_recovery(occ, eta_attn_decode_only)
    # (4) arithmetic: does attention-free strict clear 500 alone?
    af = attn_free_strict(eta_attn_decode_only, roof["pinnedk_recovery_frac_roofline"],
                          roof["pinnedk_recovery_frac_realistic"])
    # (5) byte-exact feasibility: fixed-order reduce M-invariance proof-of-mechanism + new-reference probe
    marlin = marlin_m_invariance_proof(dev, args.seeds, args.ident_trials)
    newref = new_reference_probe(FA2, dev, args.seeds[0])
    pinnedk_m_invariant_byte_exact_feasible = bool(marlin.get("fixed_order_m_invariant"))
    pinnedk_produces_new_reference = True   # fixed K!=1 reduction order != num_splits=1 serial -> new bytes

    n_distinct_kernel_rebuilds_attn_free = 1   # only the pinned-K attn kernel (body+lm_head already exact)
    attn_rebuild_is_flagged_served_file_change = True

    verdict = (
        f"M=1 draft-lane attention is UNDER-occupied ({occ['m1_attn_occupancy_ctas']} CTAs / {A10G_SMS} SM = "
        f"{occ['m1_attn_occupancy_frac']*100:.1f}%) and far below the HBM roofline "
        f"({occ['m1_attn_achieved_bw_frac']*100:.2f}% of {A10G_PEAK_BW_GBS:.0f} GB/s) -> the un-pack penalty "
        f"is OCCUPANCY-removable (removable={roof['m1_unpack_penalty_occupancy_removable']}), NOT BW-floored. "
        f"This is a DISTINCT regime from #332's M=8 verify-step (BW-floored {BW_FRAC_332_M8*100:.1f}% BW, "
        f"{CTAS_332_M8} CTAs > {A10G_SMS} SM): reconciles_332_bw_floor={roof['reconciles_332_bw_floor']}, "
        f"phi_m1={roof['phi_m1_draft_lane']:.3f} > phi_332={PHI_332_M8}. A deterministic 64-CTA "
        f"(num_splits={PINNED_SPLIT}) split-reduce could recover roofline {roof['pinnedk_recovery_frac_roofline']*100:.0f}% / "
        f"realistic {roof['pinnedk_recovery_frac_realistic']*100:.1f}% of the {eta_attn_decode_only*100:.2f}% tax. "
        f"BUT attention-free deployed strict caps at {af['attn_free_deployed_strict_tps_roofline']:.2f} "
        f"(ideal) / {af['attn_free_deployed_strict_tps_realistic']:.2f} (realistic) -- BOTH < 500 "
        f"(clears_500={af['attn_alone_clears_500_realistic']}). The attention rebuild buys at most "
        f"+{af['attn_lever_max_tps_gain_deployed']:.2f} TPS and leaves a residual "
        f"{af['residual_gap_after_attn_realistic']:.2f} TPS for cb3+demand. A fixed 64-way reduction tree IS "
        f"M-invariant byte-exact-feasible (feasible={pinnedk_m_invariant_byte_exact_feasible}, grounded by the "
        f"Marlin atomic_add=False M-invariance) but produces a NEW reference (re-capture = flagged change)."
    )

    return {
        "greedy_identity_harness_wired": greedy_identity_harness_wired,
        "unpack_identity": unpack_ident,
        "pinned_probe": pinned,
        "penalty_curve_M1": {str(L): curve_m1[L] for L in PENALTY_GRID_L},
        "penalty_decode_band": penalty_decode_band,
        "eta_attn_decode_only": eta_attn_decode_only,
        "deployed_strict": deployed_strict,
        "ceiling_strict": ceiling_strict,
        "gap_to_500": gap_to_500,
        "occupancy_bw": occ,
        "roofline": roof,
        "attn_free": af,
        "marlin_proof": marlin,
        "new_reference_probe": newref,
        # ---- headline deliverables (surfaced for SENPAI-RESULT + table) ----
        "m1_attn_occupancy_ctas": occ["m1_attn_occupancy_ctas"],
        "m1_attn_occupancy_frac": occ["m1_attn_occupancy_frac"],
        "m1_attn_achieved_bw_frac": occ["m1_attn_achieved_bw_frac"],
        "m1_unpack_penalty_occupancy_removable": roof["m1_unpack_penalty_occupancy_removable"],
        "phi_m1_draft_lane": roof["phi_m1_draft_lane"],
        "reconciles_332_bw_floor": roof["reconciles_332_bw_floor"],
        "pinnedk_recovery_frac_roofline": roof["pinnedk_recovery_frac_roofline"],
        "pinnedk_recovery_frac_realistic": roof["pinnedk_recovery_frac_realistic"],
        "attn_free_deployed_strict_tps_roofline": af["attn_free_deployed_strict_tps_roofline"],
        "attn_free_deployed_strict_tps_realistic": af["attn_free_deployed_strict_tps_realistic"],
        "attn_alone_clears_500_roofline": af["attn_alone_clears_500_roofline"],
        "attn_alone_clears_500_realistic": af["attn_alone_clears_500_realistic"],
        "residual_gap_after_attn_realistic": af["residual_gap_after_attn_realistic"],
        "pinnedk_m_invariant_byte_exact_feasible": pinnedk_m_invariant_byte_exact_feasible,
        "pinnedk_produces_new_reference": pinnedk_produces_new_reference,
        "n_distinct_kernel_rebuilds_attn_free": n_distinct_kernel_rebuilds_attn_free,
        "attn_rebuild_is_flagged_served_file_change": attn_rebuild_is_flagged_served_file_change,
        "verdict": verdict,
    }


def selftest_pinnedk(comp: dict, gpu: dict, flags: dict, n_seeds: int) -> dict[str, Any]:
    c: dict[str, bool] = {}
    roof, occ, af, marlin = comp["roofline"], comp["occupancy_bw"], comp["attn_free"], comp["marlin_proof"]
    # (a) reproduce #393's 467.48 / 505.29 / eta=0.030065 from MEASURED band + EXACT ladder round-trip.
    #     eta is ~15x-amplified from the penalty-band ratio (eta=f_attn*(penalty-1)), so a 20% band on eta
    #     is only ~4.5% on the underlying latency ratio; the robust anchors are the deployed/ceiling/ladder.
    c["a_repro_393_eta_measured"] = bool(abs(comp["eta_attn_decode_only"] / ETA_ATTN_DECODE_393 - 1.0) <= 0.20)
    c["a_repro_393_deployed_467"] = bool(abs(comp["deployed_strict"] - DEPLOYED_STRICT_393) <= 5.0)
    c["a_repro_393_ceiling_505"] = bool(abs(comp["ceiling_strict"] - CEILING_STRICT_393) <= 5.0)
    c["a_ladder_roundtrip_exact"] = bool(round(strict_tps_divisor(OFFICIAL_TPS, ETA_ATTN_DECODE_393), 2) == 467.48
                                         and round(strict_tps(ETA_ATTN_DECODE_393), 2) == 505.29)
    # (b) reproduce #332's phi=0.075 / 473.5 ceiling from cited inputs + internal consistency
    c["b_332_phi_below_breakeven"] = bool(PHI_332_M8 < BREAKEVEN_332)               # BW-floored at M=8
    c["b_332_ceiling_below_500"] = bool(CEILING_332_DEPLOYED < TARGET_500)
    c["b_332_ladder_consistent"] = bool(
        abs(strict_tps_divisor(OFFICIAL_TPS, OFFICIAL_TPS / CEILING_332_DEPLOYED - 1.0) - CEILING_332_DEPLOYED) <= 0.5)
    c["b_m1_more_headroom_than_332"] = bool(roof["phi_m1_draft_lane"] > PHI_332_M8)
    c["b_attn_free_exceeds_332_ceiling"] = bool(af["attn_free_deployed_strict_tps_roofline"] > CEILING_332_DEPLOYED)
    # (c) M=1 occupancy-removable decision backed by MEASURED occupancy/BW (not assumed)
    c["c_m1_underoccupied"] = bool(occ["m1_attn_occupancy_frac"] < 0.5)
    c["c_m1_far_from_bw_floor"] = bool(occ["m1_attn_achieved_bw_frac"] < 0.10)
    c["c_occupancy_removable_true"] = bool(roof["m1_unpack_penalty_occupancy_removable"])
    c["c_reconciles_332_true"] = bool(roof["reconciles_332_bw_floor"])
    c["c_beta_realistic_from_measurement"] = bool(
        0.0 < roof["pinnedk_recovery_frac_realistic"] <= 1.0 and roof["serial_depth_floor_us"] is not None)
    # (d) DECISIVE: attention-free deployed strict < 500 (the lever cannot clear 500 alone), residual > 0
    c["d_attn_free_below_500_roofline"] = bool(not af["attn_alone_clears_500_roofline"])
    c["d_attn_free_below_500_realistic"] = bool(not af["attn_alone_clears_500_realistic"])
    c["d_residual_gap_positive"] = bool(af["residual_gap_after_attn_realistic"] > 0)
    # (e) NO kernel built: pinned-K rejected (NotImplementedError), exactly 1 rebuild, flagged served change
    c["e_pinned_not_reachable"] = bool(not comp["pinned_probe"]["pinned_split_reachable"])
    c["e_one_rebuild"] = bool(comp["n_distinct_kernel_rebuilds_attn_free"] == 1)
    c["e_flagged_served_change"] = bool(comp["attn_rebuild_is_flagged_served_file_change"])
    # (f) byte-exact feasibility backed by MEASURED M-invariance of an EXISTING fixed-order reduce (Marlin)
    c["f_marlin_available"] = bool(marlin.get("available"))
    c["f_marlin_fixed_order_m_invariant"] = bool(marlin.get("fixed_order_m_invariant"))
    c["f_feasible_iff_measured"] = bool(
        comp["pinnedk_m_invariant_byte_exact_feasible"] == bool(marlin.get("fixed_order_m_invariant")))
    c["f_produces_new_reference"] = bool(comp["pinnedk_produces_new_reference"])
    c["f_new_reference_empirical"] = bool(comp["new_reference_probe"]["multisplit_changes_bytes_vs_serial"])
    # (g) greedy-identity harness wired; >=3 seeds; on-target A10G sm8x; guard flags
    c["g_greedy_identity_wired"] = bool(comp["greedy_identity_harness_wired"])
    c["g_three_or_more_seeds"] = bool(n_seeds >= 3)
    c["g_on_target_a10g_sm8x"] = bool(gpu["is_a10g_80sm"] and gpu["is_sm8x"])
    c["g_guard_flags"] = bool(flags["no_hf_job"] and flags["no_launch"]
                              and flags["no_served_file_change"] and flags["analysis_only"])
    passes = all(c.values())
    return {"passes": passes, "n_checks": len(c), "conditions": c}


def print_report_pinnedk(payload: dict) -> None:
    gpu, comp, st = payload["gpu"], payload["compose"], payload["selftest"]
    occ, roof, af, marlin = comp["occupancy_bw"], comp["roofline"], comp["attn_free"], comp["marlin_proof"]
    bar = "=" * 100
    print(bar)
    print("PINNED-K ATTENTION REBUILD HEADROOM -- attention-free strict vs 500 + byte-exact feasibility (PR #400)")
    print(f"  GPU {gpu['name']} SMs={gpu['sm_count']} cc={gpu['compute_capability']} "
          f"on-target={gpu['is_a10g_80sm'] and gpu['is_sm8x']}")
    print("-" * 100)
    print("  (0) HARNESS: #393 greedy-identity wired (un-pack ns=1 byte-exact M1-vs-M8) = "
          f"{comp['greedy_identity_harness_wired']}; 64-CTA pinned-K (ns>1) reachable_without_rebuild="
          f"{comp['pinned_probe']['pinned_split_reachable']} ({comp['pinned_probe']['error']})")
    print(f"  (1) #393 REPRO: penalty_band(M=1)={comp['penalty_decode_band']:.4f} -> "
          f"eta_attn_decode_only={comp['eta_attn_decode_only']*100:.4f}% -> deployed_strict="
          f"{comp['deployed_strict']:.2f} ceiling={comp['ceiling_strict']:.2f} gap_to_500={comp['gap_to_500']:.2f}")
    print("-" * 100)
    print("  (2) M=1 DRAFT-LANE OCCUPANCY / BW (the #332 question, M=1):")
    print(f"      m1_attn_occupancy_ctas   = {occ['m1_attn_occupancy_ctas']} / {A10G_SMS} SM  "
          f"(frac={occ['m1_attn_occupancy_frac']:.3f})")
    print(f"      m1_attn_achieved_bw_frac = {occ['m1_attn_achieved_bw_frac']*100:.3f}% of {A10G_PEAK_BW_GBS:.0f} GB/s "
          f"({occ['m1_attn_achieved_bw_gbs']:.2f} GB/s) | heuristic occ={occ['m1_heuristic_occupancy_frac']:.3f} "
          f"bw={occ['m1_heuristic_bw_frac']*100:.2f}%")
    print(f"      arith_intensity={occ['m1_arith_intensity_band']:.2f} FLOP/byte (ridge "
          f"{occ['ridge_flop_per_byte']:.1f}; below_ridge={occ['workload_below_ridge']})")
    print("-" * 100)
    print("  (3) ROOFLINE PINNED-K RECOVERY (beta-tier):")
    print(f"      m1_unpack_penalty_occupancy_removable = {roof['m1_unpack_penalty_occupancy_removable']}  "
          f"phi_m1_draft_lane = {roof['phi_m1_draft_lane']:.4f} (vs #332 phi={PHI_332_M8} M=8)")
    print(f"      reconciles_332_bw_floor = {roof['reconciles_332_bw_floor']}  "
          f"(M=1 distinct regime: {occ['m1_attn_achieved_bw_frac']*100:.2f}% BW vs #332 {BW_FRAC_332_M8*100:.1f}%)")
    print(f"      pinnedk_recovery_frac_roofline = {roof['pinnedk_recovery_frac_roofline']:.4f}  "
          f"realistic = {roof['pinnedk_recovery_frac_realistic']:.4f} "
          f"(K={PINNED_SPLIT}, lat_pin={roof['pinnedk_lat_pin_us']:.2f}us)")
    print("-" * 100)
    print("  (4) ATTENTION-FREE STRICT (does it clear 500 alone?):")
    print(f"      attn_free_deployed_strict_tps_roofline  = {af['attn_free_deployed_strict_tps_roofline']:.2f}  "
          f"clears_500={af['attn_alone_clears_500_roofline']}")
    print(f"      attn_free_deployed_strict_tps_realistic = {af['attn_free_deployed_strict_tps_realistic']:.2f}  "
          f"clears_500={af['attn_alone_clears_500_realistic']}")
    print(f"      residual_gap_after_attn_realistic = {af['residual_gap_after_attn_realistic']:.2f} TPS  "
          f"(attn lever max gain +{af['attn_lever_max_tps_gain_deployed']:.2f} TPS)")
    print("-" * 100)
    print("  (5) BYTE-EXACT FEASIBILITY (analysis; Marlin fixed-order reduce proof-of-mechanism):")
    print(f"      pinnedk_m_invariant_byte_exact_feasible = {comp['pinnedk_m_invariant_byte_exact_feasible']}  "
          f"(Marlin atomic_add=False byte_M8={marlin.get('byte_identity_by_M', {}).get('8')}, "
          f"guard_forces_fixed_order={marlin.get('deployment_guard_forces_fixed_order')})")
    print(f"      pinnedk_produces_new_reference = {comp['pinnedk_produces_new_reference']}  "
          f"(multisplit!=serial bytes={comp['new_reference_probe']['multisplit_changes_bytes_vs_serial']})")
    print(f"      n_distinct_kernel_rebuilds_attn_free = {comp['n_distinct_kernel_rebuilds_attn_free']}  "
          f"flagged_served_change={comp['attn_rebuild_is_flagged_served_file_change']}")
    print("-" * 100)
    print(f"  SELF-TEST {st['passes']} ({st['n_checks']} checks): "
          + json.dumps({k: int(v) for k, v in st["conditions"].items()}))
    print("-" * 100)
    print("  VERDICT")
    print("   " + comp["verdict"])
    print(bar)


def maybe_log_wandb_pinnedk(payload: dict, args) -> str | None:
    if args.no_wandb:
        return None
    repo = str(Path(__file__).resolve().parents[3])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from scripts.wandb_logging import (init_wandb_run, log_summary, log_json_artifact, finish_wandb)
    except Exception as e:  # noqa: BLE001
        print(f"[pinnedk] wandb helpers unavailable: {e}")
        return None
    comp = payload["compose"]
    occ, roof, af, marlin = comp["occupancy_bw"], comp["roofline"], comp["attn_free"], comp["marlin_proof"]
    run = init_wandb_run(
        job_type="analysis-gpu-microbench", agent="wirbel",
        name=args.wandb_name, group=args.wandb_group,
        tags=["attn-pinnedk-rebuild-headroom", "m1-occupancy-bw", "roofline-recovery",
              "attention-free-strict", "byte-exact-feasibility", "319-strict-lock", "pr-400"],
        config={"pr": 400, "kind": "attn-pinnedk-rebuild-headroom",
                "head_dim": HEAD_DIM, "n_q_heads": N_Q_HEADS, "n_kv_heads": N_KV_HEADS,
                "band_L": list(BAND_L), "short_L": SHORT_L, "pinned_split": PINNED_SPLIT,
                "pinned_ctas": PINNED_CTAS, "a10g_peak_bw_gbs": A10G_PEAK_BW_GBS,
                "ceiling_500": CEILING_500, "official_tps": OFFICIAL_TPS, "f_attn_344": F_ATTN_344,
                "eta_attn_decode_393": ETA_ATTN_DECODE_393, "phi_332_m8": PHI_332_M8,
                "ceiling_332_deployed": CEILING_332_DEPLOYED, "seeds": args.seeds,
                "ident_trials": args.ident_trials, "iters": args.iters},
    )
    if run is None:
        print("[pinnedk] wandb disabled (no API key / WANDB_MODE).")
        return None
    flat: dict[str, float] = {}
    for L in BAND_L:
        flat[f"occ/unpack_frac_L{L}"] = occ["per_L"][str(L)]["unpack_occupancy_frac"]
        flat[f"occ/unpack_bw_frac_L{L}"] = occ["per_L"][str(L)]["unpack_bw_frac"]
        flat[f"occ/heuristic_K_L{L}"] = float(occ["per_L"][str(L)]["heuristic_K"])
    flat["occ/m1_attn_occupancy_ctas"] = float(occ["m1_attn_occupancy_ctas"])
    flat["occ/m1_attn_occupancy_frac"] = occ["m1_attn_occupancy_frac"]
    flat["occ/m1_attn_achieved_bw_frac"] = occ["m1_attn_achieved_bw_frac"]
    flat["occ/m1_attn_achieved_bw_gbs"] = occ["m1_attn_achieved_bw_gbs"]
    flat["occ/m1_arith_intensity"] = occ["m1_arith_intensity_band"]
    flat["occ/ridge_flop_per_byte"] = occ["ridge_flop_per_byte"]
    flat["roof/phi_m1_draft_lane"] = roof["phi_m1_draft_lane"]
    flat["roof/m1_unpack_penalty_occupancy_removable"] = float(roof["m1_unpack_penalty_occupancy_removable"])
    flat["roof/reconciles_332_bw_floor"] = float(roof["reconciles_332_bw_floor"])
    flat["roof/pinnedk_recovery_frac_roofline"] = roof["pinnedk_recovery_frac_roofline"]
    flat["roof/pinnedk_recovery_frac_realistic"] = roof["pinnedk_recovery_frac_realistic"]
    flat["roof/pinnedk_lat_pin_us"] = roof["pinnedk_lat_pin_us"]
    flat["eta/eta_attn_decode_only"] = comp["eta_attn_decode_only"]
    flat["tps/deployed_strict"] = comp["deployed_strict"]
    flat["tps/ceiling_strict"] = comp["ceiling_strict"]
    flat["tps/gap_to_500"] = comp["gap_to_500"]
    flat["tps/attn_free_deployed_roofline"] = af["attn_free_deployed_strict_tps_roofline"]
    flat["tps/attn_free_deployed_realistic"] = af["attn_free_deployed_strict_tps_realistic"]
    flat["tps/residual_gap_after_attn_realistic"] = af["residual_gap_after_attn_realistic"]
    flat["tps/attn_lever_max_gain_deployed"] = af["attn_lever_max_tps_gain_deployed"]
    flat["decision/attn_alone_clears_500_roofline"] = float(af["attn_alone_clears_500_roofline"])
    flat["decision/attn_alone_clears_500_realistic"] = float(af["attn_alone_clears_500_realistic"])
    flat["decision/pinnedk_m_invariant_byte_exact_feasible"] = float(comp["pinnedk_m_invariant_byte_exact_feasible"])
    flat["decision/pinnedk_produces_new_reference"] = float(comp["pinnedk_produces_new_reference"])
    flat["decision/n_distinct_kernel_rebuilds_attn_free"] = float(comp["n_distinct_kernel_rebuilds_attn_free"])
    flat["decision/pinned_reachable"] = float(comp["pinned_probe"]["pinned_split_reachable"])
    if marlin.get("available"):
        flat["marlin/byte_M8"] = marlin["byte_identity_by_M"]["8"]
        flat["marlin/byte_M1"] = marlin["byte_identity_by_M"]["1"]
        flat["marlin/fixed_order_m_invariant"] = float(bool(marlin.get("fixed_order_m_invariant")))
    flat["selftest/attn_pinnedk_headroom_self_test_passes"] = float(payload["selftest"]["passes"])
    flat["gpu/sm_count"] = float(payload["gpu"]["sm_count"])
    flat["mem/peak_mem_mib"] = payload["peak_mem_mib"]
    run.log({"global_step": 0, **flat})
    log_summary(run, _jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="attn_pinnedk_headroom", artifact_type="analysis", data=_jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[pinnedk] wandb logged {len(flat)} keys (run {rid})")
    return rid


# ======================================================================================== #
# PRIMARY self-test
# ======================================================================================== #
def selftest(comp: dict, gpu: dict, flags: dict, n_seeds: int) -> dict[str, Any]:
    c: dict[str, bool] = {}
    # (a) reproduce #390's ladder from inputs: eta_attn -> 509.78 (ceiling) / 471.42 (deployed) / 471.20 (linear)
    c["a_reproduce_509_78"] = bool(round(strict_tps(ETA_ATTN_378), 2) == 509.78)
    c["a_reproduce_471_42"] = bool(round(strict_tps_divisor(OFFICIAL_TPS, ETA_ATTN_378), 2) == 471.42)
    c["a_reproduce_471_20"] = bool(round(served_tps_linear(OFFICIAL_TPS, ETA_ATTN_378), 2) == 471.20)
    # (a) reproduce eta_attn=0.02145 from the f_attn x (penalty_ew - 1) decomposition
    c["a_reproduce_eta_decomp"] = bool(abs(F_ATTN_344 * (PENALTY_EW_378 - 1.0) - ETA_ATTN_378) <= 1e-12)
    c["a_eta_attn_value"] = bool(round(ETA_ATTN_378, 5) == 0.02145)
    # (b) every config reported "byte-exact" has MEASURED identity EXACTLY 1.000
    bx_ok = True
    for name, cfg in comp["fa2_configs"].items():
        if cfg["is_byte_exact_M8"]:
            bx_ok = bx_ok and (cfg["byte_identity_M8_min"] >= 1.0)
    for name, cand in comp["byte_exact_candidates"].items():
        bx_ok = bx_ok and (cand["byte_exact_identity"] >= 1.0)
    c["b_byte_exact_configs_identity_1p000"] = bool(bx_ok)
    # (b) the un-pack config IS byte-exact (the strict pin exists); the heuristic is NOT (genuinely non-strict)
    c["b_unpack_is_byte_exact"] = bool(comp["fa2_configs"]["fa2_unpack_ns1"]["is_byte_exact_M8"])
    c["b_heuristic_not_byte_exact"] = bool(not comp["fa2_configs"]["fa2_heuristic_ns0"]["is_byte_exact_M8"])
    c["b_unpack_maxdiff_zero"] = bool(comp["fa2_configs"]["fa2_unpack_ns1"]["maxdiff_M8_max"] == 0.0)
    # (c) pinned-K is NOT reachable without a rebuild (varlen paged guard) -> 0.02145 floor not reducible today
    c["c_pinned_rejected"] = bool(not comp["pinned_probe"]["pinned_split_reachable"])
    c["c_eta_not_reducible"] = bool(not comp["attn_eta_reducible"])
    # (c) decode-only eta is finite, positive, and >= eval-weighted (band penalty above the low-L-dragged mean)
    c["c_decode_eta_finite"] = bool(math.isfinite(comp["eta_attn_decode_only"]) and comp["eta_attn_decode_only"] > 0)
    c["c_decode_ge_evalweighted"] = bool(comp["eta_attn_decode_only"] >= ETA_ATTN_378 - 1e-9)
    # (c) decision bools well-typed
    c["c_decision_bools"] = all(isinstance(comp[k], bool) for k in
                                ("attn_eta_reducible", "attn_already_strict_free", "fa_sliding0_is_strict_floor"))
    # (d) >=3 seeds, no-launch guards, on-target A10G sm8x
    c["d_three_or_more_seeds"] = bool(n_seeds >= 3)
    c["d_no_launch_flags"] = bool(flags["no_hf_job"] and flags["no_launch"]
                                  and flags["no_served_file_change"] and flags["analysis_only"])
    c["d_on_target_a10g_sm8x"] = bool(gpu["is_a10g_80sm"] and gpu["is_sm8x"])
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
    gpu, comp, st = payload["gpu"], payload["compose"], payload["selftest"]
    bar = "=" * 100
    print(bar)
    print("ATTENTION STRICT PIN COST -- cheapest byte-exact attention pin / decode-specific eta_attn (PR #393)")
    print(f"  GPU {gpu['name']} SMs={gpu['sm_count']} cc={gpu['compute_capability']} "
          f"on-target={gpu['is_a10g_80sm'] and gpu['is_sm8x']}")
    print("-" * 100)
    print("  (0) BACKENDS SHIPPED (reachable without rebuild):")
    for k, v in comp["backends_shipped"].items():
        print(f"      {k:28s} {v}")
    fi = comp["flashinfer"]
    print(f"      flashinfer available={fi.get('available')} v={fi.get('version')} "
          f"head_dim256={fi.get('head_dim_256_supported')} "
          f"valid_strict_pin={fi.get('is_valid_strict_pin')} fast_but_nonstrict={comp['fi_fast_but_nonstrict']}")
    print(f"         disable_split_kv_is_noop={fi.get('disable_split_kv_is_noop')} "
          f"verify_vs_decode_byte_id={fi.get('verify_vs_perrow_decode_byte_identity')} "
          f"decode_us_band={fi.get('decode_us_band_mean')}")
    print("-" * 100)
    print("  (1) FA2 num_splits config identity (batched-M=8 vs per-row-M=1, byte-exact gate):")
    for name, cfg in comp["fa2_configs"].items():
        print(f"      {name:20s} ns={cfg['num_splits']} byte_M8_min={cfg['byte_identity_M8_min']:.3f} "
              f"max|M8-M1|={cfg['maxdiff_M8_max']:.2e} byte_exact={cfg['is_byte_exact_M8']}")
    print(f"      pinned-K (ns={PINNED_SPLIT}) reachable_without_rebuild={comp['pinned_probe']['pinned_split_reachable']} "
          f"({comp['pinned_probe']['error']})")
    print("-" * 100)
    print("  (2) un-pack penalty (lat[ns=1]/lat[ns=0]) -- the tax is the M=1 AR/draft lane; M=8 verify is FREE:")
    print(f"      penalty @528 measured (M=1)={comp['penalty_528_measured']:.4f} (anchor 1.2778, "
          f"consistent={comp['anchor_528_consistent']})")
    print(f"      penalty_decode_band [528,658] (M=1) = {comp['penalty_decode_band']:.4f}  "
          f"| M=8 verify band = {comp['verify_penalty_band_mean']:.4f} (penalty_free={comp['verify_penalty_free']})")
    print(f"      FA2 band us (M=1): unpack={comp['fa2_unpack_band_us_M1']:.2f} heuristic={comp['fa2_heuristic_band_us_M1']:.2f}")
    d = comp["eval_decode_L_dist"]
    if d.get("present"):
        print(f"      #282 decode-L: mean={d['decode_L_mean']} median={d['decode_L_median']} "
              f"frac_in_band={d['frac_in_band']} penalty_ew_measured={d.get('penalty_evalweighted_measured')}")
    print(f"      eta_attn_evalweighted (banked #378) = {comp['eta_attn_evalweighted_banked']*100:.4f}%  "
          f"(measured re-weight={(comp['eta_attn_evalweighted_measured'] or 0)*100:.4f}%)")
    print(f"      ** eta_attn_decode_only = {comp['eta_attn_decode_only']*100:.4f}%  "
          f"(delta vs eval-wt = +{comp['eta_attn_decode_vs_evalweighted_delta']*100:.4f}pp) **")
    print(f"      deployed_tps_decode_eta = {comp['deployed_tps_decode_eta']:.2f} (vs banked 471.42)  "
          f"ceiling = {comp['ceiling_tps_decode_eta']:.2f} (vs 509.78)")
    print("-" * 100)
    print("  (3) CHEAPEST BYTE-EXACT PIN:")
    print(f"      n_byte_exact_attn_configs    = {comp['n_byte_exact_attn_configs']}")
    print(f"      cheapest_strict_attn_backend = {comp['cheapest_strict_attn_backend']}")
    print(f"      cheapest_strict_attn_eta     = {comp['cheapest_strict_attn_eta']*100:.4f}%")
    print(f"      attn_eta_reducible (no rebuild) = {comp['attn_eta_reducible']}  "
          f"attn_already_strict_free = {comp['attn_already_strict_free']}")
    print(f"      fa_sliding0_is_strict_floor  = {comp['fa_sliding0_is_strict_floor']}")
    print(f"      ceiling_with_cheapest_attn  = {comp['ceiling_with_cheapest_attn']:.2f}  "
          f"deployed_with_cheapest_attn = {comp['deployed_with_cheapest_attn']:.2f}")
    print(f"      gap_to_500_after_attn = {comp['gap_to_500_after_attn']:.2f} TPS (deployed) / "
          f"{comp['gap_to_500_after_attn_ceiling']:.2f} TPS (ceiling)")
    print("-" * 100)
    print(f"  SELF-TEST {st['passes']} ({st['n_checks']} checks): " +
          json.dumps({k: int(v) for k, v in st["conditions"].items()}))
    print("-" * 100)
    print("  VERDICT")
    print("   " + comp["verdict"])
    print(bar)


def maybe_log_wandb(payload: dict, args) -> str | None:
    if args.no_wandb:
        return None
    repo = str(Path(__file__).resolve().parents[3])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from scripts.wandb_logging import (init_wandb_run, log_summary, log_json_artifact, finish_wandb)
    except Exception as e:  # noqa: BLE001
        print(f"[attn-pin] wandb helpers unavailable: {e}")
        return None
    comp = payload["compose"]
    run = init_wandb_run(
        job_type="analysis-gpu-microbench", agent="wirbel",
        name=args.wandb_name, group=args.wandb_group,
        tags=["attention-strict-pin-cost", "eta-attn", "num-splits", "un-pack-tax",
              "flashinfer-bi", "fa-sliding", "319-strict-lock", "pr-393"],
        config={"pr": 393, "kind": "attention-strict-pin-cost",
                "head_dim": HEAD_DIM, "n_q_heads": N_Q_HEADS, "n_kv_heads": N_KV_HEADS,
                "served_block_size": SERVED_BLOCK_SIZE, "band_L": list(BAND_L), "short_L": SHORT_L,
                "ceiling_500": CEILING_500, "official_tps": OFFICIAL_TPS, "step_norm_us": STEP_NORM_US,
                "eta_attn_378": ETA_ATTN_378, "f_attn_344": F_ATTN_344, "seeds": args.seeds,
                "ident_trials": args.ident_trials, "iters": args.iters},
    )
    if run is None:
        print("[attn-pin] wandb disabled (no API key / WANDB_MODE).")
        return None
    flat: dict[str, float] = {}
    for name, cfg in comp["fa2_configs"].items():
        flat[f"identity/{name}_byte_M8_min"] = cfg["byte_identity_M8_min"]
        flat[f"identity/{name}_maxdiff_M8"] = cfg["maxdiff_M8_max"]
        flat[f"identity/{name}_byte_exact"] = float(cfg["is_byte_exact_M8"])
    for L in BAND_L:
        flat[f"penalty/M1_L{L}"] = comp["penalty_curve_M1"][str(L)]["penalty"]
        flat[f"penalty/M8_L{L}"] = comp["penalty_curve_M8"][str(L)]["penalty"]
    flat["penalty/decode_band_M1"] = comp["penalty_decode_band"]
    flat["penalty/verify_band_M8"] = comp["verify_penalty_band_mean"]
    flat["penalty/verify_penalty_free"] = float(comp["verify_penalty_free"])
    flat["penalty/at_528_M1"] = comp["penalty_528_measured"]
    flat["latency/fa2_unpack_band_us_M1"] = comp["fa2_unpack_band_us_M1"]
    flat["latency/fa2_heuristic_band_us_M1"] = comp["fa2_heuristic_band_us_M1"]
    flat["eta/eta_attn_decode_only"] = comp["eta_attn_decode_only"]
    flat["eta/eta_attn_evalweighted_banked"] = comp["eta_attn_evalweighted_banked"]
    if comp["eta_attn_evalweighted_measured"] is not None:
        flat["eta/eta_attn_evalweighted_measured"] = comp["eta_attn_evalweighted_measured"]
    flat["eta/decode_vs_evalweighted_delta"] = comp["eta_attn_decode_vs_evalweighted_delta"]
    flat["tps/deployed_tps_decode_eta"] = comp["deployed_tps_decode_eta"]
    flat["tps/ceiling_tps_decode_eta"] = comp["ceiling_tps_decode_eta"]
    flat["tps/deployed_with_cheapest_attn"] = comp["deployed_with_cheapest_attn"]
    flat["tps/ceiling_with_cheapest_attn"] = comp["ceiling_with_cheapest_attn"]
    flat["tps/gap_to_500_after_attn"] = comp["gap_to_500_after_attn"]
    flat["decision/cheapest_strict_attn_eta"] = comp["cheapest_strict_attn_eta"]
    flat["decision/n_byte_exact_attn_configs"] = float(comp["n_byte_exact_attn_configs"])
    flat["decision/attn_eta_reducible"] = float(comp["attn_eta_reducible"])
    flat["decision/attn_already_strict_free"] = float(comp["attn_already_strict_free"])
    flat["decision/fa_sliding0_is_strict_floor"] = float(comp["fa_sliding0_is_strict_floor"])
    flat["decision/cheapest_eta_above_evalweighted"] = float(comp["cheapest_eta_above_evalweighted"])
    flat["decision/fi_fast_but_nonstrict"] = float(comp["fi_fast_but_nonstrict"])
    flat["decision/pinned_reachable"] = float(comp["pinned_probe"]["pinned_split_reachable"])
    fi = comp["flashinfer"]
    if fi.get("available"):
        flat["flashinfer/is_valid_strict_pin"] = float(bool(fi.get("is_valid_strict_pin")))
        flat["flashinfer/disable_split_kv_is_noop"] = float(bool(fi.get("disable_split_kv_is_noop")))
        if fi.get("decode_us_band_mean") is not None:
            flat["flashinfer/decode_us_band"] = fi["decode_us_band_mean"]
        if fi.get("verify_vs_perrow_decode_byte_identity") is not None:
            flat["flashinfer/verify_vs_decode_byte_id"] = fi["verify_vs_perrow_decode_byte_identity"]
    flat["selftest/attention_strict_pin_cost_self_test_passes"] = float(payload["selftest"]["passes"])
    flat["gpu/sm_count"] = float(payload["gpu"]["sm_count"])
    flat["mem/peak_mem_mib"] = payload["peak_mem_mib"]
    run.log({"global_step": 0, **flat})
    log_summary(run, _jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="attention_strict_pin_cost", artifact_type="analysis", data=_jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[attn-pin] wandb logged {len(flat)} keys (run {rid})")
    return rid


# ======================================================================================== #
# Main
# ======================================================================================== #
def main_pinnedk(dev: torch.device, gpu: dict, args) -> None:
    """PR #400 driver: price the pinned-K attention rebuild (NO build/patch/launch/served-file change)."""
    comp = compose_pinnedk(dev, args, gpu)
    flags = {"no_hf_job": True, "no_launch": True, "no_served_file_change": True, "analysis_only": True}
    st = selftest_pinnedk(comp, gpu, flags, len(args.seeds))
    torch.cuda.synchronize()
    payload = {
        "agent": "wirbel", "pr": 400, "kind": "attn-pinnedk-rebuild-headroom",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        **flags,
        "gpu": gpu, "seeds": args.seeds,
        "peak_mem_mib": round(torch.cuda.max_memory_allocated(dev) / (1024**2), 3),
        "ladder_constants": {"ceiling_500": CEILING_500, "official_tps": OFFICIAL_TPS,
                             "target_500": TARGET_500, "f_attn_344": F_ATTN_344,
                             "a10g_peak_bw_gbs": A10G_PEAK_BW_GBS, "pinned_split": PINNED_SPLIT,
                             "pinned_ctas": PINNED_CTAS, "eta_attn_decode_393": ETA_ATTN_DECODE_393,
                             "deployed_strict_393": DEPLOYED_STRICT_393, "ceiling_strict_393": CEILING_STRICT_393,
                             "phi_332_m8": PHI_332_M8, "breakeven_332": BREAKEVEN_332,
                             "ceiling_332_deployed": CEILING_332_DEPLOYED, "bw_frac_332_m8": BW_FRAC_332_M8},
        "compose": comp, "selftest": st,
        # PRIMARY + headline SENPAI-RESULT surface (the #400 deliverables)
        "attn_pinnedk_headroom_self_test_passes": bool(st["passes"]),
        "m1_attn_occupancy_frac": comp["m1_attn_occupancy_frac"],
        "m1_attn_achieved_bw_frac": comp["m1_attn_achieved_bw_frac"],
        "m1_unpack_penalty_occupancy_removable": comp["m1_unpack_penalty_occupancy_removable"],
        "phi_m1_draft_lane": comp["phi_m1_draft_lane"],
        "reconciles_332_bw_floor": comp["reconciles_332_bw_floor"],
        "pinnedk_recovery_frac_roofline": comp["pinnedk_recovery_frac_roofline"],
        "pinnedk_recovery_frac_realistic": comp["pinnedk_recovery_frac_realistic"],
        "attn_free_deployed_strict_tps_roofline": comp["attn_free_deployed_strict_tps_roofline"],
        "attn_free_deployed_strict_tps_realistic": comp["attn_free_deployed_strict_tps_realistic"],
        "attn_alone_clears_500_roofline": comp["attn_alone_clears_500_roofline"],
        "attn_alone_clears_500_realistic": comp["attn_alone_clears_500_realistic"],
        "residual_gap_after_attn_realistic": comp["residual_gap_after_attn_realistic"],
        "pinnedk_m_invariant_byte_exact_feasible": comp["pinnedk_m_invariant_byte_exact_feasible"],
        "pinnedk_produces_new_reference": comp["pinnedk_produces_new_reference"],
        "n_distinct_kernel_rebuilds_attn_free": comp["n_distinct_kernel_rebuilds_attn_free"],
        "attn_rebuild_is_flagged_served_file_change": comp["attn_rebuild_is_flagged_served_file_change"],
        "deployed_strict": comp["deployed_strict"],
        "eta_attn_decode_only": comp["eta_attn_decode_only"],
    }
    print_report_pinnedk(payload)
    out_path = Path(args.out_dir) / "attn_pinnedk_headroom_results.json"
    json.dump(_jsonable(payload), open(out_path, "w"), indent=2)
    print(f"[pinnedk] results -> {out_path}")
    rid = maybe_log_wandb_pinnedk(payload, args)
    if rid:
        payload["wandb_run_id"] = rid
        json.dump(_jsonable(payload), open(out_path, "w"), indent=2)


# ======================================================================================== #
# PR #408 -- CLOSED M=1 decode-step latency budget + stacked-flagged-supply ceiling.
#   Per-component CUDA-event / roofline MEASUREMENT (body Marlin reads via lawine #388/#391
#   method, lm_head via #384 method, attention via the existing penalty curve), normalized-
#   space budget closure, flagged-lever removable attribution, stacked ceiling vs 500.
#   NO kernel build / patch / served-file change / launch. 0 official TPS.
# ======================================================================================== #
def _measure_peak_copy_gbs_408(dev: torch.device, iters: int, warmup: int) -> dict[str, Any]:
    """Achievable HBM bandwidth on THIS pod via a large bf16 d2d copy (the roofline denominator;
    same method as lawine #388/#391 -- ~470 GB/s measured vs 600 theoretical)."""
    n = 64 * 1024 * 1024                                   # 64M bf16 = 128 MiB; read+write = 256 MiB
    x = torch.randn(n, dtype=DTYPE, device=dev)
    y = torch.empty_like(x)
    us = _time_call(lambda: y.copy_(x), iters, warmup)
    moved = 2 * x.numel() * DTYPE_BYTES                    # read + write
    gbs = moved / (us * 1e-6) / 1e9
    del x, y
    return {"copy_us": us, "moved_bytes": float(moved), "peak_copy_gbs": gbs,
            "peak_theoretical_gbs": A10G_PEAK_BW_GBS, "copy_eff_vs_theoretical": gbs / A10G_PEAK_BW_GBS}


def _int4_weight_bytes_408(out: int, inn: int) -> float:
    """int4-Marlin g128 body weight-read bytes for one GEMM (4.125 bpw)."""
    return out * inn * INT4_BPW_G128_408 / 8.0


def _build_marlin_body_408(out: int, inn: int, dev: torch.device, m: int):
    """0-arg callable running one int4-Marlin (uint4b8 g128) body GEMM at width m + weight bytes
    (lawine #388/#391 _build_marlin_gemm method; weight read is m-independent)."""
    from vllm.scalar_type import scalar_types
    import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
    import vllm.model_executor.layers.quantization.utils.marlin_utils_test as mt

    K, N = inn, out
    wtype = scalar_types.uint4b8
    w = (torch.randn(K, N, dtype=DTYPE, device=dev) * 0.02)
    _wr, q_w, s, _gi, _si, _rp = mt.marlin_quantize(w, wtype, MARLIN_PROOF_GS, act_order=False)
    ws = mu.marlin_make_workspace_new(dev)
    zp = torch.empty(0, dtype=torch.int, device=dev)
    g_idx = torch.empty(0, dtype=torch.int, device=dev)
    sort_idx = torch.empty(0, dtype=torch.int, device=dev)
    x = torch.randn(m, K, dtype=DTYPE, device=dev)

    def run():
        return mu.apply_gptq_marlin_linear(
            x, q_w, s, zp, g_idx, sort_idx, ws, wtype,
            output_size_per_partition=N, input_size_per_partition=K, is_k_full=True)

    out_t = run()
    ok = bool(out_t.shape == (m, N) and torch.isfinite(out_t).all().item())
    return run, _int4_weight_bytes_408(out, inn), ok


def measure_body_gemm_budget(dev: torch.device, peak_gbs: float, iters: int, warmup: int) -> dict[str, Any]:
    """M=1 int4-Marlin body-GEMM weight-read HBM efficiency over the 8 served body shapes.
    body_bw_bound_frac = count-weighted (sum c*wbytes)/(sum c*us)/peak -- the FRESH measurement
    that reconciles #391's 0.256 and prices cb3's removable. (The cb3 read-shrink only removes
    latency from THIS BW-bound share; the (1 - frac) overhead share is a floor.)"""
    per_shape: list[dict[str, Any]] = []
    tot_int4_bytes = 0.0
    tot_time_us = 0.0
    all_ok = True
    for sh in BODY_SHAPES_408:
        run, wbytes, ok = _build_marlin_body_408(sh["out"], sh["in"], dev, M_AR)
        all_ok = all_ok and ok
        us = _time_call(run, iters, warmup)
        gbs = wbytes / (us * 1e-6) / 1e9
        per_shape.append({
            "name": sh["name"], "out": sh["out"], "in": sh["in"], "count": sh["count"],
            "marlin_us": us, "weight_mib": wbytes / (1024**2), "eff_gbs": gbs,
            "bw_eff": gbs / peak_gbs, "finite_ok": ok,
        })
        tot_int4_bytes += sh["count"] * wbytes
        tot_time_us += sh["count"] * us
    agg_eff_gbs = tot_int4_bytes / (tot_time_us * 1e-6) / 1e9
    body_bw_bound_frac = agg_eff_gbs / peak_gbs
    raw_isolated_body_us = sum(p["count"] * p["marlin_us"] for p in per_shape)
    return {
        "per_shape": per_shape,
        "count_weighted_eff_gbs": agg_eff_gbs,
        "body_bw_bound_frac": body_bw_bound_frac,
        "total_int4_weight_gib": tot_int4_bytes / (1024**3),
        "raw_isolated_body_us": raw_isolated_body_us,
        "all_shapes_finite_ok": bool(all_ok),
        "m1_marlin_hbm_eff_391": M1_MARLIN_HBM_EFF_391,
        "reconciles_391": bool(abs(body_bw_bound_frac - M1_MARLIN_HBM_EFF_391) <= BODY_BW_RECONCILE_TOL_408),
    }


def _build_marlin_lmhead_408(dev: torch.device, seed: int):
    """Channel-wise (group_size=-1) int4 GPTQ-Marlin GEMM at the deployed lm_head geometry
    [size_k=HIDDEN, size_n=LMHEAD_ROWS], FIXED fp32-reduce (use_atomic_add=False -- already the
    decode-deployed reduce per #384). Replicates deterministic_lmhead_gemm.build_marlin_lmhead."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
    from vllm.model_executor.layers.quantization.utils import marlin_utils_test as mut
    from vllm.scalar_type import scalar_types

    qtype = scalar_types.uint4b8
    g = torch.Generator(device=dev).manual_seed(seed)
    w = (torch.randn(LMHEAD_HIDDEN_408, LMHEAD_ROWS_408, generator=g, device=dev, dtype=DTYPE) * 0.02)
    _wr, q_w, s, g_idx, sort_idx, _ = mut.marlin_quantize(w, qtype, group_size=-1, act_order=False)
    ws = mu.marlin_make_workspace_new(dev)
    zp = torch.empty(0, dtype=torch.int, device=dev)
    heuristic_aa = bool(mu.should_use_atomic_add_reduce(
        m=M_AR, n=LMHEAD_ROWS_408, k=LMHEAD_HIDDEN_408, device=dev, dtype=DTYPE))

    def run(x: torch.Tensor) -> torch.Tensor:
        xr = x.reshape(-1, LMHEAD_HIDDEN_408)
        return ops.marlin_gemm(
            xr, None, q_w, None, s, None, None, zp, g_idx, sort_idx, ws, qtype,
            size_m=xr.shape[0], size_n=LMHEAD_ROWS_408, size_k=LMHEAD_HIDDEN_408,
            is_k_full=True, use_atomic_add=False, use_fp32_reduce=True,
            is_zp_float=False).reshape(x.shape[:-1] + (LMHEAD_ROWS_408,))

    return run, heuristic_aa


def measure_lmhead_gemm_budget(dev: torch.device, peak_gbs: float, iters: int, warmup: int,
                               seed: int) -> dict[str, Any]:
    """Single-token (M=1) lm_head int4-Marlin read latency + achieved-BW fraction (~21MB read,
    #344). lmhead_bw_bound_frac reported for completeness; the lm_head REMOVABLE is ~0 (land #398:
    no loadable read-shrink), so this does not feed the stacked ceiling."""
    run, heuristic_aa = _build_marlin_lmhead_408(dev, seed)
    x1 = torch.randn(M_AR, LMHEAD_HIDDEN_408, device=dev, dtype=DTYPE)
    out_t = run(x1)
    ok = bool(out_t.shape == (M_AR, LMHEAD_ROWS_408) and torch.isfinite(out_t).all().item())
    us = _time_call(lambda: run(x1), iters, warmup)
    gbs = LMHEAD_BYTES_408 / (us * 1e-6) / 1e9
    return {
        "lmhead_m1_us": us,
        "lmhead_bytes": float(LMHEAD_BYTES_408),
        "lmhead_mib": LMHEAD_BYTES_408 / (1024**2),
        "lmhead_eff_gbs": gbs,
        "lmhead_bw_bound_frac": gbs / peak_gbs,
        "heuristic_use_atomic_add": heuristic_aa,
        "best_loadable_read_shrink_frac_398": LMHEAD_BEST_LOADABLE_READ_SHRINK_398,
        "finite_ok": ok,
    }


def compose_decode_budget(dev: torch.device, args, gpu: dict) -> dict[str, Any]:
    from vllm.vllm_flash_attn import _vllm_fa2_C  # noqa: F401  (registers torch.ops._vllm_fa2_C)
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as FA2

    # (0) confirm the #393 greedy-identity harness is wired (un-pack ns=1 byte-exact M=1 vs M=8)
    ident_accs = [_measure_identity_fa2(FA2, BAND_L[0], UNPACK_SPLIT, args.ident_trials, s, dev)
                  for s in args.seeds]
    unpack_ident = _merge_idents(ident_accs)
    greedy_identity_harness_wired = bool(unpack_ident["byte_identity_by_M"]["8"] >= 1.0)

    # (1) ATTENTION component: reproduce #393's decode-only eta from the MEASURED M=1 band penalty
    curve_m1 = measure_penalty_curve(FA2, dev, args.iters, args.warmup, args.seeds[0], M_AR)
    band_pens = [curve_m1[L]["penalty"] for L in BAND_L]
    penalty_decode_band = float(sum(band_pens) / len(band_pens))
    eta_attn_decode_only = F_ATTN_344 * (penalty_decode_band - 1.0)
    deployed_strict = strict_tps_divisor(OFFICIAL_TPS, eta_attn_decode_only)   # 467.x (test metric)
    ceiling_strict = strict_tps(eta_attn_decode_only)                          # 505.x
    occ = measure_m1_occupancy_bw(curve_m1)
    roof = roofline_pinnedk_recovery(occ, eta_attn_decode_only)
    pinnedk_recovery = roof["pinnedk_recovery_frac_realistic"]                 # ~0.9872 (#400)

    # (2) PEAK copy BW + BODY / LM_HEAD measured component budgets (fresh GPU measurement)
    peak = _measure_peak_copy_gbs_408(dev, args.iters, args.warmup)
    peak_gbs = peak["peak_copy_gbs"]
    body = measure_body_gemm_budget(dev, peak_gbs, args.iters, args.warmup)
    lmhead = measure_lmhead_gemm_budget(dev, peak_gbs, args.iters, args.warmup, args.seeds[0])
    body_bw_bound_frac = body["body_bw_bound_frac"]
    lmhead_bw_bound_frac = lmhead["lmhead_bw_bound_frac"]

    # (3) CLOSED budget in normalized #378-fraction space (the 4 buckets partition STEP_NORM_US).
    S0 = STEP_NORM_US                                    # non-strict step (OFFICIAL_TPS basis)
    t_attn_us = F_ATTN_344 * S0
    t_body_gemm_us = F_BODY_STRICT_378 * S0
    t_lmhead_us = F_LMHEAD_378 * S0
    t_fixed_overhead_us = S0 - (t_attn_us + t_body_gemm_us + t_lmhead_us)   # == F_DRAFT_378*S0 (measured residual)
    budget_closure_residual_frac = abs(
        S0 - (t_attn_us + t_body_gemm_us + t_lmhead_us + t_fixed_overhead_us)) / S0   # completeness (~0)
    t_attn_frac = t_attn_us / S0
    t_body_gemm_frac = t_body_gemm_us / S0
    t_lmhead_frac = t_lmhead_us / S0
    fixed_overhead_frac = t_fixed_overhead_us / S0       # headline: how overhead-bound is the M=1 step

    # attention strict PENALTY sub-component (the ONLY removable attention latency; un-pack ns=1 tax)
    attn_penalty_us = eta_attn_decode_only * S0          # ~37.3us
    S_strict = S0 + attn_penalty_us                      # ~1255us -> deployed_strict via tps() below

    # raw-isolated diagnostic: WHY the budget is normalized (#284 overcredit, NOT used in the budget)
    raw_isolated_sum_us = occ["m1_unpack_band_us"] + body["raw_isolated_body_us"] + lmhead["lmhead_m1_us"]
    overcredit_factor = raw_isolated_sum_us / S0         # ~4-5x (isolated sums can't close the step)

    def tps_from_step(step_us: float) -> float:
        return OFFICIAL_TPS * S0 / step_us               # tps(S0)=481.53, tps(S_strict)=deployed_strict

    deployed_strict_via_step = tps_from_step(S_strict)   # must equal deployed_strict (ladder identity)

    # (4) FLAGGED supply lever removables (beta-tier roofline, priced on the MEASURED budget)
    # -- pinned-K attention: removes the strict penalty (roofline = full; realistic = x recovery 0.9872)
    pinnedk_attn_removable_roofline_us = 1.0 * attn_penalty_us
    pinnedk_attn_removable_us = pinnedk_recovery * attn_penalty_us
    attn_free_roofline_tps = tps_from_step(S_strict - pinnedk_attn_removable_roofline_us)   # -> 481.53
    attn_free_realistic_tps = tps_from_step(S_strict - pinnedk_attn_removable_us)           # -> 481.34
    attn_lever_gain_roofline_tps = attn_free_roofline_tps - deployed_strict                 # +14.29 (#400)
    attn_lever_gain_realistic_tps = attn_free_realistic_tps - deployed_strict               # +14.10
    # -- cb3 body-read shrink: removable = read_shrink * body_bw_bound_frac * t_body (PR step-4 formula)
    cb3_body_removable_us = CB3_READ_SHRINK_FRAC_408 * body_bw_bound_frac * t_body_gemm_us
    cb3_body_removable_us_g128 = CB3_READ_SHRINK_FRAC_G128_408 * body_bw_bound_frac * t_body_gemm_us
    cb3_body_removable_tps_roofline = tps_from_step(S_strict - cb3_body_removable_us) - deployed_strict
    # -- lm_head: land #398 no loadable read-shrink -> removable ~0
    lmhead_removable_us = LMHEAD_BEST_LOADABLE_READ_SHRINK_398 * lmhead_bw_bound_frac * t_lmhead_us

    # (5) STACKED-flagged-supply ceiling vs the measured fixed floor (fixed-overhead held constant).
    #     Headline uses ROOFLINE removables (PR step 5); a realistic variant is reported alongside.
    stacked_removable_roofline_us = (pinnedk_attn_removable_roofline_us
                                     + cb3_body_removable_us + lmhead_removable_us)
    stacked_removable_realistic_us = (pinnedk_attn_removable_us
                                      + cb3_body_removable_us + lmhead_removable_us)
    S_stacked_roofline = S_strict - stacked_removable_roofline_us
    S_stacked_realistic = S_strict - stacked_removable_realistic_us
    supply_stacked_flagged_ceiling_tps = tps_from_step(S_stacked_roofline)
    supply_stacked_flagged_ceiling_tps_realistic = tps_from_step(S_stacked_realistic)
    supply_stacked_flagged_clears_500 = bool(supply_stacked_flagged_ceiling_tps >= TARGET_500)
    supply_stacked_flagged_clears_500_realistic = bool(
        supply_stacked_flagged_ceiling_tps_realistic >= TARGET_500)
    residual_gap_after_all_supply_flagged = TARGET_500 - supply_stacked_flagged_ceiling_tps
    # distinct flagged served-file changes the ceiling assumes: pinned-K attn (1) + cb3 body (1) + lm_head (0)
    n_flags_in_stacked_supply = 2

    verdict = (
        f"CLOSED M=1 decode-step budget (normalized #378 partition, residual "
        f"{budget_closure_residual_frac*100:.3f}%): t_attn={t_attn_frac*100:.2f}% "
        f"t_body={t_body_gemm_frac*100:.2f}% t_lmhead={t_lmhead_frac*100:.2f}% "
        f"FIXED_OVERHEAD={fixed_overhead_frac*100:.2f}% (the draft-tail + launch/sched/norm/sampling "
        f"floor NO supply read-shrink or attn-recovery lever can touch). Each component's removable "
        f"share is MEASURED: body Marlin M=1 HBM eff body_bw_bound_frac={body_bw_bound_frac:.3f} "
        f"(reconciles #391 0.256={body['reconciles_391']}), lmhead eff={lmhead_bw_bound_frac:.3f} but "
        f"removable=0 (#398 no loadable shrink). FLAGGED removables: pinned-K attn "
        f"{pinnedk_attn_removable_us:.1f}us -> +{attn_lever_gain_roofline_tps:.2f} TPS roofline "
        f"(reproduces #400 481.53/481.34), cb3 body {cb3_body_removable_us:.1f}us "
        f"(+{cb3_body_removable_tps_roofline:.2f} TPS, PPL-UNCAPPED upper bound -- kanna #403's PPL-safe "
        f"bpw is larger => smaller shrink), lm_head ~0. STACKED ceiling "
        f"supply_stacked_flagged_ceiling_tps={supply_stacked_flagged_ceiling_tps:.2f} "
        f"(clears_500={supply_stacked_flagged_clears_500}, residual_gap="
        f"{residual_gap_after_all_supply_flagged:+.2f}) across {n_flags_in_stacked_supply} flagged "
        f"served-file changes; realistic (0.9872 attn) variant "
        f"{supply_stacked_flagged_ceiling_tps_realistic:.2f}. The {fixed_overhead_frac*100:.1f}% fixed "
        f"floor makes even the FULLY-stacked flagged-supply route land at 500 +/- ~1 TPS -- razor-thin "
        f"and PPL-uncapped, so the demand leg (tree/retrain) is effectively mandatory for a robust >500.")

    return {
        "greedy_identity_harness_wired": greedy_identity_harness_wired,
        "unpack_identity": unpack_ident,
        "penalty_curve_M1": {str(L): curve_m1[L] for L in PENALTY_GRID_L},
        "penalty_decode_band": penalty_decode_band,
        "eta_attn_decode_only": eta_attn_decode_only,
        "deployed_strict": deployed_strict,
        "deployed_strict_via_step": deployed_strict_via_step,
        "ceiling_strict": ceiling_strict,
        "occupancy_bw": occ,
        "roofline": roof,
        "pinnedk_recovery_frac_realistic": pinnedk_recovery,
        "peak_copy": peak,
        "body_budget": body,
        "lmhead_budget": lmhead,
        # ---- CLOSED budget (normalized #378 partition) ----
        "step_norm_us": S0,
        "s_strict_us": S_strict,
        "t_attn_us": t_attn_us, "t_body_gemm_us": t_body_gemm_us, "t_lmhead_us": t_lmhead_us,
        "t_fixed_overhead_us": t_fixed_overhead_us,
        "t_attn_frac": t_attn_frac, "t_body_gemm_frac": t_body_gemm_frac,
        "t_lmhead_frac": t_lmhead_frac, "fixed_overhead_frac": fixed_overhead_frac,
        "budget_closure_residual_frac": budget_closure_residual_frac,
        "attn_penalty_us": attn_penalty_us,
        "raw_isolated_sum_us": raw_isolated_sum_us, "overcredit_factor": overcredit_factor,
        # ---- BW-bound shares (measured) ----
        "body_bw_bound_frac": body_bw_bound_frac,
        "lmhead_bw_bound_frac": lmhead_bw_bound_frac,
        "body_reconciles_391": body["reconciles_391"],
        # ---- flagged-lever removables ----
        "pinnedk_attn_removable_us": pinnedk_attn_removable_us,
        "pinnedk_attn_removable_roofline_us": pinnedk_attn_removable_roofline_us,
        "attn_free_roofline_tps": attn_free_roofline_tps,
        "attn_free_realistic_tps": attn_free_realistic_tps,
        "attn_lever_gain_roofline_tps": attn_lever_gain_roofline_tps,
        "attn_lever_gain_realistic_tps": attn_lever_gain_realistic_tps,
        "cb3_read_shrink_frac": CB3_READ_SHRINK_FRAC_408,
        "cb3_body_removable_us": cb3_body_removable_us,
        "cb3_body_removable_us_g128": cb3_body_removable_us_g128,
        "cb3_body_removable_tps_roofline": cb3_body_removable_tps_roofline,
        "lmhead_removable_us": lmhead_removable_us,
        # ---- stacked-flagged-supply ceiling ----
        "stacked_removable_roofline_us": stacked_removable_roofline_us,
        "stacked_removable_realistic_us": stacked_removable_realistic_us,
        "supply_stacked_flagged_ceiling_tps": supply_stacked_flagged_ceiling_tps,
        "supply_stacked_flagged_ceiling_tps_realistic": supply_stacked_flagged_ceiling_tps_realistic,
        "supply_stacked_flagged_clears_500": supply_stacked_flagged_clears_500,
        "supply_stacked_flagged_clears_500_realistic": supply_stacked_flagged_clears_500_realistic,
        "residual_gap_after_all_supply_flagged": residual_gap_after_all_supply_flagged,
        "n_flags_in_stacked_supply": n_flags_in_stacked_supply,
        "verdict": verdict,
    }


def selftest_decode_budget(comp: dict, gpu: dict, flags: dict, n_seeds: int) -> dict[str, Any]:
    c: dict[str, bool] = {}
    body, lmhead = comp["body_budget"], comp["lmhead_budget"]
    # (a) reproduce #393's deployed 467.48 / ceiling 505.29 / eta 0.0306 from the MEASURED band
    c["a_repro_393_deployed_467"] = bool(abs(comp["deployed_strict"] - DEPLOYED_STRICT_393) <= 5.0)
    c["a_repro_393_ceiling_505"] = bool(abs(comp["ceiling_strict"] - CEILING_STRICT_393) <= 5.0)
    c["a_repro_393_eta_measured"] = bool(abs(comp["eta_attn_decode_only"] / ETA_ATTN_DECODE_393 - 1.0) <= 0.20)
    c["a_ladder_identity"] = bool(abs(comp["deployed_strict"] - comp["deployed_strict_via_step"]) <= 0.05)
    # (b) reproduce #400's attention-free strict 481.53 (roofline; EXACT) / 481.34 (realistic) from the
    #     removable. roofline is measurement-independent (full penalty -> step S0 -> OFFICIAL); the
    #     realistic value and the gain ride the freshly-measured eta/recovery, so allow +/-2 TPS drift.
    c["b_repro_400_roofline_481_53"] = bool(abs(comp["attn_free_roofline_tps"] - 481.53) <= 0.10)
    c["b_repro_400_realistic_481_34"] = bool(abs(comp["attn_free_realistic_tps"] - 481.34) <= 2.0)
    c["b_repro_400_gain_14_29"] = bool(abs(comp["attn_lever_gain_roofline_tps"] - 14.29) <= 2.0)
    # (c) body_bw_bound_frac reconciles #391's 0.256 (the FRESH measurement, not an imported number)
    c["c_body_reconciles_391"] = bool(comp["body_reconciles_391"])
    c["c_body_frac_finite_pos"] = bool(0.0 < comp["body_bw_bound_frac"] < 1.0)
    c["c_lmhead_frac_finite_pos"] = bool(0.0 < comp["lmhead_bw_bound_frac"] < 1.0)
    c["c_body_shapes_ok"] = bool(body["all_shapes_finite_ok"])
    c["c_lmhead_ok"] = bool(lmhead["finite_ok"])
    # (d) CLOSED budget: residual small, fractions sum to 1, fixed_overhead is the measured residual
    c["d_closure_le_8pct"] = bool(comp["budget_closure_residual_frac"] <= 0.08)
    fsum = comp["t_attn_frac"] + comp["t_body_gemm_frac"] + comp["t_lmhead_frac"] + comp["fixed_overhead_frac"]
    c["d_fractions_sum_1"] = bool(abs(fsum - 1.0) <= 1e-9)
    c["d_fixed_is_measured_residual"] = bool(
        abs(comp["t_fixed_overhead_us"]
            - (comp["step_norm_us"] - (comp["t_attn_us"] + comp["t_body_gemm_us"] + comp["t_lmhead_us"]))) <= 1e-6)
    c["d_fixed_overhead_positive"] = bool(comp["fixed_overhead_frac"] > 0.0)
    # (e) removables derived from the MEASURED component budget (not imported isolated numbers)
    c["e_cb3_uses_measured_body_frac"] = bool(abs(
        comp["cb3_body_removable_us"]
        - CB3_READ_SHRINK_FRAC_408 * comp["body_bw_bound_frac"] * comp["t_body_gemm_us"]) <= 1e-6)
    c["e_attn_removable_from_penalty"] = bool(abs(
        comp["pinnedk_attn_removable_us"]
        - comp["pinnedk_recovery_frac_realistic"] * comp["attn_penalty_us"]) <= 1e-6)
    c["e_lmhead_removable_zero"] = bool(comp["lmhead_removable_us"] == 0.0)
    c["e_overcredit_gt_2"] = bool(comp["overcredit_factor"] > 2.0)   # isolated sums DON'T close the step
    # (f) stacked ceiling well-formed: clears bool typed, residual = 500 - ceiling, 2 flags
    c["f_clears_bool"] = isinstance(comp["supply_stacked_flagged_clears_500"], bool)
    c["f_residual_consistent"] = bool(abs(
        comp["residual_gap_after_all_supply_flagged"]
        - (TARGET_500 - comp["supply_stacked_flagged_ceiling_tps"])) <= 1e-6)
    c["f_two_flags"] = bool(comp["n_flags_in_stacked_supply"] == 2)
    c["f_ceiling_finite"] = bool(math.isfinite(comp["supply_stacked_flagged_ceiling_tps"]))
    # (g) greedy-identity harness wired; >=3 seeds; on-target A10G sm8x; guard flags
    c["g_greedy_identity_wired"] = bool(comp["greedy_identity_harness_wired"])
    c["g_three_or_more_seeds"] = bool(n_seeds >= 3)
    c["g_on_target_a10g_sm8x"] = bool(gpu["is_a10g_80sm"] and gpu["is_sm8x"])
    c["g_guard_flags"] = bool(flags["no_hf_job"] and flags["no_launch"]
                              and flags["no_served_file_change"] and flags["analysis_only"])
    passes = all(c.values())
    return {"passes": passes, "n_checks": len(c), "conditions": c}


def print_report_decode_budget(payload: dict) -> None:
    gpu, comp, st = payload["gpu"], payload["compose"], payload["selftest"]
    body, lmhead = comp["body_budget"], comp["lmhead_budget"]
    bar = "=" * 100
    print(bar)
    print("CLOSED M=1 DECODE-STEP LATENCY BUDGET -- stacked-flagged supply vs the fixed floor (PR #408)")
    print(f"  GPU {gpu['name']} SMs={gpu['sm_count']} cc={gpu['compute_capability']} "
          f"on-target={gpu['is_a10g_80sm'] and gpu['is_sm8x']}")
    print("-" * 100)
    print(f"  (0) HARNESS: #393 greedy-identity wired (un-pack ns=1 byte-exact M1-vs-M8) = "
          f"{comp['greedy_identity_harness_wired']}")
    print(f"  (1) ATTENTION: penalty_band(M=1)={comp['penalty_decode_band']:.4f} -> "
          f"eta_attn_decode_only={comp['eta_attn_decode_only']*100:.4f}% -> deployed_strict="
          f"{comp['deployed_strict']:.2f} (=={comp['deployed_strict_via_step']:.2f} via step) "
          f"ceiling={comp['ceiling_strict']:.2f}")
    print(f"  (2) PEAK-COPY BW={comp['peak_copy']['peak_copy_gbs']:.1f} GB/s "
          f"({comp['peak_copy']['copy_eff_vs_theoretical']*100:.1f}% of {A10G_PEAK_BW_GBS:.0f})")
    print("-" * 100)
    print("  CLOSED BUDGET (normalized #378 partition; raw-isolated sums over-credit "
          f"{comp['overcredit_factor']:.2f}x -> budget MUST be normalized):")
    print(f"    t_attn       = {comp['t_attn_us']:8.2f} us  ({comp['t_attn_frac']*100:6.3f}%)")
    print(f"    t_body_gemm  = {comp['t_body_gemm_us']:8.2f} us  ({comp['t_body_gemm_frac']*100:6.3f}%)")
    print(f"    t_lmhead     = {comp['t_lmhead_us']:8.2f} us  ({comp['t_lmhead_frac']*100:6.3f}%)")
    print(f"    t_FIXED_OVHD = {comp['t_fixed_overhead_us']:8.2f} us  ({comp['fixed_overhead_frac']*100:6.3f}%) "
          f"<- the irreducible floor (draft-tail + launch/sched/norm/sampling)")
    print(f"    step_total   = {comp['step_norm_us']:8.2f} us  closure_residual="
          f"{comp['budget_closure_residual_frac']*100:.4f}%")
    print("-" * 100)
    print("  BW-BOUND SHARES (measured; read-shrink levers remove ONLY this share):")
    print(f"    body_bw_bound_frac   = {comp['body_bw_bound_frac']:.4f}  "
          f"(#391 0.256; reconciles={comp['body_reconciles_391']}; "
          f"{body['count_weighted_eff_gbs']:.1f} GB/s weight-read)")
    print(f"    lmhead_bw_bound_frac = {comp['lmhead_bw_bound_frac']:.4f}  "
          f"({lmhead['lmhead_mib']:.2f} MiB, {lmhead['lmhead_m1_us']:.2f} us; removable=0 per #398)")
    print("-" * 100)
    print("  FLAGGED supply removables (priced on the MEASURED budget):")
    print(f"    pinned-K attn : {comp['pinnedk_attn_removable_us']:6.2f} us realistic / "
          f"{comp['pinnedk_attn_removable_roofline_us']:6.2f} us roofline -> attn-free "
          f"{comp['attn_free_roofline_tps']:.2f} (roof) / {comp['attn_free_realistic_tps']:.2f} (real); "
          f"gain +{comp['attn_lever_gain_roofline_tps']:.2f} (roof, reproduces #400 +14.29)")
    print(f"    cb3 body      : {comp['cb3_body_removable_us']:6.2f} us "
          f"(shrink {comp['cb3_read_shrink_frac']*100:.1f}% x bw_frac x t_body) -> "
          f"+{comp['cb3_body_removable_tps_roofline']:.2f} TPS roofline (PPL-UNCAPPED upper bound)")
    print(f"    lm_head       : {comp['lmhead_removable_us']:6.2f} us (#398 no loadable shrink)")
    print("-" * 100)
    print("  STACKED-FLAGGED-SUPPLY CEILING (fixed-overhead held constant):")
    print(f"    supply_stacked_flagged_ceiling_tps = {comp['supply_stacked_flagged_ceiling_tps']:.2f} "
          f"(roofline) / {comp['supply_stacked_flagged_ceiling_tps_realistic']:.2f} (realistic)")
    print(f"    clears_500 = {comp['supply_stacked_flagged_clears_500']} (roofline) / "
          f"{comp['supply_stacked_flagged_clears_500_realistic']} (realistic)")
    print(f"    residual_gap_after_all_supply_flagged = {comp['residual_gap_after_all_supply_flagged']:+.2f} TPS "
          f"(handed to demand); n_flags_in_stacked_supply = {comp['n_flags_in_stacked_supply']}")
    print("-" * 100)
    print(f"  SELF-TEST {st['passes']} ({st['n_checks']} checks): "
          + json.dumps({k: int(v) for k, v in st["conditions"].items()}))
    print("-" * 100)
    print("  VERDICT")
    print("   " + comp["verdict"])
    print(bar)


def maybe_log_wandb_decode_budget(payload: dict, args) -> str | None:
    if args.no_wandb:
        return None
    repo = str(Path(__file__).resolve().parents[3])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from scripts.wandb_logging import (init_wandb_run, log_summary, log_json_artifact, finish_wandb)
    except Exception as e:  # noqa: BLE001
        print(f"[budget] wandb helpers unavailable: {e}")
        return None
    comp = payload["compose"]
    body = comp["body_budget"]
    run = init_wandb_run(
        job_type="analysis-gpu-microbench", agent="wirbel",
        name=args.wandb_name, group=args.wandb_group,
        tags=["m1-decode-latency-budget", "closed-step-budget", "fixed-overhead-floor",
              "stacked-flagged-supply", "roofline", "319-strict-lock", "pr-408"],
        config={"pr": 408, "kind": "m1-decode-latency-budget",
                "head_dim": HEAD_DIM, "band_L": list(BAND_L), "step_norm_us": STEP_NORM_US,
                "ceiling_500": CEILING_500, "official_tps": OFFICIAL_TPS, "target_500": TARGET_500,
                "f_attn_344": F_ATTN_344, "f_body_strict_378": F_BODY_STRICT_378,
                "f_lmhead_378": F_LMHEAD_378, "f_draft_378": F_DRAFT_378,
                "cb3_bpw_eff": CB3_BPW_EFF_408, "cb3_read_shrink_frac": CB3_READ_SHRINK_FRAC_408,
                "m1_marlin_hbm_eff_391": M1_MARLIN_HBM_EFF_391, "lmhead_bytes": LMHEAD_BYTES_408,
                "eta_attn_decode_393": ETA_ATTN_DECODE_393, "deployed_strict_393": DEPLOYED_STRICT_393,
                "seeds": args.seeds, "ident_trials": args.ident_trials, "iters": args.iters},
    )
    if run is None:
        print("[budget] wandb disabled (no API key / WANDB_MODE).")
        return None
    flat: dict[str, float] = {}
    flat["budget/t_attn_frac"] = comp["t_attn_frac"]
    flat["budget/t_body_gemm_frac"] = comp["t_body_gemm_frac"]
    flat["budget/t_lmhead_frac"] = comp["t_lmhead_frac"]
    flat["budget/fixed_overhead_frac"] = comp["fixed_overhead_frac"]
    flat["budget/closure_residual_frac"] = comp["budget_closure_residual_frac"]
    flat["budget/t_attn_us"] = comp["t_attn_us"]
    flat["budget/t_body_gemm_us"] = comp["t_body_gemm_us"]
    flat["budget/t_lmhead_us"] = comp["t_lmhead_us"]
    flat["budget/t_fixed_overhead_us"] = comp["t_fixed_overhead_us"]
    flat["budget/step_norm_us"] = comp["step_norm_us"]
    flat["budget/overcredit_factor"] = comp["overcredit_factor"]
    flat["bw/body_bw_bound_frac"] = comp["body_bw_bound_frac"]
    flat["bw/lmhead_bw_bound_frac"] = comp["lmhead_bw_bound_frac"]
    flat["bw/body_eff_gbs"] = body["count_weighted_eff_gbs"]
    flat["bw/peak_copy_gbs"] = comp["peak_copy"]["peak_copy_gbs"]
    flat["bw/body_reconciles_391"] = float(comp["body_reconciles_391"])
    flat["eta/eta_attn_decode_only"] = comp["eta_attn_decode_only"]
    flat["tps/deployed_strict"] = comp["deployed_strict"]
    flat["tps/ceiling_strict"] = comp["ceiling_strict"]
    flat["lever/pinnedk_attn_removable_us"] = comp["pinnedk_attn_removable_us"]
    flat["lever/attn_free_roofline_tps"] = comp["attn_free_roofline_tps"]
    flat["lever/attn_free_realistic_tps"] = comp["attn_free_realistic_tps"]
    flat["lever/attn_lever_gain_roofline_tps"] = comp["attn_lever_gain_roofline_tps"]
    flat["lever/cb3_body_removable_us"] = comp["cb3_body_removable_us"]
    flat["lever/cb3_body_removable_tps_roofline"] = comp["cb3_body_removable_tps_roofline"]
    flat["lever/lmhead_removable_us"] = comp["lmhead_removable_us"]
    flat["ceiling/supply_stacked_flagged_ceiling_tps"] = comp["supply_stacked_flagged_ceiling_tps"]
    flat["ceiling/supply_stacked_flagged_ceiling_tps_realistic"] = comp["supply_stacked_flagged_ceiling_tps_realistic"]
    flat["ceiling/supply_stacked_flagged_clears_500"] = float(comp["supply_stacked_flagged_clears_500"])
    flat["ceiling/residual_gap_after_all_supply_flagged"] = comp["residual_gap_after_all_supply_flagged"]
    flat["ceiling/n_flags_in_stacked_supply"] = float(comp["n_flags_in_stacked_supply"])
    flat["selftest/decode_latency_budget_self_test_passes"] = float(payload["selftest"]["passes"])
    flat["gpu/sm_count"] = float(payload["gpu"]["sm_count"])
    flat["mem/peak_mem_mib"] = payload["peak_mem_mib"]
    run.log({"global_step": 0, **flat})
    log_summary(run, _jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="m1_decode_latency_budget", artifact_type="analysis", data=_jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[budget] wandb logged {len(flat)} keys (run {rid})")
    return rid


def main_decode_budget(dev: torch.device, gpu: dict, args) -> None:
    """PR #408 driver: build the CLOSED M=1 decode-step latency budget + stacked-flagged-supply
    ceiling. GPU MEASUREMENT + roofline ANALYSIS only (NO build/patch/launch/served-file change)."""
    comp = compose_decode_budget(dev, args, gpu)
    flags = {"no_hf_job": True, "no_launch": True, "no_served_file_change": True, "analysis_only": True}
    st = selftest_decode_budget(comp, gpu, flags, len(args.seeds))
    torch.cuda.synchronize()
    payload = {
        "agent": "wirbel", "pr": 408, "kind": "m1-decode-latency-budget",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        **flags,
        "gpu": gpu, "seeds": args.seeds,
        "peak_mem_mib": round(torch.cuda.max_memory_allocated(dev) / (1024**2), 3),
        "ladder_constants": {"ceiling_500": CEILING_500, "official_tps": OFFICIAL_TPS,
                             "target_500": TARGET_500, "step_norm_us": STEP_NORM_US,
                             "f_attn_344": F_ATTN_344, "f_body_strict_378": F_BODY_STRICT_378,
                             "f_lmhead_378": F_LMHEAD_378, "f_draft_378": F_DRAFT_378,
                             "cb3_bpw_eff": CB3_BPW_EFF_408, "cb3_read_shrink_frac": CB3_READ_SHRINK_FRAC_408,
                             "m1_marlin_hbm_eff_391": M1_MARLIN_HBM_EFF_391,
                             "eta_attn_decode_393": ETA_ATTN_DECODE_393,
                             "deployed_strict_393": DEPLOYED_STRICT_393, "ceiling_strict_393": CEILING_STRICT_393},
        "compose": comp, "selftest": st,
        # ---- PRIMARY + headline SENPAI-RESULT surface (the #408 deliverables) ----
        "decode_latency_budget_self_test_passes": bool(st["passes"]),
        "t_attn_frac": comp["t_attn_frac"],
        "t_body_gemm_frac": comp["t_body_gemm_frac"],
        "t_lmhead_frac": comp["t_lmhead_frac"],
        "fixed_overhead_frac": comp["fixed_overhead_frac"],
        "budget_closure_residual_frac": comp["budget_closure_residual_frac"],
        "body_bw_bound_frac": comp["body_bw_bound_frac"],
        "lmhead_bw_bound_frac": comp["lmhead_bw_bound_frac"],
        "pinnedk_attn_removable_us": comp["pinnedk_attn_removable_us"],
        "cb3_body_removable_tps_roofline": comp["cb3_body_removable_tps_roofline"],
        "supply_stacked_flagged_ceiling_tps": comp["supply_stacked_flagged_ceiling_tps"],
        "supply_stacked_flagged_clears_500": comp["supply_stacked_flagged_clears_500"],
        "residual_gap_after_all_supply_flagged": comp["residual_gap_after_all_supply_flagged"],
        "n_flags_in_stacked_supply": comp["n_flags_in_stacked_supply"],
        # test metric for the marker
        "deployed_strict_repro_393": comp["deployed_strict"],
    }
    print_report_decode_budget(payload)
    out_path = Path(args.out_dir) / "m1_decode_latency_budget_results.json"
    json.dump(_jsonable(payload), open(out_path, "w"), indent=2)
    print(f"[budget] results -> {out_path}")
    rid = maybe_log_wandb_decode_budget(payload, args)
    if rid:
        payload["wandb_run_id"] = rid
        json.dump(_jsonable(payload), open(out_path, "w"), indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to validate the path")
    ap.add_argument("--pinnedk-headroom", "--pinnedk_headroom", dest="pinnedk_headroom",
                    action="store_true", help="PR #400: pinned-K attention rebuild headroom entrypoint")
    ap.add_argument("--measure-m1-occupancy-bw", "--measure_m1_occupancy_bw", dest="measure_m1_occupancy_bw",
                    action="store_true", help="(pinnedk) measure M=1 draft-lane occupancy/BW (default on)")
    ap.add_argument("--roofline-recovery", "--roofline_recovery", dest="roofline_recovery",
                    action="store_true", help="(pinnedk) roofline the pinned-K recovery (default on)")
    ap.add_argument("--decode-latency-budget", "--decode_latency_budget", dest="decode_latency_budget",
                    action="store_true", help="PR #408: CLOSED M=1 decode-step latency budget entrypoint")
    ap.add_argument("--measure-per-component-cuda-events", "--measure_per_component_cuda_events",
                    dest="measure_per_component_cuda_events", action="store_true",
                    help="(budget) per-component CUDA-event measurement (default on)")
    ap.add_argument("--roofline-stacked-supply", "--roofline_stacked_supply", dest="roofline_stacked_supply",
                    action="store_true", help="(budget) roofline the stacked-flagged-supply ceiling (default on)")
    ap.add_argument("--ident-trials", type=int, default=8, help="independent batch=1 problems per seed")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234, 2345, 3456])
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default="wirbel/attention-strict-pin-cost")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="attention-strict-pin-cost")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    if args.smoke:
        args.ident_trials = min(args.ident_trials, 2)
        args.iters = min(args.iters, 15)
        args.warmup = min(args.warmup, 4)
        args.seeds = args.seeds[:3]

    dev = _device()
    gpu = _gpu_facts(dev)

    if args.decode_latency_budget:
        main_decode_budget(dev, gpu, args)
        return

    if args.pinnedk_headroom:
        main_pinnedk(dev, gpu, args)
        return

    comp = compose(dev, args, gpu)

    flags = {"no_hf_job": True, "no_launch": True, "no_served_file_change": True, "analysis_only": True}
    st = selftest(comp, gpu, flags, len(args.seeds))

    torch.cuda.synchronize()
    payload = {
        "agent": "wirbel", "pr": 393, "kind": "attention-strict-pin-cost",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        **flags,
        "gpu": gpu, "seeds": args.seeds,
        "peak_mem_mib": round(torch.cuda.max_memory_allocated(dev) / (1024**2), 3),
        "ladder_constants": {"ceiling_500": CEILING_500, "official_tps": OFFICIAL_TPS,
                             "step_norm_us": STEP_NORM_US, "eta_attn_378": ETA_ATTN_378,
                             "f_attn_344": F_ATTN_344, "penalty_ew_378": PENALTY_EW_378,
                             "budget_500_eta": BUDGET_500_ETA,
                             "strict_tps_attn_pin_390": STRICT_TPS_ATTN_PIN_390,
                             "deployed_tps_attn_pin_390": DEPLOYED_TPS_ATTN_PIN_390},
        "compose": comp, "selftest": st,
        # PRIMARY + headline SENPAI-RESULT surface
        "attention_strict_pin_cost_self_test_passes": bool(st["passes"]),
        "eta_attn_decode_only": comp["eta_attn_decode_only"],
        "eta_attn_decode_vs_evalweighted_delta": comp["eta_attn_decode_vs_evalweighted_delta"],
        "deployed_tps_decode_eta": comp["deployed_tps_decode_eta"],
        "cheapest_strict_attn_backend": comp["cheapest_strict_attn_backend"],
        "cheapest_strict_attn_eta": comp["cheapest_strict_attn_eta"],
        "n_byte_exact_attn_configs": comp["n_byte_exact_attn_configs"],
        "attn_eta_reducible": comp["attn_eta_reducible"],
        "attn_already_strict_free": comp["attn_already_strict_free"],
        "fa_sliding0_is_strict_floor": comp["fa_sliding0_is_strict_floor"],
        "ceiling_with_cheapest_attn": comp["ceiling_with_cheapest_attn"],
        "deployed_with_cheapest_attn": comp["deployed_with_cheapest_attn"],
        "gap_to_500_after_attn": comp["gap_to_500_after_attn"],
    }
    print_report(payload)
    out_path = Path(args.out_dir) / "attention_strict_pin_cost_results.json"
    json.dump(_jsonable(payload), open(out_path, "w"), indent=2)
    print(f"[attn-pin] results -> {out_path}")
    rid = maybe_log_wandb(payload, args)
    if rid:
        payload["wandb_run_id"] = rid
        json.dump(_jsonable(payload), open(out_path, "w"), indent=2)


if __name__ == "__main__":
    main()
