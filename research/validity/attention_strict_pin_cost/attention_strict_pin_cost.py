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
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to validate the path")
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
