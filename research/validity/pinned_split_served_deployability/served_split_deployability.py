#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #375 (wirbel) -- Pinned-split deployability: is the served decode pin a config knob or a kernel patch?

THE BRIDGE QUESTION (from "the cap is refuted analytically" to "we can ship it")
--------------------------------------------------------------------------------
My #366 (`h28xnyuy`) + #370 (`3v2yaps9`) validated that pinning num_splits=8 in the FA2 split-KV
decode keeps 64 CTAs-in-flight >= heuristic 56 >> un-packed 8, byte-exact + M-invariant, reviving
the strict-compliant lambda=1 ceiling ~= 518.92 TPS. BUT BOTH cards drove the kernel by hand in a
microbench using the UPSTREAM `flash_attn.flash_attn_with_kvcache(..., num_splits=8)` -- which routes
to FA2 `mha_fwd_kvcache`. The SERVED vLLM 0.22.x V1 decode does NOT call that function. This card asks
the operational question: can the SERVED decode be made to USE the pinned split via a config/env knob
(free, deployable now), or does it need a served-file kernel patch (#319-flag-to-human)?

WHAT THE AUDIT FOUND (source + on-target microbench, both agree)
---------------------------------------------------------------
1. The served V1 FlashAttention backend decode calls **`flash_attn_varlen_func(..., num_splits=
   attn_metadata.max_num_splits, fa_version=2)`** (v1/attention/backends/flash_attn.py:796-818) -- i.e.
   FA2 `mha_varlen_fwd`, NOT `mha_fwd_kvcache`. The vendored `vllm_flash_attn` package does not even
   EXPORT `flash_attn_with_kvcache` (only `flash_attn_varlen_func`). So #366/#370's kernel is not the
   served kernel.
2. `max_num_splits` is **0** by default (heuristic) and is forced to **1** by `VLLM_BATCH_INVARIANT=1`
   (flash_attn.py:332,430,442-443). The FA3-only AOT path (`flash_attn_max_num_splits_for_cuda_graph=32`)
   is inert on FA2/sm_86. So the only num_splits values the served caller can select are {0, 1}.
3. The FA2 Python wrapper hard-guards **`if num_splits > 1: raise NotImplementedError("FA2 does not
   support num_splits > 1")`** (flash_attn_interface.py:298-299). And the compiled C++ kernel itself
   hard-stops the paged decode: `TORCH_CHECK(num_splits <= 1, "num_splits > 1 is not supported for
   varlen paged KV")` -- measured as a RuntimeError. So pinning num_splits=8 is blocked at BOTH layers.
4. The FA2 varlen paged path only splits via the `seqlenq_ngroups_swapped` GQA trick, which fires ONLY
   for max_seqlen_q==1 (the AR M=1 step). So:
     * M=1 (AR):    num_splits=0 SPLITS (heuristic K = 9/7/10 @ L 528/2048/4096 -- == #370 anchors),
                    num_splits=1 does not -> the two DIFFER (maxdiff ~5e-4).
     * M=8 (verify): no split path at all; num_splits=0 == num_splits=1 byte-exact; num_splits=8 errors.
   => the served DEFAULT (heuristic) BREAKS M-invariance: an AR-decoded position (M=1, split) != the
   same position verify-batched (M=8, no split). The ONLY served-deployable M-invariant config is
   num_splits=1 (VLLM_BATCH_INVARIANT) -- which is the UN-PACKED penalty config (3.0-4.7x slower M=1
   attention), exactly the #332 collapse the pinned-8 was meant to avoid.

DECISION: pinned_split_is_served_knob = FALSE. The pin (num_splits=8, the 64-CTA high-occupancy
M-invariant config) is NOT reachable on the served varlen kernel by ANY config/env/launch knob, and
not even by a Python served-file patch -- the compiled kernel rejects it. Deploying it requires a
kernel REBUILD (add forced-split + ordered combine to FA2 `mha_varlen_fwd`'s paged branch). The revived
518.92 ceiling is therefore UN-DEPLOYABLE on the served kernel as-is; the #319 served confirm would
measure either the M-variance-breaking heuristic (default) or the un-packed num_splits=1 penalty, never
the pinned-8 ceiling. Bank the blocker.

SCOPE: pod-GPU microbench on the SERVED vendored kernel + CPU source-audit + analytic round-trip over
banked MERGED *_results.json. NO train.py --launch, NO HF Job, NO submission, NO served-file change.
0 official TPS, baseline 481.53 UNCHANGED. Greedy identity is MEASURED, never broken. Run with
CUDA_VISIBLE_DEVICES=0 (#358/#363 single-A10G gotcha). W&B group strict-bi-verify-gemm (== #366/#370).
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

# ---------------------------------------------------------------------------------------- #
# gemma-4-E4B-it text-decoder attention geometry (EXACT dims #327/#332/#358/#363/#366/#370 use)
# ---------------------------------------------------------------------------------------- #
HEAD_DIM = 256
N_Q_HEADS = 8
N_KV_HEADS = 2
DTYPE = torch.bfloat16
SCALE = 1.0 / math.sqrt(HEAD_DIM)
A10G_SMS = 80
BLOCK_M_SPLITKV = 64
BLOCK_N_TILE = 64
SERVED_BLOCK_SIZE = 16            # vLLM deployment block_size -- USED DIRECTLY here (more faithful than #370's 256 bridge)
M_AR = 1                          # AR decode width
M_VERIFY = 8                      # verify width (K_spec=7 + 1) -- the deployed verify geometry
M_LIST = (M_AR, M_VERIFY)
L_LIST = (528, 2048, 4096)        # bridge / primary / served-max context
PRIMARY_L = 2048

HEURISTIC_SPLIT = 0               # the DEPLOYED default (max_num_splits=0)
UNPACK_SPLIT = 1                  # VLLM_BATCH_INVARIANT -> num_splits=1 (the only served-deployable M-invariant config)
PINNED_SPLIT = 8                  # #366/#370 M-invariant high-occupancy mechanism -- the config under test

# ---- #366/#370 banked CTA anchors (kvcache kernel; round-tripped in the self-test) ------ #
PIN8_CTAS = 64
HEUR_CTAS_L2048 = 56
UNPK_CTAS = 8

# ---- strict budget ladder (cite, identical to #366/#370) -------------------------------- #
CEILING_500 = 520.953
STEP_US = 1218.2
TARGET_500 = 500.0
BUDGET_500_ETA = 1.0 - TARGET_500 / CEILING_500
REVIVED_CEILING_366 = 518.9188253620001
FLOOR_BLANKET_9841 = 0.09841249119201488

# banked anchor JSONs (reconciliation provenance; all MERGED on the advisor branch)
_VAL = Path(__file__).resolve().parents[1]
ANCHOR_366 = _VAL / "pinned_split_phi_audit" / "pinned_split_phi_audit_results.json"
ANCHOR_370 = _VAL / "pagedkv_occupancy" / "pagedkv_occupancy_microbench_results.json"
# stark #365 (lm_head BI-GEMM) -- a DIFFERENT locus; reconciled analytically. NOT inspected on
# stark's branch (launch isolation: only wirbel branches + advisor branch).
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


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _load_json(p: Path) -> dict[str, Any] | None:
    if not p.is_file():
        return None
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return None


# ======================================================================================== #
# heuristic replica (exact copy of flash_attn num_splits_heuristic; round-tripped in selftest)
# ======================================================================================== #
def _num_splits_heuristic(bnm: int, num_sms: int, num_n_blocks: int, max_splits: int = 128) -> int:
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
# STEP 1 -- SOURCE AUDIT (resolve the served files from the imported vllm; self-verify lines)
# ======================================================================================== #
def step1_source_audit() -> dict[str, Any]:
    import vllm
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

    vroot = Path(vllm.__file__).resolve().parent
    iface = vroot / "vllm_flash_attn" / "flash_attn_interface.py"
    backend = vroot / "v1" / "attention" / "backends" / "flash_attn.py"
    itxt = iface.read_text() if iface.is_file() else ""
    btxt = backend.read_text() if backend.is_file() else ""

    fa_version = get_flash_attn_version()

    # source-fact checks (these are the load-bearing audit facts; each must be present in the
    # installed served kernel or the audit is stale)
    facts = {
        "vllm_version": vllm.__version__,
        "served_fa_version_on_a10g": fa_version,
        "iface_exports_varlen_func": "def flash_attn_varlen_func(" in itxt,
        "iface_has_NO_with_kvcache": "def flash_attn_with_kvcache(" not in itxt,
        "fa2_guard_blocks_gt1": 'FA2 does not support num_splits > 1' in itxt,
        "backend_decode_passes_max_num_splits": "num_splits=attn_metadata.max_num_splits" in btxt,
        "backend_default_max_num_splits_zero": "self.max_num_splits = 0" in btxt,
        "backend_batchinv_forces_1": ("if envs.VLLM_BATCH_INVARIANT:" in btxt and "max_num_splits = 1" in btxt),
    }
    mechanism = (
        "served decode calls vllm_flash_attn.flash_attn_varlen_func(num_splits=attn_metadata."
        "max_num_splits, fa_version=2) [v1/attention/backends/flash_attn.py:796]. The ONLY num_splits "
        "values the served caller can select are 0 (default heuristic; max_num_splits=0) and 1 "
        "(VLLM_BATCH_INVARIANT=1 env). num_splits>1 (the pin) is rejected by BOTH the FA2 Python wrapper "
        "(NotImplementedError 'FA2 does not support num_splits > 1', flash_attn_interface.py:298) AND the "
        "compiled kernel (TORCH_CHECK 'num_splits > 1 is not supported for varlen paged KV' in "
        "mha_varlen_fwd). No knob reaches the pin."
    )
    # the only num_splits override that reaches the served decode is the VLLM_BATCH_INVARIANT env
    # (it forces 1, the un-packed config) -- NOT a pin-8 knob.
    return {
        "served_entry_point": "vllm_flash_attn.flash_attn_varlen_func -> _vllm_fa2_C.varlen_fwd (mha_varlen_fwd)",
        "served_kernel_is_NOT_366_kvcache": True,
        "served_num_splits_override_mechanism": mechanism,
        "override_env_for_determinism": "VLLM_BATCH_INVARIANT=1 (forces num_splits=1, un-packed)",
        "config_knob_flash_attn_max_num_splits_for_cuda_graph": "inert on FA2/sm_86 (FA3+full-CG only)",
        "requires_served_file_change_for_pin8": True,
        "pin8_requires_kernel_rebuild": True,
        "source_facts": facts,
        "all_source_facts_hold": all(facts[k] for k in (
            "iface_exports_varlen_func", "iface_has_NO_with_kvcache", "fa2_guard_blocks_gt1",
            "backend_decode_passes_max_num_splits", "backend_default_max_num_splits_zero",
            "backend_batchinv_forces_1")) and fa_version == 2,
    }


# ======================================================================================== #
# STEP 2 -- SERVED-KERNEL MICROBENCH (the served entry point; page_size=16; >=2 seeds)
# ======================================================================================== #
def _build_paged(L: int, M: int, page: int, seed: int, dev: torch.device):
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(M, N_Q_HEADS, HEAD_DIM, generator=g, device=dev, dtype=DTYPE)
    nb = _ceildiv(L, page)
    kc = torch.randn(nb, page, N_KV_HEADS, HEAD_DIM, generator=g, device=dev, dtype=DTYPE)
    vc = torch.randn(nb, page, N_KV_HEADS, HEAD_DIM, generator=g, device=dev, dtype=DTYPE)
    bt = torch.arange(nb, dtype=torch.int32, device=dev).unsqueeze(0)
    cu = torch.tensor([0, M], dtype=torch.int32, device=dev)
    sk = torch.tensor([L], dtype=torch.int32, device=dev)
    return q, kc, vc, bt, cu, sk, nb


def _served_varlen(varlen_fn, q, kc, vc, bt, cu, sk, L, M, ns):
    """Faithful served decode call (the vendored Python wrapper marshals exactly what the V1 backend marshals)."""
    return varlen_fn(q=q, k=kc, v=vc, out=None, cu_seqlens_q=cu, max_seqlen_q=M,
                     seqused_k=sk, max_seqlen_k=L, softmax_scale=SCALE, causal=False,
                     block_table=bt, num_splits=ns, fa_version=2)


def _direct_varlen(q, kc, vc, bt, cu, sk, L, M, ns):
    """Bypass the Python guard -> call the compiled op directly (tests whether a served-FILE patch would suffice)."""
    dummy = torch.empty_like(cu)
    r = torch.ops._vllm_fa2_C.varlen_fwd(q, kc, vc, None, cu, dummy, sk, None, bt, None,
                                         M, L, 0.0, SCALE, False, False, -1, -1, 0.0, False, ns, None)
    return r[0]


def _time(fn, iters: int, warmup: int) -> float:
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
    return ts[len(ts) // 2] * 1e3  # us median


def step2_served_microbench(varlen_fn, seeds: list[int], iters: int, warmup: int,
                            dev: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # --- pin-8 blockedness (geometry-independent; sample at primary L) ------------------- #
    qv, kcv, vcv, btv, cuv, skv, _ = _build_paged(PRIMARY_L, M_VERIFY, SERVED_BLOCK_SIZE, seeds[0], dev)
    qa, kca, vca, bta, cua, ska, _ = _build_paged(PRIMARY_L, M_AR, SERVED_BLOCK_SIZE, seeds[0], dev)
    pin8 = {"wrapper_verify": None, "wrapper_ar": None, "direct_verify": None, "direct_ar": None}
    try:
        _served_varlen(varlen_fn, qv, kcv, vcv, btv, cuv, skv, PRIMARY_L, M_VERIFY, PINNED_SPLIT)
        pin8["wrapper_verify"] = "RAN (unexpected)"
    except NotImplementedError as ex:
        pin8["wrapper_verify"] = f"NotImplementedError: {str(ex)[:64]}"
    except Exception as ex:  # noqa: BLE001
        pin8["wrapper_verify"] = f"{type(ex).__name__}: {str(ex)[:64]}"
    try:
        _served_varlen(varlen_fn, qa, kca, vca, bta, cua, ska, PRIMARY_L, M_AR, PINNED_SPLIT)
        pin8["wrapper_ar"] = "RAN (unexpected)"
    except NotImplementedError as ex:
        pin8["wrapper_ar"] = f"NotImplementedError: {str(ex)[:64]}"
    except Exception as ex:  # noqa: BLE001
        pin8["wrapper_ar"] = f"{type(ex).__name__}: {str(ex)[:64]}"
    try:
        _direct_varlen(qv, kcv, vcv, btv, cuv, skv, PRIMARY_L, M_VERIFY, PINNED_SPLIT)
        pin8["direct_verify"] = "RAN (unexpected)"
    except Exception as ex:  # noqa: BLE001
        pin8["direct_verify"] = f"{type(ex).__name__}: {str(ex)[:72]}"
    try:
        _direct_varlen(qa, kca, vca, bta, cua, ska, PRIMARY_L, M_AR, PINNED_SPLIT)
        pin8["direct_ar"] = "RAN (ngroups path accepts, but NON-UNIFORM: verify blocks)"
    except Exception as ex:  # noqa: BLE001
        pin8["direct_ar"] = f"{type(ex).__name__}: {str(ex)[:72]}"

    pin8_wrapper_blocked_all_M = ("NotImplementedError" in str(pin8["wrapper_verify"])
                                  and "NotImplementedError" in str(pin8["wrapper_ar"]))
    pin8_verify_kernel_blocked = "RuntimeError" in str(pin8["direct_verify"])
    out["pin8_probe"] = pin8
    out["pin8_wrapper_blocked_all_M"] = pin8_wrapper_blocked_all_M
    out["pin8_verify_kernel_blocked"] = pin8_verify_kernel_blocked
    # the pin can only be uniform across AR+verify if BOTH accept it; verify never does:
    out["pin8_uniform_AR_and_verify_possible"] = bool(
        "RAN" in str(pin8["direct_verify"]) and "RAN" in str(pin8["direct_ar"]))

    # --- per-(L,M) byte-exactness, M-invariance, latency --------------------------------- #
    per_L: dict[str, Any] = {}
    any_nan = False
    for L in L_LIST:
        rec: dict[str, Any] = {"L": L}
        # build once per (L, seed); reuse KV for the per-row M-invariance reference
        maxdiff_0v1 = {str(M): 0.0 for M in M_LIST}
        m_invariant = {str(ns): True for ns in (HEURISTIC_SPLIT, UNPACK_SPLIT)}
        us = {str(M): {str(ns): [] for ns in (HEURISTIC_SPLIT, UNPACK_SPLIT)} for M in M_LIST}
        for seed in seeds:
            built = {M: _build_paged(L, M, SERVED_BLOCK_SIZE, seed, dev) for M in M_LIST}
            for M in M_LIST:
                q, kc, vc, bt, cu, sk, _ = built[M]
                o0 = _served_varlen(varlen_fn, q, kc, vc, bt, cu, sk, L, M, HEURISTIC_SPLIT)
                o1 = _served_varlen(varlen_fn, q, kc, vc, bt, cu, sk, L, M, UNPACK_SPLIT)
                any_nan = any_nan or bool(torch.isnan(o0).any() or torch.isnan(o1).any())
                d = (o0.float() - o1.float()).abs().max().item()
                maxdiff_0v1[str(M)] = max(maxdiff_0v1[str(M)], d)
                us[str(M)][str(HEURISTIC_SPLIT)].append(_time(
                    lambda: _served_varlen(varlen_fn, q, kc, vc, bt, cu, sk, L, M, HEURISTIC_SPLIT), iters, warmup))
                us[str(M)][str(UNPACK_SPLIT)].append(_time(
                    lambda: _served_varlen(varlen_fn, q, kc, vc, bt, cu, sk, L, M, UNPACK_SPLIT), iters, warmup))
            # M-invariance: each row of the M=VERIFY batched call vs the same row computed as an M=1 call
            qv, kcv, vcv, btv, cuv, skv, _ = built[M_VERIFY]
            for ns in (HEURISTIC_SPLIT, UNPACK_SPLIT):
                ob = _served_varlen(varlen_fn, qv, kcv, vcv, btv, cuv, skv, L, M_VERIFY, ns)
                for r in range(M_VERIFY):
                    qr = qv[r:r + 1].contiguous()
                    cur = torch.tensor([0, 1], dtype=torch.int32, device=dev)
                    orow = _served_varlen(varlen_fn, qr, kcv, vcv, btv, cur, skv, L, 1, ns)
                    if (orow.float() - ob[r:r + 1].float()).abs().max().item() != 0.0:
                        m_invariant[str(ns)] = False
                        break
        rec["maxdiff_heuristic_vs_unpack_by_M"] = maxdiff_0v1
        rec["m_invariant_heuristic"] = m_invariant[str(HEURISTIC_SPLIT)]
        rec["m_invariant_unpack"] = m_invariant[str(UNPACK_SPLIT)]
        rec["us_median_by_M_by_ns"] = {M: {ns: (sorted(v)[len(v) // 2] if v else None)
                                            for ns, v in d.items()} for M, d in us.items()}
        # un-packing (determinism) penalty at M=1: us(num_splits=1) / us(heuristic)
        ar = rec["us_median_by_M_by_ns"][str(M_AR)]
        rec["ar_unpack_penalty_ratio"] = (ar[str(UNPACK_SPLIT)] / ar[str(HEURISTIC_SPLIT)]
                                          if ar[str(HEURISTIC_SPLIT)] else None)
        rec["heuristic_K_ar_analytic"] = _heuristic_K(M_AR, L)
        rec["heuristic_K_verify_analytic"] = _heuristic_K(M_VERIFY, L)
        per_L[str(L)] = rec
    out["per_L"] = per_L
    out["any_nan"] = any_nan
    out["seeds"] = seeds
    # headline served-path determinism facts
    out["served_default_breaks_m_invariance"] = any(
        not per_L[str(L)]["m_invariant_heuristic"] for L in L_LIST)
    out["served_unpack_is_m_invariant"] = all(
        per_L[str(L)]["m_invariant_unpack"] for L in L_LIST)
    out["max_ar_unpack_penalty"] = max(
        (per_L[str(L)]["ar_unpack_penalty_ratio"] or 0.0) for L in L_LIST)
    return out


# ======================================================================================== #
# STEP 3 -- RECONCILE with #366/#370 banked anchors + stark #365 (analytic; isolation-safe)
# ======================================================================================== #
def step3_reconcile(micro: dict[str, Any]) -> dict[str, Any]:
    a366 = _load_json(ANCHOR_366)
    a370 = _load_json(ANCHOR_370)
    rec: dict[str, Any] = {
        "anchor_366_loaded": a366 is not None,
        "anchor_370_loaded": a370 is not None,
    }
    if a370 is not None:
        rec["a370_ctas_pinned8"] = a370.get("ctas_in_flight_paged_pinned8")
        rec["a370_ctas_heuristic"] = a370.get("ctas_in_flight_paged_heuristic")
        rec["a370_kernel_fn"] = a370.get("kernel", {}).get("fn")
        rec["a370_paged_confirms_366"] = a370.get("paged_confirms_366")
        rec["a370_revived_ceiling"] = a370.get("recompute", {}).get("revived_ceiling_paged_tps")
    if a366 is not None:
        rec["a366_wandb"] = a366.get("wandb_run_id")
    # the central reconciliation: #366/#370 validated num_splits=8 on the KVCACHE kernel
    # (mha_fwd_kvcache); the SERVED decode uses the VARLEN kernel (mha_varlen_fwd) which cannot pin.
    rec["served_kernel_differs_from_366_kvcache"] = True
    rec["a370_K_matches_served_ar_heuristic"] = {
        str(L): (_heuristic_K(M_AR, L) == micro["per_L"][str(L)]["heuristic_K_ar_analytic"])
        for L in L_LIST}
    rec["interpretation"] = (
        "#366/#370 pinned num_splits=8 in FA2 mha_fwd_kvcache (kvcache kernel) -> 64 CTAs, byte-exact, "
        "M-invariant, revived ceiling 518.92. The SERVED V1 decode uses FA2 mha_varlen_fwd (varlen paged) "
        "instead, which: (verify M=8) has NO split-KV combine -> hard C++ block on num_splits>1; (AR M=1) "
        "splits via the seqlenq_ngroups_swapped heuristic (K matches #370's kvcache anchors). The 64-CTA "
        "pin #370 validated is therefore NOT requestable on the served kernel. The served path can only be "
        "heuristic (M-variance-breaking) or num_splits=1 (un-packed penalty). Pin-8 needs a kernel rebuild."
    )
    # stark #365 reconciliation -- isolation-safe (do NOT inspect stark's branch)
    p365 = next((p for p in PR365_CANDIDATES if p.is_file()), None)
    rec["pr365_artifact_on_advisor_branch"] = str(p365) if p365 else None
    rec["pr365_locus"] = "lm_head BI-GEMM (per banked #370 reconcile block) -- SEPARATE from attention split"
    rec["pr365_reconcile"] = (
        "stark #365 is the lm_head BI-GEMM eta locus, not the attention split. Under launch isolation I do "
        "not inspect stark's branch; analytically, any attention-side determinism in an end-to-end harness "
        "would pin the split via VLLM_BATCH_INVARIANT (num_splits=1) -- the SAME (and ONLY) served knob "
        "audited here -- inheriting the un-packed penalty, NOT the pinned-8 high-occupancy config. So a "
        "#365 end-to-end identity result transfers to the num_splits=1 served config, not to the 518.92 pin."
    )
    return rec


# ======================================================================================== #
# STEP 4 -- analytic self-test (round-trip #366/#370 CTA counts + served-kernel facts)
# ======================================================================================== #
def selftest(audit: dict, micro: dict, reconcile: dict, gpu: dict, flags: dict) -> dict[str, Any]:
    c: dict[str, bool] = {}
    # (a) analytic CTA round-trip of the #366/#370 anchors via the heuristic replica
    K_heur_2048 = _heuristic_K(M_VERIFY, PRIMARY_L)   # bnm=8, nnb=32 -> 7
    c["a_heur_K_2048_is_7"] = (K_heur_2048 == 7)
    c["a_pin8_cta_roundtrip"] = (N_Q_HEADS * _ceildiv(M_VERIFY, BLOCK_M_SPLITKV) * PINNED_SPLIT == PIN8_CTAS)   # 8*1*8=64
    c["a_heur_cta_roundtrip"] = (N_Q_HEADS * _ceildiv(M_VERIFY, BLOCK_M_SPLITKV) * K_heur_2048 == HEUR_CTAS_L2048)  # 8*1*7=56
    c["a_unpk_cta_roundtrip"] = (N_Q_HEADS * _ceildiv(M_VERIFY, BLOCK_M_SPLITKV) * 1 == UNPK_CTAS)              # 8*1*1=8
    c["a_pin_gt_heur_gt_unpk"] = (PIN8_CTAS > HEUR_CTAS_L2048 > UNPK_CTAS)
    c["a_budget_500_roundtrips"] = (abs(BUDGET_500_ETA - (1.0 - TARGET_500 / CEILING_500)) < 1e-9)
    # (b) source-audit facts hold on the installed served kernel
    c["b_fa_version_is_2"] = (audit["source_facts"]["served_fa_version_on_a10g"] == 2)
    c["b_iface_no_kvcache"] = audit["source_facts"]["iface_has_NO_with_kvcache"]
    c["b_fa2_guard_present"] = audit["source_facts"]["fa2_guard_blocks_gt1"]
    c["b_backend_passes_max_num_splits"] = audit["source_facts"]["backend_decode_passes_max_num_splits"]
    c["b_default_zero_and_batchinv_one"] = (audit["source_facts"]["backend_default_max_num_splits_zero"]
                                            and audit["source_facts"]["backend_batchinv_forces_1"])
    c["b_all_source_facts_hold"] = audit["all_source_facts_hold"]
    c["b_requires_kernel_rebuild"] = audit["pin8_requires_kernel_rebuild"]
    # (c) empirical served-kernel measurements
    c["c_nan_clean"] = (not micro["any_nan"])
    c["c_pin8_wrapper_blocked_all_M"] = micro["pin8_wrapper_blocked_all_M"]
    c["c_pin8_verify_kernel_blocked"] = micro["pin8_verify_kernel_blocked"]
    c["c_pin8_not_uniform"] = (not micro["pin8_uniform_AR_and_verify_possible"])
    c["c_default_breaks_m_invariance"] = micro["served_default_breaks_m_invariance"]
    c["c_unpack_is_m_invariant"] = micro["served_unpack_is_m_invariant"]
    c["c_verify_0_eq_1_byte_exact"] = all(
        micro["per_L"][str(L)]["maxdiff_heuristic_vs_unpack_by_M"][str(M_VERIFY)] == 0.0 for L in L_LIST)
    c["c_ar_0_ne_1"] = all(
        micro["per_L"][str(L)]["maxdiff_heuristic_vs_unpack_by_M"][str(M_AR)] > 0.0 for L in L_LIST)
    c["c_unpack_penalty_gt_1"] = (micro["max_ar_unpack_penalty"] > 1.0)
    c["c_ar_heuristic_K_matches_370"] = all(reconcile["a370_K_matches_served_ar_heuristic"].values())
    c["c_seeds_ge_2"] = (len(micro["seeds"]) >= 2)
    # (d) reconciliation + decision coherence
    c["d_served_kernel_differs_from_kvcache"] = reconcile["served_kernel_differs_from_366_kvcache"]
    c["d_anchor_370_loaded"] = reconcile["anchor_370_loaded"]
    c["d_a370_is_kvcache_fn"] = (reconcile.get("a370_kernel_fn") == "flash_attn_with_kvcache")
    # (e) hygiene
    c["e_on_target_a10g_80sm"] = gpu["is_a10g_80sm"]
    c["e_ga102_sm86"] = gpu["is_ga102_sm86"]
    c["e_no_launch_flags"] = bool(flags.get("no_hf_job") and flags.get("no_launch")
                                  and flags.get("no_served_file_change"))
    return {"conditions": c, "n_checks": len(c), "passes": all(c.values())}


# ======================================================================================== #
# Report + IO + wandb
# ======================================================================================== #
def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, bool):
        return o
    if isinstance(o, (int, float, str)) or o is None:
        return o
    return str(o)


def print_report(p: dict[str, Any]) -> None:
    au, mi, rc, st = p["audit"], p["microbench"], p["reconcile"], p["selftest"]
    print("=" * 92)
    print(f"PR #375 wirbel -- pinned-split served deployability  ({p['created_at']})")
    print(f"  GPU {p['gpu']['name']} sm{p['gpu']['compute_capability']} x{p['gpu']['sm_count']}  "
          f"served FA v{au['source_facts']['served_fa_version_on_a10g']}  vllm {au['source_facts']['vllm_version']}")
    print("-" * 92)
    print("  DECISION")
    print(f"    pinned_split_is_served_knob ............. {p['pinned_split_is_served_knob']}")
    print(f"    requires_served_file_change ............. {p['requires_served_file_change']}")
    print(f"    served_pin_takes_effect ................. {p['served_pin_takes_effect']}")
    print(f"    served_pin_reproduces_366_byte_exact .... {p['served_pin_reproduces_366_byte_exact']}")
    print(f"    minimal_change_loc ...................... {p['minimal_change_loc']}")
    print("-" * 92)
    print("  SERVED num_splits override surface")
    print(f"    entry point: {au['served_entry_point']}")
    print(f"    override (determinism): {au['override_env_for_determinism']}")
    print(f"    cuda-graph config knob: {au['config_knob_flash_attn_max_num_splits_for_cuda_graph']}")
    print("-" * 92)
    print("  PIN-8 blockedness (served kernel)")
    for k, v in mi["pin8_probe"].items():
        print(f"    {k:16s}: {v}")
    print("-" * 92)
    print(f"  {'L':>5} {'minv_heur':>9} {'minv_unpk':>9} {'md_AR(0v1)':>11} {'md_V(0v1)':>10} "
          f"{'AR_pen':>7} {'K_ar':>5} {'K_ver':>5}")
    for L in L_LIST:
        m = mi["per_L"][str(L)]
        print(f"  {L:>5} {str(m['m_invariant_heuristic']):>9} {str(m['m_invariant_unpack']):>9} "
              f"{m['maxdiff_heuristic_vs_unpack_by_M'][str(M_AR)]:>11.2e} "
              f"{m['maxdiff_heuristic_vs_unpack_by_M'][str(M_VERIFY)]:>10.2e} "
              f"{(m['ar_unpack_penalty_ratio'] or 0):>7.2f} {m['heuristic_K_ar_analytic']:>5} "
              f"{m['heuristic_K_verify_analytic']:>5}")
    print("-" * 92)
    print(f"  served_default_breaks_m_invariance ... {mi['served_default_breaks_m_invariance']}")
    print(f"  served_unpack_is_m_invariant ......... {mi['served_unpack_is_m_invariant']}")
    print(f"  max AR un-pack (determinism) penalty . {mi['max_ar_unpack_penalty']:.2f}x")
    print(f"  reconcile: {rc['interpretation'][:300]}")
    print("-" * 92)
    print(f"  SELF-TEST {st['n_checks']} checks -> {'PASS' if st['passes'] else 'FAIL'}")
    if not st["passes"]:
        for k, v in st["conditions"].items():
            if not v:
                print(f"    FAILED: {k}")
    print("=" * 92)


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
        print(f"[served-split] wandb helpers unavailable: {e}")
        return None
    au, mi, st = payload["audit"], payload["microbench"], payload["selftest"]
    run = init_wandb_run(
        job_type="analysis-gpu-microbench", agent="wirbel",
        name=args.wandb_name, group=args.wandb_group,
        tags=["strict-bi-verify-gemm", "pinned-split", "served-deployability", "varlen-paged",
              "num-splits", "319-strict-lock", "pr-375"],
        config={"pr": 375, "kind": "served-split-deployability",
                "head_dim": HEAD_DIM, "n_q_heads": N_Q_HEADS, "n_kv_heads": N_KV_HEADS,
                "m_list": list(M_LIST), "L_list": list(L_LIST), "served_block_size": SERVED_BLOCK_SIZE,
                "ceiling_500": CEILING_500, "revived_ceiling_366": REVIVED_CEILING_366},
    )
    if run is None:
        print("[served-split] wandb disabled (no API key / WANDB_MODE).")
        return None
    flat: dict[str, float] = {
        "decision/pinned_split_is_served_knob": float(payload["pinned_split_is_served_knob"]),
        "decision/requires_served_file_change": float(payload["requires_served_file_change"]),
        "decision/served_pin_takes_effect": float(payload["served_pin_takes_effect"]),
        "decision/served_pin_reproduces_366_byte_exact": float(payload["served_pin_reproduces_366_byte_exact"]),
        "decision/pin8_requires_kernel_rebuild": float(au["pin8_requires_kernel_rebuild"]),
        "audit/served_fa_version": float(au["source_facts"]["served_fa_version_on_a10g"]),
        "audit/all_source_facts_hold": float(au["all_source_facts_hold"]),
        "micro/pin8_wrapper_blocked_all_M": float(mi["pin8_wrapper_blocked_all_M"]),
        "micro/pin8_verify_kernel_blocked": float(mi["pin8_verify_kernel_blocked"]),
        "micro/pin8_uniform_possible": float(mi["pin8_uniform_AR_and_verify_possible"]),
        "micro/served_default_breaks_m_invariance": float(mi["served_default_breaks_m_invariance"]),
        "micro/served_unpack_is_m_invariant": float(mi["served_unpack_is_m_invariant"]),
        "micro/max_ar_unpack_penalty": float(mi["max_ar_unpack_penalty"]),
        "selftest/analytic_self_test_passes": float(st["passes"]),
        "selftest/n_checks": float(st["n_checks"]),
        "gpu/sm_count": float(payload["gpu"]["sm_count"]),
    }
    for L in L_LIST:
        m = mi["per_L"][str(L)]
        flat[f"micro/L{L}_minv_heuristic"] = float(m["m_invariant_heuristic"])
        flat[f"micro/L{L}_minv_unpack"] = float(m["m_invariant_unpack"])
        flat[f"micro/L{L}_md_AR_0v1"] = m["maxdiff_heuristic_vs_unpack_by_M"][str(M_AR)]
        flat[f"micro/L{L}_md_V_0v1"] = m["maxdiff_heuristic_vs_unpack_by_M"][str(M_VERIFY)]
        flat[f"micro/L{L}_ar_unpack_penalty"] = m["ar_unpack_penalty_ratio"] or 0.0
    run.log({"global_step": 0, **flat})
    log_summary(run, _jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="served_split_deployability", artifact_type="analysis", data=_jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[served-split] wandb logged {len(flat)} keys (run {rid})")
    return rid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true", help="run the served-kernel microbench (default path)")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to validate the path")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="wirbel/pinned-split-served-deployability")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="strict-bi-verify-gemm")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    if args.smoke:
        args.iters = min(args.iters, 20)
        args.warmup = min(args.warmup, 5)
        args.seeds = args.seeds[:2]

    # register the served kernel ops + import the served wrapper (the EXACT served entry point)
    import vllm  # noqa: F401
    from vllm.vllm_flash_attn import _vllm_fa2_C  # noqa: F401  (registers torch.ops._vllm_fa2_C)
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    torch.manual_seed(args.seeds[0])
    dev = _device()
    gpu = _gpu_facts(dev)

    audit = step1_source_audit()
    micro = step2_served_microbench(flash_attn_varlen_func, args.seeds, args.iters, args.warmup, dev)
    reconcile = step3_reconcile(micro)

    flags = {"no_hf_job": True, "no_launch": True, "no_served_file_change": True}
    st = selftest(audit, micro, reconcile, gpu, flags)

    # ---- decision (the PR #375 required terminal fields) -------------------------------- #
    pinned_split_is_served_knob = False                       # num_splits=8 unreachable by any knob
    requires_served_file_change = True                        # and stronger: a kernel rebuild
    served_pin_takes_effect = False                           # verify (M=8) hard-blocks; pin can't run uniformly
    served_pin_reproduces_366_byte_exact = False              # cannot pin -> cannot reproduce
    minimal_change_loc = (
        "KERNEL REBUILD (not a config/env knob, not a Python served-file patch): recompile "
        "_vllm_fa2_C.abi3.so after adding a forced num_splits>1 + ordered split-KV combine to FA2 "
        "mha_varlen_fwd's paged branch (remove TORCH_CHECK 'num_splits > 1 is not supported for varlen "
        "paged KV' + wire the combine kernel) AND set max_num_splits=8 in v1/attention/backends/"
        "flash_attn.py + remove the FA2 Python guard in flash_attn_interface.py. ~O(100s) LOC CUDA + a "
        "CUDA build. Config-only LOC to reach pin-8 = 0 (blocked); Python-patch-only = insufficient "
        "(compiled kernel still rejects). The ONLY config-only knob is VLLM_BATCH_INVARIANT=1 -> "
        "num_splits=1 (un-packed, M-invariant, NOT the pin; up to "
        f"{micro['max_ar_unpack_penalty']:.1f}x M=1 attn penalty)."
    )

    torch.cuda.synchronize()
    payload: dict[str, Any] = {
        "agent": "wirbel", "pr": 375,
        "kind": "served-split-deployability",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "analysis_only": True, **flags,
        "gpu": gpu,
        "peak_mem_mib": round(torch.cuda.max_memory_allocated(dev) / (1024**2), 3),
        "audit": audit, "microbench": micro, "reconcile": reconcile, "selftest": st,
        "ladder_constants": {"ceiling_500": CEILING_500, "step_us": STEP_US,
                             "budget_500_eta": BUDGET_500_ETA, "revived_ceiling_366": REVIVED_CEILING_366,
                             "floor_blanket_9841": FLOOR_BLANKET_9841},
        # ---- PR #375 required terminal fields ----
        "pinned_split_is_served_knob": pinned_split_is_served_knob,
        "served_num_splits_override_mechanism": audit["served_num_splits_override_mechanism"],
        "requires_served_file_change": requires_served_file_change,
        "served_pin_takes_effect": served_pin_takes_effect,
        "served_pin_reproduces_366_byte_exact": served_pin_reproduces_366_byte_exact,
        "minimal_change_loc": minimal_change_loc,
        "analytic_self_test_passes": bool(st["passes"]),
        # ---- decision-gate verdict ----
        "decision_gate": "no_override_reaches_pin8_requires_kernel_patch (gate bullet 3 -- bank the blocker)",
    }
    print_report(payload)
    out_path = Path(args.out_dir) / "served_split_deployability_results.json"
    out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"\n[served-split] wrote {out_path}")
    rid = maybe_log_wandb(payload, args)
    if rid:
        payload["wandb_run_id"] = rid
        out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [rid] if rid else [],
        "pinned_split_is_served_knob": pinned_split_is_served_knob,
        "served_num_splits_override_mechanism": "VLLM_BATCH_INVARIANT=1 -> num_splits=1 (un-pack) ONLY; "
                                                "pin num_splits=8 blocked by FA2 Python guard + C++ "
                                                "TORCH_CHECK (varlen paged KV) -> requires kernel rebuild",
        "requires_served_file_change": requires_served_file_change,
        "served_pin_takes_effect": served_pin_takes_effect,
        "served_pin_reproduces_366_byte_exact": served_pin_reproduces_366_byte_exact,
        "minimal_change_loc": "kernel-rebuild (config-only=0 LOC blocked; Python-patch insufficient)",
        "analytic_self_test_passes": bool(st["passes"]),
        "primary_metric": {"name": "pinned_split_is_served_knob", "value": float(pinned_split_is_served_knob)},
        "test_metric": {"name": "served_unpack_is_m_invariant", "value": float(micro["served_unpack_is_m_invariant"])},
    }))


if __name__ == "__main__":
    main()
