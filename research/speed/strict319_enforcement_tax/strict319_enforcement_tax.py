#!/usr/bin/env python
"""PR #683 — Decompose the +4.84 ms/step strict-#319 *enforcement tax*.

denken cost-model / decomposition card (analysis_only; NO HF Job, NO submission,
official_tps=0, fires=0). Builds on denken #677
(``specdec_amortization_ceiling``, run ``hj2afh4j``) which established, on the
strict int4-Marlin spec-verify path (QAT MTP drafter, K=5 / verify width M=6):

    T_step(K)      = T0_strict + d * K     (graph mode)
    T0_strict      = 12.98 ms  (K->0 verify-base intercept, BI=1 + recompute-rescue)
    d              = 1.421 ms / draft
    T_step(K=5)    = 20.085 ms  (== E*/localTPS = 3.4741/172.74)
    T_AR_deployed  =  8.139 ms  (BI=0, no rescue, M=1; #677 D676 d674_step_wall_us)
    +4.84 ms       = T0_strict - T_AR_deployed   (the *enforcement tax* this card splits)

The enforcement tax has two physically distinct sources:

  * **BI component == the attention-reduction-pin tax.** land #680 (run 5iy1mhe4,
    LOSSLESS_VERIFY_NEEDS_KERNEL) isolated the deployed g=128 / M=6 int4 Marlin verify
    GEMM as byte-identical across verify width M (max_abs_diff=0.0) — the strict-#319
    break is the flash split-KV ATTENTION reduction, propagated through the M-invariant
    body + lm_head, NOT the GEMM. ``VLLM_BATCH_INVARIANT=1`` overrides aten ops only
    (no-op on ``ops.marlin_gemm``; lawine #675 / globalflag #484), so the measured
    T_step(BI=1) − T_step(BI=0) delta is ENTIRELY the cost of pinning that attention
    reduction. The blanket BI=1 (−16% AR) is one way to pin it; ubel #491/#484 price a
    TARGETED attention-only pin at ~5.1% (trending free with #363 fixed-split-KV), so
    we bracket the BI component between the 16% blanket and the 5.1% targeted figure.
  * **recompute-rescue component** — the blanket aten BI pin does not by itself close
    strict-#319 on the deployed flash path (measured: both BI arms remain DIVERGENT with
    rescue off), so the deployed strict submission additionally pays a recompute-rescue
    to restore greedy-token identity. stark #669 is making that rescue cheaper; lawine
    #681 / land #680 (attention pinned losslessly) makes bi_removed=1 realizable.

This card measures the BI component directly on the strict int4-Marlin path at K=5
(BI on/off, recompute-rescue OFF — the ``int4_mtp_batchinv`` submission is the
DIVERGENT no-rescue path, so BI on/off there isolates BI cleanly), parametrizes the
rescue tax ``r_ms`` (does NOT re-measure stark's), and renders the 4-scenario
official-equiv decision table + the ``rescue_tax_ms_for_stock_clears`` threshold.

Local A10G profiling only; numbers are local probes, NOT official a10g-small TPS.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# #677 anchors (denken-owned, banked run hj2afh4j). All graph-mode.
# ---------------------------------------------------------------------------
REF_OFFICIAL_TPS = 126.378       # strict int4 AR baseline to beat (submissions/int4_g128_lmhead, PR #4)
PLUS10_BAR = 136.378             # +10 official bar
LOCAL_TO_OFFICIAL = 0.870        # #677 calibration (local 172.74 -> official 150.3)
PLUS10_LOCAL = PLUS10_BAR / LOCAL_TO_OFFICIAL  # 156.76 local

KSTAR = 5                        # optimal K (verify width M=6)
ESTAR = 3.4741                   # QAT MTP e_accept at K=5 (#677)
LOCALTPS_STRICT_K5 = 172.74491495249353
T0_STRICT_MS = 12.98             # K->0 verify-base intercept (BI=1 + rescue)
D_DRAFT_MS_PER_K = 1.421         # per-draft slope
TSTEP_STRICT_K5_MS = 1000.0 * ESTAR / LOCALTPS_STRICT_K5  # 20.111 (== T0+d*K within rounding)
T_AR_DEPLOYED_MS = 8.138943890282619  # #677 D676 d674_step_wall_us (BI=0, no rescue, M=1)
ENFORCE_TAX_BASE_MS = T0_STRICT_MS - T_AR_DEPLOYED_MS  # 4.841  (the +4.84 this card splits)

# In-scope cheap-drafter operating points (#677 INSCOPE_POINTS), official-equiv tps
# at the *status-quo* strict T_step (full BI + full rescue). Stock just-misses +10,
# top_k64 just-clears. INSCOPE_E_CAP=3.38 is the in-scope acceptance ceiling.
INSCOPE = {
    "stock":   {"e_accept": 3.33, "official_status_quo": 136.12},
    "top_k64": {"e_accept": 3.38, "official_status_quo": 137.14},
}
STOCK_E = INSCOPE["stock"]["e_accept"]
INSCOPE_E_CAP = 3.38

# int4_mtp_batchinv EAGER BI on/off (research/int4_mtp_batchinv/arms; exact strict
# config minus rescue, DIVERGENT both arms). tps = emitted (accepted) tok/s over the
# 16384-token decode. Used ONLY for the RELATIVE BI tax cross-check (eager absolute
# ms is launch-inflated ~3.7x and is NOT the deliverable).
EAGER = {
    "ar_bi0_tps": 16384 / 730.5577533245087,   # 22.427  (M=1, BI=0)
    "ar_bi1_tps": 16384 / 799.1732819080353,   # 20.501  (M=1, BI=1)
    "spec_bi0_tps": 16384 / 339.33888268470764,  # 48.282 (K=6, BI=0)
    "spec_bi1_tps": 16384 / 403.78374671936035,  # 40.577 (K=6, BI=1)
    "spec_k": 6,
    "e_accept_k6": 3.589,   # draft_bi_detax qat K=6 (BI ~invariant: qat 3.589 / qat_rep 3.563)
}
# lawine #675 anchor: blanket VLLM_BATCH_INVARIANT=1 drops AR M=1 tps ~16% (advisor #683).
LAWINE_AR_BI_TPS_DROP = 0.16

# land #680 (run 5iy1mhe4, LOSSLESS_VERIFY_NEEDS_KERNEL): the deployed g=128 / M=6 int4
# Marlin verify GEMM is byte-identical across verify width M (max_abs_diff=0.0 under every
# reduction knob), yet the full-forward M=6 verify logit differs in bits from M=1 AR at
# ~90% of positions -> the strict-#319 break is the flash split-KV ATTENTION reduction
# (propagated through the M-invariant body + lm_head), NOT the GEMM. And VLLM_BATCH_INVARIANT
# overrides aten ops only (no-op on ops.marlin_gemm), so the measured T_step(BI=1)-T_step(BI=0)
# delta is ENTIRELY the cost of pinning that attention reduction. => bi_tax_ms IS the
# attention-reduction-pin tax; the "BI removed?" axis literally means "is the attention
# split-KV reduction pinned losslessly?". The verify-GEMM determinism is free-by-kernel.
VERIFY_GEMM_BYTE_IDENTICAL_ACHIEVABLE = 1
BI_AXIS_PHYSICAL = "attention_split_kv_reduction_pin"

# The blanket BI=1 (-16% AR) is ONE way to pin the attention reduction. ubel #491/#484
# price a TARGETED attention-only pin at ~5.1% tps (~6.4 TPS), trending toward free with
# #363 fixed-split-KV. That targeted pin is the realistic cost of bi_removed=1 (achieve
# determinism via the cheap attention pin, not by dropping determinism). We bracket the
# enforcement-tax BI component between the 16% blanket and this 5.1% targeted figure.
TARGETED_ATTN_PIN_TPS_DROP = 0.051


# ---------------------------------------------------------------------------
# Speed law
# ---------------------------------------------------------------------------
def official_tps(e_accept: float, tstep_ms: float) -> float:
    """#677 calibration: localTPS = 1000*E/tstep_ms ; official = 0.870*localTPS."""
    return LOCAL_TO_OFFICIAL * 1000.0 * e_accept / tstep_ms


def tstep_for_official(e_accept: float, official: float) -> float:
    """Invert official_tps -> the tstep_ms that yields `official` at `e_accept`."""
    return LOCAL_TO_OFFICIAL * 1000.0 * e_accept / official


def eager_relative() -> dict[str, float]:
    """Relative BI tax from the eager int4_mtp arms (mode-portable; absolute is not)."""
    spec_time_infl = EAGER["spec_bi0_tps"] / EAGER["spec_bi1_tps"]   # 1.190
    ar_time_infl = EAGER["ar_bi0_tps"] / EAGER["ar_bi1_tps"]         # 1.094
    return {
        "spec_bi_time_inflation": spec_time_infl,
        "spec_bi_tps_drop": 1.0 - 1.0 / spec_time_infl,              # 0.160
        "ar_bi_time_inflation": ar_time_infl,
        "ar_bi_tps_drop": 1.0 - 1.0 / ar_time_infl,                  # 0.086
    }


# ---------------------------------------------------------------------------
# Decomposition: split TSTEP_STRICT_K5 into raw / bi_tax / rescue (r0)
# ---------------------------------------------------------------------------
def decompose(tstep_raw_ms: float, tstep_bi_ms: float, source: str) -> dict[str, Any]:
    """Given the K=5 graph-mode spec base BI off (raw) and BI on (bi), both
    recompute-rescue OFF, split the strict K=5 step.

      tstep_raw  = T_step_raw          (BI=0, no rescue)   -- the floor
      tstep_bi   = T_step_raw + bi_tax (BI=1, no rescue)
      bi_tax     = tstep_bi - tstep_raw
      r0         = TSTEP_STRICT_K5 - tstep_bi   (realized recompute-rescue on QAT)
    Identity check: tstep_raw + bi_tax + r0 == TSTEP_STRICT_K5.
    """
    bi_tax = tstep_bi_ms - tstep_raw_ms
    r0 = TSTEP_STRICT_K5_MS - tstep_bi_ms
    return {
        "source": source,
        "tstep_raw_ms": tstep_raw_ms,
        "tstep_bi_ms": tstep_bi_ms,
        "bi_tax_ms": bi_tax,
        "bi_tax_frac_of_enforcement": bi_tax / ENFORCE_TAX_BASE_MS,
        "rescue_r0_ms": r0,
        "rescue_r0_frac_of_enforcement": r0 / ENFORCE_TAX_BASE_MS,
        "identity_resid_ms": (tstep_raw_ms + bi_tax + r0) - TSTEP_STRICT_K5_MS,
    }


def base_decomposition(meas_ar: dict[str, Any] | None) -> dict[str, Any]:
    """Split the +4.84 ms *base* (M=1 / K->0) enforcement tax into BI + rescue.

    This is the clean "fraction of +4.84" view: bi_tax_base + rescue_base == 4.84,
    so the fractions sum to 100%. (The K=5 full-step bi_tax is LARGER than
    bi_tax_base because the bf16 drafter's K forwards are fully BI-taxed on top of
    the verify base; rescue is verify-side / drafter-independent so it is the same
    at the base and at K=5.)

    measured: bi_tax_base = T_ar(BI=1) - T_ar(BI=0) from the AR M=1 pair.
    fallback: band from [eager AR -8.6% ... lawine #675 -16%] on the 8.139 ms AR step.
    """
    rel = eager_relative()
    lo = T_AR_DEPLOYED_MS * (rel["ar_bi_time_inflation"] - 1.0)               # ~0.765
    hi = T_AR_DEPLOYED_MS * (1.0 / (1.0 - LAWINE_AR_BI_TPS_DROP) - 1.0)       # ~1.551
    if meas_ar is not None:
        bi_base = meas_ar["bi_tax_base_ms"]
        source = "measured_graph"
    else:
        bi_base = 0.5 * (lo + hi)
        source = "cost_model_band_mid"
    rescue_base = ENFORCE_TAX_BASE_MS - bi_base
    return {
        "source": source,
        "bi_tax_base_ms": bi_base,
        "bi_tax_base_frac_of_enforcement": bi_base / ENFORCE_TAX_BASE_MS,
        "rescue_base_ms": rescue_base,
        "rescue_base_frac_of_enforcement": rescue_base / ENFORCE_TAX_BASE_MS,
        "band_eager_lo_ms": lo, "band_lawine_hi_ms": hi,
    }


def cost_model_decomposition() -> dict[str, Any]:
    """Cost-model fallback for (tstep_raw, tstep_bi) when no graph measurement.

    Anchors the spec BI *fraction* from the eager arms (+19% time / -16% tps,
    mode-portable) onto the QAT strict K=5 step, and brackets the realized rescue
    r0 from the AR-base BI tax band [eager -8.6%, lawine -16%].

    Two unknowns (tstep_raw, r0) -> 1 ratio (tstep_bi = 1.19*tstep_raw) + the AR-base
    BI band pins a bracket. We report a band; the graph measurement supersedes it.
    """
    rel = eager_relative()
    spec_infl = rel["spec_bi_time_inflation"]   # 1.190
    # AR-base BI tax band -> base bi component in ms on the 8.139 ms AR step.
    base_bi_lo = T_AR_DEPLOYED_MS * (rel["ar_bi_time_inflation"] - 1.0)        # ~0.765 (eager -8.6%)
    base_bi_hi = T_AR_DEPLOYED_MS * (1.0 / (1.0 - LAWINE_AR_BI_TPS_DROP) - 1.0)  # ~1.551 (lawine -16%)
    # Spec K=5 bi_tax >= base bi (drafter, bf16, fully BI-taxed, adds on top). The
    # drafter-side BI tax scales the spec inflation: tstep_bi = spec_infl * tstep_raw.
    # Solve with TSTEP_STRICT_K5 = tstep_raw + bi_tax + r0, bi_tax=(spec_infl-1)*tstep_raw.
    # Bracket r0 in [base_bi_hi ... TSTEP-... ]: take rescue ~ enforcement - base_bi as
    # the central rescue prior, giving a central tstep_raw.
    out: dict[str, Any] = {"source": "cost_model", "eager_relative": rel,
                           "base_bi_lo_ms": base_bi_lo, "base_bi_hi_ms": base_bi_hi}
    band = {}
    for name, base_bi in (("lo", base_bi_lo), ("hi", base_bi_hi)):
        # central prior: rescue r0 ~ enforcement_base - base_bi (rescue is the
        # non-BI remainder of the +4.84 base tax, verify-side, drafter-independent).
        r0 = ENFORCE_TAX_BASE_MS - base_bi
        tstep_bi = TSTEP_STRICT_K5_MS - r0
        tstep_raw = tstep_bi / spec_infl
        band[name] = decompose(tstep_raw, tstep_bi, f"cost_model_{name}")
    out["band"] = band
    # point estimate = midpoint of the bi_tax band
    bi_mid = 0.5 * (band["lo"]["bi_tax_ms"] + band["hi"]["bi_tax_ms"])
    r0_mid = 0.5 * (band["lo"]["rescue_r0_ms"] + band["hi"]["rescue_r0_ms"])
    tstep_bi_mid = TSTEP_STRICT_K5_MS - r0_mid
    tstep_raw_mid = tstep_bi_mid - bi_mid
    point = decompose(tstep_raw_mid, tstep_bi_mid, "cost_model_point")
    point["base"] = base_decomposition(None)
    out["point"] = point
    return out


# ---------------------------------------------------------------------------
# Cheap targeted-attention-pin bracket (ubel #491/#484)
# ---------------------------------------------------------------------------
def cheap_attn_pin_ms(decomp: dict[str, Any]) -> float:
    """Targeted attention-only pin tax on the spec K=5 step (the realistic cost of
    bi_removed=1 via ubel #491/#484, ~5.1% tps vs the 16% blanket BI=1).

    Scaled from the MEASURED blanket bi_tax by the price ratio (5.1% / 16%). This is a
    CONSERVATIVE (upper-bound) cheap pin: the measured blanket bi_tax is drafter-inclusive
    (BI=1 also taxes the bf16 drafter's aten ops), whereas the targeted pin touches only
    the verify-side split-KV attention reduction, so the true cheap pin is <= this and
    trends toward 0 with #363 fixed-split-KV. Using the upper bound keeps the threshold
    and the BI_REMOVAL_SUFFICIENT verdict conservative.
    """
    scale = TARGETED_ATTN_PIN_TPS_DROP / LAWINE_AR_BI_TPS_DROP   # 0.051 / 0.16 = 0.319
    return decomp["bi_tax_ms"] * scale


# ---------------------------------------------------------------------------
# Decision table (4 scenarios) + stock-clears threshold
# ---------------------------------------------------------------------------
def _stock_raw_tstep(decomp: dict[str, Any]) -> float:
    """Stock-drafter raw step = stock status-quo step minus (bi_tax + r0).

    Stock status-quo step embeds the SAME enforcement tax (BI=1 + rescue) as QAT;
    rescue r0 is verify-side (drafter-independent) so it transfers exactly; bi_tax
    transfers as absolute ms (verify-side part exact; drafter-side part is the QAT
    approximation -- flagged). The stock drafter is ~1.2 ms/step heavier than QAT,
    which is captured by anchoring on the #677 stock official point (136.12).
    """
    stock_sq = tstep_for_official(STOCK_E, INSCOPE["stock"]["official_status_quo"])  # 21.28 ms
    return stock_sq - decomp["bi_tax_ms"] - decomp["rescue_r0_ms"]


def decision_table(decomp: dict[str, Any], r_ms_values: list[float],
                   e_accept: float = STOCK_E) -> dict[str, Any]:
    """4-scenario official-equiv ceiling + margin over +10, as a function of r_ms.

    Scenarios (cheap stock drafter, K=5). "BI removed" == the attention split-KV
    reduction is pinned by a means OTHER than the blanket BI=1 (land #680):
      S0 status quo         : T = raw + bi_tax    + r0      (blanket attn pin, full rescue)
      S1 width-inv only     : T = raw + r0                  (attn determinism FREE, rescue kept)
      S1_cheap_attn_pin     : T = raw + bi_cheap  + r0      (TARGETED attn pin ~5.1%, rescue kept)
      S2 cheaper rescue only: T = raw + bi_tax    + r_ms    (blanket attn pin, cheaper rescue)
      S3 both               : T = raw + r_ms                (attn determinism FREE, cheaper rescue)
    raw = stock-drafter raw step. r0 = realized status-quo rescue. S1 is the optimistic
    (determinism-free) bound; S1_cheap_attn_pin is the realistic bi_removed=1 cost via the
    ubel #491/#484 targeted pin (the one that actually competes).
    """
    raw = _stock_raw_tstep(decomp)
    bi = decomp["bi_tax_ms"]
    bi_cheap = cheap_attn_pin_ms(decomp)
    r0 = decomp["rescue_r0_ms"]

    def row(tstep: float) -> dict[str, float]:
        o = official_tps(e_accept, tstep)
        return {"tstep_ms": tstep, "official_tps": o, "margin_over_plus10": o - PLUS10_BAR,
                "clears_plus10": o >= PLUS10_BAR}

    s0 = row(raw + bi + r0)
    s1 = row(raw + r0)
    s1_cheap = row(raw + bi_cheap + r0)
    table = {
        "e_accept": e_accept, "stock_raw_tstep_ms": raw, "bi_tax_ms": bi,
        "bi_tax_cheap_attn_pin_ms": bi_cheap, "rescue_r0_ms": r0,
        "S0_status_quo": s0,
        "S1_width_inv_only": s1,
        "S1_cheap_attn_pin": s1_cheap,
        "S2_cheaper_rescue_only": {f"r_ms={r:g}": row(raw + bi + r) for r in r_ms_values},
        "S3_both": {f"r_ms={r:g}": row(raw + r) for r in r_ms_values},
    }
    return table


def rescue_tax_for_stock_clears(decomp: dict[str, Any]) -> dict[str, Any]:
    """Threshold r_ms (at K=5, BI removed) below which stock (E=3.33) clears +10.

    870*STOCK_E/(raw + attn_pin + r_ms) >= 136.378
      r_ms <= tstep_for_official(STOCK_E, PLUS10_BAR) - raw - attn_pin

    HEADLINE (advisor #683): reported against the CHEAP targeted attention pin
    (attn_pin = bi_cheap), the determinism mechanism that actually competes. Also
    reported determinism-FREE (attn_pin=0, optimistic) and blanket-BI-kept (attn_pin=
    bi_tax, pessimistic) as the bracket ends. RESCUE_BOUND keys off the best case (free).
    """
    raw = _stock_raw_tstep(decomp)
    bi = decomp["bi_tax_ms"]
    bi_cheap = cheap_attn_pin_ms(decomp)
    budget = tstep_for_official(STOCK_E, PLUS10_BAR)   # 21.243 ms
    thresh_cheap = budget - raw - bi_cheap
    thresh_free = budget - raw
    thresh_blanket = budget - raw - bi
    return {
        "stock_raw_tstep_ms": raw,
        "tstep_budget_for_plus10_ms": budget,
        "bi_tax_cheap_attn_pin_ms": bi_cheap,
        "rescue_tax_ms_for_stock_clears": thresh_cheap,                # headline (cheap pin)
        "rescue_tax_ms_for_stock_clears_bi_free": thresh_free,         # optimistic bracket
        "rescue_tax_ms_for_stock_clears_blanket_bi": thresh_blanket,   # pessimistic bracket
        "stock_clears_at_zero_rescue": thresh_free >= 0.0,             # S3 best case -> RESCUE_BOUND
        "stock_clears_at_zero_rescue_cheap_pin": thresh_cheap >= 0.0,
    }


def verdict(decomp: dict[str, Any], thresh: dict[str, Any]) -> dict[str, Any]:
    """STRICT319_TAX_DECOMPOSED + BI_REMOVAL_SUFFICIENT / RESCUE_BOUND flags."""
    raw = _stock_raw_tstep(decomp)
    r0 = decomp["rescue_r0_ms"]
    bi_cheap = cheap_attn_pin_ms(decomp)
    # S1: width-invariance ONLY (attn determinism free, rescue unchanged at r0) -> clears?
    s1_official = official_tps(STOCK_E, raw + r0)
    # Realistic S1: bi_removed=1 via the cheap targeted attention pin (the competing one).
    s1_cheap_official = official_tps(STOCK_E, raw + bi_cheap + r0)
    bi_removal_sufficient = s1_official >= PLUS10_BAR
    bi_removal_sufficient_cheap_pin = s1_cheap_official >= PLUS10_BAR
    # RESCUE_BOUND: even both unlocks (S3 at r_ms=0, determinism free) can't make stock clear.
    rescue_bound = not thresh["stock_clears_at_zero_rescue"]
    flags = ["STRICT319_TAX_DECOMPOSED"]
    if bi_removal_sufficient:
        flags.append("BI_REMOVAL_SUFFICIENT")
    if rescue_bound:
        flags.append("RESCUE_BOUND")
    return {
        "verdict": "|".join(flags),
        "bi_removal_sufficient": bi_removal_sufficient,
        "bi_removal_sufficient_cheap_pin": bi_removal_sufficient_cheap_pin,
        "s1_width_inv_only_official": s1_official,
        "s1_margin_over_plus10": s1_official - PLUS10_BAR,
        "s1_cheap_attn_pin_official": s1_cheap_official,
        "s1_cheap_attn_pin_margin_over_plus10": s1_cheap_official - PLUS10_BAR,
        "rescue_bound": rescue_bound,
    }


def build_card(decomp: dict[str, Any], r_ms_values: list[float]) -> dict[str, Any]:
    table = decision_table(decomp, r_ms_values)
    thresh = rescue_tax_for_stock_clears(decomp)
    vrd = verdict(decomp, thresh)
    return {
        "anchors": {
            "ref_official_tps": REF_OFFICIAL_TPS, "plus10_bar": PLUS10_BAR,
            "local_to_official": LOCAL_TO_OFFICIAL, "plus10_local": PLUS10_LOCAL,
            "kstar": KSTAR, "estar": ESTAR, "tstep_strict_k5_ms": TSTEP_STRICT_K5_MS,
            "t_ar_deployed_ms": T_AR_DEPLOYED_MS, "enforcement_tax_base_ms": ENFORCE_TAX_BASE_MS,
            "bi_axis_physical": BI_AXIS_PHYSICAL,
            "verify_gemm_byte_identical_achievable": VERIFY_GEMM_BYTE_IDENTICAL_ACHIEVABLE,
            "blanket_bi_ar_tps_drop": LAWINE_AR_BI_TPS_DROP,
            "targeted_attn_pin_tps_drop": TARGETED_ATTN_PIN_TPS_DROP,
            "land680_run": "5iy1mhe4",
        },
        "decomposition": decomp,
        "decision_table": table,
        "threshold": thresh,
        "verdict": vrd,
    }


# ---------------------------------------------------------------------------
# Graph-mode measurement (boots the int4_mtp_batchinv submission; local only)
# ---------------------------------------------------------------------------
def _import_serve_profile():
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.local_validation import serve_profile  # noqa: E402
    return serve_profile


def measure_arm(label: str, k: int, bi: int, *, server_python: Path, out_dir: Path,
                num_prompts: int, output_len: int) -> dict[str, Any]:
    """Boot int4_mtp_batchinv at (K=k, VLLM_BATCH_INVARIANT=bi), graph mode, and
    derive T_step from the prefill-isolated single-stream probe + e_accept.

    k=0 -> NUM_SPECULATIVE_TOKENS=0 -> plain int4 M=1 AR (e_accept=1). rescue is OFF
    in this submission for all arms (it is the DIVERGENT no-rescue path).
    """
    sp = _import_serve_profile()
    out_dir.mkdir(parents=True, exist_ok=True)
    submission = REPO_ROOT / "submissions" / "int4_mtp_batchinv"
    extra_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "VLLM_BATCH_INVARIANT": str(bi),
        "NUM_SPECULATIVE_TOKENS": str(k),
        "DRAFTER_MODEL": os.environ.get("DRAFTER_MODEL",
                                        "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant"),
        "MAX_MODEL_LEN": "4096",
        # graph mode: ENFORCE_EAGER left unset (defaults 0) so CUDA graphs are ON.
    }
    t0 = time.time()
    res = sp.run_timing_pass(
        submission, server_python, out_dir, label,
        num_prompts=num_prompts, output_len=output_len, extra_env=extra_env,
    )
    res["boot_decode_wall_s"] = time.time() - t0
    # e_accept: Prometheus first, then server-log fallback; AR arm -> 1.0
    e_accept = None
    sm = res.get("spec_metrics") or {}
    if isinstance(sm, dict):
        e_accept = sm.get("e_accept_mean_acceptance_length")
    sl = res.get("spec_log") or {}
    if e_accept is None and isinstance(sl, dict):
        e_accept = sl.get("e_accept_exact") or sl.get("e_accept_interval_mean")
    if k == 0:
        e_accept = 1.0   # M=1 AR: one token per step by definition
    probe = res.get("tps_probe") or {}
    decode_tps = probe.get("decode_tps_single_stream") if isinstance(probe, dict) else None
    tstep_probe_ms = (1000.0 * e_accept / decode_tps) if (e_accept and decode_tps) else None
    # decode-summary cross-check (includes a little prefill -> upper bound)
    summ = res.get("decode_summary") or {}
    dur = summ.get("duration_s")
    ntok = summ.get("num_completion_tokens")
    tps_decode = (ntok / dur) if (dur and ntok) else None
    tstep_decode_ms = (1000.0 * e_accept / tps_decode) if (e_accept and tps_decode) else None
    return {
        "label": label, "k": k, "bi": bi, "e_accept": e_accept,
        "decode_tps_single_stream": decode_tps, "tstep_probe_ms": tstep_probe_ms,
        "decode_tps_summary": tps_decode, "tstep_decode_ms": tstep_decode_ms,
        "num_completion_tokens": ntok, "duration_s": dur,
        "steptime": res.get("steptime"), "boot_decode_wall_s": res["boot_decode_wall_s"],
        "server_log": res.get("server_log"),
    }


def _arm_tstep_ms(arm: dict[str, Any]) -> tuple[float | None, str]:
    """Pick the #677-comparable per-step time: eval-workload decode (long-seq,
    prefill-negligible) when the run is long enough, else the short-seq probe.

    Both arms share the identical probe/decode workload, so the BI *difference*
    is basis-independent; the eval-decode basis is what matches the #677 20.11 ms
    strict K=5 anchor for the absolute floor.
    """
    ntok = arm.get("num_completion_tokens") or 0
    td = arm.get("tstep_decode_ms")
    if ntok >= 800 and td and 3.0 < td < 80.0:
        return td, "eval_decode"
    return arm.get("tstep_probe_ms"), "probe"


def run_measurement(arms: str, *, server_python: Path, out_dir: Path,
                    num_prompts: int, output_len: int) -> dict[str, Any]:
    """Measure (raw, bi) at K=5; optionally the AR M=1 base pair (lawine check)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    plan = [("spec_bi0", KSTAR, 0), ("spec_bi1", KSTAR, 1)]
    if arms == "all":
        plan += [("ar_bi0", 0, 0), ("ar_bi1", 0, 1)]
    for label, k, bi in plan:
        arm = measure_arm(label, k, bi, server_python=server_python,
                          out_dir=out_dir, num_prompts=num_prompts, output_len=output_len)
        results[label] = arm
        (out_dir / f"arm_{label}.json").write_text(json.dumps(arm, indent=2, default=str))
        print(f"[arm {label}] e_accept={arm['e_accept']} "
              f"tstep_decode={arm['tstep_decode_ms']} tstep_probe={arm['tstep_probe_ms']} "
              f"ntok={arm['num_completion_tokens']}", flush=True)
    return results


def measured_decomposition(meas: dict[str, Any]) -> dict[str, Any]:
    """Hybrid decomposition: MEASURED graph-mode BI *time-inflation* relatives
    applied onto #677's absolute anchors, with rescue as the verify-side remainder.

    Why relatives, not raw absolutes: the int4_mtp serve-path on an 8-prompt eval
    subset runs ~1.3x slower per step than #677's fuller-set basis, so a naive
    tstep_bi - tstep_raw mixes bases and yields a (nonphysical) negative rescue.
    The BI *time inflation* (ratio) is basis-portable (probe & eval-decode agree to
    <0.5% for spec), and rescue is verify-side / drafter-independent, so:

      bi_tax_base = T_AR_deployed * (ar_time_inflation - 1)         [M=1, E=1 clean]
      rescue r0   = +4.84 - bi_tax_base                            [verify-side]
      tstep_bi(K5)= TSTEP_STRICT_K5 - r0                           [BI=1, no rescue]
      tstep_raw   = tstep_bi(K5) / spec_time_inflation             [BI=0, no rescue]
      bi_tax(K5)  = tstep_bi(K5) - tstep_raw
    Identity: tstep_raw + bi_tax(K5) + r0 == TSTEP_STRICT_K5.
    """
    sb0, sb1 = meas["spec_bi0"], meas["spec_bi1"]
    spec_raw, _ = _arm_tstep_ms(sb0)
    spec_bi, _ = _arm_tstep_ms(sb1)
    spec_infl = spec_bi / spec_raw
    rel = {
        "spec_time_inflation_eval_decode": spec_infl,
        "spec_time_inflation_probe": sb1["tstep_probe_ms"] / sb0["tstep_probe_ms"],
        "spec_bi_tps_drop": 1.0 - 1.0 / spec_infl,
    }
    if "ar_bi0" in meas and "ar_bi1" in meas:
        ar0, _ = _arm_tstep_ms(meas["ar_bi0"])
        ar1, _ = _arm_tstep_ms(meas["ar_bi1"])
        ar_infl = ar1 / ar0
        rel["ar_time_inflation_eval_decode"] = ar_infl
        rel["ar_time_inflation_probe"] = (meas["ar_bi1"]["tstep_probe_ms"]
                                          / meas["ar_bi0"]["tstep_probe_ms"])
        rel["ar_bi_tps_drop"] = 1.0 - 1.0 / ar_infl
        rel["ar_basis"] = "measured_graph_eval_decode_E1"
    else:
        ar_infl = 1.0 / (1.0 - LAWINE_AR_BI_TPS_DROP)   # lawine #675 fallback
        rel["ar_time_inflation_eval_decode"] = None
        rel["ar_basis"] = "lawine_675_fallback"

    bi_tax_base = T_AR_DEPLOYED_MS * (ar_infl - 1.0)
    r0 = ENFORCE_TAX_BASE_MS - bi_tax_base
    tstep_bi = TSTEP_STRICT_K5_MS - r0
    tstep_raw = tstep_bi / spec_infl
    decomp = decompose(tstep_raw, tstep_bi, "measured_hybrid")
    decomp["measured_relatives"] = rel
    decomp["measured_arms_eval_decode_ms"] = {
        "spec_raw": spec_raw, "spec_bi": spec_bi,
        "ar_raw": meas.get("ar_bi0", {}).get("tstep_decode_ms"),
        "ar_bi": meas.get("ar_bi1", {}).get("tstep_decode_ms"),
    }
    decomp["measured_arms_probe_ms"] = {
        "spec_raw": sb0["tstep_probe_ms"], "spec_bi": sb1["tstep_probe_ms"],
    }
    decomp["e_accept_measured"] = {"spec_raw": sb0["e_accept"], "spec_bi": sb1["e_accept"]}
    meas_ar = {"bi_tax_base_ms": bi_tax_base, "ar_time_inflation": ar_infl,
               "ar_bi_tps_drop": 1.0 - 1.0 / ar_infl,
               "basis": "measured_graph_relative_on_677_AR_8.14ms"}
    decomp["ar_base"] = meas_ar
    decomp["base"] = base_decomposition(meas_ar)
    return decomp


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------
def log_wandb(card: dict[str, Any], wandb_name: str | None, wandb_group: str | None,
              meas: dict[str, Any] | None) -> str | None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] unavailable: {exc}", flush=True)
        return None
    decomp = card["decomposition"]
    thr = card["threshold"]
    vrd = card["verdict"]
    tbl = card["decision_table"]
    scalars: dict[str, Any] = {
        "analysis_only": 1, "official_tps": 0, "fires": 0,
        "bi_tax_ms": decomp["bi_tax_ms"],
        "bi_tax_frac_of_enforcement": decomp["bi_tax_frac_of_enforcement"],
        "bi_tax_base_ms": decomp["base"]["bi_tax_base_ms"],
        "bi_tax_base_frac_of_enforcement": decomp["base"]["bi_tax_base_frac_of_enforcement"],
        "rescue_base_ms": decomp["base"]["rescue_base_ms"],
        "rescue_r0_ms": decomp["rescue_r0_ms"],
        "T_step_raw_ms": decomp["tstep_raw_ms"],
        "T_step_bi_ms": decomp["tstep_bi_ms"],
        "tstep_strict_k5_ms": TSTEP_STRICT_K5_MS,
        "enforcement_tax_base_ms": ENFORCE_TAX_BASE_MS,
        # headline threshold == reported against the CHEAP targeted attention pin (#683)
        "rescue_tax_ms_for_stock_clears": thr["rescue_tax_ms_for_stock_clears"],
        "rescue_tax_ms_for_stock_clears_bi_free": thr["rescue_tax_ms_for_stock_clears_bi_free"],
        "rescue_tax_ms_for_stock_clears_blanket_bi": thr["rescue_tax_ms_for_stock_clears_blanket_bi"],
        "stock_raw_tstep_ms": thr["stock_raw_tstep_ms"],
        "bi_removal_sufficient": int(vrd["bi_removal_sufficient"]),
        "bi_removal_sufficient_cheap_pin": int(vrd["bi_removal_sufficient_cheap_pin"]),
        "rescue_bound": int(vrd["rescue_bound"]),
        "s1_width_inv_only_official": vrd["s1_width_inv_only_official"],
        "s1_cheap_attn_pin_official": vrd["s1_cheap_attn_pin_official"],
        "s1_cheap_attn_pin_margin_over_plus10": vrd["s1_cheap_attn_pin_margin_over_plus10"],
        # attention-pin axis durable facts (land #680, run 5iy1mhe4)
        "bi_axis_physical": BI_AXIS_PHYSICAL,
        "verify_gemm_byte_identical_achievable": VERIFY_GEMM_BYTE_IDENTICAL_ACHIEVABLE,
        "blanket_bi_ar_tps_drop": LAWINE_AR_BI_TPS_DROP,
        "targeted_attn_pin_tps_drop": TARGETED_ATTN_PIN_TPS_DROP,
        "bi_tax_cheap_attn_pin_ms": thr["bi_tax_cheap_attn_pin_ms"],
        "decomp_source": decomp["source"],
        "ref_official_tps": REF_OFFICIAL_TPS, "plus10_bar": PLUS10_BAR,
        "local_to_official": LOCAL_TO_OFFICIAL, "plus10_local": PLUS10_LOCAL,
        "verdict": vrd["verdict"],
        # per-scenario official ceilings (stock E=3.33)
        "S0_status_quo_official": tbl["S0_status_quo"]["official_tps"],
        "S0_status_quo_margin": tbl["S0_status_quo"]["margin_over_plus10"],
        "S1_width_inv_official": tbl["S1_width_inv_only"]["official_tps"],
        "S1_width_inv_margin": tbl["S1_width_inv_only"]["margin_over_plus10"],
        "S1_cheap_attn_pin_official": tbl["S1_cheap_attn_pin"]["official_tps"],
        "S1_cheap_attn_pin_margin": tbl["S1_cheap_attn_pin"]["margin_over_plus10"],
    }
    if meas and "ar_base" in card["decomposition"]:
        ab = card["decomposition"]["ar_base"]
        scalars["bi_tax_base_ms"] = ab["bi_tax_base_ms"]
        scalars["bi_base_tps_drop"] = ab.get("ar_bi_tps_drop", ab.get("bi_base_tps_drop"))
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        name=wandb_name or "denken/strict319-enforcement-tax",
        group=wandb_group or "strict319-enforcement-tax-denken",
        config={"pr": 683, "card": "strict319_enforcement_tax", "kstar": KSTAR},
    )
    wandb.log(scalars)
    wandb.summary.update(scalars)
    if meas:
        wandb.summary.update({"measurement": json.dumps(meas, default=str)})
    wandb.summary.update({"card": json.dumps(card, default=str)})
    rid = run.id
    run.finish()
    return rid


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def self_test() -> None:
    tol = 1e-6
    # 1. speed-law round-trip
    o = official_tps(ESTAR, TSTEP_STRICT_K5_MS)
    assert abs(o - 150.30) < 0.5, o
    assert abs(tstep_for_official(ESTAR, o) - TSTEP_STRICT_K5_MS) < 1e-6
    # 2. enforcement base tax == +4.84
    assert abs(ENFORCE_TAX_BASE_MS - 4.841) < 0.01, ENFORCE_TAX_BASE_MS
    # 3. T0 + d*K consistency with E/localTPS
    assert abs((T0_STRICT_MS + D_DRAFT_MS_PER_K * KSTAR) - TSTEP_STRICT_K5_MS) < 0.05
    # 4. decompose identity
    d = decompose(13.0, 14.0, "test")
    assert abs(d["identity_resid_ms"]) < tol, d
    assert abs(d["bi_tax_ms"] - 1.0) < tol
    assert abs(d["rescue_r0_ms"] - (TSTEP_STRICT_K5_MS - 14.0)) < tol
    # 5. stock status-quo official reproduces #677 point
    stock_sq = tstep_for_official(STOCK_E, INSCOPE["stock"]["official_status_quo"])
    assert abs(official_tps(STOCK_E, stock_sq) - 136.12) < 1e-6
    # 6. stock just-misses, top_k64 just-clears at status quo
    assert INSCOPE["stock"]["official_status_quo"] < PLUS10_BAR
    assert INSCOPE["top_k64"]["official_status_quo"] > PLUS10_BAR
    # 7. threshold monotonicity: smaller raw -> larger allowable rescue
    d_big = decompose(11.0, 13.5, "t")   # bi=2.5, big tax
    d_small = decompose(13.5, 14.0, "t")  # bi=0.5, small tax
    th_big = rescue_tax_for_stock_clears(d_big)
    th_small = rescue_tax_for_stock_clears(d_small)
    # bigger bi_tax removed -> smaller stock_raw -> larger threshold
    assert th_big["rescue_tax_ms_for_stock_clears"] > th_small["rescue_tax_ms_for_stock_clears"]
    # 8. cost-model band ordering & identity
    cm = cost_model_decomposition()
    assert cm["band"]["lo"]["bi_tax_ms"] < cm["band"]["hi"]["bi_tax_ms"]
    for k in ("lo", "hi", "point" if "point" in cm else "lo"):
        bb = cm["band"].get(k) or cm.get(k)
        assert abs(bb["identity_resid_ms"]) < 1e-6
    # 9. card builds & verdict well-formed
    card = build_card(cm["point"], [0.0, 1.5, 3.0])
    assert card["verdict"]["verdict"].startswith("STRICT319_TAX_DECOMPOSED")
    assert "S0_status_quo" in card["decision_table"]
    assert "S1_cheap_attn_pin" in card["decision_table"]
    # 10. eager relative sanity (+19% spec, +9.4% AR)
    rel = eager_relative()
    assert abs(rel["spec_bi_tps_drop"] - 0.160) < 0.005, rel
    assert abs(rel["ar_bi_tps_drop"] - 0.086) < 0.005, rel
    # 11. cheap attention pin bracket: 0 < cheap < blanket; thresholds & S1 ordered
    dd = decompose(13.0, 17.0, "t")   # bi_tax = 4.0
    bc = cheap_attn_pin_ms(dd)
    assert 0.0 < bc < dd["bi_tax_ms"], bc
    assert abs(bc - dd["bi_tax_ms"] * (0.051 / 0.16)) < 1e-9
    th = rescue_tax_for_stock_clears(dd)
    # blanket-kept (pessimistic) < cheap pin (headline) < determinism-free (optimistic)
    assert (th["rescue_tax_ms_for_stock_clears_blanket_bi"]
            < th["rescue_tax_ms_for_stock_clears"]
            < th["rescue_tax_ms_for_stock_clears_bi_free"]), th
    vd = verdict(dd, th)
    # S0 (blanket) <= S1_cheap (targeted pin) <= S1 (free), all at stock E
    assert (card["decision_table"]["S0_status_quo"]["official_tps"]
            <= card["decision_table"]["S1_cheap_attn_pin"]["official_tps"]
            <= card["decision_table"]["S1_width_inv_only"]["official_tps"])
    assert vd["s1_cheap_attn_pin_official"] <= vd["s1_width_inv_only_official"]
    print("SELF-TEST PASS (11 checks)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-gpu", action="store_true", help="cost-model card only (no boot)")
    ap.add_argument("--measure", action="store_true", help="boot arms to ground bi_tax/raw")
    ap.add_argument("--from-arms", default=None, help="rebuild card from saved arm_*.json (no boot)")
    ap.add_argument("--smoke", action="store_true", help="tiny 1-arm boot to validate path")
    ap.add_argument("--arms", choices=["spec", "all"], default="spec")
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--server-python", default=str(REPO_ROOT / ".venv" / "bin" / "python"))
    ap.add_argument("--out-dir", default="/tmp/strict319_tax_arms")
    ap.add_argument("--r-ms", default="0.0,1.5,3.0", help="comma rescue tax values for table")
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_group", default="strict319-enforcement-tax-denken")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--log-card", default=None, help="load a saved card JSON and log to W&B (no boot)")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if args.log_card:
        card = json.loads(Path(args.log_card).read_text())
        rid = log_wandb(card, args.wandb_name, args.wandb_group, card.get("measurement"))
        print(f"[wandb] run id: {rid}")
        return

    out_dir = Path(args.out_dir)
    server_python = Path(args.server_python)
    r_ms_values = [float(x) for x in args.r_ms.split(",") if x.strip()]

    if args.smoke:
        m = measure_arm("smoke_spec_bi0", KSTAR, 0, server_python=server_python,
                        out_dir=out_dir, num_prompts=2, output_len=64)
        print(json.dumps(m, indent=2, default=str))
        return

    meas = None
    if args.from_arms:
        meas = {}
        for f in sorted(Path(args.from_arms).glob("arm_*.json")):
            label = f.stem.replace("arm_", "")
            meas[label] = json.loads(f.read_text())
        if "spec_bi0" not in meas or "spec_bi1" not in meas:
            raise SystemExit(f"need spec_bi0/spec_bi1 arm JSONs in {args.from_arms}")
        decomp = measured_decomposition(meas)
    elif args.measure:
        meas = run_measurement(args.arms, server_python=server_python, out_dir=out_dir,
                               num_prompts=args.num_prompts, output_len=args.output_len)
        decomp = measured_decomposition(meas)
    else:
        decomp = cost_model_decomposition()["point"]

    card = build_card(decomp, r_ms_values)
    if meas:
        card["measurement"] = meas
    # persist the card next to the harness (small JSON; server logs stay in /tmp)
    card_path = HERE / "strict319_enforcement_tax.json"
    card_path.write_text(json.dumps(card, indent=2, default=str))
    print(json.dumps(card, indent=2, default=str))
    print(f"\n[card] wrote {card_path}")

    if not args.no_wandb:
        rid = log_wandb(card, args.wandb_name, args.wandb_group, meas)
        if rid:
            print(f"[wandb] run id: {rid}")


if __name__ == "__main__":
    main()
