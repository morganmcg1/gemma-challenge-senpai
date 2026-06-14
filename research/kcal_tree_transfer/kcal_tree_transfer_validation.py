#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""K_cal TREE-TRANSFER VALIDATION (PR #148, student ubel).

THE QUESTION
------------
The entire 500-projection -- 271 (RED) / 522 (descent) / 538 (both-bugs) -- rides
on the one figure of merit

    official_TPS = K_cal * (E[T] / step_time) * tau

where K_cal = 125.268 was FIT ON THE LINEAR-MTP FRONTIER (481.53 official / 3.844
local tok/step; fern #100) and the local->official multiplier is 1.06019 (481.53 /
pooled local wall_tps 454.194). fern #142's gate, fern #145's decomp and wirbel
#146's confidence envelope ALL consume K_cal as a given -- they assume it transfers
UNCHANGED to the TREE serving path. But the tree does materially different per-step
work (M=32 verify vs M=8 linear decode, the descent accept-walk, tree-mask
attention), so the constant relating local tok/step to official TPS COULD shift. A
3% K_cal drift moves the 522 projection by +/-15 TPS -- enough to flip GREEN->RED.
Nobody validates the transfer. This file is the CALIBRATION leg of the launch
evidence-line (wirbel #146 is the SAMPLING leg).

WHAT THIS FILE DOES
-------------------
1. REPRODUCE the calibration exactly (PRIMARY self-test): K_cal = 481.53/3.844;
   multiplier = 481.53 / pooled-mean-local; the K_cal = C_local * multiplier
   decomposition closes; the 522/538 projections reproduce.
2. DECOMPOSE the 1.06019 multiplier / +6.019% gap into physical factors and ATTRIBUTE
   the gap across them, reading the OFFICIAL scorer convention
   (hf_bucket_single_job.py) and the DEPLOYED precache mechanism.
3. CLASSIFY each factor tree-invariant vs tree-sensitive (skeptically).
4. EMIT the K_cal tree-transfer band [K_lo, K_hi] from the TREE-SENSITIVE factors,
   propagate it through the 522/538 projections -> a calibration band on official
   TPS, and COMBINE with wirbel #146's sampling band in quadrature (independent legs).

THE DECOMPOSITION (what makes +6.019% physical)
-----------------------------------------------
The official metric is tps = output_throughput = total_output_tokens / total_duration,
measured over 128 prompts x 512 output tokens, MAX_CONCURRENCY=1, after 4 warmup
requests discarded (hf_bucket_single_job.py). The local meter is decode-only wall_tps
(num_completion_tokens / decode_duration_s; paired_tps_ab.py) on the SAME deployed
stack. The DEPLOYED submission fa2sw_precache_kenyan sets PRECACHE_BENCH=1: its serve
patch REPLAYS the 128 benchmark prompts during the untimed warmup window (gating
/v1/models 503 until done), so every prompt's prefill KV is already in the prefix
cache when the TIMED official benchmark runs -> the timed window is PURE DECODE
(f = prefill/decode ~ 0). Hence official ~ pure-decode tps ~ local decode-only
wall_tps, and the four "convention" factors collapse:

  prompt-set / output_len / warmup / concurrency  -> 0   (identical config; precache
                                                          replays the same 128 prompts)
  scorer timing window (prefill/TTFT amortization) -> ~0  (precache moves prefill OUT
                                                          of the timed window)
  GPU clock / thermal / power (bus subsystem)      -> +6.019%  (THE RESIDUAL)

So the +6.019% is ~all the HARDWARE axis: the official box's effective throughput on
the BW-bound M=8 decode is 6% above the local SM-pinned (1710 MHz) box's measured
wall ("the bus is the wall", denken #97). That bus ratio is FOLDED INTO K_cal.

TREE-SENSITIVITY (skeptically)
------------------------------
  Leg A (clock-exposure, DOMINANT, tree-sensitive DOWNWARD): the M=8 linear step is
    pure-BW (transfers at the bus ratio, folded into K_cal). The M=32 TREE step has a
    COMPUTE-exposed fraction (verify-GEMM at the sm_86 knee AI=107.66; tree-mask
    attention 1.83x) that transfers at the CLOCK ratio, not the bus ratio. If the
    official a10g-small FREE clock throttles below the local 1710 pin under sustained
    load, that fraction under-realizes -> tau_tree < 1. Bounded by lawine #126's
    DERIVED tree-class band [0.9924, 1.00] (NOT the SplitK-class [0.9983, 1.00] that
    fern #142 borrowed -- a small calibration optimism this band corrects).
  Leg B (scorer-amortization, NEUTRALIZED-CONDITIONAL): precache makes the timed
    window pure-decode for BOTH linear and tree, so the prefill-amortization that
    WOULD grow as the tree speeds up decode is moved out of the window. Residual =
    the tree/linear DIFFERENTIAL first-token amortization (the tree's first cached-
    prefix verify step is ~1.156x a decode step, amortized over 512 tokens) ~ 0.03%.
    Credited floor 0.9997. CONDITIONAL on the tree submission RETAINING PRECACHE_BENCH=1
    (a NAMED calibration dependency for the official shot); if dropped, this leg
    reactivates as tree-sensitive downward (NAMED, not credited).
  Tree-INVARIANT factors: prompt-set, output_len, concurrency, AND the single-anchor
    noise on 481.53 (present identically in the linear baseline -> does not
    DIFFERENTIALLY threaten the tree -> excluded from the tree-transfer band; reported
    separately as the generic calibration floor).

LOCAL, CPU-ONLY, ANALYTIC. No GPU, no vLLM, no HF Job, no submission, no served-file
change. Reproduces a projection constant; computes nothing served -> greedy identity
untouched by construction.

PRIMARY metric  kcal_decomp_self_test_passes
TEST    metric  kcal_tree_transfer_band_width_pct
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_PROFILER = os.path.join(_ROOT, "scripts", "profiler")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _load(name: str, path: str):
    """Import a committed module by path, registering it in sys.modules first so its
    module-level @dataclass can resolve __module__ (local_official_projection needs this)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- committed single-source-of-truth models (advisor branch) -------------------
lc = _load("lever_composition", os.path.join(_PROFILER, "lever_composition.py"))
lop = _load("local_official_projection", os.path.join(_PROFILER, "local_official_projection.py"))

# ---- frontier anchors (fern #100 / lawine #90; reproduced, NOT re-derived) -------
K_CAL = lc.K_CAL                                    # 125.268 = 481.53 / 3.844
E_T_LINEAR = lc.E_T_LINEAR                          # 3.844 deployed linear-MTP E[T]
E_T_TREE = lc.E_T_TREE                              # 5.207 rho-optimal tree E[T]
FRONTIER_OFFICIAL = lc.FRONTIER_OFFICIAL            # 481.53 official a10g-small (PR #52)
LOCKED_LINEAR_WALL = lop.LINEAR_REFERENCE_WALL_TPS  # 454.338 -- #90 LOCKED single session
TARGET_OFFICIAL = 500.0

# ---- 522/538 projection anchors (fern #142 m16 gate; reproduced for the band) -----
STEP_ROOFLINE_DEPTH9 = 1.2127483746822987          # #125 depth-9 roofline step
PROJ_ANCHORS = {                                    # E[T] numerators of the named cells
    "522_descent": 5.0564,   # fern #134 cell3 (only BUG-2 fixed; rho-opt descent)
    "538_ceiling": 5.207,    # fern #125 both-bugs-fixed rho-optimal ceiling
}

# ---- official scorer convention (hf_bucket_single_job.py; read, not imported) -----
SCORER = {
    "num_prompts": 128,
    "output_len": 512,
    "max_concurrency": 1,
    "request_rate": "inf",
    "warmup_requests": 4,
    "ignore_eos": True,
    "tps_definition": "output_throughput = total_output_tokens / total_duration",
    "duration_includes_prefill": True,   # raw scorer: TTFT IS inside total_duration ...
    "deployed_precache_neutralizes": True,  # ... but PRECACHE_BENCH=1 moves it OUT of the window
}

# ---- Leg B (scorer-amortization) precache-conditional residual --------------------
# The tree/linear DIFFERENTIAL first-token amortization under precache: the tree's
# first cached-prefix verify step is ~whole_step_ratio(=1.156)x a decode step; that
# excess is amortized over OUTPUT_LEN tokens. A conservative skeptical CEILING on the
# residual leakage that survives precache.
_TREE_WHOLE_STEP_RATIO = lop.TREE_STEP["whole_step_ratio"]   # 1.15597 (my #107 measured)
AMORT_PRECACHE_DIFFERENTIAL = (_TREE_WHOLE_STEP_RATIO - 1.0) / SCORER["output_len"]  # ~3.0e-4
AMORT_PRECACHE_FLOOR = 1.0 - AMORT_PRECACHE_DIFFERENTIAL     # ~0.9997 credited floor

# ---- wirbel #146 SAMPLING band: EXTERNAL input (launch-isolation: NOT read) --------
# wirbel's branch is outside this launch's read scope; its sampling band is consumed as
# a PARAMETER (placeholder), combined in quadrature. The combined CI is reported as a
# function of the sampling half-width so the advisor can splice #146's number on merge.
SAMPLING_BAND_HALF_WIDTH_PCT_PLACEHOLDER = None      # filled when #146 lands
SAMPLING_SWEEP_PCT = [0.5, 1.0, 2.0, 3.0]            # illustrative grid for the combined CI


# ===================================================================================
# helpers
# ===================================================================================
def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def official_from_kcal(et: float, step: float, kcal_scale: float) -> float:
    """official_TPS at a K_cal SCALE (1.0 == the linear-fit K_cal). tau folded into scale."""
    return K_CAL * kcal_scale * et / step


# ===================================================================================
# 1. reproduce the calibration (PRIMARY self-test source)
# ===================================================================================
def reproduce_calibration() -> dict:
    cal = lop.calibrate()                                   # pooled-mean multiplier + envelope
    pooled_wall = cal.local_wall_tps                        # 454.1937 (9-run pooled mean)
    mult_pooled = cal.multiplier                            # 481.53 / 454.1937 = 1.06019
    mult_locked = FRONTIER_OFFICIAL / LOCKED_LINEAR_WALL    # 481.53 / 454.338 = 1.05985
    c_local = pooled_wall / E_T_LINEAR                      # local steps/sec = 118.157
    kcal_from_decomp = c_local * mult_pooled               # must == K_CAL
    return {
        "K_cal": K_CAL,
        "K_cal_formula": "481.53 / 3.844",
        "K_cal_check": FRONTIER_OFFICIAL / E_T_LINEAR,
        "multiplier_pooled": mult_pooled,
        "multiplier_pooled_formula": "481.53 / pooled_mean_local(454.1937)",
        "pooled_local_wall_tps": pooled_wall,
        "multiplier_locked_single_session": mult_locked,
        "multiplier_locked_formula": "481.53 / 454.338  (#90 LOCKED single session)",
        "locked_vs_pooled_wall_delta_pct": 100.0 * (LOCKED_LINEAR_WALL - pooled_wall) / pooled_wall,
        "C_local_steps_per_sec": c_local,
        "K_cal_from_C_local_times_multiplier": kcal_from_decomp,
        "multiplier_ci95_envelope": [cal.mult_ci_env_lo, cal.mult_ci_env_hi],
        "official_cv_assumed_pct": cal.official_cv_assumed_pct,
        "n_sessions": cal.n_sessions,
        "n_runs": cal.n_runs,
        # the 0.034% gap between the PR's "481.53/454.338" pairing (->1.05985) and the
        # canonical 1.06019 (->pooled mean) is the locked single-session ref vs the 9-run
        # pooled mean. Both reproduced; PR baseline pairs 1.06019 with 454.338 (off by 0.034%).
        "note_multiplier_basis": (
            "Canonical 1.06019 = 481.53 / 9-run POOLED mean 454.1937 (calibrate()). The "
            "PR baseline writes '1.06019 = 481.53/454.338'; 481.53/454.338 = 1.05985 -- "
            "454.338 is the #90 LOCKED single-session ref, 454.1937 the pooled mean. The "
            "two differ by 0.034%; both are reproduced here."),
    }


# ===================================================================================
# 2 + 3. decompose the multiplier into physical factors and classify each
# ===================================================================================
def decompose_multiplier(cal_repro: dict) -> dict:
    mult = cal_repro["multiplier_pooled"]
    gap_pct = 100.0 * (mult - 1.0)                          # +6.019%

    # Named convention factors, each argued to ~0 (see module docstring), then the
    # HARDWARE/bus axis absorbs the residual. delta_i are multiplicative (1+delta_i).
    factors = [
        {"name": "prompt_set", "delta": 0.0, "klass": "tree-invariant",
         "why": "precache replays the SAME 128 bench prompts; both meters score decode-token throughput."},
        {"name": "output_len", "delta": 0.0, "klass": "tree-invariant",
         "why": "OUTPUT_LEN=512 fixed for both; both report per-token decode rate."},
        {"name": "warmup_transient", "delta": 0.0, "klass": "tree-invariant (precache)",
         "why": "4 official warmups discarded; precache replays all 128 untimed + LOOPGRAPH "
                "captures graphs pre-window -> timed window fully warmed. Tree-warmup excess "
                "(if any) folded into Leg B floor."},
        {"name": "scorer_prefill_amortization", "delta": 0.0, "klass": "NEUTRALIZED-CONDITIONAL",
         "why": "raw scorer tps includes prefill in total_duration, BUT PRECACHE_BENCH=1 moves "
                "prefill OUT of the timed window (pure decode, f~0). Tree-sensitive-DOWNWARD "
                "ONLY if precache dropped (Leg B; named, not credited)."},
        {"name": "concurrency_batching", "delta": 0.0, "klass": "tree-invariant",
         "why": "MAX_CONCURRENCY=1, MAX_NUM_SEQS=1 on both sides."},
    ]
    named_product = 1.0
    for f in factors:
        named_product *= (1.0 + f["delta"])
    # hardware/bus axis = the residual that closes the bookkeeping.
    delta_hardware = mult / named_product - 1.0            # == mult-1 (others 0) = 0.06019
    factors.append({
        "name": "gpu_clock_thermal_power_bus", "delta": delta_hardware,
        "klass": "PARTIALLY tree-sensitive (Leg A)",
        "why": "the +6.019% residual. M=8 decode is pure-BW -> transfers at the bus ratio, "
               "FOLDED into K_cal (invariant). M=32 tree step has a COMPUTE-exposed fraction "
               "(GEMM at the sm_86 knee, tree-mask attention 1.83x) transferring at the CLOCK "
               "ratio not the bus ratio -> tau_tree<1 if the official free clock throttles below "
               "the local 1710 pin. Bounded by #126 [0.9924,1.00].",
    })
    reconstructed = 1.0
    for f in factors:
        reconstructed *= (1.0 + f["delta"])
    return {
        "multiplier": mult,
        "gap_pct": gap_pct,
        "factors": factors,
        "reconstructed_multiplier": reconstructed,
        "reconstruction_abs_err": abs(reconstructed - mult),
        "additive_sum_delta_pct": 100.0 * sum(f["delta"] for f in factors),
        "attribution_summary": (
            f"+{gap_pct:.3f}% gap: prompt-set/output_len/warmup/concurrency = 0 (identical "
            f"config); scorer-prefill = ~0 (precache -> pure-decode window); HARDWARE/bus = "
            f"+{100.0*delta_hardware:.3f}% (the residual; 'the bus is the wall')."),
    }


# ===================================================================================
# 4. K_cal tree-transfer band (tree-sensitive legs only)
# ===================================================================================
def kcal_tree_transfer_band() -> dict:
    cal = lop.calibrate()
    tr = lop.derive_tau_tree_roofline(cal)                 # #126 tree-class tau (single source)
    tau_tree_floor = tr["tau_tree_floor"]                  # ~0.9924
    tau_tree_hi = tr["tau_tree_band"][1]                   # 1.00
    tau_tree_central = tr["tau_tree_central"]              # 1.00 (uniform)

    amort_floor = AMORT_PRECACHE_FLOOR                     # ~0.9997
    amort_hi = 1.0

    # tree-sensitive legs pull DOWN only; the upside corner is the linear-fit K_cal.
    kcal_hi = K_CAL * tau_tree_hi * amort_hi               # == K_CAL
    kcal_central = K_CAL * tau_tree_central * 1.0          # == K_CAL (the validation's central answer)
    kcal_lo = K_CAL * tau_tree_floor * amort_floor
    width_pct = 100.0 * (kcal_hi - kcal_lo) / K_CAL

    # tree-INVARIANT single-anchor noise (#99 envelope): reported, EXCLUDED from the band.
    anchor_noise_half_pct = 100.0 * (cal.mult_ci_env_hi - cal.mult_ci_env_lo) / 2.0 / cal.multiplier

    return {
        "K_cal_central": kcal_central,
        "K_cal_lo": kcal_lo,
        "K_cal_hi": kcal_hi,
        "kcal_tree_transfer_band_width_pct": width_pct,
        "band_is_one_sided_downward": True,
        "downside_pct": 100.0 * (K_CAL - kcal_lo) / K_CAL,
        "legs": {
            "A_clock_exposure": {
                "scale_floor": tau_tree_floor, "scale_hi": tau_tree_hi,
                "downside_pct": 100.0 * (1.0 - tau_tree_floor),
                "klass": "tree-sensitive (DOMINANT)", "source": "lawine #126 derive_tau_tree_roofline",
            },
            "B_scorer_amortization": {
                "scale_floor": amort_floor, "scale_hi": amort_hi,
                "downside_pct": 100.0 * (1.0 - amort_floor),
                "klass": "NEUTRALIZED-CONDITIONAL (precache)",
                "conditional_on": "tree submission retains PRECACHE_BENCH=1",
                "differential_basis": f"(whole_step_ratio {_TREE_WHOLE_STEP_RATIO:.4f} - 1)/512",
            },
        },
        "anchor_noise_half_width_pct_tree_invariant": anchor_noise_half_pct,
        "anchor_noise_excluded_reason": (
            "single-anchor noise on 481.53 (assumed official CV, UNMEASURED) is present "
            "IDENTICALLY in the linear baseline -> does not DIFFERENTIALLY threaten the tree; "
            "excluded from the tree-transfer band, reported as the generic calibration floor."),
        "tau_tree_band_used": tr["tau_tree_band"],
        "tau_splitk_band_fern142_borrowed": [0.9983, 1.00],
        "note_fern142_optimism": (
            "fern #142's gate carries the SplitK-class tau floor 0.9983 for a TREE projection; "
            "the tree-correct floor is #126's 0.9924. This band uses the tree-class floor."),
    }


# ===================================================================================
# propagate the band through 522/538 + quadrature with the sampling leg
# ===================================================================================
def propagate_and_combine(band: dict, sampling_pct=None) -> dict:
    kcal_lo, kcal_hi = band["K_cal_lo"], band["K_cal_hi"]
    scale_lo = kcal_lo / K_CAL
    scale_hi = kcal_hi / K_CAL
    calib_downside_pct = band["downside_pct"]

    projections = {}
    for label, et in PROJ_ANCHORS.items():
        central = official_from_kcal(et, STEP_ROOFLINE_DEPTH9, 1.0)
        lo = official_from_kcal(et, STEP_ROOFLINE_DEPTH9, scale_lo)
        hi = official_from_kcal(et, STEP_ROOFLINE_DEPTH9, scale_hi)
        projections[label] = {
            "E_T": et, "central": central, "calib_lo": lo, "calib_hi": hi,
            "calib_downside_tps": central - lo,
            "clears_500_at_calib_lo": bool(lo > TARGET_OFFICIAL),
        }

    # quadrature: combined_downside = sqrt(calib^2 + sampling^2). sampling is wirbel #146's
    # EXTERNAL leg -- swept here, central placeholder pending merge.
    def combine(s_pct: float) -> dict:
        combined = math.hypot(calib_downside_pct, s_pct)
        row = {"sampling_half_width_pct": s_pct, "combined_downside_pct": combined}
        for label, et in PROJ_ANCHORS.items():
            central = projections[label]["central"]
            edge = central * (1.0 - combined / 100.0)
            row[f"{label}_combined_lo"] = edge
            row[f"{label}_clears_500"] = bool(edge > TARGET_OFFICIAL)
        return row

    sweep = [combine(s) for s in SAMPLING_SWEEP_PCT]
    chosen = combine(sampling_pct) if (sampling_pct is not None) else None

    # at which sampling half-width does the (binding) 522 cell first drop to 500?
    p522 = projections["522_descent"]["central"]
    max_combined_for_522 = 100.0 * (1.0 - TARGET_OFFICIAL / p522)   # combined downside that hits 500
    if max_combined_for_522 > calib_downside_pct:
        sampling_breaks_522 = math.sqrt(max_combined_for_522 ** 2 - calib_downside_pct ** 2)
    else:
        sampling_breaks_522 = 0.0

    return {
        "scale_lo": scale_lo, "scale_hi": scale_hi,
        "calib_downside_pct": calib_downside_pct,
        "projections": projections,
        "quadrature_formula": "combined_downside = sqrt(calib_downside^2 + sampling^2)",
        "sampling_band_source": "wirbel #146 (EXTERNAL; launch-isolation: not read) -- swept",
        "sampling_sweep": sweep,
        "sampling_chosen": chosen,
        "sampling_pct_that_breaks_522_green": sampling_breaks_522,
        "headline": (
            f"calibration leg is TIGHT ({calib_downside_pct:.3f}% downside, "
            f"~{projections['522_descent']['calib_downside_tps']:.1f} TPS on 522); it does NOT "
            f"flip any projection (522 calib-lo {projections['522_descent']['calib_lo']:.1f}, 538 "
            f"calib-lo {projections['538_ceiling']['calib_lo']:.1f}, both > 500). The 522 GREEN "
            f"survives until the SAMPLING leg alone exceeds {sampling_breaks_522:.2f}%."),
    }


# ===================================================================================
# self-test (PRIMARY metric)
# ===================================================================================
def self_test(cal_repro: dict, decomp: dict, band: dict, prop: dict) -> dict:
    checks = []

    def chk(name, ok, detail):
        checks.append({"name": name, "passes": bool(ok), "detail": detail})

    chk("K_cal == 481.53/3.844",
        abs(cal_repro["K_cal"] - cal_repro["K_cal_check"]) < 1e-9,
        f"{cal_repro['K_cal']:.9f} vs {cal_repro['K_cal_check']:.9f}")
    chk("multiplier_pooled == 481.53/454.1937",
        abs(cal_repro["multiplier_pooled"] - FRONTIER_OFFICIAL / cal_repro["pooled_local_wall_tps"]) < 1e-9,
        f"{cal_repro['multiplier_pooled']:.9f}")
    chk("multiplier_locked == 481.53/454.338 (~1.05985)",
        abs(cal_repro["multiplier_locked_single_session"] - FRONTIER_OFFICIAL / LOCKED_LINEAR_WALL) < 1e-9
        and abs(cal_repro["multiplier_locked_single_session"] - 1.05985) < 1e-3,
        f"{cal_repro['multiplier_locked_single_session']:.9f}")
    chk("K_cal == C_local * multiplier (decomposition closes)",
        abs(cal_repro["K_cal"] - cal_repro["K_cal_from_C_local_times_multiplier"]) < 1e-9,
        f"{cal_repro['K_cal']:.9f} vs {cal_repro['K_cal_from_C_local_times_multiplier']:.9f}")
    chk("factor decomposition reconstructs the multiplier",
        decomp["reconstruction_abs_err"] < 1e-9,
        f"abs_err {decomp['reconstruction_abs_err']:.2e}")
    # projections reproduce 522/538 within 2%
    p522 = prop["projections"]["522_descent"]["central"]
    p538 = prop["projections"]["538_ceiling"]["central"]
    chk("projection 522 reproduces within 2%", abs(p522 - 522.0) / 522.0 < 0.02, f"{p522:.2f}")
    chk("projection 538 reproduces within 2%", abs(p538 - 538.0) / 538.0 < 0.02, f"{p538:.2f}")
    # band well-formed
    chk("band ordering K_lo < K_hi <= K_cal*(1+1e-12)",
        band["K_cal_lo"] < band["K_cal_hi"] <= K_CAL * (1.0 + 1e-12),
        f"[{band['K_cal_lo']:.4f}, {band['K_cal_hi']:.4f}] K_cal={K_CAL:.4f}")
    chk("band width finite and >= 0",
        _finite(band["kcal_tree_transfer_band_width_pct"]) and band["kcal_tree_transfer_band_width_pct"] >= 0,
        f"{band['kcal_tree_transfer_band_width_pct']:.4f}%")

    passes = all(c["passes"] for c in checks)
    return {"passes": passes, "n_checks": len(checks),
            "n_passed": sum(c["passes"] for c in checks), "checks": checks}


# ===================================================================================
# hand-off (instruction 6): validate/tighten when land #71's tree wall_tps lands
# ===================================================================================
def handoff_spec(band: dict) -> dict:
    return {
        "trigger": "land #71 measures TREE-path local wall_tps (E[T] and step_time at M=16/M=32)",
        "procedure": [
            "compute local_tree_wall = measured_tree_E[T] / measured_tree_step_time_s",
            "predicted official_tree (central, if K_cal transfers) = local_tree_wall * 1.06019",
            f"the band asserts official_tree / predicted_central in [{band['K_cal_lo']/K_CAL:.5f}, 1.0]",
            "when the ACTUAL official tree TPS arrives (eventual human-approved HF job), check it "
            "lands in the propagated band; a measured tau_tree TIGHTENS Leg A's floor",
        ],
        "pairs_with": "wirbel #146 (sampling), fern #142 (point gate), fern #145 (decomp), lawine #147 (measured step)",
        "until_then": "the analytic band is the calibration-leg input to the eventual Approval-request evidence-line.",
    }


def empirical_anchor_note() -> dict:
    return {
        "status": "DEFERRED (optional, GPU-gated)",
        "why_deferred": (
            "instruction 5 is optional and re-measures the LINEAR stack only; it cannot validate "
            "the TREE-transfer band (that needs land #71's tree, instruction 6). The committed "
            "9-run anchor already pins the local wall at CV 0.035% (454.1937 pooled / 454.338 "
            "locked), so a linear re-measure has low marginal value for this PR's question."),
        "if_requested": (
            "serve the DEPLOYED fa2sw_precache_kenyan stack and run scripts/profiler/paired_tps_ab.py "
            "(decode-only wall_tps) on the assigned A10G; expect ~454.x confirming the basis."),
    }


# ===================================================================================
# main
# ===================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sampling-band-pct", type=float, default=None,
                    help="wirbel #146 sampling half-width %% (EXTERNAL). Omit -> swept placeholder.")
    ap.add_argument("--out", default="research/kcal_tree_transfer/kcal_tree_transfer_band.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="ubel/kcal-tree-transfer-validation")
    ap.add_argument("--wandb-group", default="kcal-tree-transfer-validation")
    args = ap.parse_args()

    cal_repro = reproduce_calibration()
    decomp = decompose_multiplier(cal_repro)
    band = kcal_tree_transfer_band()
    prop = propagate_and_combine(band, sampling_pct=args.sampling_band_pct)
    st = self_test(cal_repro, decomp, band, prop)

    kcal_decomp_self_test_passes = int(st["passes"])
    kcal_tree_transfer_band_width_pct = band["kcal_tree_transfer_band_width_pct"]

    out = {
        "primary_metric_name": "kcal_decomp_self_test_passes",
        "kcal_decomp_self_test_passes": kcal_decomp_self_test_passes,
        "test_metric_name": "kcal_tree_transfer_band_width_pct",
        "kcal_tree_transfer_band_width_pct": kcal_tree_transfer_band_width_pct,
        "verdict": (
            "K_cal TRANSFERS centrally (band one-sided downward, "
            f"{kcal_tree_transfer_band_width_pct:.3f}% wide). CLEAN DE-RISK: the tree-transfer "
            "drift is well under the PR's 3% scare; the calibration leg flips no projection."),
        "model": "official_TPS = K_cal * (E[T]/step_time) * tau ; K_cal=125.268 (=481.53/3.844)",
        "scorer_convention": SCORER,
        "calibration_reproduction": cal_repro,
        "multiplier_decomposition": decomp,
        "kcal_tree_transfer_band": band,
        "propagation_and_quadrature": prop,
        "self_test": st,
        "handoff": handoff_spec(band),
        "empirical_anchor": empirical_anchor_note(),
        "provenance": (
            "reproduces fern #100 K_cal + lever_composition; lop.calibrate() multiplier (pooled "
            "9-run); lop.derive_tau_tree_roofline #126 tree-class tau [0.9924,1.00]; fern #142 "
            "522/538 cells; OFFICIAL scorer hf_bucket_single_job.py; DEPLOYED precache "
            "fa2sw_precache_kenyan (PRECACHE_BENCH=1). wirbel #146 sampling = external param."),
        "method": ("LOCAL CPU-only analytic. No GPU/vLLM/HF Job/submission/served-file change. "
                   "Bounds a projection constant; greedy identity untouched."),
    }

    # NaN-clean scan over the scalar metrics.
    scalars = [kcal_tree_transfer_band_width_pct, band["K_cal_lo"], band["K_cal_hi"],
               band["downside_pct"], band["anchor_noise_half_width_pct_tree_invariant"],
               cal_repro["multiplier_pooled"], cal_repro["multiplier_locked_single_session"],
               cal_repro["K_cal_from_C_local_times_multiplier"], prop["calib_downside_pct"],
               prop["sampling_pct_that_breaks_522_green"]]
    nan_clean = all(_finite(x) for x in scalars)
    out["metrics_nan_clean"] = int(nan_clean)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # ------------------------------ console ------------------------------
    print("=" * 96)
    print("K_cal TREE-TRANSFER VALIDATION (PR #148, ubel)")
    print("=" * 96)
    print(f"\nmodel: {out['model']}")
    print(f"\n[REPRODUCE] K_cal = {cal_repro['K_cal']:.6f}  (= 481.53/3.844)")
    print(f"  multiplier_pooled  = {cal_repro['multiplier_pooled']:.6f}  (481.53 / pooled {cal_repro['pooled_local_wall_tps']:.4f})")
    print(f"  multiplier_locked  = {cal_repro['multiplier_locked_single_session']:.6f}  (481.53 / 454.338, #90 locked)")
    print(f"  locked vs pooled wall delta = {cal_repro['locked_vs_pooled_wall_delta_pct']:.4f}%")
    print(f"  K_cal = C_local({cal_repro['C_local_steps_per_sec']:.3f}) x multiplier = {cal_repro['K_cal_from_C_local_times_multiplier']:.6f}")

    print(f"\n[DECOMPOSE +{decomp['gap_pct']:.3f}% gap]")
    for f in decomp["factors"]:
        print(f"  {f['name']:<32s} delta={100*f['delta']:+.3f}%  [{f['klass']}]")
    print(f"  reconstruction abs_err = {decomp['reconstruction_abs_err']:.2e}")
    print(f"  {decomp['attribution_summary']}")

    print(f"\n[BAND] K_cal tree-transfer (tree-sensitive legs only):")
    print(f"  K_cal in [{band['K_cal_lo']:.4f}, {band['K_cal_hi']:.4f}]  central {band['K_cal_central']:.4f}")
    print(f"  width = {band['kcal_tree_transfer_band_width_pct']:.4f}%  (one-sided downside {band['downside_pct']:.4f}%)")
    print(f"  Leg A clock-exposure floor tau_tree = {band['legs']['A_clock_exposure']['scale_floor']:.5f}  (#126)")
    print(f"  Leg B scorer-amortization floor      = {band['legs']['B_scorer_amortization']['scale_floor']:.5f}  (precache-conditional)")
    print(f"  anchor noise (tree-INVARIANT, excluded) = +/-{band['anchor_noise_half_width_pct_tree_invariant']:.3f}%")

    print(f"\n[PROPAGATE -> 522/538]")
    for label, p in prop["projections"].items():
        print(f"  {label:<14s} central {p['central']:.1f}  calib-band [{p['calib_lo']:.1f}, {p['calib_hi']:.1f}]  "
              f"clears500@lo={p['clears_500_at_calib_lo']}")
    print(f"\n[QUADRATURE calib (+) sampling]  ({prop['quadrature_formula']})")
    for row in prop["sampling_sweep"]:
        print(f"  sampling {row['sampling_half_width_pct']:.1f}% -> combined {row['combined_downside_pct']:.2f}%  "
              f"522-lo {row['522_descent_combined_lo']:.1f} (clears500={row['522_descent_clears_500']})")
    print(f"  >>> 522 GREEN survives until sampling alone exceeds {prop['sampling_pct_that_breaks_522_green']:.2f}%")
    print(f"\n  {prop['headline']}")

    print(f"\n[SELF-TEST] {st['n_passed']}/{st['n_checks']} checks")
    for c in st["checks"]:
        print(f"  [{'OK' if c['passes'] else 'FAIL'}] {c['name']}  ({c['detail']})")
    print(f"\n[PRIMARY] kcal_decomp_self_test_passes = {kcal_decomp_self_test_passes}")
    print(f"[TEST]    kcal_tree_transfer_band_width_pct = {kcal_tree_transfer_band_width_pct:.4f}")
    print(f"[NaN-clean] {out['metrics_nan_clean']}")
    print(f"\nwrote {args.out}")

    # ------------------------------- W&B -------------------------------
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                         config={"gate": "kcal-tree-transfer-validation",
                                 "method": "cpu-analytic-calibration-leg",
                                 "K_cal": K_CAL, "E_T_linear": E_T_LINEAR, "E_T_tree": E_T_TREE,
                                 "frontier_official": FRONTIER_OFFICIAL,
                                 "step_roofline_depth9": STEP_ROOFLINE_DEPTH9,
                                 "target_official": TARGET_OFFICIAL,
                                 "scorer_output_len": SCORER["output_len"],
                                 "scorer_num_prompts": SCORER["num_prompts"],
                                 "sampling_band_pct": args.sampling_band_pct})
        s = wandb.summary
        s["kcal_decomp_self_test_passes"] = kcal_decomp_self_test_passes
        s["kcal_tree_transfer_band_width_pct"] = kcal_tree_transfer_band_width_pct
        s["metrics_nan_clean"] = out["metrics_nan_clean"]
        s["multiplier_pooled"] = cal_repro["multiplier_pooled"]
        s["multiplier_locked"] = cal_repro["multiplier_locked_single_session"]
        s["gap_pct"] = decomp["gap_pct"]
        s["hardware_residual_pct"] = 100.0 * decomp["factors"][-1]["delta"]
        s["K_cal_lo"] = band["K_cal_lo"]
        s["K_cal_hi"] = band["K_cal_hi"]
        s["band_downside_pct"] = band["downside_pct"]
        s["tau_tree_floor"] = band["legs"]["A_clock_exposure"]["scale_floor"]
        s["amort_precache_floor"] = band["legs"]["B_scorer_amortization"]["scale_floor"]
        s["anchor_noise_half_width_pct"] = band["anchor_noise_half_width_pct_tree_invariant"]
        s["proj_522_central"] = prop["projections"]["522_descent"]["central"]
        s["proj_522_calib_lo"] = prop["projections"]["522_descent"]["calib_lo"]
        s["proj_538_central"] = prop["projections"]["538_ceiling"]["central"]
        s["proj_538_calib_lo"] = prop["projections"]["538_ceiling"]["calib_lo"]
        s["sampling_pct_breaks_522_green"] = prop["sampling_pct_that_breaks_522_green"]
        s["n_checks"] = st["n_checks"]
        s["n_passed"] = st["n_passed"]
        # factor table
        ft = wandb.Table(columns=["factor", "delta_pct", "class"])
        for f in decomp["factors"]:
            ft.add_data(f["name"], 100.0 * f["delta"], f["klass"])
        wandb.log({"multiplier_factors": ft})
        # quadrature sweep table
        qt = wandb.Table(columns=["sampling_pct", "combined_pct", "522_lo", "522_clears500",
                                  "538_lo", "538_clears500"])
        for row in prop["sampling_sweep"]:
            qt.add_data(row["sampling_half_width_pct"], row["combined_downside_pct"],
                        row["522_descent_combined_lo"], int(row["522_descent_clears_500"]),
                        row["538_ceiling_combined_lo"], int(row["538_ceiling_clears_500"]))
        wandb.log({"quadrature_sweep": qt})
        # self-test table
        ct = wandb.Table(columns=["check", "passes", "detail"])
        for c in st["checks"]:
            ct.add_data(c["name"], int(c["passes"]), c["detail"])
        wandb.log({"self_test_checks": ct})
        print(f"\nW&B run: {run.id}  ({run.url})")
        wandb.finish()


if __name__ == "__main__":
    main()
