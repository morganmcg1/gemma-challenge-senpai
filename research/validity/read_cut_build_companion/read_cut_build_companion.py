#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Is the #287 read-cut a FREE body-side companion to the EAGLE-3 build? (PR #302)

LOCAL CPU-analytic over banked W&B numbers. NO served-file change, NO fresh wall
A/B, NO HF Job, NO submission. NOT a launch, NOT a build, NOT open2. 0 TPS.
BASELINE stays 481.53.

CONTEXT. fern #287 (17en3hus) closed the read-side STANDALONE door: the maximum
PPL-safe body-read-byte cut is 8.431% (config mixed_int3_demote16L, proj deployed
PPL 2.3975 <= 2.42), and re-attributed on denken #283's MEASURED 38% read-fraction
it does NOT reach 500 alone (read_reduction_lever_clears_500 = False). The sole >500
path is the human-gated EAGLE-3 build, which raises the NUMERATOR (E[T], via a better
greedy-token-identical drafter) and leaves the verify-path BODY unchanged -> build
PPL stays at the deployed 2.3772.

QUESTION. "Does not clear 500 STANDALONE" is not "is a free COMPANION to the build."
Is the #287 read-cut a FREE body-side companion that stacks its TPS ON TOP of the
build (the read-side analog of lawine's SAM drafter-companion #296/#300), or does it
UNDER-REALIZE / REGRESS the way stark #273 (51bdsbpw) found every body-side deviation
from the K=7-optimized ONEGRAPH graph regresses on the wall (K4-vs-K7 = -8.629%,
realization ratio K4 = -2.018 / K5 = -0.864, NEGATIVE)?

METHOD (analytic, over banked ratios; NOT a fresh wall A/B -- that is stark's #298
harness/lane):
 1. PPL-headroom fit. Build leaves the body unchanged (greedy-identical drafter:
    verify rejects any draft token != target greedy argmax -> drafter precision moves
    E[T], NOT PPL) -> build PPL = deployed 2.3772, headroom to gate 2.42 = 0.0428.
    read-cut PPL cost = 2.3975 - 2.3772 = 0.0203 <= 0.0428 (FITS, body-only clause).
 2. Compose the read-cut TPS credit (8.43% byte cut x denken #283's 38% read-fraction)
    on the deployed AND build operating points (the read-cut shortens the SAME absolute
    verify-path body read on EVERY step, build or not).
 3. Realization-discount it by stark #273's banked body-side realization ratios. The
    read-cut RE-QUANTIZES the verify-path weights (int4->int3 on 16 layers) -> it is a
    body-side deviation from the K=7-tuned ONEGRAPH, exactly stark #273's class. Every
    banked body-side ratio is NEGATIVE.
 4. Verdict. free companion (ratio >= 0.8, stacks positive TPS) vs doubly-closed
    (ratio <= 0, regresses like static-K: not standalone (#287), not a companion).

HONESTY. 0 TPS. This prices WHERE #287's read-frontier lands relative to the gated
build (free companion vs doubly-closed) in wall-realized (stark #273-discounted) terms,
NOT idealized-composition terms. It does NOT produce a >=500 build and does NOT change
the served checkpoint (fa2sw_precache_kenyan). The realization discount is stark #273's
banked-ratio PRIOR applied analytically; a fresh wall A/B of the read-cut is stark's
#298 lane, NOT this leg. The PPL headroom assumes the build leaves the body unchanged
(if it co-quantizes the body for VRAM -- ubel #299's lane -- the headroom is shared;
flagged assumes_body_unchanged=True). Launch gate stays land #245 MEASURED >=500 at
lambda_hat>=0.9780 AND PPL<=2.42 AND VRAM<=24GB, human-approval-gated.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# BANKED IMPORTS -- do NOT re-derive. Each carries its source run + PR shorthand.
# (self-test (d): every value must round to its PR-stated shorthand EXACTLY.)
# --------------------------------------------------------------------------- #
# fern #287 (17en3hus) -- the read-frontier this leg prices as a companion.
MAX_PPL_SAFE_READ_REDUCTION_PCT = 8.431155407899594   # PR 8.431
READ_CUT_CONFIG = "mixed_int3_demote16L"
READ_CUT_PROJ_PPL = 2.3975206940224814                # PR 2.3975
READ_REDUCTION_LEVER_CLEARS_500 = False               # #287 standalone verdict
TPS_AT_MAX_SAFE_DEPLOYED = 497.4881944805805          # #287 deployed read-cut TPS

# denken #283 (vmxuwxm0) -- the MEASURED read-fraction re-attribution basis.
BODY_READ_FRACTION = 0.3804642716594112               # PR 0.38 (read frac of wall)
DENKEN283_READ_FLOOR_US = 3037.203622326286           # measured body-read time / step
DENKEN283_WALL_US = 7982.887878221502                 # honest per-step wall (deployed)
HBM_BOUND_CEILING_TPS = 1265.6378952477885            # PR 1265.6 (pure-read ceiling)

# stark #273 (51bdsbpw) -- the realization-ratio method + body-side regression prior.
# realization_ratio = measured_wall_delta_pct / composed_gain_pct. Body-side deviations
# from the K=7 ONEGRAPH graph REGRESS: every ratio is NEGATIVE.
K4_VS_K7_MEASURED_DELTA_PCT = -8.628968285857523      # PR -8.629
K4_COMPOSED_GAIN_PCT = 4.27653374979271               # K4 composition predicted +4.28%
K4_REALIZATION_RATIO = -2.01774820233227              # PR -2.018 (load-bearing precedent)
K5_REALIZATION_RATIO = -0.8641778297435041            # PR -0.864
K3_REALIZATION_RATIO = -6.184909243990122
K6_REALIZATION_RATIO = -0.6102631065916517
STARK273_BODY_SIDE_RATIOS = {                         # banked body-side realization ratios
    3: K3_REALIZATION_RATIO, 4: K4_REALIZATION_RATIO,
    5: K5_REALIZATION_RATIO, 6: K6_REALIZATION_RATIO,
}

# wirbel #290 / fern #281 -- the build target the companion would stack on.
BUILD_ET_TARGET = 4.9029                               # PR 4.9029 (step-banked build E[T])
PUBLIC_FLOOR_ET = 4.966                                # PR 4.966 (fern #281 public floor)

# kanna #217 (vgovdrjc) -- official anchors.
OFFICIAL_BASELINE_TPS = 481.53                         # PR 481.53
E_T_DEPLOYED = 3.844                                   # PR 3.844
K_CAL = 125.268                                        # PR 125.268
STEP_US = 1218.2                                       # PR 1218.2
LAMBDA1_CEILING_TPS = 520.953                          # PR 520.953 (lambda=1 TPS ceiling)
PRIVATE_VERIFIED_TPS = 460.85                          # private-verified reference

# Gates.
DEPLOYED_INT4_PPL = 2.3772                             # PR 2.3772 (build PPL, body unchanged)
PPL_GATE = 2.42                                        # PR 2.42
TARGET_TPS = 500.0
LAUNCH_LAMBDA_HAT = 0.9780                             # land #245 measured launch gate
DEPLOYED_CHECKPOINT = "fa2sw_precache_kenyan"

OUTDIR = Path("/workspace/senpai/target/research/validity/read_cut_build_companion")

# Classification thresholds for the realization ratio (PR step 2/3).
REALIZES_RATIO = 0.8                                   # >= -> free companion
PARTIAL_LO = 0.0                                       # (0, 0.8) -> partial
# <= 0 -> regresses (the static-K class -- doubly-closed)

CAVEATS = [
    "0_TPS: this leg PRICES whether the #287 read-cut is a free body-side companion to "
    "the EAGLE-3 build; it produces NO >=500 build and adds 0 TPS. BASELINE stays 481.53.",
    "ANALYTIC_NOT_A_WALL_AB: the realization discount is stark #273's (51bdsbpw) banked "
    "body-side realization-ratio PRIOR applied ANALYTICALLY over banked numbers. A fresh "
    "wall A/B of the read-cut is stark's #298 harness/lane, NOT this leg.",
    "ASSUMES_BODY_UNCHANGED: the 0.0428 PPL headroom assumes the EAGLE-3 build leaves the "
    "verify-path body unchanged (greedy-token-identical drafter -> precision moves E[T], not "
    "PPL). If the build co-quantizes the body to free VRAM (ubel #299's lane), this headroom "
    "is SHARED with the body-quant -> flagged assumes_body_unchanged=True.",
    "BODY_SIDE_DEVIATION_TRANSFER: the read-cut re-quantizes verify-path weights "
    "(int4->int3 on 16 layers) -> it is a body-side deviation from the K=7-optimized "
    "ONEGRAPH graph, stark #273's class (which alters the verify GEMM/fusion, not just K). "
    "The realization ratio is transferred on that class membership, NOT measured for the "
    "read-cut specifically.",
    "READ_FRACTION_DEPLOYED_DILUTES_ON_BUILD: denken #283's 38% read-fraction is a DEPLOYED "
    "measurement (E[T]=3.844). The build's higher E[T]=4.9029 lengthens the step, so the "
    "SAME absolute body read is a SMALLER wall-fraction on the build (~31%) -> the idealized "
    "8.43%x38% credit is OPTIMISTIC on the build before any realization discount.",
    "DO_NOT_SERVE: NO served-file change, NO re-quantization of the served checkpoint, NOT a "
    "launch, NOT a build, NOT open2, NO HF Job, NO submission. Launch gate stays land #245 "
    "MEASURED >=500 at lambda_hat>=0.9780 AND PPL<=2.42 AND VRAM<=24GB, human-approval-gated.",
]


def log(*a: Any) -> None:
    print("[rcbc]", *a, flush=True)


# --------------------------------------------------------------------------- #
# 1. PPL-headroom fit (body unchanged -> build PPL = deployed anchor).
# --------------------------------------------------------------------------- #
def ppl_headroom_fit() -> dict[str, Any]:
    headroom = PPL_GATE - DEPLOYED_INT4_PPL                    # 2.42 - 2.3772 = 0.0428
    read_cut_cost = READ_CUT_PROJ_PPL - DEPLOYED_INT4_PPL      # 2.3975 - 2.3772 = 0.0203
    fits = read_cut_cost <= headroom
    return {
        "build_ppl_body_unchanged": DEPLOYED_INT4_PPL,
        "ppl_gate": PPL_GATE,
        "ppl_headroom": headroom,
        "read_cut_ppl_cost": read_cut_cost,
        "read_cut_fits_ppl_headroom": bool(fits),
        "headroom_residual_after_read_cut": headroom - read_cut_cost,
        "assumes_body_unchanged": True,
    }


# --------------------------------------------------------------------------- #
# 2. Compose the read-cut TPS credit on the deployed AND build operating points.
# --------------------------------------------------------------------------- #
def tps_after_read_cut(baseline_tps: float, read_frac: float) -> float:
    """A body-read-byte cut of fraction X shrinks only f_read of the step wall (the rest
    -- draft + verify-compute + host -- is fixed): tps(X) = baseline / (1 - f_read*X)."""
    x = MAX_PPL_SAFE_READ_REDUCTION_PCT / 100.0
    return baseline_tps / (1.0 - read_frac * x)


def compose_read_credit() -> dict[str, Any]:
    x = MAX_PPL_SAFE_READ_REDUCTION_PCT / 100.0               # 0.08431
    f_dep = BODY_READ_FRACTION                                # 0.38046 (deployed, measured)

    # PR definition: composed_read_credit_pct = 8.43% byte cut x 38% read-fraction
    # (the wall-fraction the read-cut removes from the step).
    composed_read_credit_pct = 100.0 * x * f_dep              # ~3.2076 %

    # Deployed operating point: the #287-banked TPS uplift (cross-check).
    tps_dep = tps_after_read_cut(OFFICIAL_BASELINE_TPS, f_dep)
    uplift_dep_pct = 100.0 * (tps_dep / OFFICIAL_BASELINE_TPS - 1.0)   # ~3.314 %
    uplift_dep_tps = tps_dep - OFFICIAL_BASELINE_TPS                   # ~+15.96 TPS
    deployed_crosscheck_ok = abs(tps_dep - TPS_AT_MAX_SAFE_DEPLOYED) < 1e-6

    # Build operating point. The read-cut removes the SAME absolute body read (body
    # unchanged); the build raises E[T]=4.9029 -> longer step -> the read is a SMALLER
    # wall-fraction. Bracket the (unbanked) build TPS by [TARGET, lambda=1 ceiling].
    build_tps_lo, build_tps_hi = TARGET_TPS, LAMBDA1_CEILING_TPS
    build_step_us_lo = 1e6 * BUILD_ET_TARGET / build_tps_hi  # smallest step (highest TPS)
    build_step_us_hi = 1e6 * BUILD_ET_TARGET / build_tps_lo  # largest step (lowest TPS)
    build_read_frac_lo = DENKEN283_READ_FLOOR_US / build_step_us_hi  # ~0.31 (most diluted)
    build_read_frac_hi = DENKEN283_READ_FLOOR_US / build_step_us_lo  # ~0.323
    build_read_frac_mid = 0.5 * (build_read_frac_lo + build_read_frac_hi)

    # Idealized build credit per the PR (carry the deployed 38% read-fraction onto the
    # build) AND the honest diluted build credit (use the build's own read-fraction).
    composed_build_credit_idealized_pct = composed_read_credit_pct      # PR: carry 38%
    composed_build_credit_diluted_pct = 100.0 * x * build_read_frac_mid  # honest (~2.6%)

    # Idealized build uplift TPS at the build floor (TARGET=500, the build's banked floor).
    tps_build_floor = tps_after_read_cut(TARGET_TPS, f_dep)
    uplift_build_floor_tps = tps_build_floor - TARGET_TPS

    return {
        "read_cut_byte_cut_pct": MAX_PPL_SAFE_READ_REDUCTION_PCT,
        "deployed_read_fraction": f_dep,
        "composed_read_credit_pct": composed_read_credit_pct,          # 8.43% x 38%
        "deployed": {
            "operating_tps": OFFICIAL_BASELINE_TPS,
            "tps_after_read_cut": tps_dep,
            "uplift_pct": uplift_dep_pct,
            "uplift_tps": uplift_dep_tps,
            "matches_287_bank": bool(deployed_crosscheck_ok),
        },
        "build": {
            "build_et_target": BUILD_ET_TARGET,
            "build_tps_bracket": [build_tps_lo, build_tps_hi],
            "build_step_us_bracket": [build_step_us_lo, build_step_us_hi],
            "build_read_fraction_bracket": [build_read_frac_lo, build_read_frac_hi],
            "build_read_fraction_mid": build_read_frac_mid,
            "composed_credit_idealized_pct": composed_build_credit_idealized_pct,
            "composed_credit_diluted_pct": composed_build_credit_diluted_pct,
            "idealized_uplift_at_floor_tps": uplift_build_floor_tps,
        },
        # The build-invariant TPS-uplift % (the stark #273-consistent "composed gain"):
        # the read-cut removes wall-fraction phi -> uplift = phi/(1-phi).
        "composed_uplift_pct": uplift_dep_pct,
    }


# --------------------------------------------------------------------------- #
# 3. Realization-discount by stark #273's body-side ratios (the PRIOR, not a wall A/B).
# --------------------------------------------------------------------------- #
def realization_discount(composed_uplift_pct: float) -> dict[str, Any]:
    """realization_ratio = measured_wall_delta / composed_gain (stark #273 convention).
    Transfer the body-side prior onto the read-cut (same ONEGRAPH-deviation class)."""
    ratios = STARK273_BODY_SIDE_RATIOS
    ratio_values = list(ratios.values())
    ratio_min = min(ratio_values)
    ratio_max = max(ratio_values)
    ratio_mean = sum(ratio_values) / len(ratio_values)

    # Headline prior = K4 (-2.018): the load-bearing body-side precedent. K4's composition
    # predicted +4.28% (clears 500) but the wall REFUTED it to -8.63% -- directly analogous
    # to the read-cut whose composition predicts a positive credit.
    read_cut_realization_ratio = K4_REALIZATION_RATIO

    # Apply the ratio: realized_uplift = composed_uplift * ratio (stark #273 algebra).
    def realized_pct(r: float) -> float:
        return composed_uplift_pct * r

    realized_uplift_pct = realized_pct(read_cut_realization_ratio)
    realized_pct_bracket = sorted(realized_pct(r) for r in (ratio_min, ratio_max))
    realized_pct_mean = realized_pct(ratio_mean)

    # Absolute realized TPS credit at three anchors (all share the sign of the ratio).
    def realized_tps(anchor: float, r: float) -> float:
        return anchor * realized_pct(r) / 100.0

    realized_tps_deployed = realized_tps(OFFICIAL_BASELINE_TPS, read_cut_realization_ratio)
    realized_tps_build_floor = realized_tps(TARGET_TPS, read_cut_realization_ratio)
    realized_tps_lambda_ceiling = realized_tps(LAMBDA1_CEILING_TPS, read_cut_realization_ratio)

    if read_cut_realization_ratio >= REALIZES_RATIO:
        classification = "realizes"
    elif read_cut_realization_ratio > PARTIAL_LO:
        classification = "partial"
    else:
        classification = "regresses"

    return {
        "read_cut_realization_ratio": read_cut_realization_ratio,
        "stark273_body_side_ratios": ratios,
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
        "ratio_mean": ratio_mean,
        "all_body_side_ratios_negative": all(r <= 0 for r in ratio_values),
        "realized_uplift_pct": realized_uplift_pct,
        "realized_uplift_pct_bracket": realized_pct_bracket,
        "realized_uplift_pct_at_mean_ratio": realized_pct_mean,
        # PR-required headline: realized TPS credit "on the build" (build floor = 500 anchor).
        "read_cut_wall_realized_tps_credit": realized_tps_build_floor,
        "realized_tps_credit_deployed_anchor": realized_tps_deployed,
        "realized_tps_credit_build_floor_anchor": realized_tps_build_floor,
        "realized_tps_credit_lambda_ceiling_anchor": realized_tps_lambda_ceiling,
        "classification": classification,
    }


# --------------------------------------------------------------------------- #
# Constants exact-import check (self-test (d)).
# --------------------------------------------------------------------------- #
def constants_exact() -> tuple[bool, dict[str, Any]]:
    """Every banked import must round to its PR-stated shorthand EXACTLY (no drift)."""
    # (precise_value, pr_shorthand, decimals)
    expect = {
        "MAX_PPL_SAFE_READ_REDUCTION_PCT": (MAX_PPL_SAFE_READ_REDUCTION_PCT, 8.431, 3),
        "READ_CUT_PROJ_PPL": (READ_CUT_PROJ_PPL, 2.3975, 4),
        "DEPLOYED_INT4_PPL": (DEPLOYED_INT4_PPL, 2.3772, 4),
        "BODY_READ_FRACTION": (BODY_READ_FRACTION, 0.38, 2),
        "HBM_BOUND_CEILING_TPS": (HBM_BOUND_CEILING_TPS, 1265.6, 1),
        "K4_VS_K7_MEASURED_DELTA_PCT": (K4_VS_K7_MEASURED_DELTA_PCT, -8.629, 3),
        "K4_REALIZATION_RATIO": (K4_REALIZATION_RATIO, -2.018, 3),
        "K5_REALIZATION_RATIO": (K5_REALIZATION_RATIO, -0.864, 3),
        "BUILD_ET_TARGET": (BUILD_ET_TARGET, 4.9029, 4),
        "PUBLIC_FLOOR_ET": (PUBLIC_FLOOR_ET, 4.966, 3),
        "OFFICIAL_BASELINE_TPS": (OFFICIAL_BASELINE_TPS, 481.53, 2),
        "E_T_DEPLOYED": (E_T_DEPLOYED, 3.844, 3),
        "K_CAL": (K_CAL, 125.268, 3),
        "STEP_US": (STEP_US, 1218.2, 1),
        "LAMBDA1_CEILING_TPS": (LAMBDA1_CEILING_TPS, 520.953, 3),
        "PPL_GATE": (PPL_GATE, 2.42, 2),
        "TARGET_TPS": (TARGET_TPS, 500.0, 1),
    }
    detail, all_ok = {}, True
    for k, (have, want, dec) in expect.items():
        ok = round(float(have), dec) == round(float(want), dec)
        detail[k] = {"have": have, "want_shorthand": want, "decimals": dec, "ok": ok}
        all_ok = all_ok and ok
    return all_ok, detail


# --------------------------------------------------------------------------- #
# Compose verdict + PRIMARY self-test.
# --------------------------------------------------------------------------- #
def run() -> dict[str, Any]:
    ppl = ppl_headroom_fit()
    comp = compose_read_credit()
    disc = realization_discount(comp["composed_uplift_pct"])

    composed_uplift_pct = comp["composed_uplift_pct"]
    realized_uplift_pct = disc["realized_uplift_pct"]
    ratio = disc["read_cut_realization_ratio"]
    realized_tps_credit = disc["read_cut_wall_realized_tps_credit"]

    # VERDICT. free companion iff the read-cut REALIZES as positive stackable TPS
    # (ratio >= 0.8 AND the realized credit is materially positive vs a ~2% gate).
    material_gate_pct = 2.0
    realizes_positive = (ratio >= REALIZES_RATIO) and (realized_uplift_pct >= material_gate_pct)
    read_cut_is_free_build_companion = bool(realizes_positive)

    # ---- PRIMARY self-test ----
    const_ok, const_detail = constants_exact()

    # (a) headroom arithmetic round-trips.
    a_headroom = round(ppl["ppl_headroom"], 4) == 0.0428
    a_cost = round(ppl["read_cut_ppl_cost"], 4) == 0.0203
    a_fits = ppl["read_cut_ppl_cost"] <= ppl["ppl_headroom"]
    # (b) composed credit + ratio finite; the discounted credit <= the composed credit.
    b_finite = math.isfinite(composed_uplift_pct) and math.isfinite(ratio)
    b_discount_le_composed = realized_uplift_pct <= composed_uplift_pct
    # (c) all physical TPS/PPL operating points >= 0 and finite (the realized credit is
    #     negative BY DESIGN -- a regression -- so it is excluded from the >=0 set).
    physical = [
        OFFICIAL_BASELINE_TPS, TARGET_TPS, LAMBDA1_CEILING_TPS, PRIVATE_VERIFIED_TPS,
        HBM_BOUND_CEILING_TPS, TPS_AT_MAX_SAFE_DEPLOYED, comp["deployed"]["tps_after_read_cut"],
        DEPLOYED_INT4_PPL, READ_CUT_PROJ_PPL, PPL_GATE, ppl["ppl_headroom"],
        ppl["read_cut_ppl_cost"], composed_uplift_pct, comp["composed_read_credit_pct"],
        E_T_DEPLOYED, BUILD_ET_TARGET, PUBLIC_FLOOR_ET,
    ]
    c_nonneg = all(v >= 0.0 for v in physical)
    everything = physical + [
        realized_uplift_pct, ratio, realized_tps_credit,
        disc["realized_tps_credit_deployed_anchor"],
        disc["realized_tps_credit_lambda_ceiling_anchor"],
    ]
    c_nan_clean = all(math.isfinite(v) for v in everything)
    # (d) constants imported EXACT.
    d_const = const_ok
    # (e) leg carries the 0-TPS + analytic-not-a-wall-A/B + assumes-body-unchanged caveats.
    blob = " ".join(CAVEATS)
    e_caveats = (
        len(CAVEATS) >= 6
        and "0_TPS" in blob and "ANALYTIC_NOT_A_WALL_AB" in blob
        and "ASSUMES_BODY_UNCHANGED" in blob and ppl["assumes_body_unchanged"] is True
    )

    checks = {
        "a_headroom_eq_0428": bool(a_headroom),
        "a_read_cut_cost_eq_0203": bool(a_cost),
        "a_cost_fits_headroom": bool(a_fits),
        "b_composed_and_ratio_finite": bool(b_finite),
        "b_discounted_le_composed": bool(b_discount_le_composed),
        "c_physical_values_nonneg": bool(c_nonneg),
        "c_all_values_nan_clean": bool(c_nan_clean),
        "d_constants_imported_exact": bool(d_const),
        "e_caveats_present": bool(e_caveats),
        # consistency guard: the companion verdict must agree with the ratio sign-class.
        "verdict_consistent_with_ratio": (
            read_cut_is_free_build_companion == (disc["classification"] == "realizes")
        ),
    }
    self_test_passes = all(checks.values())

    # one-line build-portfolio recommendation.
    if read_cut_is_free_build_companion:
        recommendation = (
            "STACK the #287 read-cut as a free body-side build companion: it FITS the "
            f"0.0428 PPL headroom and REALIZES (ratio {ratio:+.3f}) as +{realized_uplift_pct:.2f}% "
            "stackable TPS on the build."
        )
    else:
        recommendation = (
            "BUILD ALONE; bank the #287 read-cut as a CLOSED companion (NOT an integration "
            f"target): it FITS the 0.0428 PPL headroom but REGRESSES on the wall (ratio "
            f"{ratio:+.3f} <= 0, realized {realized_uplift_pct:+.2f}%) the same way stark #273's "
            "body-side static-K deviations did -> doubly-closed (not standalone #287, not a "
            "companion this leg)."
        )

    handoff = (
        f"The #287 read-cut FITS the EAGLE-3 build's 0.0428 PPL headroom (cost "
        f"{ppl['read_cut_ppl_cost']:.4f}, body unchanged), but on the wall it "
        + (
            f"REALIZES as a free body-side companion (ratio {ratio:+.3f}, +{realized_uplift_pct:.2f}% TPS)"
            if read_cut_is_free_build_companion else
            f"REGRESSES like static-K (ratio {ratio:+.3f} <= 0, doubly-closed)"
        )
        + ", so the build portfolio should "
        + ("STACK the read-cut as a free companion" if read_cut_is_free_build_companion
           else "treat it as a CLOSED companion and build alone")
        + ", resolving where #287's read-frontier lands relative to the gated build."
    )

    out = {
        "primary_metric": {"name": "read_cut_build_companion_self_test_passes",
                           "value": bool(self_test_passes)},
        "test_metric": {"name": "read_cut_is_free_build_companion",
                        "value": bool(read_cut_is_free_build_companion)},
        # headline verdict fields
        "read_cut_build_companion_self_test_passes": bool(self_test_passes),
        "read_cut_is_free_build_companion": bool(read_cut_is_free_build_companion),
        "read_cut_fits_ppl_headroom": ppl["read_cut_fits_ppl_headroom"],
        "read_cut_wall_realized_tps_credit": realized_tps_credit,
        "read_cut_realization_ratio": ratio,
        "realization_classification": disc["classification"],
        "composed_read_credit_pct": comp["composed_read_credit_pct"],
        "composed_uplift_pct": composed_uplift_pct,
        "realized_uplift_pct": realized_uplift_pct,
        "build_portfolio_recommendation": recommendation,
        "handoff": handoff,
        # sub-results
        "ppl_headroom_fit": ppl,
        "compose": comp,
        "realization": disc,
        "assumes_body_unchanged": True,
        "read_reduction_lever_clears_500_standalone": READ_REDUCTION_LEVER_CLEARS_500,
        # gate/anchor echo
        "official_baseline_tps": OFFICIAL_BASELINE_TPS,
        "target_tps": TARGET_TPS,
        "lambda1_ceiling_tps": LAMBDA1_CEILING_TPS,
        "private_verified_tps": PRIVATE_VERIFIED_TPS,
        "launch_gate": {
            "measured_tps_ge": TARGET_TPS, "lambda_hat_ge": LAUNCH_LAMBDA_HAT,
            "ppl_le": PPL_GATE, "vram_le_gb": 24, "human_approval_gated": True,
            "deployed_checkpoint": DEPLOYED_CHECKPOINT,
        },
        "constants_exact": const_ok,
        "constants_detail": const_detail,
        "caveats": CAVEATS,
        "self_test": {"passes": bool(self_test_passes), "checks": checks},
    }
    return out


# --------------------------------------------------------------------------- #
# Output: plot + wandb.
# --------------------------------------------------------------------------- #
def make_plot(out: dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- left: PPL-headroom fit ----
    ppl = out["ppl_headroom_fit"]
    gate = ppl["ppl_gate"]
    build = ppl["build_ppl_body_unchanged"]
    cut = build + ppl["read_cut_ppl_cost"]
    axL.axhline(gate, color="k", ls="--", lw=1.3, label=f"PPL gate {gate}")
    axL.bar(["build\n(body unchanged)"], [build], color="tab:blue", width=0.5,
            label=f"build PPL {build}")
    axL.bar(["+ #287 read-cut"], [cut], color="tab:green", width=0.5,
            label=f"read-cut PPL {cut:.4f}")
    axL.annotate(f"headroom\n{ppl['ppl_headroom']:.4f}", (0, build),
                 xytext=(0, build + 0.012), ha="center", fontsize=8,
                 arrowprops=dict(arrowstyle="<->", color="gray"))
    axL.annotate(f"cost {ppl['read_cut_ppl_cost']:.4f}\n(FITS {ppl['read_cut_fits_ppl_headroom']})",
                 (1, build), xytext=(1, build + 0.012), ha="center", fontsize=8, color="tab:green")
    axL.set_ylim(build - 0.01, gate + 0.015)
    axL.set_ylabel("projected deployed PPL")
    axL.set_title("(1) read-cut FITS the build PPL headroom\n(body unchanged: 0.0203 <= 0.0428)")
    axL.legend(fontsize=8, loc="lower left")
    axL.grid(alpha=0.3, axis="y")

    # ---- right: composed credit -> realization discount ----
    disc = out["realization"]
    composed = out["composed_uplift_pct"]
    realized = out["realized_uplift_pct"]
    rb = disc["realized_uplift_pct_bracket"]
    axR.axhline(0, color="k", lw=0.8)
    axR.bar(["idealized\ncomposition"], [composed], color="tab:green", width=0.5,
            label=f"composed +{composed:.2f}%\n(8.43% x 38%)")
    axR.bar(["stark #273\nwall-realized"], [realized], color="tab:red", width=0.5,
            label=f"realized {realized:+.2f}%\n(ratio {disc['read_cut_realization_ratio']:+.3f})")
    axR.errorbar([1], [realized], yerr=[[realized - rb[0]], [rb[1] - realized]],
                 fmt="none", ecolor="darkred", capsize=5,
                 label=f"ratio range [{disc['ratio_min']:.2f},{disc['ratio_max']:.2f}] (all<0)")
    verdict = "FREE COMPANION" if out["read_cut_is_free_build_companion"] else "DOUBLY-CLOSED (regresses)"
    axR.set_ylabel("TPS-uplift % on the build")
    axR.set_title(f"(2) realization discount -> {verdict}\n"
                  f"read_cut_is_free_build_companion = {out['read_cut_is_free_build_companion']}")
    axR.legend(fontsize=8, loc="upper right")
    axR.grid(alpha=0.3, axis="y")

    fig.suptitle("PR #302: is the #287 read-cut a FREE body-side companion to the EAGLE-3 build? "
                 "(analytic, 0 TPS)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    log("wrote plot", path)


def log_wandb(args, out: dict[str, Any]):
    if args.no_wandb:
        return None
    sys.path.insert(0, "/workspace/senpai/target")
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        log(f"wandb import failed ({exc}); skipping")
        return None
    run = wandb_logging.init_wandb_run(
        job_type="read-cut-build-companion", agent="fern",
        name=args.wandb_name or "fern/read-cut-build-companion",
        group=args.wandb_group or "read-cut-build-companion",
        tags=["validity", "read-cut", "build-companion", "realization-ratio",
              "analytic", "0-tps", "pr302"],
        config={
            "read_cut_is_free_build_companion": out["read_cut_is_free_build_companion"],
            "read_cut_fits_ppl_headroom": out["read_cut_fits_ppl_headroom"],
            "read_cut_realization_ratio": out["read_cut_realization_ratio"],
            "composed_uplift_pct": out["composed_uplift_pct"],
            "ppl_gate": PPL_GATE, "official_baseline_tps": OFFICIAL_BASELINE_TPS,
        },
    )
    if run is None:
        log("wandb disabled; skipping")
        return None
    import wandb
    flat = {
        "read_cut_build_companion_self_test_passes": 1.0 if out["read_cut_build_companion_self_test_passes"] else 0.0,
        "read_cut_is_free_build_companion": 1.0 if out["read_cut_is_free_build_companion"] else 0.0,
        "read_cut_fits_ppl_headroom": 1.0 if out["read_cut_fits_ppl_headroom"] else 0.0,
        "read_cut_wall_realized_tps_credit": out["read_cut_wall_realized_tps_credit"],
        "read_cut_realization_ratio": out["read_cut_realization_ratio"],
        "composed_read_credit_pct": out["composed_read_credit_pct"],
        "composed_uplift_pct": out["composed_uplift_pct"],
        "realized_uplift_pct": out["realized_uplift_pct"],
        "ppl_headroom": out["ppl_headroom_fit"]["ppl_headroom"],
        "read_cut_ppl_cost": out["ppl_headroom_fit"]["read_cut_ppl_cost"],
        "headroom_residual_after_read_cut": out["ppl_headroom_fit"]["headroom_residual_after_read_cut"],
        "realized_tps_credit_deployed_anchor": out["realization"]["realized_tps_credit_deployed_anchor"],
        "realized_tps_credit_lambda_ceiling_anchor": out["realization"]["realized_tps_credit_lambda_ceiling_anchor"],
        "ratio_min": out["realization"]["ratio_min"],
        "ratio_max": out["realization"]["ratio_max"],
        "ratio_mean": out["realization"]["ratio_mean"],
        "build_read_fraction_mid": out["compose"]["build"]["build_read_fraction_mid"],
        "composed_build_credit_diluted_pct": out["compose"]["build"]["composed_credit_diluted_pct"],
        "global_step": 0,
    }
    for k, v in out["self_test"]["checks"].items():
        flat[f"selftest/{k}"] = 1.0 if v else 0.0
    run.log(flat)

    # realization-ratio transfer table
    rtbl = wandb.Table(columns=["k_config", "stark273_realization_ratio",
                                "read_cut_realized_uplift_pct_if_applied"])
    for k, r in out["realization"]["stark273_body_side_ratios"].items():
        rtbl.add_data(int(k), r, out["composed_uplift_pct"] * r)
    run.log({"realization_ratio_transfer": rtbl, "global_step": 0})

    plot_path = OUTDIR / "read_cut_build_companion.png"
    if plot_path.exists():
        run.log({"companion_plot": wandb.Image(str(plot_path)), "global_step": 0})
    run.summary.update({k: v for k, v in flat.items() if k != "global_step"})
    run.summary.update({
        "realization_classification": out["realization_classification"],
        "build_portfolio_recommendation": out["build_portfolio_recommendation"],
        "handoff": out["handoff"],
    })
    rid = run.id
    run.finish()
    log("wandb run", rid)
    return rid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-name", "--wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--self-test", "--self_test", action="store_true",
                    help="no-op alias: the run always evaluates the PRIMARY self-test")
    args = ap.parse_args()

    out = run()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "read_cut_build_companion_results.json").write_text(json.dumps(out, indent=2))
    log("wrote JSON")
    make_plot(out, OUTDIR / "read_cut_build_companion.png")
    rid = log_wandb(args, out)
    out["wandb_run_id"] = rid
    (OUTDIR / "read_cut_build_companion_results.json").write_text(json.dumps(out, indent=2))

    log("=" * 72)
    log(f"PRIMARY read_cut_build_companion_self_test_passes = {out['read_cut_build_companion_self_test_passes']}")
    log(f"TEST    read_cut_is_free_build_companion         = {out['read_cut_is_free_build_companion']}")
    log(f"read_cut_fits_ppl_headroom = {out['read_cut_fits_ppl_headroom']}  "
        f"(cost {out['ppl_headroom_fit']['read_cut_ppl_cost']:.4f} <= headroom "
        f"{out['ppl_headroom_fit']['ppl_headroom']:.4f})")
    log(f"composed credit  = +{out['composed_uplift_pct']:.3f}% TPS-uplift  "
        f"(8.43% x 38% = {out['composed_read_credit_pct']:.3f}% wall-fraction)")
    log(f"realization ratio = {out['read_cut_realization_ratio']:+.4f}  "
        f"-> classification = {out['realization_classification']}")
    log(f"realized credit  = {out['realized_uplift_pct']:+.3f}%  "
        f"= {out['read_cut_wall_realized_tps_credit']:+.2f} TPS (build-floor anchor)")
    log(f"recommendation: {out['build_portfolio_recommendation']}")
    log("checks: " + json.dumps(out["self_test"]["checks"]))


if __name__ == "__main__":
    main()
