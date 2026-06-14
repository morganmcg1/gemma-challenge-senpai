#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Launch-sigma UNIT REBASE: clean-1-sigma footing -> re-solved GO trigger (PR #204).

WHAT THIS IS
------------
ubel #201 (MERGED) banked the de-dup x realistic-ICC sigma-inflation mechanism
cleanly (anchors exact, self-test 7/7), but its COMBINED launch sigma mixed unit
footings: the dominant ACCEPTANCE leg sat on a 95%-two-sided CI HALF-WIDTH footing
(z=1.96), while sigma_hw and sigma_private sat on a 1-sigma footing -- then ALL
three were quadrature-summed and the result multiplied by z_p95 (one-sided) in
LCB(mu)=mu - z*sigma. That DOUBLE-COUNTS z on the acceptance leg. This PR re-bases
EVERY leg onto a single clean 1-sigma footing (the #194 convention), re-solves the
GO trigger, and reports its direction relative to the lambda=1 ceiling 520.953 --
the central GO/NO-GO verdict fern #185 is waiting on.

THE CRUX -- IS THE ACCEPTANCE LEG A 1-SIGMA OR A HALF-WIDTH?  (answer: HALF-WIDTH)
--------------------------------------------------------------------------------
The acceptance leg traces, footing-preserving, to a 95%-two-sided HALF-WIDTH:

  #175 sampling CI ........ "tps_finite_sample_ci_HALFWIDTH (both-bugs, z=1.96)"
        |                   (#195.axis_sigmas.sampling.source -- explicit z=1.96)
  #187 de-dup ............. h_in(3.710 HW) (+) h_out(5.178 HW), overlap-corrected
        |                   -> overlap_corrected_same_bench 5.31870  ("h_" = HALF-WIDTH)
  #190 sqrt(D)=2.100 ...... DIMENSIONLESS ratio (22.905 HW / 10.906 HW) -> preserves footing
        v
  #201 acceptance leg ..... 5.31870 (HW, iid) * 2.100 = 11.17004 (HW, realistic-ICC)

So 11.170 is a 95%-two-sided HALF-WIDTH, NOT a 1-sigma. #194 (the CLEAN anchor)
treats the SAME #175 lineage correctly: sigma_sample_1sigma = 10.906 / z2(1.95996)
= 5.5645, then break-even = 500 + z1(1.64485) * hypot(5.5645, sigma_hw) = 512.157.
#201 skipped the /z2 step on the acceptance leg.

THE CLEAN REBASE (convention A: all-1-sigma, then x z_p95 one-sided)
-------------------------------------------------------------------
  acc_1sigma = acc_HALFWIDTH / z2  =  11.17004 / 1.95996  =  5.69911 TPS
  sigma_hw   = 4.86447 (already 1-sigma; kanna #188 std, #194 "sigma_hw_1sigma_tps")
  sigma_priv = 0.88392 (kept 1-sigma per PR instruction 3; negligible-footing leg)
  sigma_launch_clean = hypot(acc_1sigma, sigma_hw, sigma_priv) [+ rho corner]
  mu_clears_500_clean = 500 + z1 * sigma_launch_clean

Because the mis-based leg is the DOMINANT one and the fix DIVIDES it down by z2,
the combined sigma SHRINKS (12.215 -> 7.545) and the GO trigger DROPS (520.09 ->
512.41): the rebase is LESS conservative, the OPPOSITE sign to #201's "~+3.14 TPS
more conservative" scoping. The honest sign is the whole point (PR step 4).

THE TWO SELF-CONSISTENT CONVENTIONS (PR step 2 -- anchor BOTH so direction can't drift)
---------------------------------------------------------------------------------------
  (A) all-1-sigma-then-xz : LCB = mu - z * hypot(sigma_i)              [#194 convention]
  (B) all-half-width      : LCB = mu - hypot(HW_i), HW_i = z * sigma_i [#190 convention]
With the SAME z linking HW_i = z*sigma_i, A and B give the IDENTICAL LCB (hypot is
1-homogeneous: hypot(z*sigma_i) = z*hypot(sigma_i)). #201's bug was carrying ONE leg
as a z2-based half-width through a z1 LCB -- a THIRD, inconsistent, basis.

SCOPE
-----
LOCAL CPU-ONLY analytic unit re-basing over EXISTING MERGED #201/#195/#194/#190
curves. No GPU / vLLM / HF Job / submission / served-file change. Takes NO official
draws, authorizes none. BASELINE stays 481.53; greedy identity untouched; adds 0
TPS (PRIMARY = self-test). NOT a launch. The rho(*,hw) [-0.3,+0.3] band still needs
land #71's co-log (a separate lever, unchanged here).

SELF-TEST (PR step 6 -- PRIMARY)
-------------------------------
(a) convention A and B give the same LCB on a consistently-based vector (err ~0);
(b) sigma_priv->0, sigma_hw->0, single acceptance leg -> LCB = mu - z*sigma_acc
    (textbook one-axis one-sided P95);
(c) re-basing an ALREADY-all-1-sigma vector is a NO-OP (idempotent);
(d) the rebase machinery reconstructs #194's 512.157 break-even from the raw #175
    sampling half-width (the #194 anchor SURVIVES the rebase);
(e) worst-case >= central after rebase;
(f) NaN-clean across all reported scalars.
PRIMARY = unit_rebase_self_test_passes (bool);
TEST    = mu_clears_500_clean_central (float TPS).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # research/validity/launch_sigma_unit_rebase -> repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# source-of-truth artifacts (imported verbatim; one source per constant).
# We re-read ubel's OWN banked #201 curve (read-only) for the launch legs, and
# the #194 redraw_budget for the clean break-even anchor + its explicit footing
# decomposition (the canonical half-width -> 1-sigma conversion).
# ---------------------------------------------------------------------------
CLOSURE_201 = os.path.join(_ROOT, "research/validity/launch_sigma_closure/launch_sigma_closure_results.json")
REDRAW_194 = os.path.join(_ROOT, "research/validity/redraw_budget/redraw_budget_results.json")
CICOV_195 = os.path.join(_ROOT, "research/validity/ci_axis_covariance/ci_axis_covariance_results.json")

# z conventions. A 95% TWO-SIDED CI half-width is z2*sigma; a one-sided P95 LCB is
# mu - z1*sigma. The clean rebase recovers a 1-sigma from a 95% half-width via /z2,
# then applies z1 in the LCB -- exactly #194's principled mix.
Z95_ONE_SIDED = 1.6448536269514722  # scipy.stats.norm.ppf(0.95)   -> LCB multiplier
Z95_TWO_SIDED = 1.959963984540054   # scipy.stats.norm.ppf(0.975)  -> 95% CI half-width / sigma
TARGET = 500.0
RHO_PLUS = 0.3                      # #195 bounded rho(*,hw) worst-case corner (PSD-admissible)
ANCHOR_TOL_TPS = 0.5               # self-test (d): |reconstruction - #194 break-even| <= 0.5 TPS
IDENT_TOL = 1e-9                   # convention A==B / idempotence numerical tolerance


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _hypot(vec: list[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in vec))


def _combined_sigma_rho(sig: np.ndarray, rho: float) -> float:
    """Combined sigma over 3 axes with a common off-diagonal rho (PSD-admissible)."""
    R = np.full((sig.size, sig.size), rho, dtype=float)
    np.fill_diagonal(R, 1.0)
    C = np.outer(sig, sig) * R
    return math.sqrt(max(float(np.ones(sig.size) @ C @ np.ones(sig.size)), 0.0))


# ---------------------------------------------------------------------------
# Step 0 -- import the banked legs (NOT re-derived).
# ---------------------------------------------------------------------------
def import_banked() -> dict[str, Any]:
    c201 = _load(CLOSURE_201)
    legs201 = c201["legs"]
    recon201 = c201["dedup_x_icc"]
    lcb201 = c201["lcb"]

    rb = _load(REDRAW_194)
    dec194 = rb["decomposition"]  # keys live one level down: dec194[<name>]

    cov195 = _load(CICOV_195)
    samp195 = cov195["axis_sigmas"]["sampling"]

    out = {
        # --- #201 launch legs (the curve being re-based) ---
        "acc_iid_halfwidth": float(recon201["acceptance_sigma_dedup_iid"]),            # 5.318697 (HW, iid)
        "acc_realistic_halfwidth": float(recon201["acceptance_sigma_dedup_realistic_icc"]),  # 11.170041 (HW, realistic)
        "sqrt_design_effect": float(recon201["sqrt_design_effect_inflation"]),         # 2.100146 (dimensionless)
        "sigma_hw": float(legs201["sigma_hw"]),                                        # 4.864469 (1-sigma)
        "sigma_private": float(legs201["sigma_private"]),                              # 0.883918
        "combined_201_central": float(c201["combined_sigma_launch_central"]),          # 12.215326 (mixed basis)
        "combined_201_worstcase": float(c201["combined_sigma_launch_worstcase"]),      # 13.795648 (mixed basis)
        "mu_201_central": float(c201["mu_clears_500_central"]),                        # 520.092424 (mixed basis)
        "mu_201_worstcase": float(c201["mu_clears_500_worstcase"]),                    # 522.691822 (mixed basis)
        "combined_dedup_195": float(legs201["combined_dedup_195"]),                    # 7.261743 (#195 iid de-dup, mixed)
        "lambda1_ceiling": float(lcb201["lambda1_ceiling_mu"]),                        # 520.952732 (#194)
        # --- #194 clean anchor + its explicit footing decomposition ---
        "mu_break_even_194": float(rb["mu_single_shot_safe_tps"]),                     # 512.157071 (clean all-1-sigma anchor)
        "sigma_draw_194": float(rb["budget"]["sigma_draw_tps"]),                       # 7.390974 (clean per-draw 1-sigma)
        "sampling_halfwidth_175": float(dec194["sampling_halfwidth_95_both_bugs_tps"]),  # 10.906182 (HW)
        "sigma_sample_1sigma_194": float(dec194["sigma_sample_1sigma_tps"]),             # 5.564481 (= 10.906/z2)
        "sigma_hw_1sigma_194": float(dec194["sigma_hw_1sigma_tps"]),                     # 4.864469
        "z95_two_sided_194": float(dec194["z95_two_sided"]),                             # 1.959964 (confirms z2 footing)
        # --- provenance strings for the footing audit ---
        "prov_sampling_175": str(samp195["source"]),
    }
    return out


# ---------------------------------------------------------------------------
# Step 1 -- diagnose the mixed basis precisely (leg_footing_audit).
# ---------------------------------------------------------------------------
def leg_footing_audit(b: dict[str, Any]) -> dict[str, Any]:
    z2 = Z95_TWO_SIDED
    audit = {
        "acceptance": {
            "value": b["acc_realistic_halfwidth"],
            "footing": "halfwidth_95_two_sided",
            "z_applied": z2,
            "needs_rebase": True,
            "clean_1sigma": b["acc_realistic_halfwidth"] / z2,
            "provenance": (
                "#175 '%s' -> #187 overlap-corrected de-dup of h_in(3.710 HW)+h_out(5.178 HW) "
                "= 5.31870 HW (iid) -> #190 sqrt(D)=2.100 (DIMENSIONLESS ratio, footing-preserving) "
                "-> 11.17004 HW (realistic). The footing was set by #175's z=1.96 CI and never divided out."
                % b["prov_sampling_175"]
            ),
        },
        "hardware": {
            "value": b["sigma_hw"],
            "footing": "one_sigma",
            "z_applied": 1.0,
            "needs_rebase": False,
            "clean_1sigma": b["sigma_hw"],
            "provenance": "kanna #188 sigma_oneshot = hypot(within, between) -- a standard deviation; "
            "#194 labels it 'sigma_hw_1sigma_tps' = 4.864469. Already 1-sigma.",
        },
        "private": {
            "value": b["sigma_private"],
            "footing": "one_sigma_per_instruction",
            "z_applied": 1.0,
            "needs_rebase": False,
            "clean_1sigma": b["sigma_private"],
            "provenance": "stark #176/#191 drop CI mapped to TPS. PR instruction 3 fixes this leg at "
            "1-sigma. NOTE (honest flag): #195 derives it as drop_halfwidth_pp(0.17)*slope, so IF the "
            "underlying drop CI is 95%-two-sided this leg would also carry a half-width footing; re-basing "
            "it (/z2 -> 0.451) moves the clean central trigger by < 0.06 TPS (negligible vs the acceptance "
            "leg's 7.68 TPS) and does NOT change the verdict -- see private_footing_sensitivity.",
        },
    }
    # the bug in one line: #201 carried 'acceptance' as a z2-based half-width but multiplied the
    # quadrature by z1 in the LCB -> a third (inconsistent) basis, double-counting z on the dominant leg.
    audit["_mixed_basis_diagnosis"] = (
        "MIXED: acceptance on a 95%%-two-sided HALF-WIDTH footing (z2=%.5f) vs hardware/private on 1-sigma; "
        "#201 quadrature-summed them and applied z1=%.5f in LCB(mu)=mu-z1*sigma. The acceptance leg's "
        "z2 is thereby never divided out -> its contribution to the LCB shift is over-stated by a factor "
        "of z2 (~1.96x). The fix: divide the acceptance half-width by z2 to a clean 1-sigma."
        % (Z95_TWO_SIDED, Z95_ONE_SIDED)
    )
    audit["_crux_acceptance_5p3187_is_a_halfwidth"] = True
    return audit


# ---------------------------------------------------------------------------
# Step 2 -- anchor BOTH self-consistent conventions; confirm identical LCB.
# ---------------------------------------------------------------------------
def conventions_self_consistency(sig_vec_1sigma: list[float], rho: float = 0.0) -> dict[str, Any]:
    z = Z95_ONE_SIDED
    sig = np.array(sig_vec_1sigma, dtype=float)
    # convention A: all-1-sigma, combine in quadrature(+rho), then x z one-sided.
    sigma_comb_A = _combined_sigma_rho(sig, rho)
    lcb_A = TARGET - z * sigma_comb_A
    # convention B: every leg a 95% (here: the SAME one-sided z) half-width HW_i = z*sigma_i,
    # combine in quadrature(+rho), LCB = mu - HW_comb (no further x z).
    hw = z * sig
    hw_comb_B = _combined_sigma_rho(hw, rho)
    lcb_B = TARGET - hw_comb_B
    return {
        "convention_A_lcb": lcb_A,
        "convention_B_lcb": lcb_B,
        "abs_err": abs(lcb_A - lcb_B),
        "identical": bool(abs(lcb_A - lcb_B) < IDENT_TOL),
        "sigma_comb_A": sigma_comb_A,
        "halfwidth_comb_B": hw_comb_B,
        "note": "hypot is 1-homogeneous: hypot(z*sigma_i) = z*hypot(sigma_i), so A and B coincide "
        "exactly WHEN the SAME z links HW_i=z*sigma_i. #201 violated this by carrying one leg's HW on "
        "z2 while the LCB used z1 -- a third, inconsistent basis.",
    }


# ---------------------------------------------------------------------------
# Helper -- the rebase operator: divide half-width-footed legs by z2, keep 1-sigma legs.
# Idempotent on an all-1-sigma vector (self-test c).
# ---------------------------------------------------------------------------
def rebase_legs_to_1sigma(values: list[float], footings: list[str]) -> list[float]:
    out = []
    for v, f in zip(values, footings):
        if f.startswith("halfwidth"):
            out.append(v / Z95_TWO_SIDED)
        else:
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# Step 3 -- re-solve the clean GO trigger (THE DELIVERABLE).
# ---------------------------------------------------------------------------
def resolve_clean_trigger(b: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    z = Z95_ONE_SIDED
    # put every leg on a clean 1-sigma footing.
    acc_1s = audit["acceptance"]["clean_1sigma"]   # 11.170 / z2 = 5.69911
    hw_1s = audit["hardware"]["clean_1sigma"]
    pv_1s = audit["private"]["clean_1sigma"]
    sig = np.array([acc_1s, hw_1s, pv_1s], dtype=float)

    comb_central = _combined_sigma_rho(sig, 0.0)
    comb_worstcase = _combined_sigma_rho(sig, RHO_PLUS)
    mu_central = TARGET + z * comb_central
    mu_worstcase = TARGET + z * comb_worstcase

    ceiling = b["lambda1_ceiling"]
    delta_central = mu_central - b["mu_201_central"]
    delta_worstcase = mu_worstcase - b["mu_201_worstcase"]

    # private-footing sensitivity: if private were ALSO a half-width (the honest flag in the audit).
    pv_alt = b["sigma_private"] / Z95_TWO_SIDED
    comb_central_pvalt = _combined_sigma_rho(np.array([acc_1s, hw_1s, pv_alt]), 0.0)
    mu_central_pvalt = TARGET + z * comb_central_pvalt

    return {
        "acceptance_1sigma_clean": acc_1s,
        "combined_sigma_launch_clean_central": comb_central,
        "combined_sigma_launch_clean_worstcase": comb_worstcase,
        "mu_clears_500_clean_central": mu_central,       # <-- TEST
        "mu_clears_500_clean_worstcase": mu_worstcase,
        "lambda1_ceiling_mu": ceiling,
        "delta_mu_rebase_central": delta_central,        # signed; negative => LESS conservative
        "delta_mu_rebase_worstcase": delta_worstcase,
        "clean_central_vs_ceiling": "BELOW" if mu_central < ceiling else "ABOVE",
        "clean_worstcase_vs_ceiling": "BELOW" if mu_worstcase < ceiling else "ABOVE",
        "central_headroom_below_ceiling_tps": ceiling - mu_central,
        "worstcase_headroom_below_ceiling_tps": ceiling - mu_worstcase,
        "lambda1_clears_500_clean_central": bool(mu_central <= ceiling),
        "lambda1_clears_500_clean_worstcase": bool(mu_worstcase <= ceiling),
        "does_lambda1_clear_500_at_p95_centrally": "YES" if mu_central <= ceiling else "NO",
        "private_footing_sensitivity": {
            "mu_central_if_private_also_halfwidth": mu_central_pvalt,
            "shift_vs_primary_tps": mu_central_pvalt - mu_central,
            "note": "negligible; does not change the verdict",
        },
    }


# ---------------------------------------------------------------------------
# Step 4 -- direction reconciliation with #201's own scoping.
# ---------------------------------------------------------------------------
def direction_reconciliation(b: dict[str, Any], trig: dict[str, Any]) -> dict[str, Any]:
    # #201 scoped the fix as ~ sigma_hw*(z1-1) TPS MORE conservative.
    predicted_shift = b["sigma_hw"] * (Z95_ONE_SIDED - 1.0)   # +3.13687, "more conservative"
    actual_shift = trig["delta_mu_rebase_central"]             # -7.68231, LESS conservative
    sign_matches = bool((predicted_shift > 0) == (actual_shift > 0))
    return {
        "predicted_shift_tps": predicted_shift,
        "predicted_direction": "more_conservative_trigger_UP",
        "actual_shift_tps": actual_shift,
        "actual_direction": "less_conservative_trigger_DOWN" if actual_shift < 0 else "more_conservative_trigger_UP",
        "rebase_direction_matches_prediction": sign_matches,
        "mechanism_note": (
            "SIGN BACKWARDS. The sigma_hw*(z-1) heuristic assumed the fix bumps the SMALL hardware/private "
            "legs UP to the acceptance leg's footing (more conservative). But the mis-based leg is the "
            "DOMINANT acceptance leg (11.170, a z2 half-width), and the clean fix DIVIDES it DOWN by z2 to "
            "5.699 (1-sigma). Acceptance dominates the quadrature, so the combined sigma SHRINKS "
            "12.215 -> 7.545 and the trigger DROPS by 7.68 TPS. The heuristic had the wrong leg (hw, not "
            "acceptance), the wrong direction (up, not down), and ~2.4x the wrong magnitude."
        ),
    }


# ---------------------------------------------------------------------------
# Step 5 -- anchor errors (close the #201 logging-granularity gap).
# ---------------------------------------------------------------------------
def anchor_errors(b: dict[str, Any]) -> dict[str, Any]:
    # #195 de-dup iid (as-published, mixed basis): hypot(acc_iid_HW, sigma_hw, sigma_priv) -> 7.2617.
    repro_195 = _hypot([b["acc_iid_halfwidth"], b["sigma_hw"], b["sigma_private"]])
    err_195 = abs(repro_195 - b["combined_dedup_195"])
    # #194 break-even, reconstructed END-TO-END through the clean rebase machinery: take the raw #175
    # sampling HALF-WIDTH 10.906, rebase /z2 -> 5.5645 (1-sigma), hypot with sigma_hw (1-sigma) -> 7.391,
    # LCB map 500 + z1*7.391 -> 512.157. This proves the #194 anchor SURVIVES the rebase.
    sigma_sample_1s = b["sampling_halfwidth_175"] / Z95_TWO_SIDED
    sigma_draw_recon = _hypot([sigma_sample_1s, b["sigma_hw"]])
    mu_194_recon = TARGET + Z95_ONE_SIDED * sigma_draw_recon
    err_194 = abs(mu_194_recon - b["mu_break_even_194"])
    return {
        "anchor_err_195_dedup": err_195,
        "anchor_repro_195_dedup_tps": repro_195,
        "anchor_target_195_dedup_tps": b["combined_dedup_195"],
        "anchor_err_194_breakeven": err_194,
        "anchor_repro_194_breakeven_tps": mu_194_recon,
        "anchor_repro_194_sigma_draw_tps": sigma_draw_recon,
        "anchor_target_194_breakeven_tps": b["mu_break_even_194"],
        "anchor_194_note": "#194 uses the UN-de-duped sampling axis (10.906 HW -> 5.5645 1-sigma); the "
        "de-duped LAUNCH acceptance axis (5.31870 HW -> 2.71367 1-sigma) is a DIFFERENT, smaller quantity. "
        "The anchor validates that the clean z-convention + LCB map (mu=500+z1*sigma_1sigma) survive the "
        "rebase, which is exactly what #201's combined sigma violated on its acceptance leg.",
    }


# ---------------------------------------------------------------------------
# Step 6 -- self-test (PRIMARY).
# ---------------------------------------------------------------------------
def self_test(b: dict[str, Any], audit: dict[str, Any], trig: dict[str, Any],
              anchors: dict[str, Any]) -> dict[str, Any]:
    z = Z95_ONE_SIDED

    # (a) convention A and B give the same LCB on a consistently-based vector (central + a rho corner).
    sc0 = conventions_self_consistency([3.1, 2.2, 0.7], rho=0.0)
    sc1 = conventions_self_consistency([5.69911, 4.86447, 0.88392], rho=RHO_PLUS)
    a_ok = bool(sc0["identical"] and sc1["identical"])

    # (b) single acceptance leg, sigma_hw->0, sigma_priv->0: LCB = mu - z*sigma_acc (textbook one-axis).
    acc_1s = audit["acceptance"]["clean_1sigma"]
    single = _combined_sigma_rho(np.array([acc_1s, 0.0, 0.0]), 0.0)
    b_lcb_machine = TARGET - z * single
    b_lcb_textbook = TARGET - z * acc_1s
    b_ok = bool(abs(b_lcb_machine - b_lcb_textbook) < IDENT_TOL and abs(single - acc_1s) < IDENT_TOL)

    # (c) re-basing an ALREADY-all-1-sigma vector is a no-op (idempotent).
    already_1s = [acc_1s, b["sigma_hw"], b["sigma_private"]]
    rebased_again = rebase_legs_to_1sigma(already_1s, ["one_sigma", "one_sigma", "one_sigma"])
    c_ok = bool(all(abs(x - y) < IDENT_TOL for x, y in zip(already_1s, rebased_again)))
    # and: re-basing the ORIGINAL mixed vector twice == once (operator is a projection).
    once = rebase_legs_to_1sigma([b["acc_realistic_halfwidth"], b["sigma_hw"], b["sigma_private"]],
                                 ["halfwidth_95_two_sided", "one_sigma", "one_sigma"])
    twice = rebase_legs_to_1sigma(once, ["one_sigma", "one_sigma", "one_sigma"])
    c_ok = bool(c_ok and all(abs(x - y) < IDENT_TOL for x, y in zip(once, twice)))

    # (d) the #194 break-even (clean all-1-sigma anchor) is reconstructed through the rebase machinery.
    d_ok = bool(anchors["anchor_err_194_breakeven"] <= ANCHOR_TOL_TPS)

    # (e) worst-case >= central after rebase.
    e_ok = bool(trig["combined_sigma_launch_clean_worstcase"] >= trig["combined_sigma_launch_clean_central"] - 1e-12
                and trig["mu_clears_500_clean_worstcase"] >= trig["mu_clears_500_clean_central"] - 1e-12)

    # (f) NaN-clean across all reported scalars.
    scalars = [
        acc_1s, trig["combined_sigma_launch_clean_central"], trig["combined_sigma_launch_clean_worstcase"],
        trig["mu_clears_500_clean_central"], trig["mu_clears_500_clean_worstcase"],
        trig["delta_mu_rebase_central"], trig["delta_mu_rebase_worstcase"],
        sc0["convention_A_lcb"], sc0["convention_B_lcb"], sc1["convention_A_lcb"], sc1["convention_B_lcb"],
        b_lcb_machine, b_lcb_textbook, anchors["anchor_err_195_dedup"], anchors["anchor_err_194_breakeven"],
        anchors["anchor_repro_194_breakeven_tps"], trig["central_headroom_below_ceiling_tps"],
        trig["worstcase_headroom_below_ceiling_tps"],
    ]
    f_ok = all(_finite(x) for x in scalars)

    checks = {
        "a_convention_A_equals_B": a_ok,
        "b_single_leg_reduces_to_textbook_p95": b_ok,
        "c_rebase_idempotent_on_all_1sigma": c_ok,
        "d_194_break_even_survives_rebase": d_ok,
        "e_worstcase_ge_central": e_ok,
        "f_nan_clean": f_ok,
    }
    passes = all(checks.values())
    return {
        "unit_rebase_self_test_passes": bool(passes),   # <-- PRIMARY
        "checks": checks,
        "evidence": {
            "a_conv_AB_err_central": sc0["abs_err"],
            "a_conv_AB_err_rho_corner": sc1["abs_err"],
            "b_single_leg_lcb_machine_tps": b_lcb_machine,
            "b_single_leg_lcb_textbook_tps": b_lcb_textbook,
            "c_idempotent": c_ok,
            "d_anchor_err_194_tps": anchors["anchor_err_194_breakeven"],
            "anchor_err_195_dedup": anchors["anchor_err_195_dedup"],   # explicit per PR step 5
            "anchor_err_194_breakeven": anchors["anchor_err_194_breakeven"],  # explicit per PR step 5
            "n_scalars_checked": len(scalars),
        },
    }


def _build_result(b, audit, sc_demo, trig, direction, anchors, st) -> dict[str, Any]:
    handoff = (
        "fern #185: after clean-1-sigma rebase the launch GO trigger is mu >= %.2f central / %.2f "
        "worst-case (Delta %.2f vs #201's 520.09); lambda=1 (ceiling %.2f) DOES clear 500 at P95 "
        "centrally (headroom +%.2f) and even worst-case (+%.2f); fern should wire THIS clean trigger, "
        "and the only remaining lever to close the central<->worst-case gap is land #71's co-log "
        "(n=385) retiring rho(*,hw)."
        % (trig["mu_clears_500_clean_central"], trig["mu_clears_500_clean_worstcase"],
           trig["delta_mu_rebase_central"], trig["lambda1_ceiling_mu"],
           trig["central_headroom_below_ceiling_tps"], trig["worstcase_headroom_below_ceiling_tps"])
    )
    return {
        "pr": 204,
        "metric_primary": "unit_rebase_self_test_passes",
        "metric_test": "mu_clears_500_clean_central",
        "unit_rebase_self_test_passes": st["unit_rebase_self_test_passes"],
        "mu_clears_500_clean_central": trig["mu_clears_500_clean_central"],
        "mu_clears_500_clean_worstcase": trig["mu_clears_500_clean_worstcase"],
        "combined_sigma_launch_clean_central": trig["combined_sigma_launch_clean_central"],
        "combined_sigma_launch_clean_worstcase": trig["combined_sigma_launch_clean_worstcase"],
        "delta_mu_rebase_central": trig["delta_mu_rebase_central"],
        "does_lambda1_clear_500_at_p95_centrally": trig["does_lambda1_clear_500_at_p95_centrally"],
        "rebase_direction_matches_prediction": direction["rebase_direction_matches_prediction"],
        "anchor_err_195_dedup": anchors["anchor_err_195_dedup"],
        "anchor_err_194_breakeven": anchors["anchor_err_194_breakeven"],
        "law": "acc is a 95%-two-sided HALF-WIDTH (z2=1.95996); clean 1-sigma = acc/z2; "
        "sigma_launch_clean = hypot(acc/z2, sigma_hw, sigma_private)[+rho]; LCB(mu) = mu - z1*sigma_launch_clean; "
        "GO trigger = 500 + z1*sigma_launch_clean.",
        "imported_legs_201": {
            "acc_iid_halfwidth": b["acc_iid_halfwidth"],
            "acc_realistic_halfwidth": b["acc_realistic_halfwidth"],
            "sqrt_design_effect": b["sqrt_design_effect"],
            "sigma_hw": b["sigma_hw"],
            "sigma_private": b["sigma_private"],
            "combined_201_central": b["combined_201_central"],
            "mu_201_central": b["mu_201_central"],
            "mu_201_worstcase": b["mu_201_worstcase"],
            "lambda1_ceiling": b["lambda1_ceiling"],
            "mu_break_even_194": b["mu_break_even_194"],
            "z95_two_sided_194_confirms_footing": b["z95_two_sided_194"],
        },
        "leg_footing_audit": audit,
        "conventions_self_consistency_demo": sc_demo,
        "clean_trigger": trig,
        "direction_reconciliation": direction,
        "anchors": anchors,
        "self_test": st,
        "handoff": handoff,
        "scope": "Pure CPU-only unit re-basing over ubel's banked #201 launch-sigma curve: re-foots the "
        "mixed-basis combined launch sigma onto a single clean 1-sigma footing (#194 convention), re-solves "
        "the GO trigger, reports its signed direction vs the lambda=1 ceiling 520.95. Takes NO official "
        "draws, authorizes none. BASELINE stays 481.53; adds 0 TPS; greedy identity untouched. The "
        "rho(*,hw) [-0.3,+0.3] band still needs land #71's co-log (separate lever). NOT a launch.",
        "public_evidence_used": [
            "ubel #201 (spau6tch, MERGED) launch_sigma_closure: the combined launch sigma 12.2153 central / "
            "13.7956 worstcase, mu_clear 520.09/522.69, mixed half-width/1-sigma basis -- the curve re-based here.",
            "ubel #194 (redraw_budget): the EXPLICIT clean-1-sigma decomposition sigma_sample_1sigma = "
            "10.906 / z95_two_sided(1.95996) = 5.5645, break-even 512.157 = 500 + z1(1.64485)*7.391 -- the "
            "convention-A anchor proving the #175 sampling axis is a z=1.96 half-width.",
            "ubel #195 (3658ncbe) ci_axis_covariance: the #175 sampling axis source string "
            "'tps_finite_sample_ci_HALFWIDTH (both-bugs, B=16384, z=1.96)' and denken #187's overlap-corrected "
            "de-dup (h_in/h_out half-widths) -> the acceptance leg's half-width lineage.",
            "wirbel #190 (fva6o4ug) icc_neff: sqrt(design_effect)=2.100 is a DIMENSIONLESS halfwidth-ratio "
            "(22.905 HW / 10.906 HW) -> footing-preserving, so the realistic acceptance leg inherits #175's "
            "half-width footing.",
        ],
        "method": "LOCAL CPU-only analytic unit re-basing over EXISTING MERGED results; no GPU/vLLM/HF Job/"
        "submission/served-file change. BASELINE stays 481.53; adds 0 TPS. Greedy identity untouched. NOT a launch.",
        "convention_note": "Clean convention A (#194): every leg a 1-sigma; a 95%-two-sided CI half-width is "
        "z2*sigma so 1-sigma = HW/z2; LCB(mu) = mu - z1*sigma_combined (z1 one-sided P95). Convention B "
        "(all-half-width, #190) gives the IDENTICAL LCB when the same z links HW=z*sigma. #201 carried the "
        "acceptance leg on z2 but applied z1 -- a third, inconsistent basis.",
        "metrics_nan_clean": 1 if st["checks"]["f_nan_clean"] else 0,
    }


def _log_wandb(args, result: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[unit-rebase] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="launch-sigma-unit-rebase", agent="ubel",
            name=args.wandb_name or "ubel/launch-sigma-unit-rebase",
            group=args.wandb_group,
            tags=["launch-sigma", "unit-rebase", "footing", "covariance", "composition-pinning", "pr204"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic", "z_one_sided_p95": Z95_ONE_SIDED,
                    "z_two_sided_95": Z95_TWO_SIDED, "rho_plus_worstcase": RHO_PLUS},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[unit-rebase] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[unit-rebase] wandb disabled; skipping", flush=True)
        return
    try:
        trig = result["clean_trigger"]
        direction = result["direction_reconciliation"]
        audit = result["leg_footing_audit"]
        st = result["self_test"]
        flat = {
            "acceptance_realistic_halfwidth": result["imported_legs_201"]["acc_realistic_halfwidth"],
            "acceptance_1sigma_clean": trig["acceptance_1sigma_clean"],
            "combined_sigma_launch_clean_central": trig["combined_sigma_launch_clean_central"],
            "combined_sigma_launch_clean_worstcase": trig["combined_sigma_launch_clean_worstcase"],
            "combined_201_central_mixed": result["imported_legs_201"]["combined_201_central"],
            "mu_clears_500_clean_central": trig["mu_clears_500_clean_central"],
            "mu_clears_500_clean_worstcase": trig["mu_clears_500_clean_worstcase"],
            "mu_201_central_mixed": result["imported_legs_201"]["mu_201_central"],
            "lambda1_ceiling_mu": trig["lambda1_ceiling_mu"],
            "delta_mu_rebase_central": trig["delta_mu_rebase_central"],
            "delta_mu_rebase_worstcase": trig["delta_mu_rebase_worstcase"],
            "central_headroom_below_ceiling_tps": trig["central_headroom_below_ceiling_tps"],
            "worstcase_headroom_below_ceiling_tps": trig["worstcase_headroom_below_ceiling_tps"],
            "lambda1_clears_500_clean_central": 1.0 if trig["lambda1_clears_500_clean_central"] else 0.0,
            "lambda1_clears_500_clean_worstcase": 1.0 if trig["lambda1_clears_500_clean_worstcase"] else 0.0,
            "predicted_shift_tps": direction["predicted_shift_tps"],
            "actual_shift_tps": direction["actual_shift_tps"],
            "rebase_direction_matches_prediction": 1.0 if direction["rebase_direction_matches_prediction"] else 0.0,
            "acceptance_needs_rebase": 1.0 if audit["acceptance"]["needs_rebase"] else 0.0,
            "anchor_err_195_dedup": result["anchors"]["anchor_err_195_dedup"],
            "anchor_err_194_breakeven": result["anchors"]["anchor_err_194_breakeven"],
            # per-leg + per-check booleans (close #201 logging-granularity gap)
            "self_test_a_convention_A_equals_B": 1.0 if st["checks"]["a_convention_A_equals_B"] else 0.0,
            "self_test_b_single_leg_textbook": 1.0 if st["checks"]["b_single_leg_reduces_to_textbook_p95"] else 0.0,
            "self_test_c_idempotent": 1.0 if st["checks"]["c_rebase_idempotent_on_all_1sigma"] else 0.0,
            "self_test_d_194_anchor_survives": 1.0 if st["checks"]["d_194_break_even_survives_rebase"] else 0.0,
            "self_test_e_worstcase_ge_central": 1.0 if st["checks"]["e_worstcase_ge_central"] else 0.0,
            "self_test_f_nan_clean": 1.0 if st["checks"]["f_nan_clean"] else 0.0,
            "unit_rebase_self_test_passes": 1.0 if st["unit_rebase_self_test_passes"] else 0.0,
        }
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="launch_sigma_unit_rebase", artifact_type="launch-sigma-unit-rebase", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[unit-rebase] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    trig = result["clean_trigger"]
    direction = result["direction_reconciliation"]
    audit = result["leg_footing_audit"]
    st = result["self_test"]
    print("\n[unit-rebase] ===== LAUNCH-SIGMA UNIT REBASE (PR #204) =====", flush=True)
    print("  leg footing audit:", flush=True)
    for leg in ("acceptance", "hardware", "private"):
        a = audit[leg]
        print(f"    {leg:11s} value={a['value']:8.4f}  footing={a['footing']:24s} "
              f"z_applied={a['z_applied']:.5f}  needs_rebase={a['needs_rebase']}", flush=True)
    print("  conventions self-consistency (clean vector):", flush=True)
    sc = result["conventions_self_consistency_demo"]
    print(f"    convention_A_lcb={sc['convention_A_lcb']:.6f}  convention_B_lcb={sc['convention_B_lcb']:.6f}  "
          f"err={sc['abs_err']:.2e}  identical={sc['identical']}", flush=True)
    print("  CLEAN re-solved trigger:", flush=True)
    print(f"    acceptance 1-sigma (clean)          = {trig['acceptance_1sigma_clean']:8.4f} TPS  (= 11.170 / z2)", flush=True)
    print(f"    combined_sigma_launch_clean_central = {trig['combined_sigma_launch_clean_central']:8.4f} TPS  (was 12.2153 mixed)", flush=True)
    print(f"    combined_sigma_launch_clean_worstcase = {trig['combined_sigma_launch_clean_worstcase']:8.4f} TPS  (was 13.7956)", flush=True)
    print(f"    mu_clears_500_clean_central   = {trig['mu_clears_500_clean_central']:8.3f} TPS  <-- TEST  (was 520.092)", flush=True)
    print(f"    mu_clears_500_clean_worstcase = {trig['mu_clears_500_clean_worstcase']:8.3f} TPS  (was 522.692)", flush=True)
    print(f"    lambda=1 ceiling              = {trig['lambda1_ceiling_mu']:8.3f} TPS", flush=True)
    print(f"    delta_mu_rebase (central)     = {trig['delta_mu_rebase_central']:+8.3f} TPS  (negative = LESS conservative)", flush=True)
    print(f"    clean central vs ceiling      = {trig['clean_central_vs_ceiling']}  "
          f"(headroom +{trig['central_headroom_below_ceiling_tps']:.2f} TPS)", flush=True)
    print(f"    >>> DOES lambda=1 clear 500 at P95 CENTRALLY?  {trig['does_lambda1_clear_500_at_p95_centrally']}  "
          f"(and worst-case clears = {trig['lambda1_clears_500_clean_worstcase']})", flush=True)
    print("  direction reconciliation:", flush=True)
    print(f"    predicted {direction['predicted_shift_tps']:+.3f} ({direction['predicted_direction']}) | "
          f"actual {direction['actual_shift_tps']:+.3f} ({direction['actual_direction']}) | "
          f"matches={direction['rebase_direction_matches_prediction']}", flush=True)
    print("  anchors:", flush=True)
    print(f"    anchor_err_195_dedup     = {result['anchors']['anchor_err_195_dedup']:.2e}", flush=True)
    print(f"    anchor_err_194_breakeven = {result['anchors']['anchor_err_194_breakeven']:.2e}", flush=True)
    print(f"\n  SELF-TEST (PRIMARY) passes = {st['unit_rebase_self_test_passes']}  "
          f"mu_clears_500_clean_central (TEST) = {trig['mu_clears_500_clean_central']:.4f} TPS", flush=True)
    for k, v in st["checks"].items():
        print(f"    [{'ok' if v else 'XX'}] {k}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Launch-sigma unit rebase: clean-1-sigma footing -> re-solved GO trigger (PR #204)")
    ap.add_argument("--out", default=os.path.join(_HERE, "launch_sigma_unit_rebase_results.json"))
    ap.add_argument("--wandb-name", "--wandb_name", default="ubel/launch-sigma-unit-rebase")
    ap.add_argument("--wandb-group", "--wandb_group", default="launch-sigma-unit-rebase")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    b = import_banked()
    audit = leg_footing_audit(b)
    sc_demo = conventions_self_consistency([5.69911, 4.86447, 0.88392], rho=0.0)
    trig = resolve_clean_trigger(b, audit)
    direction = direction_reconciliation(b, trig)
    anchors = anchor_errors(b)
    st = self_test(b, audit, trig, anchors)

    result = _build_result(b, audit, sc_demo, trig, direction, anchors, st)
    result["elapsed_s"] = round(time.time() - t0, 4)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    _print(result)
    print(f"\n[unit-rebase] HANDOFF: {result['handoff']}", flush=True)
    print(f"[unit-rebase] artifacts -> {args.out}", flush=True)
    _log_wandb(args, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
