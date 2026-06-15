#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""EAGLE-3 honest-coverage -> E[T] -> M-tile coupling (PR #337, stark).

THE GOVERNING QUESTION (a second, independent failure mode under fern #325's compliant-500 envelope)
-----------------------------------------------------------------------------------------------------
stark #331 (b48rmwjq) proved fern #325's compliant-500 envelope is sub-cliff-safe AT E[T]=6.11, which
the deployed LINEAR spine reaches because its top-4 rank coverage is cov4=0.6532, giving per-depth
c_eff = a1 + (1-a1)*cov4 = 0.7731 + 0.2269*0.6532 = 0.9213 and (chain law) E[T]=6.11. But lawine #330
(hfrscdai) measured the FUSION head's honest top-4 effective acceptance at c_eff=0.8903 -- a shortfall
(its underlying rank coverage is only cov4_fusion = (0.8903-0.7731)/0.2269 = 0.5166, vs the 0.6532 the
linear spine carries). stark #331's own follow-up flagged the coupling: a lower fusion coverage forces
a higher tree width W to restore E[T], and higher W crosses the M=32 Marlin verify-GEMM tile cliff.

This card prices that coupling and asks a sharper question than the binary identity-bar miss
(0.8903 < 0.9213): does the honest fusion coverage collapse the E[T] LEVER -- and therefore fern
#325's envelope -- below 500 on the acceptance axis ALONE? Two operating points are available to the
honest fusion head:
  (A) STAY sub-cliff at W=4 (M=29, step x1) and accept the lower E[T]=E[T](0.8903)=5.52; or
  (B) WIDEN to restore E[T]=6.11, which needs cov -> 0.6532 hence W>=5 hence M>=36 > knee 32, so the
      verify step inflates by the measured cliff multiplier mu=1.16981 (stark #331's M=32->33 ratio).
The effective TPS lever is E[T]/step. (A) gives 5.52/1 = 5.52; (B) gives 6.11/1.16981 = 5.22. So the
+16.98% cliff penalty (B) EXCEEDS the +10.7% E[T] gain from restoring coverage: STAY-SUB-CLIFF WINS,
and the honest envelope is fern #325's banked corners scaled by E[T](0.8903)/E[T](0.9213) ~ 0.903.

THE RESULT
----------
honest_envelope_central ~ 470.3  and  honest_envelope_worst ~ 445.0 -- BOTH BELOW 500. The honest
fusion coverage collapses fern #325's compliant-500 envelope on the E[T] axis itself, a SECOND failure
mode independent of the binary identity/acceptance bar. The crossover cliff multiplier at which the two
operating points tie is mu_tie = E[T](0.9213)/E[T](0.8903) = 1.1076; the real cliff (1.16981) is well
past it, so widening can never recover the lost E[T] without a net loss.

HAND-OFF
--------
This output feeds fern #335's joint compliant-500 AND-gate demand axis, refining its binary
[c_eff >= 0.9213] coverage gate into a CONTINUOUS E[T]-consequence: a coverage shortfall does not just
fail a threshold, it scales the whole envelope by E[T](c_eff_honest)/E[T](0.9213).

LOCAL, CPU-ONLY, ANALYTIC. 0 GPU, no model forward, no training, no publish, no HF Job, no submission,
no served-file change, no official draw. BASELINE stays 481.53; adds 0 TPS -- it re-prices fern #325's
banked envelope under lawine #330's honest coverage. Imports verbatim: fern #325 (ledger constants:
lambda ceiling 520.95, banked central 586.08 / worst 492.87), stark #331 (M=W*K+1 tile map, knee 32,
mu=1.16981, chain law, a1=0.7731, cov4=0.6532, a1_required_611=0.9213), lawine #330 (honest fusion
c_eff=0.8903), fern #34 (native a1~0.7714/0.7731). Re-derives nothing measured. NOT a launch / build.

PRIMARY metric  cov_et_tile_self_test_passes
TEST    metric  honest_envelope_worst   (worst-case compliant TPS under honest fusion coverage)
REPORT          honest_envelope_clears_500  (worst corner; the binding compliant-500 test)
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Acceptance topology + chain law (stark #331 / denken #304 / wirbel #79) — imported verbatim.
# --------------------------------------------------------------------------- #
K_SPEC = 7                              # deployed speculative depth (ubel #311); chain-law K
M_VERIFY_DEPLOYED = 8                   # deployed linear-chain verify width (ubel #311): M=1*7+1
A1_REQUIRED_611 = 0.9213011665456927   # denken #304 uniform per-depth c_eff reaching E[T]=6.11 (the bar)
RAW_A1_DEMAND = 0.7730729805683441     # denken #320 rank-1 acceptance a1=0.7731 (salvage anchor)
COV4 = 0.6531976066516435              # wirbel #79 top-4 rank coverage of the deployed LINEAR spine
NATIVE_A1 = 0.7714                     # fern #34 / denken #320 measured native rank-1 acceptance
SIZE29_TREE_CORPUS = 29                # lawine #101 size-29 tree node count (W=4 corpus anchor)
E_T_BUILD_FREE = 6.1112149873699195    # free build target E[T]=6.11 (wirbel #295); == E[T](0.9213)

# --------------------------------------------------------------------------- #
# lawine #330 (hfrscdai) — honest FUSION head measured top-4 effective acceptance.
# 0.8903 is the per-depth c_eff the fusion draft actually delivers (not the linear 0.9213).
# --------------------------------------------------------------------------- #
HONEST_FUSION_C_EFF = 0.8903           # lawine #330 measured fusion top-4 c_eff (the shortfall)

# --------------------------------------------------------------------------- #
# Marlin int4 W4A16 verify-GEMM tile cliff (directly measured; A10G, ctx=256; stark #331).
# --------------------------------------------------------------------------- #
TILE_BLOCK = 16                        # Marlin thread_m_blocks = ceil(M / 16)
KNEE_MSTAR = 32                        # last M before the 2->3 block cliff (cliff at M=33)
T_STEP_M32 = 12.811936378479004        # measured step @ M=32 (ms)
T_STEP_M33 = 14.98748779296875         # measured step @ M=33 (ms); 2->3 block cliff
T_STEP_M48 = 15.265167713165283        # measured step @ M=48 (ms)
T_STEP_M49 = 18.134016036987305        # measured step @ M=49 (ms); 3->4 block second cliff

# --------------------------------------------------------------------------- #
# fern #325 compliant-500 banked envelope (imported verbatim; both corners at E[T]=6.11).
# central is CAP-BOUND (= lambda ceiling); worst is the uncapped private-tax corner.
# --------------------------------------------------------------------------- #
LAMBDA_CEIL = 520.9527323111674        # int4-spec batch-invariant verify ceiling (wirbel #216/#227/#235)
PRIV_TPS_CENTRAL_318 = 586.0766391463308   # fern #318 banked central (pre-cap)
PRIV_TPS_WORST_318 = 492.865273281899      # fern #318 banked worst (uncapped)
FERN325_CENTRAL = min(PRIV_TPS_CENTRAL_318, LAMBDA_CEIL)   # 520.95 (cap-bound)
FERN325_WORST = min(PRIV_TPS_WORST_318, LAMBDA_CEIL)       # 492.87 (uncapped)
TARGET = 500.0
BASELINE_TPS = 481.53

TOL_RT = 1e-9
TOL_REPRO = 2e-2          # E[T](0.9213) vs banked 6.11 (rounding of the bar)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Core laws (stark #331 conventions).
# --------------------------------------------------------------------------- #
def e_t(c_eff: float, k: int = K_SPEC) -> float:
    """Chain-law expected accepted tokens: E[T] = 1 + sum_{d=1..K} c_eff^d (stark #331)."""
    return 1.0 + sum(c_eff ** d for d in range(1, k + 1))


def thread_m_blocks(m: int) -> int:
    """Marlin int4 W4A16 verify-GEMM tile count = ceil(M / 16)."""
    return -(-int(m) // TILE_BLOCK)


def tree_m(width: int, depth: int = K_SPEC) -> int:
    """Accepted-tree node count presented to one verify pass: M = W*K + 1 (stark #331)."""
    return width * depth + 1


def c_eff_from_cov(cov: float, a1: float = RAW_A1_DEMAND) -> float:
    """Salvage relation (denken #320): c_eff = a1 + (1-a1)*cov."""
    return a1 + (1.0 - a1) * cov


def cov_from_c_eff(c_eff: float, a1: float = RAW_A1_DEMAND) -> float:
    """Invert the salvage relation: cov = (c_eff - a1) / (1 - a1)."""
    return (c_eff - a1) / (1.0 - a1)


def cliff_mu() -> float:
    """Measured 2->3 block step ratio at the M=33 cliff (verify-GEMM denominator multiplier)."""
    return T_STEP_M33 / T_STEP_M32


# --------------------------------------------------------------------------- #
# (D1) E[T](c_eff) curve — linear 0.9213 vs honest fusion 0.8903.
# --------------------------------------------------------------------------- #
def deliverable1_et_curve() -> dict[str, Any]:
    et_linear = e_t(A1_REQUIRED_611)            # should reproduce 6.11 (== E_T_BUILD_FREE)
    et_fusion = e_t(HONEST_FUSION_C_EFF)        # honest fusion E[T] at W=4 sub-cliff
    et_native = e_t(NATIVE_A1)                  # native rank-1-only chain (context)
    # monotonicity probe over a fine c_eff grid.
    grid = [i / 100.0 for i in range(0, 101)]
    ets = [e_t(c) for c in grid]
    monotone = all(ets[i + 1] >= ets[i] - TOL_RT for i in range(len(ets) - 1))
    return {
        "chain_law": "E[T] = 1 + sum_{d=1..K} c_eff^d   (K=7, stark #331 convention)",
        "K": K_SPEC,
        "c_eff_linear": A1_REQUIRED_611,
        "c_eff_fusion_honest": HONEST_FUSION_C_EFF,
        "et_linear_0p9213": et_linear,
        "et_fusion_0p8903": et_fusion,
        "et_native_a1_only": et_native,
        "et_linear_reproduces_611": bool(abs(et_linear - E_T_BUILD_FREE) <= TOL_REPRO),
        "et_drop_from_shortfall": et_linear - et_fusion,
        "et_drop_pct": 100.0 * (et_linear - et_fusion) / et_linear,
        "monotone_increasing_in_c_eff": bool(monotone),
        "note": ("the honest fusion c_eff=0.8903 yields E[T]={:.4f}, a {:.2f}% drop from the linear "
                 "E[T]={:.4f} at c_eff=0.9213. E[T] is strictly increasing in c_eff, so coverage "
                 "shortfall maps monotonically to a lower E[T] lever.".format(
                     et_fusion, 100.0 * (et_linear - et_fusion) / et_linear, et_linear)),
    }


# --------------------------------------------------------------------------- #
# (D2) Coverage -> width -> tile map: what W restores c_eff=0.9213, and does it cross M=32?
# --------------------------------------------------------------------------- #
def deliverable2_cov_width_tile() -> dict[str, Any]:
    a1 = RAW_A1_DEMAND
    cov_needed = cov_from_c_eff(A1_REQUIRED_611, a1)       # == COV4 by construction (linear coverage)
    cov4_fusion = cov_from_c_eff(HONEST_FUSION_C_EFF, a1)  # ~0.5166 (PR seed 0.5165 used rounded a1)
    cov_shortfall = cov_needed - cov4_fusion

    # tile-map anchors (stark #331): M = W*K + 1.
    m_w1 = tree_m(1)        # deployed: 8
    m_w4 = tree_m(4)        # honest sub-cliff: 29
    m_w5 = tree_m(5)        # smallest widening past W=4: 36 (> knee 32)

    # MODEL-FREE LOWER BOUND on the restoring width:
    # coverage is monotone increasing in W; cov4_fusion < cov_needed; so restoring c_eff=0.9213
    # requires strictly more candidates than W=4, i.e. W >= 5 (integer). W=5 -> M=36 > 32 -> crosses.
    w_min_restore = 5
    m_min_restore = tree_m(w_min_restore)
    crosses_min = bool(m_min_restore > KNEE_MSTAR)

    # LABELED SENSITIVITY (not load-bearing): geometric saturation of residual coverage toward 1.0,
    # anchored at the single fusion seed with the cov(W=1)=0 convention (width-1 = rank-1, no salvage):
    #   cov_W = 1 - rho^(W-1),  rho fixed by cov4_fusion = 1 - rho^3.
    rho = (1.0 - cov4_fusion) ** (1.0 / (4 - 1))
    # solve 1 - rho^(W-1) = cov_needed  ->  W = 1 + ln(1-cov_needed)/ln(rho)
    w_geom_cont = 1.0 + math.log(1.0 - cov_needed) / math.log(rho)
    w_geom_ceil = math.ceil(w_geom_cont - TOL_RT)
    m_geom = tree_m(w_geom_ceil)
    crosses_geom = bool(m_geom > KNEE_MSTAR)

    return {
        "salvage_relation": "c_eff(W) = a1 + (1-a1)*cov_W   (a1=0.7731, denken #320)",
        "a1": a1,
        "cov_needed_for_0p9213": cov_needed,
        "cov_needed_equals_linear_cov4": bool(abs(cov_needed - COV4) <= TOL_RT),
        "cov4_fusion_seed": cov4_fusion,
        "cov4_fusion_seed_pr_rounded": 0.5165,
        "cov_shortfall": cov_shortfall,
        "tile_map": {
            "formula": "M = W*K + 1 ; thread_m_blocks = ceil(M/16) ; knee_Mstar=32 (cliff @ M=33)",
            "M_W1_deployed": m_w1, "M_W4_subcliff": m_w4, "M_W5_widen": m_w5,
            "blocks_W4": thread_m_blocks(m_w4), "blocks_W5": thread_m_blocks(m_w5),
            "deployed_anchor_ok": bool(m_w1 == M_VERIFY_DEPLOYED),
            "subcliff_anchor_ok": bool(m_w4 == SIZE29_TREE_CORPUS and m_w4 <= KNEE_MSTAR),
        },
        "restore_lower_bound": {
            "w_min_restore": w_min_restore,
            "m_min_restore": m_min_restore,
            "crosses_knee_32": crosses_min,
            "argument": ("coverage is monotone in W and cov4_fusion=%.4f < cov_needed=%.4f, so "
                         "restoring c_eff=0.9213 needs W>=5; the smallest such tree (W=5) already "
                         "presents M=36 > knee 32 -> the M=33 cliff is unavoidable for ANY "
                         "restoration." % (cov4_fusion, cov_needed)),
        },
        "restore_geom_sensitivity": {
            "model": "cov_W = 1 - rho^(W-1), rho from cov4_fusion (cov(W=1)=0 convention)",
            "rho": rho,
            "w_continuous": w_geom_cont,
            "w_ceil": w_geom_ceil,
            "m_at_w_ceil": m_geom,
            "crosses_knee_32": crosses_geom,
            "note": ("point-estimate only; not load-bearing. Even this saturating model needs "
                     "W~%.2f (M=%d) > knee 32, so the lower bound (W>=5) and the estimate agree: "
                     "restoration crosses the cliff. A larger W only worsens the supra-cliff case "
                     "(W=7 -> M=50 would also cross the M=49 second cliff)." % (w_geom_cont, m_geom)),
        },
        "verdict_restoration_crosses_cliff": bool(crosses_min and crosses_geom),
    }


# --------------------------------------------------------------------------- #
# (D3) Sub-cliff vs supra-cliff decision: which operating point yields higher compliant-TPS?
# --------------------------------------------------------------------------- #
def deliverable3_ab_decision(d1: dict) -> dict[str, Any]:
    mu = cliff_mu()
    et_a = d1["et_fusion_0p8903"]      # (A) STAY sub-cliff: lower E[T], step x1
    et_b = d1["et_linear_0p9213"]      # (B) WIDEN supra-cliff: E[T] restored to 6.11, step /mu

    lever_a = et_a / 1.0               # effective TPS lever = E[T] / step
    lever_b = et_b / mu
    a_wins = bool(lever_a > lever_b)

    # crossover cliff multiplier at which (A) and (B) tie: et_a = et_b / mu_tie.
    mu_tie = et_b / et_a

    # realized compliant-TPS at each operating point (scale / divide fern #325's banked corners):
    scale_a = et_a / et_b              # (A) lower-E[T] lever on the banked (E[T]=6.11) corners
    central_a = FERN325_CENTRAL * scale_a
    worst_a = FERN325_WORST * scale_a
    central_b = FERN325_CENTRAL / mu   # (B) E[T] restored, step inflated by the cliff
    worst_b = FERN325_WORST / mu

    return {
        "mu_cliff": mu,
        "operating_point_A_subcliff": {
            "desc": "W=4, M=29, E[T]=E[T](0.8903), step x1 (sub-cliff)",
            "E_T": et_a, "step_mult": 1.0, "effective_lever": lever_a,
            "compliant_central": central_a, "compliant_worst": worst_a,
        },
        "operating_point_B_supracliff": {
            "desc": "W>=5, M>=36, E[T] restored to 6.11, step /mu (supra-cliff)",
            "E_T": et_b, "step_mult": mu, "effective_lever": lever_b,
            "compliant_central": central_b, "compliant_worst": worst_b,
        },
        "A_wins": a_wins,
        "winner": "A_subcliff" if a_wins else "B_supracliff",
        "lever_margin_A_minus_B": lever_a - lever_b,
        "lever_ratio_A_over_B": lever_a / lever_b,
        "mu_tie_crossover": mu_tie,
        "real_mu_exceeds_tie": bool(mu > mu_tie),
        "note": ("(A) sub-cliff lever {:.4f} vs (B) supra-cliff lever {:.4f}: the +{:.2f}% cliff "
                 "penalty exceeds the +{:.2f}% E[T] gain, so STAY-SUB-CLIFF wins. They would tie at "
                 "mu_tie={:.4f}; the real cliff mu={:.5f} is past it. Compliant worst: A={:.2f} > "
                 "B={:.2f}.".format(lever_a, lever_b, 100.0 * (mu - 1.0),
                                    100.0 * (et_b / et_a - 1.0), mu_tie, mu, worst_a, worst_b)),
    }


# --------------------------------------------------------------------------- #
# (D4) Honest envelope re-price: fern #325 corners x (E[T]_honest / E[T]_611).
# --------------------------------------------------------------------------- #
def deliverable4_honest_envelope(d1: dict, d3: dict) -> dict[str, Any]:
    et_honest = d1["et_fusion_0p8903"]      # the WINNING sub-cliff operating point's E[T]
    et_611 = d1["et_linear_0p9213"]         # fern #325's envelope reference E[T]=6.11
    scale = et_honest / et_611

    honest_central = FERN325_CENTRAL * scale
    honest_worst = FERN325_WORST * scale
    clears_central = bool(honest_central >= TARGET - TOL_RT)
    clears_worst = bool(honest_worst >= TARGET - TOL_RT)
    return {
        "fern325_central_at_611": FERN325_CENTRAL,
        "fern325_worst_at_611": FERN325_WORST,
        "et_611_reference": et_611,
        "et_honest_subcliff": et_honest,
        "scale_et_honest_over_611": scale,
        "honest_envelope_central": honest_central,
        "honest_envelope_worst": honest_worst,          # TEST metric
        "honest_envelope_clears_500_central": clears_central,
        "honest_envelope_clears_500_worst": clears_worst,
        "honest_envelope_clears_500": clears_worst,      # worst corner = binding compliant-500 test
        "both_corners_below_500": bool(not clears_central and not clears_worst),
        "central_headroom_pct": 100.0 * (honest_central - TARGET) / TARGET,
        "worst_headroom_pct": 100.0 * (honest_worst - TARGET) / TARGET,
        "cap_subtlety_note": (
            "deliverable scales the BANKED corners (central is cap-bound at the lambda ceiling) by "
            "the E[T] lever ratio, per the PR's lever convention. A strict re-price holding the "
            "lambda ceiling FIXED would leave central cap-bound near {:.1f} (still >500), so the "
            "WORST corner ({:.1f}) is the binding compliant-500 test either way -- and it is "
            "below 500.".format(LAMBDA_CEIL, honest_worst)),
    }


# --------------------------------------------------------------------------- #
# (D5) Verdict + hand-off to fern #335.
# --------------------------------------------------------------------------- #
def deliverable5_verdict(d2: dict, d3: dict, d4: dict) -> dict[str, Any]:
    collapses = bool(d4["both_corners_below_500"] or not d4["honest_envelope_clears_500"])
    verdict = "ENVELOPE-COLLAPSES-ON-E[T]-AXIS" if collapses else "ENVELOPE-HOLDS"
    why = ("the honest fusion c_eff=0.8903 (lawine #330) drops E[T] to {:.4f}; restoring E[T]=6.11 "
           "needs W>=5 (M>=36 > knee 32) so the cliff (mu={:.5f}) inflates the step, and the +16.98% "
           "cliff penalty exceeds the +{:.2f}% E[T] gain -> STAY-SUB-CLIFF wins at E[T]={:.4f}. "
           "Scaling fern #325's banked envelope by {:.4f} gives central {:.2f} / worst {:.2f} -- "
           "BOTH below 500.".format(
               d3["operating_point_A_subcliff"]["E_T"], d3["mu_cliff"],
               100.0 * (d3["operating_point_B_supracliff"]["E_T"]
                        / d3["operating_point_A_subcliff"]["E_T"] - 1.0),
               d4["et_honest_subcliff"], d4["scale_et_honest_over_611"],
               d4["honest_envelope_central"], d4["honest_envelope_worst"]))
    return {
        "verdict": verdict,
        "honest_coverage_collapses_envelope": collapses,
        "why": why,
        "second_failure_mode_note": (
            "this is a SECOND, independent failure mode: the binary identity/acceptance bar already "
            "fails (0.8903 < 0.9213), but here the SAME shortfall ALSO collapses the compliant-500 "
            "envelope on the E[T] LEVER axis -- the demand miss does not need the identity bar to "
            "kill compliant-500."),
        "handoff_to_fern_335": (
            "feeds fern #335's joint compliant-500 AND-gate DEMAND axis: refine its binary "
            "[c_eff >= 0.9213] coverage gate into the CONTINUOUS consequence honest_envelope = "
            "fern_envelope x E[T](c_eff_honest)/E[T](0.9213). At c_eff=0.8903 the demand axis is "
            "RED (worst {:.2f} < 500) before any private-tax/identity term is applied.".format(
                d4["honest_envelope_worst"])),
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def _selftests(d1: dict, d2: dict, d3: dict, d4: dict, vd: dict) -> dict[str, Any]:
    mu = d3["mu_cliff"]
    conditions = {
        # (1) chain law reproduces the banked E[T]=6.11 at c_eff=0.9213 (stark #331).
        "01_et_0p9213_reproduces_611": bool(d1["et_linear_reproduces_611"]
                                            and abs(d1["et_linear_0p9213"] - E_T_BUILD_FREE) <= TOL_REPRO),
        # (2) honest fusion E[T](0.8903) is ~5.52 and strictly below the linear E[T].
        "02_et_0p8903_is_5p52": bool(abs(d1["et_fusion_0p8903"] - 5.5176) <= 1e-3
                                     and d1["et_fusion_0p8903"] < d1["et_linear_0p9213"]),
        # (3) E[T] is monotone increasing in c_eff.
        "03_et_monotone_in_c_eff": bool(d1["monotone_increasing_in_c_eff"]),
        # (4) chain law spot-check: E[T](c) recomputed independently matches.
        "04_chain_law_spotcheck": bool(
            abs(e_t(0.9) - (1.0 + sum(0.9 ** d for d in range(1, K_SPEC + 1)))) <= TOL_RT),
        # (5) M = W*K+1 deployed anchor: W=1 -> M=8.
        "05_M_anchor_deployed_8": bool(tree_m(1) == M_VERIFY_DEPLOYED
                                       and d2["tile_map"]["deployed_anchor_ok"]),
        # (6) M = W*K+1 sub-cliff anchor: W=4 -> M=29 (== size-29 corpus), 2 blocks, sub-cliff.
        "06_M_anchor_subcliff_29": bool(tree_m(4) == SIZE29_TREE_CORPUS
                                        and thread_m_blocks(29) == 2 and 29 <= KNEE_MSTAR
                                        and d2["tile_map"]["subcliff_anchor_ok"]),
        # (7) cov needed to restore 0.9213 equals the linear spine's cov4 (self-consistency).
        "07_cov_needed_equals_cov4": bool(d2["cov_needed_equals_linear_cov4"]),
        # (8) salvage relation round-trips: c_eff(cov4_fusion) == 0.8903.
        "08_salvage_roundtrip": bool(
            abs(c_eff_from_cov(d2["cov4_fusion_seed"]) - HONEST_FUSION_C_EFF) <= TOL_RT),
        # (9) restoration crosses the cliff: W>=5 -> M>=36 > 32 (model-free lower bound).
        "09_restore_crosses_cliff_lower_bound": bool(
            d2["restore_lower_bound"]["crosses_knee_32"]
            and d2["restore_lower_bound"]["m_min_restore"] == 36),
        # (10) geometric sensitivity agrees: estimate also > knee 32.
        "10_restore_crosses_cliff_geom": bool(d2["restore_geom_sensitivity"]["crosses_knee_32"]
                                              and d2["verdict_restoration_crosses_cliff"]),
        # (11) cliff mu comes from the measured tile data and is > 1.
        "11_mu_from_measured_tile": bool(abs(mu - (T_STEP_M33 / T_STEP_M32)) <= TOL_RT and mu > 1.0),
        # (12) A/B decision internally consistent: A wins, lever_A > lever_B, A_worst > B_worst.
        "12_ab_decision_consistent": bool(
            d3["A_wins"] and d3["winner"] == "A_subcliff"
            and d3["operating_point_A_subcliff"]["effective_lever"]
            > d3["operating_point_B_supracliff"]["effective_lever"]
            and d3["operating_point_A_subcliff"]["compliant_worst"]
            > d3["operating_point_B_supracliff"]["compliant_worst"]),
        # (13) crossover mu recovered: mu_tie = E[T](0.9213)/E[T](0.8903), and real mu exceeds it.
        "13_mu_tie_recovered": bool(
            abs(d3["mu_tie_crossover"] - d1["et_linear_0p9213"] / d1["et_fusion_0p8903"]) <= TOL_RT
            and d3["real_mu_exceeds_tie"] and not (mu <= d3["mu_tie_crossover"])),
        # (14) supra-cliff corners reproduce stark #331's "if crossed" numbers (445.33 / 421.32).
        "14_supracliff_matches_331": bool(
            abs(d3["operating_point_B_supracliff"]["compliant_central"] - LAMBDA_CEIL / mu) <= TOL_RT
            and abs(d3["operating_point_B_supracliff"]["compliant_worst"] - FERN325_WORST / mu) <= TOL_RT),
        # (15) honest envelope == fern corners x (E[T]_honest/E[T]_611).
        "15_honest_envelope_scaling": bool(
            abs(d4["honest_envelope_central"]
                - FERN325_CENTRAL * (d1["et_fusion_0p8903"] / d1["et_linear_0p9213"])) <= TOL_RT
            and abs(d4["honest_envelope_worst"]
                    - FERN325_WORST * (d1["et_fusion_0p8903"] / d1["et_linear_0p9213"])) <= TOL_RT),
        # (16) both clears_500 booleans correct: both corners below 500.
        "16_both_clears_500_false": bool(
            (not d4["honest_envelope_clears_500_central"])
            and (not d4["honest_envelope_clears_500_worst"])
            and d4["both_corners_below_500"]
            and d4["honest_envelope_central"] < TARGET and d4["honest_envelope_worst"] < TARGET),
        # (17) verdict is the collapse verdict and flags the second failure mode.
        "17_verdict_collapse": bool(vd["verdict"] == "ENVELOPE-COLLAPSES-ON-E[T]-AXIS"
                                    and vd["honest_coverage_collapses_envelope"]),
        # (18) NaN-clean (set by caller).
        "18_nan_clean": True,
    }
    return {
        "conditions": conditions,
        "cov_et_tile_self_test_passes": bool(all(conditions.values())),
        "n_checks": len(conditions),
        "detail": {
            "et_linear": d1["et_linear_0p9213"], "et_fusion": d1["et_fusion_0p8903"],
            "mu": mu, "mu_tie": d3["mu_tie_crossover"],
            "honest_worst": d4["honest_envelope_worst"],
        },
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    d1 = deliverable1_et_curve()
    d2 = deliverable2_cov_width_tile()
    d3 = deliverable3_ab_decision(d1)
    d4 = deliverable4_honest_envelope(d1, d3)
    vd = deliverable5_verdict(d2, d3, d4)
    st = _selftests(d1, d2, d3, d4, vd)

    headline = {
        "cov_et_tile_self_test_passes": bool(st["cov_et_tile_self_test_passes"]),   # PRIMARY
        "honest_envelope_worst": d4["honest_envelope_worst"],                         # TEST
        "honest_envelope_central": d4["honest_envelope_central"],
        "honest_envelope_clears_500": d4["honest_envelope_clears_500"],               # REPORT
        "et_linear_0p9213": d1["et_linear_0p9213"],
        "et_fusion_0p8903": d1["et_fusion_0p8903"],
        "mu_cliff": d3["mu_cliff"],
        "mu_tie_crossover": d3["mu_tie_crossover"],
        "winner": d3["winner"],
        "restoration_crosses_cliff": d2["verdict_restoration_crosses_cliff"],
        "verdict": vd["verdict"],
        "honest_coverage_collapses_envelope": vd["honest_coverage_collapses_envelope"],
    }
    return {
        "headline": headline,
        "deliverable1_et_curve": d1,
        "deliverable2_cov_width_tile": d2,
        "deliverable3_ab_decision": d3,
        "deliverable4_honest_envelope": d4,
        "deliverable5_verdict": vd,
        "self_test": st,
        "imports": {
            "provenance": (
                "fern #325 (banked compliant-500 envelope: lambda ceiling 520.95, central 586.08, "
                "worst 492.87, both @E[T]=6.11) x stark #331 b48rmwjq (M=W*K+1 tile map, knee 32, "
                "mu=1.16981 from measured M=32->33 step, chain law E[T]=1+sum c^d, a1=0.7731, "
                "cov4=0.6532, a1_required_611=0.9213) x lawine #330 hfrscdai (honest fusion top-4 "
                "c_eff=0.8903) x fern #34 gua9x68j (native a1~0.7714/0.7731). All run-ids in "
                "wandb-applied-ai-team/gemma-challenge-senpai."),
            "caveats": [
                "DERIVED, not measured: no EAGLE-3 fusion checkpoint runs here; this prices the E[T] "
                "and tile consequences the measured coverages (linear 0.6532, fusion 0.8903 c_eff) "
                "imply under stark #331's tile map. NOT a running EagleProposer.",
                "the exact restoring width W is model-dependent; the verdict rests only on the "
                "model-free lower bound W>=5 (M>=36 > knee 32), which any monotone coverage curve "
                "satisfies. The geometric estimate (W~5.4, M~43) is a labeled sensitivity that only "
                "strengthens the supra-cliff loss.",
                "deliverable 4 scales the banked corners by the E[T] lever ratio per the PR. The "
                "central corner is cap-bound at the lambda ceiling; a strict cap-fixed re-price "
                "leaves central >500 but the WORST corner is below 500 either way (the binding test).",
                "0 TPS / re-pricing property: depends only on the chain law, integer node counts, the "
                "measured tile boundary, and fern #325's banked corners -- not tensor values. NOT a "
                "launch / build / served-file change / HF Job / submission.",
            ],
        },
    }


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B.
# --------------------------------------------------------------------------- #
def _nan_paths(node: Any, p: str = "result") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            bad += _nan_paths(v, f"{p}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            bad += _nan_paths(v, f"{p}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        bad.append(p)
    return bad


def _print_report(syn: dict) -> None:
    h = syn["headline"]
    d1 = syn["deliverable1_et_curve"]
    d2 = syn["deliverable2_cov_width_tile"]
    d3 = syn["deliverable3_ab_decision"]
    d4 = syn["deliverable4_honest_envelope"]
    vd, st = syn["deliverable5_verdict"], syn["self_test"]
    A, B = d3["operating_point_A_subcliff"], d3["operating_point_B_supracliff"]
    print("\n" + "=" * 98, flush=True)
    print("EAGLE-3 COV -> E[T] -> M-TILE COUPLING (PR #337, stark) — honest fusion 0.8903", flush=True)
    print("=" * 98, flush=True)
    print("  (D1) E[T](c_eff) = 1 + sum_{d=1..7} c_eff^d", flush=True)
    print(f"      linear  c_eff=0.9213 -> E[T]={d1['et_linear_0p9213']:.4f}  (reproduces 6.11: "
          f"{d1['et_linear_reproduces_611']})", flush=True)
    print(f"      fusion  c_eff=0.8903 -> E[T]={d1['et_fusion_0p8903']:.4f}  (-{d1['et_drop_pct']:.2f}% "
          f"lever)", flush=True)
    print("-" * 98, flush=True)
    print("  (D2) COVERAGE -> WIDTH -> TILE", flush=True)
    print(f"      cov needed for 0.9213 = {d2['cov_needed_for_0p9213']:.4f} (== linear cov4: "
          f"{d2['cov_needed_equals_linear_cov4']}) ; fusion cov4 = {d2['cov4_fusion_seed']:.4f}", flush=True)
    lb = d2["restore_lower_bound"]
    print(f"      restore needs W>={lb['w_min_restore']} -> M={lb['m_min_restore']} > knee 32 -> "
          f"crosses={lb['crosses_knee_32']}  (geom est W~{d2['restore_geom_sensitivity']['w_continuous']:.2f}, "
          f"M={d2['restore_geom_sensitivity']['m_at_w_ceil']})", flush=True)
    print("-" * 98, flush=True)
    print("  (D3) SUB-CLIFF (A) vs SUPRA-CLIFF (B) DECISION", flush=True)
    print(f"      (A) E[T]={A['E_T']:.4f} step x1     lever={A['effective_lever']:.4f}  "
          f"worst={A['compliant_worst']:.2f}", flush=True)
    print(f"      (B) E[T]={B['E_T']:.4f} step /{d3['mu_cliff']:.4f} lever={B['effective_lever']:.4f}  "
          f"worst={B['compliant_worst']:.2f}", flush=True)
    print(f"      winner={d3['winner']}  mu_tie={d3['mu_tie_crossover']:.4f} (real mu="
          f"{d3['mu_cliff']:.5f} > tie: {d3['real_mu_exceeds_tie']})", flush=True)
    print("-" * 98, flush=True)
    print("  (D4) HONEST ENVELOPE = fern #325 banked x (E[T]_honest / E[T]_611)", flush=True)
    print(f"      scale={d4['scale_et_honest_over_611']:.4f} -> central={d4['honest_envelope_central']:.2f} "
          f"({d4['central_headroom_pct']:+.2f}%)  worst={d4['honest_envelope_worst']:.2f} "
          f"({d4['worst_headroom_pct']:+.2f}%)", flush=True)
    print(f"      clears_500: central={d4['honest_envelope_clears_500_central']} "
          f"worst={d4['honest_envelope_clears_500_worst']}  (both below 500: "
          f"{d4['both_corners_below_500']})", flush=True)
    print("-" * 98, flush=True)
    print(f"  (D5) VERDICT: {vd['verdict']}", flush=True)
    print(f"      {vd['why']}", flush=True)
    print("-" * 98, flush=True)
    print(f"  PRIMARY cov_et_tile_self_test_passes = {st['cov_et_tile_self_test_passes']} "
          f"({st['n_checks']} checks)", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print("=" * 98 + "\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[eagle3-cov-et-tile] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    h = syn["headline"]
    d1, d2 = syn["deliverable1_et_curve"], syn["deliverable2_cov_width_tile"]
    d3, d4 = syn["deliverable3_ab_decision"], syn["deliverable4_honest_envelope"]
    vd, st = syn["deliverable5_verdict"], syn["self_test"]
    run = init_wandb_run(
        job_type="eagle3-cov-et-tile-coupling",
        agent="stark",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["eagle3-cov-et-tile-coupling", "issue-192", "eagle3", "marlin-tile-cliff",
              "honest-coverage", "compliant-500", "validity-gate", "bank-the-analysis"],
        config={
            "K_spec": K_SPEC, "a1": RAW_A1_DEMAND, "cov4_linear": COV4,
            "a1_required_611": A1_REQUIRED_611, "honest_fusion_c_eff": HONEST_FUSION_C_EFF,
            "knee_Mstar": KNEE_MSTAR, "lambda_ceiling": LAMBDA_CEIL,
            "fern325_central": FERN325_CENTRAL, "fern325_worst": FERN325_WORST,
            "e_t_build_free": E_T_BUILD_FREE, "baseline_tps": BASELINE_TPS, "target": TARGET,
            "wandb_group": args.wandb_group,
            "source_runs": "fern#325, stark#331(b48rmwjq), lawine#330(hfrscdai), fern#34(gua9x68j)",
        },
    )
    if run is None:
        print("[eagle3-cov-et-tile] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "cov_et_tile_self_test_passes": int(bool(st["cov_et_tile_self_test_passes"])),   # PRIMARY
        "honest_envelope_worst": d4["honest_envelope_worst"],                              # TEST
        "honest_envelope_central": d4["honest_envelope_central"],
        "honest_envelope_clears_500": int(bool(d4["honest_envelope_clears_500"])),         # REPORT
        "honest_envelope_clears_500_central": int(bool(d4["honest_envelope_clears_500_central"])),
        "both_corners_below_500": int(bool(d4["both_corners_below_500"])),
        "et_linear_0p9213": d1["et_linear_0p9213"],
        "et_fusion_0p8903": d1["et_fusion_0p8903"],
        "et_drop_pct": d1["et_drop_pct"],
        "cov_needed_for_0p9213": d2["cov_needed_for_0p9213"],
        "cov4_fusion_seed": d2["cov4_fusion_seed"],
        "cov_shortfall": d2["cov_shortfall"],
        "w_min_restore": d2["restore_lower_bound"]["w_min_restore"],
        "m_min_restore": d2["restore_lower_bound"]["m_min_restore"],
        "w_geom_estimate": d2["restore_geom_sensitivity"]["w_continuous"],
        "m_geom_estimate": d2["restore_geom_sensitivity"]["m_at_w_ceil"],
        "restoration_crosses_cliff": int(bool(d2["verdict_restoration_crosses_cliff"])),
        "mu_cliff": d3["mu_cliff"],
        "mu_tie_crossover": d3["mu_tie_crossover"],
        "real_mu_exceeds_tie": int(bool(d3["real_mu_exceeds_tie"])),
        "lever_A_subcliff": d3["operating_point_A_subcliff"]["effective_lever"],
        "lever_B_supracliff": d3["operating_point_B_supracliff"]["effective_lever"],
        "compliant_worst_A_subcliff": d3["operating_point_A_subcliff"]["compliant_worst"],
        "compliant_worst_B_supracliff": d3["operating_point_B_supracliff"]["compliant_worst"],
        "A_wins": int(bool(d3["A_wins"])),
        "scale_et_honest_over_611": d4["scale_et_honest_over_611"],
        "central_headroom_pct": d4["central_headroom_pct"],
        "worst_headroom_pct": d4["worst_headroom_pct"],
        "verdict_collapses": int(bool(vd["honest_coverage_collapses_envelope"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_cov_et_tile_coupling_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[eagle3-cov-et-tile] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="eagle3-cov-et-tile-coupling")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 337, "agent": "stark",
        "kind": "eagle3-cov-et-tile-coupling", "analysis_only": True, "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["18_nan_clean"] = not nan_paths
    syn["self_test"]["cov_et_tile_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    syn["headline"]["cov_et_tile_self_test_passes"] = syn["self_test"][
        "cov_et_tile_self_test_passes"]
    if nan_paths:
        print(f"[eagle3-cov-et-tile] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_cov_et_tile_coupling_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[eagle3-cov-et-tile] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["cov_et_tile_self_test_passes"] and payload["nan_clean"])
        print(f"[eagle3-cov-et-tile] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
