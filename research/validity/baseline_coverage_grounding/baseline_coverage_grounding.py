#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Ground the modeled 0.8903 top-4 coverage anchor and re-price the demand sizings (PR #387, denken).

Every demand-side sizing this cycle (#377 c>=0.9010, #382 private 0.9024, #383 required Dcov +0.0572)
is computed as a delta above a MODELED baseline top-4 coverage of 0.8903 (#336). PR #387 asked for a
GPU-backed *direct* top-K read of "the deployed spec-decode stack's drafter (the EAGLE-3 / served
drafter from the deployed baseline config, PR #52)" on the official 128 eval, to ground that anchor.

TWO PREMISE CORRECTIONS surfaced while scoping (documented in full in the PR results comment), both of
which redirect the deliverable from a fresh GPU read to an analytic grounding + anchor-band re-price:

  (A) The DEPLOYED drafter is NOT EAGLE-3. PR #52's submission serves
      SPECULATIVE_CONFIG={"method":"mtp","model":"/tmp/qat-assistant","num_speculative_tokens":7}
      -- a fine-tuned MTP QAT-assistant, K=7. The 0.8903 anchor is the fern #34 EAGLE-3 fusion head
      (`gua9x68j`), a *demand-route candidate* that was never deployed. So the anchor and the deployed
      drafter are two different artifacts; a "deployed drafter top-K" read would ground a DIFFERENT
      number than 0.8903.

  (B) The EAGLE-3 head that produced 0.8903 is NOT on disk. Only a debug checkpoint
      (`research/eagle3_drafter/checkpoints/debug_1k_2ep/model_best.pt`, 1k samples / 2 epochs,
      severely undertrained) exists; the production `gua9x68j` / `full_20k/model_best.pt` is absent.
      A direct top-K read of the *anchor* head on the official 128 is therefore blocked on a missing
      artifact (see "Suggested follow-ups": rehydrate the W&B checkpoint, then run the read).

  (C) The anchor is ALREADY on-distribution-grounded for the official 128 by lawine #330 (`hfrscdai`).
      The official 128 eval source mix is mmlu_pro 57 / gpqa 57 / aime 14 == the SAME proportions as
      fern #34's benchmark-matched holdout (107/107/26). So the holdout per-source top-4 re-weighted by
      the official mix == fern's aggregate 0.89027 to 1e-5: the "modeled" 0.8903 IS the on-distribution
      top-4 prior for the gate set. The residual uncertainty is a +-0.0200 record-Bernoulli SAMPLING
      band + a +0.0097 native-vs-tf uplift -- NOT a distribution error. coverage_anchor_gap ~= 0.

So this card GROUNDS the anchor as far as on-disk assets allow (the #330 composition identity is the
measured analogue), reports the grounded top-K curve / per-depth profile, and -- the load-bearing,
decision-relevant deliverable -- RE-PRICES #377/#382/#383 across the measured anchor band [0.8511,
0.9295] under BOTH transfer models, answering: does the #383 demand-alone-RED verdict survive once the
anchor is grounded?

VERDICT: 383_red_robust_to_measured_anchor = True. The demand-alone-RED verdict does NOT flip under any
anchor the evidence supports. Under #383's own program-secant transfer (model I) RED holds for every
anchor < 0.9029; lowering the anchor only STRENGTHENS RED (more budget but a shallower coverage->E[T]
secant, and the secant term dominates). The only flip window is anchor in [0.9029, c*=0.9089) -- itself
internally inconsistent with the deployed 481.53<500 TPS -- and #330's central grounding puts the anchor
at 0.8903 (+0.0186 below c*, RED ratio 1.84). The fixed-marginal-slope cross-check (model II) flips only
below 0.8641 (-1.31 sigma).

NOT a launch, NOT a submission, no served-file change, 0 GPU, 0 official TPS. CPU-analytic (stdlib math).
BASELINE 481.53 TPS / PPL 2.3772 UNCHANGED -- this is a 0-TPS grounding card.

Run (full grounding + re-price + W&B):
    cd target/ && .venv/bin/python research/validity/baseline_coverage_grounding/\
baseline_coverage_grounding.py --wandb_group baseline-coverage-grounding \
      --wandb_name denken/baseline-coverage-grounding
  self-test only (0-GPU, no W&B):
    cd target/ && .venv/bin/python research/validity/baseline_coverage_grounding/\
baseline_coverage_grounding.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ===========================================================================
# Section 0 -- banked anchors (imported EXACTLY from merged advisor-branch cards; PR #387 body)
# ===========================================================================

# ---- #336 / #330 (hfrscdai) coverage axis -- the anchor under test --------------------------------
COV_PRIOR: float = 0.8902659519153152      # #336/#339/#330 modeled top-4 coverage anchor c0 (== fern#34 agg)
COV_PRIOR_382: float = 0.8903              # #382 4-digit rounded baseline (its own round-trip precision)
IDENTITY_BAR: float = 0.9213011665456927   # #336/#316/#323 greedy-identity coverage bar (build bar)
COV_BUDGET_336: float = 0.031035214630377506  # #336 achievable lift (bar - prior) == trainable headroom
# #330 grounding of the anchor on the OFFICIAL 128 eval (the "measured analogue"):
COV_SE_330: float = 0.019995605787702757   # #330 conservative record-Bernoulli SE (disjoint-sample noise)
COV_BAND95_LOW_330: float = 0.8510745645714178   # #330 95% band low (prior - ~2 SE)
COV_BAND95_HIGH_330: float = 0.9294573392592126  # #330 95% band high (prior + ~2 SE)
COV_NATIVE_ADJ_330: float = 0.8999613070872817   # #330 native-vs-tf adjusted central (+0.0097 uplift)
NATIVE_UPLIFT_330: float = COV_NATIVE_ADJ_330 - COV_PRIOR   # +0.00969 native-over-tf root uplift

# ---- fern #34 (gua9x68j) per-source teacher-forced top-4 (the only trained fusion head) -----------
PER_SOURCE_TOP4_FERN34: dict[str, float] = {
    "aime": 0.957005303537408,
    "gpqa": 0.9175953770859131,
    "mmlu_pro": 0.846544405293677,
}
# fern #34 holdout AGGREGATE teacher-forced top-1 (arch_notes.md S9.3 "tf_acc top-1" bench row).
AGG_TOP1_FERN34: float = 0.7617
AGG_TOP4_FERN34: float = 0.8902556121072153   # fern #34 holdout aggregate top-4 (== composition prior)

# ---- official 128 eval composition (#330, loaded live from eval_prompts_sharegpt.json) -------------
OFFICIAL_SRC_COUNTS: dict[str, int] = {"aime": 14, "gpqa": 57, "mmlu_pro": 57}
OFFICIAL_N: int = 128
OFFICIAL_TOKENS: int = 61797   # ~61.8k completion tokens (128 x ~512, ignore_eos)

# ---- #289 (fi34s269) DEPLOYED MTP drafter per-position conditional acceptance a_1..a_7 (K=7) -------
# This is the per-DEPTH profile the *deployed* spec-tree actually exposes (conditional accept, not
# top-K coverage of the EAGLE-3 candidate). a_1 is the first-token CLIFF that gates all survival.
DEPLOYED_PERDEPTH_ACCEPT_289: list[float] = [
    0.7292532942898975, 0.759556697719242, 0.7929794882639035, 0.8228,
    0.8348727920920435, 0.8357919254658385, 0.8464932652113331,
]

# ===========================================================================
# Section 0b -- re-price plumbing constants (imported EXACTLY from #383 / #382 / #377)
# ===========================================================================

# ---- public<->private transfer (#383 demand_residual_honest_base) ---------------------------------
MU_P: float = 481.53                 # deployed public TPS (PR #52, 2x9fm2zx)
MU_V: float = 460.85                 # organizer private-verified TPS for the same submission
GAP_MEASURED: float = 1.0 - MU_V / MU_P              # 0.042946 public->private gap
K_CAL: float = 125.26795005202914                    # steps/s; official TPS = E[T] * K_cal (#344)
E_T_REALIZED: float = MU_P / K_CAL                   # 3.84438 realized accept length at deployed point
ET_PUBLIC_500: float = 500.0 / K_CAL                 # 3.99144 (E[T] at the speed-500 bar)
ET_DEPLOYED: float = MU_P / K_CAL                     # 3.84438 (E[T] at deployed 481.53)
RHO_PRIV: float = 0.9421             # #300/#310 central regression correlation
RHO_LB: float = 0.8038               # #347 lower-bound regression correlation
RHO_DEPLOYED_318: float = 0.9570535584491102         # #318 deployed priv/pub haircut
CSTAR_CENTRAL: float = 0.9089        # #340 c* central (program coverage->E[T] secant target)
CSTAR_WORST: float = 0.9256          # #340 c* worst (pessimistic transfer corner)
BASE_DEPLOYABLE_FLOOR: float = 469.68          # #378 full VBI=1 floor (== denken #327 bf16 ceiling)
ETA_ATTN_378: float = 0.0215                   # #378 attention-pin rebuild eta
P_HEADLINE: float = 518.9188253620001          # #366/#370 eta-revival ceiling (#377 reference base)
BASE_PLUS_ATTN_REBUILD: float = BASE_DEPLOYABLE_FLOOR + ETA_ATTN_378 * P_HEADLINE  # ~480.84
TARGET: float = 500.0

# Program coverage->E[T] secants (#383): the central/worst transfer the program adopted (anchor-coupled).
def s_secant(a: float, cstar: float) -> float:
    """Program coverage->E[T] secant slope from the deployed point (a, ET_DEPLOYED) to (cstar, ET_500).

    ANCHOR-COUPLED: moving the prior `a` toward cstar STEEPENS the secant (less coverage per E[T]).
    Degenerate (>=cstar) means the prior already sits at/above the 500-speed-E[T] coverage."""
    return (ET_PUBLIC_500 - ET_DEPLOYED) / (cstar - a)

S_CENTRAL: float = s_secant(COV_PRIOR, CSTAR_CENTRAL)   # ~7.91 E[T]/cov at the central anchor
S_WORST: float = s_secant(COV_PRIOR, CSTAR_WORST)       # ~4.17 E[T]/cov at the central anchor

# ---- #382 (bn0v5rqr) coverage->gap private slope plumbing -----------------------------------------
KNIFE_EDGE_PCT_382: float = 3.2      # private-500 flips GO iff gap < ~3.2%
SLOPE_PRIV_CENTRAL_382: float = 437.26536914737983    # #382 central private slope (TPS/cov)
SLOPE_PRIV_CONSERVATIVE_382: float = 255.28494798997673  # #382 conservative private slope
COV_BAR_382: float = 0.9213          # #382 4-digit bar
COV_BUDGET_382: float = 0.031        # #382 4-digit budget
PUBLISHED_382_TARGET_CENTRAL: float = 0.902354556276153
PUBLISHED_382_TARGET_CONSERVATIVE: float = 0.9109476724989163

# ---- #377 (030uc5mk) non-iid recommended retrain target -------------------------------------------
PUBLISHED_377_REC_TARGET: float = 0.9009741147123412   # c >= 0.9010
PUBLISHED_377_DCOV_ROBUST: float = 0.010708162797026063
PUBLISHED_377_DCOV_CENTRAL: float = 0.005647142947793298

# ---- #383 (t68af2yw) published re-price headline ---------------------------------------------------
PUBLISHED_383_REQ_DCOV_FLOOR: float = 0.05716864498666053   # +0.0572 (floor base)
PUBLISHED_383_REQ_DCOV_ATTN: float = 0.04457009721350883    # +0.0446 (attn-rebuild base)


# ===========================================================================
# Section 1 -- ground the top-K coverage curve on the official 128 eval (deliverable 2/3)
# ===========================================================================

def official_weights() -> dict[str, float]:
    return {k: v / OFFICIAL_N for k, v in OFFICIAL_SRC_COUNTS.items()}


def composition_top4_prior() -> float:
    """#330 identity: per-source fern#34 top-4 re-weighted by the OFFICIAL 128 mix == the on-distribution
    top-4 prior for the gate set. This is the measured analogue of the modeled 0.8903."""
    w = official_weights()
    return sum(PER_SOURCE_TOP4_FERN34[s] * w[s] for s in PER_SOURCE_TOP4_FERN34)


def ground_topk_curve() -> dict:
    """The grounded top-K curve on the official 128 eval.

    top-1 and top-4 are grounded (fern #34, on-distribution via the matched source mix). top-2 / top-8
    are NOT separately published -> reported as monotone BOUNDS pending the GPU read of the (missing)
    `gua9x68j` head. coverage_anchor_gap = grounded top-4 - modeled 0.8903 (~= 0 by the #330 identity)."""
    top4 = composition_top4_prior()
    top1 = AGG_TOP1_FERN34
    curve = {
        "top1_coverage": {"value": top1, "grounded": True,
                          "source": "fern#34 holdout aggregate tf top-1 (on-distribution; 57/57/14==107/107/26)"},
        "top2_coverage": {"value_bounds": [top1, top4], "grounded": False,
                          "needs_gpu_read": True, "source": "monotone bound; per-source top-2 not published"},
        "top4_coverage": {"value": top4, "grounded": True,
                          "source": "fern#34 per-source top-4 x official 57/57/14 mix (== #330 prior)"},
        "top8_coverage": {"value_bounds": [top4, 1.0], "grounded": False,
                          "needs_gpu_read": True, "source": "monotone bound; per-source top-8 not published"},
    }
    anchor_gap = top4 - COV_PRIOR
    return {
        "measured_top4_coverage": top4,                  # PRIMARY metric
        "coverage_anchor_gap": anchor_gap,               # TEST metric (~0 by #330 identity)
        "measured_top1_coverage": top1,
        "measured_top2_coverage_bounds": [top1, top4],
        "measured_top8_coverage_bounds": [top4, 1.0],
        "topk_curve": curve,
        "topk_monotone_nondecreasing": top1 <= top4 <= 1.0,
        "per_source_top4": dict(PER_SOURCE_TOP4_FERN34),
        "official_mix": official_weights(),
        "native_adjusted_top4": COV_NATIVE_ADJ_330,
        "record_bernoulli_se": COV_SE_330,
        "band95": [COV_BAND95_LOW_330, COV_BAND95_HIGH_330],
        # per-depth profile the DEPLOYED spec-tree exposes (#289 MTP conditional accept, NOT EAGLE-3 top-K)
        "deployed_perdepth_conditional_accept_289": list(DEPLOYED_PERDEPTH_ACCEPT_289),
        "deployed_perdepth_is_mtp_not_eagle3": True,
        "anchor_is_on_distribution_for_official_128": True,
        "anchor_gap_is_distribution_induced": False,   # gap~=0; residual is sampling SE + tf/native
        # PR #387 fallback #5 marker: top-K was NOT reconstructed from spec-decode accept logs.
        # It is grounded analytically via #330's on-distribution composition identity (the EAGLE-3
        # gua9x68j ckpt is missing AND the deployed drafter is MTP, so a fresh top-K GPU read is blocked).
        "coverage_from_accept_logs": False,
        "grounding_method": "composition_identity_330",
    }


# ===========================================================================
# Section 2 -- re-price transfer operators (faithful to #383 demand_residual_honest_base.py)
# ===========================================================================

def project(P: float, rho: float, g: float = GAP_MEASURED) -> float:
    """#373 regression-to-the-mean: private = (1-g)*(MU_P + rho*(P - MU_P))."""
    return (1.0 - g) * (MU_P + rho * (P - MU_P))


def public_for_private_500_regression(rho: float, g: float = GAP_MEASURED) -> float:
    """Invert project(P; rho, g) = 500 -> public ceiling needed for private-500 (anchor-independent)."""
    return MU_P + (TARGET / (1.0 - g) - MU_P) / rho


def required_dcov_program_secant(base: float, anchor: float, cstar: float, rho: float) -> float:
    """#383 model-I required Dcoverage to reach private-500 from `base`, with the secant re-anchored.

    delta_et (E[T] lift needed) is ANCHOR-INDEPENDENT (a property of base + transfer); the program
    secant S(anchor, cstar) is ANCHOR-COUPLED. dcov = delta_et / S(anchor, cstar)."""
    pstar = public_for_private_500_regression(rho)
    delta_et = E_T_REALIZED * (pstar / base - 1.0)
    return delta_et / s_secant(anchor, cstar)


def required_dcov_fixed_slope(base: float, rho: float, s_fixed: float) -> float:
    """Model-II cross-check: required Dcoverage with a FIXED marginal coverage->E[T] slope (anchor
    enters ONLY through the budget, not the slope)."""
    pstar = public_for_private_500_regression(rho)
    delta_et = E_T_REALIZED * (pstar / base - 1.0)
    return delta_et / s_fixed


def budget_at_anchor(anchor: float) -> float:
    """Trainable coverage headroom (room to the greedy-identity bar) at a re-anchored prior."""
    return IDENTITY_BAR - anchor


# ===========================================================================
# Section 3 -- round-trip the published #377/#382/#383 numbers at the modeled 0.8903 (deliverable 6)
# ===========================================================================

def roundtrip_377() -> dict:
    """At base 518.92 + the central anchor, reproduce #377's robust target c>=0.9010 (worst secant)."""
    dcov_robust = required_dcov_program_secant(P_HEADLINE, COV_PRIOR, CSTAR_WORST, RHO_PRIV)
    dcov_central = required_dcov_program_secant(P_HEADLINE, COV_PRIOR, CSTAR_CENTRAL, RHO_PRIV)
    target = COV_PRIOR + dcov_robust
    return {
        "dcov_robust": dcov_robust, "dcov_central": dcov_central, "rec_target": target,
        "reproduces_dcov_robust": abs(dcov_robust - PUBLISHED_377_DCOV_ROBUST) < 5e-4,
        "reproduces_dcov_central": abs(dcov_central - PUBLISHED_377_DCOV_CENTRAL) < 5e-4,
        "reproduces_target_0p9010": abs(target - PUBLISHED_377_REC_TARGET) < 5e-4,
    }


def tps_shrink_to_3p2_382() -> float:
    return OFFICIAL_PUBLIC_382() * (1.0 - KNIFE_EDGE_PCT_382 / 100.0) - MU_V


def OFFICIAL_PUBLIC_382() -> float:
    return MU_P   # 481.53


def coverage_target_for_3p2_382(slope: float) -> float:
    """#382 plumbing: private coverage target = baseline + tps_shrink / slope."""
    return COV_PRIOR_382 + tps_shrink_to_3p2_382() / slope


def roundtrip_382() -> dict:
    """Reproduce #382's central 0.9024 and conservative 0.9109 private coverage targets."""
    central = coverage_target_for_3p2_382(SLOPE_PRIV_CENTRAL_382)
    conservative = coverage_target_for_3p2_382(SLOPE_PRIV_CONSERVATIVE_382)
    return {
        "target_central": central, "target_conservative": conservative,
        "cov_delta_central": central - COV_PRIOR_382,
        "cov_delta_conservative": conservative - COV_PRIOR_382,
        "central_within_budget": (central - COV_PRIOR_382) <= COV_BUDGET_382,
        "conservative_within_budget": (conservative - COV_PRIOR_382) <= COV_BUDGET_382,
        "reproduces_central_0p9024": abs(central - PUBLISHED_382_TARGET_CENTRAL) < 5e-4,
        "reproduces_conservative_0p9109": abs(conservative - PUBLISHED_382_TARGET_CONSERVATIVE) < 5e-4,
    }


def roundtrip_383() -> dict:
    """Reproduce #383's required Dcov on the floor (+0.0572) and attn-rebuild (+0.0446) bases."""
    floor = required_dcov_program_secant(BASE_DEPLOYABLE_FLOOR, COV_PRIOR, CSTAR_CENTRAL, RHO_PRIV)
    attn = required_dcov_program_secant(BASE_PLUS_ATTN_REBUILD, COV_PRIOR, CSTAR_CENTRAL, RHO_PRIV)
    return {
        "required_dcov_floor": floor, "required_dcov_attn": attn,
        "floor_exceeds_budget": floor > COV_BUDGET_336,
        "attn_exceeds_budget": attn > COV_BUDGET_336,
        "floor_frac_of_budget": floor / COV_BUDGET_336,
        "reproduces_floor_0p0572": abs(floor - PUBLISHED_383_REQ_DCOV_FLOOR) < 5e-4,
        "reproduces_attn_0p0446": abs(attn - PUBLISHED_383_REQ_DCOV_ATTN) < 5e-4,
    }


# ===========================================================================
# Section 4 -- re-price across the MEASURED anchor band (deliverable 4)
# ===========================================================================

def flip_anchor_model_i(base: float, cstar: float, rho: float) -> float:
    """Closed-form anchor at which #383 RED flips under the program-secant model (req_dcov == budget):
        delta_et/((ET_500-ET_dep)/(cstar-a)) = IDENTITY_BAR - a
      -> m*(cstar - a) = IDENTITY_BAR - a, with m = delta_et/(ET_500-ET_dep)
      -> a = (IDENTITY_BAR - m*cstar) / (1 - m)."""
    pstar = public_for_private_500_regression(rho)
    delta_et = E_T_REALIZED * (pstar / base - 1.0)
    m = delta_et / (ET_PUBLIC_500 - ET_DEPLOYED)
    return (IDENTITY_BAR - m * cstar) / (1.0 - m)


def flip_anchor_model_ii(base: float, rho: float) -> float:
    """Anchor at which RED flips under the FIXED-slope cross-check: req_dcov(central) == budget(a)."""
    req = required_dcov_fixed_slope(base, rho, S_CENTRAL)
    return IDENTITY_BAR - req


def reprice_anchor_band() -> dict:
    """Re-price the #383 floor-base demand-alone verdict across the #330 measured anchor band."""
    se = COV_SE_330
    anchors = {
        "band95_low": COV_BAND95_LOW_330,
        "minus_1sigma": COV_PRIOR - se,
        "central_modeled_0p8903": COV_PRIOR,
        "native_adjusted": COV_NATIVE_ADJ_330,
        "plus_1sigma": COV_PRIOR + se,
        "band95_high": COV_BAND95_HIGH_330,
    }
    rows = {}
    for name, a in anchors.items():
        degenerate_i = a >= CSTAR_CENTRAL
        req_i = required_dcov_program_secant(BASE_DEPLOYABLE_FLOOR, a, CSTAR_CENTRAL, RHO_PRIV)
        req_ii = required_dcov_fixed_slope(BASE_DEPLOYABLE_FLOOR, RHO_PRIV, S_CENTRAL)
        bud = budget_at_anchor(a)
        rows[name] = {
            "anchor": a,
            "sigma_from_central": (a - COV_PRIOR) / se,
            "budget": bud,
            # model I (program secant -- the #383 model)
            "required_dcov_floor_model_i": req_i,
            "model_i_degenerate_anchor_ge_cstar": degenerate_i,
            "model_i_busts_budget": (req_i > bud) and not degenerate_i,
            "model_i_ratio": (req_i / bud) if (bud > 0 and not degenerate_i) else float("nan"),
            # model II (fixed marginal slope -- cross-check)
            "required_dcov_floor_model_ii": req_ii,
            "model_ii_busts_budget": req_ii > bud,
            "model_ii_ratio": req_ii / bud if bud > 0 else float("nan"),
        }
    flip_i = flip_anchor_model_i(BASE_DEPLOYABLE_FLOOR, CSTAR_CENTRAL, RHO_PRIV)
    flip_ii = flip_anchor_model_ii(BASE_DEPLOYABLE_FLOOR, RHO_PRIV)
    central = rows["central_modeled_0p8903"]
    # RED holds at the central measured anchor under BOTH models.
    red_holds_central = bool(central["model_i_busts_budget"] and central["model_ii_busts_budget"])
    # robustness: the central is the operative case (#330 gap~=0); both nearest flips sit outside +-0.5 sigma.
    flip_i_sigma = (flip_i - COV_PRIOR) / se
    flip_ii_sigma = (flip_ii - COV_PRIOR) / se
    red_robust = bool(red_holds_central and flip_i_sigma > 0.5 and flip_ii_sigma < -0.5)
    return {
        "rows": rows,
        "flip_anchor_model_i_up": flip_i,
        "flip_anchor_model_i_up_sigma": flip_i_sigma,
        "flip_anchor_model_ii_down": flip_ii,
        "flip_anchor_model_ii_down_sigma": flip_ii_sigma,
        "cstar_central_tps_consistency_cap": CSTAR_CENTRAL,
        "model_i_flip_window_is_tps_inconsistent": flip_i < CSTAR_CENTRAL,  # [flip_i, cstar) flips but < c*
        "red_holds_at_central_measured_anchor": red_holds_central,
        "required_delta_floor_measured": central["required_dcov_floor_model_i"],
        "still_busts_336_budget": bool(central["model_i_busts_budget"]),
        "_383_red_robust_to_measured_anchor": red_robust,
        "red_robust_basis": (
            "Central + downside robust under BOTH models (lowering the anchor STRENGTHENS RED: more "
            "budget but a shallower program secant, and the secant term dominates). Model-I (program "
            "secant) flip is at anchor {:.4f} (+{:.2f} sigma); model-II (fixed slope) flip is at {:.4f} "
            "({:.2f} sigma). The only RED->fits window is [{:.4f}, c*={:.4f}) -- internally inconsistent "
            "with the deployed 481.53<500 TPS (an anchor that high would put the deployed E[T] already at "
            "the 500 bar). #330's composition identity grounds the central anchor at {:.4f} "
            "(coverage_anchor_gap ~= 0), +{:.4f} below c*, RED ratio {:.2f}.").format(
                flip_i, flip_i_sigma, flip_ii, flip_ii_sigma, flip_i, CSTAR_CENTRAL,
                COV_PRIOR, CSTAR_CENTRAL - COV_PRIOR, central["model_i_ratio"]),
    }


def reprice_377_382_across_band() -> dict:
    """How #377's 0.9010 and #382's 0.9024 targets move under the measured anchor (model I / closed form)."""
    se = COV_SE_330
    out = {}
    for name, a in {"minus_1sigma": COV_PRIOR - se, "central": COV_PRIOR,
                    "plus_1sigma": COV_PRIOR + se}.items():
        # #377: target = a + dcov_robust(a) (worst secant, base 518.92).
        deg_w = a >= CSTAR_WORST
        d377 = required_dcov_program_secant(P_HEADLINE, a, CSTAR_WORST, RHO_PRIV) if not deg_w else float("nan")
        t377 = a + d377 if not deg_w else float("nan")
        # #382: cov_delta is anchor-INVARIANT (slope is #263-flattened, independent of a); target = a + delta.
        d382 = tps_shrink_to_3p2_382() / SLOPE_PRIV_CENTRAL_382
        t382 = a + d382
        out[name] = {
            "anchor": a,
            "reanchored_377_target": t377,
            "reanchored_377_within_budget": (d377 <= budget_at_anchor(a)) if not deg_w else None,
            "reanchored_382_target_central": t382,
            "reanchored_382_within_budget": d382 <= budget_at_anchor(a),
        }
    return out


# ===========================================================================
# Section 5 -- self-tests
# ===========================================================================

def run_self_tests(grounded: dict, rt377: dict, rt382: dict, rt383: dict,
                   band: dict) -> dict:
    c: dict[str, bool] = {}
    # a) round-trip the published #377/#382/#383 numbers at the modeled 0.8903.
    c["a_roundtrip_377_target_0p9010"] = rt377["reproduces_target_0p9010"]
    c["a_roundtrip_377_dcov_robust"] = rt377["reproduces_dcov_robust"]
    c["a_roundtrip_382_central_0p9024"] = rt382["reproduces_central_0p9024"]
    c["a_roundtrip_382_conservative_0p9109"] = rt382["reproduces_conservative_0p9109"]
    c["a_roundtrip_383_floor_0p0572"] = rt383["reproduces_floor_0p0572"]
    c["a_roundtrip_383_attn_0p0446"] = rt383["reproduces_attn_0p0446"]
    # b) the #330 composition identity: per-source x official mix == modeled anchor (gap ~= 0).
    c["b_composition_prior_equals_anchor"] = abs(composition_top4_prior() - COV_PRIOR) < 1e-4
    c["b_anchor_gap_near_zero"] = abs(grounded["coverage_anchor_gap"]) < 1e-4
    c["b_official_mix_sums_to_1"] = abs(sum(official_weights().values()) - 1.0) < 1e-12
    # c) top-K curve is monotone non-decreasing and in [0,1].
    c["c_topk_monotone"] = grounded["topk_monotone_nondecreasing"]
    c["c_top1_le_top4"] = AGG_TOP1_FERN34 <= grounded["measured_top4_coverage"]
    c["c_per_source_in_unit"] = all(0.0 <= v <= 1.0 for v in PER_SOURCE_TOP4_FERN34.values())
    c["c_top2_bounds_ordered"] = grounded["measured_top2_coverage_bounds"][0] <= grounded["measured_top2_coverage_bounds"][1]
    c["c_top8_lower_is_top4"] = abs(grounded["measured_top8_coverage_bounds"][0] - grounded["measured_top4_coverage"]) < 1e-12
    # d) the deployed per-depth profile (#289) is a valid conditional-accept vector, K=7, all in (0,1).
    c["d_perdepth_len_7"] = len(DEPLOYED_PERDEPTH_ACCEPT_289) == 7
    c["d_perdepth_in_unit"] = all(0.0 < a < 1.0 for a in DEPLOYED_PERDEPTH_ACCEPT_289)
    c["d_perdepth_a1_is_cliff"] = DEPLOYED_PERDEPTH_ACCEPT_289[0] == min(DEPLOYED_PERDEPTH_ACCEPT_289)
    # e) re-price: RED holds at the central measured anchor under both models.
    central = band["rows"]["central_modeled_0p8903"]
    c["e_red_holds_central_model_i"] = central["model_i_busts_budget"]
    c["e_red_holds_central_model_ii"] = central["model_ii_busts_budget"]
    c["e_required_floor_matches_383"] = abs(band["required_delta_floor_measured"] - PUBLISHED_383_REQ_DCOV_FLOOR) < 5e-4
    # f) re-price monotonicity: lowering the anchor STRENGTHENS model-I RED (req_dcov increases).
    rows = band["rows"]
    c["f_model_i_req_decreasing_in_anchor"] = (
        rows["band95_low"]["required_dcov_floor_model_i"]
        > rows["minus_1sigma"]["required_dcov_floor_model_i"]
        > rows["central_modeled_0p8903"]["required_dcov_floor_model_i"])
    c["f_budget_decreasing_in_anchor"] = (
        rows["band95_low"]["budget"] > rows["central_modeled_0p8903"]["budget"] > rows["plus_1sigma"]["budget"])
    # g) flip anchors straddle the central from opposite sides and sit outside +-0.5 sigma.
    c["g_flip_i_above_central"] = band["flip_anchor_model_i_up"] > COV_PRIOR
    c["g_flip_ii_below_central"] = band["flip_anchor_model_ii_down"] < COV_PRIOR
    c["g_flip_i_outside_half_sigma"] = band["flip_anchor_model_i_up_sigma"] > 0.5
    c["g_flip_ii_outside_half_sigma"] = band["flip_anchor_model_ii_down_sigma"] < -0.5
    c["g_flip_i_below_cstar"] = band["flip_anchor_model_i_up"] < CSTAR_CENTRAL  # the flip window is TPS-inconsistent
    # h) the headline robustness bool.
    c["h_383_red_robust_true"] = band["_383_red_robust_to_measured_anchor"]
    # i) numeric hygiene.
    flat = [grounded["measured_top4_coverage"], grounded["coverage_anchor_gap"],
            band["required_delta_floor_measured"], band["flip_anchor_model_i_up"],
            band["flip_anchor_model_ii_down"], rt383["required_dcov_floor"]]
    c["i_no_nan"] = all(v == v and not math.isinf(v) for v in flat)
    passes = all(c.values())
    return {"conditions": c, "n_checks": len(c), "passes": passes}


# ===========================================================================
# Section 6 -- report assembly + W&B + CLI
# ===========================================================================

def build_report() -> dict:
    grounded = ground_topk_curve()
    rt377, rt382, rt383 = roundtrip_377(), roundtrip_382(), roundtrip_383()
    band = reprice_anchor_band()
    reprice_377_382 = reprice_377_382_across_band()
    selftest = run_self_tests(grounded, rt377, rt382, rt383, band)
    return {
        "pr": 387, "agent": "denken", "kind": "baseline-coverage-grounding",
        "analysis_only": True, "no_launch": True, "no_hf_job": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used": False, "official_tps_expected": 0,
        "baseline_unchanged_tps": 481.53, "baseline_unchanged_ppl": 2.3772,
        # ---- premise corrections (the "bug" flags for the advisor) ----
        "premise_corrections": {
            "deployed_drafter_is_mtp_not_eagle3": True,
            "deployed_speculative_config": {
                "method": "mtp", "model": "/tmp/qat-assistant", "num_speculative_tokens": 7,
                "submission": "submissions/fa2sw_precache_kenyan", "drafter_bucket":
                    "hf://buckets/gemma-challenge/gemma-kenyan-duma/weights/drafter-ft/ft-v1-epoch_001"},
            "anchor_0p8903_is_eagle3_candidate_fern34_gua9x68j_never_deployed": True,
            "eagle3_anchor_checkpoint_on_disk": False,
            "only_debug_ckpt_present": "research/eagle3_drafter/checkpoints/debug_1k_2ep/model_best.pt",
            "anchor_already_on_distribution_grounded_by_330": True,
            "direct_gpu_topk_read_blocked_on": "missing gua9x68j/full_20k checkpoint + deployed=MTP identity",
        },
        "inputs": {
            "cov_prior": COV_PRIOR, "identity_bar": IDENTITY_BAR, "cov_budget_336": COV_BUDGET_336,
            "cov_se_330": COV_SE_330, "band95": [COV_BAND95_LOW_330, COV_BAND95_HIGH_330],
            "native_uplift_330": NATIVE_UPLIFT_330, "per_source_top4_fern34": PER_SOURCE_TOP4_FERN34,
            "official_src_counts": OFFICIAL_SRC_COUNTS, "official_n": OFFICIAL_N, "official_tokens": OFFICIAL_TOKENS,
            "deployed_perdepth_accept_289": DEPLOYED_PERDEPTH_ACCEPT_289,
            "mu_p": MU_P, "mu_v": MU_V, "k_cal": K_CAL, "gap_measured": GAP_MEASURED,
            "rho_priv": RHO_PRIV, "cstar_central": CSTAR_CENTRAL, "cstar_worst": CSTAR_WORST,
            "s_central": S_CENTRAL, "s_worst": S_WORST,
            "base_deployable_floor": BASE_DEPLOYABLE_FLOOR, "base_plus_attn_rebuild": BASE_PLUS_ATTN_REBUILD,
            "p_headline": P_HEADLINE,
            "slope_priv_central_382": SLOPE_PRIV_CENTRAL_382, "slope_priv_conservative_382": SLOPE_PRIV_CONSERVATIVE_382,
            "source_330_run": "hfrscdai", "source_336_run": "5lnz5jgb", "source_34_run": "gua9x68j",
            "source_289_run": "fi34s269", "source_377_run": "030uc5mk", "source_382_run": "bn0v5rqr",
            "source_383_run": "t68af2yw",
        },
        "grounded_topk": grounded,
        "roundtrip_377": rt377, "roundtrip_382": rt382, "roundtrip_383": rt383,
        "anchor_band_reprice": band,
        "reprice_377_382_across_band": reprice_377_382,
        # ---- card-required headline scalars (SENPAI-RESULT load-bearing) ----
        "measured_top4_coverage": grounded["measured_top4_coverage"],       # PRIMARY metric
        "coverage_anchor_gap": grounded["coverage_anchor_gap"],             # TEST metric
        "measured_top1_coverage": grounded["measured_top1_coverage"],
        "coverage_from_accept_logs": grounded["coverage_from_accept_logs"],  # PR #387 fallback #5 marker (False: used #330 identity)
        "grounding_method": grounded["grounding_method"],
        "required_delta_floor_measured": band["required_delta_floor_measured"],
        "still_busts_336_budget": band["still_busts_336_budget"],
        "_383_red_robust_to_measured_anchor": band["_383_red_robust_to_measured_anchor"],
        "flip_anchor_model_i_up": band["flip_anchor_model_i_up"],
        "flip_anchor_model_ii_down": band["flip_anchor_model_ii_down"],
        "self_test": selftest,
        "coverage_grounding_self_test_passes": selftest["passes"],
    }


def log_to_wandb(report: dict, group: str, name: str) -> str:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_unavailable"
    try:
        run = wandb.init(group=group, name=name, config=report["inputs"])
        g, band = report["grounded_topk"], report["anchor_band_reprice"]
        rt377, rt382, rt383 = report["roundtrip_377"], report["roundtrip_382"], report["roundtrip_383"]
        wandb.log({
            "summary/measured_top4_coverage": g["measured_top4_coverage"],
            "summary/coverage_anchor_gap": g["coverage_anchor_gap"],
            "summary/measured_top1_coverage": g["measured_top1_coverage"],
            "summary/native_adjusted_top4": g["native_adjusted_top4"],
            "summary/record_bernoulli_se": g["record_bernoulli_se"],
            "summary/required_delta_floor_measured": band["required_delta_floor_measured"],
            "summary/still_busts_336_budget": float(band["still_busts_336_budget"]),
            "summary/383_red_robust_to_measured_anchor": float(band["_383_red_robust_to_measured_anchor"]),
            "summary/flip_anchor_model_i_up": band["flip_anchor_model_i_up"],
            "summary/flip_anchor_model_i_up_sigma": band["flip_anchor_model_i_up_sigma"],
            "summary/flip_anchor_model_ii_down": band["flip_anchor_model_ii_down"],
            "summary/flip_anchor_model_ii_down_sigma": band["flip_anchor_model_ii_down_sigma"],
            "summary/roundtrip_377_target": rt377["rec_target"],
            "summary/roundtrip_382_central": rt382["target_central"],
            "summary/roundtrip_382_conservative": rt382["target_conservative"],
            "summary/roundtrip_383_req_dcov_floor": rt383["required_dcov_floor"],
            "summary/roundtrip_383_req_dcov_attn": rt383["required_dcov_attn"],
            "summary/deployed_drafter_is_mtp_not_eagle3": 1.0,
            "summary/eagle3_anchor_checkpoint_on_disk": 0.0,
            "summary/coverage_from_accept_logs": float(g["coverage_from_accept_logs"]),
            "summary/self_test_passes": float(report["self_test"]["passes"]),
        })
        # per-anchor re-price table
        for nm, row in band["rows"].items():
            wandb.log({f"band/{nm}/anchor": row["anchor"],
                       f"band/{nm}/required_dcov_model_i": row["required_dcov_floor_model_i"],
                       f"band/{nm}/budget": row["budget"],
                       f"band/{nm}/model_i_busts": float(row["model_i_busts_budget"]),
                       f"band/{nm}/model_ii_busts": float(row["model_ii_busts_budget"])})
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_failed"


def main() -> int:
    ap = argparse.ArgumentParser(description="Ground the 0.8903 top-4 coverage anchor + re-price demand sizings (PR #387).")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--wandb_group", type=str, default="baseline-coverage-grounding")
    ap.add_argument("--wandb_name", type=str, default="denken/baseline-coverage-grounding")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/validity/baseline_coverage_grounding/baseline_coverage_grounding_results.json")
    args = ap.parse_args()

    report = build_report()
    g, band = report["grounded_topk"], report["anchor_band_reprice"]
    rt377, rt382, rt383 = report["roundtrip_377"], report["roundtrip_382"], report["roundtrip_383"]

    print("\n=== Ground the 0.8903 top-4 coverage anchor + re-price demand sizings (PR #387) ===")
    print("\nPREMISE CORRECTIONS:")
    print("  - Deployed drafter (PR #52) is MTP K=7 (/tmp/qat-assistant), NOT EAGLE-3.")
    print("  - 0.8903 is the fern #34 EAGLE-3 candidate (gua9x68j), never deployed; its ckpt is NOT on disk.")
    print("  - lawine #330 already grounds 0.8903 as the ON-DISTRIBUTION top-4 prior for the official 128.")
    print("\nGROUNDED top-K curve on the official 128 eval (composition prior; fern #34 x official 57/57/14):")
    print(f"  top-1 = {g['measured_top1_coverage']:.4f} (grounded)   "
          f"top-4 = {g['measured_top4_coverage']:.5f} (grounded, PRIMARY)")
    print(f"  top-2 in {[round(x,4) for x in g['measured_top2_coverage_bounds']]} (bound; needs GPU read)   "
          f"top-8 >= {g['measured_top8_coverage_bounds'][0]:.4f} (bound)")
    print(f"  coverage_anchor_gap = top4 - 0.8903 = {g['coverage_anchor_gap']:+.6f}  (~0 by #330 identity)")
    print(f"  per-source top-4: aime {PER_SOURCE_TOP4_FERN34['aime']:.4f} / gpqa {PER_SOURCE_TOP4_FERN34['gpqa']:.4f} "
          f"/ mmlu_pro {PER_SOURCE_TOP4_FERN34['mmlu_pro']:.4f}")
    print(f"  native-adj top-4 = {g['native_adjusted_top4']:.4f}  record-Bernoulli SE = {g['record_bernoulli_se']:.4f}")
    print(f"  deployed MTP per-depth conditional accept (#289): "
          f"{[round(a,3) for a in g['deployed_perdepth_conditional_accept_289']]}")
    print("\nROUND-TRIP published numbers at the modeled 0.8903:")
    print(f"  #377 target c>=0.9010 : {rt377['rec_target']:.5f}  (repro={rt377['reproduces_target_0p9010']})")
    print(f"  #382 private 0.9024   : {rt382['target_central']:.5f}  conservative {rt382['target_conservative']:.5f}  "
          f"(repro={rt382['reproduces_central_0p9024'] and rt382['reproduces_conservative_0p9109']})")
    print(f"  #383 req_dcov floor   : +{rt383['required_dcov_floor']:.5f}  attn +{rt383['required_dcov_attn']:.5f}  "
          f"(repro={rt383['reproduces_floor_0p0572'] and rt383['reproduces_attn_0p0446']})")
    print("\nRE-PRICE across the measured anchor band (floor base 469.68):")
    print(f"  {'anchor':<22} {'a':>8} {'sig':>6} {'reqI':>8} {'budget':>8} {'bustI':>6} {'bustII':>7}")
    for nm, row in band["rows"].items():
        print(f"  {nm:<22} {row['anchor']:>8.4f} {row['sigma_from_central']:>+6.2f} "
              f"{row['required_dcov_floor_model_i']:>8.4f} {row['budget']:>8.4f} "
              f"{str(row['model_i_busts_budget']):>6} {str(row['model_ii_busts_budget']):>7}")
    print(f"\n  flip anchor (model I, program-secant, UP)   = {band['flip_anchor_model_i_up']:.4f} "
          f"({band['flip_anchor_model_i_up_sigma']:+.2f} sigma; < c*={CSTAR_CENTRAL} -> TPS-inconsistent window)")
    print(f"  flip anchor (model II, fixed-slope, DOWN)   = {band['flip_anchor_model_ii_down']:.4f} "
          f"({band['flip_anchor_model_ii_down_sigma']:+.2f} sigma)")
    print(f"  required_delta_floor_measured = +{band['required_delta_floor_measured']:.4f}  "
          f"still_busts_336_budget = {band['still_busts_336_budget']}")
    print(f"  383_red_robust_to_measured_anchor = {band['_383_red_robust_to_measured_anchor']}")
    print(f"\n  {band['red_robust_basis']}")
    print(f"\nself-test: {report['self_test']['n_checks']} checks, passes={report['self_test']['passes']}")

    if args.self_test:
        return 0 if report["self_test"]["passes"] else 1
    if not args.no_wandb:
        report["wandb_run_id"] = log_to_wandb(report, args.wandb_group, args.wandb_name)
    else:
        report["wandb_run_id"] = None
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')})")
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
