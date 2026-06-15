#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Grounding the DELIVERABILITY of the +0.0107 demand-side coverage-retrain target (PR #380, denken).

denken #377 (`030uc5mk`) sized the demand-side closer: retrain the EAGLE-3 fusion head to coverage
c >= 0.9010 (Delta-cov +0.0107 robust / +0.00565 central) closes the #373 +5.44 TPS private-500
residual, within #336's +0.031 budget. BUT that sizing rests on two ungrounded assumptions the
ENTIRE now-primary demand-side route inherits:

  (1) DELIVERABILITY. p_softkd_reasoning_retrain_delivers ~= 1.0 came from #339's delivery
      distribution N(0.0385, 0.00742). Is N(0.0385, 0.00742) grounded in an actual coverage-lift
      checkpoint / EAGLE-3 retraining literature, or is it itself an un-grounded model?
  (2) THE TRANSFER EFFICIENCY kappa. The +0.0107 depends on the realized coverage->accept transfer
      kappa ~= 0.672 (backed out of #340's c* corners). kappa was INFERRED, not measured. If kappa
      is actually lower, the required dcov rises and could exit #336's budget.

This card grounds BOTH. It is the de-risk (or RED flag) on the weakest link of the route fern #357
is about to bank.

WHAT kappa IS (from #377, recomputed here from first principles, self-test 'a'): the realized
coverage->E[T] conversion slope as a FRACTION of the kappa=1 uniform-additive passthrough bound.

    S_uniform = dE[l]/dc when every per-position conditional a_d shifts by +dc  (= 11.781, kappa=1)
    S_realized(kappa) = kappa * S_uniform
    kappa_central = S_program_central / S_uniform = 7.913 / 11.781 = 0.6716   (#377, int4-ct transfer)
    kappa_worst   = S_program_worst   / S_uniform = 4.173 / 11.781 = 0.3541   (worst c* corner)

The required NOMINAL coverage lift to close a residual dE[T] is required_dcov(kappa) = dE[T] /
(kappa * S_uniform): a 1/kappa curve. The +0.0107 "robust" target is just required_dcov(kappa_worst)
and the +0.00565 "central" is required_dcov(kappa_central) -- two points on ONE curve. kappa_breakeven
is the kappa at which required_dcov crosses #336's +0.031 budget.

CPU-ANALYTIC. No training, no checkpoint, no model forward, no served-file change, 0 official TPS.
Greedy/PPL untouched (an EAGLE-3 drafter only PROPOSES; emission is the verify argmax -> byte-exact
greedy regardless of drafter quality). The OPTIONAL local-GPU kappa-probe (--gpu --kappa-probe) is
gated and identity-safe; it is NOT run by default because the analytic kappa is well-bounded.

Run:
    cd target/ && python research/validity/coverage_retrain_deliverability/\
coverage_retrain_deliverability.py --ground-delivery --kappa-sweep \
      --wandb_group strict-bi-verify-gemm --wandb_name denken/coverage-deliverability
Self-test only (0-GPU, no W&B):
    ... coverage_retrain_deliverability.py --self-test
Reanalyze from the banked JSON (0-GPU, no recompute drift):
    ... coverage_retrain_deliverability.py --reanalyze
"""
from __future__ import annotations

import argparse
import json
import sys
from math import erf, sqrt
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "coverage_retrain_deliverability_results.json"

# ===========================================================================
# Section 0 -- banked anchors (all from MERGED advisor-branch cards / #377 deliverable)
# ===========================================================================

# ---- #377 non-iid conversion (030uc5mk) -- the sizing this card grounds --------------------------
DELTA_ET_CENTRAL_373: float = 0.04468363487955586     # #373 lever-b central residual dE[T] (+5.44 TPS)
DELTA_ET_CONSERVATIVE_373: float = 0.17631393110269514  # #373 conservative residual dE[T] (+17.96 TPS)
S_UNIFORM_CENTRAL: float = 11.781120641460003          # #377 kappa=1 uniform-additive passthrough bound
S_PROGRAM_CENTRAL: float = 7.912609135742992           # #377 program c*-central secant
S_PROGRAM_WORST: float = 4.172857261001455             # #377 program c*-worst secant
KAPPA_CENTRAL_377: float = 0.6716346752190123          # #377 implied realized transfer (S_c / S_unif)
DCOV_ROBUST_377: float = 0.010708162797026063          # #377 robust target (central resid, worst slope)
DCOV_CENTRAL_377: float = 0.005647142947793298         # #377 central target (central resid, central slope)

# ---- coverage scalars (#330 measured prior; #336 budget; #340 c* corners) ------------------------
COV_PRIOR: float = 0.8902659519153152                  # #330 MEASURED honest top-4 ROOT prior (fern #34)
IDENTITY_BAR: float = 0.9213011665456927               # #316/#336 greedy-identity coverage bar
COV_BUDGET_336: float = 0.031035214630377506           # #336 achievable lift (bar - prior)
CSTAR_CENTRAL: float = 0.9089                            # #340 c* central
CSTAR_WORST: float = 0.9256                              # #340 c* worst

# ---- #336 recipe bands (krroookz) -- the LITERATURE-ANCHORED per-lever priors --------------------
# Each band [low, high] is a literature-grounded EXPECTED-VALUE prior (NOT measured here -- #336's
# own words). #336 explicitly flags the top-4-tail attribution as "reasoned inference, not a
# per-paper ablation". Citations: DistillSpec (arXiv:2310.08461), OSD (arXiv:2310.07177),
# Medusa (arXiv:2401.10774), EAGLE-3 (arXiv:2503.01840).
LEVER_SOFTKD_MEAN: float = 0.030
LEVER_SOFTKD_BAND: tuple[float, float] = (0.015, 0.045)
LEVER_REASONING_MEAN: float = 0.025
LEVER_REASONING_BAND: tuple[float, float] = (0.010, 0.040)
NONADDITIVITY_HAIRCUT: float = 0.70                     # #336 combo haircut (shared mmlu_pro drag)

# ---- #339 delivery distribution (0aq16szh) -- mean + the THREE correlation spreads ---------------
# mean = haircut * (softkd_mean + reasoning_mean) = 0.70 * 0.055 = 0.0385 (combo central).
# sigma derivations (#339): per-lever sigma = (hi-lo)/4 = 0.0075 each.
#   independent (rho=0):  0.70*sqrt(0.0075^2+0.0075^2) = 0.00742  <- #377 USED THIS (TIGHTEST)
#   rho=+0.5:             0.70*0.0075*sqrt(3)          = 0.00909
#   comonotonic (rho=1):  0.70*(0.0075+0.0075)         = 0.01050  <- == #336's reported band
RECIPE_MEAN_DCOV: float = 0.0385
RECIPE_SIGMA_INDEP: float = 0.00742                    # #377's choice -- the OPTIMISTIC (rho=0) spread
RECIPE_SIGMA_RHO05: float = 0.00909
RECIPE_SIGMA_COMONO: float = 0.01050                   # the realistic correlated spread (#336 band)

# ---- #323 achievability literature anchor (ceddxj20) -- the FULLY-TRAINED head coverage ----------
# Cross-paper estimate for a FULLY-TRAINED published EAGLE-style head (a1~0.765): top-4 coverage
# 0.913 [0.899, 0.929]. Citations: EAGLE-3 (2503.01840), EAGLE-1 (2401.15077), HASS (2408.15766),
# Medusa, KOALA (2408.08146). The lift a published fully-trained head achieves over our 0.8903 prior
# is +0.0227 [+0.0087, +0.0387] -- a MORE DIRECTLY literature-grounded (and LOWER) delivery anchor
# than #336's modeled +0.0385 combination.
LIT_TOP4_CENTRAL: float = 0.913
LIT_TOP4_RANGE: tuple[float, float] = (0.899, 0.929)

# ---- PR #380 independent literature-verification pass (the de-risk this card owns) ----------------
# A focused re-read of the load-bearing citations sharpens the audit on three decision-critical points
# the upstream cards (#336/#339) did NOT surface:
#   (V1) NO cited paper reports top-4 COVERAGE (top-k hit-rate). All report acceptance-rate alpha or
#        accept-length tau (sampling metrics). The +0.0385 top-4-coverage number is an INFERENCE from
#        acceptance-rate gains via an ASSUMED (never-measured) transfer -> the metric itself is not
#        literature-measured.
#   (V2) The closest controlled A/B CONTRADICTS the soft-KD +0.030 component: EAGLE-1 (2401.15077)
#        ran DistillSpec-style logit distillation as a baseline, found it gave "only modest
#        improvements", and CHOSE CE-based feature regression because logit-KD UNDERPERFORMED. HASS's
#        top-K distillation gain is folded into a richer pipeline (unattributed). Defensible soft-KD
#        coverage lift: ~+0.005..+0.015, not +0.030.
#   (V3) The reasoning-DATA +0.025 is optimistic: EAGLE-1 reports "low sensitivity to training data"
#        (~3.6% speedup from a data-quality swap); HASS finds 1/4 of ShareGPT ~= full set. Defensible
#        data lift: ~+0.005..+0.010. And diminishing returns bite: KD benefit shrinks as the
#        student-teacher gap narrows (Mirzadeh 2020), and our head is already CAPABLE at 0.89.
# Net: #339's +0.0385 sits at the OPTIMISTIC tail (0.8903+0.0385=0.9288 ~ TOP of the published
# [0.899,0.929] fully-trained-head range). The DEFENSIBLE central for a FINE-TUNE (not a from-scratch
# retrain) of the 0.8903 head is the lower [+0.009, +0.029] range, central ~+0.016.
LIT_FINETUNE_RANGE: tuple[float, float] = (0.009, 0.029)   # PR #380 research: realistic fine-tune lift
DEFENSIBLE_FINETUNE_MEAN: float = 0.016    # conservative-central (logit-KD-underperforms + diminishing returns)
DEFENSIBLE_FINETUNE_SD: float = 0.006      # ~+-2sigma spans the research fine-tune range
PESSIMISTIC_FINETUNE_MEAN: float = 0.012   # low-central corner (EAGLE-1 contra + capable-head saturation)
PESSIMISTIC_FINETUNE_SD: float = 0.005
SOFTKD_DEFENSIBLE_BAND: tuple[float, float] = (0.005, 0.015)   # V2: vs #336's +0.030 central
DATA_DEFENSIBLE_BAND: tuple[float, float] = (0.005, 0.010)     # V3: vs #336's +0.025 central

TARGET_TPS: float = 500.0


# ===========================================================================
# Section 1 -- delivery-distribution grounding audit
# ===========================================================================

def classify_grounding() -> dict:
    """Trace #336's +0.031 budget and #339's N(0.0385, 0.00742) to their source and classify.

    Provenance chain (all MERGED advisor-branch cards):
      #330 hfrscdai  -> MEASURED honest top-4 ROOT prior 0.8903 (fern #34 eval). Grounded.
      #336 krroookz  -> per-lever Delta-cov bands (soft-KD +0.030, reasoning +0.025) with REAL
                        paper citations, combined as 0.70*(0.030+0.025)=0.0385. #336's own words:
                        "literature-grounded expected-value priors a retrain would confirm -- they
                        are NOT measured here" and the top-4-tail attribution is "reasoned inference,
                        not a per-paper ablation".
      #339 0aq16szh  -> convolved #336's bands into N(0.0385, sigma); sigma=0.00742 is the
                        INDEPENDENT (rho=0) spread -- the TIGHTEST/most optimistic of three.
      #377 030uc5mk  -> used mean 0.0385 AND the optimistic sigma 0.00742.

    Verdict: the levers and their magnitudes ARE cited to real papers (DistillSpec/OSD/Medusa/
    EAGLE-3) -> NOT modeled-only. But NO coverage-lift checkpoint has been run -> NOT
    measured-checkpoint. Classification: LITERATURE-ANCHORED, with two honest modeled layers on top
    (the acceptance->top-4-coverage attribution, and the 0.70 non-additivity haircut), and #377's
    sigma is the optimistic rho=0 spread.
    """
    # sanity: reconstruct the #339 mean from #336's levers + haircut (round-trip the provenance).
    reconstructed_mean = NONADDITIVITY_HAIRCUT * (LEVER_SOFTKD_MEAN + LEVER_REASONING_MEAN)
    sigma_lever = (LEVER_SOFTKD_BAND[1] - LEVER_SOFTKD_BAND[0]) / 4.0  # #339: (hi-lo)/4 = 0.0075
    sigma_indep = NONADDITIVITY_HAIRCUT * sqrt(sigma_lever ** 2 + sigma_lever ** 2)
    sigma_comono = NONADDITIVITY_HAIRCUT * (sigma_lever + sigma_lever)
    # where #339's +0.0385 mean lands inside the published fully-trained-head range [0.899, 0.929].
    cov_post_339 = COV_PRIOR + RECIPE_MEAN_DCOV
    frac_in_lit_range = (cov_post_339 - LIT_TOP4_RANGE[0]) / (LIT_TOP4_RANGE[1] - LIT_TOP4_RANGE[0])
    return {
        # The ENUM: a real literature anchor EXISTS (the published fully-trained-head coverage 0.913
        # [0.899,0.929] and the EAGLE-family fine-tune-sensitivity evidence) -> NOT a fiction, NOT
        # modeled-only. But it anchors a LOWER lift than #339's +0.0385, which is optimistic-modeled.
        "delivery_distribution_grounding": "literature-anchored",
        "original_339_distribution_optimistic": True,
        "original_339_grounding_if_taken_literally": "modeled-only",  # the +0.0385 magnitude itself
        "provenance_chain": [
            "#330 hfrscdai: MEASURED top-4 ROOT prior 0.8903 (fern #34 gua9x68j eval)",
            "#336 krroookz: per-lever literature priors soft-KD +0.030 / reasoning +0.025, "
            "combo 0.70*(0.030+0.025)=0.0385 (cited DistillSpec/OSD/Medusa/EAGLE-3; "
            "top-4-tail attribution = reasoned inference, NOT a per-paper ablation)",
            "#339 0aq16szh: N(0.0385, sigma); sigma=0.00742 is the rho=0 INDEPENDENT spread "
            "(tightest of indep/rho0.5/comonotonic 0.00742/0.00909/0.01050)",
            "#377 030uc5mk: adopted mean 0.0385 AND the optimistic sigma 0.00742 -> P(deliver)~1.0",
        ],
        "is_measured_checkpoint": False,   # no coverage-lift retrain has been run
        "is_modeled_only": False,          # a real literature ceiling anchor exists (#323 + EAGLE-family)
        # PR #380 independent verification of WHY #339's +0.0385 is optimistic (V1/V2/V3):
        "verification_findings": [
            "V1: NO cited paper (DistillSpec/OSD/Medusa/EAGLE-1/EAGLE-3/HASS/KOALA) reports top-4 "
            "COVERAGE; all report acceptance-rate alpha / accept-length tau. The +0.0385 top-4 number "
            "is an inference from acceptance gains via an ASSUMED transfer -> the metric is unmeasured.",
            "V2: EAGLE-1 (2401.15077) ran logit-KD as a baseline, found 'only modest improvements', "
            "and CHOSE CE feature-regression because logit-KD UNDERPERFORMED -> the closest A/B "
            "CONTRADICTS the soft-KD +0.030 (defensible ~+0.005..+0.015).",
            "V3: reasoning-DATA +0.025 is optimistic: EAGLE-1 'low sensitivity to training data' "
            "(~3.6% speedup), HASS 1/4-data~=full; diminishing returns as the capable 0.89 head "
            "narrows the teacher gap (defensible ~+0.005..+0.010).",
            "Net: 0.8903+0.0385=0.9288 ~ TOP of the published [0.899,0.929] range (best-case, not "
            "central). Defensible FINE-TUNE central ~+0.016, range [+0.009,+0.029].",
        ],
        "modeled_layers_on_top": [
            "acceptance/speedup -> top-4-coverage attribution (ASSUMED transfer; no paper measures it)",
            "0.70 non-additivity haircut (shared mmlu_pro drag; #336 modeling assumption)",
            "sigma=0.00742 is the rho=0 OPTIMISTIC spread; realistic correlated spread up to 0.0105",
            "the soft-KD +0.030 magnitude is CONTRADICTED by EAGLE-1's own logit-KD A/B (V2)",
        ],
        "cov_post_339_in_lit_range_frac": frac_in_lit_range,   # ~0.93 -> near the TOP of [0.899,0.929]
        "reconstructed_mean_from_336_levers": reconstructed_mean,
        "reconstructed_sigma_indep": sigma_indep,
        "reconstructed_sigma_comono": sigma_comono,
        "mean_roundtrips": abs(reconstructed_mean - RECIPE_MEAN_DCOV) < 1e-9,
        "sigma_indep_roundtrips": abs(sigma_indep - RECIPE_SIGMA_INDEP) < 5e-5,
    }


def build_defensible_distribution() -> dict:
    """Re-derive a DEFENSIBLE (conservative) delivery distribution from the PR #380 literature pass.

    Two literature anchors, from most-optimistic to conservative:

      (A) FROM-SCRATCH CEILING: a fully-trained-FROM-SCRATCH published EAGLE-style head reaches top-4
          coverage 0.913 [0.899,0.929] (#323 ceddxj20). Delta over our 0.8903 prior = +0.0227
          [+0.0087,+0.0387]. This is the OPTIMISTIC-defensible anchor -- it assumes our retrain
          matches a from-scratch head's ceiling.

      (B) FINE-TUNE REALISTIC (the headline DEFENSIBLE distribution): we are FINE-TUNING an existing
          0.8903 head, not retraining from scratch, so (A) is an upper bound. The PR #380 pass gives
          a defensible fine-tune range [+0.009,+0.029] after discounting for V2 (EAGLE-1 found
          logit-KD UNDERPERFORMS CE) and V3 (low data-sensitivity + diminishing returns on a capable
          head). Conservative-central +0.016, sd 0.006 (~+-2sigma spans the range). This is LOWER
          than both #339's +0.0385 and the from-scratch ceiling +0.0227 -> the genuinely conservative
          delivery the route must survive.
    """
    # (A) from-scratch ceiling anchor.
    ceiling_mean = LIT_TOP4_CENTRAL - COV_PRIOR
    ceiling_lo = LIT_TOP4_RANGE[0] - COV_PRIOR
    ceiling_hi = LIT_TOP4_RANGE[1] - COV_PRIOR
    ceiling_sd = (ceiling_hi - ceiling_lo) / (2.0 * 1.6448536269514722)  # p05/p95 reading
    return {
        # headline DEFENSIBLE = the fine-tune-realistic (B), the conservative case the route must clear.
        "name": "lit-anchored-fine-tune-realistic",
        "mean": DEFENSIBLE_FINETUNE_MEAN,
        "sd": DEFENSIBLE_FINETUNE_SD,
        "band": list(LIT_FINETUNE_RANGE),
        "provenance": (
            "PR #380 literature pass: a soft-KD + reasoning-data FINE-TUNE of the existing 0.8903 head "
            "(NOT a from-scratch retrain) defensibly lifts top-4 coverage by [+0.009,+0.029], central "
            "~+0.016. Discounted from #336's +0.0385 because (V2) EAGLE-1 (2401.15077) found logit-KD "
            "UNDERPERFORMS CE feature-regression and (V3) EAGLE-1 reports low training-data sensitivity "
            "(~3.6% speedup) + diminishing KD returns as the capable 0.89 head narrows the teacher gap. "
            "No cited paper measures top-4 coverage directly (V1)."),
        # secondary anchors, reported for the spectrum.
        "ceiling_from_scratch": {
            "name": "lit-anchored-from-scratch-ceiling",
            "mean": ceiling_mean, "sd": ceiling_sd, "band": [ceiling_lo, ceiling_hi],
            "provenance": (
                "#323 ceddxj20 published fully-trained-FROM-SCRATCH head top-4 0.913 [0.899,0.929] "
                "minus the #330-MEASURED 0.8903 prior; the OPTIMISTIC-defensible upper bound."),
        },
        "pessimistic": {
            "name": "lit-anchored-pessimistic-low-central",
            "mean": PESSIMISTIC_FINETUNE_MEAN, "sd": PESSIMISTIC_FINETUNE_SD,
            "provenance": (
                "low-central corner: EAGLE-1 logit-KD contra (V2) + capable-head saturation (V3) "
                "push the fine-tune central toward the bottom of [+0.009,+0.029]."),
        },
        "original_339": {"mean": RECIPE_MEAN_DCOV, "sd": RECIPE_SIGMA_INDEP,
                         "name": "modeled-339-optimistic"},
    }


# ===========================================================================
# Section 2 -- kappa-robustness sweep + breakeven
# ===========================================================================

def required_dcov(kappa: float, delta_et: float = DELTA_ET_CENTRAL_373) -> float:
    """Nominal coverage lift to close residual delta_et at transfer efficiency kappa.

    S_realized(kappa) = kappa * S_uniform_central; required_dcov = delta_et / S_realized.
    A 1/kappa curve: lower transfer -> larger required coverage lift.
    """
    return delta_et / (kappa * S_UNIFORM_CENTRAL)


def kappa_breakeven(budget: float = COV_BUDGET_336,
                    delta_et: float = DELTA_ET_CENTRAL_373) -> float:
    """The kappa at which required_dcov(kappa) == budget (the +0.0107 target exits #336's +0.031)."""
    # required_dcov(kappa) = delta_et/(kappa*S_uniform) = budget  ->  kappa = delta_et/(budget*S_uniform)
    return delta_et / (budget * S_UNIFORM_CENTRAL)


def norm_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1.0 + erf((x - mu) / (sigma * sqrt(2.0))))


def p_deliver(req: float, mu: float, sigma: float) -> float:
    """P(delivered nominal coverage lift >= required req) under N(mu, sigma)."""
    return 1.0 - norm_cdf(req, mu, sigma)


def build_kappa_sweep(defensible: dict) -> dict:
    """Sweep kappa over [floor, 1.0]; locate breakeven; report margin and p_deliver at breakeven."""
    kbreak = kappa_breakeven()
    kbreak_conservative = kappa_breakeven(delta_et=DELTA_ET_CONSERVATIVE_373)  # +17.96 TPS sensitivity
    kappa_margin = KAPPA_CENTRAL_377 - kbreak
    kappa_worst_corner = S_PROGRAM_WORST / S_UNIFORM_CENTRAL                   # 0.3541 -- robust target's kappa
    margin_from_worst_corner = kappa_worst_corner - kbreak

    # required_dcov at breakeven == budget (by construction); p_deliver of THAT under both dists.
    req_at_break = required_dcov(kbreak)
    p_break_optimistic = p_deliver(req_at_break, RECIPE_MEAN_DCOV, RECIPE_SIGMA_INDEP)
    p_break_defensible = p_deliver(req_at_break, defensible["mean"], defensible["sd"])

    # the sweep grid (floor below breakeven to show the crossing; plausible floor = worst corner).
    grid = [0.10, 0.122214, 0.15, 0.20, 0.25, 0.3541, 0.40, 0.50, 0.6716, 0.80, 1.00]
    sweep = []
    for k in grid:
        req = required_dcov(k)
        sweep.append({
            "kappa": k,
            "required_dcov": req,
            "within_336_budget": req <= COV_BUDGET_336,
            "frac_of_336_budget": req / COV_BUDGET_336,
            "p_deliver_optimistic_339": p_deliver(req, RECIPE_MEAN_DCOV, RECIPE_SIGMA_INDEP),
            "p_deliver_defensible": p_deliver(req, defensible["mean"], defensible["sd"]),
        })
    return {
        "kappa_central_377": KAPPA_CENTRAL_377,
        "kappa_worst_corner_robust_target": kappa_worst_corner,
        "kappa_floor_plausible": kappa_worst_corner,   # the program's own worst c* corner
        "kappa_breakeven": kbreak,
        "kappa_breakeven_conservative_residual": kbreak_conservative,
        "kappa_margin": kappa_margin,                  # PR def: 0.672 - kappa_breakeven
        "kappa_margin_from_worst_corner": margin_from_worst_corner,
        "required_dcov_at_breakeven": req_at_break,
        "p_deliver_at_kappa_breakeven": p_break_optimistic,        # under #339 (matches #377 basis)
        "p_deliver_at_kappa_breakeven_defensible": p_break_defensible,
        "breakeven_below_plausible_floor": kbreak < kappa_worst_corner,
        "central_kappa_comfortably_above_breakeven": kappa_margin > 0.05,
        "sweep": sweep,
        "note": (
            "required_dcov(kappa) = dE[T]/(kappa*S_uniform) is a 1/kappa curve. The +0.0107 robust "
            "target = required_dcov(kappa_worst=0.354); the +0.00565 central = required_dcov("
            "kappa_central=0.672). Breakeven (required_dcov == +0.031 budget) sits at kappa=0.122 -- "
            "BELOW even the worst c* corner 0.354, so the target stays within budget across the ENTIRE "
            "plausible kappa range. Margin vs central 0.672 is 0.55; vs the worst corner 0.354 is 0.23."),
    }


# ===========================================================================
# Section 3 -- deliverability under BOTH the optimistic and the defensible distributions
# ===========================================================================

def build_deliverability(defensible: dict) -> dict:
    """P(retrain delivers) for the robust (+0.0107) and central (+0.00565) targets across the FULL
    delivery spectrum: optimistic #339 -> from-scratch ceiling -> defensible fine-tune -> pessimistic.

    Decision split: the CENTRAL target is small enough to survive every distribution; the ROBUST
    target (the one fern #357 wants to bank as c>=0.9010) degrades below the 0.90 confidence bar once
    the delivery mean drops to the defensible FINE-TUNE level, because #339's +0.0385 was optimistic.
    """
    targets = {"robust": DCOV_ROBUST_377, "central": DCOV_CENTRAL_377}
    ceiling = defensible["ceiling_from_scratch"]
    pess = defensible["pessimistic"]

    def deliver_under(mu: float, sigma: float) -> dict:
        return {k: p_deliver(req, mu, sigma) for k, req in targets.items()}

    p_optimistic = deliver_under(RECIPE_MEAN_DCOV, RECIPE_SIGMA_INDEP)        # #339 / #377 basis
    p_optimistic_comono = deliver_under(RECIPE_MEAN_DCOV, RECIPE_SIGMA_COMONO)  # #339 mean, realistic spread
    p_ceiling = deliver_under(ceiling["mean"], ceiling["sd"])                 # from-scratch ceiling +0.0227
    p_defensible = deliver_under(defensible["mean"], defensible["sd"])        # FINE-TUNE realistic +0.016
    p_pessimistic = deliver_under(pess["mean"], pess["sd"])                   # low-central +0.012

    # Per-target survival bars. CONFIDENCE bar = 0.90 (the #339/#377 'P(deliver)~1.0' confidence claim).
    central_survives = p_defensible["central"] > 0.90 and p_pessimistic["central"] > 0.85
    robust_survives = p_defensible["robust"] > 0.90                          # FAILS: ~0.81 < 0.90
    return {
        "targets": targets,
        "confidence_bar": 0.90,
        "spectrum": {
            "optimistic_339_indep": {"mean": RECIPE_MEAN_DCOV, "sd": RECIPE_SIGMA_INDEP, "p": p_optimistic},
            "optimistic_339_comono": {"mean": RECIPE_MEAN_DCOV, "sd": RECIPE_SIGMA_COMONO,
                                      "p": p_optimistic_comono},
            "from_scratch_ceiling": {"mean": ceiling["mean"], "sd": ceiling["sd"], "p": p_ceiling},
            "defensible_fine_tune": {"mean": defensible["mean"], "sd": defensible["sd"], "p": p_defensible},
            "pessimistic_low_central": {"mean": pess["mean"], "sd": pess["sd"], "p": p_pessimistic},
        },
        "p_optimistic_339": p_optimistic,
        "p_defensible_conservative": p_defensible,
        # headline required fields (under the DEFENSIBLE fine-tune distribution):
        "p_softkd_reasoning_retrain_delivers_robust": p_defensible["robust"],
        "p_softkd_reasoning_retrain_delivers_central": p_defensible["central"],
        "p_softkd_reasoning_retrain_delivers_robust_optimistic_339": p_optimistic["robust"],
        "p_softkd_reasoning_retrain_delivers_central_optimistic_339": p_optimistic["central"],
        "p_softkd_reasoning_retrain_delivers_robust_ceiling": p_ceiling["robust"],
        "central_target_survives_conservative": bool(central_survives),
        "robust_target_survives_conservative": bool(robust_survives),
        # The headline bool tracks the ROBUST target -- the one fern #357 banks (c>=0.9010).
        "deliverability_survives_conservative": bool(robust_survives),
        "note": (
            "CENTRAL +0.00565 target: P(deliver) stays >=0.90 across the whole spectrum down to the "
            "pessimistic corner -> ROBUSTLY deliverable. ROBUST +0.0107 target: P falls from ~1.0 "
            "(#339 optimistic) -> ~0.91 (from-scratch ceiling) -> ~0.81 (defensible FINE-TUNE) -> "
            "~0.60 (pessimistic) -> it does NOT clear the 0.90 confidence bar once delivery is sized "
            "at the defensible fine-tune level, because #339's +0.0385 was optimistic (V1/V2/V3). The "
            "robust target's deliverability is the BINDING residual uncertainty."),
    }


# ===========================================================================
# Section 4 -- decisive verdict
# ===========================================================================

def build_verdict(grounding: dict, ksweep: dict, deliver: dict) -> dict:
    grounding_ok = grounding["delivery_distribution_grounding"] in (
        "literature-anchored", "measured-checkpoint")
    margin_ok = ksweep["kappa_margin"] > 0.05 and ksweep["breakeven_below_plausible_floor"]
    robust_survives = deliver["robust_target_survives_conservative"]
    central_survives = deliver["central_target_survives_conservative"]

    # recipe_is_real: the LEVERS are literature-documented (NOT a fiction) and a positive coverage
    # lift IS empirically expected (the from-scratch ceiling +0.0227 is a real anchor) -> True. But
    # "real" here = literature-anchored prior, NOT a measured checkpoint, AND NOT the +0.0385
    # magnitude (which V2/EAGLE-1 contradicts). The recipe delivers SOMETHING positive; whether it
    # delivers the ROBUST +0.0107 at >=0.90 confidence is the open question.
    recipe_is_real = bool(grounding_ok)  # not a fiction; levers cited, ceiling anchor real
    # GREEN requires the ROBUST target (the banked sizing) to survive conservative delivery.
    green = bool(grounding_ok and margin_ok and robust_survives)
    # YELLOW: kappa-axis robust + central target safe, but robust target's delivery is marginal.
    yellow = bool(grounding_ok and margin_ok and central_survives and not robust_survives)

    if green:
        verdict = "GREEN"
        next_step = (
            "route de-risked end-to-end; fern #357 may bank c>=0.9010. Optional: #352 retrain to "
            "convert literature-anchored -> measured.")
    elif yellow:
        verdict = "YELLOW"
        next_step = (
            "SPLIT BANK. kappa-transfer axis is ROBUST (breakeven 0.122 << worst-corner 0.354) and "
            "the CENTRAL +0.00565 target (c>=0.8959) is deliverable at >=0.90 confidence across the "
            "whole literature-grounded delivery spectrum -> fern #357 can bank the CENTRAL target "
            "now. The ROBUST +0.0107 target (c>=0.9010) is the BINDING OPEN ITEM: its deliverability "
            "rests on #339's +0.0385 delivery distribution, which the PR #380 pass shows is OPTIMISTIC "
            "(V1 no paper measures top-4 coverage; V2 EAGLE-1 found logit-KD UNDERPERFORMS CE; V3 low "
            "data-sensitivity + diminishing returns) -- under a defensible FINE-TUNE delivery "
            "(central +0.016) P(deliver robust) ~= 0.81 < 0.90. CHEAPEST REAL PROOF before banking "
            "the robust target: (1) kanna #294 Phase-1 a2>=0.83 cheap pre-check (de-risks the prior, "
            "no retrain); then (2) the #352-priced ~25 A10G-GPU-hr (~3 h on the 8x node) soft-KD + "
            "reasoning-trace FINE-TUNE + wirbel #79 RANKPROBE_W=4 coverage re-measure on the OFFICIAL "
            "128 eval -> a DIRECT top-4-coverage-lift measurement (the metric no paper reports). NOTE: "
            "the optional local-GPU kappa-probe is the WRONG tool -- kappa is robust; the weak link is "
            "DELIVERY, which needs a coverage-lift pilot, not a transfer measurement.")
    else:
        verdict = "RED"
        next_step = (
            "deliverability unproven -- BINDING OPEN ITEM. Cheapest real-retrain proof: the #352 "
            "~25 A10G-GPU-hr soft-KD + reasoning-trace retrain + #79 RANKPROBE coverage re-measure "
            "before fern #357 banks the target.")
    return {
        "recipe_is_real": recipe_is_real,
        "recipe_is_real_meaning": (
            "TRUE = not a fiction: the levers (soft-KD, reasoning-data) are documented and a positive "
            "lift is empirically expected (the +0.0227 from-scratch ceiling is a real anchor). NOT a "
            "measured checkpoint, and NOT the +0.0385 magnitude (V2/EAGLE-1 contradicts the soft-KD "
            "component); the defensible FINE-TUNE central is ~+0.016."),
        "deliverability_survives_conservative": robust_survives,
        "central_target_survives_conservative": central_survives,
        "robust_target_survives_conservative": robust_survives,
        "delivery_distribution_grounding": grounding["delivery_distribution_grounding"],
        "kappa_breakeven": ksweep["kappa_breakeven"],
        "kappa_margin": ksweep["kappa_margin"],
        "green_bar_met": green,
        "yellow": yellow,
        "verdict": verdict,
        "recommended_next_step": next_step,
        "demand_route_derisked": green,
        "demand_route_derisked_central_only": bool(yellow or green),
    }


# ===========================================================================
# Section 5 -- OPTIONAL local-GPU kappa-probe (gated; NOT run by default)
# ===========================================================================

def measure_kappa_gpu(proxy: str, eval_prompts: int) -> dict:
    """Identity-safe viability kappa-probe on the int4-ct proxy drafter (gated behind --gpu).

    NOT run by default: the analytic kappa is WELL-BOUNDED (breakeven 0.122 < worst corner 0.354 <
    central 0.672; margin 0.55). Per the PR, the GPU leg is conditional on kappa being
    under-determined -- it is not. Stub provided for completeness; running it would do a short
    soft-KD fine-tune step on the proxy over a held-out slice and measure realized coverage lift per
    unit KD signal -> measured_kappa. Drafter-only; spec emission = verify argmax = byte-exact, so
    identity is untouched regardless of drafter quality. No submission, no served-file change."""
    try:
        import os
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return {"ran": False, "reason": f"gpu/torch unavailable: {exc}"}
    return {"ran": False, "reason": "stub: analytic kappa well-bounded (breakeven 0.122 << central "
            "0.672); GPU kappa-probe not required. Set up the proxy drafter + held-out slice to run."}


# ===========================================================================
# Section 6 -- self-tests
# ===========================================================================

def run_self_tests(grounding: dict, defensible: dict, ksweep: dict,
                   deliver: dict, verdict: dict) -> dict:
    c: dict[str, bool] = {}
    # a) kappa identity: recompute #377's kappa from S_program_central / S_uniform.
    c["a_kappa_central_roundtrips_377"] = abs(
        S_PROGRAM_CENTRAL / S_UNIFORM_CENTRAL - KAPPA_CENTRAL_377) < 1e-9
    # a2) required_dcov reproduces #377's robust and central targets at the worst/central kappas.
    c["a2_required_dcov_central_roundtrips"] = abs(
        required_dcov(KAPPA_CENTRAL_377) - DCOV_CENTRAL_377) < 1e-9
    c["a2_required_dcov_robust_roundtrips"] = abs(
        required_dcov(S_PROGRAM_WORST / S_UNIFORM_CENTRAL) - DCOV_ROBUST_377) < 1e-9
    # b) grounding: enum literature-anchored (real ceiling anchor) but #339 flagged optimistic.
    c["b_grounding_is_literature_anchored"] = (
        grounding["delivery_distribution_grounding"] == "literature-anchored")
    c["b_not_measured_not_modeled_only"] = (
        not grounding["is_measured_checkpoint"]) and (not grounding["is_modeled_only"])
    c["b_339_flagged_optimistic"] = grounding["original_339_distribution_optimistic"]
    c["b_mean_roundtrips_336"] = grounding["mean_roundtrips"]
    c["b_sigma_indep_roundtrips_339"] = grounding["sigma_indep_roundtrips"]
    # b2) #339's +0.0385 lands in the TOP quartile of the published [0.899,0.929] range (optimistic).
    c["b2_339_near_top_of_lit_range"] = grounding["cov_post_339_in_lit_range_frac"] > 0.75
    c["b2_verification_findings_carried"] = len(grounding["verification_findings"]) >= 3
    # c) defensible FINE-TUNE distribution is LOWER than #336 AND lower than the from-scratch ceiling.
    c["c_defensible_mean_below_336"] = defensible["mean"] < RECIPE_MEAN_DCOV
    c["c_defensible_below_ceiling"] = defensible["mean"] < defensible["ceiling_from_scratch"]["mean"]
    c["c_defensible_mean_positive"] = defensible["mean"] > 0.0
    c["c_defensible_in_finetune_range"] = (
        LIT_FINETUNE_RANGE[0] <= defensible["mean"] <= LIT_FINETUNE_RANGE[1])
    # d) kappa breakeven: below the plausible floor (worst corner) and margin comfortably positive.
    c["d_breakeven_below_worst_corner"] = ksweep["kappa_breakeven"] < ksweep[
        "kappa_worst_corner_robust_target"]
    c["d_kappa_margin_positive"] = ksweep["kappa_margin"] > 0.05
    c["d_breakeven_req_equals_budget"] = abs(
        ksweep["required_dcov_at_breakeven"] - COV_BUDGET_336) < 1e-9
    c["d2_breakeven_below_red_062"] = ksweep["kappa_breakeven"] < 0.62
    # e) THE DECISIVE SPLIT: central target survives conservative delivery; robust target does NOT.
    c["e_central_survives_conservative"] = deliver["central_target_survives_conservative"]
    c["e_robust_does_not_survive_conservative"] = not deliver["robust_target_survives_conservative"]
    c["e_p_central_defensible_high"] = deliver["p_softkd_reasoning_retrain_delivers_central"] > 0.90
    c["e_p_robust_defensible_below_bar"] = deliver["p_softkd_reasoning_retrain_delivers_robust"] < 0.90
    # e2) the spectrum is monotone in delivery mean for the robust target (optimistic > ceiling >
    #     defensible > pessimistic).
    sp = deliver["spectrum"]
    robust_chain = [sp["optimistic_339_indep"]["p"]["robust"], sp["from_scratch_ceiling"]["p"]["robust"],
                    sp["defensible_fine_tune"]["p"]["robust"], sp["pessimistic_low_central"]["p"]["robust"]]
    c["e2_robust_spectrum_monotone"] = all(
        robust_chain[i] >= robust_chain[i + 1] - 1e-12 for i in range(len(robust_chain) - 1))
    # e3) central target stays >= 0.85 even at the pessimistic corner.
    c["e3_central_pessimistic_above_085"] = sp["pessimistic_low_central"]["p"]["central"] > 0.85
    # f) sweep monotonicity: required_dcov strictly decreasing in kappa; p_deliver increasing.
    reqs = [row["required_dcov"] for row in ksweep["sweep"]]
    c["f_required_dcov_monotone_decreasing"] = all(
        reqs[i] > reqs[i + 1] for i in range(len(reqs) - 1))
    pdef = [row["p_deliver_defensible"] for row in ksweep["sweep"]]
    c["f_p_deliver_monotone_increasing"] = all(
        pdef[i] <= pdef[i + 1] + 1e-12 for i in range(len(pdef) - 1))
    # g) within-budget crossing is exactly at breakeven (first within-budget row >= kappa_breakeven).
    first_within = next(row["kappa"] for row in ksweep["sweep"] if row["within_336_budget"])
    c["g_first_within_budget_at_or_above_breakeven"] = first_within >= ksweep["kappa_breakeven"] - 1e-9
    # h) verdict wiring: YELLOW (not GREEN, not RED) given the central/robust split.
    c["h_verdict_is_yellow"] = verdict["verdict"] == "YELLOW"
    c["h_green_not_met"] = not verdict["green_bar_met"]
    c["h_recipe_is_real_true"] = verdict["recipe_is_real"] is True
    c["h_next_step_names_proof"] = "#352" in verdict["recommended_next_step"]
    # i) budget identity: #336 budget == identity bar - prior.
    c["i_budget_is_bar_minus_prior"] = abs(
        COV_BUDGET_336 - (IDENTITY_BAR - COV_PRIOR)) < 1e-9
    # j) numeric hygiene.
    flat = [defensible["mean"], defensible["sd"], ksweep["kappa_breakeven"], ksweep["kappa_margin"],
            deliver["p_softkd_reasoning_retrain_delivers_robust"],
            deliver["p_softkd_reasoning_retrain_delivers_central"]]
    c["j_no_nan"] = all(v == v for v in flat)
    c["j_probs_in_unit_interval"] = all(
        0.0 <= sp[case]["p"][t] <= 1.0 for case in sp for t in ("robust", "central"))
    passes = all(c.values())
    return {"conditions": c, "n_checks": len(c), "passes": passes}


# ===========================================================================
# Section 7 -- report assembly + W&B + CLI
# ===========================================================================

def build_report(args) -> dict:
    grounding = classify_grounding()
    defensible = build_defensible_distribution()
    ksweep = build_kappa_sweep(defensible)
    deliver = build_deliverability(defensible)
    verdict = build_verdict(grounding, ksweep, deliver)
    gpu = {"ran": False, "reason": "CPU-analytic default; --gpu/--kappa-probe not set"}
    if getattr(args, "gpu", False) and getattr(args, "kappa_probe", False):
        gpu = measure_kappa_gpu(getattr(args, "proxy", "google/gemma-4-E4B-it-qat-w4a16-ct"),
                                getattr(args, "eval_prompts", 128))
    selftest = run_self_tests(grounding, defensible, ksweep, deliver, verdict)

    return {
        "pr": 380, "agent": "denken", "kind": "coverage-retrain-deliverability",
        "analysis_only": True, "no_launch": True, "no_hf_job": True,
        "no_served_file_change": True, "official_tps_expected": 0,
        "inputs": {
            "delta_et_central_373": DELTA_ET_CENTRAL_373,
            "delta_et_conservative_373": DELTA_ET_CONSERVATIVE_373,
            "s_uniform_central_377": S_UNIFORM_CENTRAL,
            "s_program_central_377": S_PROGRAM_CENTRAL,
            "s_program_worst_377": S_PROGRAM_WORST,
            "kappa_central_377": KAPPA_CENTRAL_377,
            "dcov_robust_377": DCOV_ROBUST_377, "dcov_central_377": DCOV_CENTRAL_377,
            "cov_prior_330": COV_PRIOR, "identity_bar": IDENTITY_BAR,
            "cov_budget_336": COV_BUDGET_336,
            "cstar_central_340": CSTAR_CENTRAL, "cstar_worst_340": CSTAR_WORST,
            "recipe_mean_dcov_339": RECIPE_MEAN_DCOV,
            "recipe_sigma_indep_339": RECIPE_SIGMA_INDEP,
            "recipe_sigma_rho05_339": RECIPE_SIGMA_RHO05,
            "recipe_sigma_comono_339": RECIPE_SIGMA_COMONO,
            "lit_top4_central_323": LIT_TOP4_CENTRAL, "lit_top4_range_323": list(LIT_TOP4_RANGE),
            "source_377": "030uc5mk", "source_336": "krroookz", "source_339": "0aq16szh",
            "source_323": "ceddxj20", "source_330": "hfrscdai",
        },
        "grounding_audit": grounding,
        "defensible_delivery_distribution": defensible,
        "kappa_sweep": ksweep,
        "deliverability": deliver,
        "verdict": verdict,
        "gpu_kappa_probe_leg": gpu,
        # ----- card-required headline scalars -----
        "delivery_distribution_grounding": grounding["delivery_distribution_grounding"],
        "kappa_breakeven": ksweep["kappa_breakeven"],
        "p_deliver_at_kappa_breakeven": ksweep["p_deliver_at_kappa_breakeven"],
        "kappa_margin": ksweep["kappa_margin"],
        "p_softkd_reasoning_retrain_delivers_robust": deliver[
            "p_softkd_reasoning_retrain_delivers_robust"],
        "p_softkd_reasoning_retrain_delivers_central": deliver[
            "p_softkd_reasoning_retrain_delivers_central"],
        "recipe_is_real": verdict["recipe_is_real"],
        "deliverability_survives_conservative": verdict["deliverability_survives_conservative"],
        "recommended_next_step": verdict["recommended_next_step"],
        "measured_kappa": gpu.get("measured_kappa") if gpu.get("ran") else None,
        # ----- GO/NO-GO + SENPAI-RESULT metrics -----
        "demand_route_derisked": verdict["demand_route_derisked"],
        "primary_metric_kappa_margin": ksweep["kappa_margin"],
        "self_test": selftest,
        "deliverability_self_test_passes": selftest["passes"],
    }


def print_report(report: dict) -> None:
    g, d, k, dl, v = (report["grounding_audit"], report["defensible_delivery_distribution"],
                      report["kappa_sweep"], report["deliverability"], report["verdict"])
    print("\n=== Coverage-retrain DELIVERABILITY grounding (PR #380, denken) ===")
    print(f"\n[1] delivery_distribution_grounding = {g['delivery_distribution_grounding'].upper()}")
    print(f"    measured-checkpoint? {g['is_measured_checkpoint']}  modeled-only? {g['is_modeled_only']}")
    print(f"    mean roundtrips #336 0.0385: {g['mean_roundtrips']}; sigma_indep roundtrips #339 "
          f"0.00742: {g['sigma_indep_roundtrips']}")
    print(f"    modeled layers on top: {len(g['modeled_layers_on_top'])} (top-4 attribution / 0.70 "
          f"haircut / optimistic rho=0 sigma)")
    print(f"\n[1b] defensible_delivery_distribution = N({d['mean']:.4f}, {d['sd']:.4f})  "
          f"band {[round(x,4) for x in d['band']]}")
    print(f"     ({d['name']}; LOWER+wider than #336's N(0.0385, 0.00742))")
    print(f"\n[2] kappa-robustness sweep:")
    print(f"    kappa_central (#377)            : {k['kappa_central_377']:.4f}")
    print(f"    kappa_worst_corner (robust tgt) : {k['kappa_worst_corner_robust_target']:.4f}  "
          f"(plausible floor)")
    print(f"    kappa_breakeven                 : {k['kappa_breakeven']:.4f}  "
          f"(required_dcov == +0.031 budget)")
    print(f"    kappa_margin (0.672 - breakeven): {k['kappa_margin']:.4f}")
    print(f"    breakeven below plausible floor : {k['breakeven_below_plausible_floor']}  "
          f"(0.122 < 0.354 worst corner)")
    print(f"    p_deliver_at_kappa_breakeven    : {k['p_deliver_at_kappa_breakeven']:.4f} (opt #339) / "
          f"{k['p_deliver_at_kappa_breakeven_defensible']:.4f} (defensible)")
    print(f"\n[3] P(retrain delivers) -- robust +0.0107 / central +0.00565:")
    sp = dl["spectrum"]
    print(f"    optimistic #339  N(0.0385,0.00742): {sp['optimistic_339_indep']['p']['robust']:.4f} / "
          f"{sp['optimistic_339_indep']['p']['central']:.4f}")
    print(f"    from-scratch ceiling N({sp['from_scratch_ceiling']['mean']:.4f},"
          f"{sp['from_scratch_ceiling']['sd']:.4f}): {sp['from_scratch_ceiling']['p']['robust']:.4f} / "
          f"{sp['from_scratch_ceiling']['p']['central']:.4f}")
    print(f"    DEFENSIBLE fine-tune N({d['mean']:.4f},{d['sd']:.4f}): "
          f"{sp['defensible_fine_tune']['p']['robust']:.4f} / "
          f"{sp['defensible_fine_tune']['p']['central']:.4f}")
    print(f"    pessimistic     N({sp['pessimistic_low_central']['mean']:.4f},"
          f"{sp['pessimistic_low_central']['sd']:.4f}): {sp['pessimistic_low_central']['p']['robust']:.4f} / "
          f"{sp['pessimistic_low_central']['p']['central']:.4f}")
    print(f"    central_target_survives_conservative = {dl['central_target_survives_conservative']}")
    print(f"    robust_target_survives_conservative  = {dl['robust_target_survives_conservative']}")
    print(f"\n[5] VERDICT = {v['verdict']}")
    print(f"    recipe_is_real                       = {v['recipe_is_real']}")
    print(f"    deliverability_survives_conservative = {v['deliverability_survives_conservative']}")
    print(f"    green_bar_met                        = {v['green_bar_met']}")
    print(f"    recommended_next_step: {v['recommended_next_step'][:160]}...")
    print(f"\n>>> deliverability_self_test_passes = {report['self_test']['passes']} "
          f"({report['self_test']['n_checks']} checks)")


def log_to_wandb(report: dict, group: str, name: str) -> str:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_unavailable"
    try:
        run = wandb.init(group=group, name=name, config=report["inputs"])
        g, d, k, dl, v = (report["grounding_audit"], report["defensible_delivery_distribution"],
                          report["kappa_sweep"], report["deliverability"], report["verdict"])
        grounding_rank = {"modeled-only": 0, "literature-anchored": 1, "measured-checkpoint": 2}
        wandb.log({
            "summary/delivery_grounding_rank": grounding_rank[g["delivery_distribution_grounding"]],
            "summary/defensible_delivery_mean": d["mean"],
            "summary/defensible_delivery_sd": d["sd"],
            "summary/kappa_central_377": k["kappa_central_377"],
            "summary/kappa_worst_corner": k["kappa_worst_corner_robust_target"],
            "summary/kappa_breakeven": k["kappa_breakeven"],
            "summary/kappa_breakeven_conservative_residual": k["kappa_breakeven_conservative_residual"],
            "summary/kappa_margin": k["kappa_margin"],
            "summary/kappa_margin_from_worst_corner": k["kappa_margin_from_worst_corner"],
            "summary/p_deliver_at_kappa_breakeven": k["p_deliver_at_kappa_breakeven"],
            "summary/p_deliver_at_kappa_breakeven_defensible": k["p_deliver_at_kappa_breakeven_defensible"],
            "summary/p_deliver_robust_optimistic": dl["p_optimistic_339"]["robust"],
            "summary/p_deliver_central_optimistic": dl["p_optimistic_339"]["central"],
            "summary/p_deliver_robust_defensible": dl["p_softkd_reasoning_retrain_delivers_robust"],
            "summary/p_deliver_central_defensible": dl["p_softkd_reasoning_retrain_delivers_central"],
            "summary/p_deliver_robust_comonotonic": dl["spectrum"]["optimistic_339_comono"]["p"]["robust"],
            "summary/p_deliver_robust_ceiling": dl["spectrum"]["from_scratch_ceiling"]["p"]["robust"],
            "summary/p_deliver_robust_pessimistic": dl["spectrum"]["pessimistic_low_central"]["p"]["robust"],
            "summary/p_deliver_central_pessimistic": dl["spectrum"]["pessimistic_low_central"]["p"]["central"],
            "summary/central_target_survives_conservative": float(
                dl["central_target_survives_conservative"]),
            "summary/robust_target_survives_conservative": float(
                dl["robust_target_survives_conservative"]),
            "summary/deliverability_survives_conservative": float(
                dl["deliverability_survives_conservative"]),
            "summary/recipe_is_real": float(v["recipe_is_real"]),
            "summary/green_bar_met": float(v["green_bar_met"]),
            "summary/verdict_is_yellow": float(v["yellow"]),
            "summary/demand_route_derisked": float(v["demand_route_derisked"]),
            "summary/demand_route_derisked_central_only": float(v["demand_route_derisked_central_only"]),
            "summary/deliverability_self_test_passes": float(report["self_test"]["passes"]),
        })
        for row in k["sweep"]:
            wandb.log({
                "sweep/kappa": row["kappa"],
                "sweep/required_dcov": row["required_dcov"],
                "sweep/frac_of_336_budget": row["frac_of_336_budget"],
                "sweep/p_deliver_optimistic": row["p_deliver_optimistic_339"],
                "sweep/p_deliver_defensible": row["p_deliver_defensible"],
                "sweep/within_336_budget": float(row["within_336_budget"]),
            })
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_failed"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Coverage-retrain deliverability grounding for the demand-side route (PR #380).")
    ap.add_argument("--ground-delivery", action="store_true",
                    help="(default behavior) audit + re-ground the delivery distribution")
    ap.add_argument("--kappa-sweep", action="store_true",
                    help="(default) sweep kappa and locate breakeven")
    ap.add_argument("--gpu", action="store_true", help="OPTIONAL: enable the local-GPU kappa-probe leg")
    ap.add_argument("--kappa-probe", action="store_true", help="OPTIONAL: run the kappa-probe (needs --gpu)")
    ap.add_argument("--proxy", type=str, default="google/gemma-4-E4B-it-qat-w4a16-ct")
    ap.add_argument("--eval-prompts", type=int, default=128)
    ap.add_argument("--self-test", action="store_true", help="run self-test only (0-GPU, no W&B)")
    ap.add_argument("--reanalyze", action="store_true",
                    help="re-print the verdict from the banked JSON (0-GPU, no recompute)")
    ap.add_argument("--wandb_group", type=str, default="strict-bi-verify-gemm")
    ap.add_argument("--wandb_name", type=str, default="denken/coverage-deliverability")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if args.reanalyze:
        path = Path(args.out)
        if not path.exists():
            print(f"--reanalyze: no banked JSON at {path}", file=sys.stderr)
            return 1
        report = json.loads(path.read_text())
        print_report(report)
        print(f"\n[reanalyze] banked W&B run {report.get('wandb_run_id')}; "
              f"self_test_passes={report['self_test']['passes']}")
        return 0 if report["self_test"]["passes"] else 1

    report = build_report(args)
    print_report(report)

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
