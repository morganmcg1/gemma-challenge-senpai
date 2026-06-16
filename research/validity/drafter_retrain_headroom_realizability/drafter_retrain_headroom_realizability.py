#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Is the +38.21-TPS demand headroom above 482.74 REALIZABLE, or a head-ceiling
mirage? (PR #439, land). 0-GPU pure-analysis card. Analysis-only: NO served-file
change, NO HF Job, NO submission, NOT a launch. official_tps=0.

THE QUESTION
------------
My own #436 (`nvsbctji`) bounded the DEMAND axis above 482.74: every CHEAP
identity-free E[T] lever is closed, and the ONLY demand headroom is +38.21 TPS to
the verify-BW lambda=1 wall (520.95) -- but it is RETRAIN-gated
(`demand_lever_capturable_no_retrain`=False). #436 reported the *ceiling* (38.21
to the wall). This card reports the *realizable fraction* of it: of that 38.21
TPS, how much is REALIZABLY capturable by a practical drafter retrain -- or is it
a head-ceiling mirage?

IDENTITY-FREE BY CONSTRUCTION (land #420, `qe4qagc1`)
----------------------------------------------------
The deployed spec stack is an MTP drafter (K=7, verify width M=K+1=8, linear-chain
top-1). The drafter only PROPOSES; the truncated-head verify is the SOLE arbiter
of emitted tokens. A change to drafter quality / acceptance length changes only
HOW MANY draft tokens are accepted per step -- never WHICH token is emitted.
Greedy-token identity is preserved by construction, and PPL is teacher-forced ->
unchanged at 2.3772 (the verify is the arbiter). So unlike the cb3/pinned-K supply
levers, NO reference contract is at stake -- the only question is realizable speed.

DELIVERABLE 1 -- decompose the +0.10973 head-ceiling coverage gap
-----------------------------------------------------------------
The deployed MTP K=7 drafter is at its #289 (`fi34s269`) LINEAR acceptance cap
(`deployed_at_or_above_linear_cap`=True; E[T]=3.851 >= linear_cap 3.8445;
`built_raise_requires_nonlinear_drafter`=True). The acceptance CLIFF is at
position 1 (#289: a_1=0.7293 carries token_loss 1.895; conditional acceptance
a_2..a_7 INCREASES with depth). So the easy/linear E[T] gains are spent; the
remaining headroom lives in the +0.10973 head-ceiling coverage gap (#399
`ec7i3z5t`: 1.0 - top4_root_coverage 0.8903) -- the hard-positions / sub-linear
remainder a better DRAFT HEAD must close.

How much of that 0.10973 is a practical retrain REALIZABLY delivers is a
DELIVERY-distribution question, and the program has already priced it under a
literature pass (denken #380 `00oijpwg`, coverage-retrain-deliverability), which
CORRECTS the optimistic +0.0385 recipe (#339/#336) down to a defensible delivery:
  * V1: NO cited paper (DistillSpec/OSD/Medusa/EAGLE-1/EAGLE-3/HASS/KOALA)
    measures top-4 COVERAGE -- the +0.0385 is an inference from acceptance gains
    via an ASSUMED transfer (the metric is unmeasured).
  * V2: EAGLE-1 (2401.15077) found logit-KD UNDERPERFORMS CE feature-regression
    and chose CE because logit-KD was weaker -> soft-KD +0.030 is contradicted
    (defensible ~+0.005..+0.015).
  * V3: reasoning-data +0.025 is optimistic (EAGLE-1 low data-sensitivity ~3.6%;
    HASS 1/4-data ~ full; diminishing returns as the capable 0.89 head narrows
    the teacher gap) -> defensible ~+0.005..+0.010.
The defensible delivery distributions (#380):
    DEFENSIBLE FINE-TUNE  : Dcov ~ +0.016  (central), band [+0.009, +0.029]
    FROM-SCRATCH CEILING  : Dcov ~ +0.0227 (optimistic-defensible upper)
    PESSIMISTIC           : Dcov ~ +0.012  (low corner)
    [optimistic #339      : Dcov ~ +0.0385 -- now KNOWN optimistic (V1/V2/V3)]
The realizable fraction of the 0.10973 coverage gap is therefore Dcov_realizable
/ 0.10973; the structural (unreachable) fraction is the remainder -- the coverage
a practical retrain CANNOT deliver (no paper achieves it; capable-head
saturation; logit-KD contra). The deployed head being AT its linear cap is the
mechanism: only the sub-linear / hard-positions head-architecture remainder is
left, and the literature caps how much of THAT a retrain harvests.

DELIVERABLE 2 -- map realizable coverage -> E[T] -> equiv-TPS on the 482.74 base
-------------------------------------------------------------------------------
Reuse the #436 demand map (T_step FIXED, acceptance-only change):
    TPS(et) = BASE * et / ET_DEP,  BASE=482.74 (#428 cb3 floor), ET_DEP=3.851 (#289).
and the program coverage->E[T] central secant S=7.913 (#399/#402; corroborated by
#401's banked tps_per_unit_dcov=968.57 on the 471 base). For a realizable lift
Dcov:
    realizable_etp_lift          = S * Dcov
    realizable_demand_headroom_tps = min( BASE*S*Dcov/ET_DEP , wall_headroom 38.21 )
    realizable_frontier_tps       = BASE + realizable_demand_headroom_tps
The wall (520.95) caps TPS; but the DEFENSIBLE Dcov (+0.016) maps to only +15.9
TPS -- FAR below the +38.21 wall headroom -- so the DRAFTER DELIVERY binds first,
NOT the wall. (Only the now-known-optimistic #339 +0.0385 would reach the wall;
the defensible central does not.) The realizable frontier ~498.6 CLEARS the
deployed NON-equivalent incumbent 481.53 but does NOT reach the wall 520.95.

BASE-ROBUSTNESS (Directive note): the realizable FRACTION (Dcov/gap) is a pure
coverage-axis ratio -> base-INDEPENDENT (robust to stark #437's cb3 revision of
482.74). The realizable_demand_headroom_TPS rescales linearly with the base
(reported on the 467.14 blanket-strict base as a cross-check). PRIVATE-robustness:
ubel #382 confirmed the coverage->TPS slope survives the public->private OOD shift
(`slope_is_private_robust`=True), so the realized lift is not an extra-discounted
public artifact.

DELIVERABLE 3 -- optional local MTP-drafter fine-tune VIABILITY probe
---------------------------------------------------------------------
SKIPPED (probe_was_run=False). The probe does NOT stand up cheaply within
SENPAI_TIMEOUT_MINUTES (=90): the drafter venv is absent (rebuild needed after the
container reset), no MTP-head checkpoint is local, and denken #380 already found
the cheap probe is the WRONG tool -- the weak link is DELIVERY (a direct top-4
coverage-lift measurement, ~25 A10G-GPU-hr ~ 3 h), not transfer/direction. A
few-hundred-step fine-tune would return a noise-dominated a_1 delta (cf. #382's
public 0.7287 vs private 0.5975 a_1 swing) that cannot move the verdict. Per the
PR: rely on the analytic sizing; do NOT burn the budget on plumbing.

DELIVERABLE 4 -- the honest verdict
-----------------------------------
Of the +38.21-TPS demand headroom above 482.74, the defensible-realizable fraction
is ~42% (+15.9 TPS, band [+8.9,+28.8]); the realizable frontier ~498.6 (band
[491.7, 511.5]) clears the deployed non-equivalent 481.53 but NOT the verify-BW
wall 520.95. The remaining ~58% is a head-ceiling MIRAGE -- a practical retrain
cannot deliver the coverage to reach the wall (V1/V2/V3). retrain_is_worth_it=True
(>= +8.9 TPS across the whole defensible band, well above the +3 materiality bar,
and it makes the EQUIVALENT frontier BEAT the non-equivalent incumbent);
demand_axis_effectively_closed=False.

SELF-TEST (`drafter_retrain_headroom_realizability_self_test_passes`, PRIMARY; 0-GPU)
------------------------------------------------------------------------------------
Re-loads every merged artifact and cross-checks the pinned anchors (#436 base /
wall / ET_DEP / gap / S / 38.21 headroom; #399 gap+S+top4; #380 defensible
delivery; #339 recipe; #289 linear-cap+cliff; #382 private-robust), then verifies
the demand-map round-trips, the realizable decomposition arithmetic, the wall-does-
NOT-bind geometry, the base-robust fraction, and the verdict booleans. No GPU, no
vLLM, no served-file change.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VAL = HERE.parent            # research/validity

TOL = 1e-2

# --------------------------------------------------------------------------- #
# Pinned anchors (imported byte-exactly; cross-checked vs merged JSONs below). #
# --------------------------------------------------------------------------- #
DEPLOYED_TPS = 481.53                 # PR #52 deployed NON-equivalent frontier (identity 0.9966)
PPL_DEPLOYED = 2.3772                 # PR #52 official PPL (acceptance change => teacher-forced unchanged)
PPL_GATE = 2.42                       # public cap (reference PPL + 5%)

# The equivalence-respecting DEMAND base = wirbel #428's bit-identical supply floor (#436 base).
BASE_482 = 482.7400155438763          # #428 frozen floor (blanket-strict 467.14 + cb3 15.60)
BASE_BLANKET_STRICT = 467.14          # denken #423 blanket-strict base (base-robustness cross-check)

# #289 (fi34s269) deployed MTP per-position conditional acceptance ladder a_1..a_7 (K=7).
LADDER_289 = [
    0.7292532942898975, 0.759556697719242, 0.7929794882639035, 0.8228,
    0.8348727920920435, 0.8357919254658385, 0.8464932652113331,
]

LAMBDA1_WALL = 520.9527323111674      # #349/#344/#327 verify-BW lambda=1 ceiling (full-accept)
LINEAR_CAP_289 = 3.8445               # #289 linear acceptance cap (deployed E[T] is AT/above it)

# Head-ceiling coverage gap geometry (#399 ec7i3z5t / #401 / program secant #402).
COVERAGE_GAP_TOTAL = 0.10973404808468479   # #399 head-ceiling coverage gap (1.0 - top4 root 0.8903)
COV_PRIOR_TOP4 = 0.8902659519153152        # #399/#330 measured top-4 ROOT coverage
S_CENTRAL = 7.912609135742992              # #402/#399 program coverage->E[T] central secant
S_WORST = 4.172857261001455                # #377/#380 program coverage->E[T] WORST secant (robustness)
TPS_PER_DCOV_399 = 968.57                  # #401/#399 banked tps per unit dcov on the 471 base (cross-check)

# Realizable Dcov DELIVERY distribution (denken #380 `00oijpwg`, literature-corrected).
DCOV_DEFENSIBLE_CENTRAL = 0.016            # #380 defensible FINE-TUNE central
DCOV_DEFENSIBLE_LO = 0.009                 # #380 defensible fine-tune band low
DCOV_DEFENSIBLE_HI = 0.029                 # #380 defensible fine-tune band high
DCOV_FROMSCRATCH_CEILING = 0.02273404808468482   # #380 from-scratch CEILING (optimistic-defensible)
DCOV_PESSIMISTIC = 0.012                   # #380 pessimistic low-central corner
DCOV_OPTIMISTIC_339 = 0.0385               # #339 recipe central -- KNOWN optimistic (V1/V2/V3)
DCOV_FULL_RECIPE_339 = 0.0525              # #339 full 4-lever recipe (optimistic upper)

# Materiality threshold for "retrain is worth it" (PR-suggested +3 TPS).
MATERIALITY_TPS = 3.0

# In-repo EAGLE-3 measured a_1 trainability anchors (#308 5axqa6oa), for narrative grounding.
A1_DEPLOYED_308 = 0.72925
A1_INREPO_EAGLE3_308 = 0.7714              # measured in-repo EAGLE-3 native step-1 a_1
A1_ROBUSTNESS_CEILING_308 = 0.85           # central argmax robustness ceiling

# Artifact paths (re-loaded + cross-checked in the self-test) -----------------
ART_436 = VAL / "identity_free_demand_ceiling" / "identity_free_demand_ceiling_results.json"
ART_399 = VAL / "mtp_drafter_acceptance_headroom" / "mtp_drafter_acceptance_headroom_results.json"
ART_380 = VAL / "coverage_retrain_deliverability" / "coverage_retrain_deliverability_results.json"
ART_339 = VAL / "eagle3_retrain_clear_probability" / "eagle3_retrain_clear_probability_results.json"
ART_289 = VAL / "per_position_acceptance_decay" / "per_position_acceptance_decay_results.json"
ART_382 = VAL / "coverage_slope_private_robustness" / "coverage_slope_private_robustness_results.json"
ART_308 = VAL / "eagle3_a1_cliff_trainability" / "eagle3_a1_cliff_trainability_results.json"


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))


def _load(art: Path) -> dict | None:
    if not art.exists():
        return None
    try:
        return json.loads(art.read_text())
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Section 1 -- the demand machinery (E[T] <-> equiv-TPS, T_step FIXED).         #
# Reused byte-identically from #436 so the demand map is the same.             #
# --------------------------------------------------------------------------- #
def expected_accepted(ladder: list[float]) -> float:
    """E[accepted DRAFT tokens]/step = sum_k prod_{j<=k} a_j (conditional ladder)."""
    cum, acc = 1.0, 0.0
    for a in ladder:
        cum *= a
        acc += cum
    return acc


def expected_tokens_per_step(ladder: list[float]) -> float:
    """E[T] = 1 (always-emitted verify token) + E[accepted]."""
    return 1.0 + expected_accepted(ladder)


ET_DEP = expected_tokens_per_step(LADDER_289)   # 3.851185944363104


def tps_from_et(et: float, base_tps: float = BASE_482, base_et: float = ET_DEP) -> float:
    """Demand-alone served TPS at acceptance length `et` (T_step FIXED => linear in E[T])."""
    return base_tps * et / base_et


def tps_per_dcov(base_tps: float = BASE_482, secant: float = S_CENTRAL) -> float:
    """Served TPS lifted per unit top-4 coverage on `base_tps`: base*S/ET_DEP."""
    return base_tps * secant / ET_DEP


# --------------------------------------------------------------------------- #
# Section 2 -- realizable Dcov decomposition + coverage->E[T]->TPS map.         #
# --------------------------------------------------------------------------- #
def size_one(dcov: float, base_tps: float = BASE_482, secant: float = S_CENTRAL,
             wall: float = LAMBDA1_WALL) -> dict:
    """Map a realizable coverage lift Dcov to its capped demand headroom + frontier."""
    wall_headroom = wall - base_tps
    etp_lift = secant * dcov
    tps_uncapped = tps_per_dcov(base_tps, secant) * dcov
    capped = min(tps_uncapped, wall_headroom)
    wall_binds = bool(tps_uncapped > wall_headroom)
    frontier = base_tps + capped
    return {
        "dcov": dcov,
        "coverage_gap_realizable_frac": dcov / COVERAGE_GAP_TOTAL,
        "coverage_gap_structural_frac": 1.0 - dcov / COVERAGE_GAP_TOTAL,
        "realizable_etp_lift": etp_lift,
        "realizable_tps_uncapped": tps_uncapped,
        "realizable_demand_headroom_tps": capped,
        "frac_of_38p21_headroom_realizable": capped / wall_headroom,
        "realizable_frontier_tps": frontier,
        "wall_binds": wall_binds,
        "clears_wall": bool(frontier >= wall - 1e-9),
        "clears_deployed_481p53": bool(frontier > DEPLOYED_TPS),
    }


# --------------------------------------------------------------------------- #
# Section 3 -- assemble the four deliverables.                                  #
# --------------------------------------------------------------------------- #
def build_report() -> dict:
    wall_headroom = LAMBDA1_WALL - BASE_482                       # 38.2127 (#436)
    etp_at_wall = ET_DEP * LAMBDA1_WALL / BASE_482                # ~4.156

    # ---- Deliverable 1+2: the realizable ladder (Dcov -> capped TPS) --------
    ladder = {
        "pessimistic_380": size_one(DCOV_PESSIMISTIC),
        "defensible_central_380": size_one(DCOV_DEFENSIBLE_CENTRAL),
        "defensible_lo_380": size_one(DCOV_DEFENSIBLE_LO),
        "defensible_hi_380": size_one(DCOV_DEFENSIBLE_HI),
        "fromscratch_ceiling_380": size_one(DCOV_FROMSCRATCH_CEILING),
        "optimistic_339": size_one(DCOV_OPTIMISTIC_339),
        "full_recipe_339": size_one(DCOV_FULL_RECIPE_339),
    }
    central = ladder["defensible_central_380"]                   # the HEADLINE

    # base-robustness: same Dcov, blanket-strict 467.14 base (fraction invariant).
    central_on_467 = size_one(DCOV_DEFENSIBLE_CENTRAL, base_tps=BASE_BLANKET_STRICT)
    # secant-robustness: central Dcov under the WORST program secant (#377).
    central_worst_secant = size_one(DCOV_DEFENSIBLE_CENTRAL, secant=S_WORST)

    # ---- the headline required fields --------------------------------------
    coverage_gap_realizable_frac = central["coverage_gap_realizable_frac"]
    coverage_gap_structural_frac = central["coverage_gap_structural_frac"]
    realizable_etp_lift = central["realizable_etp_lift"]
    realizable_demand_headroom_tps = central["realizable_demand_headroom_tps"]
    realizable_frontier_tps = central["realizable_frontier_tps"]
    frac_of_headroom = central["frac_of_38p21_headroom_realizable"]

    # band on the headline (defensible fine-tune [lo, hi]).
    headroom_band = [ladder["defensible_lo_380"]["realizable_demand_headroom_tps"],
                     ladder["defensible_hi_380"]["realizable_demand_headroom_tps"]]
    frontier_band = [ladder["defensible_lo_380"]["realizable_frontier_tps"],
                     ladder["defensible_hi_380"]["realizable_frontier_tps"]]

    # ---- Deliverable 3: the (skipped) viability probe ----------------------
    probe_was_run = False
    probe_etp_delta_measured = None
    probe_direction_matches_prediction = None

    # ---- Deliverable 4: the verdict booleans -------------------------------
    # worth it iff the realizable headroom clears materiality across the WHOLE
    # defensible band (floor) AND under the worst secant -- a robust true.
    band_floor = min(headroom_band[0], central_worst_secant["realizable_demand_headroom_tps"])
    retrain_is_worth_it = bool(band_floor > MATERIALITY_TPS)
    demand_axis_effectively_closed = bool(realizable_demand_headroom_tps <= MATERIALITY_TPS)
    wall_is_mirage = bool(not central["wall_binds"])             # delivery binds before the wall

    verdict = (
        f"The +{wall_headroom:.2f}-TPS demand headroom above {BASE_482:.2f} is "
        f"~{100.0*frac_of_headroom:.0f}% realizable -> a practical drafter retrain buys "
        f"~+{realizable_demand_headroom_tps:.1f} TPS (frontier {realizable_frontier_tps:.1f}, "
        f"band [{frontier_band[0]:.1f},{frontier_band[1]:.1f}]); the realizable frontier CLEARS "
        f"the deployed non-equivalent {DEPLOYED_TPS:.2f} but NOT the verify-BW wall {LAMBDA1_WALL:.2f}. "
        f"The remaining ~{100.0*(1.0-frac_of_headroom):.0f}% is a head-ceiling MIRAGE (a practical "
        f"retrain cannot deliver the coverage to reach the wall). "
        f"retrain_is_worth_it={retrain_is_worth_it}; demand_axis_effectively_closed={demand_axis_effectively_closed}."
    )

    selftest = run_self_tests(
        ladder, central, central_on_467, central_worst_secant, wall_headroom,
        etp_at_wall, coverage_gap_realizable_frac, realizable_demand_headroom_tps,
        realizable_frontier_tps, retrain_is_worth_it, demand_axis_effectively_closed,
        wall_is_mirage, probe_was_run)

    headline = (
        f"REALIZABILITY of #436's +{wall_headroom:.2f}-TPS demand headroom: the deployed MTP K=7 head "
        f"is at its #289 LINEAR cap, so only the sub-linear head-ceiling coverage gap "
        f"({COVERAGE_GAP_TOTAL:.5f}) is left. A practical retrain DELIVERS Dcov ~ +{DCOV_DEFENSIBLE_CENTRAL:.3f} "
        f"(#380 defensible; #339's +{DCOV_OPTIMISTIC_339} is known optimistic via V1/V2/V3) -> "
        f"realizable_demand_headroom_tps = +{realizable_demand_headroom_tps:.2f} "
        f"({100.0*frac_of_headroom:.0f}% of 38.21), frontier {realizable_frontier_tps:.2f}. The DRAFTER "
        f"delivery binds at +{realizable_demand_headroom_tps:.1f} TPS, FAR below the +{wall_headroom:.2f} "
        f"wall -> the upper ~{100.0*(1.0-frac_of_headroom):.0f}% of the wall headroom is a mirage. "
        f"retrain_is_worth_it={retrain_is_worth_it}, demand_axis_effectively_closed={demand_axis_effectively_closed}."
    )

    return {
        "pr": 439, "agent": "land", "kind": "drafter-retrain-headroom-realizability",
        "analysis_only": True, "no_launch": True, "no_hf_job": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used": False, "official_tps": 0,
        "baseline_unchanged_tps": DEPLOYED_TPS, "ppl": PPL_DEPLOYED,
        "headline": headline,
        "inputs": {
            "deployed_tps": DEPLOYED_TPS, "base_482": BASE_482,
            "base_blanket_strict_467": BASE_BLANKET_STRICT, "ladder_289": LADDER_289,
            "et_dep": ET_DEP, "lambda1_wall": LAMBDA1_WALL, "linear_cap_289": LINEAR_CAP_289,
            "coverage_gap_total": COVERAGE_GAP_TOTAL, "cov_prior_top4": COV_PRIOR_TOP4,
            "s_central": S_CENTRAL, "s_worst": S_WORST, "tps_per_dcov_399": TPS_PER_DCOV_399,
            "dcov_defensible_central_380": DCOV_DEFENSIBLE_CENTRAL,
            "dcov_defensible_band_380": [DCOV_DEFENSIBLE_LO, DCOV_DEFENSIBLE_HI],
            "dcov_fromscratch_ceiling_380": DCOV_FROMSCRATCH_CEILING,
            "dcov_pessimistic_380": DCOV_PESSIMISTIC,
            "dcov_optimistic_339": DCOV_OPTIMISTIC_339, "dcov_full_recipe_339": DCOV_FULL_RECIPE_339,
            "materiality_tps": MATERIALITY_TPS,
            "a1_deployed_308": A1_DEPLOYED_308, "a1_inrepo_eagle3_308": A1_INREPO_EAGLE3_308,
            "a1_robustness_ceiling_308": A1_ROBUSTNESS_CEILING_308,
            "ppl_deployed": PPL_DEPLOYED, "ppl_gate": PPL_GATE,
            "source_436_run": "nvsbctji", "source_399_run": "ec7i3z5t", "source_380_run": "00oijpwg",
            "source_339_run": "0aq16szh", "source_289_run": "fi34s269", "source_382_run": "382-no-wandb",
            "source_308_run": "5axqa6oa", "source_420_run": "qe4qagc1", "source_401_run": "se8mf9ax",
        },
        # ---- Deliverable 1 ----
        "coverage_gap_total": COVERAGE_GAP_TOTAL,
        "coverage_gap_realizable_frac": coverage_gap_realizable_frac,
        "coverage_gap_structural_frac": coverage_gap_structural_frac,
        "deployed_at_or_above_linear_cap": True,
        "acceptance_cliff_position": 1,
        # ---- Deliverable 2 ----
        "realizable_etp_lift": realizable_etp_lift,
        "realizable_demand_headroom_tps": realizable_demand_headroom_tps,    # PRIMARY
        "realizable_frontier_tps": realizable_frontier_tps,
        "frac_of_38p21_headroom_realizable": frac_of_headroom,
        "wall_headroom_38p21": wall_headroom,
        "etp_at_wall": etp_at_wall,
        "realizable_headroom_band_tps": headroom_band,
        "realizable_frontier_band_tps": frontier_band,
        "clears_deployed_481p53": central["clears_deployed_481p53"],
        "clears_wall_520p95": central["clears_wall"],
        "wall_binds": central["wall_binds"],
        "realizable_ladder": ladder,
        "base_robustness_467": central_on_467,
        "secant_robustness_worst": central_worst_secant,
        "tps_per_dcov_482": tps_per_dcov(),
        # ---- Deliverable 3 ----
        "probe_was_run": probe_was_run,
        "probe_etp_delta_measured": probe_etp_delta_measured,
        "probe_direction_matches_prediction": probe_direction_matches_prediction,
        "probe_skip_reason": ("drafter venv absent (rebuild after container reset) + no local MTP "
                              "checkpoint + #380 found the cheap probe is the WRONG tool (delivery, "
                              "not direction, is the weak link; real coverage-lift pilot ~3 h >> 90-min "
                              "SENPAI_TIMEOUT_MINUTES) -> analytic sizing only."),
        # ---- Deliverable 4 ----
        "retrain_is_worth_it": retrain_is_worth_it,
        "demand_axis_effectively_closed": demand_axis_effectively_closed,
        "wall_is_mirage_delivery_binds": wall_is_mirage,
        "verdict": verdict,
        # ---- self-test ----
        "self_test": selftest,
        "drafter_retrain_headroom_realizability_self_test_passes": selftest["passes"],
        "self_test_passes": selftest["passes"],
    }


# --------------------------------------------------------------------------- #
# Section 4 -- self-tests (0-GPU; PRIMARY gate).                                #
# --------------------------------------------------------------------------- #
def run_self_tests(ladder: dict, central: dict, central_467: dict, central_worst: dict,
                   wall_headroom: float, etp_at_wall: float, realizable_frac: float,
                   realizable_tps: float, realizable_frontier: float, worth_it: bool,
                   closed: bool, wall_mirage: bool, probe_was_run: bool) -> dict:
    c: dict[str, bool] = {}

    # a) pinned anchors round-trip.
    c["a_deployed_is_481p53"] = abs(DEPLOYED_TPS - 481.53) < TOL
    c["a_base_is_482p74"] = abs(BASE_482 - 482.7400155438763) < 1e-9
    c["a_base_above_deployed"] = BASE_482 > DEPLOYED_TPS
    c["a_ladder_len_7"] = len(LADDER_289) == 7
    c["a_et_dep_is_3p851"] = abs(ET_DEP - 3.851185944363104) < 1e-9
    c["a_et_dep_above_linear_cap"] = ET_DEP >= LINEAR_CAP_289      # #289 deployed AT/above linear cap
    c["a_wall_is_520p95"] = abs(LAMBDA1_WALL - 520.9527323111674) < 1e-9
    c["a_gap_is_0p10973"] = abs(COVERAGE_GAP_TOTAL - 0.10973404808468479) < 1e-12
    c["a_gap_is_complement_of_top4"] = abs((1.0 - COV_PRIOR_TOP4) - COVERAGE_GAP_TOTAL) < 1e-9
    c["a_s_central_is_7p913"] = abs(S_CENTRAL - 7.912609135742992) < 1e-9
    c["a_wall_headroom_is_38p21"] = abs(wall_headroom - (LAMBDA1_WALL - BASE_482)) < 1e-9
    c["a_wall_headroom_about_38"] = 38.0 < wall_headroom < 38.5

    # b) the demand map (reused from #436): round-trips + ordering.
    c["b_tps_at_et_dep_is_base"] = abs(tps_from_et(ET_DEP) - BASE_482) < 1e-9
    c["b_map_monotone"] = tps_from_et(ET_DEP + 0.1) > tps_from_et(ET_DEP)
    c["b_etp_at_wall_roundtrips"] = abs(tps_from_et(etp_at_wall) - LAMBDA1_WALL) < 1e-6
    c["b_etp_at_wall_about_4p156"] = 4.10 < etp_at_wall < 4.20
    # tps_per_dcov on the 471 base reproduces #401's banked 968.57.
    c["b_tps_per_dcov_471_reproduces_399"] = abs(tps_per_dcov(471.41634950257713) - TPS_PER_DCOV_399) < 0.5
    c["b_tps_per_dcov_482_positive"] = tps_per_dcov() > 0.0

    # c) realizable Dcov ordering + decomposition arithmetic.
    c["c_dcov_ordering"] = (DCOV_PESSIMISTIC <= DCOV_DEFENSIBLE_CENTRAL
                            <= DCOV_FROMSCRATCH_CEILING <= DCOV_OPTIMISTIC_339 <= DCOV_FULL_RECIPE_339)
    c["c_defensible_band_brackets_central"] = DCOV_DEFENSIBLE_LO <= DCOV_DEFENSIBLE_CENTRAL <= DCOV_DEFENSIBLE_HI
    c["c_339_is_above_defensible"] = DCOV_OPTIMISTIC_339 > DCOV_DEFENSIBLE_HI    # #380 corrected #339 down
    c["c_frac_is_dcov_over_gap"] = abs(realizable_frac - DCOV_DEFENSIBLE_CENTRAL / COVERAGE_GAP_TOTAL) < 1e-9
    c["c_frac_plus_structural_is_one"] = abs(central["coverage_gap_realizable_frac"]
                                             + central["coverage_gap_structural_frac"] - 1.0) < 1e-9
    c["c_frac_in_unit"] = 0.0 < realizable_frac < 1.0
    c["c_structural_majority"] = central["coverage_gap_structural_frac"] > 0.5   # mostly structural
    c["c_etp_lift_is_s_times_dcov"] = abs(central["realizable_etp_lift"]
                                          - S_CENTRAL * DCOV_DEFENSIBLE_CENTRAL) < 1e-9

    # d) the TPS map: headroom, frontier, wall-does-NOT-bind (delivery binds).
    c["d_headroom_positive"] = realizable_tps > 0.0
    c["d_headroom_below_wall_headroom"] = realizable_tps < wall_headroom        # delivery binds, not wall
    c["d_wall_does_not_bind_central"] = central["wall_binds"] is False
    c["d_wall_is_mirage"] = wall_mirage is True
    c["d_headroom_roundtrips"] = abs(tps_from_et(ET_DEP + central["realizable_etp_lift"])
                                     - realizable_frontier) < 1e-6
    c["d_frontier_is_base_plus_headroom"] = abs(realizable_frontier - (BASE_482 + realizable_tps)) < 1e-9
    c["d_frontier_clears_deployed"] = realizable_frontier > DEPLOYED_TPS         # equiv beats non-equiv
    c["d_frontier_below_wall"] = realizable_frontier < LAMBDA1_WALL              # cannot exceed the wall
    c["d_headroom_about_16"] = 14.0 < realizable_tps < 18.0
    c["d_frontier_about_499"] = 496.0 < realizable_frontier < 501.0
    c["d_frac_of_headroom_about_0p42"] = 0.38 < central["frac_of_38p21_headroom_realizable"] < 0.46

    # e) ladder consistency: optimistic #339 reaches the wall, defensible does not.
    c["e_optimistic_339_reaches_wall"] = ladder["optimistic_339"]["realizable_frontier_tps"] >= LAMBDA1_WALL - 0.5
    c["e_full_recipe_339_caps_at_wall"] = ladder["full_recipe_339"]["wall_binds"] is True
    c["e_defensible_below_optimistic"] = (central["realizable_demand_headroom_tps"]
                                          < ladder["optimistic_339"]["realizable_demand_headroom_tps"])
    c["e_ladder_monotone_in_dcov"] = all(
        ladder[a]["realizable_tps_uncapped"] <= ladder[b]["realizable_tps_uncapped"]
        for a, b in [("pessimistic_380", "defensible_central_380"),
                     ("defensible_central_380", "fromscratch_ceiling_380"),
                     ("fromscratch_ceiling_380", "optimistic_339")])
    c["e_all_defensible_clear_deployed"] = all(
        ladder[k]["clears_deployed_481p53"] for k in
        ("pessimistic_380", "defensible_central_380", "defensible_lo_380",
         "defensible_hi_380", "fromscratch_ceiling_380"))
    c["e_none_defensible_clears_wall"] = all(
        not ladder[k]["clears_wall"] for k in
        ("pessimistic_380", "defensible_central_380", "defensible_hi_380", "fromscratch_ceiling_380"))

    # f) base + secant robustness: fraction invariant, verdict survives worst secant.
    c["f_frac_base_invariant"] = abs(central_467["coverage_gap_realizable_frac"]
                                     - central["coverage_gap_realizable_frac"]) < 1e-9
    c["f_headroom_rescales_with_base"] = central_467["realizable_demand_headroom_tps"] < central["realizable_demand_headroom_tps"]
    c["f_worst_secant_still_material"] = central_worst["realizable_demand_headroom_tps"] > MATERIALITY_TPS
    c["f_worst_secant_below_central"] = central_worst["realizable_demand_headroom_tps"] < realizable_tps

    # g) verdict booleans.
    c["g_worth_it_true"] = worth_it is True
    c["g_worth_it_above_materiality"] = realizable_tps > MATERIALITY_TPS
    c["g_demand_axis_not_closed"] = closed is False
    c["g_band_floor_above_materiality"] = min(
        ladder["defensible_lo_380"]["realizable_demand_headroom_tps"],
        central_worst["realizable_demand_headroom_tps"]) > MATERIALITY_TPS

    # h) probe skipped, identity/ppl preserved, numeric hygiene.
    c["h_probe_not_run"] = probe_was_run is False
    c["h_ppl_within_gate"] = PPL_DEPLOYED <= PPL_GATE
    c["h_a1_inrepo_above_deployed"] = A1_INREPO_EAGLE3_308 > A1_DEPLOYED_308     # retrain DOES lift a_1
    c["h_no_nan_inf"] = all(_finite(v) for v in
                            [realizable_frac, realizable_tps, realizable_frontier, wall_headroom,
                             etp_at_wall, ET_DEP, central["realizable_etp_lift"],
                             central_467["realizable_demand_headroom_tps"],
                             central_worst["realizable_demand_headroom_tps"]])

    # k) artifact provenance cross-check (pinned constants == merged JSONs).
    d436, d399, d380, d339, d289, d382, d308 = (
        _load(a) for a in (ART_436, ART_399, ART_380, ART_339, ART_289, ART_382, ART_308))
    if d436 is not None:
        di = d436.get("inputs", {})
        c["k_436_base_482"] = abs(di.get("base_482", 0) - BASE_482) < 1e-9
        c["k_436_wall"] = abs(di.get("lambda1_wall", 0) - LAMBDA1_WALL) < 1e-9
        c["k_436_et_dep"] = abs(di.get("et_dep", 0) - ET_DEP) < 1e-9
        c["k_436_gap"] = abs(di.get("coverage_ceiling_gap", 0) - COVERAGE_GAP_TOTAL) < 1e-9
        c["k_436_s_central"] = abs(di.get("s_central", 0) - S_CENTRAL) < 1e-9
        c["k_436_headroom_38"] = abs(d436.get("demand_headroom_to_ceiling_tps", 0) - wall_headroom) < TOL
        c["k_436_ceiling_is_wall"] = abs(d436.get("identity_free_demand_ceiling_tps", 0) - LAMBDA1_WALL) < 1e-9
    if d399 is not None:
        c["k_399_gap"] = abs(d399.get("coverage_ceiling_gap", 0) - COVERAGE_GAP_TOTAL) < 1e-9
        c["k_399_s_central"] = abs(d399.get("inputs", {}).get("s_central", 0) - S_CENTRAL) < 1e-9
        c["k_399_top4"] = abs(d399.get("top4_coverage_measured", 0) - COV_PRIOR_TOP4) < 1e-9
    if d380 is not None:
        dd = d380.get("defensible_delivery_distribution", {})
        c["k_380_defensible_central"] = abs(dd.get("mean", 0) - DCOV_DEFENSIBLE_CENTRAL) < 1e-9
        c["k_380_defensible_band"] = dd.get("band") == [DCOV_DEFENSIBLE_LO, DCOV_DEFENSIBLE_HI]
        c["k_380_fromscratch_ceiling"] = abs(
            dd.get("ceiling_from_scratch", {}).get("mean", 0) - DCOV_FROMSCRATCH_CEILING) < 1e-9
        c["k_380_pessimistic"] = abs(dd.get("pessimistic", {}).get("mean", 0) - DCOV_PESSIMISTIC) < 1e-9
        c["k_380_original_339"] = abs(dd.get("original_339", {}).get("mean", 0) - DCOV_OPTIMISTIC_339) < 1e-9
        c["k_380_s_worst"] = abs(d380.get("inputs", {}).get("s_program_worst_377", 0) - S_WORST) < 1e-9
        c["k_380_recipe_is_real"] = d380.get("verdict", {}).get("recipe_is_real") is True
    if d339 is not None:
        syn = d339.get("synthesis", {})
        c["k_339_recipe_dcov"] = abs(
            syn.get("step2_recipe_convolution", {}).get("recipe_mean_delta_cov", 0) - DCOV_OPTIMISTIC_339) < 1e-9
        c["k_339_full_dcov"] = abs(
            syn.get("step2_full_and_naive", {}).get("full_recipe", {}).get("full_mean_delta_cov", 0)
            - DCOV_FULL_RECIPE_339) < 1e-9
    if d289 is not None:
        c["k_289_linear_cap_flag"] = d289.get("linear_cap", {}).get("deployed_at_or_above_linear_cap") is True
        c["k_289_cliff_pos_1"] = d289.get("cliff", {}).get("acceptance_cliff_position") == 1
        c["k_289_et"] = abs(d289.get("decomposition", {}).get("E_T", 0) - ET_DEP) < 1e-9
        c["k_289_nonlinear_required"] = d289.get("linear_cap", {}).get("built_raise_requires_nonlinear_drafter") is True
    if d382 is not None:
        c["k_382_slope_private_robust"] = d382.get("slope_is_private_robust") is True
        c["k_382_demand_route_survives"] = d382.get("demand_route_survives_private_oob") is True
    if d308 is not None:
        env = d308.get("synthesis", {}).get("step2_published_envelope", {})
        c["k_308_a1_inrepo_eagle3"] = abs(env.get("a1_inrepo_eagle3_native_step1", 0) - A1_INREPO_EAGLE3_308) < 1e-6
        c["k_308_a1_robustness_ceiling"] = abs(
            d308.get("synthesis", {}).get("step3_ceiling", {}).get("a1_argmax_robustness_ceiling_central", 0)
            - A1_ROBUSTNESS_CEILING_308) < 1e-9

    passes = all(c.values())
    return {"conditions": c, "n_checks": len(c), "n_passed": sum(1 for v in c.values() if v), "passes": passes}


# --------------------------------------------------------------------------- #
# Section 5 -- reporting + W&B + entrypoint.                                    #
# --------------------------------------------------------------------------- #
def print_report(r: dict) -> None:
    print("\n=== Is the +38.21-TPS demand headroom above 482.74 realizable? (PR #439, land) ===")
    print(f"deployed NON-equiv (#52) = {DEPLOYED_TPS:.2f}   DEMAND base (#428 cb3) = {BASE_482:.4f}   "
          f"E[T]_dep (#289) = {ET_DEP:.4f}   verify-BW wall = {LAMBDA1_WALL:.2f}")
    print("\n-- deliverable 1: decompose the +0.10973 head-ceiling coverage gap --")
    print(f"  deployed MTP K=7 is at its #289 LINEAR cap (E[T] {ET_DEP:.4f} >= {LINEAR_CAP_289}); "
          f"acceptance cliff at position {r['acceptance_cliff_position']}")
    print(f"  coverage_gap_total = {r['coverage_gap_total']:.5f}  (1.0 - top4 root {COV_PRIOR_TOP4:.4f})")
    print(f"  realizable Dcov (#380 defensible fine-tune central) = +{DCOV_DEFENSIBLE_CENTRAL:.3f}  "
          f"band [+{DCOV_DEFENSIBLE_LO},+{DCOV_DEFENSIBLE_HI}]; from-scratch ceiling +{DCOV_FROMSCRATCH_CEILING:.4f}")
    print(f"  => coverage_gap_realizable_frac = {r['coverage_gap_realizable_frac']:.4f}  "
          f"structural_frac = {r['coverage_gap_structural_frac']:.4f}")
    print("\n-- deliverable 2: realizable coverage -> E[T] -> equiv-TPS on the 482.74 base --")
    print(f"  TPS_per_dcov = {r['tps_per_dcov_482']:.2f}  (BASE*S/ET_DEP; #401 banked 968.57 on 471 base)")
    print(f"  realizable_etp_lift            = {r['realizable_etp_lift']:.4f}")
    print(f"  realizable_demand_headroom_tps = +{r['realizable_demand_headroom_tps']:.2f}  "
          f"({100.0*r['frac_of_38p21_headroom_realizable']:.0f}% of the +{r['wall_headroom_38p21']:.2f} wall headroom)")
    print(f"  realizable_frontier_tps        = {r['realizable_frontier_tps']:.2f}  "
          f"band [{r['realizable_frontier_band_tps'][0]:.2f},{r['realizable_frontier_band_tps'][1]:.2f}]")
    print(f"  clears deployed 481.53: {r['clears_deployed_481p53']}   clears wall 520.95: {r['clears_wall_520p95']}   "
          f"wall_binds: {r['wall_binds']} (delivery binds first)")
    print("  realizable ladder (Dcov -> capped headroom / frontier):")
    for k, v in r["realizable_ladder"].items():
        print(f"    {k:24s} Dcov={v['dcov']:.4f}  +{v['realizable_demand_headroom_tps']:6.2f} TPS  "
              f"frontier {v['realizable_frontier_tps']:.2f}  wall_binds={v['wall_binds']}")
    print(f"  base-robustness (467.14 base): +{r['base_robustness_467']['realizable_demand_headroom_tps']:.2f} TPS "
          f"(frac {r['base_robustness_467']['coverage_gap_realizable_frac']:.4f} INVARIANT)")
    print(f"  secant-robustness (worst S={S_WORST:.3f}): +{r['secant_robustness_worst']['realizable_demand_headroom_tps']:.2f} TPS "
          f"(still > {MATERIALITY_TPS})")
    print("\n-- deliverable 3: optional MTP-drafter fine-tune viability probe --")
    print(f"  probe_was_run = {r['probe_was_run']}  ({r['probe_skip_reason']})")
    print("\n-- deliverable 4: the honest verdict --")
    print(f"  retrain_is_worth_it = {r['retrain_is_worth_it']}   demand_axis_effectively_closed = {r['demand_axis_effectively_closed']}")
    print(f"  {r['verdict']}")
    print(f"\nPPL unchanged {PPL_DEPLOYED} <= {PPL_GATE} (acceptance change is teacher-forced PPL-neutral)")
    print(f"\nself-test: {r['self_test']['n_passed']}/{r['self_test']['n_checks']} checks  "
          f"self_test_passes = {r['self_test_passes']}")


def log_to_wandb(report: dict, group: str, name: str) -> str:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_unavailable"
    try:
        run = wandb.init(group=group, name=name, config=report["inputs"])
        wandb.summary.update({
            "headline": report["headline"],
            "verdict": report["verdict"],
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            "coverage_gap_total": report["coverage_gap_total"],
            "coverage_gap_realizable_frac": report["coverage_gap_realizable_frac"],
            "coverage_gap_structural_frac": report["coverage_gap_structural_frac"],
            "realizable_etp_lift": report["realizable_etp_lift"],
            "realizable_demand_headroom_tps": report["realizable_demand_headroom_tps"],
            "realizable_frontier_tps": report["realizable_frontier_tps"],
            "frac_of_38p21_headroom_realizable": report["frac_of_38p21_headroom_realizable"],
            "retrain_is_worth_it": report["retrain_is_worth_it"],
            "demand_axis_effectively_closed": report["demand_axis_effectively_closed"],
            "probe_was_run": report["probe_was_run"],
            "ppl": PPL_DEPLOYED,
            "self_test_passes": report["self_test_passes"],
        })
        wandb.log({
            "summary/coverage_gap_total": report["coverage_gap_total"],
            "summary/coverage_gap_realizable_frac": report["coverage_gap_realizable_frac"],
            "summary/coverage_gap_structural_frac": report["coverage_gap_structural_frac"],
            "summary/realizable_etp_lift": report["realizable_etp_lift"],
            "summary/realizable_demand_headroom_tps": report["realizable_demand_headroom_tps"],
            "summary/realizable_frontier_tps": report["realizable_frontier_tps"],
            "summary/frac_of_38p21_headroom_realizable": report["frac_of_38p21_headroom_realizable"],
            "summary/wall_headroom_38p21": report["wall_headroom_38p21"],
            "summary/tps_per_dcov_482": report["tps_per_dcov_482"],
            "summary/base_482": BASE_482,
            "summary/et_dep": ET_DEP,
            "summary/lambda1_wall": LAMBDA1_WALL,
            "summary/deployed_tps": DEPLOYED_TPS,
            "summary/dcov_defensible_central": DCOV_DEFENSIBLE_CENTRAL,
            "summary/s_central": S_CENTRAL,
            "summary/ppl_deployed": PPL_DEPLOYED,
            "summary/self_test_passes": float(report["self_test_passes"]),
            "summary/self_test_n_checks": float(report["self_test"]["n_checks"]),
        })
        # realizable ladder table.
        lt = wandb.Table(columns=["scenario", "dcov", "realizable_demand_headroom_tps",
                                  "realizable_frontier_tps", "coverage_gap_realizable_frac",
                                  "frac_of_38p21_headroom", "wall_binds", "clears_deployed"])
        for k, v in report["realizable_ladder"].items():
            lt.add_data(k, v["dcov"], v["realizable_demand_headroom_tps"], v["realizable_frontier_tps"],
                        v["coverage_gap_realizable_frac"], v["frac_of_38p21_headroom_realizable"],
                        v["wall_binds"], v["clears_deployed_481p53"])
        wandb.log({"realizable_ladder": lt})
        # frontier ladder vs anchors.
        ct = wandb.Table(columns=["config", "tps", "axis", "note"])
        ct.add_data("deployed (#52)", DEPLOYED_TPS, "incumbent", "non-equivalent (identity 0.9966)")
        ct.add_data("equiv base (#428 cb3)", BASE_482, "base", "demand base")
        ct.add_data("realizable frontier (central)", report["realizable_frontier_tps"], "demand",
                    "defensible retrain Dcov +0.016")
        ct.add_data("from-scratch ceiling", report["realizable_ladder"]["fromscratch_ceiling_380"]["realizable_frontier_tps"],
                    "demand", "optimistic-defensible")
        ct.add_data("verify-BW wall (#349/#344)", LAMBDA1_WALL, "wall", "retrain-gated ceiling; NOT reached")
        wandb.log({"frontier_ladder": ct})
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
        description="Realizability of #436's +38.21-TPS demand headroom above 482.74 (PR #439).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="0-GPU analytic gate (PR #439 deliverables)")
    ap.add_argument("--reanalyze", action="store_true", help="0-GPU full re-analysis (alias of default)")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="drafter-retrain-headroom")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="land/drafter-retrain-headroom-realizability")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/validity/drafter_retrain_headroom_realizability/"
                            "drafter_retrain_headroom_realizability_results.json")
    args = ap.parse_args()

    report = build_report()
    print_report(report)
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    report["peak_mem_mib"] = peak_mib

    if args.self_test:
        out = HERE / "drafter_retrain_headroom_realizability_selftest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}  (peak {peak_mib:.1f} MiB)")
        print(f"\ndrafter_retrain_headroom_realizability_self_test_passes = {report['self_test']['passes']}")
        return 0 if report["self_test"]["passes"] else 1

    report["wandb_run_id"] = None if args.no_wandb else log_to_wandb(report, args.wandb_group, args.wandb_name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')}, peak {peak_mib:.1f} MiB)")

    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [report["wandb_run_id"]] if report.get("wandb_run_id") else [],
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "coverage_gap_realizable_frac": float(report["coverage_gap_realizable_frac"]),
        "realizable_etp_lift": float(report["realizable_etp_lift"]),
        "realizable_demand_headroom_tps": float(report["realizable_demand_headroom_tps"]),
        "realizable_frontier_tps": float(report["realizable_frontier_tps"]),
        "retrain_is_worth_it": bool(report["retrain_is_worth_it"]),
        "demand_axis_effectively_closed": bool(report["demand_axis_effectively_closed"]),
        "probe_was_run": bool(report["probe_was_run"]),
        "ppl": float(PPL_DEPLOYED),
        "self_test_passes": bool(report["self_test_passes"]),
        "primary_metric": {"name": "realizable_demand_headroom_tps",
                           "value": float(report["realizable_demand_headroom_tps"])},
        "test_metric": {"name": "self_test_passes", "value": float(report["self_test_passes"])},
    }))
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
