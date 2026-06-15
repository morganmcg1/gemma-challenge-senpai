#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PPL-only gate (#124): what LIFTING the greedy-identity gate (#192) buys on the >500 lane (PR #343, wirbel).

THE GOVERNING QUESTION (the #124/#319 decision the human needs priced)
----------------------------------------------------------------------
The greedy-token-identity gate (#192) is SELF-IMPOSED: the official scorer checks only PPL<=2.42 +
128/128 completions; it runs NO token-identity check, and the deployed 481.53 frontier itself is
~56% AR-divergent yet scores fine (issue #124). Leviathan et al. (ICML 2023, arXiv:2211.17192) prove
spec-decode preserves the output *distribution* exactly -- the guarantee is distributional, NOT
token-identical -- so byte-exact greedy identity is not a standard correctness criterion. This card
prices the load-bearing consequence: what does LIFTING #192 actually buy on the >500 lane?

KEY PRECISION (do not over-claim a free win): the deployed 481.53 is ALREADY the fast,
non-deterministic, PPL-passing config (its 56% AR-divergence proves batch-invariant determinism is OFF
in deployment). So lifting #192 does NOT change the deployed config or give a free one-run jump to ~520.
It means we never pay denken #332's determinism tax (which we'd pay ONLY to chase strict identity), and
the >500 path becomes a coverage/E[T] retrain on the already-fast config that ACTUALLY REACHES 500 (vs
being supply-capped at 473.5 under strict).

THE TWO WORLDS (re-prices banked anchors; INVERTS stark #340; re-derives NOTHING)
---------------------------------------------------------------------------------
* STRICT world (gate ON): a deterministic identity kernel pays denken #332's supply tax. The BEST the
  strict world can do, even at PERFECT coverage (central corner = lambda ceiling 520.95), is
      strict_ceiling = 520.95 * (1 - supply_floor),   supply_floor >= 0.09103 @ geometric phi
                     = 473.53 TPS  <  500
  so strict_500_reachable = False -- supply-capped below 500 for EVERY realizable deterministic
  schedule (denken #332, phi_realizable >= 1), AND coverage must additionally clear the identity bar
  0.9213. The strict-gated lane is DEAD end-to-end (supply RED + demand insufficient).
* PPL-ONLY world (gate LIFTED, #124): no determinism is required -> the supply-phi tax VANISHES -> the
  operative envelope is stark #340's DEMAND-only map (anchors already EXCLUDE the supply tax):
      E[T](c)            = 1 + sum_{d=1..7} c^d
      envelope_X(c)      = X_ANCHOR * E[T](c) / E[T](0.9213),   X in {central 520.95, worst 492.87}
  Coverage need only keep PPL <= 2.42 (NOT clear the 0.9213 identity bar). So the PPL-only >500 lane is
  NOT supply-capped; it is reachable purely via COVERAGE.

THE LIFT (priced at the MEASURED coverage; the deliverable the #124 decision needs)
-----------------------------------------------------------------------------------
At measured fusion coverage c=0.8903 (lawine #330) the PPL-only envelope is 470.35 / 444.99 (stark
#337) -- BOTH < 500, so the EXISTING head does NOT give a free >500 even gate-lifted. Solving env(c)=500
on the demand-only map gives the SAME roots as stark #340 (supply tax just absent):
    c*_central = 0.9089  =>  coverage_lift_for_ppl_only_central_500 = +0.0186 (<= lawine #336 +0.031 -> WITHIN budget)
    c*_worst   = 0.9256  =>  coverage_lift_for_ppl_only_worst_500   = +0.0353 (>  +0.031          -> OVER  budget, marginal)
Note c*_central=0.9089 < identity_bar 0.9213: PPL-only central-500 needs LESS coverage than the strict
identity bar AND no supply revival.

THE #124 DELTA (one table, one sentence)
----------------------------------------
  STRICT:    >500 IMPOSSIBLE  (supply-capped 473.5; no retrain reaches it).
  PPL-ONLY:  >500 ACHIEVABLE  via a coverage retrain to 0.9089 (central) / 0.9256 (worst).
Lifting #192 converts the >500 lane from IMPOSSIBLE (strict) to a SIZED, FEASIBLE coverage-retrain
target (central within the +0.031 budget; worst marginally past it).

PPL SAFETY (cite, do not re-run): wirbel #324 (pespixw1) showed the M=8 argmax divergence is
structurally DECOUPLED from PPL -- PPL is a prompt_logprobs reference-forward over fixed token-IDs, so
it passes by construction (PPL stays 2.3772 <= 2.42, M-binary). The PPL-only config is the SAME deployed
serve that ALREADY passes the official scorer, so the gate-lift introduces NO new PPL risk. Caveat: the
literal served greedy-rate would need an HF Job (gated, ubel #322) -- not drawn here.

LOCAL, CPU-ONLY, ANALYTIC. 0 GPU, no model forward, no training, no publish, no HF Job, no submission,
no served-file change, no official draw. BASELINE stays 481.53; adds 0 TPS. Imports verbatim: stark #340
jwv1vbug (demand-only envelope + c*_central 0.9089 / c*_worst 0.9256 / identity bar 0.9213), stark #337
lbuirkpt (E[T](c) chain law, anchors 520.95/492.87, honest corners 470.35/444.99 @0.8903), denken #332
y5cl0ena (supply floor 0.09103 @ geometric phi, strict ceiling 473.5, phi_realizable >= 1), lawine #330
hfrscdai (cov prior 0.8903, identity bar 0.9213), lawine #336 krroookz (+0.031 retrain budget), wirbel
#324 pespixw1 (PPL-decoupling 2.3772, M-binary). NOT a launch / build / submission / open2.

PRIMARY metric  ppl_only_envelope_self_test_passes
TEST    metric  coverage_lift_for_ppl_only_central_500   (coverage delta from 0.8903 to PPL-only central-500)
TEST    metric  ppl_only_500_reachable_via_coverage      (central-500 lift fits the +0.031 retrain budget)
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Imported EXACT from banked W&B runs. Re-derive NOTHING. The PR's displayed forms (0.9213, 520.95,
# 492.87, 0.8903, 473.5) are round-to-display values, asserted exact in the self-test.
# --------------------------------------------------------------------------- #
K_SPEC = 7                              # deployed speculative depth (chain-law K; ubel #311)
IDENTITY_BAR = 0.9213011665456927      # strict greedy-identity per-depth c_eff bar (lawine #330)
COV_PRIOR = 0.8903                     # lawine #330 (hfrscdai) honest fusion top-4 c_eff (MEASURED)
E_T_AT_IDENTITY = 6.111214987369918    # stark #337 E[T](0.9213) == 1 + sum_{d=1..7} 0.92130117^d

# stark #337 / stark #340 compliant-500 banked corners, both at E[T]=6.1112 (the DEMAND-only anchors;
# they already EXCLUDE the supply tax). central is CAP-BOUND (= lambda ceiling 520.95); worst uncapped.
CENTRAL_ANCHOR = 520.9527323111674     # stark #337 fern325_central_at_611 (cap-bound) == lambda ceiling
WORST_ANCHOR = 492.865273281899        # stark #337 fern325_worst_at_611 (uncapped private-tax)
LAMBDA_CEIL = 520.9527323111674        # int4-spec batch-invariant verify ceiling (== central cap)

# stark #337 banked honest corners at COV_PRIOR (for an EXACT #337 round-trip).
HONEST_CENTRAL_337 = 470.347938447151
HONEST_WORST_337 = 444.9888652889661

# stark #340 (jwv1vbug) banked inverse roots -- the demand-only env(c)=500 crossings. The PPL-only
# world re-solves the SAME demand envelope (supply tax absent), so these round-trip EXACTLY.
C_STAR_CENTRAL_340 = 0.9089363308345582
C_STAR_WORST_340 = 0.925603648491971

# denken #332 (y5cl0ena) STRICT-world supply tax: the determinism floor at geometric phi and the
# resulting strict ceiling. phi_realizable >= 1 => this floor is the BEST a deterministic schedule does.
SUPPLY_FLOOR_GEO = 0.09103155435261377   # forgone-parallelism fraction @ geometric phi (M=8 SDPA)
STRICT_CEILING_332 = 473.5295953446407   # 520.95 * (1 - 0.09103) -- denken #332 compliant_ceiling_at_geo
SUPPLY_REVIVE_BREAKEVEN = 0.255          # phi recovery needed to revive >500 under strict (NOT reachable)

# lawine #336 (krroookz) retrain head-coverage lift budget: 0.9213 - 0.8903 = +0.031.
RETRAIN_LIFT_BUDGET = 0.031

# wirbel #324 (pespixw1) PPL-decoupling: PPL stays at the deployed value, structurally decoupled from
# emission (prompt_logprobs reference-forward over fixed token-IDs), M-binary.
PPL_DEPLOYED = 2.3772
PPL_GATE = 2.42

TARGET = 500.0
BASELINE_TPS = 481.53

TOL_EXACT = 1e-9          # anchor round-trip / import-exact checks
TOL_332 = 1e-6           # reproduce denken #332 strict ceiling
TOL_337 = 1e-6           # reproduce stark #337 banked honest corners
TOL_340 = 1e-6           # reproduce stark #340 banked inverse roots
TOL_ROOT = 1e-7          # inverse root residual (|env(c*) - 500|)
TOL_DISPLAY_C = 5e-5     # full-precision constant rounds to its displayed 4-dp form
TOL_DISPLAY_TPS = 5e-3   # full-precision anchor rounds to its displayed 2-dp form


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Core laws (stark #337/#340 conventions) + the demand-only envelope and the strict supply tax.
# --------------------------------------------------------------------------- #
def e_t(c: float, k: int = K_SPEC) -> float:
    """Chain-law expected accepted tokens: E[T] = 1 + sum_{d=1..K} c^d (stark #337)."""
    return 1.0 + sum(c ** d for d in range(1, k + 1))


def envelope_central(c: float) -> float:
    """PPL-only (DEMAND-only) central envelope: supply tax ABSENT, scales with coverage c."""
    return CENTRAL_ANCHOR * e_t(c) / E_T_AT_IDENTITY


def envelope_worst(c: float) -> float:
    """PPL-only (DEMAND-only) worst (uncapped private-tax) envelope."""
    return WORST_ANCHOR * e_t(c) / E_T_AT_IDENTITY


def strict_ceiling(anchor: float, supply_floor: float = SUPPLY_FLOOR_GEO) -> float:
    """STRICT-world ceiling: anchor (best-case perfect-coverage corner) * (1 - supply_floor).

    denken #332 proved phi_realizable >= 1, so supply_floor is the MINIMUM a deterministic schedule
    pays; hence anchor*(1-supply_floor) is the SUPREMUM of the strict envelope over all coverage c.
    """
    return anchor * (1.0 - supply_floor)


def solve_c_for_envelope(env_fn: Callable[[float], float], target_env: float,
                         lo: float = 0.0, hi: float = 1.0, iters: int = 200) -> float:
    """Monotone bisection: c in [lo,hi] with env_fn(c) == target_env (env_fn strictly increasing)."""
    f_lo, f_hi = env_fn(lo) - target_env, env_fn(hi) - target_env
    if f_lo > 0.0 or f_hi < 0.0:
        return float("nan")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if env_fn(mid) < target_env:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# (D1) Establish the two worlds against the banked envelope.
# --------------------------------------------------------------------------- #
def deliverable1_two_worlds() -> dict[str, Any]:
    # STRICT world: best-case ceiling at perfect coverage (central corner = lambda ceiling), taxed.
    strict_ceil_central = strict_ceiling(CENTRAL_ANCHOR)
    strict_ceil_worst = strict_ceiling(WORST_ANCHOR)
    strict_best = max(strict_ceil_central, strict_ceil_worst)   # central is the cap-bound best corner
    strict_500_reachable = bool(strict_best >= TARGET)
    # PPL-ONLY world: demand-only envelope, supply tax 0 -> round-trip anchors @ identity and prior.
    central_at_bar = envelope_central(IDENTITY_BAR)
    worst_at_bar = envelope_worst(IDENTITY_BAR)
    central_at_prior = envelope_central(COV_PRIOR)
    worst_at_prior = envelope_worst(COV_PRIOR)
    return {
        "strict_world": {
            "law": "strict_ceiling_X = X_ANCHOR * (1 - supply_floor); supremum over coverage (phi_realizable>=1)",
            "supply_floor_geometric_phi": SUPPLY_FLOOR_GEO,
            "strict_ceiling_central": strict_ceil_central,
            "strict_ceiling_worst": strict_ceil_worst,
            "strict_best_ceiling": strict_best,
            "roundtrips_denken332_473p5": bool(abs(strict_ceil_central - STRICT_CEILING_332) <= TOL_332),
            "strict_500_reachable": strict_500_reachable,
            "strict_ceiling_gap_to_500": TARGET - strict_best,
            "also_requires_identity_bar": IDENTITY_BAR,
            "supply_revive_breakeven_phi": SUPPLY_REVIVE_BREAKEVEN,
            "note": ("STRICT pays denken #332's determinism tax. Even at PERFECT coverage (central "
                     "corner = lambda ceiling 520.95) the best the strict world does is "
                     "520.95*(1-0.09103) = 473.53 < 500: supply-capped below 500 for EVERY realizable "
                     "deterministic schedule (phi_realizable>=1). strict_500_reachable=False. Coverage "
                     "must ALSO clear the identity bar 0.9213 -- a second binding constraint."),
        },
        "ppl_only_world": {
            "law": "envelope_X(c) = X_ANCHOR * E[T](c)/E[T](0.9213); supply tax = 0 (no determinism)",
            "supply_tax": 0.0,
            "anchor_roundtrip_identity": {
                "envelope_central_at_identity": central_at_bar,
                "envelope_worst_at_identity": worst_at_bar,
                "matches_central_anchor": bool(abs(central_at_bar - CENTRAL_ANCHOR) <= TOL_EXACT),
                "matches_worst_anchor": bool(abs(worst_at_bar - WORST_ANCHOR) <= TOL_EXACT),
            },
            "honest_corner_roundtrip_337": {
                "envelope_central_at_prior": central_at_prior,
                "envelope_worst_at_prior": worst_at_prior,
                "matches_337_central_470p35": bool(abs(central_at_prior - HONEST_CENTRAL_337) <= TOL_337),
                "matches_337_worst_444p99": bool(abs(worst_at_prior - HONEST_WORST_337) <= TOL_337),
                "both_below_500": bool(central_at_prior < TARGET and worst_at_prior < TARGET),
            },
            "gate_requirement": "PPL <= 2.42 only (NOT the 0.9213 identity bar)",
            "note": ("PPL-ONLY drops determinism -> supply-phi tax VANISHES -> operative envelope is "
                     "stark #340's DEMAND-only map (anchors 520.95/492.87 already EXCLUDE the supply "
                     "tax). The >500 lane is NOT supply-capped; it is reachable purely via coverage."),
        },
    }


# --------------------------------------------------------------------------- #
# (D2) Price the PPL-only envelope at the MEASURED coverage and solve the lift (PRIMARY/TEST).
# --------------------------------------------------------------------------- #
def deliverable2_price_and_lift() -> dict[str, Any]:
    # (a) PPL-only envelope at the MEASURED coverage 0.8903: the EXISTING head, gate-lifted.
    ppl_only_central_at_prior = envelope_central(COV_PRIOR)
    ppl_only_worst_at_prior = envelope_worst(COV_PRIOR)
    existing_head_gives_free_500 = bool(ppl_only_central_at_prior >= TARGET)
    # (b) solve the demand-only env(c)=500 roots (SAME as stark #340 -- supply tax absent).
    c_star_central = solve_c_for_envelope(envelope_central, TARGET)
    c_star_worst = solve_c_for_envelope(envelope_worst, TARGET)
    res_central = envelope_central(c_star_central) - TARGET
    res_worst = envelope_worst(c_star_worst) - TARGET
    # (c) the coverage lifts from the measured 0.8903 prior.
    lift_central = c_star_central - COV_PRIOR           # TEST metric
    lift_worst = c_star_worst - COV_PRIOR
    central_within_budget = bool(lift_central <= RETRAIN_LIFT_BUDGET + TOL_EXACT)
    worst_within_budget = bool(lift_worst <= RETRAIN_LIFT_BUDGET + TOL_EXACT)
    return {
        "ppl_only_central_at_prior_0p8903": ppl_only_central_at_prior,
        "ppl_only_worst_at_prior_0p8903": ppl_only_worst_at_prior,
        "existing_head_gives_free_500_even_gate_lifted": existing_head_gives_free_500,
        "c_star_central_for_500": c_star_central,
        "c_star_worst_for_500": c_star_worst,
        "roundtrips_340_c_star_central": bool(abs(c_star_central - C_STAR_CENTRAL_340) <= TOL_340),
        "roundtrips_340_c_star_worst": bool(abs(c_star_worst - C_STAR_WORST_340) <= TOL_340),
        "root_residual_central": res_central,
        "root_residual_worst": res_worst,
        "roots_in_unit_interval": bool(0.0 < c_star_central < 1.0 and 0.0 < c_star_worst < 1.0),
        "central_root_valid": bool(_finite(c_star_central) and abs(res_central) <= TOL_ROOT),
        "worst_root_valid": bool(_finite(c_star_worst) and abs(res_worst) <= TOL_ROOT),
        "coverage_lift_for_ppl_only_central_500": lift_central,    # TEST metric (expect ~0.0186)
        "coverage_lift_for_ppl_only_worst_500": lift_worst,        # expect ~0.0353
        "ppl_only_central_500_within_budget": central_within_budget,   # expect True
        "ppl_only_worst_500_within_budget": worst_within_budget,       # expect False
        "retrain_lift_budget_336": RETRAIN_LIFT_BUDGET,
        "c_star_central_below_identity_bar": bool(c_star_central < IDENTITY_BAR),
        "note": ("at MEASURED coverage 0.8903 the PPL-only envelope is {:.2f}/{:.2f} -- BOTH < 500, so "
                 "the EXISTING head does NOT give a free >500 even gate-lifted. Solving env(c)=500 "
                 "gives c*_central={:.4f} (+{:.4f} from 0.8903, WITHIN +0.031) and c*_worst={:.4f} "
                 "(+{:.4f}, OVER +0.031). c*_central < identity bar 0.9213: PPL-only central-500 needs "
                 "LESS coverage than the strict identity bar AND no supply revival.".format(
                     ppl_only_central_at_prior, ppl_only_worst_at_prior,
                     c_star_central, lift_central, c_star_worst, lift_worst)),
    }


# --------------------------------------------------------------------------- #
# (D3) Verify the PPL gate holds in the gate-lifted config (cite wirbel #324; do not re-run).
# --------------------------------------------------------------------------- #
def deliverable3_ppl_gate_holds() -> dict[str, Any]:
    ppl_passes = bool(PPL_DEPLOYED <= PPL_GATE)
    return {
        "ppl_deployed": PPL_DEPLOYED,
        "ppl_gate": PPL_GATE,
        "ppl_passes_in_gate_lifted_config": ppl_passes,
        "ppl_margin": PPL_GATE - PPL_DEPLOYED,
        "decoupling_source": "wirbel #324 (pespixw1)",
        "decoupling_claim": (
            "wirbel #324 (pespixw1) showed the M=8 argmax divergence is structurally DECOUPLED from "
            "PPL: PPL is a prompt_logprobs reference-forward over FIXED token-IDs, so it passes by "
            "construction regardless of emission divergence (ppl_delta_under_eagle3_verify=0.0). PPL "
            "stays {:.4f} <= 2.42, M-binary (M2=M4=M6=M8 identical).".format(PPL_DEPLOYED)),
        "same_deployed_serve": (
            "the PPL-only config is the SAME deployed serve that ALREADY passes the official scorer "
            "(481.53 frontier, PPL 2.3772, 128/128). Lifting #192 introduces NO new PPL risk -- it "
            "removes a SELF-IMPOSED identity check the scorer never ran."),
        "served_greedy_rate_caveat": (
            "the literal served greedy-rate (the thing #192 would have measured) needs an HF Job "
            "(gated, ubel #322) -- NOT drawn here. This card prices the ENVELOPE consequence of the "
            "gate-lift, not a fresh served-token draw."),
    }


# --------------------------------------------------------------------------- #
# (D4) Crystallize the #124 delta -- one table + one sentence.
# --------------------------------------------------------------------------- #
def deliverable4_issue124_delta(d1: dict, d2: dict) -> dict[str, Any]:
    strict = d1["strict_world"]
    cc = d2["c_star_central_for_500"]
    cw = d2["c_star_worst_for_500"]
    table = [
        {
            "world": "STRICT (gate ON, #192)",
            "supply_tax": "denken #332 floor 0.09103 @ geometric phi (phi_realizable>=1)",
            "best_ceiling_tps": strict["strict_best_ceiling"],
            "500_status": "IMPOSSIBLE",
            "reason": "supply-capped at 473.5 < 500 even at perfect coverage; no retrain reaches it",
            "coverage_target": "N/A (supply-capped before coverage matters)",
        },
        {
            "world": "PPL-ONLY (gate LIFTED, #124)",
            "supply_tax": "0 (no determinism required)",
            "best_ceiling_tps": LAMBDA_CEIL,
            "500_status": "ACHIEVABLE",
            "reason": "demand-only envelope; reachable purely via coverage retrain",
            "coverage_target": "central c*={:.4f} (+{:.4f}, within +0.031); worst c*={:.4f} (+{:.4f}, marginal)".format(
                cc, d2["coverage_lift_for_ppl_only_central_500"], cw,
                d2["coverage_lift_for_ppl_only_worst_500"]),
        },
    ]
    one_sentence = (
        "Lifting #192 converts the >500 lane from IMPOSSIBLE under strict identity (supply-capped at "
        "473.5 TPS, no retrain reaches it) to ACHIEVABLE under PPL-only via a sized, feasible coverage "
        "retrain to c*_central=0.9089 (+0.0186, within lawine #336's +0.031 budget) / c*_worst=0.9256 "
        "(+0.0353, marginally past it).")
    impossible_to_feasible = bool(
        (not d1["strict_world"]["strict_500_reachable"])
        and d2["ppl_only_central_500_within_budget"])
    return {
        "delta_table": table,
        "one_sentence": one_sentence,
        "impossible_to_feasible_conversion": impossible_to_feasible,
        "strict_500_reachable": d1["strict_world"]["strict_500_reachable"],
        "ppl_only_500_reachable_via_coverage": d2["ppl_only_central_500_within_budget"],  # TEST metric
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def _selftests(d1: dict, d2: dict, d3: dict, d4: dict) -> dict[str, Any]:
    strict = d1["strict_world"]
    ppl = d1["ppl_only_world"]
    rt = ppl["anchor_roundtrip_identity"]
    h337 = ppl["honest_corner_roundtrip_337"]
    cc = d2["c_star_central_for_500"]
    cw = d2["c_star_worst_for_500"]
    conditions = {
        # (a) STRICT ceiling 473.5 round-trips denken #332 within tol; strict_500_reachable=False.
        "a_strict_ceiling_roundtrips_332": bool(
            strict["roundtrips_denken332_473p5"]
            and abs(strict["strict_ceiling_central"] - STRICT_CEILING_332) <= TOL_332),
        "a_strict_500_not_reachable": bool(not strict["strict_500_reachable"]),
        # (b) PPL-only envelope round-trips stark #340 anchors (520.95/492.87) and #337 corners
        #     (470.35/444.99 @0.8903) within tol.
        "b_ppl_only_anchor_roundtrip_340": bool(
            rt["matches_central_anchor"] and rt["matches_worst_anchor"]
            and abs(rt["envelope_central_at_identity"] - CENTRAL_ANCHOR) <= TOL_EXACT
            and abs(rt["envelope_worst_at_identity"] - WORST_ANCHOR) <= TOL_EXACT),
        "b_ppl_only_honest_corner_roundtrip_337": bool(
            h337["matches_337_central_470p35"] and h337["matches_337_worst_444p99"]
            and h337["both_below_500"]),
        # (c) c*_central=0.9089 / c*_worst=0.9256 reproduced (round-trip stark #340 + solve env=500).
        "c_c_star_reproduces_340": bool(
            d2["roundtrips_340_c_star_central"] and d2["roundtrips_340_c_star_worst"]
            and d2["central_root_valid"] and d2["worst_root_valid"]
            and abs(envelope_central(cc) - TARGET) <= TOL_ROOT
            and abs(envelope_worst(cw) - TARGET) <= TOL_ROOT),
        # (d) PPL-decoupling caveat carried (wirbel #324: PPL 2.3772 <= 2.42, decoupled, M-binary).
        "d_ppl_decoupling_caveat_carried": bool(
            d3["ppl_passes_in_gate_lifted_config"] and PPL_DEPLOYED <= PPL_GATE
            and "pespixw1" in d3["decoupling_source"]
            and "DECOUPLED" in d3["decoupling_claim"]),
        # (e) the #124 delta table is NaN-clean (set by caller) + the impossible->feasible conversion.
        "e_issue124_delta_nan_clean": True,
        "e_impossible_to_feasible": bool(d4["impossible_to_feasible_conversion"]),
        # (f) TEST: coverage_lift_for_ppl_only_central_500 ~ 0.0186 and within the +0.031 budget.
        "f_central_lift_within_budget": bool(
            d2["ppl_only_central_500_within_budget"]
            and abs(d2["coverage_lift_for_ppl_only_central_500"] - 0.0186) <= 5e-4),
        # (g) worst-500 lift ~ 0.0353 and OVER the +0.031 budget (marginal).
        "g_worst_lift_over_budget": bool(
            (not d2["ppl_only_worst_500_within_budget"])
            and abs(d2["coverage_lift_for_ppl_only_worst_500"] - 0.0353) <= 5e-4),
        # (h) TEST bool: ppl_only_500_reachable_via_coverage == True.
        "h_ppl_only_500_reachable_via_coverage": bool(d4["ppl_only_500_reachable_via_coverage"]),
        # (i) imports EXACT: constants match banked AND round to displayed forms.
        "i_imports_exact": bool(
            abs(IDENTITY_BAR - 0.9213) <= TOL_DISPLAY_C
            and abs(COV_PRIOR - 0.8903) <= TOL_DISPLAY_C
            and abs(CENTRAL_ANCHOR - 520.95) <= TOL_DISPLAY_TPS
            and abs(WORST_ANCHOR - 492.87) <= TOL_DISPLAY_TPS
            and abs(STRICT_CEILING_332 - 473.53) <= TOL_DISPLAY_TPS
            and abs(SUPPLY_FLOOR_GEO - 0.09103) <= 1e-5
            and abs(C_STAR_CENTRAL_340 - 0.9089) <= TOL_DISPLAY_C
            and abs(C_STAR_WORST_340 - 0.9256) <= TOL_DISPLAY_C),
        # (j) NaN-clean (set by caller).
        "j_nan_clean": True,
        # (k) roots in (0,1) (valid coverage).
        "k_roots_valid_coverage": bool(d2["roots_in_unit_interval"]),
        # (l) structural: PPL-only ceiling (520.95) strictly above the strict ceiling (473.5) -- the
        #     supply tax is exactly what the gate-lift removes.
        "l_ppl_only_ceiling_above_strict": bool(LAMBDA_CEIL > strict["strict_best_ceiling"]),
        # (m) c*_central below the strict identity bar (PPL-only central-500 needs LESS coverage).
        "m_c_star_central_below_identity_bar": bool(d2["c_star_central_below_identity_bar"]),
    }
    return {
        "conditions": conditions,
        "ppl_only_envelope_self_test_passes": bool(all(conditions.values())),
        "n_checks": len(conditions),
        "detail": {
            "strict_best_ceiling": strict["strict_best_ceiling"],
            "strict_500_reachable": strict["strict_500_reachable"],
            "c_star_central": cc, "c_star_worst": cw,
            "coverage_lift_for_ppl_only_central_500": d2["coverage_lift_for_ppl_only_central_500"],
            "ppl_only_500_reachable_via_coverage": d4["ppl_only_500_reachable_via_coverage"],
        },
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    d1 = deliverable1_two_worlds()
    d2 = deliverable2_price_and_lift()
    d3 = deliverable3_ppl_gate_holds()
    d4 = deliverable4_issue124_delta(d1, d2)
    st = _selftests(d1, d2, d3, d4)

    cc = d2["c_star_central_for_500"]
    cw = d2["c_star_worst_for_500"]
    handoff = (
        "LIFTING #192 (PPL-only world, #124) converts the >500 lane from IMPOSSIBLE under strict "
        "identity (supply-capped 473.5 < 500 at geometric phi, denken #332; no retrain reaches it) to "
        "ACHIEVABLE via a coverage retrain on the SAME already-fast deployed serve: c*_central={:.4f} "
        "(+{:.4f} from the measured 0.8903, WITHIN lawine #336's +0.031 budget) / c*_worst={:.4f} "
        "(+{:.4f}, marginally past it). The existing head does NOT give a free >500 even gate-lifted "
        "(470.35/444.99 @0.8903). PPL stays 2.3772 <= 2.42 (wirbel #324 decoupling) -- the gate-lift "
        "adds no PPL risk. The #124 deliverable: lifting #192 sizes the >500 path as a feasible "
        "coverage-retrain target, NOT a build/launch.".format(
            cc, d2["coverage_lift_for_ppl_only_central_500"], cw,
            d2["coverage_lift_for_ppl_only_worst_500"]))

    headline = {
        "ppl_only_envelope_self_test_passes": bool(st["ppl_only_envelope_self_test_passes"]),  # PRIMARY
        "coverage_lift_for_ppl_only_central_500": d2["coverage_lift_for_ppl_only_central_500"],  # TEST
        "ppl_only_500_reachable_via_coverage": d4["ppl_only_500_reachable_via_coverage"],        # TEST
        "strict_best_ceiling": d1["strict_world"]["strict_best_ceiling"],
        "strict_500_reachable": d1["strict_world"]["strict_500_reachable"],
        "c_star_central_for_500": cc,
        "c_star_worst_for_500": cw,
        "coverage_lift_for_ppl_only_worst_500": d2["coverage_lift_for_ppl_only_worst_500"],
        "ppl_only_central_500_within_budget": d2["ppl_only_central_500_within_budget"],
        "ppl_only_worst_500_within_budget": d2["ppl_only_worst_500_within_budget"],
        "impossible_to_feasible_conversion": d4["impossible_to_feasible_conversion"],
        "ppl_deployed": PPL_DEPLOYED,
    }
    return {
        "headline": headline,
        "deliverable1_two_worlds": d1,
        "deliverable2_price_and_lift": d2,
        "deliverable3_ppl_gate_holds": d3,
        "deliverable4_issue124_delta": d4,
        "self_test": st,
        "handoff": handoff,
        "imports": {
            "provenance": (
                "stark #340 jwv1vbug (demand-only envelope envelope_X(c)=X_ANCHOR*E[T](c)/E[T](0.9213), "
                "c*_central 0.9089 / c*_worst 0.9256, identity bar 0.9213) x stark #337 lbuirkpt "
                "(E[T](c)=1+sum_{d=1..7} c^d chain law, anchors central 520.95 / worst 492.87 "
                "@E[T]=6.1112, honest corners 470.35/444.99 @0.8903) x denken #332 y5cl0ena (supply "
                "floor 0.09103 @ geometric phi, strict ceiling 473.5, phi_realizable>=1, revive "
                "breakeven 0.255) x lawine #330 hfrscdai (cov prior 0.8903, identity bar 0.9213) x "
                "lawine #336 krroookz (+0.031 retrain budget) x wirbel #324 pespixw1 (PPL-decoupling "
                "2.3772, ppl_delta=0.0, M-binary). All run-ids in "
                "wandb-applied-ai-team/gemma-challenge-senpai."),
            "caveats": [
                "INVERTS stark #340 into the PPL-only (gate-lifted) world: re-prices NOTHING measured. "
                "The PPL-only envelope is stark #340's SAME demand-only map (the anchors already "
                "EXCLUDE the supply tax); the only change is that the strict world ADDITIONALLY pays "
                "denken #332's supply tax (cap 473.5) while the PPL-only world does NOT. No EAGLE-3 "
                "fusion checkpoint runs here; NOT a running EagleProposer.",
                "the deployed 481.53 is ALREADY the fast, non-deterministic, PPL-passing config (56% "
                "AR-divergent). Lifting #192 does NOT change the deployed config or give a free "
                "one-run jump to ~520 -- it means we never pay the determinism tax and the >500 path "
                "becomes a coverage retrain on the already-fast config. NOT a free win.",
                "STRICT-world ceiling 473.5 is the SUPREMUM over coverage (central corner = lambda "
                "ceiling, taxed). denken #332 proved phi_realizable>=1, so no deterministic schedule "
                "beats it; strict_500_reachable=False is robust, not a single-point estimate.",
                "PPL safety is STRUCTURAL (wirbel #324): PPL is a prompt_logprobs reference-forward "
                "over fixed token-IDs, decoupled from emission. The literal served greedy-rate would "
                "need an HF Job (gated, ubel #322) -- NOT drawn here.",
                "coverage c is per-depth effective acceptance c_eff (the E[T](c) axis), same units as "
                "the identity bar 0.9213, cov prior 0.8903, and lawine #336's +0.031. NOT a launch / "
                "build / served-file change / HF Job / submission / open2.",
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
    d1 = syn["deliverable1_two_worlds"]
    d2 = syn["deliverable2_price_and_lift"]
    d3 = syn["deliverable3_ppl_gate_holds"]
    d4 = syn["deliverable4_issue124_delta"]
    st = syn["self_test"]
    strict = d1["strict_world"]
    print("\n" + "=" * 98, flush=True)
    print("PPL-ONLY GATE (#124) — what LIFTING greedy-identity (#192) buys on >500 (PR #343, wirbel)", flush=True)
    print("=" * 98, flush=True)
    print("  (D1) TWO WORLDS", flush=True)
    print(f"      STRICT (gate ON):  best ceiling = 520.95*(1-{strict['supply_floor_geometric_phi']:.5f}) "
          f"= {strict['strict_best_ceiling']:.2f}  -> 500 reachable: {strict['strict_500_reachable']} "
          f"(round-trips #332: {strict['roundtrips_denken332_473p5']})", flush=True)
    print(f"      PPL-ONLY (lifted): supply tax = 0  -> demand-only envelope (anchors 520.95/492.87), "
          f"gate = PPL<=2.42 only", flush=True)
    print("-" * 98, flush=True)
    print("  (D2) PRICE @ MEASURED 0.8903 + SOLVE THE LIFT  (PRIMARY/TEST)", flush=True)
    print(f"      PPL-only @0.8903: central={d2['ppl_only_central_at_prior_0p8903']:.2f}  "
          f"worst={d2['ppl_only_worst_at_prior_0p8903']:.2f}  "
          f"(free >500? {d2['existing_head_gives_free_500_even_gate_lifted']})", flush=True)
    print(f"      c*_central = {d2['c_star_central_for_500']:.6f}  lift = "
          f"+{d2['coverage_lift_for_ppl_only_central_500']:.4f}  within +0.031: "
          f"{d2['ppl_only_central_500_within_budget']}", flush=True)
    print(f"      c*_worst   = {d2['c_star_worst_for_500']:.6f}  lift = "
          f"+{d2['coverage_lift_for_ppl_only_worst_500']:.4f}  within +0.031: "
          f"{d2['ppl_only_worst_500_within_budget']}", flush=True)
    print("-" * 98, flush=True)
    print("  (D3) PPL GATE IN GATE-LIFTED CONFIG  (wirbel #324 pespixw1)", flush=True)
    print(f"      PPL deployed = {d3['ppl_deployed']:.4f} <= gate {d3['ppl_gate']:.2f}: "
          f"{d3['ppl_passes_in_gate_lifted_config']}  (decoupled, M-binary; SAME deployed serve)", flush=True)
    print("-" * 98, flush=True)
    print("  (D4) THE #124 DELTA", flush=True)
    for row in d4["delta_table"]:
        print(f"      {row['world']:<28} ceiling={row['best_ceiling_tps']:.2f}  "
              f"500={row['500_status']}", flush=True)
    print(f"      impossible->feasible conversion: {d4['impossible_to_feasible_conversion']}", flush=True)
    print(f"      >> {d4['one_sentence']}", flush=True)
    print("-" * 98, flush=True)
    print(f"  HAND-OFF: {syn['handoff']}", flush=True)
    print("-" * 98, flush=True)
    print(f"  PRIMARY ppl_only_envelope_self_test_passes = "
          f"{st['ppl_only_envelope_self_test_passes']} ({st['n_checks']} checks)", flush=True)
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
        print(f"[ppl-only-gate-500-envelope] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    d1 = syn["deliverable1_two_worlds"]
    d2 = syn["deliverable2_price_and_lift"]
    d3 = syn["deliverable3_ppl_gate_holds"]
    d4 = syn["deliverable4_issue124_delta"]
    st = syn["self_test"]
    strict = d1["strict_world"]
    run = init_wandb_run(
        job_type="ppl-only-gate-500-envelope",
        agent="wirbel",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["ppl-only-gate-500-envelope", "issue-124", "issue-192", "issue-319", "eagle3",
              "gate-lift", "demand-envelope", "supply-tax", "compliant-500", "validity-gate",
              "bank-the-analysis"],
        config={
            "K_spec": K_SPEC, "identity_bar": IDENTITY_BAR, "cov_prior": COV_PRIOR,
            "central_anchor": CENTRAL_ANCHOR, "worst_anchor": WORST_ANCHOR,
            "lambda_ceil": LAMBDA_CEIL, "e_t_at_identity": E_T_AT_IDENTITY,
            "supply_floor_geometric_phi": SUPPLY_FLOOR_GEO, "strict_ceiling_332": STRICT_CEILING_332,
            "supply_revive_breakeven": SUPPLY_REVIVE_BREAKEVEN,
            "c_star_central_340": C_STAR_CENTRAL_340, "c_star_worst_340": C_STAR_WORST_340,
            "retrain_lift_budget_336": RETRAIN_LIFT_BUDGET,
            "ppl_deployed": PPL_DEPLOYED, "ppl_gate": PPL_GATE,
            "target": TARGET, "baseline_tps": BASELINE_TPS, "wandb_group": args.wandb_group,
            "source_runs": ("stark#340(jwv1vbug), stark#337(lbuirkpt), denken#332(y5cl0ena), "
                            "lawine#330(hfrscdai), lawine#336(krroookz), wirbel#324(pespixw1)"),
        },
    )
    if run is None:
        print("[ppl-only-gate-500-envelope] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "ppl_only_envelope_self_test_passes": int(bool(st["ppl_only_envelope_self_test_passes"])),  # PRIMARY
        "coverage_lift_for_ppl_only_central_500": d2["coverage_lift_for_ppl_only_central_500"],      # TEST
        "ppl_only_500_reachable_via_coverage": int(bool(d4["ppl_only_500_reachable_via_coverage"])),  # TEST
        "strict_best_ceiling": strict["strict_best_ceiling"],
        "strict_ceiling_central": strict["strict_ceiling_central"],
        "strict_500_reachable": int(bool(strict["strict_500_reachable"])),
        "strict_ceiling_gap_to_500": strict["strict_ceiling_gap_to_500"],
        "supply_floor_geometric_phi": SUPPLY_FLOOR_GEO,
        "ppl_only_central_at_prior_0p8903": d2["ppl_only_central_at_prior_0p8903"],
        "ppl_only_worst_at_prior_0p8903": d2["ppl_only_worst_at_prior_0p8903"],
        "existing_head_gives_free_500": int(bool(d2["existing_head_gives_free_500_even_gate_lifted"])),
        "c_star_central_for_500": d2["c_star_central_for_500"],
        "c_star_worst_for_500": d2["c_star_worst_for_500"],
        "coverage_lift_for_ppl_only_worst_500": d2["coverage_lift_for_ppl_only_worst_500"],
        "ppl_only_central_500_within_budget": int(bool(d2["ppl_only_central_500_within_budget"])),
        "ppl_only_worst_500_within_budget": int(bool(d2["ppl_only_worst_500_within_budget"])),
        "c_star_central_below_identity_bar": int(bool(d2["c_star_central_below_identity_bar"])),
        "root_residual_central": d2["root_residual_central"],
        "root_residual_worst": d2["root_residual_worst"],
        "ppl_deployed": PPL_DEPLOYED,
        "ppl_passes_in_gate_lifted_config": int(bool(d3["ppl_passes_in_gate_lifted_config"])),
        "ppl_margin": d3["ppl_margin"],
        "impossible_to_feasible_conversion": int(bool(d4["impossible_to_feasible_conversion"])),
        "retrain_lift_budget_336": RETRAIN_LIFT_BUDGET,
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="ppl_only_gate_500_envelope_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[ppl-only-gate-500-envelope] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="ppl-only-gate-500-envelope")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 343, "agent": "wirbel",
        "kind": "ppl-only-gate-500-envelope", "analysis_only": True, "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    # propagate NaN-clean into the two self-test conditions that assert it.
    syn["self_test"]["conditions"]["e_issue124_delta_nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["j_nan_clean"] = not nan_paths
    syn["self_test"]["ppl_only_envelope_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    syn["headline"]["ppl_only_envelope_self_test_passes"] = syn["self_test"][
        "ppl_only_envelope_self_test_passes"]
    if nan_paths:
        print(f"[ppl-only-gate-500-envelope] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ppl_only_gate_500_envelope_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[ppl-only-gate-500-envelope] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["ppl_only_envelope_self_test_passes"] and payload["nan_clean"])
        print(f"[ppl-only-gate-500-envelope] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
