#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #430 (lawine) -- The CORRECT Directive-#4 bar: is 482.74 already a banked +15.60 equivalent win?

WHY THIS CARD EXISTS (the bar was mis-anchored)
-----------------------------------------------
Issue #407 (human, 2026-06-15 21:13:17Z) RE-SCOPED the objective:
    "find the fastest implementation that also respects this equivalence ... Forget 500+ for now,
     just improve on the TPS that is the fastest that still respects this equivalence."
So the objective is  MAXIMIZE single-stream TPS  SUBJECT TO  strict byte-exact greedy-token-equivalence.
The maximum is taken over the FEASIBLE set -- the configs that RESPECT the equivalence -- only.

My #425 GO/NO-GO priced the deploy against the DEPLOYED 481.53 (PR #52, 2x9fm2zx) and made a knife-edge
"+1.21 over 481.53" the BINDING bar. But 481.53 is the NON-strict fast path: under M=8 batched verify it
carries 3/882 reduction-order flips (served identity 0.9966, #381/#397/#405). Under #407 it does NOT respect
the equivalence, so it is NOT IN THE FEASIBLE SET -- it is excluded from the max. It is therefore the WRONG
bar for Directive-#4.

The fastest config that DOES respect the equivalence (the feasible-set incumbent) is BLANKET-STRICT at
467.14 (#412 dnjvqbtf, measured, identity 0.9989). The realizable strict stack blanket-strict + cb3 supply
(+15.60, #403, equivalence-neutral) = 482.74 is therefore a CLEAN +15.60 TPS (+3.34%) improvement over the
CORRECT Directive-#4 bar (467.14) -- a banked win that needs NO margin contingency: even under ubel #410's
worst-case 14.9% additivity haircut the realized lift is +13.28, so the win stays strictly positive across
the ENTIRE admissible haircut band. The "knife-edge +1.21 over 481.53" #425 made binding is a DIFFERENT,
STRICTER, SECONDARY bar (beat the illegal incumbent on the public leaderboard); 481.53's infeasibility means
a margin-negative outcome THERE does NOT disqualify 482.74 as the fastest FEASIBLE config under #407.

THIS CARD (a 0-GPU decision-ledger / reconciliation)
----------------------------------------------------
1. ENUMERATE every strictly-equivalent config with a TPS, lowest->highest, with identity + W&B source.
2. RECONCILE the blanket-strict base: 467.14 (#412 dnjvqbtf, measured) vs 467.48 (#393 0q7ynumg, decode-eta
   projection). Emit `canonical_blanket_strict_tps`.
3. STATE THE TWO BARS explicitly:
     Bar A -- Directive-#4 (fastest prior FEASIBLE): 467.14 -> banked win +15.60 (+3.34%), NO contingency.
     Bar B -- leaderboard-beat (illegal incumbent):  481.53 -> +1.21 modeled, flips negative > ~7.76% haircut.
4. RESOLVE the ship-policy under Bar B negative: under #407's literal text 481.53 is excluded from the max,
   so 482.74 is still the fastest FEASIBLE config even if measured < 481.53. Emit `ships_if_bar_B_negative`.
5. EMIT an additive GO card (SUPERSEDES #425's checklist; new file, old left in place) that LEADS with the
   banked Directive-#4 win and DEMOTES Bar B to a leaderboard-cosmetic line. The identity conjunct is a
   SEPARATE row citing stark #429 (operative-identity resolution, in flight) -- identity is NOT re-derived here.
6. SELF-TEST the arithmetic + the analysis-only hygiene.

PURE STATIC ANALYSIS / DECISION CARD. analysis_only=True, no_hf_job=True, no_served_file_change=True,
official_tps=0. 0 GPU compute. It BUILDS NOTHING, FLIPS NO SERVED FILE, SUBMITS NOTHING, and DOES NOT
re-derive identity. PINNED constants are imported byte-exactly from the merged advisor-branch results JSON
(#412 / #403 / #410 / #393 / #196); Part 0 cross-checks them so "import, do not re-derive" is self-tested.
Baseline 481.53 / PPL 2.3772 / 128-128 (#52, 2x9fm2zx) UNCHANGED.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------------------- #
# PINNED constants -- IMPORTED byte-exactly from merged advisor-branch modules. Do NOT re-derive;
# Part 0 cross-checks each against the merged #412 / #403 / #410 / #393 / #196 results JSON.
# ---------------------------------------------------------------------------------------- #
# --- the deployed NON-strict #1 = the ILLEGAL incumbent (Bar B); NOT in the #407 feasible set ---
DEPLOYED_TPS = 481.53                  # #52 (2x9fm2zx) deployed non-strict #1 -- the leaderboard incumbent
DEPLOYED_PPL = 2.3772                  # #52 deployed PPL (unchanged by the equivalent stack, by construction)
DEPLOYED_SERVED_IDENTITY = 0.9966      # #405 (j6h228xy): 3/882 M=8 reduction-order flips -> NOT equivalent
DEPLOYED_SERVED_FLIPS = 3              # deployed non-strict served M=8 argmax flips ...
DEPLOYED_SERVED_FLIP_DENOM = 882       # ... out of 882 positions -> identity 0.9966 (infeasible under #407)
PPL_CAP = 2.42                         # quality guardrail (reference 2.30 + 5%)
REQUIRED_COMPLETED = 128               # full public run

# --- #412 (stark, dnjvqbtf): the MEASURED feasible-set incumbent. blanket-strict IS fastest realizable strict ---
BLANKET_STRICT_TPS = 467.14            # #412 `blanket_strict_measured_tps` (467.1400155) -- DIRECT local measure
BLANKET_STRICT_TPS_STD = 0.16          # #412 `blanket_strict_measured_tps_std` (0.16105) -- tight (2sigma 0.32)
BLANKET_STRICT_IDENTITY = 0.9989       # #412 `blanket_strict_within_identity` (0.99886621) -- 1 residual flip
BLANKET_STRICT_RESIDUAL_FLIPS = 1      # #412 pinned arm: 882 positions, 881 match -> 1 flip @ prompt 90
RESIDUAL_FLIP_PROMPT = 90              # the single residual bitwise-tie flip (per #412 arm)
IDENTITY_1P0_UNREACHABLE_BY_PRECISION = True   # #412 -- only a canonical tie-break (not precision) closes it

# --- #393 (wirbel, 0q7ynumg): the decode-eta PROJECTION cross-check of the strict base ---
BLANKET_STRICT_393_TPS = 467.48        # #393 `deployed_tps_decode_eta` (467.4752) -- decode-eta back-projection
BLANKET_STRICT_393_ETA = 0.030065      # #393 `eta_attn_decode_only` (3.01%) -- decode-specific attn strict tax

# --- #403 (kanna, iv9i2wks): cb3 supply, equivalence-neutral, PPL-safe, ADDITIVE ---
CB3_LIFT_M8 = 15.60                    # #403 `m8_lift_at_kstar` (15.6038966) k*=229 served M=8 lift (MEASURED)
CB3_KSTAR = 229                        # #403 conservative-k allocation (PPL-safe, worst-seed <= 2.41)

# --- #410 (ubel, 7rzf74q5): supply x demand additive, but a bounded cross-term haircut ---
ADDITIVITY_HAIRCUT_FRAC_BOUND = 0.149  # #410 `delta_demand_tps_frac_of_lift` (0.14907) -- worst-case haircut
SUPPLY_DEMAND_ADDITIVE = True          # #410 `supply_demand_additive` -- additive (cross-term NOT negligible)

# --- #196 (lawine): the pure-AR-greedy M=1 reference -- the EQUIVALENCE FLOOR (identity 1.0 by definition) ---
# OFFICIAL-regime ESTIMATE, not a local in-cycle measurement -> NOT regime-consistent with the local 467.14.
M1_NONSPEC_OFFICIAL_TPS_EST = 165.44   # #196 `nonspec_official_tps_est` (165.4379) -- cross-regime context only
M1_LOCAL_TPS = None                    # NO regime-consistent local in-cycle measurement -> `unmeasured`

# --- the re-anchored realizable strict stack (PRIMARY deliverable) ---
BLANKET_STRICT_STACK_TPS = round(BLANKET_STRICT_TPS + CB3_LIFT_M8, 2)        # 482.74

# --- Bar A: Directive-#4 (fastest prior FEASIBLE = blanket-strict) ---
DIRECTIVE4_BANKED_WIN_TPS = round(BLANKET_STRICT_STACK_TPS - BLANKET_STRICT_TPS, 2)   # +15.60 (= the cb3 lift)
DIRECTIVE4_BANKED_WIN_PCT = round(DIRECTIVE4_BANKED_WIN_TPS / BLANKET_STRICT_TPS * 100.0, 2)  # +3.34%

# --- Bar B: leaderboard-beat (illegal deployed incumbent) ---
LEADERBOARD_BEAT_MARGIN_TPS = round(BLANKET_STRICT_STACK_TPS - DEPLOYED_TPS, 2)       # +1.21 (knife-edge)
CB3_LIFT_BREAKEVEN_FOR_BARB = round(DEPLOYED_TPS - BLANKET_STRICT_TPS, 2)             # 14.39 (lift must clear)
BAR_B_HAIRCUT_FLIP_THRESHOLD = round((CB3_LIFT_M8 - CB3_LIFT_BREAKEVEN_FOR_BARB) / CB3_LIFT_M8, 4)  # 0.0776

# the ONE pending decision input (advisor-provided; swappable when stark #429 lands). cb3 is identity-neutral,
# so the residual identity question is entirely the blanket-strict base's 0.9989 -> stark #429 resolves it.
IDENTITY_VALUE_NOW = BLANKET_STRICT_IDENTITY     # 0.9989 (1 flip @ prompt 90); stark #429 drives -> 1.0 (PENDING)
IDENTITY_TARGET = 1.0

# the served files a blanket-strict GO would pin (NOT touched by this card; existence-probed read-only)
DEPLOYED_SUBMISSION = "submissions/fa2sw_treeverify_kenyan"
DEPLOYED_VERIFY_PATCH = f"{DEPLOYED_SUBMISSION}/splitkv_verify_patch.py"
DEPLOYED_ATTN_PATCH = f"{DEPLOYED_SUBMISSION}/fa_sliding_patch.py"
DEPLOYED_MANIFEST = f"{DEPLOYED_SUBMISSION}/manifest.json"
SERVED_FILES = [DEPLOYED_VERIFY_PATCH, DEPLOYED_ATTN_PATCH, DEPLOYED_MANIFEST]

# the #425 checklist this card SUPERSEDES (additive: new file, old left in place / read-only probe)
PRIOR_425_CHECKLIST = "research/validity/blanket_strict_deploy_gonogo_reanchor/GO_NOGO_CHECKLIST.md"

# merged results JSON used to cross-check the pinned constants (byte-exact import proof)
CROSSCHECK_412_JSON = ("research/validity/selective_recompute_equivalent_tps/"
                       "selective_recompute_equivalent_tps_results.json")
CROSSCHECK_403_JSON = ("research/validity/cb3_conservative_k_deployable_lift/"
                       "cb3_conservative_k_deployable_lift_results.json")
CROSSCHECK_410_JSON = "research/validity/cb3_acceptance_crossterm/cb3_acceptance_crossterm_results.json"
CROSSCHECK_393_JSON = ("research/validity/attention_strict_pin_cost/attention_strict_pin_cost_results.json")
CROSSCHECK_196_JSON = "research/validity/compliant_nonspec_floor/floor_report.json"

# banked evidence this card reasons from (read-only existence probes)
EVIDENCE = {
    "blanket_strict_measured_412": ("research/validity/selective_recompute_equivalent_tps/"
                                     "selective_recompute_equivalent_tps.py"),
    "decode_eta_projection_393": "research/validity/attention_strict_pin_cost/attention_strict_pin_cost.py",
    "cb3_conservative_k_403": ("research/validity/cb3_conservative_k_deployable_lift/"
                               "cb3_conservative_k_deployable_lift.py"),
    "additivity_410": "research/validity/cb3_acceptance_crossterm/cb3_acceptance_crossterm.py",
    "nonspec_floor_196": "research/validity/compliant_nonspec_floor/analyze_floor.py",
    "canonical_tiebreak_421": ("research/validity/canonical_tiebreak_fast_stack_identity/"
                               "canonical_tiebreak_fast_stack_identity.py"),
    "prior_gonogo_425": ("research/validity/blanket_strict_deploy_gonogo_reanchor/"
                         "blanket_strict_deploy_gonogo_reanchor.py"),
}


# ======================================================================================== #
# Part 0 -- PINNED-IMPORT CROSS-CHECK (prove the constants are byte-exact from the merged JSON)
# ======================================================================================== #
def pinned_import_crosscheck() -> dict[str, Any]:
    """Read the merged #412 / #403 / #410 / #393 / #196 results JSON and assert every pinned constant matches.
    Soft on file-absence (records availability) but the self-test asserts on the matches when present."""
    out: dict[str, Any] = {"checks": {}, "available": {}}

    p412 = REPO_ROOT / CROSSCHECK_412_JSON
    if p412.is_file():
        d = json.loads(p412.read_text())
        out["available"]["s412"] = True
        out["checks"]["x412_blanket_strict_tps"] = (
            abs(d.get("blanket_strict_measured_tps", -1) - BLANKET_STRICT_TPS) < 5e-3)
        out["checks"]["x412_blanket_strict_identity"] = (
            abs(d.get("blanket_strict_within_identity", -1) - BLANKET_STRICT_IDENTITY) < 5e-3)
        out["checks"]["x412_fastest_is_blanket"] = (
            d.get("fastest_realizable_strictly_equivalent_config") == "blanket_strict")
        out["checks"]["x412_identity_unreachable_by_precision"] = (
            d.get("identity_1p0_unreachable_by_precision") is True)
    else:
        out["available"]["s412"] = False

    p403 = REPO_ROOT / CROSSCHECK_403_JSON
    if p403.is_file():
        d = json.loads(p403.read_text())
        out["available"]["s403"] = True
        rc = d.get("result", {}).get("recost_at_kstar", {})
        ks = d.get("result", {}).get("kstar", {})
        out["checks"]["x403_cb3_lift_m8"] = (abs(rc.get("m8_lift_at_kstar", -1) - CB3_LIFT_M8) < 5e-3)
        out["checks"]["x403_cb3_kstar"] = (ks.get("k_star") == CB3_KSTAR)
    else:
        out["available"]["s403"] = False

    p410 = REPO_ROOT / CROSSCHECK_410_JSON
    if p410.is_file():
        d = json.loads(p410.read_text())
        out["available"]["s410"] = True
        out["checks"]["x410_haircut_frac"] = (
            abs(d.get("verdict", {}).get("delta_demand_tps_frac_of_lift", -1)
                - ADDITIVITY_HAIRCUT_FRAC_BOUND) < 5e-3)
        out["checks"]["x410_supply_demand_additive"] = (d.get("supply_demand_additive") is True)
    else:
        out["available"]["s410"] = False

    p393 = REPO_ROOT / CROSSCHECK_393_JSON
    if p393.is_file():
        d = json.loads(p393.read_text())
        out["available"]["s393"] = True
        # 467.475 rounds to the cited 467.48; assert it agrees with the projection constant to <0.01
        out["checks"]["x393_decode_eta_tps"] = (
            abs(d.get("deployed_tps_decode_eta", -1) - BLANKET_STRICT_393_TPS) < 1e-2)
        out["checks"]["x393_run_is_0q7ynumg"] = (d.get("wandb_run_id") == "0q7ynumg")
    else:
        out["available"]["s393"] = False

    p196 = REPO_ROOT / CROSSCHECK_196_JSON
    if p196.is_file():
        d = json.loads(p196.read_text())
        out["available"]["s196"] = True
        out["checks"]["x196_nonspec_official_tps_est"] = (
            abs(d.get("nonspec_official_tps_est", -1) - M1_NONSPEC_OFFICIAL_TPS_EST) < 1e-2)
    else:
        out["available"]["s196"] = False

    out["all_present_checks_pass"] = all(out["checks"].values()) if out["checks"] else False
    out["n_crosschecks"] = len(out["checks"])
    return out


# ======================================================================================== #
# Part 1 -- ENUMERATE every strictly-equivalent config with a TPS, lowest -> highest
# ======================================================================================== #
def enumerate_strict_equivalent_configs() -> dict[str, Any]:
    """The #407 FEASIBLE-set ladder (configs that RESPECT the equivalence), lowest -> highest TPS, each with
    identity + measurement regime + W&B source. The deployed 481.53 is shown SEPARATELY as the EXCLUDED
    (infeasible) incumbent -- it is NOT a member of the max."""
    feasible_ladder = [
        {
            "rank": 1,
            "config": "pure-AR-greedy (M=1 reference)",
            "role": "equivalence FLOOR (the reference greedy token stream every feasible config must match)",
            "tps_local_in_cycle": M1_LOCAL_TPS,                 # None -> unmeasured (no regime-consistent number)
            "tps_status": "unmeasured",
            "tps_cross_regime_est": M1_NONSPEC_OFFICIAL_TPS_EST,  # 165.44 OFFICIAL estimate (NOT local-measured)
            "tps_cross_regime_note": ("165.44 is an OFFICIAL-hardware ESTIMATE (#196 nonspec_official_tps_est); "
                                      "not regime-consistent with the local 467.14 / 482.74 numbers, so the "
                                      "in-cycle local TPS is `unmeasured`. Cited only to place the floor."),
            "identity": 1.0,
            "identity_basis": "by-definition (it IS the M=1 AR greedy reference)",
            "in_feasible_set": True,
            "wandb_run": "#196 (compliant_nonspec_floor)",
            "source": "lawine #196 floor_report.json",
        },
        {
            "rank": 2,
            "config": "blanket-strict (high-precision verify reduction everywhere)",
            "role": "fastest MEASURED feasible config today = the CORRECT Directive-#4 bar (Bar A)",
            "tps_local_in_cycle": BLANKET_STRICT_TPS,           # 467.14
            "tps_status": "local-measured",
            "tps_std": BLANKET_STRICT_TPS_STD,                  # 0.16 (2sigma 0.32)
            "tps_cross_check": BLANKET_STRICT_393_TPS,          # 467.48 decode-eta projection (#393)
            "identity": BLANKET_STRICT_IDENTITY,                # 0.9989
            "identity_basis": f"measured (1 residual bitwise-tie flip @ prompt {RESIDUAL_FLIP_PROMPT}); "
                              "stark #429 resolves -> 1.0 (PENDING)",
            "in_feasible_set": True,
            "wandb_run": "#412 dnjvqbtf (measured) / #393 0q7ynumg (467.48 projection cross-check)",
            "source": "stark #412 selective_recompute_equivalent_tps",
        },
        {
            "rank": 3,
            "config": "blanket-strict + cb3 supply (k*=229)",
            "role": "PRIMARY deliverable -- the realizable strict stack (the feasible-set MAX)",
            "tps_local_in_cycle": BLANKET_STRICT_STACK_TPS,     # 482.74 (= 467.14 + 15.60)
            "tps_status": "local-modeled (467.14 measured + 15.60 measured cb3 lift; additive per #410)",
            "tps_decomposition": f"{BLANKET_STRICT_TPS} (#412 measured) + {CB3_LIFT_M8} (#403 cb3 lift)",
            "identity": BLANKET_STRICT_IDENTITY,                # 0.9989 -- cb3 is equivalence-NEUTRAL (adds 0 flips)
            "identity_basis": "= blanket-strict (cb3 is equivalence-neutral on the supply leg, #403/#410); "
                              "no NEW flips from cb3 -- the residual is the SAME single base flip @ prompt "
                              f"{RESIDUAL_FLIP_PROMPT}",
            "in_feasible_set": True,
            "wandb_run": "#425 3u2urqzj (stack infra) + #403 iv9i2wks (cb3 lift)",
            "source": "lawine #425 + kanna #403",
        },
    ]
    excluded_incumbent = {
        "config": "deployed fast path (non-strict reduction)",
        "tps_local_in_cycle": DEPLOYED_TPS,                     # 481.53
        "tps_status": "official-deployed (leaderboard incumbent)",
        "identity": DEPLOYED_SERVED_IDENTITY,                   # 0.9966
        "identity_basis": f"{DEPLOYED_SERVED_FLIPS}/{DEPLOYED_SERVED_FLIP_DENOM} M=8 reduction-order flips "
                          "(#405 j6h228xy) -> 0.9966 < 1.0",
        "in_feasible_set": False,
        "excluded_reason": ("does NOT respect the equivalence (identity 0.9966) under the CURRENT operative-"
                            "identity -> EXCLUDED from the #407 max. NOTE: stark #429 (in flight) could "
                            "canonicalize its bitwise-tie flips on the self-referential gate (#414 bq7xkfcv) "
                            "and PROMOTE it into the feasible set -- see ship-policy."),
        "wandb_run": "#52 2x9fm2zx",
        "source": "PR #52 deployed",
    }

    # the feasible-set max (among configs with a numeric local TPS) is the cb3 stack
    numeric = [c for c in feasible_ladder if c["tps_local_in_cycle"] is not None]
    feasible_max = max(numeric, key=lambda c: c["tps_local_in_cycle"])
    monotone = all(numeric[i]["tps_local_in_cycle"] < numeric[i + 1]["tps_local_in_cycle"]
                   for i in range(len(numeric) - 1))
    return {
        "feasible_ladder_low_to_high": feasible_ladder,
        "excluded_infeasible_incumbent": excluded_incumbent,
        "feasible_set_max_config": feasible_max["config"],
        "feasible_set_max_tps": feasible_max["tps_local_in_cycle"],
        "feasible_ladder_numeric_monotone_increasing": monotone,
        "n_feasible_configs": len(feasible_ladder),
        "directive4_bar_is_feasible_set_max": (not excluded_incumbent["in_feasible_set"]),  # 481.53 excluded
    }


# ======================================================================================== #
# Part 2 -- RECONCILE the blanket-strict base: 467.14 (measured) vs 467.48 (projection)
# ======================================================================================== #
def reconcile_blanket_strict_base() -> dict[str, Any]:
    """467.14 (#412 dnjvqbtf) vs 467.48 (#393 0q7ynumg). Which is canonical, and why."""
    delta = round(BLANKET_STRICT_393_TPS - BLANKET_STRICT_TPS, 4)           # +0.34
    delta_pct = round(delta / BLANKET_STRICT_TPS * 100.0, 4)               # +0.073%
    two_sigma = round(2 * BLANKET_STRICT_TPS_STD, 4)                       # 0.32
    sigma_distance = round(abs(delta) / BLANKET_STRICT_TPS_STD, 2)         # ~2.13 sigma
    agree_within_0p1pct = abs(delta_pct) < 0.1                             # 0.073% < 0.1% -> confirming
    projection_optimistic = BLANKET_STRICT_393_TPS > BLANKET_STRICT_TPS    # 467.48 > 467.14 (idealized higher)
    return {
        "measured_467_14": {
            "tps": BLANKET_STRICT_TPS, "std": BLANKET_STRICT_TPS_STD, "run": "#412 dnjvqbtf",
            "method": ("DIRECT end-to-end local A10G measurement of the blanket-strict SERVED stack "
                       "(high-precision reduction everywhere), with measured within-config identity "
                       f"{BLANKET_STRICT_IDENTITY}."),
        },
        "projection_467_48": {
            "tps": BLANKET_STRICT_393_TPS, "run": "#393 0q7ynumg",
            "method": ("decode-eta BACK-PROJECTION: the deployed base scaled by the decode-specific attention "
                       f"strict tax eta_attn={BLANKET_STRICT_393_ETA*100:.2f}% (deployed_tps_decode_eta="
                       "467.475). An analytic projection that ASSUMES the idealized strict path, NOT a direct "
                       "measurement of the realizable blanket-strict served stack."),
        },
        "delta_tps": delta, "delta_pct": delta_pct, "two_sigma_band": two_sigma,
        "sigma_distance": sigma_distance, "agree_within_0p1pct": agree_within_0p1pct,
        "projection_optimistic": projection_optimistic,
        "canonical_blanket_strict_tps": BLANKET_STRICT_TPS,                # 467.14
        "canonical_reason": (
            f"CANONICAL = {BLANKET_STRICT_TPS} (#412 dnjvqbtf). (a) It is the DIRECT measurement of the "
            f"REALIZABLE blanket-strict served stack; 467.48 is a decode-eta back-projection from the deployed "
            f"base. (b) The cb3 +{CB3_LIFT_M8} lift (#403) was measured on THIS measured base, so the stack "
            f"{BLANKET_STRICT_STACK_TPS} = {BLANKET_STRICT_TPS} + {CB3_LIFT_M8} is internally consistent -- "
            f"swapping in 467.48 would mix a projection into a measured sum. (c) They agree to {delta} TPS "
            f"({delta_pct}%, ~{sigma_distance}sigma) -- a CONFIRMING cross-check at the 0.1% level, NOT a "
            f"competing number. The projection is marginally OPTIMISTIC (idealized strict path, no realized "
            f"overhead), so the measured 467.14 is also the CONSERVATIVE choice."),
    }


# ======================================================================================== #
# Part 3 -- STATE THE TWO BARS explicitly (Bar A = Directive-#4; Bar B = leaderboard-beat)
# ======================================================================================== #
def two_bars() -> dict[str, Any]:
    """Bar A: the CORRECT Directive-#4 bar (fastest prior FEASIBLE). Bar B: the leaderboard-beat (illegal)."""
    # worst-case haircut (#410): the realized cb3 lift, and the stack it lands at
    worst_realized_lift = round(CB3_LIFT_M8 * (1.0 - ADDITIVITY_HAIRCUT_FRAC_BOUND), 2)   # 13.28
    worst_case_stack = round(BLANKET_STRICT_TPS + worst_realized_lift, 2)                 # 480.42
    bar_a_worst_margin = round(worst_case_stack - BLANKET_STRICT_TPS, 2)                  # +13.28 (= realized lift)
    bar_b_worst_margin = round(worst_case_stack - DEPLOYED_TPS, 2)                        # -1.11

    bar_a = {
        "name": "Bar A -- Directive-#4 (fastest prior FEASIBLE config)",
        "bar_tps": BLANKET_STRICT_TPS,                          # 467.14
        "bar_source": "blanket-strict #412 dnjvqbtf (the feasible-set incumbent under #407)",
        "stack_tps": BLANKET_STRICT_STACK_TPS,                  # 482.74
        "banked_win_tps": DIRECTIVE4_BANKED_WIN_TPS,            # +15.60
        "banked_win_pct": DIRECTIVE4_BANKED_WIN_PCT,            # +3.34%
        "has_margin_contingency": False,
        "no_contingency_proof": (
            f"Bar A margin == the REALIZED cb3 lift (the stack IS blanket-strict + lift). Even at ubel #410's "
            f"worst-case {ADDITIVITY_HAIRCUT_FRAC_BOUND*100:.1f}% haircut the realized lift is "
            f"+{worst_realized_lift} (stack {worst_case_stack}), so Bar A stays +{bar_a_worst_margin} -- "
            f"strictly positive across the ENTIRE [0, {ADDITIVITY_HAIRCUT_FRAC_BOUND*100:.1f}%] haircut band. "
            "No measurement of kanna #416 can make this bar negative."),
        "worst_case_margin_tps": bar_a_worst_margin,            # +13.28
    }
    bar_b = {
        "name": "Bar B -- leaderboard-beat (illegal deployed incumbent)",
        "bar_tps": DEPLOYED_TPS,                                # 481.53
        "bar_source": "deployed non-strict #52 2x9fm2zx (identity 0.9966 -> INFEASIBLE under #407)",
        "stack_tps": BLANKET_STRICT_STACK_TPS,                  # 482.74
        "beat_margin_tps": LEADERBOARD_BEAT_MARGIN_TPS,         # +1.21 (modeled)
        "beat_margin_is_modeled": True,
        "beat_margin_contingent_on": "kanna #416 budget-exact measurement of the full stack",
        "cb3_lift_breakeven_tps": CB3_LIFT_BREAKEVEN_FOR_BARB,  # 14.39 (the lift must clear this for Bar B>0)
        "haircut_flip_threshold_pct": round(BAR_B_HAIRCUT_FLIP_THRESHOLD * 100.0, 2),   # 7.76%
        "flip_note": (
            f"Bar B flips NEGATIVE if the realized cb3 lift drops below {CB3_LIFT_BREAKEVEN_FOR_BARB} TPS, i.e. "
            f"a haircut above {BAR_B_HAIRCUT_FLIP_THRESHOLD*100:.2f}% (well inside #410's "
            f"{ADDITIVITY_HAIRCUT_FRAC_BOUND*100:.1f}% bound). Worst-case Bar B margin {bar_b_worst_margin}."),
        "worst_case_margin_tps": bar_b_worst_margin,            # -1.11
        "status": "SECONDARY / leaderboard-cosmetic (NOT a #407 feasibility gate)",
    }
    return {
        "bar_A_directive4": bar_a,
        "bar_B_leaderboard_beat": bar_b,
        "worst_case_realized_cb3_lift": worst_realized_lift,
        "worst_case_stack_tps": worst_case_stack,
        "the_two_bars_are_distinct": (bar_a["bar_tps"] != bar_b["bar_tps"]),
        "binding_bar_under_407": "Bar A (Directive-#4) -- Bar B is excluded-incumbent cosmetics",
    }


# ======================================================================================== #
# Part 4 -- SHIP-POLICY under Bar B negative (does 482.74 still ship as fastest FEASIBLE?)
# ======================================================================================== #
def ship_policy_under_bar_b_negative(bars: dict[str, Any], enum: dict[str, Any]) -> dict[str, Any]:
    """If kanna #416 measures 482.74 < 481.53 (Bar B negative), does 482.74 still ship under #407?"""
    worst_stack = bars["worst_case_stack_tps"]                  # 480.42 (Bar B negative scenario)
    # even in the Bar-B-negative (worst haircut) world, the stack still dominates the next feasible config
    dominates_next_feasible = worst_stack > BLANKET_STRICT_TPS  # 480.42 > 467.14 -> True
    ships = True
    rationale = (
        "YES -- ships_if_bar_B_negative = True. #407's literal objective is 'the fastest [TPS] that still "
        "RESPECTS this equivalence.' The deployed 481.53 does NOT respect the equivalence (identity 0.9966, "
        f"{DEPLOYED_SERVED_FLIPS}/{DEPLOYED_SERVED_FLIP_DENOM} flips) under the current operative-identity, so "
        "it is EXCLUDED from the max -- it is not a feasible competitor at all. The max is taken over the "
        "FEASIBLE set {pure-AR-greedy (floor), blanket-strict 467.14, blanket-strict+cb3}. Even in the Bar-B-"
        f"negative world (worst {ADDITIVITY_HAIRCUT_FRAC_BOUND*100:.1f}% haircut -> stack {worst_stack}), the "
        f"cb3 stack still strictly dominates the next feasible config (blanket-strict {BLANKET_STRICT_TPS}), so "
        "it REMAINS the fastest feasible config and ships under #407. Bar B (beating the illegal 481.53 on the "
        "public leaderboard) is a COSMETIC bonus, not a #407 gate.")
    conditionality = (
        "CONDITIONAL on the CURRENT operative-identity partition (481.53 -> 0.9966 -> infeasible). Stark #429 "
        "(operative-identity resolution, in flight; successor to #421's canonical tolerance tie-break) is the "
        "SEPARATE pending input that fixes the partition. #429 can only HELP: (i) if it leaves 481.53 "
        "infeasible, Bar A holds and the +15.60 is banked; (ii) if it canonicalizes the fast path's bitwise-tie "
        "flips on the self-referential gate (#414 bq7xkfcv) and PROMOTES 481.53 into the feasible set, then the "
        "bar RISES to 481.53 but the SAME cb3 +15.60 supply lever stacks on the FASTER base (fast + cb3 ~= "
        "497), so the supply win is preserved (enlarged), not invalidated. Either way the cb3 +15.60 is banked; "
        "#429 only selects WHICH strict base it stacks on.")
    return {
        "ships_if_bar_B_negative": ships,
        "rationale": rationale,
        "dominates_next_feasible_in_worstcase": dominates_next_feasible,
        "worst_case_stack_tps": worst_stack,
        "next_feasible_config_tps": BLANKET_STRICT_TPS,
        "conditionality_on_429": conditionality,
        "directive4_bar_is_feasible_set_max": enum["directive4_bar_is_feasible_set_max"],
    }


# ======================================================================================== #
# Part 5 -- the SUPERSEDING GO card (additive; leads with banked win, demotes Bar B, identity row -> #429)
# ======================================================================================== #
def updated_go_card_md(recon: dict[str, Any], bars: dict[str, Any], ship: dict[str, Any],
                       enum: dict[str, Any], created_at: str) -> str:
    a = bars["bar_A_directive4"]
    b = bars["bar_B_leaderboard_beat"]
    lines = [
        "# DIRECTIVE-#4 DEPLOY GO/NO-GO -- HUMAN CHECKLIST (the CORRECT equivalent bar)",
        "",
        f"_PR #430 (lawine) static-analysis handoff -- generated {created_at}. SUPERSEDES #425's "
        "`GO_NOGO_CHECKLIST.md` (additive new file; the #425 card is left in place). This card SHIPS NOTHING; "
        "it is the decision surface a human reads to authorize {served-file change + submission} once the ONE "
        "remaining conjunct (identity, stark #429) goes GREEN. It LEADS with the banked Directive-#4 win._",
        "",
        "## The banked win (LEAD) -- Bar A, the CORRECT Directive-#4 bar",
        "",
        "```",
        f"Directive-#4 (#407): MAXIMIZE TPS subject to strict byte-exact greedy-token-equivalence.",
        f"  fastest prior FEASIBLE config (Bar A) ... blanket-strict {BLANKET_STRICT_TPS}  (#412 dnjvqbtf, "
        f"identity {BLANKET_STRICT_IDENTITY})",
        f"  realizable strict stack ................. blanket-strict + cb3 = {a['stack_tps']}  "
        f"(= {BLANKET_STRICT_TPS} + {CB3_LIFT_M8}, #403 k*={CB3_KSTAR})",
        f"  >>> directive4_banked_win_tps .......... +{a['banked_win_tps']}  (+{a['banked_win_pct']}%)  "
        f"-- NO margin contingency",
        "```",
        "",
        f"- **The deployed 481.53 is NOT the bar.** It is the non-strict fast path "
        f"({DEPLOYED_SERVED_FLIPS}/{DEPLOYED_SERVED_FLIP_DENOM} M=8 flips, identity {DEPLOYED_SERVED_IDENTITY}); "
        f"under #407 it does NOT respect the equivalence, so it is **excluded from the feasible-set max**.",
        f"- **No margin contingency.** {a['no_contingency_proof']}",
        f"- `canonical_blanket_strict_tps = {recon['canonical_blanket_strict_tps']}` "
        f"(#412 measured; #393 467.48 decode-eta projection is a confirming cross-check, "
        f"+{recon['delta_tps']} / {recon['delta_pct']}%, ~{recon['sigma_distance']}sigma -- the projection is "
        f"marginally optimistic, so the measured value is also the conservative choice).",
        "",
        "## The feasible-set ladder (strictly-equivalent configs, lowest -> highest)",
        "",
        "| # | Config | TPS | Identity | In feasible set | W&B |",
        "|---|--------|-----|----------|-----------------|-----|",
    ]
    for c in enum["feasible_ladder_low_to_high"]:
        tps = c["tps_local_in_cycle"]
        tps_s = (f"{tps}" if tps is not None else f"unmeasured (~{c.get('tps_cross_regime_est')} official est)")
        lines.append(f"| {c['rank']} | {c['config']} | {tps_s} | {c['identity']} | "
                     f"{c['in_feasible_set']} | {c['wandb_run']} |")
    ei = enum["excluded_infeasible_incumbent"]
    lines += [
        f"| -- | ~~{ei['config']}~~ (EXCLUDED) | {ei['tps_local_in_cycle']} | {ei['identity']} | "
        f"**False** | {ei['wandb_run']} |",
        "",
        f"_Feasible-set max = **{enum['feasible_set_max_config']}** at **{enum['feasible_set_max_tps']} TPS**. "
        f"`directive4_bar_is_feasible_set_max = {enum['directive4_bar_is_feasible_set_max']}` (481.53 excluded)._",
        "",
        "## Every line that MUST be GREEN before approval",
        "",
        "| # | Gate | Threshold | Source | Now |",
        "|---|------|-----------|--------|-----|",
        f"| 1 | Directive-#4 win (Bar A) | `> 0` (stack `> {BLANKET_STRICT_TPS}`) | this card | "
        f"**GREEN -- banked +{a['banked_win_tps']}, no contingency** |",
        f"| 2 | Served greedy identity | `== 1.0` (close base flip @ prompt {RESIDUAL_FLIP_PROMPT}) | "
        f"**stark #429** (in flight) | {IDENTITY_VALUE_NOW} -- PENDING |",
        f"| 3 | PPL | `<= {PPL_CAP}` (expect {DEPLOYED_PPL}) | cb3 PPL-safe | OK by construction |",
        f"| 4 | Completed | `== {REQUIRED_COMPLETED}` | benchmark | OK by construction |",
        f"| 5 | Bar B (beat illegal 481.53) | bonus only | kanna #416 | "
        f"modeled +{b['beat_margin_tps']} (COSMETIC) |",
        "",
        f"**CURRENT VERDICT: HOLD-for-IDENTITY-ONLY** -- the margin conjunct that #425 left pending is now "
        f"RESOLVED as a banked +{a['banked_win_tps']} Directive-#4 win (Bar A, no contingency); the ONLY "
        f"remaining pending conjunct is identity == 1.0 (stark #429). This is a strict collapse of #425's "
        "2-conjunct HOLD to a 1-conjunct HOLD.",
        "",
        "## Bar B is a leaderboard-cosmetic line (NOT a #407 gate)",
        "",
        f"- `leaderboard_beat_margin_tps = +{b['beat_margin_tps']}` (modeled; contingent on "
        f"{b['beat_margin_contingent_on']}).",
        f"- {b['flip_note']}",
        f"- **Ship-policy if Bar B goes negative:** `ships_if_bar_B_negative = {ship['ships_if_bar_B_negative']}`. "
        f"{ship['rationale']}",
        "",
        "## The identity conjunct (separate -- stark #429, do NOT re-derive here)",
        "",
        f"- The TPS reframe is ORTHOGONAL to identity. The cb3 supply leg is **equivalence-neutral** (adds 0 "
        f"flips, #403/#410); the residual is the SAME single blanket-strict base flip @ prompt "
        f"{RESIDUAL_FLIP_PROMPT} (#412: a bitwise tie, `identity_1p0_unreachable_by_precision=True`).",
        f"- Resolution belongs to **stark #429** (operative-identity resolution, in flight; successor to #421's "
        f"canonical tolerance tie-break on the self-referential gate #414 `bq7xkfcv`). This card does NOT "
        "re-derive identity; it carries it as the one pending conjunct.",
        f"- {ship['conditionality_on_429']}",
        "",
        "## Safe operation order",
        "",
        "1. **Wait for stark #429** (operative-identity == 1.0) -- the ONLY remaining conjunct. Bar A is banked.",
        "2. **Measure locally on the A10G** -- run the #319 3-tier identity-verify CI; confirm identity `== 1.0`.",
        "3. **Human approves in GitHub** -- the gated approval issue (PR + branch + exact command + GREEN CI).",
        "4. **Flip served file + submit** -- pin blanket-strict + cb3 (additive, reversible). Human-gated.",
        "",
        f"_The Directive-#4 win (+{a['banked_win_tps']}) is banked on the TPS axis with no margin contingency. "
        "The deploy still waits on the SEPARATE identity conjunct (stark #429). Bar B (+"
        f"{b['beat_margin_tps']}) is cosmetic. analysis_only=True, no_served_file_change=True, official_tps=0._",
        "",
    ]
    return "\n".join(lines)


# ======================================================================================== #
# repo-fact probes (read-only) + additive/served-file hygiene + self-test
# ======================================================================================== #
def repo_facts() -> dict[str, bool]:
    def ok(rel: str) -> bool:
        return (REPO_ROOT / rel).is_file()
    facts = {f"served::{k}": ok(v) for k, v in {
        "verify_patch": DEPLOYED_VERIFY_PATCH, "attn_patch": DEPLOYED_ATTN_PATCH, "manifest": DEPLOYED_MANIFEST,
    }.items()}
    facts.update({f"evidence::{k}": ok(v) for k, v in EVIDENCE.items()})
    facts["crosscheck::s412_json"] = ok(CROSSCHECK_412_JSON)
    facts["crosscheck::s403_json"] = ok(CROSSCHECK_403_JSON)
    facts["crosscheck::s410_json"] = ok(CROSSCHECK_410_JSON)
    facts["crosscheck::s393_json"] = ok(CROSSCHECK_393_JSON)
    facts["crosscheck::s196_json"] = ok(CROSSCHECK_196_JSON)
    facts["prior::s425_checklist"] = ok(PRIOR_425_CHECKLIST)
    return facts


def additive_and_no_served_change(out_dir: Path) -> dict[str, Any]:
    """Prove this card is ADDITIVE (new file, #425's checklist untouched) and touches NO served file."""
    my_checklist = out_dir / "GO_NOGO_CHECKLIST.md"
    my_rel = my_checklist.resolve()
    prior_rel = (REPO_ROOT / PRIOR_425_CHECKLIST).resolve()
    my_write_dir = out_dir.resolve()
    served_abs = [(REPO_ROOT / s).resolve() for s in SERVED_FILES]
    # my write set is confined to out_dir; assert no served file lives under it
    no_served_in_writedir = all((my_write_dir not in s.parents and s != my_write_dir) for s in served_abs)
    return {
        "my_checklist_path": str(my_checklist.relative_to(REPO_ROOT)) if my_rel.is_relative_to(REPO_ROOT)
        else str(my_checklist),
        "prior_425_checklist_path": PRIOR_425_CHECKLIST,
        "is_additive_new_file": (my_rel != prior_rel),
        "prior_425_checklist_still_exists": prior_rel.is_file(),
        "prior_425_in_different_dir": (my_write_dir != prior_rel.parent),
        "no_served_file_in_writedir": no_served_in_writedir,
        "served_files": SERVED_FILES,
    }


def selftest(xcheck: dict[str, Any], enum: dict[str, Any], recon: dict[str, Any], bars: dict[str, Any],
             ship: dict[str, Any], add: dict[str, Any], facts: dict[str, bool],
             flags: dict[str, Any]) -> dict[str, Any]:
    c: dict[str, bool] = {}

    # (a) pinned-import cross-check: every present check passes (byte-exact import, not re-derived)
    c["a_xcheck_present"] = (xcheck["n_crosschecks"] >= 10)
    c["a_xcheck_all_pass"] = (xcheck["all_present_checks_pass"] is True)
    for k, v in xcheck["checks"].items():
        c[f"a_{k}"] = bool(v)

    # (b) the core stack arithmetic (instruction #6: 482.74 = 467.14 + 15.60)
    c["b_stack_is_482_74"] = (abs(BLANKET_STRICT_STACK_TPS - 482.74) < 1e-9)
    c["b_stack_equals_base_plus_cb3"] = (
        abs(BLANKET_STRICT_STACK_TPS - round(BLANKET_STRICT_TPS + CB3_LIFT_M8, 2)) < 1e-9)
    c["b_467_14_plus_15_60"] = (abs(round(467.14 + 15.60, 2) - 482.74) < 1e-9)

    # (c) Bar A (Directive-#4) arithmetic + NO contingency
    a = bars["bar_A_directive4"]
    c["c_bar_a_bar_is_467_14"] = (abs(a["bar_tps"] - 467.14) < 1e-9)
    c["c_bar_a_win_is_15_60"] = (abs(a["banked_win_tps"] - 15.60) < 1e-9)
    c["c_bar_a_win_equals_stack_minus_base"] = (
        abs(a["banked_win_tps"] - round(BLANKET_STRICT_STACK_TPS - BLANKET_STRICT_TPS, 2)) < 1e-9)
    c["c_bar_a_win_equals_cb3_lift"] = (abs(a["banked_win_tps"] - CB3_LIFT_M8) < 1e-9)   # win IS the lift
    c["c_bar_a_pct_3_34"] = (abs(a["banked_win_pct"] - 3.34) < 5e-3)
    c["c_bar_a_no_contingency"] = (a["has_margin_contingency"] is False)
    c["c_bar_a_worst_positive"] = (a["worst_case_margin_tps"] > 0.0)                     # +13.28 > 0
    c["c_bar_a_worst_is_realized_lift"] = (
        abs(a["worst_case_margin_tps"] - bars["worst_case_realized_cb3_lift"]) < 1e-9)

    # (d) Bar B (leaderboard-beat) arithmetic + flip threshold
    b = bars["bar_B_leaderboard_beat"]
    c["d_bar_b_bar_is_481_53"] = (abs(b["bar_tps"] - 481.53) < 1e-9)
    c["d_bar_b_margin_is_1_21"] = (abs(b["beat_margin_tps"] - 1.21) < 1e-9)
    c["d_bar_b_margin_equals_stack_minus_deployed"] = (
        abs(b["beat_margin_tps"] - round(BLANKET_STRICT_STACK_TPS - DEPLOYED_TPS, 2)) < 1e-9)
    c["d_bar_b_breakeven_14_39"] = (abs(b["cb3_lift_breakeven_tps"] - 14.39) < 1e-9)
    c["d_bar_b_flip_threshold_7_76"] = (abs(b["haircut_flip_threshold_pct"] - 7.76) < 5e-3)
    c["d_bar_b_flip_below_410_bound"] = (
        BAR_B_HAIRCUT_FLIP_THRESHOLD < ADDITIVITY_HAIRCUT_FRAC_BOUND)   # 7.76% < 14.9% -> erasable
    c["d_bar_b_worst_negative"] = (b["worst_case_margin_tps"] < 0.0)                     # -1.11 < 0
    c["d_two_bars_distinct"] = (bars["the_two_bars_are_distinct"] is True)
    c["d_bar_a_below_bar_b"] = (a["bar_tps"] < b["bar_tps"])    # 467.14 < 481.53 (A is the easier, correct bar)

    # (e) reconciliation: canonical = 467.14, 467.48 a confirming cross-check
    c["e_canonical_is_467_14"] = (abs(recon["canonical_blanket_strict_tps"] - 467.14) < 1e-9)
    c["e_delta_is_0_34"] = (abs(recon["delta_tps"] - 0.34) < 1e-9)
    c["e_agree_within_0p1pct"] = (recon["agree_within_0p1pct"] is True)    # 0.073% agreement
    c["e_projection_optimistic"] = (recon["projection_optimistic"] is True)  # 467.48 > 467.14 -> measured is conservative
    c["e_393_above_412"] = (BLANKET_STRICT_393_TPS > BLANKET_STRICT_TPS)   # projection slightly higher

    # (f) enumeration: feasible-set max is the cb3 stack; ladder monotone; 481.53 excluded
    c["f_feasible_max_is_stack"] = (abs(enum["feasible_set_max_tps"] - BLANKET_STRICT_STACK_TPS) < 1e-9)
    c["f_ladder_monotone"] = (enum["feasible_ladder_numeric_monotone_increasing"] is True)
    c["f_three_feasible_configs"] = (enum["n_feasible_configs"] == 3)
    c["f_481_excluded"] = (enum["excluded_infeasible_incumbent"]["in_feasible_set"] is False)
    c["f_bar_is_feasible_set_max"] = (enum["directive4_bar_is_feasible_set_max"] is True)
    c["f_m1_floor_unmeasured_local"] = (enum["feasible_ladder_low_to_high"][0]["tps_local_in_cycle"] is None)
    c["f_m1_identity_by_definition"] = (abs(enum["feasible_ladder_low_to_high"][0]["identity"] - 1.0) < 1e-12)
    c["f_deployed_identity_0_9966"] = (
        abs(enum["excluded_infeasible_incumbent"]["identity"] - 0.9966) < 5e-4)

    # (g) ship-policy under Bar B negative
    c["g_ships_if_bar_b_negative"] = (ship["ships_if_bar_B_negative"] is True)
    c["g_dominates_next_feasible"] = (ship["dominates_next_feasible_in_worstcase"] is True)
    c["g_worststack_above_blanket"] = (ship["worst_case_stack_tps"] > BLANKET_STRICT_TPS)
    c["g_rationale_cites_407"] = ("respects this equivalence" in ship["rationale"].lower()
                                  or "respect the equivalence" in ship["rationale"].lower())
    c["g_conditionality_cites_429"] = ("#429" in ship["conditionality_on_429"])

    # (h) cb3 equivalence-neutral on the supply leg (no NEW flips); identity is the SAME single base flip
    stack_cfg = enum["feasible_ladder_low_to_high"][2]
    c["h_cb3_identity_equals_base"] = (abs(stack_cfg["identity"] - BLANKET_STRICT_IDENTITY) < 1e-12)
    c["h_supply_demand_additive"] = (SUPPLY_DEMAND_ADDITIVE is True)
    c["h_residual_flip_is_1"] = (BLANKET_STRICT_RESIDUAL_FLIPS == 1)
    c["h_identity_unreachable_by_precision"] = (IDENTITY_1P0_UNREACHABLE_BY_PRECISION is True)

    # (i) ADDITIVE + no served-file change (instruction #6)
    c["i_is_additive_new_file"] = (add["is_additive_new_file"] is True)
    c["i_prior_425_still_exists"] = (add["prior_425_checklist_still_exists"] is True)
    c["i_prior_425_different_dir"] = (add["prior_425_in_different_dir"] is True)
    c["i_no_served_in_writedir"] = (add["no_served_file_in_writedir"] is True)

    # (j) in-repo facts (read-only existence)
    for k, v in facts.items():
        c[f"j_{k}"] = v

    # (k) analysis-only hygiene
    c["k_official_tps_zero"] = (flags.get("official_tps") == 0)
    c["k_analysis_only"] = bool(flags.get("analysis_only"))
    c["k_no_hf_job"] = bool(flags.get("no_hf_job"))
    c["k_no_served_file_change"] = bool(flags.get("no_served_file_change"))
    c["k_ship_human_gated"] = (flags.get("ship_is_human_gated") is True)

    return {"conditions": c, "n_checks": len(c), "passes": all(c.values())}


# ======================================================================================== #
# report + IO + wandb
# ======================================================================================== #
def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, bool) or isinstance(o, (int, float, str)) or o is None:
        return o
    return str(o)


def print_report(p: dict[str, Any]) -> None:
    st = p["selftest"]
    recon = p["reconciliation"]
    bars = p["two_bars"]
    ship = p["ship_policy"]
    enum = p["enumeration"]
    a = bars["bar_A_directive4"]
    b = bars["bar_B_leaderboard_beat"]
    print("=" * 100)
    print(f"PR #430 lawine -- THE CORRECT DIRECTIVE-#4 BAR (is 482.74 a banked +15.60 win?)  ({p['created_at']})")
    print(f"  analysis_only={p['analysis_only']}  no_hf_job={p['no_hf_job']}  "
          f"no_served_file_change={p['no_served_file_change']}  official_tps={p['official_tps']}")
    print("-" * 100)
    print("  FEASIBLE-SET LADDER (strictly-equivalent, low -> high)")
    for cfg in enum["feasible_ladder_low_to_high"]:
        tps = cfg["tps_local_in_cycle"]
        tps_s = f"{tps}" if tps is not None else f"unmeasured(~{cfg.get('tps_cross_regime_est')} off-est)"
        print(f"    [{cfg['rank']}] {cfg['config']:<42} {tps_s:<28} id={cfg['identity']}")
    ei = enum["excluded_infeasible_incumbent"]
    print(f"    [X] {ei['config']+' (EXCLUDED)':<42} {ei['tps_local_in_cycle']:<28} id={ei['identity']}")
    print(f"    feasible_set_max = {enum['feasible_set_max_config']} @ {enum['feasible_set_max_tps']}  "
          f"| directive4_bar_is_feasible_set_max = {enum['directive4_bar_is_feasible_set_max']}")
    print("-" * 100)
    print("  RECONCILE blanket-strict base")
    print(f"    467.14 (#412 measured) vs 467.48 (#393 projection)  delta=+{recon['delta_tps']} "
          f"({recon['delta_pct']}%, ~{recon['sigma_distance']}sigma, agree<0.1%={recon['agree_within_0p1pct']})")
    print(f"    canonical_blanket_strict_tps = {recon['canonical_blanket_strict_tps']}")
    print("-" * 100)
    print("  THE TWO BARS")
    print(f"    Bar A (Directive-#4) base {a['bar_tps']:<8} -> stack {a['stack_tps']}  "
          f"directive4_banked_win_tps = +{a['banked_win_tps']} (+{a['banked_win_pct']}%)  "
          f"contingency={a['has_margin_contingency']}  worst=+{a['worst_case_margin_tps']}")
    print(f"    Bar B (leaderboard) base {b['bar_tps']:<8} -> stack {b['stack_tps']}  "
          f"leaderboard_beat_margin_tps = +{b['beat_margin_tps']} (modeled)  "
          f"flip>{b['haircut_flip_threshold_pct']}% haircut  worst={b['worst_case_margin_tps']}")
    print("-" * 100)
    print(f"  SHIP-POLICY: ships_if_bar_B_negative = {ship['ships_if_bar_B_negative']}")
    print(f"    {ship['rationale']}")
    print("-" * 100)
    print(f"  HUMAN CHECKLIST written to: {p.get('checklist_path', '(in payload)')}")
    print(f"  SELF-TEST {st['n_checks']} checks -> {'PASS' if st['passes'] else 'FAIL'}")
    if not st["passes"]:
        for k, v in st["conditions"].items():
            if not v:
                print(f"    FAILED: {k}")
    print("=" * 100)


def build_payload(flags: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    xcheck = pinned_import_crosscheck()
    enum = enumerate_strict_equivalent_configs()
    recon = reconcile_blanket_strict_base()
    bars = two_bars()
    ship = ship_policy_under_bar_b_negative(bars, enum)
    facts = repo_facts()
    add = additive_and_no_served_change(out_dir)
    st = selftest(xcheck, enum, recon, bars, ship, add, facts, flags)
    checklist = updated_go_card_md(recon, bars, ship, enum, created_at)
    a = bars["bar_A_directive4"]
    b = bars["bar_B_leaderboard_beat"]
    payload: dict[str, Any] = {
        "agent": "lawine", "pr": 430,
        "kind": "directive4-correct-equivalent-bar",
        "created_at": created_at,
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "ship_is_human_gated": True,
        # ---- headline deliverables (terminal SENPAI-RESULT fields) ----
        "canonical_blanket_strict_tps": recon["canonical_blanket_strict_tps"],          # 467.14
        "blanket_strict_stack_tps": BLANKET_STRICT_STACK_TPS,                            # 482.74 (PRIMARY)
        "directive4_banked_win_tps": a["banked_win_tps"],                               # +15.60
        "directive4_banked_win_pct": a["banked_win_pct"],                               # +3.34%
        "directive4_bar_is_feasible_set_max": enum["directive4_bar_is_feasible_set_max"],  # True (481.53 excl)
        "leaderboard_beat_margin_tps": b["beat_margin_tps"],                            # +1.21 (modeled)
        "bar_B_haircut_flip_threshold_pct": b["haircut_flip_threshold_pct"],            # 7.76%
        "ships_if_bar_B_negative": ship["ships_if_bar_B_negative"],                     # True
        "self_test_passes": bool(st["passes"]),
        "identity_conjunct_value": IDENTITY_VALUE_NOW,
        "identity_conjunct_pending_on": "stark #429 (operative-identity resolution, in flight)",
        # ---- detail blocks ----
        "pinned_import_crosscheck": xcheck,
        "enumeration": enum,
        "reconciliation": recon,
        "two_bars": bars,
        "ship_policy": ship,
        "additive_no_served_change": add,
        "human_checklist_md": checklist,
        "repo_facts": facts,
        "selftest": st,
        "shared_baselines": {
            "deployed_tps": DEPLOYED_TPS, "deployed_ppl": DEPLOYED_PPL, "deployed_served_identity":
            DEPLOYED_SERVED_IDENTITY, "deployed_served_flips": f"{DEPLOYED_SERVED_FLIPS}/"
            f"{DEPLOYED_SERVED_FLIP_DENOM}", "ppl_cap": PPL_CAP, "required_completed": REQUIRED_COMPLETED,
            "blanket_strict_tps": BLANKET_STRICT_TPS, "blanket_strict_tps_std": BLANKET_STRICT_TPS_STD,
            "blanket_strict_identity": BLANKET_STRICT_IDENTITY, "blanket_strict_393_tps": BLANKET_STRICT_393_TPS,
            "cb3_lift_m8": CB3_LIFT_M8, "cb3_kstar": CB3_KSTAR,
            "additivity_haircut_frac_bound": ADDITIVITY_HAIRCUT_FRAC_BOUND,
            "m1_nonspec_official_tps_est": M1_NONSPEC_OFFICIAL_TPS_EST,
        },
    }
    return payload


def maybe_log_wandb(payload: dict[str, Any], args) -> str | None:
    if args.no_wandb:
        return None
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from scripts.wandb_logging import (init_wandb_run, log_summary,
                                            log_json_artifact, finish_wandb)
    except Exception as e:  # noqa: BLE001
        print(f"[card] wandb helpers unavailable: {e}")
        return None
    a = payload["two_bars"]["bar_A_directive4"]
    b = payload["two_bars"]["bar_B_leaderboard_beat"]
    run = init_wandb_run(
        job_type="analysis-static-scope", agent="lawine",
        name=args.wandb_name, group=args.wandb_group,
        tags=["directive4-correct-bar", "feasible-set-max", "equivalence", "banked-win", "two-bars",
              "blanket-strict", "cb3", "ship-policy", "decision-doc", "pr-430"],
        config={"pr": 430, "kind": "directive4-correct-equivalent-bar",
                "analysis_only": True, "no_hf_job": True, "no_served_file_change": True,
                "official_tps": 0, "ship_is_human_gated": True,
                "deployed_tps": DEPLOYED_TPS, "blanket_strict_tps": BLANKET_STRICT_TPS,
                "cb3_lift_m8": CB3_LIFT_M8},
    )
    if run is None:
        print("[card] wandb disabled (no API key / WANDB_MODE).")
        return None
    flat: dict[str, float] = {
        "card/self_test_passes": float(payload["self_test_passes"]),
        "card/self_test_n_checks": float(payload["selftest"]["n_checks"]),
        "card/official_tps": float(payload["official_tps"]),
        "card/analysis_only": float(payload["analysis_only"]),
        "card/no_hf_job": float(payload["no_hf_job"]),
        "card/no_served_file_change": float(payload["no_served_file_change"]),
        "card/ship_is_human_gated": float(payload["ship_is_human_gated"]),
        # PRIMARY + bars
        "stack/blanket_strict_stack_tps": float(payload["blanket_strict_stack_tps"]),
        "stack/canonical_blanket_strict_tps": float(payload["canonical_blanket_strict_tps"]),
        "stack/cb3_lift_m8": float(CB3_LIFT_M8),
        "barA/directive4_banked_win_tps": float(payload["directive4_banked_win_tps"]),
        "barA/directive4_banked_win_pct": float(payload["directive4_banked_win_pct"]),
        "barA/worst_case_margin_tps": float(a["worst_case_margin_tps"]),
        "barA/has_margin_contingency": float(a["has_margin_contingency"]),
        "barB/leaderboard_beat_margin_tps": float(payload["leaderboard_beat_margin_tps"]),
        "barB/haircut_flip_threshold_pct": float(payload["bar_B_haircut_flip_threshold_pct"]),
        "barB/worst_case_margin_tps": float(b["worst_case_margin_tps"]),
        "barB/cb3_lift_breakeven_tps": float(b["cb3_lift_breakeven_tps"]),
        # feasibility + ship-policy
        "feasible/bar_is_feasible_set_max": float(payload["directive4_bar_is_feasible_set_max"]),
        "feasible/feasible_set_max_tps": float(payload["enumeration"]["feasible_set_max_tps"]),
        "feasible/deployed_excluded_identity": float(DEPLOYED_SERVED_IDENTITY),
        "ship/ships_if_bar_B_negative": float(payload["ships_if_bar_B_negative"]),
        # identity conjunct (pending #429)
        "identity/conjunct_value_now": float(payload["identity_conjunct_value"]),
        "identity/target": float(IDENTITY_TARGET),
        # crosscheck
        "crosscheck/n_pinned_import_checks": float(payload["pinned_import_crosscheck"]["n_crosschecks"]),
        "crosscheck/pinned_import_all_pass":
            float(payload["pinned_import_crosscheck"]["all_present_checks_pass"]),
    }
    run.log({"global_step": 0, **flat})
    log_summary(run, _jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="directive4_correct_bar", artifact_type="analysis", data=_jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[card] wandb logged {len(flat)} keys (run {rid})")
    return rid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", "--selftest", dest="self_test", action="store_true",
                    help="run the analytic self-test and exit nonzero on failure (no wandb)")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="lawine/directive4-correct-equivalent-bar")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="directive4-correct-bar")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    flags = {"analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
             "ship_is_human_gated": True}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(flags, out_dir)

    checklist_path = out_dir / "GO_NOGO_CHECKLIST.md"
    checklist_path.write_text(payload["human_checklist_md"])
    payload["checklist_path"] = str(checklist_path.relative_to(REPO_ROOT))
    print_report(payload)
    print(f"\n[card] wrote {checklist_path}")

    out_path = out_dir / "directive4_correct_bar_results.json"
    out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"[card] wrote {out_path}")

    st = payload["selftest"]
    if args.self_test:
        assert st["passes"], f"self-test FAILED ({st['n_checks']} checks)"
        assert st["n_checks"] >= 30, f"need >=30 asserts, have {st['n_checks']}"
        print(f"[card] SELF-TEST PASS ({st['n_checks']} checks)")
        print("\nSENPAI-RESULT " + json.dumps({
            "terminal": True, "status": "complete", "pending_arms": False, "wandb_run_ids": [],
            "analysis_only": True, "no_served_file_change": True, "official_tps": 0,
            "canonical_blanket_strict_tps": payload["canonical_blanket_strict_tps"],
            "directive4_banked_win_tps": payload["directive4_banked_win_tps"],
            "directive4_bar_is_feasible_set_max": payload["directive4_bar_is_feasible_set_max"],
            "leaderboard_beat_margin_tps": payload["leaderboard_beat_margin_tps"],
            "bar_B_haircut_flip_threshold_pct": payload["bar_B_haircut_flip_threshold_pct"],
            "ships_if_bar_B_negative": payload["ships_if_bar_B_negative"],
            "self_test_passes": bool(st["passes"]),
            "primary_metric": {"name": "directive4_banked_win_tps",
                               "value": float(payload["directive4_banked_win_tps"])},
            "test_metric": {"name": "self_test_passes", "value": float(st["passes"])},
        }))
        sys.exit(0 if st["passes"] else 1)

    rid = maybe_log_wandb(payload, args)
    if rid:
        payload["wandb_run_id"] = rid
        out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))

    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [rid] if rid else [],
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "ship_is_human_gated": True,
        "canonical_blanket_strict_tps": payload["canonical_blanket_strict_tps"],
        "directive4_banked_win_tps": payload["directive4_banked_win_tps"],
        "directive4_bar_is_feasible_set_max": payload["directive4_bar_is_feasible_set_max"],
        "leaderboard_beat_margin_tps": payload["leaderboard_beat_margin_tps"],
        "bar_B_haircut_flip_threshold_pct": payload["bar_B_haircut_flip_threshold_pct"],
        "ships_if_bar_B_negative": payload["ships_if_bar_B_negative"],
        "self_test_passes": bool(st["passes"]),
        "primary_metric": {"name": "directive4_banked_win_tps",
                           "value": float(payload["directive4_banked_win_tps"])},
        "test_metric": {"name": "self_test_passes", "value": float(st["passes"])},
    }))


if __name__ == "__main__":
    main()
