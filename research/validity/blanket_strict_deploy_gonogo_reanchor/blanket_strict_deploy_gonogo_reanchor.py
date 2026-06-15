#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #425 (lawine) -- Blanket-strict deploy GO/NO-GO: re-anchor off the REFUTED selrec leg.

WHY THIS CARD EXISTS (the re-anchor)
------------------------------------
My #417 deploy-surface ledger and #419 shippable GO/NO-GO were both priced on the MODELED selective-
recompute (selrec) fastest-equivalent (combined bracket [492.08, 494.08], modeled margin +11.55). stark
#412 has since MEASURED that leg and REFUTED it:
  * selrec as a realizable runtime wrapper costs 384.11 TPS (97.42-TPS tax) -- the ~2.6-TPS #397 model is
    reachable ONLY inside a fused conditional-precision kernel (a served-kernel edit), not a wrapper;
  * AND selrec is identity-DEGRADING (served identity 0.9853, 13 residual flips) -- it picks the strict
    tie-break, not the served one, at bitwise-tie positions, so it NET introduces flips.
  * #412 verdict: `fastest_realizable_strictly_equivalent_config = blanket_strict` (NOT selrec);
    `selective_beats_blanket = False`; the disputed verify flips are bitwise ties unreachable by precision.

So the realizable fastest STRICTLY-equivalent config is no longer selrec. It is BLANKET-STRICT (high-precision
verify reduction everywhere) at 467.14 (#412-corrected base) PLUS cb3 supply +15.60 (#403, equivalence-neutral,
k*=229) = 482.74. That is a knife-edge +1.2 TPS over the non-strict deployed 481.53 -- but it carries the
byte-identity guarantee the deployed config LACKS. This card re-anchors the #419 GO/NO-GO decision surface onto
that realizable stack, removes the selrec leg, and exposes the honest contingency.

THE RE-ANCHORED DECISION (a conjunction of TWO pending measurements)
-------------------------------------------------------------------
The deployed #1 (481.53) is non-strict and gives NO identity guarantee. The strict stack's ONLY reasons to
ship are (a) it is FASTER than 481.53 and (b) it is byte-identical. So the GO/NO-GO is a function of exactly
two pending inputs, and a GO needs BOTH GREEN:

  GO := (measured_margin_tps > 0)        # kanna #416: measured(blanket-strict+cb3) stack > 481.53   [PENDING]
        AND (identity_value == 1.0)       # stark #421: canonical tie-break closes the prompt-90 flip   [PENDING]
        AND (ppl <= 2.42)                 # quality guardrail (cb3 PPL-safe, 2.3772 unchanged)
        AND (completed == 128)            # full public run

  * conjunct 1 (MARGIN) is KNIFE-EDGE: modeled +1.21, but cb3's +15.60 stacks onto an already-drafter-active
    base, so ubel #410's supply x demand cross-term (additive, but <=14.9% haircut) applies. The margin
    survives only if the realized haircut on cb3's lift stays below ~7.8% (the lift must clear 14.39 of its
    15.60); a haircut in (7.8%, 14.9%] erases it. kanna #416's MEASURED full-stack TPS is the binding scalar.
  * conjunct 2 (IDENTITY) is ONE flip from 1.0: blanket-strict is 0.9989 (1 residual bitwise-tie flip @
    prompt 90). #412 proves precision CANNOT close it (`identity_1p0_unreachable_by_precision=True`); only a
    canonical tie-break (stark #421) can. NOTE #405 (merged) showed a GLOBAL lowest-id rule is RED (introduces
    14 new flips because the M=1 AR reference is not uniformly lowest-id); #421 must apply the canonical rule
    ONLY at true bitwise ties (m1_self_gap=0.0), where it is safe -- a discrete, closable-by-construction fix.

  Until BOTH land GREEN the honest verdict is HOLD-for-conjunction.

PURE STATIC ANALYSIS / DECISION CARD. analysis_only=True, no_hf_job=True, no_served_file_change=True,
official_tps=0. 0 GPU compute. It BUILDS NOTHING, FLIPS NO SERVED FILE, SUBMITS NOTHING. It produces the
decision surface that would justify FLAGGING a (still human-approval-gated) submission once both conjuncts go
GREEN; it does NOT perform one. The flag a GO would entail selects BLANKET-STRICT, NOT selrec.

PINNED constants are imported byte-exactly from the merged advisor-branch results JSON (#412 / #403 / #410 /
#417 / #419); Part 0 cross-checks them so "import, do not re-derive" is literally self-tested. Baseline 481.53
/ PPL 2.3772 / 128-128 UNCHANGED (#52, 2x9fm2zx).
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
# Part 0 cross-checks each against the merged #412 / #403 / #410 / #417 / #419 results JSON.
# ---------------------------------------------------------------------------------------- #
# --- the deployed NON-strict #1 = the SHIP-BREAKEVEN (no identity guarantee) ---
DEPLOYED_TPS = 481.53                  # #52 (2x9fm2zx) deployed non-strict #1 -- UNCHANGED ship-breakeven
DEPLOYED_PPL = 2.3772                  # #52 deployed PPL (unchanged by the equivalent stack, by construction)
PPL_CAP = 2.42                         # quality guardrail (reference 2.30 + 5%)
REQUIRED_COMPLETED = 128               # full public run
DEPLOYED_SERVED_FLIPS = 3              # deployed non-strict served M=8 argmax flips ...
DEPLOYED_SERVED_FLIP_DENOM = 882       # ... out of 882 positions ...
DEPLOYED_SERVED_IDENTITY = 0.9966      # ... -> identity 0.9966 (NO guarantee; the strict stack drives -> 1.0)

# --- #412 (stark): the MEASURED re-anchor. blanket-strict IS the fastest realizable strict config. ---
BLANKET_STRICT_TPS = 467.14            # #412 `blanket_strict_measured_tps` (467.1400155) -- corrected base
BLANKET_STRICT_TPS_STD = 0.16          # #412 `blanket_strict_measured_tps_std` (0.16105) -- tight
BLANKET_STRICT_IDENTITY = 0.9989       # #412 `blanket_strict_within_identity` (0.99886621) -- 1 residual flip
BLANKET_STRICT_RESIDUAL_FLIPS = 1      # #412 pinned arm: 882 positions, 881 match -> 1 flip @ prompt 90
RESIDUAL_FLIP_PROMPT = 90              # the single residual bitwise-tie flip (per #425 body / #412 arm)
SELREC_REALIZABLE_TPS = 384.11         # #412 `selective_recompute_measured_tps` -- REFUTED (97.42 tax)
SELREC_IDENTITY = 0.9853               # #412 `served_identity_after_selective` -- REFUTED (identity-degrading)
SELREC_TAX_TPS = 97.42                 # #412 `selective_tax_tps` (97.4155) -- realizable wrapper tax
IDENTITY_1P0_UNREACHABLE_BY_PRECISION = True   # #412 -- only a canonical tie-break (not precision) closes it

# --- #403 (kanna): cb3 supply, equivalence-neutral, PPL-safe, ADDITIVE ---
CB3_LIFT_M8 = 15.60                    # #403 `m8_lift_at_kstar` (15.6038966) k*=229 served M=8 lift (MEASURED)
CB3_KSTAR = 229                        # #403 conservative-k allocation (PPL-safe, worst-seed <= 2.41)

# --- #410 (ubel): supply x demand additive, but a bounded cross-term haircut ---
ADDITIVITY_HAIRCUT_FRAC_BOUND = 0.149  # #410 `delta_demand_tps_frac_of_lift` (0.14907) -- worst-case haircut
SUPPLY_DEMAND_ADDITIVE = True          # #410 `supply_demand_additive` -- additive (cross-term NOT negligible)

# --- the re-anchored realizable strict stack ---
BLANKET_STRICT_STACK_TPS = round(BLANKET_STRICT_TPS + CB3_LIFT_M8, 2)        # 482.74 (PRIMARY deliverable)
MARGIN_OVER_DEPLOYED_TPS = round(BLANKET_STRICT_STACK_TPS - DEPLOYED_TPS, 2)  # +1.21 (the knife-edge)

# the two PENDING decision inputs (advisor-provided, one-line swappable when the upstream cards land)
MEASURED_MARGIN_TPS = None             # kanna #416 `fastest_equivalent_tps` - 481.53 (PENDING)
IDENTITY_VALUE_NOW = BLANKET_STRICT_IDENTITY    # stark #421 closes prompt-90 flip -> 1.0 (PENDING; 0.9989 now)
IDENTITY_TARGET = 1.0

# the deployed served stack (the served files a blanket-strict GO would pin); the cb3 lever is additive (#417)
DEPLOYED_SUBMISSION = "submissions/fa2sw_treeverify_kenyan"
DEPLOYED_VERIFY_PATCH = f"{DEPLOYED_SUBMISSION}/splitkv_verify_patch.py"
DEPLOYED_ATTN_PATCH = f"{DEPLOYED_SUBMISSION}/fa_sliding_patch.py"
DEPLOYED_MANIFEST = f"{DEPLOYED_SUBMISSION}/manifest.json"
STRICT_VERIFY_FLAG = "STRICT_VERIFY_REDUCTION"   # ON=blanket-strict, OFF=today's-served. NOT a selrec selector.

# merged results JSON used to cross-check the pinned constants (byte-exact import proof)
CROSSCHECK_412_JSON = ("research/validity/selective_recompute_equivalent_tps/"
                       "selective_recompute_equivalent_tps_results.json")
CROSSCHECK_403_JSON = ("research/validity/cb3_conservative_k_deployable_lift/"
                       "cb3_conservative_k_deployable_lift_results.json")
CROSSCHECK_410_JSON = "research/validity/cb3_acceptance_crossterm/cb3_acceptance_crossterm_results.json"
CROSSCHECK_419_JSON = ("research/validity/shippable_equivalent_go_nogo/"
                       "shippable_equivalent_go_nogo_results.json")

# the #319 3-tier identity-verify harness (carried forward from #417/#419; in-repo paths)
HARNESS = {
    "per_gemm_byte_exact_390": "research/validity/strict_ceiling_corrected_rollup/strict_ceiling_corrected_rollup.py",
    "decode_width_e2e_381": "research/validity/decodewidth_e2e_identity/decodewidth_e2e_identity.py",
    "self_ref_reference_319": "scripts/local_validation/gen_greedy_reference.py",
    "self_ref_compare_319": "scripts/local_validation/greedy_gate.py",
    "self_ref_interlock_319": "scripts/validity/greedy_identity_interlock.py",
}

# banked evidence this card reasons from (read-only existence probes)
EVIDENCE = {
    "selrec_refuted_412": ("research/validity/selective_recompute_equivalent_tps/"
                           "selective_recompute_equivalent_tps.py"),
    "cb3_conservative_k_403": ("research/validity/cb3_conservative_k_deployable_lift/"
                               "cb3_conservative_k_deployable_lift.py"),
    "additivity_410": "research/validity/cb3_acceptance_crossterm/cb3_acceptance_crossterm.py",
    "deploy_surface_417": "research/validity/equivalent_stack_deploy_surface/equivalent_stack_deploy_surface.py",
    "go_nogo_419": "research/validity/shippable_equivalent_go_nogo/shippable_equivalent_go_nogo.py",
    "tiebreak_red_405": "research/validity/argmax_tiebreak_zero_cost_semantic/argmax_tiebreak_zero_cost_semantic.py",
}


# ======================================================================================== #
# Part 0 -- PINNED-IMPORT CROSS-CHECK (prove the constants are byte-exact from the merged JSON)
# ======================================================================================== #
def pinned_import_crosscheck() -> dict[str, Any]:
    """Read the merged #412 / #403 / #410 / #419 results JSON and assert every pinned constant matches.
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
        out["checks"]["x412_selrec_realizable_tps"] = (
            abs(d.get("selective_recompute_measured_tps", -1) - SELREC_REALIZABLE_TPS) < 5e-3)
        out["checks"]["x412_selrec_identity"] = (
            abs(d.get("served_identity_after_selective", -1) - SELREC_IDENTITY) < 5e-3)
        out["checks"]["x412_selrec_tax"] = (
            abs(d.get("selective_tax_tps", -1) - SELREC_TAX_TPS) < 5e-3)
        out["checks"]["x412_fastest_is_blanket"] = (
            d.get("fastest_realizable_strictly_equivalent_config") == "blanket_strict")
        out["checks"]["x412_selrec_not_beats_blanket"] = (d.get("selective_beats_blanket") is False)
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

    p419 = REPO_ROOT / CROSSCHECK_419_JSON
    if p419.is_file():
        d = json.loads(p419.read_text())
        sb = d.get("shared_baselines", {})
        out["available"]["s419"] = True
        out["checks"]["x419_deployed_tps"] = (abs(sb.get("deployed_tps", -1) - DEPLOYED_TPS) < 1e-9)
        out["checks"]["x419_ppl_cap"] = (abs(sb.get("ppl_cap", -1) - PPL_CAP) < 1e-9)
        out["checks"]["x419_cb3_kstar"] = (sb.get("cb3_kstar") == CB3_KSTAR)
    else:
        out["available"]["s419"] = False

    out["all_present_checks_pass"] = all(out["checks"].values()) if out["checks"] else False
    out["n_crosschecks"] = len(out["checks"])
    return out


# ======================================================================================== #
# Part 1 -- the RE-ANCHORED GO/NO-GO predicate: a conjunction of TWO pending inputs
# ======================================================================================== #
def deploy_gonogo_predicate(
    measured_margin_tps: float | None,
    identity_value: float | None,
    ppl: float | None,
    completed: int | None,
    *,
    ppl_cap: float = PPL_CAP,
    required_completed: int = REQUIRED_COMPLETED,
) -> dict[str, Any]:
    """GO iff (measured_margin_tps > 0) AND (identity_value == 1.0) AND (ppl <= cap) AND (completed == 128).

    PURE FUNCTION of the two pending decision inputs (kanna #416 margin, stark #421 identity) plus the two
    by-construction guardrails. A None numeric input fails its conjunct (cannot ship on a missing measurement).
    Returns the verdict + each conjunct so the human sees EXACTLY which line is red.
    """
    conjunct_margin_green = bool(measured_margin_tps is not None and measured_margin_tps > 0.0)
    conjunct_identity_green = bool(identity_value is not None and abs(identity_value - IDENTITY_TARGET) < 1e-12)
    checks = {
        "margin_positive": conjunct_margin_green,        # kanna #416: measured stack > 481.53
        "identity_is_1p0": conjunct_identity_green,      # stark #421: tie-break closes prompt-90 flip
        "ppl_within_cap": bool(ppl is not None and ppl <= ppl_cap),
        "completed_full": bool(completed == required_completed),
    }
    ship = all(checks.values())
    # verdict: GO if all green; HOLD-for-conjunction if blocked ONLY by a pending input; else NO-GO.
    pending = (measured_margin_tps is None) or (conjunct_identity_green is False)
    if ship:
        verdict = "GO"
    elif pending and checks["ppl_within_cap"] and checks["completed_full"]:
        verdict = "HOLD-for-conjunction"
    else:
        verdict = "NO-GO"
    return {
        "ship": ship, "verdict": verdict, "checks": checks,
        "conjunct_margin_green": conjunct_margin_green,
        "conjunct_identity_green": conjunct_identity_green,
        "inputs": {"measured_margin_tps": measured_margin_tps, "identity_value": identity_value,
                   "ppl": ppl, "completed": completed},
        "thresholds": {"margin_breakeven": 0.0, "identity_target": IDENTITY_TARGET,
                       "ppl_cap": ppl_cap, "required_completed": required_completed},
    }


def evaluate_decision_surface() -> dict[str, Any]:
    """Pre-compute the verdict on current best estimates AND state what each input must reach for GO."""
    eq_kwargs = dict(ppl=DEPLOYED_PPL, completed=REQUIRED_COMPLETED)

    # --- CURRENT state: both conjuncts PENDING (margin un-measured; identity 0.9989 < 1.0) -> HOLD ---
    current = deploy_gonogo_predicate(MEASURED_MARGIN_TPS, IDENTITY_VALUE_NOW, **eq_kwargs)

    # --- scenario grid (the decision surface) ---
    # the modeled (zero-haircut) margin and the worst-case (#410 14.9% haircut) margin
    modeled_margin = MARGIN_OVER_DEPLOYED_TPS                                    # +1.21
    worst_haircut_cb3 = round(CB3_LIFT_M8 * (1.0 - ADDITIVITY_HAIRCUT_FRAC_BOUND), 2)   # 13.28
    worst_case_stack = round(BLANKET_STRICT_TPS + worst_haircut_cb3, 2)         # 480.42
    worst_case_margin = round(worst_case_stack - DEPLOYED_TPS, 2)               # -1.11
    # the breakeven haircut: cb3 lift must clear (481.53 - 467.14) = 14.39; below that margin goes negative
    cb3_lift_breakeven = round(DEPLOYED_TPS - BLANKET_STRICT_TPS, 2)            # 14.39
    breakeven_haircut_frac = round((CB3_LIFT_M8 - cb3_lift_breakeven) / CB3_LIFT_M8, 4)  # ~0.0776

    scenarios = {
        # if the haircut is benign AND identity closes -> GO
        "modeled_margin_and_identity_closed": deploy_gonogo_predicate(
            modeled_margin, IDENTITY_TARGET, **eq_kwargs),
        # if identity closes but the haircut erases the margin -> NO-GO (the margin contingency bites)
        "worstcase_haircut_and_identity_closed": deploy_gonogo_predicate(
            worst_case_margin, IDENTITY_TARGET, **eq_kwargs),
        # if the margin holds but identity does NOT close -> NO-GO (the identity contingency bites)
        "modeled_margin_and_identity_open": deploy_gonogo_predicate(
            modeled_margin, IDENTITY_VALUE_NOW, **eq_kwargs),
        # exactly at the margin breakeven (measured stack == 481.53) -> NO-GO (strict >)
        "margin_at_breakeven_identity_closed": deploy_gonogo_predicate(
            0.0, IDENTITY_TARGET, **eq_kwargs),
    }

    # what each input must reach for GO
    go_requirements = {
        "measured_margin_tps": "> 0  (i.e. measured(blanket-strict + cb3) stack > 481.53; equivalently the "
                               f"realized cb3 lift must clear {cb3_lift_breakeven} TPS, a haircut < "
                               f"{breakeven_haircut_frac * 100:.1f}% on the modeled +{CB3_LIFT_M8})",
        "identity_value": "== 1.0  (stark #421's canonical tie-break closes the single prompt-90 bitwise-tie "
                          f"flip; currently {IDENTITY_VALUE_NOW})",
        "ppl": f"<= {PPL_CAP}  (by construction; cb3 PPL-safe, 2.3772 unchanged)",
        "completed": f"== {REQUIRED_COMPLETED}  (full public run)",
    }

    return {
        "predicate_definition": ("GO iff (measured_margin_tps > 0) AND (identity_value == 1.0) AND "
                                 "(ppl <= 2.42) AND (completed == 128)"),
        "current_verdict": current,
        "blanket_strict_stack_tps": BLANKET_STRICT_STACK_TPS,
        "margin_over_deployed_tps": MARGIN_OVER_DEPLOYED_TPS,
        "modeled_margin_tps": modeled_margin,
        "worst_case_haircut_stack_tps": worst_case_stack,
        "worst_case_haircut_margin_tps": worst_case_margin,
        "cb3_lift_breakeven_tps": cb3_lift_breakeven,
        "breakeven_haircut_frac": breakeven_haircut_frac,
        "additivity_haircut_frac_bound": ADDITIVITY_HAIRCUT_FRAC_BOUND,
        "margin_survives_only_if_haircut_below_breakeven": True,
        "scenarios": scenarios,
        "go_requirements": go_requirements,
    }


def predicate_enforces_all_conjuncts() -> dict[str, bool]:
    """Prove the predicate is a true AND-gate: flipping ANY single conjunct to bad -> not GO."""
    good = dict(measured_margin_tps=1.21, identity_value=1.0, ppl=DEPLOYED_PPL, completed=REQUIRED_COMPLETED)
    base = deploy_gonogo_predicate(**good)
    bad_margin = deploy_gonogo_predicate(**{**good, "measured_margin_tps": 0.0})       # at breakeven
    bad_margin_neg = deploy_gonogo_predicate(**{**good, "measured_margin_tps": -1.11})  # worst-case haircut
    bad_identity = deploy_gonogo_predicate(**{**good, "identity_value": 0.9989})        # 1 flip short
    bad_ppl = deploy_gonogo_predicate(**{**good, "ppl": 2.50})
    bad_done = deploy_gonogo_predicate(**{**good, "completed": 127})
    none_margin = deploy_gonogo_predicate(**{**good, "measured_margin_tps": None})
    return {
        "base_is_go": (base["ship"] is True and base["verdict"] == "GO"),
        "bad_margin_is_not_go": (bad_margin["ship"] is False),       # strict >: at breakeven NOT GO
        "bad_margin_neg_is_not_go": (bad_margin_neg["ship"] is False),
        "bad_identity_is_not_go": (bad_identity["ship"] is False),   # enforces identity == 1.0
        "bad_ppl_is_nogo": (bad_ppl["ship"] is False and bad_ppl["verdict"] == "NO-GO"),
        "bad_completed_is_nogo": (bad_done["ship"] is False and bad_done["verdict"] == "NO-GO"),
        "none_margin_holds": (none_margin["ship"] is False),         # cannot ship on a missing measurement
    }


# ======================================================================================== #
# Part 2 -- the BINDING CONTINGENCY: rank the two pending conjuncts + the "what kills this" sentence
# ======================================================================================== #
def binding_contingency_analysis(ds: dict[str, Any]) -> dict[str, Any]:
    """Of the two conjuncts, which is more likely to fail? Rank them honestly and name what kills the deploy."""
    breakeven_pct = ds["breakeven_haircut_frac"] * 100.0
    bound_pct = ADDITIVITY_HAIRCUT_FRAC_BOUND * 100.0
    # MARGIN risk: a continuous measurement; the failure region (breakeven, bound] is a LARGE slice of the
    # [0, bound] haircut band -> roughly half the admissible band kills it. High likelihood.
    margin_fail_band_width_pct = round(bound_pct - breakeven_pct, 2)            # ~7.1 points of the 14.9 band
    margin_fail_band_frac_of_bound = round(margin_fail_band_width_pct / bound_pct, 3)  # ~0.48
    # IDENTITY risk: a DISCRETE, closable-by-construction fix -- a canonical tie-break applied ONLY at the
    # single true bitwise tie (prompt 90) is a deterministic semantic. #405 RED was the GLOBAL rule; the
    # true-tie-only rule does not introduce new flips. Lower likelihood (but non-zero: #421 must land).
    ranking = ["margin", "identity"]   # most -> least likely to fail
    what_kills_this = (
        "If kanna #416 measures the cb3-over-blanket-strict additivity haircut above ~"
        f"{breakeven_pct:.1f}% (well inside ubel #410's <=" f"{bound_pct:.1f}% bound), the strict stack lands "
        f"at or below the deployed 481.53, the +{ds['margin_over_deployed_tps']} evaporates, and there is then "
        "NO TPS reason to ship the strict config at all -- a byte-identity guarantee with zero speed upside is "
        "a NO-GO. The identity flip is the lesser risk: it is one canonical tie-break (stark #421) at a single "
        "true bitwise tie, a discrete fix #412 shows precision cannot do but a tie-break can.")
    return {
        "ranking_most_to_least_likely_to_fail": ranking,
        "binding_contingency": "margin",
        "binding_contingency_reason": (
            f"The MARGIN conjunct is knife-edge (+{ds['margin_over_deployed_tps']}) and depends on a CONTINUOUS "
            f"measurement: the realized cb3 lift must clear {ds['cb3_lift_breakeven_tps']} TPS (haircut < "
            f"{breakeven_pct:.1f}%), but #410 admits a haircut up to {bound_pct:.1f}% -- so the failure region "
            f"({breakeven_pct:.1f}%, {bound_pct:.1f}%] is ~{margin_fail_band_frac_of_bound * 100:.0f}% of the "
            "admissible band. The IDENTITY conjunct is one DISCRETE canonical tie-break from 1.0 at a single "
            "true bitwise tie -- closable-by-construction (and #405 already isolated the failure mode of the "
            "naive global rule, so #421's true-tie-only rule is the de-risked successor)."),
        "margin_fail_band_width_pct": margin_fail_band_width_pct,
        "margin_fail_band_frac_of_bound": margin_fail_band_frac_of_bound,
        "what_kills_this_deploy": what_kills_this,
        "identity_risk_note": (
            "#412 `identity_1p0_unreachable_by_precision=True`: no attention-precision lever closes the "
            "prompt-90 flip (it is a bitwise tie). #405 (merged) showed a GLOBAL lowest-id rule is RED "
            "(14 new flips, M=1 ref not uniformly lowest-id); #421 must canonicalize ONLY true ties."),
    }


# ======================================================================================== #
# Part 3 -- the EXACT deploy config a GO entails (blanket-strict pin + cb3, NOT selrec)
# ======================================================================================== #
def deploy_config_spec() -> dict[str, Any]:
    """State precisely what served-file change a GO would entail, so a ready submission proposal can be flagged
    to the human the moment both conjuncts go GREEN. The flag selects BLANKET-STRICT, never selrec."""
    attention_pin = {
        "subsystem": "attention_kernel / verify-reduction",
        "served_files": [DEPLOYED_VERIFY_PATCH, DEPLOYED_ATTN_PATCH, DEPLOYED_MANIFEST + " (env declaration)"],
        "change": ("PIN the served verify/attention reduction to BLANKET-STRICT (high-precision reduction "
                   "EVERYWHERE -- the strict reference path), NOT the deployed non-strict fast reduction and "
                   "NOT the refuted selrec eps-near-tie kernel. No new kernel is built: blanket-strict reuses "
                   "the high-precision reduction the verify already supports, applied unconditionally."),
        "flag": STRICT_VERIFY_FLAG,
        "flag_on": "1 (ship): blanket-strict reduction -> byte-identical to the strict reference (after #421 "
                   "tie-break, served flips -> 0, identity 1.0)",
        "flag_off": "0 (rollback): today's-served non-strict reduction -> the deployed 481.53 behavior",
        "selects_selrec": False,
    }
    cb3_supply = {
        "subsystem": "body_gemm_quant",
        "served_files": [
            f"submissions/cb3_<name>/kernels/cb3_qtip_kernel-*.whl  (prebuilt sm_86 cb3 QTIP/QuIP# dequant GEMM)",
            f"submissions/cb3_<name>/cb3_quant_patch.py  (@register_quantization_config(\"cb3\"); NO in-tree edit)",
            f"submissions/cb3_<name>/manifest.json  (fork: +cb3 kernel dep, WEIGHTS_BUCKET -> cb3-baked ckpt)",
            f"submissions/cb3_<name>/serve.py  (fork: +setup_cb3_path)",
            f"submissions/cb3_<name>/sitecustomize.py  (fork: import cb3_quant_patch)",
            f"<cb3 bucket>/config.json  (remote NEW cb3-baked checkpoint, quant_method=\"cb3\")",
        ],
        "change": ("ADD the cb3 body-read-shrink supply lever (k*=229, +15.60 M=8, PPL-safe, equivalence-"
                   "neutral) as 6 ADDITIVE files / 0 in-place edits, per the #417 ledger. Orthogonal subsystem "
                   "to the attention pin -> stacks additively in ONE combined submission + ONE checkpoint."),
        "touches_checkpoint": True,
        "equivalence_neutral": True,
    }
    return {
        "deploy_config_summary": ("PIN blanket-strict verify reduction (flag " + STRICT_VERIFY_FLAG +
                                  "=1) + ADD cb3 supply (k*=229). Selects BLANKET-STRICT, NOT selrec. "
                                  "Whole stack additive + reversible (flag flip / bucket flip); human-gated."),
        "attention_pin": attention_pin,
        "cb3_supply": cb3_supply,
        "selrec_excluded": True,
        "selrec_excluded_reason": ("stark #412 REFUTED selrec: 384.11 realizable (97.42 tax) AND identity-"
                                   "degrading (0.9853). The flag must NOT bind selrec on any path; the strict "
                                   "leg is blanket-strict (467.14), the fastest REALIZABLE strict config."),
        "whole_stack_reversible": True,
        "deploy_is_human_gated": True,
        "ppl_unchanged": DEPLOYED_PPL,
    }


# ======================================================================================== #
# Part 4 -- the human-handoff GO/NO-GO checklist (markdown emitter)
# ======================================================================================== #
def human_checklist_md(ds: dict[str, Any], bc: dict[str, Any], dc: dict[str, Any], created_at: str) -> str:
    cur = ds["current_verdict"]
    lines = [
        "# BLANKET-STRICT DEPLOY GO/NO-GO -- HUMAN CHECKLIST (re-anchored off the refuted selrec leg)",
        "",
        f"_PR #425 (lawine) static-analysis handoff -- generated {created_at}. This card SHIPS NOTHING; it is",
        "the decision surface a human reads to authorize {served-file change + leaderboard submission} once",
        "BOTH pending conjuncts go GREEN. The flag a GO entails selects BLANKET-STRICT, NOT selrec._",
        "",
        "## The decision (executable predicate -- a conjunction of TWO pending inputs)",
        "",
        "```",
        "GO iff (measured_margin_tps > 0)    # kanna #416: measured(blanket-strict+cb3) > 481.53   [PENDING]",
        "       AND (identity_value == 1.0)   # stark #421: canonical tie-break closes prompt-90      [PENDING]",
        "       AND (ppl <= 2.42)             # cb3 PPL-safe, 2.3772 unchanged",
        "       AND (completed == 128)        # full public run",
        "```",
        "",
        f"- Realizable strict stack: **blanket-strict {BLANKET_STRICT_TPS} + cb3 +{CB3_LIFT_M8} = "
        f"{ds['blanket_strict_stack_tps']} TPS** (knife-edge **+{ds['margin_over_deployed_tps']}** over the "
        f"deployed non-strict {DEPLOYED_TPS}, WITH a byte-identity guarantee the deployed config lacks).",
        f"- Ship-breakeven (NO-GO boundary): **{DEPLOYED_TPS} TPS** (deployed non-strict #1, PR #52 `2x9fm2zx`).",
        f"- **CURRENT VERDICT: {cur['verdict']}** -- conjunct_margin_green={cur['conjunct_margin_green']} "
        f"(kanna #416 un-measured), conjunct_identity_green={cur['conjunct_identity_green']} "
        f"(blanket-strict {IDENTITY_VALUE_NOW}, 1 flip @ prompt {RESIDUAL_FLIP_PROMPT}).",
        f"- selrec is EXCLUDED: stark #412 refuted it ({SELREC_REALIZABLE_TPS} realizable / {SELREC_IDENTITY} "
        "identity). Do NOT price or flag the deploy on selrec.",
        "",
        "## Every line that MUST be GREEN before approval",
        "",
        "| # | Gate | Threshold | Source | Now |",
        "|---|------|-----------|--------|-----|",
        f"| 1 | Measured stack beats breakeven | `margin > 0` (stack `> {DEPLOYED_TPS}`) | kanna #416 | "
        f"PENDING (modeled +{ds['margin_over_deployed_tps']}) |",
        f"| 2 | Served greedy identity | `== 1.0` (1 flip @ prompt {RESIDUAL_FLIP_PROMPT} -> 0) | stark #421 "
        f"canonical tie-break | {IDENTITY_VALUE_NOW} |",
        f"| 3 | PPL | `<= {PPL_CAP}` (expect {DEPLOYED_PPL}) | cb3 PPL-safe | OK by construction |",
        f"| 4 | Completed | `== {REQUIRED_COMPLETED}` | benchmark | OK by construction |",
        f"| 5 | Whole-stack reversible | flag `{STRICT_VERIFY_FLAG}=0` (verify) / cb3 bucket flip | #417 + this "
        "card | OK |",
        "",
        "## The binding contingency (what kills this deploy)",
        "",
        f"**Ranked most -> least likely to fail: {bc['ranking_most_to_least_likely_to_fail']}.** "
        f"Binding = **{bc['binding_contingency'].upper()}**.",
        "",
        f"> {bc['what_kills_this_deploy']}",
        "",
        f"- Margin knife-edge: the realized cb3 lift must clear **{ds['cb3_lift_breakeven_tps']} TPS** "
        f"(haircut < **{ds['breakeven_haircut_frac'] * 100:.1f}%**); #410 admits up to "
        f"**{ADDITIVITY_HAIRCUT_FRAC_BOUND * 100:.1f}%** -> the failure region is ~"
        f"{bc['margin_fail_band_frac_of_bound'] * 100:.0f}% of the admissible band. Worst-case stack "
        f"{ds['worst_case_haircut_stack_tps']} (margin {ds['worst_case_haircut_margin_tps']}).",
        f"- Identity: {bc['identity_risk_note']}",
        "",
        "## The exact deploy config a GO entails (BLANKET-STRICT, not selrec)",
        "",
        f"{dc['deploy_config_summary']}",
        "",
        f"- **Attention/verify pin:** {dc['attention_pin']['change']}",
        f"  - flag `{dc['attention_pin']['flag']}` ON -> {dc['attention_pin']['flag_on']}",
        f"  - flag `{dc['attention_pin']['flag']}` OFF -> {dc['attention_pin']['flag_off']}",
        f"  - selects_selrec = {dc['attention_pin']['selects_selrec']}",
        f"- **cb3 supply:** {dc['cb3_supply']['change']}",
        "",
        "## Safe operation order",
        "",
        "1. **Wait for BOTH conjuncts** -- kanna #416 (measured margin > 0) AND stark #421 (identity 1.0).",
        "2. **Measure locally on the A10G** -- run the #319 3-tier identity-verify CI; confirm identity `== "
        "1.0` and the measured stack `> 481.53`.",
        "3. **Human approves in GitHub** -- the gated approval issue (PR + branch + exact command + GREEN CI).",
        "4. **Flip served file + submit** -- pin blanket-strict (flag ON) + cb3 and submit. Human-gated.",
        "",
        f"_Until BOTH conjuncts are GREEN, the honest verdict is **{cur['verdict']}**. Rollback at any point = "
        f"re-submit the prior package, or flip `{STRICT_VERIFY_FLAG}=0` to drop the strict pin while keeping cb3._",
        "",
    ]
    return "\n".join(lines)


# ======================================================================================== #
# repo-fact probes (read-only) + self-test
# ======================================================================================== #
def repo_facts() -> dict[str, bool]:
    def ok(rel: str) -> bool:
        return (REPO_ROOT / rel).is_file()
    facts = {f"deployed::{k}": ok(v) for k, v in {
        "verify_patch": DEPLOYED_VERIFY_PATCH, "attn_patch": DEPLOYED_ATTN_PATCH, "manifest": DEPLOYED_MANIFEST,
    }.items()}
    facts.update({f"harness::{k}": ok(v) for k, v in HARNESS.items()})
    facts.update({f"evidence::{k}": ok(v) for k, v in EVIDENCE.items()})
    facts["crosscheck::s412_json"] = ok(CROSSCHECK_412_JSON)
    facts["crosscheck::s403_json"] = ok(CROSSCHECK_403_JSON)
    facts["crosscheck::s410_json"] = ok(CROSSCHECK_410_JSON)
    facts["crosscheck::s419_json"] = ok(CROSSCHECK_419_JSON)
    return facts


def selftest(xcheck: dict[str, Any], ds: dict[str, Any], enforce: dict[str, bool],
             bc: dict[str, Any], dc: dict[str, Any], facts: dict[str, bool],
             flags: dict[str, Any]) -> dict[str, Any]:
    c: dict[str, bool] = {}

    # (a) pinned-import cross-check: every present check passes (byte-exact import, not re-derived)
    c["a_xcheck_present"] = (xcheck["n_crosschecks"] >= 10)
    c["a_xcheck_all_pass"] = (xcheck["all_present_checks_pass"] is True)
    for k, v in xcheck["checks"].items():
        c[f"a_{k}"] = bool(v)

    # (b) the re-anchored stack arithmetic
    c["b_stack_is_482_74"] = (abs(ds["blanket_strict_stack_tps"] - 482.74) < 1e-9)
    c["b_stack_equals_base_plus_cb3"] = (
        abs(ds["blanket_strict_stack_tps"] - round(BLANKET_STRICT_TPS + CB3_LIFT_M8, 2)) < 1e-9)
    c["b_margin_is_1_21"] = (abs(ds["margin_over_deployed_tps"] - 1.21) < 1e-9)
    c["b_margin_positive_but_knifeedge"] = (0.0 < ds["margin_over_deployed_tps"] < 2.0)
    c["b_worst_case_margin_negative"] = (ds["worst_case_haircut_margin_tps"] < 0.0)
    c["b_breakeven_haircut_below_bound"] = (
        ds["breakeven_haircut_frac"] < ADDITIVITY_HAIRCUT_FRAC_BOUND)   # 7.8% < 14.9% -> erasable
    c["b_cb3_lift_breakeven_14_39"] = (abs(ds["cb3_lift_breakeven_tps"] - 14.39) < 1e-9)

    # (c) the predicate definition + current HOLD verdict
    c["c_predicate_def"] = ("measured_margin_tps > 0" in ds["predicate_definition"]
                            and "identity_value == 1.0" in ds["predicate_definition"]
                            and "ppl <= 2.42" in ds["predicate_definition"]
                            and "completed == 128" in ds["predicate_definition"])
    cur = ds["current_verdict"]
    c["c_current_is_hold"] = (cur["verdict"] == "HOLD-for-conjunction")
    c["c_current_not_ship"] = (cur["ship"] is False)
    c["c_conjunct_margin_pending_false"] = (cur["conjunct_margin_green"] is False)
    c["c_conjunct_identity_pending_false"] = (cur["conjunct_identity_green"] is False)
    c["c_identity_now_0_9989"] = (abs(IDENTITY_VALUE_NOW - 0.9989) < 5e-3)
    c["c_residual_flip_is_1"] = (BLANKET_STRICT_RESIDUAL_FLIPS == 1)

    # (d) scenario grid: both-green -> GO; either contingency bites -> NO-GO; breakeven -> NO-GO
    sc = ds["scenarios"]
    c["d_both_green_is_go"] = (sc["modeled_margin_and_identity_closed"]["verdict"] == "GO")
    c["d_haircut_bites_is_nogo"] = (sc["worstcase_haircut_and_identity_closed"]["ship"] is False)
    c["d_identity_open_is_not_go"] = (sc["modeled_margin_and_identity_open"]["ship"] is False)
    c["d_margin_breakeven_is_not_go"] = (sc["margin_at_breakeven_identity_closed"]["ship"] is False)

    # (e) predicate is a true AND-gate
    c["e_base_go"] = enforce["base_is_go"]
    c["e_bad_margin_not_go"] = enforce["bad_margin_is_not_go"]
    c["e_bad_margin_neg_not_go"] = enforce["bad_margin_neg_is_not_go"]
    c["e_bad_identity_not_go"] = enforce["bad_identity_is_not_go"]
    c["e_bad_ppl_nogo"] = enforce["bad_ppl_is_nogo"]
    c["e_bad_completed_nogo"] = enforce["bad_completed_is_nogo"]
    c["e_none_margin_holds"] = enforce["none_margin_holds"]

    # (f) binding contingency = margin (ranked above identity), with a "what kills this" sentence
    c["f_binding_is_margin"] = (bc["binding_contingency"] == "margin")
    c["f_ranking_margin_first"] = (bc["ranking_most_to_least_likely_to_fail"][0] == "margin")
    c["f_ranking_identity_second"] = (bc["ranking_most_to_least_likely_to_fail"][1] == "identity")
    c["f_what_kills_mentions_haircut"] = ("haircut" in bc["what_kills_this_deploy"].lower())
    c["f_what_kills_mentions_481"] = ("481.53" in bc["what_kills_this_deploy"])

    # (g) deploy config selects blanket-strict, NOT selrec
    c["g_selrec_excluded"] = (dc["selrec_excluded"] is True)
    c["g_attn_not_selrec"] = (dc["attention_pin"]["selects_selrec"] is False)
    c["g_flag_is_strict_verify"] = (dc["attention_pin"]["flag"] == STRICT_VERIFY_FLAG)
    c["g_flag_not_selective_recompute"] = ("SELECTIVE_RECOMPUTE" not in dc["attention_pin"]["flag"])
    c["g_cb3_in_config"] = ("cb3" in dc["deploy_config_summary"].lower())
    c["g_blanket_in_summary"] = ("blanket-strict" in dc["deploy_config_summary"].lower())
    c["g_human_gated"] = (dc["deploy_is_human_gated"] is True)
    c["g_reversible"] = (dc["whole_stack_reversible"] is True)
    c["g_ppl_unchanged"] = (abs(dc["ppl_unchanged"] - DEPLOYED_PPL) < 1e-9)

    # (h) selrec refutation carried (NOT priced into the stack)
    c["h_selrec_refuted_slower"] = (SELREC_REALIZABLE_TPS < BLANKET_STRICT_TPS)
    c["h_selrec_identity_degraded"] = (SELREC_IDENTITY < BLANKET_STRICT_IDENTITY)
    c["h_blanket_is_fastest_realizable_strict"] = (BLANKET_STRICT_TPS > SELREC_REALIZABLE_TPS)
    c["h_identity_unreachable_by_precision"] = (IDENTITY_1P0_UNREACHABLE_BY_PRECISION is True)

    # (i) in-repo facts (read-only existence)
    for k, v in facts.items():
        c[f"i_{k}"] = v

    # (j) analysis-only hygiene
    c["j_official_tps_zero"] = (flags.get("official_tps") == 0)
    c["j_analysis_only"] = bool(flags.get("analysis_only"))
    c["j_no_hf_job"] = bool(flags.get("no_hf_job"))
    c["j_no_served_file_change"] = bool(flags.get("no_served_file_change"))
    c["j_ship_human_gated"] = (flags.get("ship_is_human_gated") is True)

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
    ds = p["decision_surface"]
    bc = p["binding_contingency_block"]
    dc = p["deploy_config"]
    cur = ds["current_verdict"]
    print("=" * 100)
    print(f"PR #425 lawine -- BLANKET-STRICT DEPLOY GO/NO-GO (re-anchor off refuted selrec)  ({p['created_at']})")
    print(f"  analysis_only={p['analysis_only']}  no_hf_job={p['no_hf_job']}  "
          f"no_served_file_change={p['no_served_file_change']}  official_tps={p['official_tps']}")
    print(f"  deployed/ship-breakeven {DEPLOYED_TPS} (non-strict, NO guarantee) | blanket-strict {BLANKET_STRICT_TPS}"
          f" (#412) | cb3 +{CB3_LIFT_M8} (#403 k*={CB3_KSTAR}) | selrec {SELREC_REALIZABLE_TPS} REFUTED (#412)")
    print("-" * 100)
    print("  RE-ANCHORED STACK")
    print(f"    blanket_strict_stack_tps ........ {ds['blanket_strict_stack_tps']}  (= {BLANKET_STRICT_TPS} + "
          f"{CB3_LIFT_M8})  [PRIMARY]")
    print(f"    margin_over_deployed_tps ........ +{ds['margin_over_deployed_tps']}  (knife-edge)")
    print(f"    worst-case haircut stack ........ {ds['worst_case_haircut_stack_tps']}  "
          f"(margin {ds['worst_case_haircut_margin_tps']}; haircut {ADDITIVITY_HAIRCUT_FRAC_BOUND*100:.1f}%)")
    print(f"    cb3 lift breakeven .............. {ds['cb3_lift_breakeven_tps']}  "
          f"(haircut breakeven {ds['breakeven_haircut_frac']*100:.1f}%)")
    print("-" * 100)
    print("  GO/NO-GO PREDICATE (two pending conjuncts)")
    print(f"    {ds['predicate_definition']}")
    print(f"    CURRENT VERDICT ................. {cur['verdict']}")
    print(f"      conjunct_margin_green ......... {cur['conjunct_margin_green']}  (kanna #416 PENDING)")
    print(f"      conjunct_identity_green ....... {cur['conjunct_identity_green']}  (stark #421 PENDING; "
          f"{IDENTITY_VALUE_NOW} now, 1 flip @ prompt {RESIDUAL_FLIP_PROMPT})")
    print("    scenario grid:")
    for k, v in ds["scenarios"].items():
        print(f"      {k:<42} -> {v['verdict']}")
    print("-" * 100)
    print(f"  BINDING CONTINGENCY = {bc['binding_contingency'].upper()}  "
          f"(ranked {bc['ranking_most_to_least_likely_to_fail']})")
    print(f"    {bc['what_kills_this_deploy']}")
    print("-" * 100)
    print("  DEPLOY CONFIG (a GO entails)")
    print(f"    {dc['deploy_config_summary']}")
    print(f"    selrec_excluded = {dc['selrec_excluded']}  flag = {dc['attention_pin']['flag']} "
          f"(selects_selrec={dc['attention_pin']['selects_selrec']})")
    print("-" * 100)
    print(f"  HUMAN CHECKLIST written to: {p.get('checklist_path', '(in payload)')}")
    print(f"  SELF-TEST {st['n_checks']} checks -> {'PASS' if st['passes'] else 'FAIL'}")
    if not st["passes"]:
        for k, v in st["conditions"].items():
            if not v:
                print(f"    FAILED: {k}")
    print("=" * 100)


def build_payload(flags: dict[str, Any]) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    xcheck = pinned_import_crosscheck()
    ds = evaluate_decision_surface()
    enforce = predicate_enforces_all_conjuncts()
    bc = binding_contingency_analysis(ds)
    dc = deploy_config_spec()
    facts = repo_facts()
    st = selftest(xcheck, ds, enforce, bc, dc, facts, flags)
    checklist = human_checklist_md(ds, bc, dc, created_at)
    cur = ds["current_verdict"]
    payload: dict[str, Any] = {
        "agent": "lawine", "pr": 425,
        "kind": "blanket-strict-deploy-gonogo-reanchor",
        "created_at": created_at,
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "ship_is_human_gated": True,
        # ---- headline deliverables (PR W&B summary fields) ----
        "blanket_strict_stack_tps": ds["blanket_strict_stack_tps"],          # PRIMARY
        "margin_over_deployed_tps": ds["margin_over_deployed_tps"],
        "go_nogo_verdict": cur["verdict"],
        "conjunct_margin_green": cur["conjunct_margin_green"],
        "conjunct_identity_green": cur["conjunct_identity_green"],
        "identity_conjunct_value": IDENTITY_VALUE_NOW,
        "residual_flip_count": BLANKET_STRICT_RESIDUAL_FLIPS,
        "binding_contingency": bc["binding_contingency"],
        "binding_contingency_what_kills_this": bc["what_kills_this_deploy"],
        "deploy_config": dc["deploy_config_summary"],
        "selrec_excluded": dc["selrec_excluded"],
        "deploy_gonogo_self_test_passes": bool(st["passes"]),
        # ---- detail blocks ----
        "pinned_import_crosscheck": xcheck,
        "decision_surface": ds,
        "predicate_enforces_all_conjuncts": enforce,
        "binding_contingency_block": bc,
        "deploy_config": dc,
        "human_checklist_md": checklist,
        "repo_facts": facts,
        "selftest": st,
        "shared_baselines": {
            "deployed_tps": DEPLOYED_TPS, "deployed_ppl": DEPLOYED_PPL, "ppl_cap": PPL_CAP,
            "required_completed": REQUIRED_COMPLETED, "deployed_served_identity": DEPLOYED_SERVED_IDENTITY,
            "deployed_served_flips": f"{DEPLOYED_SERVED_FLIPS}/{DEPLOYED_SERVED_FLIP_DENOM}",
            "blanket_strict_tps": BLANKET_STRICT_TPS, "blanket_strict_tps_std": BLANKET_STRICT_TPS_STD,
            "blanket_strict_identity": BLANKET_STRICT_IDENTITY,
            "selrec_realizable_tps": SELREC_REALIZABLE_TPS, "selrec_identity": SELREC_IDENTITY,
            "selrec_tax_tps": SELREC_TAX_TPS,
            "cb3_lift_m8": CB3_LIFT_M8, "cb3_kstar": CB3_KSTAR,
            "additivity_haircut_frac_bound": ADDITIVITY_HAIRCUT_FRAC_BOUND,
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
    ds = payload["decision_surface"]
    run = init_wandb_run(
        job_type="analysis-static-scope", agent="lawine",
        name=args.wandb_name, group=args.wandb_group,
        tags=["blanket-strict-deploy-gonogo", "reanchor", "selrec-refuted", "go-nogo-predicate",
              "conjunction", "binding-contingency", "cb3", "identity-verify", "decision-doc", "pr-425"],
        config={"pr": 425, "kind": "blanket-strict-deploy-gonogo-reanchor",
                "analysis_only": True, "no_hf_job": True, "no_served_file_change": True,
                "official_tps": 0, "ship_is_human_gated": True,
                "deployed_tps": DEPLOYED_TPS, "blanket_strict_tps": BLANKET_STRICT_TPS,
                "cb3_lift_m8": CB3_LIFT_M8, "selrec_realizable_tps": SELREC_REALIZABLE_TPS},
    )
    if run is None:
        print("[card] wandb disabled (no API key / WANDB_MODE).")
        return None
    flat: dict[str, float] = {
        "card/self_test_passes": float(payload["deploy_gonogo_self_test_passes"]),
        "card/self_test_n_checks": float(payload["selftest"]["n_checks"]),
        "card/official_tps": float(payload["official_tps"]),
        "card/analysis_only": float(payload["analysis_only"]),
        "card/no_hf_job": float(payload["no_hf_job"]),
        "card/no_served_file_change": float(payload["no_served_file_change"]),
        "card/ship_is_human_gated": float(payload["ship_is_human_gated"]),
        # PRIMARY + margin
        "stack/blanket_strict_stack_tps": float(payload["blanket_strict_stack_tps"]),
        "stack/margin_over_deployed_tps": float(payload["margin_over_deployed_tps"]),
        "stack/blanket_strict_tps": float(BLANKET_STRICT_TPS),
        "stack/cb3_lift_m8": float(CB3_LIFT_M8),
        "stack/deployed_breakeven_tps": float(DEPLOYED_TPS),
        "stack/worst_case_haircut_stack_tps": float(ds["worst_case_haircut_stack_tps"]),
        "stack/worst_case_haircut_margin_tps": float(ds["worst_case_haircut_margin_tps"]),
        "stack/cb3_lift_breakeven_tps": float(ds["cb3_lift_breakeven_tps"]),
        "stack/breakeven_haircut_frac": float(ds["breakeven_haircut_frac"]),
        "stack/additivity_haircut_frac_bound": float(ADDITIVITY_HAIRCUT_FRAC_BOUND),
        # conjunction
        "conjunct/margin_green": float(payload["conjunct_margin_green"]),
        "conjunct/identity_green": float(payload["conjunct_identity_green"]),
        "conjunct/identity_value_now": float(payload["identity_conjunct_value"]),
        "conjunct/residual_flip_count": float(payload["residual_flip_count"]),
        "conjunct/identity_target": float(IDENTITY_TARGET),
        # selrec refutation
        "selrec/excluded": float(payload["selrec_excluded"]),
        "selrec/realizable_tps": float(SELREC_REALIZABLE_TPS),
        "selrec/identity": float(SELREC_IDENTITY),
        "selrec/tax_tps": float(SELREC_TAX_TPS),
        # crosscheck
        "crosscheck/n_pinned_import_checks": float(payload["pinned_import_crosscheck"]["n_crosschecks"]),
        "crosscheck/pinned_import_all_pass":
            float(payload["pinned_import_crosscheck"]["all_present_checks_pass"]),
    }
    # verdict as an integer code for plotting (2=GO, 1=HOLD, 0=NO-GO)
    vcode = {"GO": 2.0, "HOLD-for-conjunction": 1.0, "NO-GO": 0.0}.get(payload["go_nogo_verdict"], -1.0)
    flat["verdict/go_nogo_code"] = vcode
    run.log({"global_step": 0, **flat})
    log_summary(run, _jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="blanket_strict_deploy_gonogo_reanchor", artifact_type="analysis",
                      data=_jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[card] wandb logged {len(flat)} keys (run {rid})")
    return rid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", "--selftest", dest="self_test", action="store_true",
                    help="run the analytic self-test and exit nonzero on failure (no wandb)")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="lawine/blanket-strict-deploy-gonogo-reanchor")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="blanket-strict-deploy")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    flags = {"analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
             "ship_is_human_gated": True}
    payload = build_payload(flags)

    out_dir = Path(args.out_dir)
    checklist_path = out_dir / "GO_NOGO_CHECKLIST.md"
    checklist_path.write_text(payload["human_checklist_md"])
    payload["checklist_path"] = str(checklist_path.relative_to(REPO_ROOT))
    print_report(payload)
    print(f"\n[card] wrote {checklist_path}")

    out_path = out_dir / "blanket_strict_deploy_gonogo_reanchor_results.json"
    out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"[card] wrote {out_path}")

    st = payload["selftest"]
    if args.self_test:
        assert st["passes"], f"self-test FAILED ({st['n_checks']} checks)"
        assert st["n_checks"] >= 30, f"need >=30 asserts, have {st['n_checks']}"
        print(f"[card] SELF-TEST PASS ({st['n_checks']} checks)")
        print("\nSENPAI-RESULT " + json.dumps({
            "terminal": True, "status": "complete", "pending_arms": False, "wandb_run_ids": [],
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            "deploy_gonogo_self_test_passes": bool(st["passes"]),
            "primary_metric": {"name": "blanket_strict_stack_tps",
                               "value": float(payload["blanket_strict_stack_tps"])},
            "test_metric": {"name": "deploy_gonogo_self_test_passes", "value": float(st["passes"])},
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
        "go_nogo_verdict": payload["go_nogo_verdict"],
        "blanket_strict_stack_tps": payload["blanket_strict_stack_tps"],
        "margin_over_deployed_tps": payload["margin_over_deployed_tps"],
        "conjunct_margin_green": payload["conjunct_margin_green"],
        "conjunct_identity_green": payload["conjunct_identity_green"],
        "selrec_excluded": payload["selrec_excluded"],
        "deploy_gonogo_self_test_passes": bool(st["passes"]),
        "primary_metric": {"name": "blanket_strict_stack_tps",
                           "value": float(payload["blanket_strict_stack_tps"])},
        "test_metric": {"name": "deploy_gonogo_self_test_passes", "value": float(st["passes"])},
    }))


if __name__ == "__main__":
    main()
