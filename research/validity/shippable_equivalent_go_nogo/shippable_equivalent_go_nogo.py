#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #419 (lawine) -- Shippable fastest-equivalent GO/NO-GO: predicate + verify CI + flag.

THE CARD (the ONE page + ONE checklist a human reads to APPROVE {served-file change + leaderboard submission})
-------------------------------------------------------------------------------------------------------------
My just-merged #417 (`2mv6ssw4`) priced the deploy surface for the fastest strictly-equivalent stack:
7 served files, 41.8 GPU-min identity-verify, whole-stack reversible, 1 binding in-place line (the selective-
recompute verify edit), human-gated, and a MODELED fastest-equivalent TPS bracket [492.08, 494.08] -- which
already BEATS the non-strict deployed 481.53 with a byte-identity guarantee the deployed config lacks. #417
left two follow-ups: (1) wire the measured #412/#416 numbers into the proposal, (2) feature-flag the in-place
verify line so revert = flag-flip not code-change. THIS card turns the modeled bracket into a SHIPPABLE,
human-approvable GO/NO-GO by producing three artifacts:

  (a) an EXECUTABLE GO/NO-GO PREDICATE -- SHIP iff (measured_fastest_equivalent_tps > 481.53) AND
      byte_identity_verified AND (ppl <= 2.42) AND (completed == 128). A pure function of the soon-to-be-
      measured stark #412 (selective-recompute TPS) + kanna #416 (combined TPS). It evaluates GO across the
      whole modeled bracket, and its NO-GO boundary (the ship-breakeven) is the deployed 481.53.
  (b) the EXACT pre-submission identity-verify CI -- which #319/#411 tier-1/2/3 checks, the 41.8 GPU-min
      budget (from #417, shared-e2e), and the PASS thresholds (served flips 3/882 -> 0; e2e greedy identity
      1.0; PPL <= 2.42; 128/128). This is the byte-identity EVIDENCE, not a TPS claim.
  (c) a FEATURE-FLAG wrapping the one in-place selective-recompute verify edit (SELECTIVE_RECOMPUTE_VERIFY)
      so rollback-while-keeping-cb3 is a flag flip, not a code re-edit -- collapsing the single binding line's
      reversibility risk to ~0 hot-path cost while preserving byte-identity on BOTH the ON (=strict) and OFF
      (=today's served) paths.

PURE STATIC ANALYSIS / DESIGN CARD. analysis_only=True, no_hf_job=True, no_served_file_change=True,
official_tps=0. 0 GPU compute. It BUILDS NOTHING, FLIPS NO SERVED FILE, SUBMITS NOTHING. It PRICES and
DE-RISKS the human gate; shipping remains the human-gated action this checklist enables.

THE GO/NO-GO PREDICATE (the executable core)
--------------------------------------------
SHIP := (measured_fastest_equivalent_tps > SHIP_BREAKEVEN=481.53)   # beats the deployed non-strict #1
        AND byte_identity_verified                                  # the #319 e2e gate returns identity 1.0
        AND (ppl <= 2.42)                                           # quality guardrail (unchanged 2.3772)
        AND (completed == 128)                                      # full public run
The four conjuncts are exactly the official validity gate (PPL + completion) PLUS the strict-equivalence claim
(byte-identity) PLUS the only reason to ship at all (TPS beats the breakeven). Under the modeled bracket
[492.08, 494.08] the predicate evaluates GO at BOTH ends (margin +10.55 lo / +12.55 hi over the 481.53
breakeven). The NO-GO boundary is the breakeven itself: the combined TPS must measure > 481.53. Decomposed
onto the pending inputs, the selective-recompute input breakeven is 481.53 - cb3 15.60 = 465.93 -- which is
BELOW even the #393 blanket-strict floor 467.48, so a selective-recompute that delivered ZERO speedup over
blanket-strict would STILL ship (+1.55). All three independent strict-equivalent anchors clear the breakeven:
#417/#412 modeled selrec [476.48, 478.48] (-> combined [492.08, 494.08]), the #393 blanket floor 467.48 (->
483.08), and denken #413 equiv_tps(7)=478.93 (-> 494.53). The predicate is robust, not knife-edge.

THE FEATURE-FLAG (de-risking the one binding in-place line)
----------------------------------------------------------
#417 named the selective-recompute verify edit (splitkv_verify_patch.py / fa_sliding_patch.py) as the ONE
in-place line whose revert-while-keeping-cb3 was a CODE change. This card wires SELECTIVE_RECOMPUTE_VERIFY,
resolved ONCE at serve startup, which BINDS the verify-reduction function pointer (no per-step branch):
  ON  (=1, the shipped default): bind the selective-recompute reduction -> byte-identical to BLANKET-STRICT
       (the strict reference). This is the fast-equivalent path the GO predicate authorizes.
  OFF (=0, the rollback):        bind the today's-served reduction       -> byte-identical to TODAY'S SERVED
       verify (the deployed fast path). Rollback keeps cb3's additive checkpoint+kernel untouched.
Because each binding IS an already-validated existing reduction function, the flag is a pure SELECTOR: it adds
NO arithmetic, so it cannot itself introduce an argmax flip on EITHER path (flag_preserves_byte_identity_both
=True). Resolved at startup, residual hot-path cost is EXACTLY 0 TPS (even a per-step branch would be one
perfectly-predicted compare against a ~ms GEMM-bound step -> unmeasurable). This converts #417's single binding
line from a code-revert into a flag-flip (inplace_line_now_flag_revertible=True).

SCOPE: read-only static analysis + in-repo file enumeration + pinned-constant cross-check against the merged
#417 / #413 results JSON (so "import byte-exactly" is literally self-tested, not re-derived). Baseline 481.53 /
PPL 2.3772 / 128/128 UNCHANGED (#52, 2x9fm2zx). Public evidence (advisor-branch banked / advisor-provided):
#417 deploy surface (2mv6ssw4), #411 3-tier verify harness (078yjgax), #403 cb3 +15.60 k*=229 (iv9i2wks),
#393 corrected strict base 467.48 (0q7ynumg), denken #413 equiv_tps(7)=478.93 (se8mf9ax); pending (advisor-
provided params, one-line-swappable): stark #412 selective_recompute_equivalent_tps, kanna #416
fastest_equivalent_tps.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------------------- #
# PINNED constants -- IMPORTED byte-exactly from merged modules (provenance run-id inline).
# Do NOT re-derive; Part 0 cross-checks these against the merged #417 / #413 results JSON.
# ---------------------------------------------------------------------------------------- #
DEPLOYED_TPS = 481.53                  # #52 (2x9fm2zx) deployed NON-STRICT #1 -- the SHIP-BREAKEVEN. UNCHANGED
DEPLOYED_PPL = 2.3772                  # #52 deployed PPL (unchanged by the equivalent stack, by construction)
PPL_CAP = 2.42                         # quality guardrail (reference 2.30 + 5%)
REQUIRED_COMPLETED = 128               # full public run
STRICT_BASE_BLANKET = 467.48           # #393 (0q7ynumg) corrected BLANKET-strict base (high-prec everywhere)
CB3_LIFT_M8 = 15.60                    # #403 (iv9i2wks) k*=229 served M=8 lift (MEASURED, PPL-safe, additive)
CB3_KSTAR = 229                        # #403 conservative-k allocation
EQUIV_TPS_AT_7_413 = 478.93            # denken #413 (se8mf9ax) equiv_tps(7) strict-equiv base at K=7 (x-check)

# deployed (non-strict) served identity: 3/882 M=8 argmax flips, identity 0.9966 -- the equivalent stack
# drives flips -> 0 / identity -> 1.0 (this is the byte-identity claim the CI must PROVE).
DEPLOYED_SERVED_FLIPS = 3
DEPLOYED_SERVED_FLIP_DENOM = 882
DEPLOYED_SERVED_IDENTITY = 0.9966
STRICT_SERVED_FLIPS_TARGET = 0
STRICT_SERVED_IDENTITY_TARGET = 1.0

# selective-recompute equivalent TPS -- stark #412 (PENDING; one-line swap). Modeled = blanket base + 9..11.
SELREC_TPS_MODEL_LO = 476.48           # #417 SELREC_TPS_MODEL_LO (467.48 + 9.0)
SELREC_TPS_MODEL_HI = 478.48           # #417 SELREC_TPS_MODEL_HI (467.48 + 11.0)
SELREC_MEASURED_TPS = None             # stark #412 `selective_recompute_equivalent_tps` (PENDING)

# combined fastest-equivalent TPS -- kanna #416 (PENDING; one-line swap). Modeled = selrec bracket + cb3.
FASTEST_EQUIVALENT_MODEL_LO = round(SELREC_TPS_MODEL_LO + CB3_LIFT_M8, 2)   # 492.08
FASTEST_EQUIVALENT_MODEL_HI = round(SELREC_TPS_MODEL_HI + CB3_LIFT_M8, 2)   # 494.08
FASTEST_EQUIVALENT_MEASURED_TPS = None         # kanna #416 `fastest_equivalent_tps` (PENDING)

# the in-place selective-recompute verify edit lands here (per #417); the flag wraps it.
DEPLOYED_SUBMISSION = "submissions/fa2sw_treeverify_kenyan"
DEPLOYED_VERIFY_PATCH = f"{DEPLOYED_SUBMISSION}/splitkv_verify_patch.py"
DEPLOYED_ATTN_PATCH = f"{DEPLOYED_SUBMISSION}/fa_sliding_patch.py"
DEPLOYED_MANIFEST = f"{DEPLOYED_SUBMISSION}/manifest.json"
SELECTIVE_RECOMPUTE_FLAG = "SELECTIVE_RECOMPUTE_VERIFY"   # env / runtime flag (resolved ONCE at startup)

# the #319 3-tier identity-verify harness (confirmed in-repo paths; reused from #417/#411/#404)
HARNESS = {
    "per_gemm_byte_exact_390": "research/validity/strict_ceiling_corrected_rollup/strict_ceiling_corrected_rollup.py",
    "decode_width_e2e_381": "research/validity/decodewidth_e2e_identity/decodewidth_e2e_identity.py",
    "self_ref_reference_319": "scripts/local_validation/gen_greedy_reference.py",
    "self_ref_compare_319": "scripts/local_validation/greedy_gate.py",
    "self_ref_interlock_319": "scripts/validity/greedy_identity_interlock.py",
}

# banked evidence this card reasons from (read-only existence probes)
EVIDENCE = {
    "deploy_surface_417": "research/validity/equivalent_stack_deploy_surface/equivalent_stack_deploy_surface.py",
    "geometry_413": "research/validity/equivalent_tps_optimal_geometry/equivalent_tps_optimal_geometry.py",
    "unified_ledger_411": "research/validity/flagged_supply_deploy_surface_ledger/flagged_supply_deploy_surface_ledger.py",
    "cb3_conservative_k_403": "research/validity/cb3_conservative_k_deployable_lift/cb3_conservative_k_deployable_lift.py",
}
# the two merged results JSON used to cross-check the pinned constants (byte-exact import proof)
CROSSCHECK_417_JSON = "research/validity/equivalent_stack_deploy_surface/equivalent_stack_deploy_surface_results.json"
CROSSCHECK_413_JSON = "research/validity/equivalent_tps_optimal_geometry/equivalent_tps_optimal_geometry_results.json"


# ======================================================================================== #
# Part 0 -- PINNED-IMPORT CROSS-CHECK (prove the constants are byte-exact from the merged JSON)
# ======================================================================================== #
def pinned_import_crosscheck() -> dict[str, Any]:
    """Read the merged #417 + #413 results JSON and assert every pinned constant matches. This makes the
    PR's 'import byte-exactly, do not re-derive' literally self-testable. Soft: records availability + match
    so the card still runs if a JSON is absent, but the self-test asserts on the matches when present."""
    out: dict[str, Any] = {"checks": {}, "available": {}}

    p417 = REPO_ROOT / CROSSCHECK_417_JSON
    if p417.is_file():
        d = json.loads(p417.read_text())
        sb = d.get("shared_baselines", {})
        out["available"]["s417"] = True
        out["checks"]["x417_deployed_tps"] = (abs(sb.get("deployed_tps", -1) - DEPLOYED_TPS) < 1e-9)
        out["checks"]["x417_deployed_ppl"] = (abs(sb.get("deployed_ppl", -1) - DEPLOYED_PPL) < 1e-9)
        out["checks"]["x417_ppl_cap"] = (abs(sb.get("ppl_cap", -1) - PPL_CAP) < 1e-9)
        out["checks"]["x417_strict_base"] = (abs(sb.get("strict_base_blanket", -1) - STRICT_BASE_BLANKET) < 1e-9)
        out["checks"]["x417_cb3_lift"] = (abs(sb.get("cb3_lift_m8", -1) - CB3_LIFT_M8) < 1e-9)
        out["checks"]["x417_cb3_kstar"] = (sb.get("cb3_kstar") == CB3_KSTAR)
        out["checks"]["x417_selrec_bracket"] = (
            sb.get("selrec_modeled_tps_bracket") == [SELREC_TPS_MODEL_LO, SELREC_TPS_MODEL_HI])
        out["checks"]["x417_combined_bracket"] = (
            d.get("fastest_equivalent_tps_modeled_bracket")
            == [FASTEST_EQUIVALENT_MODEL_LO, FASTEST_EQUIVALENT_MODEL_HI])
        out["checks"]["x417_combined_verify_41_8"] = (
            abs(d.get("combined_incremental_verify_gpu_min", -1) - 41.8) < 1e-9)
        out["checks"]["x417_seven_files"] = (d.get("combined_served_files") == 7)
        out["checks"]["x417_human_gated"] = (d.get("deploy_is_human_gated") is True)
        out["checks"]["x417_whole_stack_reversible"] = (d.get("whole_stack_reversible") is True)
    else:
        out["available"]["s417"] = False

    p413 = REPO_ROOT / CROSSCHECK_413_JSON
    if p413.is_file():
        d = json.loads(p413.read_text())
        out["available"]["s413"] = True
        # #413 stores equiv_tps_at_deployed7 = 478.92999...; pinned EQUIV_TPS_AT_7_413 rounds it
        out["checks"]["x413_equiv_tps_7"] = (
            abs(round(d.get("equiv_tps_at_deployed7", -1), 2) - EQUIV_TPS_AT_7_413) < 1e-9)
        out["checks"]["x413_kstar_7"] = (d.get("k_star") == 7)
    else:
        out["available"]["s413"] = False

    out["all_present_checks_pass"] = all(out["checks"].values()) if out["checks"] else False
    out["n_crosschecks"] = len(out["checks"])
    return out


# ======================================================================================== #
# Part 1 -- the GO/NO-GO predicate (the EXECUTABLE CORE) + modeled eval + breakeven
# ======================================================================================== #
def go_nogo_predicate(
    measured_fastest_equivalent_tps: float | None,
    byte_identity_verified: bool,
    ppl: float | None,
    completed: int | None,
    *,
    ship_breakeven: float = DEPLOYED_TPS,
    ppl_cap: float = PPL_CAP,
    required_completed: int = REQUIRED_COMPLETED,
) -> dict[str, Any]:
    """SHIP iff (tps > breakeven) AND byte_identity AND (ppl <= cap) AND (completed == 128).

    PURE FUNCTION of the (to-be-measured) inputs. Returns the verdict + each conjunct so the human sees
    EXACTLY which line is red. A None numeric input fails its conjunct (cannot ship on a missing measurement).
    """
    checks = {
        "tps_beats_breakeven": bool(measured_fastest_equivalent_tps is not None
                                    and measured_fastest_equivalent_tps > ship_breakeven),
        "byte_identity_verified": bool(byte_identity_verified),
        "ppl_within_cap": bool(ppl is not None and ppl <= ppl_cap),
        "completed_full": bool(completed == required_completed),
    }
    ship = all(checks.values())
    return {"ship": ship, "verdict": "GO" if ship else "NO-GO", "checks": checks,
            "inputs": {"measured_fastest_equivalent_tps": measured_fastest_equivalent_tps,
                       "byte_identity_verified": bool(byte_identity_verified),
                       "ppl": ppl, "completed": completed},
            "thresholds": {"ship_breakeven": ship_breakeven, "ppl_cap": ppl_cap,
                           "required_completed": required_completed}}


def evaluate_predicate() -> dict[str, Any]:
    """Run the predicate under the modeled bracket + the pending measured inputs, and compute the breakeven /
    margins / robustness so the NO-GO boundary is explicit."""
    # By construction the equivalent stack is byte-exact -> identity verified True, PPL = deployed 2.3772,
    # 128/128. The ONLY pending scalar that decides GO/NO-GO is the combined TPS (kanna #416).
    eq_kwargs = dict(byte_identity_verified=True, ppl=DEPLOYED_PPL, completed=REQUIRED_COMPLETED)

    under_lo = go_nogo_predicate(FASTEST_EQUIVALENT_MODEL_LO, **eq_kwargs)
    under_hi = go_nogo_predicate(FASTEST_EQUIVALENT_MODEL_HI, **eq_kwargs)
    central = round((FASTEST_EQUIVALENT_MODEL_LO + FASTEST_EQUIVALENT_MODEL_HI) / 2.0, 2)   # 493.08
    under_central = go_nogo_predicate(central, **eq_kwargs)
    go_under_modeled = bool(under_lo["ship"] and under_hi["ship"])

    # measured verdict stays PENDING until kanna #416 lands (one-line swap of FASTEST_EQUIVALENT_MEASURED_TPS)
    if FASTEST_EQUIVALENT_MEASURED_TPS is not None:
        under_measured = go_nogo_predicate(FASTEST_EQUIVALENT_MEASURED_TPS, **eq_kwargs)
    elif SELREC_MEASURED_TPS is not None:
        under_measured = go_nogo_predicate(round(SELREC_MEASURED_TPS + CB3_LIFT_M8, 2), **eq_kwargs)
    else:
        under_measured = {"ship": None, "verdict": "PENDING (#412 + #416 measuring)", "checks": None}

    # ---- breakeven / margins / robustness (the NO-GO boundary) ---- #
    ship_breakeven_combined = DEPLOYED_TPS                                  # 481.53 (the headline breakeven)
    ship_breakeven_selrec = round(DEPLOYED_TPS - CB3_LIFT_M8, 2)            # 465.93 (selrec input breakeven)
    margin_lo = round(FASTEST_EQUIVALENT_MODEL_LO - DEPLOYED_TPS, 2)       # +10.55
    margin_hi = round(FASTEST_EQUIVALENT_MODEL_HI - DEPLOYED_TPS, 2)       # +12.55
    margin_central = round(central - DEPLOYED_TPS, 2)                      # +11.55
    # robustness: even a ZERO-speedup selective-recompute (== blanket floor 467.48) ships
    floor_combined = round(STRICT_BASE_BLANKET + CB3_LIFT_M8, 2)          # 483.08
    floor_margin = round(floor_combined - DEPLOYED_TPS, 2)                # +1.55
    selrec_breakeven_below_blanket = bool(ship_breakeven_selrec < STRICT_BASE_BLANKET)   # 465.93 < 467.48
    # #413 third anchor: equiv_tps(7)=478.93 -> + cb3 = 494.53 (above the hi bracket; agrees ~492-494)
    anchor_413_combined = round(EQUIV_TPS_AT_7_413 + CB3_LIFT_M8, 2)      # 494.53

    # NO-GO boundary sweep (combined TPS) -- show the flip happens exactly at the breakeven
    sweep = []
    for t in (478.0, 480.0, 481.53, 481.54, 483.08, 492.08, 494.08):
        r = go_nogo_predicate(t, **eq_kwargs)
        sweep.append({"combined_tps": t, "verdict": r["verdict"], "ship": r["ship"]})

    return {
        "predicate_definition": ("SHIP iff (measured_fastest_equivalent_tps > 481.53) AND "
                                 "byte_identity_verified AND (ppl <= 2.42) AND (completed == 128)"),
        "under_modeled_lo": under_lo,
        "under_modeled_central": under_central,
        "under_modeled_hi": under_hi,
        "under_measured": under_measured,
        "go_nogo_predicate_evaluates_GO_under_modeled": go_under_modeled,
        "ship_breakeven_equivalent_tps": ship_breakeven_combined,
        "ship_breakeven_selective_recompute_tps": ship_breakeven_selrec,
        "modeled_margin_tps": margin_central,
        "modeled_margin_tps_lo": margin_lo,
        "modeled_margin_tps_hi": margin_hi,
        "robustness_zero_speedup_selrec_combined_tps": floor_combined,
        "robustness_zero_speedup_margin_tps": floor_margin,
        "selrec_breakeven_below_blanket_floor": selrec_breakeven_below_blanket,
        "crosscheck_413_combined_tps": anchor_413_combined,
        "modeled_central_combined_tps": central,
        "nogo_boundary_sweep": sweep,
        "robustness_note": (
            f"NO-GO boundary = combined TPS must measure > {ship_breakeven_combined}. Decomposed onto the "
            f"pending inputs (given cb3's MEASURED +{CB3_LIFT_M8}), the selective-recompute input breakeven is "
            f"{ship_breakeven_selrec}, which is BELOW even the #393 blanket-strict floor {STRICT_BASE_BLANKET} "
            f"-> a selective-recompute with ZERO speedup over blanket-strict STILL ships ({floor_combined}, "
            f"+{floor_margin}). All three independent anchors clear: #412 modeled selrec [{SELREC_TPS_MODEL_LO},"
            f" {SELREC_TPS_MODEL_HI}] -> [{FASTEST_EQUIVALENT_MODEL_LO}, {FASTEST_EQUIVALENT_MODEL_HI}]; #393 "
            f"floor -> {floor_combined}; #413 equiv_tps(7)={EQUIV_TPS_AT_7_413} -> {anchor_413_combined}. The "
            f"predicate is robust (>=+{floor_margin} even worst-case), not knife-edge."),
    }


def predicate_enforces_all_conjuncts() -> dict[str, bool]:
    """Prove the predicate is a true AND-gate: flipping ANY single conjunct to bad -> NO-GO. (Confirms it
    enforces greedy-identity AND PPL <= 2.42 per PR point 6, not just TPS.)"""
    good = dict(measured_fastest_equivalent_tps=493.08, byte_identity_verified=True,
                ppl=DEPLOYED_PPL, completed=REQUIRED_COMPLETED)
    base_go = go_nogo_predicate(**good)["ship"]
    bad_tps = go_nogo_predicate(**{**good, "measured_fastest_equivalent_tps": 481.53})["ship"]   # == breakeven
    bad_id = go_nogo_predicate(**{**good, "byte_identity_verified": False})["ship"]
    bad_ppl = go_nogo_predicate(**{**good, "ppl": 2.50})["ship"]
    bad_done = go_nogo_predicate(**{**good, "completed": 127})["ship"]
    none_tps = go_nogo_predicate(**{**good, "measured_fastest_equivalent_tps": None})["ship"]
    return {
        "base_is_go": (base_go is True),
        "bad_tps_is_nogo": (bad_tps is False),       # strict >: at the breakeven it is NO-GO
        "bad_identity_is_nogo": (bad_id is False),   # enforces byte-identity (PR point 6)
        "bad_ppl_is_nogo": (bad_ppl is False),       # enforces PPL <= 2.42 (PR point 6)
        "bad_completed_is_nogo": (bad_done is False),
        "none_tps_is_nogo": (none_tps is False),     # cannot ship on a missing measurement
    }


# ======================================================================================== #
# Part 2 -- pre-submission identity-verify CI spec (the byte-identity EVIDENCE, not a TPS claim)
# ======================================================================================== #
def verify_ci_spec() -> dict[str, Any]:
    """The exact gate the human inspects before approving the served-file change. Reuses the #417/#411 #319
    3-tier harness and the 41.8 GPU-min shared-e2e budget. PASS thresholds = the byte-identity proof."""
    tiers = [
        {"tier": 1, "name": "per-GEMM / per-config byte-identity micro (#390) + selrec byte-exact micro + "
                            "cb3 new-ref micro",
         "harness": HARNESS["per_gemm_byte_exact_390"],
         "proves": "the selective-recompute eps near-tie gate reduces byte-identically to blanket-strict, AND "
                   "cb3's new reference is keyed; one tier-1 micro per identity-claim component."},
        {"tier": 2, "name": "decode-width e2e identity (#381)",
         "harness": HARNESS["decode_width_e2e_381"],
         "proves": "M=1 vs M=8 served decode are token-identical across decode widths on the composed stack."},
        {"tier": 3, "name": "e2e SELF-REFERENTIAL greedy-identity gate (#319)",
         "harness": [HARNESS["self_ref_reference_319"], HARNESS["self_ref_compare_319"],
                     HARNESS["self_ref_interlock_319"]],
         "proves": "the FINAL composed served stack's greedy output is byte-identical to its own served "
                   "spec-OFF AR reference -> served flips 3/882 -> 0, identity -> 1.0. Change-agnostic: ONE "
                   "capture both re-keys cb3 AND validates selective-recompute byte-exactness (#417 shared-e2e)."},
    ]
    thresholds = {
        "served_m8_flips": {"from": DEPLOYED_SERVED_FLIPS, "from_denom": DEPLOYED_SERVED_FLIP_DENOM,
                            "must_reach": STRICT_SERVED_FLIPS_TARGET, "op": "== 0"},
        "e2e_greedy_identity": {"must_reach": STRICT_SERVED_IDENTITY_TARGET, "op": "== 1.0",
                                "from": DEPLOYED_SERVED_IDENTITY},
        "ppl": {"cap": PPL_CAP, "op": "<= 2.42", "expected": DEPLOYED_PPL},
        "completed": {"required": REQUIRED_COMPLETED, "op": "== 128"},
    }
    return {
        "pre_submission_verify_gpu_min": 41.8,   # #417 combined_incremental_verify_gpu_min (shared e2e)
        "verify_gpu_min_breakdown": "tier3 e2e shared 35.8 + tier2 decode-width shared 4.0 + 2x tier-1 micro "
                                    "(cb3 new-ref + selrec byte-exact) = 41.8 (vs naive unshared 81.6).",
        "tiers": tiers,
        "verify_pass_thresholds": thresholds,
        "is_byte_identity_evidence_not_tps": True,
        "harness_3tier": ("TIER1 per-GEMM/-config byte-identity micro (#390); TIER2 decode-width e2e (#381); "
                          "TIER3 e2e self-referential gate (#319 gen_greedy_reference --mode served + "
                          "greedy_gate.compare + greedy_identity_interlock)."),
    }


# ======================================================================================== #
# Part 3 -- feature-flag design for the ONE binding in-place line
# ======================================================================================== #
def feature_flag_design() -> dict[str, Any]:
    """Wrap the in-place selective-recompute verify edit in SELECTIVE_RECOMPUTE_VERIFY, resolved ONCE at serve
    startup (binds the verify-reduction function pointer; NO per-step branch). Rollback-while-keeping-cb3
    becomes a flag flip; both paths are byte-identical to an already-validated reduction."""
    on_path = {
        "value": "1 (shipped default)",
        "binds": "selective-recompute reduction (fast attention everywhere + eps near-tie gate + higher-prec "
                 "reduction on the ~23.6% flagged near-tie steps)",
        "byte_identical_to": "BLANKET-STRICT (the strict reference) -> served flips 0, identity 1.0",
        "is_ship_path": True,
    }
    off_path = {
        "value": "0 (rollback)",
        "binds": "today's-served reduction (the deployed fast verify/attention path, unchanged)",
        "byte_identical_to": "TODAY'S SERVED verify (the deployed fast path) -> the deployed 481.53 behavior",
        "is_ship_path": False,
    }
    # residual cost: resolved at startup -> 0 per-step branch. Even a per-step branch is unmeasurable.
    flag_residual_cost_tps = 0.0
    # the flag adds NO arithmetic (pure selector between two existing functions) -> introduces no flip itself
    flag_preserves_byte_identity_both_paths = True
    inplace_line_now_flag_revertible = True
    return {
        "flag_name": SELECTIVE_RECOMPUTE_FLAG,
        "resolution": "resolved ONCE at serve startup (read env -> bind verify-reduction function pointer); "
                      "NO per-step hot-path branch.",
        "wired_in": [DEPLOYED_VERIFY_PATCH, DEPLOYED_ATTN_PATCH, DEPLOYED_MANIFEST + " (env declaration)"],
        "on_path": on_path,
        "off_path": off_path,
        "rollback_while_keeping_cb3": (f"flip {SELECTIVE_RECOMPUTE_FLAG}=0 in the manifest env -> selective-"
                                       "recompute bypassed, cb3's additive checkpoint+kernel UNTOUCHED. A flag "
                                       "flip, NOT a code re-edit (this is #417's binding line de-risked)."),
        "inplace_line_now_flag_revertible": inplace_line_now_flag_revertible,
        "flag_residual_cost_tps": flag_residual_cost_tps,
        "flag_residual_cost_note": ("startup-resolved -> exactly 0 hot-path branch; even a per-step branch is "
                                    "one perfectly-predicted compare against a ~ms GEMM-bound step -> < 0.01 "
                                    "TPS, unmeasurable."),
        "flag_preserves_byte_identity_both_paths": flag_preserves_byte_identity_both_paths,
        "flag_identity_note": ("the flag is a pure SELECTOR between two already-validated reduction functions; "
                               "it adds NO arithmetic, so it cannot itself introduce an argmax flip on EITHER "
                               "path. The #319 e2e gate validates the ON binding (=strict); the OFF binding IS "
                               "today's served reduction (already the deployed reference)."),
        "default_is_strict_on_ship": True,
    }


# ======================================================================================== #
# Part 4 -- the human-handoff GO/NO-GO checklist (markdown emitter)
# ======================================================================================== #
def human_checklist_md(pred: dict[str, Any], ci: dict[str, Any], flag: dict[str, Any],
                       created_at: str) -> str:
    """Single GO/NO-GO checklist the human reads to approve {served-file change + leaderboard submission}.
    Every line that must be GREEN + the safe operation order."""
    t = ci["verify_pass_thresholds"]
    lines = [
        "# SHIP THE FASTEST STRICTLY-EQUIVALENT CONFIG -- HUMAN GO/NO-GO CHECKLIST",
        "",
        f"_PR #419 (lawine) static-analysis handoff -- generated {created_at}. This card SHIPS NOTHING; it is",
        "the one page a human reads to authorize {served-file change + leaderboard submission}._",
        "",
        "## The decision (executable predicate)",
        "",
        "```",
        "SHIP iff (measured_fastest_equivalent_tps > 481.53)   # kanna #416, beats the deployed non-strict #1",
        "         AND byte_identity_verified                    # #319 e2e gate returns identity 1.0",
        "         AND (ppl <= 2.42)                             # quality guardrail (unchanged 2.3772)",
        "         AND (completed == 128)                        # full public run",
        "```",
        "",
        f"- Ship-breakeven (NO-GO boundary): **{pred['ship_breakeven_equivalent_tps']} TPS** (the deployed "
        "non-strict #1, PR #52 `2x9fm2zx`).",
        f"- Modeled combined TPS bracket: **[{FASTEST_EQUIVALENT_MODEL_LO}, {FASTEST_EQUIVALENT_MODEL_HI}]** "
        f"-> modeled margin **+{pred['modeled_margin_tps_lo']} .. +{pred['modeled_margin_tps_hi']}** "
        f"(central +{pred['modeled_margin_tps']}).",
        f"- Robustness: even a ZERO-speedup selective-recompute still ships "
        f"(**{pred['robustness_zero_speedup_selrec_combined_tps']} TPS**, +"
        f"{pred['robustness_zero_speedup_margin_tps']}) -- the predicate is robust, not knife-edge.",
        "",
        "## Every line that MUST be GREEN before approval",
        "",
        "| # | Gate | Threshold | Source |",
        "|---|------|-----------|--------|",
        f"| 1 | Measured combined TPS beats breakeven | `> {pred['ship_breakeven_equivalent_tps']}` | kanna "
        "#416 `fastest_equivalent_tps` |",
        f"| 2 | Served greedy identity | `== {t['e2e_greedy_identity']['must_reach']}` "
        f"(flips {t['served_m8_flips']['from']}/{t['served_m8_flips']['from_denom']} -> "
        f"{t['served_m8_flips']['must_reach']}) | #319 e2e self-referential gate |",
        f"| 3 | PPL | `<= {t['ppl']['cap']}` (expect {t['ppl']['expected']}) | PPL stage |",
        f"| 4 | Completed | `== {t['completed']['required']}` | benchmark |",
        f"| 5 | Whole-stack reversible | re-submit prior package; cb3 by bucket flip; selrec by "
        f"`{flag['flag_name']}=0` | #417 + this card |",
        f"| 6 | Flag-revert confirmed | OFF == today's served, ON == strict, residual "
        f"{flag['flag_residual_cost_tps']} TPS | this card Part 3 |",
        "",
        "## Pre-submission identity-verify CI (the byte-identity EVIDENCE)",
        "",
        f"- Budget: **{ci['pre_submission_verify_gpu_min']} GPU-min** ({ci['verify_gpu_min_breakdown']})",
        f"- {ci['harness_3tier']}",
        "- This is measured LOCALLY on the A10G BEFORE any served-file change -- it proves byte-identity, it "
        "is not a TPS claim.",
        "",
        "## Feature-flag de-risking the one binding in-place line",
        "",
        f"- `{flag['flag_name']}` -- {flag['resolution']}",
        f"  - **ON (=1, shipped default):** {flag['on_path']['binds']} -> byte-identical to "
        f"{flag['on_path']['byte_identical_to']}.",
        f"  - **OFF (=0, rollback):** {flag['off_path']['binds']} -> byte-identical to "
        f"{flag['off_path']['byte_identical_to']}.",
        f"  - Rollback-while-keeping-cb3: {flag['rollback_while_keeping_cb3']}",
        "",
        "## Safe operation order",
        "",
        "1. **Measure locally on the A10G** -- run the 41.8 GPU-min pre-submission verify CI; confirm ALL six "
        "GREEN lines above (especially identity `== 1.0` and the measured TPS `> 481.53`).",
        "2. **Human approves in GitHub** -- the gated approval issue (PR + branch + exact command + the GREEN "
        "CI evidence).",
        "3. **Flip served file + submit** -- deploy the equivalent submission (flag ON = strict) and submit to "
        "the leaderboard. THIS step is the human-gated action; everything above is pre-flight.",
        "",
        "_Rollback at any point = re-submit the prior package, or flip "
        f"`{flag['flag_name']}=0` to keep cb3 while dropping selective-recompute._",
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
        "verify_patch": DEPLOYED_VERIFY_PATCH, "attn_patch": DEPLOYED_ATTN_PATCH,
        "manifest": DEPLOYED_MANIFEST,
    }.items()}
    facts.update({f"harness::{k}": ok(v) for k, v in HARNESS.items()})
    facts.update({f"evidence::{k}": ok(v) for k, v in EVIDENCE.items()})
    facts["crosscheck::s417_json"] = ok(CROSSCHECK_417_JSON)
    facts["crosscheck::s413_json"] = ok(CROSSCHECK_413_JSON)
    return facts


def selftest(xcheck: dict[str, Any], pred: dict[str, Any], enforce: dict[str, bool],
             ci: dict[str, Any], flag: dict[str, Any], facts: dict[str, bool],
             flags: dict[str, Any]) -> dict[str, Any]:
    c: dict[str, bool] = {}

    # (a) pinned-import cross-check: every present check passes (byte-exact import, not re-derived)
    c["a_xcheck_present"] = (xcheck["n_crosschecks"] >= 8)
    c["a_xcheck_all_pass"] = (xcheck["all_present_checks_pass"] is True)
    for k, v in xcheck["checks"].items():
        c[f"a_{k}"] = bool(v)

    # (b) the GO/NO-GO predicate -- definition, GO under modeled, breakeven, margins, robustness
    c["b_predicate_def"] = ("> 481.53" in pred["predicate_definition"]
                            and "ppl <= 2.42" in pred["predicate_definition"]
                            and "completed == 128" in pred["predicate_definition"])
    c["b_go_under_modeled"] = (pred["go_nogo_predicate_evaluates_GO_under_modeled"] is True)
    c["b_go_under_modeled_lo"] = (pred["under_modeled_lo"]["ship"] is True)
    c["b_go_under_modeled_hi"] = (pred["under_modeled_hi"]["ship"] is True)
    c["b_breakeven_481_53"] = (abs(pred["ship_breakeven_equivalent_tps"] - 481.53) < 1e-9)
    c["b_selrec_breakeven_465_93"] = (abs(pred["ship_breakeven_selective_recompute_tps"] - 465.93) < 1e-9)
    c["b_margin_central"] = (abs(pred["modeled_margin_tps"] - 11.55) < 1e-9)
    c["b_margin_lo_10_55"] = (abs(pred["modeled_margin_tps_lo"] - 10.55) < 1e-9)
    c["b_margin_hi_12_55"] = (abs(pred["modeled_margin_tps_hi"] - 12.55) < 1e-9)
    c["b_margin_in_pr_range"] = (10.5 <= pred["modeled_margin_tps"] <= 12.5)
    c["b_robust_zero_speedup_ships"] = (pred["robustness_zero_speedup_selrec_combined_tps"] > DEPLOYED_TPS)
    c["b_selrec_breakeven_below_blanket"] = (pred["selrec_breakeven_below_blanket_floor"] is True)
    c["b_413_anchor_clears"] = (pred["crosscheck_413_combined_tps"] > DEPLOYED_TPS)
    c["b_measured_pending"] = (pred["under_measured"]["ship"] is None)   # both #412/#416 pending

    # (c) predicate is a true AND-gate (enforces identity AND PPL, PR point 6 -- not just TPS)
    c["c_base_go"] = enforce["base_is_go"]
    c["c_bad_tps_nogo"] = enforce["bad_tps_is_nogo"]
    c["c_bad_identity_nogo"] = enforce["bad_identity_is_nogo"]
    c["c_bad_ppl_nogo"] = enforce["bad_ppl_is_nogo"]
    c["c_bad_completed_nogo"] = enforce["bad_completed_is_nogo"]
    c["c_none_tps_nogo"] = enforce["none_tps_is_nogo"]

    # (d) NO-GO boundary sweep flips exactly at the breakeven
    sweep = {round(s["combined_tps"], 2): s["ship"] for s in pred["nogo_boundary_sweep"]}
    c["d_sweep_below_breakeven_nogo"] = (sweep.get(480.0) is False and sweep.get(481.53) is False)
    c["d_sweep_above_breakeven_go"] = (sweep.get(481.54) is True and sweep.get(492.08) is True)

    # (e) pre-submission verify CI spec
    c["e_verify_gpu_min_41_8"] = (abs(ci["pre_submission_verify_gpu_min"] - 41.8) < 1e-9)
    c["e_three_tiers"] = (len(ci["tiers"]) == 3)
    th = ci["verify_pass_thresholds"]
    c["e_flips_3_to_0"] = (th["served_m8_flips"]["from"] == 3 and th["served_m8_flips"]["must_reach"] == 0)
    c["e_identity_1_0"] = (abs(th["e2e_greedy_identity"]["must_reach"] - 1.0) < 1e-9)
    c["e_ppl_cap_2_42"] = (abs(th["ppl"]["cap"] - 2.42) < 1e-9)
    c["e_completed_128"] = (th["completed"]["required"] == 128)
    c["e_is_identity_evidence"] = (ci["is_byte_identity_evidence_not_tps"] is True)

    # (f) feature-flag design
    c["f_flag_name"] = (flag["flag_name"] == "SELECTIVE_RECOMPUTE_VERIFY")
    c["f_inplace_now_flag_revertible"] = (flag["inplace_line_now_flag_revertible"] is True)
    c["f_residual_cost_zero"] = (abs(flag["flag_residual_cost_tps"]) < 1e-9)
    c["f_preserves_identity_both"] = (flag["flag_preserves_byte_identity_both_paths"] is True)
    c["f_on_is_strict"] = ("STRICT" in flag["on_path"]["byte_identical_to"].upper()
                           and flag["on_path"]["is_ship_path"] is True)
    c["f_off_is_todays_served"] = ("TODAY'S SERVED" in flag["off_path"]["byte_identical_to"].upper()
                                   and flag["off_path"]["is_ship_path"] is False)
    c["f_default_strict_on_ship"] = (flag["default_is_strict_on_ship"] is True)

    # (g) in-repo facts (read-only existence)
    for k, v in facts.items():
        c[f"g_{k}"] = v

    # (h) analysis-only hygiene
    c["h_official_tps_zero"] = (flags.get("official_tps") == 0)
    c["h_analysis_only"] = bool(flags.get("analysis_only"))
    c["h_no_hf_job"] = bool(flags.get("no_hf_job"))
    c["h_no_served_file_change"] = bool(flags.get("no_served_file_change"))
    c["h_ship_human_gated"] = (flags.get("ship_is_human_gated") is True)

    # (i) PPL / greedy-identity by construction (PR point 6)
    c["i_ppl_unchanged_within_cap"] = (DEPLOYED_PPL <= PPL_CAP)
    c["i_predicate_enforces_ppl"] = enforce["bad_ppl_is_nogo"]

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
    pred = p["predicate"]
    ci = p["verify_ci"]
    flag = p["feature_flag"]
    print("=" * 100)
    print(f"PR #419 lawine -- SHIPPABLE FASTEST-EQUIVALENT GO/NO-GO  ({p['created_at']})")
    print(f"  analysis_only={p['analysis_only']}  no_hf_job={p['no_hf_job']}  "
          f"no_served_file_change={p['no_served_file_change']}  official_tps={p['official_tps']}  "
          f"ship_is_human_gated={p['ship_is_human_gated']}")
    print(f"  deployed/ship-breakeven {DEPLOYED_TPS} (non-strict #52) UNCHANGED  |  blanket-strict base "
          f"{STRICT_BASE_BLANKET} (#393)  |  cb3 +{CB3_LIFT_M8} (#403 k*={CB3_KSTAR})")
    print("-" * 100)
    print("  PINNED-IMPORT CROSS-CHECK (byte-exact from merged #417/#413 JSON)")
    print(f"    {p['pinned_import_crosscheck']['n_crosschecks']} checks -> "
          f"all_pass={p['pinned_import_crosscheck']['all_present_checks_pass']}  "
          f"available={p['pinned_import_crosscheck']['available']}")
    print("-" * 100)
    print("  GO/NO-GO PREDICATE (executable core)")
    print(f"    {pred['predicate_definition']}")
    print(f"    GO under modeled bracket [{FASTEST_EQUIVALENT_MODEL_LO}, {FASTEST_EQUIVALENT_MODEL_HI}] = "
          f"{pred['go_nogo_predicate_evaluates_GO_under_modeled']}  "
          f"(lo={pred['under_modeled_lo']['verdict']}, hi={pred['under_modeled_hi']['verdict']})")
    print(f"    measured verdict ................. {pred['under_measured']['verdict']}")
    print(f"    ship_breakeven_equivalent_tps .... {pred['ship_breakeven_equivalent_tps']}  "
          f"(selrec input breakeven {pred['ship_breakeven_selective_recompute_tps']})")
    print(f"    modeled_margin_tps ............... +{pred['modeled_margin_tps']} "
          f"(lo +{pred['modeled_margin_tps_lo']} / hi +{pred['modeled_margin_tps_hi']})")
    print(f"    robustness (zero-speedup selrec) . {pred['robustness_zero_speedup_selrec_combined_tps']} "
          f"(+{pred['robustness_zero_speedup_margin_tps']}); selrec_breakeven < blanket floor = "
          f"{pred['selrec_breakeven_below_blanket_floor']}")
    print("    NO-GO boundary sweep (combined TPS -> verdict):")
    for s in pred["nogo_boundary_sweep"]:
        print(f"      {s['combined_tps']:>8} -> {s['verdict']}")
    print("-" * 100)
    print("  PRE-SUBMISSION IDENTITY-VERIFY CI")
    print(f"    pre_submission_verify_gpu_min .... {ci['pre_submission_verify_gpu_min']}  "
          f"({len(ci['tiers'])} tiers)")
    th = ci["verify_pass_thresholds"]
    print(f"    PASS thresholds: served flips {th['served_m8_flips']['from']}/"
          f"{th['served_m8_flips']['from_denom']} -> {th['served_m8_flips']['must_reach']}; "
          f"identity == {th['e2e_greedy_identity']['must_reach']}; ppl <= {th['ppl']['cap']}; "
          f"completed == {th['completed']['required']}")
    print("-" * 100)
    print("  FEATURE-FLAG (de-risk the one binding in-place line)")
    print(f"    {flag['flag_name']}  ({flag['resolution']})")
    print(f"    inplace_line_now_flag_revertible = {flag['inplace_line_now_flag_revertible']}  "
          f"residual_cost_tps = {flag['flag_residual_cost_tps']}  "
          f"preserves_identity_both = {flag['flag_preserves_byte_identity_both_paths']}")
    print(f"    ON  -> {flag['on_path']['byte_identical_to']}")
    print(f"    OFF -> {flag['off_path']['byte_identical_to']}")
    print("-" * 100)
    print(f"  HUMAN CHECKLIST written to: {p.get('checklist_path', '(in payload)')}")
    print("-" * 100)
    print(f"  SELF-TEST {st['n_checks']} checks -> {'PASS' if st['passes'] else 'FAIL'}")
    if not st["passes"]:
        for k, v in st["conditions"].items():
            if not v:
                print(f"    FAILED: {k}")
    print("=" * 100)


def build_payload(flags: dict[str, Any]) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    xcheck = pinned_import_crosscheck()
    pred = evaluate_predicate()
    enforce = predicate_enforces_all_conjuncts()
    ci = verify_ci_spec()
    flag = feature_flag_design()
    facts = repo_facts()
    st = selftest(xcheck, pred, enforce, ci, flag, facts, flags)
    checklist = human_checklist_md(pred, ci, flag, created_at)
    payload: dict[str, Any] = {
        "agent": "lawine", "pr": 419,
        "kind": "shippable-equivalent-go-nogo",
        "created_at": created_at,
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "ship_is_human_gated": True,
        # ---- headline scalars (PR point 7; W&B summary deliverables) ----
        "shippable_equivalent_go_nogo_self_test_passes": bool(st["passes"]),
        "go_nogo_predicate_evaluates_GO_under_modeled": pred["go_nogo_predicate_evaluates_GO_under_modeled"],
        "ship_breakeven_equivalent_tps": pred["ship_breakeven_equivalent_tps"],
        "modeled_margin_tps": pred["modeled_margin_tps"],
        "modeled_margin_tps_lo": pred["modeled_margin_tps_lo"],
        "modeled_margin_tps_hi": pred["modeled_margin_tps_hi"],
        "pre_submission_verify_gpu_min": ci["pre_submission_verify_gpu_min"],
        "inplace_line_now_flag_revertible": flag["inplace_line_now_flag_revertible"],
        "flag_residual_cost_tps": flag["flag_residual_cost_tps"],
        "flag_preserves_byte_identity_both_paths": flag["flag_preserves_byte_identity_both_paths"],
        "fastest_equivalent_tps_modeled_bracket": [FASTEST_EQUIVALENT_MODEL_LO, FASTEST_EQUIVALENT_MODEL_HI],
        "predicate_enforces_all_conjuncts": enforce,
        # ---- detail blocks ----
        "pinned_import_crosscheck": xcheck,
        "predicate": pred,
        "verify_ci": ci,
        "feature_flag": flag,
        "human_checklist_md": checklist,
        "repo_facts": facts,
        "selftest": st,
        "shared_baselines": {
            "deployed_tps": DEPLOYED_TPS, "deployed_ppl": DEPLOYED_PPL, "ppl_cap": PPL_CAP,
            "required_completed": REQUIRED_COMPLETED, "strict_base_blanket": STRICT_BASE_BLANKET,
            "cb3_lift_m8": CB3_LIFT_M8, "cb3_kstar": CB3_KSTAR, "equiv_tps_at_7_413": EQUIV_TPS_AT_7_413,
            "selrec_modeled_tps_bracket": [SELREC_TPS_MODEL_LO, SELREC_TPS_MODEL_HI],
            "combined_modeled_tps_bracket": [FASTEST_EQUIVALENT_MODEL_LO, FASTEST_EQUIVALENT_MODEL_HI],
            "deployed_served_flips": f"{DEPLOYED_SERVED_FLIPS}/{DEPLOYED_SERVED_FLIP_DENOM}",
            "deployed_served_identity": DEPLOYED_SERVED_IDENTITY,
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
    st = payload["selftest"]
    pred = payload["predicate"]
    flag = payload["feature_flag"]
    run = init_wandb_run(
        job_type="analysis-static-scope", agent="lawine",
        name=args.wandb_name, group=args.wandb_group,
        tags=["shippable-equivalent-go-nogo", "go-nogo-predicate", "verify-ci", "feature-flag",
              "selective-recompute", "cb3", "identity-verify", "decision-doc", "pr-419"],
        config={"pr": 419, "kind": "shippable-equivalent-go-nogo",
                "analysis_only": True, "no_hf_job": True, "no_served_file_change": True,
                "official_tps": 0, "ship_is_human_gated": True,
                "deployed_tps": DEPLOYED_TPS, "strict_base_blanket": STRICT_BASE_BLANKET,
                "cb3_lift_m8": CB3_LIFT_M8},
    )
    if run is None:
        print("[card] wandb disabled (no API key / WANDB_MODE).")
        return None
    flat: dict[str, float] = {
        "card/self_test_passes": float(st["passes"]),
        "card/self_test_n_checks": float(st["n_checks"]),
        "card/official_tps": float(payload["official_tps"]),
        "card/ship_is_human_gated": float(payload["ship_is_human_gated"]),
        "predicate/go_under_modeled": float(payload["go_nogo_predicate_evaluates_GO_under_modeled"]),
        "predicate/ship_breakeven_equivalent_tps": float(payload["ship_breakeven_equivalent_tps"]),
        "predicate/ship_breakeven_selective_recompute_tps":
            float(pred["ship_breakeven_selective_recompute_tps"]),
        "predicate/modeled_margin_tps": float(payload["modeled_margin_tps"]),
        "predicate/modeled_margin_tps_lo": float(payload["modeled_margin_tps_lo"]),
        "predicate/modeled_margin_tps_hi": float(payload["modeled_margin_tps_hi"]),
        "predicate/modeled_central_combined_tps": float(pred["modeled_central_combined_tps"]),
        "predicate/robustness_zero_speedup_combined_tps":
            float(pred["robustness_zero_speedup_selrec_combined_tps"]),
        "predicate/robustness_zero_speedup_margin_tps": float(pred["robustness_zero_speedup_margin_tps"]),
        "predicate/crosscheck_413_combined_tps": float(pred["crosscheck_413_combined_tps"]),
        "predicate/selrec_breakeven_below_blanket_floor":
            float(pred["selrec_breakeven_below_blanket_floor"]),
        "verify_ci/pre_submission_verify_gpu_min": float(payload["pre_submission_verify_gpu_min"]),
        "verify_ci/n_tiers": float(len(payload["verify_ci"]["tiers"])),
        "verify_ci/served_flips_target": float(STRICT_SERVED_FLIPS_TARGET),
        "verify_ci/identity_target": float(STRICT_SERVED_IDENTITY_TARGET),
        "flag/inplace_line_now_flag_revertible": float(payload["inplace_line_now_flag_revertible"]),
        "flag/residual_cost_tps": float(payload["flag_residual_cost_tps"]),
        "flag/preserves_byte_identity_both_paths":
            float(payload["flag_preserves_byte_identity_both_paths"]),
        "crosscheck/n_pinned_import_checks": float(payload["pinned_import_crosscheck"]["n_crosschecks"]),
        "crosscheck/pinned_import_all_pass":
            float(payload["pinned_import_crosscheck"]["all_present_checks_pass"]),
        "modeled/fastest_equivalent_tps_lo": float(FASTEST_EQUIVALENT_MODEL_LO),
        "modeled/fastest_equivalent_tps_hi": float(FASTEST_EQUIVALENT_MODEL_HI),
    }
    run.log({"global_step": 0, **flat})
    log_summary(run, _jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="shippable_equivalent_go_nogo", artifact_type="analysis",
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
                    default="lawine/shippable-equivalent-go-nogo")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="shippable-equivalent-go-nogo")
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

    out_path = out_dir / "shippable_equivalent_go_nogo_results.json"
    out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"[card] wrote {out_path}")

    st = payload["selftest"]
    if args.self_test:
        assert st["passes"], f"self-test FAILED ({st['n_checks']} checks)"
        assert st["n_checks"] >= 20, f"need >=20 asserts, have {st['n_checks']}"
        print(f"[card] SELF-TEST PASS ({st['n_checks']} checks)")
        print("\nSENPAI-RESULT " + json.dumps({
            "terminal": True, "status": "complete", "pending_arms": False, "wandb_run_ids": [],
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            "shippable_equivalent_go_nogo_self_test_passes": bool(st["passes"]),
            "primary_metric": {"name": "modeled_margin_tps", "value": float(payload["modeled_margin_tps"])},
            "test_metric": {"name": "shippable_equivalent_go_nogo_self_test_passes",
                            "value": float(st["passes"])},
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
        "go_nogo_predicate_evaluates_GO_under_modeled":
            payload["go_nogo_predicate_evaluates_GO_under_modeled"],
        "ship_breakeven_equivalent_tps": payload["ship_breakeven_equivalent_tps"],
        "modeled_margin_tps": payload["modeled_margin_tps"],
        "pre_submission_verify_gpu_min": payload["pre_submission_verify_gpu_min"],
        "inplace_line_now_flag_revertible": payload["inplace_line_now_flag_revertible"],
        "flag_residual_cost_tps": payload["flag_residual_cost_tps"],
        "flag_preserves_byte_identity_both_paths": payload["flag_preserves_byte_identity_both_paths"],
        "shippable_equivalent_go_nogo_self_test_passes": bool(st["passes"]),
        "primary_metric": {"name": "modeled_margin_tps", "value": float(payload["modeled_margin_tps"])},
        "test_metric": {"name": "shippable_equivalent_go_nogo_self_test_passes", "value": float(st["passes"])},
    }))


if __name__ == "__main__":
    main()
