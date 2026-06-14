#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Launch-packet refresh: enthrone both-bugs as the primary GO (PR #179).

WHAT THIS IS
------------
My #167 packet (MERGED) and my #174 verdict-flip (MERGED) settled that, at the
SHIPPED launch-realized decode step 1.2182, descent-only-first is a knife-edge MISS
(LCB(P>=0.9)=499.97, P(clear-500)=89.94%, -0.035 TPS) while both-bugs is the robust
GO (LCB 514.88, P 99.59%). But the reviewer-facing packet I own still reads
descent-only-FIRST. This refresh RE-ISSUES the single go/no-go artifact with the
FLIPPED verdict as the headline -- both-bugs primary GO -- and assembles the now
nearly-complete leg set into one current dependency ledger so the human
`Approval request: HF job` decision is made against the real geometry.

PURE-ANALYTIC, CPU-ONLY. No GPU / vLLM / HF Job / submission / kernel build /
served-file change. BASELINE stays 481.53; greedy untouched; this banks the
analysis (PRIMARY = self-test, adds 0 TPS). It IMPORTS every merged leg VERBATIM
(my #174 verdict engine -> #167 packet -> #155 consolidator + #162 frontier; lawine
#168 step anchors; kanna #159 sigma_hw; denken #172 E[T] + floor; stark #164 native
private drop; denken #166 PPL stamp) and does NOT re-derive them. It does NOT file
the issue or authorize a launch.

THE TWO-AXIS QUADRATURE (why #174's numbers ARE the full-quadrature numbers)
---------------------------------------------------------------------------
The launch confidence factors into two INDEPENDENT axes:
  (1) PROJECTION axis = the 3-term input-band quadrature
      combined = sqrt(sampling^2 + calibration^2 + step_anchor^2)   (#155/#167/#174).
      At shipped 1.2182: descent P(clear-500)=0.8994 (LCB 499.97); both 0.9959 (LCB 514.88).
  (2) HARDWARE axis = kanna #159 sigma_hw = 4.86 TPS, cross-allocation-dominated.
      A single official draw clears the hardware axis with P~0.869; BEST-OF-2 official
      draws lift it to P=0.983 >= 0.90. So sigma_hw is RETIRED by a best-of-2 launch
      protocol -- it does NOT subtract from the projection-axis LCB.
The "full quadrature (sigma_hw (+) input bands)" launch is GO iff the projection axis
clears (P>=0.9 on the input bands) AND the hardware axis clears (best-of-2 -> P>=0.9).
Because best-of-2 retires the hardware axis, the PROJECTION axis is the binding
constraint, so #174's 499.97 / 514.88 remain the headline geometry. We ALSO report the
naive-fold sensitivity (sigma_hw as a literal 4th quadrature term) to show the verdict is
INVARIANT to how sigma_hw is composed: descent-only misses the shipped step either way,
both-bugs is GO either way.

STEP FRAMINGS (PR step 1) -- all three from lawine #168's step-anchor reconciliation:
  roofline 1.2127  (graphed-overlap floor; recovered only by a fully-graphed attn build)
  shipped  1.2182  (measured-idle-hidden overlap; the LAUNCH-REALIZED served reality)
  scatter-LP 1.2047 (#154 argmax-only decode; applies ONLY if that build ships)

SELF-TEST (PR step 4 -- PRIMARY = launch_packet_refresh_self_test_passes)
-------------------------------------------------------------------------
(a) reproduces #174's shipped-step verdict (descent LCB 499.97 / P 0.8994; both LCB
    514.88 / P 0.9959) within tolerance, AND cross-checks against #174's banked JSON;
(b) every LANDED ledger leg value matches its merged source PR artifact;
(c) the headline recommendation is both-bugs-GO with the four named gates
    (land #71 kernel, darwin boot-fix, PRECACHE_BENCH=1, human Approval-request) and
    the explicit human-approval deferral;
(d) the #154 restoration delta (+3.96 TPS LCB, imported from #174) is reported;
(e) NaN-clean.
TEST = both_bugs_launch_lcb_tps (both-bugs LCB(P>=0.9) at the shipped 1.2182).

This is my launch decision-geometry / packet-consolidation lane (the refresh of my own
#167/#174). Distinct from every leg it consumes. NOT open2 (launch economics, not drafter
architecture). NOT a launch. Serves nothing -> greedy identity untouched.
"""
from __future__ import annotations

import argparse
import glob as _glob
import importlib.util
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# Import my #174 verdict engine VERBATIM (it transitively loads #167 packet ->
# #155 consolidator + #162 frontier). One source of truth per constant.
# ============================================================================
v174 = _load("conservative_step_launch_verdict",
             os.path.join(_HERE, "conservative_step_launch_verdict.py"))

# engine + loaders (reused exactly, not re-derived)
cell = v174.cell
cons = v174.cons
SamplingModel = v174.SamplingModel
build_joint_b_dict = v174.build_joint_b_dict
b_dict_for_depth1 = v174.b_dict_for_depth1
DEPTH1_DESCENT_ONLY = v174.DEPTH1_DESCENT_ONLY
load_stark151_retention = v174.load_stark151_retention
load_kcal_band = v174.load_kcal_band
load_stark156_pinned = v174.load_stark156_pinned
load_lawine161_anchors = v174.load_lawine161_anchors
load_step163_realizable = v174.load_step163_realizable
load_denken166_ppl = v174.load_denken166_ppl

# constants (verbatim)
K_CAL = v174.K_CAL                       # 125.268
TARGET_OFFICIAL = v174.TARGET_OFFICIAL   # 500.0
FRONTIER_OFFICIAL = cons.FRONTIER_OFFICIAL  # 481.53
Z_P90 = v174.Z_P90_ONESIDED              # 1.281552
STEP_BASE = v174.STEP_MEASURED_DEPTH9    # 1.2182 (base step for the sampling CI)
STEP_REL_DEFAULT = v174.STEP_REL_1SIGMA_DEFAULT  # 0.005
P_GO = v174.P_GO                         # 0.9 conservative one-shot bar

# committed leg source artifacts (imported, not re-derived)
STEP168_JSON = os.path.join(_ROOT, "research/spec_cost_model/step_anchor_reconciliation.json")
SIGMA_HW_JSON = os.path.join(_ROOT, "research/validity/hw_variance_envelope/envelope.json")
DENKEN172_GLOB = os.path.join(_ROOT, "research/validity/descent_et_audit/runs/*/descent_et_dp_audit_result.json")
STARK164_JSON = os.path.join(_ROOT, "research/validity/descent_vs_bothbugs_private/results.json")
V174_JSON = os.path.join(_ROOT, "research/spec_cost_model/conservative_step_launch_verdict_results.json")

# robust-vs-marginal headroom above 500 (TPS) for the projection-axis LCB
ROBUST_LCB_MARGIN_TPS = 2.0

# the four named launch gates (PR step 3)
LAUNCH_GATES = [
    "land #71 builds the both-bugs descending accept-prep kernel (the GO-path gating build)",
    "kanna's darwin _IncludedRouter boot-validation startup-500 fix folded into the serve harness",
    "PRECACHE_BENCH=1 set on the served path",
    "a human-approved `Approval request: HF job` issue",
]
HUMAN_DEFERRAL = ("This packet INFORMS but does NOT authorize a launch; a human must "
                  "approve the filed `Approval request: HF job` issue before any spend.")


def _finite(x, default: float = 0.0) -> float:
    try:
        return float(x) if (x is not None and math.isfinite(float(x))) else default
    except (TypeError, ValueError):
        return default


def _rel_err(v, ref):
    return abs(v - ref) / abs(ref) if ref else float("inf")


# ============================================================================
# Leg-source loaders (verify LANDED values against their merged artifacts).
# ============================================================================
def load_step168_framings(path: str = STEP168_JSON) -> dict:
    """lawine #168 -- the THREE step framings PR #179 names, from ONE source."""
    with open(path) as f:
        j = json.load(f)
    anchors = {a["name"]: a["value"] for a in j["step1_anchors"]}
    return {
        "roofline": float(anchors["roofline_ideal_overlap"]),                 # 1.2127
        "shipped": float(j["step2_launch_realized_step"]["both_bugs"]),       # 1.2182
        "scatter_lp": float(anchors["scatter_lp_reduced_decode_path"]),       # 1.2047
        "band_half_width_pct": float(j["step2_launch_realized_step"]["band"]["half_width_pct"]),
        "e_t_descent_only": float(j["constants"]["e_t_descent_only"]),
        "e_t_both_bugs": float(j["constants"]["e_t_both_bugs"]),
        "clear_500_bar_overlap": float(j["step3_propagation"]["clear_500_bar_overlap"]),
        "clear_500_bar_roofline": float(j["step3_propagation"]["clear_500_bar_roofline"]),
    }


def load_kanna159_sigma_hw(path: str = SIGMA_HW_JSON) -> dict:
    """kanna #159 -- sigma_hw envelope + best-of-N hardware-axis re-draw budget."""
    with open(path) as f:
        j = json.load(f)
    prop = j["propagation"]
    bo = prop["redraw_budget_hardware"]["best_of_n"]
    ladder = {int(r["n"]): float(r["p_best_of_n"]) for r in bo["ladder"]}
    return {
        "sigma_hw_tps": float(prop["sigma_hw_tps"]),
        "sigma_hw_pct": float(prop["sigma_hw_pct"]),
        "central_tps": float(prop["central_tps"]),
        "cross_allocation_dominated": bool(
            j["envelope"]["sigma_within_pct"] < 0.1 * j["envelope"]["sigma_cross_pct"]),
        "p_single_draw": float(bo["p_single"]),
        "best_of_2": float(ladder.get(2, float("nan"))),
        "n_for_p90": int(bo["n_for_target"]),
        "redraw_caveat": prop["redraw_budget_hardware"]["caveat"],
    }


def load_denken172_et(glob_pat: str = DENKEN172_GLOB) -> dict:
    """denken #172 -- descent E[T] central + adversarial self-KV floor."""
    hits = sorted(_glob.glob(glob_pat))
    with open(hits[-1]) as f:
        j = json.load(f)
    lb = j["propagate"]["lower_bound"]["overlap_1p2182"]
    return {
        "descent_central": float(j["descent_only_E_T_recomputed"]),   # 5.0564
        "both_bugs": float(j["both_bugs_E_T"]),                        # 5.2070
        "descent_floor": float(j["descent_only_E_T_lower_bound"]),     # 3.5346
        "floor_official_tps": float(lb["official_tps"]),               # 363.5
        "floor_clears_500": bool(lb["clears_500"]),                    # False
        "lambda_star_clear500_overlap": float(
            j["lower_bound"]["lambda_star_clear500_overlap"]),         # 0.9085 (>=91% spread recovery)
        "verdict": j["verdict"],                                       # BOUNDED-NOT-ROBUST
        "path": hits[-1],
    }


def load_stark164_private(path: str = STARK164_JSON) -> dict:
    """stark #164 -- native private-drop CI + descent worst-case tau-low."""
    with open(path) as f:
        j = json.load(f)
    h = j["headline"]
    # tau_low = the WITHIN-worst-proxy private retention (the worst proxy's tps_taulow / its
    # tps_central), matching PR #179's pinned tau in [0.9924, 1.0]. The worst proxy is the one
    # with the lowest tau-low TPS (native_code: 504.6105/508.4586 = 0.99243).
    proxies = [p for p in j["proxies"] if "descent_only" in p]
    worst = min(proxies, key=lambda p: p["descent_only"]["tps_taulow"])
    tau_low = float(worst["descent_only"]["tps_taulow"]) / float(worst["descent_only"]["tps_central"])
    return {
        "descent_drop_mid_pct": float(h["tree_private_drop_pct_native_ci"]),     # 2.041
        "descent_drop_band_pct": [float(x) for x in h["tree_private_drop_pct_native_ci_band"]],  # [1.871,2.210]
        "descent_taulow_min_tps": float(h["descent_only_tps_taulow_min"]),       # 504.61
        "descent_worst_margin_to_500": float(h["descent_only_worst_margin_to_500"]),  # 8.46
        "both_bugs_worst_margin_to_500": float(h["both_bugs_worst_margin_to_500"]),    # 23.4
        "descent_private_safe": bool(h["descent_only_private_safe_native"]),     # True
        "tau_low": tau_low,                                                       # ~0.9924 (PR-pinned)
    }


def load_v174_banked(path: str = V174_JSON) -> dict:
    """My #174 verdict (import faithfulness cross-check + the +3.96 restoration delta)."""
    with open(path) as f:
        j = json.load(f)
    cons_cell = j["step1_three_step_instantiation"]["conservative_1p2182"]
    fs = j["step2_first_shot_settlement"]
    return {
        "descent_shipped": cons_cell["descent_only"],
        "both_shipped": cons_cell["both_bugs"],
        "restoration_154_lcb_tps": float(fs["cost_of_not_shipping_154_lcb_tps"]),   # +3.96
        "restoration_154_proj_tps": float(fs["cost_of_not_shipping_154_proj_tps"]),
        "realizable_descent_lcb": float(
            j["step1_three_step_instantiation"]["realizable_1p2086"]["descent_only"]["lcb_p90"]),  # 503.92
        "wandb_run_id": j.get("wandb_run_id"),
    }


# ============================================================================
# sigma_hw naive-fold sensitivity (transparency): treat sigma_hw as a literal
# 4th quadrature term and recompute the projection-axis LCB / P(clear-500).
# Shows the verdict is INVARIANT to how sigma_hw is composed.
# ============================================================================
def fold_sigma_hw(c: dict, sigma_hw_pct: float) -> dict:
    combined3 = float(c["combined_rel_1sigma"])
    proj = float(c["proj_private_tps"])
    combined4 = math.hypot(combined3, sigma_hw_pct / 100.0)
    sigma4 = proj * combined4
    lcb4 = proj - Z_P90 * sigma4
    # P(clear-500) on the projection axis with the widened sigma
    p4 = 0.5 * (1.0 + math.erf((proj - TARGET_OFFICIAL) / (sigma4 * math.sqrt(2.0))))
    return {
        "combined_rel_4term": combined4,
        "lcb_p90_4term": lcb4,
        "p_clear_500_4term": p4,
        "go_4term": "GO" if (p4 >= P_GO and lcb4 >= TARGET_OFFICIAL) else "HOLD",
    }


# ============================================================================
# The reviewer-facing dependency ledger (PR step 2): one row per leg, with the
# current value/band, owner PR, and status (LANDED / IN-FLIGHT / PENDING-BUILD).
# ============================================================================
def build_dependency_ledger(framings, sigma_hw, denken172, stark164, denken166) -> dict:
    d166 = "M=32 batched-verify worst-case PPL <= 2.42"
    if denken166 is not None and "worst_case" in denken166:
        d166 = (f"M=32 worst-case PPL {denken166['worst_case']:.4f} <= {denken166['cap']:.2f} "
                f"(margin {denken166['margin']:.4f})")

    legs = [
        # ---- Numerator (E[T]) ----
        {"axis": "Numerator E[T]", "leg": "descent 5.0564 / both-bugs 5.2070",
         "owner_pr": "#160/#165/#172", "status": "LANDED",
         "value": f"descent {denken172['descent_central']:.4f} / both {denken172['both_bugs']:.4f}",
         "note": (f"denken #172 caveat: central is OPTIMISTIC; adversarial self-KV floor "
                  f"{denken172['descent_floor']:.4f} -> {denken172['floor_official_tps']:.1f} TPS "
                  f"FAILS 500 ({denken172['verdict']}); clearing needs >= "
                  f"{denken172['lambda_star_clear500_overlap']*100:.0f}% deep-spine spread recovery")},
        {"axis": "Numerator E[T] de-risk", "leg": "realistic-floor refinement",
         "owner_pr": "denken (in-flight)", "status": "IN-FLIGHT",
         "value": "central +- realistic floor (replaces the adversarial 3.5346 worst case)",
         "note": "the de-risk for the #172 floor caveat -- turns the modeled floor measured"},
        # ---- Denominator (step) ----
        {"axis": "Denominator step", "leg": f"reconciled shipped {framings['shipped']:.4f}",
         "owner_pr": "#168", "status": "LANDED",
         "value": (f"roofline {framings['roofline']:.4f} <-> shipped {framings['shipped']:.4f} "
                   f"(+-{framings['band_half_width_pct']:.2f}%); scatter-LP {framings['scatter_lp']:.4f} if #154 ships"),
         "note": "lawine #168 collapses #136/#154/#161 to one launch-realized step"},
        {"axis": "Denominator step confirm", "leg": "descent-walk step-neutrality",
         "owner_pr": "#173", "status": "IN-FLIGHT",
         "value": f"confirms the built descent kernel realizes {framings['shipped']:.4f}",
         "note": "#168 reconciles analytically; #173 confirms the descent walk realizes it"},
        # ---- Hardware (sigma_hw) ----
        {"axis": "Hardware sigma_hw", "leg": f"{sigma_hw['sigma_hw_tps']:.2f} TPS envelope",
         "owner_pr": "#159", "status": "LANDED",
         "value": (f"sigma_hw {sigma_hw['sigma_hw_tps']:.2f} TPS ({sigma_hw['sigma_hw_pct']:.2f}%), "
                   f"cross-allocation-dominated; best-of-2 official draws -> P="
                   f"{sigma_hw['best_of_2']:.4f} >= 0.90 on the hardware axis"),
         "note": ("RETIRED by a best-of-2 launch protocol -- a SEPARATE axis, NOT folded into the "
                  "projection LCB (single-draw hardware-axis P=" f"{sigma_hw['p_single_draw']:.3f})")},
        # ---- Finite-sample CI ----
        {"axis": "Finite-sample CI", "leg": "numerator 2nd-moment sampling term",
         "owner_pr": "#175 (wirbel)", "status": "PENDING",
         "value": "composes in quadrature with the input-band sampling term",
         "note": "arms the finite-sample slot of the projection-axis CI"},
        # ---- Validity ----
        {"axis": "Validity", "leg": "PPL <= 2.42 / 128 / greedy-exact",
         "owner_pr": "#166/#150/#158, Issue #124", "status": "LANDED",
         "value": f"official PPL 2.377 <= 2.42; {d166}; 128/128; greedy-exact (Issue #124 RESOLVED)",
         "note": "the quality contract -- official 2.3772 (BASELINE) under cap; denken #166 bounds the M=32 batched-verify worst case at the loaded 2.4134"},
        # ---- Private drop ----
        {"axis": "Private drop", "leg": f"native CI mid {stark164['descent_drop_mid_pct']:.2f}%",
         "owner_pr": "#164", "status": "LANDED",
         "value": (f"descent native drop {stark164['descent_drop_mid_pct']:.2f}% "
                   f"[{stark164['descent_drop_band_pct'][0]:.2f},{stark164['descent_drop_band_pct'][1]:.2f}]; "
                   f"descent worst tau-low {stark164['descent_taulow_min_tps']:.1f} TPS (clears 500)"),
         "note": ("descent-only is private-SAFE at the native worst case "
                  f"(margin +{stark164['descent_worst_margin_to_500']:.1f} TPS)")},
        {"axis": "Private drop stress", "leg": "adverse-skew private stress",
         "owner_pr": "#176 (stark)", "status": "IN-FLIGHT",
         "value": "stresses the private drop under an adverse domain skew",
         "note": "hardens the #164 native CI against worst-case private-mix"},
        # ---- Launch-boot ----
        {"axis": "Launch-boot", "leg": "darwin _IncludedRouter startup-500 fix",
         "owner_pr": "kanna boot-validation (in-flight)", "status": "IN-FLIGHT",
         "value": "fixes a startup-500 in _IncludedRouter; a NEW hard serve dependency",
         "note": "a hard serve gate alongside PRECACHE_BENCH=1 -- must be folded into the harness"},
        # ---- Build ----
        {"axis": "Build", "leg": "both-bugs descending accept-prep kernel",
         "owner_pr": "#71 (land)", "status": "PENDING-BUILD",
         "value": "the GO-path gating build (measured tuple: E[T], rho2, lambda, mu, step, ppl, boots, completed)",
         "note": "the kernel this packet gates -- WIP; the GO is conditional on it building"},
    ]
    by_status = {}
    for l in legs:
        by_status.setdefault(l["status"], []).append(l["axis"] + " :: " + l["leg"])
    return {
        "legs": legs,
        "n_landed": sum(1 for l in legs if l["status"] == "LANDED"),
        "n_in_flight": sum(1 for l in legs if l["status"] == "IN-FLIGHT"),
        "n_pending": sum(1 for l in legs if l["status"] == "PENDING"),
        "n_pending_build": sum(1 for l in legs if l["status"] == "PENDING-BUILD"),
        "by_status": by_status,
    }


# ============================================================================
# The launch recommendation (PR step 3): both-bugs primary GO + the descent-only
# restoration path + the build-recommendation call.
# ============================================================================
def build_recommendation(both_shipped, desc_shipped, v174b, framings, table) -> dict:
    restoration = v174b["restoration_154_lcb_tps"]            # +3.96
    desc_restored_lcb = v174b["realizable_descent_lcb"]       # 503.92
    desc_scatterlp = table["scatter_lp"]["descent_only"]      # 1.2047 framing
    return {
        "headline": "both-bugs is the robust GO",
        "both_bugs_launch_lcb_tps": float(both_shipped["lcb_p90"]),    # 514.88
        "both_bugs_p_clear_500": float(both_shipped["p_clear_500"]),    # 0.9959
        "recommended_first_shot": "both-bugs",
        "launch_go": both_shipped["launch_go_p90"],
        "gates": LAUNCH_GATES,
        "human_approval_deferral": HUMAN_DEFERRAL,
        "descent_only_restoration": {
            "path": "ship #154's argmax-only decode (then descent-only is the simpler build)",
            "restoration_lcb_tps": restoration,                        # +3.96 (imported, #174)
            "descent_restored_lcb_at_realizable": desc_restored_lcb,   # 503.92 GO
            "descent_lcb_at_scatter_lp_1p2047": float(desc_scatterlp["lcb_p90"]),
            "note": (
                f"Shipping #154's argmax-only decode adds +{restoration:.2f} TPS LCB "
                f"(shipped->realizable), restoring descent-only to GO (LCB {desc_restored_lcb:.2f}); "
                f"the full scatter-LP framing 1.2047 lifts descent-only to LCB "
                f"{desc_scatterlp['lcb_p90']:.1f}. So +{restoration:.2f} is the CONSERVATIVE restoration."),
        },
        "build_recommendation": {
            "changes_build_rec": False,
            "primary": ("land #71's both-bugs kernel -- robust GO at ALL three framings "
                        f"(LCB {both_shipped['lcb_p90']:.1f} shipped), no #154 dependency"),
            "alternative": ("descent-only (simpler build, no BUG-1 spine) + #154 argmax-only serve patch -- "
                            f"restores GO at +{restoration:.2f} TPS (LCB {desc_restored_lcb:.1f}) BUT thinner "
                            "margin, conditional on #154 shipping, and carries the #172 BOUNDED-NOT-ROBUST "
                            "E[T] floor caveat"),
            "verdict": ("build recommendation UNCHANGED: land #71's both-bugs kernel is the gating build for "
                        "the robust GO; the descent-only+#154 path is a viable simpler-build fallback only "
                        "once #154 ships and the descent E[T] realistic-floor is confirmed"),
        },
    }


# ============================================================================
# Self-test (PR step 4 -- PRIMARY).
# ============================================================================
def self_test(table, ledger, recommendation, v174b, sources, sigma_hw_fold) -> dict:
    # #174 banked shipped-step references
    REF_DESC_LCB, REF_DESC_P = 499.96519706601964, 0.8994368544296176
    REF_BOTH_LCB, REF_BOTH_P = 514.877540689496, 0.9958868982631068

    desc = table["shipped"]["descent_only"]
    both = table["shipped"]["both_bugs"]

    # (a) reproduce #174's shipped verdict within tol AND cross-check against #174's JSON.
    a_repro = bool(
        _rel_err(desc["lcb_p90"], REF_DESC_LCB) <= 1e-4 and abs(desc["p_clear_500"] - REF_DESC_P) <= 5e-3
        and _rel_err(both["lcb_p90"], REF_BOTH_LCB) <= 1e-4 and abs(both["p_clear_500"] - REF_BOTH_P) <= 5e-3)
    a_xcheck = bool(
        _rel_err(desc["lcb_p90"], v174b["descent_shipped"]["lcb_p90"]) <= 1e-9
        and _rel_err(both["lcb_p90"], v174b["both_shipped"]["lcb_p90"]) <= 1e-9)
    a_knife = bool(desc["launch_go_p90"] != "GO" and both["launch_go_p90"] == "GO")
    assert_a = bool(a_repro and a_xcheck and a_knife)

    # (b) every LANDED ledger leg value matches its merged source artifact.
    f = sources["framings"]
    g = sources["sigma_hw"]
    d172 = sources["denken172"]
    s164 = sources["stark164"]
    checks_b = {
        "e_t_descent_5p0564": _rel_err(d172["descent_central"], 5.056404568844709) <= 1e-9,
        "e_t_both_5p2070": _rel_err(d172["both_bugs"], 5.206954309441963) <= 1e-9,
        "e_t_floor_3p5346": _rel_err(d172["descent_floor"], 3.5345806333738627) <= 1e-9,
        "e_t_consistent_lawine168": (_rel_err(f["e_t_descent_only"], d172["descent_central"]) <= 1e-9
                                     and _rel_err(f["e_t_both_bugs"], d172["both_bugs"]) <= 1e-9),
        "step_shipped_1p2182": _rel_err(f["shipped"], 1.2181727676912677) <= 1e-9,
        "step_roofline_1p2127": _rel_err(f["roofline"], 1.2127483746822987) <= 1e-9,
        "step_scatter_lp_1p2047": _rel_err(f["scatter_lp"], 1.2046765466054148) <= 1e-9,
        "sigma_hw_4p86": _rel_err(g["sigma_hw_tps"], 4.864468814937121) <= 1e-6,
        "sigma_hw_best_of_2_ge_p90": bool(g["best_of_2"] >= P_GO),
        "sigma_hw_cross_dominated": bool(g["cross_allocation_dominated"]),
        "private_drop_mid_2p04": _rel_err(s164["descent_drop_mid_pct"], 2.040947456110656) <= 1e-6,
        "private_descent_taulow_504p6": _rel_err(s164["descent_taulow_min_tps"], 504.6105118808893) <= 1e-6,
        "private_descent_safe": bool(s164["descent_private_safe"]),
        "k_cal_125p268": _rel_err(K_CAL, 125.26795005202914) <= 1e-9,
        "frontier_481p53": _rel_err(FRONTIER_OFFICIAL, 481.53) <= 1e-6,
    }
    assert_b = bool(all(checks_b.values()))

    # (c) headline recommendation is both-bugs-GO with the four gates + human deferral.
    assert_c = bool(
        recommendation["headline"] == "both-bugs is the robust GO"
        and recommendation["recommended_first_shot"] == "both-bugs"
        and recommendation["launch_go"] == "GO"
        and len(recommendation["gates"]) == 4
        and all(isinstance(x, str) and x for x in recommendation["gates"])
        and "does NOT authorize" in recommendation["human_approval_deferral"])

    # (d) the #154 restoration delta (+3.96 TPS LCB) is reported (imported from #174).
    restoration = recommendation["descent_only_restoration"]["restoration_lcb_tps"]
    assert_d = bool(abs(restoration - 3.9566734582135723) <= 1e-6 and abs(restoration - 3.96) <= 0.05)

    # (e) NaN-clean across every headline numeric.
    nums = []
    for fr in table:
        for topo in ("descent_only", "both_bugs"):
            c = table[fr][topo]
            nums += [c["official_tps"], c["proj_private_tps"], c["p_clear_500"], c["lcb_p90"]]
    nums += [recommendation["both_bugs_launch_lcb_tps"], recommendation["both_bugs_p_clear_500"], restoration]
    for k in ("descent_only", "both_bugs"):
        nums += [sigma_hw_fold[k]["lcb_p90_4term"], sigma_hw_fold[k]["p_clear_500_4term"]]
    assert_e = bool(all(x is not None and math.isfinite(x) for x in nums))

    passes = int(bool(assert_a and assert_b and assert_c and assert_d and assert_e))
    return {
        "passes": bool(passes),
        "launch_packet_refresh_self_test_passes": passes,
        "assert_a_reproduces_174_shipped_verdict": {
            "ok": assert_a, "repro_within_tol": a_repro, "xcheck_vs_174_json": a_xcheck,
            "knife_edge_descent_miss_both_go": a_knife,
            "desc_lcb": desc["lcb_p90"], "desc_p": desc["p_clear_500"],
            "both_lcb": both["lcb_p90"], "both_p": both["p_clear_500"],
            "ref_desc_lcb": REF_DESC_LCB, "ref_both_lcb": REF_BOTH_LCB,
            "expect": "shipped descent LCB 499.97/P 0.8994, both LCB 514.88/P 0.9959; descent miss + both GO"},
        "assert_b_landed_legs_match_sources": {"ok": assert_b, "checks": checks_b,
            "expect": "every LANDED ledger value matches its merged source PR artifact"},
        "assert_c_both_bugs_go_with_four_gates": {
            "ok": assert_c, "headline": recommendation["headline"],
            "n_gates": len(recommendation["gates"]), "gates": recommendation["gates"],
            "human_deferral_present": "does NOT authorize" in recommendation["human_approval_deferral"],
            "expect": "headline both-bugs-GO, 4 named gates, explicit human-approval deferral"},
        "assert_d_154_restoration_reported": {
            "ok": assert_d, "restoration_lcb_tps": restoration,
            "descent_restored_lcb": recommendation["descent_only_restoration"]["descent_restored_lcb_at_realizable"],
            "expect": "+3.96 TPS LCB (imported from #174), restoring descent-only to GO"},
        "assert_e_nan_clean": {"ok": assert_e, "n_numbers_checked": len(nums)},
    }


# ============================================================================
# Reviewer-facing packet markdown (PR step 5) -- the single go/no-go artifact.
# ============================================================================
def render_packet_md(table, ledger, recommendation, sigma_hw, sigma_hw_fold, framings, denken172, tau_low) -> str:
    def row(label, c):
        return (f"| {label} | {c['official_tps']:.1f} | {c['proj_private_tps']:.1f} | "
                f"{c['p_clear_500']*100:.2f}% | {c['lcb_p90']:.2f} | {c['launch_go_p90']} |")

    sd, ss, sl = table["shipped"]["descent_only"], table["shipped"]["both_bugs"], None
    rd, rb = table["roofline"]["descent_only"], table["roofline"]["both_bugs"]
    cd, cb = table["scatter_lp"]["descent_only"], table["scatter_lp"]["both_bugs"]
    rec = recommendation
    landed = "\n".join(f"- [LANDED]  {l['axis']} ({l['owner_pr']}) -- {l['value']}"
                       for l in ledger["legs"] if l["status"] == "LANDED")
    inflight = "\n".join(f"- [IN-FLIGHT] {l['axis']} ({l['owner_pr']}) -- {l['value']}"
                         for l in ledger["legs"] if l["status"] == "IN-FLIGHT")
    pend = "\n".join(f"- [{l['status']}] {l['axis']} ({l['owner_pr']}) -- {l['value']}"
                     for l in ledger["legs"] if l["status"] in ("PENDING", "PENDING-BUILD"))
    gates = "\n".join(f"  {i+1}. {g}" for i, g in enumerate(rec["gates"]))

    return f"""\
### Approval request: HF job for tree submission -- REFRESHED go/no-go packet (PRE-FILLED DRAFT, NOT FILED)

**HEADLINE: both-bugs is the robust GO** -- LCB(P>=0.9) {ss['lcb_p90']:.2f} TPS, P(clear-500) {ss['p_clear_500']*100:.2f}%
at the shipped launch-realized step {framings['shipped']:.4f}. descent-only-first is a knife-edge MISS at the
shipped step (LCB {sd['lcb_p90']:.2f}, P {sd['p_clear_500']*100:.2f}%, -{TARGET_OFFICIAL - sd['lcb_p90']:.3f} TPS).

**{HUMAN_DEFERRAL}**

**Composition:** `official = K_cal * (E[T]/step) * tau` (K_cal={K_CAL:.3f}, tau in [{tau_low:.4f}, 1.0]);
clear-500 bar E[T] >= {framings['clear_500_bar_overlap']:.3f} (shipped). TWO independent launch axes:
- PROJECTION axis = 3-term input-band quadrature sqrt(sampling^2 + calibration^2 + step_anchor^2).
- HARDWARE axis = kanna #159 sigma_hw {sigma_hw['sigma_hw_tps']:.2f} TPS (cross-allocation-dominated),
  RETIRED by best-of-2 official draws -> P={sigma_hw['best_of_2']:.4f} >= 0.90 (does NOT subtract from the LCB).

**Projection geometry at the three step framings (full-recovery corner lambda=mu=1, pinned 1.80% drop):**

| framing / topology | official | proj_private | P(clear 500) | LCB(P>=0.9) | launch |
|---|---|---|---|---|---|
{row(f"roofline {framings['roofline']:.4f} -- descent-only", rd)}
{row(f"roofline {framings['roofline']:.4f} -- both-bugs", rb)}
{row(f"shipped {framings['shipped']:.4f} -- descent-only (MISS)", sd)}
{row(f"shipped {framings['shipped']:.4f} -- both-bugs (GO)", ss)}
{row(f"scatter-LP {framings['scatter_lp']:.4f} -- descent-only", cd)}
{row(f"scatter-LP {framings['scatter_lp']:.4f} -- both-bugs", cb)}

both-bugs is GO at all three framings (LCB {min(rb['lcb_p90'], ss['lcb_p90'], cb['lcb_p90']):.1f} ->
{max(rb['lcb_p90'], ss['lcb_p90'], cb['lcb_p90']):.1f}); descent-only is the knife-edge miss at the shipped
step ONLY (GO at roofline {rd['lcb_p90']:.1f} and scatter-LP {cd['lcb_p90']:.1f}).

**MATERIAL CAVEAT -- numerator E[T] floor (denken #172):** the central numerator (descent {denken172['descent_central']:.4f} /
both {denken172['both_bugs']:.4f}) is the OPTIMISTIC full-recovery value used above. denken #172's adversarial self-KV
floor {denken172['descent_floor']:.4f} projects to {denken172['floor_official_tps']:.1f} TPS ({denken172['verdict']}) -- it FAILS 500.
The {ss['lcb_p90']:.0f}-class GO REQUIRES the deep-spine spread to recover to >= {denken172['lambda_star_clear500_overlap']*100:.0f}% of the
rho-optimal rising ladder (i.e. openevolve cause #2, depth>0 self-KV starvation, must be a FIXABLE build defect, not intrinsic).
The denken realistic-floor refinement (IN-FLIGHT) converts this modeled floor to measured -- it is the highest-leverage open de-risk.

**sigma_hw composition is verdict-INVARIANT.** Whether sigma_hw is RETIRED by best-of-2 (headline, P>=0.9 on a
separate axis) or naively FOLDED as a 4th quadrature term:
- both-bugs: LCB {ss['lcb_p90']:.1f}/P {ss['p_clear_500']*100:.1f}% (best-of-2) vs LCB
  {sigma_hw_fold['both_bugs']['lcb_p90_4term']:.1f}/P {sigma_hw_fold['both_bugs']['p_clear_500_4term']*100:.1f}%
  (naive-fold) -> GO either way.
- descent-only: LCB {sd['lcb_p90']:.1f}/P {sd['p_clear_500']*100:.1f}% (best-of-2) vs LCB
  {sigma_hw_fold['descent_only']['lcb_p90_4term']:.1f}/P {sigma_hw_fold['descent_only']['p_clear_500_4term']*100:.1f}%
  (naive-fold) -> MISS either way.

**Descent-only restoration path:** {rec['descent_only_restoration']['note']}
**Build recommendation:** {rec['build_recommendation']['verdict']}.

**Launch gates (ALL required before the filed issue is approved):**
{gates}

**Dependency ledger ({ledger['n_landed']} LANDED / {ledger['n_in_flight']} IN-FLIGHT / {ledger['n_pending']} PENDING / {ledger['n_pending_build']} PENDING-BUILD):**
{landed}
{inflight}
{pend}

**Truly-unmeasurable residual:** an organizer tree re-run on the REAL private set (no proxy reproduces it).
**Gating build:** land #71 (the both-bugs descending accept-prep kernel).
"""


# ============================================================================
# Driver.
# ============================================================================
def run(args) -> dict:
    t0 = time.time()

    # ---- imported operating-point constants + leg sources ----
    pinned = load_stark156_pinned()
    lawine = load_lawine161_anchors()
    denken166 = load_denken166_ppl()
    framings = load_step168_framings()
    sigma_hw = load_kanna159_sigma_hw()
    denken172 = load_denken172_et()
    stark164 = load_stark164_private()
    v174b = load_v174_banked()

    # ---- assemble the #155/#162 machinery ONCE at base_step (combined_rel step-invariant) ----
    stark151 = load_stark151_retention(step=args.base_step)
    knots = stark151["knots"]
    kcal_band = load_kcal_band()
    sampling = SamplingModel(n_steps=args.n_steps, n_boot=args.n_boot, seed=args.seed,
                             step=args.base_step, step_rel_hw=args.step_rel_half_width)
    b_both = build_joint_b_dict(args.rho_json, args.oracle_json)
    b_desc = b_dict_for_depth1(b_both, DEPTH1_DESCENT_ONLY)
    ok_validity = {"ppl": 2.39, "boots": True, "completed": 128}  # isolate the TPS verdict

    e_t_desc, e_t_both = lawine["e_t_descent_only"], lawine["e_t_both_bugs"]
    d_desc, d_both = pinned["drop_descent_only"], pinned["drop_both_bugs"]

    # ---- STEP 1: instantiate descent-only AND both-bugs at the THREE #179 framings ----
    steps = [("roofline", framings["roofline"]),
             ("shipped", framings["shipped"]),
             ("scatter_lp", framings["scatter_lp"])]
    table = {}
    for name, step in steps:
        desc = cell(e_t_desc, "descent_only", d_desc, step, b_desc, knots, sampling, kcal_band, ok_validity)
        both = cell(e_t_both, "both_bugs", d_both, step, b_both, knots, sampling, kcal_band, ok_validity)
        table[name] = {"step": step, "descent_only": desc, "both_bugs": both}

    both_shipped = table["shipped"]["both_bugs"]
    desc_shipped = table["shipped"]["descent_only"]
    both_bugs_launch_lcb_tps = _finite(both_shipped["lcb_p90"])

    # restoration uses the realizable 1.2086 cell too (for completeness); +3.96 imported from #174.
    realizable = load_step163_realizable()
    table_realizable = cell(e_t_desc, "descent_only", d_desc, realizable["step_realizable"],
                            b_desc, knots, sampling, kcal_band, ok_validity)

    # ---- sigma_hw naive-fold sensitivity (transparency; headline uses best-of-2) ----
    sigma_hw_fold = {
        "descent_only": fold_sigma_hw(desc_shipped, sigma_hw["sigma_hw_pct"]),
        "both_bugs": fold_sigma_hw(both_shipped, sigma_hw["sigma_hw_pct"]),
    }

    # ---- STEP 2: dependency ledger ----
    ledger = build_dependency_ledger(framings, sigma_hw, denken172, stark164, denken166)

    # ---- STEP 3: launch recommendation (both-bugs primary GO) ----
    recommendation = build_recommendation(both_shipped, desc_shipped, v174b, framings, table)

    # ---- STEP 4: self-test (PRIMARY) ----
    sources = {"framings": framings, "sigma_hw": sigma_hw, "denken172": denken172, "stark164": stark164}
    st = self_test(table, ledger, recommendation, v174b, sources, sigma_hw_fold)
    launch_packet_refresh_self_test_passes = st["launch_packet_refresh_self_test_passes"]

    # ---- STEP 5: packet markdown + hand-off ----
    packet_md = render_packet_md(table, ledger, recommendation, sigma_hw, sigma_hw_fold, framings, denken172,
                                 stark164["tau_low"])
    handoff = (
        f"REFRESHED go/no-go packet: both-bugs is the robust GO (LCB(P>=0.9)={both_bugs_launch_lcb_tps:.2f}, "
        f"P={both_shipped['p_clear_500']*100:.2f}%) at the shipped step {framings['shipped']:.4f}; descent-only-first "
        f"is a knife-edge MISS (LCB {desc_shipped['lcb_p90']:.2f}). Gated on land #71's kernel, kanna's darwin "
        f"boot-fix, PRECACHE_BENCH=1, and a human-approved Approval request. Shipping #154's argmax-only decode "
        f"(+{recommendation['descent_only_restoration']['restoration_lcb_tps']:.2f} TPS LCB) restores the simpler "
        f"descent-only build to GO. {HUMAN_DEFERRAL} Truly-unmeasurable residual: an organizer tree re-run on the "
        f"real private set. Gating build: land #71.")

    state = "ARMED" if launch_packet_refresh_self_test_passes else "SELF-TEST-FAIL"
    out = {
        "primary_metric_name": "launch_packet_refresh_self_test_passes",
        "launch_packet_refresh_self_test_passes": launch_packet_refresh_self_test_passes,
        "test_metric_name": "both_bugs_launch_lcb_tps",
        "both_bugs_launch_lcb_tps": both_bugs_launch_lcb_tps,
        "gate_state": state,
        "headline": recommendation["headline"],
        "operating_point": {
            "private_drop_descent_only_pct": pinned["drop_descent_only_pct"],
            "private_drop_both_bugs_pct": pinned["drop_both_bugs_pct"],
            "step_roofline": framings["roofline"],
            "step_shipped_launch_realized": framings["shipped"],
            "step_scatter_lp_conditional": framings["scatter_lp"],
            "k_cal": K_CAL, "tau_low": stark164["tau_low"], "tau_high": 1.0,
            "clear_500_bar_et_shipped": framings["clear_500_bar_overlap"],
            "source": "stark #156 pinned drop + lawine #168 step anchors (roofline/overlap/scatter-LP)",
        },
        "step1_three_framing_geometry": {
            name: {"step": table[name]["step"],
                   "descent_only": _cell_summary(table[name]["descent_only"]),
                   "both_bugs": _cell_summary(table[name]["both_bugs"])}
            for name in ("roofline", "shipped", "scatter_lp")},
        "step1b_sigma_hw_two_axis": {
            "hardware_axis": sigma_hw,
            "naive_fold_sensitivity": sigma_hw_fold,
            "interpretation": (
                "Headline RETIRES sigma_hw via best-of-2 official draws (P>=0.9 on a separate hardware axis); "
                "the naive-fold columns show the verdict is INVARIANT to composition -- descent-only misses the "
                "shipped step either way, both-bugs is GO either way."),
        },
        "step2_dependency_ledger": ledger,
        "step3_recommendation": recommendation,
        "step4_self_test": st,
        "step5_packet_markdown": packet_md,
        "handoff": handoff,
        "imported_legs": {
            "fern_174_verdict": {"wandb": v174b["wandb_run_id"],
                                 "descent_shipped_lcb": v174b["descent_shipped"]["lcb_p90"],
                                 "both_shipped_lcb": v174b["both_shipped"]["lcb_p90"],
                                 "restoration_154_lcb_tps": v174b["restoration_154_lcb_tps"]},
            "lawine_168_step_anchors": [framings["roofline"], framings["shipped"], framings["scatter_lp"]],
            "kanna_159_sigma_hw_tps": sigma_hw["sigma_hw_tps"],
            "kanna_159_best_of_2": sigma_hw["best_of_2"],
            "denken_172_et": [denken172["descent_central"], denken172["both_bugs"], denken172["descent_floor"]],
            "stark_164_private": [stark164["descent_drop_mid_pct"], stark164["descent_taulow_min_tps"]],
            "denken_166_ppl_banked": denken166 is not None,
        },
        "uncertainty_model": {
            "projection_quadrature": "combined = sqrt(sampling^2 + calibration^2 + step_anchor^2)",
            "hardware_axis": "kanna #159 sigma_hw retired by best-of-2 official draws (separate axis)",
            "finite_sample_pending": "wirbel #175 numerator 2nd-moment sampling term -- PENDING",
            "launch_bar": "P(clear 500) >= 0.9 on the projection axis AND best-of-2 hardware clearance",
            "z_p90_one_sided": Z_P90,
            "combined_rel_descent": _finite(desc_shipped["combined_rel_1sigma"]),
            "combined_rel_both": _finite(both_shipped["combined_rel_1sigma"]),
        },
        "provenance": (
            "REFRESHES my #167/#174 launch packet with both-bugs as the PRIMARY GO. Imports VERBATIM: my #174 "
            "verdict engine (-> #167 packet -> #155 consolidator + #162 frontier), lawine #168 step anchors "
            "(roofline/overlap/scatter-LP), kanna #159 sigma_hw + best-of-2, denken #172 E[T]+floor, stark #164 "
            "native private drop, denken #166 PPL stamp. One source of truth per constant -- imports, does not "
            "re-derive."),
        "method": (
            "LOCAL CPU-only analytic consolidation; no GPU/vLLM/HF Job/submission/kernel build/served-file change. "
            "BASELINE stays 481.53; adds 0 TPS (banks the analysis: PRIMARY = self-test). Greedy identity untouched. "
            "Does NOT file the issue or authorize a launch. NOT open2."),
        "metrics_nan_clean": int(st["assert_e_nan_clean"]["ok"]),
        "wandb_run_id": None,
        "wandb_url": None,
        "elapsed_s": None,
    }
    out["_realizable_restoration_lcb_recomputed"] = _finite(table_realizable["lcb_p90"])

    _print_console(out)

    if args.wandb and not args.no_wandb:
        try:
            rid, rurl = _log_wandb(args, out, table, ledger, sigma_hw_fold)
            out["wandb_run_id"], out["wandb_url"] = rid, rurl
        except Exception as e:  # noqa: BLE001
            print(f"[launch-packet-refresh] W&B logging failed (non-fatal): {e!r}", flush=True)

    out["elapsed_s"] = round(time.time() - t0, 4)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def _cell_summary(c: dict) -> dict:
    keys = ("topo", "E_T_input", "r_tree", "proj_private_tps", "p_clear_500", "lcb_p90",
            "combined_rel_1sigma", "geom_tps_public", "consolidator_conf99_verdict",
            "launch_go_p90", "official_tps")
    return {k: c[k] for k in keys if k in c}


def _print_console(out):
    print("=" * 104)
    print("LAUNCH-PACKET REFRESH: both-bugs as PRIMARY GO (PR #179)")
    print("=" * 104)
    op = out["operating_point"]
    print(f"\nPinned drop {op['private_drop_descent_only_pct']:.2f}% desc / {op['private_drop_both_bugs_pct']:.2f}% both; "
          f"steps: roofline {op['step_roofline']:.4f} / shipped {op['step_shipped_launch_realized']:.4f} (SHIPPED) / "
          f"scatter-LP {op['step_scatter_lp_conditional']:.4f}\n")
    print("[STEP 1] Three-framing projection geometry (lambda=mu=1):")
    for name in ("roofline", "shipped", "scatter_lp"):
        t = out["step1_three_framing_geometry"][name]
        print(f"  --- {name} (step {t['step']:.4f}) ---")
        for label, k in (("descent-only", "descent_only"), ("both-bugs", "both_bugs")):
            d = t[k]
            print(f"    {label:12s} official {d['official_tps']:6.1f}  proj {d['proj_private_tps']:6.1f}  "
                  f"P(>=500)={d['p_clear_500']*100:6.2f}%  LCB90={d['lcb_p90']:7.2f}  launch={d['launch_go_p90']}")
    sh = out["step1b_sigma_hw_two_axis"]
    print(f"\n[STEP 1b] sigma_hw two-axis: hardware {sh['hardware_axis']['sigma_hw_tps']:.2f} TPS, "
          f"best-of-2 P={sh['hardware_axis']['best_of_2']:.4f} (>=0.90); naive-fold descent "
          f"LCB {sh['naive_fold_sensitivity']['descent_only']['lcb_p90_4term']:.1f}/"
          f"both LCB {sh['naive_fold_sensitivity']['both_bugs']['lcb_p90_4term']:.1f} -> verdict invariant")
    led = out["step2_dependency_ledger"]
    print(f"\n[STEP 2] Dependency ledger: {led['n_landed']} LANDED / {led['n_in_flight']} IN-FLIGHT / "
          f"{led['n_pending']} PENDING / {led['n_pending_build']} PENDING-BUILD")
    rec = out["step3_recommendation"]
    print(f"\n[STEP 3] RECOMMENDATION: {rec['headline']} -- LCB {rec['both_bugs_launch_lcb_tps']:.2f}, "
          f"P {rec['both_bugs_p_clear_500']*100:.2f}%; #154 restoration +{rec['descent_only_restoration']['restoration_lcb_tps']:.2f} TPS LCB")
    print(f"  build rec changes? {rec['build_recommendation']['changes_build_rec']} -- {rec['build_recommendation']['verdict']}")
    st = out["step4_self_test"]
    print(f"\n[STEP 4] SELF-TEST (PRIMARY):")
    for k, lbl in (("assert_a_reproduces_174_shipped_verdict", "(a) reproduces #174 shipped verdict + xcheck"),
                   ("assert_b_landed_legs_match_sources", "(b) LANDED legs match sources"),
                   ("assert_c_both_bugs_go_with_four_gates", "(c) both-bugs-GO + 4 gates + deferral"),
                   ("assert_d_154_restoration_reported", "(d) #154 restoration +3.96 reported"),
                   ("assert_e_nan_clean", "(e) NaN-clean")):
        print(f"  {lbl} -> {'OK' if st[k]['ok'] else 'FAIL'}")
    print(f"  => launch_packet_refresh_self_test_passes = {out['launch_packet_refresh_self_test_passes']}")
    print(f"\n[PRIMARY] launch_packet_refresh_self_test_passes = {out['launch_packet_refresh_self_test_passes']}")
    print(f"[TEST]    both_bugs_launch_lcb_tps = {out['both_bugs_launch_lcb_tps']:.4f}")
    print(f"[STATE]   {out['gate_state']}")


def _log_wandb(args, out, table, ledger, sigma_hw_fold):
    import wandb
    op = out["operating_point"]
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                     config={"instrument": "launch-packet-refresh", "method": "cpu-analytic-consolidation",
                             "K_cal": K_CAL, "base_step": args.base_step,
                             "step_roofline": op["step_roofline"], "step_shipped": op["step_shipped_launch_realized"],
                             "step_scatter_lp": op["step_scatter_lp_conditional"],
                             "private_drop_descent_pct": op["private_drop_descent_only_pct"],
                             "private_drop_both_pct": op["private_drop_both_bugs_pct"],
                             "launch_bar_p_go": P_GO, "target_official": TARGET_OFFICIAL,
                             "frontier_official": FRONTIER_OFFICIAL})
    s = wandb.summary
    s["launch_packet_refresh_self_test_passes"] = out["launch_packet_refresh_self_test_passes"]
    s["both_bugs_launch_lcb_tps"] = out["both_bugs_launch_lcb_tps"]
    s["gate_state"] = out["gate_state"]
    s["metrics_nan_clean"] = out["metrics_nan_clean"]
    rec = out["step3_recommendation"]
    s["headline"] = rec["headline"]
    s["both_bugs_p_clear_500"] = rec["both_bugs_p_clear_500"]
    s["recommended_first_shot"] = rec["recommended_first_shot"]
    s["restoration_154_lcb_tps"] = rec["descent_only_restoration"]["restoration_lcb_tps"]
    s["build_rec_changes"] = int(rec["build_recommendation"]["changes_build_rec"])
    sh = out["step1b_sigma_hw_two_axis"]["hardware_axis"]
    s["sigma_hw_tps"] = sh["sigma_hw_tps"]
    s["sigma_hw_best_of_2"] = sh["best_of_2"]
    for name in ("roofline", "shipped", "scatter_lp"):
        t = out["step1_three_framing_geometry"][name]
        for topo in ("descent_only", "both_bugs"):
            d = t[topo]
            pre = f"{name}_{topo}"
            s[f"{pre}_official"] = d["official_tps"]
            s[f"{pre}_proj_private"] = d["proj_private_tps"]
            s[f"{pre}_p_clear_500"] = d["p_clear_500"]
            s[f"{pre}_lcb_p90"] = d["lcb_p90"]
            s[f"{pre}_launch_go"] = d["launch_go_p90"]
    s["ledger_n_landed"] = ledger["n_landed"]
    s["ledger_n_in_flight"] = ledger["n_in_flight"]
    s["ledger_n_pending"] = ledger["n_pending"]
    s["ledger_n_pending_build"] = ledger["n_pending_build"]
    for k in ("assert_a_reproduces_174_shipped_verdict", "assert_b_landed_legs_match_sources",
              "assert_c_both_bugs_go_with_four_gates", "assert_d_154_restoration_reported", "assert_e_nan_clean"):
        s[f"selftest_{k}"] = int(out["step4_self_test"][k]["ok"])

    gt = wandb.Table(columns=["framing", "step", "topology", "official", "proj_private",
                              "p_clear_500", "lcb_p90", "launch_go"])
    for name in ("roofline", "shipped", "scatter_lp"):
        t = out["step1_three_framing_geometry"][name]
        for topo in ("descent_only", "both_bugs"):
            d = t[topo]
            gt.add_data(name, t["step"], topo, d["official_tps"], d["proj_private_tps"],
                        d["p_clear_500"], d["lcb_p90"], d["launch_go_p90"])
    wandb.log({"three_framing_geometry": gt})

    lt = wandb.Table(columns=["axis", "leg", "owner_pr", "status", "value", "note"])
    for l in ledger["legs"]:
        lt.add_data(l["axis"], l["leg"], l["owner_pr"], l["status"], l["value"], l["note"])
    wandb.log({"dependency_ledger": lt})

    rid, rurl = run.id, run.url
    print(f"\nW&B run: {rid}  ({rurl})", flush=True)
    wandb.finish()
    return rid, rurl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-step", type=float, default=STEP_BASE,
                    help="base step for the sampling CI (1.2182); combined_rel is step-invariant.")
    ap.add_argument("--step-rel-half-width", type=float, default=STEP_REL_DEFAULT,
                    help="step-anchor 1-sigma relative (default 0.5%%).")
    ap.add_argument("--n-steps", type=int, default=cons.env.ORACLE_STEPS)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=174,
                    help="seed=174 reproduces #174's sampling CI exactly (sigma is bootstrap-independent).")
    ap.add_argument("--rho-json", default=cons.RHO_OPT_JSON)
    ap.add_argument("--oracle-json", default=cons.ORACLE_LIVE_JSON)
    ap.add_argument("--out", default="research/launch/packet_refresh/launch_packet_refresh_results.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", "--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb-entity", "--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb-name", "--wandb_name", default="fern/launch-packet-refresh-bothbugs")
    ap.add_argument("--wandb-group", "--wandb_group", default="launch-packet-refresh-bothbugs")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
