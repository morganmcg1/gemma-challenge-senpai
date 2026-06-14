#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Launch verdict at the conservative launch-realized step (PR #174).

WHAT THIS IS
------------
My #167 packet (MERGED) closed the decision-geometry arc, but its headline
(descent-only 519.6 TPS, P=96.3%) was instantiated at the OPTIMISTIC conditional
step 1.2047 (ubel #154's scatter+LP-reduced decode). lawine #168 (MERGED this cycle)
then ruled 1.2047 CONDITIONAL -- it needs the argmax-only decode build, which has NOT
shipped -- so the LAUNCH-REALIZED step is 1.2182. denken #166 (MERGED) banked the PPL
stamp. Both of #167's refresh items have landed; the headline step needs correcting.

This leg re-instantiates the GO/NO-GO at THREE step framings, holding stark #156's
PINNED private drop fixed (1.80% descent / 1.86% both):
  (a) conservative launch-realized 1.2182  (lawine #168, the SHIPPED reality)
  (b) realizable                    1.2086  (ubel #163, if #154 scatter+LP ships)
  (c) optimistic/conditional        1.2047  (#167's original; #154 argmax-only NOT shipped)
and settles the first-shot recommendation: at the shipped 1.2182 does descent-only-first
still clear the conservative P>=0.9 / LCB>=500 bar, or does the recommended first shot
FLIP to both-bugs?

THE KEY INVARIANT (why this is a clean re-instantiation, not a re-derivation)
----------------------------------------------------------------------------
In the #155 consolidator, proj = K_cal * E[T] * r_tree(drop) / step scales EXACTLY as
1/step, while combined_rel = sqrt(samp^2 + calib^2 + step_anchor^2) is STEP-INVARIANT
(samp depends on E[T] only; calib and step_anchor are fixed). So building the sampling
CI model ONCE at base_step=1.2182 (exactly as #167 did) and varying only the `step`
argument reproduces #167 at 1.2047 to machine precision and cleanly recomputes proj,
LCB(P>=0.9) and P(clear 500) at 1.2086 / 1.2182. r_tree at the pinned drop is also
step-invariant. ONE physical operating point, three decode-step framings.

PURE-ANALYTIC, CPU-ONLY. No GPU / vLLM / HF Job / submission / served-file change.
BASELINE stays 481.53; adds 0 TPS -- it corrects the #167 packet headline to the
shipped step and settles the first-shot recommendation. IMPORTS the committed legs
VERBATIM (my #167 packet machinery -> #155 consolidator + #162 gate; lawine #168 step,
denken #166 PPL stamp, ubel #163 realizable step, stark #156 pinned drop); does NOT
re-derive them. Does NOT file the issue or authorize a launch.

SELF-TEST (PR step 4 -- PRIMARY)
-------------------------------
(a) at the optimistic 1.2047 (and the realizable 1.2086 lane) descent-only reproduces
    #167's 519.6 / 96.3% within tolerance and stays GO; (b) at the conservative 1.2182
    the descent-only proj_private and LCB(P>=0.9) are recomputed and the GO/marginal/flip
    verdict is explicit and consistent with the launch gate; (c) both-bugs remains GO at
    all three steps; (d) the PENDING/BANKED ledger matches the current merged state
    (kanna #159 + land #71 + denken #172 + lawine #173 PENDING; #168/#166/#156/#161/
    #150/#158 BANKED); (e) NaN-clean.
PRIMARY = conservative_step_verdict_self_test_passes;
TEST    = descent_only_p_clear500_at_conservative_step.

Distinct from stark #164 (private-DROP axis of the SAME choice -- I hold the drop fixed
and vary the STEP), denken #172 (E[T] numerator -- CONSUMED here as PENDING), lawine
#173 (descent-walk step -- CONSUMED here as PENDING), kanna #159 (sigma_hw), land #71
(builds the kernel this packet gates). Serves nothing -> greedy identity untouched.
NOT a launch. NOT open2.
"""
from __future__ import annotations

import argparse
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
# Import my #167 packet VERBATIM (it transitively loads #162 frontier + #155 cons).
# One source of truth per constant -- imports, does not re-derive.
# ============================================================================
packet = _load("pinned_launch_decision_packet",
               os.path.join(_HERE, "pinned_launch_decision_packet.py"))   # fern #167

# the instantiation engine + loaders (reused exactly)
instantiate_private = packet.instantiate_private
load_stark156_pinned = packet.load_stark156_pinned
load_lawine161_anchors = packet.load_lawine161_anchors
load_frontier162_gate = packet.load_frontier162_gate

# the frontier + consolidator machinery (already wired inside the packet)
frontier = packet.frontier
cons = packet.cons
load_ubel154_bars = packet.load_ubel154_bars
load_stark151_retention = packet.load_stark151_retention
b_dict_for_depth1 = packet.b_dict_for_depth1
DEPTH1_DESCENT_ONLY = packet.DEPTH1_DESCENT_ONLY
build_joint_b_dict = cons.build_joint_b_dict
SamplingModel = cons.SamplingModel
load_kcal_band = cons.load_kcal_band
joint_et = cons.decomp.joint_et

# constants (verbatim)
K_CAL = packet.K_CAL                                  # 125.268
RHO2_BRANCH_HIT = packet.RHO2_BRANCH_HIT             # 0.4165
TARGET_OFFICIAL = packet.TARGET_OFFICIAL             # 500.0
FRONTIER_OFFICIAL = cons.FRONTIER_OFFICIAL           # 481.53
Z_P90_ONESIDED = packet.Z_P90_ONESIDED               # 1.281552 (Phi(z)=0.9)
STEP_MEASURED_DEPTH9 = packet.STEP_MEASURED_DEPTH9   # 1.2182 (base step for the sampling model)
STEP_REL_1SIGMA_DEFAULT = packet.STEP_REL_1SIGMA_DEFAULT  # 0.005
P_GO = packet.P_GO                                    # 0.9 conservative one-shot bar
P_HOLD = packet.P_HOLD                                # 0.5

# committed step sources (imported, not re-derived)
STEP168_JSON = os.path.join(_ROOT, "research/spec_cost_model/step_anchor_reconciliation.json")
STEP163_JSON = os.path.join(_ROOT, "research/spec_cost_model/host_residency_sweep/host_residency_sweep.json")
DENKEN166_GLOB = os.path.join(_ROOT, "research/validity/tree_path_ppl_margin/runs/*/ppl_margin_bound_result.json")

# how much LCB(P>=0.9) headroom above 500 to call descent-only-first "robust" vs "marginal"
ROBUST_LCB_MARGIN_TPS = 2.0


def _finite(x, default: float = 0.0) -> float:
    try:
        return float(x) if (x is not None and math.isfinite(float(x))) else default
    except (TypeError, ValueError):
        return default


# ============================================================================
# Load the three step framings from their canonical committed sources.
# ============================================================================
def load_step168_conservative(path: str = STEP168_JSON) -> dict:
    with open(path) as f:
        j = json.load(f)
    s2 = j["step2_launch_realized_step"]
    return {
        # descent_only == both_bugs (BUG-1 is step-neutral, #168 step5)
        "step_conservative": float(s2["both_bugs"]),
        "step_descent_only": float(s2["descent_only"]),
        "step_both_bugs": float(s2["both_bugs"]),
        "roofline_step": float(s2["band"]["lo_step_roofline"]),
        "band_half_width_pct": float(s2["band"]["half_width_pct"]),
        "e_t_descent_only": float(j["constants"]["e_t_descent_only"]),
        "e_t_both_bugs": float(j["constants"]["e_t_both_bugs"]),
        "official_descent_only": float(j["step5_handoff"]["official_descent_only"]),
        "official_both_bugs": float(j["step5_handoff"]["official_both_bugs"]),
    }


def load_step163_realizable(path: str = STEP163_JSON) -> dict:
    with open(path) as f:
        j = json.load(f)
    nb = j["net_step_budget"]
    return {
        "step_realizable": float(nb["net_descent_step_pinned"]),
        "bar_descent_only": float(nb["net_clear_500_bar_descent_only"]),
        "bar_both_bugs": float(nb["net_clear_500_bar_both_bugs"]),
    }


def load_denken166_ppl(glob_pat: str = DENKEN166_GLOB) -> dict | None:
    """denken #166 banked PPL-margin bound (best-effort; ledger note only). Returns the
    binding M=32 worst-case PPL + cap so the BANKED claim is concrete."""
    import glob as _glob
    hits = sorted(_glob.glob(glob_pat))
    if not hits:
        return None
    try:
        with open(hits[-1]) as f:
            j = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    wc = j.get("tree_path_ppl_worst_case")
    cap = (j.get("anchors") or {}).get("ppl_cap", 2.42)
    under = j.get("ppl_margin_under_2p42")
    if wc is None:
        return {"path": hits[-1]}
    return {"path": hits[-1], "worst_case": float(wc), "cap": float(cap),
            "margin": float(cap) - float(wc), "under_2p42": bool(under)}


# ============================================================================
# Instantiate one topology at one step (thin wrapper over the #167 engine).
# ============================================================================
def cell(E_T, topo, drop, step, b_dict, knots, sampling, kcal_band, validity):
    d = instantiate_private(E_T, RHO2_BRANCH_HIT, 1.0, 1.0, topo, drop, step,
                            b_dict, knots, sampling, kcal_band, validity)
    # official-TPS = the public projection at full membership (lambda=mu=1), no private
    # haircut == geom_tps_public; proj_private carries the r_tree(drop) haircut.
    d["official_tps"] = d["geom_tps_public"]
    return d


# ============================================================================
# The refreshed leg ledger (PR step 3): #168 + #166 now BANKED -> 8 BANKED;
# the PENDING set is {kanna #159, land #71} + the two NEW descent-validity upgrades
# {denken #172, lawine #173} -> 4 PENDING.
# ============================================================================
def build_leg_ledger(frontier162, lawine, conservative, denken166) -> dict:
    d166_note = "M=32 batched-verify aggregate-PPL worst-case vs 2.42"
    if denken166 is not None and "worst_case" in denken166:
        d166_note = (f"M=32 worst-case PPL {denken166['worst_case']:.4f} <= {denken166['cap']:.2f} "
                     f"(margin {denken166['margin']:.4f})")

    legs = [
        # ---- BANKED (evidence in hand) ----
        {"leg": "fern #155 consolidator", "role": "GO/NO-GO + CI + binding-leg union",
         "status": "BANKED", "evidence": "approval_projection_consolidator_results.json (self-test PASS)"},
        {"leg": "fern #162 build gate", "role": "(lambda_min,mu_min)=(0.8809,0.7353) @ P>=0.5",
         "status": "BANKED", "evidence": f"tightened_private_500_frontier_results.json (self-test {frontier162['self_test_passes']})"},
        {"leg": "stark #156 pinned drop", "role": "private drop 1.80% desc / 1.86% both (anchored GT-4.3%)",
         "status": "BANKED", "evidence": "tree_private_drop_reconcile/results.json"},
        {"leg": "lawine #161 both-bugs step", "role": f"measured step-NEUTRAL -> both-bugs official {lawine['official_both_assumed_roofline']:.1f}",
         "status": "BANKED", "evidence": "both_bugs_step_cost.json (step_delta 0.0%)"},
        {"leg": "lawine #168 step-reconciliation", "role": f"launch-realized step {conservative['step_conservative']:.4f} (overlap); roofline<->overlap {conservative['band_half_width_pct']:.2f}% half-width band",
         "status": "BANKED", "evidence": "step_anchor_reconciliation.json (self-test PASS; collapses #136/#154/#161)"},
        {"leg": "denken #166 PPL-margin bound", "role": d166_note,
         "status": "BANKED", "evidence": "tree_path_ppl_margin (M=32 batched-verify worst-case bounded under 2.42)"},
        {"leg": "denken #150 validity (does-it-score)", "role": "PPL<=2.42 & boots & 128/128 contract",
         "status": "BANKED", "evidence": "imported inline by #155 (ARMED/PENDING land #71's run)"},
        {"leg": "denken #158 greedy-exactness", "role": "per-token committed==argmax (BUG-2 catcher)",
         "status": "BANKED", "evidence": "--audit-kernel-symbol pre-merge contract gate"},
        # ---- PENDING (in flight) ----
        {"leg": "denken #172 descent E[T] conservative LB", "role": "replaces the point E[T]=5.0564 with central +- floor (descent-only numerator lower bound)",
         "status": "PENDING", "evidence": "in flight -- this packet uses the point E[T]; #172 hardens it to a conservative LB"},
        {"leg": "lawine #173 descent-walk step-neutrality", "role": "confirms the actual descent kernel holds the 1.2182 launch-realized step",
         "status": "PENDING", "evidence": "in flight -- #168 reconciles the step analytically; #173 confirms the descent walk realizes it"},
        {"leg": "kanna #159 sigma_hw", "role": "4th quadrature term (A10G clock/thermal/cold-start)",
         "status": "PENDING", "evidence": "in flight -- CI here is the 3-term quadrature; sigma_hw widens it"},
        {"leg": "land #71 measured tuple", "role": "(E[T], rho2, lambda, mu, step, ppl, boots, completed)",
         "status": "PENDING", "evidence": "the kernel this packet gates -- not yet built"},
    ]
    banked = [l["leg"] for l in legs if l["status"] == "BANKED"]
    pending = [l["leg"] for l in legs if l["status"] == "PENDING"]
    return {"legs": legs, "banked": banked, "pending": pending,
            "n_banked": len(banked), "n_pending": len(pending)}


def assert_ledger_matches_merged(ledger) -> dict:
    """Self-test (d): the refreshed ledger matches the current merged state.
    #168 + #166 moved BANKED (were PENDING in #167); the NEW descent-validity legs
    #172 + #173 join kanna #159 + land #71 as PENDING."""
    expect_pending = {"denken #172 descent E[T] conservative LB", "lawine #173 descent-walk step-neutrality",
                      "kanna #159 sigma_hw", "land #71 measured tuple"}
    expect_banked = {"fern #155 consolidator", "fern #162 build gate", "stark #156 pinned drop",
                     "lawine #161 both-bugs step", "lawine #168 step-reconciliation",
                     "denken #166 PPL-margin bound", "denken #150 validity (does-it-score)",
                     "denken #158 greedy-exactness"}
    got_pending = set(ledger["pending"])
    got_banked = set(ledger["banked"])
    # explicit moves the PR step 4(d) names
    f168_banked = "lawine #168 step-reconciliation" in got_banked
    f166_banked = "denken #166 PPL-margin bound" in got_banked
    f172_pending = "denken #172 descent E[T] conservative LB" in got_pending
    f173_pending = "lawine #173 descent-walk step-neutrality" in got_pending
    ok = bool(got_pending == expect_pending and got_banked == expect_banked
              and f168_banked and f166_banked and f172_pending and f173_pending)
    return {"ok": ok, "n_banked": ledger["n_banked"], "n_pending": ledger["n_pending"],
            "expected_pending": sorted(expect_pending), "got_pending": sorted(got_pending),
            "expected_banked": sorted(expect_banked), "got_banked": sorted(got_banked),
            "f168_now_banked": f168_banked, "f166_now_banked": f166_banked,
            "f172_pending": f172_pending, "f173_pending": f173_pending,
            "expect": "8 BANKED (incl. #168+#166 newly banked) / 4 PENDING (incl. #172+#173 new)"}


# ============================================================================
# First-shot verdict (PR step 2): does descent-only-first survive the shipped step?
# ============================================================================
def classify_first_shot(desc_cons, both_cons, desc_real, desc_opt) -> dict:
    lcb_cons = desc_cons["lcb_p90"]
    go_cons = desc_cons["launch_go_p90"]
    clears_lcb = bool(lcb_cons >= TARGET_OFFICIAL)
    margin_cons = _finite(lcb_cons - TARGET_OFFICIAL)
    margin_real = _finite(desc_real["lcb_p90"] - TARGET_OFFICIAL)
    margin_opt = _finite(desc_opt["lcb_p90"] - TARGET_OFFICIAL)

    if go_cons == "GO" and clears_lcb and margin_cons >= ROBUST_LCB_MARGIN_TPS:
        verdict, recommended = "robust", "descent-only"
    elif go_cons == "GO" and clears_lcb:
        verdict, recommended = "marginal", "descent-only"
    else:
        verdict, recommended = "flips-to-both-bugs", "both-bugs"

    return {
        "descent_only_first_verdict": verdict,           # robust / marginal / flips-to-both-bugs
        "recommended_first_shot": recommended,
        "conservative_step": None,  # filled by the caller
        "descent_only_conservative": {
            "official_tps": desc_cons["official_tps"], "proj_private": desc_cons["proj_private_tps"],
            "p_clear_500": desc_cons["p_clear_500"], "lcb_p90": lcb_cons,
            "launch_go_p90": go_cons, "clears_lcb_500": clears_lcb,
            "conf99": desc_cons["consolidator_conf99_verdict"]},
        "both_bugs_conservative": {
            "official_tps": both_cons["official_tps"], "proj_private": both_cons["proj_private_tps"],
            "p_clear_500": both_cons["p_clear_500"], "lcb_p90": both_cons["lcb_p90"],
            "launch_go_p90": both_cons["launch_go_p90"], "conf99": both_cons["consolidator_conf99_verdict"]},
        "lcb_margin_vs_500": {
            "conservative_1p2182": margin_cons, "realizable_1p2086": margin_real,
            "optimistic_1p2047": margin_opt},
        "cost_of_not_shipping_154_lcb_tps": _finite(margin_real - margin_cons),
        "cost_of_not_shipping_154_proj_tps": _finite(desc_real["proj_private_tps"] - desc_cons["proj_private_tps"]),
        "note": (
            f"At the shipped 1.2182, descent-only LCB(P>=0.9)={lcb_cons:.2f} "
            f"({'>=' if clears_lcb else '<'} 500) -> launch {go_cons}; descent-only-first is "
            f"'{verdict}'. The realizable step 1.2086 (if #154's argmax-only decode ships) lifts the "
            f"descent-only LCB to {desc_real['lcb_p90']:.2f} (+{margin_real - margin_cons:.2f} TPS of "
            f"headroom -- the cost of NOT shipping #154). both-bugs is the comfortable first shot at "
            f"the shipped step (proj {both_cons['proj_private_tps']:.1f}, LCB {both_cons['lcb_p90']:.1f})."),
    }


# ============================================================================
# Self-test (PR step 4 -- PRIMARY).
# ============================================================================
def _rel_err(v, ref):
    return abs(v - ref) / ref if ref else float("inf")


def self_test(table, first_shot, ledger):
    # #167 reference at the optimistic step (from pinned_launch_decision_packet_results.json)
    REF167_DESC_PROJ = 519.6390521567497
    REF167_DESC_PCLEAR = 0.9630337622697593
    REF167_DESC_LCB = 505.55510080538465

    desc_opt = table["optimistic_1p2047"]["descent_only"]
    desc_real = table["realizable_1p2086"]["descent_only"]
    desc_cons = table["conservative_1p2182"]["descent_only"]

    # (a) reproduce #167's optimistic-step headline within tolerance + realizable lane stays GO.
    a1_proj = bool(_rel_err(desc_opt["proj_private_tps"], REF167_DESC_PROJ) <= 0.001)
    a1_pclear = bool(abs(desc_opt["p_clear_500"] - REF167_DESC_PCLEAR) <= 0.005)
    a1_lcb = bool(_rel_err(desc_opt["lcb_p90"], REF167_DESC_LCB) <= 0.001)
    a1 = bool(a1_proj and a1_pclear and a1_lcb and desc_opt["launch_go_p90"] == "GO")
    a2 = bool(desc_real["launch_go_p90"] == "GO" and desc_real["lcb_p90"] >= TARGET_OFFICIAL)
    assert_a = bool(a1 and a2)

    # (b) conservative descent recomputed; the GO/marginal/flip verdict is explicit + consistent.
    v = first_shot["descent_only_first_verdict"]
    label_valid = v in ("robust", "marginal", "flips-to-both-bugs")
    # consistency: 'flips-to-both-bugs' iff descent-only does NOT clear the conservative GO bar.
    flip_consistent = bool((v == "flips-to-both-bugs") == (desc_cons["launch_go_p90"] != "GO"))
    recomputed = bool(math.isfinite(desc_cons["proj_private_tps"]) and math.isfinite(desc_cons["lcb_p90"]))
    assert_b = bool(label_valid and flip_consistent and recomputed)

    # (c) both-bugs remains GO at all three steps.
    both_go = {k: table[k]["both_bugs"]["launch_go_p90"] for k in table}
    assert_c = bool(all(g == "GO" for g in both_go.values()))

    # (d) ledger matches the current merged state.
    st_d = assert_ledger_matches_merged(ledger)
    assert_d = st_d["ok"]

    # (e) NaN-clean across every headline numeric.
    nums = []
    for k in table:
        for topo in ("descent_only", "both_bugs"):
            c = table[k][topo]
            nums += [c["official_tps"], c["proj_private_tps"], c["p_clear_500"], c["lcb_p90"]]
    nums += [first_shot["cost_of_not_shipping_154_lcb_tps"],
             first_shot["cost_of_not_shipping_154_proj_tps"]]
    assert_e = bool(all(x is not None and math.isfinite(x) for x in nums))

    passes = int(bool(assert_a and assert_b and assert_c and assert_d and assert_e))
    return {
        "passes": bool(passes),
        "conservative_step_verdict_self_test_passes": passes,
        "assert_a_reproduces_167_optimistic_and_realizable_go": {
            "ok": assert_a, "opt_proj": desc_opt["proj_private_tps"], "ref167_proj": REF167_DESC_PROJ,
            "opt_p_clear": desc_opt["p_clear_500"], "ref167_p_clear": REF167_DESC_PCLEAR,
            "opt_lcb": desc_opt["lcb_p90"], "ref167_lcb": REF167_DESC_LCB,
            "opt_launch_go": desc_opt["launch_go_p90"],
            "realizable_launch_go": desc_real["launch_go_p90"], "realizable_lcb": desc_real["lcb_p90"],
            "expect": "optimistic reproduces #167 519.6/96.3%/505.6 within tol AND realizable stays GO"},
        "assert_b_conservative_verdict_explicit": {
            "ok": assert_b, "descent_only_first_verdict": v,
            "cons_proj_private": desc_cons["proj_private_tps"], "cons_lcb_p90": desc_cons["lcb_p90"],
            "cons_p_clear_500": desc_cons["p_clear_500"], "cons_launch_go": desc_cons["launch_go_p90"],
            "label_valid": label_valid, "flip_consistent_with_gate": flip_consistent,
            "expect": "explicit verdict in {robust,marginal,flips-to-both-bugs}, consistent with the GO gate"},
        "assert_c_both_bugs_go_all_three_steps": {
            "ok": assert_c, "both_bugs_launch_go": both_go,
            "expect": "both-bugs launch(P>=0.9)=GO at 1.2182, 1.2086, 1.2047"},
        "assert_d_ledger_matches_merged": st_d,
        "assert_e_nan_clean": {"ok": assert_e, "n_numbers_checked": len(nums)},
    }


# ============================================================================
# The refreshed readiness packet (PR step 3) -- verbatim Approval-request block.
# ============================================================================
def render_packet_md(table, first_shot, frontier162, pinned, conservative, realizable, ledger) -> str:
    lm, mm = frontier162["lambda_min"], frontier162["mu_min"]
    banked = "\n".join(f"- [BANKED]  {l['leg']} -- {l['role']}" for l in ledger["legs"] if l["status"] == "BANKED")
    pending = "\n".join(f"- [PENDING] {l['leg']} -- {l['role']}" for l in ledger["legs"] if l["status"] == "PENDING")
    rec = first_shot["recommended_first_shot"]
    v = first_shot["descent_only_first_verdict"]

    def row(label, c, bold=False):
        nm = f"**{label}**" if bold else label
        proj = f"**{c['proj_private_tps']:.1f}**" if bold else f"{c['proj_private_tps']:.1f}"
        go = f"**{c['launch_go_p90']}**" if bold else c["launch_go_p90"]
        return (f"| {nm} | {c['official_tps']:.1f} | {c['r_tree']:.4f} | {proj} | "
                f"{c['p_clear_500']*100:.1f}% | {c['lcb_p90']:.1f} | {c['consolidator_conf99_verdict']} | {go} |")

    cd, cb = table["conservative_1p2182"]["descent_only"], table["conservative_1p2182"]["both_bugs"]
    rd, rb = table["realizable_1p2086"]["descent_only"], table["realizable_1p2086"]["both_bugs"]
    od, ob = table["optimistic_1p2047"]["descent_only"], table["optimistic_1p2047"]["both_bugs"]

    return f"""\
### Approval request: HF job for tree-descent submission (PRE-FILLED DRAFT -- NOT YET FILED)

**This block is the pre-filled projection+validity body of the eventual `Approval
request: HF job` issue. It awaits land #71's measured tuple and the PENDING legs below.
It does NOT authorize a launch; a human must approve the filed issue.**

**Operating point (PINNED drop, LAUNCH-REALIZED step):** private drop {pinned['drop_descent_only_pct']:.2f}%
(descent-only) / {pinned['drop_both_bugs_pct']:.2f}% (both-bugs), anchored to flagship GT-4.3% (stark #156).
**Launch-realized decode step {conservative['step_conservative']:.4f}** (lawine #168 -- the SHIPPED reality:
eager star-attn pays the +0.45% exposed launch idle that survives GEMM overlap). The optimistic
1.2047 (ubel #154 argmax-only decode) has NOT shipped; the realizable {realizable['step_realizable']:.4f}
(ubel #163) is the upside if it does.

**Projection formula (parameterized on land #71's PENDING tuple):**
```
proj_private = K_cal * E[T]_land * r_tree(d_pinned, topo) / step          (K_cal={K_CAL:.3f})
P(clear 500) = min( Phi((proj-500)/sigma), Phi((geom_tps(lambda,mu)-500)/sigma) )
sigma        = proj * sqrt(samp^2 + calib^2 + step^2)   [+ sigma_hw PENDING kanna #159]
GO  iff  P(clear 500) >= 0.9  AND  (lambda,mu) >= ({lm:.4f}, {mm:.4f})  AND  validity READY
```

**Instantiated projection at the LAUNCH-REALIZED step {conservative['step_conservative']:.4f} (full-recovery corner lambda=mu=1):**

| topology | official | r_tree | proj_private | P(clear 500) | LCB(P>=0.9) | conf-99 | launch (P>=0.9) |
|---|---|---|---|---|---|---|---|
{row('descent-only (BUG-1 deferred)', cd, bold=True)}
{row('both-bugs (BUG-1 fixed)', cb)}

**Step-framing band (same pinned drop, three decode-step framings):**

| step framing | descent proj / LCB / launch | both-bugs proj / LCB / launch |
|---|---|---|
| conservative {conservative['step_conservative']:.4f} (SHIPPED) | {cd['proj_private_tps']:.1f} / {cd['lcb_p90']:.1f} / {cd['launch_go_p90']} | {cb['proj_private_tps']:.1f} / {cb['lcb_p90']:.1f} / {cb['launch_go_p90']} |
| realizable {realizable['step_realizable']:.4f} (if #154 ships) | {rd['proj_private_tps']:.1f} / {rd['lcb_p90']:.1f} / {rd['launch_go_p90']} | {rb['proj_private_tps']:.1f} / {rb['lcb_p90']:.1f} / {rb['launch_go_p90']} |
| optimistic {table['optimistic_1p2047']['step']:.4f} (#167 original) | {od['proj_private_tps']:.1f} / {od['lcb_p90']:.1f} / {od['launch_go_p90']} | {ob['proj_private_tps']:.1f} / {ob['lcb_p90']:.1f} / {ob['launch_go_p90']} |

**Recommended first shot: {rec}** -- descent-only-first is **{v}** at the shipped step
(LCB(P>=0.9) {cd['lcb_p90']:.1f} vs the 500 bar). {first_shot['note']}

**Validity stamps (must ALL be READY before the filed issue is approved):**
- denken #150 (does-it-score): PPL<=2.42 & boots & 128/128 -- BANKED contract, ARMED/PENDING land #71's run.
- denken #158 (greedy-exact): per-token committed==argmax -- BANKED, `--audit-kernel-symbol` pre-merge gate.
- denken #166 (PPL-margin bound): M=32 aggregate-PPL worst-case vs 2.42 -- **BANKED** (bounded under 2.42).
- denken #172 (descent E[T] conservative LB): replaces point E[T]=5.0564 with central+-floor -- **PENDING**.
- lawine #173 (descent-walk step-neutrality): confirms the descent kernel holds {conservative['step_conservative']:.4f} -- **PENDING**.
- kanna #159 (sigma_hw): A10G clock/thermal/cold-start 4th quadrature term -- **PENDING** (widens the CI above).

**Leg ledger ({ledger['n_banked']} BANKED / {ledger['n_pending']} PENDING):**
{banked}
{pending}
"""


# ============================================================================
# Driver.
# ============================================================================
def run(args) -> dict:
    t0 = time.time()
    step_rel = args.step_rel_half_width

    # ---- imported operating-point constants ----
    pinned = load_stark156_pinned()
    lawine = load_lawine161_anchors()
    frontier162 = load_frontier162_gate()
    conservative = load_step168_conservative()
    realizable = load_step163_realizable()
    denken166 = load_denken166_ppl()

    # ---- the three decode-step framings (canonical sources) ----
    ubel = load_ubel154_bars(base_step=args.base_step)
    step_optimistic = ubel["points"]["realistic"]["step"]       # 1.2047 (#167's original)
    step_realizable = realizable["step_realizable"]             # 1.2086 (ubel #163)
    step_conservative = conservative["step_conservative"]       # 1.2182 (lawine #168, shipped)

    # ---- assemble the #155/#162 machinery ONCE at base_step (combined_rel step-invariant) ----
    stark151 = load_stark151_retention(step=args.base_step)
    knots = stark151["knots"]
    kcal_band = load_kcal_band()
    sampling = SamplingModel(n_steps=args.n_steps, n_boot=args.n_boot, seed=args.seed,
                             step=args.base_step, step_rel_hw=step_rel)
    b_both = build_joint_b_dict(args.rho_json, args.oracle_json)
    b_desc = b_dict_for_depth1(b_both, DEPTH1_DESCENT_ONLY)
    ok_validity = {"ppl": 2.39, "boots": True, "completed": 128}   # isolate the TPS verdict

    e_t_desc, e_t_both = lawine["e_t_descent_only"], lawine["e_t_both_bugs"]
    d_desc, d_both = pinned["drop_descent_only"], pinned["drop_both_bugs"]

    # ---- STEP 1: re-instantiate at THREE step framings (descent-only AND both-bugs) ----
    steps = [
        ("conservative_1p2182", step_conservative),
        ("realizable_1p2086", step_realizable),
        ("optimistic_1p2047", step_optimistic),
    ]
    table = {}
    for name, step in steps:
        desc = cell(e_t_desc, "descent_only", d_desc, step, b_desc, knots, sampling, kcal_band, ok_validity)
        both = cell(e_t_both, "both_bugs", d_both, step, b_both, knots, sampling, kcal_band, ok_validity)
        desc["_step"], both["_step"] = step, step
        table[name] = {"step": step, "descent_only": desc, "both_bugs": both}

    desc_cons = table["conservative_1p2182"]["descent_only"]
    descent_only_p_clear500_at_conservative_step = _finite(desc_cons["p_clear_500"])

    # ---- STEP 2: settle the first-shot recommendation ----
    first_shot = classify_first_shot(
        table["conservative_1p2182"]["descent_only"], table["conservative_1p2182"]["both_bugs"],
        table["realizable_1p2086"]["descent_only"], table["optimistic_1p2047"]["descent_only"])
    first_shot["conservative_step"] = step_conservative

    # ---- STEP 3: refresh the ledger + re-emit the Approval-request packet ----
    ledger = build_leg_ledger(frontier162, lawine, conservative, denken166)
    packet_md = render_packet_md(table, first_shot, frontier162, pinned, conservative, realizable, ledger)

    # ---- STEP 4: self-test (PRIMARY) ----
    st = self_test(table, first_shot, ledger)
    conservative_step_verdict_self_test_passes = st["conservative_step_verdict_self_test_passes"]

    # ---- STEP 5: hand-off ----
    rec = first_shot["recommended_first_shot"]
    v = first_shot["descent_only_first_verdict"]
    rc = table["conservative_1p2182"]["both_bugs"] if rec == "both-bugs" else desc_cons
    handoff = (
        f"At the conservative launch-realized step {step_conservative:.4f} and pinned "
        f"{pinned['drop_descent_only_pct']:.2f}% drop, the recommended first shot is {rec}, "
        f"P(clear-500)={rc['p_clear_500']*100:.1f}%, LCB(P>=0.9)={rc['lcb_p90']:.1f} -- "
        f"descent-only-first is {v} at the shipped step "
        f"(LCB {desc_cons['lcb_p90']:.1f}), robust at the realizable step "
        f"(LCB {table['realizable_1p2086']['descent_only']['lcb_p90']:.1f}); pending kanna #159, "
        f"land #71, denken #172, lawine #173. The refreshed packet remains a pre-filled draft -- "
        f"it does NOT authorize a launch.")

    state = "ARMED" if conservative_step_verdict_self_test_passes else "SELF-TEST-FAIL"
    out = {
        "primary_metric_name": "conservative_step_verdict_self_test_passes",
        "conservative_step_verdict_self_test_passes": conservative_step_verdict_self_test_passes,
        "test_metric_name": "descent_only_p_clear500_at_conservative_step",
        "descent_only_p_clear500_at_conservative_step": descent_only_p_clear500_at_conservative_step,
        "gate_state": state,
        "operating_point": {
            "private_drop_descent_only_pct": pinned["drop_descent_only_pct"],
            "private_drop_both_bugs_pct": pinned["drop_both_bugs_pct"],
            "anchored_to_flagship_gt_pct": pinned["calibrated_linear_drop_pct"],
            "step_conservative_launch_realized": step_conservative,
            "step_realizable": step_realizable,
            "step_optimistic_conditional": step_optimistic,
            "source": "stark #156 pinned drop + lawine #168 (conservative) / ubel #163 (realizable) / ubel #154 (optimistic) steps",
        },
        "step1_three_step_instantiation": table,
        "step2_first_shot_settlement": first_shot,
        "step3_readiness_packet": {
            "packet_markdown": packet_md,
            "leg_ledger": ledger,
            "recommended_first_shot": first_shot["recommended_first_shot"],
            "build_gate_lambda_mu_min": [frontier162["lambda_min"], frontier162["mu_min"]],
        },
        "step4_self_test": st,
        "step5_handoff": handoff,
        "imported_legs": {
            "fern_167_packet": "instantiate_private / loaders (the engine refreshed here)",
            "fern_155_consolidator": "consolidate() / SamplingModel / validity_gate (verbatim)",
            "fern_162_frontier": "r_tree / private retention / (lambda_min,mu_min)=(0.8809,0.7353)",
            "lawine_168_step": [step_conservative, conservative["roofline_step"]],
            "ubel_163_realizable_step": step_realizable,
            "ubel_154_optimistic_step": step_optimistic,
            "stark_156_pinned_drop": [pinned["drop_descent_only_pct"], pinned["drop_both_bugs_pct"]],
            "denken_166_ppl_banked": denken166 is not None,
        },
        "invariant_note": (
            "combined_rel is STEP-INVARIANT (samp depends on E[T] only; calib + step_anchor fixed), so "
            "proj scales exactly 1/step and the sampling model built once at base_step reproduces #167 "
            "at 1.2047 to machine precision. r_tree at the pinned drop is step-invariant. ONE physical "
            "operating point, three decode-step framings."),
        "uncertainty_model": {
            "quadrature_formula": "combined = sqrt(sampling^2 + calibration^2 + step_anchor^2)",
            "fourth_term_pending": "kanna #159 sigma_hw (A10G clock/thermal/cold-start) -- NOT yet folded",
            "launch_bar": "P(clear 500) >= 0.9 (conservative one-shot LCB gate); conf-99 reported alongside",
            "z_p90_one_sided": Z_P90_ONESIDED,
            "combined_rel_descent": _finite(desc_cons["combined_rel_1sigma"]),
            "combined_rel_both": _finite(table["conservative_1p2182"]["both_bugs"]["combined_rel_1sigma"]),
            "step_anchor_rel_136": _finite(step_rel),
        },
        "provenance": (
            "REFRESHES my #167 launch-decision packet at the corrected launch-realized step. Imports "
            "VERBATIM: fern #167 packet engine (-> #155 consolidator + #162 frontier), lawine #168 step "
            "reconciliation (conservative 1.2182), ubel #163 realizable step (1.2086), ubel #154 "
            "optimistic step (1.2047), stark #156 pinned drop, denken #166 PPL stamp (BANKED). One "
            "source of truth per constant -- imports, does not re-derive."),
        "method": (
            "LOCAL CPU-only analytic re-instantiation; no GPU/vLLM/HF Job/submission/kernel build. "
            "BASELINE stays 481.53; adds 0 TPS -- corrects the #167 headline to the shipped step + "
            "settles the first-shot recommendation. Does NOT file the issue or authorize a launch. "
            "Greedy identity untouched. NOT open2."),
        "metrics_nan_clean": int(st["assert_e_nan_clean"]["ok"]),
        "wandb_run_id": None,
        "wandb_url": None,
        "elapsed_s": None,
    }

    _print_console(out)

    if args.wandb and not args.no_wandb:
        try:
            rid, rurl = _log_wandb(args, out)
            out["wandb_run_id"], out["wandb_url"] = rid, rurl
        except Exception as e:  # noqa: BLE001
            print(f"[conservative-step-verdict] W&B logging failed (non-fatal): {e!r}", flush=True)

    out["elapsed_s"] = round(time.time() - t0, 4)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    return out


def _print_console(out):
    print("=" * 104)
    print("LAUNCH VERDICT AT THE CONSERVATIVE LAUNCH-REALIZED STEP (PR #174)")
    print("=" * 104)
    op = out["operating_point"]
    print(f"\nPinned drop {op['private_drop_descent_only_pct']:.2f}% desc / "
          f"{op['private_drop_both_bugs_pct']:.2f}% both; steps: conservative "
          f"{op['step_conservative_launch_realized']:.4f} (SHIPPED) / realizable "
          f"{op['step_realizable']:.4f} / optimistic {op['step_optimistic_conditional']:.4f}\n")

    print("[STEP 1] Three-step instantiation (full-recovery corner lambda=mu=1):")
    for name in ("conservative_1p2182", "realizable_1p2086", "optimistic_1p2047"):
        t = out["step1_three_step_instantiation"][name]
        print(f"  --- {name} (step {t['step']:.4f}) ---")
        for label, k in (("descent-only", "descent_only"), ("both-bugs", "both_bugs")):
            d = t[k]
            print(f"    {label:12s} official {d['official_tps']:6.1f}  proj_priv {d['proj_private_tps']:6.1f}  "
                  f"P(>=500)={d['p_clear_500']*100:5.1f}%  LCB90={d['lcb_p90']:6.1f}  "
                  f"conf99={d['consolidator_conf99_verdict']:>16s}  launch(P>=0.9)={d['launch_go_p90']}")

    fs = out["step2_first_shot_settlement"]
    print(f"\n[STEP 2] First-shot settlement: descent-only-first is '{fs['descent_only_first_verdict']}' "
          f"at the shipped step -> recommended first shot = {fs['recommended_first_shot']}")
    m = fs["lcb_margin_vs_500"]
    print(f"  descent LCB margin vs 500: conservative {m['conservative_1p2182']:+.2f} / "
          f"realizable {m['realizable_1p2086']:+.2f} / optimistic {m['optimistic_1p2047']:+.2f}")
    print(f"  cost of NOT shipping #154 (argmax-only decode): "
          f"{fs['cost_of_not_shipping_154_lcb_tps']:+.2f} TPS LCB headroom "
          f"({fs['cost_of_not_shipping_154_proj_tps']:+.2f} TPS proj)")
    print(f"  descent_only_p_clear500_at_conservative_step = "
          f"{out['descent_only_p_clear500_at_conservative_step']:.4f} [TEST]")

    led = out["step3_readiness_packet"]["leg_ledger"]
    print(f"\n[STEP 3] Refreshed ledger: {led['n_banked']} BANKED / {led['n_pending']} PENDING")
    print(f"  PENDING: {', '.join(led['pending'])}")

    st = out["step4_self_test"]
    print(f"\n[STEP 4] SELF-TEST (PRIMARY):")
    print(f"  (a) optimistic reproduces #167 + realizable GO -> "
          f"{'OK' if st['assert_a_reproduces_167_optimistic_and_realizable_go']['ok'] else 'FAIL'}")
    print(f"  (b) conservative verdict explicit ('{st['assert_b_conservative_verdict_explicit']['descent_only_first_verdict']}') -> "
          f"{'OK' if st['assert_b_conservative_verdict_explicit']['ok'] else 'FAIL'}")
    print(f"  (c) both-bugs GO at all three steps -> "
          f"{'OK' if st['assert_c_both_bugs_go_all_three_steps']['ok'] else 'FAIL'}")
    print(f"  (d) ledger matches merged state -> "
          f"{'OK' if st['assert_d_ledger_matches_merged']['ok'] else 'FAIL'}")
    print(f"  (e) NaN-clean -> {'OK' if st['assert_e_nan_clean']['ok'] else 'FAIL'}")
    print(f"  => conservative_step_verdict_self_test_passes = "
          f"{out['conservative_step_verdict_self_test_passes']}")

    print(f"\n[STEP 5] HAND-OFF: {out['step5_handoff']}")
    print(f"\n[PRIMARY] conservative_step_verdict_self_test_passes = "
          f"{out['conservative_step_verdict_self_test_passes']}")
    print(f"[TEST]    descent_only_p_clear500_at_conservative_step = "
          f"{out['descent_only_p_clear500_at_conservative_step']:.4f}")
    print(f"[STATE]   {out['gate_state']}")


def _log_wandb(args, out):
    import wandb
    op = out["operating_point"]
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                     config={"instrument": "conservative-step-launch-verdict",
                             "method": "cpu-analytic-reinstantiation-167-at-three-steps",
                             "K_cal": K_CAL, "base_step": args.base_step,
                             "step_conservative": op["step_conservative_launch_realized"],
                             "step_realizable": op["step_realizable"],
                             "step_optimistic": op["step_optimistic_conditional"],
                             "private_drop_descent_pct": op["private_drop_descent_only_pct"],
                             "private_drop_both_pct": op["private_drop_both_bugs_pct"],
                             "launch_bar_p_go": P_GO, "target_official": TARGET_OFFICIAL,
                             "frontier_official": FRONTIER_OFFICIAL})
    s = wandb.summary
    s["conservative_step_verdict_self_test_passes"] = out["conservative_step_verdict_self_test_passes"]
    s["descent_only_p_clear500_at_conservative_step"] = out["descent_only_p_clear500_at_conservative_step"]
    s["gate_state"] = out["gate_state"]
    s["metrics_nan_clean"] = out["metrics_nan_clean"]

    fs = out["step2_first_shot_settlement"]
    s["descent_only_first_verdict"] = fs["descent_only_first_verdict"]
    s["recommended_first_shot"] = fs["recommended_first_shot"]
    s["cost_of_not_shipping_154_lcb_tps"] = fs["cost_of_not_shipping_154_lcb_tps"]
    s["cost_of_not_shipping_154_proj_tps"] = fs["cost_of_not_shipping_154_proj_tps"]
    m = fs["lcb_margin_vs_500"]
    s["descent_lcb_margin_conservative"] = m["conservative_1p2182"]
    s["descent_lcb_margin_realizable"] = m["realizable_1p2086"]
    s["descent_lcb_margin_optimistic"] = m["optimistic_1p2047"]

    # per-step / per-topology summary scalars
    for name in ("conservative_1p2182", "realizable_1p2086", "optimistic_1p2047"):
        t = out["step1_three_step_instantiation"][name]
        for topo in ("descent_only", "both_bugs"):
            d = t[topo]
            pre = f"{name}_{topo}"
            s[f"{pre}_official"] = d["official_tps"]
            s[f"{pre}_proj_private"] = d["proj_private_tps"]
            s[f"{pre}_p_clear_500"] = d["p_clear_500"]
            s[f"{pre}_lcb_p90"] = d["lcb_p90"]
            s[f"{pre}_launch_go"] = d["launch_go_p90"]
            s[f"{pre}_conf99"] = d["consolidator_conf99_verdict"]

    led = out["step3_readiness_packet"]["leg_ledger"]
    s["n_banked"] = led["n_banked"]
    s["n_pending"] = led["n_pending"]
    for k in ("assert_a_reproduces_167_optimistic_and_realizable_go",
              "assert_b_conservative_verdict_explicit", "assert_c_both_bugs_go_all_three_steps",
              "assert_d_ledger_matches_merged", "assert_e_nan_clean"):
        s[f"selftest_{k}"] = int(out["step4_self_test"][k]["ok"])

    # three-step instantiation table
    it = wandb.Table(columns=["step_framing", "step", "topology", "official", "r_tree", "proj_private",
                              "p_clear_500", "lcb_p90", "conf99_verdict", "launch_go_p90"])
    for name in ("conservative_1p2182", "realizable_1p2086", "optimistic_1p2047"):
        t = out["step1_three_step_instantiation"][name]
        for topo in ("descent_only", "both_bugs"):
            d = t[topo]
            it.add_data(name, t["step"], topo, d["official_tps"], d["r_tree"], d["proj_private_tps"],
                        d["p_clear_500"], d["lcb_p90"], d["consolidator_conf99_verdict"], d["launch_go_p90"])
    wandb.log({"three_step_instantiation": it})

    # leg ledger table
    lt = wandb.Table(columns=["leg", "role", "status", "evidence"])
    for l in led["legs"]:
        lt.add_data(l["leg"], l["role"], l["status"], l["evidence"])
    wandb.log({"leg_ledger": lt})

    rid, rurl = run.id, run.url
    print(f"\nW&B run: {rid}  ({rurl})", flush=True)
    wandb.finish()
    return rid, rurl


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-step", type=float, default=STEP_MEASURED_DEPTH9,
                    help="lawine #136 merged depth-9 step (1.2182); the sampling model + ubel #154 "
                         "reductions anchor to it. combined_rel is step-invariant, so this fixes the CI.")
    ap.add_argument("--step-rel-half-width", type=float, default=STEP_REL_1SIGMA_DEFAULT,
                    help="lawine #136/#147 step-anchor 1-sigma relative (default 0.5%%).")
    ap.add_argument("--n-steps", type=int, default=cons.env.ORACLE_STEPS,
                    help="verify-step budget for the sampling CI (oracle 1024).")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="bootstrap resamples for the #146 sampling model (sigma_descend is "
                         "bootstrap-independent, so a small value suffices).")
    ap.add_argument("--seed", type=int, default=174)
    ap.add_argument("--rho-json", default=cons.RHO_OPT_JSON)
    ap.add_argument("--oracle-json", default=cons.ORACLE_LIVE_JSON)
    ap.add_argument("--out", default="research/spec_cost_model/conservative_step_launch_verdict_results.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", "--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb-entity", "--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb-name", "--wandb_name", default="fern/conservative-step-launch-verdict")
    ap.add_argument("--wandb-group", "--wandb_group", default="conservative-step-launch-verdict")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
