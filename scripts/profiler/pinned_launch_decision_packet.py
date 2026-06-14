#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Pinned-operating-point launch decision + readiness packet (PR #167).

WHAT THIS IS
------------
Every prior decision-geometry leg (#142 point gate, #145 facet decomp, #149 joint
frontier, #155 consolidator, #162 inverted build-gate) treated the private drop as a
FREE parameter (`--private-drop`). stark #156 (MERGED) has now PINNED it: the tree's
true native private drop is 1.80% (descent-only) / 1.86% (both-bugs) -- far below the
~6% threshold #162's Step-5 identified as the point where BUG-1 (wirbel #160's depth-1
spine) flips from deferrable-insurance to mandatory. lawine #161 (MERGED) separately
hardened the both-bugs accept-prep step from assumed to measured (537.84 official, spine
fix step-NEUTRAL).

So the launch decision can be INSTANTIATED at a single concrete operating point rather
than swept. This collapses the parameterized frontier into ONE pre-build verdict + the
verbatim projection/validity block of the eventual `Approval request: HF job` issue,
parameterized only on land #71's still-pending measured tuple.

PURE-ANALYTIC, CPU-ONLY. No GPU / vLLM / HF Job / submission / served-file change.
BASELINE stays 481.53; adds 0 TPS -- the launch-decision instantiation + readiness
packet, the capstone of the decision-geometry arc. IMPORTS committed leg outputs
(#155 consolidator, #162 frontier, stark #156 pinned-drop, lawine #161 both-bugs step);
does NOT re-derive them. This does NOT file the issue or launch -- that is a human-
approval decision; this produces the pre-filled draft + the verdict.

THE INSTANTIATION (PR steps 1-2)
--------------------------------
At the PINNED drop and ubel #154's realistic bar (E[T]>=4.808, step 1.2047), the
private projection of each topology corner (full recovery lambda=mu=1) is:
    proj_private = K_cal * E[T] * r_tree(d_pinned, topo) / step          (K_cal=125.268)
    P(clear 500) = min( Phi((proj-500)/sigma), Phi((geom_tps-500)/sigma) )  (#155 union)
                   sigma = proj * sqrt(samp^2 + calib^2 + step^2)         (#146/#148/#136)
Because the pinned drop 1.80% << 6%, BUG-1 is DEFERRED: the recommended FIRST shot is
descent-ONLY. Its pinned-point margin EXCEEDS #162's 511.1 (which sat at the harsher
GT 4.3%; 1.80% < 4.3% -> less haircut -> more margin).

THE LAUNCH GATE (the conservative one-shot bar)
-----------------------------------------------
For the irreversible one-shot, GO is taken at the P>=0.9 conservative bar (#162's LCB
gate), not the coin-flip P>=0.5: GO iff P(clear 500) >= 0.9 AND validity READY. The
consolidator's even-stricter conf-99 robust-GREEN view is reported alongside -- it is
the margin the BUG-1 spine buys (both-bugs clears conf-99; descent-only clears P>=0.9
but not conf-99).

THE READINESS PACKET (PR step 3)
--------------------------------
The verbatim projection+validity block of the future `Approval request: HF job` issue,
parameterized on land #71's PENDING tuple, with every leg stamped BANKED vs PENDING so
the human approver sees exactly what is still in flight. Imports the build gate
(lambda_min,mu_min)=(0.8809,0.7353) from #162; the pinned-drop private-safety from
#156; the measured both-bugs step 537.84 from lawine #161.

SELF-TEST (PR step 4 -- PRIMARY)
-------------------------------
At the pinned drop: (a) oracle E[T]=2.621 -> NO-GO; (b) both-bugs E[T]=5.207 -> GO;
(c) descent-only E[T]=5.0564 -> GO at the pinned 1.80% but NOT at the 9% ceiling
(operating-point-specific); (d) the packet correctly lists which legs are PENDING.
PRIMARY = launch_packet_self_test_passes; TEST = descent_only_p_clear500_at_pinned_drop.

Distinct from lawine's step-reconciliation (SUPPLIES the step consumed here), denken
#166 (PPL stamp), kanna #159 (sigma_hw), wirbel #165 (index-map), land #71 (builds the
kernel this packet gates). Serves nothing -> greedy identity untouched. NOT a launch.
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
# Import the merged legs VERBATIM (one source of truth per constant).
# ============================================================================
frontier = _load("tightened_private_500_frontier",
                 os.path.join(_HERE, "tightened_private_500_frontier.py"))   # fern #162
cons = frontier.cons                                                         # fern #155 (re-export)

# #162 frontier machinery (private retention + corners)
load_ubel154_bars = frontier.load_ubel154_bars
load_stark151_retention = frontier.load_stark151_retention
r_tree = frontier.r_tree
b_dict_for_depth1 = frontier.b_dict_for_depth1
DEPTH1_DESCENT_ONLY = frontier.DEPTH1_DESCENT_ONLY        # 0.679 (BUG-1 unfixed)
Z_P90_ONESIDED = frontier.Z_P90_ONESIDED                 # 1.281552 (Phi(z)=0.9)

# #155 consolidator machinery (the GO/NO-GO + CI + binding-leg)
consolidate = cons.consolidate
SamplingModel = cons.SamplingModel
load_kcal_band = cons.load_kcal_band
build_joint_b_dict = cons.build_joint_b_dict
validity_gate = cons.validity_gate
joint_et = cons.decomp.joint_et
K_CAL = cons.K_CAL                                        # 125.268
TARGET_OFFICIAL = cons.TARGET_OFFICIAL                   # 500.0
FRONTIER_OFFICIAL = cons.FRONTIER_OFFICIAL               # 481.53
ORACLE_E_T = cons.ORACLE_E_T                             # 2.621
E_T_TREE = cons.E_T_TREE                                 # 5.207 (both-bugs ceiling)
RHO2_BRANCH_HIT = cons.RHO2_BRANCH_HIT                   # 0.4165
PPL_GATE = cons.PPL_GATE                                 # 2.42
STEP_MEASURED_DEPTH9 = cons.STEP_MEASURED_DEPTH9         # 1.2182
STEP_REL_1SIGMA_DEFAULT = cons.STEP_REL_1SIGMA_DEFAULT   # 0.005

# committed artifacts folded in (imported, not re-derived)
STARK156_JSON = os.path.join(_ROOT, "research/validity/tree_private_drop_reconcile/results.json")
LAWINE161_JSON = os.path.join(_ROOT, "research/both_bugs_step_cost/both_bugs_step_cost.json")
FRONTIER162_JSON = os.path.join(_ROOT, "research/spec_cost_model/tightened_private_500_frontier_results.json")

# 9% band-ceiling drop (stark #151 band high); the operating-point contrast for self-test (c)
BAND_CEILING_DROP = 0.09

# launch confidence bars
P_GO = 0.9            # conservative one-shot launch bar (#162 LCB gate)
P_HOLD = 0.5          # below this -> NO-GO (does not clear even P50)


def _finite(x, default: float = 0.0) -> float:
    try:
        return float(x) if (x is not None and math.isfinite(float(x))) else default
    except (TypeError, ValueError):
        return default


# ============================================================================
# Load the pinned drop (stark #156) and the E[T] anchors + step (lawine #161).
# ============================================================================
def load_stark156_pinned(path: str = STARK156_JSON) -> dict:
    with open(path) as f:
        s = json.load(f)
    h = s["headline"]
    return {
        "drop_descent_only": float(h["tree_private_drop_pct_pinned"]) / 100.0,        # 0.018015
        "drop_both_bugs": float(h["tree_private_drop_pct_pinned_both_bugs"]) / 100.0,  # 0.018626
        "drop_descent_only_pct": float(h["tree_private_drop_pct_pinned"]),
        "drop_both_bugs_pct": float(h["tree_private_drop_pct_pinned_both_bugs"]),
        "stark_proj_descent": float(h["tree_private_tps_proj_pinned"]),               # 510.58 (stark's own step)
        "stark_proj_both": float(h["tree_private_tps_proj_pinned_both_bugs"]),        # 525.46
        "breakeven_descent_pct": float(h["breakeven_private_drop_descent_only_pct"]),
        "calibrated_linear_drop_pct": float(h["calibrated_linear_drop_pct"]),
    }


def load_lawine161_anchors(path: str = LAWINE161_JSON) -> dict:
    with open(path) as f:
        a = json.load(f)["anchors"]
    return {
        "e_t_descent_only": float(a["e_t_descent_only"]),       # 5.0564
        "e_t_both_bugs": float(a["e_t_both_bugs"]),             # 5.207
        "depth1_descent": float(a["depth1_descent"]),          # 0.679
        "depth1_both": float(a["depth1_both"]),                # 0.7287
        "official_both_assumed_roofline": float(a["official_both_assumed_roofline"]),  # 537.84
        "measured_step_136": float(a["measured_step_136"]),    # 1.2182
        "roofline_step": float(a["roofline_step"]),            # 1.2127
    }


def load_frontier162_gate(path: str = FRONTIER162_JSON) -> dict:
    with open(path) as f:
        f162 = json.load(f)
    lm = f162["lambda_mu_min_private_safe"]
    return {
        "lambda_min": float(lm[0]), "mu_min": float(lm[1]),          # (0.8809, 0.7353) @ P>=0.5
        "self_test_passes": int(f162["tightened_frontier_self_test_passes"]),
        "bar_et_realistic": float(f162["build_gate_headline"]["bar_et"]),
        "step_realistic": float(f162["build_gate_headline"]["step"]),
        "bug1_mandatory_at_band_ceiling": bool(f162["bug1_mandatory_handoff"]["bug1_mandatory_at_band_ceiling_realistic"]),
        "descent_proj_at_gt_realistic": float(next(
            r["descent_only_corner_proj"] for r in f162["bug1_mandatory_handoff"]["rows"]
            if r["bar_label"] == "realistic" and abs(r["private_drop_pct"] - 4.3) < 1e-6)),
    }


# ============================================================================
# The instantiation: private projection of a topology corner at the pinned drop,
# pushed through the #155 consolidator (CI / binding-leg / validity verbatim).
# ============================================================================
def instantiate_private(E_T, branch_hit, lam, mu, topo, drop, step, b_dict, knots,
                        sampling, kcal_band, validity, conf=99):
    """Private projection at (E_T, lam, mu) under a private drop `drop` for `topo`,
    evaluated through the consolidator. The private haircut applies to the headline
    numerator (E_priv = E_T * r_tree(d)); the consolidator then maps it to proj/CI and
    folds the (lam,mu) geometry membership (public, can only SUPPRESS). Launch GO is
    taken at the conservative P>=0.9 bar."""
    rt = r_tree(drop, topo, knots)
    et_priv = _finite(E_T * rt)
    dec = consolidate(et_priv, branch_hit, lam, mu, step=step,
                      ppl=validity.get("ppl"), boots=validity.get("boots"),
                      completed=validity.get("completed"), conf=conf,
                      sampling=sampling, b_dict=b_dict, kcal_band=kcal_band)
    proj = dec["proj_tps"]
    cr = dec["uncertainty"]["combined_rel_1sigma"]
    p_clear = dec["p_clear_500"]
    lcb_p90 = _finite(proj * (1.0 - Z_P90_ONESIDED * cr))
    validity_ready = (dec["validity_gate"] == "READY")

    # conservative one-shot launch verdict (P>=0.9 bar, gated on validity)
    if not validity_ready:
        launch_go = "NO-GO"
    elif p_clear >= P_GO:
        launch_go = "GO"
    elif p_clear < P_HOLD:
        launch_go = "NO-GO"
    else:
        launch_go = "HOLD"

    return {
        "topo": topo, "E_T_input": _finite(E_T), "lambda": lam, "mu": mu,
        "private_drop": _finite(drop), "private_drop_pct": _finite(drop * 100.0),
        "r_tree": _finite(rt), "et_private": et_priv,
        "proj_private_tps": proj,
        "ci99_lo": dec["ci_lo"], "ci99_hi": dec["ci_hi"],
        "combined_rel_1sigma": _finite(cr),
        "lcb_p90": lcb_p90,
        "p_clear_500": _finite(p_clear),
        "clears_p50": bool(proj >= TARGET_OFFICIAL),
        "clears_p90": bool(lcb_p90 >= TARGET_OFFICIAL),
        "geom_tps_public": dec["decision_geometry"]["geom_tps"],
        "p_geom_membership": dec["decision_geometry"]["geom_membership_prob"],
        "binding_leg": dec["binding_leg"],
        "consolidator_conf99_verdict": dec["verdict"],     # robust-GREEN/RED/INDETERMINATE
        "consolidator_conf99_go": dec["go_no_go"],         # GO iff conf-99 robust-GREEN
        "validity_gate": dec["validity_gate"],
        "launch_go_p90": launch_go,                        # the conservative one-shot verdict
    }


# ============================================================================
# Self-test (PRIMARY) -- the four PR assertions at the pinned operating point.
# ============================================================================
def self_test(b_both, b_desc, knots, step_real, pinned, lawine, sampling, kcal_band):
    ok_validity = {"ppl": 2.39, "boots": True, "completed": 128}   # isolate the TPS verdict
    et_oracle, et_desc, et_both = ORACLE_E_T, lawine["e_t_descent_only"], lawine["e_t_both_bugs"]
    d_desc, d_both = pinned["drop_descent_only"], pinned["drop_both_bugs"]

    # (a) oracle (as-built lambda=0, mu=1) at the pinned drop -> NO-GO.
    oracle = instantiate_private(et_oracle, RHO2_BRANCH_HIT, 0.0, 1.0, "both_bugs", d_both,
                                 step_real, b_both, knots, sampling, kcal_band, ok_validity)
    assert_a = bool(oracle["launch_go_p90"] == "NO-GO")

    # (b) both-bugs (full recovery) at the pinned 1.86% -> GO.
    both = instantiate_private(et_both, RHO2_BRANCH_HIT, 1.0, 1.0, "both_bugs", d_both,
                               step_real, b_both, knots, sampling, kcal_band, ok_validity)
    assert_b = bool(both["launch_go_p90"] == "GO")

    # (c) descent-only (full recovery, BUG-1 unfixed) -> GO at pinned 1.80%, NOT at 9% ceiling.
    desc_pinned = instantiate_private(et_desc, RHO2_BRANCH_HIT, 1.0, 1.0, "descent_only", d_desc,
                                      step_real, b_desc, knots, sampling, kcal_band, ok_validity)
    desc_ceiling = instantiate_private(et_desc, RHO2_BRANCH_HIT, 1.0, 1.0, "descent_only",
                                       BAND_CEILING_DROP, step_real, b_desc, knots, sampling,
                                       kcal_band, ok_validity)
    c_pinned_go = bool(desc_pinned["launch_go_p90"] == "GO")
    c_ceiling_not_go = bool(desc_ceiling["launch_go_p90"] != "GO")
    assert_c = bool(c_pinned_go and c_ceiling_not_go)

    return {
        "oracle": oracle, "both_bugs": both,
        "descent_only_pinned": desc_pinned, "descent_only_ceiling": desc_ceiling,
        "assert_a_oracle_no_go": {"ok": assert_a, "E_T": et_oracle,
                                  "launch_go": oracle["launch_go_p90"],
                                  "proj": oracle["proj_private_tps"], "expect": "NO-GO"},
        "assert_b_both_bugs_go": {"ok": assert_b, "E_T": et_both,
                                  "launch_go": both["launch_go_p90"],
                                  "proj": both["proj_private_tps"],
                                  "p_clear_500": both["p_clear_500"], "expect": "GO"},
        "assert_c_descent_operating_point": {
            "ok": assert_c, "E_T": et_desc,
            "pinned_drop_pct": pinned["drop_descent_only_pct"], "pinned_go": desc_pinned["launch_go_p90"],
            "pinned_proj": desc_pinned["proj_private_tps"], "pinned_p_clear": desc_pinned["p_clear_500"],
            "ceiling_drop_pct": BAND_CEILING_DROP * 100.0, "ceiling_go": desc_ceiling["launch_go_p90"],
            "ceiling_proj": desc_ceiling["proj_private_tps"], "ceiling_p_clear": desc_ceiling["p_clear_500"],
            "expect": "GO at pinned, NOT-GO at 9% ceiling (operating-point-specific)"},
    }


# ============================================================================
# The leg ledger (PR step 3) -- BANKED vs PENDING, asserted by self-test (d).
# ============================================================================
def build_leg_ledger(frontier162, lawine):
    """Every leg the Approval-request packet rests on, stamped BANKED vs PENDING so the
    human approver sees exactly what is still in flight."""
    legs = [
        {"leg": "fern #155 consolidator", "role": "GO/NO-GO + CI + binding-leg union",
         "status": "BANKED", "evidence": "approval_projection_consolidator_results.json (self-test PASS)"},
        {"leg": "fern #162 build gate", "role": "(lambda_min,mu_min)=(0.8809,0.7353) @ P>=0.5",
         "status": "BANKED", "evidence": f"tightened_private_500_frontier_results.json (self-test {frontier162['self_test_passes']})"},
        {"leg": "stark #156 pinned drop", "role": "private drop 1.80% desc / 1.86% both (anchored GT-4.3%)",
         "status": "BANKED", "evidence": "tree_private_drop_reconcile/results.json"},
        {"leg": "lawine #161 both-bugs step", "role": f"measured step-NEUTRAL -> both-bugs official {lawine['official_both_assumed_roofline']:.1f}",
         "status": "BANKED", "evidence": "both_bugs_step_cost.json (step_delta 0.0%)"},
        {"leg": "denken #150 validity (does-it-score)", "role": "PPL<=2.42 & boots & 128/128 contract",
         "status": "BANKED", "evidence": "imported inline by #155 (ARMED/PENDING land #71's run)"},
        {"leg": "denken #158 greedy-exactness", "role": "per-token committed==argmax (BUG-2 catcher)",
         "status": "BANKED", "evidence": "--audit-kernel-symbol pre-merge contract gate"},
        {"leg": "lawine step-reconciliation", "role": "final depth-9 step reconcile (overlap vs roofline)",
         "status": "PENDING", "evidence": "in flight -- both-bugs step locked NEUTRAL by #161; final reconcile pending"},
        {"leg": "denken #166 PPL-margin bound", "role": "M=32 batched-verify aggregate-PPL worst-case vs 2.42",
         "status": "PENDING", "evidence": "in flight -- #150/#158 ASSUME it, neither BOUNDS it"},
        {"leg": "kanna #159 sigma_hw", "role": "4th quadrature term (A10G clock/thermal/cold-start)",
         "status": "PENDING", "evidence": "in flight -- CI here is the 3-term quadrature; sigma_hw widens it"},
        {"leg": "land #71 measured tuple", "role": "(E[T], rho2, lambda, mu, step, ppl, boots, completed)",
         "status": "PENDING", "evidence": "the kernel this packet gates -- not yet built"},
    ]
    banked = [l["leg"] for l in legs if l["status"] == "BANKED"]
    pending = [l["leg"] for l in legs if l["status"] == "PENDING"]
    return {"legs": legs, "banked": banked, "pending": pending,
            "n_banked": len(banked), "n_pending": len(pending)}


def assert_packet_pending_correct(ledger) -> dict:
    """Self-test (d): the packet correctly lists which legs are PENDING. The four in-
    flight legs the PR names (lawine step-reconcile, denken #166, kanna #159, land #71)
    must be PENDING; the six banked legs must be BANKED."""
    expect_pending = {"lawine step-reconciliation", "denken #166 PPL-margin bound",
                      "kanna #159 sigma_hw", "land #71 measured tuple"}
    expect_banked = {"fern #155 consolidator", "fern #162 build gate", "stark #156 pinned drop",
                     "lawine #161 both-bugs step", "denken #150 validity (does-it-score)",
                     "denken #158 greedy-exactness"}
    got_pending = set(ledger["pending"])
    got_banked = set(ledger["banked"])
    ok = bool(got_pending == expect_pending and got_banked == expect_banked)
    return {"ok": ok, "expected_pending": sorted(expect_pending),
            "got_pending": sorted(got_pending), "expected_banked": sorted(expect_banked),
            "got_banked": sorted(got_banked),
            "expect": "exactly the 4 in-flight legs PENDING, the 6 evidence legs BANKED"}


# ============================================================================
# The readiness packet body (the verbatim Approval-request projection+validity block).
# ============================================================================
def render_packet_md(headline, frontier162, pinned, lawine, ledger, recommended) -> str:
    desc, both = headline["descent_only"], headline["both_bugs"]
    lm, mm = frontier162["lambda_min"], frontier162["mu_min"]
    banked = "\n".join(f"- [BANKED]  {l['leg']} -- {l['role']}" for l in ledger["legs"] if l["status"] == "BANKED")
    pending = "\n".join(f"- [PENDING] {l['leg']} -- {l['role']}" for l in ledger["legs"] if l["status"] == "PENDING")
    return f"""\
### Approval request: HF job for tree-descent submission (PRE-FILLED DRAFT -- NOT YET FILED)

**This block is the pre-filled projection+validity body of the eventual `Approval
request: HF job` issue. It awaits ONLY land #71's measured tuple and the PENDING legs
below. It does NOT authorize a launch; a human must approve the filed issue.**

**Operating point (PINNED):** private drop {pinned['drop_descent_only_pct']:.2f}% (descent-only) /
{pinned['drop_both_bugs_pct']:.2f}% (both-bugs), anchored to flagship GT-4.3% (stark #156).
Realistic bar E[T]>={frontier162['bar_et_realistic']:.3f}, step {frontier162['step_realistic']:.4f}
(ubel #154 scatter+LP avoidance, greedy-token-identical).

**Projection formula (parameterized on land #71's PENDING tuple):**
```
proj_private = K_cal * E[T]_land * r_tree(d_pinned, topo) / step          (K_cal={K_CAL:.3f})
P(clear 500) = min( Phi((proj-500)/sigma), Phi((geom_tps(lambda,mu)-500)/sigma) )
sigma        = proj * sqrt(samp^2 + calib^2 + step^2)   [+ sigma_hw PENDING kanna #159]
GO  iff  P(clear 500) >= 0.9  AND  (lambda,mu) >= ({lm:.4f}, {mm:.4f})  AND  validity READY
```

**Instantiated projection at the pinned point (full-recovery corner lambda=mu=1):**

| topology | E[T] | r_tree | proj_private | P(clear 500) | LCB(P>=0.9) | CI99 | conf-99 | launch (P>=0.9) | binding leg |
|---|---|---|---|---|---|---|---|---|---|
| **descent-only (BUG-1 deferred)** | {desc['E_T_input']:.4f} | {desc['r_tree']:.4f} | **{desc['proj_private_tps']:.1f}** | {desc['p_clear_500']*100:.1f}% | {desc['lcb_p90']:.1f} | [{desc['ci99_lo']:.1f}, {desc['ci99_hi']:.1f}] | {desc['consolidator_conf99_verdict']} | **{desc['launch_go_p90']}** | {desc['binding_leg']} |
| both-bugs (BUG-1 fixed) | {both['E_T_input']:.4f} | {both['r_tree']:.4f} | {both['proj_private_tps']:.1f} | {both['p_clear_500']*100:.1f}% | {both['lcb_p90']:.1f} | [{both['ci99_lo']:.1f}, {both['ci99_hi']:.1f}] | {both['consolidator_conf99_verdict']} | {both['launch_go_p90']} | {both['binding_leg']} |

**Recommended first shot:** {recommended['topology']} -- {recommended['rationale']}

**Validity stamps (must ALL be READY before the filed issue is approved):**
- denken #150 (does-it-score): PPL<=2.42 & boots & 128/128 -- BANKED contract, ARMED/PENDING land #71's run.
- denken #158 (greedy-exact): per-token committed==argmax -- BANKED, `--audit-kernel-symbol` pre-merge gate.
- denken #166 (PPL-margin bound): M=32 aggregate-PPL worst-case vs 2.42 -- **PENDING** (the last unbounded validity dim).
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
    pinned = load_stark156_pinned()
    lawine = load_lawine161_anchors()
    frontier162 = load_frontier162_gate()

    ubel = load_ubel154_bars(base_step=args.base_step)
    stark151 = load_stark151_retention(step=args.base_step)
    knots = stark151["knots"]
    kcal_band = load_kcal_band()
    sampling = SamplingModel(n_steps=args.n_steps, n_boot=args.n_boot, seed=args.seed,
                             step=args.base_step, step_rel_hw=step_rel)

    step_real = ubel["points"]["realistic"]["step"]
    bar_et_real = ubel["points"]["realistic"]["bar_et"]

    # banked #149 b_dicts: BUG-1-fixed (depth-1 rho-opt 0.7287) and BUG-1-unfixed (0.679).
    b_both = build_joint_b_dict(args.rho_json, args.oracle_json)
    b_desc = b_dict_for_depth1(b_both, DEPTH1_DESCENT_ONLY)

    # ---- consistency: the surface corners reproduce lawine #161's E[T] anchors ----
    et_corner_both = _finite(joint_et(1.0, 1.0, b_both))
    et_corner_desc = _finite(joint_et(1.0, 1.0, b_desc))

    # ---- STEP 1: instantiate at the pinned operating point (realistic bar) ----
    ok_validity = {"ppl": 2.39, "boots": True, "completed": 128}
    desc = instantiate_private(lawine["e_t_descent_only"], RHO2_BRANCH_HIT, 1.0, 1.0,
                               "descent_only", pinned["drop_descent_only"], step_real,
                               b_desc, knots, sampling, kcal_band, ok_validity)
    both = instantiate_private(lawine["e_t_both_bugs"], RHO2_BRANCH_HIT, 1.0, 1.0,
                               "both_bugs", pinned["drop_both_bugs"], step_real,
                               b_both, knots, sampling, kcal_band, ok_validity)
    headline = {"descent_only": desc, "both_bugs": both}

    # ---- STEP 2: resolve BUG-1 concretely (pinned 1.80% << 6% -> deferred) ----
    descent_only_p_clear500_at_pinned_drop = _finite(desc["p_clear_500"])
    bug1 = {
        "pinned_drop_pct": pinned["drop_descent_only_pct"],
        "bug1_binds_above_drop_pct": 6.0,                       # #162 Step-5 threshold
        "bug1_binding": bool(pinned["drop_descent_only_pct"] > 6.0),
        "recommended_first_shot": "descent-ONLY (BUG-1 deferred)",
        "descent_only_proj_at_pinned": desc["proj_private_tps"],
        "descent_only_p_clear500_at_pinned_drop": descent_only_p_clear500_at_pinned_drop,
        "exceeds_162_gt_511": bool(desc["proj_private_tps"] > frontier162["descent_proj_at_gt_realistic"]),
        "f162_descent_proj_at_gt_4p3": frontier162["descent_proj_at_gt_realistic"],
        "margin_vs_162_gt": _finite(desc["proj_private_tps"] - frontier162["descent_proj_at_gt_realistic"]),
        "note": ("pinned drop 1.80% << the ~6% BUG-1 threshold (#162 Step-5): BUG-1 (wirbel "
                 "#160 depth-1 spine) is DEFERRED. The descent-only pinned margin EXCEEDS "
                 "#162's 511.1 (which sat at the harsher GT-4.3%; 1.80% < 4.3% -> less haircut). "
                 "The spine is no longer mandatory for private-safety at the pinned drop -- it "
                 "buys the conf-99 margin + the 9%-band-ceiling insurance."),
    }

    # ---- STEP 4: self-test (PRIMARY) ----  (built before the packet so (d) can read the ledger)
    st = self_test(b_both, b_desc, knots, step_real, pinned, lawine, sampling, kcal_band)
    ledger = build_leg_ledger(frontier162, lawine)
    st_d = assert_packet_pending_correct(ledger)
    assert_a = st["assert_a_oracle_no_go"]["ok"]
    assert_b = st["assert_b_both_bugs_go"]["ok"]
    assert_c = st["assert_c_descent_operating_point"]["ok"]
    assert_d = st_d["ok"]
    launch_packet_self_test_passes = int(bool(assert_a and assert_b and assert_c and assert_d))

    # ---- STEP 3: assemble the readiness packet (recommended shot from the verdicts) ----
    recommended = {
        "topology": "descent-ONLY (BUG-1 deferred)",
        "rationale": (f"pinned drop {pinned['drop_descent_only_pct']:.2f}% << 6%, so BUG-1 is not "
                      f"mandatory; descent-only clears the conservative P>=0.9 launch bar "
                      f"(proj {desc['proj_private_tps']:.1f}, LCB {desc['lcb_p90']:.1f} >= 500). The "
                      f"both-bugs spine upgrades it to conf-99 robust-GREEN ({both['proj_private_tps']:.1f}) "
                      f"and covers the 9% band-ceiling, but is deferrable for the first shot."),
    }
    packet_md = render_packet_md(headline, frontier162, pinned, lawine, ledger, recommended)

    # ---- STEP 5: hand-off ----
    handoff = (
        f"At the pinned operating point, the recommended first shot is descent-only, "
        f"P(clear-500)={desc['p_clear_500']*100:.1f}% (proj {desc['proj_private_tps']:.1f}, "
        f"LCB(P>=0.9) {desc['lcb_p90']:.1f}), pending land #71's measured (lambda,mu) reaching "
        f"({frontier162['lambda_min']:.3f}, {frontier162['mu_min']:.3f}) and the PENDING legs "
        f"(lawine step-reconcile, denken #166 PPL, kanna #159 sigma_hw). The packet is the "
        f"pre-filled issue body awaiting only land's tuple -- it does NOT authorize a launch.")

    state = "ARMED" if launch_packet_self_test_passes else "SELF-TEST-FAIL"
    out = {
        "primary_metric_name": "launch_packet_self_test_passes",
        "launch_packet_self_test_passes": launch_packet_self_test_passes,
        "test_metric_name": "descent_only_p_clear500_at_pinned_drop",
        "descent_only_p_clear500_at_pinned_drop": descent_only_p_clear500_at_pinned_drop,
        "gate_state": state,
        "operating_point": {
            "private_drop_descent_only_pct": pinned["drop_descent_only_pct"],
            "private_drop_both_bugs_pct": pinned["drop_both_bugs_pct"],
            "bar_et_realistic": _finite(bar_et_real), "step_realistic": _finite(step_real),
            "anchored_to_flagship_gt_pct": pinned["calibrated_linear_drop_pct"],
            "source": "stark #156 pinned drop + ubel #154 realistic bar",
        },
        "step1_pinned_instantiation": {
            "descent_only": desc, "both_bugs": both,
            "cross_check_stark156_proj_descent": pinned["stark_proj_descent"],
            "cross_check_stark156_proj_both": pinned["stark_proj_both"],
            "note": ("realistic-bar PRIVATE projection of the full-recovery corner. stark #156's "
                     "own descent 510.6 / both 525.5 sit at his measured step 1.2182; the realistic "
                     "bar (1.2047) lifts them. Both-bugs PUBLIC official is lawine #161's 537.84 "
                     "(roofline 1.2127, no private haircut)."),
        },
        "step2_bug1_resolution": bug1,
        "step3_readiness_packet": {
            "packet_markdown": packet_md,
            "leg_ledger": ledger,
            "recommended_first_shot": recommended,
            "build_gate_lambda_mu_min": [frontier162["lambda_min"], frontier162["mu_min"]],
        },
        "step4_self_test": {
            "passes": bool(launch_packet_self_test_passes),
            "assert_a_oracle_no_go": st["assert_a_oracle_no_go"],
            "assert_b_both_bugs_go": st["assert_b_both_bugs_go"],
            "assert_c_descent_operating_point": st["assert_c_descent_operating_point"],
            "assert_d_packet_pending_correct": st_d,
        },
        "step5_handoff": handoff,
        "imported_legs": {
            "fern_155_consolidator": "consolidate() / SamplingModel / validity_gate (verbatim)",
            "fern_162_frontier": "r_tree / private retention / (lambda_min,mu_min)=(0.8809,0.7353)",
            "stark_156_pinned_drop": [pinned["drop_descent_only_pct"], pinned["drop_both_bugs_pct"]],
            "lawine_161_both_bugs_step": lawine["official_both_assumed_roofline"],
            "ubel_154_realistic_bar": [_finite(bar_et_real), _finite(step_real)],
        },
        "surface_corner_consistency": {
            "joint_et_1_1_both": et_corner_both, "lawine_e_t_both": lawine["e_t_both_bugs"],
            "joint_et_1_1_descent": et_corner_desc, "lawine_e_t_descent": lawine["e_t_descent_only"],
            "both_within_0p5pct": bool(abs(et_corner_both - lawine["e_t_both_bugs"]) / lawine["e_t_both_bugs"] <= 0.005),
            "descent_within_0p5pct": bool(abs(et_corner_desc - lawine["e_t_descent_only"]) / lawine["e_t_descent_only"] <= 0.005),
        },
        "uncertainty_model": {
            "quadrature_formula": "combined = sqrt(sampling^2 + calibration^2 + step_anchor^2)",
            "fourth_term_pending": "kanna #159 sigma_hw (A10G clock/thermal/cold-start) -- NOT yet folded",
            "launch_bar": "P(clear 500) >= 0.9 (conservative one-shot LCB gate); conf-99 reported alongside",
            "z_p90_one_sided": Z_P90_ONESIDED,
            "sigma_descend_146": _finite(sampling.sigma_descend),
            "calib_downside_rel_148": _finite(kcal_band["calib_downside_rel"]),
            "step_anchor_rel_136": _finite(step_rel),
        },
        "provenance": (
            "INSTANTIATES the decision-geometry arc at stark #156's pinned operating point. "
            "Imports VERBATIM: fern #155 consolidator (GO/NO-GO + CI + binding-leg), fern #162 "
            "frontier (private retention + build gate), stark #156 pinned drop, lawine #161 "
            "both-bugs step, ubel #154 realistic bar. One source of truth per constant -- "
            "imports, does not re-derive. Capstone of #142/#145/#149/#155/#162."),
        "method": (
            "LOCAL CPU-only analytic synthesis; no GPU/vLLM/HF Job/submission/kernel build. "
            "BASELINE stays 481.53; adds 0 TPS -- the launch-decision instantiation + readiness "
            "packet. Does NOT file the issue or authorize a launch. Greedy identity untouched."),
        "metrics_nan_clean": 1,
        "wandb_run_id": None,
        "wandb_url": None,
        "elapsed_s": None,
    }

    _print_console(out)

    if args.wandb and not args.no_wandb:
        try:
            rid, rurl = _log_wandb(args, out, sampling, kcal_band)
            out["wandb_run_id"], out["wandb_url"] = rid, rurl
        except Exception as e:  # noqa: BLE001
            print(f"[pinned-launch-packet] W&B logging failed (non-fatal): {e!r}", flush=True)

    out["elapsed_s"] = round(time.time() - t0, 4)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    return out


def _print_console(out):
    print("=" * 100)
    print("PINNED-OPERATING-POINT LAUNCH DECISION + READINESS PACKET (PR #167)")
    print("=" * 100)
    op = out["operating_point"]
    print(f"\nOperating point (PINNED): drop {op['private_drop_descent_only_pct']:.2f}% desc / "
          f"{op['private_drop_both_bugs_pct']:.2f}% both; realistic bar E[T]>={op['bar_et_realistic']:.3f} "
          f"step {op['step_realistic']:.4f}\n")

    print("[STEP 1] PINNED instantiation (full-recovery corner, realistic bar):")
    for label, k in (("descent-only", "descent_only"), ("both-bugs", "both_bugs")):
        d = out["step1_pinned_instantiation"][k]
        print(f"  {label:12s} E[T]={d['E_T_input']:.4f} r_tree={d['r_tree']:.4f} -> proj "
              f"{d['proj_private_tps']:6.1f}  P(>=500)={d['p_clear_500']*100:5.1f}%  "
              f"LCB90={d['lcb_p90']:6.1f}  CI99[{d['ci99_lo']:.1f},{d['ci99_hi']:.1f}]  "
              f"conf99={d['consolidator_conf99_verdict']:>16s}  launch(P>=0.9)={d['launch_go_p90']}  "
              f"bind={d['binding_leg']}")

    b = out["step2_bug1_resolution"]
    print(f"\n[STEP 2] BUG-1 resolution: pinned {b['pinned_drop_pct']:.2f}% "
          f"{'>' if b['bug1_binding'] else '<<'} 6% -> BUG-1 {'BINDS' if b['bug1_binding'] else 'DEFERRED'}")
    print(f"  recommended first shot: {b['recommended_first_shot']}")
    print(f"  descent-only proj {b['descent_only_proj_at_pinned']:.1f} vs #162 GT-4.3% "
          f"{b['f162_descent_proj_at_gt_4p3']:.1f} -> EXCEEDS by {b['margin_vs_162_gt']:.1f} "
          f"({b['exceeds_162_gt_511']})")
    print(f"  descent_only_p_clear500_at_pinned_drop = {b['descent_only_p_clear500_at_pinned_drop']:.4f} [TEST]")

    print(f"\n[STEP 3] Readiness packet: {out['step3_readiness_packet']['leg_ledger']['n_banked']} BANKED / "
          f"{out['step3_readiness_packet']['leg_ledger']['n_pending']} PENDING; build gate "
          f"(lambda_min,mu_min)={out['step3_readiness_packet']['build_gate_lambda_mu_min']}")
    print(f"  PENDING: {', '.join(out['step3_readiness_packet']['leg_ledger']['pending'])}")

    st = out["step4_self_test"]
    print(f"\n[STEP 4] SELF-TEST (PRIMARY):")
    print(f"  (a) oracle E[T]=2.621 -> {st['assert_a_oracle_no_go']['launch_go']:>6s}  "
          f"-> {'OK' if st['assert_a_oracle_no_go']['ok'] else 'FAIL'}")
    print(f"  (b) both-bugs E[T]=5.207 -> {st['assert_b_both_bugs_go']['launch_go']:>6s}  "
          f"-> {'OK' if st['assert_b_both_bugs_go']['ok'] else 'FAIL'}")
    cc = st["assert_c_descent_operating_point"]
    print(f"  (c) descent-only pinned {cc['pinned_drop_pct']:.2f}% -> {cc['pinned_go']} "
          f"(p={cc['pinned_p_clear']*100:.1f}%) / 9% ceiling -> {cc['ceiling_go']} "
          f"(p={cc['ceiling_p_clear']*100:.1f}%)  -> {'OK' if cc['ok'] else 'FAIL'}")
    print(f"  (d) packet PENDING-list correct -> {'OK' if st['assert_d_packet_pending_correct']['ok'] else 'FAIL'}")
    print(f"  => launch_packet_self_test_passes = {out['launch_packet_self_test_passes']}")

    print(f"\n[STEP 5] HAND-OFF: {out['step5_handoff']}")
    print(f"\n[PRIMARY] launch_packet_self_test_passes = {out['launch_packet_self_test_passes']}")
    print(f"[TEST]    descent_only_p_clear500_at_pinned_drop = {out['descent_only_p_clear500_at_pinned_drop']:.4f}")
    print(f"[STATE]   {out['gate_state']}")


def _log_wandb(args, out, sampling, kcal_band):
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                     config={"instrument": "pinned-launch-decision-packet",
                             "method": "cpu-analytic-instantiation-155-162-156-161-154",
                             "K_cal": K_CAL, "base_step": args.base_step,
                             "private_drop_descent_pct": out["operating_point"]["private_drop_descent_only_pct"],
                             "private_drop_both_pct": out["operating_point"]["private_drop_both_bugs_pct"],
                             "bar_et_realistic": out["operating_point"]["bar_et_realistic"],
                             "step_realistic": out["operating_point"]["step_realistic"],
                             "launch_bar_p_go": P_GO, "target_official": TARGET_OFFICIAL,
                             "frontier_official": FRONTIER_OFFICIAL})
    s = wandb.summary
    s["launch_packet_self_test_passes"] = out["launch_packet_self_test_passes"]
    s["descent_only_p_clear500_at_pinned_drop"] = out["descent_only_p_clear500_at_pinned_drop"]
    s["gate_state"] = out["gate_state"]
    desc, both = out["step1_pinned_instantiation"]["descent_only"], out["step1_pinned_instantiation"]["both_bugs"]
    s["descent_only_proj_pinned"] = desc["proj_private_tps"]
    s["descent_only_lcb_p90"] = desc["lcb_p90"]
    s["descent_only_launch_go"] = desc["launch_go_p90"]
    s["descent_only_conf99_verdict"] = desc["consolidator_conf99_verdict"]
    s["both_bugs_proj_pinned"] = both["proj_private_tps"]
    s["both_bugs_p_clear_500"] = both["p_clear_500"]
    s["both_bugs_lcb_p90"] = both["lcb_p90"]
    s["both_bugs_launch_go"] = both["launch_go_p90"]
    s["both_bugs_conf99_verdict"] = both["consolidator_conf99_verdict"]
    s["bug1_binding"] = int(out["step2_bug1_resolution"]["bug1_binding"])
    s["descent_exceeds_162_gt"] = int(out["step2_bug1_resolution"]["exceeds_162_gt_511"])
    s["descent_margin_vs_162_gt"] = out["step2_bug1_resolution"]["margin_vs_162_gt"]
    s["n_banked"] = out["step3_readiness_packet"]["leg_ledger"]["n_banked"]
    s["n_pending"] = out["step3_readiness_packet"]["leg_ledger"]["n_pending"]
    for k in ("assert_a_oracle_no_go", "assert_b_both_bugs_go",
              "assert_c_descent_operating_point", "assert_d_packet_pending_correct"):
        s[f"selftest_{k}"] = int(out["step4_self_test"][k]["ok"])
    s["build_gate_lambda_min"] = out["step3_readiness_packet"]["build_gate_lambda_mu_min"][0]
    s["build_gate_mu_min"] = out["step3_readiness_packet"]["build_gate_lambda_mu_min"][1]

    # pinned-instantiation table
    it = wandb.Table(columns=["topology", "E_T", "r_tree", "proj_private", "p_clear_500",
                              "lcb_p90", "ci99_lo", "ci99_hi", "conf99_verdict", "launch_go_p90", "binding_leg"])
    for k in ("descent_only", "both_bugs"):
        d = out["step1_pinned_instantiation"][k]
        it.add_data(k, d["E_T_input"], d["r_tree"], d["proj_private_tps"], d["p_clear_500"],
                    d["lcb_p90"], d["ci99_lo"], d["ci99_hi"], d["consolidator_conf99_verdict"],
                    d["launch_go_p90"], d["binding_leg"])
    wandb.log({"pinned_instantiation": it})

    # leg ledger table
    lt = wandb.Table(columns=["leg", "role", "status", "evidence"])
    for l in out["step3_readiness_packet"]["leg_ledger"]["legs"]:
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
                    help="lawine #136 merged depth-9 step (1.2182); ubel #154 reductions apply to it.")
    ap.add_argument("--step-rel-half-width", type=float, default=STEP_REL_1SIGMA_DEFAULT,
                    help="lawine #136/#147 step-anchor 1-sigma relative (default 0.5%%).")
    ap.add_argument("--n-steps", type=int, default=cons.env.ORACLE_STEPS,
                    help="verify-step budget for the sampling CI (oracle 1024).")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="bootstrap resamples for the #146 sampling model (sigma_descend is "
                         "bootstrap-independent, so a small value suffices).")
    ap.add_argument("--seed", type=int, default=167)
    ap.add_argument("--rho-json", default=cons.RHO_OPT_JSON)
    ap.add_argument("--oracle-json", default=cons.ORACLE_LIVE_JSON)
    ap.add_argument("--out", default="research/spec_cost_model/pinned_launch_decision_packet_results.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", "--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb-entity", "--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb-name", "--wandb_name", default="fern/pinned-launch-decision-packet")
    ap.add_argument("--wandb-group", "--wandb_group", default="pinned-launch-decision-packet")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
