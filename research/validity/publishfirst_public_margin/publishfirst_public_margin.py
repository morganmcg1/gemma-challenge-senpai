#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Publish-first public margin: the public GO margin at the #124 publish-first floor (PR #234).

WHAT THIS IS
------------
My #229 (`bz2b3fw8`, MERGED) proved `VALIDITY_BINDS_SPEED_ALWAYS_CLEARS`: at the PRIVATE
validity bar lambda_hat=0.9780 the public speed clears the #218 worst-case trigger 513.557 by
+2.367, and the smallest lambda where the worst-case speed clears is
`lambda_speed_clears=0.9675 < 0.9780`. I explicitly flagged -- and did NOT compute -- the cross:
"if a future gate lands the build below 0.9780, the speed margin should be re-scored THERE."

Issue #124 (human green-light, advisor-ruling RESOLVED) retired the PRIVATE validity bar 0.9780
as a SELF-BLOCKER: publish-first, organisers adjudicate validity POST-HOC; the official scorer
checks TPS>=500 / PPL<=2.42 / 128/128 only (no token-identity check). So the build no longer has
to clear 0.9780 to LAUNCH -- it has to clear the PUBLIC milestone. The operative launch gate is
therefore the PUBLIC clear-500 condition, and the binding acceptance is

    lambda_public_gate = max(lambda_speed_clears = 0.9675, lambda_floor_publish_first)

where `lambda_floor_publish_first` is kanna #228's publish-first lambda floor (DERIVED there; here
carried as a PARAMETER over the plausible band [0.9675, 0.9780], NOT blocked on).

THE MAP (imported VERBATIM from #229/#222; NOT re-derived)
---------------------------------------------------------
This leg imports #222's `binding_gate` module DIRECTLY (the same module #229 imports) and reuses
its ceiling-anchored reach-DP map, so the deliverable round-trips #229/#222 bit-exactly:

    mu_pub_speed(lambda) = K_cal*(E[T](lambda)/step)*tau_anchor = 520.953 * E[T](lambda)/E[T](1)

The #218 sigma band [7.6113, 7.7425, 8.2423] -> GO-triggers [512.519, 512.735, 513.557], the
520.953 int4-spec ceiling (#204), `lambda_speed_clears=0.9675` (#229), and the +2.367 validity-bar
margin (#229) are imported UNCHANGED. The order-stat is folded at N=1 (kanna #217 `vgovdrjc`:
`T(N)=T_base+sigma_sel*E[Z_(N:N)]`, `T(1)=T_base` -> NO winner's curse; `n_star_launch=1`;
best-of-N HARMFUL/flat) so the public margin is the single-shot N=1 margin against the #218 T_base.

THE DELIVERABLE
---------------
  public_go_margin(lambda)[corner] = mu_pub_speed(lambda) - go_trigger[corner]   (the curve)
  lambda_public_clears[corner] = smallest lambda where the margin crosses zero    (= #229 0.9675 wc)
  public_go_margin_at_floor    = worst-case-sigma margin at lambda=0.9675          (TEST; ~0 breakeven)
  which sub-gate binds lambda_public_gate over lambda_floor_publish_first in [0.9675, 0.9780]:
      SPEED sub-gate (0.9675) binds while lambda_floor <= 0.9675; else the FLOOR sub-gate binds.

The KEY FRAME: the PUBLIC speed trigger is the launch-relevant quantity under #124; the PRIVATE
0.9780 bar and the #226/#227 compliant-lane analytics are the POST-HOC-DEFENCE packet, NOT a
pre-launch gate.

SCOPE
-----
LOCAL CPU-ONLY analytic re-score of the PUBLIC GO margin across the publish-first lambda-floor
regime [0.9675, 0.9780], completing the cross #229 flagged. No GPU / vLLM / HF Job / submission /
served-file change / official draw. BASELINE stays 481.53; greedy/PPL untouched; adds 0 TPS
(PRIMARY = self-test). The mu_pub_speed map, sigma band, and triggers are imported unchanged.
Authorizes nothing. Supplies fern #231's GO-card the public-side margin under the publish-first
posture; consumes kanna #228's floor as a parameter. NOT a launch. NOT open2.

SELF-TEST (PR step 5 -- PRIMARY)
--------------------------------
(a) mu_pub_speed(1) round-trips the #204 ceiling 520.953 (err 0.0; both regimes, by anchor);
(b) at lambda=0.9780 the worst-case margin round-trips #229's +2.367;
(c) lambda_public_clears(worstcase-sigma) round-trips #229's 0.9675;
(d) mu_pub_speed MONOTONE INCREASING in lambda;
(e) public_go_margin MONOTONE DECREASING in sigma;
(f) public_go_margin(lambda_public_gate) >= 0  <=>  the P95 LCB clears 500 (the build clears public);
(g) NaN-clean across all reported scalars.
PRIMARY = publishfirst_public_margin_self_test_passes (bool);
TEST    = public_go_margin_at_floor (float TPS; worst-case-sigma margin at the speed-floor 0.9675).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import resource
import sys
import time
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
# publishfirst_public_margin -> validity -> research -> repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Source-of-truth artifacts (imported VERBATIM; one source per constant).
# ---------------------------------------------------------------------------
BINDING_GATE_PY = os.path.join(_ROOT, "research/validity/binding_gate/binding_gate.py")
SPEED_MARGIN_229 = os.path.join(
    _ROOT, "research/validity/speed_margin_at_validity_bar/speed_margin_at_validity_bar_results.json")
LAUNCH_TRIGGER_217 = os.path.join(
    _ROOT, "research/validity/launch_trigger/launch_trigger_calculator_results.json")

# The publish-first lambda-floor regime (kanna #228 derives the floor; we PARAMETERIZE over it).
SPEED_FLOOR_NOMINAL = 0.9675          # nominal speed floor (#229 lambda_speed_clears rounded)
VALIDITY_BAR_NOMINAL = 0.9780         # the (now-retired-as-self-blocker) private validity bar
FLOOR_BAND = (0.9675, 0.9780)         # the publish-first lambda_floor_publish_first band
# step-3 table: lambda_floor_publish_first sweep points.
FLOOR_GRID = (0.9675, 0.9700, 0.9750, 0.9780)

HEADLINE_REGIME = "both_bugs"         # the conservative regime (matches #229's TEST regime)
ROUNDTRIP_TOL = 1e-9                  # ceiling round-trip tolerance
RECONCILE_TOL = 1e-6                  # tolerance for reproducing #229's banked numbers
SIGMA_CORNERS = ("tight", "central", "worstcase")  # ascending sigma
TARGET = 500.0                        # the public clear-500 milestone


def _import(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# Import #222's binding_gate DIRECTLY -- reuse its ceiling-anchored reach-DP map so the
# deliverable round-trips #229/#222 bit-exactly (the strongest possible "import, do NOT re-derive").
BG = _import("binding_gate", BINDING_GATE_PY)


# ---------------------------------------------------------------------------
# Step 0 -- import the banked legs (NOT re-derived) + #229's verdict + #217's order-stat.
# ---------------------------------------------------------------------------
def import_context() -> dict[str, Any]:
    b = BG.import_banked()  # #218 sigma band + triggers + #204 ceiling + #208 bar + #213 spines
    r229 = _load(SPEED_MARGIN_229)
    r217 = _load(LAUNCH_TRIGGER_217)
    # kanna #217 order-stat (trigger_reconcile): N=1 is optimal, T(1)=T_base (no winner's curse).
    tr = r217["legs"]["trigger_reconcile"] if "legs" in r217 else _find_trigger_reconcile(r217)
    return {
        "banked": b,
        "pr229": {
            "lambda_speed_clears_worstcase": float(r229["lambda_speed_clears_worstcase"]),  # 0.96746847
            "speed_margin_at_validity_bar_worstcase": float(r229["speed_margin_at_validity_bar_worstcase"]),  # 2.3666
            "speed_margin_at_validity_bar_central": float(r229["speed_margin_at_validity_bar_central"]),     # 3.1887
            "speed_margin_at_validity_bar_tight": float(r229["speed_margin_at_validity_bar_tight"]),         # 3.4046
            "mu_pub_speed_at_validity_bar": float(r229["mu_pub_speed_at_validity_bar"]),                     # 515.924
            "lambda_speed_clears_descent_only": float(
                r229["per_regime"]["descent_only"]["lambda_speed_clears_worstcase"]),                        # 0.96740
            "speed_margin_worstcase_descent_only": float(
                r229["per_regime"]["descent_only"]["speed_margin_at_validity_bar_worstcase"]),               # 2.3778
        },
        "pr217": {
            "n_star_launch": int(tr["n_star_launch"]),                # 1
            "law": str(tr["law"]),                                    # T(N)=T_base+sigma_sel*E[Z_(N:N)]; T(1)=T_base
            "best_of_n_is_harmful": bool(tr["best_of_n_is_harmful"]),  # True
        },
    }


def _find_trigger_reconcile(r217: dict[str, Any]) -> dict[str, Any]:
    """Locate the #217 trigger_reconcile block regardless of nesting (defensive import)."""
    stack = [r217]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "n_star_launch" in node and "best_of_n_is_harmful" in node:
                return node
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    raise KeyError("could not locate the #217 trigger_reconcile order-stat block")


def _sigma_trigger_map(b: dict[str, Any]) -> dict[str, dict[str, float]]:
    """The #218 grounded (sigma, GO-trigger) at each corner; trigger = 500 + z1*sigma (= T(1) at N=1)."""
    return {
        "tight": {"sigma": b["combined_sigma_tight"], "trigger": b["trigger_tight"]},
        "central": {"sigma": b["combined_sigma_central"], "trigger": b["trigger_central"]},
        "worstcase": {"sigma": b["combined_sigma_worstcase"], "trigger": b["trigger_worstcase"]},
    }


# ---------------------------------------------------------------------------
# Step 1-2 -- the public GO margin curve over the publish-first floor regime.
# ---------------------------------------------------------------------------
def _margin_at(mp: "BG.LambdaMuMap", lam: float, sig_trig: dict[str, dict[str, float]],
               z1: float) -> dict[str, Any]:
    """public_go_margin(lambda)[corner] + the implied P95 LCB (margin = LCB - 500, algebraic)."""
    mu = mp.mu_pub(lam)
    row: dict[str, Any] = {"lambda": lam, "E_T": mp.et_of_lambda(lam), "mu_pub_speed": mu}
    for c in SIGMA_CORNERS:
        trig = sig_trig[c]["trigger"]
        sig = sig_trig[c]["sigma"]
        margin = mu - trig                  # = (mu - z1*sigma) - 500 = LCB - 500
        lcb = mu - z1 * sig                 # the one-sided P95 lower confidence bound
        row[c] = {
            "trigger": trig,
            "public_go_margin": margin,
            "lcb_p95": lcb,
            "lcb_minus_500": lcb - TARGET,
            "clears_500": bool(margin >= 0.0),
        }
    return row


def public_margin_curve(mp: "BG.LambdaMuMap", sig_trig: dict[str, dict[str, float]],
                        z1: float) -> dict[str, Any]:
    lo, hi = FLOOR_BAND
    # a fine grid across the floor regime + the table points + context points up to 1.0.
    n = 21
    grid_floor = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    context = [VALIDITY_BAR_NOMINAL, 0.99, 1.0]
    grid = sorted(set(round(x, 10) for x in (*grid_floor, *FLOOR_GRID, *context)))
    rows = [_margin_at(mp, lam, sig_trig, z1) for lam in grid]

    # lambda where the margin crosses zero, per corner (mu monotone increasing).
    lambda_public_clears = {}
    for c in SIGMA_CORNERS:
        lam_clear = mp.solve_lambda_for_mu(sig_trig[c]["trigger"])
        lam_clear_val = lam_clear if lam_clear is not None else float("nan")
        lambda_public_clears[c] = {
            "lambda_public_clears": lam_clear_val,
            "roundtrip_mu": (mp.mu_pub(lam_clear_val) if _finite(lam_clear_val) else float("nan")),
            "roundtrip_trigger": sig_trig[c]["trigger"],
        }
    return {"grid": grid, "rows": rows, "lambda_public_clears": lambda_public_clears}


# ---------------------------------------------------------------------------
# Step 3 -- which sub-gate binds lambda_public_gate + the order-stat (the deliverable table).
# ---------------------------------------------------------------------------
def binding_subgate_table(mp: "BG.LambdaMuMap", sig_trig: dict[str, dict[str, float]],
                          lambda_speed_clears: float, z1: float) -> dict[str, Any]:
    """For lambda_floor_publish_first across the band: which sub-gate binds lambda_public_gate, the
    resulting worst-case/central public margin, and clears-500 (N=1 single-shot, order-stat folded).

    The SPEED sub-gate binds while lambda_floor_publish_first <= lambda_speed_clears (=0.96747, the
    precise worst-case speed floor), pinning the gate there with margin 0 (breakeven). Above it the
    FLOOR sub-gate binds with strictly positive margin. The 4 PR-required points {0.9675, 0.9700,
    0.9750, 0.9780} are augmented with a below-floor point (0.9650) and the precise speed floor so
    BOTH regimes are visible; the PR-nominal 0.9675 sits a hair ABOVE the 0.96747 precise floor.
    """
    trig_wc = sig_trig["worstcase"]["trigger"]
    trig_c = sig_trig["central"]["trigger"]
    # below-floor demo (0.9650) + the precise speed floor + the 4 PR-required points.
    points = [
        (0.9650, False, "below-floor demo"),
        (lambda_speed_clears, False, "precise speed floor (breakeven)"),
        *[(f, True, "PR-required") for f in FLOOR_GRID],
    ]
    rows = []
    for floor, pr_required, label in points:
        # the publish-first launch gate is the MAX of the two sub-gates.
        gate = max(lambda_speed_clears, floor)
        # which sub-gate binds: speed while floor <= the precise speed floor; else the floor.
        binding = "speed" if floor <= lambda_speed_clears + ROUNDTRIP_TOL else "floor"
        mu_gate = mp.mu_pub(gate)
        margin_wc = mu_gate - trig_wc
        margin_c = mu_gate - trig_c
        rows.append({
            "lambda_floor_publish_first": floor,
            "pr_required": pr_required,
            "label": label,
            "lambda_public_gate": gate,
            "binding_subgate": binding,
            "mu_pub_speed_at_gate": mu_gate,
            "worstcase_public_margin": margin_wc,
            "central_public_margin": margin_c,
            "worstcase_lcb_p95": mu_gate - z1 * sig_trig["worstcase"]["sigma"],
            "clears_500": bool(margin_wc >= 0.0),
        })
    return {
        "rows": rows,
        "speed_subgate_binds_iff_floor_le": lambda_speed_clears,
        "order_stat_note": (
            "N=1 single-shot (kanna #217 n_star_launch=1; T(1)=T_base, NO winner's curse; "
            "best-of-N HARMFUL -- it LIFTS the seen public trigger for zero private gain). The "
            "#218 triggers ARE the T(1)=T_base triggers, so the public margin is the N=1 margin."
        ),
    }


# ---------------------------------------------------------------------------
# Step 2 headline -- public_go_margin_at_floor (TEST) at both endpoints + per-parameter.
# ---------------------------------------------------------------------------
def headline_margins(mp: "BG.LambdaMuMap", sig_trig: dict[str, dict[str, float]],
                     lambda_speed_clears: float, bar_precise: float) -> dict[str, Any]:
    trig_wc = sig_trig["worstcase"]["trigger"]
    trig_c = sig_trig["central"]["trigger"]
    trig_t = sig_trig["tight"]["trigger"]

    # the speed-floor endpoint: lambda = lambda_speed_clears (worst-case) -> margin == 0 by construction.
    mu_floor = mp.mu_pub(lambda_speed_clears)
    margin_at_floor_wc = mu_floor - trig_wc                       # ~0 (breakeven) -- the TEST number
    # the nominal speed floor 0.9675 (rounding cross-check; lambda_speed_clears ~ 0.96747).
    mu_floor_nom = mp.mu_pub(SPEED_FLOOR_NOMINAL)
    margin_at_floor_nom_wc = mu_floor_nom - trig_wc

    # the validity-bar endpoint: lambda = 0.9780 -> #229's +2.367 (worst-case).
    mu_bar = mp.mu_pub(bar_precise)
    margin_at_bar_wc = mu_bar - trig_wc

    return {
        "public_go_margin_at_floor": margin_at_floor_wc,             # TEST: worst-case-sigma @ 0.9675
        "public_go_margin_at_floor_nominal_0p9675": margin_at_floor_nom_wc,
        "public_go_margin_at_floor_central": mu_floor - trig_c,
        "public_go_margin_at_floor_tight": mu_floor - trig_t,
        "lambda_speed_clears_used": lambda_speed_clears,
        "mu_pub_speed_at_floor": mu_floor,
        "public_go_margin_at_validity_bar": margin_at_bar_wc,        # #229 +2.367 (worst-case)
        "public_go_margin_at_validity_bar_central": mu_bar - trig_c,
        "public_go_margin_at_validity_bar_tight": mu_bar - trig_t,
        "mu_pub_speed_at_validity_bar": mu_bar,
    }


def evaluate_regime(ctx: dict[str, Any], mp: "BG.LambdaMuMap") -> dict[str, Any]:
    b = ctx["banked"]
    z1 = b["z1_one_sided_p95"]
    bar_precise = b["validity_bar"]
    sig_trig = _sigma_trigger_map(b)
    # the worst-case speed floor: where the worst-case-sigma margin crosses zero (= #229's 0.9675).
    lam_speed_clears = mp.solve_lambda_for_mu(sig_trig["worstcase"]["trigger"])
    lam_speed_clears = lam_speed_clears if lam_speed_clears is not None else float("nan")

    curve = public_margin_curve(mp, sig_trig, z1)
    table = binding_subgate_table(mp, sig_trig, lam_speed_clears, z1)
    head = headline_margins(mp, sig_trig, lam_speed_clears, bar_precise)
    return {
        "lambda_speed_clears_worstcase": lam_speed_clears,
        "curve": curve,
        "subgate_table": table,
        "headline": head,
        "tau_anchor": mp.tau_anchor,
    }


# ---------------------------------------------------------------------------
# Step 3 reconcile -- round-trip #229 (pin lambda_speed_clears + the +2.367 margin).
# ---------------------------------------------------------------------------
def reconcile_229(ctx: dict[str, Any], per_regime: dict[str, Any]) -> dict[str, Any]:
    pr229 = ctx["pr229"]
    head = per_regime[HEADLINE_REGIME]
    lam_clear = per_regime[HEADLINE_REGIME]["lambda_speed_clears_worstcase"]
    margin_bar = head["headline"]["public_go_margin_at_validity_bar"]

    lam_clear_resid = abs(lam_clear - pr229["lambda_speed_clears_worstcase"])
    margin_bar_resid = abs(margin_bar - pr229["speed_margin_at_validity_bar_worstcase"])
    return {
        "pr229_lambda_speed_clears_worstcase": pr229["lambda_speed_clears_worstcase"],
        "recomputed_lambda_speed_clears_worstcase": lam_clear,
        "lambda_speed_clears_roundtrip_resid": lam_clear_resid,
        "roundtrips_229_lambda_speed_clears_0p9675": bool(lam_clear_resid < RECONCILE_TOL),
        "pr229_speed_margin_at_validity_bar_worstcase": pr229["speed_margin_at_validity_bar_worstcase"],
        "recomputed_public_margin_at_validity_bar_worstcase": margin_bar,
        "margin_at_validity_bar_roundtrip_resid": margin_bar_resid,
        "roundtrips_229_margin_2p367": bool(margin_bar_resid < RECONCILE_TOL),
        "note": (
            "the publish-first PUBLIC margin curve is the SAME mu_pub_speed map and the SAME #218 "
            "triggers as #229; the speed-floor endpoint IS #229's lambda_speed_clears=0.9675 (where "
            "the worst-case margin is 0 by construction) and the validity-bar endpoint IS #229's "
            "+2.367. This leg only RE-PARAMETERIZES #229 by lambda_floor_publish_first; no new map."
        ),
    }


# ---------------------------------------------------------------------------
# Step 5 -- self-test (PRIMARY).
# ---------------------------------------------------------------------------
def self_test(ctx: dict[str, Any], maps: dict[str, "BG.LambdaMuMap"],
              per_regime: dict[str, Any], recon: dict[str, Any]) -> dict[str, Any]:
    b = ctx["banked"]
    ceiling = b["lambda1_ceiling_mu"]
    floor = b["lambda_floor"]
    z1 = b["z1_one_sided_p95"]
    sig_trig = _sigma_trigger_map(b)

    # (a) mu_pub_speed(1) round-trips 520.953 EXACTLY (both regimes; anchoring construction).
    a_errs = {reg: abs(maps[reg].mu_pub(1.0) - ceiling) for reg in BG.REGIMES}
    a_ok = bool(all(e < ROUNDTRIP_TOL for e in a_errs.values()))

    # (b) at lambda=0.9780 the worst-case margin round-trips #229's +2.367.
    b_ok = bool(recon["roundtrips_229_margin_2p367"])

    # (c) lambda_public_clears(worstcase-sigma) round-trips #229's 0.9675.
    c_ok = bool(recon["roundtrips_229_lambda_speed_clears_0p9675"])

    # (d) mu_pub_speed MONOTONE INCREASING in lambda (fine grid over the floor regime), both regimes.
    lo, hi = FLOOR_BAND
    d_mono = {}
    for reg in BG.REGIMES:
        mp = maps[reg]
        grid = [lo + (1.0 - lo) * i / 80.0 for i in range(81)]  # floor regime up to the ceiling
        mus = [mp.mu_pub(x) for x in grid]
        diffs = [mus[i + 1] - mus[i] for i in range(len(mus) - 1)]
        d_mono[reg] = {
            "monotone_increasing": bool(all(dd >= -1e-12 for dd in diffs)),
            "strictly_increasing": bool(all(dd > 0.0 for dd in diffs)),
        }
    d_ok = bool(all(m["monotone_increasing"] for m in d_mono.values()))

    # (e) public_go_margin MONOTONE DECREASING in sigma (headline regime, at the validity bar + floor).
    mph = maps[HEADLINE_REGIME]
    e_detail = {}
    e_ok = True
    for lam_tag, lam in (("validity_bar", b["validity_bar"]), ("floor", SPEED_FLOOR_NOMINAL)):
        mu = mph.mu_pub(lam)
        sig_seq = [sig_trig[c]["sigma"] for c in SIGMA_CORNERS]            # ascending sigma
        margin_seq = [mu - sig_trig[c]["trigger"] for c in SIGMA_CORNERS]  # should descend
        sig_asc = all(sig_seq[i] <= sig_seq[i + 1] + 1e-15 for i in range(len(sig_seq) - 1))
        marg_desc = all(margin_seq[i] >= margin_seq[i + 1] - 1e-12 for i in range(len(margin_seq) - 1))
        e_detail[lam_tag] = {"sigma_ascending": sig_asc, "margin_descending": marg_desc,
                             "margin_seq": margin_seq}
        e_ok = e_ok and sig_asc and marg_desc
    e_ok = bool(e_ok)

    # (f) public_go_margin(lambda) >= 0  <=>  P95 LCB clears 500 (algebraic identity margin=LCB-500),
    #     tested at points BELOW the floor (margin<0), at the floor (~0), and above (margin>0).
    f_detail = []
    f_ok = True
    probe = [0.95, 0.96, SPEED_FLOOR_NOMINAL, 0.9700, b["validity_bar"], 0.99, 1.0]
    for lam in probe:
        mu = mph.mu_pub(lam)
        for c in SIGMA_CORNERS:
            trig = sig_trig[c]["trigger"]
            sig = sig_trig[c]["sigma"]
            margin = mu - trig
            lcb_clears = bool((mu - z1 * sig) >= TARGET)
            margin_clears = bool(margin >= 0.0)
            # the algebraic identity: margin = (mu - z1*sigma) - 500, so the booleans must agree.
            iff_holds = bool(margin_clears == lcb_clears
                             and abs(margin - ((mu - z1 * sig) - TARGET)) < 1e-9)
            f_detail.append({"lambda": lam, "corner": c, "margin": margin,
                             "lcb_clears_500": lcb_clears, "iff_holds": iff_holds})
            f_ok = f_ok and iff_holds
    # the iff must exercise BOTH branches (some clears, some not) to be meaningful.
    saw_clear = any(d["margin"] >= 0.0 for d in f_detail)
    saw_fail = any(d["margin"] < 0.0 for d in f_detail)
    f_ok = bool(f_ok and saw_clear and saw_fail)

    # (g) NaN-clean across all reported scalars.
    scalars: list[Any] = [ceiling, floor, z1, *a_errs.values(),
                          recon["lambda_speed_clears_roundtrip_resid"],
                          recon["margin_at_validity_bar_roundtrip_resid"]]
    for reg in BG.REGIMES:
        pr = per_regime[reg]
        h = pr["headline"]
        scalars += [pr["lambda_speed_clears_worstcase"], pr["tau_anchor"],
                    h["public_go_margin_at_floor"], h["public_go_margin_at_validity_bar"],
                    h["mu_pub_speed_at_floor"], h["mu_pub_speed_at_validity_bar"]]
        for r in pr["subgate_table"]["rows"]:
            scalars += [r["lambda_public_gate"], r["worstcase_public_margin"],
                        r["central_public_margin"], r["worstcase_lcb_p95"]]
        for c in SIGMA_CORNERS:
            scalars.append(pr["curve"]["lambda_public_clears"][c]["lambda_public_clears"])
    g_ok = bool(all(_finite(x) for x in scalars))

    checks = {
        "a_mu_pub_speed_lambda1_roundtrips_ceiling_520p953": a_ok,
        "b_margin_at_validity_bar_roundtrips_229_2p367": b_ok,
        "c_lambda_public_clears_worstcase_roundtrips_229_0p9675": c_ok,
        "d_mu_pub_speed_monotone_increasing_in_lambda": d_ok,
        "e_public_go_margin_monotone_decreasing_in_sigma": e_ok,
        "f_margin_nonneg_iff_lcb_clears_500": f_ok,
        "g_nan_clean": g_ok,
    }
    passes = bool(all(checks.values()))
    test_value = per_regime[HEADLINE_REGIME]["headline"]["public_go_margin_at_floor"]
    return {
        "publishfirst_public_margin_self_test_passes": passes,   # <-- PRIMARY
        "public_go_margin_at_floor": test_value,                 # <-- TEST
        "checks": checks,
        "evidence": {
            "a_ceiling_roundtrip_errs": a_errs,
            "b_margin_resid": recon["margin_at_validity_bar_roundtrip_resid"],
            "c_lambda_speed_clears_resid": recon["lambda_speed_clears_roundtrip_resid"],
            "d_monotonicity": d_mono,
            "e_sigma_margin": e_detail,
            "f_iff_rows": f_detail,
            "f_saw_clear_and_fail": bool(saw_clear and saw_fail),
            "n_scalars_checked": len(scalars),
        },
    }


# ---------------------------------------------------------------------------
# Assemble.
# ---------------------------------------------------------------------------
def _build_result(ctx, maps, per_regime, recon, st) -> dict[str, Any]:
    b = ctx["banked"]
    head = per_regime[HEADLINE_REGIME]
    desc = per_regime["descent_only"]
    h = head["headline"]
    table = head["subgate_table"]
    lam_clear = head["lambda_speed_clears_worstcase"]

    # the binding sub-gate over the floor band: speed binds while floor <= the precise speed floor.
    binding_over_band = (
        "the SPEED sub-gate binds for lambda_floor_publish_first <= lambda_speed_clears=%.5f "
        "(pinning lambda_public_gate at the speed floor, worst-case margin 0 -- the P95 LCB sits "
        "exactly at 500); for lambda_floor_publish_first > %.5f the FLOOR sub-gate binds with "
        "strictly positive margin. The PR-nominal 0.9675 sits a hair ABOVE the precise speed floor "
        "(margin +0.007 ~ breakeven)." % (lam_clear, lam_clear)
    )

    handoff = (
        "fern #231 + kanna #228: under #124 publish-first the operative launch gate is the PUBLIC "
        "speed trigger lambda_public_gate = max(lambda_speed_clears=0.9675, "
        "lambda_floor_publish_first); at the speed-floor endpoint 0.9675 the worst-case public "
        "margin is %+0.3f TPS (BREAKEVEN -- the P95 LCB sits exactly at 500) and at the validity "
        "bar 0.9780 it is %+0.3f TPS (#229), rising monotonically between. Over the floor band "
        "[0.9675, 0.9780] %s So fern reads the public-milestone margin at the #124 floor (the "
        "private 0.9780 bar + #226/#227 compliant analytics are POST-HOC defence, NOT a pre-launch "
        "gate). n_star_launch=1 (best-of-N HARMFUL) -> the margin is the single-shot N=1 public margin."
        % (h["public_go_margin_at_floor"], h["public_go_margin_at_validity_bar"], binding_over_band)
    )

    return {
        "pr": 234,
        "metric_primary": "publishfirst_public_margin_self_test_passes",
        "metric_test": "public_go_margin_at_floor",
        "publishfirst_public_margin_self_test_passes":
            st["publishfirst_public_margin_self_test_passes"],
        "public_go_margin_at_floor": st["public_go_margin_at_floor"],
        # the publish-first regime frame.
        "lambda_public_gate_law": "lambda_public_gate = max(lambda_speed_clears=0.9675, lambda_floor_publish_first)",
        "lambda_speed_clears_worstcase": lam_clear,
        "lambda_speed_clears_nominal": SPEED_FLOOR_NOMINAL,
        "lambda_floor_publish_first_band": list(FLOOR_BAND),
        "binding_subgate_over_floor_band": binding_over_band,
        # the headline margins at both endpoints (worst-case sigma) + the curve table.
        "public_go_margin_at_floor_0p9675": h["public_go_margin_at_floor"],
        "public_go_margin_at_floor_central": h["public_go_margin_at_floor_central"],
        "public_go_margin_at_floor_tight": h["public_go_margin_at_floor_tight"],
        "public_go_margin_at_validity_bar_0p9780": h["public_go_margin_at_validity_bar"],
        "public_go_margin_at_validity_bar_central": h["public_go_margin_at_validity_bar_central"],
        "mu_pub_speed_at_floor": h["mu_pub_speed_at_floor"],
        "mu_pub_speed_at_validity_bar": h["mu_pub_speed_at_validity_bar"],
        "subgate_table": table,
        "lambda_public_clears": head["curve"]["lambda_public_clears"],
        # robustness: both regimes.
        "public_go_margin_at_floor_descent_only": desc["headline"]["public_go_margin_at_floor"],
        "public_go_margin_at_validity_bar_descent_only":
            desc["headline"]["public_go_margin_at_validity_bar"],
        "lambda_speed_clears_descent_only": desc["lambda_speed_clears_worstcase"],
        "regime_robust": bool(
            abs(head["headline"]["public_go_margin_at_validity_bar"]
                - desc["headline"]["public_go_margin_at_validity_bar"]) < 0.1),
        # order-stat (kanna #217).
        "order_stat": ctx["pr217"],
        # reconcile #229.
        "reconcile_229": recon,
        # per-regime detail + curve.
        "per_regime": per_regime,
        "law": (
            "mu_pub_speed(lambda) = K_cal*(E[T](lambda)/step)*tau_anchor = 520.953 * "
            "E[T](lambda)/E[T](1) (imported #229/#222 map). public_go_margin(lambda)[corner] = "
            "mu_pub_speed(lambda) - GO_trigger[corner], GO_trigger = 500 + z1*combined_sigma[corner] "
            "= T(1) (kanna #217 N=1 order-stat, no winner's curse), z1=1.64485. Under #124 "
            "publish-first the launch gate is lambda_public_gate = max(lambda_speed_clears=0.9675, "
            "lambda_floor_publish_first); the build clears the public milestone iff "
            "public_go_margin(lambda_public_gate) >= 0."
        ),
        "imported": {
            "speed_floor_nominal": SPEED_FLOOR_NOMINAL,
            "validity_bar_nominal": VALIDITY_BAR_NOMINAL,
            "validity_bar_precise": b["validity_bar"],
            "lambda_floor": b["lambda_floor"],
            "trigger_tight": b["trigger_tight"],
            "trigger_central": b["trigger_central"],
            "trigger_worstcase": b["trigger_worstcase"],
            "combined_sigma_tight": b["combined_sigma_tight"],
            "combined_sigma_central": b["combined_sigma_central"],
            "combined_sigma_worstcase": b["combined_sigma_worstcase"],
            "lambda1_ceiling_mu": b["lambda1_ceiling_mu"],
            "z1_one_sided_p95": b["z1_one_sided_p95"],
            "K_cal": b["K_cal"],
            "step": b["step"],
            "tau_anchor_both_bugs": maps["both_bugs"].tau_anchor,
            "tau_anchor_descent_only": maps["descent_only"].tau_anchor,
            "pr229_lambda_speed_clears": ctx["pr229"]["lambda_speed_clears_worstcase"],
            "pr229_speed_margin_at_validity_bar_worstcase":
                ctx["pr229"]["speed_margin_at_validity_bar_worstcase"],
            "pr217_n_star_launch": ctx["pr217"]["n_star_launch"],
        },
        "honest_band": (
            "(a) the mu_pub_speed map, the #218 sigma band [7.6113,7.7425,8.2423], the "
            "[512.519,512.735,513.557] triggers, the 520.953 ceiling, #229's lambda_speed_clears="
            "0.9675 and +2.367, and #217's n_star=1 are ALL imported UNCHANGED -- this leg only "
            "RE-PARAMETERIZES #229 by lambda_floor_publish_first. (b) under #124 (advisor-ruling "
            "RESOLVED, no dissent) the PUBLIC milestone (TPS>=500/PPL<=2.42/128/128 by the official "
            "scorer) is the launch-relevant gate; the PRIVATE 0.9780 bar and the #226/#227 "
            "compliant-lane analytics are the POST-HOC-DEFENCE packet, NOT a pre-launch gate -- "
            "this is the load-bearing framing. (c) kanna #228 DERIVES lambda_floor_publish_first; "
            "we carry it as a PARAMETER over [0.9675, 0.9780] and do NOT block on it. (d) evaluating "
            "the speed map at the PRIVATE lambda UNDERSTATES true public mu_pub (public acceptance "
            ">= private) -> the margins are CONSERVATIVE; the true public margin is higher. (e) "
            "best-of-N is HARMFUL (kanna #217): N=1 is optimal, so the margin is the single-shot "
            "margin against the un-inflated T_base trigger."
        ),
        "handoff": handoff,
        "scope": (
            "LOCAL CPU-only analytic re-score of the PUBLIC GO margin across the publish-first "
            "lambda-floor regime [0.9675, 0.9780], completing the cross #229 flagged. Imports "
            "#229/#222's mu_pub_speed map DIRECTLY (round-trips its 0.9675 and +2.367 bit-exactly) "
            "and folds kanna #217's N=1 order-stat. No GPU/vLLM/HF Job/submission/served-file "
            "change/official draw. BASELINE stays 481.53; adds 0 TPS (PRIMARY = self-test); "
            "greedy/PPL untouched. Authorizes nothing. NOT a launch. NOT open2."
        ),
        "public_evidence_used": [
            "ubel #229 (bz2b3fw8, speed_margin_at_validity_bar, MERGED) -- the cross this leg "
            "completes: VALIDITY_BINDS_SPEED_ALWAYS_CLEARS, lambda_speed_clears=0.9675 < 0.9780, "
            "speed_margin_at_validity_bar_worstcase=+2.367; its mu_pub_speed map (via #222's "
            "binding_gate) is imported DIRECTLY and re-parameterized by lambda_floor_publish_first.",
            "ubel #222 (yw7i2ece, binding_gate, MERGED) -- the ceiling-anchored E[T](lambda)->mu_pub "
            "reach-DP map imported directly (round-trips the 520.953 ceiling and 515.924 bar).",
            "ubel #218 (0ug7vd7d, interleg_rho, MERGED) -- the grounded combined-sigma band "
            "[7.6113,7.7425,8.2423] and GO-trigger band [512.519,512.735,513.557] the margin is "
            "scored against; imported unchanged (= T(1) at N=1).",
            "ubel #204 (launch_sigma_unit_rebase, MERGED) -- the int4-spec lambda=1 ceiling "
            "mu_pub(1)=520.953 the map's top endpoint is anchored to (round-trip self-test).",
            "kanna #217 (vgovdrjc, trigger_reconcile, MERGED) -- the order-stat T(N)=T_base+"
            "sigma_sel*E[Z_(N:N)] with n_star_launch=1 (T(1)=T_base, no winner's curse; best-of-N "
            "HARMFUL): the public margin is the single-shot N=1 margin, folded here.",
            "human Issue #124 (publish-first green-light, advisor-ruling RESOLVED, no dissent) -- "
            "retires the PRIVATE validity bar 0.9780 as a self-blocker; the official scorer checks "
            "TPS>=500/PPL<=2.42/128/128 only -> the PUBLIC milestone is the operative launch gate.",
        ],
        "method": (
            "LOCAL CPU-only analytic synthesis: imports #229/#222's mu_pub_speed map directly and "
            "re-scores the PUBLIC GO margin as a curve over lambda in [0.9675, 0.9780] at the three "
            "#218 sigma corners, identifies which sub-gate binds lambda_public_gate = max(0.9675, "
            "lambda_floor_publish_first), and folds kanna #217's N=1 order-stat. Round-trips #229's "
            "0.9675 and +2.367 bit-exactly. No GPU/vLLM/HF Job/submission/served-file change. "
            "BASELINE stays 481.53; adds 0 TPS. Greedy/PPL untouched. NOT a launch."
        ),
        "metrics_nan_clean": 1 if st["checks"]["g_nan_clean"] else 0,
        "self_test": st,
    }


# ---------------------------------------------------------------------------
# W&B logging (mirrors #229; never fatal).
# ---------------------------------------------------------------------------
def _log_wandb(args, result: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[publishfirst-margin] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="launch-sigma-publishfirst-public-margin", agent="ubel",
            name=args.wandb_name or "ubel/publishfirst-public-margin",
            group=args.wandb_group,
            tags=["launch-sigma", "publishfirst-public-margin", "public-go-margin", "speed-trigger",
                  "publish-first", "et-lambda-map", "winners-curse-redraw-budget", "pr234"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic",
                    "z_one_sided_p95": result["imported"]["z1_one_sided_p95"],
                    "speed_floor_nominal": SPEED_FLOOR_NOMINAL,
                    "validity_bar_nominal": VALIDITY_BAR_NOMINAL,
                    "lambda_floor_publish_first_band": list(FLOOR_BAND),
                    "trigger_worstcase": result["imported"]["trigger_worstcase"],
                    "trigger_central": result["imported"]["trigger_central"],
                    "trigger_tight": result["imported"]["trigger_tight"],
                    "lambda1_ceiling_mu": result["imported"]["lambda1_ceiling_mu"],
                    "headline_regime": HEADLINE_REGIME, "n_star_launch": result["order_stat"]["n_star_launch"],
                    "K_cal": result["imported"]["K_cal"], "step": result["imported"]["step"]},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[publishfirst-margin] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[publishfirst-margin] wandb disabled; skipping", flush=True)
        return
    try:
        st = result["self_test"]
        flat = {
            # PRIMARY + TEST
            "publishfirst_public_margin_self_test_passes":
                1.0 if st["publishfirst_public_margin_self_test_passes"] else 0.0,
            "public_go_margin_at_floor": result["public_go_margin_at_floor"],
            # headline margins at both endpoints (worst-case sigma) + central.
            "public_go_margin_at_floor_0p9675": result["public_go_margin_at_floor_0p9675"],
            "public_go_margin_at_floor_central": result["public_go_margin_at_floor_central"],
            "public_go_margin_at_floor_tight": result["public_go_margin_at_floor_tight"],
            "public_go_margin_at_validity_bar_0p9780": result["public_go_margin_at_validity_bar_0p9780"],
            "public_go_margin_at_validity_bar_central": result["public_go_margin_at_validity_bar_central"],
            "mu_pub_speed_at_floor": result["mu_pub_speed_at_floor"],
            "mu_pub_speed_at_validity_bar": result["mu_pub_speed_at_validity_bar"],
            # the publish-first regime frame.
            "lambda_speed_clears_worstcase": result["lambda_speed_clears_worstcase"],
            "lambda_speed_clears_nominal": result["lambda_speed_clears_nominal"],
            # robustness: descent-only.
            "public_go_margin_at_floor_descent_only": result["public_go_margin_at_floor_descent_only"],
            "public_go_margin_at_validity_bar_descent_only":
                result["public_go_margin_at_validity_bar_descent_only"],
            "regime_robust": 1.0 if result["regime_robust"] else 0.0,
            # order-stat (kanna #217).
            "n_star_launch": float(result["order_stat"]["n_star_launch"]),
            "best_of_n_is_harmful": 1.0 if result["order_stat"]["best_of_n_is_harmful"] else 0.0,
            # reconcile #229.
            "roundtrips_229_margin_2p367":
                1.0 if result["reconcile_229"]["roundtrips_229_margin_2p367"] else 0.0,
            "roundtrips_229_lambda_speed_clears_0p9675":
                1.0 if result["reconcile_229"]["roundtrips_229_lambda_speed_clears_0p9675"] else 0.0,
            # imported gates.
            "trigger_worstcase": result["imported"]["trigger_worstcase"],
            "trigger_central": result["imported"]["trigger_central"],
            "trigger_tight": result["imported"]["trigger_tight"],
            "lambda1_ceiling_mu": result["imported"]["lambda1_ceiling_mu"],
            "tau_anchor_both_bugs": result["imported"]["tau_anchor_both_bugs"],
            # per-check booleans.
            "self_test_a_ceiling_roundtrip":
                1.0 if st["checks"]["a_mu_pub_speed_lambda1_roundtrips_ceiling_520p953"] else 0.0,
            "self_test_b_margin_roundtrips_229":
                1.0 if st["checks"]["b_margin_at_validity_bar_roundtrips_229_2p367"] else 0.0,
            "self_test_c_lambda_clears_roundtrips_229":
                1.0 if st["checks"]["c_lambda_public_clears_worstcase_roundtrips_229_0p9675"] else 0.0,
            "self_test_d_monotone_in_lambda":
                1.0 if st["checks"]["d_mu_pub_speed_monotone_increasing_in_lambda"] else 0.0,
            "self_test_e_margin_monotone_in_sigma":
                1.0 if st["checks"]["e_public_go_margin_monotone_decreasing_in_sigma"] else 0.0,
            "self_test_f_margin_iff_lcb":
                1.0 if st["checks"]["f_margin_nonneg_iff_lcb_clears_500"] else 0.0,
            "self_test_g_nan_clean": 1.0 if st["checks"]["g_nan_clean"] else 0.0,
        }
        # per-floor-parameter table rows (headline regime).
        for r in result["subgate_table"]["rows"]:
            key = f"floor_{str(r['lambda_floor_publish_first']).replace('.', 'p')}"
            flat[f"{key}_worstcase_margin"] = r["worstcase_public_margin"]
            flat[f"{key}_central_margin"] = r["central_public_margin"]
            flat[f"{key}_binding_is_speed"] = 1.0 if r["binding_subgate"] == "speed" else 0.0
            flat[f"{key}_clears_500"] = 1.0 if r["clears_500"] else 0.0
        # per-sigma-corner lambda_public_clears (headline regime).
        for c in SIGMA_CORNERS:
            flat[f"lambda_public_clears_{c}"] = \
                result["lambda_public_clears"][c]["lambda_public_clears"]
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="publishfirst_public_margin",
            artifact_type="launch-sigma-publishfirst-public-margin", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[publishfirst-margin] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    st = result["self_test"]
    head = result["per_regime"][HEADLINE_REGIME]
    print("\n[publishfirst-margin] ===== PUBLISH-FIRST PUBLIC MARGIN at the #124 floor (PR #234) =====",
          flush=True)
    print(f"  map: mu_pub_speed(lambda) = 520.953*E[T](lambda)/E[T](1)  (imported #229/#222)", flush=True)
    print(f"  gate: lambda_public_gate = max(lambda_speed_clears={result['lambda_speed_clears_worstcase']:.4f}, "
          f"lambda_floor_publish_first)   [#124 publish-first: PUBLIC milestone is the launch gate]", flush=True)
    print(f"  order-stat (kanna #217): n_star_launch={result['order_stat']['n_star_launch']}  "
          f"best_of_n_harmful={result['order_stat']['best_of_n_is_harmful']}  -> N=1 single-shot margin",
          flush=True)
    print("  public_go_margin curve at endpoints (both_bugs, worst-case sigma):", flush=True)
    print(f"    floor   lambda={result['lambda_speed_clears_worstcase']:.4f}  "
          f"mu_pub={result['mu_pub_speed_at_floor']:8.3f}  "
          f"margin={result['public_go_margin_at_floor_0p9675']:+.4f}  <- TEST (breakeven)", flush=True)
    print(f"    bar     lambda={VALIDITY_BAR_NOMINAL:.4f}  mu_pub={result['mu_pub_speed_at_validity_bar']:8.3f}  "
          f"margin={result['public_go_margin_at_validity_bar_0p9780']:+.4f}  (#229)", flush=True)
    print("  which sub-gate binds (lambda_floor_publish_first sweep, both_bugs; * = PR-required):",
          flush=True)
    for r in result["subgate_table"]["rows"]:
        star = "*" if r["pr_required"] else " "
        print(f"   {star}floor={r['lambda_floor_publish_first']:.5f}  gate={r['lambda_public_gate']:.5f}  "
              f"binds={r['binding_subgate']:5s}  wc_margin={r['worstcase_public_margin']:+.4f}  "
              f"c_margin={r['central_public_margin']:+.4f}  clears500={r['clears_500']}  "
              f"[{r['label']}]", flush=True)
    print(f"  binding over band: {result['binding_subgate_over_floor_band']}", flush=True)
    print(f"  regime-robust (both_bugs vs descent_only @ bar): {result['regime_robust']}  "
          f"(descent_only floor margin={result['public_go_margin_at_floor_descent_only']:+.4f}, "
          f"bar margin={result['public_go_margin_at_validity_bar_descent_only']:+.4f})", flush=True)
    print(f"\n  SELF-TEST (PRIMARY) passes = {st['publishfirst_public_margin_self_test_passes']}  "
          f"public_go_margin_at_floor (TEST) = {st['public_go_margin_at_floor']:+.4f} TPS", flush=True)
    for k, v in st["checks"].items():
        print(f"    [{'ok' if v else 'XX'}] {k}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Publish-first public margin at the #124 floor (PR #234)")
    ap.add_argument("--out", default=os.path.join(_HERE, "publishfirst_public_margin_results.json"))
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name",
                    default="ubel/publishfirst-public-margin")
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="winners-curse-redraw-budget")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--self-test", action="store_true", help="exit non-zero if the self-test fails")
    args = ap.parse_args(argv)

    t0 = time.time()
    ctx = import_context()
    maps = BG.build_maps(ctx["banked"])
    per_regime = {reg: evaluate_regime(ctx, maps[reg]) for reg in BG.REGIMES}
    recon = reconcile_229(ctx, per_regime)
    st = self_test(ctx, maps, per_regime, recon)

    result = _build_result(ctx, maps, per_regime, recon, st)
    result["elapsed_s"] = round(time.time() - t0, 4)
    result["peak_mem_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 3)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    _print(result)
    print(f"\n[publishfirst-margin] HANDOFF: {result['handoff']}", flush=True)
    print(f"[publishfirst-margin] artifacts -> {args.out}", flush=True)
    _log_wandb(args, result)

    if args.self_test:
        ok = st["publishfirst_public_margin_self_test_passes"] and result["metrics_nan_clean"] == 1
        print(f"[publishfirst-margin] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
