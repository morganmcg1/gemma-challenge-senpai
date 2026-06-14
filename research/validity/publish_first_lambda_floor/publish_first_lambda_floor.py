#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Publish-first lambda-floor: the built-lambda where the private MEAN crosses 500 (PR #228).

WHAT THIS IS
------------
The POINT-estimate companion to kanna #224's P95 private-bar finding. #224 (`1081oc84`,
MERGED) proved 500-PRIVATE is NOT reachable at the physical ceiling at P95 (the public mean
needed, 535.139, exceeds the int4-spec lambda=1 ceiling 520.953 by ~14.2 TPS). But the human
green-lit PUBLISH-FIRST (#124): launch on the POINT estimate, accept the single-draw risk. So
the launch-relevant threshold is NOT the 0.9780 P95 build bar (#191) but the LOWER built-lambda
at which the private MEAN first crosses 500. This leg pins that number.

THE MECHANISM (imported, NOT re-derived)
----------------------------------------
The private point estimate is the public point estimate times the public->private multiplier:

    private_mean(lambda) = K_cal * (E[T](lambda) / step_int4) * tau * f_priv

with E[T](lambda) the banked self-KV reach-DP (#175/#184 via the #183 lambda-acceptance card,
the SAME object fern #185's launch_trigger reads), and the launch composition pinned:
K_cal=125.268 (#148/#169), step_int4=1.2182 (#168), tau in [0.9924, 1.0] (#181), f_priv=0.969107
(#217). The only lambda-dependent factor is E[T](lambda); K_cal/step/tau/f_priv are constants.

THE PUBLIC-NUMBER CONVENTION (the load-bearing choice -- read carefully)
-----------------------------------------------------------------------
"public point estimate" has TWO readings in this lane, and the publish-first floor depends on
which one the human's gate uses:

  CENTRAL  (point estimate)     public_central(1) = K_cal*E[T](1)/step*1.0 = 535.433  (#191)
  LCB-CEIL (P95-conservative)   public_lcb(1)     = 520.953                            (#204/#217)

The human chose PUBLISH-FIRST = the LESS conservative gate (launch on the point estimate, accept
the draw risk). The point estimate is the CENTRAL public number, so the publish-first gate is

    private_mean_central(lambda) = public_central(lambda) * f_priv      (HEADLINE)

which round-trips #191's banked private_central_lambda1 = 535.433 * 0.969107 = 518.892 at
lambda=1, and crosses 500 at

    lambda_floor_publish_first = lambda where private_mean_central = 500

This EQUALS #191's already-banked `lambda_star_central_private` = 0.9138270633254315 (cross-checked
to ~1e-9 below). The 0.9780 P95 bar (#191 `lambda_star_lcb_private`) is the SAME composition read at
the public LCB; the gap between them, ~0.064 in lambda, is precisely the public-sampling confidence
haircut -- "the draw risk the human accepted in #124".

THE PR'S ROUND-TRIP NOTE (504.86) IS THE CONSERVATIVE READING (reported, NOT the headline)
------------------------------------------------------------------------------------------
The PR's round-trip note `private_mean(1) = 520.953 * 0.969107 ~= 504.86` uses the public-LCB
CEILING (520.953), i.e. the P95-conservative public number -- a STRICTER input than the
point-estimate gate the human actually chose. Under that reading the floor is ~0.9778, which
COLLIDES with the 0.9780 P95 bar (gap ~0.0002): 520.953*f_priv = 504.86 is numerically #191's
`private_lcb_lambda1` (the public-LCB-discounted private projection), so its "mean" floor sits on
top of the P95 bar by construction. We carry this conservative reading as an explicit reconciliation
row (and self-test it), but the publish-first POINT-estimate floor the human's #124 gate requires is
the CENTRAL one, 0.9138. See `convention_reconciliation`.

LOCAL CPU-only analytic inversion of the banked private-mean curve. No GPU / vLLM / HF Job /
submission / served-file change / official draw. BASELINE stays 481.53; greedy/PPL untouched;
adds 0 TPS; authorizes nothing. Imports #224/#217/#204/#191 + the #183/#175/#184 reach-DP VERBATIM;
does NOT re-derive E[T](lambda), K_cal, step, tau, f_priv, the ceiling, or the 0.9780 bar. The
POINT-estimate companion; the P95-private bar STAYS the 0.9780 (we report both). NOT a launch.

PRIMARY metric  publish_first_lambda_floor_self_test_passes
TEST    metric  lambda_floor_publish_first   (== #191 lambda_star_central_private ~= 0.9138)
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
from typing import Any, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # -> repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# source-of-truth artifacts (imported verbatim; one source per constant)
# ---------------------------------------------------------------------------
TRIGGER_RECONCILE_217 = os.path.join(
    _ROOT, "research/validity/trigger_reconcile/trigger_reconcile_results.json")
PRIVATE_BUILD_BAR_191 = os.path.join(
    _ROOT, "research/validity/private_build_bar/results.json")
PRIVATE_BAR_REACH_224 = os.path.join(
    _ROOT, "research/validity/private_bar_reachability/private_bar_reachability_results.json")

TARGET = 500.0
Z1 = 1.6448536269514722  # one-sided P95 (#204)
TAU_CENTRAL = 1.0
TAU_CONS = 0.9924318649123313  # #181 tau_low corner
TOPO = "both_bugs"             # the deployed stack + the binding 0.9780 bar are both-bugs (#191)

# land #71's achievable-lambda band (PR step 1)
LAMBDA_BAND = [0.8572, 0.95, 0.9780112973731208, 0.997, 1.0]

TOL = 1e-6
TOL_XCHK = 1e-7   # cross-check vs #191's banked lambda_star_central_private
EXACT_TOL = 1e-9


# ---------------------------------------------------------------------------
# import the banked E[T](lambda) reach-DP (#175/#184 via the #183 card)
# ---------------------------------------------------------------------------
def _load_mod(name: str, relpath: str):
    path = os.path.join(_ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_et_curve() -> dict[str, Any]:
    """The #183 lambda-acceptance card -> E[T](lambda) reach-DP. Same object fern #185 reads.

    E[T] is tau-independent (the reach-DP pmf-mean of the constant-lambda acceptance spine); tau
    enters only the K_cal*E[T]/step*tau TPS composition, not E[T] itself.
    """
    v179 = _load_mod("launch_packet_refresh", "scripts/profiler/launch_packet_refresh.py")
    v178 = _load_mod("realistic_selfkv_floor",
                     "research/oracle_readout/realistic_selfkv_floor/realistic_selfkv_floor.py")
    v183 = _load_mod("lambda_acceptance_card",
                     "research/oracle_readout/lambda_acceptance_card/lambda_acceptance_card.py")
    d172 = v178.D172
    anchors = d172.load_anchors(
        d172.DEFAULT_BUG2_ANCHOR, d172.DEFAULT_TOPO_JSON, d172.DEFAULT_ACCEPT_JSON,
        d172.DEFAULT_RANKCOV_JSON, d172.DEFAULT_DECOMP_JSON)
    ctx = v183.build_topologies(anchors)
    topo = ctx["topo"][TOPO]
    qf, qF = topo["q_floor"], topo["q_full"]

    def et_of_lambda(lam: float) -> float:
        return float(v183.metrics_at(ctx, lam, qf, qF, TAU_CENTRAL)["E_T"])

    return {
        "et_of_lambda": et_of_lambda,
        "k_cal": float(v179.K_CAL),       # 125.26795005202914 (#148/#169)
        "step_int4": float(v179.STEP_BASE),  # 1.2182 (#168)
    }


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _finite(x: Any) -> float:
    if x is None or not math.isfinite(float(x)):
        raise ValueError(f"non-finite value: {x!r}")
    return float(x)


def import_banked() -> dict[str, Any]:
    d217 = _load(TRIGGER_RECONCILE_217)["import_banked"]
    b191 = _load(PRIVATE_BUILD_BAR_191)["synthesis"]["per_topology"]["both_bugs"]
    c191 = _load(PRIVATE_BUILD_BAR_191)["synthesis"]["constants"]
    d224 = _load(PRIVATE_BAR_REACH_224)

    out = {
        # ---- #204/#217 launch composition + ceiling ----
        "f_priv": _finite(d217["f_priv"]),                 # 0.969106920637722 (#217 assumed)
        "ceiling_lcb": _finite(d217["lambda1_ceiling"]),   # 520.9527323111674 (public LCB @ lambda=1)
        "sigma_draw": _finite(d217["sigma_draw"]),         # 7.390974474817942 (fresh private draw)
        "mu_safe_fresh": _finite(d217["mu_safe_fresh"]),   # 512.15707117161 = 500 + z1*sigma_draw
        # ---- #191 both-bugs composition (the canonical public-central + the two private bars) ----
        "public_central_lambda1": _finite(b191["public_central_lambda1"]),   # 535.4330096522525
        "public_lcb_lambda1": _finite(b191["public_lcb_lambda1"]),           # 520.9527323111674
        "private_central_lambda1": _finite(b191["private_central_lambda1_taulow"]),  # 518.891835
        "private_lcb_lambda1": _finite(b191["private_lcb_lambda1_taulow"]),  # 504.858898
        "lambda_star_central_private_191": _finite(b191["lambda_star_central_private"]),  # 0.9138270633
        "lambda_star_lcb_private_191": _finite(b191["lambda_star_lcb_private"]),   # 0.9780112973731208
        "public_bar_both_bugs_191": _finite(c191["public_bar_both_bugs"]),   # 0.9052283680740145
        # ---- #224 grounding (the f_priv sensitivity anchor) ----
        "f_priv_grounded": _finite(d224["f_priv_grounded_point"]),   # 0.9570535584491102
        "mu_ceiling_needed_224": _finite(d224["mu_ceiling_needed"]), # 535.1394043208535 (P95 target)
        "private_500_reachable_at_ceiling_p95_224":
            bool(d224["private_500_reachable_at_physical_ceiling"]),  # False
    }
    # provenance self-checks (not gates): the two #191 anchors == K_cal*E[T](1)/step * {ceil_lcb} * f_priv
    out["ceiling_consistent_217_vs_191"] = abs(out["ceiling_lcb"] - out["public_lcb_lambda1"])
    return out


# ---------------------------------------------------------------------------
# the private-mean curves (CENTRAL = headline; CONSERVATIVE-ceiling = reconciliation)
# ---------------------------------------------------------------------------
def make_curves(curve: dict[str, Any], imp: dict[str, Any]):
    et_of_lambda = curve["et_of_lambda"]
    k_cal, step = curve["k_cal"], curve["step_int4"]
    et1 = et_of_lambda(1.0)
    ceil_lcb = imp["ceiling_lcb"]

    def public_central(lam: float, tau: float = TAU_CENTRAL) -> float:
        return k_cal * et_of_lambda(lam) / step * tau

    def private_mean_central(lam: float, f_priv: float, tau: float = TAU_CENTRAL) -> float:
        """The HEADLINE publish-first point estimate: public CENTRAL * f_priv."""
        return public_central(lam, tau) * f_priv

    def private_mean_conservative(lam: float, f_priv: float) -> float:
        """The PR round-trip reading: public-LCB CEILING * f_priv, scaled by the E[T] shape."""
        return ceil_lcb * (et_of_lambda(lam) / et1) * f_priv

    return private_mean_central, private_mean_conservative, public_central, et1


def solve_floor(mean_fn: Callable[[float], float], lo: float = 0.5, hi: float = 1.0,
                target: float = TARGET) -> float | None:
    """Smallest lambda in [lo,hi] with mean_fn(lambda)=target, by monotone bisection."""
    f_lo = mean_fn(lo) - target
    f_hi = mean_fn(hi) - target
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if (f_lo > 0.0) == (f_hi > 0.0):
        return None  # not bracketed (target below floor or above ceiling on [lo,hi])
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (mean_fn(mid) - target > 0.0) == (f_lo > 0.0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# step 1: the publish-first private-mean curve over land #71's band
# ---------------------------------------------------------------------------
def mean_curve(mean_central, mean_cons, et_of_lambda, imp: dict[str, Any]) -> dict[str, Any]:
    f_priv = imp["f_priv"]
    rows = []
    for lam in LAMBDA_BAND:
        mc = mean_central(lam, f_priv)
        mk = mean_cons(lam, f_priv)
        rows.append({
            "lambda": lam,
            "E_T": et_of_lambda(lam),
            "private_mean_central": mc,
            "private_mean_central_clears_500": bool(mc >= TARGET),
            "private_mean_conservative": mk,
            "private_mean_conservative_clears_500": bool(mk >= TARGET),
        })
    return {
        "topology": TOPO,
        "rows": rows,
        "anchor_central_lambda1": rows[-1]["private_mean_central"],       # 518.892
        "anchor_conservative_lambda1": rows[-1]["private_mean_conservative"],  # 504.859
        "anchor_central_roundtrips_191":
            abs(rows[-1]["private_mean_central"] - imp["private_central_lambda1"]),
        "anchor_conservative_roundtrips_191":
            abs(rows[-1]["private_mean_conservative"] - imp["private_lcb_lambda1"]),
    }


# ---------------------------------------------------------------------------
# step 2: the publish-first lambda-floor (the deliverable)
# ---------------------------------------------------------------------------
def lambda_floor(mean_central, mean_cons, imp: dict[str, Any]) -> dict[str, Any]:
    f_priv = imp["f_priv"]
    p95_bar = imp["lambda_star_lcb_private_191"]   # 0.9780112973731208

    floor_central = solve_floor(lambda lam: mean_central(lam, f_priv))
    floor_cons = solve_floor(lambda lam: mean_cons(lam, f_priv))

    gap_central = p95_bar - floor_central if floor_central is not None else None
    gap_cons = p95_bar - floor_cons if floor_cons is not None else None

    # cross-check the headline floor against #191's already-banked central bar
    xchk_191 = abs(floor_central - imp["lambda_star_central_private_191"])

    return {
        # HEADLINE deliverable (the human's publish-first POINT-estimate gate)
        "lambda_floor_publish_first": floor_central,                  # ~0.9138
        "lambda_gap_pe_vs_p95": gap_central,                          # ~0.0642
        "built_lambda_for_publish_first_go": floor_central,
        # cross-check: equals #191's lambda_star_central_private
        "lambda_floor_central_xcheck_191_resid": xchk_191,            # ~1e-9
        "lambda_star_central_private_191": imp["lambda_star_central_private_191"],
        # the conservative-ceiling reading (the PR's 504.86 round-trip anchor)
        "lambda_floor_conservative_ceiling": floor_cons,             # ~0.9778
        "lambda_gap_conservative_vs_p95": gap_cons,                  # ~0.0002 (collapses)
        # the P95 bar (imported, unchanged)
        "p95_private_bar": p95_bar,                                   # 0.9780112973731208
        "public_bar": imp["public_bar_both_bugs_191"],               # 0.9052283680740145
    }


# ---------------------------------------------------------------------------
# step 3: reconcile the two gates -> the three built-lambda regimes
# ---------------------------------------------------------------------------
def gate_regimes(floor: dict[str, Any], mean_central, imp: dict[str, Any]) -> dict[str, Any]:
    f_priv = imp["f_priv"]
    lf = floor["lambda_floor_publish_first"]
    p95 = floor["p95_private_bar"]

    def regime(lam: float) -> str:
        if lam >= p95:
            return "both_clear"            # publish-first GO + P95-private valid
        if lam >= lf:
            return "publish_first_go_p95_hold"  # mean>=500, P95 HOLD (the accepted draw risk)
        return "publish_first_no_go"        # mean<500

    table = []
    for lam in sorted(set(LAMBDA_BAND + [lf, p95])):
        table.append({
            "lambda": lam,
            "private_mean_central": mean_central(lam, f_priv),
            "regime": regime(lam),
        })
    return {
        "gate_regime_vs_lambda": table,
        "regime_defs": {
            "both_clear": "lambda >= 0.9780 -> publish-first GO AND P95-private valid",
            "publish_first_go_p95_hold":
                f"{lf:.4f} <= lambda < 0.9780 -> publish-first GO (mean>=500), P95-private HOLD "
                "(the single-draw risk the human accepted in #124)",
            "publish_first_no_go": f"lambda < {lf:.4f} -> publish-first NO-GO (mean<500)",
        },
        "built_lambda_for_publish_first_go": lf,
    }


# ---------------------------------------------------------------------------
# step 4: honest band -- f_priv sensitivity (dlambda_floor/df_priv)
# ---------------------------------------------------------------------------
def sensitivity(mean_central, mean_cons, public_central, imp: dict[str, Any]) -> dict[str, Any]:
    f0 = imp["f_priv"]                 # 0.969107 (assumed)
    fg = imp["f_priv_grounded"]        # 0.957054 (#224 grounded point)

    def floor_at(f_priv: float) -> float | None:
        return solve_floor(lambda lam: mean_central(lam, f_priv))

    floor_f0 = floor_at(f0)
    floor_fg = floor_at(fg)
    floor_f1 = floor_at(1.0)

    # central numeric derivative dlambda_floor/df_priv at the assumed f_priv
    h = 1e-4
    dlam_df = (floor_at(f0 + h) - floor_at(f0 - h)) / (2.0 * h)

    # does the publish-first MEAN still clear at lambda=1 under each reading x each f_priv?
    central_mean_1_f0 = public_central(1.0) * f0
    central_mean_1_fg = public_central(1.0) * fg
    cons_mean_1_f0 = mean_cons(1.0, f0)
    cons_mean_1_fg = mean_cons(1.0, fg)

    return {
        "f_priv_assumed": f0,
        "f_priv_grounded_224": fg,
        "lambda_floor_at_f_priv_assumed": floor_f0,       # ~0.9138
        "lambda_floor_at_f_priv_grounded": floor_fg,      # rises (~0.945)
        "lambda_floor_at_f_priv_one": floor_f1,           # drops (sensitivity sane)
        "dlambda_floor_df_priv": dlam_df,                 # NEGATIVE: lower f_priv => higher floor
        "floor_rises_under_grounding": bool(floor_fg > floor_f0),
        "floor_drops_at_fpriv_one": bool(floor_f1 < floor_f0),
        # central reading: publish-first mean clears at lambda=1 even under grounded f_priv
        "central_mean_at_lambda1_assumed": central_mean_1_f0,   # 518.89 >= 500
        "central_mean_at_lambda1_grounded": central_mean_1_fg,  # 512.44 >= 500 (still clears!)
        "central_clears_at_lambda1_grounded": bool(central_mean_1_fg >= TARGET),
        # conservative reading: mean DROPS below 500 at lambda=1 under grounded f_priv (the #224 finding)
        "conservative_mean_at_lambda1_assumed": cons_mean_1_f0,    # 504.86 >= 500
        "conservative_mean_at_lambda1_grounded": cons_mean_1_fg,   # 498.58 < 500 (#224)
        "conservative_clears_at_lambda1_grounded": bool(cons_mean_1_fg >= TARGET),
        "note": (
            "f_priv=0.969107 is #217's tree-only multiplier; #224 GROUNDED it to 0.957054 from the "
            "one hard #52 paired draw (the assumed value is OPTIMISTIC by ~1.2 pts). dlambda_floor/"
            "df_priv < 0: a LOWER f_priv RAISES the publish-first floor. CENTRAL reading: even at the "
            "grounded f_priv the publish-first MEAN still clears at lambda=1 (512.44 >= 500), the floor "
            "just rises. CONSERVATIVE-ceiling reading: at the grounded f_priv even lambda=1 MISSES "
            "(520.953*0.957054 = 498.58 < 500) -- exactly #224's private_mean_at_ceiling finding, which "
            "is why the conservative reading is the wrong gate for #124's point-estimate launch."
        ),
    }


# ---------------------------------------------------------------------------
# the convention reconciliation (the honest frame -- the load-bearing section)
# ---------------------------------------------------------------------------
def convention_reconciliation(floor: dict[str, Any], mean_curve_d: dict[str, Any],
                              imp: dict[str, Any]) -> dict[str, Any]:
    return {
        "central_reading": {
            "name": "publish_first_point_estimate (HEADLINE)",
            "public_number": "central 535.433 (#191 public_central_lambda1)",
            "private_mean_lambda1": mean_curve_d["anchor_central_lambda1"],   # 518.892
            "lambda_floor": floor["lambda_floor_publish_first"],             # 0.9138
            "gap_to_p95": floor["lambda_gap_pe_vs_p95"],                     # 0.0642
            "equals_191_central_bar_resid": floor["lambda_floor_central_xcheck_191_resid"],
            "why": (
                "PUBLISH-FIRST (#124) = launch on the POINT estimate, accept the draw risk. The point "
                "estimate is the CENTRAL public number 535.433, so private_mean = 535.433*f_priv = "
                "518.89 at lambda=1, crossing 500 at 0.9138 = #191's lambda_star_central_private. The "
                "0.9780 P95 bar is the SAME #191 composition read at the public LCB; the 0.064 gap "
                "between them is the public-sampling confidence haircut = the accepted draw risk."
            ),
        },
        "conservative_ceiling_reading": {
            "name": "PR round-trip note (504.86) -- reconciliation row, NOT the gate",
            "public_number": "LCB ceiling 520.953 (#204/#217, P95-conservative)",
            "private_mean_lambda1": mean_curve_d["anchor_conservative_lambda1"],  # 504.859
            "lambda_floor": floor["lambda_floor_conservative_ceiling"],          # 0.9778
            "gap_to_p95": floor["lambda_gap_conservative_vs_p95"],               # 0.0002 (collapses)
            "why": (
                "520.953*0.969107 = 504.86 is numerically #191's private_lcb_lambda1 (the public-LCB-"
                "discounted private projection). Using it as the 'mean' puts the floor (0.9778) on top "
                "of the 0.9780 P95 bar -- a near-tautology, gap ~0.0002. It also uses a STRICTER public "
                "input (the P95-low ceiling) than the point-estimate gate the human chose, so it is the "
                "wrong reading for #124. Carried + self-tested for completeness."
            ),
        },
        "verdict": (
            "The publish-first POINT-estimate floor the human's #124 gate requires is the CENTRAL "
            f"reading: lambda_floor_publish_first = {floor['lambda_floor_publish_first']:.6f} "
            f"(= #191 lambda_star_central_private), gap {floor['lambda_gap_pe_vs_p95']:.4f} below the "
            "0.9780 P95 both-bugs bar. The PR's 504.86 round-trip note is the conservative-ceiling "
            "projection (floor ~0.9778), which collides with the P95 bar and is stricter than the "
            "chosen gate; it is reported as a reconciliation row, not the deliverable."
        ),
    }


# ---------------------------------------------------------------------------
# step 5: self-test (PRIMARY)
# ---------------------------------------------------------------------------
def self_test(mean_central, mean_cons, public_central, floor: dict[str, Any],
              sens: dict[str, Any], mean_curve_d: dict[str, Any],
              imp: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    evid: dict[str, Any] = {}
    f_priv = imp["f_priv"]

    # (a) round-trips: central anchor == #191 private_central (518.89); conservative note >= 500
    a_central = abs(mean_curve_d["anchor_central_lambda1"] - imp["private_central_lambda1"])
    a_cons_clears = mean_curve_d["anchor_conservative_lambda1"] >= TARGET
    a_cons_resid = abs(mean_curve_d["anchor_conservative_lambda1"] - imp["private_lcb_lambda1"])
    checks["a_central_anchor_roundtrips_191_518"] = bool(a_central < TOL)
    checks["a_conservative_note_504_clears_500"] = bool(a_cons_clears and a_cons_resid < TOL)
    evid["a_central_resid"] = a_central
    evid["a_conservative_anchor"] = mean_curve_d["anchor_conservative_lambda1"]

    # (b) private_mean monotone INCREASING in lambda (both readings)
    grid = [0.80 + 0.02 * k for k in range(11)]  # 0.80..1.00
    inc_c = all(mean_central(grid[i], f_priv) < mean_central(grid[i + 1], f_priv)
                for i in range(len(grid) - 1))
    inc_k = all(mean_cons(grid[i], f_priv) < mean_cons(grid[i + 1], f_priv)
                for i in range(len(grid) - 1))
    checks["b_private_mean_monotone_increasing"] = bool(inc_c and inc_k)
    evid["b_inc_central"] = bool(inc_c)
    evid["b_inc_conservative"] = bool(inc_k)

    # (c) round-trip the inversion: private_mean(floor)=500 to tol (both readings)
    c_central = abs(mean_central(floor["lambda_floor_publish_first"], f_priv) - TARGET)
    c_cons = abs(mean_cons(floor["lambda_floor_conservative_ceiling"], f_priv) - TARGET)
    checks["c_floor_roundtrips_500"] = bool(c_central < TOL and c_cons < TOL)
    evid["c_central_resid"] = c_central
    evid["c_conservative_resid"] = c_cons

    # (d) ordering: both floors < 0.9780 (point-estimate easier than / not above the P95 bar)
    d_central = floor["lambda_floor_publish_first"] < imp["lambda_star_lcb_private_191"]
    d_cons = floor["lambda_floor_conservative_ceiling"] < imp["lambda_star_lcb_private_191"]
    checks["d_floor_below_p95_bar"] = bool(d_central and d_cons)
    evid["d_gap_central"] = floor["lambda_gap_pe_vs_p95"]

    # (e) sensitivity sane: f_priv->1 LOWERS the floor; grounded f_priv RAISES it
    checks["e_sensitivity_sane"] = bool(sens["floor_drops_at_fpriv_one"]
                                        and sens["floor_rises_under_grounding"]
                                        and sens["dlambda_floor_df_priv"] < 0.0)
    evid["e_dlambda_df_priv"] = sens["dlambda_floor_df_priv"]

    # (f) cross-check: headline floor == #191 lambda_star_central_private (external validity)
    checks["f_central_floor_equals_191_bar"] = bool(
        floor["lambda_floor_central_xcheck_191_resid"] < TOL_XCHK)
    evid["f_xcheck_191_resid"] = floor["lambda_floor_central_xcheck_191_resid"]

    # (g) NaN-clean: every reported scalar finite
    def _all_finite(obj) -> bool:
        if isinstance(obj, bool) or obj is None:
            return True
        if isinstance(obj, (int, float)):
            return math.isfinite(obj)
        if isinstance(obj, str):
            return True
        if isinstance(obj, dict):
            return all(_all_finite(v) for v in obj.values())
        if isinstance(obj, list):
            return all(_all_finite(v) for v in obj)
        return True
    g_clean = (_all_finite(floor) and _all_finite(sens) and _all_finite(mean_curve_d))
    checks["g_nan_clean"] = bool(g_clean)

    passes = all(checks.values())
    return {
        "publish_first_lambda_floor_self_test_passes": bool(passes),
        "checks": checks,
        "evidence": evid,
        "n_checks": len(checks),
    }


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------
def run() -> dict[str, Any]:
    t0 = time.time()
    curve = _build_et_curve()
    imp = import_banked()
    mean_central, mean_cons, public_central, et1 = make_curves(curve, imp)
    et_of_lambda = curve["et_of_lambda"]

    mc = mean_curve(mean_central, mean_cons, et_of_lambda, imp)
    floor = lambda_floor(mean_central, mean_cons, imp)
    regimes = gate_regimes(floor, mean_central, imp)
    sens = sensitivity(mean_central, mean_cons, public_central, imp)
    recon = convention_reconciliation(floor, mc, imp)
    st = self_test(mean_central, mean_cons, public_central, floor, sens, mc, imp)

    peak_mem_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    lf = floor["lambda_floor_publish_first"]
    gap = floor["lambda_gap_pe_vs_p95"]
    handoff = (
        f"fern #185 + land #71 + Issue #124: the publish-first POINT-estimate gate (the human's "
        f"chosen #124 launch) clears once the BUILT lambda >= {lf:.6f} (private MEAN reaches 500, "
        f"= #191's lambda_star_central_private), which is {gap:.4f} below the 0.9780 P95 both-bugs "
        f"bar -- so land #71's build target for the human's gate is {lf:.4f}; the band "
        f"[{lf:.4f}, 0.9780) is publish-first GO / P95-private HOLD (the accepted single-draw risk). "
        f"NOTE the PR's round-trip note 520.953*0.969107=504.86 is the conservative public-LCB-ceiling "
        f"projection (floor ~0.9778, collides with the P95 bar); the point-estimate gate uses the "
        f"CENTRAL public number 535.433 -> 518.89 -> floor {lf:.4f}. Under #224's grounded f_priv=0.957 "
        f"the central mean still clears at lambda=1 (512.44>=500); only the conservative reading misses."
    )

    result = {
        "pr": 228,
        "metric_primary": "publish_first_lambda_floor_self_test_passes",
        "metric_test": "lambda_floor_publish_first",
        "publish_first_lambda_floor_self_test_passes":
            st["publish_first_lambda_floor_self_test_passes"],
        "lambda_floor_publish_first": lf,
        # headlines
        "lambda_gap_pe_vs_p95": gap,
        "built_lambda_for_publish_first_go": lf,
        "lambda_floor_conservative_ceiling": floor["lambda_floor_conservative_ceiling"],
        "lambda_star_central_private_191": imp["lambda_star_central_private_191"],
        "p95_private_bar": floor["p95_private_bar"],
        "lambda_floor_central_xcheck_191_resid": floor["lambda_floor_central_xcheck_191_resid"],
        # sections
        "import_banked": imp,
        "composition": {"k_cal": curve["k_cal"], "step_int4": curve["step_int4"],
                        "tau_central": TAU_CENTRAL, "tau_cons": TAU_CONS,
                        "topology": TOPO, "E_T_lambda1": et1},
        "private_mean_vs_lambda": mc,
        "lambda_floor": floor,
        "gate_regimes": regimes,
        "sensitivity": sens,
        "convention_reconciliation": recon,
        "self_test": st,
        "handoff": handoff,
        "scope": (
            "LOCAL CPU-only analytic inversion of the banked private-mean curve to the publish-first "
            "lambda-floor. Takes NO official draws, authorizes none. BASELINE stays 481.53; adds 0 TPS; "
            "greedy/PPL untouched. Imports #224/#217/#204/#191 + the #183/#175/#184 reach-DP VERBATIM; "
            "does NOT re-derive E[T](lambda), K_cal, step, tau, f_priv, the ceiling, or the 0.9780 bar. "
            "The POINT-estimate companion to #224's P95 finding; the P95-private bar STAYS 0.9780 (both "
            "reported). Directly serves Issue #124's publish-first gate + fern #185 + land #71. NOT a launch."
        ),
        "public_evidence_used": [
            "kanna #224 (private_bar_reachability, MERGED `1081oc84`): mu_ceiling_needed=535.139, "
            "private_500_reachable_at_physical_ceiling=0, grounded f_priv=0.957054 -- the P95 finding "
            "this leg gives the POINT-estimate companion to.",
            "kanna #217 (trigger_reconcile, MERGED `vgovdrjc`): f_priv=0.969107, lambda=1 ceiling "
            "520.953 (public LCB), sigma_draw=7.391, mu_safe_fresh=512.157.",
            "stark #191 (private_build_bar, MERGED): both-bugs public_central_lambda1=535.433, "
            "public_lcb_lambda1=520.953, lambda_star_central_private=0.9138 (the headline floor this "
            "leg re-derives independently), lambda_star_lcb_private=0.9780 (the P95 bar).",
            "launch composition: K_cal=125.268 (#148/#169), step_int4=1.2182 (#168), tau in "
            "[0.9924,1.0] (#181), E[T](lambda) reach-DP (#175/#184 via the #183 lambda-acceptance card).",
            "Issue #124 (publish-first green-light): launch on the POINT estimate, accept the draw risk.",
        ],
        "method": (
            "LOCAL CPU-only analytic inversion over EXISTING MERGED results + the live #183 reach-DP "
            "card; no GPU/vLLM/HF Job/submission/served-file/draw. BASELINE stays 481.53; adds 0 TPS. "
            "Greedy identity untouched. NOT a launch."
        ),
        "metrics_nan_clean": 1 if st["checks"]["g_nan_clean"] else 0,
        "peak_mem_mib": peak_mem_mib,
        "elapsed_s": round(time.time() - t0, 4),
    }
    return result


# ---------------------------------------------------------------------------
# wandb
# ---------------------------------------------------------------------------
def _log_wandb(args, result: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[pflf] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="winners-curse-redraw-budget", agent="kanna",
            name=args.wandb_name or "kanna/publish-first-lambda-floor",
            group=args.wandb_group,
            tags=["publish-first", "lambda-floor", "private-mean", "point-estimate",
                  "winners-curse", "launch-trigger", "pr228"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic", "target_tps": 500.0,
                    "topology": TOPO, "imports_pr": [224, 217, 204, 191, 183, 175, 184, 124]},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[pflf] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[pflf] wandb disabled; skipping", flush=True)
        return
    try:
        floor = result["lambda_floor"]
        sens = result["sensitivity"]
        mc = result["private_mean_vs_lambda"]
        st = result["self_test"]
        flat = {
            "publish_first_lambda_floor_self_test_passes":
                1.0 if result["publish_first_lambda_floor_self_test_passes"] else 0.0,
            # headline
            "lambda_floor_publish_first": result["lambda_floor_publish_first"],
            "lambda_gap_pe_vs_p95": result["lambda_gap_pe_vs_p95"],
            "built_lambda_for_publish_first_go": result["built_lambda_for_publish_first_go"],
            "lambda_star_central_private_191": result["lambda_star_central_private_191"],
            "lambda_floor_central_xcheck_191_resid": result["lambda_floor_central_xcheck_191_resid"],
            "p95_private_bar": result["p95_private_bar"],
            # conservative reading
            "lambda_floor_conservative_ceiling": floor["lambda_floor_conservative_ceiling"],
            "lambda_gap_conservative_vs_p95": floor["lambda_gap_conservative_vs_p95"],
            # anchors
            "anchor_central_lambda1": mc["anchor_central_lambda1"],
            "anchor_conservative_lambda1": mc["anchor_conservative_lambda1"],
            "anchor_central_roundtrips_191": mc["anchor_central_roundtrips_191"],
            "anchor_conservative_roundtrips_191": mc["anchor_conservative_roundtrips_191"],
            # sensitivity
            "lambda_floor_at_f_priv_assumed": sens["lambda_floor_at_f_priv_assumed"],
            "lambda_floor_at_f_priv_grounded": sens["lambda_floor_at_f_priv_grounded"],
            "lambda_floor_at_f_priv_one": sens["lambda_floor_at_f_priv_one"],
            "dlambda_floor_df_priv": sens["dlambda_floor_df_priv"],
            "central_mean_at_lambda1_grounded": sens["central_mean_at_lambda1_grounded"],
            "conservative_mean_at_lambda1_grounded": sens["conservative_mean_at_lambda1_grounded"],
            # composition
            "k_cal": result["composition"]["k_cal"],
            "step_int4": result["composition"]["step_int4"],
            "E_T_lambda1": result["composition"]["E_T_lambda1"],
            "f_priv": result["import_banked"]["f_priv"],
            "f_priv_grounded_224": result["import_banked"]["f_priv_grounded"],
            "ceiling_lcb": result["import_banked"]["ceiling_lcb"],
            "public_central_lambda1": result["import_banked"]["public_central_lambda1"],
            "metrics_nan_clean": float(result["metrics_nan_clean"]),
            "peak_mem_mib": result["peak_mem_mib"],
        }
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="publish_first_lambda_floor",
            artifact_type="winners-curse-redraw-budget", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[pflf] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    floor = result["lambda_floor"]
    sens = result["sensitivity"]
    mc = result["private_mean_vs_lambda"]
    recon = result["convention_reconciliation"]
    st = result["self_test"]
    print("\n[pflf] ===== PUBLISH-FIRST LAMBDA-FLOOR  private_mean(lambda)=K_cal*E[T]/step*tau*f_priv  (PR #228) =====",
          flush=True)
    print(f"  composition: K_cal={result['composition']['k_cal']:.5f}  step={result['composition']['step_int4']:.4f}  "
          f"E[T](1)={result['composition']['E_T_lambda1']:.5f}  f_priv={result['import_banked']['f_priv']:.6f}  topo={TOPO}",
          flush=True)
    print(f"\n  private-mean curve over land #71's band (central = HEADLINE / conservative = PR note):", flush=True)
    for r in mc["rows"]:
        print(f"    lambda={r['lambda']:.5f}  E[T]={r['E_T']:.5f}  central={r['private_mean_central']:8.3f} "
              f"({'GO' if r['private_mean_central_clears_500'] else '--'})  "
              f"conserv={r['private_mean_conservative']:8.3f} ({'GO' if r['private_mean_conservative_clears_500'] else '--'})",
              flush=True)
    print(f"\n  DELIVERABLE  lambda_floor_publish_first (CENTRAL) = {floor['lambda_floor_publish_first']:.6f}", flush=True)
    print(f"    == #191 lambda_star_central_private {floor['lambda_star_central_private_191']:.6f}  "
          f"(resid {floor['lambda_floor_central_xcheck_191_resid']:.2e})", flush=True)
    print(f"    gap to 0.9780 P95 bar  lambda_gap_pe_vs_p95 = {floor['lambda_gap_pe_vs_p95']:.5f}", flush=True)
    print(f"    [reconciliation] conservative-ceiling floor = {floor['lambda_floor_conservative_ceiling']:.6f}  "
          f"(gap {floor['lambda_gap_conservative_vs_p95']:.5f} -- collides w/ P95 bar)", flush=True)
    print(f"\n  SENSITIVITY  dlambda_floor/df_priv = {sens['dlambda_floor_df_priv']:.4f}  "
          f"(grounded f_priv {sens['f_priv_grounded_224']:.4f} -> floor {sens['lambda_floor_at_f_priv_grounded']:.5f})",
          flush=True)
    print(f"    central mean @lambda=1 grounded = {sens['central_mean_at_lambda1_grounded']:.2f} "
          f"({'clears' if sens['central_clears_at_lambda1_grounded'] else 'MISS'})   "
          f"conservative @lambda=1 grounded = {sens['conservative_mean_at_lambda1_grounded']:.2f} "
          f"({'clears' if sens['conservative_clears_at_lambda1_grounded'] else 'MISS (#224)'})", flush=True)
    print(f"\n  VERDICT: {recon['verdict']}", flush=True)
    print(f"\n  SELF-TEST (PRIMARY) passes = {st['publish_first_lambda_floor_self_test_passes']}", flush=True)
    for k, v in st["checks"].items():
        print(f"    [{'ok' if v else '!! FAILED'}] {k}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_HERE, "publish_first_lambda_floor_results.json"))
    ap.add_argument("--self-test", action="store_true",
                    help="run the self-test (always runs; sets nonzero exit on failure)")
    ap.add_argument("--wandb-name", "--wandb_name", default="kanna/publish-first-lambda-floor")
    ap.add_argument("--wandb-group", "--wandb_group", default="winners-curse-redraw-budget")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    result = run()

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"[pflf] wrote {args.out}", flush=True)

    _print(result)
    _log_wandb(args, result)

    if not result["publish_first_lambda_floor_self_test_passes"]:
        print("[pflf] SELF-TEST FAILED", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
