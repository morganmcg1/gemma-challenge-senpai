#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Publish-first ACCEPTED-RISK curve: how much private-draw risk did #124 actually accept? (PR #237).

WHAT THIS IS
------------
The human's #124 green-light accepted "the single-draw risk" of publish-first as a QUALITATIVE
posture. kanna #228 (`352ifoi8`, MERGED) pinned the publish-first BAND endpoints
[lambda_floor=0.913827, P95-bar=0.978011) but delivered the band's EDGES, not its RISK MAGNITUDE.
This leg quantifies HOW MUCH risk that posture is, as a function of the BUILT lambda, by composing
two banked legs:

    private_draw(lambda) ~ Normal( mu = private_mean(lambda),  sigma = sigma_draw = 7.391 )

    P(clears 500 | lambda) = Phi( (private_mean(lambda) - 500) / sigma_draw )
    risk(lambda)           = 1 - P(clears 500 | lambda)              # the accepted-risk curve

with private_mean(lambda) the EXACT #228 central point-estimate curve (imported verbatim, not
re-derived: public_central(lambda)*f_priv = K_cal*E[T](lambda)/step*tau*f_priv) and sigma_draw the
#217 (`vgovdrjc`) fresh private-draw sigma. The curve IS the "how much risk did #124 accept" object.

THE THREE HEADLINE RISKS (across the publish-first band)
-------------------------------------------------------
  accepted_risk_at_floor      risk(0.913827)  ~= 0.5000  (mean == 500 -> coin-flip; the anchor)
  accepted_risk_at_speed_gate risk(0.967468)  the OPERATIVE public launch lambda (ubel #229);
                                              THE number entailed by launching at the public gate
  accepted_risk_at_p95        risk(0.978011)  ~= 0.030   (stark #191's LCB-on-lambda bar)

RECONCILING THE TWO 0.9780 CONSTRUCTIONS (the deliverable)
----------------------------------------------------------
stark #191's `lambda_star_lcb_private` = 0.978011 is the LCB-ON-LAMBDA bar: the lambda whose private
mean, computed at the PUBLIC LCB (520.953, a ~14 TPS haircut vs the central 535.433), reaches 500.
Our draw model gives a DRAW-CLEARS-500 probability around the CENTRAL mean. They are related but NOT
identical. We compute `lambda_risk5` (the lambda where risk(lambda) = 0.05, i.e. the central mean
sits z1*sigma_draw above 500 == mu_safe_fresh = 512.157) and compare it to 0.978011:

    lambda_risk5 ~= 0.970  <  0.978011  (the LCB-bar)        gap ~= -0.008

The LCB-bar is the MORE conservative construction (it discounts PUBLIC sampling uncertainty); the
draw-risk only accounts for the PRIVATE single-draw sigma. So at the 0.9780 LCB-bar the actual
draw-risk is SMALLER than 5% (~0.030). The draw-risk risk(lambda) is the DECISION-RELEVANT quantity;
the LCB-bar is the conservative PRE-REGISTRATION. Both reported; the draw-risk is operative for #124.

LOCAL CPU-only analytic composition over EXISTING MERGED legs. No GPU / vLLM / HF Job / submission /
served-file change / official draw. BASELINE stays 481.53; greedy/PPL untouched; adds 0 TPS;
authorizes nothing. Imports #228 (private_mean curve, VERBATIM via module import), #217 (sigma_draw,
mu_safe_fresh, f_priv), #191 (the 0.9780 LCB-bar), #224 (grounded f_priv), #229 (the speed gate). Does
NOT re-derive private_mean, sigma_draw, the f_priv worst-case BLEND (stark #233), or the integration
(fern #231). The draw-risk DISTRIBUTION around the imported private-mean POINT is the only new object.

PRIMARY metric  publishfirst_accepted_risk_self_test_passes
TEST    metric  accepted_risk_at_floor   (the risk magnitude at the 0.913827 floor ~= 0.5)
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
PFLF_228 = os.path.join(
    _ROOT, "research/validity/publish_first_lambda_floor/publish_first_lambda_floor.py")
TRIGGER_RECONCILE_217 = os.path.join(
    _ROOT, "research/validity/trigger_reconcile/trigger_reconcile_results.json")
SPEED_MARGIN_229 = os.path.join(
    _ROOT, "research/validity/speed_margin_at_validity_bar/speed_margin_at_validity_bar_results.json")

TARGET = 500.0
Z1 = 1.6448536269514722  # one-sided P95 == Phi^{-1}(0.95) (#204)

# the publish-first band lambda grid (PR step 2): floor, mid, speed-gate, P95-bar, near-1, 1
# the floor + p95-bar + speed-gate are filled with the exact banked values at runtime.
LAMBDA_GRID_NAMES = ["floor", "0.9500", "speed_gate", "p95_bar", "0.9970", "1.0000"]

TOL = 1e-6
TOL_XCHK = 1e-7
SIGMA_SWEEP = (0.8, 1.0, 1.2)   # +-20% robustness (PR step 4b)


# ---------------------------------------------------------------------------
# import the #228 private-mean machinery VERBATIM (one source for private_mean(lambda))
# ---------------------------------------------------------------------------
def _load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _finite(x: Any) -> float:
    if x is None or not math.isfinite(float(x)):
        raise ValueError(f"non-finite value: {x!r}")
    return float(x)


def norm_cdf(x: float) -> float:
    """Standard-normal CDF Phi(x) via erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# compose the imports: private_mean(lambda) [#228] + sigma_draw [#217] + the gates
# ---------------------------------------------------------------------------
def build_composition() -> dict[str, Any]:
    pflf = _load_mod("publish_first_lambda_floor", PFLF_228)
    curve = pflf._build_et_curve()          # #183 reach-DP -> E[T](lambda) (same object fern reads)
    imp228 = pflf.import_banked()           # #217/#191/#224 banked constants, verbatim
    mean_central, mean_cons, public_central, et1 = pflf.make_curves(curve, imp228)

    # sigma_draw + mu_safe_fresh straight from #217's banked leg (the fresh private-draw sigma).
    d217 = _load_json(TRIGGER_RECONCILE_217)["import_banked"]
    sigma_draw = _finite(d217["sigma_draw"])          # 7.390974474817942
    mu_safe_fresh = _finite(d217["mu_safe_fresh"])    # 512.15707117161 == 500 + Z1*sigma_draw

    # the operative public speed gate (ubel #229 both-bugs worstcase headline) -- rounds to 0.9675.
    lam_speed = _finite(_load_json(SPEED_MARGIN_229)["lambda_speed_clears_worstcase"])

    f_priv = imp228["f_priv"]                          # 0.969106920637722 (assumed)
    f_priv_grounded = imp228["f_priv_grounded"]        # 0.9570535584491102 (#224 grounded)
    p95_bar = imp228["lambda_star_lcb_private_191"]    # 0.9780112973731208 (the LCB-on-lambda bar)
    floor = pflf.solve_floor(lambda lam: mean_central(lam, f_priv))   # 0.9138270633 (mean==500)

    return {
        "pflf": pflf,
        "mean_central": mean_central,        # private_mean(lambda, f_priv) -- the #228 central curve
        "public_central": public_central,
        "solve_floor": pflf.solve_floor,
        "et1": et1,
        "imp228": imp228,
        "sigma_draw": sigma_draw,
        "mu_safe_fresh": mu_safe_fresh,
        "lam_speed": lam_speed,
        "f_priv": f_priv,
        "f_priv_grounded": f_priv_grounded,
        "p95_bar": p95_bar,
        "floor": floor,
        "k_cal": curve["k_cal"],
        "step_int4": curve["step_int4"],
    }


# ---------------------------------------------------------------------------
# the draw model (PR step 1): private_draw(lambda) ~ Normal(private_mean(lambda), sigma_draw)
# ---------------------------------------------------------------------------
def make_risk(comp: dict[str, Any]):
    mean_central = comp["mean_central"]
    sigma0 = comp["sigma_draw"]

    def p_clear(lam: float, f_priv: float, sigma: float | None = None) -> float:
        s = sigma if sigma is not None else sigma0
        return norm_cdf((mean_central(lam, f_priv) - TARGET) / s)

    def risk(lam: float, f_priv: float, sigma: float | None = None) -> float:
        return 1.0 - p_clear(lam, f_priv, sigma)

    return p_clear, risk


def solve_lambda_risk5(comp: dict[str, Any], f_priv: float,
                       sigma: float | None = None) -> float | None:
    """The lambda at which risk(lambda) == 0.05, i.e. central mean == 500 + Z1*sigma.

    Uses the exact banked one-sided-P95 constant Z1 (== Phi^{-1}(0.95)) so the result round-trips
    #217's mu_safe_fresh (= 500 + Z1*sigma_draw) to machine precision -- the same Z1 that built it.
    """
    s = sigma if sigma is not None else comp["sigma_draw"]
    target_mean = TARGET + Z1 * s
    return comp["solve_floor"](lambda lam: comp["mean_central"](lam, f_priv), target=target_mean)


# ---------------------------------------------------------------------------
# step 2: the accepted-risk curve over the publish-first band (the core)
# ---------------------------------------------------------------------------
def accepted_risk_curve(comp: dict[str, Any], p_clear, risk) -> dict[str, Any]:
    fp, fg = comp["f_priv"], comp["f_priv_grounded"]
    floor, p95, speed = comp["floor"], comp["p95_bar"], comp["lam_speed"]
    lambdas = [floor, 0.95, speed, p95, 0.997, 1.0]

    def regime(lam: float) -> str:
        if lam >= p95:
            return "both_clear"
        if lam >= floor:
            return "publish_first_go_p95_hold"
        return "publish_first_no_go"

    def note(name: str) -> str:
        return {
            "floor": "mean==500 -> coin-flip (the strongest anchor; risk==0.5 by construction)",
            "0.9500": "interior of the publish-first band (mean>500, single-draw risk held)",
            "speed_gate": "OPERATIVE public launch lambda (ubel #229); risk entailed by #124 launch",
            "p95_bar": "stark #191 LCB-on-lambda bar; draw-risk here is SMALLER than 5% (over-protect)",
            "0.9970": "near the lambda=1 spine; private mean ~518.2",
            "1.0000": "lambda=1 spine; private mean == #228 anchor 518.892",
        }[name]

    rows = []
    for lam, name in zip(lambdas, LAMBDA_GRID_NAMES):
        mu_a = comp["mean_central"](lam, fp)
        mu_g = comp["mean_central"](lam, fg)
        rows.append({
            "name": name,
            "lambda": lam,
            "private_mean_assumed": mu_a,
            "p_clear_assumed": p_clear(lam, fp),
            "risk_assumed": risk(lam, fp),
            "private_mean_grounded": mu_g,
            "p_clear_grounded": p_clear(lam, fg),
            "risk_grounded": risk(lam, fg),
            "regime": regime(lam),
            "construction_note": note(name),
        })
    return {"topology": "both_bugs", "sigma_draw": comp["sigma_draw"], "rows": rows}


# ---------------------------------------------------------------------------
# step 3: reconcile the two 0.9780 constructions (LCB-on-lambda vs draw-clears-500)
# ---------------------------------------------------------------------------
def reconcile_two_constructions(comp: dict[str, Any], p_clear, risk) -> dict[str, Any]:
    fp = comp["f_priv"]
    p95 = comp["p95_bar"]
    lambda_risk5 = solve_lambda_risk5(comp, fp)
    mu_at_risk5 = comp["mean_central"](lambda_risk5, fp)

    return {
        "lambda_risk5": lambda_risk5,                      # ~0.970 (draw-clears-500 at 95%)
        "lambda_risk5_minus_p95bar": lambda_risk5 - p95,   # ~-0.008 (BELOW the LCB-bar)
        "p95_lcb_on_lambda_bar": p95,                      # 0.978011 (stark #191)
        "mu_at_lambda_risk5": mu_at_risk5,                 # == mu_safe_fresh 512.157
        "mu_safe_fresh": comp["mu_safe_fresh"],
        "lambda_risk5_roundtrips_mu_safe": abs(mu_at_risk5 - comp["mu_safe_fresh"]),
        "risk_at_p95_bar": risk(p95, fp),                  # ~0.030 (< 0.05: LCB-bar over-protects)
        "table": [
            {
                "lambda": comp["floor"],
                "construction": "draw-risk floor (mean==500)",
                "private_mean": comp["mean_central"](comp["floor"], fp),
                "p_clears_500": p_clear(comp["floor"], fp),
                "risk": risk(comp["floor"], fp),
                "note": "central mean touches 500 -> coin-flip draw-risk",
            },
            {
                "lambda": lambda_risk5,
                "construction": "draw-clears-500 at 95% (risk==0.05)",
                "private_mean": mu_at_risk5,
                "p_clears_500": p_clear(lambda_risk5, fp),
                "risk": risk(lambda_risk5, fp),
                "note": "central mean == mu_safe_fresh 512.157 == 500 + z1*sigma_draw",
            },
            {
                "lambda": p95,
                "construction": "LCB-on-lambda bar (stark #191)",
                "private_mean": comp["mean_central"](p95, fp),
                "p_clears_500": p_clear(p95, fp),
                "risk": risk(p95, fp),
                "note": "lambda whose PUBLIC-LCB private mean reaches 500; draw-risk here < 0.05",
            },
        ],
        "operative_construction": "draw_risk",
        "verdict": (
            "The two 0.9780 constructions are NOT the same object. stark #191's "
            f"lambda_star_lcb_private={p95:.6f} is the LCB-ON-LAMBDA bar: the lambda whose private "
            "mean computed at the PUBLIC LCB (520.953, a ~14 TPS haircut below the central 535.433) "
            "reaches 500 -- a conservative PRE-REGISTRATION that discounts public sampling "
            "uncertainty. The draw-clears-500-at-95% lambda is "
            f"lambda_risk5={lambda_risk5:.6f} (central mean == mu_safe_fresh 512.157 == 500 + "
            f"z1*sigma_draw), which sits {lambda_risk5 - p95:+.6f} BELOW the LCB-bar. Equivalently, "
            f"at the 0.9780 LCB-bar the actual single-draw risk is only {risk(p95, fp):.4f} (< 0.05) "
            "because the LCB-bar over-protects relative to the private-draw sigma. OPERATIVE for the "
            "human's #124 decision is the draw-risk risk(lambda) (the chance the actual private draw "
            "fails its bar); the LCB-bar is the conservative pre-registration."
        ),
    }


# ---------------------------------------------------------------------------
# step 4: robustness -- f_priv axis (grounded) + sigma_draw +-20% sweep
# ---------------------------------------------------------------------------
def robustness(comp: dict[str, Any], p_clear, risk) -> dict[str, Any]:
    fp, fg = comp["f_priv"], comp["f_priv_grounded"]
    speed, p95, floor = comp["lam_speed"], comp["p95_bar"], comp["floor"]
    sigma0 = comp["sigma_draw"]

    # (a) the f_priv axis: assumed vs grounded at the operative speed gate (the calibration sensitivity)
    f_priv_axis = {
        "accepted_risk_at_speed_gate_assumed": risk(speed, fp),     # ~0.058
        "accepted_risk_at_speed_gate_grounded": risk(speed, fg),    # ~0.239
        "accepted_risk_at_p95_assumed": risk(p95, fp),
        "accepted_risk_at_p95_grounded": risk(p95, fg),
        "floor_assumed": floor,
        "floor_grounded": comp["solve_floor"](lambda lam: comp["mean_central"](lam, fg)),
        "grounded_curve_strictly_above_assumed": None,   # filled below
        "note": (
            "f_priv=0.969107 is #217's tree-only multiplier; #224 GROUNDED it to 0.957054 from the "
            "one hard #52 paired draw. Lower f_priv -> lower private mean -> HIGHER risk at every "
            "lambda. At the operative speed gate the accepted risk roughly QUADRUPLES (0.058 -> 0.239) "
            "under grounding -- the dominant calibration sensitivity for the human's decision."
        ),
    }
    # confirm the grounded curve sits strictly above the assumed curve over a grid
    grid = [floor + (1.0 - floor) * k / 20.0 for k in range(21)]
    f_priv_axis["grounded_curve_strictly_above_assumed"] = bool(
        all(risk(l, fg) > risk(l, fp) for l in grid))

    # (b) sigma_draw +-20% sweep: confirm DIRECTION stable (risk monotone in sigma; lambda_risk5 monotone)
    sigma_rows = []
    for sc in SIGMA_SWEEP:
        s = sigma0 * sc
        lr5 = solve_lambda_risk5(comp, fp, sigma=s)
        sigma_rows.append({
            "sigma_scale": sc,
            "sigma_draw": s,
            "lambda_risk5": lr5,
            "lambda_risk5_below_p95bar": bool(lr5 < p95),
            "accepted_risk_at_speed_gate": risk(speed, fp, sigma=s),
            "accepted_risk_at_p95": risk(p95, fp, sigma=s),
        })
    # direction stable: lambda_risk5 monotone increasing in sigma; risk@speed monotone increasing
    lr5_seq = [r["lambda_risk5"] for r in sigma_rows]
    risk_speed_seq = [r["accepted_risk_at_speed_gate"] for r in sigma_rows]
    sigma_sweep = {
        "rows": sigma_rows,
        "lambda_risk5_monotone_increasing_in_sigma":
            bool(all(lr5_seq[i] < lr5_seq[i + 1] for i in range(len(lr5_seq) - 1))),
        "risk_at_speed_monotone_increasing_in_sigma":
            bool(all(risk_speed_seq[i] < risk_speed_seq[i + 1] for i in range(len(risk_speed_seq) - 1))),
        "verdict_direction_stable": None,   # filled below
        "note": (
            "Larger sigma_draw -> more single-draw spread -> higher risk at fixed lambda and a higher "
            "lambda needed to reach any fixed risk. The DIRECTION (risk up in sigma, lambda_risk5 up in "
            "sigma) is stable across +-20%. The 5%-risk lambda straddles the 0.9780 LCB-bar across the "
            "band (it exceeds it only at +20% sigma), so the speed-gate risk is near-5% and "
            "sigma-sensitive in MAGNITUDE but robust in ORDERING."
        ),
    }
    sigma_sweep["verdict_direction_stable"] = bool(
        sigma_sweep["lambda_risk5_monotone_increasing_in_sigma"]
        and sigma_sweep["risk_at_speed_monotone_increasing_in_sigma"])

    return {"f_priv_axis": f_priv_axis, "sigma_sweep": sigma_sweep}


# ---------------------------------------------------------------------------
# step 5: self-test (PRIMARY)
# ---------------------------------------------------------------------------
def self_test(comp: dict[str, Any], p_clear, risk, curve_d: dict[str, Any],
              recon: dict[str, Any], robust: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    evid: dict[str, Any] = {}
    fp, fg = comp["f_priv"], comp["f_priv_grounded"]
    floor, p95 = comp["floor"], comp["p95_bar"]
    mean_central = comp["mean_central"]

    # (a) P(clears 500 | floor) == 0.5 within 1e-6 (mean == threshold -> coin-flip)
    a_mu = abs(mean_central(floor, fp) - TARGET)
    a_p = abs(p_clear(floor, fp) - 0.5)
    checks["a_floor_is_coinflip"] = bool(a_mu < TOL and a_p < TOL)
    evid["a_mu_floor_resid"] = a_mu
    evid["a_p_clear_floor"] = p_clear(floor, fp)

    # (b) private_mean(lambda) round-trips #228's anchors (518.892 @1, mean==500 @floor)
    b_anchor = abs(mean_central(1.0, fp) - comp["imp228"]["private_central_lambda1"])
    b_floor = abs(mean_central(floor, fp) - TARGET)
    checks["b_private_mean_roundtrips_228"] = bool(b_anchor < TOL and b_floor < TOL)
    evid["b_anchor_resid"] = b_anchor
    evid["b_mu_lambda1"] = mean_central(1.0, fp)

    # (c) P(clears 500|lambda) monotone INCREASING, risk(lambda) monotone DECREASING in lambda
    grid = [0.90 + 0.005 * k for k in range(21)]   # 0.90..1.00
    inc_p = all(p_clear(grid[i], fp) < p_clear(grid[i + 1], fp) for i in range(len(grid) - 1))
    dec_r = all(risk(grid[i], fp) > risk(grid[i + 1], fp) for i in range(len(grid) - 1))
    checks["c_p_clear_up_risk_down"] = bool(inc_p and dec_r)
    evid["c_p_inc"] = bool(inc_p)
    evid["c_risk_dec"] = bool(dec_r)

    # (d) risk(0.9780) small AND strictly below the naive 5% (LCB-bar over-protects) -> in (0, 0.05)
    r_p95 = risk(p95, fp)
    checks["d_risk_at_p95_small_below_5pct"] = bool(0.0 < r_p95 < 0.05)
    evid["d_risk_at_p95"] = r_p95

    # (e) grounded-f_priv curve strictly ABOVE assumed-f_priv curve (more risk) everywhere
    checks["e_grounded_above_assumed"] = bool(
        robust["f_priv_axis"]["grounded_curve_strictly_above_assumed"])
    evid["e_risk_speed_assumed"] = robust["f_priv_axis"]["accepted_risk_at_speed_gate_assumed"]
    evid["e_risk_speed_grounded"] = robust["f_priv_axis"]["accepted_risk_at_speed_gate_grounded"]

    # (f) reconciliation cross-check: lambda_risk5 round-trips mu_safe_fresh; sits below the LCB-bar
    checks["f_lambda_risk5_roundtrips_mu_safe"] = bool(
        recon["lambda_risk5_roundtrips_mu_safe"] < TOL
        and recon["lambda_risk5_minus_p95bar"] < 0.0)
    evid["f_lambda_risk5"] = recon["lambda_risk5"]
    evid["f_mu_safe_resid"] = recon["lambda_risk5_roundtrips_mu_safe"]

    # (g) sigma-sweep direction stable
    checks["g_sigma_sweep_direction_stable"] = bool(
        robust["sigma_sweep"]["verdict_direction_stable"])

    # (h) NaN-clean: every reported scalar finite
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
    checks["h_nan_clean"] = bool(
        _all_finite(curve_d) and _all_finite(recon) and _all_finite(robust))

    passes = all(checks.values())
    return {
        "publishfirst_accepted_risk_self_test_passes": bool(passes),
        "checks": checks,
        "evidence": evid,
        "n_checks": len(checks),
    }


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------
def run() -> dict[str, Any]:
    t0 = time.time()
    comp = build_composition()
    p_clear, risk = make_risk(comp)

    fp, fg = comp["f_priv"], comp["f_priv_grounded"]
    floor, p95, speed = comp["floor"], comp["p95_bar"], comp["lam_speed"]

    curve_d = accepted_risk_curve(comp, p_clear, risk)
    recon = reconcile_two_constructions(comp, p_clear, risk)
    robust = robustness(comp, p_clear, risk)
    st = self_test(comp, p_clear, risk, curve_d, recon, robust)

    accepted_risk_at_floor = risk(floor, fp)
    accepted_risk_at_speed_gate = risk(speed, fp)
    accepted_risk_at_speed_gate_grounded = risk(speed, fg)
    accepted_risk_at_p95 = risk(p95, fp)

    peak_mem_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    handoff = (
        f"fern #231 + ubel #234 + Issue #124: under #124 publish-first, launching at the OPERATIVE "
        f"public gate lambda={speed:.4f} (ubel #229) entails a private-draw fail-its-bar risk of "
        f"accepted_risk_at_speed_gate={accepted_risk_at_speed_gate:.4f} (assumed f_priv) / "
        f"{accepted_risk_at_speed_gate_grounded:.4f} (grounded #224 f_priv); the accepted-risk curve "
        f"runs from ~0.50 at the {floor:.4f} floor (mean==500, coin-flip) to "
        f"{accepted_risk_at_p95:.4f} at the {p95:.4f} P95 LCB-bar. lambda_risk5={recon['lambda_risk5']:.4f} "
        f"(draw-risk==5%) reconciles the draw-risk construction with stark #191's LCB-on-lambda bar: it "
        f"sits {recon['lambda_risk5_minus_p95bar']:+.4f} below 0.9780 because the LCB-bar discounts "
        f"PUBLIC sampling uncertainty while the draw-risk only carries the PRIVATE single-draw sigma. So "
        f"the human can weigh the TPS gain of a lower-lambda build against a QUANTIFIED private-draw "
        f"risk, not a qualitative posture -- and should note the assumed/grounded f_priv gap roughly "
        f"quadruples that risk at the operative gate."
    )

    result = {
        "pr": 237,
        "metric_primary": "publishfirst_accepted_risk_self_test_passes",
        "metric_test": "accepted_risk_at_floor",
        "publishfirst_accepted_risk_self_test_passes":
            st["publishfirst_accepted_risk_self_test_passes"],
        # headline risks
        "accepted_risk_at_floor": accepted_risk_at_floor,                       # ~0.5000
        "accepted_risk_at_speed_gate": accepted_risk_at_speed_gate,             # ~0.0583
        "accepted_risk_at_speed_gate_grounded": accepted_risk_at_speed_gate_grounded,  # ~0.2394
        "accepted_risk_at_p95": accepted_risk_at_p95,                           # ~0.0297
        # reconciliation
        "lambda_risk5": recon["lambda_risk5"],                                  # ~0.9700
        "lambda_risk5_minus_p95bar": recon["lambda_risk5_minus_p95bar"],        # ~-0.0080
        # the gates / band
        "lambda_floor_publish_first": floor,
        "lambda_speed_gate": speed,
        "p95_private_bar": p95,
        "sigma_draw": comp["sigma_draw"],
        "mu_safe_fresh": comp["mu_safe_fresh"],
        # draw model frame
        "draw_model": {
            "law": "private_draw(lambda) ~ Normal(mu=private_mean(lambda), sigma=sigma_draw)",
            "clear_event": "private_draw >= 500",
            "p_clear": "Phi((private_mean(lambda) - 500) / sigma_draw)",
            "risk": "1 - P(clears 500 | lambda)",
            "sigma_draw": comp["sigma_draw"],
            "f_priv_assumed": fp,
            "f_priv_grounded": fg,
        },
        # composition provenance (imported VERBATIM, not re-derived)
        "import_banked": {
            "private_mean_curve_src": "kanna #228 publish_first_lambda_floor (module import)",
            "sigma_draw_src": "kanna #217 trigger_reconcile (vgovdrjc)",
            "p95_lcb_bar_src": "stark #191 private_build_bar (lambda_star_lcb_private)",
            "f_priv_grounded_src": "kanna #224 private_bar_reachability (1081oc84)",
            "speed_gate_src": "ubel #229 speed_margin_at_validity_bar (bz2b3fw8, both-bugs worstcase)",
            "k_cal": comp["k_cal"], "step_int4": comp["step_int4"], "et_lambda1": comp["et1"],
            "f_priv": fp, "f_priv_grounded": fg, "sigma_draw": comp["sigma_draw"],
            "mu_safe_fresh": comp["mu_safe_fresh"],
            "private_central_lambda1": comp["imp228"]["private_central_lambda1"],
            "lambda_star_lcb_private_191": comp["imp228"]["lambda_star_lcb_private_191"],
            "lambda_star_central_private_191": comp["imp228"]["lambda_star_central_private_191"],
        },
        # sections
        "accepted_risk_curve": curve_d,
        "reconcile_0p9780": recon,
        "robustness": robust,
        "self_test": st,
        "handoff": handoff,
        "scope": (
            "LOCAL CPU-only analytic composition of the banked private-mean curve (#228) with the "
            "fresh private-draw sigma (#217) into the accepted-risk curve P(private draw fails 500 | "
            "built lambda). Takes NO official draws, authorizes none. BASELINE stays 481.53; adds 0 "
            "TPS; greedy/PPL untouched. Imports #228/#217/#191/#224/#229 VERBATIM; does NOT re-derive "
            "private_mean, sigma_draw, the f_priv worst-case BLEND (stark #233), or the integration "
            "(fern #231). The draw-risk DISTRIBUTION around the imported private-mean POINT is the only "
            "new object. Quantifies the qualitative 'accepted single-draw risk' of #124. NOT a launch."
        ),
        "public_evidence_used": [
            "kanna #228 (publish_first_lambda_floor, MERGED `352ifoi8`): private_mean(lambda) central "
            "curve, lambda_floor_publish_first=0.913827, P95-bar 0.978011, f_priv=0.969107 -- the band "
            "endpoints this leg fills with a risk MAGNITUDE. Imported as a module (bit-exact).",
            "kanna #217 (trigger_reconcile, MERGED `vgovdrjc`): sigma_draw=7.391 (fresh private-draw "
            "sigma), mu_safe_fresh=512.157 == 500 + z1*sigma_draw -- the draw-model sigma.",
            "stark #191 (private_build_bar, MERGED): lambda_star_lcb_private=0.978011 (the LCB-on-lambda "
            "bar this leg reconciles against the draw-clears-500 probability).",
            "kanna #224 (private_bar_reachability, MERGED `1081oc84`): grounded f_priv=0.957054 -- the "
            "calibration sensitivity axis (risk roughly quadruples at the speed gate under grounding).",
            "ubel #229 (speed_margin_at_validity_bar, MERGED `bz2b3fw8`): lambda_speed_clears=0.9675 "
            "(both-bugs worstcase) -- the operative public launch gate under #124; the headline risk.",
            "Issue #124 (publish-first green-light): launch on the POINT estimate, accept the draw risk "
            "-- this leg quantifies HOW MUCH risk that posture is.",
        ],
        "method": (
            "LOCAL CPU-only analytic composition over EXISTING MERGED legs (the #228 private-mean "
            "module + the #217/#191/#224/#229 banked scalars). No GPU/vLLM/HF Job/submission/served-"
            "file/draw. BASELINE stays 481.53; adds 0 TPS. Greedy identity untouched. NOT a launch."
        ),
        "metrics_nan_clean": 1 if st["checks"]["h_nan_clean"] else 0,
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
        print(f"[par] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="winners-curse-redraw-budget", agent="kanna",
            name=args.wandb_name or "kanna/publishfirst-accepted-risk",
            group=args.wandb_group,
            tags=["publish-first", "accepted-risk", "draw-risk", "private-mean", "sigma-draw",
                  "winners-curse", "reconcile-0p9780", "pr237"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic", "target_tps": 500.0,
                    "topology": "both_bugs", "imports_pr": [228, 217, 191, 224, 229, 124]},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[par] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[par] wandb disabled; skipping", flush=True)
        return
    try:
        recon = result["reconcile_0p9780"]
        fpa = result["robustness"]["f_priv_axis"]
        ss = result["robustness"]["sigma_sweep"]
        flat = {
            "publishfirst_accepted_risk_self_test_passes":
                1.0 if result["publishfirst_accepted_risk_self_test_passes"] else 0.0,
            # headline risks
            "accepted_risk_at_floor": result["accepted_risk_at_floor"],
            "accepted_risk_at_speed_gate": result["accepted_risk_at_speed_gate"],
            "accepted_risk_at_speed_gate_grounded": result["accepted_risk_at_speed_gate_grounded"],
            "accepted_risk_at_p95": result["accepted_risk_at_p95"],
            # reconciliation
            "lambda_risk5": result["lambda_risk5"],
            "lambda_risk5_minus_p95bar": result["lambda_risk5_minus_p95bar"],
            "lambda_risk5_roundtrips_mu_safe": recon["lambda_risk5_roundtrips_mu_safe"],
            "risk_at_p95_bar": recon["risk_at_p95_bar"],
            # gates / band
            "lambda_floor_publish_first": result["lambda_floor_publish_first"],
            "lambda_speed_gate": result["lambda_speed_gate"],
            "p95_private_bar": result["p95_private_bar"],
            "sigma_draw": result["sigma_draw"],
            "mu_safe_fresh": result["mu_safe_fresh"],
            # robustness
            "floor_grounded": fpa["floor_grounded"],
            "grounded_curve_strictly_above_assumed":
                1.0 if fpa["grounded_curve_strictly_above_assumed"] else 0.0,
            "sigma_sweep_direction_stable": 1.0 if ss["verdict_direction_stable"] else 0.0,
            # composition
            "k_cal": result["import_banked"]["k_cal"],
            "step_int4": result["import_banked"]["step_int4"],
            "et_lambda1": result["import_banked"]["et_lambda1"],
            "f_priv": result["import_banked"]["f_priv"],
            "f_priv_grounded": result["import_banked"]["f_priv_grounded"],
            "metrics_nan_clean": float(result["metrics_nan_clean"]),
            "peak_mem_mib": result["peak_mem_mib"],
        }
        # per-lambda risk curve as a wandb Table for plotting
        try:
            import wandb
            tbl = wandb.Table(columns=["name", "lambda", "private_mean_assumed", "risk_assumed",
                                       "private_mean_grounded", "risk_grounded", "regime"])
            for r in result["accepted_risk_curve"]["rows"]:
                tbl.add_data(r["name"], r["lambda"], r["private_mean_assumed"], r["risk_assumed"],
                             r["private_mean_grounded"], r["risk_grounded"], r["regime"])
            flat["accepted_risk_curve_table"] = tbl
        except Exception as exc:  # noqa: BLE001
            print(f"[par] wandb table skipped ({exc})", flush=True)
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="publishfirst_accepted_risk",
            artifact_type="winners-curse-redraw-budget", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[par] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    curve = result["accepted_risk_curve"]
    recon = result["reconcile_0p9780"]
    fpa = result["robustness"]["f_priv_axis"]
    ss = result["robustness"]["sigma_sweep"]
    st = result["self_test"]
    print("\n[par] ===== PUBLISH-FIRST ACCEPTED-RISK  risk(lambda)=1-Phi((private_mean(lambda)-500)/sigma_draw)  (PR #237) =====",
          flush=True)
    print(f"  draw model: private_draw ~ Normal(private_mean(lambda), sigma_draw={result['sigma_draw']:.4f})  "
          f"f_priv={result['draw_model']['f_priv_assumed']:.6f}", flush=True)
    print(f"\n  accepted-risk curve over the publish-first band (assumed f_priv | grounded #224 f_priv):", flush=True)
    for r in curve["rows"]:
        print(f"    {r['name']:>10}  lambda={r['lambda']:.6f}  mu={r['private_mean_assumed']:8.3f}  "
              f"P_clear={r['p_clear_assumed']:.4f}  risk={r['risk_assumed']:.4f}  "
              f"| grounded risk={r['risk_grounded']:.4f}  [{r['regime']}]", flush=True)
    print(f"\n  HEADLINES:", flush=True)
    print(f"    accepted_risk_at_floor      = {result['accepted_risk_at_floor']:.6f}  (mean==500 -> coin-flip)", flush=True)
    print(f"    accepted_risk_at_speed_gate = {result['accepted_risk_at_speed_gate']:.6f}  (lambda={result['lambda_speed_gate']:.4f} ubel #229; THE #124 number)", flush=True)
    print(f"      grounded f_priv           = {result['accepted_risk_at_speed_gate_grounded']:.6f}", flush=True)
    print(f"    accepted_risk_at_p95        = {result['accepted_risk_at_p95']:.6f}  (lambda={result['p95_private_bar']:.4f})", flush=True)
    print(f"\n  RECONCILE the two 0.9780 constructions:", flush=True)
    print(f"    lambda_risk5 (draw-risk==5%) = {recon['lambda_risk5']:.6f}  vs  LCB-on-lambda bar 0.978011  "
          f"(delta {recon['lambda_risk5_minus_p95bar']:+.6f})", flush=True)
    print(f"    mu(lambda_risk5)={recon['mu_at_lambda_risk5']:.4f} == mu_safe_fresh {recon['mu_safe_fresh']:.4f} "
          f"(resid {recon['lambda_risk5_roundtrips_mu_safe']:.2e})", flush=True)
    print(f"    risk at the 0.9780 LCB-bar = {recon['risk_at_p95_bar']:.4f}  (< 0.05: the LCB-bar over-protects)", flush=True)
    print(f"    operative construction = {recon['operative_construction']}", flush=True)
    print(f"\n  ROBUSTNESS:", flush=True)
    print(f"    f_priv axis: risk@speed assumed {fpa['accepted_risk_at_speed_gate_assumed']:.4f} -> grounded "
          f"{fpa['accepted_risk_at_speed_gate_grounded']:.4f}  (floor {fpa['floor_assumed']:.4f} -> {fpa['floor_grounded']:.4f})", flush=True)
    print(f"    sigma +-20% sweep: lambda_risk5 monotone={ss['lambda_risk5_monotone_increasing_in_sigma']}  "
          f"risk@speed monotone={ss['risk_at_speed_monotone_increasing_in_sigma']}  direction_stable={ss['verdict_direction_stable']}",
          flush=True)
    for r in ss["rows"]:
        print(f"      sigma*{r['sigma_scale']:.1f}={r['sigma_draw']:.4f}  lambda_risk5={r['lambda_risk5']:.6f}  "
              f"risk@speed={r['accepted_risk_at_speed_gate']:.4f}", flush=True)
    print(f"\n  VERDICT: {recon['verdict']}", flush=True)
    print(f"\n  SELF-TEST (PRIMARY) passes = {st['publishfirst_accepted_risk_self_test_passes']}", flush=True)
    for k, v in st["checks"].items():
        print(f"    [{'ok' if v else '!! FAILED'}] {k}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_HERE, "publishfirst_accepted_risk_results.json"))
    ap.add_argument("--self-test", action="store_true",
                    help="run the self-test (always runs; sets nonzero exit on failure)")
    ap.add_argument("--wandb-name", "--wandb_name", default="kanna/publishfirst-accepted-risk")
    ap.add_argument("--wandb-group", "--wandb_group", default="winners-curse-redraw-budget")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    result = run()

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"[par] wrote {args.out}", flush=True)

    _print(result)
    _log_wandb(args, result)

    if not result["publishfirst_accepted_risk_self_test_passes"]:
        print("[par] SELF-TEST FAILED", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
