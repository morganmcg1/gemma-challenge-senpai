#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""f_priv-DISTRIBUTION integrated private-draw risk band: one number, not two point estimates (PR #239).

WHAT THIS IS
------------
kanna #237 (`8x7i38jh`, MERGED) delivered the accepted-risk curve but pinned f_priv at TWO point
estimates -- risk(lambda=0.9675)=0.0583 (assumed f_priv=0.969107) and 0.2394 (grounded #224
f_priv=0.957054) -- a ~4x spread it flagged as the binding calibration uncertainty. The human's #124
decision needs ONE integrated number, not two bookends. This leg puts a DISTRIBUTION on f_priv across
its realizable range [grounded 0.957054, assumed 0.969107] and PROPAGATES it through #237's banked
draw model into a SINGLE integrated private-draw risk plus a credible band:

    private_draw(lambda; f_priv) ~ Normal( mu = private_mean(lambda; f_priv),  sigma = sigma_draw )
    risk(lambda; f_priv)         = 1 - Phi( (private_mean(lambda; f_priv) - 500) / sigma_draw )

    integrated_risk(lambda)      = E_{f_priv ~ prior}[ risk(lambda; f_priv) ]            (the new object)

private_mean(lambda; f_priv) is the EXACT #228 central curve imported VERBATIM via #237's
build_composition() (private_mean = public_central(lambda) * f_priv, linear in f_priv); sigma_draw is
#217's fresh private-draw sigma (7.391). NOTHING is re-derived -- the only new object is the
EXPECTATION of #237's risk(lambda; .) over a prior on f_priv.

THE TWO PRIORS (PR step 1 -- stated explicitly)
-----------------------------------------------
The realizable f_priv interval is [f_lo, f_hi] = [grounded #224 0.957054, assumed 0.969107]; we
normalise u = (f_priv - f_lo) / (f_hi - f_lo) in [0, 1] and use the Beta(k, 1) family
(density k*u^{k-1}, CDF u^k, quantile p^{1/k}, mean k/(k+1)) so every quantile is closed-form and no
scipy is needed.

  (a) UNIFORM  (max-entropy / agnostic):           Beta(1, 1)  -> E[f_priv] = midpoint
        The agnostic anchor. Makes no claim about where in [f_lo, f_hi] the truth sits.

  (b) DIVERGENCE-INFORMED (near-greedy shape prior): Beta(2, 1) -> density rises toward f_hi
        lawine #232 (`nxwv6pam`) measured a 0.73% int4 token-divergence (identity 0.9927) -- the
        deployed verify is NEAR-GREEDY, so the int4 decode-drop is SMALL and the realizable f_priv
        mass should lean toward the CLEAN tree-only ceiling f_hi rather than the single noisy grounded
        draw f_lo. Beta(2, 1) is the MINIMAL monotone-increasing density encoding that tilt without
        over-committing to a concentration the 0.73% does NOT pin down (token-divergence and the
        throughput multiplier are different physical quantities). A lean-strength sweep k in
        {1,2,3,5} is reported so the human sees the full sensitivity: as k grows (stronger near-greedy
        belief) integrated_risk -> risk(f_hi) = 0.0583 (#237's optimistic bookend); k=1 (uniform) is
        the agnostic upper anchor.

THE DELIVERABLE
---------------
  integrated_risk_at_speed_gate  E[risk(0.9675; f_priv)]  -- ONE number replacing #237's 0.0583/0.2394
                                 spread; sits strictly BETWEEN the two bookends, nearer 0.0583 under (b).
  + a 5-95% credible band, the integrated_risk(lambda) map over the publish-first band, and
  lambda_integrated_risk5 (the lambda where integrated risk hits 5%) vs #237's point lambda_risk5=0.9700.

LOCAL CPU-only analytic propagation over EXISTING MERGED legs. No GPU / vLLM / HF Job / submission /
served-file change / official draw. BASELINE stays 481.53; greedy/PPL untouched; adds 0 TPS;
authorizes nothing. Imports #237 (VERBATIM, which imports #228/#217/#191/#224/#229). Does NOT re-derive
private_mean, sigma_draw, the f_priv worst-case BLEND (stark #233 = the bar's LOCATION), or the
integration card (fern). The EXPECTATION of risk over a prior on f_priv is the only new object.

PRIMARY metric  fpriv_distribution_risk_self_test_passes
TEST    metric  integrated_risk_at_speed_gate   (uniform prior; the single decision-grade number)
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
# source-of-truth artifact: import #237's composition VERBATIM (which imports #228/#217/#191/#224/#229)
# ---------------------------------------------------------------------------
PAR_237 = os.path.join(
    _ROOT, "research/validity/publishfirst_accepted_risk/publishfirst_accepted_risk.py")

TARGET = 500.0
Z1 = 1.6448536269514722  # one-sided P95 == Phi^{-1}(0.95) (#204); only used for provenance notes

# the publish-first band lambda grid (PR step 3): floor, mid, speed-gate, P95-bar, lambda=1
# floor / speed_gate / p95_bar are filled with the EXACT banked values at runtime.
LAMBDA_GRID_NAMES = ["floor", "0.9500", "speed_gate", "p95_bar", "1.0000"]

# divergence-informed shape prior anchors (lawine #232 `nxwv6pam`, provided in the PR body; NOT
# inspected from its branch -- used as documented scalars to MOTIVATE the lean, not to pin k).
DIVERGENCE_INT4 = 0.0073          # measured int4 token-divergence (near-greedy)
IDENTITY_INT4 = 0.9927            # 1 - divergence == near-greedy identity fraction
K_DIVERGENCE_INFORMED = 2         # Beta(2,1): minimal monotone tilt toward the clean ceiling f_hi
K_LEAN_SWEEP = (1, 2, 3, 5)       # agnostic -> stronger near-greedy belief (sensitivity)

# stark #233 (`pszvrf2a`, provided in PR body) cross-link context (NOT my object -- the bar's LOCATION)
FPRIV_BREAKEVEN_233 = 0.9598      # f_priv worst-case break-even (stark #233)

N_GRID = 8001                     # u-grid points for the prior expectation (trapezoid)
TOL = 1e-6
TOL_XCHK = 1e-9                   # provenance cross-check vs #237 bookends (must be bit-tight)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _finite(x: Any) -> float:
    if x is None or not math.isfinite(float(x)):
        raise ValueError(f"non-finite value: {x!r}")
    return float(x)


def norm_cdf(x: float) -> float:
    """Standard-normal CDF Phi(x) via erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# the imported model (#237): risk(lambda; f_priv) = 1 - Phi((public_central(lambda)*f_priv - 500)/sigma)
# ---------------------------------------------------------------------------
def build_model() -> dict[str, Any]:
    par = _load_mod("publishfirst_accepted_risk", PAR_237)
    comp = par.build_composition()            # #237's composition (imports #228/#217/#191/#224/#229)
    p_clear, risk = par.make_risk(comp)       # risk(lam, f_priv, sigma=None) -- the banked draw model

    f_lo = _finite(comp["f_priv_grounded"])   # 0.9570535584491102  (grounded #224)  -- support min
    f_hi = _finite(comp["f_priv"])            # 0.969106920637722   (assumed)         -- support max
    sigma_draw = _finite(comp["sigma_draw"])  # 7.390974474817942   (#217)
    speed = _finite(comp["lam_speed"])        # 0.9674684694454245  (ubel #229)
    floor = _finite(comp["floor"])            # 0.9138270633254324  (#228, mean==500 @ f_hi)
    p95 = _finite(comp["p95_bar"])            # 0.9780112973731208  (stark #191 LCB-on-lambda bar)

    public_central = comp["public_central"]   # public_central(lambda); private_mean = public*f_priv

    # confirm linearity private_mean(lambda, f) == public_central(lambda)*f at two anchors, then use
    # the fast linear path (faithful to the imported curve, just avoids recomputing public_central).
    for lam in (speed, 1.0):
        pub = public_central(lam)
        for f in (f_lo, f_hi):
            if abs(comp["mean_central"](lam, f) - pub * f) > 1e-9:
                raise AssertionError("private_mean is not public_central*f_priv -- linearity broken")

    def risk_at(pub: float, f_priv: float) -> float:
        """risk for a given public_central(lambda)=pub and f_priv -- identical to #237's risk(lam,f)."""
        return 1.0 - norm_cdf((pub * f_priv - TARGET) / sigma_draw)

    return {
        "comp": comp, "risk": risk, "public_central": public_central, "risk_at": risk_at,
        "f_lo": f_lo, "f_hi": f_hi, "sigma_draw": sigma_draw,
        "speed": speed, "floor": floor, "p95": p95,
        "k_cal": comp["k_cal"], "step_int4": comp["step_int4"],
    }


# ---------------------------------------------------------------------------
# the Beta(k,1) prior family on u=(f_priv-f_lo)/(f_hi-f_lo) (closed-form quantile/mean; trapezoid E)
# ---------------------------------------------------------------------------
def f_of_u(u: float, f_lo: float, f_hi: float) -> float:
    return f_lo + u * (f_hi - f_lo)


def prior_density_u(u: float, k: float) -> float:
    """Beta(k,1) density on [0,1]: k*u^{k-1} (== 1 for k=1; == 0 at u=0 for k>1)."""
    return k * (u ** (k - 1))


def prior_quantile_u(p: float, k: float) -> float:
    """Inverse CDF of Beta(k,1): u such that u^k = p -> p^{1/k} (exact)."""
    return p ** (1.0 / k)


def prior_mean_u(k: float) -> float:
    """E[u] under Beta(k,1) == k/(k+1) (exact)."""
    return k / (k + 1.0)


def _u_grid(n: int) -> list[float]:
    return [i / (n - 1) for i in range(n)]


def integrated_risk(model: dict[str, Any], lam: float, k: float,
                    ugrid: list[float] | None = None) -> float:
    """E_{f_priv ~ Beta(k,1) on [f_lo,f_hi]}[ risk(lam; f_priv) ] by trapezoid over u, normalised."""
    pub = model["public_central"](lam)
    f_lo, f_hi, risk_at = model["f_lo"], model["f_hi"], model["risk_at"]
    ug = ugrid if ugrid is not None else _u_grid(N_GRID)
    num = 0.0
    den = 0.0
    prev_g = risk_at(pub, f_of_u(ug[0], f_lo, f_hi)) * prior_density_u(ug[0], k)
    prev_d = prior_density_u(ug[0], k)
    for i in range(1, len(ug)):
        u = ug[i]
        dens = prior_density_u(u, k)
        g = risk_at(pub, f_of_u(u, f_lo, f_hi)) * dens
        h = u - ug[i - 1]
        num += 0.5 * (g + prev_g) * h
        den += 0.5 * (dens + prev_d) * h
        prev_g, prev_d = g, dens
    return num / den   # normalised weighted average (den ~= 1; divide out quadrature drift)


def credible_band(model: dict[str, Any], lam: float, k: float) -> dict[str, float]:
    """5-95% credible band on risk(lam; f_priv) induced by the f_priv prior (closed-form quantiles).

    risk is monotone DECREASING in f_priv, so the 5th pct of risk is at the 95th pct of f_priv and the
    95th pct of risk is at the 5th pct of f_priv.
    """
    pub = model["public_central"](lam)
    f_lo, f_hi, risk_at = model["f_lo"], model["f_hi"], model["risk_at"]
    u05, u95 = prior_quantile_u(0.05, k), prior_quantile_u(0.95, k)
    f_q05, f_q95 = f_of_u(u05, f_lo, f_hi), f_of_u(u95, f_lo, f_hi)   # low f, high f
    band_lo = risk_at(pub, f_q95)   # high f_priv -> low risk  (5th pct of risk)
    band_hi = risk_at(pub, f_q05)   # low  f_priv -> high risk (95th pct of risk)
    return {"band_lo": band_lo, "band_hi": band_hi, "f_q05": f_q05, "f_q95": f_q95}


def solve_lambda_integrated_risk5(model: dict[str, Any], k: float,
                                  target_risk: float = 0.05) -> float | None:
    """The lambda at which integrated_risk(lambda; k) == target_risk, by bisection on [floor, 1.0].

    integrated_risk is monotone DECREASING in lambda, so a unique root exists when it brackets.
    """
    lo, hi = model["floor"], 1.0
    ug = _u_grid(N_GRID)
    f = lambda lam: integrated_risk(model, lam, k, ugrid=ug) - target_risk
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if (flo > 0.0) == (fhi > 0.0):
        return None   # no sign change -> target not bracketed in [floor, 1]
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < 1e-13 or (hi - lo) < 1e-13:
            return mid
        if (fm > 0.0) == (flo > 0.0):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# step 2-3: the integrated-risk lambda map + speed-gate headline (both priors)
# ---------------------------------------------------------------------------
def lambda_map(model: dict[str, Any]) -> dict[str, Any]:
    floor, speed, p95 = model["floor"], model["speed"], model["p95"]
    lambdas = [floor, 0.95, speed, p95, 1.0]
    ug = _u_grid(N_GRID)
    ku, kd = 1, K_DIVERGENCE_INFORMED   # uniform, divergence-informed

    def regime(lam: float) -> str:
        if lam >= p95:
            return "both_clear"
        if lam >= floor:
            return "publish_first_go_p95_hold"
        return "publish_first_no_go"

    rows = []
    for lam, name in zip(lambdas, LAMBDA_GRID_NAMES):
        ir_u = integrated_risk(model, lam, ku, ugrid=ug)
        ir_d = integrated_risk(model, lam, kd, ugrid=ug)
        bu = credible_band(model, lam, ku)
        bd = credible_band(model, lam, kd)
        rows.append({
            "name": name, "lambda": lam, "regime": regime(lam),
            "integrated_risk_uniform": ir_u,
            "band_uniform": [bu["band_lo"], bu["band_hi"]],
            "integrated_risk_divinformed": ir_d,
            "band_divinformed": [bd["band_lo"], bd["band_hi"]],
            "f_priv_q05": bu["f_q05"], "f_priv_q95": bu["f_q95"],
        })
    return {"prior_uniform_k": ku, "prior_divinformed_k": kd, "rows": rows}


def lean_sweep(model: dict[str, Any]) -> dict[str, Any]:
    """How integrated_risk_at_speed_gate moves as the near-greedy lean sharpens (k=1..5)."""
    speed, f_hi = model["speed"], model["f_hi"]
    pub = model["public_central"](speed)
    risk_at = model["risk_at"]
    ug = _u_grid(N_GRID)
    rows = []
    for k in K_LEAN_SWEEP:
        rows.append({
            "k": k,
            "prior": "uniform (agnostic)" if k == 1 else f"divergence-informed Beta({k},1)",
            "E_f_priv": f_of_u(prior_mean_u(k), model["f_lo"], f_hi),
            "integrated_risk_at_speed_gate": integrated_risk(model, speed, k, ugrid=ug),
        })
    return {
        "rows": rows,
        "limit_k_to_inf": risk_at(pub, f_hi),   # full near-greedy belief -> #237 assumed bookend 0.0583
        "note": (
            "Monotone DOWN in k: stronger near-greedy belief pushes f_priv mass toward the clean "
            "ceiling f_hi, so integrated_risk_at_speed_gate falls from the uniform/agnostic value "
            "toward risk(f_hi)=0.0583 (#237's assumed bookend). k is a modeling choice the 0.73% "
            "MOTIVATES but does NOT pin; the headline divergence-informed prior uses the minimal "
            f"tilt k={K_DIVERGENCE_INFORMED}."
        ),
    }


# ---------------------------------------------------------------------------
# step 4: self-test (PRIMARY)
# ---------------------------------------------------------------------------
def self_test(model: dict[str, Any], mp: dict[str, Any],
              speed_head: dict[str, Any], floor_head: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    evid: dict[str, Any] = {}
    f_lo, f_hi, speed = model["f_lo"], model["f_hi"], model["speed"]
    risk_at, pub_speed = model["risk_at"], model["public_central"](model["speed"])
    ug = _u_grid(N_GRID)

    # provenance: the two #237 bookends are risk at the support endpoints (must be bit-exact).
    bookend_assumed = risk_at(pub_speed, f_hi)    # == #237 accepted_risk_at_speed_gate 0.0583
    bookend_grounded = risk_at(pub_speed, f_lo)   # == #237 grounded 0.2394
    evid["bookend_assumed"] = bookend_assumed
    evid["bookend_grounded"] = bookend_grounded
    checks["g_bookends_match_237"] = bool(
        abs(bookend_assumed - 0.05831773945416474) < TOL_XCHK
        and abs(bookend_grounded - 0.2394311004235805) < TOL_XCHK)

    # (a) integrated_risk_at_speed_gate (uniform) strictly between the two #237 bookends.
    ir_u = speed_head["integrated_risk_uniform"]
    checks["a_uniform_between_bookends"] = bool(bookend_assumed < ir_u < bookend_grounded)
    evid["a_integrated_risk_speed_uniform"] = ir_u
    evid["a_bookend_lo"] = bookend_assumed
    evid["a_bookend_hi"] = bookend_grounded

    # (b) divergence-informed gives a LOWER integrated risk than uniform (mass leans to clean f_hi).
    ir_d = speed_head["integrated_risk_divinformed"]
    checks["b_divinformed_below_uniform"] = bool(ir_d < ir_u)
    evid["b_integrated_risk_speed_divinformed"] = ir_d
    evid["b_gap_uniform_minus_divinformed"] = ir_u - ir_d

    # (c) at lambda=floor integrated risk is in the coin-flip-or-worse region (>= 0.5).
    #     the floor was LOCATED at the optimistic assumed f_hi (mean==500), so every realizable
    #     f_priv <= f_hi makes the floor sub-500 -> risk strictly above 0.5.
    ir_floor = floor_head["integrated_risk_uniform"]
    checks["c_floor_coinflip_or_worse"] = bool((0.5 - TOL) <= ir_floor < 0.85)
    evid["c_integrated_risk_floor_uniform"] = ir_floor

    # (d) integrated_risk(lambda) monotone DECREASING in lambda (both priors), on a fine grid.
    lam_grid = [model["floor"] + (1.0 - model["floor"]) * j / 40.0 for j in range(41)]
    def _monotone_dec(k: float) -> bool:
        seq = [integrated_risk(model, lam, k, ugrid=ug) for lam in lam_grid]
        return all(seq[i] > seq[i + 1] for i in range(len(seq) - 1))
    dec_u = _monotone_dec(1)
    dec_d = _monotone_dec(K_DIVERGENCE_INFORMED)
    checks["d_monotone_decreasing_in_lambda"] = bool(dec_u and dec_d)
    evid["d_monotone_uniform"] = bool(dec_u)
    evid["d_monotone_divinformed"] = bool(dec_d)

    # (e) the credible band brackets the integrated point estimate at every grid lambda, both priors.
    bracket_ok = True
    for r in mp["rows"]:
        lo_u, hi_u = r["band_uniform"]
        lo_d, hi_d = r["band_divinformed"]
        if not (lo_u <= r["integrated_risk_uniform"] <= hi_u):
            bracket_ok = False
        if not (lo_d <= r["integrated_risk_divinformed"] <= hi_d):
            bracket_ok = False
    checks["e_band_brackets_point"] = bool(bracket_ok)

    # (f) NaN-clean: every reported scalar finite.
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
    checks["f_nan_clean"] = bool(_all_finite(mp) and _all_finite(speed_head) and _all_finite(floor_head))

    passes = all(checks.values())
    return {
        "fpriv_distribution_risk_self_test_passes": bool(passes),
        "checks": checks, "evidence": evid, "n_checks": len(checks),
    }


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------
def run() -> dict[str, Any]:
    t0 = time.time()
    model = build_model()
    f_lo, f_hi, speed, floor, p95 = (
        model["f_lo"], model["f_hi"], model["speed"], model["floor"], model["p95"])
    ku, kd = 1, K_DIVERGENCE_INFORMED

    mp = lambda_map(model)
    speed_head = next(r for r in mp["rows"] if r["name"] == "speed_gate")
    floor_head = next(r for r in mp["rows"] if r["name"] == "floor")
    sweep = lean_sweep(model)
    st = self_test(model, mp, speed_head, floor_head)

    # the deliverables
    integrated_risk_at_speed_gate = speed_head["integrated_risk_uniform"]                 # TEST metric
    integrated_risk_at_speed_gate_divinformed = speed_head["integrated_risk_divinformed"]
    band_speed_uniform = speed_head["band_uniform"]
    band_speed_divinformed = speed_head["band_divinformed"]
    lambda_integrated_risk5_uniform = solve_lambda_integrated_risk5(model, ku)
    lambda_integrated_risk5_divinformed = solve_lambda_integrated_risk5(model, kd)

    # stark #233 cross-link context (NOT my object): where the break-even f_priv sits on my risk axis.
    pub_speed = model["public_central"](speed)
    u_breakeven = (FPRIV_BREAKEVEN_233 - f_lo) / (f_hi - f_lo)
    risk_at_breakeven_speed = model["risk_at"](pub_speed, FPRIV_BREAKEVEN_233)

    peak_mem_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    lr5_u = lambda_integrated_risk5_uniform
    lr5_d = lambda_integrated_risk5_divinformed
    handoff = (
        f"fern decision card + Issue #124: integrating over f_priv in [{f_lo:.6f} grounded #224, "
        f"{f_hi:.6f} assumed] gives ONE accepted private-draw risk at the operative gate lambda=0.9675 "
        f"of integrated_risk_at_speed_gate={integrated_risk_at_speed_gate:.4f} (uniform) / "
        f"{integrated_risk_at_speed_gate_divinformed:.4f} (divergence-informed, lower), with a 5-95% "
        f"band of [{band_speed_uniform[0]:.4f},{band_speed_uniform[1]:.4f}] (uniform) -- replacing "
        f"#237's 4x two-point spread (0.0583/0.2394) with a single decision-grade number, and lawine "
        f"#232's 0.73% near-greedy verify pulls it toward the optimistic end. The 5%-integrated-risk "
        f"build lambda is lambda_integrated_risk5={lr5_u:.4f} (uniform) / {lr5_d:.4f} "
        f"(divergence-informed) vs #237's point lambda_risk5=0.9700 -- the f_priv uncertainty RAISES "
        f"the lambda needed to hold 5% draw risk above the point estimate."
    )

    result = {
        "pr": 239,
        "metric_primary": "fpriv_distribution_risk_self_test_passes",
        "metric_test": "integrated_risk_at_speed_gate",
        "fpriv_distribution_risk_self_test_passes":
            st["fpriv_distribution_risk_self_test_passes"],
        # ---- the single decision-grade deliverable (PR step 2) ----
        "integrated_risk_at_speed_gate": integrated_risk_at_speed_gate,                   # uniform
        "integrated_risk_at_speed_gate_divinformed": integrated_risk_at_speed_gate_divinformed,
        "credible_band_speed_uniform": band_speed_uniform,                               # [5%,95%]
        "credible_band_speed_divinformed": band_speed_divinformed,
        # ---- #237 two-point bookends this leg integrates over ----
        "bookend_assumed_0p969": st["evidence"]["bookend_assumed"],                       # 0.0583
        "bookend_grounded_0p957": st["evidence"]["bookend_grounded"],                     # 0.2394
        "integrated_between_bookends": bool(
            st["evidence"]["bookend_assumed"] < integrated_risk_at_speed_gate
            < st["evidence"]["bookend_grounded"]),
        # ---- the lambda where integrated risk hits 5% (PR step 3) ----
        "lambda_integrated_risk5_uniform": lambda_integrated_risk5_uniform,
        "lambda_integrated_risk5_divinformed": lambda_integrated_risk5_divinformed,
        "lambda_risk5_point_237": 0.9699990336265527,
        "lambda_integrated_risk5_uniform_minus_237": (
            None if lr5_u is None else lr5_u - 0.9699990336265527),
        # ---- the support + priors ----
        "f_priv_support": [f_lo, f_hi],
        "f_priv_support_rounded": [0.957, 0.969],
        "prior_uniform_k": ku,
        "prior_divinformed_k": kd,
        "divergence_int4": DIVERGENCE_INT4,
        "identity_int4": IDENTITY_INT4,
        # ---- the gates / draw model frame ----
        "lambda_floor_publish_first": floor,
        "lambda_speed_gate": speed,
        "p95_private_bar": p95,
        "sigma_draw": model["sigma_draw"],
        "draw_model": {
            "law": "private_draw(lambda; f_priv) ~ Normal(mu=public_central(lambda)*f_priv, sigma=sigma_draw)",
            "clear_event": "private_draw >= 500",
            "risk": "1 - Phi((public_central(lambda)*f_priv - 500)/sigma_draw)",
            "integrated_risk": "E_{f_priv~prior}[ risk(lambda; f_priv) ]",
            "sigma_draw": model["sigma_draw"],
            "f_priv_lo_grounded": f_lo,
            "f_priv_hi_assumed": f_hi,
            "prior_family": "Beta(k,1) on u=(f_priv-f_lo)/(f_hi-f_lo): density k*u^{k-1}, quantile p^{1/k}",
        },
        # ---- stark #233 cross-link (context only; NOT my object) ----
        "stark233_crosslink": {
            "f_priv_breakeven_233": FPRIV_BREAKEVEN_233,
            "u_breakeven_in_support": u_breakeven,
            "risk_at_breakeven_speed_gate": risk_at_breakeven_speed,
            "note": (
                "stark #233's f_priv_breakeven=0.9598 sits at u={:.3f} of my realizable support; the "
                "draw-risk there at the speed gate is {:.4f}. This is a CONTEXT cross-link to the bar's "
                "LOCATION (stark's worst-case BLEND), NOT my object -- I own the draw-risk DISTRIBUTION "
                "around the f_priv point.".format(u_breakeven, risk_at_breakeven_speed)),
        },
        # ---- sections ----
        "lambda_map": mp,
        "lean_sweep": sweep,
        "self_test": st,
        # ---- composition provenance (imported VERBATIM, not re-derived) ----
        "import_banked": {
            "model_src": "kanna #237 publishfirst_accepted_risk (8x7i38jh, module import: "
                         "build_composition + make_risk)",
            "private_mean_curve_src": "kanna #228 publish_first_lambda_floor (via #237)",
            "sigma_draw_src": "kanna #217 trigger_reconcile (vgovdrjc, via #237)",
            "p95_lcb_bar_src": "stark #191 private_build_bar (via #237)",
            "f_priv_grounded_src": "kanna #224 private_bar_reachability (1081oc84, via #237)",
            "speed_gate_src": "ubel #229 speed_margin_at_validity_bar (bz2b3fw8, via #237)",
            "divergence_src": "lawine #232 int4 token-identity (nxwv6pam): 0.73% divergence / 0.9927 identity",
            "fpriv_breakeven_src": "stark #233 private_fpriv_worstcase (pszvrf2a): 0.9598 (context only)",
            "k_cal": model["k_cal"], "step_int4": model["step_int4"],
            "sigma_draw": model["sigma_draw"],
            "f_priv_assumed": f_hi, "f_priv_grounded": f_lo,
        },
        "handoff": handoff,
        "scope": (
            "LOCAL CPU-only analytic propagation of a DISTRIBUTION on f_priv through kanna #237's banked "
            "draw model into ONE integrated private-draw risk band. Takes NO official draws, authorizes "
            "none. BASELINE stays 481.53; adds 0 TPS; greedy/PPL untouched. Imports #237 VERBATIM (which "
            "imports #228/#217/#191/#224/#229); does NOT re-derive private_mean, sigma_draw, the f_priv "
            "worst-case BLEND (stark #233 = the bar's LOCATION), or the integration card (fern). The "
            "EXPECTATION of #237's risk(lambda;.) over a prior on f_priv is the only new object. NOT a "
            "launch. NOT open2."
        ),
        "public_evidence_used": [
            "kanna #237 (publishfirst_accepted_risk, `8x7i38jh`): the risk(lambda; f_priv) draw model, "
            "sigma_draw=7.391, the 0.0583/0.2394 two-point bookends this leg integrates over, and "
            "lambda_risk5=0.9700 -- imported as a module (bit-exact, the bookends reproduce to <1e-9).",
            "kanna #228 (publish_first_lambda_floor, MERGED `352ifoi8`): private_mean(lambda)=public_"
            "central(lambda)*f_priv central curve and the publish-first band [0.913827, 0.978011) (via #237).",
            "kanna #224 (private_bar_reachability, MERGED `1081oc84`): grounded f_priv=0.957054 -- the "
            "support's LOW endpoint.",
            "lawine #232 (int4 token-identity, `nxwv6pam`): 0.73% int4 token-divergence / 0.9927 identity "
            "-- the near-greedy shape prior that leans the f_priv mass toward the clean ceiling f_hi.",
            "stark #233 (private_fpriv_worstcase, `pszvrf2a`): realizable f_priv worst-case [0.957,0.969] "
            "and break-even 0.9598 -- the support endpoints (context cross-link; stark owns the BLEND).",
            "ubel #229 (speed_margin_at_validity_bar, `bz2b3fw8`): the operative public gate lambda=0.9675 "
            "(via #237) -- where the headline integrated risk is reported.",
            "Issue #124 (publish-first green-light): the decision this leg supplies a single integrated "
            "risk number for (replacing #237's 4x two-point spread).",
        ],
        "method": (
            "LOCAL CPU-only analytic propagation over EXISTING MERGED legs (the #237 module, which imports "
            "the #228 private-mean curve + #217/#191/#224/#229 scalars). Beta(k,1) prior on f_priv; "
            "E[risk] by normalised trapezoid (N=8001) over u; closed-form quantile bands. No GPU/vLLM/HF "
            "Job/submission/served-file/draw. BASELINE stays 481.53; adds 0 TPS. Greedy identity untouched. "
            "NOT a launch."
        ),
        "metrics_nan_clean": 1 if st["checks"]["f_nan_clean"] else 0,
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
            name=args.wandb_name or "kanna/fpriv-distribution-risk",
            group=args.wandb_group,
            tags=["f-priv-distribution", "integrated-risk", "credible-band", "draw-risk",
                  "winners-curse", "publish-first", "pr239"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic", "target_tps": 500.0,
                    "f_priv_support": result["f_priv_support"],
                    "prior_uniform_k": result["prior_uniform_k"],
                    "prior_divinformed_k": result["prior_divinformed_k"],
                    "divergence_int4": result["divergence_int4"],
                    "imports_pr": [237, 228, 217, 191, 224, 233, 232, 229, 124]},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[par] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[par] wandb disabled; skipping", flush=True)
        return
    try:
        flat = {
            "fpriv_distribution_risk_self_test_passes":
                1.0 if result["fpriv_distribution_risk_self_test_passes"] else 0.0,
            # the deliverables
            "integrated_risk_at_speed_gate": result["integrated_risk_at_speed_gate"],
            "integrated_risk_at_speed_gate_divinformed":
                result["integrated_risk_at_speed_gate_divinformed"],
            "credible_band_speed_uniform_lo": result["credible_band_speed_uniform"][0],
            "credible_band_speed_uniform_hi": result["credible_band_speed_uniform"][1],
            "credible_band_speed_divinformed_lo": result["credible_band_speed_divinformed"][0],
            "credible_band_speed_divinformed_hi": result["credible_band_speed_divinformed"][1],
            "bookend_assumed_0p969": result["bookend_assumed_0p969"],
            "bookend_grounded_0p957": result["bookend_grounded_0p957"],
            "integrated_between_bookends": 1.0 if result["integrated_between_bookends"] else 0.0,
            "lambda_integrated_risk5_uniform": result["lambda_integrated_risk5_uniform"],
            "lambda_integrated_risk5_divinformed": result["lambda_integrated_risk5_divinformed"],
            "lambda_risk5_point_237": result["lambda_risk5_point_237"],
            "lambda_integrated_risk5_uniform_minus_237":
                result["lambda_integrated_risk5_uniform_minus_237"],
            # gates / model
            "lambda_floor_publish_first": result["lambda_floor_publish_first"],
            "lambda_speed_gate": result["lambda_speed_gate"],
            "p95_private_bar": result["p95_private_bar"],
            "sigma_draw": result["sigma_draw"],
            "divergence_int4": result["divergence_int4"],
            "identity_int4": result["identity_int4"],
            "risk_at_breakeven_speed_gate": result["stark233_crosslink"]["risk_at_breakeven_speed_gate"],
            "metrics_nan_clean": float(result["metrics_nan_clean"]),
            "peak_mem_mib": result["peak_mem_mib"],
        }
        # per-lambda integrated-risk map as a wandb Table for plotting
        try:
            import wandb
            tbl = wandb.Table(columns=[
                "name", "lambda", "integrated_risk_uniform", "band_uniform_lo", "band_uniform_hi",
                "integrated_risk_divinformed", "band_divinformed_lo", "band_divinformed_hi", "regime"])
            for r in result["lambda_map"]["rows"]:
                tbl.add_data(r["name"], r["lambda"], r["integrated_risk_uniform"],
                             r["band_uniform"][0], r["band_uniform"][1],
                             r["integrated_risk_divinformed"], r["band_divinformed"][0],
                             r["band_divinformed"][1], r["regime"])
            flat["integrated_risk_map_table"] = tbl
            swp = wandb.Table(columns=["k", "prior", "E_f_priv", "integrated_risk_at_speed_gate"])
            for r in result["lean_sweep"]["rows"]:
                swp.add_data(r["k"], r["prior"], r["E_f_priv"], r["integrated_risk_at_speed_gate"])
            flat["lean_sweep_table"] = swp
        except Exception as exc:  # noqa: BLE001
            print(f"[par] wandb table skipped ({exc})", flush=True)
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="fpriv_distribution_risk",
            artifact_type="winners-curse-redraw-budget", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[par] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    st = result["self_test"]
    print("\n[par] ===== f_priv-DISTRIBUTION INTEGRATED PRIVATE-DRAW RISK  "
          "integrated_risk(lambda)=E_{f_priv~prior}[risk(lambda;f_priv)]  (PR #239) =====", flush=True)
    print(f"  support f_priv in [{result['f_priv_support'][0]:.6f} grounded #224, "
          f"{result['f_priv_support'][1]:.6f} assumed]   sigma_draw={result['sigma_draw']:.4f}", flush=True)
    print(f"  priors: uniform Beta(1,1) | divergence-informed Beta({result['prior_divinformed_k']},1) "
          f"(lawine #232 0.73% near-greedy -> lean to clean ceiling)", flush=True)
    print(f"\n  integrated_risk(lambda) MAP  (uniform [band] | divergence-informed [band]):", flush=True)
    for r in result["lambda_map"]["rows"]:
        print(f"    {r['name']:>10}  lambda={r['lambda']:.6f}  "
              f"U={r['integrated_risk_uniform']:.4f} [{r['band_uniform'][0]:.4f},{r['band_uniform'][1]:.4f}]  "
              f"| D={r['integrated_risk_divinformed']:.4f} [{r['band_divinformed'][0]:.4f},{r['band_divinformed'][1]:.4f}]  "
              f"[{r['regime']}]", flush=True)
    print(f"\n  HEADLINE  integrated_risk_at_speed_gate (lambda=0.9675):", flush=True)
    print(f"    uniform            = {result['integrated_risk_at_speed_gate']:.6f}  "
          f"band [{result['credible_band_speed_uniform'][0]:.4f},{result['credible_band_speed_uniform'][1]:.4f}]", flush=True)
    print(f"    divergence-informed= {result['integrated_risk_at_speed_gate_divinformed']:.6f}  "
          f"band [{result['credible_band_speed_divinformed'][0]:.4f},{result['credible_band_speed_divinformed'][1]:.4f}]", flush=True)
    print(f"    #237 bookends: assumed {result['bookend_assumed_0p969']:.4f} | grounded "
          f"{result['bookend_grounded_0p957']:.4f}   integrated_between_bookends="
          f"{result['integrated_between_bookends']}", flush=True)
    print(f"\n  lambda_integrated_risk5:  uniform={result['lambda_integrated_risk5_uniform']:.6f}  "
          f"divergence-informed={result['lambda_integrated_risk5_divinformed']:.6f}  "
          f"(vs #237 point lambda_risk5=0.9700)", flush=True)
    print(f"\n  LEAN SWEEP (integrated_risk_at_speed_gate vs near-greedy belief k):", flush=True)
    for r in result["lean_sweep"]["rows"]:
        print(f"    k={r['k']}  {r['prior']:<28}  E[f_priv]={r['E_f_priv']:.6f}  "
              f"risk={r['integrated_risk_at_speed_gate']:.6f}", flush=True)
    print(f"    k->inf (full near-greedy) -> {result['lean_sweep']['limit_k_to_inf']:.6f} "
          f"(== #237 assumed bookend)", flush=True)
    print(f"\n  SELF-TEST (PRIMARY) passes = {st['fpriv_distribution_risk_self_test_passes']}", flush=True)
    for k, v in st["checks"].items():
        print(f"    [{'ok' if v else '!! FAILED'}] {k}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_HERE, "fpriv_distribution_risk_results.json"))
    ap.add_argument("--self-test", action="store_true",
                    help="run the self-test (always runs; sets nonzero exit on failure)")
    ap.add_argument("--wandb-name", "--wandb_name", default="kanna/fpriv-distribution-risk")
    ap.add_argument("--wandb-group", "--wandb_group", default="winners-curse-redraw-budget")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    result = run()

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"[par] wrote {args.out}", flush=True)

    _print(result)
    _log_wandb(args, result)

    if not result["fpriv_distribution_risk_self_test_passes"]:
        print("[par] SELF-TEST FAILED", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
