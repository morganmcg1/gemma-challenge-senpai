#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Speed margin AT the validity bar: does speed clear at lambda_hat=0.9780? (PR #229).

WHAT THIS IS
------------
My #222 (`yw7i2ece`, MERGED) delivered the launch-gate ORDERING: the VALIDITY gate binds,
not speed -- at the lambda_hat>=0.9780 validity bar the build shows public mu_pub=515.924,
which clears the #218 worst-case speed trigger 513.557 with a +2.367 worst-case margin. But
that +2.367 is LOAD-BEARING for the launch, and the launch-critical scenario is the build
landing at the MINIMUM passing validity lambda_hat (0.9780 EXACTLY), where E[T](lambda_hat)
-- and therefore speed -- is LOWEST. If #222's +2.367 had been scored at a HIGHER lambda_hat
(the spine 0.997 or the ceiling 1.0), the TRUE margin at the validity bar would be thinner,
and there could be a band [0.9780, X) where validity PASSES but speed FAILS -- a second
binding region fern #185 would have to carry.

THE QUESTION (the single launch-confidence-relevant one #222 leaves)
-------------------------------------------------------------------
Is the speed margin still POSITIVE at lambda_hat=0.9780 (the marginal validity lambda) under
the #218 grounded worst-case sigma band [7.6113, 8.2423] -> triggers [512.519 tight, 512.735
central, 513.557 worst]? Either HARDEN "validity binds, speed always clears" or SURFACE the
validity-passes-speed-fails band [0.9780, X).

THE MAP (imported VERBATIM from #222; NOT re-derived)
-----------------------------------------------------
This leg imports #222's `binding_gate` module DIRECTLY and reuses its ceiling-anchored
reach-DP map, so the deliverable round-trips #222 bit-exactly:

    mu_pub_speed(lambda_hat) = K_cal*(E[T](lambda_hat)/step)*tau_anchor
                             = 520.953 * E[T](lambda_hat)/E[T](1)        # identical by anchor

E[T](lambda_hat) is the #213 per-depth linear self-KV-recovery blend of the banked
floor/ceiling spines through the #175/#184 reach-DP (pmf-mean); K_cal=125.268 (#148/#169),
step=1.2182 (#168). tau_anchor folds the int4-ceiling calibration into one multiplier so the
lambda=1 endpoint round-trips the imported #204 int4-spec ceiling 520.953 EXACTLY (both_bugs
tau_anchor=0.9707 anchors the optimistic 536.66 raw both-bugs ceiling DOWN to the conservative
int4 520.953; descent_only tau_anchor=1.0006). The served-fraction tau band [0.9924,1.0] (#181)
is the physical tau; tau_anchor carries it PLUS the int4-ceiling calibration -- imported, not
re-derived. The sigmas, the 0.9780 bar, the [512.519,512.735,513.557] triggers, and the 520.953
ceiling are imported UNCHANGED.

THE DELIVERABLE
---------------
  mu_pub_speed_at_validity_bar = mu_pub_speed(0.9780)                                (the speed)
  speed_margin_at_validity_bar_worstcase = mu_pub_speed(0.9780) - 513.557            (the margin)
  speed_clears_at_validity_bar[corner] for corner in {tight, central, worstcase}     (3 booleans)
  HEADLINE: VALIDITY_BINDS_SPEED_ALWAYS_CLEARS (worst-case margin >= 0 even at lambda=0.9780)
            vs SPEED_FAILS_BAND_ABOVE_VALIDITY (surface X = lambda_speed_clears, the smallest
            lambda where speed clears worst-case -> the band [0.9780, X)).

RECONCILE WITH #222's +2.367 (the honest frame)
-----------------------------------------------
PIN the lambda at which #222's speed_margin_worstcase=2.367 was scored. Reading #222's
`evaluate_regime` (mu_bar = mp.mu_pub(validity_bar)) and its banked result
(mu_pub_at_validity_bar=515.924 at the forward-grid lambda_hat=0.9779783 row,
is_validity_bar=true), #222 ALREADY scored at lambda_hat=0.9780. We DEMONSTRATE the pin:
mu_pub_speed(0.9780) reproduces #222's 515.924 and the +2.367 EXACTLY ->
margin_drop_from_222_to_validity_bar = 0, and the band claim is SETTLED.

SCOPE
-----
LOCAL CPU-ONLY analytic stress-test of #222's speed-clears-at-validity claim at the marginal
validity lambda=0.9780 under #218's grounded worst-case sigma. No GPU / vLLM / HF Job /
submission / served-file change / official draw. BASELINE stays 481.53; greedy/PPL untouched;
adds 0 TPS (PRIMARY = self-test). The E[T](lambda) reach-DP, sigma band, and 513.557 trigger
are imported unchanged. Authorizes nothing. Directly hardens (or sharpens) the launch-gate
ordering for fern #185. NOT a launch. NOT open2.

SELF-TEST (PR step 5 -- PRIMARY)
--------------------------------
(a) mu_pub_speed(1) round-trips the #204 ceiling 520.953 to tol (both regimes; by anchor);
(b) mu_pub_speed(lambda_hat) MONOTONE INCREASING in lambda_hat (speed rises with acceptance);
(c) speed_margin_at_validity_bar MONOTONE DECREASING in sigma_regime (worst-sigma is tightest);
(d) the headline boolean is CONSISTENT with the sign of speed_margin_at_validity_bar_worstcase;
(e) #222 scored at lambda=0.9780 (pinned) -> margin_drop >= 0 and we reproduce 2.367 to tol;
(f) NaN-clean across all reported scalars.
PRIMARY = speed_margin_at_validity_bar_self_test_passes (bool);
TEST    = speed_margin_at_validity_bar_worstcase (float TPS, both_bugs -- the conservative regime).
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
# speed_margin_at_validity_bar -> validity -> research -> repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Source-of-truth artifacts (imported VERBATIM; one source per constant).
# ---------------------------------------------------------------------------
BINDING_GATE_PY = os.path.join(_ROOT, "research/validity/binding_gate/binding_gate.py")
BINDING_GATE_RESULTS = os.path.join(_ROOT, "research/validity/binding_gate/binding_gate_results.json")

# The PR's three lambda evaluation points.
VALIDITY_BAR_NOMINAL = 0.9780      # the marginal passing validity lambda (#191/#208)
SPINE_LAMBDA = 0.997               # PR's high-acceptance comparison point ("the spine 0.997")
CEILING_LAMBDA = 1.0               # the int4-spec ceiling lambda (#204)

HEADLINE_REGIME = "both_bugs"      # the conservative regime (matches #222's TEST regime)
ROUNDTRIP_TOL = 1e-9               # ceiling round-trip tolerance
RECONCILE_TOL = 1e-6               # tolerance for reproducing #222's banked numbers
SIGMA_CORNERS = ("tight", "central", "worstcase")  # ascending sigma


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
# deliverable round-trips #222 bit-exactly (the strongest possible "import, do NOT re-derive").
BG = _import("binding_gate", BINDING_GATE_PY)


# ---------------------------------------------------------------------------
# Step 0 -- import the banked legs (NOT re-derived) + #222's own banked verdict.
# ---------------------------------------------------------------------------
def import_context() -> dict[str, Any]:
    b = BG.import_banked()  # #218 sigma band + triggers + #204 ceiling + #208 bar + #213 spines
    r222 = _load(BINDING_GATE_RESULTS)
    return {
        "banked": b,
        "pr222": {
            "mu_pub_at_validity_bar": float(r222["mu_pub_at_validity_bar"]),       # 515.9240665878908
            "speed_margin_at_validity_bar": float(r222["speed_margin_at_validity_bar"]),  # 2.3666... (worst-case)
            "binding_gate": r222["binding_gate"],                                  # "validity"
            "lambda_hat_speed_worstcase": float(r222["lambda_hat_speed"]),         # 0.9674684694454245
            "lambda_hat_build_target": float(r222["lambda_hat_build_target"]),     # 0.9779783323491393
            "per_corner_margin": {
                c: float(r222["per_regime"][HEADLINE_REGIME]["binding_vs_trigger"][c]
                         ["speed_margin_at_validity_bar"])
                for c in SIGMA_CORNERS
            },
        },
    }


def _sigma_trigger_map(b: dict[str, Any]) -> dict[str, dict[str, float]]:
    """The #218 grounded (sigma, GO-trigger) at each corner; trigger = 500 + z1*sigma."""
    return {
        "tight": {"sigma": b["combined_sigma_tight"], "trigger": b["trigger_tight"]},
        "central": {"sigma": b["combined_sigma_central"], "trigger": b["trigger_central"]},
        "worstcase": {"sigma": b["combined_sigma_worstcase"], "trigger": b["trigger_worstcase"]},
    }


# ---------------------------------------------------------------------------
# Step 1 -- speed at the validity bar vs the spine vs the ceiling (the mechanism).
# ---------------------------------------------------------------------------
def speed_at_lambdas(mp: "BG.LambdaMuMap", ceiling_mu: float, bar_precise: float) -> dict[str, Any]:
    # The validity bar is the PRECISE d198_coupled_bar (#208); 0.9780 is its nominal rounding.
    # #222 scored at the precise bar, so we evaluate there (reproduces 515.924/2.367 exactly) and
    # carry the nominal 0.9780 as a labeled cross-check (the rounding sensitivity).
    pts = {"validity_bar": bar_precise, "validity_bar_nominal": VALIDITY_BAR_NOMINAL,
           "spine": SPINE_LAMBDA, "ceiling": CEILING_LAMBDA}
    rows = {}
    for tag, lam in pts.items():
        mu_ratio = mp.mu_pub(lam)               # 520.953 * E[T](lam)/E[T](1)
        mu_official = mp.official_tps_form(lam)  # K_cal*(E[T](lam)/step)*tau_anchor (identical form)
        rows[tag] = {
            "lambda_hat": lam,
            "E_T": mp.et_of_lambda(lam),
            "mu_pub_speed": mu_ratio,
            "mu_pub_speed_official_form": mu_official,
            "official_form_cross_check_err": abs(mu_official - mu_ratio),
            "gap_below_ceiling": ceiling_mu - mu_ratio,
        }
    return rows


# ---------------------------------------------------------------------------
# Step 2 -- the margin at the validity bar at each sigma corner (the deliverable).
# ---------------------------------------------------------------------------
def margin_at_validity_bar(mp: "BG.LambdaMuMap", sig_trig: dict[str, dict[str, float]],
                           bar_precise: float) -> dict[str, Any]:
    mu_bar = mp.mu_pub(bar_precise)                  # precise d198_coupled_bar (#222 scored here)
    mu_bar_nominal = mp.mu_pub(VALIDITY_BAR_NOMINAL)  # nominal 0.9780 (rounding cross-check)
    corners = {}
    for c in SIGMA_CORNERS:
        trig = sig_trig[c]["trigger"]
        margin = mu_bar - trig
        # smallest lambda where mu_pub_speed clears this trigger (mu monotone increasing).
        lam_clear = mp.solve_lambda_for_mu(trig)
        lam_clear_val = lam_clear if lam_clear is not None else float("nan")
        corners[c] = {
            "sigma": sig_trig[c]["sigma"],
            "trigger": trig,
            "mu_pub_speed_at_validity_bar": mu_bar,
            "speed_margin_at_validity_bar": margin,
            "speed_margin_at_validity_bar_nominal_0p9780": mu_bar_nominal - trig,
            "speed_clears_at_validity_bar": bool(margin >= 0.0),
            "lambda_speed_clears": lam_clear_val,                 # X: smallest lambda clearing speed
            "lambda_speed_clears_roundtrip_mu": (mp.mu_pub(lam_clear_val)
                                                 if _finite(lam_clear_val) else float("nan")),
            "validity_passes_speed_fails_band_above_bar":
                bool(_finite(lam_clear_val) and lam_clear_val > bar_precise),
        }
    worst = corners["worstcase"]
    headline = ("VALIDITY_BINDS_SPEED_ALWAYS_CLEARS"
                if worst["speed_margin_at_validity_bar"] >= 0.0
                else "SPEED_FAILS_BAND_ABOVE_VALIDITY")
    # the band X: only meaningful if worst-case speed FAILS at the bar; else empty (X < bar).
    lam_speed_clears_worst = worst["lambda_speed_clears"]
    band_X = (lam_speed_clears_worst
              if headline == "SPEED_FAILS_BAND_ABOVE_VALIDITY" else float("nan"))
    return {
        "mu_pub_speed_at_validity_bar": mu_bar,
        "corners": corners,
        "speed_margin_at_validity_bar_worstcase": worst["speed_margin_at_validity_bar"],   # TEST
        "speed_margin_at_validity_bar_central": corners["central"]["speed_margin_at_validity_bar"],
        "speed_margin_at_validity_bar_tight": corners["tight"]["speed_margin_at_validity_bar"],
        "speed_clears_at_validity_bar_worstcase": worst["speed_clears_at_validity_bar"],
        "speed_clears_at_validity_bar_all_corners":
            bool(all(corners[c]["speed_clears_at_validity_bar"] for c in SIGMA_CORNERS)),
        "headline": headline,
        "lambda_speed_clears_worstcase": lam_speed_clears_worst,
        "fern_gate": "single (validity; speed always clears)" if headline.startswith("VALIDITY")
                     else "double (validity + speed band [0.9780, X))",
        "band_X_lambda_speed_clears": band_X,
        "no_validity_passes_speed_fails_band":
            bool(not worst["validity_passes_speed_fails_band_above_bar"]),
    }


# ---------------------------------------------------------------------------
# Step 3 -- reconcile with #222's +2.367 (pin the lambda; the honest frame).
# ---------------------------------------------------------------------------
def reconcile_222(mp: "BG.LambdaMuMap", pr222: dict[str, Any],
                  my_margin_worstcase: float, bar_precise: float) -> dict[str, Any]:
    bar = bar_precise                       # the precise d198_coupled_bar #222 scored at
    mu_bar = mp.mu_pub(bar)
    # PIN: #222's mu_pub_at_validity_bar must equal mu_pub(0.9780). Forward + inverse.
    fwd_resid = abs(mu_bar - pr222["mu_pub_at_validity_bar"])
    inv_lambda = mp.solve_lambda_for_mu(pr222["mu_pub_at_validity_bar"])
    inv_lambda_val = inv_lambda if inv_lambda is not None else float("nan")
    inv_resid = abs(inv_lambda_val - bar) if _finite(inv_lambda_val) else float("inf")
    scored_at_validity_bar = bool(fwd_resid < RECONCILE_TOL)
    # margin drop: #222's worst-case margin minus mine at the bar (0 since #222 scored AT the bar).
    margin_drop = pr222["speed_margin_at_validity_bar"] - my_margin_worstcase
    return {
        "pr222_mu_pub_at_validity_bar": pr222["mu_pub_at_validity_bar"],
        "pr222_speed_margin_worstcase": pr222["speed_margin_at_validity_bar"],
        "recomputed_mu_pub_at_validity_bar": mu_bar,
        "recomputed_speed_margin_worstcase": my_margin_worstcase,
        "pin_lambda_222_scored_at": bar if scored_at_validity_bar else inv_lambda_val,
        "pr222_scored_at_validity_bar": scored_at_validity_bar,
        "forward_pin_resid": fwd_resid,
        "inverse_pin_lambda": inv_lambda_val,
        "inverse_pin_resid": inv_resid,
        "margin_drop_from_222_to_validity_bar": margin_drop,
        "roundtrips_222_margin_2p367":
            bool(abs(my_margin_worstcase - pr222["speed_margin_at_validity_bar"]) < RECONCILE_TOL),
        "note": (
            "#222 scored speed_margin_worstcase AT lambda_hat=0.9780 (mu_pub_at_validity_bar="
            "515.924 = mu_pub(0.9780); forward-grid row is_validity_bar=true). So the +2.367 IS "
            "the validity-bar margin, NOT a higher-lambda margin -> margin_drop=0, band SETTLED."
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

    # (a) mu_pub_speed(1) round-trips 520.953 EXACTLY (both regimes; anchoring construction).
    a_errs = {reg: abs(maps[reg].mu_pub(1.0) - ceiling) for reg in BG.REGIMES}
    a_ok = bool(all(e < ROUNDTRIP_TOL for e in a_errs.values()))

    # (b) mu_pub_speed(lambda_hat) MONOTONE INCREASING in lambda_hat (fine grid, both regimes).
    b_mono = {}
    for reg in BG.REGIMES:
        mp = maps[reg]
        grid = [floor + (1.0 - floor) * i / 80.0 for i in range(81)]
        mus = [mp.mu_pub(x) for x in grid]
        diffs = [mus[i + 1] - mus[i] for i in range(len(mus) - 1)]
        b_mono[reg] = {
            "monotone_increasing": bool(all(d >= -1e-12 for d in diffs)),
            "strictly_increasing": bool(all(d > 0.0 for d in diffs)),
        }
    b_ok = bool(all(m["monotone_increasing"] for m in b_mono.values()))

    # (c) speed_margin_at_validity_bar MONOTONE DECREASING in sigma_regime (headline regime).
    corners = per_regime[HEADLINE_REGIME]["corners"]
    sig_seq = [corners[c]["sigma"] for c in SIGMA_CORNERS]           # ascending sigma
    margin_seq = [corners[c]["speed_margin_at_validity_bar"] for c in SIGMA_CORNERS]
    sigma_ascending = all(sig_seq[i] <= sig_seq[i + 1] + 1e-15 for i in range(len(sig_seq) - 1))
    margin_descending = all(margin_seq[i] >= margin_seq[i + 1] - 1e-12 for i in range(len(margin_seq) - 1))
    c_ok = bool(sigma_ascending and margin_descending)

    # (d) headline boolean CONSISTENT with the sign of the worst-case margin (headline regime).
    head = per_regime[HEADLINE_REGIME]
    margin_wc = head["speed_margin_at_validity_bar_worstcase"]
    headline_is_clear = head["headline"] == "VALIDITY_BINDS_SPEED_ALWAYS_CLEARS"
    d_ok = bool(headline_is_clear == (margin_wc >= 0.0))

    # (e) #222 scored at lambda=0.9780 (pinned) -> margin_drop >= 0 and reproduce 2.367 to tol.
    e_ok = bool(recon["pr222_scored_at_validity_bar"]
                and recon["margin_drop_from_222_to_validity_bar"] >= -RECONCILE_TOL
                and recon["roundtrips_222_margin_2p367"])

    # (f) NaN-clean across all reported scalars.
    scalars: list[Any] = [ceiling, floor, margin_wc, *a_errs.values(), *sig_seq, *margin_seq,
                          recon["forward_pin_resid"], recon["margin_drop_from_222_to_validity_bar"]]
    for reg in BG.REGIMES:
        pr = per_regime[reg]
        scalars += [pr["mu_pub_speed_at_validity_bar"],
                    pr["speed_margin_at_validity_bar_worstcase"],
                    pr["speed_margin_at_validity_bar_central"],
                    pr["speed_margin_at_validity_bar_tight"]]
        for c in SIGMA_CORNERS:
            scalars += [pr["corners"][c]["trigger"], pr["corners"][c]["lambda_speed_clears"]]
    f_ok = bool(all(_finite(x) for x in scalars))

    checks = {
        "a_mu_pub_speed_lambda1_roundtrips_ceiling_520p953": a_ok,
        "b_mu_pub_speed_monotone_increasing_in_lambda_hat": b_ok,
        "c_speed_margin_monotone_decreasing_in_sigma": c_ok,
        "d_headline_consistent_with_margin_sign": d_ok,
        "e_222_scored_at_bar_margin_drop_nonneg_roundtrips_2p367": e_ok,
        "f_nan_clean": f_ok,
    }
    passes = bool(all(checks.values()))
    return {
        "speed_margin_at_validity_bar_self_test_passes": passes,          # <-- PRIMARY
        "speed_margin_at_validity_bar_worstcase": margin_wc,              # <-- TEST
        "checks": checks,
        "evidence": {
            "a_ceiling_roundtrip_errs": a_errs,
            "b_monotonicity": b_mono,
            "c_sigma_ascending": sigma_ascending,
            "c_margin_descending": margin_descending,
            "c_sigma_seq": sig_seq,
            "c_margin_seq": margin_seq,
            "d_headline": head["headline"],
            "d_margin_worstcase": margin_wc,
            "e_pr222_scored_at_validity_bar": recon["pr222_scored_at_validity_bar"],
            "e_margin_drop": recon["margin_drop_from_222_to_validity_bar"],
            "n_scalars_checked": len(scalars),
        },
    }


# ---------------------------------------------------------------------------
# Assemble.
# ---------------------------------------------------------------------------
def _build_result(ctx, maps, per_regime, recon, st) -> dict[str, Any]:
    b = ctx["banked"]
    head = per_regime[HEADLINE_REGIME]
    margin_wc = head["speed_margin_at_validity_bar_worstcase"]
    headline = head["headline"]
    fern_gate = head["fern_gate"]
    band_X = head["band_X_lambda_speed_clears"]
    lam_speed_clears_wc = head["lambda_speed_clears_worstcase"]

    handoff = (
        "fern #185: your #222 'validity binds, speed clears' HOLDS at the marginal validity "
        "lambda_hat=0.9780 under the #218 worst-case sigma (8.2423 -> trigger 513.557) with "
        "speed_margin_at_validity_bar=%+0.3f TPS (central %+0.3f, tight %+0.3f) -- the speed "
        "trigger clears with positive margin at the LOWEST passing validity lambda, where speed "
        "is lowest. The acceptance that clears the speed trigger worst-case is lambda=%.4f, BELOW "
        "the 0.9780 bar, so the launch is %s -- there is NO [0.9780, X) validity-passes-speed-"
        "fails band, and fern carries %s."
        % (margin_wc, head["speed_margin_at_validity_bar_central"],
           head["speed_margin_at_validity_bar_tight"], lam_speed_clears_wc,
           "SINGLE-GATED on validity (speed always clears)"
           if headline.startswith("VALIDITY") else "DOUBLE-GATED",
           fern_gate)
    )

    speed_pts = head["speed_pts"]
    return {
        "pr": 229,
        "metric_primary": "speed_margin_at_validity_bar_self_test_passes",
        "metric_test": "speed_margin_at_validity_bar_worstcase",
        "speed_margin_at_validity_bar_self_test_passes":
            st["speed_margin_at_validity_bar_self_test_passes"],
        "speed_margin_at_validity_bar_worstcase": margin_wc,
        "headline": headline,
        "fern_gate": fern_gate,
        "band_X_lambda_speed_clears": band_X,
        "no_validity_passes_speed_fails_band": head["no_validity_passes_speed_fails_band"],
        # step 1 -- speed at the three lambdas (headline regime).
        "mu_pub_speed_at_validity_bar": head["mu_pub_speed_at_validity_bar"],
        "mu_pub_speed_at_spine_0p997": speed_pts["spine"]["mu_pub_speed"],
        "mu_pub_speed_at_ceiling": speed_pts["ceiling"]["mu_pub_speed"],
        "gap_below_ceiling_at_validity_bar": speed_pts["validity_bar"]["gap_below_ceiling"],
        # step 2 -- the margin at each sigma corner (headline regime).
        "speed_margin_at_validity_bar_central": head["speed_margin_at_validity_bar_central"],
        "speed_margin_at_validity_bar_tight": head["speed_margin_at_validity_bar_tight"],
        "speed_clears_at_validity_bar_all_corners": head["speed_clears_at_validity_bar_all_corners"],
        "lambda_speed_clears_worstcase": lam_speed_clears_wc,
        # step 3 -- reconcile with #222.
        "reconcile_222": recon,
        "margin_drop_from_222_to_validity_bar": recon["margin_drop_from_222_to_validity_bar"],
        # per-regime detail.
        "per_regime": per_regime,
        "law": (
            "mu_pub_speed(lambda_hat) = K_cal*(E[T](lambda_hat)/step)*tau_anchor "
            "= 520.953 * E[T](lambda_hat)/E[T](1) (imported #222 map). speed_margin_at_validity_"
            "bar[corner] = mu_pub_speed(0.9780) - GO_trigger[corner], GO_trigger = 500 + "
            "z1*combined_sigma[corner], z1=1.64485. HEADLINE = VALIDITY_BINDS_SPEED_ALWAYS_CLEARS "
            "iff speed_margin_at_validity_bar_worstcase >= 0."
        ),
        "imported": {
            "validity_bar_nominal": VALIDITY_BAR_NOMINAL,
            "validity_bar_precise": b["validity_bar"],
            "spine_lambda": SPINE_LAMBDA,
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
        },
        "honest_band": (
            "(a) E[T](lambda), K_cal=125.268, step=1.2182, tau_anchor, the #218 sigma band "
            "[7.6113,8.2423], the [512.519,512.735,513.557] triggers and the 520.953 ceiling are "
            "imported UNCHANGED. (b) the validity bar 0.9780 is the P95 both-bugs private bar "
            "(#191/#208); if the build lands at the LOWER publish-first floor "
            "(kanna's lambda_floor_publish_first, the #124 gate), the relevant speed margin is "
            "THERE -- we note the cross, we do NOT compute kanna's number. (c) all three sigma "
            "corners are reported so fern picks the conservative default (worst-case 513.557). "
            "(d) evaluating the speed map at the PRIVATE 0.9780 UNDERSTATES true public mu_pub "
            "(public acceptance >= private) -> the +2.367 is itself CONSERVATIVE; true margin is "
            "higher -> speed clears even more robustly."
        ),
        "handoff": handoff,
        "scope": (
            "LOCAL CPU-only analytic stress-test of #222's speed-clears-at-validity claim at the "
            "marginal validity lambda=0.9780 under #218's grounded worst-case sigma. Imports "
            "#222's binding_gate map DIRECTLY (round-trips its 2.367 bit-exactly). No GPU/vLLM/HF "
            "Job/submission/served-file change/official draw. BASELINE stays 481.53; adds 0 TPS "
            "(PRIMARY = self-test); greedy/PPL untouched. The E[T](lambda) reach-DP, sigma band, "
            "and 513.557 trigger imported unchanged. Authorizes nothing. NOT a launch. NOT open2."
        ),
        "public_evidence_used": [
            "ubel #222 (yw7i2ece, binding_gate, MERGED) -- the launch-gate ordering this leg "
            "hardens: binding_gate_is_validity=1, mu_pub_at_validity_bar=515.924, "
            "speed_margin_worstcase=2.367 scored AT lambda_hat=0.9780; its ceiling-anchored "
            "E[T](lambda)->mu_pub reach-DP map is imported DIRECTLY here.",
            "ubel #218 (0ug7vd7d, interleg_rho, MERGED) -- the grounded combined-sigma band "
            "[7.6113,8.2423] and GO-trigger band [512.519 tight, 512.735 central, 513.557 worst] "
            "the margin is scored against; imported unchanged.",
            "ubel #204 (launch_sigma_unit_rebase, MERGED) -- the int4-spec lambda=1 ceiling "
            "mu_pub(1)=520.953 the map's top endpoint is anchored to (round-trip self-test).",
            "stark #208/#191 -- the worst-case private go-bar / P95 both-bugs private validity "
            "bar 0.9780 (the marginal validity lambda this leg evaluates speed at).",
            "wirbel #213/#175/#184 -- the banked per-regime floor/ceiling self-KV-recovery spines "
            "+ the reach-DP E[T](lambda_hat) SHAPE, imported via #222 (reproduces #213's E[T]).",
        ],
        "method": (
            "LOCAL CPU-only analytic synthesis: imports #222's binding_gate ceiling-anchored "
            "reach-DP map directly and re-scores the speed margin AT the marginal validity "
            "lambda=0.9780 against the #218 three-corner sigma triggers; pins the lambda at which "
            "#222's 2.367 was scored (=0.9780) and reproduces it bit-exactly. No GPU/vLLM/HF "
            "Job/submission/served-file change. BASELINE stays 481.53; adds 0 TPS. Greedy/PPL "
            "untouched. NOT a launch."
        ),
        "metrics_nan_clean": 1 if st["checks"]["f_nan_clean"] else 0,
        "self_test": st,
    }


# ---------------------------------------------------------------------------
# W&B logging (mirrors #222; never fatal).
# ---------------------------------------------------------------------------
def _log_wandb(args, result: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[speed-margin-bar] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="launch-sigma-speed-margin-at-validity-bar", agent="ubel",
            name=args.wandb_name or "ubel/speed-margin-at-validity-bar",
            group=args.wandb_group,
            tags=["launch-sigma", "speed-margin-at-validity-bar", "validity-bar", "speed-trigger",
                  "et-lambda-map", "winners-curse-redraw-budget", "pr229"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic",
                    "z_one_sided_p95": result["imported"]["z1_one_sided_p95"],
                    "validity_bar_nominal": VALIDITY_BAR_NOMINAL, "spine_lambda": SPINE_LAMBDA,
                    "trigger_worstcase": result["imported"]["trigger_worstcase"],
                    "trigger_central": result["imported"]["trigger_central"],
                    "trigger_tight": result["imported"]["trigger_tight"],
                    "lambda1_ceiling_mu": result["imported"]["lambda1_ceiling_mu"],
                    "headline_regime": HEADLINE_REGIME,
                    "K_cal": result["imported"]["K_cal"], "step": result["imported"]["step"]},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[speed-margin-bar] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[speed-margin-bar] wandb disabled; skipping", flush=True)
        return
    try:
        st = result["self_test"]
        head = result["per_regime"][HEADLINE_REGIME]
        desc = result["per_regime"]["descent_only"]
        flat = {
            # PRIMARY + TEST
            "speed_margin_at_validity_bar_self_test_passes":
                1.0 if st["speed_margin_at_validity_bar_self_test_passes"] else 0.0,
            "speed_margin_at_validity_bar_worstcase": result["speed_margin_at_validity_bar_worstcase"],
            # headline verdict
            "headline_validity_binds_speed_always_clears":
                1.0 if result["headline"] == "VALIDITY_BINDS_SPEED_ALWAYS_CLEARS" else 0.0,
            "speed_clears_at_validity_bar_all_corners":
                1.0 if result["speed_clears_at_validity_bar_all_corners"] else 0.0,
            "no_validity_passes_speed_fails_band":
                1.0 if result["no_validity_passes_speed_fails_band"] else 0.0,
            "lambda_speed_clears_worstcase": result["lambda_speed_clears_worstcase"],
            # step 1 -- speed at the three lambdas
            "mu_pub_speed_at_validity_bar": result["mu_pub_speed_at_validity_bar"],
            "mu_pub_speed_at_spine_0p997": result["mu_pub_speed_at_spine_0p997"],
            "mu_pub_speed_at_ceiling": result["mu_pub_speed_at_ceiling"],
            "gap_below_ceiling_at_validity_bar": result["gap_below_ceiling_at_validity_bar"],
            # step 2 -- margin per sigma corner
            "speed_margin_at_validity_bar_central": result["speed_margin_at_validity_bar_central"],
            "speed_margin_at_validity_bar_tight": result["speed_margin_at_validity_bar_tight"],
            # step 3 -- reconcile with #222
            "margin_drop_from_222_to_validity_bar": result["margin_drop_from_222_to_validity_bar"],
            "pr222_scored_at_validity_bar":
                1.0 if result["reconcile_222"]["pr222_scored_at_validity_bar"] else 0.0,
            "roundtrips_222_margin_2p367":
                1.0 if result["reconcile_222"]["roundtrips_222_margin_2p367"] else 0.0,
            "pin_lambda_222_scored_at": result["reconcile_222"]["pin_lambda_222_scored_at"],
            # descent_only cross-check
            "speed_margin_at_validity_bar_worstcase_descent_only":
                desc["speed_margin_at_validity_bar_worstcase"],
            "mu_pub_speed_at_validity_bar_descent_only": desc["mu_pub_speed_at_validity_bar"],
            # imported gates
            "trigger_worstcase": result["imported"]["trigger_worstcase"],
            "trigger_central": result["imported"]["trigger_central"],
            "trigger_tight": result["imported"]["trigger_tight"],
            "combined_sigma_worstcase": result["imported"]["combined_sigma_worstcase"],
            "combined_sigma_tight": result["imported"]["combined_sigma_tight"],
            "lambda1_ceiling_mu": result["imported"]["lambda1_ceiling_mu"],
            "tau_anchor_both_bugs": result["imported"]["tau_anchor_both_bugs"],
            # per-check booleans
            "self_test_a_ceiling_roundtrip":
                1.0 if st["checks"]["a_mu_pub_speed_lambda1_roundtrips_ceiling_520p953"] else 0.0,
            "self_test_b_monotone":
                1.0 if st["checks"]["b_mu_pub_speed_monotone_increasing_in_lambda_hat"] else 0.0,
            "self_test_c_margin_monotone_in_sigma":
                1.0 if st["checks"]["c_speed_margin_monotone_decreasing_in_sigma"] else 0.0,
            "self_test_d_headline_consistent":
                1.0 if st["checks"]["d_headline_consistent_with_margin_sign"] else 0.0,
            "self_test_e_222_roundtrip":
                1.0 if st["checks"]["e_222_scored_at_bar_margin_drop_nonneg_roundtrips_2p367"] else 0.0,
            "self_test_f_nan_clean": 1.0 if st["checks"]["f_nan_clean"] else 0.0,
        }
        # per-sigma-corner margin rows (headline regime).
        for c in SIGMA_CORNERS:
            cc = head["corners"][c]
            flat[f"speed_clears_at_validity_bar_{c}"] = 1.0 if cc["speed_clears_at_validity_bar"] else 0.0
            flat[f"speed_margin_{c}"] = cc["speed_margin_at_validity_bar"]
            flat[f"lambda_speed_clears_{c}"] = cc["lambda_speed_clears"]
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="speed_margin_at_validity_bar",
            artifact_type="launch-sigma-speed-margin-at-validity-bar", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[speed-margin-bar] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    st = result["self_test"]
    head = result["per_regime"][HEADLINE_REGIME]
    sp = head["speed_pts"]
    print("\n[speed-margin-bar] ===== SPEED MARGIN AT THE VALIDITY BAR lambda=0.9780 (PR #229) =====",
          flush=True)
    print(f"  map: mu_pub_speed(lambda) = K_cal*(E[T]/step)*tau_anchor = 520.953*E[T](lambda)/E[T](1)  "
          f"(imported #222)", flush=True)
    print("  speed at the three lambdas (both_bugs):", flush=True)
    for tag in ("validity_bar", "spine", "ceiling"):
        r = sp[tag]
        print(f"    {tag:13s} lambda={r['lambda_hat']:.4f}  E[T]={r['E_T']:.5f}  "
              f"mu_pub_speed={r['mu_pub_speed']:8.3f}  gap_below_ceiling={r['gap_below_ceiling']:+.3f}",
              flush=True)
    print(f"  speed_margin_at_validity_bar by sigma corner (both_bugs):", flush=True)
    for c in SIGMA_CORNERS:
        cc = head["corners"][c]
        print(f"    {c:9s} sigma={cc['sigma']:.4f} trig={cc['trigger']:8.3f}  "
              f"margin={cc['speed_margin_at_validity_bar']:+.3f}  clears={cc['speed_clears_at_validity_bar']}  "
              f"lambda_speed_clears={cc['lambda_speed_clears']:.4f}", flush=True)
    print(f"  HEADLINE: {result['headline']}  -> fern carries: {result['fern_gate']}", flush=True)
    print(f"  reconcile #222: scored_at_bar={result['reconcile_222']['pr222_scored_at_validity_bar']}  "
          f"margin_drop={result['margin_drop_from_222_to_validity_bar']:+.6f}  "
          f"roundtrips_2.367={result['reconcile_222']['roundtrips_222_margin_2p367']}", flush=True)
    print(f"\n  SELF-TEST (PRIMARY) passes = {st['speed_margin_at_validity_bar_self_test_passes']}  "
          f"speed_margin_at_validity_bar_worstcase (TEST) = "
          f"{st['speed_margin_at_validity_bar_worstcase']:+.4f} TPS", flush=True)
    for k, v in st["checks"].items():
        print(f"    [{'ok' if v else 'XX'}] {k}", flush=True)


def evaluate_regime(ctx: dict[str, Any], mp: "BG.LambdaMuMap") -> dict[str, Any]:
    b = ctx["banked"]
    bar_precise = b["validity_bar"]
    sig_trig = _sigma_trigger_map(b)
    speed_pts = speed_at_lambdas(mp, b["lambda1_ceiling_mu"], bar_precise)
    margins = margin_at_validity_bar(mp, sig_trig, bar_precise)
    out = dict(margins)
    out["speed_pts"] = speed_pts
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Speed margin at the validity bar lambda=0.9780 (PR #229)")
    ap.add_argument("--out", default=os.path.join(_HERE, "speed_margin_at_validity_bar_results.json"))
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name",
                    default="ubel/speed-margin-at-validity-bar")
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="winners-curse-redraw-budget")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--self-test", action="store_true", help="exit non-zero if the self-test fails")
    args = ap.parse_args(argv)

    t0 = time.time()
    ctx = import_context()
    maps = BG.build_maps(ctx["banked"])
    per_regime = {reg: evaluate_regime(ctx, maps[reg]) for reg in BG.REGIMES}
    recon = reconcile_222(maps[HEADLINE_REGIME], ctx["pr222"],
                          per_regime[HEADLINE_REGIME]["speed_margin_at_validity_bar_worstcase"],
                          ctx["banked"]["validity_bar"])
    st = self_test(ctx, maps, per_regime, recon)

    result = _build_result(ctx, maps, per_regime, recon, st)
    result["elapsed_s"] = round(time.time() - t0, 4)
    result["peak_mem_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 3)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    _print(result)
    print(f"\n[speed-margin-bar] HANDOFF: {result['handoff']}", flush=True)
    print(f"[speed-margin-bar] artifacts -> {args.out}", flush=True)
    _log_wandb(args, result)

    if args.self_test:
        ok = st["speed_margin_at_validity_bar_self_test_passes"] and result["metrics_nan_clean"] == 1
        print(f"[speed-margin-bar] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
