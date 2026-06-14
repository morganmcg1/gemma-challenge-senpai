#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""TPS-vs-private-risk exchange rate along the build-lambda axis (PR #240).

WHAT THIS IS
------------
The launch-sigma lane has priced TPS(lambda) and private-draw risk(lambda) on SEPARATE axes.
Issue #124's actual decision is the TRADE between them: "is the extra speed of a given build worth
its private-draw risk?". This leg composes the two banked curves into the EXCHANGE RATE

    dTPS/drisk(lambda) = (dTPS/dlambda) / (drisk/dlambda)        # TPS per unit private-draw risk

along the build-lambda axis, so the human can read the speed/risk relationship as a SLOPE rather
than two unrelated numbers. CPU-only, 0 TPS, no draw, no launch -- BANK-THE-ANALYSIS.

THE TWO CURVES (imported VERBATIM as modules; NOT re-derived)
------------------------------------------------------------
  TPS(lambda)  = my #234 / #222 public-speed map: ubel binding_gate
                 `mu_pub(lambda) = 520.953 * E[T](lambda)/E[T](1) = K_cal*(E[T](lambda)/step)*tau`
                 (K_cal=125.268, step=1.2182, both_bugs regime; round-trips the lambda=1 ceiling
                 520.953 EXACTLY). Evaluable on [lambda_floor=0.342, 1.0].
  risk(lambda) = kanna #237 accepted-risk curve: publishfirst_accepted_risk
                 `risk(lambda) = 1 - Phi((private_mean(lambda) - 500)/sigma_draw)`, assumed and
                 grounded f_priv (the #224 calibration axis). risk is the chance the PRIVATE TPS
                 draw fails to clear its 500 bar.

Both modules are imported and called directly, so TPS(lambda) round-trips #234's anchors and
risk(lambda) round-trips #237's curve BIT-FOR-BIT (a provenance lock; the only new object is the
ratio of their lambda-derivatives).

THE HONEST FINDING (the headline -- it CONTRADICTS the PR's directional premise)
--------------------------------------------------------------------------------
The PR hypothesis assumes "a lower-lambda build buys more TPS but accepts more private-draw risk"
(self-test (b): "TPS monotone DOWN in lambda"). The banked maps say the OPPOSITE about TPS:

    dTPS/dlambda  > 0  EVERYWHERE  (higher acceptance lambda -> higher E[T] -> higher public TPS)
    drisk/dlambda < 0  EVERYWHERE  (higher acceptance lambda -> higher private mean -> lower risk)

So along the build-lambda axis TPS and private-clearance are CO-MONOTONE -- both IMPROVE as lambda
rises -- and therefore

    dTPS/drisk(lambda) < 0  EVERYWHERE.

There is NO speed-for-risk trade to optimise on this axis: lowering lambda to "buy speed" actually
LOSES TPS *and* ADDS risk. Concretely, dropping from the 0.9780 P95 bar to the 0.9675 speed gate
LOSES ~2.37 public TPS while RAISING accepted private-draw risk by ~2.9pp -- strictly dominated.
The headline `tps_per_pct_risk_at_speed_gate` is therefore NEGATIVE (~-0.64 TPS/pp, assumed
f_priv): each +1pp of accepted risk near the gate comes with a TPS LOSS, not a gain. The most
TPS-efficient AND least-risky publish-first build within [0.9138, 0.9780] is the TOP of the band
(efficient_lambda = 0.9780), not an interior point -- because a dominated axis has no interior
frontier optimum. The genuine #124 tension is build-lambda vs. how HARD a high-lambda build is to
land (land #71), which lives OUTSIDE this speed/risk composition.

This leg reports the exchange rate faithfully (negative) and flags the premise contradiction; it
does NOT bend the imported maps to manufacture a positive slope.

SCOPE
-----
LOCAL CPU-only analytic composition over EXISTING legs (#234/#222 mu_pub map + #237 risk curve).
No GPU / vLLM / HF Job / submission / served-file change / official draw. BASELINE stays 481.53;
adds 0 TPS (PRIMARY = self-test); greedy/PPL untouched. Authorizes nothing. NOT a launch. NOT open2.

SELF-TEST (PR step 4 -- PRIMARY)
-------------------------------
(a) TPS(lambda) round-trips the composition anchors: ceiling 520.953 @lambda=1 (and #234's
    513.557 @ the speed gate); risk(lambda) round-trips #237's curve at the grid;
(b) OBSERVED monotonicity: TPS strictly INCREASING in lambda AND risk strictly DECREASING in
    lambda over the grid  =>  dTPS/drisk < 0 EVERYWHERE (definite sign). NOTE: this is the
    OPPOSITE TPS direction to the PR's assumed "TPS down in lambda"; we record
    `pr_premise_tps_decreasing_in_lambda_holds = False`;
(c) the exchange rate under grounded f_priv DIFFERS from assumed (grounded risk is steeper);
(d) efficient_lambda in the publish-first band [0.9138, 0.9780];
(e) NaN-clean across all reported scalars.
PRIMARY = tps_risk_exchange_rate_self_test_passes (bool);
TEST    = tps_per_pct_risk_at_speed_gate (float TPS/pp, assumed f_priv, at lambda=0.9675).
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
# tps_risk_exchange_rate -> validity -> research -> repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# source-of-truth artifacts (imported verbatim; one source per curve).
# ---------------------------------------------------------------------------
BINDING_GATE_PY = os.path.join(_ROOT, "research/validity/binding_gate/binding_gate.py")
ACCEPTED_RISK_PY = os.path.join(
    _ROOT, "research/validity/publishfirst_accepted_risk/publishfirst_accepted_risk.py")
PUBLIC_MARGIN_234 = os.path.join(
    _ROOT, "research/validity/publishfirst_public_margin/publishfirst_public_margin_results.json")
ACCEPTED_RISK_237 = os.path.join(
    _ROOT, "research/validity/publishfirst_accepted_risk/publishfirst_accepted_risk_results.json")

TPS_REGIME = "both_bugs"          # the conservative regime (matches #234's TEST regime)
CEILING_NOMINAL = 520.953         # the int4-spec lambda=1 ceiling (#204); precise value imported
PUBLISH_FIRST_BAND = (0.9138, 0.9780)  # the publish-first band efficient_lambda must lie in
H_DERIV = 1e-6                    # finite-difference step for the lambda-derivatives
ROUNDTRIP_TOL = 1e-9             # ceiling / source round-trip tolerance
RISK_ROUNDTRIP_TOL = 1e-9        # #237 risk-curve round-trip tolerance


def _import(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
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


# Import the two banked curves DIRECTLY (the strongest "import, do NOT re-derive").
BG = _import("binding_gate", BINDING_GATE_PY)
PAR = _import("publishfirst_accepted_risk", ACCEPTED_RISK_PY)


# ---------------------------------------------------------------------------
# Step 0 -- build the two curves + the gate constants.
# ---------------------------------------------------------------------------
def build_curves() -> dict[str, Any]:
    # --- TPS(lambda): my #234 / #222 mu_pub public-speed map (both_bugs regime). ---
    banked = BG.import_banked()
    tps_map = BG.build_maps(banked)[TPS_REGIME]
    ceiling = banked["lambda1_ceiling_mu"]            # 520.9527323111674 (#204)
    lambda_domain_floor = banked["lambda_floor"]      # 0.3418647166361965 (map domain floor)

    # --- risk(lambda): kanna #237 accepted-risk curve (assumed + grounded f_priv). ---
    comp = PAR.build_composition()
    _, risk = PAR.make_risk(comp)
    f_priv = comp["f_priv"]                            # 0.969106920637722 (assumed)
    f_priv_grounded = comp["f_priv_grounded"]          # 0.9570535584491102 (#224 grounded)
    floor = comp["floor"]                              # 0.9138270633254324 (mean==500)
    speed = comp["lam_speed"]                          # 0.9674684694454245 (ubel #229 gate)
    p95 = comp["p95_bar"]                              # 0.9780112973731208 (stark #191 LCB-bar)

    def tps(lam: float) -> float:
        return tps_map.mu_pub(lam)

    def risk_a(lam: float) -> float:
        return risk(lam, f_priv)

    def risk_g(lam: float) -> float:
        return risk(lam, f_priv_grounded)

    return {
        "tps_map": tps_map,
        "tps": tps,
        "risk_a": risk_a,
        "risk_g": risk_g,
        "ceiling": float(ceiling),
        "lambda_domain_floor": float(lambda_domain_floor),
        "k_cal": float(banked["K_cal"]),
        "step": float(banked["step"]),
        "tau_anchor": float(tps_map.tau_anchor),
        "f_priv": float(f_priv),
        "f_priv_grounded": float(f_priv_grounded),
        "sigma_draw": float(comp["sigma_draw"]),
        "floor": float(floor),
        "speed_gate": float(speed),
        "p95_bar": float(p95),
    }


# ---------------------------------------------------------------------------
# the lambda-derivatives (central in the interior, one-sided at the domain edges).
# ---------------------------------------------------------------------------
def deriv(f: Callable[[float], float], x: float, lo: float, hi: float, h: float = H_DERIV) -> float:
    """Finite-difference df/dx that always stays inside [lo, hi] (one-sided at the edges)."""
    xp = min(x + h, hi)
    xm = max(x - h, lo)
    return (f(xp) - f(xm)) / (xp - xm)


# ---------------------------------------------------------------------------
# Step 1-2 -- the two curves + the exchange rate on the shared lambda grid.
# ---------------------------------------------------------------------------
def exchange_table(cur: dict[str, Any]) -> dict[str, Any]:
    tps, risk_a, risk_g = cur["tps"], cur["risk_a"], cur["risk_g"]
    lo, hi = cur["lambda_domain_floor"], 1.0
    # the shared grid: exact banked floor / speed-gate / p95-bar + the three context points.
    grid = [
        ("floor", cur["floor"]),
        ("0.9500", 0.95),
        ("speed_gate", cur["speed_gate"]),
        ("p95_bar", cur["p95_bar"]),
        ("0.9970", 0.997),
        ("1.0000", 1.0),
    ]
    rows = []
    for name, lam in grid:
        dtps = deriv(tps, lam, lo, hi)
        dra = deriv(risk_a, lam, lo, hi)
        drg = deriv(risk_g, lam, lo, hi)
        # exchange rate dTPS/drisk = (dTPS/dlambda)/(drisk/dlambda); per-pp = x*0.01.
        xa = dtps / dra if dra != 0.0 else float("nan")
        xg = dtps / drg if drg != 0.0 else float("nan")
        rows.append({
            "name": name,
            "lambda": lam,
            "tps": tps(lam),
            "risk_assumed": risk_a(lam),
            "risk_grounded": risk_g(lam),
            "dtps_dlambda": dtps,
            "drisk_dlambda_assumed": dra,
            "drisk_dlambda_grounded": drg,
            "dtps_drisk_assumed": xa,                 # TPS per unit risk (risk in [0,1])
            "dtps_drisk_grounded": xg,
            "tps_per_pct_risk_assumed": xa * 0.01,    # TPS per +1pp risk
            "tps_per_pct_risk_grounded": xg * 0.01,
        })
    return {"regime": TPS_REGIME, "rows": rows}


# ---------------------------------------------------------------------------
# Step 3 -- the decision read: the 0.9780 bar -> 0.9675 gate secant (the words).
# ---------------------------------------------------------------------------
def decision_read(cur: dict[str, Any]) -> dict[str, Any]:
    tps, risk_a, risk_g = cur["tps"], cur["risk_a"], cur["risk_g"]
    bar, gate = cur["p95_bar"], cur["speed_gate"]
    d_tps = tps(gate) - tps(bar)                  # < 0: dropping lambda LOSES TPS
    d_risk_a = risk_a(gate) - risk_a(bar)         # > 0: dropping lambda ADDS risk
    d_risk_g = risk_g(gate) - risk_g(bar)
    secant_pp_a = d_tps / (d_risk_a * 100.0) if d_risk_a != 0.0 else float("nan")
    secant_pp_g = d_tps / (d_risk_g * 100.0) if d_risk_g != 0.0 else float("nan")
    words = (
        "Dropping from the 0.9780 P95 bar to the 0.9675 speed gate changes public TPS by "
        f"{d_tps:+.3f} (a LOSS) while changing accepted private-draw risk by {d_risk_a * 100:+.2f}pp "
        f"(assumed f_priv) / {d_risk_g * 100:+.2f}pp (grounded). Because TPS and clearance both fall "
        "with lambda, the move is strictly dominated: it buys NEGATIVE TPS per +1pp of extra risk "
        f"({secant_pp_a:+.3f} TPS/pp assumed). There is no speed-for-risk trade on the build-lambda "
        "axis -- a faster build is the SAME as a safer build (both want higher lambda)."
    )
    return {
        "from_lambda": bar,
        "to_lambda": gate,
        "delta_tps": d_tps,
        "delta_risk_assumed_pp": d_risk_a * 100.0,
        "delta_risk_grounded_pp": d_risk_g * 100.0,
        "secant_tps_per_pct_risk_assumed": secant_pp_a,
        "secant_tps_per_pct_risk_grounded": secant_pp_g,
        "words": words,
    }


def efficient_point(table: dict[str, Any], band: tuple[float, float]) -> dict[str, Any]:
    """The most TPS-efficient / least-risky build IN the publish-first band.

    The axis is co-monotone (TPS up, risk down in lambda), so the dominant point is the TOP of the
    band: it simultaneously maximises TPS and minimises risk. There is no interior frontier optimum
    because the speed/risk axis is dominated, not a trade-off curve.
    """
    lo, hi = band
    in_band = [r for r in table["rows"] if lo - 1e-9 <= r["lambda"] <= hi + 1e-9]
    # max TPS in band == min risk in band (co-monotone) == top of the band.
    best = max(in_band, key=lambda r: r["lambda"])
    # the PR's LITERAL "most TPS per unit risk" = argmax of the (negative) dTPS/drisk over the band.
    # On a dominated axis this degenerates to the LOWEST-lambda point (least-negative rate), which is
    # the WORST build (min TPS, max risk) -- reported for faithfulness, flagged as degenerate.
    max_ratio = max(in_band, key=lambda r: r["dtps_drisk_assumed"])
    return {
        "efficient_lambda": best["lambda"],
        "efficient_lambda_name": best["name"],
        "efficient_tps": best["tps"],
        "efficient_risk_assumed": best["risk_assumed"],
        "rationale": (
            "co-monotone axis (TPS increasing, risk decreasing in lambda) -> the most TPS-efficient "
            "AND least-risky build in the band is its TOP endpoint; no interior optimum exists "
            "because a dominated axis has no trade-off frontier."
        ),
        # the PR's literal max-ratio reading (degenerate on a dominated axis): the least-negative
        # dTPS/drisk sits at the band FLOOR, which is simultaneously the worst TPS and worst risk.
        "max_ratio_lambda": max_ratio["lambda"],
        "max_ratio_lambda_name": max_ratio["name"],
        "max_ratio_dtps_drisk_assumed": max_ratio["dtps_drisk_assumed"],
        "max_ratio_tps": max_ratio["tps"],
        "max_ratio_risk_assumed": max_ratio["risk_assumed"],
        "max_ratio_is_degenerate_worst_build": bool(
            max_ratio["lambda"] <= best["lambda"] + 1e-12
            and max_ratio["tps"] <= best["tps"] + 1e-9),
        "band": list(band),
    }


# ---------------------------------------------------------------------------
# provenance round-trips: TPS vs #234, risk vs #237 (bit-for-bit locks).
# ---------------------------------------------------------------------------
def provenance(cur: dict[str, Any]) -> dict[str, Any]:
    r234 = _load(PUBLIC_MARGIN_234)
    r237 = _load(ACCEPTED_RISK_237)
    tps = cur["tps"]

    # TPS round-trips #234's mu_pub at the speed gate + the ceiling (same lambda, same map).
    tps_speed_234 = float(r234["mu_pub_speed_at_floor"])            # 513.5574577506176 @ 0.9674685
    tps_speed_resid = abs(tps(cur["speed_gate"]) - tps_speed_234)
    tps_ceiling_resid = abs(tps(1.0) - cur["ceiling"])

    # risk round-trips #237's accepted-risk curve at every shared grid lambda.
    rows237 = {r["name"]: r for r in r237["accepted_risk_curve"]["rows"]}
    risk_resid = {}
    for name in ("floor", "0.9500", "speed_gate", "p95_bar", "0.9970", "1.0000"):
        lam = rows237[name]["lambda"]
        risk_resid[name] = {
            "assumed": abs(cur["risk_a"](lam) - rows237[name]["risk_assumed"]),
            "grounded": abs(cur["risk_g"](lam) - rows237[name]["risk_grounded"]),
        }
    risk_max_resid = max(
        max(v["assumed"], v["grounded"]) for v in risk_resid.values())

    return {
        "tps_speed_gate_234": tps_speed_234,
        "tps_speed_gate_recomputed": tps(cur["speed_gate"]),
        "tps_speed_gate_resid": tps_speed_resid,
        "tps_ceiling_resid": tps_ceiling_resid,
        "tps_roundtrips_234": bool(tps_speed_resid < ROUNDTRIP_TOL
                                   and tps_ceiling_resid < ROUNDTRIP_TOL),
        "risk_resid": risk_resid,
        "risk_max_resid": risk_max_resid,
        "risk_roundtrips_237": bool(risk_max_resid < RISK_ROUNDTRIP_TOL),
    }


# ---------------------------------------------------------------------------
# Step 4 -- self-test (PRIMARY).
# ---------------------------------------------------------------------------
def self_test(cur: dict[str, Any], table: dict[str, Any], prov: dict[str, Any],
              eff: dict[str, Any]) -> dict[str, Any]:
    rows = table["rows"]
    tps = cur["tps"]

    # (a) TPS round-trips the composition anchors (ceiling @1 + #234 @ gate); risk round-trips #237.
    a_ok = bool(prov["tps_roundtrips_234"] and prov["risk_roundtrips_237"]
                and prov["tps_ceiling_resid"] < ROUNDTRIP_TOL)

    # (b) OBSERVED monotonicity over a fine grid: TPS strictly UP in lambda, risk strictly DOWN.
    lo, hi = cur["lambda_domain_floor"], 1.0
    fine = [PUBLISH_FIRST_BAND[0] + (1.0 - PUBLISH_FIRST_BAND[0]) * i / 100.0 for i in range(101)]
    tps_seq = [cur["tps"](x) for x in fine]
    ra_seq = [cur["risk_a"](x) for x in fine]
    tps_up = all(tps_seq[i + 1] > tps_seq[i] for i in range(len(tps_seq) - 1))
    risk_down = all(ra_seq[i + 1] < ra_seq[i] for i in range(len(ra_seq) - 1))
    # the resulting exchange rate must be strictly NEGATIVE at every grid node (definite sign).
    xrate_all_negative = all(r["dtps_drisk_assumed"] < 0.0 for r in rows)
    b_ok = bool(tps_up and risk_down and xrate_all_negative)
    # the PR's premise (TPS DECREASING in lambda) is the negation of what we observe.
    pr_premise_holds = bool(not tps_up)  # False: TPS is increasing, not decreasing

    # (c) the grounded exchange rate differs from assumed at the speed gate (grounded risk steeper).
    gate_row = next(r for r in rows if r["name"] == "speed_gate")
    xa_gate = gate_row["tps_per_pct_risk_assumed"]
    xg_gate = gate_row["tps_per_pct_risk_grounded"]
    grounded_risk_steeper = bool(
        abs(gate_row["drisk_dlambda_grounded"]) > abs(gate_row["drisk_dlambda_assumed"]))
    c_ok = bool(abs(xa_gate - xg_gate) > 1e-6 and grounded_risk_steeper)

    # (d) efficient_lambda in the publish-first band [floor=0.9138, p95_bar=0.9780] (exact banked
    #     endpoints; the nominal PUBLISH_FIRST_BAND rounds p95_bar 0.97801 down to 0.9780).
    lo_b, hi_b = cur["floor"], cur["p95_bar"]
    d_ok = bool(lo_b - 1e-9 <= eff["efficient_lambda"] <= hi_b + 1e-9)

    # (e) NaN-clean across all reported scalars.
    scalars: list[Any] = [cur["ceiling"], cur["sigma_draw"], cur["f_priv"], cur["f_priv_grounded"],
                          eff["efficient_lambda"], eff["efficient_tps"], xa_gate, xg_gate,
                          prov["tps_speed_gate_resid"], prov["tps_ceiling_resid"],
                          prov["risk_max_resid"]]
    for r in rows:
        scalars += [r["lambda"], r["tps"], r["risk_assumed"], r["risk_grounded"],
                    r["dtps_dlambda"], r["drisk_dlambda_assumed"], r["drisk_dlambda_grounded"],
                    r["dtps_drisk_assumed"], r["dtps_drisk_grounded"],
                    r["tps_per_pct_risk_assumed"], r["tps_per_pct_risk_grounded"]]
    e_ok = bool(all(_finite(x) for x in scalars))

    checks = {
        "a_tps_roundtrips_anchors_and_risk_roundtrips_237": a_ok,
        "b_observed_monotone_tps_up_risk_down_xrate_negative": b_ok,
        "c_grounded_exchange_rate_differs_risk_steeper": c_ok,
        "d_efficient_lambda_in_publish_first_band": d_ok,
        "e_nan_clean": e_ok,
    }
    passes = bool(all(checks.values()))
    return {
        "tps_risk_exchange_rate_self_test_passes": passes,                 # <-- PRIMARY
        "tps_per_pct_risk_at_speed_gate": xa_gate,                         # <-- TEST (assumed f_priv)
        "tps_per_pct_risk_at_speed_gate_grounded": xg_gate,
        "checks": checks,
        "pr_premise_tps_decreasing_in_lambda_holds": pr_premise_holds,     # False (premise wrong)
        "observed": {
            "tps_increasing_in_lambda": bool(tps_up),
            "risk_decreasing_in_lambda": bool(risk_down),
            "exchange_rate_negative_everywhere": bool(xrate_all_negative),
            "grounded_risk_steeper_at_gate": grounded_risk_steeper,
        },
        "evidence": {
            "tps_speed_gate_resid": prov["tps_speed_gate_resid"],
            "tps_ceiling_resid": prov["tps_ceiling_resid"],
            "risk_max_resid": prov["risk_max_resid"],
            "xa_gate": xa_gate,
            "xg_gate": xg_gate,
            "n_scalars_checked": len(scalars),
        },
        "n_checks": len(checks),
    }


# ---------------------------------------------------------------------------
# Assemble.
# ---------------------------------------------------------------------------
def _build_result(cur, table, dec, eff, prov, st) -> dict[str, Any]:
    handoff = (
        "fern decision-card + Issue #124: along the build-lambda axis each +1pp of accepted "
        f"private-draw risk near the 0.9675 gate buys tps_per_pct_risk_at_speed_gate="
        f"{st['tps_per_pct_risk_at_speed_gate']:+.4f} TPS (assumed f_priv; "
        f"{st['tps_per_pct_risk_at_speed_gate_grounded']:+.4f} grounded) -- the slope is NEGATIVE "
        "because TPS(lambda) and private-clearance are CO-MONOTONE (both rise with lambda): a faster "
        "build is the SAME as a safer build, so there is NO speed-for-risk trade to optimise on this "
        f"axis. The most TPS-efficient AND least-risky publish-first build sits at lambda="
        f"efficient_lambda={eff['efficient_lambda']:.4f} (the TOP of the band). The human should "
        "read the #124 tension as build-lambda vs. how HARD a high-lambda build is to land (land "
        "#71), NOT as speed-vs-safety -- those move together. Conditional on a measured tunable-lambda "
        "build existing (land #71); this leg authorizes nothing."
    )
    return {
        "pr": 240,
        "metric_primary": "tps_risk_exchange_rate_self_test_passes",
        "metric_test": "tps_per_pct_risk_at_speed_gate",
        "tps_risk_exchange_rate_self_test_passes":
            st["tps_risk_exchange_rate_self_test_passes"],
        "tps_per_pct_risk_at_speed_gate": st["tps_per_pct_risk_at_speed_gate"],
        "tps_per_pct_risk_at_speed_gate_grounded": st["tps_per_pct_risk_at_speed_gate_grounded"],
        # the headline finding (CONTRADICTS the PR premise).
        "exchange_rate_sign": "negative",
        "co_monotone_in_lambda": True,
        "pr_premise_tps_decreasing_in_lambda_holds":
            st["pr_premise_tps_decreasing_in_lambda_holds"],
        "observed": st["observed"],
        "efficient_lambda": eff["efficient_lambda"],
        "efficient_point": eff,
        # the deliverable table + decision read.
        "exchange_table": table,
        "decision_read": dec,
        # the law + composition constants.
        "law": (
            "TPS(lambda) = mu_pub(lambda) = 520.953*E[T](lambda)/E[T](1) = K_cal*(E[T](lambda)/step)*"
            "tau_anchor (ubel #234/#222, both_bugs). risk(lambda) = 1 - Phi((private_mean(lambda)-500)"
            "/sigma_draw) (kanna #237). exchange rate dTPS/drisk(lambda) = (dTPS/dlambda)/(drisk/"
            "dlambda); both derivatives by central finite difference (h=1e-6). dTPS/dlambda>0 and "
            "drisk/dlambda<0 everywhere => dTPS/drisk<0 everywhere (co-monotone, dominated axis)."
        ),
        "composition": {
            "k_cal": cur["k_cal"],
            "step": cur["step"],
            "tau_anchor": cur["tau_anchor"],
            "ceiling_lambda1": cur["ceiling"],
            "sigma_draw": cur["sigma_draw"],
            "f_priv_assumed": cur["f_priv"],
            "f_priv_grounded": cur["f_priv_grounded"],
            "lambda_floor_publish_first": cur["floor"],
            "lambda_speed_gate": cur["speed_gate"],
            "p95_bar": cur["p95_bar"],
            "publish_first_band": list(PUBLISH_FIRST_BAND),
            "h_deriv": H_DERIV,
        },
        "provenance": prov,
        "self_test": st,
        "handoff": handoff,
        "scope": (
            "LOCAL CPU-only analytic composition of ubel #234/#222's mu_pub TPS(lambda) map with "
            "kanna #237's risk(lambda) curve into the exchange rate dTPS/drisk along build-lambda. "
            "Imports BOTH legs as modules (TPS round-trips #234's 513.557 gate + 520.953 ceiling; "
            "risk round-trips #237's curve bit-for-bit). The ratio of the lambda-derivatives is the "
            "only new object. No GPU/vLLM/HF Job/submission/served-file change/official draw. "
            "BASELINE stays 481.53; adds 0 TPS (PRIMARY = self-test); greedy/PPL untouched. "
            "Authorizes nothing. NOT a launch. NOT open2."
        ),
        "public_evidence_used": [
            "ubel #234 (izpjgncc, publishfirst_public_margin, MERGED) -- the TPS(lambda)=mu_pub "
            "public-speed map (via #222 binding_gate): mu_pub_speed_at_floor=513.557 @ the 0.9675 "
            "speed gate, ceiling 520.953 @ lambda=1; imported as a module and round-tripped.",
            "ubel #229 (bz2b3fw8, speed_margin_at_validity_bar, MERGED) -- the operative public "
            "speed gate lambda=0.9675 (both-bugs worstcase) at which the headline exchange rate is "
            "reported.",
            "kanna #237 (8x7i38jh, publishfirst_accepted_risk) -- the risk(lambda)=1-Phi((mu(lambda)"
            "-500)/sigma_draw) accepted-risk curve (assumed + grounded #224 f_priv); imported as a "
            "module and round-tripped bit-for-bit at the shared grid.",
            "stark #191 (private_build_bar, MERGED) -- the 0.9780 P95 LCB-on-lambda bar; the upper "
            "endpoint of the publish-first band and of the decision-read secant.",
            "kanna #228 (publish_first_lambda_floor, MERGED `352ifoi8`) -- the 0.9138 publish-first "
            "floor (private mean==500); the lower band endpoint (via #237).",
            "Issue #124 (publish-first green-light) -- the qualitative speed-vs-risk decision this "
            "leg turns into a slope; the slope is NEGATIVE (co-monotone), so the tension is NOT "
            "speed-vs-safety but build-lambda vs. build difficulty (land #71).",
        ],
        "method": (
            "LOCAL CPU-only analytic composition over EXISTING legs (the #234/#222 mu_pub map module "
            "+ the #237 risk module). Computes dTPS/dlambda and drisk/dlambda by central finite "
            "difference and reports their ratio (the exchange rate) on a shared lambda grid. No GPU/"
            "vLLM/HF Job/submission/served-file/draw. BASELINE stays 481.53; adds 0 TPS. Greedy "
            "identity untouched. NOT a launch."
        ),
        "metrics_nan_clean": 1 if st["checks"]["e_nan_clean"] else 0,
    }


# ---------------------------------------------------------------------------
# W&B logging (mirrors #234/#237; never fatal).
# ---------------------------------------------------------------------------
def _log_wandb(args, result: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[xrate] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="tps-risk-exchange-rate", agent="ubel",
            name=args.wandb_name or "ubel/tps-risk-exchange-rate",
            group=args.wandb_group,
            tags=["launch-sigma", "tps-risk-exchange-rate", "exchange-rate", "publish-first",
                  "co-monotone", "et-lambda-map", "accepted-risk", "bank-the-analysis", "pr240"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic",
                    "tps_regime": TPS_REGIME,
                    "k_cal": result["composition"]["k_cal"],
                    "step": result["composition"]["step"],
                    "ceiling_lambda1": result["composition"]["ceiling_lambda1"],
                    "sigma_draw": result["composition"]["sigma_draw"],
                    "f_priv_assumed": result["composition"]["f_priv_assumed"],
                    "f_priv_grounded": result["composition"]["f_priv_grounded"],
                    "lambda_speed_gate": result["composition"]["lambda_speed_gate"],
                    "publish_first_band": result["composition"]["publish_first_band"],
                    "h_deriv": H_DERIV},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[xrate] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[xrate] wandb disabled; skipping", flush=True)
        return
    try:
        st = result["self_test"]
        flat = {
            # PRIMARY + TEST
            "tps_risk_exchange_rate_self_test_passes":
                1.0 if st["tps_risk_exchange_rate_self_test_passes"] else 0.0,
            "tps_per_pct_risk_at_speed_gate": result["tps_per_pct_risk_at_speed_gate"],
            "tps_per_pct_risk_at_speed_gate_grounded":
                result["tps_per_pct_risk_at_speed_gate_grounded"],
            # headline finding
            "co_monotone_in_lambda": 1.0 if result["co_monotone_in_lambda"] else 0.0,
            "exchange_rate_negative_everywhere":
                1.0 if st["observed"]["exchange_rate_negative_everywhere"] else 0.0,
            "tps_increasing_in_lambda": 1.0 if st["observed"]["tps_increasing_in_lambda"] else 0.0,
            "risk_decreasing_in_lambda": 1.0 if st["observed"]["risk_decreasing_in_lambda"] else 0.0,
            "pr_premise_tps_decreasing_in_lambda_holds":
                1.0 if result["pr_premise_tps_decreasing_in_lambda_holds"] else 0.0,
            "efficient_lambda": result["efficient_lambda"],
            "efficient_tps": result["efficient_point"]["efficient_tps"],
            "efficient_risk_assumed": result["efficient_point"]["efficient_risk_assumed"],
            # decision read (0.9780 -> 0.9675 secant)
            "decision_delta_tps": result["decision_read"]["delta_tps"],
            "decision_delta_risk_assumed_pp": result["decision_read"]["delta_risk_assumed_pp"],
            "decision_delta_risk_grounded_pp": result["decision_read"]["delta_risk_grounded_pp"],
            "decision_secant_tps_per_pct_risk_assumed":
                result["decision_read"]["secant_tps_per_pct_risk_assumed"],
            "decision_secant_tps_per_pct_risk_grounded":
                result["decision_read"]["secant_tps_per_pct_risk_grounded"],
            # provenance round-trips
            "tps_speed_gate_resid": result["provenance"]["tps_speed_gate_resid"],
            "tps_ceiling_resid": result["provenance"]["tps_ceiling_resid"],
            "risk_max_resid": result["provenance"]["risk_max_resid"],
            "tps_roundtrips_234": 1.0 if result["provenance"]["tps_roundtrips_234"] else 0.0,
            "risk_roundtrips_237": 1.0 if result["provenance"]["risk_roundtrips_237"] else 0.0,
            # per-check booleans
            "self_test_a_roundtrips":
                1.0 if st["checks"]["a_tps_roundtrips_anchors_and_risk_roundtrips_237"] else 0.0,
            "self_test_b_monotone_xrate_negative":
                1.0 if st["checks"]["b_observed_monotone_tps_up_risk_down_xrate_negative"] else 0.0,
            "self_test_c_grounded_differs":
                1.0 if st["checks"]["c_grounded_exchange_rate_differs_risk_steeper"] else 0.0,
            "self_test_d_efficient_in_band":
                1.0 if st["checks"]["d_efficient_lambda_in_publish_first_band"] else 0.0,
            "self_test_e_nan_clean": 1.0 if st["checks"]["e_nan_clean"] else 0.0,
        }
        # per-lambda exchange-table rows.
        for r in result["exchange_table"]["rows"]:
            key = f"lam_{r['name'].replace('.', 'p')}"
            flat[f"{key}_tps"] = r["tps"]
            flat[f"{key}_risk_assumed"] = r["risk_assumed"]
            flat[f"{key}_risk_grounded"] = r["risk_grounded"]
            flat[f"{key}_tps_per_pct_risk_assumed"] = r["tps_per_pct_risk_assumed"]
            flat[f"{key}_tps_per_pct_risk_grounded"] = r["tps_per_pct_risk_grounded"]
        wandb_logging.log_summary(run, flat, step=0)
        # the exchange table as a wandb Table for plotting.
        try:
            import wandb
            tbl = wandb.Table(columns=["name", "lambda", "tps", "risk_assumed", "risk_grounded",
                                       "dtps_dlambda", "drisk_dlambda_assumed",
                                       "tps_per_pct_risk_assumed", "tps_per_pct_risk_grounded"])
            for r in result["exchange_table"]["rows"]:
                tbl.add_data(r["name"], r["lambda"], r["tps"], r["risk_assumed"],
                             r["risk_grounded"], r["dtps_dlambda"], r["drisk_dlambda_assumed"],
                             r["tps_per_pct_risk_assumed"], r["tps_per_pct_risk_grounded"])
            wandb_logging.log_summary(run, {"exchange_table": tbl}, step=0)
        except Exception as exc:  # noqa: BLE001
            print(f"[xrate] wandb table skipped ({exc})", flush=True)
        wandb_logging.log_json_artifact(
            run, name="tps_risk_exchange_rate",
            artifact_type="tps-risk-exchange-rate", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[xrate] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    st = result["self_test"]
    print("\n[xrate] ===== TPS-vs-PRIVATE-RISK EXCHANGE RATE along build-lambda (PR #240) =====",
          flush=True)
    print("  TPS(lambda)=mu_pub (ubel #234/#222)   risk(lambda)=1-Phi((mu_priv-500)/sigma) (kanna #237)",
          flush=True)
    print(f"  exchange rate dTPS/drisk = (dTPS/dlambda)/(drisk/dlambda)   h={H_DERIV}", flush=True)
    print("\n   lambda        TPS     risk_a   risk_g   dTPS/dl   drisk_a/dl   TPS/pp_a  TPS/pp_g",
          flush=True)
    for r in result["exchange_table"]["rows"]:
        print(f"   {r['lambda']:.6f}  {r['tps']:8.3f}  {r['risk_assumed']:.4f}  "
              f"{r['risk_grounded']:.4f}  {r['dtps_dlambda']:8.2f}  {r['drisk_dlambda_assumed']:9.4f}  "
              f"{r['tps_per_pct_risk_assumed']:+8.4f}  {r['tps_per_pct_risk_grounded']:+8.4f}  "
              f"[{r['name']}]", flush=True)
    print(f"\n  HEADLINE  tps_per_pct_risk_at_speed_gate (TEST, assumed f_priv) = "
          f"{result['tps_per_pct_risk_at_speed_gate']:+.4f} TPS/pp  "
          f"(grounded {result['tps_per_pct_risk_at_speed_gate_grounded']:+.4f})", flush=True)
    print(f"  exchange_rate_sign = {result['exchange_rate_sign'].upper()}  "
          f"co_monotone_in_lambda = {result['co_monotone_in_lambda']}  "
          f"PR-premise(TPS down in lambda) holds = "
          f"{result['pr_premise_tps_decreasing_in_lambda_holds']}", flush=True)
    print(f"  efficient_lambda (in band {result['composition']['publish_first_band']}) = "
          f"{result['efficient_lambda']:.4f}  ({result['efficient_point']['efficient_lambda_name']})",
          flush=True)
    print(f"\n  DECISION READ: {result['decision_read']['words']}", flush=True)
    print(f"\n  SELF-TEST (PRIMARY) passes = {st['tps_risk_exchange_rate_self_test_passes']}", flush=True)
    for k, v in st["checks"].items():
        print(f"    [{'ok' if v else 'XX'}] {k}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="TPS-vs-private-risk exchange rate along build-lambda (PR #240)")
    ap.add_argument("--out", default=os.path.join(_HERE, "tps_risk_exchange_rate_results.json"))
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name",
                    default="ubel/tps-risk-exchange-rate")
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="issue192-reading-calibration")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--self-test", action="store_true", help="exit non-zero if the self-test fails")
    args = ap.parse_args(argv)

    t0 = time.time()
    cur = build_curves()
    table = exchange_table(cur)
    dec = decision_read(cur)
    # search the band with EXACT banked endpoints so the p95_bar (0.97801) -- the top of the
    # publish-first band -- is included (the nominal PUBLISH_FIRST_BAND rounds it to 0.9780).
    eff = efficient_point(table, (cur["floor"], cur["p95_bar"]))
    prov = provenance(cur)
    st = self_test(cur, table, prov, eff)

    result = _build_result(cur, table, dec, eff, prov, st)
    result["elapsed_s"] = round(time.time() - t0, 4)
    result["peak_mem_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 3)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    _print(result)
    print(f"\n[xrate] HANDOFF: {result['handoff']}", flush=True)
    print(f"[xrate] artifacts -> {args.out}", flush=True)
    _log_wandb(args, result)

    if args.self_test:
        ok = st["tps_risk_exchange_rate_self_test_passes"] and result["metrics_nan_clean"] == 1
        print(f"[xrate] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
