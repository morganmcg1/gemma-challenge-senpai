#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #249 (fern) -- Build-lambda-hat target: ONE bar the measured launch lambda must clear.

WHAT THIS IS
------------
When a measured build lands (land #245 tree / lawine K-1 linear / stark T-1 topology) we will
finally have a measured lambda_hat -- but right now THREE banked constraints float on the lambda
axis and they disagree, because they price DIFFERENT risk axes:

  * 0.9780112973731208  P95 VALIDITY bar          (stark #191 LCB-on-lambda, via #239 model["p95"])
  * 0.9784133623810887  worst-case-vertex floor    (#243 lambda_floor_under_measured_div_linear:
                         the lambda at which the publish-first breakeven f_priv=0.959780 still holds
                         at the adverse NLS vertex f_priv=0.96895 under lawine #232/#242's measured
                         0.73% int4 divergence -- a POINT-estimate mean=500 floor, NOT a confidence bound)
  * 0.9807516141069097  integrated-5% draw-risk     (#239 lambda_integrated_risk5_divinformed)
  * 0.9860579957087814  integrated-5% draw-risk     (#239 lambda_integrated_risk5_uniform)

These are NOT contradictory -- they live on two different risk axes:
  (i)  VALIDITY  -- will the private re-draw be a VALID (>=500) result?  The P95-LCB axis. The first
       two constraints live here and, strikingly, AGREE: 0.9780 (P95 LCB), #243 central lambda_floor
       0.978044, and #243 worst-case-vertex floor 0.978413 cluster inside a ~4e-4 band.
  (ii) DRAW-RISK -- probability the private draw lands BELOW 500.  The integrated-5% axis. The last
       two live here (0.9808 divergence-informed / 0.9861 uniform); they sit ~3e-3 ABOVE the validity
       cluster because the draw adds sigma_draw=7.391 on top of the mean -- a draw can fall below 500
       even when the mean clears it.

THE DELIVERABLE
---------------
ONE recommended build-lambda_hat target with an explicit risk statement, so the moment a build
produces a measured lambda_hat the GO/NO-GO is unambiguous instead of a three-way argument:

  build_lambda_operative_gate   = 0.9780  (P95 VALIDITY)   -- what a build MUST clear to LAUNCH.
        residual: validity P_invalid = 0.05 (the P95 LCB construction). #243's worst-case-vertex
        floor 0.978413 CONFIRMS this gate (lands +4.0e-4 above it -- a point-estimate consistency
        check, not a tighter confidence bound), so the gate is NOT raised.
  build_lambda_defended_target  = 0.9808  (DRAW-RISK)      -- what a build SHOULD clear to hold the
        f_priv-integrated draw-below-500 risk <= 5%. Uses the DIVERGENCE-INFORMED prior (0.9808),
        NOT uniform (0.9861), because lawine #232/#242 MEASURED the 0.73% near-greedy divergence --
        the f_priv mass leans to the clean ceiling; uniform ignores that evidence and over-states.

Posture is #124 publish-first (fern #238 card): publish the milestone first; organisers rule on
validity post-hoc; the private DRAW risk is ACCEPTED (post-hoc defence). So the OPERATIVE gate is
the launch trigger; the DEFENDED target is advisory headroom, not a launch blocker.

LOCAL, CPU-ONLY, ANALYTIC reconciliation over EXISTING MERGED legs. Imports #239 (which imports #237
-> #228/#217/#191/#224/#229) and #243 VERBATIM; nothing is re-derived -- the only new object is the
RECONCILIATION of the banked numbers into one gate + one defended target. No GPU / vLLM / HF Job /
submission / served-file change / official draw. BASELINE stays 481.53; adds 0 TPS; greedy/PPL
untouched; authorizes NOTHING. NOT a launch. NOT open2.

PRIMARY metric  lambda_bar_reconciliation_self_test_passes
TEST    metric  build_lambda_operative_gate   (and build_lambda_defended_target)

Run:
  cd target/ && CUDA_VISIBLE_DEVICES="" python \
    research/validity/build_lambda_bar/build_lambda_bar.py \
    --self-test --wandb_group build-lambda-bar-reconciliation --wandb_name fern/build-lambda-bar
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# --------------------------------------------------------------------------- #
# Sibling legs imported VERBATIM (NOT re-derived). Each headline is recomputed
# from the module and round-tripped against the committed results.json.
# --------------------------------------------------------------------------- #
SRC_239 = "research/validity/fpriv_distribution_risk/fpriv_distribution_risk.py"
SRC_243 = "research/validity/fpriv_worstcase_measured_div/fpriv_worstcase_measured_div.py"
JSON_239 = "research/validity/fpriv_distribution_risk/fpriv_distribution_risk_results.json"
JSON_243 = "research/validity/fpriv_worstcase_measured_div/results.json"

TARGET = 500.0
Z1_P95 = 1.6448536269514722       # Phi^{-1}(0.95): the P95 LCB construction (stark #191 / #204)
P95_RESIDUAL = 0.05               # 1 - 0.95: the validity P_invalid the P95 LCB gate accepts
DRAW_RISK_TARGET = 0.05           # the <=5% f_priv-integrated draw-below-500 the defended target holds

K_UNIFORM = 1                     # Beta(1,1) agnostic prior (#239)
K_DIVERGENCE_INFORMED = 2         # Beta(2,1) near-greedy lean (#239), motivated by the measured 0.73%

# Provenance tolerances.
TOL_PROVENANCE = 1e-6             # PR step 1: #239 / #243 headlines must reproduce to < 1e-6
TOL_VALIDITY_CLUSTER = 1.0e-3     # the validity-axis numbers (P95 / central / worst-case) agree band


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load_mod(name: str, relpath: str):
    path = os.path.join(REPO_ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_json(relpath: str) -> Any:
    with open(os.path.join(REPO_ROOT, relpath), encoding="utf-8") as fh:
        return json.load(fh)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _close(a: float, b: float, tol: float) -> bool:
    return _finite(a) and _finite(b) and abs(float(a) - float(b)) <= tol


def _nan_paths(node: Any, p: str = "result") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            bad += _nan_paths(v, f"{p}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            bad += _nan_paths(v, f"{p}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        bad.append(p)
    return bad


# --------------------------------------------------------------------------- #
# Step 0/1 -- import the two legs, recompute their headlines, round-trip them.
# --------------------------------------------------------------------------- #
def import_banked() -> dict[str, Any]:
    """Import #239 + #243 VERBATIM, recompute their headline numbers, and round-trip vs committed JSON."""
    m239 = _load_mod("fpriv_distribution_risk", SRC_239)
    m243 = _load_mod("fpriv_worstcase_measured_div", SRC_243)

    # ---- #239: the f_priv-integrated draw-risk leg ----
    model = m239.build_model()                                   # imports #237 -> #228/#217/#191/#224/#229
    p95 = float(model["p95"])                                    # stark #191 P95 LCB-on-lambda bar
    speed = float(model["speed"])                                # ubel #229 operative public speed gate
    floor = float(model["floor"])                                # #228 publish-first floor
    f_lo, f_hi = float(model["f_lo"]), float(model["f_hi"])      # grounded / assumed f_priv support
    sigma_draw = float(model["sigma_draw"])                      # #217 private-draw sigma

    lr5_uniform = float(m239.solve_lambda_integrated_risk5(model, K_UNIFORM))
    lr5_divinformed = float(m239.solve_lambda_integrated_risk5(model, K_DIVERGENCE_INFORMED))
    ir_speed_uniform = float(m239.integrated_risk(model, speed, K_UNIFORM))
    ir_speed_divinformed = float(m239.integrated_risk(model, speed, K_DIVERGENCE_INFORMED))

    # ---- #243: the worst-case-vertex f_priv floor under the measured 0.73% divergence ----
    syn = m243.synthesize()
    h243 = syn["headline"]
    lam_floor_wc = float(h243["lambda_floor_under_measured_div_linear"])     # 0.978413 worst-case-vertex
    lam_floor_central = float(h243["lambda_floor_central"])                  # 0.978044 central (#233)
    fpriv_wc = float(h243["fpriv_worstcase_under_measured_div"])             # 0.96895 worst-case f_priv
    f_breakeven = float(h243["f_priv_breakeven"])                            # 0.959780 publish-first breakeven

    # ---- provenance round-trip: recomputed == committed JSON (< 1e-6) ----
    j239 = _load_json(JSON_239)
    j243 = _load_json(JSON_243)
    h243j = j243["synthesis"]["headline"]
    roundtrip = {
        "239.lambda_integrated_risk5_uniform":
            {"recomputed": lr5_uniform, "committed": j239["lambda_integrated_risk5_uniform"]},
        "239.lambda_integrated_risk5_divinformed":
            {"recomputed": lr5_divinformed, "committed": j239["lambda_integrated_risk5_divinformed"]},
        "239.integrated_risk_at_speed_gate":
            {"recomputed": ir_speed_uniform, "committed": j239["integrated_risk_at_speed_gate"]},
        "239.integrated_risk_at_speed_gate_divinformed":
            {"recomputed": ir_speed_divinformed,
             "committed": j239["integrated_risk_at_speed_gate_divinformed"]},
        "239.p95_private_bar":
            {"recomputed": p95, "committed": j239["p95_private_bar"]},
        "243.lambda_floor_under_measured_div_linear":
            {"recomputed": lam_floor_wc, "committed": h243j["lambda_floor_under_measured_div_linear"]},
        "243.fpriv_worstcase_under_measured_div":
            {"recomputed": fpriv_wc, "committed": h243j["fpriv_worstcase_under_measured_div"]},
        "243.f_priv_breakeven":
            {"recomputed": f_breakeven, "committed": h243j["f_priv_breakeven"]},
    }
    for d in roundtrip.values():
        d["abs_err"] = abs(float(d["recomputed"]) - float(d["committed"]))
        d["round_trips"] = bool(d["abs_err"] <= TOL_PROVENANCE)

    return {
        "m239": m239, "model": model,
        "p95": p95, "speed": speed, "floor": floor,
        "f_lo": f_lo, "f_hi": f_hi, "sigma_draw": sigma_draw,
        "lr5_uniform": lr5_uniform, "lr5_divinformed": lr5_divinformed,
        "ir_speed_uniform": ir_speed_uniform, "ir_speed_divinformed": ir_speed_divinformed,
        "lam_floor_wc": lam_floor_wc, "lam_floor_central": lam_floor_central,
        "fpriv_wc": fpriv_wc, "f_breakeven": f_breakeven,
        "roundtrip": roundtrip,
    }


# --------------------------------------------------------------------------- #
# TEST functions -- the two numbers fern #238's card consumes.
# --------------------------------------------------------------------------- #
def build_lambda_operative_gate(banked: dict[str, Any]) -> float:
    """The OPERATIVE gate (VALIDITY axis): the P95 LCB-on-lambda bar a build MUST clear to launch.

    Argued at 0.9780 (the P95 bar) rather than the slightly-higher #243 worst-case-vertex floor
    0.978413 because: (1) 0.9780 is a DISTRIBUTIONAL 95% confidence bound (residual P_invalid=0.05),
    whereas 0.978413 is a POINT-estimate mean=500 floor at the adverse NLS vertex -- it CONFIRMS the
    location (lands +4.0e-4 away) but carries no extra confidence margin; (2) the gap is inside the
    modeling resolution of the two independent derivations. The P95 bar already bakes in the margin,
    so it is the conservative, well-defined operative choice.
    """
    return float(banked["p95"])


def build_lambda_defended_target(banked: dict[str, Any]) -> float:
    """The DEFENDED target (DRAW-RISK axis): the lambda holding the f_priv-integrated P(draw<500)<=5%.

    Uses the DIVERGENCE-INFORMED Beta(2,1) prior (0.9808), NOT uniform (0.9861): lawine #232/#242
    MEASURED a 0.73% near-greedy int4 divergence, so the realizable f_priv mass leans to the clean
    ceiling f_hi. Uniform ignores that measured evidence and over-states the bar.
    """
    return float(banked["lr5_divinformed"])


# --------------------------------------------------------------------------- #
# Step 1 -- the three constraints on one lambda axis (provenance + axis label).
# --------------------------------------------------------------------------- #
def constraint_table(banked: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": "P95 validity bar",
            "lambda": banked["p95"],
            "risk_axis": "VALIDITY (P95 LCB on lambda_hat)",
            "residual_risk": {"P_invalid": P95_RESIDUAL},
            "provenance": "stark #191 LCB-on-lambda, via #239 model['p95']",
            "role": "operative gate (MUST clear to launch)",
        },
        {
            "name": "worst-case-vertex floor (measured 0.73% div)",
            "lambda": banked["lam_floor_wc"],
            "risk_axis": "VALIDITY (point-estimate mean=500 @ adverse NLS vertex)",
            "residual_risk": {"note": "point estimate; mean clears 500 at the worst-case f_priv vertex"},
            "provenance": "#243 lambda_floor_under_measured_div_linear (breakeven 0.959780 holds)",
            "role": "consistency check -- confirms the operative gate (does NOT raise it)",
        },
        {
            "name": "integrated-5% draw-risk (divergence-informed)",
            "lambda": banked["lr5_divinformed"],
            "risk_axis": "DRAW-RISK (f_priv-integrated P(draw<500))",
            "residual_risk": {"P_draw_below_500": DRAW_RISK_TARGET},
            "provenance": "#239 lambda_integrated_risk5_divinformed (Beta(2,1), measured 0.73%)",
            "role": "defended target (SHOULD clear)",
        },
        {
            "name": "integrated-5% draw-risk (uniform)",
            "lambda": banked["lr5_uniform"],
            "risk_axis": "DRAW-RISK (f_priv-integrated P(draw<500))",
            "residual_risk": {"P_draw_below_500": DRAW_RISK_TARGET},
            "provenance": "#239 lambda_integrated_risk5_uniform (Beta(1,1), divergence discounted)",
            "role": "defended target IF the measured divergence evidence is discounted",
        },
    ]


# --------------------------------------------------------------------------- #
# Step 2 -- the two risk axes, stated explicitly.
# --------------------------------------------------------------------------- #
def risk_axes(banked: dict[str, Any]) -> dict[str, Any]:
    p95, central, wc = banked["p95"], banked["lam_floor_central"], banked["lam_floor_wc"]
    validity_band = [min(p95, central, wc), max(p95, central, wc)]
    return {
        "validity_axis": {
            "question": "will the private re-draw be a VALID (>=500) result?",
            "constraints_lambda": {
                "p95_lcb_bar": p95,
                "central_lambda_floor_233": central,
                "worstcase_vertex_floor_243": wc,
            },
            "cluster_band": validity_band,
            "cluster_width": validity_band[1] - validity_band[0],
            "residual_at_gate": {"P_invalid": P95_RESIDUAL},
            "note": ("Three independent validity-axis derivations -- the P95 LCB bar, #233's central "
                     "lambda_floor, and #243's worst-case-vertex floor under the measured 0.73%% "
                     "divergence -- cluster inside a ~%.1e band; the validity location is pinned at "
                     "~0.9780." % (validity_band[1] - validity_band[0])),
        },
        "draw_risk_axis": {
            "question": "probability the private DRAW lands BELOW 500?",
            "constraints_lambda": {
                "integrated_risk5_divinformed": banked["lr5_divinformed"],
                "integrated_risk5_uniform": banked["lr5_uniform"],
            },
            "sigma_draw": banked["sigma_draw"],
            "residual_at_target": {"P_draw_below_500": DRAW_RISK_TARGET},
            "note": ("The draw adds sigma_draw=%.3f on TOP of the mean: a draw can fall below 500 even "
                     "when the mean clears it, so this axis sits ABOVE the validity cluster. The two "
                     "values differ only by the f_priv prior (divergence-informed vs uniform)."
                     % banked["sigma_draw"]),
        },
        "why_not_one_number": (
            "VALIDITY prices whether lambda_hat itself is high enough (an LCB confidence statement); "
            "DRAW-RISK prices whether a Gaussian draw with sigma=%.3f around the mean lands >=500. "
            "Different objects -> the validity bar (0.9780) and the draw-risk bar (0.9808+) do not "
            "reduce to a single number." % banked["sigma_draw"]),
    }


# --------------------------------------------------------------------------- #
# Step 3/4 -- the recommendation (operative gate + defended target + headline).
# --------------------------------------------------------------------------- #
def recommendation(banked: dict[str, Any]) -> dict[str, Any]:
    m239, model = banked["m239"], banked["model"]
    operative = build_lambda_operative_gate(banked)
    defended = build_lambda_defended_target(banked)
    defended_uniform = banked["lr5_uniform"]

    # The ACCEPTED draw-below-500 residual AT the operative gate (publish-first accepts this).
    draw_at_operative_divinformed = float(m239.integrated_risk(model, operative, K_DIVERGENCE_INFORMED))
    draw_at_operative_uniform = float(m239.integrated_risk(model, operative, K_UNIFORM))
    # By construction the defended target drives the divergence-informed draw risk to 5%.
    draw_at_defended_divinformed = float(m239.integrated_risk(model, defended, K_DIVERGENCE_INFORMED))

    # #243's worst-case-vertex floor consistency margin vs the operative gate.
    wc_minus_operative = banked["lam_floor_wc"] - operative

    risk_statement = (
        "build to lambda_hat >= %.4f (OPERATIVE gate, P95 validity); at %.4f the accepted residual is: "
        "validity P_invalid = %.2f (P95 LCB construction) and draw-below-500 P = %.4f "
        "(divergence-informed) -- the draw risk is ACCEPTED under #124 publish-first as post-hoc "
        "defence. SHOULD additionally clear the DEFENDED target lambda_hat >= %.4f (divergence-informed "
        "5%% draw-risk) to drive draw-below-500 P down to %.2f."
        % (operative, operative, P95_RESIDUAL, draw_at_operative_divinformed,
           defended, DRAW_RISK_TARGET))

    return {
        "build_lambda_operative_gate": operative,                    # TEST
        "build_lambda_defended_target": defended,                    # TEST
        "build_lambda_defended_target_uniform_if_div_discounted": defended_uniform,
        "operative_gate_axis": "VALIDITY (P95 LCB on lambda_hat)",
        "operative_gate_residual": {"P_invalid": P95_RESIDUAL,
                                    "draw_below_500_divinformed_accepted": draw_at_operative_divinformed,
                                    "draw_below_500_uniform_accepted": draw_at_operative_uniform},
        "defended_target_axis": "DRAW-RISK (f_priv-integrated P(draw<500), divergence-informed)",
        "defended_target_residual": {"P_draw_below_500": draw_at_defended_divinformed},
        "worstcase_vertex_floor_243": banked["lam_floor_wc"],
        "worstcase_vertex_floor_minus_operative": wc_minus_operative,
        "headline_build_lambda_target": operative,
        "risk_statement": risk_statement,
        "posture": ("#124 publish-first: publish the milestone first; organisers rule on validity "
                    "post-hoc; private DRAW risk is ACCEPTED (post-hoc defence). The OPERATIVE gate is "
                    "the launch trigger; the DEFENDED target is advisory headroom, not a launch blocker."),
        "monotone_defended_ge_operative": bool(defended >= operative),
    }


# --------------------------------------------------------------------------- #
# Step 5 -- sensitivity.
# --------------------------------------------------------------------------- #
def sensitivity(banked: dict[str, Any]) -> dict[str, Any]:
    m239, model = banked["m239"], banked["model"]
    floor = banked["floor"]
    risk_at, public_central = model["risk_at"], model["public_central"]
    f_lo, f_hi = banked["f_lo"], banked["f_hi"]

    def solve_point_lambda_risk5(f_priv: float) -> float | None:
        """lambda where the POINT (not integrated) draw risk risk(lambda; f_priv) == 5%."""
        lo, hi = floor, 1.0
        g = lambda lam: risk_at(public_central(lam), f_priv) - DRAW_RISK_TARGET
        glo, ghi = g(lo), g(hi)
        if glo == 0.0:
            return lo
        if ghi == 0.0:
            return hi
        if (glo > 0.0) == (ghi > 0.0):
            return None
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            gm = g(mid)
            if abs(gm) < 1e-14 or (hi - lo) < 1e-14:
                return mid
            if (gm > 0.0) == (glo > 0.0):
                lo, glo = mid, gm
            else:
                hi, ghi = mid, gm
        return 0.5 * (lo + hi)

    point_assumed = solve_point_lambda_risk5(f_hi)      # ~ #237 point lambda_risk5 = 0.9700
    point_grounded = solve_point_lambda_risk5(f_lo)     # grounded endpoint -> near lambda=1

    return {
        "discount_divergence_evidence": {
            "defended_target_divergence_informed": banked["lr5_divinformed"],
            "defended_target_uniform": banked["lr5_uniform"],
            "delta_uniform_minus_divinformed": banked["lr5_uniform"] - banked["lr5_divinformed"],
            "operative_gate_unchanged": banked["p95"],
            "note": ("Discounting lawine #232/#242's measured 0.73%% (uniform prior) moves the DEFENDED "
                     "target UP from %.4f to %.4f (+%.4f); the OPERATIVE gate is unaffected (it does "
                     "not depend on the f_priv prior)."
                     % (banked["lr5_divinformed"], banked["lr5_uniform"],
                        banked["lr5_uniform"] - banked["lr5_divinformed"])),
        },
        "fpriv_endpoint_pin": {
            "point_lambda_risk5_at_assumed_f_hi": point_assumed,
            "point_lambda_risk5_at_grounded_f_lo": point_grounded,
            "f_hi_assumed": f_hi,
            "f_lo_grounded": f_lo,
            "note": ("Pinning f_priv at a POINT instead of integrating: at the assumed ceiling "
                     "f_hi=%.6f the 5%% draw-risk lambda is %.4f (== #237's point lambda_risk5 0.9700); "
                     "at the grounded floor f_lo=%.6f it rises to %.4f (near lambda=1 -- the grounded "
                     "f_priv needs almost-perfect acceptance to hold 5%% draw-risk). The integrated "
                     "defended target 0.9808 sits between these endpoints, nearer the assumed end under "
                     "the divergence-informed lean."
                     % (f_hi, point_assumed if point_assumed is not None else float("nan"),
                        f_lo, point_grounded if point_grounded is not None else float("nan"))),
        },
    }


# --------------------------------------------------------------------------- #
# self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(banked: dict[str, Any], axes: dict[str, Any], rec: dict[str, Any],
              sens: dict[str, Any]) -> dict[str, Any]:
    conditions: dict[str, Any] = {}

    # (a) provenance: every #239 / #243 headline reproduces from the module to < 1e-6.
    rt = banked["roundtrip"]
    a_ok = all(d["round_trips"] for d in rt.values())
    conditions["a_provenance_239_243_reproduce_lt_1e6"] = {
        "pass": bool(a_ok),
        "max_abs_err": max(d["abs_err"] for d in rt.values()),
        "tol": TOL_PROVENANCE,
        "failures": [k for k, d in rt.items() if not d["round_trips"]],
    }

    # (b) operative gate + defended target are each pinned to a stated risk axis with a NUMERIC residual.
    op_res = rec["operative_gate_residual"]
    df_res = rec["defended_target_residual"]
    b_ok = bool(
        isinstance(rec["operative_gate_axis"], str) and rec["operative_gate_axis"]
        and _finite(op_res["P_invalid"])
        and _finite(op_res["draw_below_500_divinformed_accepted"])
        and isinstance(rec["defended_target_axis"], str) and rec["defended_target_axis"]
        and _finite(df_res["P_draw_below_500"]))
    conditions["b_gate_and_target_pinned_to_axis_with_numeric_residual"] = {
        "pass": b_ok,
        "operative_gate_axis": rec["operative_gate_axis"],
        "operative_P_invalid": op_res["P_invalid"],
        "operative_draw_below_500_divinformed": op_res["draw_below_500_divinformed_accepted"],
        "defended_target_axis": rec["defended_target_axis"],
        "defended_P_draw_below_500": df_res["P_draw_below_500"],
    }

    # (c) monotone / consistent: defended >= operative; and the validity-axis numbers agree (< 1e-3),
    #     and discounting divergence only RAISES the defended target (uniform >= div-informed).
    op, df = rec["build_lambda_operative_gate"], rec["build_lambda_defended_target"]
    df_uni = rec["build_lambda_defended_target_uniform_if_div_discounted"]
    cluster_w = axes["validity_axis"]["cluster_width"]
    c_ok = bool(df >= op and df_uni >= df and cluster_w <= TOL_VALIDITY_CLUSTER)
    conditions["c_monotone_consistent_defended_ge_operative"] = {
        "pass": c_ok,
        "operative": op, "defended_divinformed": df, "defended_uniform": df_uni,
        "defended_ge_operative": bool(df >= op),
        "uniform_ge_divinformed": bool(df_uni >= df),
        "validity_cluster_width": cluster_w,
        "validity_cluster_tol": TOL_VALIDITY_CLUSTER,
    }

    # (d) the defended target actually delivers ~5% integrated draw-risk (construction check), and the
    #     operative gate's accepted draw-risk is strictly ABOVE 5% (it is the looser validity bar).
    df_residual = df_res["P_draw_below_500"]
    op_draw = op_res["draw_below_500_divinformed_accepted"]
    d_ok = bool(_close(df_residual, DRAW_RISK_TARGET, 1e-4) and op_draw > DRAW_RISK_TARGET)
    conditions["d_defended_holds_5pct_operative_accepts_more"] = {
        "pass": d_ok,
        "defended_integrated_draw_risk": df_residual,
        "operative_accepted_draw_risk": op_draw,
        "draw_risk_target": DRAW_RISK_TARGET,
    }

    # (e) NaN-clean across the reported payload (filled at payload level too).
    payload_numeric = {"banked_scalars": {k: v for k, v in banked.items()
                                          if k not in ("m239", "model", "roundtrip")},
                       "axes": axes, "rec": rec, "sens": sens}
    nan_paths = _nan_paths(payload_numeric, "selftest")
    e_ok = (len(nan_paths) == 0)
    conditions["e_nan_clean"] = {"pass": e_ok, "nan_paths": nan_paths}

    passes = bool(all(c["pass"] for c in conditions.values()))
    return {
        "lambda_bar_reconciliation_self_test_passes": passes,
        "conditions": conditions,
        "n_conditions": len(conditions),
    }


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run() -> dict[str, Any]:
    t0 = time.time()
    banked = import_banked()
    constraints = constraint_table(banked)
    axes = risk_axes(banked)
    rec = recommendation(banked)
    sens = sensitivity(banked)
    st = self_test(banked, axes, rec, sens)

    handoff = (
        "fern #238 card row (iii) build bar: the measured launch lambda_hat must clear the OPERATIVE "
        "gate build_lambda_operative_gate=%.4f (P95 validity; residual P_invalid=%.2f). #243's "
        "worst-case-vertex floor %.6f confirms it (+%.1e). The DEFENDED target "
        "build_lambda_defended_target=%.4f (divergence-informed 5%% draw-risk) is advisory headroom -- "
        "under #124 publish-first the draw risk (%.4f at the gate) is ACCEPTED post-hoc, so the "
        "operative gate is the launch trigger and the defended target is a SHOULD, not a MUST."
        % (rec["build_lambda_operative_gate"], P95_RESIDUAL, banked["lam_floor_wc"],
           rec["worstcase_vertex_floor_minus_operative"], rec["build_lambda_defended_target"],
           rec["operative_gate_residual"]["draw_below_500_divinformed_accepted"]))

    peak_mem_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    result = {
        "pr": 249,
        "agent": "fern",
        "kind": "build_lambda_bar_reconciliation",
        "metric_primary": "lambda_bar_reconciliation_self_test_passes",
        "metric_test": "build_lambda_operative_gate",
        "lambda_bar_reconciliation_self_test_passes":
            st["lambda_bar_reconciliation_self_test_passes"],
        # ---- the two decision-grade deliverables (TEST) ----
        "build_lambda_operative_gate": rec["build_lambda_operative_gate"],
        "build_lambda_defended_target": rec["build_lambda_defended_target"],
        "build_lambda_defended_target_uniform_if_div_discounted":
            rec["build_lambda_defended_target_uniform_if_div_discounted"],
        "headline_build_lambda_target": rec["headline_build_lambda_target"],
        "risk_statement": rec["risk_statement"],
        # ---- residuals / axes ----
        "operative_gate_axis": rec["operative_gate_axis"],
        "operative_gate_residual": rec["operative_gate_residual"],
        "defended_target_axis": rec["defended_target_axis"],
        "defended_target_residual": rec["defended_target_residual"],
        "monotone_defended_ge_operative": rec["monotone_defended_ge_operative"],
        "worstcase_vertex_floor_243": rec["worstcase_vertex_floor_243"],
        "worstcase_vertex_floor_minus_operative": rec["worstcase_vertex_floor_minus_operative"],
        # ---- sections ----
        "constraint_table": constraints,
        "risk_axes": axes,
        "recommendation": rec,
        "sensitivity": sens,
        "self_test": st,
        # ---- banked scalars + provenance round-trip ----
        "banked_scalars": {
            "p95_validity_bar": banked["p95"],
            "speed_gate": banked["speed"],
            "publish_first_floor": banked["floor"],
            "f_priv_grounded": banked["f_lo"],
            "f_priv_assumed": banked["f_hi"],
            "sigma_draw": banked["sigma_draw"],
            "lambda_integrated_risk5_uniform": banked["lr5_uniform"],
            "lambda_integrated_risk5_divinformed": banked["lr5_divinformed"],
            "integrated_risk_at_speed_gate_uniform": banked["ir_speed_uniform"],
            "integrated_risk_at_speed_gate_divinformed": banked["ir_speed_divinformed"],
            "lambda_floor_worstcase_vertex_243": banked["lam_floor_wc"],
            "lambda_floor_central_233": banked["lam_floor_central"],
            "fpriv_worstcase_under_measured_div": banked["fpriv_wc"],
            "f_priv_breakeven_publish_first": banked["f_breakeven"],
        },
        "provenance_roundtrip": banked["roundtrip"],
        "handoff": handoff,
        "scope": (
            "LOCAL CPU-only analytic RECONCILIATION over EXISTING MERGED legs: imports #239 "
            "(fpriv_distribution_risk -> #237 -> #228/#217/#191/#224/#229) and #243 "
            "(fpriv_worstcase_measured_div) VERBATIM and reconciles their banked numbers into ONE "
            "operative gate + ONE defended target on labelled risk axes. Recomputes both legs' "
            "headlines from their modules and round-trips them against the committed JSON (< 1e-6). "
            "No GPU / vLLM / HF Job / submission / served-file change / official draw. BASELINE stays "
            "481.53; adds 0 TPS; greedy/PPL untouched; authorizes NOTHING. NOT a launch. NOT open2."),
        "public_evidence_used": [
            "kanna #239 (fpriv_distribution_risk): lambda_integrated_risk5 = 0.9861 uniform / 0.9808 "
            "divergence-informed, integrated_risk_at_speed_gate, p95 validity bar 0.9780 -- imported "
            "as a module and reproduced to < 1e-6.",
            "stark #243 (fpriv_worstcase_measured_div): lambda_floor_under_measured_div_linear 0.978413 "
            "(publish-first breakeven 0.959780 holds at the worst-case NLS vertex under the measured "
            "0.73% divergence) -- imported as a module and reproduced to < 1e-6.",
            "lawine #232 / #242 (int4 token divergence): the MEASURED 0.73% near-greedy divergence that "
            "selects the divergence-informed prior (0.9808) over uniform (0.9861) for the defended target.",
            "fern #238 (launch_decision_card): the #124 publish-first GO/NO-GO card whose build-bar row "
            "(iii) consumes the operative gate; under publish-first the draw risk is ACCEPTED post-hoc.",
            "Issue #124 (publish-first green-light): the launch posture under which the operative gate is "
            "the trigger and the defended target is advisory.",
        ],
        "method": (
            "LOCAL CPU-only analytic reconciliation. Import #239 + #243 VERBATIM; recompute and "
            "round-trip their headlines; lay the three constraints on one lambda axis; separate the "
            "VALIDITY (P95 LCB) and DRAW-RISK (integrated-5%) axes; pick the operative gate (0.9780 "
            "validity) and defended target (0.9808 divergence-informed) with numeric residuals; "
            "sensitivity to discounting the divergence evidence and to the f_priv endpoint pin. No "
            "GPU/vLLM/HF Job/submission/served-file/draw. BASELINE stays 481.53; adds 0 TPS."),
        "metrics_nan_clean": 1 if st["conditions"]["e_nan_clean"]["pass"] else 0,
        "peak_mem_mib": peak_mem_mib,
        "elapsed_s": round(time.time() - t0, 4),
    }
    # payload-level NaN guard (mirrors siblings).
    nan_paths = _nan_paths(result, "result")
    result["nan_clean"] = not nan_paths
    if nan_paths:
        result["metrics_nan_clean"] = 0
        result["self_test"]["conditions"]["e_nan_clean"]["pass"] = False
        result["self_test"]["conditions"]["e_nan_clean"]["nan_paths_payload"] = nan_paths
        result["self_test"]["lambda_bar_reconciliation_self_test_passes"] = False
        result["lambda_bar_reconciliation_self_test_passes"] = False
    return result


# --------------------------------------------------------------------------- #
# wandb
# --------------------------------------------------------------------------- #
def _log_wandb(args, result: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[build-lambda-bar] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    run = wandb_logging.init_wandb_run(
        job_type="build-lambda-bar-reconciliation", agent="fern",
        name=args.wandb_name or "fern/build-lambda-bar",
        group=args.wandb_group,
        tags=["build-lambda-bar", "lambda-reconciliation", "operative-gate", "defended-target",
              "validity-axis", "draw-risk-axis", "publish-first", "issue-124", "pr249"],
        config={"baseline_tps": 481.53, "method": "cpu-only-analytic", "target_tps": 500.0,
                "build_lambda_operative_gate": result["build_lambda_operative_gate"],
                "build_lambda_defended_target": result["build_lambda_defended_target"],
                "imports_pr": [239, 243, 237, 232, 242, 238, 124]},
    )
    if run is None:
        print("[build-lambda-bar] wandb disabled; skipping", flush=True)
        return
    try:
        bs = result["banked_scalars"]
        flat = {
            "lambda_bar_reconciliation_self_test_passes":
                1.0 if result["lambda_bar_reconciliation_self_test_passes"] else 0.0,
            "build_lambda_operative_gate": result["build_lambda_operative_gate"],
            "build_lambda_defended_target": result["build_lambda_defended_target"],
            "build_lambda_defended_target_uniform_if_div_discounted":
                result["build_lambda_defended_target_uniform_if_div_discounted"],
            "headline_build_lambda_target": result["headline_build_lambda_target"],
            "operative_P_invalid": result["operative_gate_residual"]["P_invalid"],
            "operative_draw_below_500_divinformed":
                result["operative_gate_residual"]["draw_below_500_divinformed_accepted"],
            "operative_draw_below_500_uniform":
                result["operative_gate_residual"]["draw_below_500_uniform_accepted"],
            "defended_P_draw_below_500": result["defended_target_residual"]["P_draw_below_500"],
            "worstcase_vertex_floor_243": result["worstcase_vertex_floor_243"],
            "worstcase_vertex_floor_minus_operative": result["worstcase_vertex_floor_minus_operative"],
            "monotone_defended_ge_operative":
                1.0 if result["monotone_defended_ge_operative"] else 0.0,
            "p95_validity_bar": bs["p95_validity_bar"],
            "lambda_floor_central_233": bs["lambda_floor_central_233"],
            "lambda_integrated_risk5_uniform": bs["lambda_integrated_risk5_uniform"],
            "lambda_integrated_risk5_divinformed": bs["lambda_integrated_risk5_divinformed"],
            "sigma_draw": bs["sigma_draw"],
            "validity_cluster_width": result["risk_axes"]["validity_axis"]["cluster_width"],
            "provenance_max_abs_err":
                result["self_test"]["conditions"]["a_provenance_239_243_reproduce_lt_1e6"]["max_abs_err"],
            "metrics_nan_clean": float(result["metrics_nan_clean"]),
            "peak_mem_mib": result["peak_mem_mib"],
            **{f"selftest_{k}": (1.0 if v["pass"] else 0.0)
               for k, v in result["self_test"]["conditions"].items()},
        }
        try:
            import wandb
            tbl = wandb.Table(columns=["name", "lambda", "risk_axis", "role", "provenance"])
            for r in result["constraint_table"]:
                tbl.add_data(r["name"], r["lambda"], r["risk_axis"], r["role"], r["provenance"])
            flat["constraint_table"] = tbl
        except Exception as exc:  # noqa: BLE001
            print(f"[build-lambda-bar] wandb table skipped ({exc})", flush=True)
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="build_lambda_bar", artifact_type="validity", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[build-lambda-bar] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# report + main
# --------------------------------------------------------------------------- #
def _print(result: dict[str, Any]) -> None:
    st = result["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("PR #249  BUILD-lambda_hat TARGET -- one bar the measured launch lambda must clear", flush=True)
    print("=" * 100, flush=True)
    print("  three constraints on one lambda axis:", flush=True)
    for r in result["constraint_table"]:
        print(f"    {r['lambda']:.6f}  [{r['risk_axis']}]  {r['name']}", flush=True)
        print(f"               role: {r['role']}", flush=True)
    print("-" * 100, flush=True)
    va = result["risk_axes"]["validity_axis"]
    print(f"  VALIDITY axis cluster band [{va['cluster_band'][0]:.6f}, {va['cluster_band'][1]:.6f}] "
          f"width {va['cluster_width']:.2e}  (P95 / central lambda_floor / worst-case-vertex AGREE)",
          flush=True)
    print("-" * 100, flush=True)
    print(f"  build_lambda_operative_gate  = {result['build_lambda_operative_gate']:.6f}   "
          f"[{result['operative_gate_axis']}]", flush=True)
    print(f"    residual: P_invalid={result['operative_gate_residual']['P_invalid']:.4f}  "
          f"draw<500 accepted (div-informed)="
          f"{result['operative_gate_residual']['draw_below_500_divinformed_accepted']:.4f}", flush=True)
    print(f"  build_lambda_defended_target = {result['build_lambda_defended_target']:.6f}   "
          f"[{result['defended_target_axis']}]", flush=True)
    print(f"    residual: P(draw<500)={result['defended_target_residual']['P_draw_below_500']:.4f}  "
          f"(uniform-if-discounted={result['build_lambda_defended_target_uniform_if_div_discounted']:.6f})",
          flush=True)
    print("-" * 100, flush=True)
    print(f"  HEADLINE: {result['risk_statement']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  PRIMARY lambda_bar_reconciliation_self_test_passes = "
          f"{st['lambda_bar_reconciliation_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"    [{'ok' if v['pass'] else '!! FAILED'}] {k}", flush=True)
    print(f"\n  HANDOFF: {result['handoff']}\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "build_lambda_bar_results.json"))
    ap.add_argument("--self-test", "--self_test", action="store_true",
                    help="run the self-test (PRIMARY); nonzero exit on failure")
    ap.add_argument("--wandb-name", "--wandb_name", default="fern/build-lambda-bar")
    ap.add_argument("--wandb-group", "--wandb_group", default="build-lambda-bar-reconciliation")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    result = run()

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"[build-lambda-bar] wrote {args.out}", flush=True)

    _print(result)
    _log_wandb(args, result)

    if args.self_test and not result["lambda_bar_reconciliation_self_test_passes"]:
        print("[build-lambda-bar] SELF-TEST FAILED", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
