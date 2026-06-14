#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Binding gate: validity (0.9780) or speed (513.557) -- which launch gate binds? (PR #222).

WHAT THIS IS
------------
The launch has TWO gates that the SAME build acceptance must satisfy, and nobody has
resolved WHICH ONE BINDS:

  (validity) the build is valid iff its private depth-aggregate acceptance lambda_hat >= 0.9780
             (stark #208 worst-case private go-bar `d198_coupled_bar`; #215 deep-tail framing);
  (speed)    the public mu_pub >= 513.557 (my #218 grounded worst-case GO trigger; central
             512.735) -- the trigger at which a build's one-sided P95 LCB clears 500.

land #71 is building ONE checkpoint whose acceptance drives BOTH the E[T](lambda_hat)->mu_pub
speed map AND the private validity bar. The single launch-simplifying question: at the
acceptance that JUST clears the VALIDITY bar (lambda_hat=0.9780), what public mu_pub does the
build show via the E[T] map -- and does it ALREADY clear the 513.557 SPEED trigger?

  if mu_pub(0.9780) >= 513.557 -> VALIDITY binds: clearing 0.9780 launches with speed to spare,
                                  so land #71's single target is the validity bar 0.9780;
  if mu_pub(0.9780) <  513.557 -> SPEED binds: the build needs acceptance ABOVE 0.9780 to be
                                  fast enough even though validity is satisfied.

THE E[T](lambda_hat) -> mu_pub MAP (imported reach-DP SHAPE, anchored to the #204 ceiling)
-----------------------------------------------------------------------------------------
The reach-DP map is `official = K_cal*(E[T](lambda_hat)/step)*tau` (K_cal=125.268, step=1.2182;
#175/#184, the SAME DP wirbel #199/#213 used). E[T](lambda_hat) is the per-depth linear
self-KV-recovery blend over the lambda_hat in [0.342, 1.0] segment (#213's LambdaCurve):

    t(lambda_hat) = (lambda_hat - lambda_floor) / (1 - lambda_floor)        # [floor,1] -> [0,1]
    spine         = (1 - t)*floor_spine + t*ceiling_spine                   # #178/#193 recovery
    E[T](lambda_hat) = reach_DP_pmf_mean(spine)                             # #175/#184 DP

The two banked endpoint spines (#213's `floor_spine_at_lambda_hat` / `ceiling_spine`) and the
reach-DP are imported VERBATIM (the reach-DP on those spines reproduces #213's banked E[T]
bit-exactly -- a provenance lock). We do NOT re-derive E[T], K_cal, the step, or the spines.

The launch-sigma lane pins ONE lambda=1 ceiling: mu_pub(1) = 520.953 (#204/#194, the int4-spec
ceiling). The raw reach-DP E[T](1) does NOT land on 520.953 for tau in [0.9924,1.0] (both-bugs
E[T](1)=5.2189 -> 536.66 at tau=1, the OPTIMISTIC compliant-spec ceiling; descent-only
E[T](1)=5.0629 -> 520.62 at tau=1). So we ANCHOR the map's lambda=1 endpoint to the imported
#204 ceiling and carry the reach-DP only for the SHAPE:

    mu_pub(lambda_hat) = 520.953 * E[T](lambda_hat) / E[T](1)               # round-trips 520.953

Equivalently `official = K_cal*(E[T]/step)*tau_anchor` with tau_anchor = 520.953*step /
(K_cal*E[T](1)) calibrated to the ceiling: tau_anchor = 1.0006 (descent_only, ~ tau=1 -> 520.953
IS the descent-spec ceiling) / 0.9707 (both_bugs, anchoring its shape to the CONSERVATIVE
int4 520.953 rather than the optimistic 536.66). The two regimes' shapes are near-identical:
mu_pub(0.9780) = 515.92 (both_bugs) / 515.94 (descent_only) -- the verdict is regime-robust.

THE VERDICT (the deliverable)
-----------------------------
  mu_pub_at_validity_bar = mu_pub(0.9780) = 515.92  >=  513.557 trigger  -> VALIDITY BINDS
  speed_margin_at_validity_bar = 515.92 - 513.557 = +2.37 TPS

Equivalently: a build AT the validity bar shows public mu_pub=515.92, whose one-sided P95 LCB
mu_pub - z1*sigma_worst = 502.37 clears the real 500 target by +2.37 (worst-case sigma). The
acceptance that clears the SPEED trigger is lambda_hat_speed=0.9675 (worst-case) -- BELOW the
0.9780 validity bar -- so validity is the binding constraint at EVERY trigger corner
([512.519 tight, 512.735 central, 513.557 worst-case]). land #71's single build-target is 0.9780.

CONSERVATIVE FRAMING (favorable direction): lambda_hat=0.9780 is the PRIVATE acceptance bar;
the build's PUBLIC acceptance is >= its private acceptance (the public->private drop is
non-negative -- stark #203/#208: private is the lower number), so evaluating the speed map at
the PRIVATE 0.9780 UNDERSTATES the true public mu_pub -> mu_pub_at_validity_bar is a
conservative (lower-bound) speed estimate, and the true public mu_pub is HIGHER -> validity
binds even MORE robustly.

SCOPE
-----
LOCAL CPU-ONLY analytic cross of ubel's #218 speed trigger against stark #208's validity bar
via the banked E[T](lambda_hat) reach-DP map. No GPU / vLLM / HF Job / submission / official
draw / served-file change. Takes NO official draws, authorizes none. BASELINE stays 481.53;
greedy/PPL untouched; adds 0 TPS (PRIMARY = self-test). The sigma's, bar 0.9780, trigger
513.557, ceiling 520.953 are imported unchanged. NOT a launch. NOT open2.

SELF-TEST (PR step 5 -- PRIMARY)
--------------------------------
(a) mu_pub(lambda=1) = 520.953 EXACTLY (round-trip the #204 ceiling, by anchoring construction);
(b) mu_pub(lambda_hat) MONOTONE INCREASING in lambda_hat (reach-DP E[T] is monotone in the blend);
(c) mu_pub_at_validity_bar in (mu_pub at the realistic floor lambda_hat=0.342, 520.953) -- sane;
(d) binding_gate = "validity" iff mu_pub_at_validity_bar >= trigger, and then
    lambda_hat_build_target = 0.9780; else lambda_hat_speed > 0.9780 -- and the inverse map
    round-trips mu_pub(lambda_hat_speed) = 513.557 to tol (tested regardless of which binds);
(e) reproduces #218's 513.557 worst-case trigger as the speed bar (imported, asserted);
(f) NaN-clean across all reported scalars.
Provenance lock: reach-DP on the banked floor/ceiling spines reproduces #213's banked E[T].
PRIMARY = binding_gate_self_test_passes (bool);
TEST    = mu_pub_at_validity_bar (float TPS, both_bugs -- the conservative regime).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from typing import Any, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # research/validity/binding_gate -> repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# source-of-truth artifacts (imported verbatim; one source per constant).
# ---------------------------------------------------------------------------
INTERLEG_218 = os.path.join(_ROOT, "research/validity/interleg_rho/interleg_rho_results.json")
MULTIVERTEX = os.path.join(_ROOT, "research/validity/multivertex_realizability/results.json")
KERNEL_BUDGET_213 = os.path.join(_ROOT, "research/validity/kernel_budget_lambda/kernel_budget_lambda_results.json")
COMPLIANT_SPEC = os.path.join(_ROOT, "research/validity/compliant_spec_et/compliant_spec_et.py")

Z1_ONE_SIDED = 1.6448536269514722  # scipy.stats.norm.ppf(0.95) -> one-sided P95 LCB multiplier
TARGET = 500.0
VALIDITY_BAR_NOMINAL = 0.9780      # the PR's nominal bar; the precise import is d198_coupled_bar
REGIMES = ("both_bugs", "descent_only")
HEADLINE_REGIME = "both_bugs"      # the conservative regime + the bar's own topology -> TEST metric
ANCHOR_TOL = 1e-9                  # ceiling round-trip / inverse round-trip tolerance
PROV_TOL = 1e-9                    # reach-DP-on-banked-spines provenance tolerance vs #213
SANE_TOL = 1e-9


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _import(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# Import #199's compliant-spec module purely for the reach-DP (et_via_reachdp is a pure
# function of a spine -- it does NOT need the PR#86 rankprobe shard) + the pinned composition.
C = _import("compliant_spec_et", COMPLIANT_SPEC)
K_CAL = C.K_CAL          # 125.26795005202914 (#148/#169)
STEP = C.STEP            # 1.2182 (#168)


def _reachdp_et(spine: list[float]) -> float:
    """E[T] = pmf-mean of the accepted-length reach-DP on a per-depth acceptance spine (#175/#184)."""
    return float(C.et_via_reachdp(spine)["et_pmf_mean"])


# ---------------------------------------------------------------------------
# Step 0 -- import the banked legs (NOT re-derived).
# ---------------------------------------------------------------------------
def import_banked() -> dict[str, Any]:
    # --- #218 (my MERGED interleg-rho): the grounded speed triggers + combined sigma + ceiling. ---
    r218 = _load(INTERLEG_218)
    gt = r218["grounded_trigger"]
    trig_band = gt["go_trigger_grounded_band"]            # [tight, worstcase]
    sig_band = gt["combined_sigma_grounded_band"]         # [tight, loose=worstcase]

    # --- #208/#215 validity lane: the worst-case private go-bar + the realistic recovery floor. ---
    rmv = _load(MULTIVERTEX)["synthesis"]["constants"]
    validity_bar = float(rmv["d198_coupled_bar"])         # 0.9779783323491393 (#208 worst-case)
    lambda_floor = float(rmv["lambda_floor_liveprobe"])   # 0.3418647166361965 (#193)

    # --- #213 kernel-budget: the two banked endpoint spines per regime (the reach-DP SHAPE). ---
    r213 = _load(KERNEL_BUDGET_213)["synthesis"]
    lambda_floor_213 = float(r213["lambda_hat"])
    spines = {}
    et_banked_213 = {}
    for reg in REGIMES:
        rg = r213["regimes"][reg]
        spines[reg] = {
            "floor_spine": [float(x) for x in rg["floor_spine_at_lambda_hat"]],
            "ceiling_spine": [float(x) for x in rg["ceiling_spine"]],
        }
        # #213 banked endpoint E[T] (reach-DP on those spines must reproduce these).
        anc = rg["endpoint_anchors"]
        et_banked_213[reg] = {
            "et_floor": float(anc["overhead_budget_at_lambda_hat_0342"]["E_T"]),
            "et_ceiling": float(anc["overhead_budget_at_lambda_1"]["E_T"]),
        }

    out = {
        # speed gate (my #218)
        "trigger_tight": float(trig_band[0]),                          # 512.5194519580790
        "trigger_central": float(gt["go_trigger_grounded_central"]),   # 512.7353207419002
        "trigger_worstcase": float(gt["go_trigger_grounded_worstcase"]),  # 513.5574577506176
        "combined_sigma_central": float(gt["combined_sigma_grounded_central"]),    # 7.7425252516
        "combined_sigma_tight": float(sig_band[0]),                    # 7.6112863497
        "combined_sigma_worstcase": float(sig_band[1]),                # 8.2423490628
        "lambda1_ceiling_mu": float(gt["lambda1_ceiling_mu"]),         # 520.9527323112 (#204)
        "z1_one_sided_p95": float(r218["imported_legs_204"]["z1_one_sided_p95"]),  # 1.6448536270
        # validity gate (stark #208/#215)
        "validity_bar": validity_bar,
        "validity_bar_nominal": VALIDITY_BAR_NOMINAL,
        "lambda_floor": lambda_floor,
        "lambda_floor_213": lambda_floor_213,
        # reach-DP shape (#213)
        "spines": spines,
        "et_banked_213": et_banked_213,
        # composition (#175/#184/#148/#168)
        "K_cal": K_CAL,
        "step": STEP,
    }
    return out


# ---------------------------------------------------------------------------
# The E[T](lambda_hat) -> mu_pub map (ceiling-anchored reach-DP).
# ---------------------------------------------------------------------------
class LambdaMuMap:
    """mu_pub(lambda_hat) = ceiling * E[T](lambda_hat) / E[T](1), lambda_hat in [lambda_floor, 1].

    E[T] via the #213 per-depth linear self-KV-recovery blend of the banked floor/ceiling spines
    (reach-DP pmf-mean). The lambda=1 endpoint is anchored to the imported #204 ceiling 520.953,
    so mu_pub round-trips the ceiling EXACTLY (ratio=1 at lambda=1) and inherits the reach-DP's
    monotone shape below it.
    """

    def __init__(self, floor_spine: list[float], ceil_spine: list[float],
                 lambda_floor: float, ceiling_mu: float):
        self.floor = list(floor_spine)
        self.ceil = list(ceil_spine)
        self.lambda_floor = float(lambda_floor)
        self.ceiling_mu = float(ceiling_mu)
        self.ceil_ge_floor_all_depths = all(c >= f - 1e-12 for f, c in zip(self.floor, self.ceil))
        self.et_floor = _reachdp_et(self.floor)              # E[T] at lambda=lambda_floor (t=0)
        self.et_ceiling = _reachdp_et(self.ceil)             # E[T] at lambda=1 (t=1)
        # tau implied by anchoring the lambda=1 reach-DP E[T] to the imported ceiling mu.
        self.tau_anchor = self.ceiling_mu * STEP / (K_CAL * self.et_ceiling)

    def t_of_lambda(self, lam: float) -> float:
        return (lam - self.lambda_floor) / (1.0 - self.lambda_floor)

    def spine_of_lambda(self, lam: float) -> list[float]:
        t = self.t_of_lambda(lam)
        return [(1.0 - t) * f + t * c for f, c in zip(self.floor, self.ceil)]

    def et_of_lambda(self, lam: float) -> float:
        return _reachdp_et(self.spine_of_lambda(lam))

    def mu_pub(self, lam: float) -> float:
        return self.ceiling_mu * self.et_of_lambda(lam) / self.et_ceiling

    def official_tps_form(self, lam: float) -> float:
        """The PR's `official = K_cal*(E[T]/step)*tau` written with the anchor-tau -- identical
        to mu_pub() (a cross-check that the ratio form and the official form agree)."""
        return K_CAL * (self.et_of_lambda(lam) / STEP) * self.tau_anchor

    def solve_lambda_for_mu(self, target_mu: float) -> float | None:
        """Smallest lambda in [lambda_floor, 1] with mu_pub(lambda)=target (mu monotone increasing).
        None if target is outside [mu_pub(floor), ceiling]."""
        lo, hi = self.lambda_floor, 1.0
        f_lo = self.mu_pub(lo) - target_mu
        f_hi = self.mu_pub(hi) - target_mu
        if f_lo == 0.0:
            return lo
        if f_hi == 0.0:
            return hi
        if (f_lo > 0.0) == (f_hi > 0.0):
            return None
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if (self.mu_pub(mid) - target_mu > 0.0) == (f_lo > 0.0):
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def build_maps(b: dict[str, Any]) -> dict[str, LambdaMuMap]:
    return {
        reg: LambdaMuMap(
            b["spines"][reg]["floor_spine"], b["spines"][reg]["ceiling_spine"],
            b["lambda_floor"], b["lambda1_ceiling_mu"],
        )
        for reg in REGIMES
    }


# ---------------------------------------------------------------------------
# Step 1 -- the two gates in one frame (the mechanism + provenance).
# ---------------------------------------------------------------------------
def gate_frame(b: dict[str, Any]) -> dict[str, Any]:
    return {
        "launch_go_iff": "BOTH (validity: private lambda_hat >= 0.9780) AND (speed: public mu_pub >= 513.557)",
        "validity_gate": {
            "bar": b["validity_bar"],
            "bar_nominal": b["validity_bar_nominal"],
            "definition": "private depth-aggregate acceptance lambda_hat >= 0.9780",
            "provenance": "stark #208 (wi4gxxx8) worst-case private go-bar `d198_coupled_bar`; #215 deep-tail framing",
        },
        "speed_gate": {
            "trigger_worstcase": b["trigger_worstcase"],
            "trigger_central": b["trigger_central"],
            "trigger_tight": b["trigger_tight"],
            "definition": "public mu_pub >= 513.557 (the trigger at which the one-sided P95 LCB clears 500)",
            "provenance": "ubel #218 (0ug7vd7d) grounded worst-case GO trigger; 500 + z1*combined_sigma_worstcase",
        },
        "shared_driver": (
            "land #71 builds ONE checkpoint whose acceptance lambda_hat drives BOTH the "
            "E[T](lambda_hat)->mu_pub speed map AND the private validity bar."
        ),
        "map": "mu_pub(lambda_hat) = 520.953 * E[T](lambda_hat)/E[T](1); E[T] via #213 reach-DP blend (#175/#184)",
    }


# ---------------------------------------------------------------------------
# Step 2-3 -- mu_pub at the validity bar + which gate binds, per regime.
# ---------------------------------------------------------------------------
def evaluate_regime(b: dict[str, Any], mp: LambdaMuMap) -> dict[str, Any]:
    z = Z1_ONE_SIDED
    bar = b["validity_bar"]
    floor = b["lambda_floor"]
    ceiling = b["lambda1_ceiling_mu"]

    mu_bar = mp.mu_pub(bar)
    mu_floor = mp.mu_pub(floor)
    mu_one = mp.mu_pub(1.0)

    # P95 one-sided LCB of the public draw at the validity bar (#218 combined-sigma band).
    lcb_central = mu_bar - z * b["combined_sigma_central"]
    lcb_worstcase = mu_bar - z * b["combined_sigma_worstcase"]

    # binding verdict at each trigger corner.
    triggers = {
        "tight": b["trigger_tight"],
        "central": b["trigger_central"],
        "worstcase": b["trigger_worstcase"],
    }
    binding_vs_trigger: dict[str, Any] = {}
    for corner, trig in triggers.items():
        lam_speed = mp.solve_lambda_for_mu(trig)
        # mu monotone increasing -> mu_bar >= trig <=> lam_speed <= bar.
        binds_validity = bool(mu_bar >= trig)
        lam_speed_val = lam_speed if lam_speed is not None else float("nan")
        build_target = max(bar, lam_speed_val) if _finite(lam_speed_val) else bar
        binding_vs_trigger[corner] = {
            "trigger": trig,
            "mu_pub_at_validity_bar": mu_bar,
            "binding_gate": "validity" if binds_validity else "speed",
            "speed_margin_at_validity_bar": mu_bar - trig,        # >0 when validity binds
            "lambda_hat_speed": lam_speed_val,                    # lambda where mu_pub = trigger
            "lambda_hat_speed_roundtrip_mu": mp.mu_pub(lam_speed_val) if _finite(lam_speed_val) else float("nan"),
            "gap_lambda_speed_minus_bar": (lam_speed_val - bar) if _finite(lam_speed_val) else float("nan"),
            "lambda_hat_build_target": build_target,              # max(0.9780, lambda_hat_speed)
        }

    return {
        "tau_anchor": mp.tau_anchor,
        "et_floor": mp.et_floor,
        "et_validity_bar": mp.et_of_lambda(bar),
        "et_ceiling": mp.et_ceiling,
        "mu_pub_at_validity_bar": mu_bar,
        "mu_pub_at_floor": mu_floor,
        "mu_pub_at_ceiling": mu_one,
        "mu_pub_at_validity_bar_lcb_p95_central": lcb_central,
        "mu_pub_at_validity_bar_lcb_p95_worstcase": lcb_worstcase,
        "official_form_cross_check_err": abs(mp.official_tps_form(bar) - mu_bar),
        "binding_vs_trigger": binding_vs_trigger,
        # headline (worst-case trigger).
        "binding_gate": binding_vs_trigger["worstcase"]["binding_gate"],
        "speed_margin_at_validity_bar": binding_vs_trigger["worstcase"]["speed_margin_at_validity_bar"],
        "lambda_hat_speed": binding_vs_trigger["worstcase"]["lambda_hat_speed"],
        "lambda_hat_build_target": binding_vs_trigger["worstcase"]["lambda_hat_build_target"],
        "ceil_ge_floor_all_depths": mp.ceil_ge_floor_all_depths,
    }


# ---------------------------------------------------------------------------
# Step 4 -- sensitivity sweep across the #218 grounded trigger band + forward grid.
# ---------------------------------------------------------------------------
def sensitivity_sweep(b: dict[str, Any], mp: LambdaMuMap) -> dict[str, Any]:
    bar = b["validity_bar"]
    mu_bar = mp.mu_pub(bar)
    rows = []
    for corner, trig in (("tight", b["trigger_tight"]), ("central", b["trigger_central"]),
                         ("worstcase", b["trigger_worstcase"])):
        lam_speed = mp.solve_lambda_for_mu(trig)
        lam_speed_val = lam_speed if lam_speed is not None else float("nan")
        rows.append({
            "corner": corner,
            "trigger": trig,
            "binding_gate": "validity" if mu_bar >= trig else "speed",
            "lambda_hat_speed": lam_speed_val,
            "lambda_hat_build_target": (max(bar, lam_speed_val) if _finite(lam_speed_val) else bar),
            "speed_margin_at_validity_bar": mu_bar - trig,
        })
    validity_binds_every_corner = all(r["binding_gate"] == "validity" for r in rows)
    return {
        "rows": rows,
        "validity_binds_at_every_trigger_corner": validity_binds_every_corner,
        "honest_band_note": (
            "(a) the public->private drop makes mu_pub_at_validity_bar CONSERVATIVE (true public "
            "mu_pub is higher -> validity MORE likely to bind, the favorable direction); (b) the "
            "E[T](lambda_hat) reach-DP shape and the 520.953 ceiling are imported unchanged; (c) "
            "validity binds at EVERY trigger corner [tight, central, worst-case] -> the "
            "launch-simplifying headline: land #71's single target is the validity bar 0.9780, and "
            "clearing it auto-clears the speed gate."
        ),
    }


def forward_grid(b: dict[str, Any], mp: LambdaMuMap) -> dict[str, Any]:
    bar = b["validity_bar"]
    floor = b["lambda_floor"]
    grid = sorted({floor, 0.5, 0.6, 0.7, 0.8, 0.9, round(bar, 7), 0.99, 1.0})
    rows = []
    prev = None
    monotone = True
    for lam in grid:
        mu = mp.mu_pub(lam)
        if prev is not None and mu < prev - 1e-12:
            monotone = False
        prev = mu
        rows.append({
            "lambda_hat": lam,
            "E_T": mp.et_of_lambda(lam),
            "mu_pub": mu,
            "is_floor": bool(abs(lam - floor) < 1e-9),
            "is_validity_bar": bool(abs(lam - round(bar, 7)) < 1e-9),
            "is_ceiling": bool(lam == 1.0),
        })
    return {"rows": rows, "monotone_increasing": monotone}


# ---------------------------------------------------------------------------
# Step 5 -- self-test (PRIMARY).
# ---------------------------------------------------------------------------
def self_test(b: dict[str, Any], maps: dict[str, LambdaMuMap],
              per_regime: dict[str, Any]) -> dict[str, Any]:
    bar = b["validity_bar"]
    floor = b["lambda_floor"]
    ceiling = b["lambda1_ceiling_mu"]
    trig_wc = b["trigger_worstcase"]

    # (a) mu_pub(lambda=1) = 520.953 EXACTLY (round-trip the #204 ceiling), both regimes.
    a_errs = {reg: abs(maps[reg].mu_pub(1.0) - ceiling) for reg in REGIMES}
    a_ok = bool(all(e < ANCHOR_TOL for e in a_errs.values()))

    # (b) mu_pub(lambda_hat) MONOTONE INCREASING in lambda_hat (fine grid), both regimes.
    b_mono = {}
    for reg in REGIMES:
        mp = maps[reg]
        grid = [floor + (1.0 - floor) * i / 60.0 for i in range(61)]
        mus = [mp.mu_pub(x) for x in grid]
        diffs = [mus[i + 1] - mus[i] for i in range(len(mus) - 1)]
        b_mono[reg] = {
            "monotone_increasing": bool(all(d >= -1e-12 for d in diffs)),
            "strictly_increasing": bool(all(d > 0.0 for d in diffs)),
        }
    b_ok = bool(all(m["monotone_increasing"] for m in b_mono.values()))

    # (c) mu_pub_at_validity_bar in (mu_pub at floor lambda_hat=0.342, 520.953) -- sane, both regimes.
    c_detail = {}
    for reg in REGIMES:
        mu_bar = per_regime[reg]["mu_pub_at_validity_bar"]
        mu_floor = per_regime[reg]["mu_pub_at_floor"]
        c_detail[reg] = {
            "mu_pub_floor": mu_floor, "mu_pub_bar": mu_bar, "ceiling": ceiling,
            "in_open_interval": bool(mu_floor + SANE_TOL < mu_bar < ceiling - SANE_TOL),
        }
    c_ok = bool(all(d["in_open_interval"] for d in c_detail.values()))

    # (d) binding logic + inverse-map round-trip (headline regime, worst-case trigger).
    mph = maps[HEADLINE_REGIME]
    mu_bar_h = per_regime[HEADLINE_REGIME]["mu_pub_at_validity_bar"]
    binds_validity = bool(mu_bar_h >= trig_wc)
    build_target = per_regime[HEADLINE_REGIME]["binding_vs_trigger"]["worstcase"]["lambda_hat_build_target"]
    lam_speed = mph.solve_lambda_for_mu(trig_wc)
    lam_speed_val = lam_speed if lam_speed is not None else float("nan")
    roundtrip_err = abs(mph.mu_pub(lam_speed_val) - trig_wc) if _finite(lam_speed_val) else float("inf")
    if binds_validity:
        # validity binds -> build target is exactly the bar, and lambda_hat_speed < bar.
        d_logic = bool(abs(build_target - bar) < SANE_TOL and lam_speed_val < bar + SANE_TOL)
    else:
        # speed binds -> lambda_hat_speed > bar, and it IS the build target.
        d_logic = bool(lam_speed_val > bar - SANE_TOL and abs(build_target - lam_speed_val) < SANE_TOL)
    d_ok = bool(d_logic and roundtrip_err < ANCHOR_TOL)

    # (e) reproduces #218's 513.557 worst-case trigger as the speed bar.
    e_err = abs(b["trigger_worstcase"] - 513.5574577506176)
    e_ok = bool(e_err < 1e-6)

    # provenance lock: reach-DP on the banked spines reproduces #213's banked E[T].
    prov_detail = {}
    for reg in REGIMES:
        mp = maps[reg]
        prov_detail[reg] = {
            "et_floor_resid": abs(mp.et_floor - b["et_banked_213"][reg]["et_floor"]),
            "et_ceiling_resid": abs(mp.et_ceiling - b["et_banked_213"][reg]["et_ceiling"]),
            "ceil_ge_floor_all_depths": mp.ceil_ge_floor_all_depths,
        }
    prov_ok = bool(all(
        d["et_floor_resid"] < PROV_TOL and d["et_ceiling_resid"] < PROV_TOL and d["ceil_ge_floor_all_depths"]
        for d in prov_detail.values()
    ))

    # (f) NaN-clean across all reported scalars.
    scalars = [
        ceiling, trig_wc, bar, floor, mu_bar_h, lam_speed_val, roundtrip_err, e_err,
        *a_errs.values(),
    ]
    for reg in REGIMES:
        pr = per_regime[reg]
        scalars += [
            pr["mu_pub_at_validity_bar"], pr["mu_pub_at_floor"], pr["mu_pub_at_ceiling"],
            pr["mu_pub_at_validity_bar_lcb_p95_central"], pr["mu_pub_at_validity_bar_lcb_p95_worstcase"],
            pr["tau_anchor"], pr["speed_margin_at_validity_bar"], pr["lambda_hat_speed"],
            pr["lambda_hat_build_target"], pr["official_form_cross_check_err"],
        ]
    f_ok = all(_finite(x) for x in scalars)

    checks = {
        "a_mu_pub_lambda1_roundtrips_ceiling_520p953": a_ok,
        "b_mu_pub_monotone_increasing_in_lambda_hat": b_ok,
        "c_mu_pub_at_bar_between_floor_and_ceiling": c_ok,
        "d_binding_logic_and_inverse_roundtrips": d_ok,
        "e_reproduces_218_worstcase_trigger_513p557": e_ok,
        "f_nan_clean": f_ok,
        "provenance_reachdp_on_banked_spines_reproduces_213_ET": prov_ok,
    }
    passes = all(checks.values())
    return {
        "binding_gate_self_test_passes": bool(passes),                                  # <-- PRIMARY
        "mu_pub_at_validity_bar": per_regime[HEADLINE_REGIME]["mu_pub_at_validity_bar"],  # <-- TEST
        "checks": checks,
        "evidence": {
            "a_ceiling_roundtrip_errs": a_errs,
            "b_monotonicity": b_mono,
            "c_sanity": c_detail,
            "d_binds_validity": binds_validity,
            "d_lambda_hat_speed": lam_speed_val,
            "d_build_target": build_target,
            "d_inverse_roundtrip_err": roundtrip_err,
            "e_trigger_err": e_err,
            "provenance": prov_detail,
            "n_scalars_checked": len(scalars),
        },
    }


# ---------------------------------------------------------------------------
# Assemble the result.
# ---------------------------------------------------------------------------
def _build_result(b, frame, per_regime, sweep, grids, st) -> dict[str, Any]:
    head = per_regime[HEADLINE_REGIME]
    mu_bar = head["mu_pub_at_validity_bar"]
    binding = head["binding_gate"]
    margin = head["speed_margin_at_validity_bar"]
    lam_speed = head["lambda_hat_speed"]
    build_target = head["lambda_hat_build_target"]
    lcb_c = head["mu_pub_at_validity_bar_lcb_p95_central"]
    lcb_w = head["mu_pub_at_validity_bar_lcb_p95_worstcase"]

    handoff = (
        "fern #185 + land #71: at the validity bar lambda_hat=0.9780 the build shows public "
        "mu_pub = %.3f (conservative -- public acceptance >= private, so true mu_pub is higher; "
        "P95-LCB %.3f central / %.3f worst-case) vs the 513.557 speed trigger, so the %s gate "
        "binds (speed margin %+.3f TPS; the acceptance that clears the speed trigger is "
        "lambda_hat_speed=%.4f, BELOW the bar) -- land #71's single launch target is %.4f, and "
        "fern #185 reads the VALIDITY gate because clearing it AUTO-CLEARS the speed trigger at "
        "every grounded trigger corner [512.519, 512.735, 513.557]."
        % (mu_bar, lcb_c, lcb_w, binding.upper(), margin, lam_speed, build_target)
    )

    return {
        "pr": 222,
        "metric_primary": "binding_gate_self_test_passes",
        "metric_test": "mu_pub_at_validity_bar",
        "binding_gate_self_test_passes": st["binding_gate_self_test_passes"],
        "mu_pub_at_validity_bar": mu_bar,
        "binding_gate": binding,
        "speed_margin_at_validity_bar": margin,
        "lambda_hat_speed": lam_speed,
        "lambda_hat_build_target": build_target,
        "mu_pub_at_validity_bar_lcb_p95_central": lcb_c,
        "mu_pub_at_validity_bar_lcb_p95_worstcase": lcb_w,
        "validity_binds_at_every_trigger_corner":
            sweep[HEADLINE_REGIME]["validity_binds_at_every_trigger_corner"],
        "headline_regime": HEADLINE_REGIME,
        "law": (
            "mu_pub(lambda_hat) = 520.953 * E[T](lambda_hat)/E[T](1) = K_cal*(E[T](lambda_hat)/step)*tau_anchor; "
            "E[T](lambda_hat) = reach_DP_pmf_mean((1-t)*floor_spine + t*ceiling_spine), "
            "t = (lambda_hat - lambda_floor)/(1 - lambda_floor); the lambda=1 endpoint is anchored to the "
            "imported #204 ceiling 520.953, tau_anchor = 520.953*step/(K_cal*E[T](1)). VALIDITY binds iff "
            "mu_pub(0.9780) >= 513.557 (else SPEED binds at lambda_hat_speed > 0.9780)."
        ),
        "gate_frame": frame,
        "per_regime": per_regime,
        "sensitivity": sweep,
        "forward_grid": grids,
        "imported": {
            "validity_bar": b["validity_bar"],
            "validity_bar_nominal": b["validity_bar_nominal"],
            "lambda_floor": b["lambda_floor"],
            "trigger_tight": b["trigger_tight"],
            "trigger_central": b["trigger_central"],
            "trigger_worstcase": b["trigger_worstcase"],
            "combined_sigma_central": b["combined_sigma_central"],
            "combined_sigma_worstcase": b["combined_sigma_worstcase"],
            "lambda1_ceiling_mu": b["lambda1_ceiling_mu"],
            "z1_one_sided_p95": b["z1_one_sided_p95"],
            "K_cal": b["K_cal"],
            "step": b["step"],
            "et_banked_213": b["et_banked_213"],
        },
        "self_test": st,
        "handoff": handoff,
        "scope": (
            "LOCAL CPU-only analytic cross of ubel's #218 grounded speed trigger (513.557 worst-case / "
            "512.735 central) against stark #208's validity bar (0.9780) via the banked #213 E[T](lambda_hat) "
            "reach-DP map (#175/#184), the lambda=1 endpoint anchored to the imported #204 ceiling 520.953. "
            "Computes mu_pub at the validity bar and resolves WHICH gate binds. The public->private drop is "
            "carried as a CONSERVATIVE (favorable) direction (true public mu_pub is higher). Takes NO official "
            "draws, authorizes none. The sigma's, bar 0.9780, trigger 513.557, ceiling 520.953 are imported "
            "unchanged. BASELINE stays 481.53; adds 0 TPS (PRIMARY = self-test); greedy/PPL untouched. "
            "NOT a launch. NOT open2."
        ),
        "public_evidence_used": [
            "ubel #218 (0ug7vd7d, interleg_rho, MERGED) -- the grounded worst-case GO trigger 513.557 "
            "(central 512.735, tight 512.519), combined-sigma central 7.7425 / worst 8.2423, z1=1.64485, "
            "lambda=1 ceiling 520.953: the SPEED gate this leg crosses against validity.",
            "ubel #204 (launch_sigma_unit_rebase, MERGED) -- the int4-spec lambda=1 ceiling mu_pub(1)=520.953 "
            "that the E[T] map's top endpoint is anchored to.",
            "stark #208 (wi4gxxx8) -- the worst-case private go-bar `d198_coupled_bar`=0.9780 (the VALIDITY "
            "gate), imported via the multivertex-realizability banked constants.",
            "stark #215 (deeptail_bar_budget) -- the deep-tail build-bar budget framing of the 0.9780 validity bar.",
            "wirbel #213 (5o7zcj8s, kernel_budget_lambda) -- the banked per-regime floor/ceiling self-KV-recovery "
            "spines + the lambda in [0.342,1.0] linear-recovery blend; the reach-DP E[T](lambda_hat) SHAPE.",
            "wirbel #199/#175/#184 -- the accepted-length reach-DP (et_via_reachdp) that maps a per-depth "
            "acceptance spine to E[T]; imported verbatim (reproduces #213's banked E[T] bit-exactly).",
        ],
        "method": (
            "LOCAL CPU-only analytic synthesis over EXISTING MERGED results (#218 trigger + #204 ceiling + "
            "#208 bar + #213 reach-DP spines); the reach-DP is imported as a pure function of a spine (no "
            "PR#86 shard, no GPU/vLLM/HF Job/submission/served-file change). BASELINE stays 481.53; adds 0 "
            "TPS. Greedy/PPL identity untouched. NOT a launch."
        ),
        "metrics_nan_clean": 1 if st["checks"]["f_nan_clean"] else 0,
    }


# ---------------------------------------------------------------------------
# W&B logging (mirrors #218; never fatal).
# ---------------------------------------------------------------------------
def _log_wandb(args, result: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[binding-gate] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="launch-sigma-binding-gate", agent="ubel",
            name=args.wandb_name or "ubel/binding-gate",
            group=args.wandb_group,
            tags=["launch-sigma", "binding-gate", "validity-vs-speed", "validity-gate",
                  "speed-trigger", "et-lambda-map", "pr222"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic",
                    "z_one_sided_p95": Z1_ONE_SIDED, "validity_bar": result["imported"]["validity_bar"],
                    "trigger_worstcase": result["imported"]["trigger_worstcase"],
                    "trigger_central": result["imported"]["trigger_central"],
                    "lambda1_ceiling_mu": result["imported"]["lambda1_ceiling_mu"],
                    "headline_regime": HEADLINE_REGIME, "K_cal": K_CAL, "step": STEP},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[binding-gate] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[binding-gate] wandb disabled; skipping", flush=True)
        return
    try:
        st = result["self_test"]
        head = result["per_regime"][HEADLINE_REGIME]
        desc = result["per_regime"]["descent_only"]
        flat = {
            # PRIMARY + TEST
            "binding_gate_self_test_passes": 1.0 if st["binding_gate_self_test_passes"] else 0.0,
            "mu_pub_at_validity_bar": result["mu_pub_at_validity_bar"],
            # headline verdict
            "binding_gate_is_validity": 1.0 if result["binding_gate"] == "validity" else 0.0,
            "speed_margin_at_validity_bar": result["speed_margin_at_validity_bar"],
            "lambda_hat_speed": result["lambda_hat_speed"],
            "lambda_hat_build_target": result["lambda_hat_build_target"],
            "mu_pub_at_validity_bar_lcb_p95_central": result["mu_pub_at_validity_bar_lcb_p95_central"],
            "mu_pub_at_validity_bar_lcb_p95_worstcase": result["mu_pub_at_validity_bar_lcb_p95_worstcase"],
            "validity_binds_at_every_trigger_corner":
                1.0 if result["validity_binds_at_every_trigger_corner"] else 0.0,
            # both regimes (mu_pub at bar + anchor tau)
            "mu_pub_at_validity_bar_both_bugs": head["mu_pub_at_validity_bar"],
            "mu_pub_at_validity_bar_descent_only": desc["mu_pub_at_validity_bar"],
            "mu_pub_at_floor_both_bugs": head["mu_pub_at_floor"],
            "mu_pub_at_ceiling_both_bugs": head["mu_pub_at_ceiling"],
            "tau_anchor_both_bugs": head["tau_anchor"],
            "tau_anchor_descent_only": desc["tau_anchor"],
            "et_validity_bar_both_bugs": head["et_validity_bar"],
            "et_ceiling_both_bugs": head["et_ceiling"],
            # imported gates
            "trigger_worstcase": result["imported"]["trigger_worstcase"],
            "trigger_central": result["imported"]["trigger_central"],
            "trigger_tight": result["imported"]["trigger_tight"],
            "validity_bar": result["imported"]["validity_bar"],
            "lambda_floor": result["imported"]["lambda_floor"],
            "lambda1_ceiling_mu": result["imported"]["lambda1_ceiling_mu"],
            "combined_sigma_central": result["imported"]["combined_sigma_central"],
            "combined_sigma_worstcase": result["imported"]["combined_sigma_worstcase"],
            # per-check booleans
            "self_test_a_ceiling_roundtrip": 1.0 if st["checks"]["a_mu_pub_lambda1_roundtrips_ceiling_520p953"] else 0.0,
            "self_test_b_monotone": 1.0 if st["checks"]["b_mu_pub_monotone_increasing_in_lambda_hat"] else 0.0,
            "self_test_c_sane": 1.0 if st["checks"]["c_mu_pub_at_bar_between_floor_and_ceiling"] else 0.0,
            "self_test_d_binding_logic": 1.0 if st["checks"]["d_binding_logic_and_inverse_roundtrips"] else 0.0,
            "self_test_e_reproduces_218": 1.0 if st["checks"]["e_reproduces_218_worstcase_trigger_513p557"] else 0.0,
            "self_test_f_nan_clean": 1.0 if st["checks"]["f_nan_clean"] else 0.0,
            "self_test_provenance": 1.0 if st["checks"]["provenance_reachdp_on_banked_spines_reproduces_213_ET"] else 0.0,
        }
        # per-trigger-corner binding rows (headline regime).
        for r in result["sensitivity"][HEADLINE_REGIME]["rows"]:
            flat[f"binding_is_validity_{r['corner']}"] = 1.0 if r["binding_gate"] == "validity" else 0.0
            flat[f"lambda_hat_speed_{r['corner']}"] = r["lambda_hat_speed"]
            flat[f"speed_margin_{r['corner']}"] = r["speed_margin_at_validity_bar"]
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="binding_gate", artifact_type="launch-sigma-binding-gate", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[binding-gate] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    st = result["self_test"]
    head = result["per_regime"][HEADLINE_REGIME]
    print("\n[binding-gate] ===== BINDING GATE: VALIDITY (0.9780) vs SPEED (513.557) (PR #222) =====", flush=True)
    print(f"  gates: validity lambda_hat >= {result['imported']['validity_bar']:.6f} (#208) AND "
          f"public mu_pub >= {result['imported']['trigger_worstcase']:.3f} (#218 worst-case)", flush=True)
    print(f"  map: mu_pub(lambda_hat) = 520.953 * E[T](lambda_hat)/E[T](1)  (reach-DP shape, "
          f"ceiling-anchored)", flush=True)
    print("  forward grid (both_bugs):", flush=True)
    for r in result["forward_grid"][HEADLINE_REGIME]["rows"]:
        tag = ""
        if r["is_floor"]:
            tag = "  <- floor"
        elif r["is_validity_bar"]:
            tag = "  <- VALIDITY BAR"
        elif r["is_ceiling"]:
            tag = "  <- ceiling"
        print(f"    lambda_hat={r['lambda_hat']:.6f}  E[T]={r['E_T']:.5f}  mu_pub={r['mu_pub']:8.3f}{tag}", flush=True)
    print(f"  mu_pub_at_validity_bar = {result['mu_pub_at_validity_bar']:.4f}  "
          f"(LCB P95: {result['mu_pub_at_validity_bar_lcb_p95_central']:.3f} central / "
          f"{result['mu_pub_at_validity_bar_lcb_p95_worstcase']:.3f} worst-case)", flush=True)
    print(f"  tau_anchor: both_bugs={head['tau_anchor']:.6f}  "
          f"descent_only={result['per_regime']['descent_only']['tau_anchor']:.6f}", flush=True)
    print("  binding_vs_trigger (both_bugs):", flush=True)
    for corner in ("tight", "central", "worstcase"):
        v = head["binding_vs_trigger"][corner]
        print(f"    {corner:9s} trig={v['trigger']:8.3f}  binds={v['binding_gate']:8s}  "
              f"margin={v['speed_margin_at_validity_bar']:+.3f}  lambda_speed={v['lambda_hat_speed']:.4f}  "
              f"build_target={v['lambda_hat_build_target']:.4f}", flush=True)
    print(f"  VALIDITY binds at every trigger corner? "
          f"{result['validity_binds_at_every_trigger_corner']}", flush=True)
    print(f"\n  SELF-TEST (PRIMARY) passes = {st['binding_gate_self_test_passes']}  "
          f"mu_pub_at_validity_bar (TEST) = {st['mu_pub_at_validity_bar']:.4f} TPS", flush=True)
    for k, v in st["checks"].items():
        print(f"    [{'ok' if v else 'XX'}] {k}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Binding gate: validity (0.9780) or speed (513.557) -- which launch gate binds? (PR #222)")
    ap.add_argument("--out", default=os.path.join(_HERE, "binding_gate_results.json"))
    ap.add_argument("--wandb-name", "--wandb_name", default="ubel/binding-gate")
    ap.add_argument("--wandb-group", "--wandb_group", default="launch-sigma-unit-rebase")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    b = import_banked()
    maps = build_maps(b)
    frame = gate_frame(b)
    per_regime = {reg: evaluate_regime(b, maps[reg]) for reg in REGIMES}
    sweep = {reg: sensitivity_sweep(b, maps[reg]) for reg in REGIMES}
    grids = {reg: forward_grid(b, maps[reg]) for reg in REGIMES}
    st = self_test(b, maps, per_regime)

    result = _build_result(b, frame, per_regime, sweep, grids, st)
    result["elapsed_s"] = round(time.time() - t0, 4)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    _print(result)
    print(f"\n[binding-gate] HANDOFF: {result['handoff']}", flush=True)
    print(f"[binding-gate] artifacts -> {args.out}", flush=True)
    _log_wandb(args, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
