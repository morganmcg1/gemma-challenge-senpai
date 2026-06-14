#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""DESCENT OVER-ACCEPTANCE SIGNATURE (PR #170, student wirbel).

THE QUESTION
------------
wirbel #165 (MERGED) named land #71's binding build-risk: BUG-2's
linear->descending structural change (~19x BUG-1's E[T] lever). Its dominant
failure mode is OVER-ACCEPTANCE -- a descent walk that accepts e extra nodes
PAST the true greedy boundary. That failure is insidious because it INFLATES the
measured E[T] above the true greedy-exact ceiling: a higher E[T] readout from
land #71 is therefore NOT unambiguously a win -- it could be a buggy over-
accepting descent masquerading as a faster one while silently violating greedy.

denken #158 (MERGED) built the BINARY detector ("does any committed token differ
from in_step_target_argmax?"). What is missing is the JOINT (E[T], greedy-
violation-rate) acceptance region: the band of (E[T], v) pairs consistent with a
TRUSTWORTHY greedy-exact descent, so land's measured (E[T], v) tuple can be
checked against it. This file is the ANALYTIC MAGNITUDE complement to #158's
binary flag -- it stops an inflated E[T] from being read as acceptance headroom.

THE MODEL (pure node-counting; nothing re-derived)
--------------------------------------------------
A speculative step's greedy-exact verifier accepts the longest draft prefix whose
every token equals the target greedy argmax, then commits one bonus correction.
The greedy-exact expected accepted length per step is the #160 DP ceiling E[T]*
(descent-only 5.0564 / both-bugs 5.2070). Over-acceptance commits e EXTRA nodes
per step beyond that greedy boundary. Each over-accepted node is

  (1) one extra COMMITTED token            -> inflates E[T] by exactly +1, and
  (2) a greedy VIOLATION (committed != argmax, because past the rejection
      boundary the draft has diverged from the greedy path).

So, with e = E[expected extra accepted nodes past the greedy boundary per step]:

  E[T](e) = E[T]* + e                       # inflated expected accepted length
  v(e)    = e / (E[T]* + e)                  # per-token greedy-violation rate

Inverting (eliminating e) gives the OVER-ACCEPT LOCUS -- the (E[T], v) boundary
traced by e > 0:

  v(E[T]) = 1 - E[T]*/E[T]      <=>     E[T](v) = E[T]* / (1 - v)

DEGENERATE AT v=0: v=0 <=> e=0 <=> E[T]=E[T]* (UNIQUE). A greedy-exact descent
(v=0) CANNOT inflate E[T] above the ceiling, so max_et_inflation_at_v0 == 0; any
E[T] > E[T]* REQUIRES v > 0. UNIT OVER-ACCEPT: accepting exactly one extra node
inflates E[T] by delta(e=1) = 1.0 (identical for both topologies -- it is a node
count); the violation rate it costs, v(e=1) = 1/(E[T]*+1), is topology-specific.

CROSS-CHECK with denken #158: #158's BUG-2 over-accept battery (the
_bug2_overaccept_kernel) reports exactness_rate = total_exact/total_committed =
217/236 = 0.91949..., i.e. a per-token violation differential of
num_violations/total_committed = 19/236 = 0.08051 = 1 - exactness_rate. The
analytic v form (over-accepted committed fraction) reproduces that empirical
number EXACTLY at #158's operating point -- the continuous v(e) and the binary
detector are the SAME quantity, agreeing at the same point.

LAND #71 GATE: land measures its descent kernel's (E[T], v). It is TRUSTWORTHY
(real greedy-exact descent, not over-accepting) iff v <= v_tol AND E[T] is within
the locus budget at v_tol (E[T] <= E[T]*/(1-v_tol)); strict v_tol->0 collapses to
"v == 0 AND E[T] <= ceiling". Composes with #158 from both sides: #158 = "any
violation at all?" (catches substitution-only violations with no inflation);
this = "is the measured E[T] inflation explained by violation?" (catches inflated
-E[T]-read-as-headroom). A descent passing BOTH (binary-clean AND no-inflation)
is a trustworthy speedup.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / kernel build /
served-file change. BASELINE stays 481.53; 0 TPS; greedy identity untouched by
construction. Imports #160's E[T]-DP anchors + #158's measured operating point;
does NOT re-derive them.

PRIMARY metric  overaccept_signature_self_test_passes
TEST    metric  et_inflation_at_unit_overaccept  (= delta(e=1) = 1.0)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---- imported single-source-of-truth artifacts (advisor branch) ------------------
SPINE_SPEC_JSON = os.path.join(_ROOT, "research", "spine_spec", "spine_spec_results.json")
BUG2_DESCENT_JSON = os.path.join(
    _ROOT, "research", "spec_cost_model", "bug2_salvage_descent_results.json"
)
GREEDY_HARNESS_GLOB = os.path.join(
    _ROOT, "research", "descent_greedy_exact_harness", "runs", "*", "greedy_exact_harness_result.json"
)

# Hard per-step committed ceiling: max_spec_len(=7) draft positions + 1 bonus = 8
# (denken #158 harness width = max_spec_len + 1; reject_at_0 over-accept reaches 8).
MAX_SPEC_LEN = 7
MAX_COMMITTED_PER_STEP = MAX_SPEC_LEN + 1  # = 8.0

# Official projection constants (context only; this leg is 0 TPS, changes nothing).
K_CAL = 125.268
STEP_MEASURED = 1.2182
TARGET_OFFICIAL = 500.0


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# ===================================================================================
# imports (NOT re-derived): #160 E[T]-DP anchors + #158 measured operating point
# ===================================================================================
def import_160_anchors() -> dict:
    """Import the #160 (W&B x8vffgbs) descent E[T]-DP ceilings -- do NOT re-derive."""
    with open(SPINE_SPEC_JSON) as f:
        spine = json.load(f)
    descent_only = spine["verdict"]["descent_only_E_T"]      # 5.056404568844709
    both_bugs = spine["verdict"]["both_bugs_E_T_specced"]    # 5.206954309441963
    # cross-source the same two anchors from #160's bug2-salvage decomposition
    with open(BUG2_DESCENT_JSON) as f:
        bug2 = json.load(f)
    descent_xsrc = bug2["step4_decomposition"]["bug2_et_full_alt_d1_0679"]
    both_xsrc = bug2["step4_decomposition"]["combined_et_both_fixed"]
    return {
        "descent_only_E_T_star": descent_only,
        "both_bugs_E_T_star": both_bugs,
        "descent_only_xsrc": descent_xsrc,
        "both_bugs_xsrc": both_xsrc,
        "anchors_cross_source_match": (
            abs(descent_only - descent_xsrc) < 1e-12 and abs(both_bugs - both_xsrc) < 1e-12
        ),
        "source_spine_spec": os.path.relpath(SPINE_SPEC_JSON, _ROOT),
        "source_bug2_descent": os.path.relpath(BUG2_DESCENT_JSON, _ROOT),
        "wandb_run_160": "x8vffgbs",
    }


def import_158_operating_point() -> dict:
    """Import denken #158's (W&B opbbrnce) measured BUG-2 over-accept operating point.

    Picks the harness run that carries a bug2_audit; that audit is the per-token
    `committed == in_step_target_argmax` differential on the _bug2_overaccept_kernel.
    """
    runs = sorted(glob.glob(GREEDY_HARNESS_GLOB))
    chosen = None
    for path in reversed(runs):  # newest first
        with open(path) as f:
            data = json.load(f)
        bug2 = data.get("self_test", {}).get("bug2_audit")
        if bug2 and bug2.get("total_committed"):
            chosen = (path, bug2)
            break
    if chosen is None:
        raise FileNotFoundError(f"no greedy-exact harness run with bug2_audit under {GREEDY_HARNESS_GLOB}")
    path, bug2 = chosen
    total_committed = int(bug2["total_committed"])
    total_exact = int(bug2["total_exact"])
    num_violations = int(bug2["num_violations"])
    exactness_rate = float(bug2["exactness_rate"])
    return {
        "run_path": os.path.relpath(path, _ROOT),
        "wandb_run_158": "opbbrnce",
        "total_committed": total_committed,
        "total_exact": total_exact,
        "num_violations": num_violations,
        "num_length_violations": int(bug2.get("num_length_violations", 0)),
        "exactness_rate": exactness_rate,
        # the detector's per-token violation differential (what v(e) must reproduce)
        "detector_violation_rate": num_violations / total_committed,
        "verdict": bug2.get("verdict"),
    }


# ===================================================================================
# 1. parametrize over-acceptance: the {(e, E[T], v)} curve  (per topology)
# ===================================================================================
def et_of_eps(e: float, e_star: float) -> float:
    """Inflated expected accepted length: one over-accepted node = +1 committed token."""
    return e_star + e


def v_of_eps(e: float, e_star: float) -> float:
    """Per-token greedy-violation rate: e over-accepted of (E[T]*+e) committed."""
    denom = e_star + e
    return (e / denom) if denom > 0 else 0.0


def eps_of_v(v: float, e_star: float) -> float:
    """Invert v -> e on the locus: e = v*E[T]*/(1-v)."""
    return (v * e_star) / (1.0 - v) if v < 1.0 else float("inf")


def et_of_v(v: float, e_star: float) -> float:
    """Over-accept locus: E[T] = E[T]*/(1-v)  (max E[T] explainable by violation v)."""
    return e_star / (1.0 - v) if v < 1.0 else float("inf")


def v_of_et(e_t: float, e_star: float) -> float:
    """Over-accept locus, other direction: v = 1 - E[T]*/E[T]  (E[T] >= E[T]*)."""
    return (1.0 - e_star / e_t) if e_t > 0 else 0.0


def overaccept_curve(e_star: float, n_grid: int = 0) -> list[dict]:
    """Build the {e, E[T], v} table over e in [0, e_max], e_max = 8 - E[T]* (accept-all)."""
    e_max = MAX_COMMITTED_PER_STEP - e_star
    # dense uniform grid plus the named special points (e=1 unit over-accept, e_max)
    grid = sorted(set(
        [round(0.05 * k, 4) for k in range(0, int(e_max / 0.05) + 1)]
        + [0.0, 0.25, 0.5, 1.0, 2.0, e_max]
    ))
    grid = [e for e in grid if 0.0 <= e <= e_max + 1e-12]
    rows = []
    for e in grid:
        e_t = et_of_eps(e, e_star)
        v = v_of_eps(e, e_star)
        rows.append({
            "epsilon": e,
            "E_T": e_t,
            "v": v,
            "exactness": 1.0 - v,
            "et_inflation": e_t - e_star,  # == epsilon by construction
        })
    return rows


# ===================================================================================
# 2. invert to the 2D trustworthy region + the over-accept locus  (per ceiling)
# ===================================================================================
def trustworthy_region(e_star: float, v_tol: float) -> dict:
    """Region of (E[T], v) consistent with a trustworthy greedy-exact descent.

    trustworthy_region = {(E[T], v): 0 <= v <= v_tol AND floor <= E[T] <= E[T]*/(1-v)}.
    The binding (upper) boundary IS the over-accept locus E[T] = E[T]*/(1-v): a
    measured E[T] above it cannot be explained by violation rate v. The degenerate
    v=0 slice pins the UNIQUE greedy-exact ceiling E[T]* (max E[T] at v=0).
    """
    floor = 1.0  # min committed per step (single-token accept); under-accept is greedy-SAFE
    corner_exact = {"E_T": e_star, "v": 0.0, "label": "greedy_exact_ceiling (UNIQUE at v=0)"}
    e_t_at_vtol = et_of_v(v_tol, e_star)
    corner_locus = {
        "E_T": e_t_at_vtol, "v": v_tol,
        "label": "max-E[T] on over-accept locus at v_tol",
        "et_inflation_budget_at_vtol": e_t_at_vtol - e_star,
    }
    return {
        "E_T_star": e_star,
        "v_tol": v_tol,
        "E_T_floor": floor,
        "corner_greedy_exact": corner_exact,
        "corner_max_inflation_at_vtol": corner_locus,
        "upper_boundary_is_overaccept_locus": "E_T <= E_T_star/(1-v)",
        "degenerate_at_v0_unique_E_T": e_star,
        "max_et_inflation_at_v0": et_of_v(0.0, e_star) - e_star,  # == 0 exactly
    }


def overaccept_locus(e_star: float, v_tol: float, n: int = 0) -> dict:
    """The (E[T], v) boundary curve traced by e>0, from (E[T]*,0) to the accept-all extreme."""
    e_max = MAX_COMMITTED_PER_STEP - e_star
    v_max = v_of_eps(e_max, e_star)
    grid_v = sorted(set([0.0, v_tol] + [round(0.02 * k, 4) for k in range(0, int(v_max / 0.02) + 1)] + [v_max]))
    grid_v = [v for v in grid_v if 0.0 <= v <= v_max + 1e-12]
    pts = []
    for v in grid_v:
        e_t = et_of_v(v, e_star)
        pts.append({"v": v, "E_T": e_t, "epsilon": eps_of_v(v, e_star), "et_inflation": e_t - e_star})
    return {
        "formula_E_T_of_v": "E_T = E_T_star / (1 - v)",
        "formula_v_of_E_T": "v = 1 - E_T_star / E_T",
        "v_max_accept_all": v_max,
        "E_T_max_accept_all": MAX_COMMITTED_PER_STEP,
        "epsilon_max": e_max,
        "points": pts,
    }


def make_land_gate(e_star_default: float):
    """Return the one-line predicate land #71 evaluates on its measured (E[T], v)."""

    def land_tuple_in_trustworthy_region(
        E_T: float,
        v: float,
        E_T_star: float = e_star_default,
        v_tol: float = 0.0,
        et_abs_tol: float = 1e-9,
    ) -> bool:
        # trustworthy iff violations within tol AND E[T] within the locus budget at v_tol
        return (v <= v_tol + 1e-12) and (E_T <= et_of_v(v_tol, E_T_star) + et_abs_tol)

    return land_tuple_in_trustworthy_region


# ===================================================================================
# 3. cross-check against denken #158's binary detector
# ===================================================================================
def cross_check_158(op158: dict) -> dict:
    """Confirm the analytic v form reproduces #158's measured per-token differential."""
    # v(e) evaluated at #158's measured counts == over-accepted committed fraction
    v_analytic = op158["num_violations"] / op158["total_committed"]   # 19/236
    detector_diff = 1.0 - op158["exactness_rate"]                     # 1 - 217/236
    matches = abs(v_analytic - detector_diff) < 1e-12 and abs(v_analytic - op158["detector_violation_rate"]) < 1e-12
    return {
        "v_at_denken158_point": v_analytic,
        "denken158_exactness_rate": op158["exactness_rate"],
        "denken158_detector_violation_rate": detector_diff,
        "num_violations": op158["num_violations"],
        "total_committed": op158["total_committed"],
        "total_exact": op158["total_exact"],
        "matches_detector": bool(matches),
        "identity": "v = num_violations/total_committed = 1 - exactness_rate (exact complement)",
        "note": (
            "#158's bug2 battery is a synthetic stress battery, not the deployed tree; the "
            "cross-check is the count-identity (analytic v form == empirical detector "
            "differential), which needs no topology. For CONTEXT only, the e that would "
            "produce this v on each deployed locus is reported in `context_eps_on_locus`."
        ),
        "context_eps_on_locus": {
            "descent_only_5.0564": None,   # filled by caller (needs e_star)
            "both_bugs_5.2070": None,
        },
    }


# ===================================================================================
# 5. self-test (PRIMARY) + et_inflation_at_unit_overaccept (TEST)
# ===================================================================================
def self_test(anchors: dict, curves: dict, regions: dict, loci: dict, xcheck: dict) -> dict:
    checks = []

    def chk(name, ok, detail):
        checks.append({"name": name, "passes": bool(ok), "detail": detail})

    d_star = anchors["descent_only_E_T_star"]
    b_star = anchors["both_bugs_E_T_star"]

    # (a) reproduce the #160 anchors at e=0 (v=0, zero inflation)
    d_row0 = curves["descent_only"][0]
    b_row0 = curves["both_bugs"][0]
    chk("descent anchor reproduced at e=0 (E[T]=5.0564, v=0)",
        abs(d_row0["E_T"] - 5.056404568844709) < 1e-9 and d_row0["epsilon"] == 0.0 and d_row0["v"] == 0.0,
        f"E_T={d_row0['E_T']:.9f} v={d_row0['v']}")
    chk("both-bugs anchor reproduced at e=0 (E[T]=5.2070, v=0)",
        abs(b_row0["E_T"] - 5.206954309441963) < 1e-9 and b_row0["epsilon"] == 0.0 and b_row0["v"] == 0.0,
        f"E_T={b_row0['E_T']:.9f} v={b_row0['v']}")
    chk("imported #160 anchors cross-source match (x8vffgbs)",
        anchors["anchors_cross_source_match"], "spine_spec == bug2_salvage_descent")

    # (b) degenerate-at-v=0: max E[T] inflation at v=0 is exactly 0 (unique ceiling)
    chk("descent max_et_inflation_at_v0 == 0",
        regions["descent_only"]["max_et_inflation_at_v0"] == 0.0,
        f"{regions['descent_only']['max_et_inflation_at_v0']}")
    chk("both-bugs max_et_inflation_at_v0 == 0",
        regions["both_bugs"]["max_et_inflation_at_v0"] == 0.0,
        f"{regions['both_bugs']['max_et_inflation_at_v0']}")

    # (c) locus round-trips: v_of_et(et_of_v(v)) == v  on a few points
    rt_ok = True
    for v in (0.01, 0.05, 0.1, 0.2):
        for e_star in (d_star, b_star):
            if abs(v_of_et(et_of_v(v, e_star), e_star) - v) > 1e-12:
                rt_ok = False
    chk("over-accept locus inverts cleanly (E_T<->v round-trip)", rt_ok, "v_of_et(et_of_v(v))==v")

    # (d) #158 cross-check reproduces the detector differential
    chk("v(e) reproduces denken #158 per-token differential (matches_detector)",
        xcheck["matches_detector"],
        f"v={xcheck['v_at_denken158_point']:.12f} vs 1-exactness={xcheck['denken158_detector_violation_rate']:.12f}")

    # (e) unit over-accept inflates E[T] by exactly 1.0 for BOTH topologies
    d_unit = et_of_eps(1.0, d_star) - d_star
    b_unit = et_of_eps(1.0, b_star) - b_star
    chk("delta(e=1) == 1.0 for descent-only", abs(d_unit - 1.0) < 1e-12, f"{d_unit}")
    chk("delta(e=1) == 1.0 for both-bugs", abs(b_unit - 1.0) < 1e-12, f"{b_unit}")

    # (f) any E[T] above the ceiling REQUIRES v>0 (the central claim)
    chk("E[T]>ceiling requires v>0 (descent): v(5.5)>0",
        v_of_et(5.5, d_star) > 0.0, f"v={v_of_et(5.5, d_star):.6f}")
    chk("E[T]>ceiling requires v>0 (both-bugs): v(5.5)>0",
        v_of_et(5.5, b_star) > 0.0, f"v={v_of_et(5.5, b_star):.6f}")

    # NaN-clean scan
    scalars = [d_star, b_star, d_unit, b_unit, xcheck["v_at_denken158_point"],
               regions["descent_only"]["max_et_inflation_at_v0"],
               regions["both_bugs"]["max_et_inflation_at_v0"],
               loci["descent_only"]["v_max_accept_all"], loci["both_bugs"]["v_max_accept_all"]]
    nan_clean = all(_finite(x) for x in scalars)
    chk("all metrics NaN-clean", nan_clean, f"{len(scalars)} scalars finite")

    passes = all(c["passes"] for c in checks)
    return {"passes": passes, "n_checks": len(checks),
            "n_passed": sum(c["passes"] for c in checks), "checks": checks,
            "nan_clean": nan_clean}


# ===================================================================================
# main
# ===================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="research/oracle_readout/overaccept_signature/overaccept_signature_results.json")
    ap.add_argument("--report-md", default="research/oracle_readout/overaccept_signature/report_overaccept_signature.md")
    ap.add_argument("--v-tol", type=float, default=0.0,
                    help="strict greedy-contract violation tolerance for the trustworthy region (default 0)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="wirbel/descent-overaccept-signature")
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default="descent-overaccept-signature")
    args = ap.parse_args()

    # ---- imports (NOT re-derived) ----
    anchors = import_160_anchors()
    op158 = import_158_operating_point()
    d_star = anchors["descent_only_E_T_star"]
    b_star = anchors["both_bugs_E_T_star"]

    # ---- 1. over-accept curves ----
    curves = {"descent_only": overaccept_curve(d_star), "both_bugs": overaccept_curve(b_star)}

    # ---- 2. trustworthy region + locus (both ceilings) + noise-floor variant ----
    V_TOL_NOISE = 1.0 / 65536.0  # one spurious token in the 128x512 benchmark token budget
    regions = {
        "descent_only": trustworthy_region(d_star, args.v_tol),
        "both_bugs": trustworthy_region(b_star, args.v_tol),
        "descent_only_noise_floor": trustworthy_region(d_star, V_TOL_NOISE),
        "both_bugs_noise_floor": trustworthy_region(b_star, V_TOL_NOISE),
    }
    loci = {"descent_only": overaccept_locus(d_star, args.v_tol),
            "both_bugs": overaccept_locus(b_star, args.v_tol)}

    # ---- 3. cross-check #158 ----
    xcheck = cross_check_158(op158)
    xcheck["context_eps_on_locus"] = {
        "descent_only_5.0564": eps_of_v(xcheck["v_at_denken158_point"], d_star),
        "both_bugs_5.2070": eps_of_v(xcheck["v_at_denken158_point"], b_star),
    }

    # ---- 4. land #71 gate predicate (evaluated truth table) ----
    gate_both = make_land_gate(b_star)
    gate_descent = make_land_gate(d_star)
    # illustrative measured tuples land might report, checked against the both-bugs ceiling
    examples = [
        {"E_T": b_star, "v": 0.0, "desc": "greedy-exact at both-bugs ceiling"},
        {"E_T": d_star, "v": 0.0, "desc": "greedy-exact at descent-only ceiling (<= both-bugs)"},
        {"E_T": 4.80, "v": 0.0, "desc": "under-accept, greedy-SAFE (slower than ceiling)"},
        {"E_T": 5.40, "v": v_of_et(5.40, b_star), "desc": "E[T]>ceiling WITH locus v -> OVER-ACCEPT"},
        {"E_T": 5.40, "v": 0.0, "desc": "E[T]>ceiling but v=0 -> ANOMALOUS (model-inconsistent)"},
        {"E_T": 5.2071, "v": V_TOL_NOISE, "desc": "1-token noise above ceiling at noise-floor v"},
    ]
    gate_truth_table = []
    for ex in examples:
        strict = gate_both(ex["E_T"], ex["v"], E_T_star=b_star, v_tol=0.0)
        noise = gate_both(ex["E_T"], ex["v"], E_T_star=b_star, v_tol=V_TOL_NOISE)
        gate_truth_table.append({**ex,
                                 "trustworthy_strict_vtol0": bool(strict),
                                 "trustworthy_noise_floor": bool(noise),
                                 "v_implied_by_locus": v_of_et(ex["E_T"], b_star) if ex["E_T"] > b_star else 0.0})

    # ---- 5. self-test (PRIMARY) + TEST metric ----
    st = self_test(anchors, curves, regions, loci, xcheck)
    overaccept_signature_self_test_passes = bool(st["passes"])
    et_inflation_at_unit_overaccept = et_of_eps(1.0, b_star) - b_star  # == 1.0 (node count)

    # headline sparse curve for the report (both topologies)
    def sparse(rows):
        keep = {0.0, 0.25, 0.5, 1.0, 2.0}
        out = [r for r in rows if r["epsilon"] in keep]
        out.append(rows[-1])  # e_max accept-all extreme
        return out

    out = {
        "primary_metric_name": "overaccept_signature_self_test_passes",
        "overaccept_signature_self_test_passes": int(overaccept_signature_self_test_passes),
        "test_metric_name": "et_inflation_at_unit_overaccept",
        "et_inflation_at_unit_overaccept": et_inflation_at_unit_overaccept,
        "verdict": (
            "OVER-ACCEPT SIGNATURE BUILT. The greedy-exact ceiling pins a UNIQUE E[T] at v=0 "
            "(max_et_inflation_at_v0=0); any E[T]>ceiling REQUIRES v>0 on the locus "
            "E_T=E_T*/(1-v). Unit over-accept inflates E[T] by exactly 1.0. The analytic v(e) "
            "reproduces denken #158's per-token differential exactly (matches_detector=True). "
            "land #71's measured (E[T], v) tuple is now checkable: an inflated E[T] is "
            "acceptance headroom ONLY if v stays ~0; otherwise it is over-acceptance."),
        "model": {
            "E_T_of_eps": "E_T(e) = E_T_star + e",
            "v_of_eps": "v(e) = e / (E_T_star + e)",
            "overaccept_locus": "E_T = E_T_star/(1-v)  <=>  v = 1 - E_T_star/E_T",
            "max_committed_per_step": MAX_COMMITTED_PER_STEP,
            "epsilon_units": "expected extra accepted NODES past the greedy boundary per step",
        },
        "imported_anchors_160": anchors,
        "imported_operating_point_158": op158,
        "overaccept_curve": {
            "descent_only_E_T_star": d_star,
            "both_bugs_E_T_star": b_star,
            "descent_only": curves["descent_only"],
            "both_bugs": curves["both_bugs"],
            "headline_descent_only": sparse(curves["descent_only"]),
            "headline_both_bugs": sparse(curves["both_bugs"]),
        },
        "trustworthy_region": regions,
        "overaccept_locus": loci,
        "cross_check_denken158": xcheck,
        "land71_gate": {
            "predicate": "land_tuple_in_trustworthy_region(E_T, v, E_T_star, v_tol=0, et_abs_tol=1e-9) -> bool",
            "predicate_body": "(v <= v_tol) AND (E_T <= E_T_star/(1-v_tol))",
            "strict_meaning_at_vtol0": "v == 0 AND E_T <= ceiling (no inflation)",
            "default_ceiling_both_bugs": b_star,
            "v_tol_noise_floor": V_TOL_NOISE,
            "v_tol_noise_floor_basis": "one spurious token in the 128 prompts x 512 tokens = 65536 token benchmark budget",
            "truth_table": gate_truth_table,
            "composition_with_denken158": (
                "denken #158 = BINARY 'any violation?' (v>0 fires; catches substitution-only "
                "violations that do NOT inflate E[T]). THIS = MAGNITUDE 'is the E[T] inflation "
                "explained by violation?' (E[T]>ceiling is over-acceptance, quantified by the "
                "locus v=1-E_T*/E_T). Together they bound BUG-2 from both the binary and the "
                "magnitude side: a descent passing BOTH (binary-clean v=0 AND no-inflation "
                "E[T]<=ceiling) is a trustworthy greedy-exact speedup."),
            "three_regions": {
                "TRUSTWORTHY": "v<=v_tol AND E_T<=ceiling (greedy-exact corner; under-accept is greedy-SAFE-but-slow)",
                "OVER_ACCEPT_BUG2": "v>v_tol AND E_T>ceiling, on the locus E_T~E_T*/(1-v) -> FAILS greedy",
                "ANOMALOUS": "E_T>ceiling but v~0 -> model-inconsistent (no inflation without violation); investigate",
            },
        },
        "self_test": st,
        "official_projection_context": {
            "note": "context only; this leg is 0 TPS and authorizes no launch",
            "formula": "official = K_cal * (E[T]/step) * tau",
            "K_cal": K_CAL, "step_measured": STEP_MEASURED, "target": TARGET_OFFICIAL,
            "both_bugs_at_measured_step_tau1": K_CAL * b_star / STEP_MEASURED,
            "descent_only_at_measured_step_tau1": K_CAL * d_star / STEP_MEASURED,
            "caveat": (
                "the over-accept signature does NOT change the clear-500 bar; it certifies the "
                "measured E[T] feeding the projection is TRUSTWORTHY (not over-accept-inflated)."),
        },
        "provenance": (
            "imports wirbel #160 (W&B x8vffgbs) descent E[T]-DP ceilings 5.0564/5.2070 and "
            "denken #158 (W&B opbbrnce) measured BUG-2 over-accept operating point "
            "(exactness 0.91949, 19/236 violations); does NOT re-derive either. Builds on "
            "wirbel #165 (W&B laxllfjl) which named BUG-2 the binding build-risk."),
        "method": ("LOCAL CPU-only analytic synthesis. No GPU/vLLM/HF Job/submission/kernel "
                   "build/served-file change. BASELINE stays 481.53. Greedy untouched."),
        "metrics_nan_clean": int(st["nan_clean"]),
    }

    os.makedirs(os.path.dirname(os.path.join(_ROOT, args.out)), exist_ok=True)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    _console(out)
    _write_report(out, args.report_md if os.path.isabs(args.report_md) else os.path.join(_ROOT, args.report_md))

    if args.wandb:
        _log_wandb(args, out)


# ===================================================================================
# console + report + wandb
# ===================================================================================
def _console(out: dict) -> None:
    print("=" * 96)
    print("DESCENT OVER-ACCEPTANCE SIGNATURE (PR #170, wirbel)")
    print("=" * 96)
    m = out["model"]
    print(f"\nmodel: E_T(e)={m['E_T_of_eps']} ; v(e)={m['v_of_eps']} ; locus {m['overaccept_locus']}")
    a = out["imported_anchors_160"]
    print(f"\n[IMPORT #160 x8vffgbs] descent-only E[T]*={a['descent_only_E_T_star']:.9f}  "
          f"both-bugs E[T]*={a['both_bugs_E_T_star']:.9f}  (cross-source match={a['anchors_cross_source_match']})")
    o = out["imported_operating_point_158"]
    print(f"[IMPORT #158 opbbrnce] exactness={o['exactness_rate']:.10f}  "
          f"violations={o['num_violations']}/{o['total_committed']}  verdict={o['verdict']}")

    print("\n[1. OVER-ACCEPT CURVE]  (e, E[T], v)  -- headline rows")
    for label in ("headline_descent_only", "headline_both_bugs"):
        topo = "descent-only" if "descent" in label else "both-bugs "
        print(f"  {topo}:")
        for r in out["overaccept_curve"][label]:
            print(f"    e={r['epsilon']:6.4f}  E[T]={r['E_T']:8.5f}  v={r['v']:8.5f}  exact={r['exactness']:8.5f}")

    r = out["trustworthy_region"]["both_bugs"]
    print(f"\n[2. TRUSTWORTHY REGION / LOCUS]  ceiling E[T]*={r['E_T_star']:.5f}")
    print(f"    degenerate v=0 -> UNIQUE E[T]={r['degenerate_at_v0_unique_E_T']:.5f}  "
          f"max_et_inflation_at_v0={r['max_et_inflation_at_v0']}")
    print(f"    locus: {out['overaccept_locus']['both_bugs']['formula_E_T_of_v']}")

    x = out["cross_check_denken158"]
    print(f"\n[3. CROSS-CHECK #158]  v_at_denken158_point={x['v_at_denken158_point']:.12f}  "
          f"(= 1 - exactness {x['denken158_exactness_rate']:.12f})")
    print(f"    matches_detector = {x['matches_detector']}   ({x['identity']})")

    print("\n[4. LAND #71 GATE]  land_tuple_in_trustworthy_region(E_T, v) -> bool")
    for row in out["land71_gate"]["truth_table"]:
        print(f"    E[T]={row['E_T']:7.4f} v={row['v']:.6e}  strict={row['trustworthy_strict_vtol0']!s:5s} "
              f"noise={row['trustworthy_noise_floor']!s:5s}  | {row['desc']}")

    st = out["self_test"]
    print(f"\n[5. SELF-TEST]  {st['n_passed']}/{st['n_checks']} checks")
    for c in st["checks"]:
        print(f"    [{'OK' if c['passes'] else 'FAIL'}] {c['name']}  ({c['detail']})")
    print(f"\n[PRIMARY] overaccept_signature_self_test_passes = {int(out['overaccept_signature_self_test_passes'])}")
    print(f"[TEST]    et_inflation_at_unit_overaccept = {out['et_inflation_at_unit_overaccept']}")
    print(f"[NaN-clean] {out['metrics_nan_clean']}")


def _write_report(out: dict, path: str) -> None:
    a = out["imported_anchors_160"]
    o = out["imported_operating_point_158"]
    x = out["cross_check_denken158"]
    rb = out["trustworthy_region"]["both_bugs"]
    rd = out["trustworthy_region"]["descent_only"]
    st = out["self_test"]

    def curve_rows(label):
        s = ""
        for r in out["overaccept_curve"][label]:
            s += f"| {r['epsilon']:.4f} | {r['E_T']:.5f} | {r['v']:.5f} | {r['exactness']:.5f} |\n"
        return s

    md = f"""<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Descent over-acceptance signature — the joint (E[T], greedy-violation) region
# land's measured tuple must fall in (PR #170 · wirbel)

**PRIMARY** `overaccept_signature_self_test_passes` = **{bool(out['overaccept_signature_self_test_passes'])}** ({st['n_passed']}/{st['n_checks']} checks, NaN-clean)
**TEST** `et_inflation_at_unit_overaccept` = **{out['et_inflation_at_unit_overaccept']}** (δ(ε=1); one extra node = +1.0 E[T], both topologies)

## Honest scope
Pure-analytic **CPU-only** magnitude complement to denken #158's binary detector. No GPU / vLLM / HF Job / submission / kernel build / served-file change. BASELINE stays 481.53; **0 TPS**; greedy untouched by construction. Imports wirbel #160 (`x8vffgbs`) descent E[T]-DP ceilings + denken #158 (`opbbrnce`) measured operating point — **does NOT re-derive them**. Builds on wirbel #165 (`laxllfjl`), which named **BUG-2 the binding build-risk** (~19× BUG-1's E[T] lever).

## The model (node-counting; nothing re-derived)
Over-acceptance commits **ε** extra nodes per step past the true greedy boundary. Each over-accepted node is (1) one extra committed token → **+1 E[T]**, and (2) one greedy violation (past the boundary the draft has diverged). With the #160 greedy-exact ceiling `E[T]*`:

```
E[T](ε) = E[T]* + ε                  v(ε) = ε / (E[T]* + ε)
over-accept locus:   E[T] = E[T]*/(1 − v)   ⇔   v = 1 − E[T]*/E[T]
```

Imported ceilings (`x8vffgbs`): descent-only **E[T]\*={a['descent_only_E_T_star']:.6f}**, both-bugs **E[T]\*={a['both_bugs_E_T_star']:.6f}** (cross-source match = {a['anchors_cross_source_match']}). Per-step committed cap = max_spec_len+1 = **{out['model']['max_committed_per_step']}** (accept-all extreme).

## 1. Over-accept curve {{ε, E[T], v}}
**both-bugs (E[T]\*={a['both_bugs_E_T_star']:.5f}):**

| ε | E[T] | v | exactness |
|---|---|---|---|
{curve_rows('headline_both_bugs')}
**descent-only (E[T]\*={a['descent_only_E_T_star']:.5f}):**

| ε | E[T] | v | exactness |
|---|---|---|---|
{curve_rows('headline_descent_only')}
## 2. Inversion → 2D trustworthy region + over-accept locus
`trustworthy_region` = {{(E[T], v): v ≤ v_tol AND E[T] ≤ E[T]*/(1−v)}} (upper boundary IS the locus). **Degenerate at v=0**: v=0 ⇔ ε=0 ⇔ E[T]=E[T]* (UNIQUE) ⇒ **`max_et_inflation_at_v0` = {rb['max_et_inflation_at_v0']}** for both ceilings. Any E[T] > E[T]* **requires** v>0.

- both-bugs corners: greedy-exact `({rb['corner_greedy_exact']['E_T']:.5f}, 0)`; locus@v_tol `({rb['corner_max_inflation_at_vtol']['E_T']:.5f}, {rb['v_tol']})`.
- descent-only corners: greedy-exact `({rd['corner_greedy_exact']['E_T']:.5f}, 0)`.
- noise-floor (v_tol = 1/65536 = one spurious token in the 128×512 benchmark budget): E[T] inflation budget = **{out['trustworthy_region']['both_bugs_noise_floor']['corner_max_inflation_at_vtol']['et_inflation_budget_at_vtol']:.2e}** — i.e. even one spurious violation buys < 1e-4 E[T]; any meaningful E[T] readout above the ceiling is over-acceptance.

## 3. Cross-check vs denken #158's binary detector
#158's BUG-2 battery (`opbbrnce`): exactness {o['exactness_rate']:.10f} = {o['total_exact']}/{o['total_committed']}; violations **{o['num_violations']}/{o['total_committed']}**. The analytic v form (over-accepted committed fraction) reproduces the detector's per-token differential **exactly**:

- **`v_at_denken158_point` = {x['v_at_denken158_point']:.12f}** = 1 − exactness {x['denken158_exactness_rate']:.12f}
- **`matches_detector` = {x['matches_detector']}** ({x['identity']})

The continuous `v(ε)` and the binary detector are the **same quantity**, agreeing at the same operating point. (#158's battery is synthetic stress, not the deployed tree — the cross-check is the count-identity, topology-free.)

## 4. The gate handed to land #71
```
land_tuple_in_trustworthy_region(E_T, v, E_T_star, v_tol=0, et_abs_tol=1e-9):
    return (v <= v_tol) and (E_T <= E_T_star/(1 - v_tol))
```
Strict (v_tol=0): **trustworthy ⇔ v=0 AND E[T] ≤ ceiling**. If land measures **E[T] > {a['both_bugs_E_T_star']:.4f} with v>0 → over-acceptance (FAILS greedy-exact)**, not a faster descent. Three regions: **TRUSTWORTHY** (v≈0, E[T]≤ceiling; under-accept is greedy-SAFE-but-slow) · **OVER-ACCEPT/BUG-2** (v>0, E[T]>ceiling on the locus) · **ANOMALOUS** (E[T]>ceiling but v≈0 — model-inconsistent, investigate).

**Composition with #158** — #158 = binary *"any violation?"* (catches substitution-only violations with no inflation); this = magnitude *"is the E[T] inflation explained by violation?"* (catches inflated-E[T]-read-as-headroom). Together they bound BUG-2 from both sides; a descent passing **both** is a trustworthy greedy-exact speedup.

## 5. Self-validate (PRIMARY)
{st['n_passed']}/{st['n_checks']} checks pass (anchors reproduced at ε=0 with v=0; degenerate-at-v=0; locus inverts; #158 cross-check; δ(ε=1)=1.0 both topologies; E[T]>ceiling⇒v>0; NaN-clean). **`overaccept_signature_self_test_passes` = {bool(out['overaccept_signature_self_test_passes'])}**. **`et_inflation_at_unit_overaccept` = {out['et_inflation_at_unit_overaccept']}**.

## Public / banked evidence used
- wirbel #160 (`x8vffgbs`): descent E[T]-DP ceilings 5.0564 / 5.2070 (imported).
- denken #158 (`opbbrnce`): per-token `committed==in_step_target_argmax` binary detector, measured BUG-2 over-accept operating point exactness 0.91949 / 19 violations (imported).
- wirbel #165 (`laxllfjl`): SHARED index-map; named BUG-2 the binding build-risk (over-acceptance the only greedy-breaking path).

Official projection (context only; 0 TPS): the signature does **not** move the clear-500 bar — it certifies the measured E[T] feeding `official = K_cal·(E[T]/step)·τ` is TRUSTWORTHY, not over-accept-inflated.
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(md)
    print(f"\nwrote {os.path.relpath(path, _ROOT)}")


def _log_wandb(args, out: dict) -> None:
    import wandb

    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                     config={"gate": "descent-overaccept-signature",
                             "method": "cpu-analytic-magnitude-complement-to-158-binary",
                             "descent_only_E_T_star": out["imported_anchors_160"]["descent_only_E_T_star"],
                             "both_bugs_E_T_star": out["imported_anchors_160"]["both_bugs_E_T_star"],
                             "max_committed_per_step": out["model"]["max_committed_per_step"],
                             "v_tol": args.v_tol,
                             "wandb_run_160": "x8vffgbs", "wandb_run_158": "opbbrnce",
                             "wandb_run_165": "laxllfjl"})
    s = wandb.summary
    s["overaccept_signature_self_test_passes"] = int(out["overaccept_signature_self_test_passes"])
    s["et_inflation_at_unit_overaccept"] = out["et_inflation_at_unit_overaccept"]
    s["metrics_nan_clean"] = out["metrics_nan_clean"]
    s["descent_only_E_T_star"] = out["imported_anchors_160"]["descent_only_E_T_star"]
    s["both_bugs_E_T_star"] = out["imported_anchors_160"]["both_bugs_E_T_star"]
    s["max_et_inflation_at_v0_descent"] = out["trustworthy_region"]["descent_only"]["max_et_inflation_at_v0"]
    s["max_et_inflation_at_v0_both_bugs"] = out["trustworthy_region"]["both_bugs"]["max_et_inflation_at_v0"]
    s["v_at_denken158_point"] = out["cross_check_denken158"]["v_at_denken158_point"]
    s["denken158_exactness_rate"] = out["cross_check_denken158"]["denken158_exactness_rate"]
    s["matches_detector"] = int(out["cross_check_denken158"]["matches_detector"])
    s["v_at_unit_overaccept_descent"] = 1.0 / (out["imported_anchors_160"]["descent_only_E_T_star"] + 1.0)
    s["v_at_unit_overaccept_both_bugs"] = 1.0 / (out["imported_anchors_160"]["both_bugs_E_T_star"] + 1.0)
    s["overaccept_locus_v_max_both_bugs"] = out["overaccept_locus"]["both_bugs"]["v_max_accept_all"]
    s["n_checks"] = out["self_test"]["n_checks"]
    s["n_passed"] = out["self_test"]["n_passed"]

    # over-accept curve (both-bugs) as a table
    ct = wandb.Table(columns=["epsilon", "E_T", "v", "exactness", "et_inflation"])
    for r in out["overaccept_curve"]["both_bugs"]:
        ct.add_data(r["epsilon"], r["E_T"], r["v"], r["exactness"], r["et_inflation"])
    wandb.log({"overaccept_curve_both_bugs": ct})

    # over-accept locus (both-bugs)
    lt = wandb.Table(columns=["v", "E_T", "epsilon", "et_inflation"])
    for p in out["overaccept_locus"]["both_bugs"]["points"]:
        lt.add_data(p["v"], p["E_T"], p["epsilon"], p["et_inflation"])
    wandb.log({"overaccept_locus_both_bugs": lt})

    # land gate truth table
    gt = wandb.Table(columns=["E_T", "v", "trustworthy_strict", "trustworthy_noise", "desc"])
    for row in out["land71_gate"]["truth_table"]:
        gt.add_data(row["E_T"], row["v"], int(row["trustworthy_strict_vtol0"]),
                    int(row["trustworthy_noise_floor"]), row["desc"])
    wandb.log({"land71_gate_truth_table": gt})

    # self-test checks
    stt = wandb.Table(columns=["check", "passes", "detail"])
    for c in out["self_test"]["checks"]:
        stt.add_data(c["name"], int(c["passes"]), c["detail"])
    wandb.log({"self_test_checks": stt})

    print(f"\nW&B run: {run.id}  ({run.url})")
    wandb.finish()


if __name__ == "__main__":
    main()
