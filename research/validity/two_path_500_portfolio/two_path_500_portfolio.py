#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #253 (fern) -- Two-path-to-500 portfolio card: combined P(reach 500) over land's tree build (A)
+ the 5-lever linear speed portfolio (B).

WHAT THIS IS
------------
The fleet now has TWO INDEPENDENT routes to the 500 TPS milestone, and NO single banked artifact
prices their COMBINED reach. This card builds that one number under explicitly-stated PRIORS.

  PATH-A (the build route).  land #245/#71's high-risk tree-decode build -- the long pole every
      external fleet stalled on. Clears 500 iff the live build's MEASURED E[T]_both >= 4.3305
      (denken #241 operative floor) AND MEASURED min-lambda_hat q[2..9] >= 0.9780 (fern #249 / stark
      #191 operative gate), conditioned on land's projection E[T]_both=4.512 / min-lambda 0.983 being
      realized. P_A is a PRIOR: treeverify_served_gain_MEASURED_realized=0.0 (fern #238) -- there is
      NO live tree win yet. The adverse anchor is the external-fleet stall rate on this exact build.

  PATH-B (the linear-lever portfolio).  5 students are pricing INDEPENDENT linear speed levers on the
      EXISTING 481.53 stack that need NO tree build: stark #247 (OPT-Tree E[T]), lawine #246
      (FlashInfer+CUDAGraph), kanna #248 (int3 draft), ubel #250 (n-gram draft), wirbel #251
      (activation-recycle). 481.53 -> 500 is +18.47 TPS = +3.8357%. reach500_B = 1[stacked linear
      gain >= 3.8357%]. P_B is a PRIOR over the screens' PROJECTED-gain ranges (from the speed-levers
      researcher doc): NONE of the 5 screens has returned a GO verdict yet.

THE DELIVERABLE
---------------
  combined_reach500_prob = 1 - (1 - P_A) * (1 - P_B)     (the two routes priced as INDEPENDENT)

plus a route x (gate, status, reach-prob, what-collapses-the-prior) table, and a one-line readiness
statement. HONEST FRAME: both P_A and P_B are PRIORS, not facts. The card does NOT claim 500 -- it
STRUCTURES the two-route decision and names exactly which MEASUREMENT collapses each prior:
  * Path-A: land #245's ONE build run -> a MEASURED end-to-end >=500 tree-decode artifact.
  * Path-B: each screen's GO/NO-GO verdict + a kernel build clearing PPL-valid wall_tps >= 500.
Until the FIRST measured PPL-valid wall_tps >= 500 from EITHER route, fern #238's
readiness_verdict=NOT-READY is carried forward UNCHANGED (the card cannot read a prior as a delivered
win -- the same discipline as #238's treeverify_realized==0 gate).

The COMPOSITION is LINEAR: official = K_cal*(E[T]/step)*tau (kanna #217, K_cal=125.268, step=1.2182),
so a fractional gain g in (E[T]/step) is exactly a fractional gain g in official TPS -- the +3.8357%
Path-B threshold round-trips 481.53 -> 500 with no slope dependence.

LOCAL, CPU-ONLY integration over EXISTING MERGED legs (#238/#241/#244/#249) imported VERBATIM (loaded
from their committed result JSON and round-tripped scalar-for-scalar) + the 5 in-flight Path-B screens
priced as a PORTFOLIO PRIOR over the researcher doc's projected-gain ranges (NOT their unreturned
measured results). Nothing is re-derived; the only new object is the COMBINED reach. No GPU / vLLM /
HF Job / submission / served-file change / official draw. BASELINE stays 481.53; adds 0 TPS;
greedy/PPL untouched; authorizes NOTHING. NOT a launch. NOT open2. Bank-the-analysis (primary=self-test).

PRIMARY metric  two_path_portfolio_self_test_passes
TEST    metric  combined_reach500_prob

Run:
  cd target/ && CUDA_VISIBLE_DEVICES="" python \
    research/validity/two_path_500_portfolio/two_path_500_portfolio.py \
    --self-test --wandb_group launch-readiness-integration --wandb_name fern/two-path-500-portfolio
"""
from __future__ import annotations

import argparse
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
# Banked legs imported VERBATIM (loaded from their committed result JSON, NOT re-derived). Each
# headline scalar is pulled from the committed artifact and round-tripped against the hardcoded
# expected value -- the same provenance discipline as fern #238's launch_decision_card.
# --------------------------------------------------------------------------- #
JSON_238 = "research/validity/launch_decision_card/launch_decision_card_results.json"
JSON_241 = "research/validity/measured_et_shortfall/measured_et_shortfall_results.json"
JSON_244 = "research/validity/ceiling_gap_topology_headroom/results.json"
JSON_249 = "research/validity/build_lambda_bar/build_lambda_bar_results.json"

# ---- the two Path-A MEASURED gates (the ONLY two hard gates land #245's build must clear) ----
ET_FLOOR_OPERATIVE = 4.330527243789328       # denken #241 e_t_meas_floor (TPS500 binds, independent lambda)
ET_FLOOR_SELF_INSURED = 4.488950392227178    # denken #241 e_t_meas_floor_conservative (uniform-adverse coupling)
LAMBDA_GATE = 0.9780112973731208             # fern #249 operative gate == stark #191 P95 == #241 lambda_star_191

# ---- land #245/#71 projection (the PRIOR central estimate -- UNMEASURED) ----
PROJ_ET_BOTH = 4.512                         # land #71 projection E[T]_both (+18.3%)
PROJ_MIN_LAMBDA = 0.983                      # land #71 projection min-lambda q[2..9]
DELTA_MAX_TPS500 = 0.040220025755911104      # #241: measured E[T] may fall 4.022% below proj and still clear 500
LAMBDA_HEADROOM_PROJ = 0.005021667650860651  # #241: 0.983 - 0.9780 acceptance-gate margin at the projection
LAMBDA1_CEILING = 520.9527323111674          # land GO read = lambda=1 ceiling (the 4.512 projection -> 520.95)

# ---- the LINEAR official composition (kanna #217 via wirbel #199) ----
K_CAL = 125.26795005202914
STEP = 1.2182
TAU = 1.0
F_LINEAR8 = 3.8444537125748504               # linear-8 (E[T]/step) basis (#244 composition)
BASELINE_TPS = 481.53
TARGET_TPS = 500.0

TOL_PROVENANCE = 1e-9                         # committed-JSON scalars must match hardcoded expected exactly
TOL_COMPOSITION = 1e-9                        # the composition-linearity round-trip tolerance

# --------------------------------------------------------------------------- #
# PATH-A reach prior (stated explicitly; bracketed adverse / central / optimistic).
#   P_A = p_build_lands * p_ET_clears_given_build * p_lambda_clears_given_build
# The two gate-clears are MEASURED INDEPENDENTLY (denken #241 operative framing: lambda from q[2..9],
# deep-tail-protected, does NOT move with the E[T]-shortfall delta), so they multiply.
#   * p_build_lands     -- probability land's tree-decode build produces a MEASURED end-to-end
#                          artifact AT ALL (vs stalling like every external fleet -- the long pole).
#                          The ADVERSE anchor (external-fleet stall) lives here.
#   * p_ET_clears        -- P(measured E[T] >= 4.3305 | build). The projection 4.512 carries a 4.02%
#                          shortfall tolerance (denken #241), so the central case clears with margin.
#   * p_lambda_clears    -- P(measured lambda_hat >= 0.9780 | build). The projection 0.983 carries a
#                          THIN 0.005 margin, so this is the tighter of the two gate-clears.
# --------------------------------------------------------------------------- #
PATH_A_PRIORS = {
    "adverse":    {"p_build_lands": 0.30, "p_ET_clears": 0.65, "p_lambda_clears": 0.55},
    "central":    {"p_build_lands": 0.50, "p_ET_clears": 0.80, "p_lambda_clears": 0.70},
    "optimistic": {"p_build_lands": 0.70, "p_ET_clears": 0.90, "p_lambda_clears": 0.85},
}

# --------------------------------------------------------------------------- #
# PATH-B portfolio prior -- the 5 in-flight linear speed-lever screens.
# Each lever: GO-probability p_go (does the screen return a GO verdict clearing its stop condition)
# and a PROJECTED fractional-TPS gain range [lo, hi] (from the speed-levers researcher doc). Levers
# are grouped by the STEP COMPONENT they target; within a group they do NOT stack (you cannot remove
# the same time twice -- the GPU-bound step / the draft step), across groups they compose ~additively
# on the linear composition (E[T]-axis x step-axis), with an Amdahl efficiency among step-lowering
# groups. ALL priors are PESSIMISTIC-leaning because the 481.53 stack is already well-tuned and fern's
# own banked wirbel #244 certifies the verify-tree TOPOLOGY is near-exhausted (topology_lift_max=0).
#
#   group  E[T]   -> raises the E[T] numerator (topology). {stark #247 OPT-Tree}
#   group  SYS    -> lowers the GPU-bound step (kernel/launch/bandwidth). {lawine #246, wirbel #251}
#   group  DRAFT  -> lowers/removes the draft step. {kanna #248 int3, ubel #250 n-gram} (ALTERNATIVES)
# --------------------------------------------------------------------------- #
#
# REVEALED-DIFFICULTY PRIOR: the served stack has PLATEAUED at 481.53 after extensive tuning. If any
# single one of these levers were an easy, deployable, greedy-identical >=+3.84% win, it would already
# be banked -- so the stack's plateau is Bayesian evidence AGAINST easy linear gains, and every p_go is
# set well below the literature's nominal success rate. A "GO" here is the CONJUNCTION of {deployable
# on this A10G single-stream stack, greedy-IDENTICAL, PPL-valid, clears the screen's own stop condition}
# -- a hard conjunction, hence p_go in [0.20, 0.40]. The gain ranges sit at the LOW end of the
# literature (measured on A100/H100, Llama/Qwen) because the single-stream small-KV 128/128 regime is
# the doc's own "may be smaller / must profile first" case.
PATH_B_LEVERS = [
    {
        "key": "stark_247_opt_tree",
        "screen_pr": 247,
        "name": "OPT-Tree per-step adaptive topology",
        "doc_rank": "T-1 (Rank-1)",
        "group": "ET",
        "p_go": 0.28,                 # zero greedy-ID risk BUT wirbel #244 shows topology near-exhausted
        "gain_lo": 0.005, "gain_hi": 0.035,
        "greedy_id_risk": "zero",
        "rationale": ("doc T-1 headline +3-10%, but fern's banked #244 certifies the deployed depth-9 "
                      "mb-3 tree is already rho-optimal (topology_lift_max=+0.0 TPS) -- so OPT-Tree's "
                      "per-step adaptation headroom is priored DOWN hard (p_go 0.28, cap +0.5-3.5%)."),
    },
    {
        "key": "lawine_246_flashinfer_cudagraph",
        "screen_pr": 246,
        "name": "FlashInfer decode + CUDAGraph",
        "doc_rank": "K-1 (Rank-2)",
        "group": "SYS",
        "p_go": 0.40,                 # strongest lever, zero greedy-ID risk; but CUDAGraph shape
        "gain_lo": 0.02, "gain_hi": 0.08,   # conflicts, 'stop if <1%', + a competent extant backend
        "greedy_id_risk": "zero",
        "rationale": ("doc K-1 +7-18% (bit-identical fused attention + launch-overhead elimination); "
                      "the single strongest lever, but the served stack likely already uses a competent "
                      "backend, so the realizable marginal gain is priored to +2-8% with p_go 0.40."),
    },
    {
        "key": "kanna_248_int3_draft",
        "screen_pr": 248,
        "name": "int3 draft quantization (QSpec)",
        "doc_rank": "Q-1 (Rank-8)",
        "group": "DRAFT",
        "p_go": 0.30,
        "gain_lo": 0.01, "gain_hi": 0.06,
        "greedy_id_risk": "zero",     # draft quantization cannot move verify / acceptance
        "rationale": ("doc Q-1 +5-10%, but in single-stream the draft runs once per tree step and is "
                      "likely a small fraction of step time ('if draft is bandwidth-bound') -> +1-6%."),
    },
    {
        "key": "ubel_250_ngram_draft",
        "screen_pr": 250,
        "name": "n-gram / REST datastore draft",
        "doc_rank": "N-1 (Rank-13)",
        "group": "DRAFT",
        "p_go": 0.20,                 # task-dependent: match rate on diverse benchmark prompts uncertain
        "gain_lo": 0.00, "gain_hi": 0.05,
        "greedy_id_risk": "zero",
        "rationale": ("doc N-1 task-dependent: E[T]<2 if prompts are diverse NL (NO-GO), E[T]>4 if "
                      "structured. High chance of low match rate on general prompts -> p_go 0.20, "
                      "wide range with a 0% floor."),
    },
    {
        "key": "wirbel_251_activation_recycle",
        "screen_pr": 251,
        "name": "activation-recycle (HBM bandwidth)",
        "doc_rank": "Rank-4 (HBM-floor-capped)",
        "group": "SYS",
        "p_go": 0.25,                 # HBM-floor cap limits whether it clears
        "gain_lo": 0.01, "gain_hi": 0.05,
        "greedy_id_risk": "zero",
        "rationale": ("PR frame +4-8% but HBM-floor-capped; targets the same GPU-bound step as "
                      "FlashInfer (NON-stacking with lawine #246) -> priored to +1-5% with p_go 0.25."),
    },
]

PATH_B_LEVERS_BY_KEY = {lv["key"]: lv for lv in PATH_B_LEVERS}

# within-group overlap retention: how much of a SECOND same-component lever's marginal gain survives.
OVERLAP_RETAIN = {
    "ET": 0.0,        # single lever in the group
    "SYS": 0.25,      # FlashInfer + activation-recycle are complementary but mostly redundant on step
    "DRAFT": 0.0,     # int3-draft vs n-gram are MUTUALLY EXCLUSIVE draft sources (deploy one, not both)
}
ETA_STEP = 0.85       # Amdahl efficiency across the two STEP-lowering groups (SYS + DRAFT)

PATH_B_THRESHOLD = TARGET_TPS / BASELINE_TPS - 1.0   # +0.0383569 : 481.53 -> 500 on the linear composition
MC_N = 400_000
MC_SEED = 20260614
MC_SEEDS_STABILITY = (20260614, 7, 99991)            # stability cross-check seeds


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
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


def official_tps(et_over_step_ratio: float) -> float:
    """The LINEAR official composition official = K_cal*(E[T]/step)*tau."""
    return K_CAL * (et_over_step_ratio / STEP) * TAU


# --------------------------------------------------------------------------- #
# Step 0/1 -- import the four banked legs (load committed JSON, round-trip scalars).
# --------------------------------------------------------------------------- #
def import_banked() -> dict[str, Any]:
    """Load #238/#241/#244/#249 committed artifacts and round-trip their headline scalars vs the
    hardcoded expected values used by this card (provenance check, < 1e-9)."""
    j238 = _load_json(JSON_238)
    j241 = _load_json(JSON_241)
    j244 = _load_json(JSON_244)
    j249 = _load_json(JSON_249)

    s241 = j241["synthesis"]
    land = j238["land71_projection"]
    h244 = j244["synthesis"]["headline"]

    loaded = {
        # ---- denken #241: the E[T] floors + projection + shortfall tolerance + composition ----
        "241.e_t_meas_floor": s241["binding_tolerance"]["e_t_meas_floor"],
        "241.e_t_meas_floor_conservative": s241["binding_tolerance"]["e_t_meas_floor_conservative"],
        "241.delta_max_tps500": s241["binding_tolerance"]["delta_max_tps500"],
        "241.min_lambda_proj": s241["lambda_gate"]["min_lambda_proj"],
        "241.lambda_headroom_abs": s241["lambda_gate"]["headroom_abs"],
        "241.lambda1_ceiling": s241["imports"]["lambda1_ceiling"],
        "241.K_cal": s241["imports"]["K_cal"],
        "241.step_banked": s241["imports"]["step_banked"],
        "241.proj_e_t_both": s241["shortfall_table"]["rows"][0]["e_t_meas"],
        # ---- fern #249: the operative build-lambda gate ----
        "249.build_lambda_operative_gate": j249["build_lambda_operative_gate"],
        # ---- fern #238: the readiness truth + measured-gain gate + projection ----
        "238.readiness_verdict": j238["readiness_verdict"],
        "238.n_green_gates": j238["n_green_gates"],
        "238.treeverify_served_gain_MEASURED_realized": land["treeverify_served_gain_MEASURED_realized"],
        "238.projection_E_T_both": land["projection_E_T_both"],
        "238.projection_min_lambda_q2q9": land["projection_min_lambda_q2q9"],
        # ---- wirbel #244: the compliant-PRIVATE-500 lane topology verdict (context) ----
        "244.lane_reopenable": h244["lane_reopenable"],
        "244.reopener_is_coverage_not_topology": h244["reopener_is_coverage_not_topology"],
        "244.topology_lift_max_tps": h244["topology_lift_max_tps"],
        "244.operative_ceiling_tps": h244["operative_ceiling_tps"],
    }
    expected = {
        "241.e_t_meas_floor": ET_FLOOR_OPERATIVE,
        "241.e_t_meas_floor_conservative": ET_FLOOR_SELF_INSURED,
        "241.delta_max_tps500": DELTA_MAX_TPS500,
        "241.min_lambda_proj": PROJ_MIN_LAMBDA,
        "241.lambda_headroom_abs": LAMBDA_HEADROOM_PROJ,
        "241.lambda1_ceiling": LAMBDA1_CEILING,
        "241.K_cal": K_CAL,
        "241.step_banked": STEP,
        "241.proj_e_t_both": PROJ_ET_BOTH,
        "249.build_lambda_operative_gate": LAMBDA_GATE,
        "238.readiness_verdict": "NOT-READY",
        "238.n_green_gates": 2,
        "238.treeverify_served_gain_MEASURED_realized": 0.0,
        "238.projection_E_T_both": PROJ_ET_BOTH,
        "238.projection_min_lambda_q2q9": PROJ_MIN_LAMBDA,
        "244.lane_reopenable": False,
        "244.reopener_is_coverage_not_topology": True,
        "244.topology_lift_max_tps": 0.0,
        "244.operative_ceiling_tps": LAMBDA1_CEILING,
    }
    roundtrip: dict[str, Any] = {}
    for k, exp in expected.items():
        got = loaded[k]
        if isinstance(exp, bool) or isinstance(exp, str):
            ok = (got == exp)
            err = 0.0 if ok else float("nan")
        else:
            err = abs(float(got) - float(exp))
            ok = err <= TOL_PROVENANCE
        roundtrip[k] = {"loaded": got, "expected": exp, "abs_err": err, "round_trips": bool(ok)}

    return {
        "roundtrip": roundtrip,
        "readiness_verdict_carried": j238["readiness_verdict"],
        "n_green_gates_238": j238["n_green_gates"],
        "treeverify_realized": land["treeverify_served_gain_MEASURED_realized"],
    }


# --------------------------------------------------------------------------- #
# Step 1 -- PATH-A reach (the build route).
# --------------------------------------------------------------------------- #
def path_a_reach(banked: dict[str, Any]) -> dict[str, Any]:
    def compose(p: dict[str, float]) -> float:
        return p["p_build_lands"] * p["p_ET_clears"] * p["p_lambda_clears"]

    brackets = {k: compose(v) for k, v in PATH_A_PRIORS.items()}
    p_a = brackets["central"]

    # reach500_A_is_measured: is there a LIVE tree win yet? (fern #238 treeverify_realized > 0)
    reach500_A_is_measured = bool(banked["treeverify_realized"] > 0.0)

    prior_statement = (
        "P_A = p_build_lands * p_ET_clears * p_lambda_clears, with land #245/#71's projection "
        "(E[T]_both=%.3f, min-lambda %.3f) as the CENTRAL estimate and the external-fleet stall rate "
        "on the tree-decode build (the long pole) as the ADVERSE anchor. CENTRAL priors: "
        "p_build_lands=%.2f (a genuine coin-flip -- every external fleet stalled on this build), "
        "p_ET_clears=%.2f (the 4.02%% shortfall tolerance from denken #241 gives the E[T] gate room), "
        "p_lambda_clears=%.2f (the THIN 0.005 lambda margin makes this the tighter gate). The two "
        "gate-clears are MEASURED INDEPENDENTLY (denken #241 operative framing) so they multiply."
        % (PROJ_ET_BOTH, PROJ_MIN_LAMBDA,
           PATH_A_PRIORS["central"]["p_build_lands"], PATH_A_PRIORS["central"]["p_ET_clears"],
           PATH_A_PRIORS["central"]["p_lambda_clears"]))

    collapsing_measurement = (
        "land #245's ONE build run delivering a MEASURED end-to-end >=500 tree-decode artifact: "
        "measured E[T]_both >= %.4f (denken #241) AND measured min-lambda_hat q[2..9] >= %.7f "
        "(fern #249 / stark #191). Until then treeverify_served_gain_MEASURED_realized=%.1f and "
        "reach500_A_is_measured=False." % (ET_FLOOR_OPERATIVE, LAMBDA_GATE, banked["treeverify_realized"]))

    return {
        "P_A": p_a,
        "P_A_brackets": brackets,
        "priors": PATH_A_PRIORS,
        "prior_statement": prior_statement,
        "collapsing_measurement": collapsing_measurement,
        "reach500_A_is_measured": reach500_A_is_measured,
        "gates": {
            "et_floor_operative": ET_FLOOR_OPERATIVE,
            "et_floor_self_insured": ET_FLOOR_SELF_INSURED,
            "lambda_gate": LAMBDA_GATE,
            "proj_e_t_both": PROJ_ET_BOTH,
            "proj_min_lambda": PROJ_MIN_LAMBDA,
            "et_shortfall_tolerance": DELTA_MAX_TPS500,
            "lambda_headroom_proj": LAMBDA_HEADROOM_PROJ,
        },
    }


# --------------------------------------------------------------------------- #
# Step 2 -- PATH-B reach (the linear-lever portfolio).
# --------------------------------------------------------------------------- #
def _group_gain(gains: dict[str, float], group: str) -> float:
    """Within-group composition: best lever + overlap_retain * (sum of the rest). Same-component
    levers do NOT stack (you cannot remove the same step time twice)."""
    members = [g for k, g in gains.items() if PATH_B_LEVERS_BY_KEY[k]["group"] == group]
    if not members:
        return 0.0
    members_sorted = sorted(members, reverse=True)
    best = members_sorted[0]
    rest = sum(members_sorted[1:])
    return best + OVERLAP_RETAIN[group] * rest


def _stack_total(gains: dict[str, float]) -> float:
    """Across-group composition on the LINEAR official law:
       E[T]-axis gain (a) composes MULTIPLICATIVELY with the step-axis gain (b); the two step-lowering
       groups (SYS+DRAFT) combine with an Amdahl efficiency ETA_STEP (you cannot fully add two step
       reductions). total_gain = (1+a)*(1+b) - 1."""
    a = _group_gain(gains, "ET")
    b = ETA_STEP * (_group_gain(gains, "SYS") + _group_gain(gains, "DRAFT"))
    return (1.0 + a) * (1.0 + b) - 1.0


def _point_stack(go_weighted: bool) -> dict[str, Any]:
    """Deterministic point stack. go_weighted=False: 'if every lever fires at its central (mid) gain'
    (the PR's reach500_B = 1[sum g_i >= threshold] indicator). go_weighted=True: expected gain with
    each lever's central gain * p_go (a probability-discounted companion)."""
    gains = {}
    for lv in PATH_B_LEVERS:
        mid = 0.5 * (lv["gain_lo"] + lv["gain_hi"])
        gains[lv["key"]] = mid * (lv["p_go"] if go_weighted else 1.0)
    total = _stack_total(gains)
    return {"per_lever_gain": gains, "stacked_gain": total, "reach": bool(total >= PATH_B_THRESHOLD)}


def path_b_portfolio() -> dict[str, Any]:
    import numpy as np

    keys = [lv["key"] for lv in PATH_B_LEVERS]
    p_go = np.array([lv["p_go"] for lv in PATH_B_LEVERS])
    lo = np.array([lv["gain_lo"] for lv in PATH_B_LEVERS])
    hi = np.array([lv["gain_hi"] for lv in PATH_B_LEVERS])
    groups = [lv["group"] for lv in PATH_B_LEVERS]
    grp_idx = {g: [i for i, gg in enumerate(groups) if gg == g] for g in ("ET", "SYS", "DRAFT")}

    def mc_p_b(n: int, seed: int, go_scale: float = 1.0, gain_mult: float = 1.0,
              eta: float = ETA_STEP, overlap: dict[str, float] | None = None) -> float:
        ov = OVERLAP_RETAIN if overlap is None else overlap
        rng = np.random.default_rng(seed)
        p_eff = np.clip(p_go * go_scale, 0.0, 1.0)
        go = rng.random((n, len(keys))) < p_eff         # (n, L) Bernoulli GO
        raw = (lo + rng.random((n, len(keys))) * (hi - lo)) * gain_mult
        g = np.where(go, raw, 0.0)                       # gain = GO * Uniform(lo,hi)*gain_mult

        def grp(group: str) -> np.ndarray:
            idx = grp_idx[group]
            if not idx:
                return np.zeros(n)
            sub = g[:, idx]
            best = sub.max(axis=1)
            rest = sub.sum(axis=1) - best
            return best + ov[group] * rest

        a = grp("ET")
        b = eta * (grp("SYS") + grp("DRAFT"))
        total = (1.0 + a) * (1.0 + b) - 1.0
        return float((total >= PATH_B_THRESHOLD).mean())

    p_b = mc_p_b(MC_N, MC_SEED)
    se = math.sqrt(max(p_b * (1.0 - p_b), 0.0) / MC_N)
    stability = [mc_p_b(MC_N, s) for s in MC_SEEDS_STABILITY]
    spread = max(stability) - min(stability)

    # PRIOR-SENSITIVITY sweep: P_B swings with the (unobserved) prior, so report its range explicitly.
    # pessimistic -> lower GO, smaller gains, harsher Amdahl, less within-group stacking; optimistic ->
    # the reverse (overlap clamped to [0, 0.5]; DRAFT stays 0 -- int3/n-gram remain mutually exclusive).
    sweep_scenarios = {
        "pessimistic": {"go_scale": 0.70, "gain_mult": 0.80, "eta": 0.75,
                        "overlap": {"ET": 0.0, "SYS": 0.10, "DRAFT": 0.0}},
        "central":     {"go_scale": 1.00, "gain_mult": 1.00, "eta": ETA_STEP,
                        "overlap": OVERLAP_RETAIN},
        "optimistic":  {"go_scale": 1.30, "gain_mult": 1.15, "eta": 1.00,
                        "overlap": {"ET": 0.0, "SYS": 0.50, "DRAFT": 0.0}},
    }
    sweep = {name: mc_p_b(MC_N, MC_SEED, **sc) for name, sc in sweep_scenarios.items()}
    p_b_range = [min(sweep.values()), max(sweep.values())]

    # the PR's literal indicator + a probability-discounted companion.
    point_if_all_fire = _point_stack(go_weighted=False)
    point_go_weighted = _point_stack(go_weighted=True)

    # any-single-lever-alone reach (which lever can clear the threshold by itself, if it fires).
    single_alone = {}
    for lv in PATH_B_LEVERS:
        # a lone lever: ET via (1+mid_hi), step via ETA_STEP*mid_hi.
        if lv["group"] == "ET":
            hi_total = (1.0 + lv["gain_hi"]) - 1.0
            lo_total = (1.0 + lv["gain_lo"]) - 1.0
        else:
            hi_total = (1.0 + ETA_STEP * lv["gain_hi"]) - 1.0
            lo_total = (1.0 + ETA_STEP * lv["gain_lo"]) - 1.0
        single_alone[lv["key"]] = {
            "can_clear_alone_at_hi": bool(hi_total >= PATH_B_THRESHOLD),
            "can_clear_alone_at_lo": bool(lo_total >= PATH_B_THRESHOLD),
            "gain_range_total": [lo_total, hi_total],
        }

    levers_table = []
    for lv in PATH_B_LEVERS:
        levers_table.append({
            "screen_pr": lv["screen_pr"], "name": lv["name"], "doc_rank": lv["doc_rank"],
            "group": lv["group"], "p_go": lv["p_go"],
            "projected_gain_range": [lv["gain_lo"], lv["gain_hi"]],
            "greedy_id_risk": lv["greedy_id_risk"], "rationale": lv["rationale"],
        })

    prior_statement = (
        "P_B = P(stacked linear gain >= %.5f) over the 5 in-flight screens' PROJECTED-gain ranges "
        "(NOT their unreturned measured results). STACKING: levers are grouped by the step component "
        "they target; WITHIN a group they do NOT stack (SYS overlap_retain=%.2f for FlashInfer + "
        "activation-recycle; DRAFT overlap_retain=%.2f because int3-draft and n-gram are mutually "
        "exclusive draft sources); ACROSS groups the E[T]-axis composes multiplicatively with the "
        "step-axis and the two step groups carry an Amdahl efficiency ETA_STEP=%.2f. MC over GO x "
        "Uniform priors, N=%d, seed=%d."
        % (PATH_B_THRESHOLD, OVERLAP_RETAIN["SYS"], OVERLAP_RETAIN["DRAFT"], ETA_STEP, MC_N, MC_SEED))

    collapsing_measurement = (
        "each screen's GO/NO-GO verdict + a kernel build clearing PPL-valid wall_tps >= 500. NONE of "
        "the 5 screens (stark #247 / lawine #246 / kanna #248 / ubel #250 / wirbel #251) has returned "
        "a GO verdict yet -> path_b_go_returned=False.")

    return {
        "P_B": p_b,
        "P_B_mc_se": se,
        "P_B_stability_spread": spread,
        "P_B_stability_seeds": dict(zip([str(s) for s in MC_SEEDS_STABILITY], stability)),
        "P_B_prior_sweep": sweep,
        "P_B_prior_sweep_scenarios": sweep_scenarios,
        "P_B_range_over_priors": p_b_range,
        "path_b_go_returned": False,
        "threshold": PATH_B_THRESHOLD,
        "reach500_B_point_if_all_fire": point_if_all_fire,
        "reach500_B_point_go_weighted": point_go_weighted,
        "single_lever_alone": single_alone,
        "levers": levers_table,
        "overlap_model": {"overlap_retain": OVERLAP_RETAIN, "eta_step": ETA_STEP,
                          "across_group_law": "(1+a_ET)*(1+b_step)-1, b_step=ETA_STEP*(SYS+DRAFT)"},
        "prior_statement": prior_statement,
        "collapsing_measurement": collapsing_measurement,
    }


# --------------------------------------------------------------------------- #
# Step 2/3 -- the combined reach + the verdict table.
# --------------------------------------------------------------------------- #
def combined_reach(p_a: float, p_b: float) -> dict[str, Any]:
    combined = 1.0 - (1.0 - p_a) * (1.0 - p_b)
    return {
        "combined_reach500_prob": combined,
        "P_A": p_a,
        "P_B": p_b,
        "lower_bound_max": max(p_a, p_b),
        "upper_bound_sum": p_a + p_b,
        "second_shot_margin": combined - p_a,         # how much Path-B adds on top of Path-A alone
        "path_b_materially_de_risks": bool((combined - p_a) >= 0.10),
    }


def decision_table(a: dict[str, Any], b: dict[str, Any], comb: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "route": "PATH-A (tree build)",
            "gate": "measured E[T]_both >= %.4f AND measured lambda_hat q[2..9] >= %.7f"
                    % (ET_FLOOR_OPERATIVE, LAMBDA_GATE),
            "current_status": "PRIOR (UNMEASURED): treeverify_served_gain_MEASURED_realized=0.0",
            "reach_prob": a["P_A"],
            "what_collapses_prior_to_fact": a["collapsing_measurement"],
        },
        {
            "route": "PATH-B (linear-lever portfolio)",
            "gate": "stacked linear gain >= %.5f (481.53 -> 500 on official=K_cal*(E[T]/step)*tau)"
                    % b["threshold"],
            "current_status": "PRIOR (NO GO returned): path_b_go_returned=False over 5 in-flight screens",
            "reach_prob": b["P_B"],
            "what_collapses_prior_to_fact": b["collapsing_measurement"],
        },
        {
            "route": "COMBINED (1 - (1-P_A)(1-P_B))",
            "gate": "FIRST measured PPL-valid wall_tps >= 500 from EITHER route",
            "current_status": "PRIOR: neither route has a measured >=500 -> readiness stays NOT-READY",
            "reach_prob": comb["combined_reach500_prob"],
            "what_collapses_prior_to_fact": ("land #245's build run (A) OR any Path-B screen GO + kernel "
                                             "build clearing PPL-valid wall_tps >= 500 (B)"),
        },
    ]


# --------------------------------------------------------------------------- #
# self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(banked: dict[str, Any], a: dict[str, Any], b: dict[str, Any],
              comb: dict[str, Any]) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    p_a, p_b = a["P_A"], b["P_B"]
    combined = comb["combined_reach500_prob"]

    # (a) combined round-trips 1-(1-P_A)(1-P_B) and obeys the probability bounds:
    #     max(P_A,P_B) <= combined <= P_A+P_B, and combined in [0,1].
    rt = 1.0 - (1.0 - p_a) * (1.0 - p_b)
    a_ok = bool(_close(combined, rt, 1e-12)
                and (combined >= max(p_a, p_b) - 1e-12)
                and (combined <= p_a + p_b + 1e-12)
                and 0.0 <= combined <= 1.0
                and 0.0 <= p_a <= 1.0 and 0.0 <= p_b <= 1.0)
    conditions["a_combined_roundtrip_and_prob_bounds"] = {
        "pass": a_ok, "combined": combined, "roundtrip": rt,
        "lower_bound_max": max(p_a, p_b), "upper_bound_sum": p_a + p_b,
        "combined_ge_max": bool(combined >= max(p_a, p_b) - 1e-12),
        "combined_le_sum": bool(combined <= p_a + p_b + 1e-12),
    }

    # (b) P_A and P_B EACH carry an explicit stated prior + the measurement that collapses it.
    b_ok = bool(
        isinstance(a["prior_statement"], str) and len(a["prior_statement"]) > 40
        and isinstance(a["collapsing_measurement"], str) and len(a["collapsing_measurement"]) > 40
        and isinstance(b["prior_statement"], str) and len(b["prior_statement"]) > 40
        and isinstance(b["collapsing_measurement"], str) and len(b["collapsing_measurement"]) > 40)
    conditions["b_each_prior_has_statement_and_collapsing_measurement"] = {
        "pass": b_ok,
        "P_A_has_prior": bool(a["prior_statement"]),
        "P_A_has_collapsing_measurement": bool(a["collapsing_measurement"]),
        "P_B_has_prior": bool(b["prior_statement"]),
        "P_B_has_collapsing_measurement": bool(b["collapsing_measurement"]),
    }

    # (c) reach500_A_is_measured=False AND no Path-B GO returned => readiness_verdict stays NOT-READY
    #     (the card cannot read a prior as a delivered win -- same discipline as #238's
    #     treeverify_realized==0 gate).
    no_measured_win = (not a["reach500_A_is_measured"]) and (not b["path_b_go_returned"])
    readiness = banked["readiness_verdict_carried"]
    c_ok = bool(no_measured_win and readiness == "NOT-READY")
    conditions["c_no_measured_win_keeps_not_ready"] = {
        "pass": c_ok,
        "reach500_A_is_measured": a["reach500_A_is_measured"],
        "path_b_go_returned": b["path_b_go_returned"],
        "readiness_verdict": readiness,
    }

    # (d) the +3.8357% Path-B threshold round-trips 481.53 -> 500 via the LINEAR composition:
    #     (i) baseline*(1+threshold) == 500 exactly; (ii) a g-fractional gain in (E[T]/step) is a
    #     g-fractional gain in official TPS (composition linearity, slope-invariant).
    thr = b["threshold"]
    baseline_roundtrip = BASELINE_TPS * (1.0 + thr)
    g = thr
    comp_ratio = official_tps(F_LINEAR8 * (1.0 + g)) / official_tps(F_LINEAR8) - 1.0
    d_ok = bool(_close(baseline_roundtrip, TARGET_TPS, 1e-9)
                and _close(comp_ratio, g, TOL_COMPOSITION))
    conditions["d_threshold_roundtrips_481p53_to_500_via_composition"] = {
        "pass": d_ok, "threshold": thr, "baseline_times_1_plus_thr": baseline_roundtrip,
        "target": TARGET_TPS, "composition_linearity_ratio": comp_ratio,
    }

    # (e) the Path-A gates match the imported #249 (lambda) and #241 (E[T]) EXACTLY.
    rt238_249_241 = banked["roundtrip"]
    e_ok = bool(
        rt238_249_241["249.build_lambda_operative_gate"]["round_trips"]
        and _close(LAMBDA_GATE, 0.9780112973731208, 0.0)
        and rt238_249_241["241.e_t_meas_floor"]["round_trips"]
        and _close(ET_FLOOR_OPERATIVE, 4.330527243789328, 0.0)
        and all(d["round_trips"] for d in rt238_249_241.values()))
    conditions["e_path_a_gates_match_249_241_exactly"] = {
        "pass": e_ok,
        "lambda_gate": LAMBDA_GATE,
        "lambda_gate_249_round_trips": rt238_249_241["249.build_lambda_operative_gate"]["round_trips"],
        "et_floor": ET_FLOOR_OPERATIVE,
        "et_floor_241_round_trips": rt238_249_241["241.e_t_meas_floor"]["round_trips"],
        "provenance_max_abs_err": max(d["abs_err"] for d in rt238_249_241.values()
                                      if _finite(d["abs_err"])),
        "all_round_trip": all(d["round_trips"] for d in rt238_249_241.values()),
    }

    # (f) NaN-clean across the reported payload.
    payload = {"a": a, "b": b, "comb": comb, "banked_roundtrip": banked["roundtrip"]}
    nan_paths = _nan_paths(payload, "selftest")
    f_ok = (len(nan_paths) == 0)
    conditions["f_nan_clean"] = {"pass": f_ok, "nan_paths": nan_paths}

    passes = bool(all(c["pass"] for c in conditions.values()))
    return {
        "two_path_portfolio_self_test_passes": passes,
        "conditions": conditions,
        "n_conditions": len(conditions),
    }


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run() -> dict[str, Any]:
    t0 = time.time()
    banked = import_banked()
    a = path_a_reach(banked)
    b = path_b_portfolio()
    comb = combined_reach(a["P_A"], b["P_B"])
    table = decision_table(a, b, comb)
    st = self_test(banked, a, b, comb)

    readiness_statement = (
        "readiness_verdict=%s (carried UNCHANGED from fern #238). combined_reach500_prob=%.4f under "
        "stated priors (P_A=%.4f tree build, P_B=%.4f linear portfolio). Path-B adds "
        "second_shot_margin=%.4f on top of Path-A alone -> it %s materially de-risk the "
        "all-in-on-land's-build posture. The launch is STILL single-blocked on a MEASURED win: "
        "reach500_A_is_measured=False AND path_b_go_returned=False, so NEITHER route has a measured "
        "PPL-valid wall_tps >= 500 yet."
        % (banked["readiness_verdict_carried"], comb["combined_reach500_prob"], a["P_A"], b["P_B"],
           comb["second_shot_margin"],
           "DOES" if comb["path_b_materially_de_risks"] else "does NOT"))

    handoff = (
        "the team has two independent routes to 500 -- Path-A (land #245 tree build, gated "
        "E[T]_both>=%.4f AND lambda_hat>=%.4f) and Path-B (the 5-lever linear portfolio, reach iff "
        "stacked gain >=%.4f%%) -- with combined_reach500_prob=%.4f under stated priors; both are "
        "currently PROJECTIONS (no measured >=500), so readiness stays NOT-READY until the first "
        "measured PPL-valid wall_tps>=500 from EITHER path, but Path-B %s materially de-risk the "
        "all-in-on-land's-build posture (second-shot margin %.4f)."
        % (ET_FLOOR_OPERATIVE, LAMBDA_GATE, PATH_B_THRESHOLD * 100.0,
           comb["combined_reach500_prob"],
           "DOES" if comb["path_b_materially_de_risks"] else "does NOT", comb["second_shot_margin"]))

    peak_mem_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    result = {
        "pr": 253,
        "agent": "fern",
        "kind": "two_path_500_portfolio",
        "metric_primary": "two_path_portfolio_self_test_passes",
        "metric_test": "combined_reach500_prob",
        # ---- PRIMARY + TEST ----
        "two_path_portfolio_self_test_passes": st["two_path_portfolio_self_test_passes"],
        "combined_reach500_prob": comb["combined_reach500_prob"],
        # ---- the two route reach probabilities + the second-shot read ----
        "P_A": a["P_A"],
        "P_B": b["P_B"],
        "P_A_range_over_priors": [a["P_A_brackets"]["adverse"], a["P_A_brackets"]["optimistic"]],
        "P_B_range_over_priors": b["P_B_range_over_priors"],
        "combined_range_over_priors": [
            1.0 - (1.0 - a["P_A_brackets"]["adverse"]) * (1.0 - b["P_B_range_over_priors"][0]),
            1.0 - (1.0 - a["P_A_brackets"]["optimistic"]) * (1.0 - b["P_B_range_over_priors"][1]),
        ],
        "second_shot_margin": comb["second_shot_margin"],
        "path_b_materially_de_risks": comb["path_b_materially_de_risks"],
        "reach500_A_is_measured": a["reach500_A_is_measured"],
        "path_b_go_returned": b["path_b_go_returned"],
        "readiness_verdict": banked["readiness_verdict_carried"],
        "readiness_statement": readiness_statement,
        # ---- sections ----
        "path_a": a,
        "path_b": b,
        "combined": comb,
        "decision_table": table,
        "self_test": st,
        "provenance_roundtrip": banked["roundtrip"],
        "handoff": handoff,
        "scope": (
            "LOCAL CPU-only integration over EXISTING MERGED legs (#238/#241/#244/#249) loaded from "
            "their committed result JSON and round-tripped scalar-for-scalar, PLUS the 5 in-flight "
            "Path-B screens priced as a PORTFOLIO PRIOR over the speed-levers researcher doc's "
            "projected-gain ranges (NOT their unreturned measured results). The only new object is the "
            "COMBINED reach 1-(1-P_A)(1-P_B). No GPU / vLLM / HF Job / submission / served-file change "
            "/ official draw. BASELINE stays 481.53; adds 0 TPS; greedy/PPL untouched; authorizes "
            "NOTHING. NOT a launch. NOT open2."),
        "public_evidence_used": [
            "fern #238 (launch_decision_card): readiness_verdict=NOT-READY, n_green_gates=2, "
            "treeverify_served_gain_MEASURED_realized=0.0, land #71 projection E[T]_both=4.512 / "
            "min-lambda 0.983 -- loaded from the committed artifact and round-tripped.",
            "denken #241 (measured_et_shortfall): E[T] operative floor 4.3305 / self-insured 4.4890, "
            "delta_max_tps500=0.04022 (4.02% shortfall tolerance), lambda margin 0.00502, the LINEAR "
            "composition K_cal=125.268 / step=1.2182, lambda=1 ceiling 520.95 -- round-tripped.",
            "fern #249 (build_lambda_bar): operative build-lambda gate 0.9780112973731208 (== stark "
            "#191 P95 == #241 lambda_star_191) -- round-tripped.",
            "wirbel #244 (ceiling_gap_topology_headroom): compliant-PRIVATE-500 lane TOPOLOGY-DEAD "
            "(lane_reopenable=False, topology_lift_max=0.0, reopener=coverage/lambda->1) -- the banked "
            "evidence that priors stark #247 (OPT-Tree, a topology lever) DOWN.",
            "speed-levers researcher doc (research/RESEARCH_IDEAS_2026-06-14_speed-levers.md): the "
            "Ranked Priority Table supplying each Path-B lever's projected-gain range and greedy-ID "
            "risk (T-1 OPT-Tree, K-1 FlashInfer+CUDAGraph, Q-1 int3-draft, N-1 n-gram, activation-recycle).",
        ],
        "method": (
            "LOCAL CPU-only analytic integration. Load #238/#241/#244/#249 committed JSON; round-trip "
            "their headline scalars; price Path-A reach P_A = p_build_lands*p_ET_clears*p_lambda_clears "
            "(stated priors, independent gates, bracketed adverse/central/optimistic); price Path-B "
            "reach P_B = P(stacked linear gain >= 3.8357%) via Monte-Carlo over GO x Uniform projected-"
            "gain priors with within-group non-stacking + Amdahl across groups; combine "
            "1-(1-P_A)(1-P_B); carry fern #238 readiness NOT-READY forward unchanged. No "
            "GPU/vLLM/HF Job/submission/served-file/draw. BASELINE stays 481.53; adds 0 TPS."),
        "metrics_nan_clean": 1 if st["conditions"]["f_nan_clean"]["pass"] else 0,
        "peak_mem_mib": peak_mem_mib,
        "elapsed_s": round(time.time() - t0, 4),
    }
    # payload-level NaN guard.
    nan_paths = _nan_paths(result, "result")
    result["nan_clean"] = not nan_paths
    if nan_paths:
        result["metrics_nan_clean"] = 0
        result["self_test"]["conditions"]["f_nan_clean"]["pass"] = False
        result["self_test"]["conditions"]["f_nan_clean"]["nan_paths_payload"] = nan_paths
        result["self_test"]["two_path_portfolio_self_test_passes"] = False
        result["two_path_portfolio_self_test_passes"] = False
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
        print(f"[two-path-500] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    run = wandb_logging.init_wandb_run(
        job_type="two-path-500-portfolio", agent="fern",
        name=args.wandb_name or "fern/two-path-500-portfolio",
        group=args.wandb_group,
        tags=["two-path-500", "combined-reach", "path-a-tree-build", "path-b-linear-portfolio",
              "launch-readiness-integration", "publish-first", "issue-124", "pr253"],
        config={"baseline_tps": BASELINE_TPS, "target_tps": TARGET_TPS, "method": "cpu-only-analytic",
                "path_b_threshold": PATH_B_THRESHOLD, "mc_n": MC_N, "mc_seed": MC_SEED,
                "imports_pr": [238, 241, 244, 249, 217]},
    )
    if run is None:
        print("[two-path-500] wandb disabled; skipping", flush=True)
        return
    try:
        flat = {
            "two_path_portfolio_self_test_passes":
                1.0 if result["two_path_portfolio_self_test_passes"] else 0.0,
            "combined_reach500_prob": result["combined_reach500_prob"],
            "P_A": result["P_A"],
            "P_B": result["P_B"],
            "P_B_mc_se": result["path_b"]["P_B_mc_se"],
            "P_B_stability_spread": result["path_b"]["P_B_stability_spread"],
            "second_shot_margin": result["second_shot_margin"],
            "path_b_materially_de_risks": 1.0 if result["path_b_materially_de_risks"] else 0.0,
            "reach500_A_is_measured": 1.0 if result["reach500_A_is_measured"] else 0.0,
            "path_b_go_returned": 1.0 if result["path_b_go_returned"] else 0.0,
            "readiness_not_ready": 1.0 if result["readiness_verdict"] == "NOT-READY" else 0.0,
            "P_A_adverse": result["path_a"]["P_A_brackets"]["adverse"],
            "P_A_central": result["path_a"]["P_A_brackets"]["central"],
            "P_A_optimistic": result["path_a"]["P_A_brackets"]["optimistic"],
            "path_b_threshold": result["path_b"]["threshold"],
            "reach500_B_point_if_all_fire":
                1.0 if result["path_b"]["reach500_B_point_if_all_fire"]["reach"] else 0.0,
            "reach500_B_point_if_all_fire_gain":
                result["path_b"]["reach500_B_point_if_all_fire"]["stacked_gain"],
            "provenance_max_abs_err":
                result["self_test"]["conditions"]["e_path_a_gates_match_249_241_exactly"]
                ["provenance_max_abs_err"],
            "et_floor_operative": ET_FLOOR_OPERATIVE,
            "lambda_gate": LAMBDA_GATE,
            "metrics_nan_clean": float(result["metrics_nan_clean"]),
            "peak_mem_mib": result["peak_mem_mib"],
            **{f"selftest_{k}": (1.0 if v["pass"] else 0.0)
               for k, v in result["self_test"]["conditions"].items()},
        }
        try:
            import wandb
            tbl = wandb.Table(columns=["route", "gate", "current_status", "reach_prob",
                                       "what_collapses_prior_to_fact"])
            for r in result["decision_table"]:
                tbl.add_data(r["route"], r["gate"], r["current_status"], r["reach_prob"],
                             r["what_collapses_prior_to_fact"])
            flat["decision_table"] = tbl
            lev = wandb.Table(columns=["screen_pr", "name", "doc_rank", "group", "p_go",
                                       "projected_gain_range", "greedy_id_risk"])
            for r in result["path_b"]["levers"]:
                lev.add_data(r["screen_pr"], r["name"], r["doc_rank"], r["group"], r["p_go"],
                             str(r["projected_gain_range"]), r["greedy_id_risk"])
            flat["path_b_levers"] = lev
        except Exception as exc:  # noqa: BLE001
            print(f"[two-path-500] wandb table skipped ({exc})", flush=True)
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="two_path_500_portfolio", artifact_type="validity", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[two-path-500] WARN wandb logging error: {exc}", flush=True)
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
    print("PR #253  TWO-PATH-TO-500 PORTFOLIO -- combined P(reach 500) over tree build (A) + linear "
          "portfolio (B)", flush=True)
    print("=" * 100, flush=True)
    for r in result["decision_table"]:
        print(f"  [{r['route']}]  reach_prob = {r['reach_prob']:.4f}", flush=True)
        print(f"      gate:   {r['gate']}", flush=True)
        print(f"      status: {r['current_status']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  P_A (tree build)        = {result['P_A']:.4f}   "
          f"[adverse {result['path_a']['P_A_brackets']['adverse']:.3f} / central "
          f"{result['path_a']['P_A_brackets']['central']:.3f} / optimistic "
          f"{result['path_a']['P_A_brackets']['optimistic']:.3f}]", flush=True)
    sw = result["path_b"]["P_B_prior_sweep"]
    print(f"  P_B (linear portfolio)  = {result['P_B']:.4f}   "
          f"(MC se {result['path_b']['P_B_mc_se']:.4f}, stability spread "
          f"{result['path_b']['P_B_stability_spread']:.4f})", flush=True)
    print(f"      prior sweep: pessimistic {sw['pessimistic']:.4f} / central {sw['central']:.4f} / "
          f"optimistic {sw['optimistic']:.4f}", flush=True)
    print(f"  combined_reach500_prob  = {result['combined_reach500_prob']:.4f}   "
          f"(>= max {max(result['P_A'], result['P_B']):.4f}, <= sum "
          f"{result['P_A'] + result['P_B']:.4f}; prior range "
          f"[{result['combined_range_over_priors'][0]:.4f}, "
          f"{result['combined_range_over_priors'][1]:.4f}])", flush=True)
    print(f"  second_shot_margin      = {result['second_shot_margin']:.4f}   "
          f"-> Path-B {'DOES' if result['path_b_materially_de_risks'] else 'does NOT'} materially "
          f"de-risk", flush=True)
    print("-" * 100, flush=True)
    print(f"  readiness_verdict       = {result['readiness_verdict']}  "
          f"(reach500_A_is_measured={result['reach500_A_is_measured']}, "
          f"path_b_go_returned={result['path_b_go_returned']})", flush=True)
    print("-" * 100, flush=True)
    print(f"  PRIMARY two_path_portfolio_self_test_passes = "
          f"{st['two_path_portfolio_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"    [{'ok' if v['pass'] else '!! FAILED'}] {k}", flush=True)
    print(f"\n  HANDOFF: {result['handoff']}\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "two_path_500_portfolio_results.json"))
    ap.add_argument("--self-test", "--self_test", action="store_true",
                    help="run the self-test (PRIMARY); nonzero exit on failure")
    ap.add_argument("--wandb-name", "--wandb_name", default="fern/two-path-500-portfolio")
    ap.add_argument("--wandb-group", "--wandb_group", default="launch-readiness-integration")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    result = run()

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"[two-path-500] wrote {args.out}", flush=True)

    _print(result)
    _log_wandb(args, result)

    if args.self_test and not result["two_path_portfolio_self_test_passes"]:
        print("[two-path-500] SELF-TEST FAILED", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
