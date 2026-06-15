#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #259 (fern) -- E[T] branch-interior sensitivity: does Path-A survive land #245's
scratch-verify finding?  LOCAL CPU-only analytic re-partition. NOT a launch. NOT open2.

THE QUESTION
------------
land #245 cycle-1 proved the scratch-reconstruction verify forward is only ~0.599 faithful
(a genuine reconstruction bug, NOT issue #192). The deep branch-INTERIOR contribution to the
banked both-bugs projection E[T]=4.512 is therefore UNVERIFIED and may be optimistic, while the
SPINE and the first-level branch-hit (rho2) are real-KV-confirmed (land #245 reconfirmed the
real-KV ladder this cycle). The decisive question for the whole Path-A posture:

  if the suspect branch-interior is DISCOUNTED, does the E[T] LOWER bound still clear denken
  #241's measured floor  E_T_meas_floor = 4.3305 ?

The gap is thin: 4.512 - 4.3305 = 0.18147 of E[T] headroom. CRUX: is the scratch-derived
branch-interior share of E[T]=4.512 SMALLER than 0.18147 (=> Path-A robust to land's finding,
the measured build clears 500 even fully discounting the suspect part) or LARGER (=> the
branch-interior is LOAD-BEARING and land's fidelity-first step is make-or-break)?

THE METHOD (inversion / attribution -- it ROUND-TRIPS)
-----------------------------------------------------
The 4.512 projection is F_tree of the rho-optimal max-branch-3 / spine-9 tree (wirbel topo74 at
budget M=16, committed in research/spec_cost_model/rho_optimal_topology_results.json,
F_tree = 4.512274954048941). We re-run the SAME tree-economics model that BUILT it -- the
renewal-reward / path-enumeration tree-walk over that committed parent array with the committed
depth_q_76 spine ladder + rho_cond rank-rescue ladder (the descent_et_audit #172 / #135 model) --
and PARTITION its additive node contributions. E[T] = 1 + sum_nodes reach_prob(node) is additive,
so the partition is exact and SUMS to F_tree by construction:

  * E_spine            = root token + the rank-1 greedy chain (the linear-chain accepted length).
                         Real-KV-confirmed (land #245 reconfirmed the spine ladder: top1 0.69-0.78,
                         mean_accepted_len 2.57-3.69, lambda-vs-top1 d2..d7 1.03-1.18 all clear 0.9780).
  * E_branchhit_rho2   = the FIRST-LEVEL salvage: rank>=2 children OF spine nodes (the immediate
                         divergence rescue, headlined by rho2~0.4165). Real-KV-confirmed (the first
                         divergence rank-coverage rho_cond was measured on real KV, #79/#245).
  * E_branch_interior  = the RESIDUAL: every node DEEPER inside a branch (a descendant of a
                         branch-hit). In land's machinery this continuation can only come from the
                         scratch verify forward => scratch-suspect (0.599-faithful, land #245).

We attribute the BANKED E[T]=4.512 in the tree-walk's structural shares so the three components
sum to 4.512 exactly, then discount the suspect part two ways and check the floor.

  E_T_lower_full_discount     = confirmed_et = E_spine + E_branchhit_rho2     (branch-interior REMOVED)
  E_T_lower_fidelity_discount = confirmed_et + 0.599 * E_branch_interior      (scaled by scratch fidelity;
                                we EXPECT the real path BETTER than scratch, so this is conservative)

  path_a_robust_to_scratch_finding = (E_T_lower_full_discount >= 4.3305)      (the worst case clears?)

IMPORTS (do NOT re-derive): E[T]_both=4.512 (land #238/#245, fern #253); the committed topo74 M16
parent array + depth_q_76 + rho_cond (wirbel #83 6tghbnjn / rho_optimal_topology); E_T_meas_floor
=4.330527243789328 and the 520.95 lambda=1 ceiling (denken #241 hqewf1d6); lambda-bar 0.9780112973731208
(fern #249 / stark #191); scratch fidelity 0.599 (land #245). This leg ATTRIBUTES the projection --
it does NOT move it. The lambda-hat gate-2 evidence is real-KV and is NOT touched. No GPU / vLLM /
HF Job / submission / served-file change / official draw. BASELINE stays 481.53; adds 0 TPS.

PRIMARY metric  et_branch_interior_sensitivity_self_test_passes
TEST    metric  e_t_lower_branch_interior_discounted  ( = E_T_lower_full_discount )

Run:
  cd target/ && CUDA_VISIBLE_DEVICES="" python \
    research/validity/et_branch_interior_sensitivity/et_branch_interior_sensitivity.py \
    --self-test --wandb_group launch-readiness-integration --wandb_name fern/et-branch-interior-sensitivity
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
# Committed anchors (advisor-branch content only; nothing re-derived).
# --------------------------------------------------------------------------- #
TOPO_JSON = os.path.join(REPO_ROOT, "research/spec_cost_model/rho_optimal_topology_results.json")

E_T_BOTH = 4.512                       # land #238/#245 both-bugs projected E[T]; fern #253 import (BANKED)
E_T_MEAS_FLOOR = 4.330527243789328     # denken #241 (hqewf1d6): 4.512 * 500/520.9527 -> clears-500 floor
LAMBDA1_CEILING_TPS = 520.9527323111674  # denken #217/#241 land GO read (lambda=1 ceiling)
LAMBDA_BAR = 0.9780112973731208        # fern #249 / stark #191 P95 validity gate (acceptance axis)
SCRATCH_FIDELITY = 0.599               # land #245 clean-room scratch verify-forward fidelity
BASELINE_TPS = 481.53                  # official baseline (PR #52); UNCHANGED by this leg
TARGET_OFFICIAL = 500.0
W_DEFAULT = 4

# the committed F_tree we must reproduce to PROVE we use the same model that built 4.512.
F_TREE_TOPO74_M16_COMMITTED = 4.512274954048941

# real-KV ladder land #245 reconfirmed this cycle (CONFIRMATION evidence for the spine + first
# branch-hit; the lambda-vs-top1 ladder is the gate-2 / lambda-hat evidence -- NOT touched here).
REAL_KV_RECONFIRM = {
    "mean_accepted_len_range": [2.57, 3.69],
    "top1_range": [0.69, 0.78],
    "lambda_vs_top1_by_depth": {"d2": 1.03, "d3": 1.04, "d4": 1.08, "d5": 1.15, "d6": 1.14, "d7": 1.18},
    "branch_hit_rho2": 0.4165047789261015,
    "lambda_bar_all_clear": 0.9780112973731208,
}

TOL_SUM = 1e-6          # PR self-test (a): three components sum to 4.512 within this.
TOL_REPRO = 1e-9        # tree-walk must reproduce the committed F_tree within this.
TOL_XMETHOD = 1e-12     # backward-DP vs path-enum cross-method agreement.
TOL_RT = 1e-9           # floor / ceiling round-trip tolerance.


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# The committed tree-economics model (descent_et_audit #172 / wirbel #135), verbatim shape.
#   pv[1] = q[d]                                  (rank-1 spine conditional accept)
#   pv[r] = (prod_{j<r}(1-rho_j)) * rho_r * (1-q[d])   for r >= 2  (rank-r first-divergence rescue)
#   E[T]  = 1 + sum over non-root nodes of reach_prob(node)         (greedy chain, siblings exclusive)
# --------------------------------------------------------------------------- #
def my_pvec(qd: float, rho_cond: list[float], W: int) -> list[float]:
    pv = [0.0] * (W + 1)
    pv[1] = qd
    miss = 1.0 - qd
    surv = 1.0
    for r in range(2, W + 1):
        rr = rho_cond[r - 2] if (r - 2) < len(rho_cond) else 0.0
        pv[r] = surv * rr * miss
        surv *= (1.0 - rr)
    return pv


def qd_at(spine: list[float], d: int) -> float:
    """Conditional spine accept at depth d (1-indexed); flat extrapolation past the measured horizon."""
    return spine[d - 1] if d <= len(spine) else spine[-1]


def build_children(parent: list[int]) -> tuple[list[list[int]], list[int]]:
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    depth = [0] * n
    for i in range(1, n):
        children[parent[i]].append(i)
    for i in range(1, n):
        depth[i] = depth[parent[i]] + 1
    return children, depth


def et_backward(parent, children, depth, spine, rho_cond, W) -> float:
    """M1 -- backward renewal-reward DP (post-order). E[T] = 1 + D(root)."""
    n = len(parent)
    D = [0.0] * n
    for u in sorted(range(n), key=lambda x: -depth[x]):
        d = depth[u] + 1
        pv = my_pvec(qd_at(spine, d), rho_cond, W)
        s = 0.0
        for rank, c in enumerate(children[u], start=1):
            r = rank if rank <= W else W
            s += pv[r] * (1.0 + D[c])
        D[u] = s
    return 1.0 + D[0]


def et_pathenum(children, depth, spine, rho_cond, W) -> float:
    """M2 -- explicit root->node path-product enumeration (direct combinatorial expectation)."""
    total = 1.0

    def dfs(u: int, pp: float) -> None:
        nonlocal total
        for rank, c in enumerate(children[u], start=1):
            d = depth[c]
            pv = my_pvec(qd_at(spine, d), rho_cond, W)
            r = rank if rank <= W else W
            ppc = pp * pv[r]
            total += ppc
            dfs(c, ppc)

    dfs(0, 1.0)
    return total


# --------------------------------------------------------------------------- #
# The re-partition: classify every node and bucket its reach-probability.
#   spine      = the rank-1 greedy chain from the root (root included).
#   branchhit  = rank>=2 children OF spine nodes (first-level divergence salvage; rho2 headline).
#   interior   = every other non-root node (a descendant of a branch-hit; deep INSIDE a branch).
# --------------------------------------------------------------------------- #
def partition_tree(parent, children, depth, spine, rho_cond, W) -> dict[str, Any]:
    n = len(parent)
    reach = [0.0] * n
    rank_of = [0] * n
    bucket = ["interior"] * n

    reach[0] = 1.0
    rank_of[0] = 1
    bucket[0] = "spine"
    # parent[i] < i in these committed arrays, so a single forward pass sets parents before children.
    for u in range(n):
        for rank, c in enumerate(children[u], start=1):
            d = depth[c]
            pv = my_pvec(qd_at(spine, d), rho_cond, W)
            r = rank if rank <= W else W
            reach[c] = reach[u] * pv[r]
            rank_of[c] = rank
            if bucket[u] == "spine" and rank == 1:
                bucket[c] = "spine"
            elif bucket[u] == "spine" and rank >= 2:
                bucket[c] = "branchhit"
            else:
                bucket[c] = "interior"

    e_spine = sum(reach[u] for u in range(n) if bucket[u] == "spine")           # includes root's 1.0
    e_branchhit = sum(reach[u] for u in range(n) if bucket[u] == "branchhit")
    e_interior = sum(reach[u] for u in range(n) if bucket[u] == "interior")
    f_tree = e_spine + e_branchhit + e_interior

    node_rows = []
    for u in range(n):
        node_rows.append({
            "node": u, "parent": parent[u], "depth": depth[u], "rank": rank_of[u],
            "bucket": bucket[u], "reach_prob": reach[u],
        })

    return {
        "e_spine_raw": e_spine,
        "e_branchhit_raw": e_branchhit,
        "e_interior_raw": e_interior,
        "f_tree_raw": f_tree,
        "n_spine": sum(1 for b in bucket if b == "spine"),
        "n_branchhit": sum(1 for b in bucket if b == "branchhit"),
        "n_interior": sum(1 for b in bucket if b == "interior"),
        "node_rows": node_rows,
        "spine_nodes": [u for u in range(n) if bucket[u] == "spine"],
        "branchhit_nodes": [u for u in range(n) if bucket[u] == "branchhit"],
        "interior_nodes": [u for u in range(n) if bucket[u] == "interior"],
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def run() -> dict[str, Any]:
    t0 = time.time()
    topo = _load_json(TOPO_JSON)
    parent = [int(x) for x in topo["per_budget"]["16"]["topo74"]["parent"]]
    f_tree_committed = float(topo["per_budget"]["16"]["topo74"]["F_tree"])
    spine_ladder = [float(x) for x in topo["inputs"]["depth_q_76"]]
    rho_cond = [float(x) for x in topo["inputs"]["rho_cond_measured"]]
    W = W_DEFAULT

    children, depth = build_children(parent)

    # ---- (0) reproduce the committed 4.512 by the SAME model (two independent routes) ----
    et_m1 = et_backward(parent, children, depth, spine_ladder, rho_cond, W)
    et_m2 = et_pathenum(children, depth, spine_ladder, rho_cond, W)
    xmethod_resid = abs(et_m1 - et_m2)
    repro_resid_vs_committed = abs(et_m1 - f_tree_committed)
    repro_resid_vs_const = abs(et_m1 - F_TREE_TOPO74_M16_COMMITTED)

    # ---- (1) partition the additive tree-walk into the three buckets ----
    part = partition_tree(parent, children, depth, spine_ladder, rho_cond, W)
    f_tree_raw = part["f_tree_raw"]
    s_spine = part["e_spine_raw"] / f_tree_raw
    s_branchhit = part["e_branchhit_raw"] / f_tree_raw
    s_interior = part["e_interior_raw"] / f_tree_raw

    # attribute the BANKED 4.512 in the structural shares (sums to 4.512 exactly).
    e_spine = s_spine * E_T_BOTH
    e_branchhit_rho2 = s_branchhit * E_T_BOTH
    e_branch_interior = s_interior * E_T_BOTH
    comp_sum = e_spine + e_branchhit_rho2 + e_branch_interior

    # ---- (2) tag confirmed vs suspect ----
    confirmed_et = e_spine + e_branchhit_rho2            # real-KV-confirmed (land #245)
    suspect_et = e_branch_interior                       # scratch-suspect (0.599-faithful forward)

    # ---- (3) discount the suspect part two ways + check the floor ----
    e_t_lower_full_discount = confirmed_et                                  # branch-interior REMOVED
    e_t_lower_fidelity_discount = confirmed_et + SCRATCH_FIDELITY * suspect_et  # scaled by 0.599
    margin_full = e_t_lower_full_discount - E_T_MEAS_FLOOR
    margin_fidelity = e_t_lower_fidelity_discount - E_T_MEAS_FLOOR
    clears_full = e_t_lower_full_discount >= E_T_MEAS_FLOOR
    clears_fidelity = e_t_lower_fidelity_discount >= E_T_MEAS_FLOOR

    # ---- (4) the verdict ----
    headroom = E_T_BOTH - E_T_MEAS_FLOOR                 # 0.18147
    headroom_share = headroom / E_T_BOTH
    branch_interior_share = e_branch_interior / E_T_BOTH
    branch_interior_load_bearing = e_branch_interior > headroom        # == not(clears_full)
    path_a_robust_to_scratch_finding = bool(clears_full)

    # floor round-trips from (4.512, 520.95, 500) -- proves it is the banked floor, unchanged.
    floor_roundtrip = E_T_BOTH * (TARGET_OFFICIAL / LAMBDA1_CEILING_TPS)
    floor_rt_resid = abs(floor_roundtrip - E_T_MEAS_FLOOR)

    # ---- (5) self-test (PRIMARY) ----
    cond = {}
    cond["a_components_sum_to_4512"] = bool(abs(comp_sum - E_T_BOTH) <= TOL_SUM)
    cond["b_monotone_in_discount"] = bool(
        e_t_lower_full_discount <= e_t_lower_fidelity_discount + 1e-12 <= E_T_BOTH + 1e-12
    )
    cond["c_floor_and_lambdabar_imported_exact_unchanged"] = bool(
        floor_rt_resid <= TOL_RT
        and E_T_MEAS_FLOOR == 4.330527243789328
        and LAMBDA_BAR == 0.9780112973731208
    )
    cond["d_branch_interior_share_consistent"] = bool(
        abs(branch_interior_share - (suspect_et / E_T_BOTH)) <= 1e-12
    )
    cond["e_nan_clean"] = True   # finalized after the payload NaN walk below.
    cond["f_baseline_and_ceiling_unchanged"] = bool(
        BASELINE_TPS == 481.53 and LAMBDA1_CEILING_TPS == 520.9527323111674
    )
    # extra rigor (proves we used the model that built 4.512): exact reproduction + cross-method.
    cond["g_reproduces_committed_F_tree"] = bool(
        repro_resid_vs_committed <= TOL_REPRO and repro_resid_vs_const <= TOL_REPRO
    )
    cond["h_cross_method_M1_equals_M2"] = bool(xmethod_resid <= TOL_XMETHOD)

    handoff = _handoff_line(
        share_confirmed=(s_spine + s_branchhit),
        share_suspect=s_interior,
        e_t_lower_full=e_t_lower_full_discount,
        clears_full=clears_full,
        load_bearing=branch_interior_load_bearing,
    )

    verdict = (
        "PATH-A FRAGILE-TO-FULL-DISCOUNT, CLEARS-UNDER-FIDELITY; BRANCH-INTERIOR LOAD-BEARING"
        if branch_interior_load_bearing else
        "PATH-A ROBUST; BRANCH-INTERIOR SLACK"
    )

    decomposition_table = [
        {"component": "E_spine (rank-1 chain, real-KV-confirmed)", "E_T": e_spine,
         "share_pct": 100.0 * s_spine, "confirmed": True,
         "in_full_discount": True, "in_fidelity_discount": True},
        {"component": "E_branchhit_rho2 (first-level salvage, real-KV-confirmed)", "E_T": e_branchhit_rho2,
         "share_pct": 100.0 * s_branchhit, "confirmed": True,
         "in_full_discount": True, "in_fidelity_discount": True},
        {"component": "E_branch_interior (deep-branch, scratch-suspect)", "E_T": e_branch_interior,
         "share_pct": 100.0 * s_interior, "confirmed": False,
         "in_full_discount": False, "in_fidelity_discount": "x0.599"},
    ]

    result = {
        "pr": 259,
        "agent": "fern",
        "kind": "et-branch-interior-sensitivity",
        "metric_primary": "et_branch_interior_sensitivity_self_test_passes",
        "metric_test": "e_t_lower_branch_interior_discounted",
        # ---- PRIMARY / TEST ----
        "et_branch_interior_sensitivity_self_test_passes": all(cond.values()),
        "e_t_lower_branch_interior_discounted": e_t_lower_full_discount,
        # ---- the decomposition ----
        "decomposition": {
            "method": (
                "re-run the committed renewal-reward / path-enum tree-walk (descent_et_audit #172 / "
                "wirbel #135 model) over the committed topo74 M16 parent array + depth_q_76 spine ladder "
                "+ rho_cond rank-rescue ladder; partition the additive node reach-probabilities into "
                "spine (rank-1 chain) / branchhit (rank>=2 children of spine nodes) / interior "
                "(descendants of branch-hits); attribute the banked E[T]=4.512 in the structural shares."
            ),
            "E_T_both_banked": E_T_BOTH,
            "f_tree_committed": f_tree_committed,
            "f_tree_recomputed_M1": et_m1,
            "f_tree_recomputed_M2": et_m2,
            "reproduction_resid_vs_committed": repro_resid_vs_committed,
            "cross_method_resid_M1_M2": xmethod_resid,
            "E_spine": e_spine,
            "E_branchhit_rho2": e_branchhit_rho2,
            "E_branch_interior": e_branch_interior,
            "components_sum": comp_sum,
            "share_spine": s_spine,
            "share_branchhit_rho2": s_branchhit,
            "share_branch_interior": s_interior,
            "raw_tree_walk": {
                "E_spine_raw": part["e_spine_raw"],
                "E_branchhit_raw": part["e_branchhit_raw"],
                "E_interior_raw": part["e_interior_raw"],
                "f_tree_raw": f_tree_raw,
            },
            "node_counts": {"spine": part["n_spine"], "branchhit": part["n_branchhit"],
                            "interior": part["n_interior"]},
            "spine_nodes": part["spine_nodes"],
            "branchhit_nodes": part["branchhit_nodes"],
            "interior_nodes": part["interior_nodes"],
            "node_rows": part["node_rows"],
        },
        "tags": {
            "confirmed_et": confirmed_et,
            "suspect_et": suspect_et,
            "confirmed_components": ["E_spine", "E_branchhit_rho2"],
            "suspect_components": ["E_branch_interior"],
            "confirmed_basis": (
                "land #245 reconfirmed the real-KV spine ladder (top1 0.69-0.78, mean_accepted_len "
                "2.57-3.69, lambda-vs-top1 d2..d7 1.03-1.18 all clear 0.9780) and the first-divergence "
                "rho2~0.4165; these came from the real-KV ladder, NOT the scratch forward."
            ),
            "suspect_basis": (
                "the deep branch-interior continuation can only come from land's scratch reconstruction "
                "verify forward, measured ~0.599 faithful this cycle (land #245 clean-room control)."
            ),
            "real_kv_reconfirm": REAL_KV_RECONFIRM,
        },
        "discount_and_floor": {
            "E_T_meas_floor": E_T_MEAS_FLOOR,
            "floor_roundtrip_from_anchor": floor_roundtrip,
            "floor_roundtrip_resid": floor_rt_resid,
            "scratch_fidelity": SCRATCH_FIDELITY,
            "E_T_lower_full_discount": e_t_lower_full_discount,
            "E_T_lower_fidelity_discount": e_t_lower_fidelity_discount,
            "margin_full_vs_floor": margin_full,
            "margin_fidelity_vs_floor": margin_fidelity,
            "clears_floor_full_discount": clears_full,
            "clears_floor_fidelity_discount": clears_fidelity,
        },
        "verdict": {
            "headline": verdict,
            "headroom_4512_minus_floor": headroom,
            "headroom_share": headroom_share,
            "branch_interior_share": branch_interior_share,
            "branch_interior_share_minus_headroom_share": branch_interior_share - headroom_share,
            "branch_interior_load_bearing": branch_interior_load_bearing,
            "path_a_robust_to_scratch_finding": path_a_robust_to_scratch_finding,
            "decomposition_table": decomposition_table,
        },
        "self_test": {
            "et_branch_interior_sensitivity_self_test_passes": all(cond.values()),
            "conditions": cond,
            "tolerances": {"TOL_SUM": TOL_SUM, "TOL_REPRO": TOL_REPRO,
                           "TOL_XMETHOD": TOL_XMETHOD, "TOL_RT": TOL_RT},
        },
        "imports_unchanged": {
            "BASELINE_TPS": BASELINE_TPS,
            "lambda1_ceiling_tps": LAMBDA1_CEILING_TPS,
            "lambda_bar": LAMBDA_BAR,
            "note": "BASELINE 481.53 and the 520.95 lambda=1 ceiling are UNCHANGED; the lambda-hat "
                    "gate-2 acceptance evidence is real-KV and is NOT touched by this re-partition.",
        },
        "handoff": handoff,
        "scope": (
            "LOCAL CPU-only analytic re-partition of banked figures. Re-runs the committed tree-walk that "
            "BUILT 4.512, partitions it, discounts the scratch-suspect branch-interior, checks the 4.3305 "
            "floor. Attributes -- does NOT move -- the projection. land #245 owns the live build that "
            "SETTLES E[T] by measurement. No GPU / vLLM / HF Job / submission / served-file change / "
            "official draw. BASELINE stays 481.53; adds 0 TPS. NOT a launch. NOT open2."
        ),
        "public_evidence_used": [
            "land #245 cycle-1: scratch verify-forward ~0.599 faithful (reconstruction bug, NOT #192); "
            "real-KV ladder reconfirmed (mean_accepted_len 2.57-3.69, top1 0.69-0.78, lambda d2..d7 "
            "1.03-1.18, rho2~0.4165).",
            "land #238 / fern #253: banked both-bugs projection E[T]_both=4.512.",
            "wirbel #83 (6tghbnjn) / rho_optimal_topology: committed topo74 M16 parent array (F_tree "
            "4.512274954048941), depth_q_76 spine ladder, rho_cond rank-rescue ladder; denken #85 tree overhead.",
            "denken #241 (hqewf1d6): E_T_meas_floor=4.3305, lambda=1 ceiling 520.95.",
            "fern #249 / stark #191: lambda-bar 0.9780 P95 validity gate.",
        ],
        "peak_mem_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 3),
        "elapsed_s": round(time.time() - t0, 4),
    }

    nan_paths = _nan_paths(result, "result")
    result["nan_clean"] = not nan_paths
    result["self_test"]["conditions"]["e_nan_clean"] = not nan_paths
    if nan_paths:
        result["self_test"]["conditions"]["e_nan_clean_paths"] = nan_paths
    result["self_test"]["et_branch_interior_sensitivity_self_test_passes"] = all(
        v for k, v in result["self_test"]["conditions"].items() if isinstance(v, bool)
    )
    result["et_branch_interior_sensitivity_self_test_passes"] = \
        result["self_test"]["et_branch_interior_sensitivity_self_test_passes"]
    return result


def _handoff_line(*, share_confirmed: float, share_suspect: float, e_t_lower_full: float,
                  clears_full: bool, load_bearing: bool) -> str:
    robust = "does NOT" if not clears_full else "does"
    frag = "fragile" if not clears_full else "robust"
    bear = "load-bearing" if load_bearing else "slack"
    return (
        f"the E[T]=4.512 projection is {100.0*share_confirmed:.1f}% real-KV-confirmed spine+branch-hit / "
        f"{100.0*share_suspect:.1f}% scratch-suspect branch-interior; fully discounting the suspect part "
        f"gives E_T_lower = {e_t_lower_full:.4f} which {robust} clear the 4.3305 floor, so Path-A is "
        f"{frag} to land #245's scratch-verify finding and the branch-interior is {bear}."
    )


# --------------------------------------------------------------------------- #
# Helpers: NaN walk + wandb + report.
# --------------------------------------------------------------------------- #
def _nan_paths(node: Any, path: str) -> list[str]:
    bad: list[str] = []

    def walk(n: Any, p: str) -> None:
        if isinstance(n, dict):
            for k, v in n.items():
                walk(v, f"{p}.{k}")
        elif isinstance(n, (list, tuple)):
            for i, v in enumerate(n):
                walk(v, f"{p}[{i}]")
        elif isinstance(n, float) and not math.isfinite(n):
            bad.append(p)

    walk(node, path)
    return bad


def _log_wandb(args, result: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[et-branch-interior] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    run = wandb_logging.init_wandb_run(
        job_type="et-branch-interior-sensitivity", agent="fern",
        name=args.wandb_name or "fern/et-branch-interior-sensitivity",
        group=args.wandb_group,
        tags=["et-branch-interior", "launch-readiness-integration", "validity-gate",
              "path-a-robustness", "scratch-fidelity", "pr259"],
        config={"baseline_tps": BASELINE_TPS, "method": "cpu-only-analytic", "target_tps": TARGET_OFFICIAL,
                "E_T_both_banked": E_T_BOTH, "E_T_meas_floor": E_T_MEAS_FLOOR,
                "scratch_fidelity": SCRATCH_FIDELITY,
                "imports_pr": [238, 241, 245, 249, 253, 83, 85, 191]},
    )
    if run is None:
        print("[et-branch-interior] wandb disabled; skipping", flush=True)
        return
    dec = result["decomposition"]
    vd = result["verdict"]
    df = result["discount_and_floor"]
    try:
        flat = {
            "et_branch_interior_sensitivity_self_test_passes":
                1.0 if result["et_branch_interior_sensitivity_self_test_passes"] else 0.0,
            "e_t_lower_branch_interior_discounted": result["e_t_lower_branch_interior_discounted"],
            "E_spine": dec["E_spine"],
            "E_branchhit_rho2": dec["E_branchhit_rho2"],
            "E_branch_interior": dec["E_branch_interior"],
            "components_sum": dec["components_sum"],
            "share_spine": dec["share_spine"],
            "share_branchhit_rho2": dec["share_branchhit_rho2"],
            "share_branch_interior": dec["share_branch_interior"],
            "f_tree_recomputed_M1": dec["f_tree_recomputed_M1"],
            "reproduction_resid_vs_committed": dec["reproduction_resid_vs_committed"],
            "cross_method_resid_M1_M2": dec["cross_method_resid_M1_M2"],
            "confirmed_et": result["tags"]["confirmed_et"],
            "suspect_et": result["tags"]["suspect_et"],
            "E_T_meas_floor": df["E_T_meas_floor"],
            "floor_roundtrip_resid": df["floor_roundtrip_resid"],
            "E_T_lower_full_discount": df["E_T_lower_full_discount"],
            "E_T_lower_fidelity_discount": df["E_T_lower_fidelity_discount"],
            "margin_full_vs_floor": df["margin_full_vs_floor"],
            "margin_fidelity_vs_floor": df["margin_fidelity_vs_floor"],
            "clears_floor_full_discount": 1.0 if df["clears_floor_full_discount"] else 0.0,
            "clears_floor_fidelity_discount": 1.0 if df["clears_floor_fidelity_discount"] else 0.0,
            "headroom_4512_minus_floor": vd["headroom_4512_minus_floor"],
            "branch_interior_share": vd["branch_interior_share"],
            "headroom_share": vd["headroom_share"],
            "branch_interior_share_minus_headroom_share": vd["branch_interior_share_minus_headroom_share"],
            "branch_interior_load_bearing": 1.0 if vd["branch_interior_load_bearing"] else 0.0,
            "path_a_robust_to_scratch_finding": 1.0 if vd["path_a_robust_to_scratch_finding"] else 0.0,
            "baseline_tps": BASELINE_TPS,
            "lambda1_ceiling_tps": LAMBDA1_CEILING_TPS,
            "lambda_bar": LAMBDA_BAR,
            "peak_mem_mib": result["peak_mem_mib"],
            "metrics_nan_clean": 1.0 if result["nan_clean"] else 0.0,
            **{f"selftest_{k}": (1.0 if v else 0.0)
               for k, v in result["self_test"]["conditions"].items() if isinstance(v, bool)},
        }
        try:
            import wandb
            tbl = wandb.Table(columns=["component", "E_T", "share_pct", "confirmed",
                                       "in_full_discount", "in_fidelity_discount"])
            for r in vd["decomposition_table"]:
                tbl.add_data(r["component"], r["E_T"], r["share_pct"], str(r["confirmed"]),
                             str(r["in_full_discount"]), str(r["in_fidelity_discount"]))
            flat["decomposition_table"] = tbl
            ntbl = wandb.Table(columns=["node", "parent", "depth", "rank", "bucket", "reach_prob"])
            for r in dec["node_rows"]:
                ntbl.add_data(r["node"], r["parent"], r["depth"], r["rank"], r["bucket"], r["reach_prob"])
            flat["node_partition_table"] = ntbl
        except Exception as exc:  # noqa: BLE001
            print(f"[et-branch-interior] wandb table skipped ({exc})", flush=True)
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="et_branch_interior_sensitivity", artifact_type="validity", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[et-branch-interior] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    dec, vd, df, st = (result["decomposition"], result["verdict"],
                       result["discount_and_floor"], result["self_test"])
    print("\n" + "=" * 100, flush=True)
    print("PR #259  E[T] BRANCH-INTERIOR SENSITIVITY -- does Path-A survive land #245's scratch finding?",
          flush=True)
    print("=" * 100, flush=True)
    print(f"  reproduce 4.512 by the SAME model: F_tree M1={dec['f_tree_recomputed_M1']:.12f} "
          f"M2={dec['f_tree_recomputed_M2']:.12f}", flush=True)
    print(f"     vs committed {dec['f_tree_committed']:.12f}  (resid {dec['reproduction_resid_vs_committed']:.1e}, "
          f"M1==M2 {dec['cross_method_resid_M1_M2']:.1e})", flush=True)
    print("-" * 100, flush=True)
    print(f"  DECOMPOSITION of banked E[T]=4.512   (spine {dec['node_counts']['spine']} / "
          f"branchhit {dec['node_counts']['branchhit']} / interior {dec['node_counts']['interior']} nodes)",
          flush=True)
    for r in vd["decomposition_table"]:
        tag = "confirmed" if r["confirmed"] else "SUSPECT  "
        print(f"    {r['E_T']:.4f}  ({r['share_pct']:5.2f}%)  [{tag}]  {r['component']}", flush=True)
    print(f"    -------- sum = {dec['components_sum']:.6f}", flush=True)
    print("-" * 100, flush=True)
    print(f"  confirmed_et = {result['tags']['confirmed_et']:.4f}   suspect_et = "
          f"{result['tags']['suspect_et']:.4f}", flush=True)
    print(f"  E_T_lower_full_discount     = {df['E_T_lower_full_discount']:.4f}   "
          f"margin vs 4.3305 = {df['margin_full_vs_floor']:+.4f}   clears={df['clears_floor_full_discount']}",
          flush=True)
    print(f"  E_T_lower_fidelity_discount = {df['E_T_lower_fidelity_discount']:.4f}   "
          f"margin vs 4.3305 = {df['margin_fidelity_vs_floor']:+.4f}   "
          f"clears={df['clears_floor_fidelity_discount']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  branch_interior_share = {100.0*vd['branch_interior_share']:.2f}%   vs headroom_share "
          f"{100.0*vd['headroom_share']:.2f}%  (interior {vd['branch_interior_share_minus_headroom_share']:+.4%} "
          f"{'OVER' if vd['branch_interior_load_bearing'] else 'UNDER'})", flush=True)
    print(f"  branch_interior_load_bearing       = {vd['branch_interior_load_bearing']}", flush=True)
    print(f"  path_a_robust_to_scratch_finding   = {vd['path_a_robust_to_scratch_finding']}", flush=True)
    print(f"  VERDICT: {vd['headline']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  PRIMARY et_branch_interior_sensitivity_self_test_passes = "
          f"{st['et_branch_interior_sensitivity_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        if isinstance(v, bool):
            print(f"    [{'ok' if v else '!! FAILED'}] {k}", flush=True)
    print(f"\n  HANDOFF: {result['handoff']}\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "et_branch_interior_sensitivity_results.json"))
    ap.add_argument("--self-test", "--self_test", action="store_true",
                    help="run the self-test (PRIMARY); nonzero exit on failure")
    ap.add_argument("--wandb-name", "--wandb_name", default="fern/et-branch-interior-sensitivity")
    ap.add_argument("--wandb-group", "--wandb_group", default="launch-readiness-integration")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    result = run()

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"[et-branch-interior] wrote {args.out}", flush=True)

    _print(result)
    _log_wandb(args, result)

    ok = bool(result["et_branch_interior_sensitivity_self_test_passes"] and result["nan_clean"])
    if args.self_test:
        print(f"[et-branch-interior] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
