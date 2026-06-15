#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #262 (fern) -- Fidelity-safe shallow tree: does spine + first-level rho2 (NO deep
branch-interior) reach 500 at ITS OWN cheaper verify step?  LOCAL CPU-only analytic
composition of banked figures. NOT a launch. NOT open2. NO GPU / served-file change.

THE QUESTION
------------
land #245 hit a verify-fidelity wall on the DEEP branch-interior of the depth-9/branch-3
topo74 tree (scratch-reconstruction reproduces the deployed verifier argmax only ~0.599
faithfully). My own #259 proved that wall is LOAD-BEARING: the E[T]=4.512 projection
decomposes (exact additive partition of the committed tree-walk) into

    95.3% real-KV-CONFIRMED  = E_spine 4.0244 + first-level rho2 branch-hit 0.2760
     4.7% scratch-SUSPECT    = deep branch-interior 0.2117

and fully discounting the 4.7% drops E[T]_lower to confirmed_et = 4.3003, which MISSES the
4.3305 floor by -0.0302.  BUT that -0.0302 miss was computed against the FULL 16-node tree's
1.084953 ms BUILT step.

CRUX: if we DELIBERATELY BUILD ONLY the fidelity-safe sub-tree (spine + first-level rho2
branch-hit, dropping the entire scratch-suspect deep branch-interior, nodes [6,9,10]), we get
E[T] = confirmed_et = 4.3003 -- but on a SMALLER, CHEAPER tree (13 nodes, not 16). A cheaper
verify step LOWERS the E_T floor below 4.3305. DOES the fidelity-safe shallow tree clear 500 at
ITS OWN step -- sidestepping land #245's fidelity wall ENTIRELY?  This is "Path-A-lite": no deep
branch reconstruction, every accepted token real-KV-confirmed and greedy-identity-safe by
construction.

THE METHOD (analytic re-pricing of banked figures -- it ROUND-TRIPS)
--------------------------------------------------------------------
  1. SHALLOW TREE = topo74 parent array with the #259 scratch-suspect interior nodes [6,9,10]
     PRUNED. Remaining = spine(9) + first-level rho2 branch-hit(4) = 13 nodes. Its E[T] (by
     #259's exact additive partition) = E_spine + E_branchhit_rho2 = confirmed_et = 4.3003.
     Max depth and max branch width are UNCHANGED from the full tree (8 / 3) -- the only thing
     that drops is 3 interior verify ROWS (16 -> 13 candidate positions).

  2. step_shallow from the banked verify-step(M) curve (lawine #153) + the tree-overhead audit
     (denken #85 -- measured on the EXACT topo74 M16 parent array). In the M<=32 flat regime the
     verify-step marginal is PER-ROW (per-node) GEMM/sampler/argmax (denken #85: centroid_sampler
     ~M^0.75, verify_argmax ~M^0.94) while ATTENTION AMORTIZES (M8->M32 only 1.06x) -- so cost is
     ~linear in node-count, NOT depth-bound. We remove 3 per-node marginals from the banked built
     step. (BAND: if instead the verify cost were entirely fixed/depth-bound, step_shallow ==
     full step -> the #259 miss; that is the band's pessimistic upper end.)

  3. RE-DERIVE the E_T floor at step_shallow. The floor scales linearly with step:
     E_T_floor(step) = 500 * step / (K_cal * tau)  (invert official = K_cal*(E[T]/step)*tau at
     TPS=500). This round-trips the banked 4.3305 floor at the full step EXACTLY. Check
     confirmed_et = 4.3003 against E_T_floor(step_shallow) (NOT 4.3305).

IMPORTS (do NOT re-derive): #259 decomposition (E_spine, E_branchhit_rho2, E_branch_interior,
confirmed_et, the bucket node lists); the committed topo74 M16 parent array (wirbel #83 6tghbnjn);
the verify-step(M) curve (lawine #153 ma0qlpas) + tree-overhead audit (denken #85); the BUILT
full-tree step 1.084952540947906 ms and its E_T floor 4.330527243789328 + the 520.95 lambda=1
ceiling + K_cal=125.268 (denken #241 hqewf1d6 / #252); lambda-bar 0.9780112973731208 (fern #249).
The serve-layer tau tie-breaker band is kanna #260 (SECONDARY, additive, NOT load-bearing).
BASELINE stays 481.53; this leg adds 0 TPS.

PRIMARY metric  fidelity_safe_shallow_tree_self_test_passes
TEST    metric  shallow_tree_implied_tps  (= K_cal*(4.3003/step_shallow)*tau, central per-node step)

Run:
  cd target/ && CUDA_VISIBLE_DEVICES="" python \
    research/validity/fidelity_safe_shallow_tree/fidelity_safe_shallow_tree.py \
    --self-test --wandb_group launch-readiness-integration --wandb_name fern/fidelity-safe-shallow-tree
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
# Committed source files (advisor-branch content only; nothing re-derived).
# --------------------------------------------------------------------------- #
TOPO_JSON = os.path.join(REPO_ROOT, "research/spec_cost_model/rho_optimal_topology_results.json")
PR259_JSON = os.path.join(
    REPO_ROOT,
    "research/validity/et_branch_interior_sensitivity/et_branch_interior_sensitivity_results.json",
)
LAWINE_JSON = os.path.join(REPO_ROOT, "research/oracle_readout/verify_step_m_curve.json")

# ---- #259 decomposition (fern 1j099vrm) -- IMPORTED, not re-derived. -------- #
E_SPINE = 4.0243768358447065             # rank-1 greedy chain (real-KV-confirmed)
E_BRANCHHIT_RHO2 = 0.2759528587796363    # first-level rho2 salvage (real-KV-confirmed)
E_BRANCH_INTERIOR = 0.21167030537565695  # deep branch-interior (scratch-suspect, 0.599-faithful)
CONFIRMED_ET = 4.300329694624343         # = E_SPINE + E_BRANCHHIT_RHO2 (fidelity-safe E[T])
SUSPECT_NODES_259 = [6, 9, 10]           # #259 interior bucket -> PRUNED for the shallow tree
SPINE_NODES_259 = [0, 1, 4, 7, 11, 12, 13, 14, 15]
BRANCHHIT_NODES_259 = [2, 3, 5, 8]

# ---- composition anchors (denken #241 hqewf1d6 / #252) -- IMPORTED unchanged. #
E_T_BOTH = 4.512                          # full both-bugs projection (land #238/#245, fern #253)
STEP_BUILT_FULL_MS = 1.084952540947906    # built (tree-decode) step of the FULL 16-node tree
E_T_MEAS_FLOOR_FULL = 4.330527243789328   # E[T] needed to clear 500 AT the full 1.085 step
LAMBDA1_CEILING_TPS = 520.9527323111674   # lambda=1 ceiling (E[T]=4.512 @ built step)
K_CAL = 125.26795005202914                # official = K_cal*(E[T]/step)*tau  (rounds to 125.268)
TAU = 1.0                                 # central composition tau (#252: tau in [0.9924,1.0])
LAMBDA_BAR = 0.9780112973731208           # fern #249 / stark #191 P95 acceptance gate (separate axis)
BASELINE_TPS = 481.53                     # official baseline (PR #52); UNCHANGED by this leg
TARGET_OFFICIAL = 500.0

F_TREE_TOPO74_M16_COMMITTED = 4.512274954048941  # wirbel #83 committed F_tree (structure proof)

# ---- serve-layer tau tie-breaker (kanna #260) -- SECONDARY, additive, NOT load-bearing. ----- #
SERVE_LEVER_CENTRAL_PCT = 0.0006163037678542111  # +0.0616% (= +0.30 TPS on 481.53)
SERVE_LEVER_LO_PCT = 0.00018489113035626333      # +0.0185%
SERVE_LEVER_HI_PCT = 0.001848911303562633        # +0.1849% (= +0.89 TPS on 481.53)

TOL_RT_FLOOR = 1e-6     # PR self-test (c): floor formula round-trips full-step floor.
TOL_RT_ET = 1e-9        # PR self-test (b): shallow E[T] round-trips #259 confirmed_et.
TOL_STRUCT = 0          # PR self-test (a): exact node-set / count equality.


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Tree structure: re-build children/depth + re-confirm the #259 bucket partition
# (rank-1 spine / rank>=2-of-spine branch-hit / deep interior) DIRECTLY from the
# committed parent array. This re-confirms the [6,9,10] exclusion WITHOUT
# re-deriving the imported E[T] component values.
# --------------------------------------------------------------------------- #
def build_children(parent: list[int]) -> tuple[list[list[int]], list[int]]:
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    depth = [0] * n
    for i in range(1, n):
        children[parent[i]].append(i)
    for i in range(1, n):
        depth[i] = depth[parent[i]] + 1
    return children, depth


def partition_buckets(parent, children) -> dict[str, list[int]]:
    """#259 bucketing: spine = rank-1 chain from root; branchhit = rank>=2 children OF spine
    nodes; interior = everything deeper inside a branch."""
    n = len(parent)
    bucket = ["interior"] * n
    bucket[0] = "spine"
    for u in range(n):
        for rank, c in enumerate(children[u], start=1):
            if bucket[u] == "spine" and rank == 1:
                bucket[c] = "spine"
            elif bucket[u] == "spine" and rank >= 2:
                bucket[c] = "branchhit"
            else:
                bucket[c] = "interior"
    return {
        "spine": [u for u in range(n) if bucket[u] == "spine"],
        "branchhit": [u for u in range(n) if bucket[u] == "branchhit"],
        "interior": [u for u in range(n) if bucket[u] == "interior"],
        "_bucket": bucket,
    }


def shallow_tree_shape(parent, depth, keep: set[int]) -> dict[str, Any]:
    """Prune to `keep`; report the surviving tree's node count, max depth, max branch width."""
    kept = sorted(keep)
    children_kept: dict[int, list[int]] = {u: [] for u in kept}
    for i in kept:
        p = parent[i]
        if p in keep:
            children_kept[p].append(i)
    max_depth = max(depth[u] for u in kept)
    max_branch = max(len(children_kept[u]) for u in kept)
    return {
        "kept_nodes": kept,
        "n_nodes": len(kept),
        "max_depth": max_depth,
        "max_branch_width": max_branch,
        "children": {u: children_kept[u] for u in kept},
    }


# --------------------------------------------------------------------------- #
# The verify-step(M) model (lawine #153 m8-basis curve + denken #85 corroboration).
#   step_norm_m8basis is the deployed-ms step where M=8 -> ~1.0 and M=32 -> 1.2182.
#   The FULL 16-node tree's BUILT step (1.084953) is the land-GO-read step (denken
#   #252), 1.92% above the lawine M=16 microbench point -- so we use the curve for the
#   RELATIVE per-node shape and ANCHOR it to the banked built step.
# --------------------------------------------------------------------------- #
def step_model(lawine: dict, n_full: int, n_shallow: int) -> dict[str, Any]:
    curve = lawine["curve"]
    s8 = float(curve["8"]["step_norm"])    # = step_norm_m8basis at M=8
    s16 = float(curve["16"]["step_norm"])  # = step_norm_m8basis at M=16
    s24 = float(curve["24"]["step_norm"])
    s32 = float(curve["32"]["step_norm"])

    # per-node marginal in the flat regime: the [8,16] segment slope (cost of nodes 9..16).
    marg_per_node_curve = (s16 - s8) / (16 - 8)         # deployed-ms / node
    n_drop = n_full - n_shallow                          # 16 - 13 = 3 verify rows removed

    # ABSOLUTE removal (physical: each pruned row removes its GEMM/sampler/argmax marginal).
    step_reduction_abs = n_drop * marg_per_node_curve
    step_shallow_abs = STEP_BUILT_FULL_MS - step_reduction_abs

    # FRACTIONAL cross-check (ratio of curve(13)/curve(16) applied to the built step).
    s13 = s8 + (n_shallow - 8) * marg_per_node_curve     # linear interp at M=13
    ratio_13_16 = s13 / s16
    step_shallow_frac = STEP_BUILT_FULL_MS * ratio_13_16

    step_shallow_central = step_shallow_abs              # primary (more conservative of the two)

    # convexity note: the curve accelerates (16->24 slope 0.0102/node > 8->16 slope 0.0070/node),
    # so the LOCAL-at-16 marginal is larger -> dropping the last 3 nodes could save MORE. Using the
    # [8,16] average is the CONSERVATIVE central estimate.
    marg_local_16_24 = (s24 - s16) / (24 - 16)
    step_shallow_convex = STEP_BUILT_FULL_MS - n_drop * marg_local_16_24

    # denken #85 corroboration that the marginal is PER-ROW (not depth/attention bound).
    return {
        "lawine_curve_m8basis": {"8": s8, "16": s16, "24": s24, "32": s32},
        "marginal_per_node_curve_ms": marg_per_node_curve,
        "n_full": n_full,
        "n_shallow": n_shallow,
        "n_drop": n_drop,
        "step_reduction_abs_ms": step_reduction_abs,
        "step_shallow_abs_ms": step_shallow_abs,
        "step_shallow_frac_ms": step_shallow_frac,
        "step_shallow_abs_minus_frac_ms": step_shallow_abs - step_shallow_frac,
        "step_shallow_central_ms": step_shallow_central,
        "step_shallow_convexaware_ms": step_shallow_convex,
        "marginal_local_16_24_ms": marg_local_16_24,
        "step_built_full_ms": STEP_BUILT_FULL_MS,
        # BAND: [most reduction (per-node), zero reduction (= full step)].
        "step_shallow_band_ms": [step_shallow_central, STEP_BUILT_FULL_MS],
        "step_reduction_pct_central": step_reduction_abs / STEP_BUILT_FULL_MS,
        "model": (
            "verify-step marginal is PER-ROW (per-node) in the M<=32 flat regime: lawine #153 "
            "step(M) grows via per-candidate GEMM/sampler/argmax rows, and denken #85 measured "
            "(on the EXACT topo74 M16 tree) centroid_sampler ~M^0.75 / verify_argmax ~M^0.94 while "
            "attention AMORTIZES (M8->M32 1.06x). The shallow tree has the SAME max-depth (8) and "
            "max-branch (3) as the full tree, so ALL the saving is the 3 fewer verify ROWS, none "
            "from depth. step_shallow = built_step - 3 * per-node-marginal (anchored to the banked "
            "built step; lawine curve supplies only the relative per-node shape)."
        ),
        "denken85_corroboration": {
            "verify_side_us_static_M8": 24.279760718345642,
            "verify_side_us_static_M16": 41.680380925536156,
            "verify_side_marginal_us_per_node_M8_M16": (41.680380925536156 - 24.279760718345642) / 8,
            "attn_M32_over_M8_ratio": 1.0584490540530653,
            "note": "verify-side non-GEMM overhead is per-row (+2.18us/node M8->M16); attention "
                    "amortizes (1.06x) -> cost is node-count-bound, not depth-bound.",
        },
    }


# --------------------------------------------------------------------------- #
# Floor inversion + TPS composition.
#   official = K_cal*(E[T]/step)*tau   ==>   E_T_floor(step) = TARGET*step/(K_cal*tau)
# --------------------------------------------------------------------------- #
def e_t_floor(step_ms: float) -> float:
    return TARGET_OFFICIAL * step_ms / (K_CAL * TAU)


def implied_tps(e_t: float, step_ms: float) -> float:
    return K_CAL * (e_t / step_ms) * TAU


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def run() -> dict[str, Any]:
    t0 = time.time()
    topo = _load_json(TOPO_JSON)
    parent = [int(x) for x in topo["per_budget"]["16"]["topo74"]["parent"]]
    f_tree_committed = float(topo["per_budget"]["16"]["topo74"]["F_tree"])
    lawine = _load_json(LAWINE_JSON)
    pr259 = _load_json(PR259_JSON)

    n_full = len(parent)
    children, depth = build_children(parent)

    # ---- (1) re-confirm the #259 partition from the parent array (structure only) ----
    buckets = partition_buckets(parent, children)
    spine_match = buckets["spine"] == SPINE_NODES_259
    branchhit_match = buckets["branchhit"] == BRANCHHIT_NODES_259
    interior_match = buckets["interior"] == SUSPECT_NODES_259

    keep = set(range(n_full)) - set(SUSPECT_NODES_259)
    shape = shallow_tree_shape(parent, depth, keep)
    n_shallow = shape["n_nodes"]
    pruned = sorted(set(range(n_full)) - keep)

    # ---- shallow E[T] = E_spine + E_branchhit_rho2 = confirmed_et (round-trip to #259) ----
    shallow_et = E_SPINE + E_BRANCHHIT_RHO2
    confirmed_et_259_banked = float(pr259["tags"]["confirmed_et"])
    et_rt_resid_vs_const = abs(shallow_et - CONFIRMED_ET)
    et_rt_resid_vs_259 = abs(shallow_et - confirmed_et_259_banked)

    # ---- (2) step_shallow from the banked verify-step(M) curve ----
    sm = step_model(lawine, n_full=n_full, n_shallow=n_shallow)
    step_shallow = sm["step_shallow_central_ms"]
    step_band = sm["step_shallow_band_ms"]

    # ---- (3) re-derive the E_T floor at step_shallow + check 4.3003 against IT ----
    floor_full_roundtrip = e_t_floor(STEP_BUILT_FULL_MS)         # must == 4.330527243789328
    floor_rt_resid = abs(floor_full_roundtrip - E_T_MEAS_FLOOR_FULL)
    e_t_floor_shallow = e_t_floor(step_shallow)                  # the floor at the CHEAPER step
    e_t_floor_shallow_upper = e_t_floor(STEP_BUILT_FULL_MS)      # band-upper floor (= 4.3305)

    shallow_tree_clears_500 = bool(shallow_et >= e_t_floor_shallow)
    margin_vs_floor = shallow_et - e_t_floor_shallow
    shallow_tps = implied_tps(shallow_et, step_shallow)          # TEST metric (central)
    shallow_tps_upper = implied_tps(shallow_et, STEP_BUILT_FULL_MS)  # band-pessimistic (#259 miss)

    # crossover robustness: how much step reduction is REQUIRED to clear 500 vs PREDICTED.
    step_threshold = shallow_et * K_CAL * TAU / TARGET_OFFICIAL  # step at which TPS == 500
    required_reduction_pct = (STEP_BUILT_FULL_MS - step_threshold) / STEP_BUILT_FULL_MS
    predicted_reduction_pct = sm["step_reduction_pct_central"]
    fixed_fraction_breakeven = (required_reduction_pct / predicted_reduction_pct
                                if predicted_reduction_pct > 0 else float("inf"))

    # ---- (4) orthogonal serve-layer tau tie-breaker (kanna #260) -- SECONDARY ----
    shallow_tps_serve_central = shallow_tps * (1.0 + SERVE_LEVER_CENTRAL_PCT)
    shallow_tps_serve_hi = shallow_tps * (1.0 + SERVE_LEVER_HI_PCT)

    # ---- (5) verdict table ----
    verdict_table = [
        {
            "tree": "full 16-node (land #245 path)", "fidelity": "RISK (scratch-suspect deep branch)",
            "nodes": n_full, "E_T": E_T_BOTH, "step_ms": STEP_BUILT_FULL_MS,
            "E_T_floor_at_step": e_t_floor(STEP_BUILT_FULL_MS),
            "clears_500": bool(E_T_BOTH >= e_t_floor(STEP_BUILT_FULL_MS)),
            "implied_TPS": implied_tps(E_T_BOTH, STEP_BUILT_FULL_MS),
        },
        {
            "tree": "fidelity-safe 13-node shallow (Path-A-lite)", "fidelity": "SAFE (all real-KV-confirmed)",
            "nodes": n_shallow, "E_T": shallow_et, "step_ms": step_shallow,
            "E_T_floor_at_step": e_t_floor_shallow,
            "clears_500": shallow_tree_clears_500,
            "implied_TPS": shallow_tps,
        },
    ]

    # ---- (6) self-test (PRIMARY) ----
    cond: dict[str, Any] = {}
    cond["a_13node_count_excludes_suspect_6_9_10"] = bool(
        n_shallow == 13
        and len(SPINE_NODES_259) == 9 and len(BRANCHHIT_NODES_259) == 4
        and pruned == SUSPECT_NODES_259
        and spine_match and branchhit_match and interior_match
    )
    cond["b_shallow_et_roundtrips_confirmed_et_259"] = bool(
        et_rt_resid_vs_const <= TOL_RT_ET and et_rt_resid_vs_259 <= TOL_RT_ET
    )
    cond["c_floor_formula_roundtrips_full_step_4p3305"] = bool(floor_rt_resid <= TOL_RT_FLOOR)
    cond["d_monotone_cheaper_step_lower_floor"] = bool(
        step_shallow <= STEP_BUILT_FULL_MS + 1e-15
        and e_t_floor_shallow <= E_T_MEAS_FLOOR_FULL + 1e-15
    )
    cond["e_nan_clean"] = True  # finalized after the NaN walk.
    cond["f_imports_exact_unchanged"] = bool(
        BASELINE_TPS == 481.53
        and LAMBDA1_CEILING_TPS == 520.9527323111674
        and round(K_CAL, 3) == 125.268
        and LAMBDA_BAR == 0.9780112973731208
        and E_T_MEAS_FLOOR_FULL == 4.330527243789328
        and CONFIRMED_ET == 4.300329694624343
    )
    # extra rigor: prove we point at the committed topo74 (structure provenance).
    cond["g_topo74_committed_structure"] = bool(
        abs(f_tree_committed - F_TREE_TOPO74_M16_COMMITTED) <= 1e-12 and n_full == 16
    )

    handoff = _handoff_line(
        step_shallow=step_shallow, e_t_floor_shallow=e_t_floor_shallow,
        clears=shallow_tree_clears_500, shallow_tps=shallow_tps,
    )

    verdict = (
        "FIDELITY-SAFE ROUTE TO 500 EXISTS (Path-A-lite clears at its own cheaper step)"
        if shallow_tree_clears_500 else
        "NO FIDELITY-SAFE ROUTE AT SHALLOW STEP (deep branch-interior still load-bearing)"
    )

    result = {
        "pr": 262,
        "agent": "fern",
        "kind": "fidelity-safe-shallow-tree",
        "metric_primary": "fidelity_safe_shallow_tree_self_test_passes",
        "metric_test": "shallow_tree_implied_tps",
        # ---- PRIMARY / TEST ----
        "fidelity_safe_shallow_tree_self_test_passes": all(
            v for v in cond.values() if isinstance(v, bool)),
        "shallow_tree_implied_tps": shallow_tps,
        "shallow_tree_clears_500": shallow_tree_clears_500,
        # ---- the shallow tree ----
        "shallow_tree": {
            "definition": (
                "topo74 M16 parent array with the #259 scratch-suspect interior nodes [6,9,10] "
                "PRUNED; remaining = spine(9) + first-level rho2 branch-hit(4) = 13 nodes."
            ),
            "pruned_nodes": pruned,
            "kept_nodes": shape["kept_nodes"],
            "n_nodes": n_shallow,
            "max_depth": shape["max_depth"],
            "max_branch_width": shape["max_branch_width"],
            "n_spine": len(SPINE_NODES_259),
            "n_branchhit": len(BRANCHHIT_NODES_259),
            "spine_nodes": SPINE_NODES_259,
            "branchhit_nodes": BRANCHHIT_NODES_259,
            "shallow_et": shallow_et,
            "shallow_et_eq_confirmed_et": True,
            "depth_width_unchanged_vs_full": bool(
                shape["max_depth"] == max(depth) and shape["max_branch_width"] == max(
                    len(children[u]) for u in range(n_full))),
            "partition_reconfirmed": bool(spine_match and branchhit_match and interior_match),
        },
        "et_roundtrip": {
            "shallow_et": shallow_et,
            "confirmed_et_imported": CONFIRMED_ET,
            "confirmed_et_259_banked": confirmed_et_259_banked,
            "resid_vs_imported": et_rt_resid_vs_const,
            "resid_vs_259_banked": et_rt_resid_vs_259,
            "E_spine": E_SPINE,
            "E_branchhit_rho2": E_BRANCHHIT_RHO2,
            "E_branch_interior_dropped": E_BRANCH_INTERIOR,
        },
        "step_model": sm,
        "floor_and_clears": {
            "law": "official = K_cal*(E[T]/step)*tau ; E_T_floor(step) = 500*step/(K_cal*tau)",
            "K_cal": K_CAL,
            "tau": TAU,
            "step_built_full_ms": STEP_BUILT_FULL_MS,
            "floor_full_roundtrip": floor_full_roundtrip,
            "floor_full_banked_241": E_T_MEAS_FLOOR_FULL,
            "floor_roundtrip_resid": floor_rt_resid,
            "step_shallow_ms": step_shallow,
            "step_shallow_band_ms": step_band,
            "E_T_floor_shallow": e_t_floor_shallow,
            "E_T_floor_shallow_band": [e_t_floor(step_band[0]), e_t_floor(step_band[1])],
            "confirmed_et": shallow_et,
            "shallow_tree_clears_500": shallow_tree_clears_500,
            "margin_vs_floor": margin_vs_floor,
            "shallow_tps_central": shallow_tps,
            "shallow_tps_band": [implied_tps(shallow_et, step_band[0]),
                                 implied_tps(shallow_et, step_band[1])],
            "shallow_tps_pessimistic_full_step": shallow_tps_upper,
            "crossover": {
                "step_threshold_clear500_ms": step_threshold,
                "required_step_reduction_pct": required_reduction_pct,
                "predicted_step_reduction_pct": predicted_reduction_pct,
                "fixed_fraction_breakeven": fixed_fraction_breakeven,
                "note": (
                    "the shallow tree needs only a {:.3%} step reduction to clear 500; the banked "
                    "per-node verify cost predicts {:.3%} from dropping 3 of 16 rows -- so it clears "
                    "unless > {:.1%} of the verify marginal is fixed/depth-bound (denken #85 refutes: "
                    "attention amortizes 1.06x, marginal is per-row).".format(
                        required_reduction_pct, predicted_reduction_pct,
                        1.0 - fixed_fraction_breakeven)
                ),
            },
        },
        "serve_lever_secondary": {
            "source": "kanna #260 greedy-safe serve-layer tau tie-breaker (additive, NOT load-bearing)",
            "serve_lever_central_pct": SERVE_LEVER_CENTRAL_PCT,
            "serve_lever_band_pct": [SERVE_LEVER_LO_PCT, SERVE_LEVER_HI_PCT],
            "shallow_tps_with_serve_lever_central": shallow_tps_serve_central,
            "shallow_tps_with_serve_lever_hi": shallow_tps_serve_hi,
            "note": "SECONDARY / tau-term -- does NOT carry the headline; the headline rests on the "
                    "real-KV-confirmed E[T]=4.3003 at the model-compute step_shallow.",
        },
        "verdict": {
            "headline": verdict,
            "shallow_tree_clears_500": shallow_tree_clears_500,
            "fidelity_safe_route_to_500_exists": shallow_tree_clears_500,
            "table": verdict_table,
            "one_line_read": (
                "a fidelity-SAFE route to 500 {} at the shallow tree's own cheaper step -- "
                "land #245's deep-branch reconstruction is {}.".format(
                    "EXISTS" if shallow_tree_clears_500 else "DOES NOT EXIST",
                    "OPTIONAL UPSIDE" if shallow_tree_clears_500 else "the ONLY tree route")
            ),
        },
        "self_test": {
            "fidelity_safe_shallow_tree_self_test_passes": all(
                v for v in cond.values() if isinstance(v, bool)),
            "conditions": cond,
            "tolerances": {"TOL_RT_FLOOR": TOL_RT_FLOOR, "TOL_RT_ET": TOL_RT_ET,
                           "TOL_STRUCT": TOL_STRUCT},
        },
        "imports_unchanged": {
            "BASELINE_TPS": BASELINE_TPS,
            "lambda1_ceiling_tps": LAMBDA1_CEILING_TPS,
            "K_cal": K_CAL,
            "tau": TAU,
            "lambda_bar": LAMBDA_BAR,
            "E_T_meas_floor_full": E_T_MEAS_FLOOR_FULL,
            "E_T_both": E_T_BOTH,
            "confirmed_et": CONFIRMED_ET,
            "note": "BASELINE 481.53, the 520.95 lambda=1 ceiling, K_cal=125.268, lambda-bar 0.9780, "
                    "the 4.3305 full-step floor and the #259 confirmed_et 4.3003 are IMPORTED EXACTLY "
                    "and UNCHANGED. This leg re-PRICES the banked projection at a cheaper step; it "
                    "moves no measurement. land #245 owns the live build.",
        },
        "handoff": handoff,
        "scope": (
            "LOCAL CPU-only analytic composition of banked figures. Prices whether the fidelity-SAFE "
            "13-node shallow sub-tree (spine + first-level rho2, dropping the scratch-suspect deep "
            "branch-interior) clears 500 at its OWN cheaper verify step -- by re-deriving the E_T "
            "floor at step_shallow and checking confirmed_et=4.3003 against IT. No GPU / vLLM / HF "
            "Job / submission / served-file change / official draw. BASELINE stays 481.53; adds 0 "
            "TPS. land #245 owns the live measurement. NOT a launch. NOT open2."
        ),
        "public_evidence_used": [
            "fern #259 (1j099vrm): E[T]=4.512 additive decomposition -> E_spine 4.0244 / "
            "E_branchhit_rho2 0.2760 / E_branch_interior 0.2117 (suspect nodes [6,9,10]); "
            "confirmed_et 4.3003 misses the 4.3305 floor by -0.0302 at the FULL-tree step.",
            "land #245 cycle-1: scratch verify-forward ~0.599 faithful on the deep branch-interior; "
            "real-KV spine + rho2 ladder reconfirmed.",
            "wirbel #83 (6tghbnjn): committed topo74 M16 parent array (F_tree 4.512274954048941).",
            "lawine #153 (ma0qlpas): verify-step(M) curve (KNEE_AT_32; per-row GEMM/sampler/argmax "
            "marginal in the flat M<=32 regime).",
            "denken #85: tree-overhead audit on the EXACT topo74 M16 tree (centroid_sampler ~M^0.75, "
            "verify_argmax ~M^0.94, attention amortizes 1.06x -> node-count-bound, not depth-bound).",
            "denken #241 (hqewf1d6) / #252: built step 1.084953 ms, 4.3305 floor, 520.95 ceiling, "
            "K_cal 125.268.",
            "fern #249 / stark #191: lambda-bar 0.9780 P95 validity gate.",
            "kanna #260: greedy-safe serve-layer tau tie-breaker +0.0616% central (band +0.0185..+0.1849%) "
            "-- SECONDARY additive lever.",
        ],
        "peak_mem_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 3),
        "elapsed_s": round(time.time() - t0, 4),
    }

    nan_paths = _nan_paths(result, "result")
    result["nan_clean"] = not nan_paths
    result["self_test"]["conditions"]["e_nan_clean"] = not nan_paths
    if nan_paths:
        result["self_test"]["conditions"]["e_nan_clean_paths"] = nan_paths
    passes = all(v for v in result["self_test"]["conditions"].values() if isinstance(v, bool))
    result["self_test"]["fidelity_safe_shallow_tree_self_test_passes"] = passes
    result["fidelity_safe_shallow_tree_self_test_passes"] = passes
    return result


def _handoff_line(*, step_shallow: float, e_t_floor_shallow: float, clears: bool,
                  shallow_tps: float) -> str:
    verb = "clears" if clears else "misses"
    upside = "optional upside" if clears else "the only tree route"
    exists = "exists" if clears else "does not exist"
    return (
        f"the fidelity-safe 13-node shallow tree (spine + first-level rho2, no scratch-suspect deep "
        f"branch-interior) has E[T]=4.3003 at step_shallow={step_shallow:.6f} ms, whose E_T floor is "
        f"{e_t_floor_shallow:.4f}, so it {verb} 500 at implied {shallow_tps:.2f} TPS -- meaning a "
        f"fidelity-RISK-FREE route to 500 {exists} and land #245's deep-branch reconstruction is "
        f"{upside}."
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
        print(f"[fidelity-shallow] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    run = wandb_logging.init_wandb_run(
        job_type="fidelity-safe-shallow-tree", agent="fern",
        name=args.wandb_name or "fern/fidelity-safe-shallow-tree",
        group=args.wandb_group,
        tags=["fidelity-safe-shallow-tree", "launch-readiness-integration", "validity-gate",
              "path-a-lite", "shallow-tree", "step-shallow", "pr262"],
        config={"baseline_tps": BASELINE_TPS, "method": "cpu-only-analytic",
                "target_tps": TARGET_OFFICIAL, "confirmed_et": CONFIRMED_ET,
                "step_built_full_ms": STEP_BUILT_FULL_MS, "K_cal": K_CAL, "tau": TAU,
                "imports_pr": [259, 245, 83, 153, 85, 241, 252, 249, 260]},
    )
    if run is None:
        print("[fidelity-shallow] wandb disabled; skipping", flush=True)
        return
    sm = result["step_model"]
    fc = result["floor_and_clears"]
    sv = result["serve_lever_secondary"]
    st = result["shallow_tree"]
    try:
        flat = {
            "fidelity_safe_shallow_tree_self_test_passes":
                1.0 if result["fidelity_safe_shallow_tree_self_test_passes"] else 0.0,
            "shallow_tree_implied_tps": result["shallow_tree_implied_tps"],
            "shallow_tree_clears_500": 1.0 if result["shallow_tree_clears_500"] else 0.0,
            "shallow_n_nodes": st["n_nodes"],
            "shallow_max_depth": st["max_depth"],
            "shallow_max_branch_width": st["max_branch_width"],
            "shallow_et": st["shallow_et"],
            "confirmed_et": CONFIRMED_ET,
            "et_roundtrip_resid_vs_259": result["et_roundtrip"]["resid_vs_259_banked"],
            "marginal_per_node_curve_ms": sm["marginal_per_node_curve_ms"],
            "n_drop": sm["n_drop"],
            "step_reduction_abs_ms": sm["step_reduction_abs_ms"],
            "step_reduction_pct_central": sm["step_reduction_pct_central"],
            "step_shallow_central_ms": sm["step_shallow_central_ms"],
            "step_shallow_frac_ms": sm["step_shallow_frac_ms"],
            "step_shallow_convexaware_ms": sm["step_shallow_convexaware_ms"],
            "step_built_full_ms": STEP_BUILT_FULL_MS,
            "step_shallow_band_lo_ms": fc["step_shallow_band_ms"][0],
            "step_shallow_band_hi_ms": fc["step_shallow_band_ms"][1],
            "floor_full_roundtrip": fc["floor_full_roundtrip"],
            "floor_roundtrip_resid": fc["floor_roundtrip_resid"],
            "E_T_floor_shallow": fc["E_T_floor_shallow"],
            "E_T_floor_shallow_band_lo": fc["E_T_floor_shallow_band"][0],
            "E_T_floor_shallow_band_hi": fc["E_T_floor_shallow_band"][1],
            "margin_vs_floor": fc["margin_vs_floor"],
            "shallow_tps_central": fc["shallow_tps_central"],
            "shallow_tps_band_lo": fc["shallow_tps_band"][0],
            "shallow_tps_band_hi": fc["shallow_tps_band"][1],
            "shallow_tps_pessimistic_full_step": fc["shallow_tps_pessimistic_full_step"],
            "step_threshold_clear500_ms": fc["crossover"]["step_threshold_clear500_ms"],
            "required_step_reduction_pct": fc["crossover"]["required_step_reduction_pct"],
            "predicted_step_reduction_pct": fc["crossover"]["predicted_step_reduction_pct"],
            "fixed_fraction_breakeven": fc["crossover"]["fixed_fraction_breakeven"],
            "shallow_tps_with_serve_lever_central": sv["shallow_tps_with_serve_lever_central"],
            "shallow_tps_with_serve_lever_hi": sv["shallow_tps_with_serve_lever_hi"],
            "baseline_tps": BASELINE_TPS,
            "lambda1_ceiling_tps": LAMBDA1_CEILING_TPS,
            "K_cal": K_CAL,
            "tau": TAU,
            "lambda_bar": LAMBDA_BAR,
            "peak_mem_mib": result["peak_mem_mib"],
            "metrics_nan_clean": 1.0 if result["nan_clean"] else 0.0,
            **{f"selftest_{k}": (1.0 if v else 0.0)
               for k, v in result["self_test"]["conditions"].items() if isinstance(v, bool)},
        }
        try:
            import wandb
            tbl = wandb.Table(columns=["tree", "fidelity", "nodes", "E_T", "step_ms",
                                       "E_T_floor_at_step", "clears_500", "implied_TPS"])
            for r in result["verdict"]["table"]:
                tbl.add_data(r["tree"], r["fidelity"], r["nodes"], r["E_T"], r["step_ms"],
                             r["E_T_floor_at_step"], str(r["clears_500"]), r["implied_TPS"])
            flat["verdict_table"] = tbl
        except Exception as exc:  # noqa: BLE001
            print(f"[fidelity-shallow] wandb table skipped ({exc})", flush=True)
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="fidelity_safe_shallow_tree", artifact_type="validity", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[fidelity-shallow] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    st, sm, fc, sv = (result["shallow_tree"], result["step_model"],
                      result["floor_and_clears"], result["serve_lever_secondary"])
    vd, stt = result["verdict"], result["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("PR #262  FIDELITY-SAFE SHALLOW TREE -- does spine+rho2 (no deep branch) reach 500 at its "
          "own step?", flush=True)
    print("=" * 100, flush=True)
    print(f"  SHALLOW TREE: prune suspect {st['pruned_nodes']} -> {st['n_nodes']} nodes "
          f"(spine {st['n_spine']} + branchhit {st['n_branchhit']})  "
          f"max_depth {st['max_depth']}  max_branch {st['max_branch_width']}", flush=True)
    print(f"     shallow E[T] = E_spine {E_SPINE:.4f} + E_branchhit_rho2 {E_BRANCHHIT_RHO2:.4f} = "
          f"{st['shallow_et']:.6f}  (== #259 confirmed_et, resid "
          f"{result['et_roundtrip']['resid_vs_259_banked']:.1e})", flush=True)
    print(f"     depth/width UNCHANGED vs full tree: {st['depth_width_unchanged_vs_full']}  "
          f"(all saving is 3 fewer verify ROWS)", flush=True)
    print("-" * 100, flush=True)
    print(f"  STEP MODEL (per-node, lawine #153 flat regime + denken #85):", flush=True)
    print(f"     per-node marginal {sm['marginal_per_node_curve_ms']*1000:.3f} us/node  x{sm['n_drop']} "
          f"dropped = -{sm['step_reduction_abs_ms']*1000:.2f} us "
          f"({sm['step_reduction_pct_central']:.3%})", flush=True)
    print(f"     step_built_full {STEP_BUILT_FULL_MS:.6f} ms  ->  step_shallow_central "
          f"{sm['step_shallow_central_ms']:.6f} ms   BAND [{fc['step_shallow_band_ms'][0]:.6f}, "
          f"{fc['step_shallow_band_ms'][1]:.6f}]", flush=True)
    print("-" * 100, flush=True)
    print(f"  FLOOR @ step (E_T_floor = 500*step/(K_cal*tau)):", flush=True)
    print(f"     round-trip full step -> {fc['floor_full_roundtrip']:.12f}  vs banked 4.330527243789328 "
          f"(resid {fc['floor_roundtrip_resid']:.1e})", flush=True)
    print(f"     E_T_floor_shallow = {fc['E_T_floor_shallow']:.6f}   confirmed_et = {st['shallow_et']:.6f}"
          f"   margin {fc['margin_vs_floor']:+.4f}", flush=True)
    print(f"     shallow_tree_clears_500 = {fc['shallow_tree_clears_500']}   implied TPS "
          f"{fc['shallow_tps_central']:.2f}   (band [{fc['shallow_tps_band'][0]:.2f}, "
          f"{fc['shallow_tps_band'][1]:.2f}], pessimistic full-step {fc['shallow_tps_pessimistic_full_step']:.2f})",
          flush=True)
    cx = fc["crossover"]
    print(f"     crossover: need {cx['required_step_reduction_pct']:.3%} reduction, predicted "
          f"{cx['predicted_step_reduction_pct']:.3%}  (clears unless >"
          f"{1.0-cx['fixed_fraction_breakeven']:.1%} of marginal is fixed/depth)", flush=True)
    print(f"  SERVE LEVER (secondary): shallow TPS x(1+kanna#260) -> central "
          f"{sv['shallow_tps_with_serve_lever_central']:.2f}  hi {sv['shallow_tps_with_serve_lever_hi']:.2f}",
          flush=True)
    print("-" * 100, flush=True)
    print("  VERDICT TABLE  {tree | nodes | E[T] | step ms | floor@step | clears | TPS}", flush=True)
    for r in vd["table"]:
        print(f"    {r['nodes']:>3}n  E[T]={r['E_T']:.4f}  step={r['step_ms']:.6f}  "
              f"floor={r['E_T_floor_at_step']:.4f}  clears={str(r['clears_500']):>5}  "
              f"TPS={r['implied_TPS']:7.2f}  | {r['tree']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  VERDICT: {vd['headline']}", flush=True)
    print(f"  PRIMARY fidelity_safe_shallow_tree_self_test_passes = "
          f"{stt['fidelity_safe_shallow_tree_self_test_passes']}", flush=True)
    for k, v in stt["conditions"].items():
        if isinstance(v, bool):
            print(f"    [{'ok' if v else '!! FAILED'}] {k}", flush=True)
    print(f"\n  HANDOFF: {result['handoff']}\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "fidelity_safe_shallow_tree_results.json"))
    ap.add_argument("--self-test", "--self_test", action="store_true",
                    help="run the self-test (PRIMARY); nonzero exit on failure")
    ap.add_argument("--wandb-name", "--wandb_name", default="fern/fidelity-safe-shallow-tree")
    ap.add_argument("--wandb-group", "--wandb_group", default="launch-readiness-integration")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    result = run()

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"[fidelity-shallow] wrote {args.out}", flush=True)

    _print(result)
    _log_wandb(args, result)

    ok = bool(result["fidelity_safe_shallow_tree_self_test_passes"] and result["nan_clean"])
    if args.self_test:
        print(f"[fidelity-shallow] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
