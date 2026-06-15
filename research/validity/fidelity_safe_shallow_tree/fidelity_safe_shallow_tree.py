#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #262 (fern) -- Fidelity-safe shallow tree: does spine + first-level rho2 (NO deep
branch-interior) reach 500 at ITS OWN cheaper verify step?  LOCAL CPU-only analytic
composition of banked figures. NOT a launch. NOT open2. NO GPU / served-file change.

REVISION (advisor send-back -- re-ground the step basis on denken #257)
----------------------------------------------------------------------
The first pass headlined 506.27 TPS by dividing the shallow E[T]=4.3003 by step_shallow=
1.064 ms, derived as 1.085 - 0.021 off the ANALYTIC built-step anchor STEP_BUILT_FULL_MS=
1.084953 (denken #241). That 1.085 anchor was EMPIRICALLY RETIRED by denken #257 (eee4603,
merged): a forward-pass roofline measured the built tree step at 1.3458 ms CENTRAL (measured
g_d=0.0195, ~9x below the assumed 0.168), band [1.1186 (assumed-g_d optimistic), 1.4294
(measured-g_d depth9 pessimistic)]. The measured band sits ENTIRELY ABOVE the shallow tree's
break-even step (step_shallow must be <= 1.0774 ms / step_full <= ~1.098 ms to clear 500).
So this revision re-prices the SAME fidelity-safe tree at the GROUNDED step as a step-conditioned
BRACKET (not one inflated point), inverts the break-even (at 1.346 ms, what E[T] would clear
500?), and flags the near-zero lambda-hat validity headroom. Headline flips: under the grounded
central step the shallow tree implies ~406.6 TPS and MISSES 500; the required E[T]=5.288 exceeds
even the FULL tree's 4.512 by +0.776, so NO tree route (shallow OR land #245's deep branch) clears
500 at the grounded step. The structure / E[T] / floor-inversion machinery is UNCHANGED and still
self-tests 7/7; only the step BASIS is re-grounded.

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
TEST    metric  shallow_tree_implied_tps  (= K_cal*(4.3003/step_shallow_grounded_central)*tau, at the
                denken #257 GROUNDED central step 1.3458 - 0.0209 = 1.3249 ms -> ~406.6 TPS, MISSES 500)

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
LAMBDA_BAR = 0.9780112973731208           # fern #249 / stark #191 P95 ACHIEVED acceptance lambda-hat
LAMBDA_BAR_GATE = 0.9780                   # the P95 acceptance BAR (threshold the achieved lambda-hat must clear)
BASELINE_TPS = 481.53                     # official baseline (PR #52); UNCHANGED by this leg
TARGET_OFFICIAL = 500.0

# ---- denken #257 (eee4603) BUILT-STEP ROOFLINE GROUNDING -- RETIRES the 1.085 anchor. -------- #
# Forward-pass roofline measured the built tree step; g_d_measured=0.0195 is ~9x BELOW the assumed
# 0.168 the 1.085 analytic anchor bakes in. Imported EXACTLY from the committed report
#   research/validity/built_step_roofline/built_step_roofline_report.json  (advisor branch eee4603).
# These GROUND the step the E_T floor divides by. The advisor send-back requires re-pricing the
# shallow tree at THESE steps as a bracket, not at the retired 1.085.
STEP_BUILT_GROUNDED_CENTRAL_MS = 1.3458358727216921    # measured g_d=0.0195, b5 (the NEW central)
STEP_BUILT_GROUNDED_PESSIMISTIC_MS = 1.4294356405266744  # measured g_d, depth9 (band-high ~1.43)
STEP_BUILT_GROUNDED_OPTIMISTIC_MS = 1.1185888768817671   # assumed g_d=0.168, b5 (band-low ~1.12)
G_D_MEASURED_257 = 0.019498025961743392    # measured full-forward draft-overhead fraction
G_D_ASSUMED_FLEET = 0.168                  # the assumed g_d the retired 1.085 bakes in
# cross-check anchor: denken #257's OWN full-tree (E[T]=4.512) implied TPS at the grounded central
# step. My K_cal composition must reproduce this EXACTLY (machine precision) -> import provenance.
DENKEN257_FULLTREE_TPS_AT_CENTRAL = 419.96873622615647
DENKEN257_FULLTREE_CLEARS_500 = False      # report verdict: even the FULL tree misses at the grounded step

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

    # ---- (3') RE-GROUND the step basis on denken #257 (advisor send-back) ----
    # The 1.085 analytic anchor is RETIRED. Re-price the SAME shallow tree (E[T]=4.3003, same
    # per-node saving) at the GROUNDED built steps as a step-conditioned BRACKET. Each row applies
    # the SAME shallow saving (sm["step_reduction_abs_ms"], ~0.0209 ms) to its step_full basis.
    shallow_saving = sm["step_reduction_abs_ms"]

    def _bracket_row(label: str, step_full: float, basis: str, retired: bool = False) -> dict:
        step_sh = step_full - shallow_saving
        tps = implied_tps(shallow_et, step_sh)
        return {
            "label": label, "basis": basis, "retired_anchor": retired,
            "step_full_ms": step_full, "step_shallow_ms": step_sh,
            "E_T_floor_at_step_shallow": e_t_floor(step_sh),
            "margin_vs_floor": shallow_et - e_t_floor(step_sh),
            "implied_tps": tps, "clears_500": bool(tps >= TARGET_OFFICIAL),
        }

    grounded_bracket = [
        _bracket_row("retired_analytic_1p085", STEP_BUILT_FULL_MS,
                     "denken #241 analytic anchor -- RETIRED by #257", retired=True),
        _bracket_row("grounded_optimistic_1p12", STEP_BUILT_GROUNDED_OPTIMISTIC_MS,
                     "denken #257 assumed-g_d=0.168 b5 (band-low, most optimistic grounded)"),
        _bracket_row("grounded_central_1p346", STEP_BUILT_GROUNDED_CENTRAL_MS,
                     "denken #257 MEASURED g_d=0.0195 b5 (CENTRAL -- the new headline basis)"),
        _bracket_row("grounded_pessimistic_1p43", STEP_BUILT_GROUNDED_PESSIMISTIC_MS,
                     "denken #257 measured-g_d depth9 (band-high, pessimistic)"),
    ]
    # the new headline read: grounded CENTRAL row.
    central_row = next(r for r in grounded_bracket if r["label"] == "grounded_central_1p346")
    step_shallow_grounded_central = central_row["step_shallow_ms"]
    shallow_tps_grounded_central = central_row["implied_tps"]          # NEW TEST metric (was 506.27)
    shallow_clears_grounded_central = central_row["clears_500"]        # NEW headline boolean (False)
    grounded_any_clears = any(r["clears_500"] for r in grounded_bracket if not r["retired_anchor"])

    # ---- (3'') INVERT the break-even: at the grounded central step, what E[T] clears 500? ----
    # The shallow tree built at the grounded central full step runs at step_shallow_grounded_central.
    # E[T] needed to clear 500 there = E_T_floor(step_shallow_grounded_central). This is the single
    # most actionable number for land #245: how much MORE acceptance the real-path build must extract.
    e_t_needed_grounded_central = e_t_floor(step_shallow_grounded_central)         # at the shallow step
    e_t_needed_at_full_grounded_step = e_t_floor(STEP_BUILT_GROUNDED_CENTRAL_MS)   # at the full step (ref)
    delta_et_needed_vs_shallow = e_t_needed_grounded_central - shallow_et          # +0.988 over 4.3003
    delta_et_needed_vs_full = e_t_needed_grounded_central - E_T_BOTH               # +0.776 over 4.512
    # does even the FULL tree (E[T]=4.512, deep branch included) clear at the grounded central step?
    full_tree_tps_grounded_central = implied_tps(E_T_BOTH, STEP_BUILT_GROUNDED_CENTRAL_MS)
    full_tree_clears_grounded_central = bool(E_T_BOTH >= e_t_needed_at_full_grounded_step)
    # provenance: my full-tree composition must reproduce denken #257's own number EXACTLY.
    fulltree_257_crosscheck_resid = abs(full_tree_tps_grounded_central - DENKEN257_FULLTREE_TPS_AT_CENTRAL)
    # the deep branch-interior (#259) is worth only E_branch_interior; adding it back (4.3003 -> 4.512)
    # still leaves a shortfall to the required E[T] at the grounded step.
    shortfall_after_adding_deep_branch = e_t_needed_grounded_central - (shallow_et + E_BRANCH_INTERIOR)

    # ---- (3''') lambda-hat validity headroom flag ----
    lambda_hat_margin = LAMBDA_BAR - LAMBDA_BAR_GATE        # ~1.13e-5 -- essentially zero headroom
    lambda_hat_clears = bool(LAMBDA_BAR >= LAMBDA_BAR_GATE)

    # ---- (4) orthogonal serve-layer tau tie-breaker (kanna #260) -- SECONDARY ----
    # NOTE: now applied to the GROUNDED-central TPS (the honest basis), not the retired 506.27.
    shallow_tps_serve_central = shallow_tps_grounded_central * (1.0 + SERVE_LEVER_CENTRAL_PCT)
    shallow_tps_serve_hi = shallow_tps_grounded_central * (1.0 + SERVE_LEVER_HI_PCT)

    # ---- (5) verdict table -- step-conditioned bracket (full + shallow, retired vs grounded) ----
    verdict_table = [
        {
            "tree": "full 16-node @ RETIRED 1.085 step (land #245 path, old read)",
            "fidelity": "RISK (scratch-suspect deep branch)", "nodes": n_full, "E_T": E_T_BOTH,
            "step_ms": STEP_BUILT_FULL_MS, "E_T_floor_at_step": e_t_floor(STEP_BUILT_FULL_MS),
            "clears_500": bool(E_T_BOTH >= e_t_floor(STEP_BUILT_FULL_MS)),
            "implied_TPS": implied_tps(E_T_BOTH, STEP_BUILT_FULL_MS), "basis": "RETIRED",
        },
        {
            "tree": "full 16-node @ GROUNDED central 1.346 step (denken #257)",
            "fidelity": "RISK (scratch-suspect deep branch)", "nodes": n_full, "E_T": E_T_BOTH,
            "step_ms": STEP_BUILT_GROUNDED_CENTRAL_MS,
            "E_T_floor_at_step": e_t_floor(STEP_BUILT_GROUNDED_CENTRAL_MS),
            "clears_500": full_tree_clears_grounded_central,
            "implied_TPS": full_tree_tps_grounded_central, "basis": "GROUNDED",
        },
    ] + [
        {
            "tree": f"fidelity-safe 13-node shallow @ {r['label']}",
            "fidelity": "SAFE (all real-KV-confirmed)", "nodes": n_shallow, "E_T": shallow_et,
            "step_ms": r["step_shallow_ms"], "E_T_floor_at_step": r["E_T_floor_at_step_shallow"],
            "clears_500": r["clears_500"], "implied_TPS": r["implied_tps"],
            "basis": "RETIRED" if r["retired_anchor"] else "GROUNDED",
        }
        for r in grounded_bracket
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
    # ---- REVISION self-tests (the denken #257 re-grounding) ----
    # h: the grounded bracket is strictly monotone DECREASING in step_full (TPS ~ 1/step).
    grounded_only = [r for r in grounded_bracket if not r["retired_anchor"]]
    cond["h_grounded_bracket_monotone_in_step"] = bool(
        all(grounded_only[i]["step_full_ms"] < grounded_only[i + 1]["step_full_ms"]
            and grounded_only[i]["implied_tps"] > grounded_only[i + 1]["implied_tps"]
            for i in range(len(grounded_only) - 1))
    )
    # i: the inversion round-trips -- E[T]=E_T_needed at the grounded shallow step prices EXACTLY 500.
    inv_rt_resid = abs(implied_tps(e_t_needed_grounded_central, step_shallow_grounded_central)
                       - TARGET_OFFICIAL)
    cond["i_inversion_roundtrips_to_500"] = bool(inv_rt_resid <= TOL_RT_ET)
    # j: my full-tree composition REPRODUCES denken #257's own grounded full-tree TPS (419.97) EXACTLY
    #    -> the grounded step import is provenance-faithful (not a re-derivation).
    cond["j_fulltree_reproduces_denken257_419p97"] = bool(
        fulltree_257_crosscheck_resid <= TOL_RT_FLOOR)
    # k: the achieved lambda-hat clears the 0.9780 P95 bar but with ~zero (<1e-4) headroom (FLAG).
    cond["k_lambda_hat_clears_bar_zero_headroom"] = bool(
        lambda_hat_clears and 0.0 < lambda_hat_margin < 1e-4)

    handoff = _handoff_line(
        step_shallow=step_shallow_grounded_central,
        e_t_floor_shallow=e_t_needed_grounded_central,
        clears=shallow_clears_grounded_central, shallow_tps=shallow_tps_grounded_central,
        e_t_needed=e_t_needed_grounded_central, delta_vs_full=delta_et_needed_vs_full,
    )

    verdict = (
        "FIDELITY-SAFE ROUTE TO 500 EXISTS at the grounded step (clears)"
        if shallow_clears_grounded_central else
        "NO ROUTE TO 500 AT THE GROUNDED STEP -- shallow MISSES (~{:.1f} TPS) and even the FULL "
        "tree (E[T]=4.512) misses; clearing 500 needs E[T]={:.3f} (+{:.3f} over the full tree)".format(
            shallow_tps_grounded_central, e_t_needed_grounded_central, delta_et_needed_vs_full)
    )

    result = {
        "pr": 262,
        "agent": "fern",
        "kind": "fidelity-safe-shallow-tree",
        "metric_primary": "fidelity_safe_shallow_tree_self_test_passes",
        "metric_test": "shallow_tree_implied_tps",
        # ---- PRIMARY / TEST (TEST now on the denken #257 GROUNDED central basis) ----
        "fidelity_safe_shallow_tree_self_test_passes": all(
            v for v in cond.values() if isinstance(v, bool)),
        "shallow_tree_implied_tps": shallow_tps_grounded_central,
        "shallow_tree_clears_500": shallow_clears_grounded_central,
        "shallow_tree_implied_tps_retired_anchor": shallow_tps,  # the now-RETIRED 506.27 read (1.085 basis)
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
        "floor_and_clears_RETIRED_ANCHOR": {
            "BASIS_NOTE": "RETIRED: this block prices the shallow tree at the 1.085 analytic step "
                          "(denken #241), which denken #257 empirically retired. Kept for continuity / "
                          "audit of the first-pass 506.27 read. The LIVE headline is grounded_step_257.",
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
        # ============================ THE LIVE HEADLINE (denken #257 re-grounded) ============== #
        "grounded_step_257": {
            "BASIS_NOTE": "denken #257 (eee4603) forward-pass roofline RETIRED the 1.085 analytic step. "
                          "g_d_measured=0.0195 (~9x below the assumed 0.168). Re-prices the SAME shallow "
                          "tree (E[T]=4.3003, same -0.0209 ms per-node saving) at the GROUNDED built steps.",
            "g_d_measured_257": G_D_MEASURED_257,
            "g_d_assumed_fleet": G_D_ASSUMED_FLEET,
            "shallow_saving_ms": shallow_saving,
            "step_threshold_clear500_shallow_ms": step_threshold,         # 1.0774: step_shallow must be <= this
            "step_threshold_clear500_full_ms": step_threshold + shallow_saving,  # ~1.098: step_full must be <= this
            # ---- (1) the step-conditioned BRACKET (the advisor's requested deliverable) ----
            "bracket": grounded_bracket,
            "grounded_band_full_step_ms": [STEP_BUILT_GROUNDED_OPTIMISTIC_MS,
                                           STEP_BUILT_GROUNDED_PESSIMISTIC_MS],
            "grounded_central_full_step_ms": STEP_BUILT_GROUNDED_CENTRAL_MS,
            "shallow_tps_grounded_central": shallow_tps_grounded_central,
            "shallow_tps_grounded_band": [
                next(r["implied_tps"] for r in grounded_bracket if r["label"] == "grounded_optimistic_1p12"),
                next(r["implied_tps"] for r in grounded_bracket if r["label"] == "grounded_pessimistic_1p43"),
            ],
            "shallow_clears_grounded_central": shallow_clears_grounded_central,
            "grounded_any_clears": grounded_any_clears,
            # ---- (2) the INVERTED break-even (the most actionable number for land #245) ----
            "inversion": {
                "step_shallow_grounded_central_ms": step_shallow_grounded_central,
                "e_t_needed_to_clear500_at_grounded_shallow_step": e_t_needed_grounded_central,
                "e_t_needed_at_full_grounded_step": e_t_needed_at_full_grounded_step,
                "delta_et_needed_vs_shallow_4p300": delta_et_needed_vs_shallow,
                "delta_et_needed_vs_full_4p512": delta_et_needed_vs_full,
                "full_tree_tps_grounded_central": full_tree_tps_grounded_central,
                "full_tree_clears_grounded_central": full_tree_clears_grounded_central,
                "denken257_fulltree_tps_at_central": DENKEN257_FULLTREE_TPS_AT_CENTRAL,
                "fulltree_257_crosscheck_resid": fulltree_257_crosscheck_resid,
                "shortfall_after_adding_deep_branch_259": shortfall_after_adding_deep_branch,
                "note": (
                    "at the grounded central step the shallow tree needs E[T]={:.4f} to clear 500 "
                    "(+{:.4f} over the fidelity-safe 4.3003). That target EXCEEDS even the FULL tree's "
                    "E[T]=4.512 by +{:.4f}; the deep branch-interior (#259) is worth only +{:.4f}, so "
                    "adding it back still falls {:.4f} short -- NO tree route clears 500 at the grounded "
                    "step.".format(e_t_needed_grounded_central, delta_et_needed_vs_shallow,
                                   delta_et_needed_vs_full, E_BRANCH_INTERIOR,
                                   shortfall_after_adding_deep_branch)
                ),
            },
            # ---- (3) lambda-hat validity headroom FLAG ----
            "lambda_hat_flag": {
                "lambda_hat_achieved": LAMBDA_BAR,
                "lambda_bar_gate": LAMBDA_BAR_GATE,
                "lambda_hat_margin": lambda_hat_margin,
                "lambda_hat_clears": lambda_hat_clears,
                "note": (
                    "lambda-hat={:.10f} clears the {} P95 bar by only {:.2e} -- essentially ZERO "
                    "validity headroom. The E[T]={:.4f} the build must reach (point 2) must come from "
                    "genuinely better DRAFTING, NOT from relaxing acceptance: any E[T] bought by "
                    "accepting lower-prob draft tokens lowers lambda-hat below 0.9780 and BREAKS the "
                    "P95 validity gate.".format(LAMBDA_BAR, LAMBDA_BAR_GATE, lambda_hat_margin,
                                                e_t_needed_grounded_central)
                ),
            },
        },
        "serve_lever_secondary": {
            "source": "kanna #260 greedy-safe serve-layer tau tie-breaker (additive, NOT load-bearing)",
            "applied_to": "GROUNDED-central shallow TPS (the honest basis), NOT the retired 506.27",
            "serve_lever_central_pct": SERVE_LEVER_CENTRAL_PCT,
            "serve_lever_band_pct": [SERVE_LEVER_LO_PCT, SERVE_LEVER_HI_PCT],
            "shallow_tps_with_serve_lever_central": shallow_tps_serve_central,
            "shallow_tps_with_serve_lever_hi": shallow_tps_serve_hi,
            "note": "SECONDARY / tau-term -- does NOT carry the headline and does NOT rescue the miss "
                    "(+0.06% on ~406 TPS is ~+0.25 TPS). The grounded headline rests on E[T]=4.3003 at "
                    "the denken #257 grounded step.",
        },
        "verdict": {
            "headline": verdict,
            "shallow_tree_clears_500": shallow_clears_grounded_central,
            "fidelity_safe_route_to_500_exists_at_grounded_step": shallow_clears_grounded_central,
            "full_tree_clears_500_at_grounded_step": full_tree_clears_grounded_central,
            "table": verdict_table,
            "one_line_read": (
                "at denken #257's grounded central built step (1.346 ms) the fidelity-safe shallow tree "
                "implies ~{:.1f} TPS and MISSES 500; clearing 500 there needs E[T]={:.3f}, which exceeds "
                "even the FULL tree's 4.512 by +{:.3f} -- so NO tree route (shallow OR land #245's deep "
                "branch) reaches 500 at the grounded step. The first-pass 506.27 read was an artifact of "
                "the retired 1.085 anchor.".format(
                    shallow_tps_grounded_central, e_t_needed_grounded_central, delta_et_needed_vs_full)
                if not shallow_clears_grounded_central else
                "a fidelity-SAFE route to 500 EXISTS at the grounded step."
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
            "lambda_bar_gate": LAMBDA_BAR_GATE,
            "note": "BASELINE 481.53, the 520.95 lambda=1 ceiling, K_cal=125.268, lambda-hat 0.9780113, "
                    "the 4.3305 (1.085-step) floor and the #259 confirmed_et 4.3003 are IMPORTED EXACTLY "
                    "and UNCHANGED. This REVISION re-prices the banked projection at the GROUNDED (denken "
                    "#257) step -- which is HIGHER, not cheaper -- flipping the headline; it moves no "
                    "measurement. land #245 owns the live build.",
        },
        "handoff": handoff,
        "scope": (
            "LOCAL CPU-only analytic composition of banked figures. Re-prices whether the fidelity-SAFE "
            "13-node shallow sub-tree (spine + first-level rho2, dropping the scratch-suspect deep "
            "branch-interior) clears 500 at the GROUNDED built step (denken #257), as a step-conditioned "
            "bracket + inverted E[T] break-even + lambda-hat headroom flag. No GPU / vLLM / HF Job / "
            "submission / served-file change / official draw. BASELINE stays 481.53; adds 0 TPS. land "
            "#245 owns the live measurement. NOT a launch. NOT open2."
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
            "denken #241 (hqewf1d6) / #252: analytic built step 1.084953 ms, 4.3305 floor, 520.95 "
            "ceiling, K_cal 125.268 -- the 1.085 step RETIRED by #257.",
            "denken #257 (eee4603, MERGED): built-step roofline grounding -- measured g_d=0.0195 "
            "(~9x below assumed 0.168) -> built step 1.3458 ms central, band [1.1186, 1.4294]; the "
            "full E[T]=4.512 tree implies only 419.97 TPS at this step (clears_500=False). THIS LEG'S "
            "GROUNDED STEP BASIS.",
            "fern #249 / stark #191: lambda-hat 0.9780113 vs the 0.9780 P95 validity bar (margin ~1.1e-5).",
            "kanna #260: greedy-safe serve-layer tau tie-breaker +0.0616% central (band +0.0185..+0.1849%) "
            "-- SECONDARY additive lever (does not rescue the grounded miss).",
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
                  shallow_tps: float, e_t_needed: float, delta_vs_full: float) -> str:
    verb = "clears" if clears else "MISSES"
    exists = "exists" if clears else "does NOT exist"
    if clears:
        tail = "land #245's deep-branch reconstruction is optional upside."
    else:
        tail = (
            f"clearing 500 at this step needs E[T]={e_t_needed:.3f}, which exceeds even the FULL "
            f"16-node tree's E[T]=4.512 by +{delta_vs_full:.3f} -- so NEITHER the shallow tree NOR "
            f"land #245's deep-branch reconstruction reaches 500 at the grounded step."
        )
    return (
        f"on denken #257's GROUNDED central built step, the fidelity-safe 13-node shallow tree "
        f"(spine + first-level rho2) has E[T]=4.3003 at step_shallow={step_shallow:.6f} ms, whose "
        f"E_T floor is {e_t_floor_shallow:.4f}, so it {verb} 500 at implied {shallow_tps:.2f} TPS "
        f"-- a fidelity-RISK-FREE route to 500 {exists} at the grounded step; {tail}"
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
                "step_built_full_ms_RETIRED": STEP_BUILT_FULL_MS,
                "step_built_grounded_central_ms": STEP_BUILT_GROUNDED_CENTRAL_MS,
                "g_d_measured_257": G_D_MEASURED_257, "K_cal": K_CAL, "tau": TAU,
                "revision": "denken257-regrounding",
                "imports_pr": [259, 245, 83, 153, 85, 241, 252, 257, 249, 260]},
    )
    if run is None:
        print("[fidelity-shallow] wandb disabled; skipping", flush=True)
        return
    sm = result["step_model"]
    fc = result["floor_and_clears_RETIRED_ANCHOR"]
    gs = result["grounded_step_257"]
    inv = gs["inversion"]
    lf = gs["lambda_hat_flag"]
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
            # ---- GROUNDED (denken #257) -- the live headline metrics ----
            "g_d_measured_257": gs["g_d_measured_257"],
            "shallow_saving_ms": gs["shallow_saving_ms"],
            "step_built_grounded_central_ms": gs["grounded_central_full_step_ms"],
            "step_built_grounded_opt_ms": gs["grounded_band_full_step_ms"][0],
            "step_built_grounded_pess_ms": gs["grounded_band_full_step_ms"][1],
            "step_threshold_clear500_full_ms": gs["step_threshold_clear500_full_ms"],
            "shallow_tps_grounded_central": gs["shallow_tps_grounded_central"],
            "shallow_tps_grounded_opt": gs["shallow_tps_grounded_band"][0],
            "shallow_tps_grounded_pess": gs["shallow_tps_grounded_band"][1],
            "shallow_tps_retired_anchor": result["shallow_tree_implied_tps_retired_anchor"],
            "shallow_clears_grounded_central": 1.0 if gs["shallow_clears_grounded_central"] else 0.0,
            "grounded_any_clears": 1.0 if gs["grounded_any_clears"] else 0.0,
            "e_t_needed_grounded_central": inv["e_t_needed_to_clear500_at_grounded_shallow_step"],
            "delta_et_needed_vs_shallow": inv["delta_et_needed_vs_shallow_4p300"],
            "delta_et_needed_vs_full": inv["delta_et_needed_vs_full_4p512"],
            "full_tree_tps_grounded_central": inv["full_tree_tps_grounded_central"],
            "full_tree_clears_grounded_central": 1.0 if inv["full_tree_clears_grounded_central"] else 0.0,
            "fulltree_257_crosscheck_resid": inv["fulltree_257_crosscheck_resid"],
            "shortfall_after_adding_deep_branch": inv["shortfall_after_adding_deep_branch_259"],
            "lambda_hat_margin": lf["lambda_hat_margin"],
            "baseline_tps": BASELINE_TPS,
            "lambda1_ceiling_tps": LAMBDA1_CEILING_TPS,
            "K_cal": K_CAL,
            "tau": TAU,
            "lambda_bar": LAMBDA_BAR,
            "lambda_bar_gate": LAMBDA_BAR_GATE,
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
    st, sm, sv = result["shallow_tree"], result["step_model"], result["serve_lever_secondary"]
    fc = result["floor_and_clears_RETIRED_ANCHOR"]
    gs = result["grounded_step_257"]
    inv, lf = gs["inversion"], gs["lambda_hat_flag"]
    vd, stt = result["verdict"], result["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("PR #262 (REVISION)  FIDELITY-SAFE SHALLOW TREE -- re-priced at denken #257's GROUNDED step",
          flush=True)
    print("=" * 100, flush=True)
    print(f"  SHALLOW TREE: prune suspect {st['pruned_nodes']} -> {st['n_nodes']} nodes "
          f"(spine {st['n_spine']} + branchhit {st['n_branchhit']})  "
          f"max_depth {st['max_depth']}  max_branch {st['max_branch_width']}", flush=True)
    print(f"     shallow E[T] = E_spine {E_SPINE:.4f} + E_branchhit_rho2 {E_BRANCHHIT_RHO2:.4f} = "
          f"{st['shallow_et']:.6f}  (== #259 confirmed_et, resid "
          f"{result['et_roundtrip']['resid_vs_259_banked']:.1e});  per-node saving "
          f"{gs['shallow_saving_ms']*1000:.2f} us applied to each grounded step", flush=True)
    print("-" * 100, flush=True)
    print(f"  RETIRED ANCHOR (1.085 analytic): shallow step {fc['step_shallow_ms']:.6f} ms -> "
          f"{fc['shallow_tps_central']:.2f} TPS  [RETIRED by denken #257]", flush=True)
    print(f"  GROUNDED (denken #257, g_d_measured={gs['g_d_measured_257']:.4f} vs assumed "
          f"{gs['g_d_assumed_fleet']}):  step_full must be <= {gs['step_threshold_clear500_full_ms']:.4f} ms "
          f"to clear 500", flush=True)
    print("-" * 100, flush=True)
    print("  STEP-CONDITIONED BRACKET  {basis | step_full | step_shallow | E_T_floor | clears | TPS}",
          flush=True)
    for r in gs["bracket"]:
        tag = "RETIRED" if r["retired_anchor"] else "GROUNDED"
        print(f"    [{tag:>8}] {r['label']:<26} step_full={r['step_full_ms']:.6f}  "
              f"step_shallow={r['step_shallow_ms']:.6f}  floor={r['E_T_floor_at_step_shallow']:.4f}  "
              f"clears={str(r['clears_500']):>5}  TPS={r['implied_tps']:7.2f}", flush=True)
    print("-" * 100, flush=True)
    print(f"  INVERTED BREAK-EVEN @ grounded central (step_shallow={inv['step_shallow_grounded_central_ms']:.6f} ms):",
          flush=True)
    print(f"     E[T] needed to clear 500 = {inv['e_t_needed_to_clear500_at_grounded_shallow_step']:.4f}  "
          f"(+{inv['delta_et_needed_vs_shallow_4p300']:.4f} over shallow 4.3003, "
          f"+{inv['delta_et_needed_vs_full_4p512']:.4f} over FULL 4.512)", flush=True)
    print(f"     FULL tree (E[T]=4.512) @ grounded central -> {inv['full_tree_tps_grounded_central']:.2f} TPS "
          f"clears={inv['full_tree_clears_grounded_central']}  (== denken #257's 419.97, resid "
          f"{inv['fulltree_257_crosscheck_resid']:.1e})", flush=True)
    print(f"     adding the deep branch-interior (#259, +{E_BRANCH_INTERIOR:.4f}) still falls "
          f"{inv['shortfall_after_adding_deep_branch_259']:.4f} short -> NO tree route clears 500", flush=True)
    print(f"  LAMBDA-HAT FLAG: achieved {lf['lambda_hat_achieved']:.10f} vs bar {lf['lambda_bar_gate']} "
          f"-> margin {lf['lambda_hat_margin']:.2e} (~zero headroom; E[T] gain must NOT cost lambda)",
          flush=True)
    print(f"  SERVE LEVER (secondary, on grounded TPS): central "
          f"{sv['shallow_tps_with_serve_lever_central']:.2f}  hi {sv['shallow_tps_with_serve_lever_hi']:.2f} "
          f"(does NOT rescue the miss)", flush=True)
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
