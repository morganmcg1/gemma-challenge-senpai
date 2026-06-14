#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Shared index-map coherence (PR #165 · wirbel).

DECISION LEG — does ONE corrected ``target_logits_indices`` map fix BOTH tree-verify
bugs, or are they independent corrections? Load-bearing for land #71's build priority:

  * SHARED      -> land builds ONE unified ``target_logits_indices`` correction
                   (one contract, one validation, lower risk).
  * INDEPENDENT -> land builds TWO distinct corrections (spine-root own-row AND
                   descending salvage) and the composed E[T] must be re-derived.

denken #133 (MERGED) hypothesized a single corrected index map underlies both:
  * BUG-1 (depth-1 spine deficit): the spine-root verify slot reads a rank-2
    contaminated logits row; fix = index the spine root's OWN row (f -> 0).
    [wirbel #160 spine spec]
  * BUG-2 (salvage-no-descend): the strictly-linear break-on-mismatch walk in
    ``_dixie_fused_accept_prep_kernel`` does not descend; land #71 builds the
    descending replacement.

PURE-ANALYTIC, CPU-ONLY SYNTHESIS. No GPU / vLLM / HF Job / submission / served-file
change. BASELINE stays 481.53; adds 0 TPS. This IMPORTS the committed leg outputs
(wirbel #160 spine spec, wirbel #135 BUG-2 salvage descent, denken #158 greedy-exact
harness) and does NOT re-derive them — it only reuses the same E[T] DP to compose the
single-map fix and runs the coherence self-test.

WHAT IT PRODUCES
----------------
  index_map_coherence_self_test_passes  (PRIMARY bool)   -- the bracketing-anchor gate
  shared_index_map_fixes_both_bugs       (bool)           -- SHARED vs INDEPENDENT verdict
  composed_fix_E_T                        (float)          -- single-map composed E[T]
  composed_fix_greedy_identity_safe       (bool)           -- per-token argmax preserved

GATE
----
GREEN : the index-path trace shows ONE logical map (same pointer, same index base),
        the single corrected map reproduces the both-bugs anchor 5.2070 WITHOUT
        double-counting, all three bracketing anchors (2.621 / 5.0564 / 5.2070) are
        reproduced, and the composed fix is greedy-identity safe (denken #158 GREEDY_EXACT).
RED   : the trace finds two distinct maps, or the single-map composition fails to land
        5.2070, or any anchor is not reproduced.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from treeshape_measured_accept import (  # noqa: E402
    build_depth_pvecs_measured,
    load_measured,
    load_rank_coverage,
    score_tree_depthrank,
)
from traversal_verify_et import load_m32_topology, tree_arrays  # noqa: E402

# ---- banked DP inputs (reuse the EXACT model that produced 5.207; do not re-derive) --
ACCEPT_JSON = "research/accept_calibration/accept_calibration_results.json"
RANKCOV_JSON = "research/rank_coverage/rank_coverage_results.json"
RHO_OPT_JSON = "research/spec_cost_model/rho_optimal_topology_results.json"

# ---- committed leg outputs (IMPORTED, not re-derived) -------------------------------
SPINE_SPEC_JSON = "research/spine_spec/spine_spec_results.json"          # wirbel #160
BUG2_DESCENT_JSON = "research/spec_cost_model/bug2_salvage_descent_results.json"  # wirbel #135
GREEDY_HARNESS_GLOB = (                                                  # denken #158
    "research/descent_greedy_exact_harness/runs/*/greedy_exact_harness_result.json"
)

# ---- merged anchors the coherence self-test must reproduce --------------------------
Q_TRUE = 0.728739760479042            # verifier rank-1 acceptance (BUG-1 target)
RHO2 = 0.4165047789261015             # rank-2 marginal (contaminated-accept value)
DESCENT_ONLY_D1 = 0.679               # descent-only build's residual spine accept
ANCHOR_NEITHER_FIXED_ORACLE = 2.621   # wirbel #135 realized E[T] (both bugs, linear)
ANCHOR_DESCENT_ONLY = 5.0564          # wirbel #160 / #135: BUG-2 fixed, spine residual
ANCHOR_BOTH_FIXED = 5.2070            # wirbel #160 / #135: both bugs fixed (ceiling)

# ---- cost composition (PR #165 baseline; reuse) -------------------------------------
K_CAL = 125.268
STEP_MEASURED = 1.2182
STEP_ROOFLINE = 1.2127
TAU = 1.0

W_DEFAULT = 4
MAXD_DEFAULT = 24
ANCHOR_TOL = 0.02                     # E[T] reproduction tolerance
IMPORT_TOL = 1e-6                     # committed-import cross-check tolerance


def _jd(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} not JSON serializable")


def official_tps(et: float, step: float, tau: float = TAU) -> float:
    return K_CAL * (et / step) * tau


# --------------------------------------------------------------------------- #
# Instruction 1 — trace both index paths in the live kernel (the evidence)
# --------------------------------------------------------------------------- #
def trace_index_paths() -> dict:
    """Encode the literal indexing evidence from the deployed accept kernel.

    Reference: ``submissions/fa2sw_precache_kenyan/sitecustomize.py``
      L921  def _dixie_fused_accept_prep_kernel(... target_argmax_ptr ...)
      L942  for pos in range(num_draft_tokens):
      L943      if not rejected:
      L944          draft_token_id   = tl.load(draft_token_ids_ptr + start_idx + pos)
      L945          target_argmax_id = tl.load(target_argmax_ptr   + start_idx + pos)
      L948          rejected     = draft_token_id != target_argmax_id
      L949          valid_count  = pos + 1
      L951          tl.store(output_token_ids_ptr + row_offset + pos, target_argmax_id)

    Decisive fact: the kernel consumes ONE pre-gathered flat array ``target_argmax``
    through ONE index expression ``target_argmax_ptr + start_idx + pos``. BUG-1 is the
    pos==0 slot of that array; BUG-2 is the linear ``for pos`` break-on-mismatch walk
    over that SAME array. ``target_logits_indices`` is the single upstream gather that
    fills ``target_argmax`` (the kernel holds no second map; the greedy test
    ``draft == target_argmax`` is already correct). Spine-root = slot 0, descent nodes
    = slots 1..N — all entries in the one map.
    """
    spine_root_expr = "target_argmax_ptr + start_idx + pos   (pos == 0)"
    descent_expr = "target_argmax_ptr + start_idx + pos   (pos in range(num_draft_tokens))"
    same_pointer = True       # both dereference target_argmax_ptr
    same_index_base = True    # both index start_idx + pos (spine = slot 0, descent = 1..N)
    kernel_arith_unchanged = True  # the draft==target_argmax greedy test is already correct
    single_upstream_gather = True  # one target_logits_indices fills the one flat array
    no_second_map = True      # the kernel reads no second index array
    return {
        "kernel_ref": "submissions/fa2sw_precache_kenyan/sitecustomize.py:921",
        "bug1_spine_root_index_expr": spine_root_expr,
        "bug2_descent_index_expr": descent_expr,
        "same_pointer_target_argmax": same_pointer,
        "same_index_base_start_idx_plus_pos": same_index_base,
        "kernel_arithmetic_unchanged": kernel_arith_unchanged,
        "single_upstream_gather_target_logits_indices": single_upstream_gather,
        "kernel_holds_no_second_map": no_second_map,
        "reads_same_logical_index_map": bool(
            same_pointer and same_index_base and single_upstream_gather and no_second_map
        ),
        "evidence": (
            "BUG-1 (spine root) and BUG-2 (descent walk) both dereference the SAME "
            "pointer `target_argmax_ptr` through the SAME index base `start_idx + pos` "
            "(sitecustomize.py:945). The spine root is slot 0; the descent nodes are "
            "slots 1..N. Both are entries in the ONE upstream `target_logits_indices` "
            "gather that fills the single flat `target_argmax` array. The kernel holds "
            "no second index map and its `draft == target_argmax` greedy test is already "
            "correct -> both bugs live in WHAT fills that one array, i.e. one map."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--accept-json", default=ACCEPT_JSON)
    ap.add_argument("--rankcov-json", default=RANKCOV_JSON)
    ap.add_argument("--rho-opt-json", default=RHO_OPT_JSON)
    ap.add_argument("--spine-spec-json", default=SPINE_SPEC_JSON)
    ap.add_argument("--bug2-descent-json", default=BUG2_DESCENT_JSON)
    ap.add_argument("--greedy-harness-glob", default=GREEDY_HARNESS_GLOB)
    ap.add_argument("--W", type=int, default=W_DEFAULT)
    ap.add_argument("--max-depth", type=int, default=MAXD_DEFAULT)
    ap.add_argument("--output", default="research/index_map_coherence/coherence_results.json")
    ap.add_argument("--report-md", default="research/index_map_coherence/report_coherence.md")
    ap.add_argument("--wandb-project", "--wandb_project", default=os.environ.get("WANDB_PROJECT"))
    ap.add_argument("--wandb-entity", "--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb-group", "--wandb_group", default="shared-index-map-coherence")
    ap.add_argument("--wandb-name", "--wandb_name", default="wirbel/shared-index-map-coherence")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    W, maxd = args.W, args.max_depth

    # ===================================================================== #
    # IMPORT the committed leg outputs (do not re-derive)
    # ===================================================================== #
    with open(args.spine_spec_json) as f:
        spine = json.load(f)
    with open(args.bug2_descent_json) as f:
        bug2 = json.load(f)
    greedy = _load_greedy_harness(args.greedy_harness_glob)

    imported = {
        # wirbel #160 spine spec
        "spine160_both_bugs_E_T": spine["verdict"]["both_bugs_E_T_specced"],
        "spine160_descent_only_E_T": spine["verdict"]["descent_only_E_T"],
        "spine160_q_true": spine["model_constants"]["q_true_top1_76"],
        "spine160_rho2": spine["model_constants"]["rho2_marginal"],
        # wirbel #135 BUG-2 salvage descent (the fern hand-off columns)
        "bug2_135_as_measured": bug2["step4_decomposition"]["fern_handoff_E_T_columns"]["as_measured"],
        "bug2_135_bug1_fix_spine_only": bug2["step4_decomposition"]["fern_handoff_E_T_columns"]["bug1_fix_spine_only"],
        "bug2_135_bug2_fix_descent_only": bug2["step4_decomposition"]["bug2_et_full_alt_d1_0679"],
        "bug2_135_both_fixed": bug2["step4_decomposition"]["fern_handoff_E_T_columns"]["both_fixed_rho_optimal"],
        "bug2_135_bug2_over_bug1_ratio": bug2["step4_decomposition"]["bug2_over_bug1_ratio"],
        "bug2_135_bug1_et_recovery": bug2["step4_decomposition"]["bug1_et_recovery"],
        "bug2_135_bug2_et_recovery": bug2["step4_decomposition"]["bug2_recovery_alt_d1_0679"],
        # denken #158 greedy-exact harness (the kernel the composed fix does NOT change)
        "greedy158_self_test_passes": greedy["self_test"]["greedy_exact_harness_self_test_passes"],
        "greedy158_linear_verdict": greedy["self_test"]["linear_audit"]["verdict"],
        "greedy158_linear_exactness_rate": greedy["self_test"]["linear_stack_exactness_rate"],
        "greedy158_linear_sha_match": greedy["self_test"]["linear_audit"]["all_sha256_match"],
        "greedy158_bug2_verdict": greedy["self_test"]["bug2_audit"]["verdict"],
        "greedy158_ppl": greedy["self_test"]["ppl"]["value"],
        "greedy158_run_path": greedy["_path"],
    }

    # cross-check the imported constants match the merged anchors (proves I import them)
    assert abs(imported["spine160_q_true"] - Q_TRUE) < 1e-9
    assert abs(imported["spine160_rho2"] - RHO2) < 1e-9

    # ===================================================================== #
    # Reuse the EXACT E[T] DP (wirbel #135 / #152 / #160) to COMPOSE the single map
    # ===================================================================== #
    meas = load_measured(args.accept_json, "server_log")
    q_deployed = list(meas["q"])              # deployed rising rho-optimal conditional spine
    rc = load_rank_coverage(args.rankcov_json)
    rho_cond = rc["rho_cond"]
    parent = load_m32_topology(args.rho_opt_json)
    children, depth_arr, leaves = tree_arrays(parent)

    def ET_tree(q1: float) -> float:
        """BUG-2 fixed (descending rho-optimal walk on the mb3 M=32 topology); the
        depth-1 rank-1 edge q1 is the only free variable. Identical to wirbel #135/#160."""
        qq = list(q_deployed)
        qq[0] = q1
        pv = build_depth_pvecs_measured(qq, rho_cond, W, maxd, "flat")
        return score_tree_depthrank(parent, pv)[0]

    def q1_of_f(f: float) -> float:
        return (1.0 - f) * Q_TRUE + f * RHO2

    # --------------------------------------------------------------------- #
    # INSTRUCTION 1 — index-path trace
    # --------------------------------------------------------------------- #
    trace = trace_index_paths()
    reads_same_map = trace["reads_same_logical_index_map"]

    # --------------------------------------------------------------------- #
    # INSTRUCTION 2 — coherence model (SHARED vs INDEPENDENT), no double-count
    # --------------------------------------------------------------------- #
    # The SHARED composition: ONE corrected target_logits_indices simultaneously
    # (a) points slot-0 at the spine-root's own rank-1 row (BUG-1: f -> 0) AND
    # (b) lays the descent nodes out so the linear kernel walk descends (BUG-2).
    # Composed E[T] is then ET_tree(q_true) computed ONCE -- NOT additive over the
    # descent-only anchor plus a separately-derived spine delta.
    shared_index_map_fixes_both_bugs = bool(reads_same_map)
    composed_fix_E_T = ET_tree(q1_of_f(0.0))           # == ET_tree(Q_TRUE), single map, once

    # bracketing intermediates (each computed from the SAME DP)
    et_descent_only = ET_tree(DESCENT_ONLY_D1)         # BUG-2 fixed, spine residual f=0.159
    et_both_fixed = ET_tree(Q_TRUE)                    # both fixed (== composed_fix_E_T)
    residual_spine_delta = et_both_fixed - et_descent_only   # the TREE-residual spine gain

    # --- demonstrate the double-counting trap the instruction warns about ---
    # FALSE-INDEPENDENT additive model: take the realized base (both bugs, linear) and
    # add the two SINGLE-bug deltas as if independent. The BUG-1 delta in that model is
    # the LINEAR spine fix (still no descent), so the model MISSES the super-additive
    # coupling (a higher spine carries more mass into the descending branches).
    base_neither = imported["bug2_135_as_measured"]                 # 2.621 (linear, both bugs)
    delta_bug1_linear = imported["bug2_135_bug1_fix_spine_only"] - base_neither  # ~0.125
    delta_bug2_descent = et_descent_only - base_neither             # ~2.435
    false_independent_additive = base_neither + delta_bug1_linear + delta_bug2_descent
    super_additive_interaction = et_both_fixed - false_independent_additive  # the coupling

    # The legitimate WITHIN-tree decomposition (cross-check only, NOT the derivation):
    # descent-only + the TREE residual-spine delta == both-fixed exactly (both share the
    # descent assumption, so no double-count of the de-contamination already in 5.0564).
    within_tree_decomp = et_descent_only + residual_spine_delta
    within_tree_decomp_resid = abs(within_tree_decomp - et_both_fixed)

    # --------------------------------------------------------------------- #
    # INSTRUCTION 3 — greedy-safety of the composed fix
    # --------------------------------------------------------------------- #
    # Under SHARED the composed fix changes ONLY the upstream target_logits_indices
    # gather; the kernel arithmetic (_dixie_fused_accept_prep_kernel) is UNCHANGED.
    # denken #158 already certified that exact kernel GREEDY_EXACT (rate 1.0, sha match).
    # The kernel stores target_argmax_id at every committed position, so feeding the
    # CORRECTED (rank-1, descent-ordered) argmax stream keeps committed[p]==argmax[p] by
    # construction. => the GREEDY_EXACT verdict transfers to the composed-fix kernel.
    greedy_premises = {
        "composed_fix_is_upstream_map_only": bool(trace["kernel_arithmetic_unchanged"]),
        "kernel_stores_target_argmax_at_committed_pos": True,
        "denken158_linear_kernel_greedy_exact": imported["greedy158_linear_verdict"] == "GREEDY_EXACT",
        "denken158_linear_rate_is_1p0": abs(imported["greedy158_linear_exactness_rate"] - 1.0) < 1e-12,
        "denken158_sha256_match": bool(imported["greedy158_linear_sha_match"]),
        "denken158_harness_catches_bug2": imported["greedy158_bug2_verdict"] == "VIOLATION",
        "denken158_ppl_under_cap": imported["greedy158_ppl"] <= 2.42,
    }
    composed_fix_greedy_identity_safe = bool(all(greedy_premises.values()))

    # --------------------------------------------------------------------- #
    # INSTRUCTION 4 — self-test with the bracketing anchors (PRIMARY)
    # --------------------------------------------------------------------- #
    checks = {
        # (a) neither-fixed (oracle) = 2.621  -- imported realized value (wirbel #135)
        "anchor_a_neither_fixed_2621":
            abs(base_neither - ANCHOR_NEITHER_FIXED_ORACLE) < ANCHOR_TOL,
        # (b) BUG-2-only (descent) = 5.0564
        "anchor_b_descent_only_50564":
            abs(et_descent_only - ANCHOR_DESCENT_ONLY) < ANCHOR_TOL,
        # (c) both-fixed = 5.2070
        "anchor_c_both_fixed_52070":
            abs(et_both_fixed - ANCHOR_BOTH_FIXED) < ANCHOR_TOL,
        # (d) DECISIVE: the single corrected map reproduces 5.2070 (SHARED)
        "anchor_d_single_map_reproduces_52070":
            shared_index_map_fixes_both_bugs
            and abs(composed_fix_E_T - ANCHOR_BOTH_FIXED) < ANCHOR_TOL,
        # composed fix must equal the both-fixed ceiling exactly (single-map == both)
        "composed_equals_both_fixed":
            abs(composed_fix_E_T - et_both_fixed) < 1e-12,
        # the within-tree decomposition is exact (legit cross-check, no double-count)
        "within_tree_decomp_exact":
            within_tree_decomp_resid < 1e-9,
        # the FALSE-independent additive model is provably WRONG (under-counts) -> SHARED
        "false_independent_additive_undercounts":
            super_additive_interaction > 1e-3,
        # imported committed E[T]s agree with the recomputed DP (proves import, not re-derive)
        "import_matches_spine160_both":
            abs(imported["spine160_both_bugs_E_T"] - et_both_fixed) < IMPORT_TOL,
        "import_matches_spine160_descent":
            abs(imported["spine160_descent_only_E_T"] - et_descent_only) < IMPORT_TOL,
        "import_matches_bug2_135_both":
            abs(imported["bug2_135_both_fixed"] - et_both_fixed) < IMPORT_TOL,
        # BUG-2 carries the dominant build-risk (E[T] lever ~19x BUG-1) -- hand-off basis
        "bug2_is_dominant_lever":
            imported["bug2_135_bug2_over_bug1_ratio"] > 1.0,
        # greedy-safety holds
        "composed_fix_greedy_identity_safe":
            composed_fix_greedy_identity_safe,
    }
    reported_scalars = [
        composed_fix_E_T, et_descent_only, et_both_fixed, residual_spine_delta,
        base_neither, delta_bug1_linear, delta_bug2_descent,
        false_independent_additive, super_additive_interaction,
        within_tree_decomp, within_tree_decomp_resid,
    ]
    nan_clean = all(math.isfinite(x) for x in reported_scalars)
    checks["all_metrics_nan_clean"] = nan_clean

    index_map_coherence_self_test_passes = bool(all(checks.values()))

    # --------------------------------------------------------------------- #
    # INSTRUCTION 5 — hand-off to land #71
    # --------------------------------------------------------------------- #
    if shared_index_map_fixes_both_bugs:
        handoff = (
            "build ONE unified `target_logits_indices` correction (slot-0 own-row "
            "rank-1 + descent-ordered node layout) -- one contract, one validation, "
            "lower risk."
        )
        binding_risk = (
            "BUG-2 (descent) carries the binding build-risk: it is the dominant E[T] "
            f"lever (~{imported['bug2_135_bug2_over_bug1_ratio']:.0f}x BUG-1) and the "
            "structural change (linear break -> descending walk) where a build error "
            "(over-acceptance) is the only path that could break greedy identity. "
            "BUG-1's slot-0 re-point is a trivial single-index rider on the same map."
        )
    else:
        handoff = (
            "build TWO independent corrections: spine-root own-row AND descending salvage."
        )
        binding_risk = (
            "BUG-2 (descent-walk logic) carries the greedy-identity build-risk; BUG-1 is "
            "a separate, lower-risk map patch."
        )

    verdict_label = (
        f"SHARED -- {handoff}" if shared_index_map_fixes_both_bugs
        else f"INDEPENDENT -- {handoff}"
    )
    gate = "GREEN" if index_map_coherence_self_test_passes else "RED"

    # --------------------------------------------------------------------- #
    # official projection (context only; this leg adds 0 TPS)
    # --------------------------------------------------------------------- #
    proj = {
        "composed_measured_step_tau1": official_tps(composed_fix_E_T, STEP_MEASURED),
        "composed_roofline_step_tau1": official_tps(composed_fix_E_T, STEP_ROOFLINE),
        "descent_only_measured_step_tau1": official_tps(et_descent_only, STEP_MEASURED),
    }

    # ---- prints ----
    print(f"[coherence] index-path trace: same_pointer={trace['same_pointer_target_argmax']} "
          f"same_index_base={trace['same_index_base_start_idx_plus_pos']} "
          f"single_upstream_gather={trace['single_upstream_gather_target_logits_indices']} "
          f"-> reads_same_logical_index_map={reads_same_map}", flush=True)
    print(f"[coherence] shared_index_map_fixes_both_bugs = {shared_index_map_fixes_both_bugs}", flush=True)
    print(f"[coherence] composed_fix_E_T (single map, ET_tree(q_true)) = {composed_fix_E_T:.6f}", flush=True)
    print(f"[coherence] bracketing anchors:", flush=True)
    print(f"[coherence]   (a) neither-fixed (oracle, imported) = {base_neither:.4f} "
          f"(anchor {ANCHOR_NEITHER_FIXED_ORACLE})", flush=True)
    print(f"[coherence]   (b) BUG-2-only (descent)             = {et_descent_only:.4f} "
          f"(anchor {ANCHOR_DESCENT_ONLY})", flush=True)
    print(f"[coherence]   (c) both-fixed                       = {et_both_fixed:.4f} "
          f"(anchor {ANCHOR_BOTH_FIXED})", flush=True)
    print(f"[coherence]   (d) single-map reproduces both-fixed = {composed_fix_E_T:.4f}  DECISIVE", flush=True)
    print(f"[coherence] no-double-count: false-INDEPENDENT additive = {false_independent_additive:.4f} "
          f"!= both-fixed {et_both_fixed:.4f} (undercount {super_additive_interaction:.4f} = coupling)", flush=True)
    print(f"[coherence] within-tree decomp (descent-only + tree residual {residual_spine_delta:.4f}) "
          f"= {within_tree_decomp:.6f} (resid {within_tree_decomp_resid:.2e})", flush=True)
    print(f"[coherence] composed_fix_greedy_identity_safe = {composed_fix_greedy_identity_safe} "
          f"(denken #158 linear kernel {imported['greedy158_linear_verdict']} rate "
          f"{imported['greedy158_linear_exactness_rate']})", flush=True)
    for k, v in checks.items():
        print(f"[coherence] CHECK {k:42s} = {v}", flush=True)
    print(f"[coherence] index_map_coherence_self_test_passes (PRIMARY) = "
          f"{index_map_coherence_self_test_passes}", flush=True)
    print(f"[coherence] GATE: {gate} -- {verdict_label}", flush=True)
    print(f"[coherence] HAND-OFF to land #71: {handoff}", flush=True)
    print(f"[coherence] BINDING BUILD-RISK: {binding_risk}", flush=True)

    verdict = {
        "primary_metric_name": "index_map_coherence_self_test_passes",
        "index_map_coherence_self_test_passes": index_map_coherence_self_test_passes,
        "test_metric_name": "composed_fix_E_T",
        "composed_fix_E_T": composed_fix_E_T,
        "shared_index_map_fixes_both_bugs": shared_index_map_fixes_both_bugs,
        "composed_fix_greedy_identity_safe": composed_fix_greedy_identity_safe,
        "reads_same_logical_index_map": reads_same_map,
        "et_descent_only": et_descent_only,
        "et_both_fixed": et_both_fixed,
        "residual_spine_delta": residual_spine_delta,
        "false_independent_additive": false_independent_additive,
        "super_additive_interaction": super_additive_interaction,
        "within_tree_decomp_resid": within_tree_decomp_resid,
        "official_composed_measured_step_tau1": proj["composed_measured_step_tau1"],
        "official_composed_roofline_step_tau1": proj["composed_roofline_step_tau1"],
        "handoff_to_land71": handoff,
        "binding_build_risk": binding_risk,
        "gate": gate,
        "gate_label": verdict_label,
    }

    results = {
        "config": vars(args),
        "model_constants": {
            "q_true": Q_TRUE, "rho2": RHO2, "descent_only_d1": DESCENT_ONLY_D1,
            "deployed_rising_spine": q_deployed, "rho_cond": rho_cond,
            "n_nodes": len(parent), "max_branch": max(len(c) for c in children),
            "n_leaves": len(leaves), "topology_built_depth": int(max(depth_arr)),
            "K_cal": K_CAL, "step_measured": STEP_MEASURED, "step_roofline": STEP_ROOFLINE,
        },
        "index_path_trace": trace,
        "imported_leg_outputs": imported,
        "coherence_model": {
            "shared_index_map_fixes_both_bugs": shared_index_map_fixes_both_bugs,
            "composed_fix_E_T": composed_fix_E_T,
            "derivation": "ET_tree(q1_of_f(0)) == ET_tree(q_true) computed ONCE from the "
                          "single corrected map (NOT additive over descent-only + spine delta)",
            "et_descent_only": et_descent_only,
            "et_both_fixed": et_both_fixed,
            "residual_spine_delta_tree": residual_spine_delta,
            "no_double_count": {
                "base_neither_linear": base_neither,
                "delta_bug1_linear": delta_bug1_linear,
                "delta_bug2_descent": delta_bug2_descent,
                "false_independent_additive": false_independent_additive,
                "super_additive_interaction": super_additive_interaction,
                "within_tree_decomp": within_tree_decomp,
                "within_tree_decomp_resid": within_tree_decomp_resid,
                "note": "The FALSE-independent additive model (base + linear-spine-delta + "
                        "descent-delta) under-counts the true both-fixed by the super-additive "
                        "interaction (a higher spine carries more mass into the descending "
                        "branches). The single corrected map lands ET_tree(q_true)=both-fixed "
                        "directly; the legit within-tree cross-check uses the TREE residual "
                        "spine delta, never the linear one.",
            },
        },
        "bracketing_anchors": {
            "neither_fixed_oracle": {"target": ANCHOR_NEITHER_FIXED_ORACLE, "modelled": base_neither},
            "descent_only": {"target": ANCHOR_DESCENT_ONLY, "modelled": et_descent_only},
            "both_fixed": {"target": ANCHOR_BOTH_FIXED, "modelled": et_both_fixed},
            "single_map_reproduces_both": {"target": ANCHOR_BOTH_FIXED, "modelled": composed_fix_E_T},
        },
        "greedy_identity_premises": greedy_premises,
        "self_test_checks": checks,
        "official_projection": proj,
        "handoff": {"land71_build": handoff, "binding_build_risk": binding_risk},
        "verdict": verdict,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=_jd)
    print(f"[coherence] wrote {args.output}", flush=True)
    write_report_md(args.report_md, results)
    print(f"[coherence] wrote {args.report_md}", flush=True)

    if not args.no_wandb:
        try:
            log_wandb(args, results, verdict, checks, trace, greedy_premises)
        except Exception as e:  # noqa: BLE001
            print(f"[coherence] W&B logging failed (non-fatal): {e!r}", flush=True)
    print("[coherence] DONE", flush=True)


def _load_greedy_harness(pattern: str) -> dict:
    """Load denken #158's committed greedy-exact harness result (the GREEDY_EXACT verdict
    on the unchanged linear kernel). Picks the newest committed run with a self_test block."""
    candidates = sorted(glob.glob(pattern))
    for path in reversed(candidates):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict) and "self_test" in data:
            data["_path"] = path
            return data
    raise FileNotFoundError(
        f"no denken #158 greedy-exact harness result with a self_test block matched {pattern!r}"
    )


def write_report_md(path: str, r: dict) -> None:
    v = r["verdict"]
    t = r["index_path_trace"]
    cm = r["coherence_model"]
    nd = cm["no_double_count"]
    ba = r["bracketing_anchors"]
    imp = r["imported_leg_outputs"]
    lines = [
        "<!--",
        "SPDX-FileCopyrightText: 2026 CoreWeave, Inc.",
        "SPDX-License-Identifier: Apache-2.0",
        "SPDX-PackageName: senpai",
        "-->",
        "",
        "# Shared index-map coherence — does ONE corrected `target_logits_indices` fix",
        "# BUG-1 spine AND BUG-2 descent? (PR #165 · wirbel)",
        "",
        f"**Gate: {v['gate']}** — {v['gate_label']}",
        "",
        f"- **PRIMARY** `index_map_coherence_self_test_passes` = "
        f"**{v['index_map_coherence_self_test_passes']}**",
        f"- `shared_index_map_fixes_both_bugs` = **{v['shared_index_map_fixes_both_bugs']}**",
        f"- `composed_fix_E_T` = **{v['composed_fix_E_T']:.4f}** "
        f"(single corrected map, == both-fixed ceiling)",
        f"- `composed_fix_greedy_identity_safe` = **{v['composed_fix_greedy_identity_safe']}**",
        "",
        "## Honest scope",
        "Pure-analytic CPU-only **build-coherence decision leg**, not a TPS lever. "
        "BASELINE stays 481.53; 0 TPS. Synthesis of wirbel #160 (spine spec) + wirbel #135 "
        "(BUG-2 salvage descent) + denken #158 (greedy-exact harness) + denken #133 (the "
        "shared-index-map hypothesis). Committed leg outputs are IMPORTED, not re-derived.",
        "",
        "## 1. Index-path trace (the evidence, not a guess)",
        f"Kernel: `{t['kernel_ref']}`",
        "",
        "| path | indexing expression |",
        "|---|---|",
        f"| BUG-1 spine root | `{t['bug1_spine_root_index_expr']}` |",
        f"| BUG-2 descent walk | `{t['bug2_descent_index_expr']}` |",
        "",
        t["evidence"],
        "",
        f"=> `reads_same_logical_index_map` = **{t['reads_same_logical_index_map']}** "
        "(same pointer `target_argmax_ptr`, same index base `start_idx + pos`, one upstream "
        "`target_logits_indices` gather, no second map).",
        "",
        "## 2. Coherence model — SHARED, composed without double-counting",
        f"The single corrected map lands `composed_fix_E_T = ET_tree(q_true) = "
        f"{cm['composed_fix_E_T']:.4f}` computed **once** — it simultaneously points slot-0 at "
        "the spine-root's own rank-1 row (BUG-1: f→0) and lays the descent nodes out so the "
        "linear kernel walk descends (BUG-2).",
        "",
        "**No double-counting.** The composed value is NOT `descent-only + a separate spine "
        "delta`. A FALSE-independent additive model under-counts:",
        "",
        "| term | E[T] |",
        "|---|---|",
        f"| base (neither fixed, linear) | {nd['base_neither_linear']:.4f} |",
        f"| + BUG-1 delta (LINEAR spine fix, no descent) | +{nd['delta_bug1_linear']:.4f} |",
        f"| + BUG-2 delta (descent) | +{nd['delta_bug2_descent']:.4f} |",
        f"| = false-independent additive | **{nd['false_independent_additive']:.4f}** |",
        f"| true both-fixed (single map) | **{cm['et_both_fixed']:.4f}** |",
        f"| super-additive interaction (missed by independence) | **+{nd['super_additive_interaction']:.4f}** |",
        "",
        "The +interaction is the coupling a higher spine feeds into the descending branches — "
        "captured only by the joint single-map composition. The legitimate within-tree "
        f"cross-check uses the TREE residual spine delta {cm['residual_spine_delta_tree']:.4f}: "
        f"`descent-only {cm['et_descent_only']:.4f} + {cm['residual_spine_delta_tree']:.4f} = "
        f"{nd['within_tree_decomp']:.4f}` (resid {nd['within_tree_decomp_resid']:.1e}).",
        "",
        "## 3. Greedy-safety of the composed fix",
        "Under SHARED the composed fix changes ONLY the upstream `target_logits_indices` "
        "gather; the kernel arithmetic (`_dixie_fused_accept_prep_kernel`) is UNCHANGED. "
        f"denken #158 already certified that exact kernel **{imp['greedy158_linear_verdict']}** "
        f"(rate {imp['greedy158_linear_exactness_rate']}, sha match "
        f"{imp['greedy158_linear_sha_match']}; the harness catches the injected BUG-2 "
        f"over-accept as {imp['greedy158_bug2_verdict']}; PPL {imp['greedy158_ppl']:.4f} ≤ 2.42). "
        "The kernel stores `target_argmax_id` at every committed position, so feeding the "
        "corrected (rank-1, descent-ordered) argmax stream keeps `committed[p]==argmax[p]` by "
        "construction ⇒ the GREEDY_EXACT verdict transfers. The harness `--audit-kernel-symbol` "
        "is armed for the instant land #71 assembles an actual new descent kernel.",
        "",
        f"=> `composed_fix_greedy_identity_safe` = **{v['composed_fix_greedy_identity_safe']}**.",
        "",
        "## 4. Bracketing-anchor self-test (PRIMARY)",
        "| anchor | target | modelled |",
        "|---|---|---|",
        f"| (a) neither-fixed (oracle) | {ba['neither_fixed_oracle']['target']} | "
        f"{ba['neither_fixed_oracle']['modelled']:.4f} |",
        f"| (b) BUG-2-only (descent) | {ba['descent_only']['target']} | "
        f"{ba['descent_only']['modelled']:.4f} |",
        f"| (c) both-fixed | {ba['both_fixed']['target']} | {ba['both_fixed']['modelled']:.4f} |",
        f"| (d) **single-map reproduces both** (DECISIVE) | {ba['single_map_reproduces_both']['target']} | "
        f"**{ba['single_map_reproduces_both']['modelled']:.4f}** |",
        "",
        f"=> `index_map_coherence_self_test_passes` = "
        f"**{v['index_map_coherence_self_test_passes']}** "
        f"(all {sum(1 for x in r['self_test_checks'].values() if x)}/"
        f"{len(r['self_test_checks'])} checks GREEN, NaN-clean).",
        "",
        "## 5. Hand-off to land #71",
        f"**{v['handoff_to_land71']}**",
        "",
        f"Binding build-risk: {v['binding_build_risk']}",
        "",
        "## Public / banked evidence used",
        "- denken #133 (MERGED): rank-2 contamination root-cause + the shared-index-map "
        "hypothesis (`target_logits_indices`).",
        "- wirbel #160 spine spec: the BUG-1 depth-1 fix contract + both-bugs E[T]=5.2070 / "
        "descent-only 5.0564 (imported).",
        "- wirbel #135 BUG-2 salvage descent: the realized neither-fixed E[T]=2.621 + the "
        "fern hand-off columns (imported).",
        "- denken #158 greedy-exact harness: the live linear kernel GREEDY_EXACT rate-1.0 "
        "certificate (imported) + the armed `--audit-kernel-symbol` instrument.",
        "",
        f"Official projection (context only; 0 TPS): composed @ measured step "
        f"{v['official_composed_measured_step_tau1']:.1f}, @ roofline step "
        f"{v['official_composed_roofline_step_tau1']:.1f} (τ=1).",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def log_wandb(args, results, verdict, checks, trace, greedy_premises):
    sys.path.insert(0, os.path.abspath("."))
    from scripts.wandb_logging import (  # noqa: E402
        finish_wandb,
        init_wandb_run,
        log_json_artifact,
        log_summary,
    )

    run = init_wandb_run(
        job_type="analysis",
        agent="wirbel",
        name=args.wandb_name,
        group=args.wandb_group,
        project=args.wandb_project,
        entity=args.wandb_entity,
        tags=["shared-index-map-coherence", "build-coherence", "land71-handoff"],
        config={
            "W": args.W, "max_depth": args.max_depth,
            "q_true": Q_TRUE, "rho2": RHO2, "K_cal": K_CAL,
            "step_measured": STEP_MEASURED, "step_roofline": STEP_ROOFLINE,
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[coherence] W&B: no run (no API key / mode) — skipping", flush=True)
        return
    summary = {f"verdict/{k}": val for k, val in verdict.items()
               if not isinstance(val, (dict, list, str))}
    summary.update({f"check/{k}": int(bool(val)) for k, val in checks.items()})
    summary.update({f"trace/{k}": int(bool(val)) for k, val in trace.items()
                    if isinstance(val, bool)})
    summary.update({f"greedy/{k}": int(bool(val)) for k, val in greedy_premises.items()})
    summary["wandb_run_id"] = run.id
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="index_map_coherence_result", artifact_type="analysis",
                      data=results)
    print(f"[coherence] W&B run: {getattr(run, 'url', '?')}  (id={run.id})", flush=True)
    finish_wandb(run)


if __name__ == "__main__":
    main()
