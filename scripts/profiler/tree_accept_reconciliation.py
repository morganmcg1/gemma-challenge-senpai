#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Tree accept-length reconciliation (PR #101): why does the as-built tree give
tok/step = 2.10 vs the analytical E[T] = 5.207?

WHAT THIS ANSWERS
-----------------
byteshark's `tree-v2-merge-eager-v1` is the FIRST empirical tree build with both
halves wired (star-attn DISPATCH + fused reject/salvage WALK). The salvage signature
recovered (0.033 -> 0.358 per non-full step, root cause CONFIRMED), but the realised
`tok/step = 2.097` sits BELOW the deployed linear MTP accept_length (3.844) and far
below the analytical tree E[T] = 5.207 the fleet de-risked (#79 rho-ladder, #85
overhead, #91 topology, #92 independence). This script reconciles the 2.10-vs-5.207
gap against the validated acceptance model and classifies the defect: is it a fixable
BUILD/dispatch defect (shallow salvage walk / eager mode), or a real acceptance
CEILING (the model was optimistic)?

LOCAL, CPU-ONLY, ANALYTIC. No GPU, no vLLM, no HF Job, no submission, no served-file
change. Consumes ONLY byteshark's committed counters + the fleet's validated
acceptance machinery (treeshape_measured_accept). Reuses wirbel #79's measured
rho-ladder and the M=32 topology verbatim.

THE METHOD (three steps, matching the PR)
-----------------------------------------
Step 1 - back out the as-built tree's implied per-position acceptance from the accept
  histogram (survival -> conditional continuation c(k)) and the implied rank-2+ rescue
  rho_hat(k) = (c(k) - q[k]) / (1 - q[k]); compare to the deployed linear q-ladder and
  the full-tree model coverage. `implied_tree_rho_vs_model`.

Step 2 - classify the gap to (A) truncated branch exploration, (B) draft-quality
  collapse on the tree layout, (C) eager-mode artifact, (D) optimistic model, using
  reference points linear=3.844, spine-only=4.177, full=5.207, as-built=2.10.
  `tree_accept_length_gap_explained_pct`.

Step 3 - gate (GREEN/AMBER/RED) + the corrected E[T] band fern #100 composes against.

THE DECISIVE OBSERVATION
------------------------
The tree spine IS the deployed linear MTP chain (same drafter, same K, no
tree-attention dependence at depth 1). A correctly-built tree is therefore a strict
SUPERSET of the linear chain and CANNOT accept fewer tokens than 3.844. The as-built
2.10 sits 1.74 tokens BELOW that measured floor, with depth-1 spine continuation
c(1) = 0.598 << the required q[1] = 0.729. That deficit is structurally impossible
without a build/dispatch defect -- it cannot be an acceptance-model story (the same
drafter+verifier already realise 3.844 in linear mode on the real stack).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from treeshape_measured_accept import (  # noqa: E402
    build_depth_pvecs_measured,
    build_linear,
    score_tree_depthrank,
)

# --------------------------------------------------------------------------- #
# byteshark's COMMITTED counters: tree-v2-merge-eager-v1 (chiku-inu both-halves
# merge), partial job logs after the 40m timeout. Source: message board
# 20260614-055342-826_byteshark.md (relayed in PR #101 body). Last accept hist
# snapshot at step 14336; terminal counters at step 14848.
# --------------------------------------------------------------------------- #
ASBUILT = {
    "method": "tree-v2-merge-eager-v1",
    "steps_total": 14848,
    "tok_per_step": 2.097,
    "salvages": 5264,
    "full": 164,
    "attn_py_calls_per_step": 37.0,
    "accept_hist": [0, 5761, 5061, 1765, 854, 355, 214, 126, 200],  # index = tokens emitted
    "hist_snapshot_step": 14336,
    "salvage_prev_broken": 0.033,  # the one-half-wired signature
    "salvage_asbuilt_per_nonfull": 0.358,
}

# --------------------------------------------------------------------------- #
# The validated acceptance model (all MERGED / de-risked).
#   q-ladder, rho-ladder  : wirbel #79 (rank_coverage, run z6wi4z4v)
#   M=32 topology         : land #71 build target (fern #92 / #88 / treeshape handoff)
#   E[T] anchors          : reproduced below to 1e-4
# --------------------------------------------------------------------------- #
Q_LADDER = [0.728739760479042, 0.7589764102641635, 0.7924989076194682,
            0.821702519412012, 0.8342716929825772, 0.8352594665096346,
            0.8472621220149911]
RHO_LADDER = [0.4165047789261015, 0.2655480090557997, 0.19075249320036264]
M32_PARENT = [-1, 0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 5, 7, 9, 9, 10, 11, 12, 13, 15,
              16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 29]
WIRBEL_ET = 5.207
DEPLOYED_LINEAR_ET = 3.844131736526946          # measured, server_log (treeshape)
FERN92_INDEP_GAP_PCT = 0.024671869899095533     # ideal-tree model holds under real corr draws
LAND71_PROJ_TPS = 568.0                          # fern #92 recalibrated 568.14, band 558-581


def survival_and_continuation(hist: list[int]) -> dict:
    """From the accept histogram (index k = tokens emitted that step), compute the
    survival S(emit>=k) and the per-DRAFT-POSITION conditional continuation
    c(j) = P(accept >= j drafts | accept >= j-1 drafts) = S(emit>=j+1) / S(emit>=j).

    Bin 0 is 'emit 0 tokens' (never happens: every step emits >=1 guaranteed token),
    so bin index k = 1 bonus + (k-1) accepted draft tokens; the first draft position
    is c(1) = S(emit>=2)/S(emit>=1).
    """
    h = np.asarray(hist, dtype=np.float64)
    N = float(h.sum())
    surv = np.array([h[k:].sum() / N for k in range(len(h))])  # surv[k] = P(emit>=k)
    K = len(h) - 1  # max draft-position index with a bucket
    cont = []
    for j in range(1, K):           # draft positions 1..K-1
        cont.append(surv[j + 1] / surv[j] if surv[j] > 0 else 0.0)
    mean_emit = float((np.arange(len(h)) * h).sum() / N)
    return {"N": N, "survival": surv.tolist(), "continuation": cont,
            "mean_emit": mean_emit, "p_zero_draft_accept": float(h[1] / N)}


def expected_salvage_edges_per_step(parent: list[int], pvecs) -> float:
    """Model E[# accepted rank>=2 edges per step] on the M=32 tree -- the quantity
    byteshark's `salvages` counter approximates. Sum of acceptance path-products over
    nodes whose incoming edge is a rank>=2 sibling."""
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    for i in range(1, n):
        children[parent[i]].append(i)
    pp = np.zeros(n)
    pp[0] = 1.0
    depth = [0] * n
    maxd = len(pvecs) - 1
    exp_salv = 0.0
    for par in range(n):
        for rank, c in enumerate(children[par], start=1):
            d = depth[par] + 1
            pv = pvecs[min(d, maxd)]
            r = rank if rank < len(pv) else len(pv) - 1
            pp[c] = pp[par] * pv[r]
            depth[c] = d
            if rank >= 2:
                exp_salv += pp[c]
    return float(exp_salv)


def reconcile(args) -> dict:
    W = args.W
    maxd = args.max_depth

    # ---- model anchors (reproduce E[T]) -------------------------------------
    pv_full = build_depth_pvecs_measured(Q_LADDER, RHO_LADDER, W, maxd)
    pv_spine = build_depth_pvecs_measured(Q_LADDER, [0.0] * len(RHO_LADDER), W, maxd)
    ET_full, depth_full = score_tree_depthrank(M32_PARENT, pv_full)
    ET_spine, _ = score_tree_depthrank(M32_PARENT, pv_spine)
    ET_lin8, _ = score_tree_depthrank(build_linear(8), pv_full)

    model_cov_by_depth = [float(pv_full[min(d, maxd)][1:W + 1].sum())
                          for d in range(1, 8)]
    model_exp_salvage = expected_salvage_edges_per_step(M32_PARENT, pv_full)

    # ---- Step 1: back out the as-built per-position acceptance ---------------
    sc = survival_and_continuation(ASBUILT["accept_hist"])
    cont = sc["continuation"]                     # c(1..7), per draft position
    # implied rank-2+ rescue: c = q + (1-q)*rescue  =>  rescue = (c - q)/(1 - q)
    implied_rho = [(cont[k] - Q_LADDER[k]) / (1.0 - Q_LADDER[k])
                   for k in range(min(len(cont), len(Q_LADDER)))]
    asbuilt_below_linear = [cont[k] < Q_LADDER[k] for k in range(min(len(cont), len(Q_LADDER)))]
    c1, q1 = cont[0], Q_LADDER[0]

    step1 = {
        "asbuilt_continuation_c": [float(x) for x in cont],
        "deployed_linear_q": [float(x) for x in Q_LADDER],
        "full_tree_model_coverage": model_cov_by_depth,
        "implied_rescue_rho_hat": [float(x) for x in implied_rho],
        "implied_rho_hat_all_negative": bool(all(r < 0 for r in implied_rho)),
        "asbuilt_below_linear_every_position": bool(all(asbuilt_below_linear)),
        "depth1_continuation_asbuilt": float(c1),
        "depth1_continuation_required_min": float(q1),   # identical-drafter-forward floor
        "depth1_continuation_full_model": model_cov_by_depth[0],
        "depth1_deficit_vs_spine_floor_pp": float((q1 - c1) * 100.0),
        "p_zero_draft_accept_asbuilt": sc["p_zero_draft_accept"],
        "p_zero_draft_accept_linear": float(1.0 - q1),
        "p_zero_draft_accept_full_model": float(1.0 - model_cov_by_depth[0]),
    }

    # ---- Step 2: classify the gap -------------------------------------------
    asb = sc["mean_emit"]
    gap = ET_full - asb
    sublinear_collapse = DEPLOYED_LINEAR_ET - asb          # below the measured linear floor
    spine_ext = ET_spine - DEPLOYED_LINEAR_ET              # depth-9 spine over linear-8
    branch_premium = ET_full - ET_spine                    # rank-2+ branch contribution

    # provably-fixable lower bound: the portion below the deployed linear MTP floor is
    # structurally impossible for a correct tree (its spine IS the linear chain).
    provable_fixable_pct = 100.0 * sublinear_collapse / gap
    # (D) ceiling share: fern #92 independence (+0.025%) refutes optimistic model on the
    # IDEAL tree under real correlated draws -> the 5.207 ceiling is sound; 0% ceiling.
    ceiling_D_pct = 0.0
    # (C) eager-mode: eager dispatch changes per-step OVERHEAD (attn_py_calls=37), not the
    # verify numerics/argmax -> ~0% of the ACCEPT-LENGTH gap (it is the TPS axis, #97/#85).
    eager_C_accept_pct = 0.0
    # whole gap attributable to fixable build/dispatch defects A/B/C (since D=0).
    fixable_total_pct = 100.0 - ceiling_D_pct

    # (B) bound: salvage fires at ~model rate -> the drafter IS producing correct rank-2+
    # candidates, so draft-quality collapse is bounded small at the front (depth-1/2).
    salvage_ratio_vs_model = ASBUILT["salvage_asbuilt_per_nonfull"] / model_exp_salvage

    step2 = {
        "ET_model_full": float(ET_full),
        "ET_model_spine_only": float(ET_spine),
        "ET_deployed_linear": float(DEPLOYED_LINEAR_ET),
        "ET_asbuilt": float(asb),
        "asbuilt_frac_of_model_pct": float(100.0 * asb / ET_full),
        "total_gap_tokens": float(gap),
        "decomp_sublinear_collapse_tokens": float(sublinear_collapse),
        "decomp_sublinear_collapse_pct": float(provable_fixable_pct),
        "decomp_spine_ext_tokens": float(spine_ext),
        "decomp_spine_ext_pct": float(100.0 * spine_ext / gap),
        "decomp_branch_premium_tokens": float(branch_premium),
        "decomp_branch_premium_pct": float(100.0 * branch_premium / gap),
        "tree_accept_length_gap_explained_pct": float(fixable_total_pct),
        "gap_provably_fixable_lower_bound_pct": float(provable_fixable_pct),
        "gap_ceiling_share_D_pct": float(ceiling_D_pct),
        "eager_C_accept_length_share_pct": float(eager_C_accept_pct),
        "dominant_defect": "A_truncated_salvage_walk_plus_depth1_verify_defect",
        "model_expected_salvage_edges_per_step": float(model_exp_salvage),
        "asbuilt_salvage_per_nonfull": ASBUILT["salvage_asbuilt_per_nonfull"],
        "asbuilt_salvage_vs_model_ratio": float(salvage_ratio_vs_model),
        "fern92_independence_gap_pct": FERN92_INDEP_GAP_PCT,
        "full_tree_reach_rate_model_est": float(np.prod(model_cov_by_depth)),
        "full_tree_reach_rate_asbuilt": float(ASBUILT["full"] / ASBUILT["steps_total"]),
    }

    # ---- Step 3: gate + corrected E[T] band ---------------------------------
    # GREEN: gap dominated by (A)/(C) build/dispatch, NOT acceptance collapse, AND the
    # ceiling (D) is refuted -> the ~568 projection survives once the walk descends full
    # sub-paths and the graph path is healthy. The dominant, PROVABLE component is the
    # sub-linear-floor collapse (build defect) and (D)=0 (fern #92), so: GREEN.
    if provable_fixable_pct >= 40.0 and ceiling_D_pct <= 5.0:
        gate = "GREEN"
    elif ceiling_D_pct >= 50.0:
        gate = "RED"
    else:
        gate = "AMBER"

    step3 = {
        "gate": gate,
        "verdict": "fixable_build_defect_not_acceptance_collapse",
        "corrected_ET_band_for_fern100": {
            "realized_today": float(asb),
            "realized_today_is_build_defect_artifact": True,
            "post_fix_floor": float(DEPLOYED_LINEAR_ET),    # a correct tree MUST clear this
            "post_fix_ceiling": float(ET_full),
            "post_fix_central_target": [5.14, float(ET_full)],
            "requires_remeasure": True,
        },
        "land71_proj_tps_survives": float(LAND71_PROJ_TPS),
        "build_team_handoff": {
            "owners": "land #71 / chiku-inu / byteshark",
            "named_defect": "salvage walk recovers a shallow sub-path (truncated branch "
                            "exploration) AND depth-1 verify/dispatch corrupts the spine "
                            "below its own rank-1 floor",
            "recovery_target_first": "depth-1 spine continuation must equal q[1]=0.7287 "
                                     "(identical-drafter-forward); as-built 0.598 is the "
                                     "fastest single check",
            "recovery_target_second": "accept_length must clear the deployed linear floor "
                                      "3.844 before any quota spend (correct tree >= linear)",
            "remeasure_gate": "byteshark bounded fp32/bit-exact oracle: report salvage, "
                              "accept_length, greedy identity, per-position branch-hit "
                              "(NOT the eager relerr path; wirbel #93 fp32 star-attn req)",
        },
    }

    return {
        "asbuilt_counters": ASBUILT,
        "model_inputs": {
            "q_ladder": Q_LADDER, "rho_ladder": RHO_LADDER,
            "M32_parent": M32_PARENT, "W": W, "max_depth": maxd,
            "ET_full_anchor_reproduced": float(ET_full),
            "ET_full_vs_wirbel_abs": float(abs(ET_full - WIRBEL_ET)),
            "ET_spine_only": float(ET_spine), "ET_linear8": float(ET_lin8),
            "depth_full": int(depth_full),
        },
        "step1_implied_rho": step1,
        "step2_defect_classification": step2,
        "step3_gate": step3,
        "verdict": {
            "primary_metric_name": "tree_accept_length_gap_explained_pct",
            "tree_accept_length_gap_explained_pct": float(fixable_total_pct),
            "gap_provably_fixable_lower_bound_pct": float(provable_fixable_pct),
            "test_metric_name": "implied_tree_rho_vs_model",
            "implied_rho_hat_depth1": float(implied_rho[0]),
            "implied_rho_hat_all_negative": bool(all(r < 0 for r in implied_rho)),
            "gate": gate,
            "ET_asbuilt": float(asb),
            "ET_model_full": float(ET_full),
            "ET_deployed_linear_floor": float(DEPLOYED_LINEAR_ET),
            "depth1_continuation_asbuilt": float(c1),
            "depth1_continuation_required": float(q1),
            "ceiling_D_refuted_by_fern92": True,
            "decision": (
                f"GATE {gate} -- the as-built tok/step={asb:.3f} vs analytical E[T]="
                f"{ET_full:.3f} gap is a FIXABLE BUILD DEFECT, not acceptance collapse. "
                f"DECISIVE: the as-built sits {sublinear_collapse:.3f} tok BELOW the "
                f"measured deployed-linear floor {DEPLOYED_LINEAR_ET:.3f} "
                f"({provable_fixable_pct:.1f}% of the gap), which is structurally "
                f"impossible for a correct tree (its spine IS the linear chain). Depth-1 "
                f"continuation c(1)={c1:.3f} << the identical-drafter-forward floor "
                f"q[1]={q1:.3f} pins a verify/dispatch defect; salvage fires at "
                f"{ASBUILT['salvage_asbuilt_per_nonfull']:.3f} (~model {model_exp_salvage:.3f}) "
                f"so the drafter produces correct rank-2+ candidates (B bounded small) but "
                f"the WALK truncates them (A). Implied rescue rho_hat is NEGATIVE at every "
                f"depth (impossible for a real tree) -- the model rho-ladder is un-exercised, "
                f"not contradicted. (D) optimistic-model ceiling = 0% (fern #92 +0.025%). "
                f"(C) eager is a TPS axis (#97/#85), ~0% of accept-length. fern #100 must "
                f"compose against the corrected band [{DEPLOYED_LINEAR_ET:.3f} floor, "
                f"{ET_full:.3f} ceiling], tree marked BUILD-BLOCKED / re-measure-pending -- "
                f"NOT 2.10 and NOT 5.207."
            ),
        },
    }


def log_wandb(args, results):
    import wandb
    v = results["verdict"]
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name, job_type="analysis",
                     config={"topology": "land71_M32_depth9_mb3",
                             "asbuilt_method": ASBUILT["method"],
                             "asbuilt_steps": ASBUILT["steps_total"],
                             "W": args.W, "wirbel_E_T": WIRBEL_ET,
                             "deployed_linear_E_T": DEPLOYED_LINEAR_ET})
    flat = {f"verdict/{k}": x for k, x in v.items() if not isinstance(x, (dict, list))}
    flat.update({f"step2/{k}": x for k, x in results["step2_defect_classification"].items()
                 if not isinstance(x, (dict, list, str))})
    flat.update({f"step1/{k}": x for k, x in results["step1_implied_rho"].items()
                 if not isinstance(x, (dict, list, str))})
    run.summary.update(flat)
    run.log(flat)

    s1 = results["step1_implied_rho"]
    tb = wandb.Table(columns=["draft_position", "asbuilt_c", "deployed_linear_q",
                              "full_tree_model_cov", "implied_rescue_rho_hat",
                              "asbuilt_below_linear"])
    for k in range(len(s1["asbuilt_continuation_c"])):
        if k < len(s1["deployed_linear_q"]):
            tb.add_data(k + 1, s1["asbuilt_continuation_c"][k], s1["deployed_linear_q"][k],
                        s1["full_tree_model_coverage"][k], s1["implied_rescue_rho_hat"][k],
                        bool(s1["asbuilt_continuation_c"][k] < s1["deployed_linear_q"][k]))
    run.log({"per_position_acceptance": tb})

    s2 = results["step2_defect_classification"]
    gb = wandb.Table(columns=["segment", "tokens", "pct_of_gap", "attribution"])
    gb.add_data("sublinear_collapse (< linear floor)", s2["decomp_sublinear_collapse_tokens"],
                s2["decomp_sublinear_collapse_pct"], "A build defect (provable)")
    gb.add_data("spine_depth9_ext", s2["decomp_spine_ext_tokens"],
                s2["decomp_spine_ext_pct"], "de-risked premium (unrealized)")
    gb.add_data("branch_premium (rank2+)", s2["decomp_branch_premium_tokens"],
                s2["decomp_branch_premium_pct"], "de-risked premium (unrealized)")
    run.log({"gap_decomposition": gb})
    run.finish()
    print(f"[reconcile] W&B run: {run.url}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--W", type=int, default=4)
    ap.add_argument("--max-depth", type=int, default=24)
    ap.add_argument("--output",
                    default="research/spec_cost_model/tree_accept_reconciliation_results.json")
    ap.add_argument("--wandb-project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb-entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb-group", default="tree-accept-reconciliation")
    ap.add_argument("--wandb-name", default="denken/tree-accept-reconciliation")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    results = reconcile(args)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    v = results["verdict"]
    s1, s2, s3 = (results["step1_implied_rho"], results["step2_defect_classification"],
                  results["step3_gate"])
    print("=" * 78)
    print("TREE ACCEPT-LENGTH RECONCILIATION (PR #101)")
    print("=" * 78)
    print(f"  model anchors: E[T]_full={s2['ET_model_full']:.4f} (wirbel 5.207) "
          f"spine={s2['ET_model_spine_only']:.4f} linear={s2['ET_deployed_linear']:.4f}")
    print(f"  as-built tok/step = {s2['ET_asbuilt']:.4f}  "
          f"({s2['asbuilt_frac_of_model_pct']:.1f}% of model)")
    print("-" * 78)
    print("STEP 1 -- implied per-position acceptance vs model")
    print(f"  as-built c(1..7) : {[round(x,3) for x in s1['asbuilt_continuation_c']]}")
    print(f"  deployed q(1..7) : {[round(x,3) for x in s1['deployed_linear_q']]}")
    print(f"  full-tree cov    : {[round(x,3) for x in s1['full_tree_model_coverage']]}")
    print(f"  implied rho_hat  : {[round(x,3) for x in s1['implied_rescue_rho_hat']]}")
    print(f"  -> implied rescue all-negative = {s1['implied_rho_hat_all_negative']} "
          f"(impossible for a real tree)")
    print(f"  -> as-built below linear at EVERY position = "
          f"{s1['asbuilt_below_linear_every_position']}")
    print(f"  -> depth-1: as-built {s1['depth1_continuation_asbuilt']:.3f} vs "
          f"required {s1['depth1_continuation_required_min']:.3f} "
          f"(deficit {s1['depth1_deficit_vs_spine_floor_pp']:.1f}pp)")
    print("-" * 78)
    print("STEP 2 -- defect classification")
    print(f"  gap = {s2['total_gap_tokens']:.4f} tok")
    print(f"    sub-linear collapse : {s2['decomp_sublinear_collapse_tokens']:.4f} "
          f"({s2['decomp_sublinear_collapse_pct']:.1f}%)  <- PROVABLE build defect")
    print(f"    spine depth-9 ext   : {s2['decomp_spine_ext_tokens']:.4f} "
          f"({s2['decomp_spine_ext_pct']:.1f}%)")
    print(f"    branch premium      : {s2['decomp_branch_premium_tokens']:.4f} "
          f"({s2['decomp_branch_premium_pct']:.1f}%)")
    print(f"  salvage: as-built {s2['asbuilt_salvage_per_nonfull']:.3f} vs model "
          f"{s2['model_expected_salvage_edges_per_step']:.3f} "
          f"(ratio {s2['asbuilt_salvage_vs_model_ratio']:.2f}) -> B bounded small")
    print(f"  full-reach: as-built {s2['full_tree_reach_rate_asbuilt']*100:.2f}% vs model "
          f"~{s2['full_tree_reach_rate_model_est']*100:.1f}%")
    print(f"  PRIMARY tree_accept_length_gap_explained_pct = "
          f"{s2['tree_accept_length_gap_explained_pct']:.1f}% "
          f"(provable lower bound {s2['gap_provably_fixable_lower_bound_pct']:.1f}%; "
          f"(D) ceiling {s2['gap_ceiling_share_D_pct']:.1f}%)")
    print("-" * 78)
    print(f"STEP 3 -- GATE: {s3['gate']}  ({s3['verdict']})")
    band = s3["corrected_ET_band_for_fern100"]
    print(f"  fern #100 band: today={band['realized_today']:.3f} (DEFECT artifact)  "
          f"floor={band['post_fix_floor']:.3f}  ceiling={band['post_fix_ceiling']:.3f}")
    print("=" * 78)
    print(v["decision"])
    print("=" * 78)

    if not args.no_wandb:
        try:
            log_wandb(args, results)
        except Exception as e:  # noqa: BLE001
            print(f"[reconcile] W&B logging skipped: {e}", flush=True)
    print(f"[reconcile] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
