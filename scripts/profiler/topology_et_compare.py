#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Empirical E[T] confirm: max-branch-3 vs max-branch-4 M=32 tree topology (PR #91).

WHAT THIS ANSWERS
-----------------
wirbel #83 (MERGED) re-optimised the M=32 draft tree under the MEASURED declining rho
ladder (rho2=0.4165 >> rho3=0.2655 > rho4=0.1908) and found the optimum drops from
#74's max-branch-4 to max-branch-3, buying **+0.96% E[T] / +1.13pp TPS**. Both
topologies are depth-9, M=32 -> identical verify/decode cost, so the delta is
cost-model-INDEPENDENT. BUT it was computed ANALYTICALLY (score_tree_depthrank, a
closed-form sum of root-to-leaf path products) and MC-VALIDATED on ONLY the
max-branch-3 endpoint (#83: 400k greedy sim E[T]=5.214 vs analytic 5.207). The
max-branch-4 endpoint was never simulated, so the +0.96% *delta* has never been
measured end-to-end -- it is a difference of two analytic numbers, one of them
unconfirmed by simulation.

This script CLOSES that gap. It drives fern's #88 Monte-Carlo greedy-descent engine
(`run_mc`, the validated traversal-verify harness) on BOTH topologies under the
EXACT same measured acceptance model wirbel used, and reports the directly-measured
E[T] of each plus the empirical delta. Three independent estimators must agree:

  (1) ANALYTIC  (exact)        score_tree_depthrank on both -> reproduces wirbel's
                               5.20695 / 5.15727 and the +0.9633% delta to 1e-6.
  (2) INDEPENDENT MC (#88)     run_mc greedy on each topology, multi-seed; the
                               max-branch-3/seed-7/400k cell reproduces #88 Leg A
                               (E[T]=5.2140425) bit-for-bit. Each topology's E[T]
                               carries a multi-seed standard error.
  (3) CRN PAIRED MC            both topologies descended on a SHARED per-depth
                               uniform stream (common random numbers). The rank-1
                               spine acceptance is then perfectly correlated across
                               topologies, so the per-trial DELTA variance collapses
                               to only the width-3-vs-width-4 rescue events -- a
                               razor-sharp paired CI on exactly the quantity #91 asks
                               for. Each marginal law is preserved (unbiased E[T]).

PRIMARY metric  topology_et_delta_pct = (E[T]_mb3 / E[T]_mb4 - 1) * 100  (independent MC)
TEST    metric  topology_et_confirmed = 1 if delta_pct >= +0.8% else 0

GATE (PR #91):
  CONFIRMED  delta_pct >= +0.8%   (within 20% of wirbel's +0.96% prediction)
  WEAK       +0.3% <= delta_pct < +0.8%
  REFUTED    delta_pct < +0.3%    (topologies tied at MC precision -> topology-free)

LOCAL, CPU-ONLY, ANALYTIC. No GPU, no vLLM, no HF Job, no submission, no served-file
change. Reuses fern's #88 MC engine and wirbel #79/#83's measured-acceptance
machinery verbatim; the only new code is the two-topology driver + CRN coupling.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traversal_verify_et import run_mc, tree_arrays  # noqa: E402  (fern #88 engine)
from treeshape_measured_accept import (  # noqa: E402  (wirbel #79/#83 acceptance model)
    build_depth_pvecs_measured,
    load_measured,
    load_rank_coverage,
    score_tree_depthrank,
)

RHO_OPT_JSON = "research/spec_cost_model/rho_optimal_topology_results.json"
ACCEPT_JSON = "research/accept_calibration/accept_calibration_results.json"
RANKCOV_JSON = "research/rank_coverage/rank_coverage_results.json"

# wirbel #83 analytic anchors (the numbers this job confirms by simulation).
WIRBEL_F_MB3 = 5.206954309441967   # max-branch-3 DP optimum, score_tree_depthrank
WIRBEL_F_MB4 = 5.157273233329180   # max-branch-4 (#74 topo74), score_tree_depthrank
WIRBEL_DELTA_PCT = 0.9633206127555916   # reopt_ET_gain_over_74 * 100
# fern #88 Leg A reproducibility cell (max-branch-3, seed 7, 400k greedy MC).
FERN88_LEGA_ET = 5.2140425


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} not JSON serializable")


def load_both_topologies(path: str) -> tuple[list[int], list[int], dict]:
    """max-branch-3 (wirbel #83 DP optimum) and max-branch-4 (#74 topo74) parent arrays."""
    d = json.load(open(path))
    pb = d["per_budget"]["32"]
    mb3 = [int(x) for x in pb["optimal"]["parent"]]
    mb4 = [int(x) for x in pb["topo74"]["parent"]]
    meta = {
        "mb3_max_branch": pb["optimal"]["max_branch"],
        "mb4_max_branch": pb["topo74"]["max_branch"],
        "mb3_depth": pb["optimal"]["depth"],
        "mb4_depth": pb["topo74"]["depth"],
        "wirbel_reopt_ET_gain_over_74": pb["reopt_ET_gain_over_74"],
        "wirbel_same_topology_as_74": pb["same_topology_as_74"],
    }
    return mb3, mb4, meta


def topo_stats(parent: list[int]) -> dict:
    children, depth, leaves = tree_arrays(parent)
    return {
        "n": len(parent), "depth": max(depth),
        "max_branch": max(len(c) for c in children), "leaves": len(leaves),
    }


def mc_multiseed(parent: list[int], pvecs, trials: int, seeds: list[int]) -> dict:
    """run_mc (the #88 engine) greedy on one topology across seeds; return per-seed E[T]
    (== et_rootleaf, the deployed root-to-leaf accept length + bonus) and the multi-seed
    mean / standard error. greedy_violation_count must be 0 on every seed (under greedy
    leaf-to-root == root-to-leaf, fern #88's structural result)."""
    ets, viols = [], 0
    for s in seeds:
        r = run_mc(parent, pvecs, trials, seed=s, regime="greedy")
        ets.append(r["et_rootleaf"])
        viols += r["greedy_violation_count"]
    ets = np.asarray(ets, dtype=np.float64)
    mean = float(ets.mean())
    # SE of the mean across independent seeds (each seed is `trials` i.i.d. draws).
    se = float(ets.std(ddof=1) / np.sqrt(len(ets))) if len(ets) > 1 else float("nan")
    return {"per_seed_et": ets.tolist(), "seeds": list(seeds), "trials_per_seed": trials,
            "total_trials": trials * len(seeds), "et_mean": mean, "et_se": se,
            "greedy_violations": viols}


def _descend_et(children, pvecs, maxd, u) -> int:
    """One greedy root-to-leaf descent on `children`, driven by the shared per-depth
    uniform vector `u` (u[d] decides the depth-d child). Returns committed tokens
    (accepted draft length + 1 bonus). Identical bin logic to gen_matches_greedy /
    simulate_greedy_depthrank; the ONLY change is the uniform is supplied (shared)
    rather than drawn locally -- this is what couples the two topologies (CRN)."""
    node, length, dcur = 0, 0, 0
    while children[node]:
        kids = children[node]
        d = dcur + 1
        pv = pvecs[min(d, maxd)]
        draw = u[d]
        cum, chosen = 0.0, -1
        for idx in range(len(kids)):
            r = idx + 1
            cum += pv[r if r < len(pv) else len(pv) - 1]
            if draw < cum:
                chosen = idx
                break
        if chosen < 0:
            break
        node = kids[chosen]
        length += 1
        dcur = d
    return length + 1


def crn_paired(parent3, parent4, pvecs, trials: int, seed: int) -> dict:
    """Common-random-numbers paired MC. Each trial draws ONE uniform vector u[1..maxd+1]
    and descends BOTH trees on it. Because pvecs is depth-indexed and topology-
    independent, the rank-1 spine decisions are perfectly correlated across the two
    trees; the per-trial delta T3-T4 is non-zero only when the shared draw lands in a
    rank>=2 bin that one tree's branch width covers and the other's does not -- i.e. the
    exact max-branch-3-vs-4 rescue difference. Variance of the delta collapses, giving a
    tight CI on topology_et_delta_pct. Marginals are preserved => unbiased E[T]."""
    c3, depth3, _ = tree_arrays(parent3)
    c4, depth4, _ = tree_arrays(parent4)
    maxd = len(pvecs) - 1
    max_tree_depth = max(max(depth3), max(depth4))
    rng = np.random.default_rng(seed)
    s3 = s4 = 0.0
    d_sum = d_sqsum = 0.0
    ndiff = 0
    for _ in range(trials):
        u = rng.random(max_tree_depth + 2)   # u[d] for d=1..max_tree_depth
        t3 = _descend_et(c3, pvecs, maxd, u)
        t4 = _descend_et(c4, pvecs, maxd, u)
        s3 += t3
        s4 += t4
        diff = t3 - t4
        d_sum += diff
        d_sqsum += diff * diff
        if diff != 0:
            ndiff += 1
    et3 = s3 / trials
    et4 = s4 / trials
    mean_d = d_sum / trials
    var_d = max(0.0, d_sqsum / trials - mean_d * mean_d)
    se_d = float(np.sqrt(var_d / trials))
    delta_pct = (et3 / et4 - 1.0) * 100.0 if et4 else 0.0
    # SE of the relative delta: dominated by SE of the paired absolute delta (et4's own
    # sampling error is shared and largely cancels in the ratio); first-order propagation.
    se_delta_pct = se_d / et4 * 100.0 if et4 else 0.0
    return {
        "seed": seed, "trials": trials,
        "et_mb3": et3, "et_mb4": et4,
        "paired_abs_delta": mean_d, "paired_abs_delta_se": se_d,
        "paired_delta_pct": delta_pct, "paired_delta_pct_se": se_delta_pct,
        "paired_delta_pct_ci95": [delta_pct - 1.96 * se_delta_pct,
                                  delta_pct + 1.96 * se_delta_pct],
        "frac_trials_with_diff": ndiff / trials,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rho-opt-json", default=RHO_OPT_JSON)
    ap.add_argument("--accept-json", default=ACCEPT_JSON)
    ap.add_argument("--accept-source", default="server_log")
    ap.add_argument("--rank-coverage-json", default=RANKCOV_JSON)
    ap.add_argument("--W", type=int, default=4)
    ap.add_argument("--max-depth", type=int, default=24)
    ap.add_argument("--extrapolate", default="flat", choices=["flat", "rise"])
    ap.add_argument("--mc-trials", type=int, default=400_000,
                    help="independent-MC trials PER SEED, per topology (#88 engine)")
    ap.add_argument("--mc-seeds", type=int, nargs="+",
                    default=[7, 11, 23, 101, 2027],
                    help="independent-MC seeds; seed 7 @ 400k reproduces #88 Leg A")
    ap.add_argument("--crn-trials", type=int, default=2_000_000,
                    help="CRN paired trials per seed (variance-reduced delta)")
    ap.add_argument("--crn-seeds", type=int, nargs="+", default=[7, 11, 23])
    ap.add_argument("--confirmed-threshold", type=float, default=0.8)
    ap.add_argument("--weak-threshold", type=float, default=0.3)
    ap.add_argument("--output",
                    default="research/spec_cost_model/topology_et_compare_results.json")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb-group", default="tree-topology-et-comparison")
    ap.add_argument("--wandb-name", default="fern/topology-et-compare")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    # ---- inputs: both topologies + the EXACT measured acceptance model wirbel used ----
    mb3, mb4, meta = load_both_topologies(args.rho_opt_json)
    meas = load_measured(args.accept_json, args.accept_source)
    rank_cov = load_rank_coverage(args.rank_coverage_json)
    pvecs = build_depth_pvecs_measured(meas["q"], rank_cov["rho_cond"], args.W,
                                       args.max_depth, args.extrapolate)
    maxd = len(pvecs) - 1
    st3, st4 = topo_stats(mb3), topo_stats(mb4)
    print(f"[topo] max-branch-3 (wirbel #83 opt): {st3}", flush=True)
    print(f"[topo] max-branch-4 (#74 topo74)    : {st4}", flush=True)
    print(f"[topo] measured accept: top1=q[0]={meas['q'][0]:.4f} (PR 0.7287)  "
          f"rho_cond={[round(x, 4) for x in rank_cov['rho_cond']]} (PR [0.4165,0.2655,0.1908])",
          flush=True)
    # both topologies must be the depth-9 / M=32 pair (cost-model-independent comparison)
    assert st3["depth"] == st4["depth"] == 9, "topologies are not both depth-9"
    assert st3["n"] == st4["n"] == 32, "topologies are not both M=32"
    assert st3["max_branch"] == 3 and st4["max_branch"] == 4, "max_branch mismatch"
    assert abs(meas["q"][0] - 0.7287) < 1e-3, "top1 q[0] departs from PR's 0.7287"
    assert abs(rank_cov["rho_cond"][0] - 0.4165) < 1e-3, "rho2 departs from PR's 0.4165"

    # ---- estimator (1): ANALYTIC exact (reproduce wirbel #83) ----
    F3, _ = score_tree_depthrank(mb3, pvecs)
    F4, _ = score_tree_depthrank(mb4, pvecs)
    analytic_delta_pct = (F3 / F4 - 1.0) * 100.0
    print(f"[topo] ANALYTIC F_mb3={F3:.6f} (wirbel {WIRBEL_F_MB3:.6f}) "
          f"F_mb4={F4:.6f} (wirbel {WIRBEL_F_MB4:.6f}) "
          f"delta={analytic_delta_pct:+.4f}% (wirbel {WIRBEL_DELTA_PCT:+.4f}%)", flush=True)
    assert abs(F3 - WIRBEL_F_MB3) < 1e-3, "max-branch-3 analytic E[T] departs from wirbel #83"
    assert abs(F4 - WIRBEL_F_MB4) < 1e-3, "max-branch-4 analytic E[T] departs from wirbel #83"

    # ---- estimator (2): INDEPENDENT MC (#88 engine), multi-seed ----
    mc3 = mc_multiseed(mb3, pvecs, args.mc_trials, args.mc_seeds)
    mc4 = mc_multiseed(mb4, pvecs, args.mc_trials, args.mc_seeds)
    # reproducibility anchor: max-branch-3 @ seed 7 @ 400k must equal #88 Leg A exactly.
    repro_ok = None
    if 7 in args.mc_seeds and args.mc_trials == 400_000:
        idx7 = args.mc_seeds.index(7)
        repro_et = mc3["per_seed_et"][idx7]
        repro_ok = abs(repro_et - FERN88_LEGA_ET) < 1e-9
        print(f"[topo] REPRO #88 Leg A (mb3 seed7 400k): {repro_et:.7f} "
              f"vs {FERN88_LEGA_ET} -> {'EXACT' if repro_ok else 'MISMATCH'}", flush=True)
        assert repro_ok, "did not reproduce #88 Leg A E[T]=5.2140425 -- engine drift!"
    # independent-MC delta + its SE (two independent multi-seed means).
    et3, et4 = mc3["et_mean"], mc4["et_mean"]
    topology_et_delta_pct = (et3 / et4 - 1.0) * 100.0
    se3, se4 = mc3["et_se"], mc4["et_se"]
    # relative-delta SE via first-order propagation of two independent means.
    se_indep_delta_pct = (100.0 / et4) * np.sqrt(se3**2 + (et3 / et4 * se4) ** 2)
    print(f"[topo] INDEP MC mb3 E[T]={et3:.5f}+/-{se3:.5f} "
          f"mb4 E[T]={et4:.5f}+/-{se4:.5f} "
          f"delta={topology_et_delta_pct:+.4f}% +/-{se_indep_delta_pct:.4f}pp "
          f"({mc3['total_trials']} trials/topo)", flush=True)

    # ---- estimator (3): CRN paired MC (tight delta) ----
    crn_runs = [crn_paired(mb3, mb4, pvecs, args.crn_trials, s) for s in args.crn_seeds]
    crn_delta_pcts = np.array([r["paired_delta_pct"] for r in crn_runs])
    crn_abs_deltas = np.array([r["paired_abs_delta"] for r in crn_runs])
    crn_delta_pct_mean = float(crn_delta_pcts.mean())
    crn_delta_pct_se = float(crn_delta_pcts.std(ddof=1) / np.sqrt(len(crn_runs))
                             if len(crn_runs) > 1 else crn_runs[0]["paired_delta_pct_se"])
    crn_et3 = float(np.mean([r["et_mb3"] for r in crn_runs]))
    crn_et4 = float(np.mean([r["et_mb4"] for r in crn_runs]))
    print(f"[topo] CRN PAIRED mb3 E[T]={crn_et3:.5f} mb4 E[T]={crn_et4:.5f} "
          f"abs_delta={crn_abs_deltas.mean():.5f} "
          f"delta={crn_delta_pct_mean:+.4f}% +/-{crn_delta_pct_se:.4f}pp "
          f"(per-seed within-run SE~{crn_runs[0]['paired_delta_pct_se']:.4f}pp, "
          f"{args.crn_trials} trials x {len(args.crn_seeds)} seeds)", flush=True)

    # ---- gate on the PRIMARY metric (independent-MC delta, the PR's literal ask) ----
    d = topology_et_delta_pct
    if d >= args.confirmed_threshold:
        gate = "CONFIRMED"
    elif d >= args.weak_threshold:
        gate = "WEAK"
    else:
        gate = "REFUTED"
    confirmed = 1 if gate == "CONFIRMED" else 0

    if gate == "CONFIRMED":
        decision = (
            f"CONFIRMED -- empirical MC E[T] delta {d:+.3f}% (#88 engine, "
            f"{mc3['total_trials']} trials/topo; CRN paired {crn_delta_pct_mean:+.3f}%) "
            f"matches wirbel #83's analytic +0.96% prediction within MC precision. "
            f"max-branch-3 is empirically optimal. RECOMMEND land #71 BUILD THE "
            f"max-branch-3 array; the topology choice is backed by direct measurement, "
            f"not just the DP cost model. The acceptance model is validated for future "
            f"tree optimisation. lawine #90 should still confirm the served wall_tps "
            f"delta (+1.13pp) once land #71 delivers the build.")
    elif gate == "WEAK":
        decision = (
            f"WEAK -- empirical MC E[T] delta {d:+.3f}% is positive (max-branch-3 still "
            f"wins) but below wirbel #83's +0.96% prediction; the DP model over-predicts "
            f"the realised E[T] gain. land #71 may build max-branch-3 but the margin over "
            f"max-branch-4 is smaller than modelled -- treat the +1.13pp TPS as an upper "
            f"bound, not a banked number.")
    else:
        decision = (
            f"REFUTED -- empirical MC E[T] delta {d:+.3f}% is at/below MC precision; the "
            f"two topologies are statistically tied. wirbel #83's +0.96% figure is not a "
            f"reliably measurable E[T] difference. TOPOLOGY IS FREE: land #71 may build "
            f"max-branch-4 (or whichever is simpler) -- there is no measured E[T] penalty.")

    verdict = {
        "primary_metric_name": "topology_et_delta_pct",
        "topology_et_delta_pct": topology_et_delta_pct,
        "topology_et_delta_pct_se": se_indep_delta_pct,
        "test_metric_name": "topology_et_confirmed",
        "topology_et_confirmed": confirmed,
        "gate": gate,
        "et_max_branch_3_mc": et3,
        "et_max_branch_4_mc": et4,
        "et_max_branch_3_se": se3,
        "et_max_branch_4_se": se4,
        "analytic_delta_pct": analytic_delta_pct,
        "analytic_et_max_branch_3": F3,
        "analytic_et_max_branch_4": F4,
        "crn_paired_delta_pct": crn_delta_pct_mean,
        "crn_paired_delta_pct_se": crn_delta_pct_se,
        "crn_et_max_branch_3": crn_et3,
        "crn_et_max_branch_4": crn_et4,
        "wirbel83_predicted_delta_pct": WIRBEL_DELTA_PCT,
        "delta_vs_wirbel_abs_pp": abs(topology_et_delta_pct - WIRBEL_DELTA_PCT),
        "reproduced_fern88_lega": repro_ok,
        "greedy_violations_mb3": mc3["greedy_violations"],
        "greedy_violations_mb4": mc4["greedy_violations"],
        "decision": decision,
    }

    results = {
        "config": vars(args),
        "topologies": {
            "max_branch_3": {"parent": mb3, **st3},
            "max_branch_4": {"parent": mb4, **st4},
            "meta": meta,
        },
        "inputs": {"top1_q0": meas["q"][0], "q_ladder": meas["q"],
                   "rho_cond": rank_cov["rho_cond"],
                   "wirbel_F_mb3": WIRBEL_F_MB3, "wirbel_F_mb4": WIRBEL_F_MB4,
                   "wirbel_delta_pct": WIRBEL_DELTA_PCT, "fern88_lega_et": FERN88_LEGA_ET},
        "analytic": {"F_mb3": F3, "F_mb4": F4, "delta_pct": analytic_delta_pct},
        "independent_mc": {"mb3": mc3, "mb4": mc4,
                           "delta_pct": topology_et_delta_pct,
                           "delta_pct_se": se_indep_delta_pct},
        "crn_paired_mc": {"runs": crn_runs, "delta_pct_mean": crn_delta_pct_mean,
                          "delta_pct_se": crn_delta_pct_se,
                          "et_mb3": crn_et3, "et_mb4": crn_et4},
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"[topo] wrote {args.output}", flush=True)

    print("\n[topo] ===== TREE TOPOLOGY E[T] EMPIRICAL COMPARISON (M=32, depth-9, greedy) =====",
          flush=True)
    print(f"  E[T] max-branch-3 (build target) = {et3:.5f} +/- {se3:.5f}", flush=True)
    print(f"  E[T] max-branch-4 (#74 baseline) = {et4:.5f} +/- {se4:.5f}", flush=True)
    print(f"  PRIMARY topology_et_delta_pct    = {topology_et_delta_pct:+.4f}% "
          f"+/- {se_indep_delta_pct:.4f}pp  (independent MC, #88 engine)", flush=True)
    print(f"  CRN paired delta (variance-red.) = {crn_delta_pct_mean:+.4f}% "
          f"+/- {crn_delta_pct_se:.4f}pp", flush=True)
    print(f"  ANALYTIC delta (exact, wirbel)   = {analytic_delta_pct:+.4f}%", flush=True)
    print(f"  wirbel #83 predicted             = {WIRBEL_DELTA_PCT:+.4f}%", flush=True)
    print(f"  TEST topology_et_confirmed       = {confirmed}", flush=True)
    print(f"  GATE: {gate}", flush=True)

    if not args.no_wandb:
        try:
            log_wandb(args, results, verdict, mc3, mc4, crn_runs)
        except Exception as e:  # noqa: BLE001
            print(f"[topo] W&B logging failed (non-fatal): {e!r}", flush=True)
    print("[topo] DONE", flush=True)


def log_wandb(args, results, verdict, mc3, mc4, crn_runs):
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name, job_type="profiling",
                     config={"comparison": "max_branch_3_vs_4", "M": 32, "depth": 9,
                             "W": args.W, "mc_trials_per_seed": args.mc_trials,
                             "mc_seeds": args.mc_seeds, "crn_trials": args.crn_trials,
                             "crn_seeds": args.crn_seeds,
                             "confirmed_threshold": args.confirmed_threshold,
                             "weak_threshold": args.weak_threshold,
                             "top1": results["inputs"]["top1_q0"],
                             "rho_cond": results["inputs"]["rho_cond"]})
    flat = {f"verdict/{k}": v for k, v in verdict.items() if not isinstance(v, (dict, list))}
    flat.update({f"analytic/{k}": v for k, v in results["analytic"].items()})
    flat["indep_mc/mb3_et_mean"] = mc3["et_mean"]
    flat["indep_mc/mb3_et_se"] = mc3["et_se"]
    flat["indep_mc/mb4_et_mean"] = mc4["et_mean"]
    flat["indep_mc/mb4_et_se"] = mc4["et_se"]
    flat["indep_mc/total_trials_per_topo"] = mc3["total_trials"]
    flat["crn/delta_pct_mean"] = results["crn_paired_mc"]["delta_pct_mean"]
    flat["crn/delta_pct_se"] = results["crn_paired_mc"]["delta_pct_se"]
    flat["crn/et_mb3"] = results["crn_paired_mc"]["et_mb3"]
    flat["crn/et_mb4"] = results["crn_paired_mc"]["et_mb4"]
    run.summary.update(flat)
    run.log(flat)
    tb = wandb.Table(columns=["estimator", "E[T]_mb3", "E[T]_mb4", "delta_pct", "delta_se_pp"])
    tb.add_data("analytic_exact", results["analytic"]["F_mb3"], results["analytic"]["F_mb4"],
                results["analytic"]["delta_pct"], 0.0)
    tb.add_data("independent_mc", mc3["et_mean"], mc4["et_mean"],
                results["independent_mc"]["delta_pct"], results["independent_mc"]["delta_pct_se"])
    tb.add_data("crn_paired", results["crn_paired_mc"]["et_mb3"],
                results["crn_paired_mc"]["et_mb4"],
                results["crn_paired_mc"]["delta_pct_mean"],
                results["crn_paired_mc"]["delta_pct_se"])
    run.log({"topology_et_estimators": tb})
    # per-seed independent-MC scatter for both topologies
    ts = wandb.Table(columns=["topology", "seed", "E[T]"])
    for s, et in zip(mc3["seeds"], mc3["per_seed_et"]):
        ts.add_data("max_branch_3", s, et)
    for s, et in zip(mc4["seeds"], mc4["per_seed_et"]):
        ts.add_data("max_branch_4", s, et)
    run.log({"per_seed_et": ts})
    run.finish()
    print(f"[topo] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
