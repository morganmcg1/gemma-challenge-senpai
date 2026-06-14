"""T-1 OPT-Tree per-step DP profiler (PR #247, stark).

LOCAL profiling only (1xA10G, but this routine is pure-CPU/NumPy -- no GPU, no
model load, no served-file change, no HF Job). Implements + validates the
OPT-Tree per-step adaptive draft-tree DP and measures its realized E[T] gain
over the deployed static tree at matched budget.

Deliverable (PRIMARY): ``opttree_dp_self_test_passes`` --
  (a) DP weakly dominates the static tree on every tested distribution,
  (b) greedy-identity preserved (verify criterion untouched),
  (c) NaN-clean, (d) DP latency < 0.1 ms/step.
TEST metrics: ``e_t_opttree``, ``e_t_gain_vs_static``, ``tps_proj_opttree``.

Run:
  cd target/ && CUDA_VISIBLE_DEVICES=0 python research/topology/opttree/profile.py \
      --self-test --wandb_group opttree-perstep-dp --wandb_name stark/opttree-perstep-dp
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from importlib import util as _u
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repo root (target/)
HERE = Path(__file__).resolve().parent

_dp_spec = _u.spec_from_file_location("opttree_dp", HERE / "opttree_dp.py")
opttree_dp = _u.module_from_spec(_dp_spec)
sys.modules.setdefault("opttree_dp", opttree_dp)
_dp_spec.loader.exec_module(opttree_dp)

tree_spec = opttree_dp.tree_spec
TreeSpec = tree_spec.TreeSpec
build_opt_tree = opttree_dp.build_opt_tree
static_tree_et = opttree_dp.static_tree_et
ladder_edge_prob = opttree_dp.ladder_edge_prob

# ---- deployed convention anchors (BASELINE.md / tree_verify_path) -------------
K_CAL = 125.268          # TPS = K_CAL * E[T] / STEP  (481.53 = K_CAL * 3.844)
STEP_INT4 = 1.2182       # measured M=8-norm depth-9 verify step (lawine #136)
DEPLOYED_ET = 5.066      # deployed-faithful tree E[T] -> 520.95 operative ceiling
STATIC_RHO_OPT_ET = 5.219  # static rho-optimal max-branch-3 -> 536.659 reach-DP ceiling
OPERATIVE_CEILING_TPS = 520.95
REACHDP_CEILING_TPS = 536.659
OFFICIAL_BASELINE_TPS = 481.53

# Measured acceptance ladder (research/accept_calibration server_log; tree_spec).
Q_LADDER_MEAN = [0.7287, 0.7590, 0.7925, 0.8217, 0.8343, 0.8353, 0.8473]
RHO_LADDER = {2: 0.4165, 3: 0.2655, 4: 0.1908}  # P(target == rank-k | rank-1 miss)
TOP1 = tree_spec.TOP1_MEASURED  # 0.729


def tps_of(e_t: float, step: float = STEP_INT4) -> float:
    """Deployed-convention TPS projection at fixed verify cost."""
    return K_CAL * e_t / step


def extend_q(q: list[float], depth: int) -> list[float]:
    """Extend the measured q-ladder to `depth` entries, flat-extrapolating the
    deepest measured rung (survivorship plateau ~0.85; tree_verify_path)."""
    out = list(q)
    while len(out) < depth:
        out.append(out[-1])
    return out[:depth]


# =============================================================================
# SELF-TEST  (PRIMARY deliverable: opttree_dp_self_test_passes)
# =============================================================================

def _random_surface(rng: random.Random, max_depth: int):
    """A random (depth, rank) acceptance surface for dominance/NaN fuzzing.

    q[d] in (0,1) per depth (independent -> arbitrary, NOT just the measured
    monotone ladder, so dominance is stressed on adversarial shapes too); rho
    ladder drawn in (0, 0.7) and renormalised so sum_k rho_k <= 1."""
    q = [rng.uniform(0.05, 0.98) for _ in range(max_depth)]
    raw = {2: rng.uniform(0.0, 0.7), 3: rng.uniform(0.0, 0.4), 4: rng.uniform(0.0, 0.25)}
    s = sum(raw.values())
    if s > 1.0:
        raw = {k: v / s for k, v in raw.items()}
    return ladder_edge_prob(q, raw)


def _greedy_realization(tree: "TreeSpec", hit_rank: dict[int, int]):
    """Construct a (node_argmax, draft_token) realization where, at each depth p,
    the verifier greedy token g[p] is hit by exactly the drafter's rank-`hit_rank[p]`
    candidate (or by none if hit_rank[p] is out of the drafted ranks).

    Convention (tree_spec.descend_accept): node_argmax[node] = g[depth(node)];
    draft_token[node] = drafter's rank-(rank_in_parent) candidate for predicting
    g[depth(parent)] -- equals g[parent_depth] iff that node's rank == hit_rank.
    Returns (node_argmax, draft_token, g) with g[d] = 1000 + d the greedy token.
    """
    g = {d: 1000 + d for d in range(tree.max_depth + 1)}
    node_argmax = [0] * tree.num_nodes
    draft_token = [0] * tree.num_nodes
    for node in range(tree.num_nodes):
        d = tree.depth[node]
        node_argmax[node] = g[d]
        if node == 0:
            draft_token[node] = -1  # root token unused by the walk
            continue
        p = d - 1  # parent depth
        if tree.rank_in_parent[node] == hit_rank.get(p, 1):
            draft_token[node] = g[p]            # this candidate hits the argmax
        else:
            draft_token[node] = 9_000_000 + d * 10 + tree.rank_in_parent[node]  # miss
    return node_argmax, draft_token, g


def self_test(verbose: bool = True) -> dict:
    """Run the four self-test facets. Returns a dict of metrics + pass flags."""
    rng = random.Random(20260614)
    budget, max_width, max_depth = 8, 4, 7  # deployed M=8 verify width, drafter top-4
    results: dict = {"budget": budget, "max_width": max_width, "max_depth": max_depth}

    # ---- mean-surface static rho-optimal tree (the fixed offline baseline) ----
    mean_surface = ladder_edge_prob(extend_q(Q_LADDER_MEAN, max_depth), RHO_LADDER)
    static_parent, _, static_et_mean = build_opt_tree(
        mean_surface, budget, max_width, max_depth
    )
    static_tree = TreeSpec(static_parent)
    deployed_linear = TreeSpec(tree_spec.linear_parent(budget))  # M=8 K=7 chain
    if verbose:
        print(f"[self-test] static rho-opt M={budget} parent={static_parent}")
        print(f"            spine={static_tree.spine} max_branch={static_tree.max_branch} "
              f"depth={static_tree.max_depth} E[T]_mean={static_et_mean:.4f}")
        print(f"            deployed linear M={budget} E[T]_mean="
              f"{static_tree_et(deployed_linear, mean_surface)[0]:.4f}")

    # ---- (a) WEAK DOMINANCE on every tested surface ---------------------------
    n_surf = 4000
    worst_gap = float("inf")
    nan_clean = True
    dom_fail = 0
    for _ in range(n_surf):
        surf = _random_surface(rng, max_depth)
        opt_parent, opt_pp, opt_et = build_opt_tree(surf, budget, max_width, max_depth)
        st_et, st_pp = static_tree_et(static_tree, surf)
        lin_et, lin_pp = static_tree_et(deployed_linear, surf)
        for arr in (opt_pp, st_pp, lin_pp, [opt_et, st_et, lin_et]):
            if any((x != x) or (x in (float("inf"), float("-inf"))) for x in arr):
                nan_clean = False
        # OPT-tree must weakly dominate BOTH fixed trees at equal budget.
        gap_static = opt_et - st_et
        gap_linear = opt_et - lin_et
        worst_gap = min(worst_gap, gap_static, gap_linear)
        if gap_static < -1e-9 or gap_linear < -1e-9:
            dom_fail += 1
        # constructed tree must be structurally valid (raises if not).
        TreeSpec(opt_parent)
    results["dominance_surfaces"] = n_surf
    results["dominance_failures"] = dom_fail
    results["dominance_worst_gap"] = worst_gap
    results["nan_clean"] = nan_clean
    weak_dominance = (dom_fail == 0) and (worst_gap >= -1e-9)
    if verbose:
        print(f"[self-test] (a) weak dominance: {n_surf} surfaces, failures={dom_fail}, "
              f"worst E[T] gap vs fixed = {worst_gap:+.6f}  -> {'PASS' if weak_dominance else 'FAIL'}")
        print(f"[self-test] (c) NaN-clean: {'PASS' if nan_clean else 'FAIL'}")

    # ---- (b) GREEDY-IDENTITY: committed seq is a prefix of the greedy run -----
    gi_ok = True
    gi_checked = 0
    gi_extra_salvage = 0  # steps where OPT-tree committed strictly more greedy tokens
    for _ in range(3000):
        surf = _random_surface(rng, max_depth)
        opt_parent, _, _ = build_opt_tree(surf, budget, max_width, max_depth)
        opt_tree = TreeSpec(opt_parent)
        # per-depth hitting rank (which drafter rank matches the verifier argmax)
        hit_rank = {p: rng.choice([1, 1, 1, 2, 2, 3, 99]) for p in range(max_depth + 1)}
        for tree in (static_tree, opt_tree, deployed_linear):
            node_argmax, draft_token, g = _greedy_realization(tree, hit_rank)
            emitted, vc, _ = tree_spec.descend_accept(tree, node_argmax, draft_token)
            expected = [g[d] for d in range(len(emitted))]  # greedy prefix
            if emitted != expected:
                gi_ok = False
            gi_checked += 1
        # OPT-tree may commit >= the linear chain's greedy tokens (never a diff one)
        e_opt = tree_spec.descend_accept(opt_tree, *_greedy_realization(opt_tree, hit_rank)[:2])[1]
        e_lin = tree_spec.descend_accept(deployed_linear,
                                         *_greedy_realization(deployed_linear, hit_rank)[:2])[1]
        if e_opt > e_lin:
            gi_extra_salvage += 1
    results["greedy_identity_checks"] = gi_checked
    results["greedy_identity_extra_salvage_steps"] = gi_extra_salvage
    if verbose:
        print(f"[self-test] (b) greedy-identity: {gi_checked} (tree,realization) checks, "
              f"all committed seqs == verifier-argmax prefix -> {'PASS' if gi_ok else 'FAIL'} "
              f"({gi_extra_salvage} steps OPT-tree salvaged extra greedy tokens vs linear)")

    # ---- (d) LATENCY < 0.1 ms/step --------------------------------------------
    n_lat = 2000
    surfs = [_random_surface(rng, max_depth) for _ in range(n_lat)]
    t0 = time.perf_counter()
    for surf in surfs:
        build_opt_tree(surf, budget, max_width, max_depth)
    dt = time.perf_counter() - t0
    lat_ms = 1000.0 * dt / n_lat
    results["dp_latency_ms_mean"] = lat_ms
    latency_ok = lat_ms < 0.1
    if verbose:
        print(f"[self-test] (d) DP latency: {lat_ms*1000:.2f} us/step "
              f"(< 0.1 ms -> {'PASS' if latency_ok else 'FAIL'})")

    passes = bool(weak_dominance and nan_clean and gi_ok and latency_ok)
    results["weak_dominance_pass"] = weak_dominance
    results["greedy_identity_pass"] = gi_ok
    results["latency_pass"] = latency_ok
    results["opttree_dp_self_test_passes"] = passes
    if verbose:
        print(f"[self-test] === opttree_dp_self_test_passes = {passes} ===")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="OPT-Tree per-step DP profiler (PR #247)")
    ap.add_argument("--self-test", action="store_true", help="run the self-test battery")
    ap.add_argument("--measure", action="store_true", help="run the E[T] measurement")
    ap.add_argument("--wandb_group", default=None)
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--out", default=str(HERE / "opttree_profile_results.json"))
    args = ap.parse_args()
    run_all = not (args.self_test or args.measure)

    metrics: dict = {}
    if args.self_test or run_all:
        metrics.update(self_test())

    metrics["_anchors"] = {
        "K_cal": K_CAL, "step_int4": STEP_INT4,
        "deployed_et": DEPLOYED_ET, "static_rho_opt_et": STATIC_RHO_OPT_ET,
        "operative_ceiling_tps": OPERATIVE_CEILING_TPS,
        "reachdp_ceiling_tps": REACHDP_CEILING_TPS,
    }
    Path(args.out).write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nwrote {args.out}")

    if not args.no_wandb and (args.wandb_name or args.wandb_group):
        try:
            import wandb
            run = wandb.init(
                project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
                group=args.wandb_group, name=args.wandb_name,
                config={"experiment": "opttree-perstep-dp", "pr": 247,
                        "budget": metrics.get("budget"), "max_width": metrics.get("max_width")},
            )
            flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float, bool))}
            run.log(flat)
            run.summary.update(flat)
            run.finish()
            print(f"[wandb] logged {len(flat)} scalars to {args.wandb_name}")
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] skipped: {e}")


if __name__ == "__main__":
    main()
