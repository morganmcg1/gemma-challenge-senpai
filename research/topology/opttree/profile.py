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

Run (PRIMARY deliverable only):
  cd target/ && CUDA_VISIBLE_DEVICES=0 python research/topology/opttree/profile.py \
      --self-test --wandb_group opttree-perstep-dp --wandb_name stark/opttree-perstep-dp

Run (self-test + TEST metrics e_t_opttree / e_t_gain_vs_static / tps_proj_opttree;
this is the command used for the reported results -- passing no flags is equivalent):
  cd target/ && CUDA_VISIBLE_DEVICES=0 python research/topology/opttree/profile.py \
      --self-test --measure --wandb_group opttree-perstep-dp --wandb_name stark/opttree-perstep-dp
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

# Measured per-step rank-2 catch rate by drafter-entropy decile (equal-count, so
# equal-weight; pooled mean == 0.4172 == RHO_LADDER[2]).  This is the DIRECTLY
# MEASURED per-step surface-variance signal: rho2 swings 0.152..0.657 step-to-step
# with drafter confidence (research/rank_coverage/entropy_branching_results.json,
# drafter_entropy_fine10.bin_rho2, n=13491 divergences on the 128 ShareGPT prompts).
RHO2_DECILES_MEASURED = [
    0.5085, 0.6271, 0.6568, 0.5293, 0.4566, 0.4033, 0.3558, 0.2832, 0.2001, 0.1519,
]
# Measured accepted-length std (research/oracle_readout/et_second_moment;
# both-bugs sigma_L=3.0354).  This is an INTRINSIC-acceptance read: it is the std
# of the committed length under a FIXED probability pmf, so it measures the spread
# of the acceptance *process*, NOT step-to-step variation of the acceptance
# *surface*.  It is therefore NOT a calibration anchor for the q-confidence sweep
# below (and the tree is not live -- only the linear K=7 chain is -- so the tree's
# true per-step q-variance is unmeasurable on-branch).  Printed for context only.
SIGMA_L_MEASURED = 3.0354


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


# =============================================================================
# MEASUREMENT  (E[T] gain on representative measured draft distributions)
# =============================================================================

def _decile_surfaces(q_ladder, rho2_values, rho_tail):
    """Equal-weight per-step surfaces: q fixed at `q_ladder`, rho2 swept over the
    measured deciles, rho3/rho4 held at `rho_tail` (only the measured rho2
    component varies -> a clean, fully-measured lower bound on the gain)."""
    surfaces = []
    for r2 in rho2_values:
        rho = {2: r2, **rho_tail}
        surfaces.append(ladder_edge_prob(q_ladder, rho))
    return surfaces


def _exact_gain(static_tree, surfaces, budget, max_width, max_depth):
    """E[T]_static (fixed tree) vs E[T]_opttree (per-step DP), averaged equal-weight
    over `surfaces`.  Exact path-product accounting -- no Monte-Carlo noise."""
    n = len(surfaces)
    e_static = e_opt = 0.0
    worst_gap = float("inf")
    for surf in surfaces:
        fs, _ = static_tree_et(static_tree, surf)
        _, _, fo = build_opt_tree(surf, budget, max_width, max_depth)
        e_static += fs
        e_opt += fo
        worst_gap = min(worst_gap, fo - fs)
    return e_static / n, e_opt / n, worst_gap


def _shape_variance_analysis(static_tree, surfaces, dev_key, budget, max_width, max_depth):
    """Per-step tree-SHAPE variance + gain-by-deviation decomposition (advisor
    #244 follow-up: the online win can only come from per-step confidence
    variance, so quantify whether the DP actually re-shapes the tree step-to-step
    or collapses onto the static optimum, and WHERE the gain concentrates).

    `dev_key[i]` is the per-step surface deviation from the distribution mean
    (here |rho2_i - mean rho2|).  Returns:
      frac_shape_differs_from_static -- fraction of steps whose OPT-tree parent
        array != the static tree's (0 == DP always reproduces the static optimum),
      distinct_opt_shapes            -- number of unique OPT-tree shapes emitted,
      gain_low_variance_half/high_variance_half -- mean E[T] gain on the low- vs
        high-deviation half of the steps (the gain should concentrate in the
        high-deviation half if the win is variance-driven)."""
    static_shape = tuple(static_tree.parent)
    rows = []
    for i, surf in enumerate(surfaces):
        fs, _ = static_tree_et(static_tree, surf)
        opt_parent, _, fo = build_opt_tree(surf, budget, max_width, max_depth)
        rows.append({"dev": dev_key[i], "shape": tuple(opt_parent),
                     "differs": tuple(opt_parent) != static_shape, "gain": fo - fs})
    n = len(rows)
    order = sorted(range(n), key=lambda i: rows[i]["dev"])
    half = n // 2
    low_idx, high_idx = order[:half], order[half:]
    g_low = sum(rows[i]["gain"] for i in low_idx) / max(1, len(low_idx))
    g_high = sum(rows[i]["gain"] for i in high_idx) / max(1, len(high_idx))
    return {
        "n_steps": n,
        "frac_shape_differs_from_static": sum(r["differs"] for r in rows) / n,
        "distinct_opt_shapes": len({r["shape"] for r in rows}),
        "gain_low_variance_half": g_low,
        "gain_high_variance_half": g_high,
    }


def _simulate_L(tree, edge_prob, rng):
    """One stochastic committed-length draw for `tree` under `edge_prob` (the
    verifier acceptance process the deployed descend-walk realises)."""
    cur = 0
    while tree.children[cur]:
        d = tree.depth[cur] + 1
        u = rng.random()
        acc = 0.0
        nxt = -1
        for child in tree.children[cur]:
            acc += edge_prob(d, tree.rank_in_parent[child])
            if u < acc:
                nxt = child
                break
        if nxt < 0:
            break
        cur = nxt
    return tree.depth[cur] + 1  # committed length = accepted depth + bonus


def _conf_surface(q_mean, rho2_mean, rho_tail, m):
    """Per-step confidence-perturbed surface: multiplier m scales the rank-1 MISS
    gap (1-q) and the rank-2 catch.  m<1 == more confident (higher q, ... ), m>1
    == less confident.  Calibrated so E[m]=1 reproduces the mean ladder."""
    q = [max(0.001, min(0.999, 1.0 - (1.0 - q) * m)) for q in q_mean]
    rho = {2: max(0.0, min(0.95, rho2_mean * m)), **rho_tail}
    return ladder_edge_prob(q, rho)


def _qconf_sweep(q_mean, rho2_mean, rho_tail, static_tree, budget, max_width,
                 max_depth, scales, n_steps, seed):
    """EXPLORATORY sensitivity: per-step joint q+rho2 confidence variance.  For
    each spread `scale`, draw m_step ~ lognormal(mean 1) and measure the gain.
    The reported sigma_L is the MODELLED accepted-length std at that scale, shown
    for context only -- it is NOT a calibration anchor: the measured sigma_L=3.0354
    (et_second_moment) is a fixed-probability pmf read (intrinsic acceptance
    variance), so it carries no information about per-step surface variance, and
    the tree is not live (only the linear K=7 chain is) so the tree's true
    per-step q-variance is not measurable on-branch."""
    import math
    out = []
    for scale in scales:
        rng = random.Random(seed)
        e_static = e_opt = 0.0
        Ls = []
        sigma = scale
        mu = -0.5 * sigma * sigma  # so E[m] = 1
        for _ in range(n_steps):
            m = math.exp(rng.gauss(mu, sigma))
            surf = _conf_surface(q_mean, rho2_mean, rho_tail, m)
            fs, _ = static_tree_et(static_tree, surf)
            _, _, fo = build_opt_tree(surf, budget, max_width, max_depth)
            e_static += fs
            e_opt += fo
            Ls.append(_simulate_L(static_tree, surf, rng))
        n = n_steps
        mean_L = sum(Ls) / n
        var_L = sum((x - mean_L) ** 2 for x in Ls) / n
        out.append({
            "q_conf_scale": scale,
            "e_t_static": e_static / n,
            "e_t_opttree": e_opt / n,
            "e_t_gain": (e_opt - e_static) / n,
            "sigma_L_static": var_L ** 0.5,
        })
    return out


def measure(verbose: bool = True) -> dict:
    """Measure realized E[T] (static vs OPT-tree per-step) on the measured draft
    distributions, at the deployed verify budgets where the topology lever lives."""
    res: dict = {}
    # Budget configs: (label, M, max_width-cap, max_depth). M=8 is the deployed
    # linear verify width (branching does not pay there -> ~zero gain, a sanity
    # anchor); M=16/M=32 are the tree-verify build targets where the 5.066/5.219
    # E[T] anchors and the max-branch-2/3 topology lever live.
    configs = [
        ("M8_deployed", 8, 1, 7),
        ("M16_branch2", 16, 2, 9),
        ("M32_branch3", 32, 3, 9),
    ]
    rho_tail = {3: RHO_LADDER[3], 4: RHO_LADDER[4]}
    # per-step surface deviation from the distribution mean (|rho2_decile - mean|);
    # drives the high- vs low-variance gain split (advisor #244 follow-up).
    mean_rho2 = sum(RHO2_DECILES_MEASURED) / len(RHO2_DECILES_MEASURED)
    rho2_dev = [abs(r - mean_rho2) for r in RHO2_DECILES_MEASURED]

    headline = None
    for label, M, W, D in configs:
        q = extend_q(Q_LADDER_MEAN, D)
        mean_surface = ladder_edge_prob(q, {2: RHO_LADDER[2], **rho_tail})
        static_parent, _, static_et_mean = build_opt_tree(mean_surface, M, W, D)
        static_tree = TreeSpec(static_parent)

        # PRIMARY: fully-measured rho2-decile per-step variance (q at mean).
        surfaces = _decile_surfaces(q, RHO2_DECILES_MEASURED, rho_tail)
        e_static, e_opt, worst_gap = _exact_gain(static_tree, surfaces, M, W, D)
        gain = e_opt - e_static

        # SHAPE VARIANCE (advisor #244 follow-up): does the DP re-shape the tree
        # step-to-step, and where does the gain concentrate?
        shape = _shape_variance_analysis(static_tree, surfaces, rho2_dev, M, W, D)

        # EXPLORATORY sensitivity: per-step joint q+rho2 confidence variance.  This
        # is NOT measured on-branch (see SIGMA_L_MEASURED note) -- it shows how the
        # gain would grow IF unmeasured per-step q-variance existed, scanned over a
        # range of spreads.  The max over the scan is an exploratory upper band.
        sweep = _qconf_sweep(q, RHO_LADDER[2], rho_tail, static_tree, M, W, D,
                             scales=[0.0, 0.15, 0.30, 0.45, 0.60, 0.75],
                             n_steps=8000, seed=7)
        sweep_max = max(sweep, key=lambda s: s["e_t_gain"])

        cfg = {
            "budget_M": M, "max_width": W, "max_depth": D,
            "static_parent": static_parent,
            "static_max_branch": static_tree.max_branch,
            "static_depth": static_tree.max_depth,
            "e_t_static_mean_surface": static_et_mean,
            # PRIMARY (rho2-decile, fully measured on-branch, q fixed at mean):
            "e_t_static": e_static,
            "e_t_opttree": e_opt,
            "e_t_gain_vs_static": gain,
            "worst_perstep_gap": worst_gap,
            "tps_proj_static": tps_of(e_static),
            "tps_proj_opttree": tps_of(e_opt),
            "tps_gain": tps_of(e_opt) - tps_of(e_static),
            # SHAPE variance + gain-by-deviation (advisor #244 follow-up):
            "frac_shape_differs_from_static": shape["frac_shape_differs_from_static"],
            "distinct_opt_shapes": shape["distinct_opt_shapes"],
            "gain_low_variance_half": shape["gain_low_variance_half"],
            "gain_high_variance_half": shape["gain_high_variance_half"],
            # EXPLORATORY (joint q+rho2 confidence; unmeasured -- upper band only):
            "qconf_sweep": sweep,
            "e_t_gain_exploratory_max": sweep_max["e_t_gain"],
            "e_t_gain_exploratory_max_scale": sweep_max["q_conf_scale"],
            "tps_gain_exploratory_max": tps_of(sweep_max["e_t_opttree"]) - tps_of(sweep_max["e_t_static"]),
        }
        res[label] = cfg
        if label == "M32_branch3":
            headline = cfg
        if verbose:
            print(f"\n[measure] === {label}  M={M} max_branch<={W} depth<={D} ===")
            print(f"  static rho-opt (DP on mean): E[T]_mean={static_et_mean:.4f} "
                  f"branch={static_tree.max_branch} depth={static_tree.max_depth}")
            print(f"  PRIMARY (measured rho2 deciles, q fixed):")
            print(f"    E[T]_static={e_static:.4f}  E[T]_opttree={e_opt:.4f}  "
                  f"gain={gain:+.4f}  (worst per-step gap {worst_gap:+.5f})")
            print(f"    TPS_static={tps_of(e_static):.1f}  TPS_opttree={tps_of(e_opt):.1f}  "
                  f"dTPS={tps_of(e_opt)-tps_of(e_static):+.1f}")
            print(f"  SHAPE variance (advisor #244): DP picks a different shape than "
                  f"static on {shape['frac_shape_differs_from_static']*100:.0f}% of steps "
                  f"({shape['distinct_opt_shapes']} distinct shapes over {shape['n_steps']} deciles)")
            print(f"    gain | low-variance half = {shape['gain_low_variance_half']:+.4f}   "
                  f"gain | high-variance half = {shape['gain_high_variance_half']:+.4f}")
            print(f"  EXPLORATORY (joint q+rho2 confidence variance; NOT measured "
                  f"on-branch -- modelled sigma_L below is not comparable to the "
                  f"intrinsic measured sigma_L={SIGMA_L_MEASURED}):")
            for s in sweep:
                mark = "  <-- exploratory max" if s is sweep_max else ""
                print(f"    scale={s['q_conf_scale']:.2f}: sigma_L(modelled)={s['sigma_L_static']:.3f} "
                      f"gain={s['e_t_gain']:+.4f} (dTPS {tps_of(s['e_t_opttree'])-tps_of(s['e_t_static']):+.1f}){mark}")

    # headline summary (M=32 max-branch-3, where the 5.219 / 536.659 anchors live).
    # PRIMARY = the fully-measured rho2-decile gain (q fixed at mean); the stop-
    # condition is judged on THIS measured number, not on the exploratory sweep.
    res["headline_budget"] = "M32_branch3"
    res["e_t_opttree"] = headline["e_t_opttree"]
    res["e_t_gain_vs_static"] = headline["e_t_gain_vs_static"]
    res["tps_proj_opttree"] = headline["tps_proj_opttree"]
    res["tps_gain"] = headline["tps_gain"]
    res["e_t_gain_measured_floor"] = headline["e_t_gain_vs_static"]          # rho2-only (measured)
    res["e_t_gain_exploratory_max"] = headline["e_t_gain_exploratory_max"]   # unmeasured upper band
    res["frac_shape_differs_from_static"] = headline["frac_shape_differs_from_static"]
    res["gain_low_variance_half"] = headline["gain_low_variance_half"]
    res["gain_high_variance_half"] = headline["gain_high_variance_half"]
    stop = 0.05
    res["gain_clears_stop_condition"] = bool(headline["e_t_gain_vs_static"] >= stop)
    res["stop_condition_et_gain"] = stop
    if verbose:
        print(f"\n[measure] HEADLINE (M=32 max-branch-3):")
        print(f"  PRIMARY measured e_t_gain = {res['e_t_gain_measured_floor']:+.4f} "
              f"(rho2-decile, q fixed)  ->  dTPS {res['tps_gain']:+.1f}")
        print(f"  shape variance: DP re-shapes on {res['frac_shape_differs_from_static']*100:.0f}% of steps; "
              f"gain low-var half {res['gain_low_variance_half']:+.4f} vs high-var half "
              f"{res['gain_high_variance_half']:+.4f} (win is variance-driven)")
        print(f"  exploratory upper band   = {res['e_t_gain_exploratory_max']:+.4f} "
              f"(unmeasured per-step q-variance; not used for the verdict)")
        print(f"  stop-condition (measured gain>={stop}): "
              f"{'CLEARS' if res['gain_clears_stop_condition'] else 'BELOW -> DP overhead may not be worth it'}")
    return res


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
    if args.measure or run_all:
        metrics["measurement"] = measure()

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
            # self-test scalars (top-level) + flattened measurement scalars (headline
            # + per-budget primary), so every headline number is a first-class W&B key.
            flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float, bool))}
            meas = metrics.get("measurement", {})
            for k, v in meas.items():
                if isinstance(v, (int, float, bool)):
                    flat[f"measurement/{k}"] = v
                elif isinstance(v, dict):  # a per-budget config (e.g. M32_branch3)
                    for ck, cv in v.items():
                        if isinstance(cv, (int, float, bool)):
                            flat[f"measurement/{k}/{ck}"] = cv
            run.log(flat)
            run.summary.update(flat)
            run.finish()
            print(f"[wandb] logged {len(flat)} scalars to {args.wandb_name}")
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] skipped: {e}")


if __name__ == "__main__":
    main()
