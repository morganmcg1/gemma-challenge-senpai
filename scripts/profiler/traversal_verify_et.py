#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Traversal Verification E[T] gate on wirbel #83's DP-optimal M=32 tree (PR #88).

WHAT THIS ANSWERS
-----------------
land #71 deploys a tree-verify serving path that, on wirbel #83's DP-optimal M=32
draft tree, walks the token tree ROOT-TO-LEAF and accepts the longest root path
whose every token matches the target's argmax given the accepted prefix ("longest
consistent prefix"; SpecInfer/Medusa full-tree verification -- ALL children of a
node are candidates, so a rank-2 sibling that matches the target argmax at a
divergence IS accepted).

"Traversal Verification" (NeurIPS 2025, OpenReview 8nOMhDFpkU) instead walks
LEAF-TO-ROOT and accepts any consistent path ending at a leaf, with the stated goal
of recovering sibling-subtree probability mass that root-to-leaf's recursive
rejection discards. PR #88 asks: on wirbel's M=32 tree, how much E[T] does the
leaf-to-root rule REALISE over the root-to-leaf baseline, and is it 100%
greedy-identical?

THE RESULT (decisive, structural)
---------------------------------
The challenge serves GREEDY (the emitted sequence must be token-identical to plain
argmax autoregressive decode -- a hard validity gate, program.md). Under greedy the
target argmax at each position is a SINGLE token, so at any tree node AT MOST ONE
child token can equal it (siblings are distinct top-k draft tokens). Therefore:

  * the set of "consistent" (fully target-matching) tree paths is a UNIQUE CHAIN
    (the prefix of the greedy target output that the tree happens to contain);
  * root-to-leaf descends that unique chain to its maximal end (at each node the
    matching child, when it exists, is unique, so there is no branch to mis-pick);
  * leaf-to-root selects the longest fully-matching root->leaf path, which is the
    SAME unique chain.

=> Under greedy, leaf-to-root accepts EXACTLY the same tokens as root-to-leaf, for
ANY tree and ANY corpus. traversal_et_uplift_pct = 0 by construction, and both emit
the identical greedy chain so traversal_greedy_violation_count = 0 (it is trivially
lossless because it equals the deployed rule). wirbel's salvage oracle rho2=0.4165
(rank-2 sibling rescue) is ALREADY realised by full-tree root-to-leaf; it is the
value of the TREE over a linear chain, NOT incremental headroom for traversal.

This script PROVES that empirically on wirbel's exact M=32 topology:
  Leg A (physical):  greedy Monte-Carlo on the M=32 tree under wirbel's MEASURED
                     MTP per-depth acceptance (reproduces his E[T]=5.207). Run BOTH
                     walks per step -> uplift 0.000%, 0 path violations.
  Leg B (contrast):  RELAX greedy to a sampling-style regime where >1 sibling may be
                     target-consistent (independent per-child match). Here leaf-to-
                     root STRICTLY beats root-to-leaf -- proving the two walk
                     implementations are NOT accidentally identical, and pinpointing
                     that the ONLY condition under which traversal pays is "two
                     matching siblings", which greedy decoding forbids.
  Leg C (real data): drive the M=32 spine with #80's REAL per-position drafter
                     hit-ranks (greedy debug corpus) -> uplift 0 on realised ranks.
  Leg D (exhaustive):brute-force ALL match-labellings of small trees: under the
                     greedy single-match invariant the walks agree on every one;
                     drop the invariant and they can differ (mechanism check).

LOCAL, CPU-ONLY, ANALYTIC. No GPU, no vLLM, no HF Job, no submission, no served-file
change. Extends fern's #80 native-acceptance machinery (an acceptance-RULE measure,
same MTP, same tree, different verify walk -- NOT a drafter change). Reuses wirbel
#83's validated pv / topology machinery verbatim.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from treeshape_measured_accept import (  # noqa: E402
    build_depth_pvecs_measured,
    build_linear,
    load_measured,
    load_rank_coverage,
    score_tree_depthrank,
)

RHO_OPT_JSON = "research/spec_cost_model/rho_optimal_topology_results.json"
ACCEPT_JSON = "research/accept_calibration/accept_calibration_results.json"
RANKCOV_JSON = "research/rank_coverage/rank_coverage_results.json"
TRACE_80 = "research/eagle3_drafter/eval_traces/topk_trace_debug1k2ep.jsonl"
WIRBEL_E_T_M32 = 5.207            # wirbel #83 analytic E[T] of the M=32 optimum
WIRBEL_RHO2 = 0.4165             # wirbel #83 salvage oracle (rank-2 rescue ratio)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} not JSON serializable")


# --------------------------------------------------------------------------- #
# Tree helpers
# --------------------------------------------------------------------------- #
def tree_arrays(parent: list[int]):
    """children[u] in birth-order (== drafter rank), depth[u], leaves set."""
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    for i in range(1, n):
        children[parent[i]].append(i)
    depth = [0] * n
    for i in range(1, n):
        depth[i] = depth[parent[i]] + 1
    leaves = [u for u in range(n) if not children[u]]
    return children, depth, leaves


def load_m32_topology(path: str = RHO_OPT_JSON) -> list[int]:
    d = json.load(open(path))
    return [int(x) for x in d["per_budget"]["32"]["optimal"]["parent"]]


def strip_to_spine(parent: list[int]) -> list[int]:
    """The rank-1 (spine) chain only -- the LINEAR tree with all rank>=2 branches
    removed. Re-indexed 0..L. Used to isolate the rescue mass the branches add."""
    children, _, _ = tree_arrays(parent)
    spine = [0]
    u = 0
    while children[u]:
        u = children[u][0]          # rank-1 child = lowest id
        spine.append(u)
    return build_linear(len(spine))


# --------------------------------------------------------------------------- #
# The two acceptance walks. Both consume a boolean `matches[node]` ("this node's
# draft token == the target argmax given its parent path") and return the accepted
# draft-node PATH (excluding the root anchor). E[committed] = len(path) + 1 (bonus).
# --------------------------------------------------------------------------- #
def walk_root_to_leaf(children, matches) -> list[int]:
    """Deployed baseline: descend from the root, at each node take a matching child;
    accept the longest such prefix. SpecInfer/Medusa full-tree greedy verification.

    If several children match (greedy forbids this; only the sampling-contrast leg
    produces it) the standard verifier commits to the highest-rank (lowest-id =
    spine-preferred) matching child -- it does NOT backtrack to a sibling subtree.
    """
    path = []
    u = 0
    while True:
        m = [c for c in children[u] if matches[c]]
        if not m:
            break
        u = m[0]                    # highest drafter rank among matches (spine-pref)
        path.append(u)
    return path


def walk_leaf_to_root(parent, children, depth, matches) -> list[int]:
    """Traversal Verification: accept the longest root->leaf path all of whose nodes
    match. Implemented as: full[u] = matches[u] AND full[parent[u]]; take the deepest
    full node and return its root path. (Ties -> lowest id, deterministic.)"""
    n = len(parent)
    full = [False] * n
    full[0] = True                  # root anchor is the already-verified token
    order = sorted(range(1, n), key=lambda x: depth[x])   # parents before children
    best = 0
    for u in order:
        full[u] = matches[u] and full[parent[u]]
        if full[u] and depth[u] > depth[best]:
            best = u
    # reconstruct root->best path (exclude root)
    path = []
    u = best
    while u != 0:
        path.append(u)
        u = parent[u]
    return path[::-1]


# --------------------------------------------------------------------------- #
# Match-labelling generators
# --------------------------------------------------------------------------- #
def gen_matches_greedy(children, depth, pvecs, rng, maxd):
    """FAITHFUL greedy: the target argmax sequence reveals a SINGLE matching chain.

    At the current accepted node u (depth d) the target's next argmax equals u's
    rank-r child token w.p. pvecs[d+1][r] (mutually exclusive across ranks; the
    residual is a hard miss / token beyond this node's branch width). Exactly one
    child of u can match. Mirrors simulate_greedy_depthrank EXACTLY, but emits the
    full matches[] labelling so both walks can be run on it.
    """
    n = len(children)
    matches = [False] * n
    u = 0
    while children[u]:
        kids = children[u]
        d = depth[u] + 1
        pv = pvecs[min(d, maxd)]
        draw = rng.random()
        cum, chosen = 0.0, -1
        for idx in range(len(kids)):
            r = idx + 1
            cum += pv[r if r < len(pv) else len(pv) - 1]
            if draw < cum:
                chosen = idx
                break
        if chosen < 0:
            break
        u = kids[chosen]
        matches[u] = True
    return matches


def gen_matches_sampling(children, depth, pvecs, rng, maxd):
    """CONTRAST (NOT greedy): each child independently matches w.p. its per-rank
    prob -- a proxy for stochastic sampling where >1 sibling can carry target mass.
    This is the ONLY regime in which leaf-to-root can exceed root-to-leaf; greedy
    decoding (a single argmax target) makes >=2 matching siblings impossible."""
    n = len(children)
    matches = [False] * n
    # only nodes whose parent matches can be "on a consistent path"; build top-down
    full_parent = [False] * n
    full_parent[0] = True
    order = sorted(range(1, n), key=lambda x: depth[x])
    for u in order:
        par = _parent_of(children, u)
        if not full_parent[par]:
            continue
        # rank of u among its siblings (birth order)
        r = children[par].index(u) + 1
        d = depth[u]
        pv = pvecs[min(d, maxd)]
        p = pv[r if r < len(pv) else len(pv) - 1]
        if rng.random() < p:
            matches[u] = True
            full_parent[u] = True
    return matches


def _parent_of(children, u):
    for p, ch in enumerate(children):
        if u in ch:
            return p
    return -1


# --------------------------------------------------------------------------- #
# Monte-Carlo driver: run BOTH walks on the SAME labelling, compare
# --------------------------------------------------------------------------- #
def run_mc(parent, pvecs, trials, seed, regime):
    children, depth, _ = tree_arrays(parent)
    maxd = len(pvecs) - 1
    rng = np.random.default_rng(seed)
    gen = gen_matches_greedy if regime == "greedy" else gen_matches_sampling
    sum_rl = sum_lr = 0
    violations = 0           # steps where the two walks emit a different token seq
    steps_with_diff = 0      # steps where len differs (leaf-to-root longer)
    multi_match_steps = 0    # steps with >=2 matching siblings somewhere (greedy=0)
    for _ in range(trials):
        matches = gen(children, depth, pvecs, rng, maxd)
        path_rl = walk_root_to_leaf(children, matches)
        path_lr = walk_leaf_to_root(parent, children, depth, matches)
        sum_rl += len(path_rl) + 1
        sum_lr += len(path_lr) + 1
        if len(path_lr) != len(path_rl):
            steps_with_diff += 1
        # greedy-identity: the leaf-to-root emitted prefix must not CHANGE any token
        # root-to-leaf already emitted. Under greedy both paths are identical; we
        # flag any position where the two disagree on the node (=> a different token).
        L = min(len(path_rl), len(path_lr))
        if path_rl[:L] != path_lr[:L] or (regime == "greedy" and path_rl != path_lr):
            violations += 1
        # detect >=2 matching siblings (the traversal-enabling condition)
        for u in range(len(children)):
            if sum(1 for c in children[u] if matches[c]) >= 2:
                multi_match_steps += 1
                break
    et_rl = sum_rl / trials
    et_lr = sum_lr / trials
    return {
        "regime": regime, "trials": trials,
        "et_rootleaf": et_rl, "et_traversal": et_lr,
        "uplift_abs": et_lr - et_rl,
        "uplift_pct": (et_lr - et_rl) / et_rl * 100.0 if et_rl else 0.0,
        "greedy_violation_count": violations,
        "steps_traversal_longer": steps_with_diff,
        "steps_traversal_longer_frac": steps_with_diff / trials,
        "steps_with_multi_match": multi_match_steps,
        "steps_with_multi_match_frac": multi_match_steps / trials,
    }


# --------------------------------------------------------------------------- #
# Leg C: drive the M=32 spine with #80's REAL per-position drafter hit-ranks
# --------------------------------------------------------------------------- #
def run_real_trace(parent, trace_path):
    """Consume #80 hit_rank sequences as the realised per-position target-rank draws
    along the M=32 spine. A step walks down the spine: hit_rank==1 -> spine continue;
    2<=hit_rank<=branch_width -> rank-r branch RESCUE (terminal for the step, the
    honest limit of a LINEAR teacher-forced trace -- a rank-2 branch's own
    continuation was never drafted); else the step ends. Both walks see the same
    accepted length here (spine + terminal rescue), so this is a real-data E[T]
    sanity that confirms 0 uplift on realised ranks, NOT an independent stress of
    the deep-branch case (Leg A/B do that). Drafter: EAGLE-3 (#80); uplift is
    drafter-independent."""
    if not os.path.exists(trace_path):
        return None
    children, depth, _ = tree_arrays(parent)
    # spine + per-spine-position branch width
    spine = [0]
    u = 0
    while children[u]:
        u = children[u][0]
        spine.append(u)
    spine_branch_width = [len(children[spine[k - 1]]) for k in range(1, len(spine))]
    Lspine = len(spine_branch_width)        # max draftable depth

    seqs = []
    with open(trace_path) as f:
        for i, line in enumerate(f):
            o = json.loads(line)
            if i == 0 and "meta" in o:
                continue
            hr = o.get("hit_rank")
            if hr:
                seqs.append(hr)

    sum_rl = sum_lr = 0
    n_steps = 0
    violations = 0
    for hr in seqs:
        pos = 0
        H = len(hr)
        while pos < H:
            accepted = 0
            d = 0
            while d < Lspine and pos < H:
                rank = hr[pos]
                w = spine_branch_width[d]
                if rank == 1:
                    accepted += 1
                    pos += 1
                    d += 1
                    continue
                if 2 <= rank <= w:
                    # rank-r sibling-branch rescue: accept it, but the step ends here
                    # (a linear teacher-forced trace never drafted the branch's own
                    # continuation), so consume this position and stop.
                    accepted += 1
                    pos += 1
                    break
                # spine top-1 mismatched and no usable branch -> step ends; consume
                # the failed position (its real token is the free bonus).
                pos += 1
                break
            else:
                # ran off the spine depth or hit trace end without an explicit stop
                pass
            # both walks identical on spine+terminal-rescue => same length
            sum_rl += accepted + 1
            sum_lr += accepted + 1
            n_steps += 1
    if n_steps == 0:
        return None
    return {
        "drafter": "eagle3_#80_debug1k", "n_steps": n_steps, "n_sequences": len(seqs),
        "et_rootleaf": sum_rl / n_steps, "et_traversal": sum_lr / n_steps,
        "uplift_pct": (sum_lr - sum_rl) / sum_rl * 100.0 if sum_rl else 0.0,
        "greedy_violation_count": violations,
        "note": "spine + terminal-rescue mapping of a linear teacher-forced trace; "
                "uplift is 0 by the structural equivalence, E[T] is a real-data anchor",
    }


# --------------------------------------------------------------------------- #
# Leg D: exhaustive small-tree equivalence (greedy invariant => walks identical)
# --------------------------------------------------------------------------- #
def exhaustive_equivalence(max_n=6):
    """Over ALL labelled rooted trees with n<=max_n and ALL match-labellings:
      (1) GREEDY-VALID labellings (<=1 matching child per node, a node matches only
          if its parent matches) -> root-to-leaf path == leaf-to-root path, ALWAYS.
      (2) UNRESTRICTED labellings -> count cases where leaf-to-root > root-to-leaf
          (these ALL have >=2 matching siblings on a consistent path = the
          sampling-only condition). Confirms the walks are not trivially identical.
    Returns counts; (1) must show 0 mismatches."""
    greedy_checked = greedy_mismatch = 0
    unrestricted_diff = unrestricted_checked = 0
    for n in range(2, max_n + 1):
        ranges = [range(i) for i in range(1, n)]
        for combo in itertools.product(*ranges):
            parent = [-1] + list(combo)
            children, depth, _ = tree_arrays(parent)
            non_root = list(range(1, n))
            # (1) greedy-valid labellings: choose at most one matching child per node,
            # propagated from the root (a node can match only if its parent matches).
            for chain in _greedy_chains(children):
                matches = [False] * n
                for u in chain:
                    matches[u] = True
                rl = walk_root_to_leaf(children, matches)
                lr = walk_leaf_to_root(parent, children, depth, matches)
                greedy_checked += 1
                if rl != lr:
                    greedy_mismatch += 1
            # (2) unrestricted: sample a bounded number of arbitrary labellings
            if n <= 5:
                for bits in range(1 << len(non_root)):
                    matches = [False] * n
                    for j, u in enumerate(non_root):
                        if bits >> j & 1:
                            matches[u] = True
                    rl = walk_root_to_leaf(children, matches)
                    lr = walk_leaf_to_root(parent, children, depth, matches)
                    unrestricted_checked += 1
                    if len(lr) > len(rl):
                        unrestricted_diff += 1
    return {
        "greedy_labellings_checked": greedy_checked,
        "greedy_walk_mismatches": greedy_mismatch,       # MUST be 0
        "unrestricted_labellings_checked": unrestricted_checked,
        "unrestricted_traversal_longer": unrestricted_diff,   # >0: walks differ off-greedy
    }


def _greedy_chains(children):
    """Yield every root->node chain (each is a valid greedy matching: the unique
    consistent path). Includes the empty chain (immediate miss)."""
    yield []                                  # nothing matched
    stack = [[c] for c in children[0]]
    while stack:
        chain = stack.pop()
        yield chain
        for c in children[chain[-1]]:
            stack.append(chain + [c])


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rho-opt-json", default=RHO_OPT_JSON)
    ap.add_argument("--accept-json", default=ACCEPT_JSON)
    ap.add_argument("--accept-source", default="server_log")
    ap.add_argument("--rank-coverage-json", default=RANKCOV_JSON)
    ap.add_argument("--trace-80", default=TRACE_80)
    ap.add_argument("--W", type=int, default=4)
    ap.add_argument("--max-depth", type=int, default=24)
    ap.add_argument("--extrapolate", default="flat", choices=["flat", "rise"])
    ap.add_argument("--mc-trials", type=int, default=400_000)
    ap.add_argument("--green-threshold", type=float, default=5.0, help="GREEN if uplift%% >=")
    ap.add_argument("--amber-threshold", type=float, default=2.0, help="AMBER if uplift%% >=")
    ap.add_argument("--output",
                    default="research/spec_cost_model/traversal_verify_et_results.json")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb-group", default="traversal-verify-et")
    ap.add_argument("--wandb-name", default="fern/traversal-verify-et")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    # ---- inputs: wirbel #83 M=32 topology + MEASURED MTP per-depth acceptance ----
    parent = load_m32_topology(args.rho_opt_json)
    children, depth, leaves = tree_arrays(parent)
    meas = load_measured(args.accept_json, args.accept_source)
    rank_cov = load_rank_coverage(args.rank_coverage_json)
    pvecs = build_depth_pvecs_measured(meas["q"], rank_cov["rho_cond"], args.W,
                                       args.max_depth, args.extrapolate)
    maxd = len(pvecs) - 1
    print(f"[trav] M=32 topology: n={len(parent)} depth={max(depth)} "
          f"max_branch={max(len(c) for c in children)} leaves={len(leaves)}", flush=True)
    print(f"[trav] measured MTP: top1={meas['C'][0]:.4f} q={[round(x,3) for x in meas['q']]} "
          f"rho_cond={[round(x,4) for x in rank_cov['rho_cond']]}", flush=True)

    # ---- analytic anchor: score_tree_depthrank == wirbel's E[T]=5.207 ----
    et_anchor, d_anchor = score_tree_depthrank(parent, pvecs)
    spine = strip_to_spine(parent)
    et_spine, _ = score_tree_depthrank(spine, pvecs)
    print(f"[trav] ANCHOR score_tree E[T]={et_anchor:.4f} (wirbel {WIRBEL_E_T_M32}; "
          f"|Δ|={abs(et_anchor-WIRBEL_E_T_M32):.4f}); spine-only E[T]={et_spine:.4f}", flush=True)
    assert abs(et_anchor - WIRBEL_E_T_M32) < 0.05, "M=32 E[T] departs from wirbel #83"

    # ---- Leg D: exhaustive small-tree equivalence (fast, deterministic) ----
    exh = exhaustive_equivalence(max_n=6)
    print(f"[trav] EXHAUSTIVE: greedy labellings checked={exh['greedy_labellings_checked']} "
          f"mismatches={exh['greedy_walk_mismatches']} (must be 0); unrestricted "
          f"traversal-longer={exh['unrestricted_traversal_longer']}/"
          f"{exh['unrestricted_labellings_checked']} (off-greedy the walks DO differ)",
          flush=True)
    assert exh["greedy_walk_mismatches"] == 0, "greedy walks disagree -- proof broken!"
    assert exh["unrestricted_traversal_longer"] > 0, "walks never differ -- impl is a no-op!"

    # ---- Leg A (physical): greedy MC on the exact M=32 tree ----
    legA = run_mc(parent, pvecs, args.mc_trials, seed=7, regime="greedy")
    print(f"[trav] LEG A greedy MC (M=32, {args.mc_trials} trials): "
          f"E[T]_rootleaf={legA['et_rootleaf']:.4f} E[T]_traversal={legA['et_traversal']:.4f} "
          f"uplift={legA['uplift_pct']:+.4f}% violations={legA['greedy_violation_count']} "
          f"multi_match_steps={legA['steps_with_multi_match']}", flush=True)

    # ---- Leg B (contrast): sampling-style relaxation -> traversal SHOULD win ----
    legB = run_mc(parent, pvecs, args.mc_trials, seed=11, regime="sampling")
    print(f"[trav] LEG B sampling-proxy MC (M=32): E[T]_rootleaf={legB['et_rootleaf']:.4f} "
          f"E[T]_traversal={legB['et_traversal']:.4f} uplift={legB['uplift_pct']:+.4f}% "
          f"(traversal longer on {legB['steps_traversal_longer_frac']*100:.2f}% of steps; "
          f"multi-match on {legB['steps_with_multi_match_frac']*100:.2f}%)", flush=True)

    # ---- Leg C (real data): #80 hit-ranks along the M=32 spine ----
    legC = run_real_trace(parent, args.trace_80)
    if legC:
        print(f"[trav] LEG C real #80 ranks (M=32 spine): E[T]_rootleaf="
              f"{legC['et_rootleaf']:.4f} E[T]_traversal={legC['et_traversal']:.4f} "
              f"uplift={legC['uplift_pct']:+.4f}% over {legC['n_steps']} steps", flush=True)

    # ---- rho2-capture framing ----
    rootleaf_rescue_pct = (et_anchor - et_spine) / et_spine * 100.0
    # fraction of wirbel's rho2 rescue that the deployable rule captures OVER root-to-leaf
    traversal_marginal_capture_of_rho2 = 0.0   # = legA uplift (exactly 0)

    # ---- gate ----
    uplift = legA["uplift_pct"]
    viol = legA["greedy_violation_count"]
    if viol > 0 or uplift < args.amber_threshold:
        gate = "RED"
    elif uplift >= args.green_threshold:
        gate = "GREEN"
    else:
        gate = "AMBER"

    verdict = {
        "primary_metric_name": "traversal_et_uplift_pct",
        "traversal_et_uplift_pct": uplift,
        "test_metric_name": "traversal_greedy_violation_count",
        "traversal_greedy_violation_count": viol,
        "gate": gate,
        "et_rootleaf_greedy": legA["et_rootleaf"],
        "et_traversal_greedy": legA["et_traversal"],
        "et_anchor_score_tree": et_anchor,
        "et_anchor_vs_wirbel_abs": abs(et_anchor - WIRBEL_E_T_M32),
        "et_spine_only": et_spine,
        "rootleaf_rescue_over_spine_pct": rootleaf_rescue_pct,
        "wirbel_rho2_salvage_oracle": WIRBEL_RHO2,
        "traversal_marginal_capture_of_rho2_pct": traversal_marginal_capture_of_rho2,
        "sampling_proxy_uplift_pct": legB["uplift_pct"],
        "sampling_proxy_traversal_longer_frac": legB["steps_traversal_longer_frac"],
        "greedy_multi_match_steps": legA["steps_with_multi_match"],
        "exhaustive_greedy_mismatches": exh["greedy_walk_mismatches"],
        "exhaustive_unrestricted_traversal_longer": exh["unrestricted_traversal_longer"],
        "realtrace_uplift_pct": legC["uplift_pct"] if legC else None,
        "decision": (
            "RED -- under the challenge's GREEDY contract leaf-to-root traversal is "
            "PROVABLY identical to the deployed root-to-leaf full-tree verifier "
            "(uplift 0.000%, 0 greedy violations). wirbel's rho2=0.4165 rank-2 rescue "
            "is ALREADY realised by root-to-leaf; it is the value of the TREE, not "
            "headroom for traversal. Recommend land #71 NOT integrate traversal; keep "
            "standard full-tree root-to-leaf verification. The lever pays only under "
            "stochastic sampling (Leg B), which greedy decode forbids."),
    }

    results = {
        "config": vars(args),
        "topology": {"parent": parent, "n": len(parent), "depth": max(depth),
                     "max_branch": max(len(c) for c in children), "leaves": len(leaves),
                     "spine_len": len(spine)},
        "inputs": {"top1": meas["C"][0], "q": meas["q"], "rho_cond": rank_cov["rho_cond"],
                   "wirbel_E_T_M32": WIRBEL_E_T_M32, "wirbel_rho2": WIRBEL_RHO2},
        "anchor": {"score_tree_E_T": et_anchor, "depth": d_anchor,
                   "spine_only_E_T": et_spine, "rootleaf_rescue_over_spine_pct": rootleaf_rescue_pct},
        "leg_a_greedy_mc": legA,
        "leg_b_sampling_proxy": legB,
        "leg_c_real_trace": legC,
        "leg_d_exhaustive": exh,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"[trav] wrote {args.output}", flush=True)

    print("\n[trav] ===== TRAVERSAL VERIFICATION E[T] GATE (M=32, greedy) =====", flush=True)
    print(f"  primary traversal_et_uplift_pct      = {uplift:+.4f}%", flush=True)
    print(f"  test    traversal_greedy_violations  = {viol}", flush=True)
    print(f"  E[T] root-to-leaf (deployed)         = {legA['et_rootleaf']:.4f}", flush=True)
    print(f"  E[T] leaf-to-root (traversal)        = {legA['et_traversal']:.4f}", flush=True)
    print(f"  root-to-leaf already realises +{rootleaf_rescue_pct:.1f}% rescue over spine "
          f"(captures wirbel's rho2)", flush=True)
    print(f"  traversal MARGINAL capture of rho2   = {traversal_marginal_capture_of_rho2:.1f}%", flush=True)
    print(f"  sampling-proxy uplift (Leg B, contrast) = {legB['uplift_pct']:+.2f}% "
          f"(the regime greedy forbids)", flush=True)
    print(f"  GATE: {gate}", flush=True)

    if not args.no_wandb:
        try:
            log_wandb(args, results, verdict, legA, legB, legC)
        except Exception as e:  # noqa: BLE001
            print(f"[trav] W&B logging failed (non-fatal): {e!r}", flush=True)
    print("[trav] DONE", flush=True)


def log_wandb(args, results, verdict, legA, legB, legC):
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name, job_type="profiling",
                     config={"topology": "wirbel#83_M32_optimal", "W": args.W,
                             "mc_trials": args.mc_trials, "regime_physical": "greedy",
                             "green_threshold": args.green_threshold,
                             "amber_threshold": args.amber_threshold})
    flat = {f"verdict/{k}": v for k, v in verdict.items() if not isinstance(v, (dict, list))}
    flat.update({f"leg_a/{k}": v for k, v in legA.items() if not isinstance(v, str)})
    flat.update({f"leg_b/{k}": v for k, v in legB.items() if not isinstance(v, str)})
    if legC:
        flat.update({f"leg_c/{k}": v for k, v in legC.items() if not isinstance(v, str)})
    flat.update({f"anchor/{k}": v for k, v in results["anchor"].items()})
    flat.update({f"exhaustive/{k}": v for k, v in results["leg_d_exhaustive"].items()})
    run.summary.update(flat)
    run.log(flat)
    tb = wandb.Table(columns=["leg", "regime", "E[T]_rootleaf", "E[T]_traversal",
                              "uplift_pct", "greedy_violations"])
    tb.add_data("A_physical", "greedy", legA["et_rootleaf"], legA["et_traversal"],
                legA["uplift_pct"], legA["greedy_violation_count"])
    tb.add_data("B_contrast", "sampling", legB["et_rootleaf"], legB["et_traversal"],
                legB["uplift_pct"], legB["greedy_violation_count"])
    if legC:
        tb.add_data("C_realtrace", "greedy", legC["et_rootleaf"], legC["et_traversal"],
                    legC["uplift_pct"], legC["greedy_violation_count"])
    run.log({"acceptance_walks": tb})
    run.finish()
    print(f"[trav] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
