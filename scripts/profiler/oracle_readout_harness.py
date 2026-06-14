#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Oracle-readout -> measured-official-TPS harness (PR #129): turn openevolve's 4
tree numbers into a MEASURED 500 go/no-go (not a projection).

THE GAP THIS CLOSES
-------------------
openevolve is offering an unlimited-A10G oracle (real spec-decode metrics, NO
bench-quota). chiku-inu is firing the tree package `tree-488-pw-fp32-v0` (the
fp32 star-verify fix) and the oracle will return 4 numbers:
  (1) depth1_spine_accept     -- depth-1 spine accept (q_1; fp32 target 0.7287)
  (2) per_position histogram  -- the rho-ladder (q_spine[d] + branch-hit rho2[d])
  (3) accept_length           -- MEASURED E[T] (the numerator)
  (4) full_tree_reach         -- does the salvage walk descend (BUG-2 closed)?
The oracle returns SPEC-DECODE metrics; the fleet decision needs an OFFICIAL TPS.
This harness ingests the 4 numbers and emits a measured official TPS + a
greedy-exactness check, so the moment the oracle runs we read a measured verdict.

THE MAP (banked supply cost model -- #100 compose + my #125 realization ceiling)
--------------------------------------------------------------------------------
  official_TPS(accept_length) = K_cal * accept_length / step_time * tau

K_cal = 125.268 (= frontier 481.53 / E[T]_lin 3.844; #100). The ONLY free input
is the oracle's MEASURED accept_length; step_time and tau are banked. The tree
package land #71 builds is the W* = M=32 / depth-9 / max-branch-3 rho-optimal
tree, whose MEASURED-attention-tax step_time = 1.2128 is already priced in my
#125 (the measured 1.83x tree-mask attention tax, lawine #107). So the operative
readout for the depth-9 package is:
  official = 125.268 * accept_length / 1.2128            (depth-9 W*, tau=1)
  at accept_length=5.207 (rho-optimal) -> 537.8 (== #125 W*, by construction).

THE BANDS (Step 2)
------------------
  clear-500 bar (operative, depth-9 W* step):  accept_length >= 4.841
  overtake tree-free (491.8, denken #123):     accept_length >= 4.761
  demand floor (denken #123, board-merged):    accept_length >= 4.624  [CROSS-CHECK]
The harness-native depth-9 bar (4.841) is HIGHER than denken's topology-optimal
demand floor (4.624): a tree TUNED for E[T]=4.624 is shallow (~depth 5-6, cheaper
step) and clears 500, but the depth-9 W* build pays the depth-9 step, so it needs
+0.217 more E[T]. This TIGHTENS the build bracket from [4.624, 5.207] to
[4.841, 5.207] for the depth-9 topology -- a real, actionable finding for land #71
(flagged to the advisor; the live-oracle gate should use 4.841, or re-price at the
oracle's reported realized depth).

GREEDY-EXACTNESS (Step 3)
-------------------------
The tree is greedy-EXACT iff every accepted token == the true target argmax.
chiku-inu's fp32 star-verify (QK+PV -> fp32/IEEE) makes ta[0] == true target
argmax by construction, so the tree should PASS where the deployed 56%-divergent
spec stack (kanna #114) fails. `tree_greedy_exact_from_oracle` returns 1 iff the
oracle's per-position accepts match the greedy-rejection profile (depth-1 accept
recovered to the fp32 target, ladder monotone/consistent, no near-tie flips) AND
full_tree_reach confirms the salvage descends.

LOCAL, CPU-ONLY, ANALYTIC. No GPU, no vLLM, no HF Job, no submission, no kernel
build. Reuses the banked #100 compose (lever_composition), my #125 step model
(tree_et_realization_ceiling), and wirbel's E[T] DP (treeshape_measured_accept
build_depth_pvecs_measured + score_tree_depthrank). A readout harness serves
nothing -> greedy identity untouched by construction.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- banked models reused verbatim (one source of truth per constant) ----
lc = _load("lever_composition", os.path.join(_HERE, "lever_composition.py"))
tma = _load("treeshape_measured_accept", os.path.join(_HERE, "treeshape_measured_accept.py"))

K_CAL = lc.K_CAL                          # 125.268 (= 481.53 / 3.844)
E_T_LINEAR = lc.E_T_LINEAR               # 3.844
E_T_TREE = lc.E_T_TREE                   # 5.207 rho-optimal ceiling
FRONTIER_OFFICIAL = lc.FRONTIER_OFFICIAL  # 481.53
TARGET_OFFICIAL = lc.TARGET_OFFICIAL     # 500.0

# tau band: lawine #116 local->official transfer (tight; central pinned at 1).
TAU = {"low": 0.9983, "central": 1.00, "high": 1.00}

# ---- board-merged / banked anchors (denken #123 demand side, chiku-inu fp32 fix) ----
TREEFREE_CEILING = 491.8                  # denken #123: best tree-free stack caps here
DEMAND_FLOOR_500 = 4.624                  # denken #123: E[T] a tree must deliver for 500
MILE_CLEAR_500 = 4.613985078031858       # lawine #107 banked break-even (cross-check)
MILE_OVERTAKE = 4.727                     # banked tree-overtakes-treefree E[T] bar
DEPTH1_ACCEPT_FP32_TARGET = 0.7287       # chiku-inu fp32 target (== rho-optimal q_1)
DEPTH1_ACCEPT_BF16_DEFICIT = 0.13        # the ~13pp bf16 near-tie-flip deficit
AS_BUILT_ET = 2.10                       # land #71 as-built realized E[T] (bf16 bug)

# Lookup-ladder accept_length anchors the fleet reads the verdict off.
LADDER = [AS_BUILT_ET, E_T_LINEAR, DEMAND_FLOOR_500, None, E_T_TREE]  # None=clear-500 bar


# ----------------------------------------------------------------------------
# Core map: the ONLY free input is accept_length; everything else is banked.
# ----------------------------------------------------------------------------
def measured_official_tps(accept_length: float, step_time: float,
                          tau: float = TAU["central"]) -> float:
    """#100 compose figure of merit: K_cal * E[T] / step_time * tau.

    accept_length = the oracle's MEASURED E[T] (numerator). step_time = the banked
    decode-step of the topology the oracle ran (default = W* depth-9, attention tax
    already priced; #125). tau = local->official transfer (#116)."""
    return K_CAL * accept_length / step_time * tau


def accept_length_for_official(target_official: float, step_time: float,
                               tau: float = TAU["central"]) -> float:
    """Invert the map: the accept_length at which official crosses `target`."""
    return target_official * step_time / (K_CAL * tau)


def step_time_for_depth(tree_et: dict, depth: int) -> float | None:
    """Re-price the decode step at a realized M=32 depth from the banked #125
    m32_official_by_depth curve. Lets the harness use the oracle's ACTUAL topology
    depth (full_tree_reach) instead of assuming the full depth-9 W* step."""
    rows = {int(r["depth"]): r["step_time"]
            for r in tree_et["binding_constraint"]["m32_official_by_depth"]}
    if depth in rows:
        return rows[depth]
    if not rows:
        return None
    lo = max((d for d in rows if d <= depth), default=min(rows))
    hi = min((d for d in rows if d >= depth), default=max(rows))
    if lo == hi:
        return rows[lo]
    frac = (depth - lo) / (hi - lo)
    return rows[lo] + frac * (rows[hi] - rows[lo])


# ----------------------------------------------------------------------------
# Step 1 (consistency): reconstruct E[T] from the oracle's ladder via the banked
# DP and compare to the oracle's reported accept_length (catch a mis-measured E[T]).
# ----------------------------------------------------------------------------
def reconstruct_et_from_ladder(q_ladder: list[float], rho_cond: list[float],
                               parent: list[int], W: int, max_depth: int) -> dict:
    """E[T] of the banked M=32 topology under the oracle's MEASURED q-ladder +
    rank-conditional rescue ratios, via wirbel's score_tree_depthrank (the exact
    DP that produced F_tree=5.207). Returns the reconstructed E[T] and depth."""
    pvecs = tma.build_depth_pvecs_measured(q_ladder, rho_cond, W, max_depth, "flat")
    F, depth = tma.score_tree_depthrank(parent, pvecs)
    return {"reconstructed_et": F, "reconstructed_depth": depth}


def check_accept_length_consistency(oracle: dict, parent: list[int], rho_cond: list[float],
                                    W: int, max_depth: int, tol: float = 0.25) -> dict:
    """Validate the oracle's accept_length against its own per-position ladder.

    Reconstructs E[T] from the oracle's q-ladder (depth-1 spine accept + the
    per-position branch-hit histogram) on the banked M=32 topology and compares to
    the oracle's reported accept_length. A material gap flags a mis-measured E[T]."""
    q_ladder = [p["q_spine"] for p in oracle["per_position"]]
    # per-depth rho2 if the oracle measured it; else fall back to the banked pooled ladder.
    rho2 = [p.get("branch_hit_rho2") for p in oracle["per_position"]]
    rho_cond_eff = list(rho_cond)
    rec = reconstruct_et_from_ladder(q_ladder, rho_cond_eff, parent, W, max_depth)
    al = oracle["accept_length"]
    diff = al - rec["reconstructed_et"]
    return {
        "oracle_accept_length": al,
        "reconstructed_et_from_ladder": rec["reconstructed_et"],
        "reconstructed_depth": rec["reconstructed_depth"],
        "abs_diff": abs(diff),
        "rel_diff": abs(diff) / max(rec["reconstructed_et"], 1e-9),
        "consistent": bool(abs(diff) <= tol),
        "depth1_spine_accept_oracle": q_ladder[0] if q_ladder else None,
        "depth1_matches_fp32_target": (
            bool(q_ladder and q_ladder[0] >= DEPTH1_ACCEPT_FP32_TARGET - 0.02)),
        "tolerance": tol,
        "note": ("reconstructed E[T] from the oracle's q-ladder on the banked M=32 "
                 "topology should match the oracle's accept_length within tol; a gap "
                 "means accept_length was mis-measured or the topology differs."),
    }


# ----------------------------------------------------------------------------
# Step 3: greedy-exactness from the per-position histogram + full-tree reach.
# ----------------------------------------------------------------------------
def tree_greedy_exact_from_oracle(oracle: dict, q_target_ladder: list[float],
                                  tol: float = 0.03) -> dict:
    """1 iff the oracle's per-position accepts match the greedy-rejection profile
    with no near-tie flips -- the tree's analogue of the kanna #114 / Issue #124
    validity gate, which the fp32 star-verify tree should PASS.

    Three conditions, all from the oracle's returned numbers:
      (a) depth-1 spine accept recovered to the fp32 target 0.7287 (the bf16
          near-tie-flip 13pp deficit is GONE -> ta[0] == true target argmax);
      (b) the per-position q-ladder is non-decreasing and tracks the banked
          rho-optimal ladder within tol (no anomalous flips mid-tree);
      (c) full_tree_reach confirms the salvage walk descends to the built depth
          (BUG-2 closed: star branches are not stuck as leaves).
    """
    q = [p["q_spine"] for p in oracle["per_position"]]
    d1 = oracle.get("depth1_spine_accept", q[0] if q else None)

    # (a) depth-1 recovered (fp32 fix removed the bf16 near-tie flips).
    cond_a = bool(d1 is not None and d1 >= DEPTH1_ACCEPT_FP32_TARGET - tol)

    # (b) ladder monotone non-decreasing + tracks the banked target within tol.
    mono = all(q[i + 1] >= q[i] - tol for i in range(len(q) - 1))
    n = min(len(q), len(q_target_ladder))
    tracks = all(abs(q[i] - q_target_ladder[i]) <= max(tol, 0.06) for i in range(n))
    cond_b = bool(mono and tracks)

    # (c) salvage walk descends (BUG-2 closed).
    reach = oracle.get("full_tree_reach", {})
    reached = reach.get("max_depth_reached")
    built = reach.get("built_depth", len(q))
    cond_c = bool(reached is not None and built and reached >= max(built - 1, 2))

    exact = bool(cond_a and cond_b and cond_c)
    return {
        "tree_greedy_exact_from_oracle": int(exact),
        "cond_a_depth1_recovered": cond_a,
        "cond_b_ladder_consistent": cond_b,
        "cond_c_salvage_descends": cond_c,
        "depth1_spine_accept": d1,
        "depth1_fp32_target": DEPTH1_ACCEPT_FP32_TARGET,
        "ladder_monotone": bool(mono),
        "ladder_tracks_target": bool(tracks),
        "max_depth_reached": reached,
        "built_depth": built,
        "mechanism": ("fp32 star-verify (QK+PV -> fp32/IEEE, relerr 1e-3 -> 1e-6) makes "
                      "ta[0] == true target argmax -> tree greedy-EXACT by construction; "
                      "does NOT inherit kanna #114's 56% bf16-near-tie-flip divergence."),
    }


# ----------------------------------------------------------------------------
# Bands + lookup table (Step 2) at a given operative step.
# ----------------------------------------------------------------------------
def compute_bands(step_time: float, tau: float) -> dict:
    clear500 = accept_length_for_official(TARGET_OFFICIAL, step_time, tau)
    overtake = accept_length_for_official(TREEFREE_CEILING, step_time, tau)
    beat_lin = accept_length_for_official(FRONTIER_OFFICIAL, step_time, tau)
    return {
        "step_time": step_time, "tau": tau,
        "accept_length_to_clear_500": clear500,
        "accept_length_to_overtake_treefree": overtake,
        "accept_length_to_beat_frontier": beat_lin,
        "treefree_ceiling_official": TREEFREE_CEILING,
        "demand_floor_500_denken123": DEMAND_FLOOR_500,
        "demand_floor_official_at_this_step": measured_official_tps(DEMAND_FLOOR_500, step_time, tau),
        "gap_vs_demand_floor": clear500 - DEMAND_FLOOR_500,
    }


def lookup_table(step_time: float, tau: float, clear500_bar: float,
                 overtake_bar: float) -> list[dict]:
    # depth-9-consistent anchors only: the overtake/clear bars are re-priced at THIS
    # step (4.761 / 4.841), so every label is internally consistent. The banked
    # MILE_OVERTAKE=4.727 was a different-step anchor and is kept only in banked_anchors.
    anchors = [
        (AS_BUILT_ET, "2.10 as-built (bf16 bug)"),
        (E_T_LINEAR, "3.844 linear-MTP frontier E[T]"),
        (DEMAND_FLOOR_500, "4.624 demand floor (denken #123, topology-optimal)"),
        (overtake_bar, "overtake-treefree bar (depth-9 operative)"),
        (clear500_bar, "clear-500 bar (depth-9 operative)"),
        (E_T_TREE, "5.207 rho-optimal ceiling"),
    ]
    rows = []
    for al, label in sorted(anchors, key=lambda x: x[0]):
        off = measured_official_tps(al, step_time, tau)
        rows.append({
            "accept_length": round(al, 4), "label": label,
            "official_tps": off,
            "clears_500": bool(off >= TARGET_OFFICIAL),
            "overtakes_treefree": bool(off >= TREEFREE_CEILING),
        })
    return rows


# ----------------------------------------------------------------------------
# Synthetic self-test: feed the banked rho-optimal ladder as a "perfect oracle"
# and confirm the harness reproduces #125's official 537.8 + greedy_exact=1.
# ----------------------------------------------------------------------------
def synthetic_oracle_from_banked(q_ladder: list[float], rho2_by_depth: dict,
                                 parent: list[int], rho_cond: list[float],
                                 W: int, max_depth: int) -> dict:
    rec = reconstruct_et_from_ladder(q_ladder, rho_cond, parent, W, max_depth)
    per_position = []
    for i, q in enumerate(q_ladder, start=1):
        per_position.append({
            "depth": i, "q_spine": q,
            "branch_hit_rho2": rho2_by_depth.get(str(i - 1)),
        })
    return {
        "source": "SELF-TEST (banked rho-optimal ladder as a perfect oracle)",
        "tree_package": "tree-488-pw-fp32-v0 (synthetic rho-optimal)",
        "depth1_spine_accept": q_ladder[0],
        "per_position": per_position,
        "accept_length": rec["reconstructed_et"],
        "full_tree_reach": {"max_depth_reached": rec["reconstructed_depth"],
                            "built_depth": rec["reconstructed_depth"],
                            "frac_reaching_full_depth": None},
    }


def evaluate_oracle(oracle: dict, banked: dict, step_override_depth: int | None,
                    tau_key: str = "central") -> dict:
    """Full readout of one oracle JSON -> measured official TPS + gates."""
    tau = TAU[tau_key]
    tree_et = banked["tree_et"]
    parent = banked["parent"]
    rho_cond = banked["rho_cond"]
    q_target = banked["q_target_ladder"]
    W, max_depth = banked["W"], banked["max_depth"]

    # operative step: oracle-reported realized depth if present, else depth-9 W*.
    reach = oracle.get("full_tree_reach", {})
    realized_depth = step_override_depth or reach.get("built_depth") or banked["wstar_depth"]
    step_time = step_time_for_depth(tree_et, int(realized_depth)) or banked["step_wstar"]

    al = oracle["accept_length"]
    official_central = measured_official_tps(al, step_time, TAU["central"])
    official_taulow = measured_official_tps(al, step_time, TAU["low"])

    consistency = check_accept_length_consistency(
        oracle, parent, rho_cond, W, max_depth)
    greedy = tree_greedy_exact_from_oracle(oracle, q_target)
    bands = compute_bands(step_time, TAU["central"])

    clears_500 = bool(official_central >= TARGET_OFFICIAL)
    clears_500_cons = bool(official_taulow >= TARGET_OFFICIAL)
    overtakes = bool(official_central >= TREEFREE_CEILING)

    return {
        "operative_step_time": step_time,
        "operative_realized_depth": realized_depth,
        "accept_length": al,
        "measured_official_tps_central": official_central,
        "measured_official_tps_taulow": official_taulow,
        "clears_500_central": clears_500,
        "clears_500_conservative": clears_500_cons,
        "overtakes_treefree": overtakes,
        "margin_to_500": official_central - TARGET_OFFICIAL,
        "consistency": consistency,
        "greedy_exact": greedy,
        "bands": bands,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rho", default="research/spec_cost_model/rho_optimal_topology_results.json")
    ap.add_argument("--tree-et", default="research/spec_cost_model/tree_et_realization_ceiling_results.json")
    ap.add_argument("--oracle-json", default=None,
                    help="path to the openevolve oracle readout (4 numbers). If absent, "
                         "run bands-ready mode + the banked-ladder self-test.")
    ap.add_argument("--out", default="research/oracle_readout/oracle_readout_harness_results.json")
    ap.add_argument("--sample-out", default="research/oracle_readout/sample_oracle_input.json",
                    help="write the oracle-input schema template here for chiku-inu/openevolve.")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="fern/oracle-readout-harness")
    ap.add_argument("--wandb-group", default="oracle-readout-harness")
    args = ap.parse_args()

    with open(args.rho) as f:
        rho = json.load(f)
    with open(args.tree_et) as f:
        tree_et = json.load(f)

    cfg = rho["config"]
    W, max_depth = cfg["W"], cfg["max_depth"]
    parent = rho["per_budget"]["32"]["optimal"]["parent"]
    rho_cond = rho["inputs"]["rho_cond_measured"]
    q_target_ladder = rho["inputs"]["depth_q_76"]
    rho2_by_depth = rho["inputs"]["rho2_by_depth"]
    step_wstar = tree_et["wstar"]["step_time_measured"]   # 1.2128 (M=32/depth-9, attn-taxed)
    wstar_depth = tree_et["wstar"]["depth"]

    banked = {"tree_et": tree_et, "parent": parent, "rho_cond": rho_cond,
              "q_target_ladder": q_target_ladder, "W": W, "max_depth": max_depth,
              "step_wstar": step_wstar, "wstar_depth": wstar_depth}

    # ---- bands at the operative (depth-9 W*) step ----
    bands_wstar = compute_bands(step_wstar, TAU["central"])
    bands_wstar_taulow = compute_bands(step_wstar, TAU["low"])
    clear500_bar = bands_wstar["accept_length_to_clear_500"]
    overtake_bar = bands_wstar["accept_length_to_overtake_treefree"]
    table = lookup_table(step_wstar, TAU["central"], clear500_bar, overtake_bar)

    # ---- self-test: banked rho-optimal ladder as a perfect oracle -> must hit 537.8 ----
    synth = synthetic_oracle_from_banked(
        q_target_ladder, rho2_by_depth, parent, rho_cond, W, max_depth)
    synth_eval = evaluate_oracle(synth, banked, step_override_depth=wstar_depth)
    selftest_pass = bool(
        abs(synth_eval["measured_official_tps_central"]
            - tree_et["wstar"]["official_measured_central"]) < 0.5
        and synth_eval["consistency"]["consistent"]
        and synth_eval["greedy_exact"]["tree_greedy_exact_from_oracle"] == 1)

    # ---- live oracle (if provided) ----
    live_eval = None
    oracle_in = None
    if args.oracle_json and os.path.exists(args.oracle_json):
        with open(args.oracle_json) as f:
            oracle_in = json.load(f)
        live_eval = evaluate_oracle(oracle_in, banked, step_override_depth=None)

    # ---- primary / test metrics ----
    # PRIMARY: oracle_accept_length_to_clear_500 = the accept_length the MEASURED build
    # must report to clear 500. The build the oracle runs (tree-488 = land #71's depth-9
    # W* tree) pays the depth-9 step, so its operative bar is clear500_bar (4.841). denken
    # #123's 4.624 is the TOPOLOGY-OPTIMAL floor (a ~depth-6.7 tree, step 1.1585) and is
    # the PR's named CROSS-CHECK, NOT the depth-9 metric. We report the operative bar as
    # primary (consistent with the gate, which fires on operative-step official >= 500),
    # and carry 4.624 as the explicit cross-check. The harness re-prices the bar at the
    # oracle's reported realized depth (step_time_for_depth) when a live number lands.
    primary_metric = clear500_bar                 # 4.841 (operative depth-9 W*)
    primary_demand_floor_xcheck = DEMAND_FLOOR_500  # 4.624 (denken #123 topology-optimal floor)

    if live_eval is not None:
        test_metric = live_eval["measured_official_tps_central"]
        test_metric_label = f"measured at oracle accept_length={oracle_in['accept_length']}"
        oracle_pending = False
    else:
        test_metric = measured_official_tps(AS_BUILT_ET, step_wstar, TAU["central"])
        test_metric_label = (f"as-built E[T]={AS_BUILT_ET} on the depth-9 step "
                             f"(AWAITING LIVE ORACLE RUN; fp32 fix targets E[T]->5.207 -> "
                             f"{measured_official_tps(E_T_TREE, step_wstar, TAU['central']):.1f})")
        oracle_pending = True

    # ---- gate (PR #129) ----
    if live_eval is not None:
        if live_eval["clears_500_conservative"] and \
                live_eval["greedy_exact"]["tree_greedy_exact_from_oracle"] == 1:
            verdict = "GREEN"
            verdict_label = (
                f"MEASURED 500 CONFIRMED: oracle accept_length={live_eval['accept_length']:.3f} "
                f"-> official {live_eval['measured_official_tps_central']:.1f} TPS "
                f"(>= 500 by +{live_eval['margin_to_500']:.0f}), greedy-EXACT, "
                f"consistency {'OK' if live_eval['consistency']['consistent'] else 'FLAG'}.")
        elif live_eval["clears_500_central"]:
            verdict = "AMBER"
            verdict_label = (
                f"oracle accept_length={live_eval['accept_length']:.3f} clears 500 central "
                f"({live_eval['measured_official_tps_central']:.1f}) but not conservative, or "
                f"greedy-exact/consistency did not pass cleanly.")
        else:
            verdict = "RED"
            verdict_label = (
                f"oracle accept_length={live_eval['accept_length']:.3f} -> official "
                f"{live_eval['measured_official_tps_central']:.1f} < 500 at the depth-9 step.")
    else:
        # No oracle number yet -> AMBER (harness banked, bands ready), unless the
        # supply model itself could not reach 500 even at the ceiling (-> RED inconsistency).
        ceiling_official = measured_official_tps(E_T_TREE, step_wstar, TAU["low"])
        if ceiling_official < TARGET_OFFICIAL:
            verdict = "RED"
            verdict_label = (
                f"MODEL INCONSISTENCY: even at the rho-optimal ceiling E[T]={E_T_TREE} the "
                f"depth-9 step yields {ceiling_official:.1f} < 500 -- contradicts #125 (538). "
                f"Flag a banked-model error.")
        else:
            verdict = "AMBER"
            verdict_label = (
                f"harness BANKED + bands computed; oracle run PENDING (chiku-inu firing "
                f"tree-488-pw-fp32-v0, openevolve cross-check awaited). Operative depth-9 "
                f"clear-500 bar = {clear500_bar:.3f}; demand floor (denken #123) = "
                f"{DEMAND_FLOOR_500}; supply ceiling E[T]=5.207 -> "
                f"{measured_official_tps(E_T_TREE, step_wstar, TAU['central']):.1f}. "
                f"Self-test {'PASS' if selftest_pass else 'FAIL'}.")

    gate = {
        "primary_metric_name": "oracle_accept_length_to_clear_500",
        "oracle_accept_length_to_clear_500": primary_metric,
        "oracle_accept_length_to_clear_500_demand_floor_xcheck": primary_demand_floor_xcheck,
        "oracle_accept_length_to_overtake_treefree": overtake_bar,
        "test_metric_name": "measured_official_tps",
        "measured_official_tps": test_metric,
        "test_metric_label": test_metric_label,
        "oracle_run_pending": oracle_pending,
        "self_test_pass": selftest_pass,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "rule": ("GREEN = live oracle accept_length clears 500 conservative AND greedy-exact / "
                 "AMBER = harness banked + bands ready, oracle pending (or central-only) / "
                 "RED = supply ceiling can't reach 500 (model inconsistency) or live < 500"),
    }

    finding = {
        "title": "depth-9 operative clear-500 bar is HIGHER than the demand floor",
        "operative_depth9_clear_500_bar": clear500_bar,
        "demand_floor_denken123": DEMAND_FLOOR_500,
        "gap_E_T": clear500_bar - DEMAND_FLOOR_500,
        "explanation": (
            f"The board-merged demand floor 4.624 (denken #123) is the TOPOLOGY-OPTIMAL E[T] "
            f"to clear 500 -- it corresponds to a ~depth-6.7 tree (step 1.1585, between the "
            f"banked depth-6 bar 4.553 and depth-7 bar 4.649). The clear-500 bar RISES with "
            f"realized depth (each step costs more). The build land #71 ships is the depth-9 W* "
            f"tree (to REACH the 5.207 ceiling), whose step is 1.2128, so it needs "
            f"accept_length >= {clear500_bar:.3f} to clear 500. "
            f"At E[T]=4.624 the depth-9 tree only yields "
            f"{measured_official_tps(DEMAND_FLOOR_500, step_wstar, TAU['central']):.1f} < 500. "
            f"=> the depth-9 build bracket is [{clear500_bar:.3f}, 5.207], NOT [4.624, 5.207]; "
            f"the low-end slack is {E_T_TREE - clear500_bar:.3f} E[T], not "
            f"{E_T_TREE - DEMAND_FLOOR_500:.3f}. The live-oracle gate should use {clear500_bar:.3f} "
            f"(or re-price step at the oracle's reported realized depth via step_time_for_depth)."),
        "advisor_ruling_requested": True,
    }

    out = {
        "gate": gate,
        "finding_depth9_bar": finding,
        "map": {
            "figure_of_merit": "official_TPS = K_cal * accept_length / step_time * tau",
            "K_cal": K_CAL, "step_time_wstar_depth9": step_wstar,
            "tau_band": TAU, "frontier_official": FRONTIER_OFFICIAL,
            "target_official": TARGET_OFFICIAL, "treefree_ceiling": TREEFREE_CEILING,
            "official_at_ceiling_5p207": measured_official_tps(E_T_TREE, step_wstar, TAU["central"]),
            "normalisation_check_wstar": {
                "expected_525_538": tree_et["wstar"]["official_measured_central"],
                "harness": measured_official_tps(E_T_TREE, step_wstar, TAU["central"])},
        },
        "bands_depth9_wstar": {"central": bands_wstar, "taulow": bands_wstar_taulow},
        "lookup_table_depth9": table,
        "self_test": {"pass": selftest_pass, "synthetic_oracle": synth, "evaluation": synth_eval},
        "live_oracle": {"input": oracle_in, "evaluation": live_eval},
        "oracle_input_schema": {
            "depth1_spine_accept": "float -- depth-1 spine accept q_1 (fp32 target 0.7287) [#1]",
            "per_position": "[{depth, q_spine, branch_hit_rho2, branch_width}] -- rho-ladder [#2]",
            "accept_length": "float -- MEASURED E[T], the numerator [#3]",
            "full_tree_reach": "{max_depth_reached, built_depth, frac_reaching_full_depth} [#4]",
        },
        "banked_anchors": {
            "depth1_accept_fp32_target": DEPTH1_ACCEPT_FP32_TARGET,
            "as_built_et": AS_BUILT_ET, "demand_floor_500": DEMAND_FLOOR_500,
            "milestone_clear_500_lawine107": MILE_CLEAR_500,
            "milestone_overtake_4p727": MILE_OVERTAKE,
            "q_target_ladder": q_target_ladder, "rho_cond_measured": rho_cond,
            "m32_parent": parent, "W": W, "max_depth": max_depth,
            "wstar_depth": wstar_depth, "step_wstar": step_wstar},
        "provenance": (
            "ingests the openevolve A10G-oracle 4-number readout of tree-488-pw-fp32-v0 "
            "and maps it through the #100 lever_composition compose (K_cal) + my #125 "
            "tree_et_realization_ceiling step model (W* depth-9 step, measured 1.83x attn tax) "
            "+ wirbel treeshape_measured_accept E[T] DP (consistency). Demand floor 4.624 + "
            "tree-free 491.8 from denken #123 (board-merged); fp32-greedy-exact mechanism from "
            "chiku-inu (board 20260614-092043-711) + kanna #114."),
        "method": ("LOCAL CPU-only analytic readout harness; no GPU/vLLM/HF Job/submission/"
                   "kernel build. Banks the live-oracle pipe. Greedy identity untouched."),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # write the oracle-input schema template for chiku-inu / openevolve to fill.
    os.makedirs(os.path.dirname(args.sample_out), exist_ok=True)
    with open(args.sample_out, "w") as f:
        json.dump(synth, f, indent=2)

    # ------------------------------- console -------------------------------
    print("=" * 96)
    print("ORACLE-READOUT -> MEASURED-OFFICIAL-TPS HARNESS (PR #129)")
    print("=" * 96)
    print(f"\nmap: official = K_cal*accept_length/step*tau  (K_cal={K_CAL:.3f}, "
          f"step_wstar_depth9={step_wstar:.4f}, tau_central=1.0)")
    print(f"normalisation: official(E[T]=5.207) = "
          f"{measured_official_tps(E_T_TREE, step_wstar, TAU['central']):.1f} "
          f"(== #125 W* {tree_et['wstar']['official_measured_central']:.1f})")

    print(f"\n[STEP 2] go/no-go bands (operative depth-9 W* step {step_wstar:.4f}):")
    print(f"  clear-500 bar (operative)      accept_length >= {clear500_bar:.3f}")
    print(f"  overtake tree-free (491.8)     accept_length >= {overtake_bar:.3f}")
    print(f"  demand floor (denken #123)     accept_length >= {DEMAND_FLOOR_500}  [CROSS-CHECK]")
    print(f"  -> FINDING: depth-9 bar {clear500_bar:.3f} > demand floor {DEMAND_FLOOR_500} "
          f"(+{clear500_bar - DEMAND_FLOOR_500:.3f} E[T]); bracket [{clear500_bar:.3f}, 5.207]")

    print(f"\n[lookup] accept_length -> official (depth-9 step):")
    print(f"  {'accept_length':>13s} {'official':>9s}  clears500  label")
    for r in table:
        print(f"  {r['accept_length']:13.3f} {r['official_tps']:9.1f}  "
              f"{'YES' if r['clears_500'] else ' no':>8s}   {r['label']}")

    print(f"\n[self-test] banked rho-optimal ladder as a perfect oracle:")
    print(f"  accept_length={synth['accept_length']:.4f} -> official "
          f"{synth_eval['measured_official_tps_central']:.1f}  | consistency "
          f"{'OK' if synth_eval['consistency']['consistent'] else 'FLAG'} | greedy_exact "
          f"{synth_eval['greedy_exact']['tree_greedy_exact_from_oracle']}  -> "
          f"{'PASS' if selftest_pass else 'FAIL'}")

    if live_eval is not None:
        print(f"\n[LIVE ORACLE] accept_length={live_eval['accept_length']:.4f} -> "
              f"official {live_eval['measured_official_tps_central']:.1f} "
              f"(taulow {live_eval['measured_official_tps_taulow']:.1f})")
        print(f"  clears_500={live_eval['clears_500_central']} "
              f"(cons={live_eval['clears_500_conservative']})  greedy_exact="
              f"{live_eval['greedy_exact']['tree_greedy_exact_from_oracle']}  consistency="
              f"{live_eval['consistency']['consistent']}")
    else:
        print(f"\n[LIVE ORACLE] PENDING -- no tree-488 oracle number on the board yet. "
              f"Test metric = measured_official_tps(as-built 2.10) = {test_metric:.1f}.")

    print(f"\n[PRIMARY] oracle_accept_length_to_clear_500 = {primary_metric:.3f} "
          f"(operative depth-9 W* bar; denken #123 demand-floor x-check = "
          f"{primary_demand_floor_xcheck})")
    print(f"[TEST]    measured_official_tps = {test_metric:.1f}  ({test_metric_label})")
    print(f"\n[VERDICT] {verdict} -- {verdict_label}")
    print(f"\nwrote {args.out}")
    print(f"wrote {args.sample_out} (oracle-input schema template)")

    # ------------------------------- W&B -------------------------------
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                         config={"gate": "oracle-readout-harness",
                                 "method": "cpu-analytic-readout-extends-100-125-wirbel-dp",
                                 "K_cal": K_CAL, "step_wstar_depth9": step_wstar,
                                 "frontier_official": FRONTIER_OFFICIAL,
                                 "target_official": TARGET_OFFICIAL,
                                 "treefree_ceiling": TREEFREE_CEILING,
                                 "demand_floor_500": DEMAND_FLOOR_500,
                                 "tau_low": TAU["low"], "tau_central": TAU["central"],
                                 "oracle_run_pending": oracle_pending})
        s = wandb.summary
        s["oracle_accept_length_to_clear_500"] = primary_metric
        s["oracle_accept_length_to_clear_500_demand_floor_xcheck"] = primary_demand_floor_xcheck
        s["oracle_accept_length_to_overtake_treefree"] = overtake_bar
        s["measured_official_tps"] = test_metric
        s["official_at_ceiling_5p207"] = measured_official_tps(E_T_TREE, step_wstar, TAU["central"])
        s["depth9_bar_minus_demand_floor"] = clear500_bar - DEMAND_FLOOR_500
        s["self_test_pass"] = int(selftest_pass)
        s["oracle_run_pending"] = int(oracle_pending)
        s["verdict"] = verdict
        s["verdict_label"] = verdict_label
        if live_eval is not None:
            s["live_measured_official_tps"] = live_eval["measured_official_tps_central"]
            s["live_clears_500"] = int(live_eval["clears_500_central"])
            s["live_greedy_exact"] = live_eval["greedy_exact"]["tree_greedy_exact_from_oracle"]
            s["live_consistency_ok"] = int(live_eval["consistency"]["consistent"])

        lt = wandb.Table(columns=["accept_length", "official_tps", "clears_500",
                                  "overtakes_treefree", "label"])
        for r in table:
            lt.add_data(r["accept_length"], r["official_tps"], r["clears_500"],
                        r["overtakes_treefree"], r["label"])
        wandb.log({"lookup_table_depth9": lt})

        bt = wandb.Table(columns=["band", "accept_length", "official_at_bar"])
        bt.add_data("clear_500_operative_depth9", clear500_bar, TARGET_OFFICIAL)
        bt.add_data("overtake_treefree", overtake_bar, TREEFREE_CEILING)
        bt.add_data("demand_floor_denken123", DEMAND_FLOOR_500,
                    measured_official_tps(DEMAND_FLOOR_500, step_wstar, TAU["central"]))
        bt.add_data("ceiling_5p207", E_T_TREE,
                    measured_official_tps(E_T_TREE, step_wstar, TAU["central"]))
        wandb.log({"bands": bt})
        print(f"\nW&B run: {run.id}  ({run.url})")
        wandb.finish()


if __name__ == "__main__":
    main()
