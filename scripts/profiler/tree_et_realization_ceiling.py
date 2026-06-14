#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Tree E[T] REALIZATION-CEILING roofline (PR #125): can the speculative-decoding
TREE PHYSICALLY clear 500 official TPS, NET of the real M=32 wide-verify costs?

THE GAP THIS PRICES (supply side)
---------------------------------
denken #123 prices the DEMAND side: the E[T] a tree must DELIVER to clear 500.
This module prices the SUPPLY side: the MAX E[T] the tree can PHYSICALLY REALIZE
once you charge it the real wide-verify costs -- the Marlin W4A16 tile staircase
(flat M<=32, hard cliff M=33; denken #68), the MEASURED tree-mask attention tax
(lawine #107 r_attn=1.83x at M=32, which CORRECTS denken #85's optimistic 1.06x
amortisation that the #100 compose model still carries), and the drafter
tree-expansion cost (wirbel g_drafter). The figure of merit is the realised
official TPS as a function of tree size:

  official_TPS(W) = K_cal * E[T](W) / step_time(W) * tau           (#100 compose)

  step_time(M, depth) = gemm_cost_mult(M)          # Marlin staircase, #68 roofline
                      + g_drafter*(depth-7)/7       # drafter tree-expansion, wirbel
                      + attn_share*(r_attn(M) - 1)  # tree-mask attention, lawine #107

  E[T](M, depth) = rho-optimal DP tree accept-length (wirbel #79/#86 ladder,
                   MC-validated; denken #101 realisation band).

step_time is normalised so the deployed M=8 linear-MTP step = 1.0; then
official(M=8, depth=7) = K_cal*3.844/1.0 = 481.53 by construction (the frontier).
The novelty vs wirbel's rho-optimal projection (569) is the +attn_share*(r_attn-1)
term: pricing the MEASURED 1.83x attention tax pulls the realised official from
~569 down to ~538 -- the supply-side haircut the PR title references. It STILL
clears 500.

WHAT BOUNDS THE OPTIMUM (Step 2 -- the binding constraint)
----------------------------------------------------------
  * WIDTH (M): the Marlin tile-cliff. gemm_cost_mult jumps 1.098 -> 1.284 at
    M=33 (+1698.85 us/row = 14.6% of the decode step; #68 marginal_per_row).
    M=32 is the LAST width on the flat-Marlin plateau -> width is TILE-CLIFF-bound.
  * DEPTH: acceptance-saturation vs drafter-expansion. F_tree saturates (each
    extra spine token accepts with prob -> q_inf ~0.847) while the drafter cost
    grows linearly in depth -> official peaks at depth 9 and declines.
The optimum W* therefore sits AT the tile-cliff boundary (M=32) at depth 9; the
attention tax lowers the LEVEL (569->538) but does not move W*.

OUTPUTS
  PRIMARY  tree_et_realization_ceiling = E[T](W*)  (the max E[T] the tree realises
           at the official-TPS optimum, with the realisation band).
  TEST     tree_clears_500_physically  (1 iff official(W*) >= 500).
  Also     tree_headroom_to_overtake = E[T](W*) - 4.727 (the tree-overtakes-
           tree-free bar) + W* tree shape hand-off to land #71.
  VERDICT  GREEN (official(W*)>=500 w/ margin AND E[T](W*)>=4.727) / AMBER / RED.

LOCAL, CPU-ONLY, ANALYTIC. Extends my #106/#111 crossover model + #100 compose +
wirbel rho-optimal-topology (E[T](M,depth) DP) + lawine #107 measured step-ratio +
denken #68 Marlin roofline. No GPU, no vLLM, no HF Job, no submission, no kernel
build (land #71 owns the BUILD; this bounds the supply CEILING). Greedy identity
untouched by construction -- a projection model serves nothing.
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


# ---- reuse the #100 compose anchors (K_cal, E[T] anchors, attention budget) ----
lc = _load("lever_composition", os.path.join(_HERE, "lever_composition.py"))
K_CAL = lc.K_CAL                       # 125.268  (481.53 / 3.844)
E_T_LINEAR = lc.E_T_LINEAR             # 3.844    deployed M=8 linear-MTP E[T]
E_T_TREE = lc.E_T_TREE                 # 5.207    realised tree ceiling
FRONTIER_OFFICIAL = lc.FRONTIER_OFFICIAL   # 481.53
TARGET_OFFICIAL = lc.TARGET_OFFICIAL       # 500.0
ATTN_SHARE = lc.BUDGET["attention"]    # 0.08    attention slice of the M=8 step
ATTN_M32_OPTIMISTIC = lc.ATTN_TREE_AMORTIZE   # 1.06  denken #85 (OUTDATED optimistic)

# ---- step-model constants (wirbel rho-optimal cost model) ----
G_DRAFTER = 0.168                      # drafter tree-expansion slope (per depth/7)
BASE_DRAFTER_DEPTH = 7                 # deployed K=7 drafter depth (M=8 linear)
M_LINEAR = 8                           # deployed verify width (K+1)
M_TILE_PLATEAU_MAX = 32                # last width on the flat-Marlin plateau (#68)
M_TILE_CLIFF = 33                      # first width past the Marlin tile-cliff

# ---- milestone ladder (E[T] demand anchors; lawine #107 step3_breakeven + #111) ----
MILE_BEAT_LINEAR = 4.452747588028169   # E[T] to beat the linear chain at M=8
MILE_CLEAR_500 = 4.613985078031858     # corrected break-even E[T] to clear 500 (#107)
MILE_TREE_OVERTAKES = 4.727            # E[T] at which tree overtakes tree-free (#106/#111)
MILE_CEILING = 5.207                   # analytical tree E[T] ceiling

# ---- tau band: lawine #116 local->official transfer (tight; ~pinned at 1) ----
TAU = {"low": 0.9983, "central": 1.00, "high": 1.00}


# ----------------------------------------------------------------------------
# Step model
# ----------------------------------------------------------------------------
def r_attn(M: float, m32_ratio: float) -> float:
    """Tree-mask attention scaling vs the M=8 baseline, linear in tree node count
    (attention cost ~ #KV attended). Anchored (M=8 -> 1.0), (M=32 -> m32_ratio).
    m32_ratio = lawine #107 measured 1.8325 (primary) or denken #85 1.06 (optimistic)."""
    if M <= M_LINEAR:
        return 1.0
    return 1.0 + (m32_ratio - 1.0) * (M - M_LINEAR) / (M_TILE_PLATEAU_MAX - M_LINEAR)


def step_time(M: int, depth: int, gemm_cost_mult: dict, m32_attn_ratio: float) -> dict:
    """Decode step normalised to the M=8 linear step = 1.0. Three priced slices:
    Marlin GEMM staircase (#68) + drafter tree-expansion (wirbel) + tree-mask
    attention tax (lawine #107). Returns the decomposition for auditing."""
    gemm = gemm_cost_mult[str(M)]                       # already includes the M=8 1.0 base
    drafter = G_DRAFTER * (depth - BASE_DRAFTER_DEPTH) / BASE_DRAFTER_DEPTH
    attn = ATTN_SHARE * (r_attn(M, m32_attn_ratio) - 1.0)
    step = gemm + drafter + attn
    return {"step": step, "gemm_mult": gemm, "drafter_add": drafter, "attn_add": attn}


def official_tps(F_tree: float, step: float, tau: float) -> float:
    """#100 compose figure of merit: K_cal * E[T]/step_time * tau."""
    return K_CAL * (F_tree / step) * tau


def evaluate(M: int, depth: int, F_tree: float, gemm_cost_mult: dict,
             m32_attn_ratio: float, tau: float) -> dict:
    st = step_time(M, depth, gemm_cost_mult, m32_attn_ratio)
    off = official_tps(F_tree, st["step"], tau)
    return {"M": M, "depth": depth, "F_tree": F_tree, "official_tps": off,
            "step_time": st["step"], "gemm_mult": st["gemm_mult"],
            "drafter_add": st["drafter_add"], "attn_add": st["attn_add"]}


# ----------------------------------------------------------------------------
# Candidate trees (W) from wirbel's rho-optimal depth sweeps + the cliff check
# ----------------------------------------------------------------------------
def build_candidates(rho: dict) -> list[dict]:
    """One candidate per (M, depth) for which wirbel banked a rho-optimal F_tree:
    the M=8 linear frontier, the M=16 and M=32 depth sweeps (deduped by depth),
    plus the synthetic M=33 first-node-past-the-cliff demonstration."""
    cands = [{"M": M_LINEAR, "depth": BASE_DRAFTER_DEPTH, "F_tree": E_T_LINEAR,
              "label": "linear-M8 (deployed frontier)", "source": "frontier",
              "max_branch": 1}]
    for M in (16, 32):
        seen = set()
        for e in rho["per_budget"][str(M)]["depth_sweep"]:
            d = e["depth"]
            if d in seen:                       # depth caps past saturation repeat
                continue
            seen.add(d)
            cands.append({"M": M, "depth": d, "F_tree": e["F_tree"],
                          "label": f"M{M} rho-opt depth {d}", "source": "depth_sweep",
                          "max_branch": rho["per_budget"][str(M)]["optimal"]["max_branch"]
                          if d == rho["per_budget"][str(M)]["optimal"]["depth"] else None})
    # M=33: one node past the Marlin tile-cliff. Grant it the M=32 ceiling F_tree
    # PLUS the single richest extra leaf (beyond_width4 best_rank5_leaf_marginal) --
    # a generous UPPER bound on a 33-node tree's E[T] -- to show the cliff still kills it.
    f32 = rho["per_budget"]["32"]["optimal"]["F_tree"]
    extra = rho["beyond_width4"]["M32"]["best_rank5_leaf_marginal"]
    cands.append({"M": M_TILE_CLIFF, "depth": 9, "F_tree": f32 + extra,
                  "label": "M33 (one node past tile-cliff, generous F)",
                  "source": "cliff_check", "max_branch": 3})
    return cands


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rho", default="research/spec_cost_model/rho_optimal_topology_results.json")
    ap.add_argument("--step-denom", default="research/spec_cost_model/tree_step_denominator.json")
    ap.add_argument("--roofline", default="research/spec_cost_model/verify_gemm_roofline.json")
    ap.add_argument("--out", default="research/spec_cost_model/tree_et_realization_ceiling_results.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="fern/tree-et-realization-ceiling")
    ap.add_argument("--wandb-group", default="tree-et-realization-ceiling")
    args = ap.parse_args()

    with open(args.rho) as f:
        rho = json.load(f)
    with open(args.step_denom) as f:
        denom = json.load(f)
    with open(args.roofline) as f:
        roof = json.load(f)

    gemm_cost_mult = rho["inputs"]["gemm_cost_mult"]        # Marlin staircase, M-keyed
    # MEASURED tree-mask attention ratio at M=32 (lawine #107 median).
    attn_m32_measured = denom["step1_measurement"]["r_attn"]["median"]   # 1.8325
    # cliff magnitude (#68): M=32 -> M=33 adds this fraction of the decode step.
    cliff_step_pct = roof["marginal_per_row"][str(M_TILE_CLIFF)]["pct_of_decode_step"]
    cliff_us = roof["marginal_per_row"][str(M_TILE_CLIFF)]["delta_us"]

    cands = build_candidates(rho)

    # ---- Step 2: official-TPS landscape over W (measured attn, central tau) ----
    landscape = []
    for c in cands:
        meas_c = evaluate(c["M"], c["depth"], c["F_tree"], gemm_cost_mult,
                          attn_m32_measured, TAU["central"])
        meas_lo = evaluate(c["M"], c["depth"], c["F_tree"], gemm_cost_mult,
                           attn_m32_measured, TAU["low"])
        opt_hi = evaluate(c["M"], c["depth"], c["F_tree"], gemm_cost_mult,
                          ATTN_M32_OPTIMISTIC, TAU["high"])
        no_attn = evaluate(c["M"], c["depth"], c["F_tree"], gemm_cost_mult,
                           1.0, TAU["high"])    # wirbel's UNCORRECTED reference (no attn tax)
        row = {**c,
               "official_measured_central": meas_c["official_tps"],
               "official_measured_taulow": meas_lo["official_tps"],
               "official_optimistic": opt_hi["official_tps"],
               "official_no_attn_tax_ref": no_attn["official_tps"],
               "step_time_measured": meas_c["step_time"],
               "gemm_mult": meas_c["gemm_mult"],
               "drafter_add": meas_c["drafter_add"],
               "attn_add": meas_c["attn_add"],
               "clears_500_measured_central": meas_c["official_tps"] >= TARGET_OFFICIAL,
               "clears_500_measured_taulow": meas_lo["official_tps"] >= TARGET_OFFICIAL}
        landscape.append(row)

    # ---- W*: argmax official under the PRIMARY corner (measured attn, central tau),
    # restricted to the physically buildable flat-Marlin plateau (M <= 32). ----
    plateau = [r for r in landscape if r["M"] <= M_TILE_PLATEAU_MAX]
    wstar = max(plateau, key=lambda r: r["official_measured_central"])
    cliff_row = next(r for r in landscape if r["M"] == M_TILE_CLIFF)

    et_ceiling = wstar["F_tree"]                                   # PRIMARY metric
    official_wstar_central = wstar["official_measured_central"]
    official_wstar_taulow = wstar["official_measured_taulow"]
    official_wstar_opt = wstar["official_optimistic"]
    official_wstar_noattn = wstar["official_no_attn_tax_ref"]      # wirbel uncorrected ~569
    attn_haircut = official_wstar_noattn - official_wstar_central  # the supply-side haircut

    clears_500 = bool(official_wstar_central >= TARGET_OFFICIAL)   # TEST metric
    clears_500_conservative = bool(official_wstar_taulow >= TARGET_OFFICIAL)
    headroom_to_overtake = et_ceiling - MILE_TREE_OVERTAKES
    margin_to_500_central = official_wstar_central - TARGET_OFFICIAL
    margin_to_500_conservative = official_wstar_taulow - TARGET_OFFICIAL

    # ---- binding constraint (Step 2): name what stops official from going higher ----
    # WIDTH: is M=32 the plateau ceiling and does M=33 lose? DEPTH: where does the
    # M=32 official peak, and does it decline after?
    m32_rows = sorted((r for r in landscape if r["M"] == 32), key=lambda r: r["depth"])
    m32_peak = max(m32_rows, key=lambda r: r["official_measured_central"])
    depth_declines_after_peak = any(
        r["depth"] > m32_peak["depth"] and r["official_measured_central"] < m32_peak["official_measured_central"]
        for r in m32_rows)
    width_cliff_binds = wstar["M"] == M_TILE_PLATEAU_MAX and \
        cliff_row["official_measured_central"] < wstar["official_measured_central"]

    if width_cliff_binds:
        binding = "marlin-tile-cliff (width)"
    elif depth_declines_after_peak:
        binding = "acceptance-saturation / drafter-expansion (depth)"
    else:
        binding = "unbound-on-grid"
    binding_detail = (
        f"WIDTH is Marlin-tile-cliff-bound: M={M_TILE_PLATEAU_MAX} is the last width on the "
        f"flat-Marlin plateau (gemm_cost_mult {gemm_cost_mult['32']:.3f}); M={M_TILE_CLIFF} jumps to "
        f"{gemm_cost_mult['33']:.3f} (+{cliff_us:.0f} us/row = {cliff_step_pct:.1f}% of the decode step, "
        f"#68), crashing official to {cliff_row['official_measured_central']:.1f} (vs {official_wstar_central:.1f} "
        f"at W*) even when GENEROUSLY granted the M=32 ceiling F_tree + the richest extra leaf. "
        f"DEPTH is acceptance-saturation-bound: at M=32 official peaks at depth {m32_peak['depth']} "
        f"({m32_peak['official_measured_central']:.1f}) then declines as F_tree saturates "
        f"(q_inf~0.847) while the drafter cost grows linearly. The MEASURED 1.83x attention tax "
        f"lowers the LEVEL by {attn_haircut:.0f} TPS (569->{official_wstar_central:.0f}) but does NOT move W*.")

    # ---- Step 3: milestone ladder + demand placement ----
    milestones = {
        "beat_linear_4.45": MILE_BEAT_LINEAR,
        "clear_500_4.62": MILE_CLEAR_500,
        "tree_overtakes_treefree_4.727": MILE_TREE_OVERTAKES,
        "tree_et_ceiling_5.207": MILE_CEILING,
    }
    # tolerance absorbs the 5.20695 (exact DP) vs 5.207 (rounded ceiling label) gap.
    milestone_cleared = {k: bool(et_ceiling >= v - 1e-3) for k, v in milestones.items()}

    # ---- verdict (Step gate) ----
    # GREEN: official(W*) clears 500 with margin at the CONSERVATIVE corner AND
    #        E[T](W*) >= the tree-overtakes bar (4.727).
    # AMBER: clears 500 central but not conservative, or clears 500 but E[T] < 4.727.
    # RED:   official(W*) < 500 even central.
    if clears_500_conservative and et_ceiling >= MILE_TREE_OVERTAKES:
        verdict = "GREEN"
    elif clears_500 and (et_ceiling >= MILE_CLEAR_500):
        verdict = "AMBER"
    else:
        verdict = "RED"

    if verdict == "GREEN":
        verdict_label = (
            f"the tree PHYSICALLY clears 500 and overtakes tree-free. W* = M={wstar['M']} / "
            f"depth {wstar['depth']} / max-branch {wstar['max_branch']} realises E[T]={et_ceiling:.3f} "
            f"at official {official_wstar_central:.1f} TPS (conservative corner {official_wstar_taulow:.1f}, "
            f">= 500 by +{margin_to_500_conservative:.0f}). E[T] {et_ceiling:.3f} clears the "
            f"4.727 overtake bar by +{headroom_to_overtake:.3f}. Supply EXCEEDS demand "
            f"(denken #123): the constraint is the BUILD (land #71 realising the topology), "
            f"NOT the physics. The MEASURED 1.83x attention tax (correcting denken #85's 1.06x) "
            f"costs {attn_haircut:.0f} TPS but the tree still clears with room. Hand land #71 the "
            f"M=32 / depth-9 / max-branch-3 tree; do NOT exceed M=32 nodes (tile-cliff).")
    elif verdict == "AMBER":
        verdict_label = (
            f"the tree clears 500 at the central corner ({official_wstar_central:.1f}) but the "
            f"conservative corner ({official_wstar_taulow:.1f}) or the overtake bar is not cleared "
            f"with margin -- supply and the build/tau co-decide the ship.")
    else:
        verdict_label = (
            f"the tree does NOT physically clear 500 at the optimum ({official_wstar_central:.1f} < 500) "
            f"once the real wide-verify costs are charged -- escalate; the supply side binds.")

    # ---- land #71 hand-off (Step 4) ----
    handoff = {
        "consumer": "land #71 (tree-verify serving BUILD)",
        "build_target_M": wstar["M"],
        "build_target_depth": wstar["depth"],
        "build_target_max_branch": wstar["max_branch"],
        "build_target_parent_array": rho["handoff_land71"]["build_target_M32_parent"],
        "max_tree_width_on_flat_marlin_plateau": M_TILE_PLATEAU_MAX,
        "do_not_exceed_nodes": M_TILE_PLATEAU_MAX,
        "tile_cliff_node": M_TILE_CLIFF,
        "tile_cliff_step_pct": cliff_step_pct,
        "realized_et_target": et_ceiling,
        "realized_official_target_central": official_wstar_central,
        "realized_official_band": [official_wstar_taulow, official_wstar_opt],
        "note": (f"BUILD the M={wstar['M']} / depth-{wstar['depth']} / max-branch-{wstar['max_branch']} "
                 f"rho-optimal tree (parent array banked). The Marlin tile-cliff at M={M_TILE_CLIFF} is the "
                 f"HARD width ceiling: every extra node past {M_TILE_PLATEAU_MAX} costs {cliff_step_pct:.1f}% of "
                 f"the step. Realising E[T]={et_ceiling:.3f} lands official ~{official_wstar_central:.0f} "
                 f"(band {official_wstar_taulow:.0f}-{official_wstar_opt:.0f}); the residual risk is BUILD "
                 f"fidelity (denken #101 realisation band), not supply physics."),
    }

    gate = {
        "primary_metric_name": "tree_et_realization_ceiling",
        "tree_et_realization_ceiling": et_ceiling,
        "official_tps_at_wstar_central": official_wstar_central,
        "official_tps_at_wstar_band": [official_wstar_taulow, official_wstar_opt],
        "official_tps_at_wstar_no_attn_tax_ref": official_wstar_noattn,
        "measured_attention_haircut_tps": attn_haircut,
        "test_metric_name": "tree_clears_500_physically",
        "tree_clears_500_physically": int(clears_500),
        "tree_clears_500_physically_conservative": int(clears_500_conservative),
        "tree_headroom_to_overtake": headroom_to_overtake,
        "official_margin_to_500_central": margin_to_500_central,
        "official_margin_to_500_conservative": margin_to_500_conservative,
        "wstar": {"M": wstar["M"], "depth": wstar["depth"], "max_branch": wstar["max_branch"]},
        "binding_constraint": binding,
        "binding_constraint_detail": binding_detail,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "rule": ("GREEN = official(W*) clears 500 w/ margin at conservative corner AND "
                 "E[T](W*) >= 4.727 overtake bar / AMBER = central-only or no-overtake / "
                 "RED = official(W*) < 500"),
    }

    out = {
        "gate": gate,
        "step_model": {
            "figure_of_merit": "official_TPS(W) = K_cal * E[T](W) / step_time(W) * tau",
            "step_time": "gemm_cost_mult(M) [Marlin staircase #68] + g_drafter*(depth-7)/7 "
                         "[drafter expansion, wirbel] + attn_share*(r_attn(M)-1) [tree-mask, lawine #107]",
            "K_cal": K_CAL, "E_T_linear": E_T_LINEAR, "attn_share": ATTN_SHARE,
            "g_drafter": G_DRAFTER, "base_drafter_depth": BASE_DRAFTER_DEPTH,
            "gemm_cost_mult": gemm_cost_mult,
            "r_attn_M32_measured_primary": attn_m32_measured,
            "r_attn_M32_optimistic_denken85": ATTN_M32_OPTIMISTIC,
            "tau_band_lawine116": TAU,
            "normalisation_check_official_M8": official_tps(E_T_LINEAR, 1.0, 1.0),
        },
        "step2_landscape": landscape,
        "wstar": wstar,
        "cliff_check_M33": cliff_row,
        "step3_milestones": {"ladder": milestones, "cleared_by_wstar": milestone_cleared,
                             "et_realization_ceiling": et_ceiling},
        "binding_constraint": {"name": binding, "detail": binding_detail,
                               "m32_official_by_depth": [
                                   {"depth": r["depth"], "official_measured_central": r["official_measured_central"],
                                    "F_tree": r["F_tree"], "step_time": r["step_time_measured"]}
                                   for r in m32_rows],
                               "m32_peak_depth": m32_peak["depth"],
                               "cliff_us_per_row": cliff_us, "cliff_step_pct": cliff_step_pct},
        "handoff_land71": handoff,
        "anchors": {
            "frontier_official": FRONTIER_OFFICIAL, "target_official": TARGET_OFFICIAL,
            "E_T_linear": E_T_LINEAR, "E_T_tree_ceiling": E_T_TREE,
            "milestones": milestones,
        },
        "provenance": ("extends fern #106/#111 tree-vs-tree-free crossover + #100 lever_composition "
                       "compose model + wirbel rho_optimal_topology (E[T](M,depth) DP, MC-validated) + "
                       "lawine #107 tree_step_denominator (MEASURED r_attn=1.83x, correcting denken #85's "
                       "1.06x) + denken #68 verify_gemm_roofline (Marlin W4A16 staircase, M=33 cliff). "
                       "denken #123 prices the DEMAND side; this prices SUPPLY."),
        "method": ("LOCAL CPU-only analytic roofline; no GPU/vLLM/HF Job/submission/kernel build. "
                   "land #71 owns the BUILD; this bounds the supply CEILING. Greedy identity untouched."),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # ------------------------------- console -------------------------------
    print("=" * 94)
    print("TREE E[T] REALIZATION-CEILING ROOFLINE (PR #125) -- can the tree physically clear 500?")
    print("=" * 94)
    print(f"\nfigure of merit: official_TPS(W) = K_cal*E[T](W)/step_time(W)*tau  "
          f"(K_cal={K_CAL:.3f}, frontier {FRONTIER_OFFICIAL}, target {TARGET_OFFICIAL})")
    print(f"step_time(M,d) = gemm_cost_mult(M) + {G_DRAFTER}*(d-7)/7 + {ATTN_SHARE}*(r_attn(M)-1)")
    print(f"  Marlin staircase gemm_cost_mult: " +
          " ".join(f"M{k}={v:.3f}" for k, v in gemm_cost_mult.items()))
    print(f"  tree-mask attention r_attn(M=32) = {attn_m32_measured:.4f} MEASURED (lawine #107) "
          f"vs {ATTN_M32_OPTIMISTIC} optimistic (denken #85)")
    print(f"  normalisation check: official(M=8, depth=7) = {official_tps(E_T_LINEAR, 1.0, 1.0):.2f} "
          f"(== frontier {FRONTIER_OFFICIAL})")

    print(f"\n[STEP 2] official-TPS landscape over W  (measured attn / central tau)")
    print(f"  {'tree W':34s} {'E[T]':>6s} {'step':>6s} {'off(meas)':>9s} {'off(opt)':>9s} {'(no-attn)':>9s} clears500?")
    for r in sorted(landscape, key=lambda x: (x["M"], x["depth"])):
        flag = "YES" if r["clears_500_measured_central"] else "no"
        star = "  <- W*" if (r["M"] == wstar["M"] and r["depth"] == wstar["depth"]) else (
            "  <- CLIFF" if r["M"] == M_TILE_CLIFF else "")
        print(f"  {r['label'][:34]:34s} {r['F_tree']:6.3f} {r['step_time_measured']:6.3f} "
              f"{r['official_measured_central']:9.1f} {r['official_optimistic']:9.1f} "
              f"{r['official_no_attn_tax_ref']:9.1f} {flag:>5s}{star}")

    print(f"\n[STEP 2] binding constraint = {binding}")
    print(f"  M=32 official by depth (measured attn): " +
          " ".join(f"d{r['depth']}:{r['official_measured_central']:.1f}" for r in m32_rows))
    print(f"  -> peak at depth {m32_peak['depth']}; M=33 cliff crashes to "
          f"{cliff_row['official_measured_central']:.1f} (+{cliff_step_pct:.1f}% step)")

    print(f"\n[STEP 3] milestone ladder vs E[T](W*) = {et_ceiling:.3f}")
    for k, v in milestones.items():
        print(f"  {k:34s} {v:6.3f}  {'CLEARED' if milestone_cleared[k] else 'below'}")
    print(f"  tree_headroom_to_overtake = E[T](W*) - 4.727 = {headroom_to_overtake:+.3f}")

    print(f"\n[PRIMARY] tree_et_realization_ceiling = {et_ceiling:.3f}  "
          f"(official {official_wstar_central:.1f} TPS, band [{official_wstar_taulow:.1f}, {official_wstar_opt:.1f}])")
    print(f"[TEST]    tree_clears_500_physically = {int(clears_500)}  "
          f"(central margin +{margin_to_500_central:.0f}, conservative +{margin_to_500_conservative:.0f})")
    print(f"[supply haircut] MEASURED 1.83x attention tax: {official_wstar_noattn:.1f} (wirbel uncorrected) "
          f"-> {official_wstar_central:.1f} (corrected) = -{attn_haircut:.0f} TPS")
    print(f"[hand-off] land #71: build M={wstar['M']} / depth {wstar['depth']} / max-branch {wstar['max_branch']}; "
          f"do NOT exceed M={M_TILE_PLATEAU_MAX} nodes (tile-cliff)")
    print(f"\n[VERDICT] {verdict} -- {verdict_label}")
    print(f"\nwrote {args.out}")

    # ------------------------------- W&B -------------------------------
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                         config={"gate": "tree-et-realization-ceiling",
                                 "method": "cpu-analytic-roofline-extends-106-111-100-wirbel-107-68",
                                 "K_cal": K_CAL, "frontier_official": FRONTIER_OFFICIAL,
                                 "target_official": TARGET_OFFICIAL, "attn_share": ATTN_SHARE,
                                 "g_drafter": G_DRAFTER, "r_attn_M32_measured": attn_m32_measured,
                                 "r_attn_M32_optimistic": ATTN_M32_OPTIMISTIC,
                                 "tau_low": TAU["low"], "tau_high": TAU["high"]})
        s = wandb.summary
        s["tree_et_realization_ceiling"] = et_ceiling
        s["tree_clears_500_physically"] = int(clears_500)
        s["tree_clears_500_physically_conservative"] = int(clears_500_conservative)
        s["tree_headroom_to_overtake"] = headroom_to_overtake
        s["official_tps_at_wstar_central"] = official_wstar_central
        s["official_tps_at_wstar_taulow"] = official_wstar_taulow
        s["official_tps_at_wstar_optimistic"] = official_wstar_opt
        s["official_tps_at_wstar_no_attn_tax_ref"] = official_wstar_noattn
        s["measured_attention_haircut_tps"] = attn_haircut
        s["official_margin_to_500_central"] = margin_to_500_central
        s["official_margin_to_500_conservative"] = margin_to_500_conservative
        s["wstar_M"] = wstar["M"]
        s["wstar_depth"] = wstar["depth"]
        s["wstar_max_branch"] = wstar["max_branch"]
        s["binding_constraint"] = binding
        s["cliff_official_M33"] = cliff_row["official_measured_central"]
        s["cliff_step_pct"] = cliff_step_pct
        s["verdict"] = verdict
        s["verdict_label"] = verdict_label

        lt = wandb.Table(columns=["tree", "M", "depth", "E_T", "step_time",
                                  "official_measured_central", "official_optimistic",
                                  "official_no_attn_tax_ref", "clears_500"])
        for r in landscape:
            lt.add_data(r["label"], r["M"], r["depth"], r["F_tree"], r["step_time_measured"],
                        r["official_measured_central"], r["official_optimistic"],
                        r["official_no_attn_tax_ref"], bool(r["clears_500_measured_central"]))
        wandb.log({"official_landscape": lt})

        dt = wandb.Table(columns=["depth", "F_tree", "step_time", "official_measured_central"])
        for r in m32_rows:
            dt.add_data(r["depth"], r["F_tree"], r["step_time_measured"], r["official_measured_central"])
        wandb.log({"m32_official_by_depth": dt})

        mt = wandb.Table(columns=["milestone", "E_T", "cleared_by_wstar"])
        for k, v in milestones.items():
            mt.add_data(k, v, bool(milestone_cleared[k]))
        wandb.log({"milestone_ladder": mt})
        print(f"\nW&B run: {run.id}  ({run.url})")
        wandb.finish()


if __name__ == "__main__":
    main()
