#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Per-position-width tree-verify DP (PR #409, denken).

THE QUESTION (the one #402 held out of scope -- it priced only TWO fixed corner shapes):
  denken #402 (W&B 8pcyhe2r) closed the tree-verify demand leg as a STANDALONE 500-clearer: the
  honest single-forward full_fanout M(K)=1+7K is net -61.61 TPS at K=8, while the optimistic
  depth1_branch M(K)=8+(K-1) is only weakly net-positive (+71.44) and harvests just ONE position's
  top-K. But #402 evaluated only those two CORNER shapes. The net-TPS-optimal tree is almost
  certainly an INTERIOR per-position-width vector w=(w_1,...,w_7) that taper-allocates draft width
  to the positions with the steepest coverage slope (early positions, per the #289 ladder decay)
  while paying minimal verify-M tax. This card runs a DP/greedy over per-position widths to find it.

  Questions answered:
    (a) max_w net_tps(w) and the optimal shape w*.
    (b) Is max_w net_tps(w) net-POSITIVE (does ANY tree shape beat the M=8 deployed baseline)?
    (c) Combined with the kanna cb3 supply lift (#388 realized +38.02 optimistic), does
        max_w net_tps(w) + cb3_supply_lift clear 500 from the corrected 467.48 base (gap 32.53)?

THE ANSWER (decision-critical, honest):
  The verify-M tax is a STEP function of total verified rows M(w)=1+sum(w_p): the deployed split-KV
  attention re-reads KV per query-block of BLOCK_Q=4 rows, so tps_loss only jumps when M crosses a
  4-row boundary (M in 9..12 share one tax tier, 13..16 the next, ...). Within a tier extra draft
  slots are TAX-FREE, so the DP packs the steepest-slope early positions up to just below each tax
  boundary. But the coverage PRIZE is small: the entire head-ceiling band is g_max=0.1097 -> only
  ~105.6 TPS gross at FULL saturation (M=57, tax 136). The first tax tier (M<=12, tps_loss=14.36)
  buys only the steepest ~4 marginal slots -> g(w*)~=0.017 -> gross ~16 -> net ~+1.3 TPS (exact
  multiplicative). So:
    * max_w net_tps(w) ~= +1.3 TPS at w* ~= (3,2,2,1,1,1,1) (M=12) -- net-POSITIVE but NEGLIGIBLE.
    * The honest optimum is FAR below #402's optimistic depth1 corner (+71.44, which credited a
      single widened position with the FULL g_max -- an upper bound #402 itself flagged as
      unrealizable). The per-position DP confirms the tree is not a meaningful net-TPS lever: the
      verify-M tax eats the head-ceiling coverage prize at every width that buys non-trivial g.
    * (c) max_w net_tps(w) + cb3(+38.02) = ~39.3 > gap 32.53 -> clears 500, but the cb3 OPTIMISTIC
      anchor (+38.02) ALONE already exceeds the 32.53 gap; the tree contributes a negligible +1.3
      cushion. The combination clears 500 essentially on cb3's back; if kanna #403 conservative-k
      cb3 lands below 32.53, the tree's +1.3 does not rescue it.

DUAL-FRAME (why the self-test reproduces #402's corners AND the DP dominates them):
  #402's two corner numbers (-61.61, +71.44) are CEILING-frame: both were evaluated at the FULL
  head-ceiling coverage g=g_max=0.1097 (`net_tps_at_full_gap`). This card reuses the EXACT #402
  tax+gross machinery, so feeding the corner shapes at g_max reproduces -61.61 / +71.44 to <1e-9.
  In that same optimistic ceiling frame the DP trivially dominates both corners (cheapest tree
  M=9 credited with g_max -> +88). That ceiling-frame dominance is the self-test consistency check.
  The HONEST headline replaces "credit every shape with g_max" by a saturating per-position g(w)
  (Section 1): full fan-out approaches g_max, a single widened position gets a small slice, and the
  DP optimizes net_tps(w, g_honest(w)). The honest optimum (~+1.3) is the decision-driving number.

PPL/GREEDY NOTE (unchanged from #402): a greedy tree verify keeps the longest target-argmax-matching
  path, so the emitted token is EXACTLY the target greedy token -> greedy identity preserved -> PPL
  UNCHANGED at 2.3772 <= 2.42 for every shape w. The shape search changes only WHICH candidate rows
  are verified, never which token is emitted.

WHAT THIS IS / IS NOT:
  Pure-CPU analytic card (stdlib math). 0 GPU, 0 official TPS, 0 HF Job, NO served-file change, NO
  submission, NO kernel build. Imports the BANKED #402 module (tree_verify_net_tps_go_nogo) for the
  corrected base 467.475, demand secant S, E[T], the verify-M roofline tax, and the head-ceiling
  band g_max -- nothing is re-derived. The ONLY new modelling is the saturating per-position
  coverage g_honest(w) (Section 1) and the exact knapsack DP over per-position widths (Section 2).

REPRODUCE (0-GPU):
    cd target/ && .venv/bin/python -m research.validity.tree_per_position_width_dp.\
tree_per_position_width_dp --self-test
    cd target/ && .venv/bin/python -m research.validity.tree_per_position_width_dp.\
tree_per_position_width_dp --per-position-width-dp \
      --wandb_group tree-per-position-width-dp --wandb_name denken/tree-per-position-width-dp
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from pathlib import Path

# ---- import the BANKED #402 machinery (corrected base, secant, E[T], verify-M tax, g_max) ----------
from research.validity.tree_verify_net_tps_go_nogo import tree_verify_net_tps_go_nogo as t402

HERE = Path(__file__).resolve().parent

# ===========================================================================
# Section 0 -- banked anchors re-exported from #402 (NOTHING re-derived here)
# ===========================================================================
BASE_467: float = t402.BASE_467               # corrected realized strict decode base (#393 0q7ynumg)
GAP_32: float = t402.GAP_32                   # 500 - BASE_467 (strict gap-to-500)
S_CENTRAL: float = t402.S_CENTRAL             # demand secant E[T]/cov (#387, 7.912609)
E_T_REALIZED: float = t402.E_T_REALIZED       # secant-consistent realized accept length (#396)
G_MAX: float = t402.COVERAGE_CEILING_GAP      # head-ceiling band 1-top4 = 0.10973 (#401/#387)
TOP1_COVERAGE: float = t402.TOP1_COVERAGE     # program top-1 coverage 0.7617 (#387 fern#34 holdout)
TOP4_COVERAGE: float = t402.TOP4_COVERAGE     # program top-4 anchor 0.890 (#340/#387 c0)
LOCKED_PRIZE: float = t402.LOCKED_PRIZE_TOP1_TOP4  # top1->top4 prize 0.12857
LADDER_289: list[float] = list(t402.LADDER_289)    # per-position conditional acceptance a_1..a_7
F_ATTN_STEP: float = t402.F_ATTN_STEP         # M=8 verify attention = 9.5% of the served spec step
M_DEPLOYED: int = t402.M_DEPLOYED             # deployed verify rows = 8 (all-ones tree, no widening)
N_DRAFT_POS: int = 7                          # #289 ladder length (K_spec MTP draft positions)
GROSS_PER_UNIT_COV: float = t402.gross_tps_gain_per_unit_cov()   # 962.27 (corrected base)
PUBLISHED_GROSS_PER_COV_399: float = t402.PUBLISHED_GROSS_PER_COV_399  # 968.57 (old-base cross-check)

PPL_DEPLOYED: float = t402.PPL_DEPLOYED       # 2.3772
PPL_GATE: float = t402.PPL_GATE               # 2.42

# ---- banked #402 corner anchors (CEILING-frame: both at g=g_max) the self-test must reproduce ------
CORNER_FULL_FANOUT_K8: float = -61.607197607642426   # #402 net_tps_at_full_gap (full_fanout K=8)
CORNER_DEPTH1_K8: float = 71.43798577088975          # #402 net_tps_at_full_gap_optimistic (depth1 K=8)

# ---- kanna cb3 supply lift (#388 realized, optimistic anchor; #403 conservative-k re-cost pending) -
CB3_SUPPLY_LIFT_OPTIMISTIC: float = 38.02     # #388 realized cb3 supply lift (TPS), optimistic
TARGET: float = 500.0

# ---- per-position-width DP knobs ------------------------------------------------------------------
# Per-position width cap = the #401 top-k support ceiling (top-8/16 measurement). The optimum sits
# far below this (the verify-M tax binds first), so the cap is non-binding; reported as a sweep.
WIDTH_CAP: int = 16
TOL_PROV: float = 1e-6


# ===========================================================================
# Section 1 -- the verify-M tax (reused from #402) + the NEW saturating per-position coverage g(w)
# ===========================================================================
# (1a) verify-M roofline tax -- generalize #402's per-(K,shape) tau to take total M directly. The
#      #402 corner taxes are special cases: m_full_fanout(8)=57, m_depth1_branch(8)=15 (verified
#      below), so reusing t402.attn_scale guarantees byte-exact corner reproduction.

def m_of_w(w: tuple[int, ...]) -> int:
    """Total verified rows for a per-position width vector: M(w) = 1 (bonus/root row) + sum(w_p).

    all-ones w=(1,)*7 -> M = 1+7 = 8 = M_DEPLOYED (no widening). full_fanout (K,)*7 -> M = 1+7K.
    depth1 (K,1,1,1,1,1,1) -> M = 1+(K+6) = 8+(K-1). Unifies both #402 corner row-count models."""
    return 1 + sum(w)


def tstep_tax_frac_M(m: int | float) -> float:
    """T_step inflation fraction tau(M): the 9.5%-of-step attention lane scaled by the wider-M
    roofline (#402 Section 3). tau = F_ATTN_STEP * (attn_scale(M) - 1); other lanes weight-bound."""
    return F_ATTN_STEP * (t402.attn_scale(m) - 1.0)


def tps_loss_M(m: int | float, base: float = BASE_467) -> float:
    """TPS drop from inflating T_step by tau(M) at FIXED E[T]: loss = base*tau/(1+tau)."""
    tau = tstep_tax_frac_M(m)
    return base * tau / (1.0 + tau)


def net_tps_exact_M(g: float, m: int | float, base: float = BASE_467, base_et: float = E_T_REALIZED,
                    slope: float = S_CENTRAL) -> float:
    """Honest multiplicative net vs the M=8 baseline at coverage uplift g and verified-rows M:
    net = base*[(1 + S*g/E[T])/(1+tau(M)) - 1]. EXACT composition (gain reduces steps, tax inflates
    T_step; the cross term is real). Identical algebra to t402.net_tps_exact, parameterized on M."""
    tau = tstep_tax_frac_M(m)
    return base * ((1.0 + slope * g / base_et) / (1.0 + tau) - 1.0)


def net_tps_linear_M(g: float, m: int | float, base: float = BASE_467) -> float:
    """The PR's named first-order decomposition net = g*gross_per_unit_cov - tps_loss(M). Overstates
    net by the cross term vs the exact multiplicative composition; reported alongside for the card."""
    return g * GROSS_PER_UNIT_COV - tps_loss_M(m, base)


# (1b) NEW: saturating per-position coverage g_honest(w). The deployed program coverage anchor is the
#      top-4 = 0.890 (#340/#387); a width-K tree raises realized per-position accepted-coverage toward
#      the head ceiling. Model the program-coverage UPLIFT g over the 0.890 anchor as a sum of
#      per-position captures, each a geometric top-k tail decay weighted by that position's miss-room.
#
#      Saturation model (stated explicitly):
#        - position p has miss-room u_p = 1 - a_p  (a_p = #289 ladder top-1 conditional acceptance).
#          Early positions (low a_p, e.g. a_1=0.7293) have the LARGEST room -> steepest coverage slope.
#        - widening position p to width w_p captures fraction (1 - beta^(w_p-1)) of its miss-room:
#          captured_p(w_p) = u_p * (1 - beta^(w_p - 1));  captured_p(1) = 0  (top-1, no widening).
#        - beta is the shared geometric top-k tail-decay ratio, CALIBRATED from the banked program
#          anchors so the model reconciles top-1 (0.7617), top-4 (0.890) and the head ceiling exactly:
#          a program top-k chain cov(W) = top1 + (1-top1)*(1-beta^(W-1)) hits top4 at W=4 iff
#          (1-top4)/(1-top1) = beta^3  =>  beta = ((1-top4)/(1-top1))^(1/3) ~= 0.7720, and then the
#          residual head-ceiling band cov(inf)-top4 = (1-top1)*beta^3 = 1-top4 = g_max EXACTLY.
#        - the per-position uplift sum is normalized to the program ceiling g_max (its sum->inf limit
#          is sum_p u_p; full saturation of ALL positions -> g_max):
#          g_honest(w) = g_max * [ sum_p u_p*(1-beta^(w_p-1)) ] / [ sum_p u_p ].
#      => g_honest(all-ones) = 0 (perturbation sanity); g_honest(all->inf) -> g_max (ceiling).

def beta_decay() -> float:
    """Shared geometric top-k tail-decay ratio, calibrated from banked program top-1/top-4 anchors:
    (1-top4)/(1-top1) = beta^3  =>  beta = ((1-top4)/(1-top1))^(1/3). Reproduces g_max as residual."""
    return ((1.0 - TOP4_COVERAGE) / (1.0 - TOP1_COVERAGE)) ** (1.0 / 3.0)


def miss_room() -> list[float]:
    """Per-position miss-room u_p = 1 - a_p (the saturable coverage above each position's top-1)."""
    return [1.0 - a for a in LADDER_289]


def captured_sum(w: tuple[int, ...], beta: float) -> float:
    """sum_p u_p * (1 - beta^(w_p-1)): total per-position miss-room captured by width vector w."""
    u = miss_room()
    return sum(u[p] * (1.0 - beta ** (w[p] - 1)) for p in range(len(w)))


def g_honest(w: tuple[int, ...], beta: float | None = None) -> float:
    """Saturating per-position program-coverage uplift over the top-4 anchor, capped at g_max.
    g(all-ones)=0; g(all->inf)->g_max. Taper-weighted by per-position miss-room u_p=(1-a_p)."""
    if beta is None:
        beta = beta_decay()
    u_total = sum(miss_room())
    return G_MAX * captured_sum(w, beta) / u_total


# ===========================================================================
# Section 2 -- the DP: optimize per-position widths for max honest net_tps(w)
# ===========================================================================
# net_honest(w) = net_tps_exact_M(g_honest(w), M(w)). g_honest is SEPARABLE across positions and each
# per-position capture is CONCAVE in w_p (geometric diminishing returns), while the tax depends only
# on total M = 8 + slots (slots = sum(w_p - 1)). So for a fixed slot budget the coverage-max
# allocation is an exact bounded-knapsack DP over positions; net is then maximized over the (small)
# slot budget. We also run a marginal-greedy optimizer independently and assert they agree.

def net_honest(w: tuple[int, ...], beta: float | None = None) -> float:
    return net_tps_exact_M(g_honest(w, beta), m_of_w(w))


def dp_best_capture_by_slots(beta: float, width_cap: int = WIDTH_CAP,
                             n_pos: int = N_DRAFT_POS) -> tuple[dict[int, float], dict[int, tuple[int, ...]]]:
    """Exact bounded-knapsack DP. Returns, for every slot budget s (0..n_pos*(width_cap-1)):
       best_cap[s]  = max_w { captured_sum(w) : sum(w_p-1) = s, 1<=w_p<=width_cap }
       best_w[s]    = the achieving width vector.
    Concave separable -> the DP is exact (no need to enumerate the full 16^7 grid)."""
    u = miss_room()
    # dp[s] = (max captured using positions processed so far with s slots, widths-tuple)
    dp: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for p in range(n_pos):
        ndp: dict[int, tuple[float, tuple[int, ...]]] = {}
        for s, (cap_val, wv) in dp.items():
            for wp in range(1, width_cap + 1):
                ns = s + (wp - 1)
                ncap = cap_val + u[p] * (1.0 - beta ** (wp - 1))
                cur = ndp.get(ns)
                if cur is None or ncap > cur[0] + 1e-15:
                    ndp[ns] = (ncap, wv + (wp,))
        dp = ndp
    best_cap = {s: v[0] for s, v in dp.items()}
    best_w = {s: v[1] for s, v in dp.items()}
    return best_cap, best_w


def greedy_best_capture_by_slots(beta: float, max_slots: int,
                                 n_pos: int = N_DRAFT_POS) -> tuple[dict[int, float], dict[int, tuple[int, ...]]]:
    """Independent marginal-greedy: repeatedly add the slot with the largest marginal capture
    u_p*beta^(w_p-1)*(1-beta). Optimal for separable-concave; used to cross-check the DP."""
    u = miss_room()
    w = [1] * n_pos
    cap_by_slots: dict[int, float] = {0: 0.0}
    w_by_slots: dict[int, tuple[int, ...]] = {0: tuple(w)}
    cur_cap = 0.0
    for s in range(1, max_slots + 1):
        # marginal of widening position p from w[p] to w[p]+1
        best_p, best_marg = -1, -1.0
        for p in range(n_pos):
            marg = u[p] * (beta ** (w[p] - 1)) * (1.0 - beta)
            if marg > best_marg:
                best_marg, best_p = marg, p
        w[best_p] += 1
        cur_cap += best_marg
        cap_by_slots[s] = cur_cap
        w_by_slots[s] = tuple(w)
    return cap_by_slots, w_by_slots


def optimize_honest(beta: float | None = None, width_cap: int = WIDTH_CAP) -> dict:
    """Find w* = argmax_w net_honest(w) via the exact slot-budget DP; cross-check with greedy."""
    if beta is None:
        beta = beta_decay()
    u_total = sum(miss_room())
    best_cap, best_w = dp_best_capture_by_slots(beta, width_cap)

    best = {"net": float("-inf"), "slots": 0, "w": tuple([1] * N_DRAFT_POS), "g": 0.0, "m": M_DEPLOYED}
    net_by_slots: dict[int, float] = {}
    for s in sorted(best_cap):
        m = M_DEPLOYED + s
        g = G_MAX * best_cap[s] / u_total
        net = net_tps_exact_M(g, m)
        net_by_slots[s] = net
        if net > best["net"]:
            best = {"net": net, "slots": s, "w": best_w[s], "g": g, "m": m}

    # greedy cross-check over a bounded slot range that safely covers the optimum.
    max_slots_greedy = min(N_DRAFT_POS * (width_cap - 1), best["slots"] + 4 * N_DRAFT_POS)
    g_cap, g_w = greedy_best_capture_by_slots(beta, max_slots_greedy)
    greedy_best = {"net": float("-inf"), "slots": 0, "w": tuple([1] * N_DRAFT_POS)}
    for s, cval in g_cap.items():
        net = net_tps_exact_M(G_MAX * cval / u_total, M_DEPLOYED + s)
        if net > greedy_best["net"]:
            greedy_best = {"net": net, "slots": s, "w": g_w[s]}

    return {
        "beta": beta, "width_cap": width_cap, "u_total": u_total,
        "w_star": list(best["w"]), "m_star": best["m"], "slots_star": best["slots"],
        "g_star_honest": best["g"], "max_net_honest": best["net"],
        "max_net_honest_linear": net_honest_linear(tuple(best["w"]), beta),
        "net_positive": bool(best["net"] > 0.0),
        "greedy_max_net": greedy_best["net"], "greedy_w_star": list(greedy_best["w"]),
        "greedy_matches_dp": bool(abs(greedy_best["net"] - best["net"]) < 1e-6
                                  and list(greedy_best["w"]) == list(best["w"])),
        "net_by_slots": {str(s): net_by_slots[s] for s in sorted(net_by_slots) if s <= 24},
    }


def net_honest_linear(w: tuple[int, ...], beta: float | None = None) -> float:
    return net_tps_linear_M(g_honest(w, beta), m_of_w(w))


# ===========================================================================
# Section 3 -- CEILING frame: reproduce #402's two corners + the DP that dominates them
# ===========================================================================
# #402's corner numbers are at g=g_max (the optimistic full head-ceiling credit). Reusing the #402
# tax machinery, the corner shapes reproduce -61.61 / +71.44 to <1e-9. In that same ceiling frame the
# DP optimum is the CHEAPEST tree (smallest M>8) credited with g_max: M=9 -> net ~+88 >= max(corners).

def w_full_fanout(k: int) -> tuple[int, ...]:
    """full_fanout corner as a width vector: top-K at every draft position -> M=1+7K."""
    return tuple([k] * N_DRAFT_POS)


def w_depth1_branch(k: int) -> tuple[int, ...]:
    """depth1_branch corner as a width vector: top-K at ONE position, rest top-1 -> M=8+(K-1)."""
    return tuple([k] + [1] * (N_DRAFT_POS - 1))


def net_at_ceiling(w: tuple[int, ...]) -> float:
    """#402 CEILING-frame net: credit shape w with the FULL head-ceiling coverage g_max."""
    return net_tps_exact_M(G_MAX, m_of_w(w))


def optimize_ceiling(width_cap: int = WIDTH_CAP) -> dict:
    """Max net over TREES (>=1 widened position) in the optimistic g_max frame -> dominates corners.
    net_at_ceiling decreases in M, so the optimum is the smallest tree M=9 (one extra slot)."""
    # smallest tree: one position widened to 2 -> M=9.
    w_min_tree = tuple([2] + [1] * (N_DRAFT_POS - 1))
    best = {"net": net_at_ceiling(w_min_tree), "w": w_min_tree, "m": m_of_w(w_min_tree)}
    # (monotone in M; scan a few small M to confirm the smallest-M tree wins.)
    for extra in range(1, 8):
        w = tuple([1 + extra] + [1] * (N_DRAFT_POS - 1))
        n = net_at_ceiling(w)
        if n > best["net"]:
            best = {"net": n, "w": w, "m": m_of_w(w)}
    return {"max_net_ceiling": best["net"], "w_star_ceiling": list(best["w"]),
            "m_star_ceiling": best["m"]}


# ===========================================================================
# Section 4 -- combine with cb3 supply lift; does tree(w*) + cb3 clear 500?
# ===========================================================================

def combine_with_cb3(max_net_honest: float, cb3: float = CB3_SUPPLY_LIFT_OPTIMISTIC) -> dict:
    combined_lift = max_net_honest + cb3
    combined_tps = BASE_467 + combined_lift
    return {
        "cb3_supply_lift_optimistic": cb3,
        "max_net_honest": max_net_honest,
        "combined_lift_tps": combined_lift,
        "combined_projected_tps": combined_tps,
        "gap_to_500": GAP_32,
        "clears_500_combined": bool(combined_tps >= TARGET),
        "cb3_alone_clears_500": bool(BASE_467 + cb3 >= TARGET),
        "tree_marginal_cushion_tps": max_net_honest,
        "combined_vs_500_tps": combined_tps - TARGET,
    }


# ===========================================================================
# Section 5 -- self-tests (must pass; reproduces #402 corners + perturbation sanity)
# ===========================================================================

def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))


def run_self_tests(honest: dict, ceiling: dict, beta: float) -> dict:
    c: dict[str, bool] = {}

    # a) provenance: banked #402 anchors imported unchanged.
    c["a_base_is_393"] = abs(BASE_467 - 467.475218449957) < TOL_PROV
    c["a_base_plus_gap_is_500"] = abs(BASE_467 + GAP_32 - 500.0) < 1e-6
    c["a_secant_matches_387"] = abs(S_CENTRAL - 7.912609135742992) < 1e-9
    c["a_gmax_rounds_0p1097"] = round(G_MAX, 4) == 0.1097
    c["a_gross_per_cov_corrected_900s"] = 900.0 < GROSS_PER_UNIT_COV < 1000.0
    c["a_ladder_len_7"] = len(LADDER_289) == N_DRAFT_POS
    c["a_ladder_monotone"] = all(LADDER_289[i] <= LADDER_289[i + 1] for i in range(6))

    # b) M(w) row-count unifies the two #402 corner models.
    c["b_m_all_ones_is_8"] = m_of_w(tuple([1] * N_DRAFT_POS)) == M_DEPLOYED
    c["b_m_full_fanout_k8_is_57"] = m_of_w(w_full_fanout(8)) == 57
    c["b_m_full_fanout_matches_402"] = m_of_w(w_full_fanout(8)) == t402.m_full_fanout(8)
    c["b_m_depth1_k8_is_15"] = m_of_w(w_depth1_branch(8)) == 15
    c["b_m_depth1_matches_402"] = m_of_w(w_depth1_branch(8)) == int(t402.m_depth1_branch(8))

    # c) CEILING-frame corner reproduction (the headline self-test): both #402 corners to <1e-9.
    c["c_reproduce_full_fanout_corner"] = abs(net_at_ceiling(w_full_fanout(8)) - CORNER_FULL_FANOUT_K8) < 1e-6
    c["c_reproduce_depth1_corner"] = abs(net_at_ceiling(w_depth1_branch(8)) - CORNER_DEPTH1_K8) < 1e-6
    c["c_full_fanout_tight_to_402_exact"] = abs(
        net_at_ceiling(w_full_fanout(8)) - t402.net_tps_exact(G_MAX, 8, "full_fanout")) < 1e-12
    c["c_depth1_tight_to_402_exact"] = abs(
        net_at_ceiling(w_depth1_branch(8)) - t402.net_tps_exact(G_MAX, 8, "depth1_branch")) < 1e-12
    c["c_full_fanout_corner_negative"] = net_at_ceiling(w_full_fanout(8)) < 0.0
    c["c_depth1_corner_positive"] = net_at_ceiling(w_depth1_branch(8)) > 0.0

    # d) CEILING-frame optimum dominates BOTH corners (DP improves on #402's two hand-picked shapes).
    c["d_ceiling_opt_ge_full_fanout"] = ceiling["max_net_ceiling"] >= CORNER_FULL_FANOUT_K8 - 1e-6
    c["d_ceiling_opt_ge_depth1"] = ceiling["max_net_ceiling"] >= CORNER_DEPTH1_K8 - 1e-6
    c["d_ceiling_opt_ge_max_corner"] = ceiling["max_net_ceiling"] >= max(CORNER_FULL_FANOUT_K8, CORNER_DEPTH1_K8) - 1e-6

    # e) saturation model: g(all-ones)=0 (perturbation sanity), monotone, saturates to g_max.
    c["e_g_all_ones_is_zero"] = abs(g_honest(tuple([1] * N_DRAFT_POS))) < 1e-12
    c["e_beta_in_unit_interval"] = 0.0 < beta < 1.0
    c["e_beta_rounds_0p772"] = round(beta, 3) == 0.772
    c["e_g_monotone_in_width"] = g_honest((2, 1, 1, 1, 1, 1, 1)) > g_honest(tuple([1] * N_DRAFT_POS))
    c["e_g_saturates_below_gmax"] = g_honest(tuple([WIDTH_CAP] * N_DRAFT_POS)) < G_MAX
    c["e_g_full_saturation_near_gmax"] = g_honest(tuple([60] * N_DRAFT_POS)) > 0.99 * G_MAX
    # program-anchor reconciliation: a single-position program chain hits top4 at width 4.
    prog_cov_w4 = TOP1_COVERAGE + (1.0 - TOP1_COVERAGE) * (1.0 - beta ** 3)
    c["e_program_chain_hits_top4_at_w4"] = abs(prog_cov_w4 - TOP4_COVERAGE) < 1e-9
    c["e_program_residual_is_gmax"] = abs((1.0 - TOP1_COVERAGE) * beta ** 3 - G_MAX) < 1e-9

    # f) HONEST optimum: net-positive, far below the optimistic depth1 corner, DP==greedy.
    c["f_honest_opt_net_positive"] = honest["max_net_honest"] > 0.0
    c["f_honest_opt_below_depth1_corner"] = honest["max_net_honest"] < CORNER_DEPTH1_K8
    c["f_honest_opt_above_full_fanout_corner"] = honest["max_net_honest"] > CORNER_FULL_FANOUT_K8
    c["f_greedy_matches_dp"] = honest["greedy_matches_dp"]
    c["f_w_star_widens_early_positions"] = honest["w_star"][0] >= honest["w_star"][-1]
    c["f_m_star_modest"] = honest["m_star"] <= 20

    # g) net composition algebra: exact <= linear (cross term), net0 at g=0.
    c["g_exact_le_linear_at_opt"] = (honest["max_net_honest"]
                                     <= honest["max_net_honest_linear"] + 1e-9)
    c["g_net_zero_at_g_zero_m8"] = abs(net_tps_exact_M(0.0, 8)) < 1e-9
    c["g_tax_zero_at_m8"] = abs(tstep_tax_frac_M(8)) < 1e-12
    c["g_tax_step_within_block"] = abs(tps_loss_M(9) - tps_loss_M(12)) < 1e-9  # M 9..12 share a tier
    c["g_tax_jumps_across_block"] = tps_loss_M(13) > tps_loss_M(12) + 1.0

    # h) PPL/greedy identity preserved (shape search changes only verified rows, not emitted token).
    c["h_ppl_passes_gate"] = PPL_DEPLOYED <= PPL_GATE

    # i) numeric hygiene.
    flat = [beta, honest["max_net_honest"], honest["g_star_honest"], ceiling["max_net_ceiling"],
            GROSS_PER_UNIT_COV, net_tps_exact_M(G_MAX, 57), net_tps_exact_M(G_MAX, 15)]
    c["i_no_nan_inf"] = all(_finite(v) for v in flat)

    passes = all(c.values())
    return {"conditions": c, "n_checks": len(c), "n_passed": sum(1 for v in c.values() if v),
            "passes": passes}


# ===========================================================================
# Section 6 -- report assembly + W&B + CLI
# ===========================================================================

def build_report() -> dict:
    beta = beta_decay()
    honest = optimize_honest(beta)
    ceiling = optimize_ceiling()
    combine = combine_with_cb3(honest["max_net_honest"])
    selftest = run_self_tests(honest, ceiling, beta)

    # beta-sensitivity sweep (robustness of the honest verdict to the tail-decay calibration).
    beta_sweep = {}
    for bv in [0.70, beta, 0.85]:
        h = optimize_honest(bv)
        beta_sweep[f"beta_{bv:.3f}"] = {
            "beta": bv, "w_star": h["w_star"], "m_star": h["m_star"],
            "max_net_honest": h["max_net_honest"], "net_positive": h["net_positive"],
            "combined_clears_500": bool(BASE_467 + h["max_net_honest"]
                                        + CB3_SUPPLY_LIFT_OPTIMISTIC >= TARGET),
        }

    tree_shape_str = (
        "per-position width vector w=(w_1..w_7) over the 7 #289 MTP draft slots; M(w)=1+sum(w_p) "
        "verified rows. all-ones=deployed M=8 (no tree); full_fanout=(K,)*7->M=1+7K; "
        "depth1=(K,1,1,1,1,1,1)->M=8+(K-1). Honest coverage g_honest(w)=g_max*sum_p u_p(1-beta^(w_p-1))"
        "/sum_p u_p with u_p=1-a_p (per-position miss-room) and beta=((1-top4)/(1-top1))^(1/3)~=0.772 "
        "(geometric top-k tail decay calibrated to reconcile top1/top4/head-ceiling). Verify-M tax "
        "reused EXACTLY from #402 (split-KV attention, LOCKED 16-way split): tps_loss steps every "
        "BLOCK_Q=4 rows."
    )
    return {
        "pr": 409, "agent": "denken", "kind": "tree-per-position-width-dp",
        "analysis_only": True, "no_launch": True, "no_hf_job": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used": False, "official_tps": 0,
        "baseline_unchanged_tps": 481.53, "baseline_unchanged_ppl": 2.3772,
        "corrected_strict_base_tps": BASE_467, "gap_to_500_tps": GAP_32,
        "inputs": {
            "base_467_393": BASE_467, "gap_32_393": GAP_32, "s_central_387": S_CENTRAL,
            "e_t_realized_396": E_T_REALIZED, "g_max_ceiling_band_401": G_MAX,
            "top1_coverage_387": TOP1_COVERAGE, "top4_coverage_387": TOP4_COVERAGE,
            "locked_prize_top1_top4": LOCKED_PRIZE, "ladder_289": LADDER_289,
            "f_attn_step_378": F_ATTN_STEP, "m_deployed": M_DEPLOYED, "n_draft_pos": N_DRAFT_POS,
            "gross_per_unit_cov_corrected": GROSS_PER_UNIT_COV,
            "published_gross_per_cov_399": PUBLISHED_GROSS_PER_COV_399,
            "corner_full_fanout_k8_402": CORNER_FULL_FANOUT_K8, "corner_depth1_k8_402": CORNER_DEPTH1_K8,
            "cb3_supply_lift_optimistic_388": CB3_SUPPLY_LIFT_OPTIMISTIC,
            "width_cap_401_support": WIDTH_CAP, "target": TARGET,
            "ppl_deployed": PPL_DEPLOYED, "ppl_gate": PPL_GATE,
            "source_402_run": "8pcyhe2r", "source_401_run": "i2qsjyp6", "source_393_run": "0q7ynumg",
            "source_399_run": "ec7i3z5t", "source_387_run": "z8osvif8", "source_289_run": "fi34s269",
            "source_332_run": "y5cl0ena", "source_388_ref": "kanna cb3 realized +38.02",
            "source_403_ref": "kanna conservative-k cb3 re-cost (pending; not used)",
            "source_357_ref": "fern tree+cb3 GO/NO-GO (consumer; sibling)",
        },
        "tree_shape": tree_shape_str,
        "beta_decay": beta,
        # ---- card deliverable scalars ----
        "w_star_honest": honest["w_star"], "m_star_honest": honest["m_star"],
        "slots_star_honest": honest["slots_star"], "g_star_honest": honest["g_star_honest"],
        "max_net_honest": honest["max_net_honest"],
        "max_net_honest_linear": honest["max_net_honest_linear"],
        "tree_net_positive": honest["net_positive"],
        "greedy_matches_dp": honest["greedy_matches_dp"],
        "max_net_ceiling": ceiling["max_net_ceiling"], "w_star_ceiling": ceiling["w_star_ceiling"],
        "m_star_ceiling": ceiling["m_star_ceiling"],
        "combine": combine,
        "max_net_combined_projected_tps": combine["combined_projected_tps"],
        "combined_clears_500": combine["clears_500_combined"],
        "cb3_alone_clears_500": combine["cb3_alone_clears_500"],
        "net_by_slots": honest["net_by_slots"],
        "beta_sweep": beta_sweep,
        "ppl_unchanged": PPL_DEPLOYED,
        "self_test": selftest,
        "tree_per_position_width_dp_self_test_passes": selftest["passes"],
    }


def log_to_wandb(report: dict, group: str, name: str) -> str:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_unavailable"
    try:
        run = wandb.init(group=group, name=name, config=report["inputs"])
        combine = report["combine"]
        wandb.summary.update({
            "tree_shape": report["tree_shape"], "beta_decay": report["beta_decay"],
            "w_star_honest": str(report["w_star_honest"]), "m_star_honest": report["m_star_honest"],
            "g_star_honest": report["g_star_honest"], "max_net_honest": report["max_net_honest"],
            "tree_net_positive": report["tree_net_positive"],
            "max_net_ceiling": report["max_net_ceiling"],
            "max_net_combined_projected_tps": report["max_net_combined_projected_tps"],
            "combined_clears_500": report["combined_clears_500"],
            "cb3_alone_clears_500": report["cb3_alone_clears_500"],
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            "tree_per_position_width_dp_self_test_passes": report["tree_per_position_width_dp_self_test_passes"],
        })
        wandb.log({
            "summary/beta_decay": report["beta_decay"],
            "summary/m_star_honest": float(report["m_star_honest"]),
            "summary/slots_star_honest": float(report["slots_star_honest"]),
            "summary/g_star_honest": report["g_star_honest"],
            "summary/max_net_honest": report["max_net_honest"],
            "summary/max_net_honest_linear": report["max_net_honest_linear"],
            "summary/max_net_ceiling": report["max_net_ceiling"],
            "summary/cb3_supply_lift_optimistic": combine["cb3_supply_lift_optimistic"],
            "summary/combined_lift_tps": combine["combined_lift_tps"],
            "summary/max_net_combined_projected_tps": combine["combined_projected_tps"],
            "summary/combined_vs_500_tps": combine["combined_vs_500_tps"],
            "summary/gap_to_500_tps": report["gap_to_500_tps"],
            "summary/corrected_strict_base_tps": report["corrected_strict_base_tps"],
            "summary/tree_net_positive": float(report["tree_net_positive"]),
            "summary/combined_clears_500": float(report["combined_clears_500"]),
            "summary/cb3_alone_clears_500": float(report["cb3_alone_clears_500"]),
            "summary/greedy_matches_dp": float(report["greedy_matches_dp"]),
            "summary/self_test_passes": float(report["self_test"]["passes"]),
            "summary/self_test_n_checks": float(report["self_test"]["n_checks"]),
        })
        # net-vs-slots curve (the tax/coverage tradeoff shape).
        for s, net in report["net_by_slots"].items():
            wandb.log({"net_by_slots/slots": float(s), "net_by_slots/net_honest": float(net)})
        for tag, row in report["beta_sweep"].items():
            wandb.log({f"beta_sweep/{tag}/beta": row["beta"],
                       f"beta_sweep/{tag}/max_net_honest": row["max_net_honest"],
                       f"beta_sweep/{tag}/m_star": float(row["m_star"]),
                       f"beta_sweep/{tag}/combined_clears_500": float(row["combined_clears_500"])})
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_failed"


def print_report(r: dict) -> None:
    combine = r["combine"]
    print("\n=== Per-position-width tree-verify DP (PR #409, denken) ===")
    print(f"corrected strict base (#393) = {BASE_467:.4f} TPS   gap_to_500 = {GAP_32:.4f}")
    print(f"head-ceiling band g_max (#401) = {G_MAX:.4f}   gross/unit-cov = {GROSS_PER_UNIT_COV:.2f} TPS")
    print(f"saturation: beta = ((1-top4)/(1-top1))^(1/3) = {r['beta_decay']:.5f}  "
          f"(reconciles top1={TOP1_COVERAGE:.4f}, top4={TOP4_COVERAGE:.4f}, residual=g_max)")
    print("\n-- CEILING frame (reproduce #402's two corners at g=g_max) --")
    print(f"  full_fanout (8,)*7  M=57  net = {net_at_ceiling(w_full_fanout(8)):+.2f}  (#402: {CORNER_FULL_FANOUT_K8:+.2f})")
    print(f"  depth1 (8,1..1)     M=15  net = {net_at_ceiling(w_depth1_branch(8)):+.2f}  (#402: {CORNER_DEPTH1_K8:+.2f})")
    print(f"  DP optimum (ceiling, smallest tree) = {r['max_net_ceiling']:+.2f} at w={r['w_star_ceiling']} "
          f"(M={r['m_star_ceiling']}) >= max(corners)")
    print("\n-- HONEST frame (saturating g(w); the decision headline) --")
    print(f"  w* = {r['w_star_honest']}  (M={r['m_star_honest']}, slots={r['slots_star_honest']})")
    print(f"  g(w*) = {r['g_star_honest']:.5f}  (vs g_max {G_MAX:.4f})")
    print(f"  max_w net_tps(w) = {r['max_net_honest']:+.3f} TPS  (linear {r['max_net_honest_linear']:+.3f})  "
          f"net_positive = {r['tree_net_positive']}")
    print(f"  greedy==DP: {r['greedy_matches_dp']}")
    print("  net_honest by slot budget (s -> net TPS):")
    for s, net in list(r["net_by_slots"].items())[:14]:
        print(f"    slots={s:>2} M={M_DEPLOYED + int(s):>2}  net={net:+.3f}")
    print("\n-- COMBINE with cb3 supply lift (#388 +38.02 optimistic) --")
    print(f"  cb3 alone: {BASE_467:.2f} + {combine['cb3_supply_lift_optimistic']:.2f} = "
          f"{BASE_467 + combine['cb3_supply_lift_optimistic']:.2f}  clears 500? {combine['cb3_alone_clears_500']}")
    print(f"  tree(w*) + cb3: {BASE_467:.2f} + {r['max_net_honest']:.2f} + "
          f"{combine['cb3_supply_lift_optimistic']:.2f} = {combine['combined_projected_tps']:.2f}  "
          f"clears 500? {combine['clears_500_combined']}  (vs500 {combine['combined_vs_500_tps']:+.2f})")
    print(f"  tree marginal cushion over cb3 = {combine['tree_marginal_cushion_tps']:+.3f} TPS")
    print("\n-- beta sensitivity --")
    for tag, row in r["beta_sweep"].items():
        print(f"  {tag}: w*={row['w_star']} max_net={row['max_net_honest']:+.3f} "
              f"net+={row['net_positive']} combined_clears_500={row['combined_clears_500']}")
    print(f"\nPPL: greedy tree verify preserves token identity -> PPL unchanged {PPL_DEPLOYED} <= {PPL_GATE}")
    print(f"\nself-test: {r['self_test']['n_passed']}/{r['self_test']['n_checks']} checks  "
          f"passes = {r['tree_per_position_width_dp_self_test_passes']}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-position-width tree-verify DP (PR #409).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="0-GPU analytic gate (PR #409 deliverables)")
    ap.add_argument("--per-position-width-dp", action="store_true",
                    help="run the per-position-width DP analysis (alias of default)")
    ap.add_argument("--corrected-base", type=float, default=None, help="(documented; banked = 467.48)")
    ap.add_argument("--tau", type=float, default=None, help="(documented; banked #402 calibration)")
    ap.add_argument("--secant", type=float, default=None, help="(documented; banked = 962.27)")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="tree-per-position-width-dp")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="denken/tree-per-position-width-dp")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/validity/tree_per_position_width_dp/tree_per_position_width_dp_results.json")
    args = ap.parse_args()

    report = build_report()
    print_report(report)
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    report["peak_mem_mib"] = peak_mib

    if args.self_test:
        out = Path("research/validity/tree_per_position_width_dp/tree_per_position_width_dp_selftest.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}  (peak {peak_mib:.1f} MiB)")
        print(f"\ntree_per_position_width_dp_self_test_passes = {report['self_test']['passes']}")
        return 0 if report["self_test"]["passes"] else 1

    report["wandb_run_id"] = None if args.no_wandb else log_to_wandb(report, args.wandb_group, args.wandb_name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')}, peak {peak_mib:.1f} MiB)")

    combine = report["combine"]
    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [report["wandb_run_id"]] if report.get("wandb_run_id") else [],
        "no_hf_job": True, "official_tps": 0.0, "analysis_only": True, "no_served_file_change": True,
        "beta_decay": float(report["beta_decay"]),
        "w_star_honest": report["w_star_honest"], "m_star_honest": int(report["m_star_honest"]),
        "max_net_honest_tps": float(report["max_net_honest"]),
        "tree_net_positive": bool(report["tree_net_positive"]),
        "max_net_ceiling_tps": float(report["max_net_ceiling"]),
        "max_net_combined_projected_tps": float(report["max_net_combined_projected_tps"]),
        "combined_clears_500": bool(report["combined_clears_500"]),
        "cb3_alone_clears_500": bool(report["cb3_alone_clears_500"]),
        "greedy_matches_dp": bool(report["greedy_matches_dp"]),
        "tree_per_position_width_dp_self_test_passes": bool(report["tree_per_position_width_dp_self_test_passes"]),
        "primary_metric": {"name": "tree_per_position_width_dp_self_test_passes",
                           "value": float(report["tree_per_position_width_dp_self_test_passes"])},
        "test_metric": {"name": "max_net_tps_combined_vs_500",
                        "value": float(combine["combined_vs_500_tps"])},
    }))
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
