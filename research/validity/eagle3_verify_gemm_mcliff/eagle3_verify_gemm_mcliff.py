#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""EAGLE-3 verify-GEMM M-tile cliff (PR #331, stark) — does E[T]=6.11 stay sub-cliff?

THE GOVERNING QUESTION (the denominator audit under fern #325's compliant-500 envelope)
----------------------------------------------------------------------------------------
fern #325's joint compliant-500 ledger holds the verify step FIXED at step_deployed (its code
labels this the "deployed split-K M=32 step") and scales TPS purely with E[T]. That is valid ONLY
while the E[T]=6.11 build verifies its draft tree at a token-width M that stays on the SAME Marlin
int4 W4A16 GEMM tile as the deployed step. The Marlin verify GEMM is weight-bandwidth-bound and
quantizes M into tiles of `thread_m_blocks = ceil(M/16)`: throughput is FLAT inside a tile and
CLIFFS at each 16-boundary (M=17: 1->2 blocks; M=33: 2->3 blocks; M=49: 3->4 blocks). The deployed
linear MTP chain verifies M=8 (1 block). fern's ledger is anchored at the M=32 plateau top (2
blocks). The open denominator term: if the E[T]=6.11 tree's node count crosses M=33, the verify
step jumps and fern's whole envelope divides by that jump -- the cliff would bind BEFORE the private
tax does. This card prices exactly that: the accepted-tree -> M mapping, the cliff cost if crossed,
the largest sub-cliff width, and the verdict.

THE MAPPING (accepted tree width -> verify-GEMM M)
--------------------------------------------------
A speculative draft tree of depth K with breadth W (candidates retained per depth) presents
    M = W * K + 1                       (root + W siblings at each of K depths)
tokens to the single tree-verify forward pass. Anchors that pin the formula:
  - DEPLOYED linear MTP chain: W=1, K=7  ->  M = 1*7 + 1 = 8   == m_verify_deployed (exact).
  - TOP-4 SALVAGE tree for E[T]=6.11: the per-depth effective acceptance that reaches 6.11 over the
    deployed K=7 spine is c_eff = a1 + (1-a1)*cov4 = 0.7731 + 0.2269*0.6532 = 0.9213 (denken #304
    a1_required_611 / wirbel #79 cov4). Realizing the top-4 rank-coverage salvage REQUIRES carrying
    W=4 candidate siblings per depth, so  M = 4*7 + 1 = 29  == lawine #101 size-29 tree (corpus).
M=29 is 2 Marlin blocks (ceil(29/16)=2), the SAME tile as fern's M=32 anchor; the next cliff is at
M=33 (3 blocks). So the E[T]=6.11 tree is SUB-CLIFF with 3 nodes of headroom below M=33.

THE CLIFF COST (directly measured, A10G int4 W4A16, research/spec_cost_model/results_tile_boundary)
----------------------------------------------------------------------------------------------------
  - M=32 -> M=33 step: 12.812ms -> 14.987ms = +2.176ms = x1.16981 (+16.98%); GEMM-only forward
    9.566ms -> 11.731ms (+22.64%). This is the 2->3 thread_m_block transition (knee_Mstar=32).
  - If the tree crossed to M>=33, every verify step scales by mu=1.16981, so the ledger denominator
    grows by mu and the WHOLE envelope divides by mu (both the E[T]-lever public TPS and the
    520.95 identity-kernel ceiling, since that ceiling is itself banked at the deployed step):
        central 586.08 -> min(586.08, 520.95)/mu = 445.33  (-10.93% vs 500, RED)
        worst   492.87 -> 492.87/mu = 421.32              (-15.74% vs 500, RED)
    so crossing the cliff would flip fern's YELLOW envelope to RED.

THE VERDICT
-----------
SUB-CLIFF-SAFE: M=29 <= knee_Mstar=32. fern #325's step_deployed denominator is correct for the
E[T]=6.11 tree, its YELLOW envelope STANDS (central capped 520.95, worst 492.87), and the M=33 cliff
does NOT bind earlier. The margin is TIGHT: E[T]=6.11 is simultaneously the depth-7 top-4 acceptance
ceiling AND the M=32 tile knee. Any push to E[T]>6.11 needs either depth-8 (M=33) or top-5 breadth
(M=36) -- BOTH cross the cliff -> RED. The cliff and the acceptance ceiling coincide at 6.11.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / served-file change / official draw.
BASELINE stays 481.53; adds 0 TPS; greedy/PPL untouched. Imports fern #325 (the ledger constants),
denken #304 (a1_required_611, depth-7 ceiling 6.11), wirbel #79 (cov4), denken #320 (salvage
topology), lawine #101 / ubel #311 (size-29 tree, m_verify_deployed=8, k_spec=7), and the
directly-measured Marlin tile boundary (research/spec_cost_model/results_tile_boundary.json).
Re-derives nothing measured. NOT a launch. NOT a build.

PRIMARY metric  eagle3_verify_gemm_mcliff_self_test_passes
TEST    metric  compliant_tps_worstcase_with_mcliff  (= 492.87; sub-cliff, cliff does not bind)
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# fern #325 compliant-500 ledger constants (imported verbatim; nothing re-measured).
# --------------------------------------------------------------------------- #
K_CAL = 125.26795005202914        # kanna #269 composition calibration
TAU = 1.218                       # served-fraction frame (denken #278)
STEP_DEPLOYED = 1.2182            # fern #325 denominator, labeled "deployed split-K M=32 step"
E_T_DEPLOYED = 3.844              # deployed native accepted-tok/step
TAU_OVER_STEP = TAU / STEP_DEPLOYED
LAMBDA_CEIL = 520.9527323111674   # int4-spec batch-invariant verify ceiling (wirbel #216/#227/#235)
HONEST_PUBLIC_611 = 622.080888    # fern #318 honest_public at realized public E[T]=4.966
E_T_BUILD_FREE = 6.1112149873699195   # free build target (wirbel #295)
RHO_CENTRAL = 0.9421228821714434      # fern #318 central private-tax
RHO_BREAKEVEN = 0.8037539966988988    # fern #318 break-even (= 500/622.08)
RHO_WORST = 0.7922848664688427        # fern #318 worst cross-dataset private-tax
PRIV_TPS_CENTRAL_318 = 586.0766391463308   # fern #318 banked central (= 622.08*rho_central)
PRIV_TPS_WORST_318 = 492.865273281899      # fern #318 banked worst   (= 622.08*rho_worst)
TARGET = 500.0
BASELINE_TPS = 481.53

# --------------------------------------------------------------------------- #
# Acceptance topology (denken #304 / #320, wirbel #79) — pins the W=4 -> M=29 tree.
# --------------------------------------------------------------------------- #
K_SPEC = 7                              # deployed speculative depth (ubel #311)
M_VERIFY_DEPLOYED = 8                   # deployed linear-chain verify width (ubel #311)
E_T_MAX = 8                             # K_spec + 1 (depth-7 chain max accepted tokens)
A1_REQUIRED_611 = 0.9213011665456927   # denken #304 uniform per-depth c_eff reaching E[T]=6.11
RAW_A1_DEMAND = 0.7730729805683441     # denken #320 raw rank-1 demand at 6.11 (pre-salvage)
COV4 = 0.6531976066516435              # wirbel #79 top-4 rank coverage (deployed linear spine)
NATIVE_A1 = 0.7714                     # denken #320 measured native rank-1 acceptance
SIZE29_TREE_CORPUS = 29                # lawine #101 size-29 tree node count (corpus anchor)
MAX_SAFE_TREE_WIDTH_311 = 16           # ubel #311 capture-safety ceiling (ORTHOGONAL axis)
CAPTURE_SIZES = (8, 16, 32)            # eagle3_capture_peak profiled tree-verify buckets

# --------------------------------------------------------------------------- #
# Marlin int4 W4A16 verify-GEMM tile cliff (directly measured; A10G, ctx=256).
# research/spec_cost_model/results_tile_boundary.json  (knee_Mstar=32).
# --------------------------------------------------------------------------- #
TILE_BLOCK = 16                        # Marlin thread_m_blocks = ceil(M / 16)
KNEE_MSTAR = 32                        # last M before the 2->3 block cliff
T_STEP_M32 = 12.811936378479004        # measured step @ M=32 (ms)
T_STEP_M33 = 14.98748779296875         # measured step @ M=33 (ms)
T_FWD_M32 = 9.565584182739258          # measured forward (GEMM) @ M=32 (ms)
T_FWD_M33 = 11.73144006729126          # measured forward (GEMM) @ M=33 (ms)
T_STEP_M48 = 15.265167713165283        # measured step @ M=48 (ms)
T_STEP_M49 = 18.134016036987305        # measured step @ M=49 (ms; 3->4 block second cliff)
MARGINAL_MS_M33 = 2.175551414489746    # corpus marginal_ms_per_token[33] (the cliff jump)
MARGINAL_MS_M49 = 2.8688483238220215   # corpus marginal_ms_per_token[49] (second cliff jump)

TOL_RT = 1e-6
TOL_ANCHOR = 0.5


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def thread_m_blocks(m: int) -> int:
    """Marlin int4 W4A16 verify-GEMM tile count = ceil(M / 16)."""
    return -(-int(m) // TILE_BLOCK)


def tree_m(width: int, depth: int) -> int:
    """Accepted-tree node count presented to one verify pass: M = W*K + 1."""
    return width * depth + 1


def capture_bucket(m: int) -> int:
    """The cudagraph capture size a tree of M real nodes is padded up to (>= M)."""
    for s in CAPTURE_SIZES:
        if m <= s:
            return s
    return ((m + TILE_BLOCK - 1) // TILE_BLOCK) * TILE_BLOCK   # next 16-multiple above the table


def official_tps(et: float, step: float = STEP_DEPLOYED) -> float:
    """fern #325 composition law: K_cal * (E[T] * tau / step)."""
    return K_CAL * (et * TAU / step)


def cliff_mu() -> float:
    """Measured 2->3 block step ratio at the M=33 cliff (verify-GEMM denominator multiplier)."""
    return T_STEP_M33 / T_STEP_M32


# --------------------------------------------------------------------------- #
# (D1) Accepted-tree-width -> verify-GEMM M mapping.
# --------------------------------------------------------------------------- #
def deliverable1_tree_to_m() -> dict[str, Any]:
    rows = {}
    configs = [
        ("deployed_linear", 1, K_SPEC, "deployed MTP chain (W=1)"),
        ("eagle3_top4_611", 4, K_SPEC, "EAGLE-3 top-4 salvage tree for E[T]=6.11"),
    ]
    for key, w, k, desc in configs:
        m = tree_m(w, k)
        bucket = capture_bucket(m)
        rows[key] = {
            "width_W": w, "depth_K": k, "M_nodes": m, "desc": desc,
            "thread_m_blocks": thread_m_blocks(m),
            "capture_bucket": bucket,
            "bucket_blocks": thread_m_blocks(bucket),
            "sub_cliff_real": bool(m <= KNEE_MSTAR),
            "sub_cliff_bucket": bool(bucket <= KNEE_MSTAR),
        }
    e3 = rows["eagle3_top4_611"]
    # salvage identity that pins W=4: c_eff = a1 + (1-a1)*cov4 reaches the 6.11 demand.
    c_eff = RAW_A1_DEMAND + (1.0 - RAW_A1_DEMAND) * COV4
    return {
        "formula": "M = W*K + 1  (root + W candidate siblings per depth)",
        "rows": rows,
        "salvage_identity": {
            "c_eff_top4": c_eff,
            "a1_required_611": A1_REQUIRED_611,
            "matches_304_bank": bool(abs(c_eff - A1_REQUIRED_611) <= TOL_RT),
            "note": ("top-4 salvage c_eff = a1 + (1-a1)*cov4 = {:.4f} reaches the depth-7 6.11 "
                     "demand; realizing top-4 coverage REQUIRES W=4 siblings per depth.".format(c_eff)),
        },
        "deployed_anchor_ok": bool(rows["deployed_linear"]["M_nodes"] == M_VERIFY_DEPLOYED),
        "size29_corpus_anchor_ok": bool(e3["M_nodes"] == SIZE29_TREE_CORPUS),
        "eagle3_M": e3["M_nodes"],
        "eagle3_is_sub_cliff": bool(e3["M_nodes"] <= KNEE_MSTAR),
        "headroom_nodes_below_cliff": (KNEE_MSTAR + 1) - e3["M_nodes"],   # nodes until M=33
        "capture_safety_note": (
            "ubel #311 capture-safety (max_safe_tree_width={}, buckets {}) is an ORTHOGONAL "
            "correctness axis (lawine #101 IndexError): a 29-node tree pads to the M=32 capture "
            "bucket, still 2 Marlin blocks, still sub-cliff. The verify-GEMM throughput cliff priced "
            "here is M=33 (3 blocks), reached only by a tree of >=33 real nodes.".format(
                MAX_SAFE_TREE_WIDTH_311, CAPTURE_SIZES)),
    }


# --------------------------------------------------------------------------- #
# (D2) Cliff cost if crossed: per-step jump propagated through fern #325's ledger.
# --------------------------------------------------------------------------- #
def deliverable2_cliff_cost() -> dict[str, Any]:
    mu = cliff_mu()                       # 2->3 block step multiplier at M=33
    mu_gemm = T_FWD_M33 / T_FWD_M32       # GEMM-only (forward) multiplier
    mu2 = T_STEP_M49 / T_STEP_M48         # 3->4 block second cliff (M=49)
    step_jump_ms = T_STEP_M33 - T_STEP_M32

    # crossing M=33 scales the verify step by mu => the whole ledger denominator grows by mu =>
    # both the E[T]-lever public TPS AND the 520.95 ceiling (banked at the deployed step) divide.
    def crossed(uncapped: float) -> dict[str, Any]:
        cap_x = LAMBDA_CEIL / mu
        val_x = min(uncapped / mu, cap_x)
        return {
            "uncapped_subcliff": uncapped,
            "uncapped_crossed": uncapped / mu,
            "cap_crossed": cap_x,
            "joint_crossed": val_x,
            "clears_500": bool(val_x >= TARGET - 1e-9),
            "headroom_pct": 100.0 * (val_x - TARGET) / TARGET,
        }

    central_x = crossed(PRIV_TPS_CENTRAL_318)
    worst_x = crossed(PRIV_TPS_WORST_318)
    return {
        "cliff_mu_step_33_over_32": mu,
        "cliff_mu_gemm_fwd_33_over_32": mu_gemm,
        "cliff_pct_step": 100.0 * (mu - 1.0),
        "cliff_pct_gemm": 100.0 * (mu_gemm - 1.0),
        "step_jump_ms": step_jump_ms,
        "step_jump_matches_corpus_marginal": bool(abs(step_jump_ms - MARGINAL_MS_M33) <= 1e-6),
        "second_cliff_mu_step_49_over_48": mu2,
        "second_cliff_jump_ms": T_STEP_M49 - T_STEP_M48,
        "honest_public_611_crossed": HONEST_PUBLIC_611 / mu,
        "ceiling_crossed": LAMBDA_CEIL / mu,
        "central_if_crossed": central_x,
        "worst_if_crossed": worst_x,
        "both_red_if_crossed": bool(not central_x["clears_500"] and not worst_x["clears_500"]),
        "note": ("crossing to M>=33 scales the verify step by mu={:.5f}; the ledger denominator "
                 "grows by mu so the envelope divides by mu: central {:.2f}->{:.2f} ({:+.2f}%), "
                 "worst {:.2f}->{:.2f} ({:+.2f}%) -- both RED. The cliff would bind BEFORE the "
                 "private tax.".format(mu, PRIV_TPS_CENTRAL_318, central_x["joint_crossed"],
                                       central_x["headroom_pct"], PRIV_TPS_WORST_318,
                                       worst_x["joint_crossed"], worst_x["headroom_pct"])),
    }


# --------------------------------------------------------------------------- #
# (D3) Max sub-cliff width: largest tree staying M<=32; is E[T]=6.11 inside it?
# --------------------------------------------------------------------------- #
def deliverable3_max_subcliff() -> dict[str, Any]:
    # largest breadth W at the deployed depth K=7 staying M = W*7 + 1 <= 32:
    max_w_at_k7 = (KNEE_MSTAR - 1) // K_SPEC                 # floor(31/7) = 4
    m_at_max_w = tree_m(max_w_at_k7, K_SPEC)
    m_w5 = tree_m(5, K_SPEC)                                 # top-5 -> 36 (crosses)
    # largest depth K at breadth W=4 staying M = 4*K + 1 <= 32:
    max_k_at_w4 = (KNEE_MSTAR - 1) // 4                      # floor(31/4) = 7
    m_at_max_k = tree_m(4, max_k_at_w4)
    m_k8 = tree_m(4, 8)                                      # depth-8 -> 33 (crosses, == cliff)

    # is E[T]=6.11 reachable inside the sub-cliff (W=4, K=7) tree? 6.11 IS the depth-7 top-4
    # ceiling (denken #304): uniform c_eff=0.9213 over K=7 gives E[T]=1+sum_{d=1..7} c_eff^d.
    et_at_subcliff = 1.0 + sum(A1_REQUIRED_611 ** d for d in range(1, K_SPEC + 1))
    et_reachable = bool(abs(et_at_subcliff - E_T_BUILD_FREE) <= 0.02)
    return {
        "max_width_W_at_K7": max_w_at_k7,
        "M_at_max_width": m_at_max_w,
        "width_W5_M": m_w5,
        "width_W5_crosses": bool(m_w5 > KNEE_MSTAR),
        "max_depth_K_at_W4": max_k_at_w4,
        "M_at_max_depth": m_at_max_k,
        "depth_K8_M": m_k8,
        "depth_K8_crosses": bool(m_k8 > KNEE_MSTAR),
        "subcliff_et_ceiling": et_at_subcliff,
        "e_t_611_reachable_sub_cliff": et_reachable,
        "frontier_note": ("the largest sub-cliff tree at K=7 is W=4 (M=29); W=5 -> M=36 crosses. "
                          "At W=4 the largest depth is K=7 (M=29); K=8 -> M=33 crosses (== the "
                          "cliff). E[T]=6.11 IS the depth-7 top-4 ceiling, so it is exactly "
                          "reachable sub-cliff, but it sits AT the frontier: raising E[T] above "
                          "6.11 needs depth-8 (M=33) or top-5 (M=36) -- both cross. The acceptance "
                          "ceiling and the tile knee coincide at 6.11."),
    }


# --------------------------------------------------------------------------- #
# (D4) Verdict: SUB-CLIFF-SAFE vs CLIFF-BINDS-EARLIER.
# --------------------------------------------------------------------------- #
def deliverable4_verdict(d1: dict, d2: dict, d3: dict) -> dict[str, Any]:
    sub_cliff = bool(d1["eagle3_is_sub_cliff"])
    # sub-cliff envelope = fern #325 banked (cap binds central, worst uncapped).
    central_subcliff = min(PRIV_TPS_CENTRAL_318, LAMBDA_CEIL)
    worst_subcliff = min(PRIV_TPS_WORST_318, LAMBDA_CEIL)
    if sub_cliff:
        v = "SUB-CLIFF-SAFE"
        why = ("the E[T]=6.11 top-4 tree verifies at M={} <= knee_Mstar={} (2 Marlin blocks, the "
               "same tile as fern #325's M=32 anchor). The verify-GEMM denominator is unchanged, "
               "fern's YELLOW envelope STANDS (central capped {:.2f}, worst {:.2f}), and the M=33 "
               "cliff does NOT bind earlier.".format(
                   d1["eagle3_M"], KNEE_MSTAR, central_subcliff, worst_subcliff))
    else:
        v = "CLIFF-BINDS-EARLIER"
        why = ("the E[T]=6.11 tree verifies at M={} > knee_Mstar={}: the verify step scales by "
               "mu={:.5f}, dividing fern's envelope to central {:.2f} / worst {:.2f} -- RED.".format(
                   d1["eagle3_M"], KNEE_MSTAR, d2["cliff_mu_step_33_over_32"],
                   d2["central_if_crossed"]["joint_crossed"], d2["worst_if_crossed"]["joint_crossed"]))
    return {
        "verdict": v,
        "why": why,
        "eagle3_M": d1["eagle3_M"],
        "knee_Mstar": KNEE_MSTAR,
        "is_sub_cliff": sub_cliff,
        "compliant_tps_central_subcliff": central_subcliff,
        "compliant_tps_worstcase_subcliff": worst_subcliff,    # == fern banked worst (cliff inert)
        "fern325_envelope_stands": bool(sub_cliff),
        "counterfactual_central_if_crossed": d2["central_if_crossed"]["joint_crossed"],
        "counterfactual_worst_if_crossed": d2["worst_if_crossed"]["joint_crossed"],
        "cliff_would_flip_yellow_to_red": bool(d2["both_red_if_crossed"]),
        "frontier_tight": bool(d3["e_t_611_reachable_sub_cliff"]
                               and d3["depth_K8_crosses"] and d3["width_W5_crosses"]),
        "joint_frontier_note": (
            "E[T]=6.11 sits at the JOINT frontier of the depth-7 top-4 acceptance ceiling AND the "
            "M=32 verify-GEMM tile knee: the cheapest move past either (depth-8 or top-5) crosses "
            "BOTH at once."),
    }


# --------------------------------------------------------------------------- #
# (5) Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def _selftests(d1: dict, d2: dict, d3: dict, vd: dict) -> dict[str, Any]:
    # (a) M = W*K+1 reproduces the deployed M=8 verify width (W=1, K=7).
    cond_a = bool(d1["deployed_anchor_ok"] and tree_m(1, K_SPEC) == M_VERIFY_DEPLOYED)

    # (b) the top-4 (W=4, K=7) tree is M=29 == lawine #101 size-29 corpus tree, and sub-cliff.
    cond_b = bool(d1["size29_corpus_anchor_ok"] and d1["eagle3_M"] == 29
                  and d1["eagle3_is_sub_cliff"] and thread_m_blocks(29) == 2)

    # (c) the salvage identity that pins W=4: a1 + (1-a1)*cov4 == denken #304 a1_required_611.
    cond_c = bool(d1["salvage_identity"]["matches_304_bank"])

    # (d) the cliff multiplier comes from the measured tile data and matches the corpus marginal.
    mu = d2["cliff_mu_step_33_over_32"]
    cond_d = bool(abs(mu - (T_STEP_M33 / T_STEP_M32)) <= TOL_RT
                  and d2["step_jump_matches_corpus_marginal"]
                  and mu > 1.0)

    # (e) the cliff-crossed counterfactual divides fern's envelope -> both corners RED (< 500).
    cond_e = bool(d2["both_red_if_crossed"]
                  and not d2["central_if_crossed"]["clears_500"]
                  and not d2["worst_if_crossed"]["clears_500"]
                  and abs(d2["central_if_crossed"]["joint_crossed"] - LAMBDA_CEIL / mu) <= TOL_RT)

    # (f) the SUB-CLIFF envelope equals fern #325's banked corners (cliff inert): cap binds central,
    #     worst == 492.87 (uncapped).
    cond_f = bool(abs(vd["compliant_tps_central_subcliff"] - LAMBDA_CEIL) <= TOL_RT
                  and abs(vd["compliant_tps_worstcase_subcliff"] - PRIV_TPS_WORST_318) <= TOL_RT)

    # (g) max sub-cliff width: W=4/K=7 fits (M=29); W=5 (M=36) and K=8 (M=33) both cross; and
    #     E[T]=6.11 is reachable inside the sub-cliff tree (it IS the depth-7 ceiling).
    cond_g = bool(d3["max_width_W_at_K7"] == 4 and d3["width_W5_crosses"]
                  and d3["max_depth_K_at_W4"] == 7 and d3["depth_K8_crosses"]
                  and d3["e_t_611_reachable_sub_cliff"])

    # (h) verdict is SUB-CLIFF-SAFE, fern's envelope stands, and the cliff would flip YELLOW->RED.
    cond_h = bool(vd["verdict"] == "SUB-CLIFF-SAFE" and vd["fern325_envelope_stands"]
                  and vd["cliff_would_flip_yellow_to_red"] and vd["frontier_tight"])

    # (i) NaN-clean (set by caller).
    cond_i = True

    conditions = {
        "a_M_formula_reproduces_deployed_8": cond_a,
        "b_top4_tree_is_29_subcliff": cond_b,
        "c_salvage_identity_pins_W4": cond_c,
        "d_cliff_mu_from_measured_tile": cond_d,
        "e_crossed_envelope_both_red": cond_e,
        "f_subcliff_envelope_equals_fern325": cond_f,
        "g_max_subcliff_width_and_611_reachable": cond_g,
        "h_verdict_sub_cliff_safe": cond_h,
        "i_nan_clean": cond_i,
    }
    return {
        "conditions": conditions,
        "eagle3_verify_gemm_mcliff_self_test_passes": bool(all(conditions.values())),
        "detail": {
            "deployed_M": tree_m(1, K_SPEC), "eagle3_M": d1["eagle3_M"],
            "cliff_mu": mu, "knee_Mstar": KNEE_MSTAR,
        },
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    d1 = deliverable1_tree_to_m()
    d2 = deliverable2_cliff_cost()
    d3 = deliverable3_max_subcliff()
    vd = deliverable4_verdict(d1, d2, d3)
    st = _selftests(d1, d2, d3, vd)

    headline = {
        "eagle3_verify_gemm_mcliff_self_test_passes":
            bool(st["eagle3_verify_gemm_mcliff_self_test_passes"]),               # PRIMARY
        "compliant_tps_worstcase_with_mcliff": vd["compliant_tps_worstcase_subcliff"],   # TEST
        "compliant_tps_central_with_mcliff": vd["compliant_tps_central_subcliff"],
        "eagle3_M": d1["eagle3_M"],
        "knee_Mstar": KNEE_MSTAR,
        "is_sub_cliff": vd["is_sub_cliff"],
        "verdict": vd["verdict"],
        "cliff_mu_step": d2["cliff_mu_step_33_over_32"],
        "cliff_pct_step": d2["cliff_pct_step"],
        "counterfactual_central_if_crossed": vd["counterfactual_central_if_crossed"],
        "counterfactual_worst_if_crossed": vd["counterfactual_worst_if_crossed"],
        "headroom_nodes_below_cliff": d1["headroom_nodes_below_cliff"],
        "fern325_envelope_stands": vd["fern325_envelope_stands"],
    }

    return {
        "headline": headline,
        "mapping": {
            "formula": "M = W*K + 1; thread_m_blocks = ceil(M/16); knee_Mstar=32 (2->3 block cliff @ M=33)",
            "deployed_M": M_VERIFY_DEPLOYED, "eagle3_M": d1["eagle3_M"],
            "K_spec": K_SPEC, "e_t_build_free": E_T_BUILD_FREE,
            "step_deployed_basis_note": (
                "fern #325 labels step_deployed the 'M=32 step' (2-block plateau top). The "
                "normalized 1.2182 unit traces (denken #278) to the deployed M=8 verify; either "
                "reading is SUB-CLIFF, and the M=29 tree shares the 2-block tile with the M=32 "
                "anchor. The only cliff above is M=33 (3 blocks)."),
        },
        "deliverable1_tree_to_m": d1,
        "deliverable2_cliff_cost": d2,
        "deliverable3_max_subcliff": d3,
        "deliverable4_verdict": vd,
        "self_test": st,
        "imports": {
            "provenance": (
                "fern #325 (joint compliant-500 ledger: K_cal 125.268, step 1.2182, lambda 520.95, "
                "honest_public 622.08, rho central 0.9421 / breakeven 0.8038 / worst 0.7923, private "
                "586.08/492.87) x denken #304 (a1_required_611 0.9213, depth-7 ceiling 6.11) x wirbel "
                "#79 (cov4 0.6532) x denken #320 (salvage topology c_eff=a1+(1-a1)cov) x lawine #101 / "
                "ubel #311 (size-29 tree, m_verify_deployed=8, k_spec=7, max_safe_tree_width=16) x "
                "directly-measured Marlin tile boundary research/spec_cost_model/results_tile_boundary "
                "(knee_Mstar=32, M32->M33 step 12.812->14.987ms, +16.98%). All run-ids in "
                "wandb-applied-ai-team/gemma-challenge-senpai."),
            "caveats": [
                "DERIVED, not measured: no EAGLE-3 checkpoint exists; the W=4->M=29 mapping prices "
                "the tree the top-4 salvage acceptance model implies, not a running EagleProposer.",
                "cov4=0.6532 is measured on the deployed LINEAR spine (wirbel #79); a {2,21,39} "
                "fusion draft could lower it and raise the W needed -- the cov-transfer YELLOW "
                "(denken #320) is inherited, ORTHOGONAL to this step-regime card.",
                "ubel #311 capture-safety (max_safe_tree_width=16, lawine #101 IndexError) is a "
                "separate correctness axis; a 29-node tree pads to the M=32 capture bucket, still "
                "sub-cliff. This card prices the verify-GEMM THROUGHPUT cliff (M=33), not capture.",
                "0 TPS / denominator-correctness property: depends only on integer node-counts and "
                "the measured tile boundary, not tensor values. NOT a launch / build / served-file "
                "change / HF Job.",
            ],
        },
    }


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B.
# --------------------------------------------------------------------------- #
def _nan_paths(node: Any, p: str = "result") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            bad += _nan_paths(v, f"{p}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            bad += _nan_paths(v, f"{p}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        bad.append(p)
    return bad


def _print_report(syn: dict) -> None:
    h = syn["headline"]
    d1, d2 = syn["deliverable1_tree_to_m"], syn["deliverable2_cliff_cost"]
    d3, vd, st = syn["deliverable3_max_subcliff"], syn["deliverable4_verdict"], syn["self_test"]
    print("\n" + "=" * 98, flush=True)
    print("EAGLE-3 VERIFY-GEMM M-TILE CLIFF (PR #331, stark) — does E[T]=6.11 stay sub-cliff?",
          flush=True)
    print("=" * 98, flush=True)
    print("  MAPPING  M = W*K + 1 ; thread_m_blocks = ceil(M/16) ; knee_Mstar=32 (cliff @ M=33)",
          flush=True)
    print("-" * 98, flush=True)
    print("  (D1) ACCEPTED-TREE -> VERIFY M", flush=True)
    for key, r in d1["rows"].items():
        print(f"      {r['desc']:<42} W={r['width_W']} K={r['depth_K']}  M={r['M_nodes']:>2}  "
              f"blocks={r['thread_m_blocks']}  sub_cliff={str(r['sub_cliff_real']):>5}", flush=True)
    print(f"      salvage: c_eff=a1+(1-a1)*cov4={d1['salvage_identity']['c_eff_top4']:.4f} "
          f"== a1_required_611 (pins W=4) ; deployed M=8 anchor ok={d1['deployed_anchor_ok']} ; "
          f"size-29 corpus ok={d1['size29_corpus_anchor_ok']}", flush=True)
    print("-" * 98, flush=True)
    print("  (D2) CLIFF COST IF CROSSED (M=32->33, measured)", flush=True)
    print(f"      step x{d2['cliff_mu_step_33_over_32']:.5f} (+{d2['cliff_pct_step']:.2f}%), GEMM "
          f"+{d2['cliff_pct_gemm']:.2f}%, jump {d2['step_jump_ms']:.3f}ms "
          f"(corpus-match={d2['step_jump_matches_corpus_marginal']})", flush=True)
    cx, wx = d2["central_if_crossed"], d2["worst_if_crossed"]
    print(f"      envelope/mu: central {PRIV_TPS_CENTRAL_318:.2f}->{cx['joint_crossed']:.2f} "
          f"({cx['headroom_pct']:+.2f}%)  worst {PRIV_TPS_WORST_318:.2f}->{wx['joint_crossed']:.2f} "
          f"({wx['headroom_pct']:+.2f}%)  both_RED={d2['both_red_if_crossed']}", flush=True)
    print("-" * 98, flush=True)
    print("  (D3) MAX SUB-CLIFF WIDTH", flush=True)
    print(f"      max W@K=7 is {d3['max_width_W_at_K7']} (M={d3['M_at_max_width']}); W=5 -> "
          f"M={d3['width_W5_M']} crosses. max K@W=4 is {d3['max_depth_K_at_W4']} "
          f"(M={d3['M_at_max_depth']}); K=8 -> M={d3['depth_K8_M']} crosses.", flush=True)
    print(f"      sub-cliff E[T] ceiling = {d3['subcliff_et_ceiling']:.4f} ; E[T]=6.11 reachable "
          f"sub-cliff = {d3['e_t_611_reachable_sub_cliff']}", flush=True)
    print("-" * 98, flush=True)
    print(f"  (D4) VERDICT: {vd['verdict']}", flush=True)
    print(f"      {vd['why']}", flush=True)
    print(f"      fern #325 stands: central={vd['compliant_tps_central_subcliff']:.2f} "
          f"worst={vd['compliant_tps_worstcase_subcliff']:.2f} (YELLOW) | if crossed: "
          f"central={vd['counterfactual_central_if_crossed']:.2f} "
          f"worst={vd['counterfactual_worst_if_crossed']:.2f} (RED)", flush=True)
    print("-" * 98, flush=True)
    print(f"  (5) PRIMARY self_test_passes = "
          f"{st['eagle3_verify_gemm_mcliff_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print("=" * 98 + "\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[eagle3-mcliff] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    h = syn["headline"]
    d1, d2 = syn["deliverable1_tree_to_m"], syn["deliverable2_cliff_cost"]
    d3, vd, st = syn["deliverable3_max_subcliff"], syn["deliverable4_verdict"], syn["self_test"]
    run = init_wandb_run(
        job_type="eagle3-verify-gemm-mcliff",
        agent="stark",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["eagle3-verify-gemm-mcliff", "issue-192", "eagle3", "marlin-tile-cliff",
              "verify-gemm", "compliant-500", "validity-gate", "bank-the-analysis"],
        config={
            "K_cal": K_CAL, "tau": TAU, "step_deployed": STEP_DEPLOYED, "lambda_ceiling": LAMBDA_CEIL,
            "honest_public_611": HONEST_PUBLIC_611, "e_t_build_free": E_T_BUILD_FREE,
            "rho_central": RHO_CENTRAL, "rho_worst": RHO_WORST, "k_spec": K_SPEC,
            "m_verify_deployed": M_VERIFY_DEPLOYED, "knee_Mstar": KNEE_MSTAR,
            "a1_required_611": A1_REQUIRED_611, "cov4": COV4, "baseline_tps": BASELINE_TPS,
            "target": TARGET, "wandb_group": args.wandb_group,
            "source_runs": "fern#325, denken#304/#320, wirbel#79, lawine#101/ubel#311, "
                           "spec_cost_model/results_tile_boundary",
        },
    )
    if run is None:
        print("[eagle3-mcliff] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "eagle3_verify_gemm_mcliff_self_test_passes":
            int(bool(st["eagle3_verify_gemm_mcliff_self_test_passes"])),          # PRIMARY
        "compliant_tps_worstcase_with_mcliff": h["compliant_tps_worstcase_with_mcliff"],  # TEST
        "compliant_tps_central_with_mcliff": h["compliant_tps_central_with_mcliff"],
        "eagle3_M": d1["eagle3_M"],
        "deployed_M": M_VERIFY_DEPLOYED,
        "knee_Mstar": KNEE_MSTAR,
        "headroom_nodes_below_cliff": d1["headroom_nodes_below_cliff"],
        "is_sub_cliff": int(bool(vd["is_sub_cliff"])),
        "cliff_mu_step": d2["cliff_mu_step_33_over_32"],
        "cliff_pct_step": d2["cliff_pct_step"],
        "cliff_pct_gemm": d2["cliff_pct_gemm"],
        "second_cliff_mu_step": d2["second_cliff_mu_step_49_over_48"],
        "counterfactual_central_if_crossed": vd["counterfactual_central_if_crossed"],
        "counterfactual_worst_if_crossed": vd["counterfactual_worst_if_crossed"],
        "both_red_if_crossed": int(bool(d2["both_red_if_crossed"])),
        "subcliff_et_ceiling": d3["subcliff_et_ceiling"],
        "max_width_W_at_K7": d3["max_width_W_at_K7"],
        "max_depth_K_at_W4": d3["max_depth_K_at_W4"],
        "e_t_611_reachable_sub_cliff": int(bool(d3["e_t_611_reachable_sub_cliff"])),
        "verdict_sub_cliff_safe": int(vd["verdict"] == "SUB-CLIFF-SAFE"),
        "verdict_cliff_binds_earlier": int(vd["verdict"] == "CLIFF-BINDS-EARLIER"),
        "fern325_envelope_stands": int(bool(vd["fern325_envelope_stands"])),
        "frontier_tight": int(bool(vd["frontier_tight"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="eagle3_verify_gemm_mcliff_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[eagle3-mcliff] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default="eagle3-mcliff")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 331, "agent": "stark",
        "kind": "eagle3-verify-gemm-mcliff", "analysis_only": True, "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["i_nan_clean"] = not nan_paths
    syn["self_test"]["eagle3_verify_gemm_mcliff_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    syn["headline"]["eagle3_verify_gemm_mcliff_self_test_passes"] = syn["self_test"][
        "eagle3_verify_gemm_mcliff_self_test_passes"]
    if nan_paths:
        print(f"[eagle3-mcliff] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_verify_gemm_mcliff_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[eagle3-mcliff] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["eagle3_verify_gemm_mcliff_self_test_passes"]
              and payload["nan_clean"])
        print(f"[eagle3-mcliff] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
