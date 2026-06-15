#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Strict sub-saturation verify (PR #358, stark) -- does a sub-80-SM M escape the 473.5 tax?

THE GOVERNING QUESTION (the #319 STRICT contract)
-------------------------------------------------
The human reversed #124 (issue #319, 2026-06-15 10:56Z): STRICT byte-exact greedy-token-
identity is the live contract. denken #332 (y5cl0ena) priced the strict ceiling at 473.53:
the batched multi-token verify forward is OCCUPANCY-SATURATED (the deployed M=8 verify spawns
N_FULL_3D = 96 CTAs > the A10G's 80 SMs), so a DETERMINISTIC (fixed-split, ordered-combine)
schedule can keep only the non-reduction (M-query x kv-head) grid -- 6/80 SMs -- recovering
recovery_phi = 0.075 of the forgone split-KV parallelism. The rest is the determinism tax.

THE HYPOTHESIS THIS CARD TESTS (not assumes)
--------------------------------------------
*Below the 80-SM occupancy wall*, a smaller-M / narrower-tree verify has SM headroom -- the
intuition is that idle SMs leave room for a deterministic single-pass reduction WITHOUT
forgoing the parallel split, so a sub-saturation verify might pay a LOWER determinism tax
(higher recovery_phi). Does any (M, tree-shape) BELOW the wall yield recovery_phi above 0.255
(no-regression break-even) or 0.591 (>500 break-even), and does the resulting strict ceiling
  ceiling(M) = LAMBDA1_CEIL * (1 - tax(M)) * (E[T]_M / E[T]_8)
beat 473.5 -- OR does the E[T] loss from a smaller M outpace the recovery_phi gain?

THE DUAL phi NAMING (reconcile the anchors before reading the math)
-------------------------------------------------------------------
denken #332 banks GEOMETRIC_PHI_332 = 0.925 = the FORGONE split-KV fraction. This card's
recovery_phi == 1 - forgone is the *recovery* fraction the deployed M=8 schedule keeps:
recovery_phi(M=8) = 1 - 0.925 = 0.075. The break-even RECOVERY fractions (cite fern #349):
  recovery >= 0.255 -> no-regression (floor <= 7.332%, ceiling >= 481.53)
  recovery >= 0.591 -> clears 500    (floor <= 4.022%, ceiling >= 500.0)
tax(M) = FLOOR_AT_PHI1_327 * (1 - recovery_phi(M)) = the determinism floor as a step fraction;
ceiling(M) = LAMBDA1_CEIL * (1 - tax(M)) * E[T]_M/E[T]_8 (a strict ceiling). At M=8 this
round-trips denken #332 EXACTLY: recovery 0.075, floor 0.09103, ceiling 473.5296.

THE CTA-vs-M GEOMETRY (deliverable 1, denken #332's launch model)
-----------------------------------------------------------------
The vLLM unified_attention verify tiles M query rows with block_q = BLOCK_M//(q_heads/kv_heads)
= 16//4 = 4 rows/CTA -> total_num_q_blocks(M) = M//block_q + NUM_SEQS, non-reduction grid
N_nonreduction(M) = total_num_q_blocks(M) * NUM_KV_HEADS, and the adaptive 3D split-KV grid
N_full_3d(M) = N_nonreduction(M) * NUM_PAR_SOFTMAX_SEGMENTS (16-way). At M=8: 3 q-blocks ->
N_nonreduction=6, N_full_3d=96 (> 80 -> saturated). M_sat (continuous) solves N_full_3d=80 ->
M_sat = block_q*(80/(NUM_KV_HEADS*SEGMENTS) - NUM_SEQS) = 4*(80/32 - 1) = 6.0. Sub-saturation
is M < 6: M in {2,4}. A WIDER tree only ADDS verify query rows -> MORE CTAs -> deeper into
saturation, never below; the only direction under the wall is smaller M / narrower tree.

THE DECISIVE TENSION (deliverable 2-3)
--------------------------------------
recovery_phi(M) = min(1, N_nonreduction(M)/SMs) GROWS with M (more non-reduction q-blocks),
so on denken's faithful geometry sub-saturation LOWERS recovery (M=4 -> 0.050, M=2 -> 0.025,
both < the saturated 0.075) AND lowers E[T] -- a DOUBLE penalty. A steelman headroom model
(idle SMs grant a free deterministic reduction: recovery_phi_headroom(M) = min(1,
N_full_3d(M)/SMs) when sub-saturated) raises recovery at small M, but the E[T] loss still
dominates. Either way the strict ceiling at every sub-saturation M lands FAR below 473.5.
The verify slack is BANDWIDTH-bound (AI 7.88 << ridge 208; the saturated 96-CTA path still
sits at 34.9% BW, stark #345), so idle SMs cannot convert to recovery anyway -- the steelman
is generous and STILL fails.

WHAT THIS CARD DOES (CPU-analytic over banked numbers; 0 GPU, 0 TPS)
-------------------------------------------------------------------
1. CTA-vs-M map: N_nonreduction(M), N_full_3d(M), M_sat; sub- vs super-saturation per M.
2. recovery_phi(M) (faithful + steelman headroom) and tax(M); E[T]_M from the #289 a_k
   profile (truncate/extend); ceiling(M) = LAMBDA1_CEIL*(1-tax)*E[T]_M/E[T]_8 over
   M in {2,4,8,16,32}; the argmax M*; the sub-saturation-only max.
3. Verdict: sub_saturation_escapes_473 (does any M < M_sat beat 473.5?) -- honest.
4. Self-test (PRIMARY): (a) M=8 round-trips denken #332 recovery 0.075 / ceiling 473.5296
   (<=1e-6); (b) E[T]_8 round-trips #289 3.8512; (c) phi(M)+ceiling(M) NaN-clean, finite;
   (d) M_sat identified + CTA map dimensionally consistent (N_nonreduction*SEGMENTS=N_full_3d);
   (e) verdict bool set.

HONEST SCOPE
------------
0 TPS. BASELINE 481.53 unchanged. NO GPU, NO model forward, NO training, NO served-file
change, NO HF Job, NO submission, NO launch. Imports VERBATIM and re-derives nothing measured:
denken #332 y5cl0ena (recovery 0.075 / forgone 0.925, floor 0.09841, geo-floor 0.09103, strict
ceiling 473.5296, AI 7.88, 96 CTAs > 80 SMs, ceiling 520.953, >500 budget 4.022%); kanna #289
fi34s269 (a_k profile, E[T]=3.8512); fern #349 u8vmtji0 (recovery->ceiling map: 0.075->473.5 /
0.255->482.76 / 0.591->500); stark #345 (batched-verify floor method-independent). This RESOLVES
whether the occupancy wall is escapable downward; it is NOT a kernel-buildability proof. NOT a
launch / build / served-file change.

PRIMARY metric  strict_sub_saturation_self_test_passes
TEST    metric  max_strict_ceiling_over_M  (float)
                + sub_saturation_escapes_473  (bool, expect False)
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
# Imported EXACT from banked W&B runs -- DO NOT re-derive. Full precision; the
# displayed 4-dp forms (473.53, 0.075, 3.8512, ...) are asserted in the self-test.
# All runs live in wandb-applied-ai-team/gemma-challenge-senpai.
# --------------------------------------------------------------------------- #
# ---- denken #332 (y5cl0ena): the saturated strict verify tax + launch geometry ----
GEOMETRIC_PHI_332 = 0.925                       # FORGONE split-KV parallelism fraction (saturated M=8)
RECOVERY_PHI_332 = 1.0 - GEOMETRIC_PHI_332      # 0.075 = recovery this card sweeps in M
FLOOR_AT_PHI1_327 = 0.09841249119201488         # determinism floor at full forgone slack (denken #327)
FLOOR_AT_GEO_332 = 0.09103155435261377          # floor at the saturated geo phi (= FLOOR_AT_PHI1*0.925)
STRICT_COMPLIANT_CEILING_332 = 473.5295953446407  # = LAMBDA1_CEIL*(1-FLOOR_AT_GEO_332); the 473.53 cap
LAMBDA1_CEIL = 520.9527323111674                # lambda=1 step-side ceiling (int4-spec batch-inv verify)
SDPA_BW_UTIL = 0.34883864849061247              # verify-attention BW utilisation (the floor)
SDPA_AI_FLOP_PER_BYTE = 7.880597014925373       # AI 7.88 flop/byte -- << ridge -> bandwidth-bound
RIDGE_AI = 208.33333333333334                   # A10G roofline ridge (600 GB/s, fp16 compute)
N_FULL_3D_CTAS_332 = 96                          # adaptive 3D split-KV verify CTAs at M=8 (> 80 SMs)
A10G_SMS = 80                                    # A10G SM count (occupancy denominator)

# verify launch-grid constants (denken #332 / PR #279 / splitkv_verify_patch / PR #39).
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
BLOCK_M = 16                                     # vLLM Triton BLOCK_M
NUM_SEQS = 1                                     # concurrency = 1 single sequence
NUM_PAR_SOFTMAX_SEGMENTS = 16                    # 3D split-KV reduction segments
BLOCK_Q = max(1, BLOCK_M // (NUM_Q_HEADS // NUM_KV_HEADS))   # 16 // 4 = 4 query rows per CTA

# ---- kanna #289 (fi34s269): deployed EAGLE-3-lane per-position conditional acceptance + E[T] ----
A_K_EAGLE3_289 = [
    0.7292532942898975,   # a_1 (the deployed cliff)
    0.759556697719242,    # a_2
    0.7929794882639035,   # a_3
    0.8228,               # a_4
    0.8348727920920435,   # a_5
    0.8357919254658385,   # a_6
    0.8464932652113331,   # a_7
]
E_T_EAGLE3_289 = 3.851185944363104              # E[T]_8 = 1 + sum cumprod(a_k) -- the deployed head's E[T]

# ---- wirbel #213 / advisor #192: the two recovery break-even bars (cite fern #349) ----
BUDGET_LAMBDA1_FRAC_213 = 0.07331808522875782   # no-regression floor budget (omega dropping 520.953->481.53)
BUDGET_500_FRAC_192 = 0.040220025755911104       # >500 floor budget = 1 - 500/520.953 (operative bar)

# fern #349 (u8vmtji0): the recovery->ceiling calibration anchors (displayed, asserted in self-test).
FERN_MAP_349 = [
    (0.075, 473.53),   # saturated
    (0.255, 482.76),   # no-regression break-even
    (0.591, 500.0),    # >500 break-even
]

BASELINE_TPS = 481.53
TARGET = 500.0
K_CAL = 125.268                                 # official = K_cal * E[T]
M_GRID = [2, 4, 8, 16, 32]
M_DEPLOYED = 8

TOL_EXACT = 1e-9
TOL_ROUNDTRIP = 1e-6
TOL_DISPLAY = 5e-4


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Core laws.
# --------------------------------------------------------------------------- #
def et_of_M(M: int, a: list[float] = A_K_EAGLE3_289) -> float:
    """E[T]_M = 1 + sum_{k=1}^{M-1} prod_{j<=k} a_j -- the survival sum over (M-1) draft
    positions (kanna #289). For M-1 > len(a) the profile is EXTENDED by repeating the last
    a_k (acceptance saturates); for M-1 < len(a) it is TRUNCATED. M=8 uses a_1..a_7 ->
    round-trips E_T_EAGLE3_289 = 3.8512."""
    n_positions = max(0, M - 1)
    s, prod = 0.0, 1.0
    for k in range(n_positions):
        ak = a[k] if k < len(a) else a[-1]
        prod *= ak
        s += prod
    return 1.0 + s


def total_num_q_blocks(M: int) -> int:
    """vLLM unified_attention q-block count: M//BLOCK_Q + NUM_SEQS (PR #279 grid logic)."""
    return M // BLOCK_Q + NUM_SEQS


def n_nonreduction_ctas(M: int) -> int:
    """The deterministic-COMPATIBLE 2D grid: q-blocks x kv-heads. head_dim is the QK^T
    contraction (a reduction), excluded. This is what a fixed-split schedule keeps."""
    return total_num_q_blocks(M) * NUM_KV_HEADS


def n_full_3d_ctas(M: int) -> int:
    """The adaptive 3D split-KV grid: non-reduction x the 16-way KV-segment split. At M=8 = 96."""
    return n_nonreduction_ctas(M) * NUM_PAR_SOFTMAX_SEGMENTS


def m_sat_continuous() -> float:
    """The continuous M where N_full_3d(M) crosses A10G_SMS. N_full_3d = (M/BLOCK_Q + NUM_SEQS)
    * NUM_KV_HEADS * SEGMENTS = SMs -> M = BLOCK_Q*(SMs/(NUM_KV_HEADS*SEGMENTS) - NUM_SEQS)."""
    return BLOCK_Q * (A10G_SMS / (NUM_KV_HEADS * NUM_PAR_SOFTMAX_SEGMENTS) - NUM_SEQS)


def recovery_phi_faithful(M: int) -> float:
    """denken #332's faithful model: recovery = the non-reduction grid's OWN occupancy =
    min(1, N_nonreduction(M)/SMs). The deterministic schedule keeps only the non-reduction
    CTAs; the KV-split reduction is forgone. GROWS with M -> M=8 -> 6/80 = 0.075 (anchor)."""
    return min(1.0, n_nonreduction_ctas(M) / A10G_SMS)


def recovery_phi_headroom(M: int) -> float:
    """STEELMAN (generous to the hypothesis): when sub-saturated (N_full_3d < SMs), assume the
    idle SMs grant a FREE deterministic parallel reduction -> the schedule reconstructs the
    full N_full_3d grid deterministically, recovery = min(1, N_full_3d(M)/SMs). When saturated
    (N_full_3d >= SMs) no idle headroom exists -> falls back to the faithful non-reduction
    occupancy. This is the MOST optimistic defensible reading of the hypothesis's intuition;
    it is physically unlikely because the verify slack is BW-bound (AI 7.88 << ridge), not
    occupancy-bound, so idle SMs cannot convert to recovery -- carried only to show the
    verdict is robust even when we GRANT the headroom story."""
    full3d = n_full_3d_ctas(M)
    if full3d < A10G_SMS:
        return min(1.0, full3d / A10G_SMS)
    return recovery_phi_faithful(M)


def tax_of_recovery(recovery_phi: float) -> float:
    """determinism tax (floor as step fraction) = FLOOR_AT_PHI1 * forgone = FLOOR_AT_PHI1 *
    (1 - recovery_phi). Round-trips denken #332: recovery 0.075 -> 0.09103155 (FLOOR_AT_GEO)."""
    forgone = 1.0 - float(max(0.0, min(1.0, recovery_phi)))
    return FLOOR_AT_PHI1_327 * forgone


def strict_ceiling(M: int, recovery_phi: float) -> float:
    """ceiling(M) = LAMBDA1_CEIL * (1 - tax(M)) * (E[T]_M / E[T]_8). The (1-tax) factor is the
    determinism penalty; the E[T] ratio is the tokens-per-step scaling (linear-E[T] convention,
    stark #345: verify step method/M-independent -- OPTIMISTIC at large M, see caveats)."""
    tax = tax_of_recovery(recovery_phi)
    et_ratio = et_of_M(M) / et_of_M(M_DEPLOYED)
    return LAMBDA1_CEIL * (1.0 - tax) * et_ratio


def recovery_breakeven(budget_frac: float) -> float:
    """The recovery_phi that drops the floor to a budget: floor = FLOOR_AT_PHI1*(1-recovery) =
    budget -> recovery = 1 - budget/FLOOR_AT_PHI1. no-reg -> 0.255; >500 -> 0.591 (fern #349)."""
    return 1.0 - budget_frac / FLOOR_AT_PHI1_327


# --------------------------------------------------------------------------- #
# (D1) CTA-vs-M occupancy map.
# --------------------------------------------------------------------------- #
def deliverable1_cta_map() -> dict[str, Any]:
    m_sat = m_sat_continuous()
    rows = []
    for M in M_GRID:
        nnr = n_nonreduction_ctas(M)
        nf3 = n_full_3d_ctas(M)
        rows.append({
            "M": M,
            "total_num_q_blocks": total_num_q_blocks(M),
            "n_nonreduction_ctas": nnr,
            "n_full_3d_ctas": nf3,
            "saturated": bool(nf3 >= A10G_SMS),
            "sub_saturation": bool(M < m_sat),
            "partition_product_consistent": bool(nnr * NUM_PAR_SOFTMAX_SEGMENTS == nf3),
        })
    return {
        "model": "N_nonreduction(M) = (M//BLOCK_Q + NUM_SEQS) * NUM_KV_HEADS; "
                 "N_full_3d(M) = N_nonreduction(M) * NUM_PAR_SOFTMAX_SEGMENTS (16-way split-KV). "
                 "M_sat = M where N_full_3d crosses 80 SMs.",
        "block_q": BLOCK_Q, "num_seqs": NUM_SEQS, "num_kv_heads": NUM_KV_HEADS,
        "num_q_heads": NUM_Q_HEADS, "num_par_softmax_segments": NUM_PAR_SOFTMAX_SEGMENTS,
        "a10g_sms": A10G_SMS,
        "n_full_3d_at_deployed_m8": n_full_3d_ctas(M_DEPLOYED),
        "deployed_m8_reproduces_96": bool(n_full_3d_ctas(M_DEPLOYED) == N_FULL_3D_CTAS_332),
        "m_sat_continuous": m_sat,
        "sub_saturation_M": [M for M in M_GRID if M < m_sat],
        "saturated_M": [M for M in M_GRID if M >= m_sat],
        "tree_width_note": "the deployed lane is a narrow/linear chain (verify rows = M). A WIDER "
                           "tree (c-candidate Trie) adds verify query rows -> MORE q-blocks -> MORE "
                           "CTAs -> deeper saturation, never below the wall. The only direction "
                           "under the 80-SM wall is smaller M / narrower tree (the swept axis); a "
                           "c=64 wide-tree verify (up to 64 rows) would launch ~544 CTAs >> 80.",
        "wide_tree_c64_ctas_illustrative": (64 // BLOCK_Q + NUM_SEQS) * NUM_KV_HEADS
        * NUM_PAR_SOFTMAX_SEGMENTS,
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# (D2) recovery_phi(M), tax(M), E[T]_M, ceiling(M) -- the two models.
# --------------------------------------------------------------------------- #
def deliverable2_ceiling_over_M() -> dict[str, Any]:
    et8 = et_of_M(M_DEPLOYED)
    breakeven_noreg = recovery_breakeven(BUDGET_LAMBDA1_FRAC_213)   # 0.255
    breakeven_500 = recovery_breakeven(BUDGET_500_FRAC_192)         # 0.591
    m_sat = m_sat_continuous()
    rows = []
    for M in M_GRID:
        et = et_of_M(M)
        phi_f = recovery_phi_faithful(M)
        phi_h = recovery_phi_headroom(M)
        tax_f = tax_of_recovery(phi_f)
        tax_h = tax_of_recovery(phi_h)
        ceil_f = strict_ceiling(M, phi_f)
        ceil_h = strict_ceiling(M, phi_h)
        rows.append({
            "M": M, "sub_saturation": bool(M < m_sat),
            "e_t_M": et, "e_t_ratio_vs_8": et / et8,
            "recovery_phi_faithful": phi_f, "recovery_phi_headroom": phi_h,
            "tax_faithful": tax_f, "tax_headroom": tax_h,
            "ceiling_faithful": ceil_f, "ceiling_headroom": ceil_h,
            "faithful_phi_clears_noreg_breakeven": bool(phi_f >= breakeven_noreg),
            "faithful_phi_clears_500_breakeven": bool(phi_f >= breakeven_500),
            "headroom_phi_clears_noreg_breakeven": bool(phi_h >= breakeven_noreg),
            "headroom_phi_clears_500_breakeven": bool(phi_h >= breakeven_500),
            "ceiling_faithful_beats_473": bool(ceil_f > STRICT_COMPLIANT_CEILING_332),
            "ceiling_headroom_beats_473": bool(ceil_h > STRICT_COMPLIANT_CEILING_332),
        })
    # argmax over the full grid (faithful model is the headline).
    best_f = max(rows, key=lambda r: r["ceiling_faithful"])
    best_h = max(rows, key=lambda r: r["ceiling_headroom"])
    sub_rows = [r for r in rows if r["sub_saturation"]]
    best_sub_f = max(sub_rows, key=lambda r: r["ceiling_faithful"]) if sub_rows else None
    best_sub_h = max(sub_rows, key=lambda r: r["ceiling_headroom"]) if sub_rows else None
    return {
        "ceiling_law": "ceiling(M) = LAMBDA1_CEIL(520.953) * (1 - tax(M)) * (E[T]_M / E[T]_8 = "
                       "3.8512). tax(M) = FLOOR_AT_PHI1(0.09841) * (1 - recovery_phi(M)).",
        "e_t_8": et8,
        "recovery_breakeven_noreg": breakeven_noreg,
        "recovery_breakeven_500": breakeven_500,
        "rows": rows,
        # ---- the TEST metric (faithful model, full grid) ----
        "argmax_M_faithful": best_f["M"],
        "max_strict_ceiling_over_M": best_f["ceiling_faithful"],          # TEST metric (float)
        "argmax_M_headroom": best_h["M"],
        "max_strict_ceiling_over_M_headroom": best_h["ceiling_headroom"],
        # ---- the hypothesis-relevant sub-saturation-only maxima ----
        "argmax_sub_saturation_M_faithful": best_sub_f["M"] if best_sub_f else None,
        "max_sub_saturation_ceiling_faithful": best_sub_f["ceiling_faithful"] if best_sub_f else None,
        "argmax_sub_saturation_M_headroom": best_sub_h["M"] if best_sub_h else None,
        "max_sub_saturation_ceiling_headroom": best_sub_h["ceiling_headroom"] if best_sub_h else None,
        # for the report: does the faithful argmax come from sub-saturation? (No -> large M.)
        "faithful_argmax_is_sub_saturation": bool(best_f["M"] < m_sat),
        "faithful_argmax_is_largest_M": bool(best_f["M"] == max(M_GRID)),
    }


# --------------------------------------------------------------------------- #
# (D3) Verdict: does any sub-saturation M escape 473.5?
# --------------------------------------------------------------------------- #
def deliverable3_verdict(d1: dict, d2: dict) -> dict[str, Any]:
    rows = d2["rows"]
    sub_rows = [r for r in rows if r["sub_saturation"]]
    # the deliverable bool: does ANY sub-saturation M beat 473.5 under EITHER model?
    sub_escapes_faithful = bool(any(r["ceiling_faithful_beats_473"] for r in sub_rows))
    sub_escapes_headroom = bool(any(r["ceiling_headroom_beats_473"] for r in sub_rows))
    sub_saturation_escapes_473 = bool(sub_escapes_faithful or sub_escapes_headroom)
    # does any sub-saturation M clear the recovery break-even (the tax-escape question)?
    sub_clears_noreg_faithful = bool(any(r["faithful_phi_clears_noreg_breakeven"] for r in sub_rows))
    sub_clears_500_faithful = bool(any(r["faithful_phi_clears_500_breakeven"] for r in sub_rows))
    sub_clears_500_headroom = bool(any(r["headroom_phi_clears_500_breakeven"] for r in sub_rows))
    # under the faithful model, recovery_phi GROWS with M -> sub-saturation lowers it.
    faithful_phi_grows_with_M = bool(all(
        rows[i]["recovery_phi_faithful"] <= rows[i + 1]["recovery_phi_faithful"] + TOL_EXACT
        for i in range(len(rows) - 1)))
    # any M at all reaches the no-reg recovery break-even (faithful)?
    any_M_clears_noreg_faithful = bool(any(r["faithful_phi_clears_noreg_breakeven"] for r in rows))
    return {
        "sub_saturation_escapes_473": sub_saturation_escapes_473,            # TEST bool (expect False)
        "sub_escapes_faithful": sub_escapes_faithful,
        "sub_escapes_headroom": sub_escapes_headroom,
        "sub_clears_noreg_breakeven_faithful": sub_clears_noreg_faithful,
        "sub_clears_500_breakeven_faithful": sub_clears_500_faithful,
        "sub_clears_500_breakeven_headroom": sub_clears_500_headroom,
        "faithful_recovery_phi_grows_with_M": faithful_phi_grows_with_M,
        "any_M_clears_noreg_recovery_breakeven_faithful": any_M_clears_noreg_faithful,
        "occupancy_wall_escapable_downward": sub_saturation_escapes_473,
        "verdict": (
            "REFUTED: NO sub-saturation M (M in {} < M_sat={:.1f}) escapes the 473.5 strict tax. "
            "On denken #332's FAITHFUL geometry recovery_phi(M) = N_nonreduction(M)/80 GROWS with M "
            "(M=2->{:.3f}, M=4->{:.3f}, M=8->0.075), so a SMALLER M shrinks the non-reduction grid "
            "and LOWERS recovery -- a deterministic schedule has LESS, not more, parallelism to keep. "
            "Sub-saturation is thus DOUBLY penalised: lower recovery_phi (higher tax) AND lower E[T] "
            "(fewer tokens/step). The faithful sub-saturation ceilings collapse to {:.1f} (M=4) / "
            "{:.1f} (M=2), far below 473.5. Even a STEELMAN headroom model -- granting idle SMs a "
            "free deterministic reduction (recovery_phi up to {:.2f} at M=4) -- still fails: the E[T] "
            "loss (E[T]_4/E[T]_8 = {:.3f}) outpaces the recovery gain, capping the M=4 steelman "
            "ceiling at {:.1f} < 473.5. The verify slack is BANDWIDTH-bound (AI 7.88 << ridge 208; "
            "the 96-CTA saturated path still sits at 34.9% BW, stark #345), so idle SMs cannot "
            "convert to recovery anyway. The strict ceiling can exceed 473.5 ONLY at LARGER M "
            "(argmax M*={}, ceiling {:.1f}) -- but that is E[T]-driven SUPER-saturation, the OPPOSITE "
            "of the hypothesis, and rests on the linear-E[T] convention (M-independent verify step) "
            "being optimistic at large M. The occupancy wall is NOT escapable downward.".format(
                d1["sub_saturation_M"], d1["m_sat_continuous"],
                rows[0]["recovery_phi_faithful"], rows[1]["recovery_phi_faithful"],
                next(r["ceiling_faithful"] for r in rows if r["M"] == 4),
                next(r["ceiling_faithful"] for r in rows if r["M"] == 2),
                next(r["recovery_phi_headroom"] for r in rows if r["M"] == 4),
                next(r["e_t_ratio_vs_8"] for r in rows if r["M"] == 4),
                next(r["ceiling_headroom"] for r in rows if r["M"] == 4),
                d2["argmax_M_faithful"], d2["max_strict_ceiling_over_M"])),
    }


# --------------------------------------------------------------------------- #
# (D4) Caveats + scope.
# --------------------------------------------------------------------------- #
def deliverable4_caveats() -> dict[str, Any]:
    return {
        "linear_et_convention_optimistic_at_large_M": (
            "ceiling(M) uses the linear-E[T] convention (stark #345): TPS scales with E[T]_M at a "
            "M-INDEPENDENT verify step time. This is defensible for the KV-read-dominated, BW-bound "
            "verify near the deployed M, but OPTIMISTIC at large M -- a larger M adds verify query "
            "rows (activation bytes ~ M) and compute, so the real step time grows and the true "
            "ceiling at M=16/32 is BELOW the reported value. The large-M ceilings are therefore an "
            "UPPER bound; they do not change the verdict (sub-saturation still loses on both axes)."),
        "two_phi_models_bracket_the_truth": (
            "recovery_phi(M) is carried as a faithful (non-reduction occupancy, denken #332) and a "
            "steelman (idle-SM headroom) model. The faithful model is the headline (it round-trips "
            "the M=8 anchor exactly); the steelman is the most generous reading of the hypothesis. "
            "The verdict (no sub-saturation escape) holds under BOTH -- it is not a modelling artifact."),
        "bw_bound_not_occupancy_bound": (
            "the hypothesis assumes the verify is OCCUPANCY-bound (idle SMs -> free determinism). "
            "denken #332 + stark #345 measured it BANDWIDTH-bound (AI 7.88 << ridge 208; the 96-CTA "
            "saturated path STILL sits at 34.9% BW), so sub-saturation headroom does NOT convert to "
            "recovery -- the steelman over-credits the hypothesis and it STILL fails."),
        "tree_width_only_adds_ctas": (
            "a wider tree (c-candidate Trie) only ADDS verify query rows -> MORE CTAs -> deeper "
            "saturation; it cannot move a verify BELOW the 80-SM wall. The only sub-saturation lever "
            "is smaller M / narrower tree, which is the swept axis -- and it is refuted."),
        "scope": (
            "this PRICES whether the occupancy wall is escapable downward; it is NOT a kernel-"
            "buildability proof and builds NOTHING. The deterministic verify kernel stays UNBUILT + "
            "human-approval-gated regardless. 0 TPS; BASELINE 481.53 unchanged."),
        "non_collision": (
            "ORTHOGONAL to denken (body-bits / saturated phi-floor), kanna (non-batched M=1 verify), "
            "wirbel (composition), lawine (frontier baseline), fern (recovery->ceiling integrator). "
            "This owns the verify occupancy / M-shape axis."),
    }


# --------------------------------------------------------------------------- #
# (D5) Greedy-safety + analytic scope.
# --------------------------------------------------------------------------- #
def deliverable5_greedy_safety() -> dict[str, Any]:
    return {
        "card_is_cpu_analytic": True, "no_gpu": True, "no_served_change": True,
        "no_model_forward": True, "no_training": True, "no_hf_job": True,
        "zero_official_tps": True, "greedy_identity_preserved_by_construction": True,
        "note": (
            "this card builds and runs NO kernel -- it is a numeric occupancy/ceiling map over banked "
            "anchors. STRICT byte-exact greedy identity (#319) is the contract this card prices "
            "AGAINST; nothing here touches a served file, kernel, or decode path. BASELINE 481.53."),
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def _selftests(d1: dict, d2: dict, d3: dict, nan_clean: bool) -> dict[str, Any]:
    rows = d2["rows"]
    r8 = next(r for r in rows if r["M"] == 8)
    et8_recomputed = et_of_M(M_DEPLOYED)
    ceiling8_faithful = r8["ceiling_faithful"]
    conditions = {
        # (a) M=8 round-trips denken #332's saturated recovery 0.075 / ceiling 473.5296 <= 1e-6.
        "a_m8_recovery_phi_is_0p075": bool(abs(r8["recovery_phi_faithful"] - RECOVERY_PHI_332) <= TOL_ROUNDTRIP),
        "a_m8_recovery_phi_is_0p075_display": bool(abs(r8["recovery_phi_faithful"] - 0.075) <= TOL_ROUNDTRIP),
        "a_m8_tax_is_floor_at_geo_332": bool(abs(r8["tax_faithful"] - FLOOR_AT_GEO_332) <= TOL_ROUNDTRIP),
        "a_m8_ceiling_roundtrips_473": bool(
            abs(ceiling8_faithful - STRICT_COMPLIANT_CEILING_332) <= TOL_ROUNDTRIP),
        "a_m8_ceiling_is_473p53_display": bool(abs(ceiling8_faithful - 473.53) <= 5e-2),
        "a_m8_et_ratio_is_unity": bool(abs(r8["e_t_ratio_vs_8"] - 1.0) <= TOL_EXACT),
        # (b) E[T]_8 round-trips the deployed #289 3.8512 (the a_k profile is 4-dp rounded).
        "b_et8_reproduces_289": bool(abs(et8_recomputed - E_T_EAGLE3_289) <= TOL_ROUNDTRIP),
        "b_et8_is_3p8512_display": bool(abs(et8_recomputed - 3.8512) <= 2e-3),
        # (c) phi(M) + ceiling(M) NaN-clean and finite for all M.
        "c_all_phi_finite": bool(all(
            _finite(r["recovery_phi_faithful"]) and _finite(r["recovery_phi_headroom"]) for r in rows)),
        "c_all_ceiling_finite": bool(all(
            _finite(r["ceiling_faithful"]) and _finite(r["ceiling_headroom"]) for r in rows)),
        "c_all_phi_in_unit_interval": bool(all(
            0.0 <= r["recovery_phi_faithful"] <= 1.0 and 0.0 <= r["recovery_phi_headroom"] <= 1.0
            for r in rows)),
        "c_nan_clean": bool(nan_clean),
        # (d) M_sat identified + CTA map dimensionally consistent.
        "d_m_sat_is_6": bool(abs(d1["m_sat_continuous"] - 6.0) <= TOL_ROUNDTRIP),
        "d_deployed_m8_reproduces_96": bool(d1["deployed_m8_reproduces_96"]),
        "d_partition_product_consistent": bool(all(
            row["partition_product_consistent"] for row in d1["rows"])),
        "d_m2_m4_are_sub_saturation": bool(d1["sub_saturation_M"] == [2, 4]),
        "d_m8_m16_m32_saturated": bool(d1["saturated_M"] == [8, 16, 32]),
        "d_nonreduction_x_segments_eq_full3d_m8": bool(
            n_nonreduction_ctas(8) * NUM_PAR_SOFTMAX_SEGMENTS == n_full_3d_ctas(8)),
        # (e) verdict bool set + the two TEST metrics well-formed.
        "e_verdict_bool_is_set": bool(isinstance(d3["sub_saturation_escapes_473"], bool)),
        "e_verdict_is_false": bool(d3["sub_saturation_escapes_473"] is False),
        "e_max_ceiling_is_float": bool(_finite(d2["max_strict_ceiling_over_M"])),
        # (f) [extra] the recovery break-evens round-trip fern #349 (0.255 / 0.591).
        "f_breakeven_noreg_is_0p255": bool(abs(d2["recovery_breakeven_noreg"] - 0.255) <= 1e-3),
        "f_breakeven_500_is_0p591": bool(abs(d2["recovery_breakeven_500"] - 0.591) <= 1e-3),
        "f_fern_map_consistent": _fern_map_consistent(),
        # (g) [extra] the decisive structure: faithful recovery_phi GROWS with M (sub-sat lowers it).
        "g_faithful_phi_grows_with_M": bool(d3["faithful_recovery_phi_grows_with_M"]),
        "g_m4_phi_below_m8": bool(
            next(r["recovery_phi_faithful"] for r in rows if r["M"] == 4) < RECOVERY_PHI_332),
        "g_m2_phi_below_m4": bool(
            next(r["recovery_phi_faithful"] for r in rows if r["M"] == 2)
            < next(r["recovery_phi_faithful"] for r in rows if r["M"] == 4)),
        # (h) [extra] no sub-saturation M beats 473.5 under EITHER model (robust verdict).
        "h_no_sub_escape_faithful": bool(not d3["sub_escapes_faithful"]),
        "h_no_sub_escape_headroom": bool(not d3["sub_escapes_headroom"]),
        "h_m4_faithful_below_473": bool(
            next(r["ceiling_faithful"] for r in rows if r["M"] == 4) < STRICT_COMPLIANT_CEILING_332),
        "h_m4_headroom_below_473": bool(
            next(r["ceiling_headroom"] for r in rows if r["M"] == 4) < STRICT_COMPLIANT_CEILING_332),
        # (i) [extra] the faithful argmax is the LARGEST M (E[T]-driven), not sub-saturation.
        "i_faithful_argmax_not_sub_saturation": bool(not d2["faithful_argmax_is_sub_saturation"]),
        "i_faithful_argmax_is_largest_M": bool(d2["faithful_argmax_is_largest_M"]),
        # (j) [extra] constants imported exact.
        "j_constants_exact": bool(
            abs(LAMBDA1_CEIL - 520.9527323111674) < TOL_EXACT
            and abs(FLOOR_AT_PHI1_327 - 0.09841249119201488) < TOL_EXACT
            and abs(STRICT_COMPLIANT_CEILING_332 - 473.5295953446407) < TOL_ROUNDTRIP
            and A10G_SMS == 80 and N_FULL_3D_CTAS_332 == 96 and BLOCK_Q == 4),
        # (k) [extra] bandwidth-bound (the structural reason headroom cannot convert to recovery).
        "k_verify_bandwidth_bound": bool(SDPA_AI_FLOP_PER_BYTE < RIDGE_AI),
        "k_deployed_saturated": bool(N_FULL_3D_CTAS_332 > A10G_SMS),
    }
    passes = bool(all(conditions.values()))
    return {
        "conditions": conditions,
        "strict_sub_saturation_self_test_passes": passes,
        "n_checks": len(conditions),
        "detail": {
            "max_strict_ceiling_over_M": d2["max_strict_ceiling_over_M"],
            "argmax_M_faithful": d2["argmax_M_faithful"],
            "sub_saturation_escapes_473": d3["sub_saturation_escapes_473"],
            "m_sat": d1["m_sat_continuous"],
        },
    }


def _fern_map_consistent() -> bool:
    """fern #349: recovery -> ceiling at M=8 (E[T] ratio 1). ceiling = LAMBDA1_CEIL*(1-tax(rec))."""
    for rec, ceil_expected in FERN_MAP_349:
        ceil = LAMBDA1_CEIL * (1.0 - tax_of_recovery(rec))
        if abs(ceil - ceil_expected) > 5e-2:
            return False
    return True


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    d1 = deliverable1_cta_map()
    d2 = deliverable2_ceiling_over_M()
    d3 = deliverable3_verdict(d1, d2)
    d4 = deliverable4_caveats()
    d5 = deliverable5_greedy_safety()

    handoff = (
        "the strict #319 sub-saturation question: does a sub-80-SM (smaller-M / narrower-tree) verify "
        "escape denken #332's 473.5 determinism tax? NO (sub_saturation_escapes_473 = {}). On the "
        "FAITHFUL launch geometry recovery_phi(M) = N_nonreduction(M)/80 GROWS with M (M=2->0.025, "
        "M=4->0.050, M=8->0.075), so a smaller M shrinks the deterministic-compatible grid and LOWERS "
        "recovery -- sub-saturation is DOUBLY penalised (lower recovery_phi AND lower E[T]). The "
        "sub-saturation ceilings collapse to {:.1f} (M=4) / {:.1f} (M=2) << 473.5; even a STEELMAN "
        "idle-SM-headroom model (recovery up to 0.80 at M=4) fails because the E[T] loss outpaces the "
        "recovery gain (M=4 steelman {:.1f} < 473.5). The slack is BW-bound (AI 7.88 << ridge 208; the "
        "96-CTA saturated path still sits at 34.9% BW), so idle SMs cannot convert to recovery -- the "
        "occupancy wall is NOT escapable downward. The strict ceiling rises above 473.5 ONLY at LARGER "
        "M (argmax M*={}, max ceiling {:.1f}), but that is E[T]-driven SUPER-saturation (the opposite "
        "of the hypothesis) and rests on the optimistic M-independent-verify convention. CONCLUSION: "
        "narrowing below the wall does not help; only a TRUE deterministic-reduction kernel (denken's "
        "UNBUILT, human-gated artifact) or #124 moves the strict >500 lane. ANALYTIC; 0 TPS; NOT a "
        "launch / build / served-file change.".format(
            d3["sub_saturation_escapes_473"],
            next(r["ceiling_faithful"] for r in d2["rows"] if r["M"] == 4),
            next(r["ceiling_faithful"] for r in d2["rows"] if r["M"] == 2),
            next(r["ceiling_headroom"] for r in d2["rows"] if r["M"] == 4),
            d2["argmax_M_faithful"], d2["max_strict_ceiling_over_M"]))

    headline = {
        "strict_sub_saturation_self_test_passes": None,   # set after nan audit
        "max_strict_ceiling_over_M": d2["max_strict_ceiling_over_M"],                # TEST
        "sub_saturation_escapes_473": d3["sub_saturation_escapes_473"],              # TEST
        "argmax_M_faithful": d2["argmax_M_faithful"],
        "m_sat_continuous": d1["m_sat_continuous"],
        "max_sub_saturation_ceiling_faithful": d2["max_sub_saturation_ceiling_faithful"],
        "max_sub_saturation_ceiling_headroom": d2["max_sub_saturation_ceiling_headroom"],
        "strict_compliant_ceiling_332": STRICT_COMPLIANT_CEILING_332,
        "faithful_argmax_is_largest_M": d2["faithful_argmax_is_largest_M"],
        "occupancy_wall_escapable_downward": d3["occupancy_wall_escapable_downward"],
    }
    return {
        "headline": headline,
        "deliverable1_cta_map": d1,
        "deliverable2_ceiling_over_M": d2,
        "deliverable3_verdict": d3,
        "deliverable4_caveats": d4,
        "deliverable5_greedy_safety": d5,
        "handoff": handoff,
        "imports": {
            "provenance": (
                "denken #332 y5cl0ena (recovery 0.075 / forgone 0.925, floor 0.09841, geo-floor "
                "0.09103, strict ceiling 473.5296, AI 7.88, 96 CTAs > 80 SMs, ceiling 520.953, >500 "
                "budget 4.022%) x kanna #289 fi34s269 (a_k profile, E[T]=3.8512) x fern #349 u8vmtji0 "
                "(recovery->ceiling map 0.075->473.5 / 0.255->482.76 / 0.591->500) x stark #345 "
                "(batched-verify floor method-independent). All run-ids in "
                "wandb-applied-ai-team/gemma-challenge-senpai."),
            "caveats": [
                "CPU-analytic occupancy/ceiling map over banked anchors; builds and runs NO kernel, "
                "re-derives nothing measured. NOT a running verify / kernel / launch.",
                "ceiling(M) uses the linear-E[T] convention (M-independent verify step) -- optimistic "
                "at large M; the large-M ceilings are an UPPER bound and do not change the verdict.",
                "recovery_phi(M) bracketed by a faithful (non-reduction occupancy) and a steelman "
                "(idle-SM headroom) model; the no-escape verdict holds under BOTH.",
                "NOT a launch / build / served-file change / HF Job / submission.",
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


def _print_report(syn: dict, st: dict) -> None:
    d1 = syn["deliverable1_cta_map"]
    d2 = syn["deliverable2_ceiling_over_M"]
    d3 = syn["deliverable3_verdict"]
    print("\n" + "=" * 100, flush=True)
    print("STRICT SUB-SATURATION VERIFY (PR #358, stark) — does a sub-80-SM M escape the 473.5 tax?",
          flush=True)
    print("=" * 100, flush=True)
    print(f"  (D1) CTA-vs-M MAP  (M_sat={d1['m_sat_continuous']:.1f}; deployed M=8 -> "
          f"{d1['n_full_3d_at_deployed_m8']} CTAs > {d1['a10g_sms']} SMs; reproduces 96 = "
          f"{d1['deployed_m8_reproduces_96']})", flush=True)
    for r in d1["rows"]:
        tag = "sub-sat" if r["sub_saturation"] else "SATURATED"
        print(f"        - M={r['M']:>2d}  q_blocks={r['total_num_q_blocks']:>2d}  "
              f"N_nonred={r['n_nonreduction_ctas']:>2d}  N_full3d={r['n_full_3d_ctas']:>3d}  [{tag}]",
              flush=True)
    print("-" * 100, flush=True)
    print(f"  (D2) recovery_phi(M), tax(M), ceiling(M)  (E[T]_8={d2['e_t_8']:.4f}; break-evens: "
          f"no-reg>={d2['recovery_breakeven_noreg']:.3f}, 500>={d2['recovery_breakeven_500']:.3f})",
          flush=True)
    for r in d2["rows"]:
        print(f"        - M={r['M']:>2d}  E[T]={r['e_t_M']:.3f} (x{r['e_t_ratio_vs_8']:.3f})  "
              f"phi_faith={r['recovery_phi_faithful']:.3f} -> ceil_faith={r['ceiling_faithful']:6.1f}  "
              f"|  phi_head={r['recovery_phi_headroom']:.3f} -> ceil_head={r['ceiling_headroom']:6.1f}",
              flush=True)
    print(f"      max_strict_ceiling_over_M = {d2['max_strict_ceiling_over_M']:.2f} "
          f"(M*={d2['argmax_M_faithful']}, {'SUB-sat' if d2['faithful_argmax_is_sub_saturation'] else 'super-sat'})",
          flush=True)
    print(f"      max SUB-saturation ceiling = {d2['max_sub_saturation_ceiling_faithful']:.2f} "
          f"(faithful) / {d2['max_sub_saturation_ceiling_headroom']:.2f} (steelman)", flush=True)
    print("-" * 100, flush=True)
    print("  (D3) VERDICT", flush=True)
    print(f"      sub_saturation_escapes_473 = {d3['sub_saturation_escapes_473']}  (expect False)",
          flush=True)
    print(f"      faithful recovery_phi grows with M = {d3['faithful_recovery_phi_grows_with_M']}   "
          f"any M clears no-reg recovery break-even = {d3['any_M_clears_noreg_recovery_breakeven_faithful']}",
          flush=True)
    print("-" * 100, flush=True)
    print(f"  HAND-OFF: {syn['handoff']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  PRIMARY strict_sub_saturation_self_test_passes = "
          f"{st['strict_sub_saturation_self_test_passes']} ({st['n_checks']} checks)", flush=True)
    for k, v in st["conditions"].items():
        if not v:
            print(f"        - FAIL {k}: {v}", flush=True)
    print("=" * 100 + "\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> str | None:
    if not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        _w = sys.modules.get("wandb")
        if _w is not None and not hasattr(_w, "init"):
            del sys.modules["wandb"]
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[strict-sub-saturation-verify] wandb logging unavailable: {exc}", flush=True)
        return None

    syn = payload["synthesis"]
    d1 = syn["deliverable1_cta_map"]
    d2 = syn["deliverable2_ceiling_over_M"]
    d3 = syn["deliverable3_verdict"]
    st = payload["self_test"]
    run = init_wandb_run(
        job_type="validity-gate",
        agent="stark",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["strict-sub-saturation-verify", "issue-319", "strict-identity", "eagle3",
              "verify-occupancy", "cta-map", "determinism-tax", "supply-tax", "m-shape",
              "validity-gate", "bank-the-analysis", "pr-358"],
        config={
            "geometric_phi_332": GEOMETRIC_PHI_332, "recovery_phi_332": RECOVERY_PHI_332,
            "floor_at_phi1_327": FLOOR_AT_PHI1_327, "floor_at_geo_332": FLOOR_AT_GEO_332,
            "strict_compliant_ceiling_332": STRICT_COMPLIANT_CEILING_332,
            "lambda1_ceil": LAMBDA1_CEIL, "sdpa_bw_util": SDPA_BW_UTIL,
            "sdpa_ai_flop_per_byte": SDPA_AI_FLOP_PER_BYTE, "ridge_ai": RIDGE_AI,
            "n_full_3d_ctas_332": N_FULL_3D_CTAS_332, "a10g_sms": A10G_SMS,
            "block_q": BLOCK_Q, "num_kv_heads": NUM_KV_HEADS,
            "num_par_softmax_segments": NUM_PAR_SOFTMAX_SEGMENTS,
            "e_t_eagle3_head_289": E_T_EAGLE3_289,
            "budget_lambda1_frac_213": BUDGET_LAMBDA1_FRAC_213,
            "budget_500_frac_192": BUDGET_500_FRAC_192,
            "m_grid": M_GRID, "m_sat_continuous": d1["m_sat_continuous"],
            "target": TARGET, "baseline_tps": BASELINE_TPS,
            "wandb_group": args.wandb_group,
            "source_runs": "denken#332(y5cl0ena), kanna#289(fi34s269), fern#349(u8vmtji0), stark#345",
        },
    )
    if run is None:
        print("[strict-sub-saturation-verify] wandb: no run (no WANDB_API_KEY/mode) — skipping",
              flush=True)
        return None

    summary: dict[str, Any] = {
        "strict_sub_saturation_self_test_passes": int(bool(
            st["strict_sub_saturation_self_test_passes"])),                              # PRIMARY
        "max_strict_ceiling_over_M": d2["max_strict_ceiling_over_M"],                    # TEST
        "sub_saturation_escapes_473": int(bool(d3["sub_saturation_escapes_473"])),       # TEST
        "argmax_M_faithful": d2["argmax_M_faithful"],
        "argmax_M_headroom": d2["argmax_M_headroom"],
        "max_strict_ceiling_over_M_headroom": d2["max_strict_ceiling_over_M_headroom"],
        "max_sub_saturation_ceiling_faithful": d2["max_sub_saturation_ceiling_faithful"],
        "max_sub_saturation_ceiling_headroom": d2["max_sub_saturation_ceiling_headroom"],
        "m_sat_continuous": d1["m_sat_continuous"],
        "n_full_3d_at_deployed_m8": d1["n_full_3d_at_deployed_m8"],
        "faithful_recovery_phi_grows_with_M": int(bool(d3["faithful_recovery_phi_grows_with_M"])),
        "any_M_clears_noreg_recovery_breakeven_faithful": int(bool(
            d3["any_M_clears_noreg_recovery_breakeven_faithful"])),
        "faithful_argmax_is_sub_saturation": int(bool(d2["faithful_argmax_is_sub_saturation"])),
        "faithful_argmax_is_largest_M": int(bool(d2["faithful_argmax_is_largest_M"])),
        "strict_compliant_ceiling_332": STRICT_COMPLIANT_CEILING_332,
        "recovery_breakeven_noreg": d2["recovery_breakeven_noreg"],
        "recovery_breakeven_500": d2["recovery_breakeven_500"],
        "e_t_8": d2["e_t_8"],
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    # per-M curves, flattened for cross-run analysis.
    for r in d2["rows"]:
        M = r["M"]
        summary[f"et_M{M}"] = r["e_t_M"]
        summary[f"et_ratio_M{M}"] = r["e_t_ratio_vs_8"]
        summary[f"recovery_phi_faithful_M{M}"] = r["recovery_phi_faithful"]
        summary[f"recovery_phi_headroom_M{M}"] = r["recovery_phi_headroom"]
        summary[f"tax_faithful_M{M}"] = r["tax_faithful"]
        summary[f"ceiling_faithful_M{M}"] = r["ceiling_faithful"]
        summary[f"ceiling_headroom_M{M}"] = r["ceiling_headroom"]
    for row in d1["rows"]:
        M = row["M"]
        summary[f"n_nonreduction_ctas_M{M}"] = row["n_nonreduction_ctas"]
        summary[f"n_full_3d_ctas_M{M}"] = row["n_full_3d_ctas"]
        summary[f"saturated_M{M}"] = int(bool(row["saturated"]))
    summary = {k: v for k, v in summary.items()
               if v is not None and not (isinstance(v, float) and not math.isfinite(v))}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="strict_sub_saturation_verify_result", artifact_type="validity",
                      data=payload)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    print(f"[strict-sub-saturation-verify] wandb logged {len(summary)} keys (run {rid})", flush=True)
    return rid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="strict-sub-saturation-verify")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)
    if args.no_wandb:
        args.wandb_name = None

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 358, "agent": "stark",
        "kind": "strict-sub-saturation-verify", "analysis_only": True, "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths

    st = _selftests(syn["deliverable1_cta_map"], syn["deliverable2_ceiling_over_M"],
                    syn["deliverable3_verdict"], payload["nan_clean"])
    payload["self_test"] = st
    syn["headline"]["strict_sub_saturation_self_test_passes"] = st[
        "strict_sub_saturation_self_test_passes"]
    if nan_paths:
        print(f"[strict-sub-saturation-verify] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn, st)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "strict_sub_saturation_verify_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    print(f"[strict-sub-saturation-verify] wrote {out_path}", flush=True)

    rid = _maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = rid

    if args.self_test:
        ok = st["strict_sub_saturation_self_test_passes"] and payload["nan_clean"]
        if not ok:
            failed = [k for k, v in st["conditions"].items() if not v]
            print(f"[strict-sub-saturation-verify] SELF-TEST FAILED: {failed}", flush=True)
        print(f"[strict-sub-saturation-verify] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
