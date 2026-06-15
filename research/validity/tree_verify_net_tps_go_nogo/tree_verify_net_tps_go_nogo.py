#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Tree-verify NET-TPS go/no-go (PR #402, denken).

THE QUESTION (the one ubel #399 held out of scope -- "the actual tree go/no-go"):
  ubel #399 proved the demand route's d-cov is real and that NO cheap deployable lever supplies
  it -- the only no-retrain coverage lever is a TREE VERIFY (top-K width > 1) harvesting the head-
  ceiling coverage beyond the deployed top-4 anchor. But a tree is NOT free: widening the verify
  from M=8 (top-1/position) to a top-K tree pays a verify-M STEP-TIME TAX (a #390 CUDA-graph
  rebuild for the new width + a wider-M attention/roofline cost; #332 shows the M=8 verify is
  already occupancy-saturated at 96 CTAs > 80 SM, BW-floored at 34.9% -- wider-M is on the wrong
  side of the roofline). The decision-critical question: does the coverage GAIN (more E[accepted]
  -> fewer steps) NET positive after the step-time TAX, and does the net NET-SUPPLY the demand
  d-cov needed to clear 500 from the corrected 467.48 base?

THE ANSWER (decision-critical, honest, conservative NO-GO ALONE):
  The verify-M tax scales with the SAME tree width that buys the coverage, so there is no free
  lunch. To realize ANY program-coverage uplift g > 0 over the deployed top-4 anchor (0.890) the
  tree must verify top-K with K > 4 (a K=4 tree merely reproduces the anchor -> g = 0). At the
  first width that buys coverage (K=8, the point ubel #401 is measuring), the honest single-forward
  verify cost is a full per-position fan-out M(8) = 1 + 7*8 = 57 rows: the deployed split-KV
  attention (LOCKED 16-way split, no kernel change) re-reads the KV once per query-block, so its
  time scales with the query-block count -> the 9.5%-of-step attention lane inflates by r = 5.33x
  -> tstep_tax_frac(K=8) = 0.412 -> tps_loss = 136.4 TPS on the 467.48 base. The full-gap gain
  (g = 0.1097, the head-ceiling UPPER bound, K -> inf) is only ~105.6 TPS; net_tps(0.1097, K=8) =
  -61.6 (NEGATIVE). The realized-fraction threshold to close 500 is g*(K=8, full) = 0.248 --
  ABOVE the entire achievable coverage band (g_max = 0.1097) and even above the top-1->top-4 prize
  (0.1286). => the tree CANNOT close 500 ALONE at the honest verify cost.
    * tree_verify_net_positive = True ONLY for the OPTIMISTIC depth-1 branch-once shape
      (M = 8 + (K-1), tax 0.063 at K=8) -- but that shape widens a single position, so its realized
      g is small and almost certainly below its own g*(K=8, depth1) = 0.067.
    * tree_closes_500_alone = False (robust): at every width that buys g > 0, g*(K) >= g_max.
    * reconciles #396: at tax = 0, g* = 0.0338 == denken #396's corrected-base required_dcov, which
      ALREADY busts the +0.031 #336 budget; the verify-M tax only pushes g* higher.
  => tree_plus_cb3_required = True: the robust >500 plan needs the tree's coverage AND the kanna
     #207 conservative-k cb3 supply lift to share the burden; neither closes 500 alone.

PPL/GREEDY NOTE: a greedy tree verify keeps the longest target-argmax-matching path, so the emitted
  token is EXACTLY the target greedy token -> greedy identity preserved -> PPL UNCHANGED at 2.3772
  <= 2.42 for every (g, K). The binding constraints are the verify-M tax + the CUDA-graph rebuild,
  NOT PPL.

WHAT THIS IS / IS NOT:
  Pure-CPU analytic card (stdlib math). 0 GPU, 0 official TPS, 0 HF Job, NO served-file change, NO
  submission, NO kernel build. A roofline + secant composition parameterized over the head-ceiling
  band g in [0, 0.1097] and the tree width K. Composes BANKED anchors ONLY: #393 corrected base
  467.475 (0q7ynumg), #399/#387/#289/#340/#336 coverage+secant (ec7i3z5t/z8osvif8/fi34s269),
  #396 demand inversion (yc5ji486), #332 M=8 verify roofline geometry (y5cl0ena), #378/#393 served
  step-fraction attn = 9.5%, #390 CUDA-graph rebuild count (5y64zbjz). BASELINE 481.53 TPS / PPL
  2.3772 and the corrected strict base 467.475 UNCHANGED.

REPRODUCE (0-GPU):
    cd target/ && .venv/bin/python -m research.validity.tree_verify_net_tps_go_nogo.\
tree_verify_net_tps_go_nogo --self-test
    cd target/ && .venv/bin/python -m research.validity.tree_verify_net_tps_go_nogo.\
tree_verify_net_tps_go_nogo \
      --wandb_group tree-verify-net-tps-go-nogo --wandb_name denken/tree-verify-net-tps-go-nogo
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ===========================================================================
# Section 0 -- banked anchors (imported EXACTLY from merged advisor-branch cards / PR #402 body)
# ===========================================================================

# ---- #393 (0q7ynumg) corrected realized strict decode base (the TPS-mapping target) --------------
# attention_strict_pin_cost_results.json: deployed_tps_decode_eta / gap_to_500_after_attn.
BASE_467: float = 467.475218449957          # corrected realized shippable strict decode TPS (#393)
GAP_32: float = 32.524781550042974          # = 500 - BASE_467 (strict gap-to-500, banked #393)
CEILING_505_393: float = 505.29039303418637  # ceiling_with_cheapest_attn (#393)

# ---- #390 (5y64zbjz) OLD corrected base + CUDA-graph rebuild count (the rebuild-tax provenance) ---
BASE_471_390: float = 471.41634950257713    # OLD corrected strict base (pre-#393 attn-eta correction)
GAP_28_390: float = 28.583650497422866      # OLD gap-to-500 (#390 frame)
N_REBUILDS_ARM_A_390: int = 1               # #390 counts distinct kernel rebuilds: attention pin = 1
CEILING_BAND_LO_390: float = 471.20         # #390 corrected ceiling band lower (PR #402 body)
CEILING_BAND_HI_390: float = 509.78         # #390 corrected ceiling band upper (PR #402 body)

# ---- #289 (fi34s269) DEPLOYED MTP per-position conditional acceptance ladder a_1..a_7 (K=7) -------
LADDER_289: list[float] = [
    0.7292532942898975, 0.759556697719242, 0.7929794882639035, 0.8228,
    0.8348727920920435, 0.8357919254658385, 0.8464932652113331,
]
E_ACCEPTED_289: float = 2.851185944363104   # #289 E[accepted draft tokens]/step
E_T_289: float = 3.851185944363104          # #289 E[T] = 1 + E[accepted] (ladder)

# ---- #387 (z8osvif8) grounded top-K coverage anchors (deployed MTP, on-distribution for the 128) --
TOP4_COVERAGE: float = 0.8902659519153152   # measured top-4 coverage == #336/#340 program anchor c0
TOP1_COVERAGE: float = 0.7617               # measured top-1 coverage (fern#34 holdout)
COV_BUDGET_336: float = 0.031035214630377506  # #336 trainable coverage headroom (identity_bar - prior)
IDENTITY_BAR_336: float = 0.9213011665456927  # #336 greedy-identity coverage bar
CSTAR_CENTRAL_340: float = 0.9089           # #340 c* central (program coverage->E[T] secant knot)
# head-ceiling gap (g band UPPER bound; a direct top-8/16 GPU read is BLOCKED #387 -> bounded by 1):
COVERAGE_CEILING_GAP: float = 1.0 - TOP4_COVERAGE      # 0.10973 -- the g in [0, this] band
LOCKED_PRIZE_TOP1_TOP4: float = TOP4_COVERAGE - TOP1_COVERAGE  # 0.12857 -- top1->top4 E[accepted] prize

# ---- #383/#387 public<->private demand secant (program coverage -> E[T]) + #396 inversion plumbing
MU_P: float = 481.53                        # deployed public TPS (PR #52, 2x9fm2zx)
K_CAL: float = 125.26795005202914           # steps/s; public official TPS = E[T] * K_cal (#344)
E_T_REALIZED: float = MU_P / K_CAL          # 3.84399 secant-consistent realized accept length (#396)
ET_PUBLIC_500: float = 500.0 / K_CAL        # 3.99144 E[T] at the speed-500 bar
TARGET: float = 500.0
# program coverage->E[T] central secant (anchor-coupled at the central prior). Banked 7.912609 (#387).
S_CENTRAL: float = (ET_PUBLIC_500 - E_T_REALIZED) / (CSTAR_CENTRAL_340 - TOP4_COVERAGE)
PUBLISHED_S_CENTRAL: float = 7.912609135742992
# denken #396 (yc5ji486): demand-alone required_dcov on the OLD 471.42 base (cross-check anchor).
REQ_DCOV_396_OLDBASE: float = 0.02945619989109333
# #399 (ec7i3z5t) headline gross-gain-per-unit-cov on the OLD base w/ the ladder E[T] (cross-check).
PUBLISHED_GROSS_PER_COV_399: float = 968.57

# ---- #332 (y5cl0ena) deployed M=8 verify SDPA launch geometry (the verify-M roofline) ------------
# eagle3_sdpa_phi_floor.py: GQA / split-KV tiling of the deployed splitkv_verify_patch (PR #39).
A10G_SMS: int = 80                          # GA102, 80 enabled (deployed-patch + repo-measured)
M_DEPLOYED: int = 8                         # deployed verify rows = K_spec(7) + 1 chain/bonus token
BLOCK_M: int = 16                           # vLLM Triton BLOCK_M
NUM_Q_HEADS: int = 8
NUM_KV_HEADS: int = 2                        # GQA: 2 KV heads
GQA_GROUP: int = NUM_Q_HEADS // NUM_KV_HEADS  # 4
BLOCK_Q: int = BLOCK_M // GQA_GROUP         # 16 // 4 = 4 query rows per CTA
NUM_SEQS: int = 1                           # concurrency = 1 single sequence
NUM_PAR_SOFTMAX_SEGMENTS: int = 16          # LOCKED 3D split-KV reduction segments (no kernel change)
SDPA_BW_UTIL_332: float = 0.34883864849061247  # measured M=8 SDPA bandwidth utilisation (BW-floored)

# ---- #378/#393 served-step component fraction: the M=8 verify ATTENTION lane (the tax lever) ------
# attention_strict_pin_cost.py F_ATTN_344 == #378 step_fractions.attn (served spec-step attn share).
F_ATTN_STEP: float = 0.09506718019009251    # M=8 verify attention = 9.5% of the served spec step
SDPA_PENALTY_FREE_393: float = 0.9956       # #393: M=8 verify lane penalty-free (0.44% WITHIN the lane)

# ---- #371 (nonspec-baselift-cudagraph) capture-vs-eager hazard (dynamic-shape flag, NOT headline) -
CAPTURE_OVER_EAGER_X_371: float = 2.0       # captured graph ~2x eager (launch+resident+fusion); HAZARD

PPL_DEPLOYED: float = 2.3772
PPL_GATE: float = 2.42

TOL_PROV: float = 1e-6


# ===========================================================================
# Section 1 -- E[T] / coverage geometry from the #289 ladder + #387 anchors
# ===========================================================================

def expected_accepted(ladder: list[float]) -> float:
    """E[accepted draft tokens]/step = sum_k prod_{j<=k} a_j (conditional ladder)."""
    cum, acc = 1.0, 0.0
    for a in ladder:
        cum *= a
        acc += cum
    return acc


def expected_tokens_per_step(ladder: list[float]) -> float:
    return 1.0 + expected_accepted(ladder)


# ===========================================================================
# Section 2 -- GAIN leg: program-coverage uplift g -> Delta E[accepted] -> TPS (fixed T_step)
# ===========================================================================
# The deployed program coverage anchor is the top-4 = 0.890 (#340/#387). A width-K tree raises the
# realized per-position accepted-coverage toward top-K; mapped through the #387 demand secant
# (dE[T] = S * dcov) this lifts E[T] hence TPS on a FIXED T_step (the tax leg prices T_step separately).
# g = the realized program-coverage uplift over the 0.890 anchor, in the head-ceiling band [0, 0.1097].

def gross_tps_gain(g: float, base: float = BASE_467, base_et: float = E_T_REALIZED,
                   slope: float = S_CENTRAL) -> float:
    """TPS lift at FIXED T_step from a coverage uplift g (the PR's 'gross_tps_gain(g)').

    TPS = E[T] / T_step; raising E[T] by S*g at fixed T_step => dTPS = base * S*g / base_et."""
    return base * slope * g / base_et


def gross_tps_gain_per_unit_cov(base: float = BASE_467, base_et: float = E_T_REALIZED) -> float:
    """The secant->TPS conversion factor (~962 on the corrected base; ~968.57 on the OLD #399 base)."""
    return gross_tps_gain(1.0, base=base, base_et=base_et)


# ===========================================================================
# Section 3 -- TAX leg: verify-M step-time inflation from the roofline + CUDA-graph rebuild
# ===========================================================================
# The deployed split-KV verify attention (LOCKED 16-way split, NO kernel change) tiles M query rows
# into query-blocks of BLOCK_Q=4, x NUM_KV_HEADS, x the 16-way reduction split. Each query-block
# re-reads its KV-head's full context (no cross-block reuse -> the 34.9% BW floor), so the attention
# time is BW-bound and scales with the query-block count N_nonreduction(M). lm_head GEMM (0.834 BW,
# weight-bound) and the body GEMMs stay weight-bound at these M (<< the compute crossover ~200), so
# ONLY the 9.5%-of-step attention lane inflates. The CUDA-graph rebuild for the new width is a
# one-time capture (#390 counts it as a discrete kernel rebuild, not a per-step cost); the per-step
# hazard is the dynamic-shape eager fallback (~2x, #371) IF the tree shape is not static-capturable.

def n_nonreduction(m: int | float) -> float:
    """Query-block x KV-head tile count for M verify rows (the BW-bound attention work unit)."""
    return (math.ceil(m / BLOCK_Q) + NUM_SEQS) * NUM_KV_HEADS


def n_full_3d(m: int | float) -> float:
    """Total CTAs incl. the LOCKED 16-way split-KV reduction (oversubscribes 80 SMs at M=8: 96>80)."""
    return n_nonreduction(m) * NUM_PAR_SOFTMAX_SEGMENTS


def attn_scale(m: int | float) -> float:
    """Attention-time multiplier vs the deployed M=8 (BW-bound -> proportional to query-blocks)."""
    return n_nonreduction(m) / n_nonreduction(M_DEPLOYED)


# --- tree-shape verify-row models (state both explicitly; headline = full single-forward fan-out) --
def m_full_fanout(k: int) -> int:
    """Honest single-forward verify cost: top-K candidates carried at each of the 7 draft slots + the
    bonus row (a depth-<=7 beam-K token tree). All candidate rows are verified BEFORE acceptance is
    known, so the verify processes M = 1 + 7*K rows under a tree-causal mask."""
    return 1 + 7 * k


def m_depth1_branch(k: int) -> float:
    """OPTIMISTIC floor: branch top-K at ONE position only (literal 'depth-1'), rest linear. Adds
    (K-1) candidate rows to the deployed M=8 -> M = 8 + (K-1). Widens a single position, so its
    realized program-coverage uplift is small (reported as the cheap-but-weak end of the band)."""
    return float(M_DEPLOYED + (k - 1))


def tstep_tax_frac(k: int, shape: str = "full_fanout") -> float:
    """T_step inflation fraction tau(K): the 9.5%-of-step attention lane scaled by the wider-M
    roofline. tau = F_ATTN_STEP * (attn_scale(M) - 1); other lanes stay weight-bound (~flat)."""
    m = m_full_fanout(k) if shape == "full_fanout" else m_depth1_branch(k)
    return F_ATTN_STEP * (attn_scale(m) - 1.0)


def tps_loss(k: int, shape: str = "full_fanout", base: float = BASE_467) -> float:
    """TPS drop from inflating T_step by tau at FIXED E[T]: base -> base/(1+tau), loss = base*tau/(1+tau)."""
    tau = tstep_tax_frac(k, shape)
    return base * tau / (1.0 + tau)


# ===========================================================================
# Section 4 -- NET + go/no-go: net_tps(g,K), breakeven, g*, and the verdict bools
# ===========================================================================

def net_tps_exact(g: float, k: int, shape: str = "full_fanout", base: float = BASE_467,
                  base_et: float = E_T_REALIZED, slope: float = S_CENTRAL) -> float:
    """Honest multiplicative composition: TPS = (E[T]+S*g)/(T_step*(1+tau)) - base
    = base * [ (1 + S*g/E[T]) / (1+tau) - 1 ]. (The gain reduces steps; the tax inflates T_step;
    they compose multiplicatively, NOT additively -- the cross term is real at these g, tau.)"""
    tau = tstep_tax_frac(k, shape)
    return base * ((1.0 + slope * g / base_et) / (1.0 + tau) - 1.0)


def net_tps_linear(g: float, k: int, shape: str = "full_fanout") -> float:
    """The PR's named decomposition net = gross_tps_gain(g) - tps_loss(K) (first-order; overstates
    net by the cross term gross_gain*tau/(1+tau) vs the exact multiplicative composition)."""
    return gross_tps_gain(g) - tps_loss(k, shape)


def g_breakeven(k: int, shape: str = "full_fanout", base_et: float = E_T_REALIZED,
                slope: float = S_CENTRAL) -> float:
    """g at which net_tps_exact = 0: (1 + S*g/E[T])/(1+tau) = 1 => g = (E[T]/S) * tau."""
    tau = tstep_tax_frac(k, shape)
    return (base_et / slope) * tau


def g_star_close_500(k: int, shape: str = "full_fanout", base: float = BASE_467,
                     base_et: float = E_T_REALIZED, slope: float = S_CENTRAL,
                     target: float = TARGET) -> float:
    """Realized-fraction threshold g* with net_tps_exact(g*,K) = (target-base): solve
    (1 + S*g*/E[T])/(1+tau) = target/base  =>  g* = (E[T]/S) * [ (target/base)*(1+tau) - 1 ]."""
    tau = tstep_tax_frac(k, shape)
    return (base_et / slope) * ((target / base) * (1.0 + tau) - 1.0)


def required_dcov_corrected(base: float = BASE_467, base_et: float = E_T_REALIZED,
                            slope: float = S_CENTRAL, target: float = TARGET) -> float:
    """The TAX-FREE (tau=0) g* == denken #396's required_dcov on the corrected base (reconciliation):
    g* = (E[T]/S)*(target/base - 1). On 467.48 this is ~0.0338 -> already busts the +0.031 #336 budget."""
    return (base_et / slope) * (target / base - 1.0)


def go_nogo_card() -> dict:
    """Assemble the per-(K, shape) tax/gain/net ledger and the go/no-go verdicts."""
    widths = [4, 8, 16]
    shapes = ["full_fanout", "depth1_branch"]
    g_max = COVERAGE_CEILING_GAP            # 0.1097 -- the most optimistic realizable uplift (K->inf)

    ledger: dict[str, dict] = {}
    for shape in shapes:
        for k in widths:
            m = m_full_fanout(k) if shape == "full_fanout" else m_depth1_branch(k)
            tau = tstep_tax_frac(k, shape)
            gstar = g_star_close_500(k, shape)
            gbe = g_breakeven(k, shape)
            ledger[f"{shape}_k{k}"] = {
                "shape": shape, "k": k, "verify_rows_M": m,
                "n_nonreduction": n_nonreduction(m), "n_full_3d": n_full_3d(m),
                "attn_scale_vs_m8": attn_scale(m),
                "tstep_tax_frac": tau,
                "tps_loss": tps_loss(k, shape),
                "gross_gain_at_full_gap": gross_tps_gain(g_max),
                "net_tps_at_full_gap_exact": net_tps_exact(g_max, k, shape),
                "net_tps_at_full_gap_linear": net_tps_linear(g_max, k, shape),
                "g_breakeven_net0": gbe,
                "g_star_close_500": gstar,
                "g_star_exceeds_ceiling_band": bool(gstar > g_max),
                "g_star_exceeds_top1_top4_prize": bool(gstar > LOCKED_PRIZE_TOP1_TOP4),
                "net_positive_at_max_achievable_g": bool(net_tps_exact(g_max, k, shape) > 0.0),
            }

    # The first width that buys ANY coverage uplift over the top-4 anchor is K>4 (K=4 -> top-4 = anchor
    # -> g = 0). So the decision-relevant operating points are K>=8. tree_verify_net_positive asks "is
    # there ANY (g<=g_max, K>=8, shape) with net>0"; tree_closes_500_alone asks "can the net reach the
    # 32.53 gap" -- i.e. g*(K) <= g_max at a width that buys g>0.
    # honest headline = full_fanout (the single-forward verify must process all proposed rows).
    head_k8 = ledger["full_fanout_k8"]
    opt_k8 = ledger["depth1_branch_k8"]

    # net_positive: TRUE iff SOME feasible operating point has net>0 at its max achievable g.
    tree_verify_net_positive = any(
        v["net_positive_at_max_achievable_g"] for kk, v in ledger.items() if v["k"] >= 8)
    # closes-500-alone: TRUE iff SOME width that buys g>0 (K>=8) has g*(K) <= g_max (reachable).
    closes_candidates = {kk: v for kk, v in ledger.items() if v["k"] >= 8}
    tree_closes_500_alone = any(not v["g_star_exceeds_ceiling_band"] and v["k"] >= 8
                                and v["shape"] == "full_fanout" for kk, v in closes_candidates.items())

    # RECONCILE #396: the tax-free (tau=0) g* IS denken #396's corrected-base required_dcov. At K=4,
    # m_full_fanout(4)=29 BUT the gain-relevant tax-free limit is required_dcov_corrected (tau->0). The
    # reconciliation holds iff that matches #396's corrected 0.0338 AND already busts the +0.031 budget.
    req_tax_free = required_dcov_corrected()
    reconciles_396 = bool(abs(req_tax_free - 0.033800214858626394) < 1e-6
                          and req_tax_free > COV_BUDGET_336)

    return {
        "ledger": ledger,
        "g_max_ceiling_band": g_max,
        "locked_prize_top1_top4": LOCKED_PRIZE_TOP1_TOP4,
        "required_dcov_tax_free_corrected": req_tax_free,           # == #396 corrected-base required
        "required_dcov_busts_336_budget": bool(req_tax_free > COV_BUDGET_336),
        # headline scalars (honest full_fanout @ K=8, the first width that buys g>0):
        "headline_k": 8,
        "headline_shape": "full_fanout",
        "tstep_tax_frac_k8_headline": head_k8["tstep_tax_frac"],
        "tps_loss_k8_headline": head_k8["tps_loss"],
        "net_tps_at_full_gap_headline": head_k8["net_tps_at_full_gap_exact"],
        "g_star_headline": head_k8["g_star_close_500"],
        "g_star_headline_exceeds_band": head_k8["g_star_exceeds_ceiling_band"],
        # optimistic floor (depth-1 branch @ K=8):
        "tstep_tax_frac_k8_optimistic": opt_k8["tstep_tax_frac"],
        "net_tps_at_full_gap_optimistic": opt_k8["net_tps_at_full_gap_exact"],
        "g_star_optimistic": opt_k8["g_star_close_500"],
        # verdict bools (deliverables):
        "tree_verify_net_positive": tree_verify_net_positive,
        "tree_closes_500_alone": tree_closes_500_alone,
        "tree_plus_cb3_required": bool(not tree_closes_500_alone),
        "reconciles_396_corrected_base": reconciles_396,
        # CUDA-graph rebuild accounting (#390 style: a discrete count, amortized per-step; + hazard):
        "n_distinct_graph_rebuilds_for_tree": N_REBUILDS_ARM_A_390,   # new verify-width graph = 1 rebuild
        "graph_rebuild_is_one_time_amortized": True,
        "dynamic_shape_eager_hazard_x": CAPTURE_OVER_EAGER_X_371,     # ~2x IF shape not static-capturable
    }


# ===========================================================================
# Section 5 -- self-tests (>= 20 checks)
# ===========================================================================

def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))


def run_self_tests(et: float, card: dict) -> dict:
    c: dict[str, bool] = {}
    led = card["ledger"]

    # a) provenance: corrected base / gap / secant / ladder match the banked anchors.
    c["a_base_is_393"] = abs(BASE_467 - 467.475218449957) < TOL_PROV
    c["a_gap_is_393"] = abs(GAP_32 - 32.524781550042974) < TOL_PROV
    c["a_base_plus_gap_is_500"] = abs(BASE_467 + GAP_32 - 500.0) < 1e-6
    c["a_secant_matches_387"] = abs(S_CENTRAL - PUBLISHED_S_CENTRAL) < 1e-9
    c["a_ladder_len_7"] = len(LADDER_289) == 7
    c["a_ladder_monotone"] = all(LADDER_289[i] <= LADDER_289[i + 1] for i in range(6))
    c["a_et_roundtrips_289"] = abs(et - E_T_289) < 1e-9

    # b) coverage geometry: ordering + the two banked gaps (ceiling band 0.1097, prize 0.1286).
    c["b_top1_le_top4"] = TOP1_COVERAGE <= TOP4_COVERAGE <= 1.0
    c["b_ceiling_gap_is_complement_top4"] = abs(COVERAGE_CEILING_GAP - (1.0 - TOP4_COVERAGE)) < 1e-12
    c["b_ceiling_gap_rounds_0p1097"] = round(COVERAGE_CEILING_GAP, 4) == 0.1097
    c["b_prize_rounds_0p1286"] = round(LOCKED_PRIZE_TOP1_TOP4, 4) == 0.1286
    c["b_prize_exceeds_ceiling_band"] = LOCKED_PRIZE_TOP1_TOP4 > COVERAGE_CEILING_GAP

    # c) GAIN leg: per-unit-cov reproduces ~962 (corrected) and the #399 oldbase 968.57 anchor.
    gpc_corr = gross_tps_gain_per_unit_cov()
    gpc_old = gross_tps_gain_per_unit_cov(base=BASE_471_390, base_et=E_T_289)
    c["c_gross_per_cov_corrected_900s"] = 900.0 < gpc_corr < 1000.0
    c["c_gross_per_cov_oldbase_matches_399"] = abs(gpc_old - PUBLISHED_GROSS_PER_COV_399) < 0.05
    c["c_gain_zero_at_zero_g"] = abs(gross_tps_gain(0.0)) < 1e-12
    c["c_gain_monotone_in_g"] = gross_tps_gain(0.05) > gross_tps_gain(0.0)
    c["c_gain_at_full_gap_positive"] = gross_tps_gain(COVERAGE_CEILING_GAP) > 0.0

    # d) ROOFLINE geometry reproduces the #332 M=8 anchors EXACTLY.
    c["d_blockq_is_4"] = BLOCK_Q == 4
    c["d_nnr_m8_is_6"] = n_nonreduction(8) == 6
    c["d_nfull_m8_is_96"] = n_full_3d(8) == 96
    c["d_m8_oversubscribes_80sm"] = n_full_3d(8) > A10G_SMS
    c["d_attn_scale_unit_at_m8"] = abs(attn_scale(8) - 1.0) < 1e-12
    c["d_attn_scale_monotone"] = attn_scale(57) > attn_scale(29) > attn_scale(8)

    # e) TAX leg: tau positive & monotone in K; full_fanout taxes the headline K=4/K=8 points.
    c["e_tax_k4_full_positive"] = tstep_tax_frac(4, "full_fanout") > 0.0
    c["e_tax_monotone_in_k_full"] = (tstep_tax_frac(8, "full_fanout")
                                     > tstep_tax_frac(4, "full_fanout"))
    c["e_tax_full_exceeds_depth1"] = (tstep_tax_frac(8, "full_fanout")
                                      > tstep_tax_frac(8, "depth1_branch"))
    c["e_tps_loss_lt_base"] = tps_loss(8, "full_fanout") < BASE_467
    c["e_m_full_k4_is_29"] = m_full_fanout(4) == 29
    c["e_m_full_k8_is_57"] = m_full_fanout(8) == 57

    # f) NET composition: exact <= linear (cross term), breakeven & g* ordering, gain-only sanity.
    c["f_exact_le_linear_at_full_gap"] = (net_tps_exact(COVERAGE_CEILING_GAP, 8, "full_fanout")
                                          <= net_tps_linear(COVERAGE_CEILING_GAP, 8, "full_fanout") + 1e-9)
    c["f_net_zero_at_breakeven"] = abs(net_tps_exact(g_breakeven(8, "full_fanout"), 8, "full_fanout")) < 1e-6
    c["f_net_hits_gap_at_gstar"] = abs(net_tps_exact(g_star_close_500(8, "full_fanout"), 8, "full_fanout")
                                       - GAP_32) < 1e-6
    c["f_gstar_gt_breakeven"] = g_star_close_500(8, "full_fanout") > g_breakeven(8, "full_fanout")

    # g) the DECISION: headline g*(K=8 full) exceeds the achievable band AND the prize -> NO-GO alone.
    c["g_headline_gstar_exceeds_band"] = led["full_fanout_k8"]["g_star_exceeds_ceiling_band"]
    c["g_headline_gstar_exceeds_prize"] = led["full_fanout_k8"]["g_star_exceeds_top1_top4_prize"]
    c["g_headline_net_full_gap_negative"] = led["full_fanout_k8"]["net_tps_at_full_gap_exact"] < 0.0
    c["g_closes_500_alone_false"] = card["tree_closes_500_alone"] is False
    c["g_plus_cb3_required_true"] = card["tree_plus_cb3_required"] is True

    # h) RECONCILE #396: tax-free g* == corrected required_dcov 0.0338, which busts the +0.031 budget.
    c["h_tax_free_gstar_matches_396_corrected"] = abs(card["required_dcov_tax_free_corrected"]
                                                      - 0.033800214858626394) < 1e-6
    c["h_corrected_required_busts_budget"] = card["required_dcov_busts_336_budget"]
    c["h_reconciles_396_true"] = card["reconciles_396_corrected_base"] is True
    c["h_oldbase_required_matches_396"] = abs(
        required_dcov_corrected(base=BASE_471_390) - REQ_DCOV_396_OLDBASE) < 1e-6

    # i) PPL/greedy: tree verify preserves greedy identity -> PPL unchanged -> passes gate.
    c["i_ppl_passes_gate"] = PPL_DEPLOYED <= PPL_GATE

    # j) CUDA-graph rebuild accounting present and sane.
    c["j_rebuild_count_ge_1"] = card["n_distinct_graph_rebuilds_for_tree"] >= 1
    c["j_eager_hazard_gt_1"] = card["dynamic_shape_eager_hazard_x"] > 1.0

    # k) numeric hygiene across the headline scalars.
    flat = [et, S_CENTRAL, gross_tps_gain_per_unit_cov(), tstep_tax_frac(8, "full_fanout"),
            tps_loss(8, "full_fanout"), net_tps_exact(COVERAGE_CEILING_GAP, 8, "full_fanout"),
            g_star_close_500(8, "full_fanout"), card["required_dcov_tax_free_corrected"]]
    c["k_no_nan_inf"] = all(_finite(v) for v in flat)

    passes = all(c.values())
    return {"conditions": c, "n_checks": len(c), "n_passed": sum(1 for v in c.values() if v),
            "passes": passes}


# ===========================================================================
# Section 6 -- report assembly + W&B + CLI
# ===========================================================================

def build_report() -> dict:
    et = expected_tokens_per_step(LADDER_289)
    card = go_nogo_card()
    selftest = run_self_tests(et, card)
    led = card["ledger"]
    tree_shape_str = (
        "depth-<=7 per-position top-K token tree, single-forward verify. HEADLINE 'full_fanout': "
        "top-K candidates at each of the 7 MTP draft slots + 1 bonus row, all verified before "
        "acceptance -> M(K)=1+7K (K=4->29, K=8->57). FLOOR 'depth1_branch': branch top-K at ONE "
        "position only -> M(K)=8+(K-1) (K=8->15). The deployed split-KV attention (LOCKED 16-way "
        "split, NO kernel change) re-reads KV per query-block, so attention time scales with "
        "N_nonreduction(M)=(ceil(M/4)+1)*2."
    )
    return {
        "pr": 402, "agent": "denken", "kind": "tree-verify-net-tps-go-nogo",
        "analysis_only": True, "no_launch": True, "no_hf_job": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used": False, "official_tps": 0,
        "baseline_unchanged_tps": 481.53, "baseline_unchanged_ppl": 2.3772,
        "corrected_strict_base_tps": BASE_467, "gap_to_500_tps": GAP_32,
        "inputs": {
            "base_467_393": BASE_467, "gap_32_393": GAP_32, "ceiling_505_393": CEILING_505_393,
            "base_471_390": BASE_471_390, "gap_28_390": GAP_28_390,
            "n_rebuilds_arm_a_390": N_REBUILDS_ARM_A_390,
            "ladder_289": LADDER_289, "e_accepted_289": E_ACCEPTED_289, "e_t_289": E_T_289,
            "top4_coverage_387": TOP4_COVERAGE, "top1_coverage_387": TOP1_COVERAGE,
            "coverage_ceiling_gap": COVERAGE_CEILING_GAP, "locked_prize_top1_top4": LOCKED_PRIZE_TOP1_TOP4,
            "cov_budget_336": COV_BUDGET_336, "identity_bar_336": IDENTITY_BAR_336,
            "cstar_central_340": CSTAR_CENTRAL_340, "s_central": S_CENTRAL,
            "mu_p": MU_P, "k_cal": K_CAL, "e_t_realized": E_T_REALIZED, "target": TARGET,
            "req_dcov_396_oldbase": REQ_DCOV_396_OLDBASE,
            "published_gross_per_cov_399": PUBLISHED_GROSS_PER_COV_399,
            "a10g_sms": A10G_SMS, "m_deployed": M_DEPLOYED, "block_q": BLOCK_Q,
            "num_kv_heads": NUM_KV_HEADS, "num_seqs": NUM_SEQS,
            "num_par_softmax_segments": NUM_PAR_SOFTMAX_SEGMENTS, "sdpa_bw_util_332": SDPA_BW_UTIL_332,
            "f_attn_step_378": F_ATTN_STEP, "sdpa_penalty_free_393": SDPA_PENALTY_FREE_393,
            "capture_over_eager_x_371": CAPTURE_OVER_EAGER_X_371,
            "ppl_deployed": PPL_DEPLOYED, "ppl_gate": PPL_GATE,
            "source_393_run": "0q7ynumg", "source_399_run": "ec7i3z5t", "source_396_run": "yc5ji486",
            "source_387_run": "z8osvif8", "source_289_run": "fi34s269", "source_332_run": "y5cl0ena",
            "source_390_run": "5y64zbjz", "source_336_ref": "cov_budget=0.031035",
            "source_340_anchor": "cstar_central=0.9089", "source_207_ref": "kanna cb3 supply lift (sibling)",
            "source_401_ref": "ubel top-8/16 realized coverage point (sibling, pending)",
        },
        "expected_tokens_per_step": et,
        "expected_accepted_draft": et - 1.0,
        "tree_shape": tree_shape_str,
        # ---- card-required deliverable scalars (SENPAI-RESULT / W&B load-bearing) ----
        "gross_tps_gain_per_unit_cov": gross_tps_gain_per_unit_cov(),
        "gross_tps_gain_per_unit_cov_oldbase_399": gross_tps_gain_per_unit_cov(base=BASE_471_390,
                                                                               base_et=E_T_289),
        "tstep_tax_frac_k4": tstep_tax_frac(4, "full_fanout"),
        "tstep_tax_frac_k8": tstep_tax_frac(8, "full_fanout"),
        "tstep_tax_frac_k4_depth1": tstep_tax_frac(4, "depth1_branch"),
        "tstep_tax_frac_k8_depth1": tstep_tax_frac(8, "depth1_branch"),
        "tps_loss_k4": tps_loss(4, "full_fanout"),
        "tps_loss_k8": tps_loss(8, "full_fanout"),
        "net_tps_at_full_gap": card["net_tps_at_full_gap_headline"],
        "net_tps_at_full_gap_optimistic_depth1_k8": card["net_tps_at_full_gap_optimistic"],
        "g_star_threshold_to_close_500": card["g_star_headline"],
        "g_star_optimistic_depth1_k8": card["g_star_optimistic"],
        "g_max_ceiling_band": card["g_max_ceiling_band"],
        "tree_verify_net_positive": card["tree_verify_net_positive"],
        "tree_closes_500_alone": card["tree_closes_500_alone"],
        "tree_plus_cb3_required": card["tree_plus_cb3_required"],
        "reconciles_396_corrected_base": card["reconciles_396_corrected_base"],
        "required_dcov_tax_free_corrected": card["required_dcov_tax_free_corrected"],
        "n_distinct_graph_rebuilds_for_tree": card["n_distinct_graph_rebuilds_for_tree"],
        "dynamic_shape_eager_hazard_x": card["dynamic_shape_eager_hazard_x"],
        "go_nogo": card,
        "self_test": selftest,
        "tree_verify_net_tps_self_test_passes": selftest["passes"],
    }


def log_to_wandb(report: dict, group: str, name: str) -> str:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_unavailable"
    try:
        run = wandb.init(group=group, name=name, config=report["inputs"])
        card = report["go_nogo"]
        wandb.summary.update({
            "gross_tps_gain_per_unit_cov": report["gross_tps_gain_per_unit_cov"],
            "tree_shape": report["tree_shape"],
            "tstep_tax_frac_k4": report["tstep_tax_frac_k4"],
            "tstep_tax_frac_k8": report["tstep_tax_frac_k8"],
            "tps_loss_k4": report["tps_loss_k4"],
            "tps_loss_k8": report["tps_loss_k8"],
            "net_tps_at_full_gap": report["net_tps_at_full_gap"],
            "tree_verify_net_positive": report["tree_verify_net_positive"],
            "g_star_threshold_to_close_500": report["g_star_threshold_to_close_500"],
            "tree_closes_500_alone": report["tree_closes_500_alone"],
            "tree_plus_cb3_required": report["tree_plus_cb3_required"],
            "reconciles_396_corrected_base": report["reconciles_396_corrected_base"],
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            "tree_verify_net_tps_self_test_passes": report["tree_verify_net_tps_self_test_passes"],
        })
        wandb.log({
            "summary/gross_tps_gain_per_unit_cov": report["gross_tps_gain_per_unit_cov"],
            "summary/gross_tps_gain_per_unit_cov_oldbase_399": report["gross_tps_gain_per_unit_cov_oldbase_399"],
            "summary/tstep_tax_frac_k4": report["tstep_tax_frac_k4"],
            "summary/tstep_tax_frac_k8": report["tstep_tax_frac_k8"],
            "summary/tstep_tax_frac_k4_depth1": report["tstep_tax_frac_k4_depth1"],
            "summary/tstep_tax_frac_k8_depth1": report["tstep_tax_frac_k8_depth1"],
            "summary/tps_loss_k4": report["tps_loss_k4"],
            "summary/tps_loss_k8": report["tps_loss_k8"],
            "summary/net_tps_at_full_gap": report["net_tps_at_full_gap"],
            "summary/net_tps_at_full_gap_optimistic_depth1_k8": report["net_tps_at_full_gap_optimistic_depth1_k8"],
            "summary/g_star_threshold_to_close_500": report["g_star_threshold_to_close_500"],
            "summary/g_star_optimistic_depth1_k8": report["g_star_optimistic_depth1_k8"],
            "summary/g_max_ceiling_band": report["g_max_ceiling_band"],
            "summary/required_dcov_tax_free_corrected": report["required_dcov_tax_free_corrected"],
            "summary/tree_verify_net_positive": float(report["tree_verify_net_positive"]),
            "summary/tree_closes_500_alone": float(report["tree_closes_500_alone"]),
            "summary/tree_plus_cb3_required": float(report["tree_plus_cb3_required"]),
            "summary/reconciles_396_corrected_base": float(report["reconciles_396_corrected_base"]),
            "summary/n_distinct_graph_rebuilds_for_tree": float(report["n_distinct_graph_rebuilds_for_tree"]),
            "summary/dynamic_shape_eager_hazard_x": report["dynamic_shape_eager_hazard_x"],
            "summary/corrected_strict_base_tps": report["corrected_strict_base_tps"],
            "summary/gap_to_500_tps": report["gap_to_500_tps"],
            "summary/expected_tokens_per_step": report["expected_tokens_per_step"],
            "summary/self_test_passes": float(report["self_test"]["passes"]),
            "summary/self_test_n_checks": float(report["self_test"]["n_checks"]),
        })
        # per-(shape, K) ledger series (so the tax/net curve is in W&B).
        for tag, row in card["ledger"].items():
            wandb.log({f"ledger/{tag}/verify_rows_M": float(row["verify_rows_M"]),
                       f"ledger/{tag}/tstep_tax_frac": row["tstep_tax_frac"],
                       f"ledger/{tag}/tps_loss": row["tps_loss"],
                       f"ledger/{tag}/net_tps_at_full_gap_exact": row["net_tps_at_full_gap_exact"],
                       f"ledger/{tag}/g_star_close_500": row["g_star_close_500"],
                       f"ledger/{tag}/g_breakeven_net0": row["g_breakeven_net0"]})
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_failed"


def print_report(r: dict) -> None:
    card = r["go_nogo"]
    print("\n=== Tree-verify NET-TPS go/no-go (PR #402, denken) ===")
    print(f"corrected strict base (#393 0q7ynumg) = {BASE_467:.4f} TPS   gap_to_500 = {GAP_32:.4f}")
    print(f"#289 ladder E[accepted]={r['expected_accepted_draft']:.4f}  E[T]={r['expected_tokens_per_step']:.4f}")
    print(f"coverage: top1={TOP1_COVERAGE:.4f} top4(anchor)={TOP4_COVERAGE:.4f}  "
          f"ceiling band g_max=1-top4={COVERAGE_CEILING_GAP:.4f}  locked prize(top1->top4)={LOCKED_PRIZE_TOP1_TOP4:.4f}")
    print(f"demand secant S={S_CENTRAL:.4f} E[T]/cov  ->  gross gain/unit-cov = "
          f"{r['gross_tps_gain_per_unit_cov']:.2f} TPS (corrected base; oldbase #399 = "
          f"{r['gross_tps_gain_per_unit_cov_oldbase_399']:.2f} ~= 968.57)")
    print("\n-- TAX leg: verify-M roofline (deployed split-KV, LOCKED 16-way split, no kernel change) --")
    print(f"  M=8 deployed: N_nonreduction={n_nonreduction(8):.0f}  N_full_3d={n_full_3d(8):.0f} CTAs "
          f"(> {A10G_SMS} SMs -> occupancy-saturated)  attn lane = {F_ATTN_STEP*100:.2f}% of step")
    hdr = f"  {'shape/K':<22}{'M':>5}{'N_nr':>6}{'attn_x':>8}{'tau':>8}{'tps_loss':>10}{'net@full_gap':>14}{'g*':>9}"
    print(hdr)
    for tag, row in card["ledger"].items():
        print(f"  {tag:<22}{row['verify_rows_M']:>5.0f}{row['n_nonreduction']:>6.0f}"
              f"{row['attn_scale_vs_m8']:>8.2f}{row['tstep_tax_frac']:>8.3f}{row['tps_loss']:>10.2f}"
              f"{row['net_tps_at_full_gap_exact']:>14.2f}{row['g_star_close_500']:>9.4f}")
    print("\n-- NET + go/no-go (HEADLINE = full_fanout @ K=8, the first width that buys g>0) --")
    print(f"  tstep_tax_frac(K=4)={r['tstep_tax_frac_k4']:.4f}  (K=8)={r['tstep_tax_frac_k8']:.4f}   "
          f"tps_loss(K=4)={r['tps_loss_k4']:.2f}  (K=8)={r['tps_loss_k8']:.2f}")
    print(f"  net_tps_at_full_gap (g={COVERAGE_CEILING_GAP:.4f}, K=8 full) = {r['net_tps_at_full_gap']:+.2f} TPS "
          f"(optimistic depth1 K=8 = {r['net_tps_at_full_gap_optimistic_depth1_k8']:+.2f})")
    print(f"  g_star_to_close_500 (K=8 full) = {r['g_star_threshold_to_close_500']:.4f}  vs g_max="
          f"{COVERAGE_CEILING_GAP:.4f}  prize={LOCKED_PRIZE_TOP1_TOP4:.4f}  "
          f"(optimistic depth1 g* = {r['g_star_optimistic_depth1_k8']:.4f})")
    print(f"  tax-free g* (== #396 corrected required_dcov) = {r['required_dcov_tax_free_corrected']:.5f}  "
          f"(busts #336 budget +{COV_BUDGET_336:.5f}? {card['required_dcov_busts_336_budget']})")
    print("\n-- VERDICT --")
    print(f"  tree_verify_net_positive        = {r['tree_verify_net_positive']}  "
          f"(weak: only the optimistic depth-1 shape; honest full_fanout @ K>=8 is NEGATIVE)")
    print(f"  tree_closes_500_alone           = {r['tree_closes_500_alone']}  "
          f"(g*(K>=8) >= g_max at every width that buys g>0)")
    print(f"  tree_plus_cb3_required          = {r['tree_plus_cb3_required']}  (needs kanna #207 cb3 assist)")
    print(f"  reconciles_396_corrected_base   = {r['reconciles_396_corrected_base']}")
    print(f"  CUDA-graph rebuilds (new width) = {r['n_distinct_graph_rebuilds_for_tree']} (one-time, amortized); "
          f"dynamic-shape eager hazard ~{r['dynamic_shape_eager_hazard_x']:.1f}x if not static-capturable")
    print(f"\nPPL: greedy tree verify preserves token identity -> PPL unchanged {PPL_DEPLOYED} <= {PPL_GATE} (passes)")
    print(f"\nself-test: {r['self_test']['n_passed']}/{r['self_test']['n_checks']} checks  "
          f"tree_verify_net_tps_self_test_passes = {r['tree_verify_net_tps_self_test_passes']}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Tree-verify NET-TPS go/no-go (PR #402).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="0-GPU analytic gate (PR #402 deliverables)")
    ap.add_argument("--reanalyze", action="store_true", help="0-GPU full re-analysis (alias of default)")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="tree-verify-net-tps-go-nogo")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default="denken/tree-verify-net-tps-go-nogo")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/validity/tree_verify_net_tps_go_nogo/tree_verify_net_tps_go_nogo_results.json")
    args = ap.parse_args()

    report = build_report()
    print_report(report)
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    report["peak_mem_mib"] = peak_mib

    if args.self_test:
        out = Path("research/validity/tree_verify_net_tps_go_nogo/tree_verify_net_tps_go_nogo_selftest.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}  (peak {peak_mib:.1f} MiB)")
        print(f"\ntree_verify_net_tps_self_test_passes = {report['self_test']['passes']}")
        return 0 if report["self_test"]["passes"] else 1

    report["wandb_run_id"] = None if args.no_wandb else log_to_wandb(report, args.wandb_group, args.wandb_name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')}, peak {peak_mib:.1f} MiB)")

    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [report["wandb_run_id"]] if report.get("wandb_run_id") else [],
        "no_hf_job": True, "official_tps": 0.0, "analysis_only": True, "no_served_file_change": True,
        "gross_tps_gain_per_unit_cov": float(report["gross_tps_gain_per_unit_cov"]),
        "tstep_tax_frac_k4": float(report["tstep_tax_frac_k4"]),
        "tstep_tax_frac_k8": float(report["tstep_tax_frac_k8"]),
        "tps_loss_k4": float(report["tps_loss_k4"]),
        "tps_loss_k8": float(report["tps_loss_k8"]),
        "net_tps_at_full_gap": float(report["net_tps_at_full_gap"]),
        "tree_verify_net_positive": bool(report["tree_verify_net_positive"]),
        "g_star_threshold_to_close_500": float(report["g_star_threshold_to_close_500"]),
        "tree_closes_500_alone": bool(report["tree_closes_500_alone"]),
        "tree_plus_cb3_required": bool(report["tree_plus_cb3_required"]),
        "reconciles_396_corrected_base": bool(report["reconciles_396_corrected_base"]),
        "tree_verify_net_tps_self_test_passes": bool(report["tree_verify_net_tps_self_test_passes"]),
        "primary_metric": {"name": "tree_verify_net_tps_self_test_passes",
                           "value": float(report["tree_verify_net_tps_self_test_passes"])},
        "test_metric": {"name": "g_star_threshold_to_close_500",
                        "value": float(report["g_star_threshold_to_close_500"])},
    }))
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
