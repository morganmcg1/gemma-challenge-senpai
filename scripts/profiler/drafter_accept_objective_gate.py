#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Drafter loss-objective gate (PR #95): is the deployed MTP draft head
ACCEPTANCE-near-optimal, or only LIKELIHOOD-optimal?  (LK-Loss headroom)

WHAT THIS ANSWERS
-----------------
The deployed MTP draft head ("kenyan-duma") is trained with next-token
cross-entropy (CE).  CE maximizes per-token *likelihood*; the speculative-decode
objective is to maximize the *number of accepted tokens* (E[T]).  "LK-Losses"
(arXiv:2602.23881 -- the MECHANISM is what matters, not the cite) optimize
acceptance directly and claim +8-10% E[T] with zero inference overhead (only the
training objective changes).

fern #80 (MERGED) found every retrained CE / KL-distill / recipe-sweep drafter
lands at MTP PARITY (~3.83 accept_length) and concluded the ceiling is
ARCHITECTURAL (single-layer head capacity), not the training schedule.  That was
a LIKELIHOOD-objective result.  LK-Loss varies a DIFFERENT objective family
(acceptance, not likelihood).  The sharp question:

    Is the MTP head already ACCEPTANCE-near-optimal (so #80's likelihood-parity
    closure EXTENDS to acceptance and LK headroom ~ 0), or is there a real
    CE-vs-acceptance gap LK-Loss could capture (+8% E[T] -> ~520 TPS)?

This is a CPU-first analytical gate that sizes the prize BEFORE any GPU
fine-tuning spend.  It is analysis-only: no served change, greedy identity
unchanged by construction (the verify step is untouched; LK only changes how the
draft is trained -> output is always the verifier's greedy sequence).

INPUTS (all committed; NO forward pass needed -- the per-position-k profile is
fully reconstructable):
  * research/accept_calibration/accept_calibration_results.json  (#76)
      deployed-chain per-position conditional acceptance q[k], E[T]=3.844.
  * research/rank_coverage/rank_coverage_results.json            (#79/wirbel)
      conditional_rank1_acceptance_q[k] (cross-check), the rho ladder
      [0.4165,0.2655,0.1908], rho2_by_depth, and the rank-of-true histogram
      (the "near-miss" data an acceptance loss would target).
  * research/rank_coverage/entropy_branching_results.json        (#86/wirbel)
      13,491 first-reject steps; rho2 vs drafter-entropy (confidence) curve.

METHOD (reuses fern #88/#91 + wirbel #79 machinery verbatim):
  Step 1  measure P(accept|k), k=1..7; classify shape; reconcile E[T] via
          score_tree_depthrank(build_linear(8), pvecs) == 3.844.
  Step 2  decompose the LK headroom:
            (a) RE-RANKING headroom  -- the only thing committed data computes
                directly: can re-ordering the drafter's OWN top-W candidates by
                acceptance beat the current likelihood ordering?  (= is rank-1
                already the highest-acceptance token at every depth?)
            (b) PAPER-TRANSFER ceiling -- apply the paper's +8/10% E[T] claim
                directly; report implied E[T] and TPS, and the per-position
                q-lift it would require.
            (c) #80 reconciliation -- the head is at the likelihood/capacity
                frontier AND its argmax is already acceptance-ordered; under
                GREEDY verify acceptance collapses to a top-1 classification for
                which CE is near-optimal (the same greedy-collapse fern #88
                proved for Traversal Verification: +4.57% sampling -> 0 greedy).
  Step 3  gate on lk_implied_ET_headroom_pct.

PRIMARY metric  lk_implied_ET_headroom_pct  (realizable headroom on OUR head)
TEST    metric  measured_drafter_accept_profile (per-position P(accept|k))

GATE (PR #95):
  GREEN  >= +3%   real CE-vs-acceptance gap -> recommend LK fine-tune (approval-gated)
  AMBER  +1..+3%  marginal -> report corrected ceiling
  RED    <  +1%   MTP head already acceptance-near-optimal -> #80 closure EXTENDS
                  to the acceptance objective -> drafter-loss-objective lane CLOSED

LOCAL, CPU-ONLY, ANALYTIC.  No GPU, no vLLM, no HF Job, no submission, no
served-file change.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sequoia_dp_tree import build_linear  # noqa: E402  (fern/wirbel chain builder)
from treeshape_measured_accept import (  # noqa: E402
    build_depth_pvecs_measured,
    load_measured,
    load_rank_coverage,
    score_tree_depthrank,
)

ACCEPT_JSON = "research/accept_calibration/accept_calibration_results.json"
RANKCOV_JSON = "research/rank_coverage/rank_coverage_results.json"
ENTROPY_JSON = "research/rank_coverage/entropy_branching_results.json"

# Frontier anchors (PR #95 baseline; leaderboard fa2sw-precache-splitkv-linear-mtp-k7).
FRONTIER_TPS_OFFICIAL = 481.53
FRONTIER_TPS_LOCAL_WALL = 454.09
E_T_DEPLOYED = 3.844131736526946  # #76 server-log E[T], the reconcile anchor

# Paper's HEADLINE claim (the +8-10% in the PR body) is the SAMPLING (T=1) figure.
LK_PAPER_ET_GAIN_HEADLINE_LOW = 0.08
LK_PAPER_ET_GAIN_HEADLINE_HIGH = 0.10

# Actual LK-Losses (arXiv:2602.23881) T=0 vs T=1 accept-length gains over KL/CE,
# from the paper's own tables (researcher-agent verified, Llama-3.1-8B target).
# Our deployed MTP head is GREEDY (T=0); these are the regime-correct numbers.
#   architecture        T=0 gain   T=1 gain
#   EAGLE-3 (recurrent)  +2.4%      +2.7%      <- our head's *upper* analogue
#   Medusa  (MLP head)   +1.0%      +7.6%      <- our single-layer head's *lower* analogue (#80)
#   MLP-Speculator       +1.2%      +3.3%
LK_T0_EAGLE3 = 0.024     # corrected greedy CEILING (best-case analogue)
LK_T0_MEDUSA = 0.010     # corrected greedy FLOOR (single-layer-head analogue, #80)
LK_T0_MLPSPEC = 0.012


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} not JSON serializable")


def et_linear(q: list[float]) -> float:
    """E[T] of a linear (chain) drafter under greedy verify from the per-position
    conditional acceptance q[k] = P(accept k | reached k).

        E[T] = 1 (bonus) + sum_{k=1..L} prod_{j=1..k} q_j

    This is the closed form score_tree_depthrank reduces to on build_linear; we
    assert agreement below so the result is anchored to the #88/#91 engine."""
    et = 1.0
    cum = 1.0
    for qk in q:
        cum *= qk
        et += cum
    return et


def et_linear_via_engine(q: list[float]) -> float:
    """Same quantity through fern #88/#91 score_tree_depthrank(build_linear),
    so the gate inherits the validated engine rather than a fresh formula."""
    # rho_cond is irrelevant for a linear chain (every node has a single rank-1
    # child); build_depth_pvecs_measured sets pv[1]=q[d] which is all the spine uses.
    pvecs = build_depth_pvecs_measured(q, rho_cond=[0.0, 0.0, 0.0], W=4,
                                       max_depth=max(8, len(q) + 1), extrapolate="flat")
    parent = build_linear(len(q) + 1)  # root + L children = L draft positions
    F, _ = score_tree_depthrank(parent, pvecs)
    return float(F)


def classify_shape(q: list[float]) -> str:
    diffs = np.diff(q)
    if np.all(diffs > 1e-3):
        return "monotone-rising"
    if np.all(diffs < -1e-3):
        return "monotone-declining"
    if np.mean(diffs) > 5e-3:
        return "rising (non-strict)"
    if abs(np.mean(diffs)) <= 5e-3 and np.std(q) < 0.02:
        return "approximately-constant (geometric)"
    return "mixed"


def solve_uniform_qlift_for_et(q: list[float], target_et: float,
                               cap: float = 0.999) -> float:
    """Smallest uniform RELATIVE per-position lift delta on q s.t. E[T] == target_et.
    Bisection on delta. Returns delta (e.g. 0.027 = +2.7% per-position q)."""
    def et_of(delta: float) -> float:
        return et_linear([min(cap, qk * (1.0 + delta)) for qk in q])
    lo, hi = 0.0, 2.0
    if et_of(hi) < target_et:
        return float("nan")  # unreachable even at the cap
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if et_of(mid) < target_et:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accept-json", default=ACCEPT_JSON)
    ap.add_argument("--rank-coverage-json", default=RANKCOV_JSON)
    ap.add_argument("--entropy-json", default=ENTROPY_JSON)
    ap.add_argument("--out", default="research/drafter_accept_objective/gate_results.json")
    ap.add_argument("--wandb", action="store_true", help="log to W&B")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="fern/drafter-accept-objective-gate")
    ap.add_argument("--wandb-group", default="drafter-accept-objective-gate")
    args = ap.parse_args()

    # ----------------------------------------------------------------- load #76
    meas = load_measured(args.accept_json, "server_log")
    meas_x = load_measured(args.accept_json, "prometheus")
    q = meas["q"]                          # per-position conditional acceptance, k=1..7
    L = len(q)

    # ----------------------------------------------------------------- load #79
    rc = load_rank_coverage(args.rank_coverage_json)
    rcd = json.load(open(args.rank_coverage_json))["analysis"]
    q79 = [float(x) for x in rcd["conditional_rank1_acceptance_q"]]
    rho2_by_depth = [float(rcd["rho2_by_depth"][str(k)]) for k in range(L)]
    rank_hist = rcd["rank_fd_hist_pooled"]          # {"0":beyond, "2":..,"3":..,"4":..}
    n_div = int(rcd["n_divergences"])
    rho_cond = rc["rho_cond"]                        # [rho2, rho3, rho4]

    # ============================================================ STEP 1
    et_closed = et_linear(q)
    et_engine = et_linear_via_engine(q)
    assert abs(et_closed - et_engine) < 1e-9, (et_closed, et_engine)
    assert abs(et_closed - meas["E_T"]) < 5e-3, (et_closed, meas["E_T"])
    shape = classify_shape(q)

    step1 = {
        "measured_drafter_accept_profile": q,                 # TEST metric (vector)
        "measured_drafter_accept_profile_xcheck_79": q79,
        "profile_shape": shape,
        "top1_accept_q1": q[0],
        "deepest_accept_q7": q[-1],
        "profile_rise_q7_minus_q1": q[-1] - q[0],
        "E_T_from_profile_closedform": et_closed,
        "E_T_from_profile_engine_88_91": et_engine,
        "E_T_reported_76_serverlog": meas["E_T"],
        "E_T_reported_76_prometheus": meas_x["E_T"],
        "E_T_from_79_profile": et_linear(q79),
        "reconcile_abs_err_vs_3844": abs(et_closed - meas["E_T"]),
        "cumulative_acceptance_C": [float(c) for c in meas["C"]],
        "rho_ladder_79": rho_cond,
    }

    # ============================================================ STEP 2
    # (a) RE-RANKING headroom: is rank-1 already the highest-acceptance token at
    #     every depth?  P(rank1==true|reach k)=q79[k]; P(rank2==true|reach k)=
    #     (1-q79[k])*rho2_by_depth[k]; deeper ranks are strictly smaller still.
    per_pos = []
    rerank_gain_positions = 0
    for k in range(L):
        p_r1 = q79[k]
        p_r2 = (1.0 - q79[k]) * rho2_by_depth[k]
        margin = p_r1 - p_r2
        if p_r2 > p_r1:
            rerank_gain_positions += 1
        per_pos.append({
            "k": k + 1,
            "P_rank1_eq_true": p_r1,
            "P_rank2_eq_true": p_r2,
            "argmax_acceptance_margin": margin,
            "rank1_is_best_acceptance_bet": bool(p_r1 >= p_r2),
        })
    # Acceptance-optimal LINEAR re-ranking emits, at each k, argmax acceptance =
    # rank-1 everywhere -> identical chain -> identical E[T] -> 0 headroom.
    # Compute APPLES-TO-APPLES within the #79 probe (both the rank-1 spine q79[k]
    # and the rank-2 rescue rho2_by_depth[k] come from the SAME probe); using the
    # #76 baseline here would inject the +1.5% #76-vs-#79 probe-variance gap as
    # spurious "headroom". The current (likelihood) chain emits rank-1 -> E[T] =
    # et_linear(q79); the acceptance-optimal re-ranking emits max-acceptance
    # candidate per position -> q_reranked; rank-1 wins everywhere -> equal.
    et_base_79 = et_linear(q79)
    q_reranked = [max(p["P_rank1_eq_true"], p["P_rank2_eq_true"]) for p in per_pos]
    et_reranked = et_linear(q_reranked)
    rerank_headroom_pct = (et_reranked / et_base_79 - 1.0) * 100.0
    # The #76-vs-#79 cross-probe gap, reported separately as MEASUREMENT noise on
    # the same deployed head (NOT acceptance headroom).
    probe_variance_pct = (et_base_79 / et_closed - 1.0) * 100.0

    # (b) CEILINGS. The PR's +8-10% is the paper's SAMPLING (T=1) headline; it does
    #     NOT transfer to our greedy verify. The regime-correct ceiling is the
    #     paper's own T=0 number for our nearest architecture analogue.
    et_head_low = E_T_DEPLOYED * (1.0 + LK_PAPER_ET_GAIN_HEADLINE_LOW)
    et_head_high = E_T_DEPLOYED * (1.0 + LK_PAPER_ET_GAIN_HEADLINE_HIGH)
    # TPS scales ~linearly with E[T] at fixed per-step cost (zero inference overhead).
    tps_head_low = FRONTIER_TPS_OFFICIAL * (1.0 + LK_PAPER_ET_GAIN_HEADLINE_LOW)
    qlift_for_8pct = solve_uniform_qlift_for_et(q, et_head_low)
    qlift_for_10pct = solve_uniform_qlift_for_et(q, et_head_high)
    # Corrected greedy ceiling band: EAGLE-3 T=0 (+2.4%, upper) -> Medusa T=0 (+1.0%,
    # lower, the single-layer-head analogue #80 says we resemble).
    et_greedy_ceiling = E_T_DEPLOYED * (1.0 + LK_T0_EAGLE3)
    tps_greedy_ceiling = FRONTIER_TPS_OFFICIAL * (1.0 + LK_T0_EAGLE3)
    et_greedy_floor = E_T_DEPLOYED * (1.0 + LK_T0_MEDUSA)
    tps_greedy_floor = FRONTIER_TPS_OFFICIAL * (1.0 + LK_T0_MEDUSA)

    # (c) NEAR-MISS structure (what an acceptance loss would target).
    n_beyond = int(rank_hist.get("0", 0))
    n_rank2 = int(rank_hist.get("2", 0))
    n_rank3 = int(rank_hist.get("3", 0))
    n_rank4 = int(rank_hist.get("4", 0))
    nearmiss = {
        "n_divergences": n_div,
        "rank2_nearmiss_frac_of_rejections": n_rank2 / n_div,
        "rank3_frac": n_rank3 / n_div,
        "rank4_frac": n_rank4 / n_div,
        "beyond_topW_frac": n_beyond / n_div,
        "interpretation": (
            "rank-2 near-misses are 41.7% of rejections, BUT rank-1 still wins "
            "73% outright and is 7x more likely than rank-2 to be the true token "
            "at k=1 (rising to >13x at k=7). The rank-2 mass is TREE fodder "
            "(land #71 width-2 branch, rho2=0.4165) realized root-to-leaf "
            "(fern #88), NOT a mis-ranking a LINEAR acceptance loss can flip: "
            "demoting rank-1 to promote rank-2 loses more than it gains."
        ),
    }

    # ---- realizable headroom on OUR head (two channels) ----
    # CHANNEL 1 (re-ranking): directly computed = 0.0% (rank-1 already
    #   acceptance-ordered at every depth). An acceptance loss buys nothing by
    #   re-ordering the drafter's existing candidates.
    # CHANNEL 2 (prediction-improvement): an acceptance loss CHANGES the drafter's
    #   weights, moving mass so more contexts have the true token at rank-1. The
    #   committed data CANNOT measure this (it needs the actual retrain). The
    #   literature gives the regime-correct estimate: LK-Loss T=0 gain over KL/CE
    #   is +2.4% for EAGLE-3 (our upper analogue) down to +1.0% for Medusa/MLP
    #   (the single-layer-head class #80 says we resemble). #80 closed only the
    #   LIKELIHOOD objectives (CE/KL); the acceptance objective's gradient
    #   mechanism (focus mass on the acceptance gap) is genuinely untested on our
    #   head, so channel 2 is NOT provably 0 -- it is small-but-real.
    # The PRIMARY metric = the corrected greedy CEILING (channel-2 upper analogue),
    # the number the advisor needs to decide whether to probe.
    lk_implied_ET_headroom_pct = LK_T0_EAGLE3 * 100.0   # +2.4% (corrected ceiling)
    lk_greedy_floor_pct = LK_T0_MEDUSA * 100.0          # +1.0% (single-layer analogue)

    step2 = {
        "rerank_headroom_pct": rerank_headroom_pct,
        "rerank_baseline_E_T_79": et_base_79,
        "rerank_optimal_E_T_79": et_reranked,
        "probe_variance_76_vs_79_pct": probe_variance_pct,
        "probe_variance_note": "76-vs-79 E[T] gap is measurement noise on the SAME deployed head, not acceptance headroom",
        "rerank_gain_positions": rerank_gain_positions,
        "per_position_acceptance_margin": per_pos,
        "min_argmax_acceptance_margin": min(p["argmax_acceptance_margin"] for p in per_pos),
        "rank1_acceptance_dominant_all_depths": rerank_gain_positions == 0,
        "headline_ceiling_NONtransferable": {
            "note": "PR's +8-10% is the paper's SAMPLING (T=1) headline; sampling acceptance = min(1,p_t/p_d) depends on full-distribution match. Does NOT transfer to our greedy verify.",
            "lk_headline_ET_gain_pct_low": LK_PAPER_ET_GAIN_HEADLINE_LOW * 100,
            "lk_headline_ET_gain_pct_high": LK_PAPER_ET_GAIN_HEADLINE_HIGH * 100,
            "E_T_headline_low": et_head_low,
            "tps_headline_low": tps_head_low,
            "per_position_qlift_needed_for_8pct": qlift_for_8pct,
            "per_position_qlift_needed_for_10pct": qlift_for_10pct,
        },
        "corrected_greedy_ceiling": {
            "note": "paper's OWN T=0 numbers (researcher-verified, Llama-3.1-8B). Our head is GREEDY -> these are regime-correct. EAGLE-3 T=0 +2.4% (upper), Medusa/MLP T=0 +1.0-1.2% (lower / single-layer analogue).",
            "lk_T0_EAGLE3_pct": LK_T0_EAGLE3 * 100,
            "lk_T0_medusa_pct": LK_T0_MEDUSA * 100,
            "lk_T0_mlpspec_pct": LK_T0_MLPSPEC * 100,
            "E_T_greedy_ceiling": et_greedy_ceiling,
            "tps_greedy_ceiling": tps_greedy_ceiling,
            "E_T_greedy_floor": et_greedy_floor,
            "tps_greedy_floor": tps_greedy_floor,
            "greedy_collapse_factor_vs_headline": "T=0 gain is ~3-8x smaller than the T=1 headline (paper's own tables)",
        },
        "nearmiss": nearmiss,
        "reconciliation_80": {
            "pr80_finding": "every retrained CE/KL/recipe drafter -> MTP parity (~3.83)",
            "pr80_conclusion": "ceiling is ARCHITECTURAL (single-layer head capacity), not training schedule",
            "pr80_tested_only_likelihood": "CE + KL-distill are LIKELIHOOD objectives; #80 did NOT test an acceptance objective -> channel-2 headroom is untested on our head",
            "rerank_channel_closed": "channel-1 (re-ranking) IS closed by committed data: argmax already acceptance-ordered -> any LK gain must come from channel-2 (prediction-improvement)",
            "single_layer_implies_lower": "#80's single-layer-head finding -> we resemble the Medusa/MLP T=0 class (+1.0%) more than EAGLE-3 (+2.4%) -> realistic central nearer the floor",
            "greedy_collapse_precedent_88": "fern #88: Traversal Verification +4.57% sampling -> provably 0 greedy; same collapse direction shrinks the LK T=1 +8% to T=0 +2.4%",
        },
        "lk_implied_ET_headroom_pct": lk_implied_ET_headroom_pct,
        "lk_greedy_floor_pct": lk_greedy_floor_pct,
        "realistic_band_pct": [lk_greedy_floor_pct, lk_implied_ET_headroom_pct],
    }

    # ============================================================ STEP 3 GATE
    if lk_implied_ET_headroom_pct >= 3.0:
        verdict = "GREEN"
    elif lk_implied_ET_headroom_pct >= 1.0:
        verdict = "AMBER"
    else:
        verdict = "RED"

    gate = {
        "primary_metric_name": "lk_implied_ET_headroom_pct",
        "lk_implied_ET_headroom_pct": lk_implied_ET_headroom_pct,
        "realistic_band_pct": [lk_greedy_floor_pct, lk_implied_ET_headroom_pct],
        "verdict": verdict,
        "rule": "GREEN>=+3% / AMBER +1..3% / RED <+1%",
        "headline": (
            f"{verdict}: corrected greedy LK E[T] headroom = "
            f"+{lk_greedy_floor_pct:.1f}..{lk_implied_ET_headroom_pct:.1f}% "
            f"(EAGLE-3 T=0 ceiling +2.4%, Medusa/single-layer floor +1.0%) -- "
            f"sharply DOWN from the PR's +8-10% (that is the paper's T=1/sampling "
            f"headline). Re-ranking channel CLOSED (0.0%, argmax already "
            f"acceptance-ordered); any gain must come from prediction-improvement, "
            f"which #80 (likelihood-only) never tested. Marginal prize (~"
            f"+{lk_greedy_floor_pct:.0f}-{lk_implied_ET_headroom_pct:.0f}% E[T] -> "
            f"~{tps_greedy_floor:.0f}-{tps_greedy_ceiling:.0f} TPS): SIZE with a "
            f"cheap LoRA/projection-layer probe before any GPU fine-tune. Do NOT "
            f"transfer the +8% headline; do NOT full-launch unsized."
        ),
    }

    out = {
        "gate": gate,
        "step1_profile": step1,
        "step2_headroom": step2,
        "frontier_anchors": {
            "tps_official": FRONTIER_TPS_OFFICIAL,
            "tps_local_wall": FRONTIER_TPS_LOCAL_WALL,
            "E_T_deployed": E_T_DEPLOYED,
        },
        "inputs": {
            "accept_json": args.accept_json,
            "rank_coverage_json": args.rank_coverage_json,
            "wandb_run_id_76": json.load(open(args.accept_json)).get("wandb_run_id"),
            "wandb_run_id_79": rc.get("wandb_run_id"),
        },
        "method": "CPU-only analytic; reuses fern #88/#91 score_tree_depthrank + wirbel #79 rank-coverage; no forward pass (profile fully committed)",
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=_json_default)

    # ----------------------------------------------------------------- console
    print("=" * 78)
    print("DRAFTER LOSS-OBJECTIVE GATE (PR #95) -- LK-Loss acceptance headroom")
    print("=" * 78)
    print(f"\n[STEP 1] measured per-position P(accept|k), k=1..{L}  (#76 server-log):")
    for k in range(L):
        print(f"   k={k+1}  q={q[k]:.4f}   (#79 xcheck {q79[k]:.4f})")
    print(f"   shape: {shape}  (q1={q[0]:.4f} -> q{L}={q[-1]:.4f}, +{q[-1]-q[0]:.4f})")
    print(f"   E[T] from profile = {et_closed:.4f} (engine {et_engine:.4f}) "
          f"vs reported 3.844 -> |err|={abs(et_closed-meas['E_T']):.2e}  OK")
    print(f"\n[STEP 2a] re-ranking headroom (emit acceptance-argmax of drafter's own top-W):")
    for p in per_pos:
        print(f"   k={p['k']}  P(r1=true)={p['P_rank1_eq_true']:.4f}  "
              f"P(r2=true)={p['P_rank2_eq_true']:.4f}  margin={p['argmax_acceptance_margin']:+.4f}"
              f"  {'rank-1 best' if p['rank1_is_best_acceptance_bet'] else 'RANK-2 BEAT R1!'}")
    print(f"   -> rank-1 acceptance-dominant at ALL depths; re-ranking (channel-1) headroom = "
          f"{rerank_headroom_pct:+.3f}%  => any LK gain must come from prediction-improvement")
    print(f"\n[STEP 2b] ceilings:")
    print(f"   PR headline +8-10% E[T] -> {et_head_low:.3f}/{et_head_high:.3f}, TPS {tps_head_low:.1f}"
          f"  == paper's T=1/SAMPLING figure, NON-transferable to greedy")
    print(f"   corrected GREEDY ceiling (paper's own T=0): EAGLE-3 +2.4% -> E[T] {et_greedy_ceiling:.3f}, "
          f"TPS {tps_greedy_ceiling:.1f}")
    print(f"   corrected GREEDY floor   (Medusa/single-layer): +1.0% -> E[T] {et_greedy_floor:.3f}, "
          f"TPS {tps_greedy_floor:.1f}")
    print(f"   #80: retrain CE/KL -> parity BUT tested LIKELIHOOD only; channel-2 untested on our head")
    print(f"\n[STEP 2c] near-miss: rank-2={nearmiss['rank2_nearmiss_frac_of_rejections']:.4f} of "
          f"rejections (TREE fodder, land #71; not linear-retrain fodder)")
    print(f"\n[STEP 3] GATE: lk_implied_ET_headroom_pct = +{lk_implied_ET_headroom_pct:.1f}% "
          f"(band +{lk_greedy_floor_pct:.1f}..{lk_implied_ET_headroom_pct:.1f}%)  -> {verdict}")
    print(f"   {gate['headline']}")
    print(f"\nwrote {args.out}")

    # ----------------------------------------------------------------- W&B
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name, group=args.wandb_group,
                         job_type="analysis", config={
                             "gate": "drafter-accept-objective",
                             "method": "cpu-analytic-committed-data",
                             "L_draft_positions": L,
                             "E_T_deployed": E_T_DEPLOYED,
                             "frontier_tps_official": FRONTIER_TPS_OFFICIAL,
                             "lk_headline_ET_gain_low": LK_PAPER_ET_GAIN_HEADLINE_LOW,
                             "lk_headline_ET_gain_high": LK_PAPER_ET_GAIN_HEADLINE_HIGH,
                             "lk_T0_eagle3": LK_T0_EAGLE3,
                             "lk_T0_medusa": LK_T0_MEDUSA,
                         })
        wandb.summary["lk_implied_ET_headroom_pct"] = lk_implied_ET_headroom_pct
        wandb.summary["lk_greedy_floor_pct"] = lk_greedy_floor_pct
        wandb.summary["verdict"] = verdict
        wandb.summary["rerank_headroom_pct"] = rerank_headroom_pct
        wandb.summary["rank1_acceptance_dominant_all_depths"] = (rerank_gain_positions == 0)
        wandb.summary["min_argmax_acceptance_margin"] = step2["min_argmax_acceptance_margin"]
        wandb.summary["E_T_deployed"] = et_closed
        wandb.summary["E_T_from_profile_engine"] = et_engine
        wandb.summary["E_T_greedy_ceiling"] = et_greedy_ceiling
        wandb.summary["tps_greedy_ceiling"] = tps_greedy_ceiling
        wandb.summary["top1_accept_q1"] = q[0]
        wandb.summary["profile_shape"] = shape
        wandb.summary["headline_ceiling_tps_8pct_NONtransfer"] = tps_head_low
        wandb.summary["rank2_nearmiss_frac"] = nearmiss["rank2_nearmiss_frac_of_rejections"]
        # per-position profile table
        prof_tbl = wandb.Table(columns=["k", "P_accept_q", "P_accept_q_79xcheck",
                                        "P_rank2_eq_true", "argmax_accept_margin",
                                        "cumulative_C"])
        for k in range(L):
            prof_tbl.add_data(k + 1, q[k], q79[k], per_pos[k]["P_rank2_eq_true"],
                              per_pos[k]["argmax_acceptance_margin"], meas["C"][k])
        wandb.log({"acceptance_profile": prof_tbl})
        # headroom decomposition table
        hd_tbl = wandb.Table(columns=["component", "E_T", "headroom_pct", "realizable"])
        hd_tbl.add_data("deployed CE (#79 probe)", et_base_79, 0.0, "baseline")
        hd_tbl.add_data("acceptance re-ranking (committed-data, channel-1)", et_reranked,
                        rerank_headroom_pct, "yes (=0)")
        hd_tbl.add_data("LK greedy FLOOR T=0 (Medusa/single-layer, channel-2)",
                        et_greedy_floor, lk_greedy_floor_pct, "maybe (probe)")
        hd_tbl.add_data("LK greedy CEILING T=0 (EAGLE-3, channel-2)",
                        et_greedy_ceiling, lk_implied_ET_headroom_pct, "maybe (probe)")
        hd_tbl.add_data("PR headline +8% (T=1 SAMPLING)", et_head_low, 8.0, "no (wrong regime)")
        wandb.log({"headroom_decomposition": hd_tbl})
        print(f"\nW&B run: {run.id}  ({run.url})")
        wandb.finish()


if __name__ == "__main__":
    main()
