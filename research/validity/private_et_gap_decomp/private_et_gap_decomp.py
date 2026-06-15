#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Public->private E[T] gap decomposition for the trained MTP K=7 draft (PR #258, student ubel).

WHAT THIS LEG ANSWERS
---------------------
The deployed linear-MTP K=7 drafter accepts E[T]=3.844 tokens/step on the OFFICIAL public
ShareGPT eval set (accept_calibration, run 5m17r52s) but only E[T]=3.090 on the PRIVATE
ShareGPT-chat proxy (tree_private_acceptance_gap, run ytxfi6zk). That is a gap of 0.754 E[T].
This DIAGNOSTIC decomposes that gap and returns a single verdict:

    is the gap RECOVERABLE (a greedy-safe calibration / temperature / predicate mismatch -- the
    draft is merely OVER-CONFIDENT and a cheap knob restores E[T]) or INTRINSIC (a genuine
    draft predictive weakness out-of-distribution that no greedy-safe knob can touch)?

We decompose along THREE axes and reconcile, then price any lever.

AXIS 1 -- PER-POSITION (decode protocol; the PR's headline anchors 3.844 / 3.090).
    E[T] = 1 + sum_{d=1..7} C_d, where C_d = prod_{j<=d} p_j is the cumulative acceptance and
    p_j the per-depth CONDITIONAL acceptance (accept iff draft_argmax_j == target_greedy_j). The
    gap is therefore EXACTLY additive over depth: gap = sum_d (C_d^pub - C_d^priv). We report both
    the cumulative contribution gap_d and the conditional drop dp_d = p_d^pub - p_d^priv. The
    conditional drop is concentrated at the SHALLOW positions (d1 dominates) and goes NEGATIVE at
    the deep tail (private d5..d7 conditionals are HIGHER than public) -- the multiplicative
    compounding of the early loss is what spreads the cumulative gap across all depths.

AXIS 2 -- PER-CATEGORY (sglang vllm-chat-scored protocol; the only per-content-axis-resolved
    cache, public anchor E[T]=4.124, #176 native 6-axis proxies). Each of the 6 hard tails
    (code / casual / sharegpt / multilingual / math / longctx) has its own conditional ladder. We
    report each category's E[T] and its gap vs the 4.124 public anchor, and a COUNT-POOLED
    aggregate. The per-category contributions to each depth's pooled conditional drop are EXACTLY
    additive (count weights w_c = n_c / sum n_c): sum_c w_c (p_pub[d] - p_c[d]) = p_pub[d] -
    p_pooled[d]. The signature is UNIVERSAL: every category's depth-1 top-1 match q0 drops below
    public (0.758 -> 0.64..0.69) -- a discrimination loss in every content slice, not a single
    bad tail.

AXIS 3 -- DRAFT-CONFIDENCE-vs-ACCEPTANCE (the DECISIVE axis). The PR asks: bin draft-proposed
    tokens by draft probability and compare acceptance per bin public vs private; over-confidence
    (high-prob bins drop) => RECOVERABLE, mass-shift-down => INTRINSIC. We answer this with a
    THEOREM plus the cached public rank-coverage structure (run z6wi4z4v):

      * ARGMAX-INVARIANCE. Under width-1 greedy-EXACT verify the accept decision is
        1[argmax(draft_logits) == argmax(target_logits)], which is INVARIANT to every strictly
        monotone transform of the draft logits -- temperature (argmax(z/T)=argmax(z) for T>0),
        any probability THRESHOLD, any confidence re-scaling. Hence NO calibration / temperature /
        predicate knob can flip a single reject to an accept; greedy-safe predicates are strictly
        ONE-SIDED (they can only REJECT more). The calibration-recoverable share of the gap is
        therefore EXACTLY 0, independent of what the confidence bins would show -- acceptance does
        not depend on the draft's probability VALUE, only on its argmax.

      * So the gap is a DISCRIMINATION loss: the draft's top-1 token is simply WRONG more often
        OOD (q0 0.729 -> 0.599). The ONLY greedy-safe lever that can raise E[T] is WIDTH>1
        (tree / multi-candidate): when draft_argmax != target but the true token sits at draft
        rank 2..W, a tree proposes it and the verifier accepts the rank-2+ match. That is a
        SEPARATE mechanism (adding lanes, not re-calibrating) and is already priced in
        tree_private_acceptance_gap (private tree E[T] 4.38 raw -> 4.92 calibrated). We report the
        PUBLIC rank-2+ coverage (cumulative_coverage[W=4]=0.653 of divergences recoverable) as the
        size of that separate lever; the private analog needs one local GPU rank-probe forward
        (priced as a ready follow-up, NOT required for the verdict).

VERDICT  recoverable_fraction (calibration/temperature/predicate) = 0.0  =>  gap_verdict =
    INTRINSIC-WEAKNESS. A clean NEGATIVE: there is no free greedy-safe knob; the gap is genuine
    OOD discrimination weakness, and the only greedy-safe E[T] lever (width>1 tree) is a separate
    already-priced lane.

LOCAL CPU-only analytic decomposition over banked cached acceptance ladders. No GPU / vLLM / HF
Job / submission / served-file change / draft retrain. All ladders are imported UNCHANGED.
BASELINE stays 481.53. Greedy/PPL untouched. PRIMARY = self-test (adds 0 TPS).

PRIMARY metric  private_et_gap_decomp_self_test_passes
TEST    metric  recoverable_fraction  (calibration-recoverable share of the 0.754 E[T] gap)

Run:
    CUDA_VISIBLE_DEVICES="" python research/validity/private_et_gap_decomp/\
private_et_gap_decomp.py --self-test \\
        --wandb_group private-et-gap-decomp --wandb_name ubel/private-et-gap-decomp
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

# --- cached sources (imported UNCHANGED; this leg re-derives nothing) --------------------------
_PUB_DECODE_JSON = REPO_ROOT / "research/accept_calibration/accept_calibration_results.json"
_PRIV_DECODE_JSON = REPO_ROOT / "research/validity/tree_private_acceptance_gap/results.json"
_SGLANG_6AXIS_JSON = REPO_ROOT / "research/validity/private_adverse_skew/proxies_native_6axis.json"
_RANKCOV_JSON = REPO_ROOT / "research/rank_coverage/rank_coverage_results.json"

K_MTP = 7                       # speculative depth of the deployed linear-MTP draft
PUB_ET_ANCHOR = 3.844           # PR headline public anchor (accept_calibration server E[T])
PRIV_ET_ANCHOR = 3.090          # PR headline private anchor (ShareGPT-chat proxy)
ANCHOR_TOL = 1e-3               # headline "3.844 / 3.09" rounding tolerance
EXACT_TOL = 1e-9                # ladder-reconstruction / additive-residual tolerance

CATS = ["native_code", "native_casual", "native_sharegpt",
        "native_multilingual", "native_math", "native_longctx"]


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def ladder(conditional_p: list[float]) -> tuple[list[float], float]:
    """cumulative C_d = prod_{j<=d} p_j ;  E[T] = 1 + sum_d C_d."""
    cum: list[float] = []
    acc = 1.0
    for p in conditional_p:
        acc *= float(p)
        cum.append(acc)
    return cum, 1.0 + sum(cum)


# --------------------------------------------------------------------------- #
# Imports.
# --------------------------------------------------------------------------- #
def load_imports() -> dict[str, Any]:
    pub = _load(_PUB_DECODE_JSON)
    priv = _load(_PRIV_DECODE_JSON)
    sg = _load(_SGLANG_6AXIS_JSON)
    rc = _load(_RANKCOV_JSON)

    pub_cond = list(pub["server_log_metrics"]["conditional_acceptance_p"])
    pub_et_server = float(pub["server_log_metrics"]["mean_tokens_per_step_E_T"])
    priv_cond = list(priv["private_ladder"]["q_conditional_raw"])
    priv_et_cached = float(priv["private_ladder"]["linear_E_T_raw"])
    pub_et_cached_ladder = float(priv["public_ladder"]["linear_E_T"])  # tree_private public ladder

    sg_pub = list(sg["q_pub_sglang"]["conditional_p_sglang"])
    sg_cats: dict[str, dict[str, Any]] = {}
    for p in sg["proxies"]:
        sg_cats[p["name"]] = {
            "conditional_p": list(p["component"]["conditional_p_sglang"]),
            "num_drafts": float(p["component"]["num_drafts"]),
            "axis": p["axis"],
        }

    rcA = rc["analysis"]
    return {
        "pub_decode_conditional_p": pub_cond,
        "pub_decode_et_server": pub_et_server,            # 3.844131736526946 (accept_calibration)
        "pub_decode_et_cached_ladder": pub_et_cached_ladder,  # 3.8444537125748504 (tree_private)
        "priv_decode_conditional_p": priv_cond,
        "priv_decode_et_cached": priv_et_cached,          # 3.0898055282313597
        "pub_decode_run": str(pub.get("wandb_run_id")),
        "priv_decode_run": "ytxfi6zk",
        "sglang_pub_conditional_p": sg_pub,
        "sglang_cats": sg_cats,
        "rankcov": {
            "W": int(rcA["W"]),
            "n_divergences": int(rcA["n_divergences"]),
            "rho_marginal": {int(k): float(v) for k, v in rcA["rho_marginal"].items()},
            "cumulative_coverage": {int(k): float(v) for k, v in rcA["cumulative_coverage"].items()},
            "frac_true_beyond_topW": float(rcA["frac_true_beyond_topW"]),
            "rank_fd_hist_pooled": {str(k): int(v) for k, v in rcA["rank_fd_hist_pooled"].items()},
            "conditional_rank1_acceptance_q": list(rcA["conditional_rank1_acceptance_q"]),
            "run": str(rc.get("wandb_run_id")),
        },
        "provenance": (
            "PUBLIC decode ladder accept_calibration 5m17r52s (E[T]=3.844, ShareGPT eval) x PRIVATE "
            "decode ladder tree_private_acceptance_gap ytxfi6zk (E[T]=3.090, ShareGPT-chat proxy) x "
            "PER-CATEGORY sglang #176 native 6-axis proxies (public anchor 4.124) x PUBLIC rank "
            "coverage z6wi4z4v (rho ladder, beyond-topW=0.347). All ladders imported UNCHANGED."),
    }


# --------------------------------------------------------------------------- #
# AXIS 1 -- per-position (decode protocol). EXACT additive over depth.
# --------------------------------------------------------------------------- #
def axis1_per_position(imp: dict) -> dict[str, Any]:
    pub_p = imp["pub_decode_conditional_p"]
    priv_p = imp["priv_decode_conditional_p"]
    pub_C, pub_ET = ladder(pub_p)
    priv_C, priv_ET = ladder(priv_p)
    total_gap = pub_ET - priv_ET

    rows = []
    for d in range(K_MTP):
        gap_d = pub_C[d] - priv_C[d]          # cumulative contribution to E[T] gap
        dp_d = pub_p[d] - priv_p[d]           # conditional acceptance drop at depth d
        rows.append({
            "depth": d + 1,
            "p_pub": pub_p[d],
            "p_priv": priv_p[d],
            "conditional_drop": dp_d,
            "C_pub": pub_C[d],
            "C_priv": priv_C[d],
            "gap_contrib": gap_d,
            "gap_frac": gap_d / total_gap if total_gap else float("nan"),
        })
    recon = sum(r["gap_contrib"] for r in rows)
    # depth-1 share + sign structure of the conditional drop (the discrimination signature).
    d1_cond_drop = rows[0]["conditional_drop"]
    deep_improves = all(rows[d]["conditional_drop"] < 0 for d in (4, 5, 6))
    return {
        "protocol": "decode (accept_calibration / tree_private; PR headline anchors)",
        "pub_ET": pub_ET, "priv_ET": priv_ET, "total_gap": total_gap,
        "rows": rows,
        "reconstruction_sum": recon,
        "reconstruction_residual": abs(recon - total_gap),
        "d1_conditional_drop": d1_cond_drop,
        "deep_tail_conditionals_improve_on_private": bool(deep_improves),
        "note": ("conditional drop peaks at d1 ({:.4f}) and goes NEGATIVE at the deep tail (private "
                 "d5..d7 conditionals exceed public) -- the gap is an EARLY top-1 discrimination "
                 "loss compounded multiplicatively, not a deep-position failure."
                 .format(d1_cond_drop)),
    }


# --------------------------------------------------------------------------- #
# AXIS 2 -- per-category (sglang protocol). EXACT additive at the conditional-ladder level.
# --------------------------------------------------------------------------- #
def axis2_per_category(imp: dict) -> dict[str, Any]:
    pub_p = imp["sglang_pub_conditional_p"]
    pub_C, pub_ET = ladder(pub_p)
    cats = imp["sglang_cats"]

    N = sum(cats[c]["num_drafts"] for c in CATS)
    weights = {c: cats[c]["num_drafts"] / N for c in CATS}

    per_cat = []
    for c in CATS:
        cp = cats[c]["conditional_p"]
        _, et_c = ladder(cp)
        per_cat.append({
            "category": c,
            "axis": cats[c]["axis"],
            "weight": weights[c],
            "q0_top1": cp[0],
            "q0_drop_vs_pub": pub_p[0] - cp[0],
            "E_T": et_c,
            "gap_vs_pub": pub_ET - et_c,
        })

    # count-pooled aggregate conditional ladder + EXACT additive per-depth attribution.
    pooled_p = [sum(weights[c] * cats[c]["conditional_p"][d] for c in CATS) for d in range(K_MTP)]
    pooled_C, pooled_ET = ladder(pooled_p)
    per_depth_attrib = []
    max_resid = 0.0
    for d in range(K_MTP):
        contribs = {c: weights[c] * (pub_p[d] - cats[c]["conditional_p"][d]) for c in CATS}
        pooled_drop = pub_p[d] - pooled_p[d]
        resid = abs(sum(contribs.values()) - pooled_drop)
        max_resid = max(max_resid, resid)
        per_depth_attrib.append({
            "depth": d + 1,
            "pooled_conditional_drop": pooled_drop,
            "per_category_contrib": contribs,
            "additive_residual": resid,
        })

    universal_q0_drop = all(pc["q0_drop_vs_pub"] > 0 for pc in per_cat)
    return {
        "protocol": "sglang vllm-chat-scored (#176 native 6-axis; public anchor 4.124)",
        "pub_ET": pub_ET,
        "pooled_ET": pooled_ET,
        "pooled_gap_vs_pub": pub_ET - pooled_ET,
        "per_category": per_cat,
        "per_depth_additive_attribution": per_depth_attrib,
        "max_additive_residual": max_resid,
        "universal_q0_discrimination_drop": bool(universal_q0_drop),
        "note": ("every one of the 6 content categories shows a depth-1 top-1 match (q0) DROP below "
                 "the 0.758 public anchor (0.64..0.69) -- the discrimination loss is universal "
                 "across content, not a single bad tail. The count-weighted per-category "
                 "conditional drops reconstruct the pooled drop EXACTLY at every depth "
                 "(max residual {:.2e}).".format(max_resid)),
    }


# --------------------------------------------------------------------------- #
# AXIS 3 -- draft-confidence-vs-acceptance (DECISIVE). Argmax-invariance => recoverable=0.
# --------------------------------------------------------------------------- #
def axis3_confidence_rank(imp: dict) -> dict[str, Any]:
    rc = imp["rankcov"]
    W = rc["W"]
    cumcov = rc["cumulative_coverage"]

    # PUBLIC confidence-vs-recovery curve: P(true token at rank r | draft argmax wrong).
    # well-formed = bins populated, finite, non-negative, sums consistent.
    rho = rc["rho_marginal"]
    coverage_curve = []
    for r in sorted(rho.keys()):
        coverage_curve.append({
            "rank": r,
            "rho_marginal": rho[r],
            "cumulative_coverage_within_rank": cumcov[r],
        })
    bins_populated = all(v > 0 for v in rho.values()) and rc["n_divergences"] > 0
    finite_ok = all(_finite(v) for v in rho.values()) and all(_finite(v) for v in cumcov.values())

    # acceptance-vs-depth confidence proxy curve (deeper = lower draft confidence regime).
    accept_vs_depth = [{"depth": d + 1, "rank1_conditional_acceptance": q}
                       for d, q in enumerate(rc["conditional_rank1_acceptance_q"])]

    tree_recoverable_public = cumcov[W]            # rank-2+ within top-W (the SEPARATE width lever)
    beyond_topW = rc["frac_true_beyond_topW"]      # not even width-W recovers these

    # THE THEOREM: greedy-exact width-1 acceptance = 1[argmax_draft == argmax_target], invariant to
    # every strictly-monotone draft-logit transform => no calibration/temperature/threshold knob
    # changes any accept decision => calibration-recoverable share = 0 EXACTLY.
    recoverable_fraction = 0.0
    return {
        "decisive_axis": "draft-confidence-vs-acceptance under width-1 greedy-EXACT verify",
        "argmax_invariance_theorem": (
            "accept_d = 1[argmax(draft_logits_d) == argmax(target_logits_d)]. For any T>0 and any "
            "strictly-monotone g, argmax(g(z)) = argmax(z); temperature, probability thresholds and "
            "confidence re-scalings are all such transforms. Therefore NO calibration/temperature/"
            "predicate knob can flip a reject to an accept (greedy-safe predicates are strictly "
            "one-sided -- they can only reject MORE). The calibration-recoverable share of the E[T] "
            "gap is EXACTLY 0, regardless of the confidence-bin shape."),
        "recoverable_fraction": recoverable_fraction,
        "recoverable_fraction_basis": "argmax-invariance (calibration/temperature/predicate lever)",
        "public_confidence_recovery_curve": coverage_curve,
        "public_accept_vs_depth_curve": accept_vs_depth,
        "curve_bins_populated": bool(bins_populated),
        "curve_finite": bool(finite_ok),
        "tree_recoverable_fraction_public": tree_recoverable_public,
        "frac_true_beyond_topW_public": beyond_topW,
        "tree_lever_note": (
            "the ONLY greedy-safe E[T]-raising lever is WIDTH>1 (tree rank-2+ recovery), a SEPARATE "
            "mechanism (adding candidate lanes, NOT re-calibrating). On the PUBLIC set "
            "{:.1%} of divergences have the true token within draft rank 2..{} (recoverable by a "
            "width-{} tree) and {:.1%} fall beyond top-{} (irrecoverable at any width<= {}). This "
            "lever is already priced in tree_private_acceptance_gap (private tree E[T] 4.38 raw -> "
            "4.92 calibrated). The private rank-2+ analog needs one local GPU rank-probe forward."
            .format(tree_recoverable_public, W, W, beyond_topW, W, W)),
    }


# --------------------------------------------------------------------------- #
# Reconciliation across protocols + verdict.
# --------------------------------------------------------------------------- #
def reconcile(a1: dict, a2: dict) -> dict[str, Any]:
    return {
        "decode_total_gap": a1["total_gap"],
        "decode_anchors": {"pub": a1["pub_ET"], "priv": a1["priv_ET"]},
        "sglang_pooled_gap": a2["pooled_gap_vs_pub"],
        "sglang_anchors": {"pub": a2["pub_ET"], "priv_pooled": a2["pooled_ET"]},
        "agreement": (
            "both protocols agree in DIRECTION and SHAPE: the gap is driven by the depth-1 top-1 "
            "match (q0) dropping OOD -- decode q0 0.729->0.599 (drop 0.130), sglang q0 0.758->~0.66 "
            "pooled. The magnitudes differ by protocol (decode under-reads E[T] vs sglang; "
            "documented in tree_private_drop_reconcile) but the discrimination signature is "
            "identical. No protocol shows a calibration signature (a calibration signature would be "
            "invisible to greedy acceptance regardless)."),
    }


def verdict(a1: dict, a2: dict, a3: dict) -> dict[str, Any]:
    rf = a3["recoverable_fraction"]
    gap_verdict = "INTRINSIC-WEAKNESS" if rf <= EXACT_TOL else (
        "RECOVERABLE-CALIBRATION" if rf >= 1.0 - EXACT_TOL else "MIXED")
    return {
        "recoverable_fraction": rf,
        "gap_verdict": gap_verdict,
        "lever": None,
        "lever_priced_private_et_lift": 0.0,
        "tree_recoverable_fraction_public": a3["tree_recoverable_fraction_public"],
        "summary": (
            "INTRINSIC-WEAKNESS. The {:.3f} E[T] public->private gap (decode 3.844->3.090) is a "
            "DISCRIMINATION loss: the draft's top-1 token is wrong more often OOD (q0 0.729->0.599), "
            "universal across all 6 content categories and concentrated in the early conditionals "
            "(d1 drop {:.4f}; deep d5..d7 conditionals actually IMPROVE on private). Under width-1 "
            "greedy-EXACT verify, acceptance depends only on the draft ARGMAX, which is invariant to "
            "temperature / thresholds / any greedy-safe predicate -- so the "
            "calibration-recoverable_fraction = 0.0 EXACTLY. There is NO free greedy-safe lever. The "
            "only greedy-safe E[T]-raising lever is width>1 tree rank-2+ recovery (public coverage "
            "{:.1%} of divergences), a SEPARATE already-priced lane (tree private E[T] 4.38->4.92). "
            "Clean negative: the gap is genuine OOD weakness, not over-confidence."
            .format(a1["total_gap"], a1["d1_conditional_drop"],
                    a3["tree_recoverable_fraction_public"])),
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(imp: dict, a1: dict, a2: dict, a3: dict, vd: dict) -> dict[str, Any]:
    # (a) per-position EXACTLY reconstructs the decode total gap; per-category conditional
    #     attribution EXACTLY reconstructs the pooled drop at every depth.
    cond_a = bool(a1["reconstruction_residual"] < EXACT_TOL
                  and a2["max_additive_residual"] < EXACT_TOL)

    # (b) E[T] anchors match cached: public 3.844, private 3.090 (within headline rounding), AND
    #     the ladder reconstruction matches the cached ladder values exactly.
    pub_ok_headline = abs(a1["pub_ET"] - PUB_ET_ANCHOR) < ANCHOR_TOL
    priv_ok_headline = abs(a1["priv_ET"] - PRIV_ET_ANCHOR) < ANCHOR_TOL
    pub_ok_exact = abs(a1["pub_ET"] - imp["pub_decode_et_cached_ladder"]) < EXACT_TOL
    priv_ok_exact = abs(a1["priv_ET"] - imp["priv_decode_et_cached"]) < EXACT_TOL
    pub_ok_server = abs(a1["pub_ET"] - imp["pub_decode_et_server"]) < ANCHOR_TOL
    cond_b = bool(pub_ok_headline and priv_ok_headline and pub_ok_exact and priv_ok_exact
                  and pub_ok_server)

    # (c) confidence-vs-acceptance curves well-formed: bins populated, finite (monotonicity NOT
    #     required). Both the public rank-coverage curve and the accept-vs-depth proxy.
    accept_curve_ok = all(_finite(x["rank1_conditional_acceptance"])
                          for x in a3["public_accept_vs_depth_curve"])
    cond_c = bool(a3["curve_bins_populated"] and a3["curve_finite"] and accept_curve_ok)

    # (d) recoverable_fraction in [0,1] and verdict consistent with it.
    rf = vd["recoverable_fraction"]
    verdict_ok = (vd["gap_verdict"] == "INTRINSIC-WEAKNESS" and rf <= EXACT_TOL) or (
        vd["gap_verdict"] == "RECOVERABLE-CALIBRATION" and rf >= 1.0 - EXACT_TOL) or (
        vd["gap_verdict"] == "MIXED" and EXACT_TOL < rf < 1.0 - EXACT_TOL)
    cond_d = bool(0.0 <= rf <= 1.0 and verdict_ok)

    # (e) NaN-clean over the key scalars (full payload walk enforced in main()).
    key = [a1["total_gap"], a1["pub_ET"], a1["priv_ET"], a1["reconstruction_residual"],
           a2["pooled_ET"], a2["pooled_gap_vs_pub"], a2["max_additive_residual"],
           a3["recoverable_fraction"], a3["tree_recoverable_fraction_public"],
           a3["frac_true_beyond_topW_public"]]
    cond_e = all(_finite(x) for x in key)

    passes = bool(cond_a and cond_b and cond_c and cond_d and cond_e)
    return {
        "private_et_gap_decomp_self_test_passes": passes,
        "conditions": {
            "a_per_position_and_per_category_reconstruct_total_gap": cond_a,
            "b_et_anchors_match_cached_3p844_3p090": cond_b,
            "c_confidence_curves_well_formed": cond_c,
            "d_recoverable_fraction_in_unit_and_verdict_consistent": cond_d,
            "e_key_scalars_finite": cond_e,
        },
        "evidence": {
            "a_per_position_residual": a1["reconstruction_residual"],
            "a_per_category_max_residual": a2["max_additive_residual"],
            "b_pub_ET": a1["pub_ET"], "b_priv_ET": a1["priv_ET"],
            "b_pub_ET_cached_ladder": imp["pub_decode_et_cached_ladder"],
            "b_priv_ET_cached": imp["priv_decode_et_cached"],
            "b_pub_ET_server": imp["pub_decode_et_server"],
            "c_n_divergences": imp["rankcov"]["n_divergences"],
            "c_rho_bins": sorted(imp["rankcov"]["rho_marginal"].keys()),
            "d_recoverable_fraction": rf, "d_gap_verdict": vd["gap_verdict"],
        },
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    imp = load_imports()
    a1 = axis1_per_position(imp)
    a2 = axis2_per_category(imp)
    a3 = axis3_confidence_rank(imp)
    rec = reconcile(a1, a2)
    vd = verdict(a1, a2, a3)
    st = self_test(imp, a1, a2, a3, vd)
    return {
        "self_test": st,
        "test_metric": {"recoverable_fraction": vd["recoverable_fraction"]},
        "imports": imp,
        "axis1_per_position": a1,
        "axis2_per_category": a2,
        "axis3_confidence_rank": a3,
        "reconciliation": rec,
        "verdict": vd,
    }


# --------------------------------------------------------------------------- #
# NaN-clean walk.
# --------------------------------------------------------------------------- #
def _assert_nan_clean(payload: dict, path: str = "result") -> list[str]:
    bad: list[str] = []

    def walk(node: Any, p: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{p}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{p}[{i}]")
        elif isinstance(node, float) and not math.isfinite(node):
            bad.append(p)

    walk(payload, path)
    return bad


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def _print_report(syn: dict) -> None:
    a1 = syn["axis1_per_position"]
    a2 = syn["axis2_per_category"]
    a3 = syn["axis3_confidence_rank"]
    vd = syn["verdict"]
    st = syn["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("PUBLIC->PRIVATE E[T] GAP DECOMPOSITION (PR #258) -- recoverable (calibration) or "
          "intrinsic (OOD)?", flush=True)
    print("=" * 100, flush=True)
    print(f"  anchors (decode): public E[T]={a1['pub_ET']:.4f}  private E[T]={a1['priv_ET']:.4f}  "
          f"=>  total gap = {a1['total_gap']:.4f}", flush=True)
    print("-" * 100, flush=True)
    print("  AXIS 1 -- PER-POSITION (decode; EXACT additive):", flush=True)
    print("   depth | p_pub  | p_priv | cond_drop |  C_pub | C_priv | gap_contrib | gap_frac",
          flush=True)
    for r in a1["rows"]:
        print(f"     {r['depth']}   | {r['p_pub']:.4f} | {r['p_priv']:.4f} | {r['conditional_drop']:+.4f}  "
              f" | {r['C_pub']:.4f} | {r['C_priv']:.4f} |   {r['gap_contrib']:+.4f}   | {r['gap_frac']:.3f}",
              flush=True)
    print(f"   reconstruction residual = {a1['reconstruction_residual']:.2e}  (deep-tail "
          f"conditionals improve on private = {a1['deep_tail_conditionals_improve_on_private']})",
          flush=True)
    print("-" * 100, flush=True)
    print(f"  AXIS 2 -- PER-CATEGORY (sglang; public anchor {a2['pub_ET']:.4f}, pooled private "
          f"{a2['pooled_ET']:.4f}, pooled gap {a2['pooled_gap_vs_pub']:.4f}):", flush=True)
    print("   category            | weight |   q0   | q0_drop |  E[T]  | gap_vs_pub", flush=True)
    for pc in a2["per_category"]:
        print(f"   {pc['category']:<19} | {pc['weight']:.4f} | {pc['q0_top1']:.4f} | "
              f"{pc['q0_drop_vs_pub']:+.4f} | {pc['E_T']:.4f} |  {pc['gap_vs_pub']:+.4f}", flush=True)
    print(f"   universal q0 discrimination drop = {a2['universal_q0_discrimination_drop']}  "
          f"(max additive residual {a2['max_additive_residual']:.2e})", flush=True)
    print("-" * 100, flush=True)
    print("  AXIS 3 -- DRAFT-CONFIDENCE-vs-ACCEPTANCE (decisive):", flush=True)
    print(f"   argmax-invariance => recoverable_fraction (calibration) = "
          f"{a3['recoverable_fraction']:.4f}", flush=True)
    print(f"   public rank-2+ coverage (tree lever, SEPARATE) = "
          f"{a3['tree_recoverable_fraction_public']:.4f}   beyond-top{imp_W(syn)} = "
          f"{a3['frac_true_beyond_topW_public']:.4f}", flush=True)
    print(f"   confidence curve bins populated = {a3['curve_bins_populated']}  finite = "
          f"{a3['curve_finite']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  >>> recoverable_fraction = {vd['recoverable_fraction']:.4f}   gap_verdict = "
          f"{vd['gap_verdict']}   lever = {vd['lever']}", flush=True)
    print(f"  (PRIMARY) private_et_gap_decomp_self_test_passes = "
          f"{st['private_et_gap_decomp_self_test_passes']}", flush=True)
    for k, val in st["conditions"].items():
        print(f"          - {k}: {val}", flush=True)
    print(f"  VERDICT: {vd['summary']}", flush=True)
    print("=" * 100 + "\n", flush=True)


def imp_W(syn: dict) -> int:
    return int(syn["imports"]["rankcov"]["W"])


# --------------------------------------------------------------------------- #
# W&B logging (mirrors measured_et_shortfall; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[private-et-gap-decomp] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    imp = syn["imports"]
    a1 = syn["axis1_per_position"]
    a2 = syn["axis2_per_category"]
    a3 = syn["axis3_confidence_rank"]
    vd = syn["verdict"]
    st = syn["self_test"]

    run = init_wandb_run(
        job_type="validity-diagnostic",
        agent="ubel",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["private-et-gap-decomp", "validity-diagnostic", "acceptance-gap", "per-position",
              "per-category", "confidence-vs-acceptance", "argmax-invariance", "intrinsic-weakness",
              "bank-the-analysis"],
        config={
            "K_mtp": K_MTP,
            "pub_et_anchor": PUB_ET_ANCHOR,
            "priv_et_anchor": PRIV_ET_ANCHOR,
            "pub_decode_run": imp["pub_decode_run"],
            "priv_decode_run": imp["priv_decode_run"],
            "rankcov_run": imp["rankcov"]["run"],
            "imports": imp["provenance"],
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[private-et-gap-decomp] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "private_et_gap_decomp_self_test_passes":
            int(bool(st["private_et_gap_decomp_self_test_passes"])),
        "recoverable_fraction": vd["recoverable_fraction"],
        "gap_verdict_is_intrinsic": int(vd["gap_verdict"] == "INTRINSIC-WEAKNESS"),
        "decode_total_gap": a1["total_gap"],
        "decode_pub_ET": a1["pub_ET"],
        "decode_priv_ET": a1["priv_ET"],
        "per_position_reconstruction_residual": a1["reconstruction_residual"],
        "d1_conditional_drop": a1["d1_conditional_drop"],
        "deep_tail_improves_on_private": int(a1["deep_tail_conditionals_improve_on_private"]),
        "sglang_pub_ET": a2["pub_ET"],
        "sglang_pooled_ET": a2["pooled_ET"],
        "sglang_pooled_gap": a2["pooled_gap_vs_pub"],
        "per_category_max_additive_residual": a2["max_additive_residual"],
        "universal_q0_discrimination_drop": int(a2["universal_q0_discrimination_drop"]),
        "tree_recoverable_fraction_public": a3["tree_recoverable_fraction_public"],
        "frac_true_beyond_topW_public": a3["frac_true_beyond_topW_public"],
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"gap_contrib_d{r['depth']}": r["gap_contrib"] for r in a1["rows"]},
        **{f"cond_drop_d{r['depth']}": r["conditional_drop"] for r in a1["rows"]},
        **{f"cat_ET_{pc['category']}": pc["E_T"] for pc in a2["per_category"]},
        **{f"cat_q0drop_{pc['category']}": pc["q0_drop_vs_pub"] for pc in a2["per_category"]},
        **{f"selftest_{k}": int(bool(val)) for k, val in st["conditions"].items()},
    }
    summary = {k: val for k, val in summary.items()
               if not (isinstance(val, float) and not math.isfinite(val)) and val is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="private_et_gap_decomp_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[private-et-gap-decomp] wandb logged: {summary}", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="private-et-gap-decomp")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at,
        "pr": 258,
        "agent": "ubel",
        "kind": "private-et-gap-decomp",
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[private-et-gap-decomp] WARNING non-finite values at: {nan_paths}", flush=True)
    # fold nan-clean into self-test (e) and recompute PRIMARY.
    syn["self_test"]["conditions"]["e_key_scalars_finite"] = bool(
        syn["self_test"]["conditions"]["e_key_scalars_finite"] and payload["nan_clean"])
    passes = bool(all(syn["self_test"]["conditions"].values()))
    syn["self_test"]["private_et_gap_decomp_self_test_passes"] = passes

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "private_et_gap_decomp_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[private-et-gap-decomp] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    print(f"  PRIMARY private_et_gap_decomp_self_test_passes = {passes}", flush=True)
    print(f"  TEST recoverable_fraction = {syn['test_metric']['recoverable_fraction']:.4f}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        ok = passes and payload["nan_clean"]
        print(f"[private-et-gap-decomp] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
