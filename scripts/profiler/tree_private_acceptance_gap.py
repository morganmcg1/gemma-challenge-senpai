#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Tree private-acceptance gap (PR #151): does the descent-walk E[T] survive the
4-9% private distribution drop?

THE QUESTION
------------
The committed tree projection (descent-only 522, both-bugs 537.8) is priced on the
PUBLIC oracle per-position acceptance ladder. The official leaderboard verifies on a
PRIVATE prompt distribution, where the deployed frontier already drops Δ4.3% (481.53
public-verified -> 460.85 private-verified, BASELINE.md). This harness asks whether
the descent-walk E[T] -- and therefore the tree's clear-500 verdict -- survives that
private drop, by measuring the per-position drafter acceptance ladder on the
private-proxy distribution and propagating it through the banked descent-walk E[T] DP.

LOCAL, CPU-ONLY for the propagation (this file); the per-position PRIVATE ladder is
measured by `accept_calibration.py --dataset data/private_proxy_sharegpt.json` on the
single A10G (the deployed QAT MTP K=7 drafter + int4 target, real autoregressive
greedy spec-decode counters). No HF Job, no submission, no served-file change, no
training. Greedy identity / PPL untouched by construction (serves nothing here).

THE MODEL (reuses the banked machinery verbatim -- one source of truth)
----------------------------------------------------------------------
  * wirbel #135 / fern #134 descent-walk E[T] DP: score_tree_depthrank on the
    rho-optimal M=32/depth-9/max-branch-3 topology (#83/#86) with the measured
    rank-conditional rescue ladder rho_cond=[0.4165,0.2655,0.1908] (#79/#86) and a
    depth-dependent rising spine (build_depth_pvecs_measured). ET_tree(q1) overrides
    depth-1 and reproduces the committed cells:
        ET_tree(0.598)=4.811, ET_tree(0.679)=5.056 (cell3 descent-only),
        ET_tree(0.7287)=5.207 (cell4 both-bugs-fixed).
  * fern #142 official map: official = K_cal * E[T] / step * tau, K_cal=125.268,
    measured depth-9 step 1.2182 (lawine #136; roofline 1.2127 shown as reference),
    tau central 1.0 / low 0.9983. The 1.06019 local->official factor is FOLDED INTO
    K_cal (NOT applied again).

THE PROPAGATION
---------------
The deployed LINEAR served chain's per-position conditional acceptance spine IS the
tree's recovery-target ladder (PR #76 -> #135). Measuring it on the private proxy vs
the public oracle gives the per-position private drop, which we propagate:
  * both-bugs-fixed  : spine = private ladder, depth-1 = q_private[0].
  * descent-only     : spine = private ladder, depth-1 = q_private[0] * BUG1_MULT,
                       where BUG1_MULT = 0.679/0.7287 is the distribution-INDEPENDENT
                       build-plumbing depth-1 deficit (denken #133; cell3 vs cell4).
The branch rescue ratios rho_cond are UNMEASURABLE on the linear served stack (a
linear chain only proposes rank-1), so we report a band: rho_cond held at public
(branch rescue distribution-independent -> upper) vs rho_cond coupled to the spine
haircut (central).

THE CALIBRATION ANCHOR (PR step 1)
----------------------------------
The chat-proxy probe (kanna #44) reads an ~11% AGGREGATE E[T] drop -- a PESSIMISTIC
upper bound, ~2.6x the organizer ground-truth Δ4.3% (BASELINE.md). We therefore
report BOTH: the RAW-PROXY (pessimistic) projection at the measured proxy drop, AND a
GROUND-TRUTH-CALIBRATED central projection that linearly scales the per-position
deficit so the aggregate linear-E[T] drop matches a target (default 4.3%). The
calibrated ladder at fraction f of the proxy shift is
    q_cal[j] = q_public[j] - f * (q_public[j] - q_private_raw[j]),
with f chosen so the linear-E[T] drop hits each target in {4.3%, 9%, raw-proxy}.

OUTPUTS (PR #151)
-----------------
  tree_private_tps_proj   (PRIMARY) descent-only official TPS, ground-truth-calibrated
  tree_private_clears_500 (TEST)    descent-only official >= 500 at the ground truth
  tree_private_et                   descent-only private E[T] (central rho_cond)
  private E[T] drop % vs public, for descent-only and both-bugs, with a CI/band.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from treeshape_measured_accept import (  # noqa: E402
    build_depth_pvecs_measured,
    load_measured,
    load_rank_coverage,
    score_tree_depthrank,
)
from traversal_verify_et import load_m32_topology, tree_arrays  # noqa: E402


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# fern #142 gate: K_cal, official_tps_map, accept_length_for_official, step, tau, bars.
_m16 = _load("m16_measured_500_gate", os.path.join(_HERE, "m16_measured_500_gate.py"))
K_CAL = _m16.K_CAL                              # 125.268 (= 481.53 / 3.844)
official_tps_map = _m16.official_tps_map        # K_cal * E[T] / step * tau
accept_length_for_official = _m16.accept_length_for_official
STEP_ROOFLINE = _m16.STEP_ROOFLINE_DEPTH9       # 1.2127 (#125 roofline; reference)
TARGET_500 = _m16.TARGET_OFFICIAL               # 500.0
TARGET_530 = _m16.TARGET_530                    # 530.0
TAU = _m16.TAU                                  # {'low':0.9983,'central':1.0,'high':1.0}
E_T_LINEAR = _m16.E_T_LINEAR                    # 3.844 deployed linear-MTP floor
FRONTIER_OFFICIAL = _m16.FRONTIER_OFFICIAL      # 481.53

# ---- banked inputs --------------------------------------------------------- #
ACCEPT_JSON = "research/accept_calibration/accept_calibration_results.json"
RANKCOV_JSON = "research/rank_coverage/rank_coverage_results.json"
RHO_OPT_JSON = "research/spec_cost_model/rho_optimal_topology_results.json"

# ---- PR-specified measured depth-9 step (lawine #136) ---------------------- #
STEP_MEASURED_DEPTH9 = 1.2182                    # PR #151: measured step; primary

# ---- banked descent-walk anchors (self-test; the model is valid iff these hold) #
DEPTH1_PUBLIC = 0.728739760479042                # rho-optimal depth-1 (BUG-1 target)
DEPTH1_DESCENT_ONLY = 0.679                       # cell3 as-built depth-1 (BUG-1 live)
ANCHOR_ET_0598 = 4.811237948198919               # ET_tree(0.598)
ANCHOR_ET_CELL3 = 5.0564                          # ET_tree(0.679)  descent-only
ANCHOR_ET_CELL4 = 5.2068                          # ET_tree(0.7287) both-bugs-fixed
# BUG-1 build-plumbing depth-1 deficit (distribution-INDEPENDENT; cell3 vs cell4).
BUG1_MULT = DEPTH1_DESCENT_ONLY / DEPTH1_PUBLIC   # 0.9318

W_DEFAULT = 4
MAXD_DEFAULT = 24
# organizer ground-truth aggregate private drop (481.53 -> 460.85; BASELINE.md).
GROUND_TRUTH_DROP = 0.043


def _jd(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} not JSON serializable")


def cum_to_conditional(cum: list[float]) -> list[float]:
    q = [cum[0]]
    for i in range(1, len(cum)):
        prev = cum[i - 1]
        q.append(cum[i] / prev if prev > 0 else 0.0)
    return q


def linear_et_from_q(q: list[float]) -> float:
    """Spine-only linear E[T] = 1 + sum(cumulative C), C[d]=prod(q[0..d])."""
    et, c = 1.0, 1.0
    for qd in q:
        c *= qd
        et += c
    return et


class DescentModel:
    """The banked descent-walk E[T] DP (wirbel #135 / fern #134), parameterised by a
    per-position conditional spine ladder. ET_tree overrides depth-1 only."""

    def __init__(self, parent, rho_cond, W=W_DEFAULT, maxd=MAXD_DEFAULT):
        self.parent = parent
        self.rho_cond = list(rho_cond)
        self.W = W
        self.maxd = maxd
        _, depth_arr, _ = tree_arrays(parent)
        self.built_depth = max(depth_arr)

    def et(self, spine: list[float], rho_cond=None) -> float:
        rc = self.rho_cond if rho_cond is None else rho_cond
        pv = build_depth_pvecs_measured(list(spine), rc, self.W, self.maxd, "flat")
        return score_tree_depthrank(self.parent, pv)[0]

    def et_tree(self, q1: float, spine: list[float], rho_cond=None) -> float:
        qq = list(spine)
        qq[0] = q1
        return self.et(qq, rho_cond=rho_cond)


def official_band(et: float, step: float) -> dict:
    """official TPS at central + tau-low corners and the clear verdicts."""
    central = official_tps_map(et, step, TAU["central"])
    low = official_tps_map(et, step, TAU["low"])
    return {
        "official_central": central,
        "official_taulow": low,
        "clears_500_central": bool(central >= TARGET_500),
        "clears_500_conservative": bool(low >= TARGET_500),
        "clears_530_central": bool(central >= TARGET_530),
        "margin_to_500": central - TARGET_500,
    }


def build_calibrated_ladder(q_public, q_private_raw, frac):
    """q_cal[j] = q_public[j] - frac*(q_public[j]-q_private_raw[j]); frac in [0,1]."""
    n = min(len(q_public), len(q_private_raw))
    return [q_public[j] - frac * (q_public[j] - q_private_raw[j]) for j in range(n)]


def frac_for_target_drop(q_public, q_private_raw, target_drop):
    """Find frac so the linear-E[T] drop of the calibrated ladder == target_drop.
    Linear-E[T] is monotone decreasing in frac, so bisection is exact."""
    et_pub = linear_et_from_q(q_public)
    proxy_drop = (et_pub - linear_et_from_q(q_private_raw)) / et_pub
    if proxy_drop <= 1e-9:
        return 0.0, proxy_drop
    if target_drop >= proxy_drop:
        return 1.0, proxy_drop
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        q_cal = build_calibrated_ladder(q_public, q_private_raw, mid)
        drop = (et_pub - linear_et_from_q(q_cal)) / et_pub
        if drop < target_drop:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), proxy_drop


def project_one(model: DescentModel, spine_private, q_public, rho_pub, h, step):
    """Descent-only + both-bugs E[T] and official TPS for a private spine ladder.
    rho_cond band: public (upper) vs coupled h-scaled (central)."""
    rho_central = [r * h for r in rho_pub]
    out = {}
    for topo, q1 in (("descent_only", spine_private[0] * BUG1_MULT),
                     ("both_bugs", spine_private[0])):
        et_upper = model.et_tree(q1, spine_private, rho_cond=rho_pub)      # rho public
        et_central = model.et_tree(q1, spine_private, rho_cond=rho_central)  # rho coupled
        band = official_band(et_central, step)
        band_upper = official_band(et_upper, step)
        # full band: min over {central rho x tau-low}, max over {public rho x tau-central}
        tps_lo = official_tps_map(et_central, step, TAU["low"])
        tps_hi = official_tps_map(et_upper, step, TAU["central"])
        out[topo] = {
            "depth1": q1,
            "et_central_rho_coupled": et_central,
            "et_upper_rho_public": et_upper,
            "official_central": band["official_central"],
            "official_taulow": band["official_taulow"],
            "official_upper_rho_public": band_upper["official_central"],
            "tps_band": [tps_lo, tps_hi],
            "clears_500_central": band["clears_500_central"],
            "clears_500_conservative": band["clears_500_conservative"],
            "clears_500_band_low": bool(tps_lo >= TARGET_500),
            "clears_530_central": band["clears_530_central"],
            "margin_to_500": band["margin_to_500"],
        }
    return out


def run_self_test(model: DescentModel, q_public, rho_pub):
    """The model is valid iff it reproduces the banked descent-walk anchors."""
    et_0598 = model.et_tree(0.598, q_public, rho_cond=rho_pub)
    et_cell3 = model.et_tree(DEPTH1_DESCENT_ONLY, q_public, rho_cond=rho_pub)
    et_cell4 = model.et_tree(DEPTH1_PUBLIC, q_public, rho_cond=rho_pub)
    checks = {
        "ET_tree_0598": et_0598,
        "ET_tree_0598_expected": ANCHOR_ET_0598,
        "ET_tree_cell3_descent_only_0679": et_cell3,
        "ET_tree_cell3_expected": ANCHOR_ET_CELL3,
        "ET_tree_cell4_both_bugs_0729": et_cell4,
        "ET_tree_cell4_expected": ANCHOR_ET_CELL4,
        "official_cell3_step_meas": official_tps_map(et_cell3, STEP_MEASURED_DEPTH9, 1.0),
        "official_cell4_step_meas": official_tps_map(et_cell4, STEP_MEASURED_DEPTH9, 1.0),
        "official_cell3_step_roofline": official_tps_map(et_cell3, STEP_ROOFLINE, 1.0),
        "official_cell4_step_roofline": official_tps_map(et_cell4, STEP_ROOFLINE, 1.0),
        "clear_500_bar_step_meas": accept_length_for_official(TARGET_500, STEP_MEASURED_DEPTH9, 1.0),
        "clear_500_bar_step_roofline": accept_length_for_official(TARGET_500, STEP_ROOFLINE, 1.0),
    }
    ok = (abs(et_0598 - ANCHOR_ET_0598) < 0.02
          and abs(et_cell3 - ANCHOR_ET_CELL3) < 0.02
          and abs(et_cell4 - ANCHOR_ET_CELL4) < 0.02)
    checks["passes"] = bool(ok)
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accept-json", default=ACCEPT_JSON,
                    help="public per-position ladder (accept_calibration_results.json).")
    ap.add_argument("--private-json", default=None,
                    help="MEASURED private-proxy accept_calibration output "
                         "(accept_calibration.py --dataset ...). Omit for CPU self-test only.")
    ap.add_argument("--accept-source", default="server_log",
                    choices=["server_log", "prometheus"])
    ap.add_argument("--rankcov-json", default=RANKCOV_JSON)
    ap.add_argument("--rho-opt-json", default=RHO_OPT_JSON)
    ap.add_argument("--step", type=float, default=STEP_MEASURED_DEPTH9,
                    help="measured depth-9 step (lawine #136; PR #151 = 1.2182).")
    ap.add_argument("--ground-truth-drop", type=float, default=GROUND_TRUTH_DROP,
                    help="organizer ground-truth aggregate private drop (BASELINE Δ4.3%).")
    ap.add_argument("--drop-targets", type=float, nargs="+", default=[0.043, 0.09],
                    help="calibrated aggregate-drop anchors; raw-proxy is always added.")
    ap.add_argument("--W", type=int, default=W_DEFAULT)
    ap.add_argument("--max-depth", type=int, default=MAXD_DEFAULT)
    ap.add_argument("--output",
                    default="research/validity/tree_private_acceptance_gap/results.json")
    ap.add_argument("--wandb-project", "--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb-entity", "--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb-group", "--wandb_group", default="tree-private-acceptance-gap")
    ap.add_argument("--wandb-name", "--wandb_name",
                    default="stark/tree-private-acceptance-gap")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    # ---- banked model ----
    meas_pub = load_measured(args.accept_json, args.accept_source)
    q_public = list(meas_pub["q"])
    et_public_linear = linear_et_from_q(q_public)
    rc = load_rank_coverage(args.rankcov_json)
    rho_pub = rc["rho_cond"]
    parent = load_m32_topology(args.rho_opt_json)
    model = DescentModel(parent, rho_pub, W=args.W, maxd=args.max_depth)

    print(f"[private-gap] public spine q = {[round(x,4) for x in q_public]}", flush=True)
    print(f"[private-gap] public linear E[T] = {et_public_linear:.4f} "
          f"(reported {meas_pub['E_T']:.4f}); rho_cond = {rho_pub}", flush=True)

    # ---- self-test (model validity; reproduces banked descent-walk anchors) ----
    st = run_self_test(model, q_public, rho_pub)
    print(f"[private-gap] SELF-TEST: ET_tree(0.598)={st['ET_tree_0598']:.4f} "
          f"(exp {ANCHOR_ET_0598:.4f}); cell3 ET_tree(0.679)="
          f"{st['ET_tree_cell3_descent_only_0679']:.4f} (exp {ANCHOR_ET_CELL3}); "
          f"cell4 ET_tree(0.7287)={st['ET_tree_cell4_both_bugs_0729']:.4f} "
          f"(exp {ANCHOR_ET_CELL4}) -> {'PASS' if st['passes'] else 'FAIL'}", flush=True)
    print(f"[private-gap] SELF-TEST official @meas-step {args.step}: cell3="
          f"{st['official_cell3_step_meas']:.1f}, cell4={st['official_cell4_step_meas']:.1f}; "
          f"@roofline {STEP_ROOFLINE}: cell3={st['official_cell3_step_roofline']:.1f}, "
          f"cell4={st['official_cell4_step_roofline']:.1f}; clear-500 bar(meas)="
          f"{st['clear_500_bar_step_meas']:.3f}", flush=True)
    assert st["passes"], "descent-walk model does not reproduce the banked anchors"

    # public descent-walk references (for the drop %).
    et_pub_descent = model.et_tree(q_public[0] * BUG1_MULT, q_public, rho_cond=rho_pub)
    et_pub_both = model.et_tree(q_public[0], q_public, rho_cond=rho_pub)
    pub_ref = {
        "et_descent_only": et_pub_descent,
        "et_both_bugs": et_pub_both,
        "official_descent_only": official_tps_map(et_pub_descent, args.step, 1.0),
        "official_both_bugs": official_tps_map(et_pub_both, args.step, 1.0),
    }

    results = {
        "config": vars(args),
        "constants": {
            "K_cal": K_CAL, "step_measured_depth9": args.step,
            "step_roofline": STEP_ROOFLINE, "tau_band": TAU, "bug1_mult": BUG1_MULT,
            "ground_truth_drop": args.ground_truth_drop,
            "clear_500_bar_at_step": accept_length_for_official(TARGET_500, args.step, 1.0),
            "frontier_official": FRONTIER_OFFICIAL, "target_500": TARGET_500,
        },
        "public_ladder": {
            "q_conditional": q_public, "linear_E_T": et_public_linear,
            "rho_cond": rho_pub, "depth1_public": q_public[0],
        },
        "self_test": st,
        "public_descent_walk_reference": pub_ref,
        "private_pending": args.private_json is None,
    }

    verdict = {
        "self_test_passes": int(st["passes"]),
        "private_pending": args.private_json is None,
    }

    # ---- private propagation (only when the measured private ladder is provided) ----
    if args.private_json is not None:
        meas_priv = load_measured(args.private_json, args.accept_source)
        q_private_raw = list(meas_priv["q"])
        et_priv_linear = linear_et_from_q(q_private_raw)
        proxy_drop = (et_public_linear - et_priv_linear) / et_public_linear
        h_raw = float(np.mean(q_private_raw)) / float(np.mean(q_public[:len(q_private_raw)]))
        print(f"[private-gap] private RAW proxy spine q = "
              f"{[round(x,4) for x in q_private_raw]}", flush=True)
        print(f"[private-gap] private RAW linear E[T] = {et_priv_linear:.4f}; "
              f"proxy aggregate drop = {proxy_drop*100:.2f}% (pessimistic upper bound); "
              f"spine haircut h = {h_raw:.4f}", flush=True)

        # drop anchors: the calibrated targets + the raw proxy (frac=1.0).
        anchors = []
        seen = set()
        for tgt in list(args.drop_targets) + [proxy_drop]:
            key = round(tgt, 4)
            if key in seen:
                continue
            seen.add(key)
            frac, _ = frac_for_target_drop(q_public, q_private_raw, tgt)
            spine = build_calibrated_ladder(q_public, q_private_raw, frac)
            h = float(np.mean(spine)) / float(np.mean(q_public[:len(spine)]))
            label = ("raw_proxy" if abs(tgt - proxy_drop) < 1e-9
                     else f"calibrated_{tgt*100:.1f}pct")
            proj = project_one(model, spine, q_public, rho_pub, h, args.step)
            # private E[T] drop % vs the public descent-walk references
            for topo, ref in (("descent_only", et_pub_descent), ("both_bugs", et_pub_both)):
                proj[topo]["et_drop_pct_vs_public"] = (
                    (ref - proj[topo]["et_central_rho_coupled"]) / ref * 100.0)
            anchors.append({
                "label": label, "aggregate_drop_target": tgt, "frac_of_proxy_shift": frac,
                "spine": spine, "spine_haircut_h": h,
                "linear_E_T": linear_et_from_q(spine), "projection": proj,
            })
            d, b = proj["descent_only"], proj["both_bugs"]
            print(f"[private-gap] {label:<18s} drop={tgt*100:5.2f}% frac={frac:.3f}: "
                  f"descent E[T]={d['et_central_rho_coupled']:.4f}->"
                  f"{d['official_central']:.1f} TPS (clears500={d['clears_500_central']}); "
                  f"both E[T]={b['et_central_rho_coupled']:.4f}->"
                  f"{b['official_central']:.1f}", flush=True)

        # ground-truth-calibrated central = the PR headline.
        gt = next(a for a in anchors
                  if abs(a["aggregate_drop_target"] - args.ground_truth_drop) < 1e-6)
        gt_descent = gt["projection"]["descent_only"]
        raw = next(a for a in anchors if a["label"] == "raw_proxy")
        raw_descent = raw["projection"]["descent_only"]

        # PR #151 headline metrics.
        tree_private_tps_proj = gt_descent["official_central"]        # PRIMARY
        tree_private_clears_500 = gt_descent["clears_500_central"]    # TEST
        tree_private_et = gt_descent["et_central_rho_coupled"]
        # CI/band: [raw-proxy tau-low corner, ground-truth public-rho upper corner].
        band_lo = raw_descent["tps_band"][0]
        band_hi = gt_descent["tps_band"][1]

        results["private_ladder"] = {
            "q_conditional_raw": q_private_raw, "linear_E_T_raw": et_priv_linear,
            "proxy_aggregate_drop": proxy_drop, "spine_haircut_raw": h_raw,
            "measured_E_T_reported": meas_priv["E_T"], "source": args.accept_source,
            "private_json": args.private_json,
        }
        results["drop_anchors"] = anchors
        results["headline"] = {
            "tree_private_tps_proj": tree_private_tps_proj,
            "tree_private_clears_500": tree_private_clears_500,
            "tree_private_et": tree_private_et,
            "descent_only_et_drop_pct_vs_public": gt_descent["et_drop_pct_vs_public"],
            "both_bugs_et_drop_pct_vs_public":
                gt["projection"]["both_bugs"]["et_drop_pct_vs_public"],
            "tps_band_descent_only": [band_lo, band_hi],
            "raw_proxy_descent_tps": raw_descent["official_central"],
            "raw_proxy_clears_500": raw_descent["clears_500_central"],
            "ground_truth_drop": args.ground_truth_drop,
            "proxy_drop": proxy_drop,
        }
        verdict.update({
            "tree_private_tps_proj": tree_private_tps_proj,
            "tree_private_clears_500": int(tree_private_clears_500),
            "tree_private_et": tree_private_et,
            "descent_only_et_drop_pct": gt_descent["et_drop_pct_vs_public"],
            "raw_proxy_descent_tps": raw_descent["official_central"],
            "raw_proxy_clears_500": int(raw_descent["clears_500_central"]),
            "tps_band_low": band_lo, "tps_band_high": band_hi,
            "proxy_aggregate_drop": proxy_drop,
        })
        print(f"\n[private-gap] ===== PR #151 HEADLINE =====", flush=True)
        print(f"  tree_private_tps_proj (PRIMARY, descent-only, GT-{args.ground_truth_drop*100:.1f}%): "
              f"{tree_private_tps_proj:.1f} TPS", flush=True)
        print(f"  tree_private_clears_500 (TEST): {tree_private_clears_500}", flush=True)
        print(f"  tree_private_et: {tree_private_et:.4f}  "
              f"(public {et_pub_descent:.4f}, drop {gt_descent['et_drop_pct_vs_public']:.2f}%)",
              flush=True)
        print(f"  descent-only TPS band [raw-proxy taulow, GT public-rho]: "
              f"[{band_lo:.1f}, {band_hi:.1f}]", flush=True)
        print(f"  raw-proxy (pessimistic) descent-only: {raw_descent['official_central']:.1f} "
              f"TPS (clears500={raw_descent['clears_500_central']})", flush=True)

    results["verdict"] = verdict

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=_jd)
    print(f"[private-gap] wrote {args.output}", flush=True)

    if not args.no_wandb:
        try:
            log_wandb(args, results, verdict, st)
        except Exception as e:  # noqa: BLE001
            print(f"[private-gap] W&B logging failed (non-fatal): {e!r}", flush=True)
    print("[private-gap] DONE", flush=True)


def log_wandb(args, results, verdict, st):
    import wandb
    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
        name=args.wandb_name, job_type="analysis",
        config={"K_cal": K_CAL, "step": args.step, "step_roofline": STEP_ROOFLINE,
                "bug1_mult": BUG1_MULT, "ground_truth_drop": args.ground_truth_drop,
                "tau_low": TAU["low"], "tau_central": TAU["central"],
                "private_json": args.private_json,
                "private_pending": args.private_json is None,
                "W": args.W, "max_depth": args.max_depth})
    summ = {f"verdict/{k}": v for k, v in verdict.items()
            if not isinstance(v, (dict, list))}
    summ.update({f"selftest/{k}": v for k, v in st.items()
                 if not isinstance(v, (dict, list, str))})
    run.summary.update(summ)
    if "drop_anchors" in results:
        tbl = wandb.Table(columns=[
            "label", "aggregate_drop_pct", "frac", "descent_et", "descent_tps",
            "descent_clears_500", "both_et", "both_tps", "descent_et_drop_pct"])
        for a in results["drop_anchors"]:
            d = a["projection"]["descent_only"]
            b = a["projection"]["both_bugs"]
            tbl.add_data(a["label"], a["aggregate_drop_target"] * 100, a["frac_of_proxy_shift"],
                         d["et_central_rho_coupled"], d["official_central"],
                         int(d["clears_500_central"]), b["et_central_rho_coupled"],
                         b["official_central"], d["et_drop_pct_vs_public"])
        run.log({"drop_anchor_sweep": tbl})
        run.summary["wandb_run_id"] = run.id
    run.summary["wandb_run_id"] = run.id
    print(f"[private-gap] W&B run: {run.url}  (id={run.id})", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
