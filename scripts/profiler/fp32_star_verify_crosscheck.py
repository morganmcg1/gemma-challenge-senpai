#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""fp32 star-verify cross-check: does the QK+PV bf16->fp32 upcast recover the 13pp
depth-1 deficit chiku-inu localised to the star-attention VERIFY FORWARD? (PR #128)

PRE-RUN, QUOTA-CHEAP, ANALYTIC. No HF Job, no submission, no kernel build, no GPU.
Reuses ONLY banked data:
  * kanna #87 argmax-margin map (research/validity/verify_argmax_margin/<ts>/
    margin_perturb.npz): the per-position top-2 lm_head logit margin over the
    official 128x512 greedy decode (65,536 emitted positions). This IS the
    distribution the star-verify root-row argmax is taken against.
  * fern #125 tree E[T] realisation-ceiling step model + wirbel #83/#86 rho-optimal
    M=32 topology + the measured per-depth acceptance ladder (q[], rho_cond) =
    the #100 official-TPS compose.

THE QUESTION
------------
chiku-inu measured the tree-build depth-1 spine acceptance at 0.598 vs the correct
0.7287 (rank_coverage top1_76) -- a 13pp deficit that caps realised E[T] at ~2.10.
They localised it (static trace) to the star kernel's VERIFY FORWARD running in
bf16: a noisy bf16 root-row argmax that flips on near-ties and rejects the drafter's
correct depth-1 guess. Their fix: upcast QK+PV to fp32/IEEE (measured star relerr
bf16~1e-3 -> fp32~1e-6). They are about to spend a scarce quota run on
`tree-488-pw-fp32-v0`. This cross-check asks, from banked data alone:

  does a bf16 perturbation of relative magnitude ~relerr flip the depth-1 root-row
  argmax often enough (~13pp) to BE the deficit, and does fp32 drive the residual to
  ~0?

MODEL (Step 1/2)
----------------
The star-attention output carries relerr `e`; modelled (per the PR) as a per-logit
perturbation of magnitude ~e*|logit| on the lm_head logits feeding the root-row
argmax. For a position with bf16-rung top-2 logits L1>=L2 (margin m=L1-L2>=0):
  * Gaussian: delta_j ~ N(0,(e|Lj|)^2) indep -> delta2-delta1 ~ N(0, e^2(L1^2+L2^2))
    flip_prob = P(delta2-delta1 > m) = 0.5*erfc(m/(e*sqrt(L1^2+L2^2)*sqrt(2))).
  * Worst-case (model-independent UPPER bound): a relative-e perturbation can flip a
    position at all only if e*(|L1|+|L2|) > m. frac_could_flip bounds ANY flip model.
predicted_flip_frac = mean over the 65,536 positions. Compared to the 0.131 deficit.

FORWARD (Step 3)
----------------
If fp32 recovers depth-1 to q1, re-price the rho-optimal M=32/depth-9 tree:
score_tree_depthrank with pvecs[1][1]:=q1 (rest of the measured ladder + rho_cond
held) -> E[T](q1) -> official_TPS = K_cal*E[T]/step_time(W*)*tau (fern #125). Report
the MIN depth-1 that still clears 500 and whether the predicted fp32-recovered depth-1
clears it.

GATE
----
GREEN  : predicted bf16-flip ~= 13pp (+-2pp) AND fp32 residual ~0 AND fwd official>=500
AMBER  : bf16-flip explains MOST (8-11pp) of 13pp -> fp32 helps, a 2nd contributor remains
RED    : bf16-flip << 13pp -> the deficit is NOT primarily bf16 star-verify precision
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.special import erfc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from treeshape_measured_accept import (  # noqa: E402
    build_depth_pvecs_measured,
    load_measured,
    load_rank_coverage,
    score_tree_depthrank,
)
from traversal_verify_et import load_m32_topology, tree_arrays  # noqa: E402

# ---- banked inputs ----------------------------------------------------------
MARGIN_NPZ = ("research/validity/verify_argmax_margin/20260614T041541Z/"
              "margin_perturb.npz")
MARGIN_REPORT = ("research/validity/verify_argmax_margin/20260614T041541Z/"
                 "report.json")
ACCEPT_JSON = "research/accept_calibration/accept_calibration_results.json"
RANKCOV_JSON = "research/rank_coverage/rank_coverage_results.json"
RHO_OPT_JSON = "research/spec_cost_model/rho_optimal_topology_results.json"
CEILING_JSON = "research/spec_cost_model/tree_et_realization_ceiling_results.json"

# chiku-inu MEASURED star-attention output relerr (board 20260614-092043-711),
# validated locally on sm_120: bf16 ~1e-3 -> fp32 ~1e-6.
RELERR_BF16 = 1e-3
RELERR_FP32 = 1e-6

# chiku-inu localizer targets (board msg + tree-v2 stats).
DEPTH1_CORRECT = 0.728739760479042   # rank_coverage top1_76 (the "correct" depth-1)
DEPTH1_BUILT = 0.598                  # tree-488-pw-v0 measured (the bf16 build)
DEFICIT = DEPTH1_CORRECT - DEPTH1_BUILT   # 0.1307 (the 13pp to explain)

OFFICIAL_TARGET = 500.0


def _jd(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} not JSON serializable")


# --------------------------------------------------------------------------- #
# Step 1/2 -- argmax-flip-frac vs star relerr, convolved with kanna #87 margins
# --------------------------------------------------------------------------- #
def flip_frac_gaussian(L1, L2, margin, relerr):
    """Expected argmax-flip-frac: P(delta2-delta1 > margin), delta_j ~ N(0,(e|Lj|)^2).
    Ties (margin<=0) -> 0.5 (a relerr perturbation reshuffles a true tie ~half the
    time); this OVER-counts the fp32 case (deterministic tie-break matches the fp32
    reference) and is reported alongside the worst-case bound + kanna's direct 0-flip
    fp32 measurement."""
    sig = relerr * np.sqrt(L1 ** 2 + L2 ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        fp = 0.5 * erfc(margin / (sig * np.sqrt(2)))
    fp = np.where(sig > 0, fp, (margin <= 0) * 0.5)
    return float(fp.mean()), fp


def frac_could_flip_worstcase(L1, L2, margin, relerr):
    """Model-INDEPENDENT upper bound: a perturbation with |delta_j|<=e*|Lj| can flip a
    position only if e*(|L1|+|L2|) > margin. Bounds ANY flip model from above."""
    return float((relerr * (np.abs(L1) + np.abs(L2)) > margin).mean())


def relerr_for_target_flip(L1, L2, margin, target, mode):
    """Solve for the relerr that yields flip_frac == target (bisection)."""
    lo, hi = 1e-6, 3.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        ff = (flip_frac_gaussian(L1, L2, margin, mid)[0] if mode == "gaussian"
              else frac_could_flip_worstcase(L1, L2, margin, mid))
        if ff >= target:
            hi = mid
        else:
            lo = mid
    return hi


def step1_step2(npz_path, report_path):
    d = np.load(npz_path)
    L1 = d["ref_top1"].astype(np.float64)
    L2 = d["ref_top2"].astype(np.float64)
    margin = L1 - L2
    n = margin.size
    rep = json.load(open(report_path)) if os.path.exists(report_path) else {}

    ties = int((margin <= 0).sum())
    out = {
        "n_positions": n,
        "median_margin": float(np.median(margin)),
        "mean_margin": float(margin.mean()),
        "max_abs_logit": float(np.abs(L1).max()),
        "exact_tie_frac": ties / n,
        "exact_tie_count": ties,
        "margin_pcts": {f"p{p:02d}": float(np.percentile(margin, p))
                        for p in (1, 2, 5, 10, 20, 50)},
        "relerr_bf16": RELERR_BF16,
        "relerr_fp32": RELERR_FP32,
        "deficit_to_explain": DEFICIT,
    }

    # Step 1 -- bf16
    bf16_g, _ = flip_frac_gaussian(L1, L2, margin, RELERR_BF16)
    bf16_wc = frac_could_flip_worstcase(L1, L2, margin, RELERR_BF16)
    out["bf16_depth1_flip_frac_gaussian"] = bf16_g
    out["bf16_depth1_flip_frac_worstcase_bound"] = bf16_wc
    out["bf16_depth1_flip_frac_predicted"] = bf16_g          # primary point estimate
    out["bf16_explains_frac_of_deficit_gaussian"] = bf16_g / DEFICIT
    out["bf16_explains_frac_of_deficit_worstcase"] = bf16_wc / DEFICIT

    # Step 2 -- fp32 residual
    fp32_g, _ = flip_frac_gaussian(L1, L2, margin, RELERR_FP32)
    fp32_wc = frac_could_flip_worstcase(L1, L2, margin, RELERR_FP32)
    out["fp32_residual_depth1_flip_frac_gaussian"] = fp32_g
    out["fp32_residual_depth1_flip_frac_worstcase_bound"] = fp32_wc
    # kanna #87 DIRECT measurement: fp32-regime perturbations (SplitK reduction-order,
    # M-widen) flip 0/65536 argmaxes -- the star-verify fp32 upcast is the same class.
    cs = rep.get("capture_summary", {})
    out["fp32_direct_measured_flips_kanna87"] = {
        "splitk_flip_count_vs_emuS1": cs.get("splitk_flip_count_vs_emuS1"),
        "mwiden_flip_count_vs_refM8": cs.get("mwiden_flip_count_vs_refM8"),
        "note": "0/65536 flips under fp32-reduce regime (kanna #87) -> physical fp32 "
                "residual ~0; the Gaussian/worst-case fp32 numbers above only reflect "
                "unresolved bf16-cast ties whose true fp32 margin this capture does "
                "not store.",
    }
    out["fp32_residual_depth1_flip_frac"] = 0.0   # physical (kanna direct + det. tie-break)

    # relerr the deficit WOULD require (how far chiku's 1e-3 is from explaining 13pp)
    out["relerr_needed_for_deficit_gaussian"] = relerr_for_target_flip(
        L1, L2, margin, DEFICIT, "gaussian")
    out["relerr_needed_for_deficit_worstcase"] = relerr_for_target_flip(
        L1, L2, margin, DEFICIT, "worstcase")
    out["relerr_needed_over_measured_gaussian"] = (
        out["relerr_needed_for_deficit_gaussian"] / RELERR_BF16)
    out["relerr_needed_over_measured_worstcase"] = (
        out["relerr_needed_for_deficit_worstcase"] / RELERR_BF16)

    # relerr sensitivity curve (for the report / wandb)
    out["relerr_sweep"] = {}
    for e in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1.4e-2, 3e-2, 8e-2, 1e-1):
        g, _ = flip_frac_gaussian(L1, L2, margin, e)
        out["relerr_sweep"][f"{e:.0e}"] = {
            "gaussian": g, "worstcase": frac_could_flip_worstcase(L1, L2, margin, e)}
    return out


# --------------------------------------------------------------------------- #
# Step 3 -- forward recovered depth-1 -> E[T] -> official TPS (#100 compose)
# --------------------------------------------------------------------------- #
def load_step_model(ceiling_path):
    c = json.load(open(ceiling_path))["step_model"]
    return {
        "K_cal": c["K_cal"],
        "g_drafter": c["g_drafter"],
        "base_drafter_depth": c["base_drafter_depth"],
        "attn_share": c["attn_share"],
        "gemm_cost_mult": {int(k): v for k, v in c["gemm_cost_mult"].items()},
        "r_attn_M32": c["r_attn_M32_measured_primary"],
        "tau_band": c["tau_band_lawine116"],
        "norm_check_M8": c["normalisation_check_official_M8"],
    }


def step_time_wstar(sm, M=32, depth=9):
    return (sm["gemm_cost_mult"][M]
            + sm["g_drafter"] * (depth - sm["base_drafter_depth"]) / sm["base_drafter_depth"]
            + sm["attn_share"] * (sm["r_attn_M32"] - 1.0))


def step3(sm):
    parent = load_m32_topology(RHO_OPT_JSON)
    children, depth, leaves = tree_arrays(parent)
    meas = load_measured(ACCEPT_JSON, "server_log")
    rc = load_rank_coverage(RANKCOV_JSON)
    q = list(meas["q"])
    rho_cond = rc["rho_cond"]
    W, maxd = 4, 24

    def ET_tree(q1):
        qq = list(q)
        qq[0] = q1
        pv = build_depth_pvecs_measured(qq, rho_cond, W, maxd, "flat")
        return score_tree_depthrank(parent, pv)[0]

    st = step_time_wstar(sm)
    tau_c = sm["tau_band"]["central"]
    tau_lo = sm["tau_band"]["low"]

    def official(q1, tau):
        return sm["K_cal"] * ET_tree(q1) / st * tau

    def min_q1(tau):
        lo, hi = 0.0, DEPTH1_CORRECT
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if official(mid, tau) >= OFFICIAL_TARGET:
                hi = mid
            else:
                lo = mid
        return hi

    anchor = ET_tree(DEPTH1_CORRECT)
    out = {
        "topology": {"M": 32, "depth": max(depth), "n": len(parent),
                     "max_branch": max(len(c) for c in children), "leaves": len(leaves)},
        "step_time_wstar": st,
        "tau_central": tau_c, "tau_low": tau_lo,
        "anchor_ET_at_correct": anchor,
        "anchor_official_at_correct_central": official(DEPTH1_CORRECT, tau_c),
        "norm_check_official_M8": sm["norm_check_M8"],
        "min_depth1_clears_500_central": min_q1(tau_c),
        "min_depth1_clears_500_taulow": min_q1(tau_lo),
    }
    # checkpoints
    for label, q1 in (("built_0598", DEPTH1_BUILT), ("correct_0729", DEPTH1_CORRECT)):
        out[f"ET_{label}"] = ET_tree(q1)
        out[f"official_{label}_central"] = official(q1, tau_c)
        out[f"official_{label}_taulow"] = official(q1, tau_lo)
    # E[T](q1) sweep
    out["q1_sweep"] = {}
    for q1 in np.round(np.arange(0.55, 0.7401, 0.01), 4):
        out["q1_sweep"][f"{q1:.2f}"] = {
            "ET": ET_tree(float(q1)),
            "official_central": official(float(q1), tau_c),
            "official_taulow": official(float(q1), tau_lo),
        }
    out["_ET_fn_anchor_matches_fern125"] = abs(anchor - 5.207) < 0.02
    return out, ET_tree, official, st


# --------------------------------------------------------------------------- #
def gate(step12, predicted_fp32_q1, official_at_predicted_central, min_q1_central):
    bf16 = step12["bf16_depth1_flip_frac_predicted"]
    bf16_wc = step12["bf16_depth1_flip_frac_worstcase_bound"]
    fp32 = step12["fp32_residual_depth1_flip_frac"]
    # primary band checks (against the 13pp deficit)
    near = abs(bf16 - DEFICIT) <= 0.02
    most = 0.08 <= max(bf16, bf16_wc) <= 0.11
    if near and fp32 <= 0.005 and official_at_predicted_central >= OFFICIAL_TARGET:
        g, label = "GREEN", ("fp32 alone is sufficient: bf16-flip ~= 13pp, fp32 "
                             "residual ~0, forwarded official >= 500.")
    elif most:
        g, label = "AMBER", ("bf16-flip explains MOST (8-11pp) of the 13pp; fp32 helps "
                             "but a 2nd contributor remains -- name it.")
    else:
        g, label = "RED", (
            f"bf16-flip << 13pp: the worst-case (model-independent) upper bound is only "
            f"{bf16_wc*100:.2f}% and the expected flip-frac {bf16*100:.2f}%, vs a "
            f"{DEFICIT*100:.1f}pp deficit. A relerr~1e-3 perturbation CANNOT flip 13% of "
            f"argmaxes against these margins (median 4.875). The depth-1 deficit is NOT "
            f"primarily bf16 star-verify precision; fp32 alone recovers at most "
            f"~{bf16_wc*100:.1f}pp (0.598 -> ~{DEPTH1_BUILT+bf16_wc:.3f}). FLAG the build "
            f"team before they spend quota on an fp32-only run.")
    return {"gate": g, "gate_label": label,
            "fp32_recovers_depth1": int(near and fp32 <= 0.005)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--margin-npz", default=MARGIN_NPZ)
    ap.add_argument("--margin-report", default=MARGIN_REPORT)
    ap.add_argument("--ceiling-json", default=CEILING_JSON)
    ap.add_argument("--output",
                    default="research/validity/fp32_star_verify_crosscheck/results.json")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT",
                                                              "gemma-challenge-senpai"))
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY",
                                                             "wandb-applied-ai-team"))
    ap.add_argument("--wandb-group", default="fp32-star-verify-crosscheck")
    ap.add_argument("--wandb-name", default="denken/fp32-star-verify-crosscheck")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    print("[xcheck] Step 1/2: bf16/fp32 argmax-flip-frac vs kanna #87 margins", flush=True)
    s12 = step1_step2(args.margin_npz, args.margin_report)
    print(f"  positions={s12['n_positions']} median_margin={s12['median_margin']:.3f} "
          f"ties={s12['exact_tie_frac']*100:.3f}%", flush=True)
    print(f"  bf16 flip-frac: gaussian={s12['bf16_depth1_flip_frac_gaussian']*100:.3f}%  "
          f"worstcase-bound={s12['bf16_depth1_flip_frac_worstcase_bound']*100:.3f}%  "
          f"vs deficit {DEFICIT*100:.1f}pp", flush=True)
    print(f"  bf16 explains {s12['bf16_explains_frac_of_deficit_worstcase']*100:.1f}% "
          f"(worstcase) of the deficit", flush=True)
    print(f"  fp32 residual: physical~0 (kanna #87 direct 0/65536); "
          f"gaussian={s12['fp32_residual_depth1_flip_frac_gaussian']*100:.3f}%", flush=True)
    print(f"  relerr needed for 13pp: gaussian={s12['relerr_needed_for_deficit_gaussian']:.2e} "
          f"({s12['relerr_needed_over_measured_gaussian']:.0f}x measured)  "
          f"worstcase={s12['relerr_needed_for_deficit_worstcase']:.2e} "
          f"({s12['relerr_needed_over_measured_worstcase']:.0f}x)", flush=True)

    print("[xcheck] Step 3: forward depth-1 -> E[T] -> official TPS", flush=True)
    sm = load_step_model(args.ceiling_json)
    s3, ET_tree, official, st = step3(sm)
    assert s3["_ET_fn_anchor_matches_fern125"], "E[T] anchor departs from fern #125 5.207"
    print(f"  anchor E[T](0.7287)={s3['anchor_ET_at_correct']:.4f} "
          f"official={s3['anchor_official_at_correct_central']:.2f} (fern 537.84)", flush=True)
    print(f"  built  E[T](0.598) ={s3['ET_built_0598']:.4f} "
          f"official={s3['official_built_0598_central']:.2f}", flush=True)
    print(f"  MIN depth-1 to clear 500 = {s3['min_depth1_clears_500_central']:.4f} "
          f"(central) / {s3['min_depth1_clears_500_taulow']:.4f} (tau_low)", flush=True)

    # predicted fp32-recovered depth-1 = built + bf16 flip-frac that fp32 removes.
    # fp32 removes the bf16-flip contribution; its recovery is bounded by the bf16
    # flip-frac (point estimate) and its worst-case bound.
    rec_point = DEPTH1_BUILT + s12["bf16_depth1_flip_frac_predicted"]
    rec_wc = DEPTH1_BUILT + s12["bf16_depth1_flip_frac_worstcase_bound"]
    off_point = official(rec_point, s3["tau_central"])
    off_wc = official(rec_wc, s3["tau_central"])
    s3["predicted_fp32_recovered_depth1_point"] = rec_point
    s3["predicted_fp32_recovered_depth1_worstcase"] = rec_wc
    s3["official_at_predicted_fp32_point_central"] = off_point
    s3["official_at_predicted_fp32_worstcase_central"] = off_wc
    s3["predicted_fp32_clears_500_point"] = bool(off_point >= OFFICIAL_TARGET)
    s3["predicted_fp32_clears_500_worstcase"] = bool(off_wc >= OFFICIAL_TARGET)
    print(f"  predicted fp32-recovered depth-1: point={rec_point:.4f} -> "
          f"official={off_point:.2f}; worstcase={rec_wc:.4f} -> official={off_wc:.2f}",
          flush=True)

    g = gate(s12, rec_point, off_point, s3["min_depth1_clears_500_central"])
    print(f"\n[xcheck] GATE: {g['gate']}  fp32_recovers_depth1={g['fp32_recovers_depth1']}",
          flush=True)
    print(f"  {g['gate_label']}", flush=True)

    results = {
        "config": vars(args),
        "inputs": {
            "margin_npz": args.margin_npz,
            "relerr_bf16_measured_chiku": RELERR_BF16,
            "relerr_fp32_measured_chiku": RELERR_FP32,
            "depth1_correct": DEPTH1_CORRECT, "depth1_built": DEPTH1_BUILT,
            "deficit": DEFICIT,
        },
        "step1_step2_flip": s12,
        "step3_forward": s3,
        "verdict": {
            "primary_metric_name": "bf16_depth1_flip_frac_predicted",
            "bf16_depth1_flip_frac_predicted": s12["bf16_depth1_flip_frac_predicted"],
            "bf16_depth1_flip_frac_worstcase_bound": s12["bf16_depth1_flip_frac_worstcase_bound"],
            "deficit_to_explain": DEFICIT,
            "fp32_residual_depth1_flip_frac": s12["fp32_residual_depth1_flip_frac"],
            "test_metric_name": "fp32_recovers_depth1",
            "fp32_recovers_depth1": g["fp32_recovers_depth1"],
            "min_depth1_clears_500_central": s3["min_depth1_clears_500_central"],
            "official_at_correct_depth1_central": s3["anchor_official_at_correct_central"],
            "predicted_fp32_recovered_depth1_point": rec_point,
            "official_at_predicted_fp32_point_central": off_point,
            **g,
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=_jd)
    print(f"[xcheck] wrote {args.output}", flush=True)

    if not args.no_wandb:
        try:
            log_wandb(args, results)
        except Exception as e:  # noqa: BLE001
            print(f"[xcheck] W&B logging failed (non-fatal): {e!r}", flush=True)
    print("[xcheck] DONE", flush=True)


def log_wandb(args, results):
    import wandb
    s12 = results["step1_step2_flip"]
    s3 = results["step3_forward"]
    v = results["verdict"]
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name, job_type="analysis",
                     config={
                         "relerr_bf16": RELERR_BF16, "relerr_fp32": RELERR_FP32,
                         "depth1_correct": DEPTH1_CORRECT, "depth1_built": DEPTH1_BUILT,
                         "deficit": DEFICIT, "official_target": OFFICIAL_TARGET,
                         "margin_npz": args.margin_npz,
                         "topology": "wirbel#83_M32_optimal_depth9", "analytic": True,
                     })
    flat = {f"verdict/{k}": val for k, val in v.items()
            if not isinstance(val, (dict, list))}
    flat.update({f"step12/{k}": val for k, val in s12.items()
                 if not isinstance(val, (dict, list))})
    flat.update({f"step3/{k}": val for k, val in s3.items()
                 if not isinstance(val, (dict, list))})
    run.summary.update(flat)
    run.log(flat)

    # relerr sensitivity table
    t = wandb.Table(columns=["relerr", "flip_frac_gaussian", "flip_frac_worstcase",
                             "x_measured_1e-3", "vs_deficit_0131"])
    for k, val in s12["relerr_sweep"].items():
        e = float(k)
        t.add_data(e, val["gaussian"], val["worstcase"], e / RELERR_BF16,
                   val["gaussian"] / DEFICIT)
    run.log({"relerr_sensitivity": t})

    # depth-1 -> official forward table
    t2 = wandb.Table(columns=["depth1_q1", "E_T_tree", "official_central", "official_taulow"])
    for k, val in s3["q1_sweep"].items():
        t2.add_data(float(k), val["ET"], val["official_central"], val["official_taulow"])
    run.log({"depth1_to_official": t2})
    run.finish()
    print(f"[xcheck] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
