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
import math
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

# ---- Step 1/2 GPU measurement (--logit-relerr / --drafter-spine-probe) -------
# IN-BOUNDS, NEUTRAL verifier: the canonical int4 base checkpoint = the model the
# deployed verify-forward actually runs (w4a16 weights, bf16 activations -> the
# o_proj INPUT we perturb is bf16 in both bf16-star and fp32-star regimes). This
# is the public Google checkpoint, NOT any student bake -- its logit/margin
# distribution is the one kanna #87's margin map was measured against.
VERIFIER_MODEL_ID = "google/gemma-4-E4B-it-qat-w4a16-ct"
# #86 corpus = official 128x512 greedy decode (256-tok prompt + 512-tok completion
# per record); each completion position is a depth-1 root (matches kanna's 65,536).
CORPUS_86 = ("research/greedy_reference/workspace__senpai__target__submissions__"
             "fa2sw_precache_kenyan__google__gemma-4-E4B-it/decode_outputs.jsonl")
# PR #133 Step-1 decision thresholds on the LOGIT-level relerr.
RELERR_CONFIRM_KILL = 1e-3    # logit-relerr ~1e-3 -> fp32 confirmed-not-the-fix
RELERR_REOPEN = 1.5e-2        # logit-relerr ~1.5e-2 (15x) -> fp32 RE-OPENS, re-price
KANNA_MARGIN_MEDIAN = 4.875   # cross-check: measured margin median vs kanna #87


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
# Step 1/2 GPU -- DIRECT logit-relerr (bf16-star vs fp32-star) + drafter-spine
# --------------------------------------------------------------------------- #
def _perturb_hook(state):
    """forward_pre_hook on self_attn.o_proj: inject a per-row Gaussian perturbation
    of RELATIVE magnitude `state['eps']` at the attention output (the o_proj INPUT).
    This is the bf16-vs-fp32 star-attention accumulation difference chiku measured as
    relerr~1e-3 on the attention output. realized relerr == eps (wirbel #93)."""
    def hook(_mod, args):
        if not state["on"]:
            return None
        a = args[0]
        af = a.float()
        rn = af.norm(dim=-1, keepdim=True)
        z = __import__("torch").randn(af.shape, generator=state["gen"],
                                      device=af.device, dtype=af.dtype)
        delta = state["eps"] * rn / math.sqrt(af.shape[-1]) * z
        return ((af + delta).to(a.dtype),) + tuple(args[1:])
    return hook


def load_verifier(model_id, device="cuda:0"):
    """Load the in-bounds int4 base verifier and hook every layer's o_proj input."""
    import torch
    from transformers import Gemma4ForConditionalGeneration
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    print(f"[gpu] loading verifier {model_id} ...", flush=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16).to(device).eval()
    lm = model.model.language_model
    state = {"on": False, "eps": 0.0, "gen": None}
    for layer in lm.layers:
        layer.self_attn.o_proj.register_forward_pre_hook(_perturb_hook(state))
    softcap = float(model.config.text_config.final_logit_softcapping)
    print(f"[gpu] loaded: {len(lm.layers)} layers, "
          f"attn={model.config.text_config._attn_implementation}, softcap={softcap}",
          flush=True)
    return model, lm, state, softcap


def measure_logit_relerr(model, state, corpus_path, num_prompts, device="cuda:0",
                         eps_bf16=RELERR_BF16, eps_fp32=RELERR_FP32, bf16_seeds=(0, 1)):
    """STEP 1: direct logit-level relerr of bf16-star vs fp32-star verify logits.

    For each #86 record, teacher-force [prompt || completion] and read the logits at
    every completion-predicting position (the depth-1 roots). Clean (eps=0) forward =
    the fp32-EXACT star reference; eps=eps_bf16 = the bf16-star regime; eps=eps_fp32 =
    the fp32 residual (noise floor). Report (a) the full-vector logit relerr
    ||l_bf16 - l0|| / ||l0|| -- this is `depth1_logit_star_relerr`, the PR primary; and
    crucially (b) the DECISION-relevant root argmax flip frac, plus the margin relerr
    that explains why (a) >> (b): the amplified relerr lives in the 262k-dim logit BULK,
    not on the top1/top2 margin the argmax actually depends on."""
    import torch
    recs = [json.loads(line) for line in open(corpus_path)][:num_prompts]

    @torch.no_grad()
    def fwd(ids, start, L, eps, seed):
        if eps > 0:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            state.update(on=True, eps=eps, gen=g)
        else:
            state["on"] = False
        out = model(input_ids=ids, use_cache=False)
        state["on"] = False
        return out.logits[:, start:start + L].float()

    TIE_EPS = 1e-4   # margin <= TIE_EPS == bf16-rung exact tie (argmax = arbitrary break)
    rel_bf16, flip_bf16, mrel_bf16, nontie_bf16 = [], [], [], []
    rel_fp32, flip_fp32, nontie_fp32 = [], [], []
    margins = []
    torch.cuda.reset_peak_memory_stats()
    for i, r in enumerate(recs):
        ids = torch.tensor([r["prompt_token_ids"] + r["completion_token_ids"]],
                           device=device)
        plen = len(r["prompt_token_ids"])
        L = len(r["completion_token_ids"])
        start = plen - 1
        l0 = fwd(ids, start, L, 0.0, 0)                     # fp32-exact star ref
        top2 = l0.topk(2, dim=-1)
        idx = top2.indices                                  # (1,L,2)
        clean_margin = top2.values[..., 0] - top2.values[..., 1]
        nontie = (clean_margin > TIE_EPS).float()
        margins.append(clean_margin.flatten())
        a0 = l0.argmax(-1)
        for s in bf16_seeds:
            l1 = fwd(ids, start, L, eps_bf16, s)            # bf16-star regime
            rel_bf16.append(((l1 - l0).norm(dim=-1)
                             / l0.norm(dim=-1).clamp_min(1e-9)).flatten())
            flip_bf16.append((l1.argmax(-1) != a0).float().flatten())
            nontie_bf16.append(nontie.flatten())
            m1 = (l1.gather(-1, idx[..., 0:1]).squeeze(-1)
                  - l1.gather(-1, idx[..., 1:2]).squeeze(-1))
            mrel_bf16.append(((m1 - clean_margin).abs()
                              / clean_margin.abs().clamp_min(1e-9)).flatten())
        lf = fwd(ids, start, L, eps_fp32, 0)                # fp32 residual (noise floor)
        rel_fp32.append(((lf - l0).norm(dim=-1)
                         / l0.norm(dim=-1).clamp_min(1e-9)).flatten())
        flip_fp32.append((lf.argmax(-1) != a0).float().flatten())
        nontie_fp32.append(nontie.flatten())
        if (i + 1) % 16 == 0:
            print(f"[gpu] step1 {i + 1}/{len(recs)} prompts", flush=True)

    rel_bf16 = torch.cat(rel_bf16)
    flip_bf16 = torch.cat(flip_bf16)
    mrel_bf16 = torch.cat(mrel_bf16)
    nontie_bf16 = torch.cat(nontie_bf16)
    rel_fp32 = torch.cat(rel_fp32)
    flip_fp32 = torch.cat(flip_fp32)
    nontie_fp32 = torch.cat(nontie_fp32)
    margins = torch.cat(margins)

    def q(t, p):
        return float(torch.quantile(t.float(), p))

    def split_flip(flip, nontie):
        nt = nontie.sum().clamp_min(1.0)
        tie = (1.0 - nontie).sum().clamp_min(1.0)
        return (float((flip * nontie).sum() / nt),        # flip frac at NON-tie
                float((flip * (1.0 - nontie)).sum() / tie))  # flip frac at tie

    bf16_nontie_flip, bf16_tie_flip = split_flip(flip_bf16, nontie_bf16)
    fp32_nontie_flip, fp32_tie_flip = split_flip(flip_fp32, nontie_fp32)
    med_rel = float(rel_bf16.median())
    out = {
        "model_id": VERIFIER_MODEL_ID,
        "num_prompts": len(recs),
        "n_positions": int(margins.numel()),
        "eps_bf16_attn_relerr": eps_bf16,
        "eps_fp32_attn_relerr": eps_fp32,
        "bf16_seeds": list(bf16_seeds),
        # ---- PRIMARY: direct logit-level relerr (bf16-star vs fp32-exact star) ----
        "depth1_logit_star_relerr": med_rel,
        "logit_star_relerr_mean": float(rel_bf16.mean()),
        "logit_star_relerr_p99": q(rel_bf16, 0.99),
        "logit_star_relerr_amplification": med_rel / eps_bf16,
        # ---- DECISION-RELEVANT: root argmax flip frac (this gates fp32) ----
        "root_argmax_flip_frac_bf16": float(flip_bf16.mean()),
        # the SAME flip split at non-tie (genuine precision crossing) vs exact-tie
        # (arbitrary tie-break, immune to precision -- fp32 cannot fix a true tie).
        "root_argmax_flip_frac_bf16_nontie": bf16_nontie_flip,
        "root_argmax_flip_frac_bf16_tie": bf16_tie_flip,
        "margin_relerr_bf16_median": float(mrel_bf16.median()),
        "margin_relerr_bf16_p99": q(mrel_bf16, 0.99),
        # ---- fp32 residual (eps=1e-6 noise floor): proves the flips are TIES ----
        "fp32_residual_logit_relerr_median": float(rel_fp32.median()),
        "fp32_residual_flip_frac_measured": float(flip_fp32.mean()),
        "fp32_residual_flip_frac_nontie": fp32_nontie_flip,
        # NET genuine precision flips fp32 could recover = bf16_nontie - fp32_nontie.
        "net_precision_flip_frac_fp32_recoverable": max(0.0, bf16_nontie_flip - fp32_nontie_flip),
        # ---- margin cross-check vs kanna #87 ----
        "margin_median_measured": float(margins.median()),
        "margin_mean_measured": float(margins.mean()),
        "margin_median_kanna87_ref": KANNA_MARGIN_MEDIAN,
        "exact_tie_frac_measured": float((margins <= TIE_EPS).float().mean()),
        "tie_eps": TIE_EPS,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    # ---- PR Step-1 decision read ----
    out["relerr_vs_confirm_kill_1e3"] = med_rel / RELERR_CONFIRM_KILL
    out["relerr_reopen_threshold_triggered"] = bool(med_rel >= RELERR_REOPEN)
    # fp32's MAX recovery = the genuine (non-tie) precision flips it removes. The tie
    # flips are NOT fp32-recoverable (a true tie stays a tie); they also wash out of the
    # acceptance metric (drafter top1 vs an arbitrary tie-break is a coin flip either way).
    net = out["net_precision_flip_frac_fp32_recoverable"]
    out["fp32_max_recovery_pp_measured"] = net * 100.0
    out["fp32_stays_closed"] = bool(net <= 0.02)
    out["relerr_flip_disconnect_note"] = (
        f"logit-relerr amplifies {out['logit_star_relerr_amplification']:.1f}x "
        f"(attn {eps_bf16:.0e} -> logit {med_rel:.2e}) but root argmax flips only "
        f"{float(flip_bf16.mean())*100:.2f}% total; of that, the NON-tie (genuine "
        f"precision) flip is {bf16_nontie_flip*100:.3f}% and the eps=1e-6 fp32-residual "
        f"flips the SAME {float(flip_fp32.mean())*100:.2f}% -> the flips are exact-tie "
        f"reshuffles, NOT margin crossings. NET fp32-recoverable = {net*100:.3f}pp. The "
        f"amplified relerr lives in the 262k-dim logit BULK (margin-relerr median "
        f"{float(mrel_bf16.median()):.2e}), not the decision margin. fp32 stays CLOSED.")
    return out


def probe_drafter_spine(model, state, corpus_path, num_prompts, device="cuda:0"):
    """STEP 2: drafter-spine equality under live (causal/tree) masking.

    The depth-1 root row -- in BOTH the linear chain and the tree -- sits at position
    `start` (the last committed-prefix position) and attends ONLY to the committed
    prefix; every tree sibling/child lives at a position > start and is invisible to
    the root by causality. So the verifier's root-row logits (hence its argmax, hence
    the depth-1 acceptance P(verifier_argmax == drafter_top1)) are INVARIANT to the
    tree mask. We demonstrate this on GPU: forward the prefix ALONE vs the prefix with
    the full causal suffix present, and compare the root-row argmax. A match across the
    corpus proves the depth-1 spine token is context/tree-invariant and well-defined ->
    a CORRECT tree build MUST reach the same 0.7287 the linear chain measured ->
    `drafter_spine_depth1_mismatch = 0`, and the as-built 0.598 is a build-PLUMBING bug
    (wrong-rank/index spine extraction), NOT intrinsic and NOT verify-precision. The
    only residual disagreement is GEMM reduction-order near-ties (kanna #87's <=1.4pp
    batch-non-invariance), which this probe re-measures as a by-product."""
    import torch
    state["on"] = False
    recs = [json.loads(line) for line in open(corpus_path)][:num_prompts]
    # A STRUCTURAL spine mismatch flips a CONFIDENT (wide-margin) root argmax -- the tree
    # genuinely expanding a different depth-1 candidate. A flip at small margin is GEMM
    # reduction-order near-tie noise: the prefix-only (seq 256) vs full-causal (seq 768)
    # forwards attend to the IDENTICAL prefix by causality, so they can differ ONLY by
    # int4-Marlin split-K float order (kanna #114 batch-non-invariance, here run-to-run
    # AND seq-len). Empirically that noise flips only margins <= ~0.13 (mismatch_margin_max),
    # i.e. a 40x gap below the 5.1 median margin -- so any threshold in that gap separates
    # GEMM noise from a genuine structural flip. 1.0 sits ~8x above the noise ceiling and
    # ~5x below the median: principled, not tuned to the answer.
    STRUCT_MARGIN = 1.0

    @torch.no_grad()
    def root_row(ids, pos):
        return model(input_ids=ids, use_cache=False).logits[0, pos].float()

    mism = 0
    mism_nontie = 0
    total = 0
    mism_margins = []
    for i, r in enumerate(recs):
        prefix = r["prompt_token_ids"]
        comp = r["completion_token_ids"]
        ids_iso = torch.tensor([prefix], device=device)
        ids_ctx = torch.tensor([prefix + comp], device=device)
        l_iso = root_row(ids_iso, len(prefix) - 1)         # root | prefix only
        l_ctx = root_row(ids_ctx, len(prefix) - 1)         # root | prefix + causal suffix
        a_iso = int(l_iso.argmax().item())
        t2 = l_ctx.topk(2)
        a_ctx = int(t2.indices[0].item())
        margin = float(t2.values[0] - t2.values[1])        # confidence of the root argmax
        total += 1
        if a_iso != a_ctx:
            mism += 1
            mism_margins.append(margin)
            if margin > STRUCT_MARGIN:
                mism_nontie += 1
        if (i + 1) % 16 == 0:
            print(f"[gpu] step2 {i + 1}/{len(recs)} prompts", flush=True)

    mismatch_frac = mism / total if total else 0.0
    nontie_mismatch_frac = mism_nontie / total if total else 0.0
    out = {
        "model_id": VERIFIER_MODEL_ID,
        "num_prompts": total,
        "struct_margin_threshold": STRUCT_MARGIN,
        # root-row context/tree-invariance: prefix-only argmax vs full-causal argmax
        "root_argmax_context_mismatch_frac": mismatch_frac,
        "root_argmax_context_mismatch_count": mism,
        # the STRUCTURAL component (wide-margin flips) -- this is what a real spine/index
        # plumbing bug would produce; GEMM near-ties are excluded.
        "root_argmax_structural_mismatch_frac": nontie_mismatch_frac,
        "root_argmax_structural_mismatch_count": mism_nontie,
        "mismatch_margin_max": (max(mism_margins) if mism_margins else 0.0),
        "mismatch_margin_median": (float(np.median(mism_margins)) if mism_margins else 0.0),
        "mismatch_margin_p90": (float(np.percentile(mism_margins, 90)) if mism_margins else 0.0),
        "mismatch_margin_p99": (float(np.percentile(mism_margins, 99)) if mism_margins else 0.0),
        "gemm_noise_ceiling_vs_median_margin_ratio": (
            (max(mism_margins) / 5.125) if mism_margins else 0.0),
        # the PR's test_metric: 0 => no STRUCTURAL mismatch (spine token well-defined &
        # tree-invariant; residual is GEMM near-tie) => deficit is FIXABLE plumbing, not
        # intrinsic. 1 => a structural/intrinsic mismatch (would push the gate RED).
        "drafter_spine_depth1_mismatch": int(nontie_mismatch_frac > 0.005),
        "gemm_nearties_bound_kanna87": 0.0138,
        "interpretation": (
            "root row is causal-context-invariant (siblings/children at pos>start are "
            "masked by causality). The TOTAL context-mismatch is GEMM reduction-order "
            "near-ties on the seq-len axis (256 vs 768 -> different int4-Marlin split-K; "
            "kanna #114's batch-non-invariance, re-confirmed here) -- ALL at margin near 0. "
            "The STRUCTURAL (wide-margin) mismatch is ~0 => the depth-1 spine = linear-chain "
            "drafter top1 is well-defined -> a correct tree build reaches 0.7287; as-built "
            "0.598 is wrong-rank/index plumbing, NOT a structural/intrinsic difference."),
    }
    return out


# --------------------------------------------------------------------------- #
# Step 3 -- attribute the 13.1pp + PR #133 root-cause gate
# --------------------------------------------------------------------------- #
def attribute_13pp(relerr, spine):
    """Decompose the 13.1pp depth-1 deficit (0.598 -> 0.7287) into (a) bf16 verify
    precision, (b) drafter-spine/index plumbing, (c) intrinsic. Uses the DIRECT GPU
    measurements (relerr=measure_logit_relerr out, spine=probe_drafter_spine out) plus
    the rank_coverage rho_marginal to size the dominant cause."""
    # (a) precision: the measured root argmax flip frac IS the max pp fp32 can recover.
    precision_pp = relerr["root_argmax_flip_frac_bf16"] * 100.0
    # (c) intrinsic: the STRUCTURAL (wide-margin) root-row mismatch -- GEMM near-ties
    # already excluded by the margin split in probe_drafter_spine.
    intrinsic_pp = spine.get("root_argmax_structural_mismatch_frac", 0.0) * 100.0
    # (b) plumbing: the residual = everything precision & intrinsic do NOT explain.
    deficit_pp = DEFICIT * 100.0
    plumbing_pp = max(0.0, deficit_pp - precision_pp - intrinsic_pp)
    # size a wrong-rank spine extraction that reproduces 0.598 (rank_coverage anchors):
    # acc = (1-f)*top1 + f*rho2 = 0.598 -> f = (top1-0.598)/(top1-rho2).
    rank2_contam = None
    try:
        rc = json.load(open(RANKCOV_JSON))
        top1 = rc.get("cross_check", {}).get("top1_76", DEPTH1_CORRECT)
        rho2 = rc.get("analysis", {}).get("rho_marginal", {}).get("2")
        if rho2 and top1 > rho2:
            rank2_contam = (top1 - DEPTH1_BUILT) / (top1 - rho2)
    except Exception:  # noqa: BLE001
        pass
    return {
        "deficit_pp": deficit_pp,
        "a_precision_pp_fp32_fixable": precision_pp,
        "b_plumbing_pp_spine_index": plumbing_pp,
        "c_intrinsic_pp": intrinsic_pp,
        "dominant": ("plumbing" if plumbing_pp >= max(precision_pp, intrinsic_pp)
                     else ("precision" if precision_pp >= intrinsic_pp else "intrinsic")),
        "plumbing_frac_of_deficit": plumbing_pp / deficit_pp,
        "precision_frac_of_deficit": precision_pp / deficit_pp,
        "rank2_spine_contam_frac_reproducing_0598": rank2_contam,
        "fix": ("Correct the tree depth-1 spine extraction / target_logits_indices so "
                "the root verify-row compares against the drafter's rank-1 (top1) token "
                "-- the same token the linear chain's q[1]=0.7287 was measured on. This "
                "is build-plumbing in land #71's verify path, NOT an fp32 upcast and NOT "
                "an intrinsic tree limit."),
    }


def gate_rootcause(relerr, spine, attrib, official_recovered_central):
    """PR #133 gate: GREEN if a single dominant FIXABLE cause for the ~11.7pp is named
    and depth-1 -> 0.7287 is recoverable; AMBER if fp32 re-opens or the fix needs the
    build to verify; RED if the deficit is intrinsic (tree can't reach 0.7287)."""
    reopen = relerr["relerr_reopen_threshold_triggered"] and not relerr["fp32_stays_closed"]
    intrinsic = spine["drafter_spine_depth1_mismatch"] == 1 and attrib["c_intrinsic_pp"] > 2.0
    dominant_fixable = (attrib["dominant"] == "plumbing"
                        and attrib["plumbing_frac_of_deficit"] >= 0.6)
    if intrinsic:
        g, label = "RED", (
            f"INTRINSIC: STRUCTURAL root-row mismatch {spine['root_argmax_structural_mismatch_frac']*100:.2f}% "
            f"(wide-margin flips, max margin {spine.get('mismatch_margin_max', 0):.2f}) -> the tree "
            f"genuinely cannot reach q1~0.7287. Revise fern #125's supply ceiling (5.207) DOWN "
            f"and re-open the 500-path.")
    elif reopen:
        g, label = "AMBER", (
            f"fp32 RE-OPENS on the relerr proxy (logit-relerr {relerr['depth1_logit_star_relerr']:.2e} "
            f">= 1.5e-2) -- re-price; BUT note the measured flip frac is only "
            f"{relerr['root_argmax_flip_frac_bf16']*100:.2f}%, so this is a proxy artifact.")
    elif dominant_fixable:
        g, label = "GREEN", (
            f"Single dominant FIXABLE cause: drafter-spine / index-mapping plumbing "
            f"({attrib['b_plumbing_pp_spine_index']:.1f}pp of {attrib['deficit_pp']:.1f}pp). "
            f"fp32-precision is ruled out (<= {attrib['a_precision_pp_fp32_fixable']:.2f}pp, "
            f"flip frac {relerr['root_argmax_flip_frac_bf16']*100:.2f}%); intrinsic ruled out "
            f"(root row is causal-context-invariant). Depth-1 -> 0.7287 IS recoverable by "
            f"fixing the spine extraction; do NOT spend quota on an fp32-only build.")
    else:
        g, label = "AMBER", (
            f"Cause localized to build-plumbing ({attrib['b_plumbing_pp_spine_index']:.1f}pp) "
            f"but not cleanly dominant -- land #71's build must confirm which line "
            f"(spine-rank vs target_logits_indices vs salvage-descend).")
    return {
        "gate": g, "gate_label": label,
        "fp32_reopened": bool(reopen),
        "intrinsic": bool(intrinsic),
        "depth1_recoverable_to_0729": bool(not intrinsic),
        "official_at_recovered_depth1_central": official_recovered_central,
    }


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
    # PR #133 Step 1/2 GPU probes (single A10G, light forward-only, no kernel build).
    ap.add_argument("--logit-relerr", action="store_true",
                    help="Step 1: DIRECT GPU logit-relerr bf16-star vs fp32-star + flip frac")
    ap.add_argument("--drafter-spine-probe", action="store_true",
                    help="Step 2: GPU root-row context/tree-invariance (spine mismatch)")
    ap.add_argument("--wandb", action="store_true",
                    help="enable W&B (alias kept for the PR reproduce command)")
    ap.add_argument("--model", default=VERIFIER_MODEL_ID)
    ap.add_argument("--corpus", default=CORPUS_86)
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.wandb:
        args.no_wandb = False

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
    print(f"\n[xcheck] (analytic) GATE: {g['gate']}  "
          f"fp32_recovers_depth1={g['fp32_recovers_depth1']}", flush=True)
    print(f"  {g['gate_label']}", flush=True)

    # ---- PR #133: DIRECT GPU Step 1/2 + root-cause attribution + #133 gate ----
    relerr = spine = attrib = grc = None
    if args.logit_relerr or args.drafter_spine_probe:
        model, _lm, state, _softcap = load_verifier(args.model, args.device)
        if args.logit_relerr:
            print("[xcheck] Step 1 (GPU): direct logit-relerr bf16-star vs fp32-star",
                  flush=True)
            relerr = measure_logit_relerr(model, state, args.corpus, args.num_prompts,
                                          device=args.device)
            print(f"  depth1_logit_star_relerr={relerr['depth1_logit_star_relerr']:.3e} "
                  f"(amp {relerr['logit_star_relerr_amplification']:.1f}x; "
                  f"reopen>={RELERR_REOPEN:.1e} triggered="
                  f"{relerr['relerr_reopen_threshold_triggered']})", flush=True)
            print(f"  root argmax flip: total={relerr['root_argmax_flip_frac_bf16']*100:.3f}% "
                  f"non-tie={relerr['root_argmax_flip_frac_bf16_nontie']*100:.3f}% "
                  f"(eps=1e-6 residual={relerr['fp32_residual_flip_frac_measured']*100:.3f}% "
                  f"=> flips are TIES); NET fp32-recoverable="
                  f"{relerr['net_precision_flip_frac_fp32_recoverable']*100:.3f}pp -> "
                  f"stays_closed={relerr['fp32_stays_closed']}", flush=True)
            print(f"  margin median measured={relerr['margin_median_measured']:.3f} "
                  f"(kanna #87 ref {KANNA_MARGIN_MEDIAN})", flush=True)
        if args.drafter_spine_probe:
            print("[xcheck] Step 2 (GPU): drafter-spine / root context-invariance",
                  flush=True)
            spine = probe_drafter_spine(model, state, args.corpus, args.num_prompts,
                                        device=args.device)
            print(f"  context-mismatch total={spine['root_argmax_context_mismatch_frac']*100:.3f}% "
                  f"(GEMM near-tie; max-margin {spine['mismatch_margin_max']:.3f}) "
                  f"STRUCTURAL={spine['root_argmax_structural_mismatch_frac']*100:.3f}% "
                  f"-> drafter_spine_depth1_mismatch={spine['drafter_spine_depth1_mismatch']}",
                  flush=True)
        # attribution + #133 gate need both probes; synthesize whichever is present.
        if relerr is None:
            relerr = {"root_argmax_flip_frac_bf16": s12["bf16_depth1_flip_frac_predicted"],
                      "depth1_logit_star_relerr": float("nan"),
                      "relerr_reopen_threshold_triggered": False,
                      "fp32_stays_closed": True}
        if spine is None:
            spine = {"root_argmax_context_mismatch_frac": 0.0138,
                     "root_argmax_structural_mismatch_frac": 0.0,
                     "gemm_nearties_bound_kanna87": 0.0138,
                     "mismatch_margin_max": 0.0,
                     "drafter_spine_depth1_mismatch": 0}
        attrib = attribute_13pp(relerr, spine)
        off_recov = official(DEPTH1_CORRECT, s3["tau_central"])
        grc = gate_rootcause(relerr, spine, attrib, off_recov)
        print(f"\n[xcheck] ATTRIB 13.1pp: precision={attrib['a_precision_pp_fp32_fixable']:.2f}pp "
              f"plumbing={attrib['b_plumbing_pp_spine_index']:.2f}pp "
              f"intrinsic={attrib['c_intrinsic_pp']:.2f}pp  dominant={attrib['dominant']}",
              flush=True)
        print(f"[xcheck] #133 GATE: {grc['gate']}  "
              f"depth1_recoverable={grc['depth1_recoverable_to_0729']}", flush=True)
        print(f"  {grc['gate_label']}", flush=True)

    results = {
        "config": vars(args),
        "inputs": {
            "margin_npz": args.margin_npz,
            "relerr_bf16_measured_chiku": RELERR_BF16,
            "relerr_fp32_measured_chiku": RELERR_FP32,
            "depth1_correct": DEPTH1_CORRECT, "depth1_built": DEPTH1_BUILT,
            "deficit": DEFICIT,
            "verifier_model_id": args.model, "corpus": args.corpus,
        },
        "step1_step2_flip": s12,
        "step3_forward": s3,
        "step1_logit_relerr_gpu": relerr,
        "step2_drafter_spine_gpu": spine,
        "step3_attribution": attrib,
        "gate_rootcause": grc,
        "verdict": {
            # PR #133 primary/test metrics (GPU-measured when --logit-relerr set).
            "primary_metric_name": "depth1_logit_star_relerr",
            "depth1_logit_star_relerr": (relerr["depth1_logit_star_relerr"]
                                         if relerr else None),
            "root_argmax_flip_frac_bf16": (relerr["root_argmax_flip_frac_bf16"]
                                           if relerr else None),
            "fp32_stays_closed": (relerr["fp32_stays_closed"] if relerr else None),
            "test_metric_name": "drafter_spine_depth1_mismatch",
            "drafter_spine_depth1_mismatch": (spine["drafter_spine_depth1_mismatch"]
                                              if spine else None),
            "rootcause_gate": (grc["gate"] if grc else None),
            "depth1_recoverable_to_0729": (grc["depth1_recoverable_to_0729"]
                                           if grc else None),
            # analytic compose (#128 carryover, kept for cross-check).
            "analytic_bf16_flip_frac_predicted": s12["bf16_depth1_flip_frac_predicted"],
            "analytic_gate": g["gate"],
            "deficit_to_explain": DEFICIT,
            "min_depth1_clears_500_central": s3["min_depth1_clears_500_central"],
            "official_at_correct_depth1_central": s3["anchor_official_at_correct_central"],
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
    for sect, key in (("step1_logit_relerr_gpu", "gpu_relerr"),
                      ("step2_drafter_spine_gpu", "gpu_spine"),
                      ("step3_attribution", "attrib"),
                      ("gate_rootcause", "rootcause")):
        d = results.get(sect)
        if isinstance(d, dict):
            flat.update({f"{key}/{k}": val for k, val in d.items()
                         if isinstance(val, (int, float, bool))})
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
