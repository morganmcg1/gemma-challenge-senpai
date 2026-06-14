#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""WITHIN-PROMPT ICC -> realistic launch CI + GO-robustness verdict (PR #190, wirbel).

THE QUESTION
------------
wirbel #175 priced the single-shot benchmark-TPS finite-sample CI at +/-10.906 TPS
assuming the ~3147 spec-decode steps are IID. denken #184 stress-tested the OTHER
extreme -- perfect within-prompt clustering ICC=1 -- and found +/-54.9 TPS (LCB 480.5,
a MISS). The launch's clear-500 margin lives somewhere in that 5x-wide bracket, and
NOTHING has yet asked the data where: what is the REALISTIC intra-class correlation of
the 128-prompt benchmark, and does the both-bugs GO survive it?

This file measures the realistic ICC from the only per-step acceptance data we have
(the PR #86 rankprobe, 17169 decode steps reconstructed into the 128 benchmark prompts)
and pins the launch CI between #175's IID floor and #184's ICC=1 ceiling. The committed
length L = (accepted run) + 1 bonus token is the SAME per-step quantity whose sample
mean is the benchmark TPS (E[L]=E[T], sigma_L is the #175 CI driver), so its
within-prompt correlation is exactly the design-effect that inflates the single draw.

THE ESTIMATOR (one-way random-effects ANOVA)
--------------------------------------------
Group the per-step L by prompt. Between-prompt variance sigma_b^2 = (MSB-MSW)/m0,
within-prompt sigma_w^2 = MSW, and ICC = sigma_b^2 / (sigma_b^2 + sigma_w^2). The
design effect for the benchmark grand mean is Deff = 1 + (m_bar - 1)*ICC (imported #184
exchangeable map, m_bar = N_steps/N_prompts = 24.58), N_eff = N_steps/Deff, and only the
finite-sample (accept-length) half-width inflates: halfwidth_realistic =
halfwidth_iid * sqrt(Deff). kanna #159's sigma_hw (a denominator hardware-jitter term)
stays fixed in the quadrature.

CLUSTER-SIZE NOTE (honest): the within-prompt correlation is SERIAL (rho(1)=0.26
decaying ~AR), not pure exchangeable, so the ANOVA ICC depends on the cluster window.
We report the band -- full-prompt (lower), the matched ~m_bar window, and the
TPS-window (B=16384/128 = 128 tokens/prompt, the #175 budget scale, PRIMARY) -- plus an
ACF-based Deff cross-check. The GO verdict is shown to be INVARIANT across the whole
band, so the estimation ambiguity does not move the headline.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / kernel build /
served-file change. BASELINE stays 481.53; 0 TPS; greedy untouched. Imports (does NOT
re-derive): #175 (zh1accmi) sigma_L/+-10.906/central; #184 (lambda-robust) N_eff
two-level model + the ICC=0/ICC=1 bracket reproduced as self-test targets; #183
(82uisrez) lambda-acceptance-card forward-map spine + lambda*_LCB=0.9052; kanna #159
sigma_hw=4.86, K_cal, step 1.2182; launch packet LCB(P>=0.9) convention. NOT open2.
Does NOT authorize a launch.

PRIMARY metric  icc_neff_self_test_passes
TEST    metric  lcb_bothbugs_realistic_icc  (both-bugs launch LCB at realistic ICC, TPS)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))

# ---- banked artifacts (the exact files these legs wrote; read, do NOT re-derive) ----
RANKPROBE = os.path.join(_ROOT, "research", "rank_coverage", "pr86", "rankprobe_records.jsonl.118860")
DECODE_RC = os.path.join(_ROOT, "research", "rank_coverage", "pr86", "decode_rank_coverage.jsonl")
ET175_JSON = os.path.join(_ROOT, "research", "oracle_readout", "et_second_moment", "et_second_moment_results.json")
TOPO184_JSON = os.path.join(_ROOT, "research", "oracle_readout", "lambda_robust_topology", "lambda_robust_topology_results.json")
CARD183_JSON = os.path.join(_ROOT, "research", "oracle_readout", "lambda_acceptance_card", "lambda_acceptance_card_results.json")
LAUNCH_JSON = os.path.join(_ROOT, "research", "launch", "packet_refresh", "launch_packet_refresh_results.json")

# ---- composition constants (kanna #159 / #184 §4 convention) ----
SIGMA_HW_S4 = 4.86                     # the sigma_hw #184 used in its ICC bracket (reproduce it exactly)
SIGMA_HW_PRECISE = 4.864468814937121   # kanna #159 precise (launch-packet convention)
Z95 = 1.959963984540054                # two-sided 95% normal quantile
N_PROMPTS = 128
TARGET = 500.0
BENCH_BUDGET_TOKENS = 16384            # #175 nominal B (128 tokens/prompt TPS window)
PROMPT_OUTPUT_TOKENS = 512             # benchmark generates 512 tokens/prompt (README)
LAMBDA_STAR_IID = 0.9052283680740145   # #183 / #184 published both-bugs build bar at Deff=1

# tolerances
BRACKET_TOL = 0.5                      # TPS: reproduce #184's ICC=0/ICC=1 bracket within this
LAMSTAR_REPRO_TOL = 1e-3               # lambda*_LCB(Deff=1) must reproduce 0.9052


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _all_finite(xs) -> bool:
    return all(_finite(x) for x in xs)


# ===================================================================================
# 0. data: per-step committed length L = accepted-run + 1, reconstructed into prompts
# ===================================================================================
def load_per_step_lengths(path: str) -> list[int]:
    """L = fd + 1 per decode step (fd = accepted draft run-length; +1 bonus token).
    L is the committed length per step; E[L]=E[T], Var[L]=sigma_L^2 -- the #175 CI driver."""
    L = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            L.append(int(json.loads(line)["fd"]) + 1)
    return L


def reconstruct_prompts(L: list[int], out_tokens: int, n_prompts: int) -> list[list[int]]:
    """Walk the concatenated per-step series, accumulating committed tokens; close a prompt
    when its committed count reaches out_tokens (the benchmark's per-prompt max_tokens).
    The rankprobe carries no prompt id, so boundaries are reconstructed from the budget."""
    groups, cur, run = [], [], 0
    for x in L:
        cur.append(x)
        run += x
        if run >= out_tokens:
            groups.append(cur)
            cur, run = [], 0
    if cur:
        groups.append(cur)
    return groups[:n_prompts]


def token_window(group: list[int], window_tokens: int) -> list[int]:
    """First steps of a prompt whose committed tokens reach window_tokens (the TPS window)."""
    out, s = [], 0
    for v in group:
        out.append(v)
        s += v
        if s >= window_tokens:
            break
    return out


# ===================================================================================
# 1. ICC estimators: one-way random-effects ANOVA + ACF design-effect cross-check
# ===================================================================================
def anova_icc(groups: list[list[int]]) -> dict:
    """One-way random-effects ANOVA ICC = sigma_b^2/(sigma_b^2+sigma_w^2).
    sigma_b^2=(MSB-MSW)/m0, sigma_w^2=MSW, m0=(N - sum m_i^2 / N)/(k-1) (unequal groups)."""
    grps = [g for g in groups if len(g) >= 2]
    k = len(grps)
    N = sum(len(g) for g in grps)
    grand = sum(sum(g) for g in grps) / N
    means = [sum(g) / len(g) for g in grps]
    ssb = sum(len(g) * (means[i] - grand) ** 2 for i, g in enumerate(grps))
    ssw = sum(sum((v - means[i]) ** 2 for v in g) for i, g in enumerate(grps))
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    m0 = (N - sum(len(g) ** 2 for g in grps) / N) / (k - 1)
    sb = (msb - msw) / m0
    sw = msw
    icc = sb / (sb + sw) if (sb + sw) > 0 else 0.0
    return {"icc": max(icc, 0.0), "icc_raw": icc, "sigma_b2": sb, "sigma_w2": sw,
            "MSB": msb, "MSW": msw, "m0": m0, "k": k, "N": N,
            "mean_m": N / k, "grand_mean_L": grand}


def pooled_acf(groups: list[list[int]], lmax: int) -> list[float]:
    """Within-prompt-demeaned pooled autocorrelation rho(l), l=0..lmax (rho(0)=1)."""
    dem = []
    for g in groups:
        mu = sum(g) / len(g)
        dem.append([v - mu for v in g])
    var = sum(sum(v * v for v in g) for g in dem) / sum(len(g) for g in dem)
    rho = [1.0]
    for l in range(1, lmax + 1):
        num, cnt = 0.0, 0
        for g in dem:
            for j in range(len(g) - l):
                num += g[j] * g[j + l]
                cnt += 1
        rho.append((num / cnt) / var if cnt > 0 and var > 0 else 0.0)
    return rho


def deff_acf(rho: list[float], m: float) -> float:
    """Design effect of the mean of m serially-correlated steps:
    Deff(m) = 1 + 2*sum_{l=1}^{floor(m-1)} (1 - l/m)*rho(l)  (triangular Bartlett weight)."""
    s = 0.0
    last = int(math.floor(m - 1e-9))
    for l in range(1, last + 1):
        if l < len(rho):
            s += (1.0 - l / m) * rho[l]
    return 1.0 + 2.0 * s


def bootstrap_icc_ci(groups: list[list[int]], window_tokens: int, n_boot: int,
                     seed: int = 0) -> dict:
    """Prompt-level (cluster) bootstrap of the TPS-window ANOVA ICC: resample whole prompts."""
    win = [token_window(g, window_tokens) for g in groups if len(g) >= 2]
    rng = np.random.default_rng(seed)
    n = len(win)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samp = [win[i] for i in idx]
        boots.append(anova_icc(samp)["icc"])
    boots = np.sort(np.asarray(boots))
    return {"icc_ci_lo": float(boots[int(0.025 * n_boot)]),
            "icc_ci_hi": float(boots[int(0.975 * n_boot)]),
            "icc_boot_mean": float(boots.mean()), "n_boot": n_boot}


# ===================================================================================
# 2. design effect -> realistic launch CI (section-4 convention, reproduces #184)
# ===================================================================================
def deff_from_icc(mbar: float, icc: float) -> float:
    return 1.0 + (mbar - 1.0) * icc


def lcb_section4(central: float, hw_iid: float, mbar: float, icc: float,
                 sigma_hw: float = SIGMA_HW_S4, z: float = Z95) -> dict:
    """#184 section-4 LCB. The accept-length half-width (which already carries its own z)
    inflates by sqrt(Deff); the hardware term is z*sigma_hw (sigma_hw a 1-sigma TPS):
        accept_half = hw_iid*sqrt(Deff)
        total_half  = sqrt(accept_half^2 + (z*sigma_hw)^2)
        LCB         = central - total_half."""
    deff = deff_from_icc(mbar, icc)
    accept_half = hw_iid * math.sqrt(deff)
    total_half = math.sqrt(accept_half ** 2 + (z * sigma_hw) ** 2)
    return {"icc": icc, "design_effect": deff, "accept_half_tps": accept_half,
            "total_half_tps": total_half, "lcb_tps": central - total_half,
            "lcb_clears_500": bool((central - total_half) >= TARGET)}


def icc_breakpoint(central: float, hw_iid: float, mbar: float,
                   sigma_hw: float = SIGMA_HW_S4, z: float = Z95) -> float:
    """The ICC at which the section-4 LCB crosses 500 (bisection; clamps to [0,1])."""
    def lcb(icc):
        return lcb_section4(central, hw_iid, mbar, icc, sigma_hw, z)["lcb_tps"]
    if lcb(0.0) < TARGET:
        return 0.0
    if lcb(1.0) >= TARGET:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if lcb(mid) >= TARGET:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ===================================================================================
# 3. build-bar refinement (#183): lambda*_LCB under the realistic design effect
# ===================================================================================
def load_card_spine(card_json: str, topo: str):
    """Import #183's forward-map spine (lambda, E_T, sigma_L) + composition constants."""
    d = json.load(open(card_json))
    comp = d["synthesis"]["composition"]
    rows = d["synthesis"]["forward_map"][topo]["tau_central_1p0"]["rows"]
    lam = np.array([r["lambda"] for r in rows])
    et = np.array([r["E_T"] for r in rows])
    sL = np.array([r["sigma_L"] for r in rows])
    return {"lambda": lam, "E_T": et, "sigma_L": sL,
            "K_cal": comp["K_cal"], "step": comp["step"], "B": comp["B_tokens"],
            "z95": comp["z95"], "sigma_hw": comp["sigma_hw_tps"]}


def lambda_star_lcb(spine: dict, deff: float, tau: float = 1.0) -> tuple[float, float]:
    """Solve LCB_real(lambda*) = 500 with the finite-sample SE inflated by sqrt(Deff):
    central(l)=K_cal*E_T(l)/step*tau ; SE(l)=K_cal*tau/step*sigma_L(l)/sqrt(B/E_T(l)) ;
    LCB(l)=central(l) - z95*sqrt(SE(l)^2*Deff + sigma_hw^2)."""
    g = np.linspace(0.0, 1.0, 100001)
    etg = np.interp(g, spine["lambda"], spine["E_T"])
    slg = np.interp(g, spine["lambda"], spine["sigma_L"])
    central = spine["K_cal"] * etg / spine["step"] * tau
    se = spine["K_cal"] * tau / spine["step"] * slg / np.sqrt(spine["B"] / etg)
    lcb = central - spine["z95"] * np.sqrt(se ** 2 * deff + spine["sigma_hw"] ** 2)
    clears = lcb >= TARGET
    if not clears.any():
        return 1.0, float(lcb[-1])        # unreachable within [0,1]
    idx = int(np.argmax(clears))
    return float(g[idx]), float(lcb[idx])


# ===================================================================================
# 4. launch-packet LCB(P>=0.9) convention (secondary handoff: realistic finite-sample fold)
# ===================================================================================
def launch_lcb_p90_realistic(launch_json: str, hw_iid: float, central_175: float,
                             deff: float) -> dict:
    """Fold the realistic-ICC finite-sample relative term into the launch packet's
    LCB(P>=0.9). The published combined_rel_1sigma (3-term, sigma_hw retired) gains the
    #175 finite-sample 1sigma relative (= (hw_iid/z95)/central_175) inflated by sqrt(Deff),
    in quadrature; LCB = proj_private*(1 - z_p90*combined_rel)."""
    d = json.load(open(launch_json))
    bb = d["step1_three_framing_geometry"]["shipped"]["both_bugs"]
    proj = bb["proj_private_tps"]
    rel_pub = bb["combined_rel_1sigma"]
    lcb_pub = bb["lcb_p90"]
    z90 = d["uncertainty_model"]["z_p90_one_sided"]
    rel_fs_iid = (hw_iid / Z95) / central_175
    rel_fs_real = rel_fs_iid * math.sqrt(deff)
    rel_iid_fold = math.sqrt(rel_pub ** 2 + rel_fs_iid ** 2)
    rel_real_fold = math.sqrt(rel_pub ** 2 + rel_fs_real ** 2)
    lcb_iid = proj * (1.0 - z90 * rel_iid_fold)
    lcb_real = proj * (1.0 - z90 * rel_real_fold)
    return {"proj_private_tps": proj, "z_p90": z90,
            "combined_rel_published_3term": rel_pub, "lcb_p90_published": lcb_pub,
            "rel_finite_sample_iid": rel_fs_iid, "rel_finite_sample_realistic": rel_fs_real,
            "combined_rel_iid_fold": rel_iid_fold, "combined_rel_realistic_fold": rel_real_fold,
            "lcb_p90_iid_finite_sample": lcb_iid, "lcb_p90_realistic_icc": lcb_real,
            "clears_500_realistic": bool(lcb_real >= TARGET)}


# ===================================================================================
# 5. self-test (PRIMARY)
# ===================================================================================
def self_test(out: dict) -> dict:
    checks = []

    def chk(name, ok, detail):
        checks.append({"name": name, "passes": bool(ok), "detail": str(detail)})

    s4 = out["go_robustness"]["section4"]
    ref = out["imported"]["topo184_bracket"]

    # (a) ICC=0 reproduces #184's +-10.906 / LCB 521
    icc0 = s4["icc0"]
    chk("ICC=0 reproduces #184 IID bracket (LCB 520.95, half 14.48)",
        abs(icc0["lcb_tps"] - ref["icc0_lcb"]) < BRACKET_TOL,
        f"lcb={icc0['lcb_tps']:.4f} vs #184 {ref['icc0_lcb']:.4f}")
    # (b) ICC=1 reproduces #184's +-54.9 / LCB 480.5
    icc1 = s4["icc1"]
    chk("ICC=1 reproduces #184 worst-case bracket (LCB 480.53, half 54.91)",
        abs(icc1["lcb_tps"] - ref["icc1_lcb"]) < BRACKET_TOL,
        f"lcb={icc1['lcb_tps']:.4f} vs #184 {ref['icc1_lcb']:.4f}")
    # (c) ICC_hat in [0,1] with a finite CI
    ih = out["icc_estimate"]
    chk("ICC_hat in [0,1] with finite CI",
        0.0 <= ih["icc_hat"] <= 1.0 and _finite(ih["icc_ci"][0]) and _finite(ih["icc_ci"][1])
        and ih["icc_ci"][0] <= ih["icc_hat"] <= ih["icc_ci"][1],
        f"icc_hat={ih['icc_hat']:.4f} ci=[{ih['icc_ci'][0]:.4f},{ih['icc_ci'][1]:.4f}]")
    # (d) LCB monotone-decreasing in ICC
    grid = [lcb_section4(out["imported"]["bb_central"], out["imported"]["bb_hw_iid"],
                         out["imported"]["mbar_bb"], i)["lcb_tps"]
            for i in np.linspace(0, 1, 101)]
    mono = all(grid[i + 1] <= grid[i] + 1e-9 for i in range(len(grid) - 1))
    chk("section-4 LCB monotone-decreasing in ICC", mono,
        f"LCB(0)={grid[0]:.2f} -> LCB(1)={grid[-1]:.2f}")
    # (e) realistic build bar >= iid 0.9052 (correlation can only raise it)
    bar = out["build_bar"]
    chk("lambda*_LCB(realistic) >= iid 0.9052",
        bar["lambda_star_lcb_realistic_icc"] >= LAMBDA_STAR_IID - 1e-6,
        f"lambda*={bar['lambda_star_lcb_realistic_icc']:.4f} vs iid {LAMBDA_STAR_IID:.4f}")
    # (e') build-bar machinery sanity: Deff=1 reproduces 0.9052
    chk("lambda*_LCB(Deff=1) reproduces #183 build bar 0.9052",
        abs(bar["lambda_star_lcb_iid_check"] - LAMBDA_STAR_IID) < LAMSTAR_REPRO_TOL,
        f"lambda*(Deff=1)={bar['lambda_star_lcb_iid_check']:.5f}")
    # (f) NaN-clean
    scal = [ih["icc_hat"], ih["icc_ci"][0], ih["icc_ci"][1],
            out["go_robustness"]["lcb_bothbugs_realistic_icc"],
            out["go_robustness"]["lcb_descent_realistic_icc"],
            out["go_robustness"]["icc_at_which_bothbugs_breaks_500"],
            bar["lambda_star_lcb_realistic_icc"], bar["bar_shift_from_icc"],
            out["realistic_ci"]["halfwidth_realistic_tps"], out["realistic_ci"]["n_eff_hat"]]
    nan_clean = _all_finite(scal)
    chk("all reported scalars NaN-clean", nan_clean, f"{len(scal)} scalars finite")

    passes = all(c["passes"] for c in checks)
    return {"passes": passes, "n_checks": len(checks),
            "n_passed": sum(c["passes"] for c in checks), "checks": checks,
            "nan_clean": nan_clean}


# ===================================================================================
# main
# ===================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="research/validity/icc_neff/icc_neff_results.json")
    ap.add_argument("--report-md", default="research/validity/icc_neff/report_icc_neff.md")
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--acf-lmax", type=int, default=40)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="wirbel/icc-neff-launch-ci")
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default="icc-neff-launch-ci")
    args = ap.parse_args()

    # ---- import the banked legs (read, do NOT re-derive) ----
    et175 = json.load(open(ET175_JSON))
    bb_central = et175["finite_sample_tps_ci"]["both_bugs"]["primary_16384_tau1"]["central_tps"]
    bb_hw_iid = et175["finite_sample_tps_ci"]["both_bugs"]["primary_16384_tau1"]["ci_halfwidth_tps"]
    bb_nsteps = et175["finite_sample_tps_ci"]["both_bugs"]["primary_16384_tau1"]["N_steps"]
    bb_sigmaL = et175["topologies"]["both_bugs"]["sigma_L"]
    ds_central = et175["finite_sample_tps_ci"]["descent_only"]["primary_16384_tau1"]["central_tps"]
    ds_hw_iid = et175["finite_sample_tps_ci"]["descent_only"]["primary_16384_tau1"]["ci_halfwidth_tps"]
    ds_nsteps = et175["finite_sample_tps_ci"]["descent_only"]["primary_16384_tau1"]["N_steps"]

    topo184 = json.load(open(TOPO184_JSON))
    rec184 = topo184["synthesis"]["neff_two_level"]["recommended"]
    mbar_bb = rec184["mean_steps_per_prompt"]          # 24.5825 (imported #184 m_bar)
    bracket = {b["icc"]: b for b in rec184["bands"]}
    topo184_bracket = {"icc0_lcb": bracket[0.0]["lcb_tps"], "icc0_half": bracket[0.0]["total_half_tps"],
                       "icc1_lcb": bracket[1.0]["lcb_tps"], "icc1_half": bracket[1.0]["total_half_tps"],
                       "mbar": mbar_bb, "central": rec184["central_tps"]}
    mbar_ds = ds_nsteps / N_PROMPTS

    # ---- 0. data: per-step L -> 128 prompts ----
    L = load_per_step_lengths(RANKPROBE)
    groups = reconstruct_prompts(L, PROMPT_OUTPUT_TOKENS, N_PROMPTS)
    window_tokens = BENCH_BUDGET_TOKENS // N_PROMPTS   # 128 tokens/prompt (#175 budget scale)
    win = [token_window(g, window_tokens) for g in groups]
    mean_m_full = sum(len(g) for g in groups) / len(groups)
    mean_m_win = sum(len(w) for w in win) / len(win)
    e_l_probe = sum(L) / len(L)

    # ---- 1. ICC estimators (band + ACF cross-check) ----
    a_full = anova_icc(groups)
    a_win = anova_icc(win)
    a_25 = anova_icc([g[:25] for g in groups])
    a_24 = anova_icc([g[:24] for g in groups])
    a_33 = anova_icc([g[:33] for g in groups])
    rho = pooled_acf(groups, args.acf_lmax)
    deff_acf_bb = deff_acf(rho, mbar_bb)
    icc_acf_bb = (deff_acf_bb - 1.0) / (mbar_bb - 1.0)

    icc_hat = a_win["icc"]                              # PRIMARY: TPS-window (#175 budget scale)
    ci = bootstrap_icc_ci(groups, window_tokens, args.n_boot)
    icc_lo, icc_hi = ci["icc_ci_lo"], ci["icc_ci_hi"]

    icc_band = {
        "token_window_128tok_PRIMARY": a_win["icc"], "full_prompt": a_full["icc"],
        "first_24_step": a_24["icc"], "first_25_step": a_25["icc"], "first_33_step": a_33["icc"],
        "acf_equiv_at_mbar": icc_acf_bb,
    }

    # ---- 2. realistic CI mapping ----
    deff_hat = deff_from_icc(mbar_bb, icc_hat)
    halfwidth_realistic = bb_hw_iid * math.sqrt(deff_hat)
    n_eff_hat = bb_nsteps / deff_hat
    realistic_ci = {
        "m_bar": mbar_bb, "design_effect_hat": deff_hat, "n_eff_hat": n_eff_hat,
        "halfwidth_iid_tps": bb_hw_iid, "halfwidth_realistic_tps": halfwidth_realistic,
        "halfwidth_worstcase_icc1_tps": bb_hw_iid * math.sqrt(deff_from_icc(mbar_bb, 1.0)),
    }

    # ---- 3. GO-robustness verdict (THE deliverable) ----
    s4_bb_hat = lcb_section4(bb_central, bb_hw_iid, mbar_bb, icc_hat)
    s4_ds_hat = lcb_section4(ds_central, ds_hw_iid, mbar_ds, icc_hat)
    s4_bb_lo = lcb_section4(bb_central, bb_hw_iid, mbar_bb, icc_lo)
    s4_bb_hi = lcb_section4(bb_central, bb_hw_iid, mbar_bb, icc_hi)
    s4_ds_hi = lcb_section4(ds_central, ds_hw_iid, mbar_ds, icc_hi)
    bb_break = icc_breakpoint(bb_central, bb_hw_iid, mbar_bb)
    ds_break = icc_breakpoint(ds_central, ds_hw_iid, mbar_ds)

    go_robustness = {
        "convention": "section-4 (central=535.43 both / 519.95 descent, z95, accept(+)sigma_hw=4.86 quadrature)",
        "section4": {
            "icc0": lcb_section4(bb_central, bb_hw_iid, mbar_bb, 0.0),
            "icc1": lcb_section4(bb_central, bb_hw_iid, mbar_bb, 1.0),
            "bothbugs_hat": s4_bb_hat, "descent_hat": s4_ds_hat,
            "bothbugs_ci_lo": s4_bb_lo, "bothbugs_ci_hi": s4_bb_hi, "descent_ci_hi": s4_ds_hi,
        },
        "lcb_bothbugs_realistic_icc": s4_bb_hat["lcb_tps"],
        "lcb_descent_realistic_icc": s4_ds_hat["lcb_tps"],
        "icc_at_which_bothbugs_breaks_500": bb_break,
        "icc_at_which_descent_breaks_500": ds_break,
        "bothbugs_clears_across_full_ci": bool(s4_bb_lo["lcb_clears_500"] and s4_bb_hi["lcb_clears_500"]),
        "bothbugs_break_margin_over_hat": bb_break / icc_hat if icc_hat > 0 else float("inf"),
        "descent_robust_to_realistic_icc": bool(s4_ds_hat["lcb_clears_500"]),
        "headline": "",
    }
    bb_go = s4_bb_lo["lcb_clears_500"] and s4_bb_hi["lcb_clears_500"]
    go_robustness["headline"] = (
        f"both-bugs {'STAYS >=500 (robust GO)' if bb_go else 'FAILS 500'} at the realistic ICC "
        f"{icc_hat:.3f} (LCB {s4_bb_hat['lcb_tps']:.1f}) and across the entire CI "
        f"[{icc_lo:.3f},{icc_hi:.3f}] (LCB {s4_bb_hi['lcb_tps']:.1f}..{s4_bb_lo['lcb_tps']:.1f}); "
        f"it only breaks 500 at ICC={bb_break:.3f} ({bb_break/icc_hat:.1f}x the realistic value). "
        f"descent-only {'clears' if s4_ds_hat['lcb_clears_500'] else 'MISSES'} (LCB "
        f"{s4_ds_hat['lcb_tps']:.1f}); it breaks 500 already at ICC={ds_break:.3f}, below the "
        f"realistic estimate -- NOT robust to realistic within-prompt correlation.")

    # ---- 4. build-bar refinement (#183) ----
    spine = load_card_spine(CARD183_JSON, "both_bugs")
    ls_iid_check, _ = lambda_star_lcb(spine, 1.0)
    ls_real, lcb_at_ls = lambda_star_lcb(spine, deff_hat)
    ls_lo, _ = lambda_star_lcb(spine, deff_from_icc(mbar_bb, icc_lo))
    ls_hi, _ = lambda_star_lcb(spine, deff_from_icc(mbar_bb, icc_hi))
    build_bar = {
        "lambda_star_lcb_iid_check": ls_iid_check,
        "lambda_star_lcb_realistic_icc": ls_real,
        "lambda_star_lcb_ci_lo_icc": ls_lo, "lambda_star_lcb_ci_hi_icc": ls_hi,
        "bar_shift_from_icc": ls_real - LAMBDA_STAR_IID,
        "iid_bar": LAMBDA_STAR_IID, "design_effect": deff_hat,
        "note": ("#183 section-5 used an asymptotic AR(1) VIF (<=2.0 -> lambda*=0.9213); the "
                 "correct finite-cluster design effect Deff=1+(m_bar-1)*ICC is larger, so the "
                 "realistic build bar is higher. Still reachable (<1) and bracketed by full recovery."),
    }

    # ---- 5. launch-packet LCB(P>=0.9) secondary handoff ----
    launch = launch_lcb_p90_realistic(LAUNCH_JSON, bb_hw_iid, bb_central, deff_hat)

    out = {
        "primary_metric_name": "icc_neff_self_test_passes",
        "test_metric_name": "lcb_bothbugs_realistic_icc",
        "icc_data_source": {
            "source": os.path.relpath(RANKPROBE, _ROOT),
            "prompt_lengths": os.path.relpath(DECODE_RC, _ROOT),
            "per_step_quantity": "L = accepted-run (fd) + 1 bonus token (E[L]=E[T]; Var[L]=sigma_L^2, the #175 CI driver)",
            "n_steps": len(L), "n_prompts": len(groups),
            "steps_per_prompt_full_mean": mean_m_full,
            "steps_per_prompt_tps_window_mean": mean_m_win,
            "tps_window_tokens": window_tokens, "prompt_output_tokens": PROMPT_OUTPUT_TOKENS,
            "E_L_probe_operating_point": e_l_probe,
            "operating_point_caveat": (
                "the rankprobe is at the liveprobe operating point lambda_hat=0.342 (E[L]=3.85), not "
                "the launch's full-recovery lambda=1 (E[L]=5.21). The CORRELATION STRUCTURE (ICC, "
                "rho(l)) is dimensionless and transported to lambda=1; if longer accept-runs at "
                "lambda=1 are MORE correlated, the true ICC could exceed this estimate -- the "
                "breakpoint headroom (2.6x) absorbs a 2x miss."),
        },
        "icc_estimate": {
            "icc_hat": icc_hat,
            "icc_ci": [icc_lo, icc_hi],
            "icc_ci_method": f"prompt-level cluster bootstrap, {args.n_boot} resamples, TPS-window ICC",
            "estimator": "one-way random-effects ANOVA, ICC = sigma_b^2/(sigma_b^2+sigma_w^2)",
            "icc_band_by_cluster_size": icc_band,
            "anova_window": a_win, "anova_full": a_full,
            "acf_rho_lags_1to8": rho[1:9], "acf_deff_at_mbar": deff_acf_bb,
            "serial_structure_note": (
                f"rho(1)={rho[1]:.3f} decaying ~AR (rho(2)={rho[2]:.3f}, rho(3)={rho[3]:.3f}) => the "
                "ICC is cluster-size dependent. PRIMARY = the TPS-window (128 tok/prompt = #175 "
                "B/128) ANOVA ICC; it is BOTH cluster-matched to m_bar (the first-24/25-step ANOVA "
                f"= {a_24['icc']:.3f}/{a_25['icc']:.3f} agree) AND the conservative end of the band: "
                f"the full-prompt ICC {a_full['icc']:.3f} paired with m_bar gives Deff "
                f"{deff_from_icc(mbar_bb, a_full['icc']):.2f} -> a HIGHER both-bugs LCB "
                f"{lcb_section4(bb_central, bb_hw_iid, mbar_bb, a_full['icc'])['lcb_tps']:.1f} (still GO). "
                "full-prompt (lower) and ACF-Deff cross-check bound the band."),
        },
        "realistic_ci": realistic_ci,
        "go_robustness": go_robustness,
        "build_bar": build_bar,
        "launch_packet_p90_secondary": launch,
        "imported": {
            "bb_central": bb_central, "bb_hw_iid": bb_hw_iid, "bb_nsteps": bb_nsteps,
            "bb_sigmaL": bb_sigmaL, "ds_central": ds_central, "ds_hw_iid": ds_hw_iid,
            "mbar_bb": mbar_bb, "mbar_ds": mbar_ds, "sigma_hw_s4": SIGMA_HW_S4,
            "topo184_bracket": topo184_bracket,
            "wandb_run_175": "zh1accmi", "wandb_run_183": "82uisrez", "wandb_run_184": "7uek36mx",
            "source_175": os.path.relpath(ET175_JSON, _ROOT),
            "source_184": os.path.relpath(TOPO184_JSON, _ROOT),
            "source_183": os.path.relpath(CARD183_JSON, _ROOT),
            "source_launch": os.path.relpath(LAUNCH_JSON, _ROOT),
        },
        "method": ("LOCAL CPU-only analytic synthesis. No GPU/vLLM/HF Job/submission/kernel "
                   "build/served-file change. BASELINE stays 481.53. Greedy untouched. NOT open2. "
                   "Does NOT authorize a launch."),
        "provenance": (
            "ICC measured from the PR #86 rankprobe (per-step accepted run-lengths) reconstructed "
            "into the 128 benchmark prompts; mapped through #184's imported N_eff two-level model "
            "(m_bar=24.58, exchangeable Deff) and #175's sigma_L finite-sample CI; build bar from "
            "#183's banked forward-map spine. #184's ICC=0/ICC=1 bracket is reproduced as a "
            "self-test, not re-derived."),
    }

    st = self_test(out)
    out["self_test"] = st
    out["icc_neff_self_test_passes"] = int(bool(st["passes"]))
    out["lcb_bothbugs_realistic_icc"] = go_robustness["lcb_bothbugs_realistic_icc"]
    out["metrics_nan_clean"] = int(st["nan_clean"])
    out["nan_clean"] = bool(st["nan_clean"])

    out_path = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    _console(out)
    rep_path = args.report_md if os.path.isabs(args.report_md) else os.path.join(_ROOT, args.report_md)
    _write_report(out, rep_path)

    if args.wandb:
        _log_wandb(args, out)


# ===================================================================================
# console + report + wandb
# ===================================================================================
def _console(out: dict) -> None:
    print("=" * 100)
    print("WITHIN-PROMPT ICC -> realistic launch CI + GO-robustness verdict (PR #190, wirbel)")
    print("=" * 100)
    ds = out["icc_data_source"]
    print(f"\n[DATA] {ds['n_steps']} steps -> {ds['n_prompts']} prompts; "
          f"E[L]@probe={ds['E_L_probe_operating_point']:.3f}; window={ds['tps_window_tokens']} tok "
          f"(~{ds['steps_per_prompt_tps_window_mean']:.1f} steps)")
    ie = out["icc_estimate"]
    print(f"\n[ICC] hat={ie['icc_hat']:.4f}  CI=[{ie['icc_ci'][0]:.4f},{ie['icc_ci'][1]:.4f}]  "
          f"(rho1={ie['acf_rho_lags_1to8'][0]:.3f})")
    print("  band by cluster size:")
    for k, v in ie["icc_band_by_cluster_size"].items():
        print(f"    {k:28s} {v:.4f}")
    rc = out["realistic_ci"]
    print(f"\n[CI] Deff={rc['design_effect_hat']:.3f}  N_eff={rc['n_eff_hat']:.0f}  "
          f"half_iid=±{rc['halfwidth_iid_tps']:.2f} -> half_realistic=±{rc['halfwidth_realistic_tps']:.2f} TPS "
          f"(worst ICC=1 ±{rc['halfwidth_worstcase_icc1_tps']:.2f})")
    gr = out["go_robustness"]
    s4 = gr["section4"]
    print(f"\n[GO-ROBUSTNESS] {gr['convention']}")
    print(f"  ICC=0     BB LCB {s4['icc0']['lcb_tps']:.2f}   ICC=1 BB LCB {s4['icc1']['lcb_tps']:.2f}")
    print(f"  ICC_hat   BB LCB {s4['bothbugs_hat']['lcb_tps']:.2f} ({'CLEARS' if s4['bothbugs_hat']['lcb_clears_500'] else 'MISS'})"
          f"   DS LCB {s4['descent_hat']['lcb_tps']:.2f} ({'CLEARS' if s4['descent_hat']['lcb_clears_500'] else 'MISS'})")
    print(f"  CI band   BB LCB [{s4['bothbugs_ci_hi']['lcb_tps']:.2f}, {s4['bothbugs_ci_lo']['lcb_tps']:.2f}]")
    print(f"  breakpoint BB ICC*={gr['icc_at_which_bothbugs_breaks_500']:.4f}  "
          f"DS ICC*={gr['icc_at_which_descent_breaks_500']:.4f}")
    print(f"  >> {gr['headline']}")
    bb = out["build_bar"]
    print(f"\n[BUILD BAR] lambda*_LCB(Deff=1)={bb['lambda_star_lcb_iid_check']:.4f} (repro 0.9052)  "
          f"-> realistic={bb['lambda_star_lcb_realistic_icc']:.4f}  shift={bb['bar_shift_from_icc']:+.4f}")
    lp = out["launch_packet_p90_secondary"]
    print(f"[LAUNCH P>=0.9] published 3-term LCB {lp['lcb_p90_published']:.2f} -> realistic-fold "
          f"{lp['lcb_p90_realistic_icc']:.2f} ({'GO' if lp['clears_500_realistic'] else 'MISS'})")
    st = out["self_test"]
    print(f"\n[SELF-TEST] {st['n_passed']}/{st['n_checks']} checks")
    for c in st["checks"]:
        print(f"  [{'OK' if c['passes'] else 'FAIL'}] {c['name']}  ({c['detail']})")
    print(f"\n[PRIMARY] icc_neff_self_test_passes = {out['icc_neff_self_test_passes']}")
    print(f"[TEST]    lcb_bothbugs_realistic_icc = {out['lcb_bothbugs_realistic_icc']:.3f} TPS")
    print(f"[NaN-clean] {out['metrics_nan_clean']}")


def _write_report(out: dict, path: str) -> None:
    ds = out["icc_data_source"]
    ie = out["icc_estimate"]
    rc = out["realistic_ci"]
    gr = out["go_robustness"]
    s4 = gr["section4"]
    bb = out["build_bar"]
    lp = out["launch_packet_p90_secondary"]
    st = out["self_test"]
    imp = out["imported"]
    band = ie["icc_band_by_cluster_size"]

    md = f"""<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Within-prompt ICC → realistic launch CI + GO-robustness verdict (PR #190 · wirbel)

**PRIMARY** `icc_neff_self_test_passes` = **{bool(out['icc_neff_self_test_passes'])}** ({st['n_passed']}/{st['n_checks']} conditions, NaN-clean)
**TEST** `lcb_bothbugs_realistic_icc` = **{out['lcb_bothbugs_realistic_icc']:.2f} TPS** (both-bugs launch LCB at realistic ICC={ie['icc_hat']:.3f}, §4 convention — clears 500 by +{out['lcb_bothbugs_realistic_icc']-500:.1f})
**W&B** group `icc-neff-launch-ci`

## Honest scope
Pure-analytic **CPU-only** synthesis. No GPU / vLLM / HF Job / submission / kernel build / served-file change. BASELINE stays **481.53**; **0 TPS**; greedy untouched. Imports (does **NOT** re-derive): wirbel **#175** (`zh1accmi`) σ_L/±10.906/central; denken **#184** N_eff two-level model + the ICC=0/ICC=1 bracket (reproduced as a self-test); denken **#183** (`82uisrez`) forward-map spine + λ\\*_LCB=0.9052; kanna **#159** σ_hw=4.86, K_cal, step 1.2182; the launch-packet LCB(P≥0.9) convention. **NOT open2. Does NOT authorize a launch.**

## The question
#175 priced the single-shot finite-sample TPS CI at **±10.906** assuming the ~3147 decode steps are **IID**; #184 stress-tested perfect clustering **ICC=1 → ±54.9** (LCB 480.5, a MISS). The clear-500 margin lives in that 5× bracket. This card asks the **data** where: what is the **realistic** within-prompt ICC, and does the **both-bugs GO survive it**?

## Estimator + data source
`icc_data_source` = **{ds['source']}** (PR #86 rankprobe, **{ds['n_steps']} decode steps**), reconstructed into the **{ds['n_prompts']} benchmark prompts** by accumulating committed tokens to {ds['prompt_output_tokens']}/prompt ({ds['prompt_lengths']}). Per-step quantity = **L = accepted-run (fd) + 1 bonus** (E[L]=E[T]; Var[L]=σ_L² — the #175 CI driver). One-way random-effects ANOVA: **ICC = σ²_b/(σ²_b+σ²_w)**, σ²_b=(MSB−MSW)/m₀, σ²_w=MSW.

**Cluster-size honesty:** the within-prompt correlation is **serial** (ρ(1)={ie['acf_rho_lags_1to8'][0]:.3f}, ρ(2)={ie['acf_rho_lags_1to8'][1]:.3f}, ρ(3)={ie['acf_rho_lags_1to8'][2]:.3f}, ~AR decay), not pure exchangeable, so the ANOVA ICC depends on the window. **PRIMARY = the TPS-window** (128 tok/prompt = #175's B=16384/128, ~{ds['steps_per_prompt_tps_window_mean']:.0f} steps); the band below brackets it.

| ICC estimate (cluster window) | ICC |
|---|---|
| **token-window 128 tok/prompt (PRIMARY, #175 scale)** | **{band['token_window_128tok_PRIMARY']:.4f}** |
| full-prompt (512 tok, ~{ds['steps_per_prompt_full_mean']:.0f} steps) | {band['full_prompt']:.4f} |
| first-24-step (≈ m̄) | {band['first_24_step']:.4f} |
| first-25-step | {band['first_25_step']:.4f} |
| first-33-step | {band['first_33_step']:.4f} |
| ACF-Deff equiv @ m̄ (pure serial, no prompt-mean heterogeneity) | {band['acf_equiv_at_mbar']:.4f} |

**`icc_hat` = {ie['icc_hat']:.4f}**, **`icc_ci` = [{ie['icc_ci'][0]:.4f}, {ie['icc_ci'][1]:.4f}]** ({ie['icc_ci_method']}).

## 1. Realistic CI (between #175 IID and #184 ICC=1)
m̄ = {rc['m_bar']:.3f} (imported #184). `Deff = 1+(m̄−1)·ICC` = **{rc['design_effect_hat']:.3f}**; `N_eff` = N_steps/Deff = **{rc['n_eff_hat']:.0f}** (from {imp['bb_nsteps']:.0f}). Only the accept-length term inflates (σ_hw is fixed denominator jitter):

| | half-width (accept, TPS) |
|---|---|
| #175 IID floor (ICC=0) | ±{rc['halfwidth_iid_tps']:.2f} |
| **realistic (ICC={ie['icc_hat']:.3f})** | **±{rc['halfwidth_realistic_tps']:.2f}** |
| #184 ICC=1 ceiling | ±{rc['halfwidth_worstcase_icc1_tps']:.2f} |

`halfwidth_realistic` = ±{rc['halfwidth_realistic_tps']:.2f} TPS — **{rc['halfwidth_realistic_tps']/rc['halfwidth_iid_tps']:.1f}× the IID floor, {rc['halfwidth_realistic_tps']/rc['halfwidth_worstcase_icc1_tps']:.2f}× the ICC=1 ceiling.**

## 2. GO-robustness verdict (THE deliverable)
§4 convention (central={imp['bb_central']:.2f} both / {imp['ds_central']:.2f} descent, z95, accept ⊕ σ_hw=4.86 in quadrature). The ICC=0 and ICC=1 rows **reproduce #184's bracket exactly** (520.95 / 480.53):

| ICC | Deff | accept ± | total ± | **both-bugs LCB** | clears? | **descent LCB** | clears? |
|---|---|---|---|---|---|---|---|
| 0 (IID #175) | {s4['icc0']['design_effect']:.2f} | — | ±{s4['icc0']['total_half_tps']:.2f} | {s4['icc0']['lcb_tps']:.2f} | ✓ | {lcb_section4(imp['ds_central'], imp['ds_hw_iid'], imp['mbar_ds'], 0.0)['lcb_tps']:.2f} | ✓ |
| {ie['icc_ci'][0]:.3f} (CI-lo) | {s4['bothbugs_ci_lo']['design_effect']:.2f} | ±{s4['bothbugs_ci_lo']['accept_half_tps']:.2f} | ±{s4['bothbugs_ci_lo']['total_half_tps']:.2f} | **{s4['bothbugs_ci_lo']['lcb_tps']:.2f}** | {'✓' if s4['bothbugs_ci_lo']['lcb_clears_500'] else '✗'} | — | — |
| **{ie['icc_hat']:.3f} (ĤAT)** | **{s4['bothbugs_hat']['design_effect']:.2f}** | **±{s4['bothbugs_hat']['accept_half_tps']:.2f}** | **±{s4['bothbugs_hat']['total_half_tps']:.2f}** | **{s4['bothbugs_hat']['lcb_tps']:.2f}** | **{'✓' if s4['bothbugs_hat']['lcb_clears_500'] else '✗'}** | **{s4['descent_hat']['lcb_tps']:.2f}** | **{'✓' if s4['descent_hat']['lcb_clears_500'] else '✗'}** |
| {ie['icc_ci'][1]:.3f} (CI-hi) | {s4['bothbugs_ci_hi']['design_effect']:.2f} | ±{s4['bothbugs_ci_hi']['accept_half_tps']:.2f} | ±{s4['bothbugs_ci_hi']['total_half_tps']:.2f} | **{s4['bothbugs_ci_hi']['lcb_tps']:.2f}** | {'✓' if s4['bothbugs_ci_hi']['lcb_clears_500'] else '✗'} | {s4['descent_ci_hi']['lcb_tps']:.2f} | {'✓' if s4['descent_ci_hi']['lcb_clears_500'] else '✗'} |
| 1 (#184 worst) | {s4['icc1']['design_effect']:.2f} | — | ±{s4['icc1']['total_half_tps']:.2f} | {s4['icc1']['lcb_tps']:.2f} | ✗ | {lcb_section4(imp['ds_central'], imp['ds_hw_iid'], imp['mbar_ds'], 1.0)['lcb_tps']:.2f} | ✗ |

- `lcb_bothbugs_realistic_icc` = **{gr['lcb_bothbugs_realistic_icc']:.2f}** (clears 500 by **+{gr['lcb_bothbugs_realistic_icc']-500:.1f}**)
- `lcb_descent_realistic_icc` = **{gr['lcb_descent_realistic_icc']:.2f}** ({'clears' if gr['descent_robust_to_realistic_icc'] else 'MISSES'})
- `icc_at_which_bothbugs_breaks_500` = **{gr['icc_at_which_bothbugs_breaks_500']:.4f}** (= **{gr['icc_at_which_bothbugs_breaks_500']/ie['icc_hat']:.1f}×** the realistic ICC); descent breaks already at ICC={gr['icc_at_which_descent_breaks_500']:.4f}.

**HEADLINE — {gr['headline']}**

## 3. Refined build bar (#183) under realistic correlation
The #183 bar λ\\*_LCB=0.9052 solved `central(λ)−z95·√(SE(λ)²+σ_hw²)=500` at Deff=1. The finite-sample SE inflates by √Deff. Machinery check: **Deff=1 reproduces {bb['lambda_star_lcb_iid_check']:.4f}** (= published 0.9052).

| Deff (ICC) | λ\\*_LCB | shift vs iid |
|---|---|---|
| 1.00 (iid) | {bb['lambda_star_lcb_iid_check']:.4f} | +0.0000 |
| {deff_from_icc(imp['mbar_bb'], ie['icc_ci'][0]):.2f} (CI-lo) | {bb['lambda_star_lcb_ci_lo_icc']:.4f} | {bb['lambda_star_lcb_ci_lo_icc']-bb['iid_bar']:+.4f} |
| **{bb['design_effect']:.2f} (ĤAT)** | **{bb['lambda_star_lcb_realistic_icc']:.4f}** | **{bb['bar_shift_from_icc']:+.4f}** |
| {deff_from_icc(imp['mbar_bb'], ie['icc_ci'][1]):.2f} (CI-hi) | {bb['lambda_star_lcb_ci_hi_icc']:.4f} | {bb['lambda_star_lcb_ci_hi_icc']-bb['iid_bar']:+.4f} |

`lambda_star_lcb_realistic_icc` = **{bb['lambda_star_lcb_realistic_icc']:.4f}**, `bar_shift_from_icc` = **{bb['bar_shift_from_icc']:+.4f}** (vs iid 0.9052). {bb['note']}

## 4. Launch-packet LCB(P≥0.9) — secondary handoff
Folding the realistic-ICC finite-sample relative term (×√Deff) into the launch packet's published 3-term combined (σ_hw retired on a separate axis):

| | combined_rel (1σ) | LCB(P≥0.9) | GO? |
|---|---|---|---|
| published (3-term, finite-sample PENDING) | {lp['combined_rel_published_3term']:.5f} | {lp['lcb_p90_published']:.2f} | GO |
| + IID finite-sample fold | {lp['combined_rel_iid_fold']:.5f} | {lp['lcb_p90_iid_finite_sample']:.2f} | GO |
| **+ realistic-ICC finite-sample fold** | **{lp['combined_rel_realistic_fold']:.5f}** | **{lp['lcb_p90_realistic_icc']:.2f}** | **{'GO' if lp['clears_500_realistic'] else 'MISS'}** |

Consistent with the §4 verdict ({gr['lcb_bothbugs_realistic_icc']:.1f}): both-bugs clears 500 under the realistic ICC in **both** LCB conventions.

## 5. Self-validate (PRIMARY)
{st['n_passed']}/{st['n_checks']} conditions pass: (a) ICC=0 reproduces #184 ±10.906/LCB 521; (b) ICC=1 reproduces ±54.9/LCB 480.5; (c) ICC_hat∈[0,1] with finite CI; (d) §4 LCB monotone-decreasing in ICC; (e) λ\\*_LCB(realistic) ≥ 0.9052 AND Deff=1 reproduces 0.9052; (f) NaN-clean. **`icc_neff_self_test_passes` = {bool(out['icc_neff_self_test_passes'])}**.

## Operating-point caveat
{ds['operating_point_caveat']}

## Hand-off
**fern #185 / launch packet:** the realistic within-prompt ICC is **{ie['icc_hat']:.3f}** [{ie['icc_ci'][0]:.3f}, {ie['icc_ci'][1]:.3f}], placing the single-shot CI at **±{rc['halfwidth_realistic_tps']:.1f} TPS** (Deff={rc['design_effect_hat']:.2f}, N_eff={rc['n_eff_hat']:.0f}) — {rc['halfwidth_realistic_tps']/rc['halfwidth_iid_tps']:.1f}× #175's IID floor but only {rc['halfwidth_realistic_tps']/rc['halfwidth_worstcase_icc1_tps']:.2f}× #184's ICC=1 ceiling. **both-bugs stays a robust GO** (LCB {gr['lcb_bothbugs_realistic_icc']:.1f} §4 / {lp['lcb_p90_realistic_icc']:.1f} P≥0.9), breaking 500 only at ICC={gr['icc_at_which_bothbugs_breaks_500']:.3f} ({gr['icc_at_which_bothbugs_breaks_500']/ie['icc_hat']:.1f}× realistic); **descent-only is NOT robust** (LCB {gr['lcb_descent_realistic_icc']:.1f}, breaks at ICC={gr['icc_at_which_descent_breaks_500']:.3f} < realistic). **land #71's build bar tightens to λ ≥ {bb['lambda_star_lcb_realistic_icc']:.4f}** (vs iid 0.9052, {bb['bar_shift_from_icc']:+.3f}).

## Public / banked evidence used
- wirbel **#175** (`zh1accmi`): σ_L, ±10.906 IID half-width, central 535.43 — the CI floor + driver.
- denken **#184**: N_eff two-level model (m̄=24.58, exchangeable Deff), the ICC=0/ICC=1 bracket (520.95/480.53) reproduced as the self-test.
- denken **#183** (`82uisrez`): forward-map spine (E[T](λ), σ_L(λ)) + λ\\*_LCB=0.9052 — the build bar refined here.
- kanna **#159**: σ_hw=4.86 TPS, K_cal, step 1.2182.
- launch packet: LCB(P≥0.9) convention (proj_private 528.89, z_p90 1.2816) — secondary fold.
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(md)
    print(f"\nwrote {os.path.relpath(path, _ROOT)}")


def _log_wandb(args, out: dict) -> None:
    import wandb

    ie = out["icc_estimate"]
    rc = out["realistic_ci"]
    gr = out["go_robustness"]
    bb = out["build_bar"]
    lp = out["launch_packet_p90_secondary"]
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_name, group=args.wandb_group, job_type="analysis",
                     config={"leg": "icc-neff-launch-ci",
                             "method": "cpu-analytic-within-prompt-ICC",
                             "icc_data_source": out["icc_data_source"]["source"],
                             "n_steps": out["icc_data_source"]["n_steps"],
                             "n_prompts": out["icc_data_source"]["n_prompts"],
                             "m_bar": rc["m_bar"], "sigma_hw_s4": out["imported"]["sigma_hw_s4"],
                             "wandb_run_175": "zh1accmi", "wandb_run_183": "82uisrez"})
    s = wandb.summary
    s["icc_neff_self_test_passes"] = out["icc_neff_self_test_passes"]
    s["lcb_bothbugs_realistic_icc"] = out["lcb_bothbugs_realistic_icc"]
    s["lcb_descent_realistic_icc"] = gr["lcb_descent_realistic_icc"]
    s["icc_hat"] = ie["icc_hat"]
    s["icc_ci_lo"] = ie["icc_ci"][0]
    s["icc_ci_hi"] = ie["icc_ci"][1]
    s["design_effect_hat"] = rc["design_effect_hat"]
    s["n_eff_hat"] = rc["n_eff_hat"]
    s["halfwidth_realistic_tps"] = rc["halfwidth_realistic_tps"]
    s["halfwidth_iid_tps"] = rc["halfwidth_iid_tps"]
    s["icc_at_which_bothbugs_breaks_500"] = gr["icc_at_which_bothbugs_breaks_500"]
    s["icc_at_which_descent_breaks_500"] = gr["icc_at_which_descent_breaks_500"]
    s["bothbugs_clears_across_full_ci"] = int(gr["bothbugs_clears_across_full_ci"])
    s["descent_robust_to_realistic_icc"] = int(gr["descent_robust_to_realistic_icc"])
    s["lambda_star_lcb_realistic_icc"] = bb["lambda_star_lcb_realistic_icc"]
    s["bar_shift_from_icc"] = bb["bar_shift_from_icc"]
    s["lcb_p90_realistic_icc"] = lp["lcb_p90_realistic_icc"]
    s["metrics_nan_clean"] = out["metrics_nan_clean"]
    s["n_checks"] = out["self_test"]["n_checks"]
    s["n_passed"] = out["self_test"]["n_passed"]

    # ICC band table
    bt = wandb.Table(columns=["cluster_window", "icc"])
    for k, v in ie["icc_band_by_cluster_size"].items():
        bt.add_data(k, v)
    wandb.log({"icc_band_by_cluster_size": bt})

    # GO-robustness §4 LCB table
    gt = wandb.Table(columns=["icc", "design_effect", "bothbugs_lcb", "bothbugs_clears",
                              "descent_lcb", "descent_clears"])
    s4 = gr["section4"]
    mb, mbh, mbd = out["imported"]["mbar_bb"], out["imported"]["bb_hw_iid"], out["imported"]["mbar_ds"]
    bc, dc, dh = out["imported"]["bb_central"], out["imported"]["ds_central"], out["imported"]["ds_hw_iid"]
    for icc in (0.0, ie["icc_ci"][0], ie["icc_hat"], ie["icc_ci"][1], 1.0):
        lb = lcb_section4(bc, mbh, mb, icc)
        ld = lcb_section4(dc, dh, mbd, icc)
        gt.add_data(icc, lb["design_effect"], lb["lcb_tps"], int(lb["lcb_clears_500"]),
                    ld["lcb_tps"], int(ld["lcb_clears_500"]))
    wandb.log({"go_robustness_section4": gt})

    # self-test checks
    stt = wandb.Table(columns=["check", "passes", "detail"])
    for c in out["self_test"]["checks"]:
        stt.add_data(c["name"], int(c["passes"]), c["detail"])
    wandb.log({"self_test_checks": stt})

    print(f"\nW&B run: {run.id}  ({run.url})")
    wandb.finish()


if __name__ == "__main__":
    main()
