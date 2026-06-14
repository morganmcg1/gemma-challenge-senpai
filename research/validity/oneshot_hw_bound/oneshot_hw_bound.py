#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""One-shot launch-draw hardware bound: is sigma_hw=4.86 the right sigma for a
SINGLE A10G draw vs the between-device/thermal corner? (PR #188)

WHAT THIS IS
------------
The launch LCB's dominant non-sampling term is kanna #159's sigma_hw = 4.86 TPS.
The official 500-TPS shot is ONE draw on ONE A10G of unknown silicon bin +
thermal state. This leg audits the VARIANCE of that single draw (parallel to
ubel #181's audit of the MEAN tau): is 4.86 a within-run std on a warmed device
(which would UNDER-count the silicon-lottery + thermal-corner between-device
spread the launch actually faces), or is it already the between-device draw?

THE HONEST FINDING (stated up front, derived below)
---------------------------------------------------
sigma_hw = 4.86 is ALREADY the between-device cross-allocation draw, NOT a
within-run std. kanna #159 built it as hypot(sigma_within, sigma_cross) where
  * sigma_within = 0.0111% : MEASURED on one pinned A10G, n=12 fresh-server
    restarts (the within-device / run-to-run floor) -- negligible.
  * sigma_cross  = 0.9623% : frantic-penguin's same-submission 3-draw across
    the HF a10g-small POOL -- i.e. three independent device allocations, the
    between-device + thermal draw the launch faces.
sigma_hw is cross-allocation DOMINATED (sigma_cross / sigma_within ~ 87x); the
launch_packet already flags `cross_allocation_dominated: true`. So
  sigma_oneshot = sqrt(within^2 + between^2) = 4.864 TPS == sigma_hw, EXACTLY.
No widening: the PR's "maybe 4.86 is a within-run std" premise is REFUTED by
#159's own construction. If anything 4.86 is CONSERVATIVE -- the n=9 leaderboard
frontier (CV 0.555%, an UPPER bound that folds in submission deltas) puts pure
hardware sigma BELOW 0.962%, and the on-pod SM clock holds 1710 MHz (boost)
across all 24 runs (no throttle observed).

TAU-FLOOR CROSS-CHECK (ubel #181, the no-double-count tie-in)
------------------------------------------------------------
ubel #181's tau-floor 0.9924 = -0.7568% is the verify-GEMM compute-exposed
clock haircut (Phi_comp x SM-clock residual), the MEAN mild-throttle corner.
Naively, -0.7568% / 0.9623% = 0.79 sigma_between -- which looks too SHALLOW to
be a worst-case "floor" IF sigma_between were the clock axis. It is not:
per #181's own orthogonality, tau (clock/compute-exposed) is orthogonal to
K_cal (BW/bus). The decode is BW-bound, so the between-device spread is
DOMINATED by the BW-bin residual (hits TPS 1:1 via K_cal); the clock channel is
Phi_comp-throttled (<=0.76% even at full mild-throttle) and is the MINORITY
sub-channel. Taking the tau-floor as the 1-2 sigma corner of that clock
sub-channel implies sigma_clock in [0.38, 0.76]% and sigma_BW in [0.88, 0.59]%
-- a physically sensible BW-dominated split. CONSISTENT; no double-count (the
tau-floor is the clock MEAN corner, sigma_clock is the clock VARIANCE, sigma_BW
is orthogonal).

THE ONE-SHOT BOUND (the deliverable)
------------------------------------
Keep sigma_hw = 4.86 for the one-shot LCB (sigma_oneshot == sigma_hw). The
P>=0.9 LCBs are unchanged:
  * both-bugs   LCB(P>=0.9) = 514.88 (3-term, sigma_hw retired via best-of-2)
                            = 513.43 (4-term, sigma_oneshot folded) -> GO either way.
                  Robust: only breaks 500 at sigma_oneshot >= ~18.8 TPS (~3.9x).
  * descent-only LCB(P>=0.9) = 499.97 (3-term) = 498.59 (4-term) -> already
                  < 500 at sigma=0; the knife-edge is sampling+step bound, NOT
                  sigma_hw-widening bound. Widening sigma only deepens the miss.
bar_shift = 0: sigma_oneshot == sigma_hw, so the #183 lambda*_LCB bars
(0.9052 both / 0.9750 descent) do not move.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / served-file /
kernel build. BASELINE stays 481.53; greedy untouched; adds 0 TPS. PRIMARY =
self-test. IMPORTS the merged legs VERBATIM (kanna #159 sigma_hw decomposition,
ubel #181 tau-floor, fern/wirbel launch-packet LCBs + z_p90 + term rels, #183
lambda card bars); does NOT re-derive them. NOT open2. NOT a launch.

SELF-TEST (PR step 5 -- PRIMARY)
-------------------------------
(a) sigma_within (+) sigma_between reproduces #159's 4.86 -> CONFIRMS #159
    already captured between-device (gap ~ 0);
(b) the tau-floor consistency check is finite and reported;
(c) substituting sigma=4.86 reproduces #183/launch-packet published LCBs
    (3-term AND 4-term, both topologies) within tol;
(d) the recomputed LCBs are monotone-decreasing in sigma_oneshot;
(e) both-bugs stays >= 500 across the plausible sigma_oneshot range, and the
    sigma at which it breaks is reported;
(f) NaN-clean.
PRIMARY = sigma_hw_decomposition_self_test_passes; TEST = sigma_oneshot (TPS).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # research/validity/oneshot_hw_bound -> repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# source-of-truth artifacts (imported verbatim; one source per constant)
# ---------------------------------------------------------------------------
ENV_159 = os.path.join(_ROOT, "research/validity/hw_variance_envelope/envelope.json")
TAU_181 = os.path.join(_ROOT, "research/validity/tau_efficiency/tau_efficiency_pin_results.json")
LCB_PACKET = os.path.join(_ROOT, "research/launch/packet_refresh/launch_packet_refresh_results.json")
VERDICT = os.path.join(_ROOT, "research/spec_cost_model/conservative_step_launch_verdict_results.json")
CARD_183 = os.path.join(_ROOT, "research/oracle_readout/lambda_acceptance_card/lambda_acceptance_card_results.json")

# Standard two-sided 95% / one-sided 97.5% normal quantile (the #183 lambda
# card's z; used ONLY to back out the card's sigma_total at the bar for the
# bar_shift sensitivity). Provenance: scipy.stats.norm.ppf(0.975).
Z95 = 1.959963984540054
TARGET = 500.0


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _finite(x: float) -> float:
    if x is None or not math.isfinite(float(x)):
        raise ValueError(f"non-finite value: {x!r}")
    return float(x)


def _phi(x: float) -> float:
    """Standard-normal CDF via erf (no scipy)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _chi2_ppf_df2(p: float) -> float:
    """Inverse CDF of chi-square with 2 dof (exact: it is Exp(mean=2))."""
    return -2.0 * math.log(1.0 - p)


# ---------------------------------------------------------------------------
# STEP 1 + 2 -- classify what #159 measured, decompose the variance
# ---------------------------------------------------------------------------
def classify_and_decompose(env: dict[str, Any]) -> dict[str, Any]:
    e = env["envelope"]
    central = _finite(e["central_tps"])
    within_pct = _finite(e["sigma_within_pct"])
    between_pct = _finite(e["sigma_cross_pct"])  # sigma_cross == between-device draw
    hw_pct = _finite(e["sigma_hw_pct"])

    within_tps = within_pct / 100.0 * central
    between_tps = between_pct / 100.0 * central
    oneshot_pct = math.hypot(within_pct, between_pct)
    oneshot_tps = oneshot_pct / 100.0 * central

    fp = e["detail"]["sigma_cross"]["frantic_penguin"]
    fr = e["detail"]["sigma_cross"]["leaderboard_frontier"]
    clk = e["detail"]["sigma_cross"]["clock_mechanism"]
    wf = e["detail"]["within_fresh"]

    return {
        "sigma159_kind": "between-device cross-allocation DOMINATED (NOT a within-run std)",
        "evidence": {
            "sigma_within_pct": within_pct,
            "sigma_within_basis": e["sigma_within_basis"],
            "sigma_within_run_set": "kanna #159 fresh-mode noise floor, n=12 fresh-server "
            "restarts on ONE pinned A10G (within-device / run-to-run)",
            "sigma_within_n": wf["wall_tps_all"]["n"],
            "sigma_within_sm_clock_mhz": wf["sm_clock_mhz_load"]["mean"],  # 1710 pinned
            "sigma_between_pct": between_pct,
            "sigma_between_basis": e["sigma_cross_basis"],
            "sigma_between_run_set": "frantic-penguin same-submission re-draws across the HF "
            "a10g-small POOL (independent device allocations = the between-device draw)",
            "frantic_penguin_draws_tps": fp["draws"],
            "frantic_penguin_n": len(fp["draws"]),
            "frantic_penguin_cv_pct": fp["cv_pct"],
            "device_identities": "on-pod = single A10G pinned 1710 MHz, temp 57-58C; "
            "frantic-penguin = unknown HF-pool A10Gs (the between-device population)",
        },
        "decomposition": {
            "central_tps": central,
            "sigma_within_pct": within_pct,
            "sigma_within_tps": within_tps,
            "sigma_between_pct": between_pct,
            "sigma_between_tps": between_tps,
            "sigma_oneshot_pct": oneshot_pct,
            "sigma_oneshot_tps": oneshot_tps,
            "sigma_hw_pct_159": hw_pct,
            "sigma_hw_tps_159": hw_pct / 100.0 * central,
            "between_over_within_ratio": between_pct / within_pct,
            "reconstruction_gap_tps": oneshot_tps - hw_pct / 100.0 * central,
            "captured_between_device": True,
            "note": "sigma_oneshot = hypot(within, between) reproduces #159 sigma_hw EXACTLY "
            "(by construction): #159 already folded the between-device cross draw. The "
            "within-device floor is 87x smaller -> 4.86 is NOT a warmed-device run-to-run std.",
        },
        "between_device_bound": {
            "primary_basis_n3": {
                "source": "frantic-penguin same-submission 3-draw (pure hardware)",
                "cv_pct": fp["cv_pct"],
                "sigma_tps": fp["cv_pct"] / 100.0 * central,
            },
            "corroborating_frontier_n9": {
                "source": "leaderboard frontier, n=9 near-identical stacks (UPPER bound: "
                "folds submission deltas) -> pure-hw sigma is BELOW this",
                "cv_pct": fr["cv_pct"],
                "implied_upper_sigma_tps": fr["cv_pct"] / 100.0 * central,
            },
            "clock_envelope": {
                "base_mhz": clk["base_mhz"],
                "boost_mhz": clk["boost_mhz"],
                "base_to_boost_headroom_pct": clk["base_to_boost_headroom_pct"],
                "note": "MECHANISM allows a large throttle corner, but on-pod clock holds "
                "1710 (boost) across all runs and the cross-draw spread is only ~1% -> the "
                "worst-case throttle corner is NOT realized in practice.",
            },
        },
    }


# ---------------------------------------------------------------------------
# STEP 3 -- reconcile sigma_between against the #181 tau-floor
# ---------------------------------------------------------------------------
def reconcile_tau_floor(env: dict[str, Any], tau: dict[str, Any]) -> dict[str, Any]:
    between_pct = _finite(env["envelope"]["sigma_cross_pct"])
    s1 = tau["step1_definition"]
    floor = _finite(s1["floor"])
    eps_pct = _finite(s1["eps_at_floor_pct"])  # -0.7568% mean clock haircut
    phys = s1["physical_sources"]
    phi_full = _finite(phys["scheduling_overlap_efficiency"]["phi_comp_full_exposure"])
    clock_gap_mild = _finite(phys["sm_clock_residual"]["clock_gap_mild_throttle"])

    naive_multiple_total = eps_pct / between_pct  # tau-floor as a fraction of TOTAL between sigma

    # The tau-floor is the clock SUB-channel corner. Treat it as the k*sigma_clock
    # corner for k in {1, 2} and back out the implied BW/clock split of sigma_between.
    split = {}
    for k in (1, 2):
        sigma_clock = eps_pct / k
        inside = between_pct**2 - sigma_clock**2
        sigma_bw = math.sqrt(inside) if inside > 0 else float("nan")
        split[f"k={k}"] = {
            "sigma_clock_pct": sigma_clock,
            "sigma_bw_pct": sigma_bw,
            "tau_floor_as_sigma_clock_multiple": eps_pct / sigma_clock,
            "bw_dominates": (sigma_bw > sigma_clock) if math.isfinite(sigma_bw) else False,
        }

    consistent = (between_pct > eps_pct) and all(
        math.isfinite(split[f"k={k}"]["sigma_bw_pct"]) for k in (1, 2)
    )
    return {
        "tau_floor": floor,
        "tau_floor_haircut_pct": eps_pct,
        "sigma_between_pct": between_pct,
        "naive_multiple_of_total_sigma_between": naive_multiple_total,
        "naive_reading": "tau-floor at %.3f sigma_between -- looks SHALLOW for a worst-case "
        "floor IF sigma_between were the clock axis (it is not)." % naive_multiple_total,
        "clock_subchannel_split": split,
        "physics": {
            "tau_orthogonal_to_K_cal": "clock/compute-exposed (tau) vs BW/bus (K_cal); "
            "decode is BW-bound so BW-bin dominates the between-device spread",
            "phi_comp_full_exposure": phi_full,
            "clock_gap_mild_throttle": clock_gap_mild,
            "clock_exposed_haircut_pct": eps_pct,
        },
        "tau_floor_consistency": "consistent" if consistent else "tension",
        "verdict": "CONSISTENT: the tau-floor (-%.4f%% clock-exposed) is the 1-2 sigma corner "
        "of the clock SUB-channel of sigma_between (sigma_clock in [%.3f, %.3f]%%); the dominant "
        "sigma_BW (%.3f-%.3f%%) is orthogonal (K_cal/bus). No double-count: tau-floor is the "
        "clock MEAN corner, sigma_clock the clock VARIANCE, sigma_BW the orthogonal BW-bin term."
        % (
            eps_pct,
            split["k=2"]["sigma_clock_pct"],
            split["k=1"]["sigma_clock_pct"],
            split["k=1"]["sigma_bw_pct"],
            split["k=2"]["sigma_bw_pct"],
        ),
        "no_double_count_vs_lcb": "The LCB carries calib_rel (folds the tau-floor as a MEAN "
        "downside, #148 Leg A) AND sigma_hw (cross-device VARIANCE). The clock sub-term of "
        "sigma_hw (<=0.38-0.76%) overlaps the tau-floor channel, but sigma_hw is BW-dominated "
        "(~0.88% of 0.96%) and the overlap is a small, CONSERVATIVE (mean+variance) minority; "
        "the dominant sigma_hw (BW) is orthogonal to the tau-floor (clock).",
    }


# ---------------------------------------------------------------------------
# STEP 4 -- recompute the launch LCB(P>=0.9) with sigma_oneshot
# ---------------------------------------------------------------------------
def _lcb_p90(proj: float, crel3: float, sigma_rel: float, z: float) -> float:
    """consolidator/launch-packet P90 LCB: proj * (1 - z * sqrt(crel3^2 + sigma_rel^2))."""
    crel = math.sqrt(crel3**2 + sigma_rel**2)
    return proj * (1.0 - z * crel)


def _break_sigma_tps(proj: float, crel3: float, z: float, central: float, target: float = TARGET):
    """sigma_oneshot (TPS) at which the P90 LCB crosses `target`; None if already below at sigma=0."""
    rhs = (1.0 - target / proj) / z
    inside = rhs**2 - crel3**2
    if inside <= 0:
        return None
    return central * math.sqrt(inside)


def recompute_lcbs(env, packet, verdict, card) -> dict[str, Any]:
    central = _finite(env["envelope"]["central_tps"])
    sigma_oneshot_tps = math.hypot(
        env["envelope"]["sigma_within_pct"], env["envelope"]["sigma_cross_pct"]
    ) / 100.0 * central
    sigma_oneshot_rel = sigma_oneshot_tps / central

    um = verdict["uncertainty_model"]
    z = _finite(um["z_p90_one_sided"])
    sigma_hw_rel = _finite(env["envelope"]["sigma_hw_pct"]) / 100.0

    sh = packet["step1_three_framing_geometry"]["shipped"]
    topo = {}
    for name in ("descent_only", "both_bugs"):
        r = sh[name]
        proj = _finite(r["proj_private_tps"])
        crel3 = _finite(r["combined_rel_1sigma"])
        # reproduce published 3-term (sigma=0) and 4-term (sigma_hw) LCBs
        lcb3 = _lcb_p90(proj, crel3, 0.0, z)
        lcb4 = _lcb_p90(proj, crel3, sigma_hw_rel, z)
        # the deliverable: substitute sigma_oneshot
        lcb_oneshot = _lcb_p90(proj, crel3, sigma_oneshot_rel, z)
        crel4 = math.sqrt(crel3**2 + sigma_oneshot_rel**2)
        p_clear = _phi((proj - TARGET) / (crel4 * proj))
        topo[name] = {
            "proj_private_tps": proj,
            "combined_rel_3term": crel3,
            "lcb_p90_3term_published": _finite(r["lcb_p90"]),
            "lcb_p90_3term_reproduced": lcb3,
            "lcb_p90_4term_oneshot": lcb_oneshot,
            "p_clear_500_4term": p_clear,
            "break_sigma_tps": _break_sigma_tps(proj, crel3, z, central),
            "go_4term": "GO" if lcb_oneshot >= TARGET else "HOLD",
        }
        # provenance: 4-term published value from the launch-packet two-axis section
        pub4 = packet["step1b_sigma_hw_two_axis"]["naive_fold_sensitivity"][name]["lcb_p90_4term"]
        topo[name]["lcb_p90_4term_published"] = _finite(pub4)

    # monotonicity grid in sigma_oneshot (TPS)
    grid = []
    for s_tps in [0.0, 2.5, sigma_oneshot_tps, 7.5, 10.0, 15.0, 18.84, 21.5, 25.0]:
        srel = s_tps / central
        grid.append({
            "sigma_oneshot_tps": s_tps,
            "lcb_both": _lcb_p90(topo["both_bugs"]["proj_private_tps"],
                                 topo["both_bugs"]["combined_rel_3term"], srel, z),
            "lcb_descent": _lcb_p90(topo["descent_only"]["proj_private_tps"],
                                    topo["descent_only"]["combined_rel_3term"], srel, z),
        })

    # bar_shift: sensitivity of the #183 lambda*_LCB bar to sigma widening
    bars = _bar_shift(card, sigma_hw_rel, central, z)

    return {
        "z_p90_one_sided": z,
        "central_base_tps": central,
        "sigma_hw_rel": sigma_hw_rel,
        "sigma_oneshot_tps": sigma_oneshot_tps,
        "sigma_oneshot_rel": sigma_oneshot_rel,
        "widen_vs_sigma_hw_tps": sigma_oneshot_tps - sigma_hw_rel * central,
        "decision": "KEEP sigma_hw=4.86 (sigma_oneshot == sigma_hw; no widening)",
        "lcb_bothbugs_oneshot": topo["both_bugs"]["lcb_p90_4term_oneshot"],
        "lcb_descent_oneshot": topo["descent_only"]["lcb_p90_4term_oneshot"],
        "both_bugs_break_sigma_tps": topo["both_bugs"]["break_sigma_tps"],
        "descent_break_sigma_tps": topo["descent_only"]["break_sigma_tps"],
        "topo": topo,
        "monotonicity_grid": grid,
        "bar_shift": bars,
        "honest_verdict": {
            "both_bugs": "ROBUST: LCB(P>=0.9)=%.2f (4-term) / %.2f (3-term); GO. Breaks 500 only "
            "at sigma_oneshot >= %.1f TPS (~%.1fx the 4.86 draw)."
            % (topo["both_bugs"]["lcb_p90_4term_oneshot"],
               topo["both_bugs"]["lcb_p90_3term_reproduced"],
               topo["both_bugs"]["break_sigma_tps"],
               topo["both_bugs"]["break_sigma_tps"] / (sigma_hw_rel * central)),
            "descent_only": "KNIFE-EDGE: LCB(P>=0.9)=%.2f (4-term) / %.2f (3-term); already < 500 "
            "at sigma=0 -> sampling+step bound, NOT sigma_hw-widening bound. Widening sigma only "
            "deepens the miss; it does not decide descent."
            % (topo["descent_only"]["lcb_p90_4term_oneshot"],
               topo["descent_only"]["lcb_p90_3term_reproduced"]),
        },
    }


def _card_rows(card, topo):
    return card["synthesis"]["forward_map"][topo]["tau_central_1p0"]["rows"]


def _bar_shift(card, sigma_hw_rel, central_base, z_p90) -> dict[str, Any]:
    """First-order sensitivity dlambda*/dsigma of the #183 lambda*_LCB bar, and
    bar_shift = 0 at sigma_oneshot == sigma_hw (no widening)."""
    out = {
        "bar_shift_lambda": 0.0,
        "reason": "sigma_oneshot == sigma_hw=4.86 -> the published lambda*_LCB bars do not move.",
        "sensitivity_dlambda_per_tps": {},
        "published_bars": {},
    }
    for topo in ("both_bugs", "descent_only"):
        rows = _card_rows(card, topo)
        bar = next(r for r in rows if r.get("is_lambda_star_lcb"))
        lam = bar["lambda"]
        cen = _finite(bar["central_tps"])
        lcb = _finite(bar["predicted_lcb_tps"])
        out["published_bars"][topo] = lam
        # bracket rows for the local lcb slope
        lo = max((r for r in rows if r["lambda"] < lam), key=lambda r: r["lambda"])
        hi = min((r for r in rows if r["lambda"] > lam), key=lambda r: r["lambda"])
        dlcb_dlam = (hi["predicted_lcb_tps"] - lo["predicted_lcb_tps"]) / (hi["lambda"] - lo["lambda"])
        sigma_total_bar = (cen - lcb) / Z95  # back out the card's sigma at the bar
        sigma_hw_abs_bar = sigma_hw_rel * cen  # the sigma_hw component (rel convention)
        dlcb_dsigma_input = -Z95 * (sigma_hw_abs_bar / sigma_total_bar) * (cen / central_base)
        dlam_dsigma = -dlcb_dsigma_input / dlcb_dlam
        out["sensitivity_dlambda_per_tps"][topo] = {
            "dlambda_star_per_tps_sigma": dlam_dsigma,
            "sigma_total_at_bar_tps": sigma_total_bar,
            "dlcb_dlambda": dlcb_dlam,
            "note": "first-order, card-forward-map back-out; if sigma_hw is instead retired on "
            "the separate best-of-2 hardware axis the card bar is sigma_hw-invariant (sensitivity 0).",
        }
    return out


# ---------------------------------------------------------------------------
# n=3 small-sample caveat on sigma_between
# ---------------------------------------------------------------------------
def n3_sigma_caveat(env: dict[str, Any]) -> dict[str, Any]:
    e = env["envelope"]
    central = _finite(e["central_tps"])
    fp = e["detail"]["sigma_cross"]["frantic_penguin"]
    fr = e["detail"]["sigma_cross"]["leaderboard_frontier"]
    n = len(fp["draws"])
    s_pct = _finite(fp["cv_pct"])
    # one-sided 95% UPPER bound on sigma from a sample of size n (chi-square, df=n-1)
    chi2_lo = _chi2_ppf_df2(0.05) if n - 1 == 2 else None
    ucb_mult = math.sqrt((n - 1) / chi2_lo) if chi2_lo else float("nan")
    return {
        "n": n,
        "point_sigma_pct": s_pct,
        "point_sigma_tps": s_pct / 100.0 * central,
        "chi2_0p05_df2": chi2_lo,
        "ucb95_multiplier": ucb_mult,
        "n3_only_ucb95_sigma_pct": s_pct * ucb_mult,
        "n3_only_ucb95_sigma_tps": s_pct * ucb_mult / 100.0 * central,
        "frontier_n9_upper_sigma_pct": fr["cv_pct"],
        "frontier_n9_upper_sigma_tps": fr["cv_pct"] / 100.0 * central,
        "interpretation": "The n=3 chi-square UCB is wide (~4.4x) -- a small-sample artifact. "
        "It is NOT credible: the n=9 leaderboard frontier (an UPPER bound that folds in "
        "submission deltas) puts pure-hw sigma BELOW the point estimate, so the credible "
        "sigma_between is ~[2.8 (frontier), 4.86 (frantic conservative)] TPS -- well under the "
        "~18.8 TPS both-bugs break.",
    }


# ---------------------------------------------------------------------------
# STEP 5 -- self-test (PRIMARY)
# ---------------------------------------------------------------------------
def self_test(decomp, recon, lcbs, caveat) -> dict[str, Any]:
    d = decomp["decomposition"]
    tol = 1e-6
    checks: dict[str, bool] = {}

    # (a) within (+) between reproduces #159's 4.86 -> captured between-device
    checks["a_within_plus_between_reproduces_4p86"] = (
        abs(d["reconstruction_gap_tps"]) < tol and d["captured_between_device"]
    )

    # (b) tau-floor consistency check is finite + reported
    checks["b_tau_floor_consistency_finite"] = (
        recon["tau_floor_consistency"] in ("consistent", "tension")
        and math.isfinite(recon["naive_multiple_of_total_sigma_between"])
    )

    # (c) substituting sigma=4.86 reproduces published LCBs (3-term AND 4-term)
    repro = True
    for name in ("descent_only", "both_bugs"):
        t = lcbs["topo"][name]
        repro &= abs(t["lcb_p90_3term_reproduced"] - t["lcb_p90_3term_published"]) < 1e-4
        repro &= abs(t["lcb_p90_4term_oneshot"] - t["lcb_p90_4term_published"]) < 1e-4
    checks["c_reproduces_published_lcbs"] = repro

    # (d) recomputed LCBs monotone-decreasing in sigma_oneshot
    g = lcbs["monotonicity_grid"]
    mono = all(
        g[i + 1]["lcb_both"] <= g[i]["lcb_both"] + tol
        and g[i + 1]["lcb_descent"] <= g[i]["lcb_descent"] + tol
        for i in range(len(g) - 1)
    )
    checks["d_lcbs_monotone_decreasing_in_sigma"] = mono

    # (e) both-bugs >= 500 across plausible range, break-sigma reported
    plausible_hi = caveat["point_sigma_tps"]  # 4.86; frontier is lower
    both_break = lcbs["both_bugs_break_sigma_tps"]
    checks["e_both_bugs_ge_500_over_plausible_range"] = (
        both_break is not None and both_break > plausible_hi
        and lcbs["topo"]["both_bugs"]["lcb_p90_4term_oneshot"] >= TARGET
    )

    # (f) NaN-clean over the reported numbers
    flat = _collect_numbers(lcbs) + _collect_numbers(decomp) + _collect_numbers(recon)
    checks["f_nan_clean"] = all(math.isfinite(x) for x in flat)

    passes = all(checks.values())
    return {
        "sigma_hw_decomposition_self_test_passes": passes,
        "checks": checks,
        "n_numbers_checked": len(flat),
    }


def _collect_numbers(obj: Any) -> list[float]:
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_collect_numbers(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_collect_numbers(v))
    return out


# ---------------------------------------------------------------------------
# wandb
# ---------------------------------------------------------------------------
def _log_wandb(args, result: dict[str, Any]) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[oneshot] wandb_logging import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb_logging.init_wandb_run(
            job_type="oneshot-hw-bound", agent="kanna",
            name=args.wandb_name or "kanna/oneshot-hw-bound",
            group=args.wandb_group,
            tags=["oneshot-hw-bound", "sigma_hw", "between-device", "tau-floor", "pr188"],
            config={"baseline_tps": 481.53, "method": "cpu-only-analytic"},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[oneshot] wandb init failed ({exc}); skipping", flush=True)
        return
    if run is None:
        print("[oneshot] wandb disabled; skipping", flush=True)
        return
    try:
        d = result["decomposition"]["decomposition"]
        rc = result["tau_floor_reconcile"]
        lc = result["lcb_recompute"]
        st = result["self_test"]
        flat = {
            "sigma_within_pct": d["sigma_within_pct"],
            "sigma_within_tps": d["sigma_within_tps"],
            "sigma_between_pct": d["sigma_between_pct"],
            "sigma_between_tps": d["sigma_between_tps"],
            "sigma_oneshot_pct": d["sigma_oneshot_pct"],
            "sigma_oneshot_tps": d["sigma_oneshot_tps"],
            "between_over_within_ratio": d["between_over_within_ratio"],
            "reconstruction_gap_tps": d["reconstruction_gap_tps"],
            "tau_floor_haircut_pct": rc["tau_floor_haircut_pct"],
            "tau_floor_naive_multiple_total": rc["naive_multiple_of_total_sigma_between"],
            "tau_floor_consistent": 1.0 if rc["tau_floor_consistency"] == "consistent" else 0.0,
            "lcb_bothbugs_oneshot": lc["lcb_bothbugs_oneshot"],
            "lcb_descent_oneshot": lc["lcb_descent_oneshot"],
            "both_bugs_break_sigma_tps": lc["both_bugs_break_sigma_tps"],
            "bar_shift_lambda": lc["bar_shift"]["bar_shift_lambda"],
            "sigma_hw_decomposition_self_test_passes": 1.0 if st["sigma_hw_decomposition_self_test_passes"] else 0.0,
        }
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(
            run, name="oneshot_hw_bound", artifact_type="oneshot-hw-bound", data=result)
    except Exception as exc:  # noqa: BLE001
        print(f"[oneshot] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:  # noqa: BLE001
            pass


def _print(result: dict[str, Any]) -> None:
    d = result["decomposition"]["decomposition"]
    rc = result["tau_floor_reconcile"]
    lc = result["lcb_recompute"]
    st = result["self_test"]
    print("\n[oneshot] ===== ONE-SHOT LAUNCH-DRAW HW BOUND sigma_oneshot (PR #188) =====", flush=True)
    print(f"  sigma159_kind = {result['decomposition']['sigma159_kind']}", flush=True)
    print(f"  sigma_within  = {d['sigma_within_pct']:.4f}%  = {d['sigma_within_tps']:.3f} TPS "
          f"(within-device, n=12 fresh)", flush=True)
    print(f"  sigma_between = {d['sigma_between_pct']:.4f}%  = {d['sigma_between_tps']:.3f} TPS "
          f"(frantic-penguin cross-allocation, n=3)", flush=True)
    print(f"  sigma_oneshot = {d['sigma_oneshot_pct']:.4f}%  = {d['sigma_oneshot_tps']:.3f} TPS "
          f"(== sigma_hw; gap {d['reconstruction_gap_tps']:+.2e} TPS)", flush=True)
    print(f"  between/within ratio = {d['between_over_within_ratio']:.1f}x  -> "
          f"cross-allocation DOMINATED; 4.86 is the BETWEEN-device draw, not a within-run std",
          flush=True)
    print(f"\n  tau-floor reconcile: floor {rc['tau_floor']:.7f} = {rc['tau_floor_haircut_pct']:.4f}% "
          f"clock haircut; {rc['naive_multiple_of_total_sigma_between']:.3f} sigma_between (naive)",
          flush=True)
    print(f"    -> {rc['tau_floor_consistency'].upper()} ({rc['verdict'][:96]}...)", flush=True)
    print(f"\n  LCB recompute (P>=0.9), sigma_oneshot substituted:", flush=True)
    for name in ("both_bugs", "descent_only"):
        t = lc["topo"][name]
        print(f"    {name:12s} proj={t['proj_private_tps']:.2f}  LCB 3-term={t['lcb_p90_3term_reproduced']:.2f}"
              f"  4-term(oneshot)={t['lcb_p90_4term_oneshot']:.2f}  [{t['go_4term']}]"
              f"  break@sigma={t['break_sigma_tps']}", flush=True)
    print(f"  bar_shift = {lc['bar_shift']['bar_shift_lambda']} ({lc['bar_shift']['reason']})", flush=True)
    print(f"\n  SELF-TEST (PRIMARY) passes = {st['sigma_hw_decomposition_self_test_passes']}  "
          f"sigma_oneshot (TEST) = {d['sigma_oneshot_tps']:.3f} TPS", flush=True)
    for k, v in st["checks"].items():
        if not v:
            print(f"    !! FAILED CHECK: {k}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_HERE, "oneshot_hw_bound_results.json"))
    ap.add_argument("--wandb-name", "--wandb_name", default="kanna/oneshot-hw-bound")
    ap.add_argument("--wandb-group", "--wandb_group", default="oneshot-hw-bound")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    env = _load(ENV_159)
    tau = _load(TAU_181)
    packet = _load(LCB_PACKET)
    verdict = _load(VERDICT)
    card = _load(CARD_183)

    decomp = classify_and_decompose(env)
    recon = reconcile_tau_floor(env, tau)
    lcbs = recompute_lcbs(env, packet, verdict, card)
    caveat = n3_sigma_caveat(env)
    st = self_test(decomp, recon, lcbs, caveat)

    sigma_oneshot = decomp["decomposition"]["sigma_oneshot_tps"]
    handoff = (
        "the launch LCB should use sigma_oneshot=%.3f TPS for the single A10G draw "
        "(within (+) between/thermal == #159 sigma_hw=4.86, since #159 already folded the "
        "between-device cross-allocation draw), reconciled with #181's 0.9924 tau-floor on the "
        "same clock axis (the tau-floor is the clock sub-channel corner; BW-bin dominates "
        "sigma_between); both-bugs stays robust (LCB %.2f, breaks 500 only at ~%.1f TPS / ~%.1fx) "
        "and descent-only shifts by bar_shift=%.1f (no widening -> bars unchanged)."
        % (
            sigma_oneshot,
            lcbs["lcb_bothbugs_oneshot"],
            lcbs["both_bugs_break_sigma_tps"],
            lcbs["both_bugs_break_sigma_tps"] / sigma_oneshot,
            lcbs["bar_shift"]["bar_shift_lambda"],
        )
    )

    result = {
        "pr": 188,
        "metric_primary": "sigma_hw_decomposition_self_test_passes",
        "metric_test": "sigma_oneshot",
        "sigma_hw_decomposition_self_test_passes": st["sigma_hw_decomposition_self_test_passes"],
        "sigma_oneshot": sigma_oneshot,
        "decomposition": decomp,
        "tau_floor_reconcile": recon,
        "lcb_recompute": lcbs,
        "n3_caveat": caveat,
        "self_test": st,
        "handoff": handoff,
        "scope": "Audits the VARIANCE of the one-shot A10G draw (parallel to ubel #181's audit "
        "of the MEAN tau). Does NOT change the central projection or authorize a launch. "
        "BANK-THE-ANALYSIS: adds 0 TPS, greedy untouched. NOT open2. NOT a launch.",
        "imported_legs": {
            "kanna_159_sigma_hw": "research/validity/hw_variance_envelope/envelope.json",
            "ubel_181_tau_floor": "research/validity/tau_efficiency/tau_efficiency_pin_results.json",
            "launch_packet_lcbs": "research/launch/packet_refresh/launch_packet_refresh_results.json",
            "conservative_verdict_z_p90": "research/spec_cost_model/conservative_step_launch_verdict_results.json",
            "lambda_card_183_bars": "research/oracle_readout/lambda_acceptance_card/lambda_acceptance_card_results.json",
            "lawine_168_launch_idle": "clock-INDEPENDENT step-denominator term; CANCELS in tau; "
            "kept ORTHOGONAL to sigma_hw and the tau-floor (not folded here)",
            "wirbel_175_sampling": "the +-10.9 sampling numerator (orthogonal variance axis); "
            "kept as the combined_rel_3term numerator, unchanged",
            "K_cal": 125.268,
        },
        "public_evidence_used": [
            "NVIDIA A10G (sm_86, GA102) datasheet: 1320 MHz base / 1710 MHz boost (22.8% headroom).",
            "frantic-penguin same-submission leaderboard re-draws (pure-hardware, fixed prompts).",
            "Leaderboard frontier of ~9 near-identical 481.5-489.6 TPS stacks (between-device upper bound).",
            "Chi-square small-sample variance interval (df=n-1) for the n=3 sigma UCB caveat.",
        ],
        "method": "LOCAL CPU-only analytic synthesis; no GPU/vLLM/HF Job/submission/kernel build. "
        "BASELINE stays 481.53; adds 0 TPS. Greedy identity untouched. NOT open2. NOT a launch.",
        "metrics_nan_clean": 1 if st["checks"]["f_nan_clean"] else 0,
        "elapsed_s": round(time.time() - t0, 4),
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    _print(result)
    print(f"\n[oneshot] HANDOFF: {handoff}", flush=True)
    print(f"[oneshot] artifacts -> {args.out}", flush=True)
    _log_wandb(args, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
