#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Margin-aware λ-acceptance card (PR #183) — the finite-sample-LCB build bar.

denken #178 (`zjdc7hhh`) graded the descent-only / both-bugs self-KV recovery into a
POINT curve E[T](λ) and read off the clear-500 thresholds as POINT estimates
(λ*=0.909 descent-only / 0.838 both-bugs at τ=1, where the *central* TPS = 500). But
"the built kernel clears 500" must mean the FINITE-SAMPLE 95% LOWER BOUND clears 500,
not the central point. wirbel #175 (`zh1accmi`) pinned the single-draw accept-length
scatter at ±10.906 TPS (both-bugs, B=16384) — the numerator leg — and it composes in
quadrature with kanna #159's σ_hw=4.86 TPS denominator leg. So the build must show a
*higher* per-depth λ than #178's central λ* for the LCB to clear 500.

This PR derives that **margin-aware λ-acceptance card**: it composes the finite-sample
LCB as a function of λ, solves lcb_tps(λ*)=500 for the build-acceptance bar, translates
λ* into the literal per-depth `q[2..9]` ladder land #71's measured kernel must hit, and
provides the forward map (measured-λ → predicted-LCB-TPS) as a go/no-go calculator. It
is a SYNTHESIS: it IMPORTS denken #178's E[T](λ) interpolation + endpoints and wirbel
#175's accepted-length pmf → σ_L second-moment machinery, and propagates them. It does
NOT re-derive 5.0564 / 3.5346 / 5.2070, K_cal, the step, σ_L, the ±10.906 CI, or σ_hw.

LOCAL CPU-only analytic. No GPU / vLLM / HF Job / submission / kernel build / served-file
change. BASELINE stays 481.53. Greedy untouched. Bank-the-analysis (PRIMARY = self-test,
adds 0 TPS). NOT open2. NOT a launch.

------------------------------------------------------------------------------
COMPOSITION (imported, not re-derived)
------------------------------------------------------------------------------
central(λ) = K_cal·(E[T](λ)/step)·τ                          (#172/#178 composition)
H(λ)       = z95·√( SE_tps(λ)²  +  σ_hw² )                   (PR #183 full half-width)
  SE_tps(λ) = (K_cal·τ/step)·σ_L(λ)/√N_steps(λ)             (wirbel #175 numerator leg)
  N_steps(λ) = B / E[T](λ),  B=16384                        (wirbel #175 budget)
  σ_L(λ)     = √Var[L] of the accepted-length pmf at spine(λ) (wirbel #175 second moment)
  σ_hw       = 4.86 TPS                                       (kanna #159 denominator leg)
lcb_tps(λ) = central(λ) − H(λ)                               (THE margin-aware bar)

The provenance lock: σ_L(λ) is read from wirbel #175's `dp_accepted_length_pmf` on the
SAME spine #178's `et_backward` consumes, and the pmf-mean reproduces E[T](λ) to ~1e-15
at every λ (`pmf_mean_reproduces_et`). At λ=1 the *numerator-only* LCB reproduces wirbel
#175's published [524.527 both-bugs / 509.120 descent] exactly (`resid<0.5 TPS`); the
full LCB then adds σ_hw on top.

PRIMARY metric  lambda_acceptance_card_self_test_passes
TEST    metric  both_bugs_lambda_star_lcb  (margin-aware λ*, both-bugs, full LCB, τ=1)

Run:
    python -m research.oracle_readout.lambda_acceptance_card.lambda_acceptance_card \
        --self-test --wandb-name denken/lambda-acceptance-card \
        --wandb-group lambda-acceptance-card
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Import denken #178 (E[T](λ) interpolation) and wirbel #175 (σ_L pmf machinery).
# Both are imported verbatim; neither is re-derived. #178 itself imports #172's
# et_backward + composition constants, so D172 is reached through D178.
# --------------------------------------------------------------------------- #
_D178_PATH = REPO_ROOT / "research/oracle_readout/realistic_selfkv_floor/realistic_selfkv_floor.py"
_D175_PATH = REPO_ROOT / "research/oracle_readout/et_second_moment/et_second_moment.py"


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D178 = _import("realistic_selfkv_floor", _D178_PATH)
D175 = _import("et_second_moment", _D175_PATH)
D172 = D178.D172

# ---- composition constants (committed; imported, not re-derived) ---- #
K_CAL = D172.K_CAL                      # 125.268  (ubel #148 / #100, tree-invariant #169)
STEP = D172.STEP_OVERLAP                # 1.2182   (lawine #168 launch-realized step)
Z95 = D175.Z95                          # 1.959963984540054 (two-sided 95% normal quantile)
B_TOKENS = D175.BENCH_TOKENS            # 16384    (wirbel #175 primary benchmark budget)
B_TOKENS_ALT = D175.BENCH_TOKENS_ALT    # 65536    (sensitivity budget)
SIGMA_HW = 4.86                         # kanna #159 hardware-jitter denominator leg (1σ, TPS)
TARGET_OFFICIAL = D172.TARGET_OFFICIAL  # 500.0
MAXD = D175.MAXD_DEFAULT                # 24 (pvec build horizon; matches #175/#160)

TAU_CENTRAL = 1.0
TAU_CONSERVATIVE = 0.9924               # tree-class τ floor (ubel #181 not yet landed → floor)
TAU_CORNERS = (("tau_central_1p0", TAU_CENTRAL),
               ("tau_conservative_0p9924", TAU_CONSERVATIVE))

# ---- wirbel #175 published λ=1 numerator-only LCBs (reproduction targets) ---- #
WIRBEL_LCB_NUM_BOTH_BUGS = 524.527041138081
WIRBEL_LCB_NUM_DESCENT = 509.1203710494078
WIRBEL_SIGMA_L_BOTH_BUGS = 3.035436750248887
WIRBEL_SIGMA_L_DESCENT = 3.0592748771161733
WIRBEL_HALFWIDTH_BOTH_BUGS = 10.906182006867379   # ±10.906 numerator leg
RESID_TOL_TPS = 0.5                                # self-test (a) tolerance

# ---- #178 point-estimate λ* (central=500) — cross-check targets ---- #
D178_LAMBDA_STAR_DESCENT = {"tau_central_1p0": 0.9091326079857753,
                            "tau_conservative_0p9924": 0.9271221293937553}
D178_LAMBDA_STAR_BOTH_BUGS = {"tau_central_1p0": 0.8383898298915815,
                              "tau_conservative_0p9924": 0.8569525576400423}
# ---- #178 liveprobe central TPS at λ̂ (reproduction targets) ---- #
D178_LIVEPROBE_CENTRAL_DESCENT = 404.06468476135797
D178_LIVEPROBE_CENTRAL_BOTH_BUGS = 416.307156176311

# ---- serial-correlation sensitivity (honest scope): variance-inflation factors ---- #
# positive lag-1 autocorrelation ρ shrinks effective N → SE × √VIF, VIF≈(1+ρ)/(1−ρ).
SERIAL_CORR_VIF = (1.0, 1.5, 2.0)       # ρ≈0 / 0.2 / 0.33 ; SE scales by √VIF


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Topology endpoint spines (imported from #178's build_endpoints).
# --------------------------------------------------------------------------- #
def build_topologies(anchors: dict[str, Any]) -> dict[str, Any]:
    """The two committed #178 endpoint spines (q_floor, q_full) per topology, the tree
    structure, and rho_cond — all imported from #178 (which imports #172). descent-only
    and both-bugs differ ONLY at depth-1 (the BUG-1 axis); depth≥2 (the q[2..9] self-KV
    ladder) is shared and governed by the same recovery λ."""
    ep = D178.build_endpoints(anchors)
    children, depth = D172.build_children(ep["parent"])
    # descent-only: q_full depth-1 = 0.679 (BUG-1 unfixed), q_floor depth-1 = 0.674.
    q_full_d = list(ep["q_full"])
    q_floor_d = list(ep["q_floor"])
    # both-bugs: depth-1 fixed at the linear self-KV ref 0.7287 (BUG-1 fixed) for BOTH ends.
    d1_bb = ep["q_deployed"][0]
    q_full_bb = list(ep["q_deployed"])               # rising spine, depth-1 0.7287
    q_floor_bb = list(ep["q_floor"]); q_floor_bb[0] = d1_bb   # declining, depth-1 0.7287
    return {
        "ep": ep,
        "children": children,
        "depth": depth,
        "max_tree_depth": max(depth),
        "topo": {
            "descent_only": {"q_floor": q_floor_d, "q_full": q_full_d,
                             "d1_floor": q_floor_d[0], "d1_full": q_full_d[0]},
            "both_bugs": {"q_floor": q_floor_bb, "q_full": q_full_bb,
                          "d1_floor": q_floor_bb[0], "d1_full": q_full_bb[0]},
        },
    }


# --------------------------------------------------------------------------- #
# Core: E[T](λ), σ_L(λ), central TPS, finite-sample half-width, LCB.
# --------------------------------------------------------------------------- #
def metrics_at(ctx: dict, lam: float, q_floor: list[float], q_full: list[float],
               tau: float, b_tokens: int = B_TOKENS) -> dict[str, Any]:
    """Compose the finite-sample LCB at recovery λ for one topology / τ.

    E[T](λ) via #178's et_backward; σ_L(λ) via wirbel #175's accepted-length pmf on the
    SAME spine (pmf-mean reproduces E[T](λ) — the provenance lock). Numerator leg
    reproduces wirbel #175's CI; full leg adds kanna #159's σ_hw in quadrature.
    """
    ep = ctx["ep"]
    spine = D178.spine_from_profile(ep, D178.constant_lambda(len(q_full), lam), q_floor, q_full)
    et = D178.et_of_spine(ep, spine)                                   # #178 / #172 E[T](λ)

    # σ_L(λ): wirbel #175 second moment of the accepted-length pmf at this spine.
    pvecs = D175.build_depth_pvecs_measured(spine, ep["rho_cond"], ep["W"], MAXD, "flat")
    pmf, _, _, _ = D175.dp_accepted_length_pmf(ep["parent"], pvecs)
    mom = D175.pmf_moments(pmf)
    sigma_L = mom["std"]
    pmf_mean_resid = abs(mom["mean"] - et)                             # provenance lock ≈1e-15

    n_steps = b_tokens / et
    se_lbar = sigma_L / math.sqrt(n_steps)
    slope = K_CAL * tau / STEP                                         # dTPS/dE[T]
    central = slope * et
    se_tps = slope * se_lbar                                           # numerator SE (TPS)
    h_num = Z95 * se_tps                                               # wirbel #175 leg
    h_full = Z95 * math.sqrt(se_tps ** 2 + SIGMA_HW ** 2)             # ⊕ kanna #159 σ_hw
    return {
        "lambda": lam, "tau": tau, "E_T": et, "sigma_L": sigma_L,
        "pmf_mean_resid": pmf_mean_resid, "n_steps": n_steps, "se_tps": se_tps,
        "central_tps": central, "H_num": h_num, "H_full": h_full,
        "lcb_num_tps": central - h_num, "lcb_full_tps": central - h_full,
        "clears_500_lcb": bool(central - h_full >= TARGET_OFFICIAL),
        "spine": spine,
    }


def _bisect_lambda(f_at: Callable[[float], float], target: float) -> float:
    """Smallest λ∈[0,1] with f_at(λ) ≥ target (f monotone increasing). NaN if never."""
    lo, hi = 0.0, 1.0
    if f_at(hi) < target:
        return float("nan")
    if f_at(lo) >= target:
        return 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f_at(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def lambda_star_lcb(ctx: dict, q_floor: list[float], q_full: list[float], tau: float) -> float:
    """Margin-aware build bar: smallest λ whose FULL finite-sample LCB clears 500."""
    return _bisect_lambda(lambda lam: metrics_at(ctx, lam, q_floor, q_full, tau)["lcb_full_tps"],
                          TARGET_OFFICIAL)


def lambda_star_central(ctx: dict, q_floor: list[float], q_full: list[float], tau: float) -> float:
    """#178's POINT-estimate bar: smallest λ whose CENTRAL TPS hits 500 (LCB disabled)."""
    return _bisect_lambda(lambda lam: metrics_at(ctx, lam, q_floor, q_full, tau)["central_tps"],
                          TARGET_OFFICIAL)


# --------------------------------------------------------------------------- #
# Per-depth q[2..9] acceptance ladder (the literal card land #71 tests against).
# --------------------------------------------------------------------------- #
def per_depth_ladder(ctx: dict, lam: float, q_floor: list[float], q_full: list[float],
                     d_lo: int = 2, d_hi: int = 9) -> dict[str, Any]:
    """The depth-d conditional accept q_d(λ) the measured kernel must hit, for the self-KV
    depths d=2..9 (depth-1 is the separate BUG-1 axis). Uses #172's qd_at flat-extrapolation
    past the 7-entry spine (so depths 8,9 == depth-7 value), exactly as the E[T] DP does."""
    spine = D178.spine_from_profile(ctx["ep"], D178.constant_lambda(len(q_full), lam),
                                    q_floor, q_full)
    rows = []
    for d in range(d_lo, d_hi + 1):
        q_d = D172.qd_at(spine, d)
        q_fl = D172.qd_at(q_floor, d)
        q_fu = D172.qd_at(q_full, d)
        span = q_fu - q_fl
        # bind margin: how close q_d(λ*) sits to its full-recovery endpoint (smaller = more binding)
        headroom = q_fu - q_d
        rows.append({
            "depth": d, "q_floor": q_fl, "q_full": q_fu, "q_at_lambda_star": q_d,
            "span_full_minus_floor": span,
            "headroom_to_full": headroom,
            "extrapolated_flat": bool(d > len(spine)),
            "bracketed": bool(q_fl - 1e-12 <= q_d <= q_fu + 1e-12),
        })
    # most binding = the depth whose q_d(λ*) is the *largest absolute accept* it must hit,
    # i.e. the hardest per-depth target (closest to the rising-spine ceiling).
    binding = max(rows, key=lambda r: r["q_at_lambda_star"])
    return {
        "lambda_star": lam, "depths": f"{d_lo}..{d_hi}",
        "ladder": rows,
        "q_values": [r["q_at_lambda_star"] for r in rows],
        "all_bracketed": all(r["bracketed"] for r in rows),
        "most_binding_depth": binding["depth"],
        "most_binding_q": binding["q_at_lambda_star"],
    }


def lambda_from_measured_ladder(ctx: dict, q_meas_2to9: list[float], q_floor: list[float],
                                q_full: list[float], d_lo: int = 2) -> dict[str, Any]:
    """Inverse map for land #71: given a MEASURED per-depth q[2..9] ladder, the implied
    per-depth deficit-closure λ̂_d = (q_meas[d]−q_floor[d])/(q_full[d]−q_floor[d]) and the
    pooled λ̂_built = mean_d λ̂_d (generalizes #178's depth-1 liveprobe formula to all depths)."""
    per = []
    for i, q_m in enumerate(q_meas_2to9):
        d = d_lo + i
        q_fl = D172.qd_at(q_floor, d)
        q_fu = D172.qd_at(q_full, d)
        span = q_fu - q_fl
        lam_d = (q_m - q_fl) / span if abs(span) > 1e-12 else float("nan")
        per.append({"depth": d, "q_measured": q_m, "lambda_hat_d": lam_d})
    valid = [p["lambda_hat_d"] for p in per if _finite(p["lambda_hat_d"])]
    lam_built = sum(valid) / len(valid) if valid else float("nan")
    return {"per_depth": per, "lambda_hat_built": lam_built}


# --------------------------------------------------------------------------- #
# Forward map  measured-λ -> predicted LCB-TPS  (the go/no-go calculator).
# --------------------------------------------------------------------------- #
def forward_map(ctx: dict, q_floor: list[float], q_full: list[float], tau: float,
                lam_hat: float, lam_star_lcb: float, lam_star_central: float) -> dict[str, Any]:
    grid = sorted(set([0.0, 0.1, 0.2, 0.3, round(lam_hat, 5), 0.4, 0.5, 0.6, 0.7, 0.8,
                       round(lam_star_central, 5), 0.9, round(lam_star_lcb, 5), 0.95, 1.0]))
    rows = []
    prev_lcb = None
    monotone = True
    for lam in grid:
        m = metrics_at(ctx, lam, q_floor, q_full, tau)
        if prev_lcb is not None and m["lcb_full_tps"] < prev_lcb - 1e-9:
            monotone = False
        prev_lcb = m["lcb_full_tps"]
        rows.append({
            "lambda": lam, "E_T": m["E_T"], "sigma_L": m["sigma_L"],
            "central_tps": m["central_tps"], "H_full": m["H_full"],
            "predicted_lcb_tps": m["lcb_full_tps"], "predicted_lcb_clears_500": m["clears_500_lcb"],
            "is_liveprobe": bool(abs(lam - round(lam_hat, 5)) < 1e-9),
            "is_lambda_star_lcb": bool(abs(lam - round(lam_star_lcb, 5)) < 1e-9),
            "is_lambda_star_central": bool(abs(lam - round(lam_star_central, 5)) < 1e-9),
        })
    return {"tau": tau, "rows": rows, "card_is_monotone": monotone}


# --------------------------------------------------------------------------- #
# Serial-correlation sensitivity (honest scope): inflate the numerator SE.
# --------------------------------------------------------------------------- #
def sensitivity_inflated_H(ctx: dict, q_floor: list[float], q_full: list[float],
                           tau: float) -> dict[str, Any]:
    """If benchmark steps are positively serially correlated, effective N shrinks and the
    TRUE half-width is LARGER → λ* RISES. Report λ* under variance-inflation factors VIF
    (SE_tps ← SE_tps·√VIF) so land has the conservative bar too."""
    out = []
    for vif in SERIAL_CORR_VIF:
        scale = math.sqrt(vif)

        def lcb_inflated(lam: float) -> float:
            m = metrics_at(ctx, lam, q_floor, q_full, tau)
            h = Z95 * math.sqrt((scale * m["se_tps"]) ** 2 + SIGMA_HW ** 2)
            return m["central_tps"] - h

        lcb1 = lcb_inflated(1.0)
        ls = _bisect_lambda(lcb_inflated, TARGET_OFFICIAL)
        out.append({
            "vif": vif, "se_scale": scale, "implied_lag1_rho": (vif - 1.0) / (vif + 1.0),
            # None (not NaN) when even full recovery can't clear under this inflation.
            "lambda_star_lcb": (ls if math.isfinite(ls) else None),
            "clears_500_at_lambda1": bool(lcb1 >= TARGET_OFFICIAL),
            "lcb_at_lambda1": lcb1,
        })
    return {"tau": tau, "rows": out}


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize(anchors: dict[str, Any]) -> dict[str, Any]:
    ctx = build_topologies(anchors)
    topo = ctx["topo"]
    qfl_d, qfu_d = topo["descent_only"]["q_floor"], topo["descent_only"]["q_full"]
    qfl_b, qfu_b = topo["both_bugs"]["q_floor"], topo["both_bugs"]["q_full"]

    # liveprobe λ̂ (#178 formula, recomputed from the same anchors).
    lam_hat = ((D178.LIVEPROBE_WALK_TOPW0_HIT - qfl_d[0])
               / (D178.LIVEPROBE_LINEAR_TOP1 - qfl_d[0]))

    # ---------- (1) finite-sample LCB(λ): λ=1 reproduction of wirbel #175 ---------- #
    repro = {}
    for lab, (qfl, qfu, ref_lcb, ref_sig) in (
        ("descent_only", (qfl_d, qfu_d, WIRBEL_LCB_NUM_DESCENT, WIRBEL_SIGMA_L_DESCENT)),
        ("both_bugs", (qfl_b, qfu_b, WIRBEL_LCB_NUM_BOTH_BUGS, WIRBEL_SIGMA_L_BOTH_BUGS)),
    ):
        m1 = metrics_at(ctx, 1.0, qfl, qfu, TAU_CENTRAL)
        repro[lab] = {
            "E_T": m1["E_T"], "sigma_L": m1["sigma_L"],
            "central_tps": m1["central_tps"],
            "lcb_num_tps": m1["lcb_num_tps"], "lcb_full_tps": m1["lcb_full_tps"],
            "H_num": m1["H_num"], "H_full": m1["H_full"],
            "wirbel_lcb_num_ref": ref_lcb, "resid_lcb_num_vs_wirbel": abs(m1["lcb_num_tps"] - ref_lcb),
            "wirbel_sigma_L_ref": ref_sig, "resid_sigma_L_vs_wirbel": abs(m1["sigma_L"] - ref_sig),
            "pmf_mean_resid": m1["pmf_mean_resid"],
        }

    # ---------- (2) margin-aware λ* (full LCB=500) vs #178 point estimate ---------- #
    lam_star = {"descent_only": {}, "both_bugs": {}}
    lam_star_central_chk = {"descent_only": {}, "both_bugs": {}}
    for lab, (qfl, qfu, d178_pt) in (
        ("descent_only", (qfl_d, qfu_d, D178_LAMBDA_STAR_DESCENT)),
        ("both_bugs", (qfl_b, qfu_b, D178_LAMBDA_STAR_BOTH_BUGS)),
    ):
        for tag, tau in TAU_CORNERS:
            ls_lcb = lambda_star_lcb(ctx, qfl, qfu, tau)
            ls_cen = lambda_star_central(ctx, qfl, qfu, tau)
            lam_star[lab][tag] = {
                "lambda_star_lcb": ls_lcb,
                "lambda_star_central_point": ls_cen,
                "d178_point_estimate": d178_pt[tag],
                "delta_lambda_lcb_over_point": ls_lcb - d178_pt[tag],
                "lcb_bar_is_stricter": bool(ls_lcb >= d178_pt[tag] - 1e-9),
                "in_unit": bool(0.0 <= ls_lcb <= 1.0),
            }
            lam_star_central_chk[lab][tag] = abs(ls_cen - d178_pt[tag])

    both_bugs_lambda_star_lcb = lam_star["both_bugs"]["tau_central_1p0"]["lambda_star_lcb"]   # TEST
    descent_only_lambda_star_lcb = lam_star["descent_only"]["tau_central_1p0"]["lambda_star_lcb"]

    # ---------- (3) per-depth q[2..9] acceptance ladder at λ* (both-bugs) ---------- #
    ladder_bb = {tag: per_depth_ladder(ctx, lam_star["both_bugs"][tag]["lambda_star_lcb"],
                                       qfl_b, qfu_b)
                 for tag, _ in TAU_CORNERS}
    ladder_descent = {tag: per_depth_ladder(ctx, lam_star["descent_only"][tag]["lambda_star_lcb"],
                                            qfl_d, qfu_d)
                      for tag, _ in TAU_CORNERS}

    # ---------- (4) forward map (go/no-go calculator) ---------- #
    fwd = {}
    for lab, (qfl, qfu) in (("descent_only", (qfl_d, qfu_d)), ("both_bugs", (qfl_b, qfu_b))):
        fwd[lab] = {tag: forward_map(ctx, qfl, qfu, tau, lam_hat,
                                     lam_star[lab][tag]["lambda_star_lcb"],
                                     lam_star[lab][tag]["lambda_star_central_point"])
                    for tag, tau in TAU_CORNERS}

    # liveprobe λ̂ row explicit (reproduces #178 central misses) at τ=1.
    liveprobe = {}
    for lab, (qfl, qfu, ref_central) in (
        ("descent_only", (qfl_d, qfu_d, D178_LIVEPROBE_CENTRAL_DESCENT)),
        ("both_bugs", (qfl_b, qfu_b, D178_LIVEPROBE_CENTRAL_BOTH_BUGS)),
    ):
        m = metrics_at(ctx, lam_hat, qfl, qfu, TAU_CENTRAL)
        liveprobe[lab] = {
            "lambda_hat": lam_hat, "E_T": m["E_T"], "central_tps": m["central_tps"],
            "predicted_lcb_tps": m["lcb_full_tps"], "clears_500_lcb": m["clears_500_lcb"],
            "d178_central_ref": ref_central,
            "resid_central_vs_d178": abs(m["central_tps"] - ref_central),
            "far_below_lambda_star": bool(lam_hat < (descent_only_lambda_star_lcb
                                                     if lab == "descent_only"
                                                     else both_bugs_lambda_star_lcb)),
        }

    # inverse-map self-consistency demo: feed q_floor / q_full / spine(λ*) ladders back.
    inv_demo = {}
    for lab, (qfl, qfu, ls) in (
        ("descent_only", (qfl_d, qfu_d, descent_only_lambda_star_lcb)),
        ("both_bugs", (qfl_b, qfu_b, both_bugs_lambda_star_lcb)),
    ):
        sp_star = D178.spine_from_profile(ctx["ep"], D178.constant_lambda(len(qfu), ls), qfl, qfu)
        ladder_star = [D172.qd_at(sp_star, d) for d in range(2, 10)]
        ladder_floor = [D172.qd_at(qfl, d) for d in range(2, 10)]
        ladder_full = [D172.qd_at(qfu, d) for d in range(2, 10)]
        inv_demo[lab] = {
            "recover_lambda_star_from_its_own_ladder":
                lambda_from_measured_ladder(ctx, ladder_star, qfl, qfu)["lambda_hat_built"],
            "recover_0_from_floor_ladder":
                lambda_from_measured_ladder(ctx, ladder_floor, qfl, qfu)["lambda_hat_built"],
            "recover_1_from_full_ladder":
                lambda_from_measured_ladder(ctx, ladder_full, qfl, qfu)["lambda_hat_built"],
            "target_lambda_star": ls,
        }

    # ---------- serial-correlation sensitivity (conservative bar) ---------- #
    sensitivity = {lab: {tag: sensitivity_inflated_H(ctx, qfl, qfu, tau)
                         for tag, tau in TAU_CORNERS}
                   for lab, (qfl, qfu) in (("descent_only", (qfl_d, qfu_d)),
                                           ("both_bugs", (qfl_b, qfu_b)))}

    # ---------- (5) self-test (PRIMARY) ---------- #
    # (a) λ=1 numerator-only LCB reproduces wirbel 524.5/509.1 within 0.5 TPS.
    cond_a = bool(repro["both_bugs"]["resid_lcb_num_vs_wirbel"] < RESID_TOL_TPS
                  and repro["descent_only"]["resid_lcb_num_vs_wirbel"] < RESID_TOL_TPS)
    # (b) margin-aware λ* ≥ #178 point estimate for both topologies, both ∈ [0,1] (τ=1).
    cond_b = bool(
        lam_star["both_bugs"]["tau_central_1p0"]["lcb_bar_is_stricter"]
        and lam_star["descent_only"]["tau_central_1p0"]["lcb_bar_is_stricter"]
        and lam_star["both_bugs"]["tau_central_1p0"]["in_unit"]
        and lam_star["descent_only"]["tau_central_1p0"]["in_unit"]
    )
    # (c) per-depth q[2..9] ladder at λ* (both-bugs) reported + bracketed by #178 endpoints.
    cond_c = bool(ladder_bb["tau_central_1p0"]["all_bracketed"]
                  and len(ladder_bb["tau_central_1p0"]["ladder"]) == 8)
    # (d) forward map monotone (both topologies, τ=1) AND λ̂ row reproduces #178 central.
    cond_d = bool(
        fwd["both_bugs"]["tau_central_1p0"]["card_is_monotone"]
        and fwd["descent_only"]["tau_central_1p0"]["card_is_monotone"]
        and liveprobe["descent_only"]["resid_central_vs_d178"] < RESID_TOL_TPS
        and liveprobe["both_bugs"]["resid_central_vs_d178"] < RESID_TOL_TPS
    )
    # provenance lock: pmf-mean reproduces E[T] at λ=1 (≈1e-15).
    cond_prov = bool(repro["both_bugs"]["pmf_mean_resid"] < 1e-9
                     and repro["descent_only"]["pmf_mean_resid"] < 1e-9)
    # central-λ* cross-check reproduces #178 point estimates (sanity that the curve matches).
    cond_central_match = bool(all(v < 1e-3 for lab in lam_star_central_chk
                                  for v in lam_star_central_chk[lab].values()))

    self_test_passes = bool(cond_a and cond_b and cond_c and cond_d
                            and cond_prov and cond_central_match)

    card_is_monotone = bool(fwd["both_bugs"]["tau_central_1p0"]["card_is_monotone"]
                            and fwd["descent_only"]["tau_central_1p0"]["card_is_monotone"])

    handoff = _handoff_lines(both_bugs_lambda_star_lcb=both_bugs_lambda_star_lcb,
                             descent_lambda_star_lcb=descent_only_lambda_star_lcb,
                             d178_point_bb=D178_LAMBDA_STAR_BOTH_BUGS["tau_central_1p0"],
                             delta_bb=lam_star["both_bugs"]["tau_central_1p0"]["delta_lambda_lcb_over_point"],
                             lam_hat=lam_hat,
                             ladder_bb=ladder_bb["tau_central_1p0"]["q_values"],
                             binding_depth=ladder_bb["tau_central_1p0"]["most_binding_depth"],
                             liveprobe_lcb_bb=liveprobe["both_bugs"]["predicted_lcb_tps"])

    return {
        "self_test": {
            "lambda_acceptance_card_self_test_passes": self_test_passes,
            "conditions": {
                "a_lambda1_lcb_reproduces_wirbel": cond_a,
                "b_margin_aware_lambda_star_stricter_and_in_unit": cond_b,
                "c_per_depth_ladder_reported_and_bracketed": cond_c,
                "d_forward_map_monotone_and_liveprobe_reproduces_178": cond_d,
                "prov_pmf_mean_reproduces_et": cond_prov,
                "central_lambda_star_matches_178_point": cond_central_match,
            },
        },
        "test_metric": {
            "both_bugs_lambda_star_lcb": both_bugs_lambda_star_lcb,
            "descent_only_lambda_star_lcb": descent_only_lambda_star_lcb,
        },
        "composition": {
            "formula_central": "central(lam) = K_cal*(E_T(lam)/step)*tau",
            "formula_halfwidth": "H(lam) = z95*sqrt( (K_cal*tau/step*sigma_L(lam)/sqrt(N_steps))^2 + sigma_hw^2 )",
            "formula_lcb": "lcb_tps(lam) = central(lam) - H(lam)",
            "K_cal": K_CAL, "step": STEP, "z95": Z95, "B_tokens": B_TOKENS,
            "sigma_hw_tps": SIGMA_HW, "tau_central": TAU_CENTRAL,
            "tau_conservative": TAU_CONSERVATIVE, "max_tree_depth": ctx["max_tree_depth"],
        },
        "lambda_hat_liveprobe": lam_hat,
        "endpoints": {
            "descent_only": {"q_floor": qfl_d, "q_full": qfu_d,
                             "E_T_floor": metrics_at(ctx, 0.0, qfl_d, qfu_d, TAU_CENTRAL)["E_T"],
                             "E_T_full": repro["descent_only"]["E_T"]},
            "both_bugs": {"q_floor": qfl_b, "q_full": qfu_b,
                          "E_T_floor": metrics_at(ctx, 0.0, qfl_b, qfu_b, TAU_CENTRAL)["E_T"],
                          "E_T_full": repro["both_bugs"]["E_T"]},
        },
        "lcb_lambda1_reproduction": repro,
        "margin_aware_lambda_star": lam_star,
        "per_depth_acceptance_ladder": {
            "both_bugs": ladder_bb, "descent_only": ladder_descent,
            "note": ("q[2..9] = self-KV-governed depths (depth-1 is the separate BUG-1 axis). "
                     "Spine parameterizes depths 1-7; depths 8-9 are flat-extrapolated "
                     "(== depth-7 value), exactly as the imported E[T] DP's qd_at clamps."),
        },
        "forward_map": fwd,
        "forward_map_inverse": {
            "formula": "lambda_hat_d = (q_meas[d]-q_floor[d])/(q_full[d]-q_floor[d]); lambda_hat_built = mean_d",
            "self_consistency_demo": inv_demo,
            "note": "land #71 feeds its measured q[2..9] -> implied lambda_hat_built -> read predicted_lcb_tps off forward_map.",
        },
        "liveprobe_row": liveprobe,
        "card_is_monotone": card_is_monotone,
        "serial_correlation_sensitivity": sensitivity,
        "verdict": _verdict(both_bugs_lambda_star_lcb, descent_only_lambda_star_lcb,
                            D178_LAMBDA_STAR_BOTH_BUGS["tau_central_1p0"], lam_hat),
        "handoff_lines": handoff,
    }


def _verdict(bb_lcb: float, desc_lcb: float, d178_bb: float, lam_hat: float) -> str:
    return (
        f"MARGIN-AWARE-CARD-STAMPED. The finite-sample-LCB build bar is STRICTER than #178's "
        f"point estimate: both-bugs must demonstrate per-depth self-KV recovery lambda >= "
        f"{bb_lcb:.4f} (vs #178's {d178_bb:.4f} central point estimate; +{bb_lcb - d178_bb:.4f} "
        f"more recovery for the 95% LCB to clear 500), descent-only lambda >= {desc_lcb:.4f}. "
        f"The one pre-build measured point lambda_hat={lam_hat:.3f} (liveprobe) is far below "
        f"both bars. land #71's measured q[2..9] ladder is tested against the per-depth card; "
        f"the forward map converts any measured ladder to a predicted LCB-TPS go/no-go. NOT a launch."
    )


def _handoff_lines(*, both_bugs_lambda_star_lcb: float, descent_lambda_star_lcb: float,
                   d178_point_bb: float, delta_bb: float, lam_hat: float,
                   ladder_bb: list[float], binding_depth: int, liveprobe_lcb_bb: float) -> dict[str, str]:
    ladder_str = "[" + ", ".join(f"{q:.4f}" for q in ladder_bb) + "]"
    fern = (
        f"BUILD LINE (denken #183, margin-aware λ-acceptance card): land #71's kernel must "
        f"demonstrate per-depth self-KV recovery λ ≥ {both_bugs_lambda_star_lcb:.4f} (both-bugs, "
        f"finite-sample-LCB-clears-500 bar — wirbel #175's ±10.9 ⊕ kanna #159's σ_hw=4.86), "
        f"STRICTER than #178's {d178_point_bb:.4f} central point estimate by Δλ={delta_bb:+.4f}; "
        f"the measured q[2..9] ladder is tested against the per-depth card (depth {binding_depth} "
        f"most binding); the one pre-build measured point λ̂={lam_hat:.3f} (liveprobe) is far below."
    )
    land = (
        f"land #71 acceptance: hit per-depth q[2..9] ≥ {ladder_str} (both-bugs, at the LCB bar "
        f"λ*={both_bugs_lambda_star_lcb:.4f}); depth {binding_depth} is the hardest target. "
        f"Forward calculator: feed your measured q[2..9] → implied λ̂_built → read predicted "
        f"LCB-TPS; clear iff predicted-LCB ≥ 500. (At the unbuilt λ̂={lam_hat:.3f}: LCB≈"
        f"{liveprobe_lcb_bb:.0f}, a clear miss.)"
    )
    return {"fern_179_packet": fern, "land_71_calculator": land}


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
    st = syn["self_test"]
    comp = syn["composition"]
    print("\n" + "=" * 90, flush=True)
    print("MARGIN-AWARE λ-ACCEPTANCE CARD (PR #183) — the finite-sample-LCB build bar", flush=True)
    print("=" * 90, flush=True)
    print(f"  composition: central=K_cal·(E[T]/step)·τ ; H=z95·√(SE_tps² + σ_hw²) ; "
          f"σ_hw={comp['sigma_hw_tps']} TPS ; B={comp['B_tokens']}", flush=True)
    print("-" * 90, flush=True)
    print("  (1) λ=1 LCB reproduces wirbel #175:", flush=True)
    for lab in ("both_bugs", "descent_only"):
        r = syn["lcb_lambda1_reproduction"][lab]
        print(f"      {lab:<13} σ_L={r['sigma_L']:.6f} (ref {r['wirbel_sigma_L_ref']:.6f}) "
              f"central={r['central_tps']:.3f}  lcb_num={r['lcb_num_tps']:.4f} "
              f"(wirbel {r['wirbel_lcb_num_ref']:.4f}, resid {r['resid_lcb_num_vs_wirbel']:.2e}) "
              f"| lcb_full(+σ_hw)={r['lcb_full_tps']:.3f}", flush=True)
    print("-" * 90, flush=True)
    print("  (2) margin-aware λ* (full LCB=500) vs #178 point estimate:", flush=True)
    for lab in ("both_bugs", "descent_only"):
        for tag, _ in TAU_CORNERS:
            v = syn["margin_aware_lambda_star"][lab][tag]
            print(f"      {lab:<13} {tag:<22} λ*_lcb={v['lambda_star_lcb']:.4f}  "
                  f"#178_point={v['d178_point_estimate']:.4f}  Δλ={v['delta_lambda_lcb_over_point']:+.4f}  "
                  f"stricter={v['lcb_bar_is_stricter']}", flush=True)
    print(f"      TEST both_bugs_lambda_star_lcb = {syn['test_metric']['both_bugs_lambda_star_lcb']:.6f}",
          flush=True)
    print("-" * 90, flush=True)
    print("  (3) per-depth q[2..9] acceptance ladder at λ* (both-bugs, τ=1):", flush=True)
    lad = syn["per_depth_acceptance_ladder"]["both_bugs"]["tau_central_1p0"]
    for r in lad["ladder"]:
        flag = " (flat-extrap)" if r["extrapolated_flat"] else ""
        print(f"      depth {r['depth']}: q*={r['q_at_lambda_star']:.4f}  "
              f"[floor {r['q_floor']:.4f} → full {r['q_full']:.4f}]  "
              f"headroom={r['headroom_to_full']:.4f}{flag}", flush=True)
    print(f"      most-binding depth={lad['most_binding_depth']} (q={lad['most_binding_q']:.4f}); "
          f"all bracketed={lad['all_bracketed']}", flush=True)
    print("-" * 90, flush=True)
    print("  (4) forward map (both-bugs, τ=1)  λ → predicted LCB-TPS:", flush=True)
    for r in syn["forward_map"]["both_bugs"]["tau_central_1p0"]["rows"]:
        marks = []
        if r["is_liveprobe"]:
            marks.append("λ̂")
        if r["is_lambda_star_central"]:
            marks.append("#178λ*")
        if r["is_lambda_star_lcb"]:
            marks.append("LCBλ*")
        mk = (" <- " + ",".join(marks)) if marks else ""
        print(f"      λ={r['lambda']:.5f}  E[T]={r['E_T']:.4f}  LCB={r['predicted_lcb_tps']:.2f}  "
              f"clears500={r['predicted_lcb_clears_500']}{mk}", flush=True)
    print(f"      card_is_monotone={syn['card_is_monotone']}", flush=True)
    print("-" * 90, flush=True)
    print("  serial-correlation sensitivity (both-bugs, τ=1) — conservative bar:", flush=True)
    for r in syn["serial_correlation_sensitivity"]["both_bugs"]["tau_central_1p0"]["rows"]:
        print(f"      VIF={r['vif']:.1f} (ρ≈{r['implied_lag1_rho']:.2f}, SE×{r['se_scale']:.3f})  "
              f"λ*_lcb={r['lambda_star_lcb']:.4f}", flush=True)
    print("-" * 90, flush=True)
    print(f"  (5) PRIMARY lambda_acceptance_card_self_test_passes = "
          f"{st['lambda_acceptance_card_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"          - {k}: {v}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 90, flush=True)
    print(f"\n  HAND-OFF (fern #179): {syn['handoff_lines']['fern_179_packet']}", flush=True)
    print(f"\n  HAND-OFF (land #71): {syn['handoff_lines']['land_71_calculator']}\n", flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors #178; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict, anchors: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb,
            init_wandb_run,
            log_json_artifact,
            log_summary,
        )
    except Exception as exc:
        print(f"[lambda-card] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    run = init_wandb_run(
        job_type="lambda-acceptance-card",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["lambda-acceptance-card", "validity-gate", "finite-sample-lcb", "build-bar"],
        config={
            "K_cal": K_CAL, "step": STEP, "z95": Z95, "B_tokens": B_TOKENS,
            "sigma_hw_tps": SIGMA_HW, "tau_central": TAU_CENTRAL,
            "tau_conservative": TAU_CONSERVATIVE,
            "imports": "denken#178 E[T](λ) + wirbel#175 σ_L pmf + kanna#159 σ_hw",
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[lambda-card] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    st = syn["self_test"]
    tm = syn["test_metric"]
    msbb1 = syn["margin_aware_lambda_star"]["both_bugs"]["tau_central_1p0"]
    msd1 = syn["margin_aware_lambda_star"]["descent_only"]["tau_central_1p0"]
    rbb = syn["lcb_lambda1_reproduction"]["both_bugs"]
    rd = syn["lcb_lambda1_reproduction"]["descent_only"]
    lp = syn["liveprobe_row"]
    lad = syn["per_depth_acceptance_ladder"]["both_bugs"]["tau_central_1p0"]
    summary: dict[str, Any] = {
        "lambda_acceptance_card_self_test_passes": int(bool(st["lambda_acceptance_card_self_test_passes"])),
        "both_bugs_lambda_star_lcb": tm["both_bugs_lambda_star_lcb"],
        "descent_only_lambda_star_lcb": tm["descent_only_lambda_star_lcb"],
        # margin-aware vs point estimate
        "both_bugs_lambda_star_lcb_tau_cons":
            syn["margin_aware_lambda_star"]["both_bugs"]["tau_conservative_0p9924"]["lambda_star_lcb"],
        "both_bugs_d178_point_estimate": msbb1["d178_point_estimate"],
        "both_bugs_delta_lambda_over_point": msbb1["delta_lambda_lcb_over_point"],
        "descent_d178_point_estimate": msd1["d178_point_estimate"],
        "descent_delta_lambda_over_point": msd1["delta_lambda_lcb_over_point"],
        # λ=1 reproduction
        "lambda1_lcb_num_both_bugs": rbb["lcb_num_tps"],
        "lambda1_resid_lcb_num_vs_wirbel_both_bugs": rbb["resid_lcb_num_vs_wirbel"],
        "lambda1_resid_lcb_num_vs_wirbel_descent": rd["resid_lcb_num_vs_wirbel"],
        "lambda1_lcb_full_both_bugs": rbb["lcb_full_tps"],
        "lambda1_lcb_full_descent": rd["lcb_full_tps"],
        "pmf_mean_resid_both_bugs": rbb["pmf_mean_resid"],
        # liveprobe row
        "lambda_hat_liveprobe": syn["lambda_hat_liveprobe"],
        "liveprobe_central_both_bugs": lp["both_bugs"]["central_tps"],
        "liveprobe_central_descent": lp["descent_only"]["central_tps"],
        "liveprobe_lcb_both_bugs": lp["both_bugs"]["predicted_lcb_tps"],
        "liveprobe_resid_central_vs_178_both_bugs": lp["both_bugs"]["resid_central_vs_d178"],
        # ladder
        "ladder_most_binding_depth": lad["most_binding_depth"],
        "ladder_most_binding_q": lad["most_binding_q"],
        "ladder_all_bracketed": int(bool(lad["all_bracketed"])),
        "card_is_monotone": int(bool(syn["card_is_monotone"])),
        # sensitivity (VIF=2.0)
        "lambda_star_lcb_vif2_both_bugs":
            syn["serial_correlation_sensitivity"]["both_bugs"]["tau_central_1p0"]["rows"][-1]["lambda_star_lcb"],
        "K_cal": K_CAL, "sigma_hw_tps": SIGMA_HW,
        "verdict_stamped": int(syn["verdict"].startswith("MARGIN-AWARE-CARD-STAMPED")),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items() if not (isinstance(v, float) and not math.isfinite(v))}

    log_summary(run, summary, step=0)
    # per-depth ladder + forward map as tables, if helper supports it; else artifact only.
    log_json_artifact(run, name="lambda_acceptance_card_result", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[lambda-card] wandb logged: {summary}", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--bug2-anchor", type=Path, default=D172.DEFAULT_BUG2_ANCHOR)
    ap.add_argument("--topo-json", type=Path, default=D172.DEFAULT_TOPO_JSON)
    ap.add_argument("--accept-json", type=Path, default=D172.DEFAULT_ACCEPT_JSON)
    ap.add_argument("--rankcov-json", type=Path, default=D172.DEFAULT_RANKCOV_JSON)
    ap.add_argument("--decomp-json", type=Path, default=D172.DEFAULT_DECOMP_JSON)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="lambda-acceptance-card")
    args = ap.parse_args(argv)

    anchors = D172.load_anchors(
        args.bug2_anchor, args.topo_json, args.accept_json, args.rankcov_json, args.decomp_json
    )
    syn = synthesize(anchors)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at,
        "pr": 183,
        "agent": "denken",
        "kind": "lambda-acceptance-card",
        "anchors": {k: v for k, v in anchors.items() if k != "_paths"},
        "anchor_paths": anchors.get("_paths"),
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[lambda-card] WARNING non-finite values at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lambda_acceptance_card_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[lambda-card] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload, anchors)

    if args.self_test:
        ok = syn["self_test"]["lambda_acceptance_card_self_test_passes"] and payload["nan_clean"]
        print(f"[lambda-card] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
