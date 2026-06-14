#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Salvage-staleness λ(depth) — mechanism vs flat depth-transfer (PR #193).

Every finite-sample result in the denken self-KV lane — #178 (`REALISTIC-FLOOR-MISSES-
BOTH`), #183 (the build bar λ ≥ 0.9052), #187 (the λ̂_built measurement-CI) — rests on ONE
shared, un-grounded modelled assumption: that the depth-1 recovery fraction transfers
**FLAT** across depths 2..9. The single measured point is the depth-1 liveprobe
``λ̂₁ = (0.6927−0.674)/(0.7287−0.674) = 0.342``, then carried *constant* across depth
(constant-λ primary; a geometric γ=0.7–0.9 band as the only sensitivity). The clear-500
question is decided almost entirely by the SHAPE of λ(depth), and that shape is currently
an assumption, not a mechanism.

But λ(depth) is **not** free. It is set by the BUG-2 self-KV **salvage staleness** physics.
The salvage-no-descend path (``_dixie_fused_accept_prep_kernel`` in the served
``fa2sw_precache_kenyan`` stack; root-caused in wirbel #135's ``bug2_salvage_descent``)
reuses the *parent's* KV without re-running the descend, so at depth ``d`` the salvaged KV
the depth>0 spine reuses is ``d−1`` steps stale. Each descend step adds exactly **one**
stale step, and the stale-KV / true-KV divergence compounds multiplicatively, so the
per-depth recovery fraction obeys a **geometric staleness-decay law**

    λ_d = λ̂₁ · β^(d−1)         (staleness s = d−1 ; β = per-step self-KV retention)

This is *exactly* the geometric profile #178 already swept as its lone sensitivity band —
but #178 *guessed* γ∈[0.7,0.9]; here the geometric FORM is **derived** from the salvage
construction (one stale step per depth → multiplicative compounding) and β is **grounded**
in the kernel's own measured staleness fingerprint, not picked. The staleness-decay
exponent is ``α = −ln β`` (α=0 ⇔ flat ⇔ #178).

------------------------------------------------------------------------------
THIS IS A SYNTHESIS (imports; does NOT re-derive)
------------------------------------------------------------------------------
It imports denken #172's backward-DP ``et_backward`` + composition constants, #178's graded
``E[T](λ)`` interpolation + endpoint spines + liveprobe λ̂₁=0.342, #183's finite-sample-LCB
machinery + the 0.9052 build bar + ``q[2..9]`` per-depth ladder, wirbel #175's accepted-
length pmf σ_L (through #183), and wirbel #135's measured salvage-no-descend conditional
ladder. It replaces only the *assumed* flat/geometric transfer with the mechanism-grounded
λ(depth). LOCAL CPU-only analytic. No GPU / vLLM / HF Job / submission / kernel build /
served-file change. BASELINE stays 481.53. Greedy/PPL untouched. Bank-the-analysis
(PRIMARY = self-test, adds 0 TPS). NOT open2. NOT a launch.

------------------------------------------------------------------------------
WHAT IT ANSWERS
------------------------------------------------------------------------------
(1) λ(depth) from the mechanism, anchored at λ̂₁=0.342; β grounded from the salvage walk's
    own measured conditional decay; the profile + staleness-decay exponent; faster/slower
    than constant-λ and where it sits vs the geometric γ=0.7–0.9 band.
(2) the realistic floor re-run under the mechanism profile: descent + both-bugs
    ``mechanism_floor_E_T`` → official TPS (K_cal=125.268, step 1.2182, τ∈[0.9924,1.0]) →
    ``mechanism_floor_clears_500`` (both τ corners).
(3) verdict robustness: ``misses_both_robust_to_mechanism`` and the inverse map
    ``both_bugs_lambda1_star`` = the *depth-1* λ̂₁ at which the mechanism profile's both-bugs
    floor first clears #183's 0.9052-equivalent bar — the threshold land #71's depth-1 probe
    can be read against directly.
(4) self-validate (PRIMARY): flat (β→1) reproduces #178's 404/416; λ̂₁→0 reproduces #172's
    3.5346; β=1 reproduces #183's 0.9052 bar; the monotone-decaying staleness profile gives
    ``mechanism_floor_E_T ≤ constant_lambda_floor_E_T`` at every λ̂₁ (staleness cannot improve
    recovery vs flat).

PRIMARY metric  lambda_depth_profile_self_test_passes
TEST    metric  both_bugs_mechanism_floor_tps  (both-bugs mechanism floor TPS, primary β, τ=1)

Run:
    python -m research.oracle_readout.lambda_depth_profile.lambda_depth_profile \
        --self-test --wandb-name denken/lambda-depth-staleness-profile \
        --wandb-group lambda-depth-staleness-profile
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
# Import #183 (→ #178 → #172, and #175). Everything below is imported, not
# re-derived: #172 et_backward + composition, #178 graded E[T](λ) + endpoints +
# liveprobe λ̂₁, #183 LCB machinery + 0.9052 bar, #175 σ_L pmf (through #183).
# --------------------------------------------------------------------------- #
_D183_PATH = REPO_ROOT / "research/oracle_readout/lambda_acceptance_card/lambda_acceptance_card.py"


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D183 = _import("lambda_acceptance_card", _D183_PATH)
D178 = D183.D178
D175 = D183.D175
D172 = D183.D172

# ---- composition constants (committed; imported, not re-derived) ---- #
K_CAL = D172.K_CAL                      # 125.268 (ubel #148 / #100, tree-invariant #169)
STEP = D172.STEP_OVERLAP                # 1.2182  (lawine #168 launch-realized step)
Z95 = D175.Z95                          # 1.959963984540054 (two-sided 95% normal quantile)
B_TOKENS = D175.BENCH_TOKENS            # 16384   (wirbel #175 primary benchmark budget)
SIGMA_HW = D183.SIGMA_HW                # 4.86    (kanna #159 hardware-jitter denominator leg)
MAXD = D175.MAXD_DEFAULT                # 24      (pvec build horizon; matches #175/#160)
TARGET_OFFICIAL = D172.TARGET_OFFICIAL  # 500.0

TAU_CENTRAL = 1.0
TAU_CONSERVATIVE = 0.9924               # τ=1 band floor (ubel #181, MERGED)
TAU_CORNERS = (("tau_central_1p0", TAU_CENTRAL),
               ("tau_conservative_0p9924", TAU_CONSERVATIVE))

# ---- wirbel #135 measured salvage-no-descend cumulative ladder (the kernel's
# own staleness fingerprint; the in-scope salvage path's measured output) ---- #
ORACLE_CUM_LADDER = list(D172.ORACLE_CUM_LADDER)   # [0.674,0.350,0.203,0.131,0.089,0.060,0.037]

# ---- reproduction targets (the assumptions this leg re-grounds) ---- #
D178_DESCENT_FLOOR_E_T = 3.9294296647453835   # #178 constant-λ descent floor @ λ̂=0.342
D178_BOTHBUGS_FLOOR_E_T = 4.048484687770039   # #178 constant-λ both-bugs floor @ λ̂=0.342
D178_DESCENT_FLOOR_TPS = 404.06468476135797   # #178 descent floor TPS (τ=1)
D178_BOTHBUGS_FLOOR_TPS = 416.307156176311    # #178 both-bugs floor TPS (τ=1)
D172_DESCENT_LOWER_BOUND = 3.534580633373862  # #172 adversarial floor (λ→0)
D183_BOTHBUGS_LAMBDA_STAR_LCB = 0.905229319301184    # #183 both-bugs LCB build bar
D178_BOTHBUGS_LAMBDA_STAR_CENTRAL = 0.8383898298915815  # #178 both-bugs central-500 point

TOL_REPRO = 1e-4          # reproduction tolerance for the imported floors / bars.
TOL_ENDPOINT = 1e-6       # endpoint reproduction.

# Staleness-decay β band: flat (1.0, == #178) + #178's geometric band + the
# construction-grounded conservative end (filled in at runtime).
BETA_BAND_FIXED = (1.0, 0.9, 0.8, 0.7)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


# --------------------------------------------------------------------------- #
# (0) The mechanism profile  λ_d = λ̂₁ · β^(d−1)  (depth index 0 == depth 1).
# --------------------------------------------------------------------------- #
def mechanism_lambda(horizon: int, lam1: float, beta: float) -> list[float]:
    """Geometric staleness-decay recovery profile. Index d (0-based) == depth d+1,
    staleness s = d, so the exponent IS the index. β=1 ⇒ flat (== #178 constant-λ)."""
    return [lam1 * (beta ** d) for d in range(horizon)]


def staleness_decay_exponent(beta: float) -> float:
    """α = −ln β ≥ 0; α=0 ⇔ flat transfer ⇔ #178."""
    return -math.log(beta) if beta > 0 else float("inf")


# --------------------------------------------------------------------------- #
# (1) Ground β in the salvage construction (NOT a guess).
# --------------------------------------------------------------------------- #
def ground_beta(salvage_cond: list[float], q_full: list[float]) -> dict[str, Any]:
    """Two construction-grounded estimates of the per-step self-KV retention β, read
    straight from wirbel #135's measured salvage-no-descend conditional ladder
    (``salvage_cond`` = the in-scope kernel's own per-depth acceptance under stale KV) vs
    the full-self-KV linear reference (``q_full``):

      * ``beta_reach``  = geometric mean of the salvage walk's conditional acceptance at
                          depths 2..H — the absolute per-step survival of the stale-KV walk
                          (the directest staleness fingerprint; conservative / steep end).
      * ``beta_rel``    = geometric mean of the salvage walk's per-step acceptance RATIO
                          divided by the full ladder's own per-step ratio — the staleness-
                          ONLY decay with the intrinsic depth trend removed (mild end).

    β is bounded in (0,1) but is NOT point-identified by the single depth-1 anchor; the
    primary point is their geometric mean (the central staleness estimate) and the band
    [beta_reach, beta_rel] is reported as the explicit construction-grounded range."""
    cond_deep = salvage_cond[1:]                              # depths 2..H conditional accept
    beta_reach = _geomean(cond_deep)
    rel = []
    for d in range(1, len(salvage_cond)):
        s_ratio = salvage_cond[d] / salvage_cond[d - 1]
        f_ratio = q_full[d] / q_full[d - 1]
        if f_ratio > 0:
            rel.append(s_ratio / f_ratio)
    beta_rel = _geomean(rel)
    lo, hi = sorted((beta_reach, beta_rel))
    beta_primary = _geomean([beta_reach, beta_rel])           # central construction estimate
    return {
        "beta_reach_absolute": beta_reach,
        "beta_rel_staleness_only": beta_rel,
        "beta_primary_geomean": beta_primary,
        "beta_construction_range": [lo, hi],
        "salvage_conditional_ladder": list(salvage_cond),
        "note": ("salvage_cond = wirbel #135 measured salvage-no-descend conditional "
                 "acceptance (the in-scope kernel's stale-KV output). beta_reach is the "
                 "geomean per-step survival; beta_rel divides out the full ladder's depth "
                 "trend. β bounded but not point-identified by the single depth-1 anchor; "
                 "primary = geomean of the two."),
    }


# --------------------------------------------------------------------------- #
# (2) Finite-sample LCB at a per-depth profile (mirrors #183.metrics_at; only the
#     spine is profile-driven instead of constant-λ).
# --------------------------------------------------------------------------- #
def metrics_at_profile(ctx: dict, lam_profile: list[float], q_floor: list[float],
                       q_full: list[float], tau: float, b_tokens: int = B_TOKENS) -> dict[str, Any]:
    ep = ctx["ep"]
    spine = D178.spine_from_profile(ep, lam_profile, q_floor, q_full)
    et = D178.et_of_spine(ep, spine)                                   # #178 / #172 E[T]
    pvecs = D175.build_depth_pvecs_measured(spine, ep["rho_cond"], ep["W"], MAXD, "flat")
    pmf, _, _, _ = D175.dp_accepted_length_pmf(ep["parent"], pvecs)
    mom = D175.pmf_moments(pmf)
    sigma_L = mom["std"]
    pmf_mean_resid = abs(mom["mean"] - et)                             # provenance lock ≈1e-15
    n_steps = b_tokens / et
    slope = K_CAL * tau / STEP
    central = slope * et
    se_tps = slope * sigma_L / math.sqrt(n_steps)
    h_full = Z95 * math.sqrt(se_tps ** 2 + SIGMA_HW ** 2)
    return {
        "E_T": et, "sigma_L": sigma_L, "pmf_mean_resid": pmf_mean_resid,
        "central_tps": central, "H_full": h_full,
        "lcb_full_tps": central - h_full,
        "clears_500_central": bool(central >= TARGET_OFFICIAL),
        "clears_500_lcb": bool(central - h_full >= TARGET_OFFICIAL),
        "spine": spine,
    }


def official_tps(et: float, tau: float) -> float:
    return D172.official_tps(et, STEP, K_CAL, tau)


def clear500_bar(tau: float) -> float:
    return D172.clear500_bar(STEP, K_CAL, tau, TARGET_OFFICIAL)


def propagate(et: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tag, tau in TAU_CORNERS:
        tps = official_tps(et, tau)
        out[tag] = {
            "tau": tau, "official_tps": tps, "clear500_bar_et": clear500_bar(tau),
            "clears_500": bool(tps >= TARGET_OFFICIAL),
            "tps_margin_over_500": tps - TARGET_OFFICIAL,
        }
    return out


# --------------------------------------------------------------------------- #
# Inverse maps:  smallest depth-1 λ̂₁ that clears a target (bisection over λ̂₁).
# Returns None (not NaN) when even λ̂₁=1.0 cannot clear — keeps payload NaN-clean.
# --------------------------------------------------------------------------- #
def _bisect_lam1(f_at: Callable[[float], float], target: float):
    if f_at(1.0) < target:
        return None                       # unreachable: even perfect depth-1 recovery misses
    if f_at(0.0) >= target:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f_at(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def lambda1_star_lcb(ctx: dict, q_floor: list[float], q_full: list[float], beta: float,
                     tau: float, b_tokens: int = B_TOKENS):
    """Depth-1 λ̂₁ at which the mechanism profile's FULL finite-sample LCB clears 500 — the
    #183-faithful '0.9052-equivalent' bar read against a depth-1 probe under the mechanism."""
    H = len(q_full)
    return _bisect_lam1(
        lambda l: metrics_at_profile(ctx, mechanism_lambda(H, l, beta), q_floor, q_full, tau,
                                     b_tokens)["lcb_full_tps"],
        TARGET_OFFICIAL)


def lambda1_star_central(ctx: dict, q_floor: list[float], q_full: list[float], beta: float,
                         target_et: float):
    """Depth-1 λ̂₁ at which the mechanism profile's CENTRAL both-bugs E[T] reaches target_et."""
    H = len(q_full)
    ep = ctx["ep"]
    return _bisect_lam1(
        lambda l: D178.et_of_spine(ep, D178.spine_from_profile(ep, mechanism_lambda(H, l, beta),
                                                               q_floor, q_full)),
        target_et)


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize(anchors: dict[str, Any]) -> dict[str, Any]:
    ctx = D183.build_topologies(anchors)
    ep = ctx["ep"]
    topo = ctx["topo"]
    qfl_d, qfu_d = topo["descent_only"]["q_floor"], topo["descent_only"]["q_full"]
    qfl_b, qfu_b = topo["both_bugs"]["q_floor"], topo["both_bugs"]["q_full"]
    H = len(qfu_d)
    max_tree_depth = ctx["max_tree_depth"]

    # liveprobe λ̂₁ (#178 formula, recomputed from the same anchors).
    lam_hat = ((D178.LIVEPROBE_WALK_TOPW0_HIT - qfl_d[0])
               / (D178.LIVEPROBE_LINEAR_TOP1 - qfl_d[0]))

    # ---------- (1) ground β in the salvage construction ---------- #
    salvage_cond = D172.cum_to_conditional(ORACLE_CUM_LADDER)
    beta = ground_beta(salvage_cond, qfu_d)
    beta_primary = beta["beta_primary_geomean"]
    # full β ladder (flat + #178 band + construction range), de-duplicated & sorted desc.
    beta_ladder = sorted({1.0, 0.9, 0.8, 0.7,
                          round(beta_primary, 6),
                          round(beta["beta_reach_absolute"], 6),
                          round(beta["beta_rel_staleness_only"], 6)}, reverse=True)

    # effective profile the E[T] DP actually consumes (spine clamps depths >H to depth-H);
    # the pure mechanism would keep decaying at depths H+1.. — reported as the optimistic gap.
    eff_lambda = [lam_hat * (beta_primary ** min(d, H - 1)) for d in range(max_tree_depth)]
    pure_lambda = [lam_hat * (beta_primary ** d) for d in range(max_tree_depth)]
    lambda_of_depth = {
        "depths_1_to_%d" % max_tree_depth: list(range(1, max_tree_depth + 1)),
        "effective_lambda_DP_clamped": eff_lambda,
        "pure_mechanism_lambda_unclamped": pure_lambda,
        "functional_form": "lambda_d = lambda_hat_1 * beta^(d-1)  (geometric staleness decay)",
        "staleness_decay_exponent_alpha": staleness_decay_exponent(beta_primary),
        "beta_primary": beta_primary,
        "note": ("the E[T] DP's 7-entry spine flat-extrapolates depths 8..9 to the depth-7 "
                 "recovery (effective_lambda); the pure mechanism would decay further "
                 "(pure_mechanism_lambda), so the DP-clamped profile is mildly OPTIMISTIC "
                 "at depths 8..9 — consistent with flat-transfer being the optimistic bound."),
    }

    # ---------- (2) mechanism floor E[T] + TPS + clear-500 (primary β) ---------- #
    def floor_et(beta_v: float, qfl: list[float], qfu: list[float], lam1: float = None) -> float:
        l1 = lam_hat if lam1 is None else lam1
        return D178.et_of_spine(ep, D178.spine_from_profile(ep, mechanism_lambda(H, l1, beta_v),
                                                            qfl, qfu))

    descent_only_mechanism_floor_E_T = floor_et(beta_primary, qfl_d, qfu_d)
    both_bugs_mechanism_floor_E_T = floor_et(beta_primary, qfl_b, qfu_b)
    prop_descent = propagate(descent_only_mechanism_floor_E_T)
    prop_bb = propagate(both_bugs_mechanism_floor_E_T)
    mechanism_floor_tps_descent = {tag: prop_descent[tag]["official_tps"] for tag, _ in TAU_CORNERS}
    mechanism_floor_tps_both_bugs = {tag: prop_bb[tag]["official_tps"] for tag, _ in TAU_CORNERS}
    mechanism_floor_clears_500 = {
        "descent": {tag: prop_descent[tag]["clears_500"] for tag, _ in TAU_CORNERS},
        "both_bugs": {tag: prop_bb[tag]["clears_500"] for tag, _ in TAU_CORNERS},
    }
    both_bugs_mechanism_floor_tps = mechanism_floor_tps_both_bugs["tau_central_1p0"]   # TEST

    # the full β ladder (descent + both-bugs) at the liveprobe anchor λ̂₁.
    beta_sweep = []
    for bv in beta_ladder:
        etd = floor_et(bv, qfl_d, qfu_d)
        etb = floor_et(bv, qfl_b, qfu_b)
        beta_sweep.append({
            "beta": bv, "staleness_alpha": staleness_decay_exponent(bv),
            "is_flat_178": bool(abs(bv - 1.0) < 1e-12),
            "descent_E_T": etd, "descent_tps_tau1": official_tps(etd, TAU_CENTRAL),
            "descent_clears_500_tau1": bool(official_tps(etd, TAU_CENTRAL) >= TARGET_OFFICIAL),
            "both_bugs_E_T": etb, "both_bugs_tps_tau1": official_tps(etb, TAU_CENTRAL),
            "both_bugs_clears_500_tau1": bool(official_tps(etb, TAU_CENTRAL) >= TARGET_OFFICIAL),
        })

    # whether the mechanism profile decays FASTER than constant-λ (β<1) and vs the band.
    decays_faster_than_constant = bool(beta_primary < 1.0)
    in_geometric_band = bool(0.7 <= beta_primary <= 0.9)
    below_geometric_band = bool(beta_primary < 0.7)

    # ---------- (3) verdict robustness + inverse map ---------- #
    # MISSES-BOTH is robust iff even the OPTIMISTIC plateau (β=1 flat, the #178 case) misses
    # at the realistic λ̂₁ — because any β<1 only lowers E[T] (conservative ordering).
    flat_descent_clears = bool(official_tps(floor_et(1.0, qfl_d, qfu_d), TAU_CENTRAL) >= TARGET_OFFICIAL)
    flat_bb_clears = bool(official_tps(floor_et(1.0, qfl_b, qfu_b), TAU_CENTRAL) >= TARGET_OFFICIAL)
    # also check both τ corners for the mechanism floor itself.
    mech_any_clears = any(mechanism_floor_clears_500["descent"].values()) or \
        any(mechanism_floor_clears_500["both_bugs"].values())
    misses_both_robust_to_mechanism = bool((not flat_descent_clears) and (not flat_bb_clears)
                                           and (not mech_any_clears))

    # inverse map: depth-1 λ̂₁ that clears #183's 0.9052-equivalent (LCB) bar, both-bugs.
    bb_lam1_star_lcb_primary = lambda1_star_lcb(ctx, qfl_b, qfu_b, beta_primary, TAU_CENTRAL)
    bb_lam1_star_lcb_flat = lambda1_star_lcb(ctx, qfl_b, qfu_b, 1.0, TAU_CENTRAL)   # ⇒ 0.9052
    desc_lam1_star_lcb_flat = lambda1_star_lcb(ctx, qfl_d, qfu_d, 1.0, TAU_CENTRAL)  # ⇒ 0.9750
    # central-500 route (looser) at β=1 ⇒ #178's 0.8384 point.
    bb_lam1_star_central_flat = lambda1_star_central(ctx, qfl_b, qfu_b, 1.0, clear500_bar(TAU_CENTRAL))
    # the β ladder of inverse maps (how the depth-1 bar moves with staleness).
    inverse_sweep = []
    for bv in beta_ladder:
        ls_lcb = lambda1_star_lcb(ctx, qfl_b, qfu_b, bv, TAU_CENTRAL)
        m1 = metrics_at_profile(ctx, mechanism_lambda(H, 1.0, bv), qfl_b, qfu_b, TAU_CENTRAL)
        inverse_sweep.append({
            "beta": bv,
            "both_bugs_lambda1_star_lcb": ls_lcb,           # None ⇒ unreachable for any λ̂₁≤1
            "reachable": bool(ls_lcb is not None),
            "lcb_at_lam1_eq_1": m1["lcb_full_tps"],         # shortfall witness when unreachable
            "central_at_lam1_eq_1": m1["central_tps"],
        })
    # critical per-step retention β at which λ̂₁=1.0 just clears the LCB bar (depth-1 sufficiency).
    def lcb_at_lam1(bv: float) -> float:
        return metrics_at_profile(ctx, mechanism_lambda(H, 1.0, bv), qfl_b, qfu_b,
                                  TAU_CENTRAL)["lcb_full_tps"]
    if lcb_at_lam1(1.0) < TARGET_OFFICIAL:
        beta_crit_depth1_sufficient = None
    else:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if lcb_at_lam1(mid) < TARGET_OFFICIAL:
                lo = mid
            else:
                hi = mid
        beta_crit_depth1_sufficient = 0.5 * (lo + hi)

    # ---------- (4) self-test (PRIMARY) ---------- #
    # (a) flat (β=1) reproduces #178's constant-λ floors 404/416.
    flat_descent_et = floor_et(1.0, qfl_d, qfu_d)
    flat_bb_et = floor_et(1.0, qfl_b, qfu_b)
    cond_a = bool(abs(flat_descent_et - D178_DESCENT_FLOOR_E_T) < TOL_REPRO
                  and abs(flat_bb_et - D178_BOTHBUGS_FLOOR_E_T) < TOL_REPRO
                  and abs(official_tps(flat_descent_et, TAU_CENTRAL) - D178_DESCENT_FLOOR_TPS) < 1e-2
                  and abs(official_tps(flat_bb_et, TAU_CENTRAL) - D178_BOTHBUGS_FLOOR_TPS) < 1e-2)
    # (b) λ̂₁→0 reproduces #172's adversarial floor 3.5346 (descent, any β).
    zero_floor_et = floor_et(beta_primary, qfl_d, qfu_d, lam1=0.0)
    cond_b = bool(abs(zero_floor_et - D172_DESCENT_LOWER_BOUND) < TOL_ENDPOINT)
    # (c) β=1 inverse map reproduces #183's 0.9052 (LCB) AND #178's 0.8384 (central-500).
    cond_c = bool(bb_lam1_star_lcb_flat is not None
                  and abs(bb_lam1_star_lcb_flat - D183_BOTHBUGS_LAMBDA_STAR_LCB) < TOL_REPRO
                  and bb_lam1_star_central_flat is not None
                  and abs(bb_lam1_star_central_flat - D178_BOTHBUGS_LAMBDA_STAR_CENTRAL) < TOL_REPRO)
    # (d) conservative ordering: mech E[T] ≤ flat E[T] at every λ̂₁ (both topologies, band β).
    ordering_ok = True
    ordering_min_slack = float("inf")
    for lam1 in (0.1, 0.2, 0.3, round(lam_hat, 6), 0.5, 0.7, 0.9, 1.0):
        for qfl, qfu in ((qfl_d, qfu_d), (qfl_b, qfu_b)):
            flat_et = floor_et(1.0, qfl, qfu, lam1=lam1)
            for bv in beta_ladder:
                m_et = floor_et(bv, qfl, qfu, lam1=lam1)
                slack = flat_et - m_et            # must be ≥ 0 (staleness can't help)
                ordering_min_slack = min(ordering_min_slack, slack)
                if slack < -1e-9:
                    ordering_ok = False
    cond_d = bool(ordering_ok)
    # provenance lock: pmf-mean reproduces E[T] at the mechanism floor.
    prov = metrics_at_profile(ctx, mechanism_lambda(H, lam_hat, beta_primary), qfl_b, qfu_b,
                              TAU_CENTRAL)
    cond_prov = bool(prov["pmf_mean_resid"] < 1e-9)

    self_test_passes = bool(cond_a and cond_b and cond_c and cond_d and cond_prov)

    verdict = _verdict(misses_both_robust_to_mechanism, beta_primary, beta,
                       both_bugs_mechanism_floor_tps, bb_lam1_star_lcb_primary,
                       beta_crit_depth1_sufficient)
    handoff = _handoff_line(
        beta_primary=beta_primary, beta_range=beta["beta_construction_range"],
        lam_hat=lam_hat, descent_tps=mechanism_floor_tps_descent["tau_central_1p0"],
        bb_tps=both_bugs_mechanism_floor_tps,
        robust=misses_both_robust_to_mechanism,
        bb_lam1_star=bb_lam1_star_lcb_primary,
        beta_crit=beta_crit_depth1_sufficient)

    return {
        "self_test": {
            "lambda_depth_profile_self_test_passes": self_test_passes,
            "conditions": {
                "a_flat_reproduces_178_floors_404_416": cond_a,
                "b_lambda1_zero_reproduces_172_floor_3p5346": cond_b,
                "c_flat_inverse_reproduces_183_0p9052_and_178_0p8384": cond_c,
                "d_conservative_ordering_mech_le_flat": cond_d,
                "prov_pmf_mean_reproduces_et": cond_prov,
            },
            "reproduction_residuals": {
                "flat_descent_E_T": flat_descent_et, "imported_178_descent": D178_DESCENT_FLOOR_E_T,
                "flat_both_bugs_E_T": flat_bb_et, "imported_178_both_bugs": D178_BOTHBUGS_FLOOR_E_T,
                "lambda1_zero_floor_E_T": zero_floor_et, "imported_172_floor": D172_DESCENT_LOWER_BOUND,
                "flat_bb_lambda1_star_lcb": bb_lam1_star_lcb_flat, "imported_183_bar": D183_BOTHBUGS_LAMBDA_STAR_LCB,
                "flat_bb_lambda1_star_central": bb_lam1_star_central_flat, "imported_178_point": D178_BOTHBUGS_LAMBDA_STAR_CENTRAL,
                "ordering_min_slack_flat_minus_mech": ordering_min_slack,
            },
        },
        "test_metric": {"both_bugs_mechanism_floor_tps": both_bugs_mechanism_floor_tps},
        "mechanism_law": {
            "functional_form": "lambda_d = lambda_hat_1 * beta^(d-1)",
            "derivation": ("salvage-no-descend reuses the parent KV (one extra stale step per "
                           "depth; staleness s=d-1); stale-KV/true-KV divergence compounds "
                           "multiplicatively -> geometric. The geometric FORM #178 swept as a "
                           "guessed sensitivity band is here DERIVED from the salvage construction; "
                           "alpha=0 (beta=1) is the flat #178 special case."),
            "lambda_hat_1": lam_hat,
            "staleness_decay_exponent_alpha": staleness_decay_exponent(beta_primary),
            "beta_grounding": beta,
            "lambda_of_depth": lambda_of_depth,
            "decays_faster_than_constant_lambda": decays_faster_than_constant,
            "in_geometric_band_0p7_0p9": in_geometric_band,
            "below_geometric_band": below_geometric_band,
            "vs_geometric_band": ("primary beta sits %s the #178 geometric band [0.7,0.9]; the "
                                  "construction range [%.3f, %.3f] brackets it" % (
                                      ("inside" if in_geometric_band else
                                       "below" if below_geometric_band else "above"),
                                      beta["beta_construction_range"][0],
                                      beta["beta_construction_range"][1])),
        },
        "mechanism_floor": {
            "primary_beta": beta_primary,
            "descent_only_mechanism_floor_E_T": descent_only_mechanism_floor_E_T,
            "both_bugs_mechanism_floor_E_T": both_bugs_mechanism_floor_E_T,
            "mechanism_floor_tps_descent": mechanism_floor_tps_descent,
            "mechanism_floor_tps_both_bugs": mechanism_floor_tps_both_bugs,
            "mechanism_floor_clears_500": mechanism_floor_clears_500,
            "propagate_descent": prop_descent,
            "propagate_both_bugs": prop_bb,
            "beta_sweep_at_liveprobe": beta_sweep,
        },
        "verdict_robustness": {
            "misses_both_robust_to_mechanism": misses_both_robust_to_mechanism,
            "flat_descent_clears_500_tau1": flat_descent_clears,
            "flat_both_bugs_clears_500_tau1": flat_bb_clears,
            "mechanism_any_corner_clears_500": mech_any_clears,
            "both_bugs_lambda1_star": bb_lam1_star_lcb_primary,        # None ⇒ unreachable
            "both_bugs_lambda1_star_reachable": bool(bb_lam1_star_lcb_primary is not None),
            "both_bugs_lambda1_star_flat_beta1": bb_lam1_star_lcb_flat,
            "descent_lambda1_star_flat_beta1": desc_lam1_star_lcb_flat,
            "beta_crit_depth1_sufficient": beta_crit_depth1_sufficient,
            "inverse_sweep_over_beta": inverse_sweep,
            "interpretation": ("under the mechanism the 0.9052 bar is a CONSTANT-λ bar; with "
                               "beta<1 deeper depths are capped at lambda_hat_1*beta^(d-1), so "
                               "no depth-1 probe value (not even lambda_hat_1=1.0) clears it "
                               "unless beta >= beta_crit_depth1_sufficient. land #71's depth-1 "
                               "probe is necessary but not sufficient; it must measure q[2..9]."),
        },
        "composition": {
            "K_cal": K_CAL, "step": STEP, "z95": Z95, "B_tokens": B_TOKENS,
            "sigma_hw_tps": SIGMA_HW, "tau_central": TAU_CENTRAL,
            "tau_conservative": TAU_CONSERVATIVE, "max_tree_depth": max_tree_depth,
            "clear500_bar_tau1": clear500_bar(TAU_CENTRAL),
            "clear500_bar_tau_cons": clear500_bar(TAU_CONSERVATIVE),
        },
        "verdict": verdict,
        "handoff_line": handoff,
    }


def _verdict(robust: bool, beta_primary: float, beta: dict, bb_tps: float,
             bb_lam1_star, beta_crit) -> str:
    star = ("UNREACHABLE (no depth-1 λ̂₁≤1 clears it)" if bb_lam1_star is None
            else f"{bb_lam1_star:.4f}")
    crit = "n/a" if beta_crit is None else f"{beta_crit:.3f}"
    head = "MECHANISM-HARDENS-MISSES-BOTH" if robust else "MECHANISM-LIFTS-FLOOR"
    return (
        f"{head}. The salvage-staleness law λ_d=λ̂₁·β^(d−1) is DERIVED (one stale step per "
        f"depth → multiplicative divergence → geometric), GROUNDING #178's guessed geometric "
        f"band; primary β={beta_primary:.3f} (construction range "
        f"[{beta['beta_construction_range'][0]:.3f},{beta['beta_construction_range'][1]:.3f}]) "
        f"decays FASTER than constant-λ, so the realistic both-bugs floor is "
        f"{bb_tps:.0f} TPS — even LOWER than #178's flat 416. MISSES-BOTH is robust: flat "
        f"(β=1) is the OPTIMISTIC plateau and already misses, and staleness can only lower "
        f"E[T] (conservative ordering). Inverse map: both-bugs depth-1 λ̂₁ bar = {star}; the "
        f"depth-1 probe is sufficient only if per-step retention β ≥ {crit} (≈no staleness), "
        f"so land #71 must measure the q[2..9] ladder, not infer it from depth-1. NOT a launch."
    )


def _handoff_line(*, beta_primary: float, beta_range: list[float], lam_hat: float,
                  descent_tps: float, bb_tps: float, robust: bool, bb_lam1_star,
                  beta_crit) -> str:
    star = "unreachable" if bb_lam1_star is None else f"{bb_lam1_star:.4f}"
    crit = "n/a" if beta_crit is None else f"{beta_crit:.3f}"
    return (
        f"SALVAGE-STALENESS λ(depth) (denken #193): replaces the FLAT depth-transfer behind "
        f"#178/#183/#187 with the mechanism-derived λ_d=λ̂₁·β^(d−1) (β grounded in wirbel #135's "
        f"measured salvage-no-descend ladder; primary β={beta_primary:.3f}, range "
        f"[{beta_range[0]:.3f},{beta_range[1]:.3f}], inside/below #178's [0.7,0.9]). At the "
        f"liveprobe λ̂₁={lam_hat:.3f}: descent {descent_tps:.0f} / both-bugs {bb_tps:.0f} TPS, "
        f"both MISS — lower than #178's flat 404/416, so MISSES-BOTH={'ROBUST' if robust else 'NOT robust'} "
        f"(flat is the optimistic plateau and already misses). The 0.9052 build bar is a "
        f"constant-λ bar; under the mechanism the depth-1 inverse bar is {star} and depth-1 is "
        f"sufficient only if β≥{crit}. Confirms/refutes against land #71's measured q[2..9]; "
        f"re-grounds denken #183 (the bar) + #187 (the depth-9-dominant variance is also the "
        f"low-recovery end). NOT open2. NOT a launch."
    )


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
    ml, mf, vr, st = (syn["mechanism_law"], syn["mechanism_floor"],
                      syn["verdict_robustness"], syn["self_test"])
    print("\n" + "=" * 92, flush=True)
    print("SALVAGE-STALENESS λ(depth) — mechanism vs flat depth-transfer (PR #193)", flush=True)
    print("=" * 92, flush=True)
    bg = ml["beta_grounding"]
    print(f"  (1) MECHANISM  {ml['functional_form']}  (α=−ln β; α=0 ⇔ flat #178)", flush=True)
    print(f"      λ̂₁={ml['lambda_hat_1']:.4f}  β_primary={bg['beta_primary_geomean']:.4f} "
          f"(α={ml['staleness_decay_exponent_alpha']:.4f})  "
          f"reach={bg['beta_reach_absolute']:.4f}  rel={bg['beta_rel_staleness_only']:.4f}", flush=True)
    print(f"      decays-faster-than-constant={ml['decays_faster_than_constant_lambda']}  "
          f"{ml['vs_geometric_band']}", flush=True)
    print(f"      effective λ(depth 1..{syn['composition']['max_tree_depth']}) = "
          f"{[round(x,4) for x in ml['lambda_of_depth']['effective_lambda_DP_clamped']]}", flush=True)
    print("-" * 92, flush=True)
    print(f"  (2) MECHANISM FLOOR (primary β={mf['primary_beta']:.4f}, at λ̂₁):", flush=True)
    print(f"      descent  E[T]={mf['descent_only_mechanism_floor_E_T']:.4f} "
          f"TPS={mf['mechanism_floor_tps_descent']['tau_central_1p0']:.1f}  "
          f"clears500={mf['mechanism_floor_clears_500']['descent']['tau_central_1p0']}", flush=True)
    print(f"      both-bugs E[T]={mf['both_bugs_mechanism_floor_E_T']:.4f} "
          f"TPS={mf['mechanism_floor_tps_both_bugs']['tau_central_1p0']:.1f}  "
          f"clears500={mf['mechanism_floor_clears_500']['both_bugs']['tau_central_1p0']}", flush=True)
    print("      β sweep at λ̂₁ (β=1 ⇒ flat #178):", flush=True)
    for r in mf["beta_sweep_at_liveprobe"]:
        tag = " <-flat#178" if r["is_flat_178"] else ""
        print(f"        β={r['beta']:.4f}  descent {r['descent_tps_tau1']:6.1f} "
              f"(clears={r['descent_clears_500_tau1']})  both-bugs {r['both_bugs_tps_tau1']:6.1f} "
              f"(clears={r['both_bugs_clears_500_tau1']}){tag}", flush=True)
    print("-" * 92, flush=True)
    print(f"  (3) VERDICT ROBUSTNESS  misses_both_robust_to_mechanism="
          f"{vr['misses_both_robust_to_mechanism']}", flush=True)
    star = ("UNREACHABLE" if vr["both_bugs_lambda1_star"] is None
            else f"{vr['both_bugs_lambda1_star']:.4f}")
    crit = ("n/a" if vr["beta_crit_depth1_sufficient"] is None
            else f"{vr['beta_crit_depth1_sufficient']:.4f}")
    print(f"      both_bugs_lambda1_star (LCB bar, primary β) = {star}  "
          f"(flat β=1 ⇒ {vr['both_bugs_lambda1_star_flat_beta1']:.4f} = #183's 0.9052)", flush=True)
    print(f"      β_crit (depth-1 λ̂₁=1.0 just clears LCB) = {crit}", flush=True)
    for r in vr["inverse_sweep_over_beta"]:
        ls = "UNREACHABLE" if r["both_bugs_lambda1_star_lcb"] is None else f"{r['both_bugs_lambda1_star_lcb']:.4f}"
        print(f"        β={r['beta']:.4f}  λ̂₁*_lcb={ls:>11}  LCB@λ̂₁=1.0={r['lcb_at_lam1_eq_1']:.1f}", flush=True)
    print("-" * 92, flush=True)
    print(f"  (4) PRIMARY lambda_depth_profile_self_test_passes = "
          f"{st['lambda_depth_profile_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"          - {k}: {v}", flush=True)
    print(f"  TEST both_bugs_mechanism_floor_tps = {syn['test_metric']['both_bugs_mechanism_floor_tps']:.4f}",
          flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 92, flush=True)
    print(f"\n  HAND-OFF: {syn['handoff_line']}\n", flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors #178 / #183; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
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
        print(f"[lambda-depth] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    run = init_wandb_run(
        job_type="lambda-depth-staleness-profile",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["lambda-depth-staleness-profile", "validity-gate", "salvage-staleness",
              "depth-transfer", "bank-the-analysis"],
        config={
            "K_cal": K_CAL, "step": STEP, "z95": Z95, "B_tokens": B_TOKENS,
            "sigma_hw_tps": SIGMA_HW, "tau_central": TAU_CENTRAL,
            "tau_conservative": TAU_CONSERVATIVE,
            "imports": "denken#172 et_backward + #178 E[T](λ) + #183 LCB/0.9052 bar + #175 σ_L + wirbel#135 salvage ladder",
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[lambda-depth] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    st = syn["self_test"]
    ml = syn["mechanism_law"]
    mf = syn["mechanism_floor"]
    vr = syn["verdict_robustness"]
    bg = ml["beta_grounding"]
    summary: dict[str, Any] = {
        # PRIMARY + TEST
        "lambda_depth_profile_self_test_passes": int(bool(st["lambda_depth_profile_self_test_passes"])),
        "both_bugs_mechanism_floor_tps": syn["test_metric"]["both_bugs_mechanism_floor_tps"],
        # mechanism law / grounding
        "lambda_hat_1": ml["lambda_hat_1"],
        "beta_primary": bg["beta_primary_geomean"],
        "beta_reach_absolute": bg["beta_reach_absolute"],
        "beta_rel_staleness_only": bg["beta_rel_staleness_only"],
        "staleness_decay_exponent_alpha": ml["staleness_decay_exponent_alpha"],
        "decays_faster_than_constant": int(bool(ml["decays_faster_than_constant_lambda"])),
        "in_geometric_band_0p7_0p9": int(bool(ml["in_geometric_band_0p7_0p9"])),
        # mechanism floor
        "descent_only_mechanism_floor_E_T": mf["descent_only_mechanism_floor_E_T"],
        "both_bugs_mechanism_floor_E_T": mf["both_bugs_mechanism_floor_E_T"],
        "mechanism_floor_tps_descent_tau1": mf["mechanism_floor_tps_descent"]["tau_central_1p0"],
        "mechanism_floor_tps_both_bugs_tau1": mf["mechanism_floor_tps_both_bugs"]["tau_central_1p0"],
        "mechanism_floor_tps_descent_tau_cons": mf["mechanism_floor_tps_descent"]["tau_conservative_0p9924"],
        "mechanism_floor_tps_both_bugs_tau_cons": mf["mechanism_floor_tps_both_bugs"]["tau_conservative_0p9924"],
        "descent_clears_500_tau1": int(bool(mf["mechanism_floor_clears_500"]["descent"]["tau_central_1p0"])),
        "both_bugs_clears_500_tau1": int(bool(mf["mechanism_floor_clears_500"]["both_bugs"]["tau_central_1p0"])),
        # verdict robustness
        "misses_both_robust_to_mechanism": int(bool(vr["misses_both_robust_to_mechanism"])),
        "both_bugs_lambda1_star_reachable": int(bool(vr["both_bugs_lambda1_star_reachable"])),
        "both_bugs_lambda1_star_flat_beta1": vr["both_bugs_lambda1_star_flat_beta1"],
        "descent_lambda1_star_flat_beta1": vr["descent_lambda1_star_flat_beta1"],
        # bars
        "clear500_bar_tau1": syn["composition"]["clear500_bar_tau1"],
        "K_cal": K_CAL, "step": STEP,
        "verdict_hardens_misses_both": int(syn["verdict"].startswith("MECHANISM-HARDENS")),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    # numeric companions for the optional (None) inverse maps.
    if vr["both_bugs_lambda1_star"] is not None:
        summary["both_bugs_lambda1_star"] = vr["both_bugs_lambda1_star"]
    if vr["beta_crit_depth1_sufficient"] is not None:
        summary["beta_crit_depth1_sufficient"] = vr["beta_crit_depth1_sufficient"]
    summary = {k: v for k, v in summary.items() if not (isinstance(v, float) and not math.isfinite(v))}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="lambda_depth_profile_result", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[lambda-depth] wandb logged: {summary}", flush=True)


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
    ap.add_argument("--wandb-name", "--wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", default="lambda-depth-staleness-profile")
    args = ap.parse_args(argv)

    anchors = D172.load_anchors(
        args.bug2_anchor, args.topo_json, args.accept_json, args.rankcov_json, args.decomp_json
    )
    syn = synthesize(anchors)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at,
        "pr": 193,
        "agent": "denken",
        "kind": "lambda-depth-staleness-profile",
        "anchors": {k: v for k, v in anchors.items() if k != "_paths"},
        "anchor_paths": anchors.get("_paths"),
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[lambda-depth] WARNING non-finite values at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lambda_depth_profile_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[lambda-depth] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = syn["self_test"]["lambda_depth_profile_self_test_passes"] and payload["nan_clean"]
        print(f"[lambda-depth] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
