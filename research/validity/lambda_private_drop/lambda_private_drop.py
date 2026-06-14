#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""λ-dependent private drop (PR #198) — couple #176's adverse-skew private drop to
#193's salvage-staleness depth mechanism, then re-solve #191's private build bar.

THE UN-MODELLED COUPLING
------------------------
stark #191 (`jeclr39w`) set the binding launch bar at the PRIVATE-stricter
λ*_LCB,private = 0.9780 (both-bugs) by composing stark #176's adverse-vertex drop as
a CONSTANT 2.350% (both) / 2.300% (descent) applied uniformly across all λ. denken
#193 (`2clxvlr8`) proved the realized recovery is NOT flat in depth: λ_d = λ̂₁·β^(d−1)
(β_primary 0.7651). The adverse vertex is PURE non-Latin-script (W_hard 0.2904) — the
domain the int4 drafter predicts WORST at the shallow rungs. So the private drop and
the recovery λ are almost certainly NOT independent: harder-to-draft adverse tokens
concentrate their acceptance deficit at SHALLOW depth, and #193's depth mechanism
controls how much accepted mass sits shallow. This PR closes that coupling.

MODEL (one new equation; #191's forward map kept verbatim)
----------------------------------------------------------
    δ_d        = 1 − q_adv[d]/q_pub[d]                  ← #176 per-rung adverse deficit
    a_pub(d;λ) = spine_from_profile(geometric_lambda(λ,β))   ← #193 depth profile, #178 spine
    a_adv(d;λ) = a_pub(d;λ)·(1−δ_d)                      ← adverse domain on the card's tree
    drop_mech(λ) = 1 − E_T(a_adv)/E_T(a_pub)            ← #178/#172 tree E[T] (et_of_spine)
    drop(λ)    = drop_176 · drop_mech(λ)/drop_mech(1)   ← anchored: drop(1)=drop_176

    private_LCB(λ) = public_LCB(λ)·(1 − drop(λ))·τ_low  ← #191 forward map, drop now λ-coupled

  * quality component    = drop_176 (the depth-flat-δ baseline = full-recovery drop; λ-INDEPENDENT).
  * acceptance component = drop_176·(drop_mech(λ)/drop_mech(1) − 1) (depth-VARIATION of δ_d;
    0 at λ=1). Its SIGN is a model OUTPUT, not an assumption. FINDING: it is NEGATIVE — the
    shallow per-rung deficits (δ₀,δ₁,δ₂ > 0) COMPOUND multiplicatively along the accepted
    chain, so a deeper tree (higher λ) accumulates MORE total drop; at the shallow realistic
    floor (λ̂₁=0.342) LESS drop accumulates. This REFUTES the hypothesised depth-amplification
    (which predicted drop RISES at low λ). Net effect near the bar is NEGLIGIBLE, so #191's
    fixed-drop 0.9780 is VALIDATED (and is a conservative upper bound).

LOCAL CPU-only analytic. No GPU / vLLM / HF Job / submission / served-file change.
BASELINE stays 481.53. Greedy/PPL untouched. Bank-the-analysis (PRIMARY = self-test,
adds 0 TPS). NOT open2. NOT a launch.

PRIMARY metric  lambda_private_drop_self_test_passes
TEST    metric  both_bugs_lambda_star_lcb_private_coupled

Run:
    python -m research.validity.lambda_private_drop.lambda_private_drop \
        --self-test --wandb-name stark/lambda-private-drop \
        --wandb-group lambda-dependent-private-drop
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

_LAC_PATH = REPO_ROOT / "research/oracle_readout/lambda_acceptance_card/lambda_acceptance_card.py"
_D176_RESULTS = REPO_ROOT / "research/validity/private_adverse_skew/results.json"

TARGET_OFFICIAL = 500.0
DISQUALIFY_GATE_PCT = 5.0
PUBLIC_BAR_BOTH = 0.9052283680740145        # #183/#184 both-bugs public LCB build bar (τ=1)
RESID_TOL_TPS = 0.5                          # self-test tolerance (matches #176/#183/#191)
LAMBDA_STAR_191 = 0.9780112973731208        # #191 both-bugs private bar (fixed-drop)

# --- denken #193 salvage-staleness mechanism (imported; NOT re-derived) --- #
LAMBDA_FLOOR = 0.3418647166361965           # #193 liveprobe λ̂₁ (realistic floor)
BETA_PRIMARY = 0.765124365433998            # #193 primary β (geomean of construction range)
BETA_RANGE = (0.616486595380561, 0.9495993894553337)   # #193 construction range
BETA_CRIT = 0.9648839148878561              # #193 depth-1-sufficiency β


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- import #183 card + #178 spine machinery (public leg; NOT re-derived) --- #
LAC = _import("lambda_acceptance_card", _LAC_PATH)
D178 = LAC.D178
D172 = LAC.D172


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _at(vec: list[float], i: int) -> float:
    return vec[i] if i < len(vec) else vec[-1]


# --------------------------------------------------------------------------- #
# Imports.
# --------------------------------------------------------------------------- #
def load_176() -> dict[str, Any]:
    """Import stark #176's adverse-vertex per-rung acceptance + worst-corner tree drop,
    byte-from the banked results.json. NOT re-derived."""
    with _D176_RESULTS.open(encoding="utf-8") as fh:
        r = json.load(fh)
    av = r["adverse_vertex"]
    const = r["constants"]
    q_pub = list(r["q_pub_sglang"])           # public per-rung conditional acceptance
    q_adv = list(av["q_native"])              # adverse-vertex per-rung (mixture: pub + non-Latin hard)
    # per-rung adverse deficit (frame-independent ratio; transfers onto the card's tree):
    r_d = [a / p for a, p in zip(q_adv, q_pub)]
    delta_d = [1.0 - x for x in r_d]
    return {
        "tau_low": const["tau_low"],
        "tau_central": const["tau_central"],
        "target_500": const["target_500"],
        "K_cal": const["K_cal"],
        "step": const["step"],
        "gt_drop_pct": const["gt_drop_pct"],
        "decode_drop_pct": av["achieved_decode_drop_pct"],     # 4.2946 (λ-independent DQ anchor)
        "W_hard": av["W_hard"],
        "adverse_axis": av["kind"],
        "q_pub": q_pub,
        "q_adv": q_adv,
        "r_d": r_d,
        "delta_d": delta_d,
        "drop_descent_176": av["descent_tree_drop_pct"] / 100.0,   # 0.022999781
        "drop_both_176": av["both_tree_drop_pct"] / 100.0,         # 0.023502817
        "ref_descent_taulow_adverse": av["descent_tps_taulow"],
        "ref_both_taulow_adverse": av["both_tps_taulow"],
    }


def build_public_ctx() -> dict[str, Any]:
    """Execute #183's build_topologies on its default anchors → ctx + per-topology spines."""
    anchors = D172.load_anchors(
        D172.DEFAULT_BUG2_ANCHOR, D172.DEFAULT_TOPO_JSON, D172.DEFAULT_ACCEPT_JSON,
        D172.DEFAULT_RANKCOV_JSON, D172.DEFAULT_DECOMP_JSON,
    )
    ctx = LAC.build_topologies(anchors)
    topo = ctx["topo"]
    return {
        "ctx": ctx,
        "descent_only": (topo["descent_only"]["q_floor"], topo["descent_only"]["q_full"]),
        "both_bugs": (topo["both_bugs"]["q_floor"], topo["both_bugs"]["q_full"]),
    }


# --------------------------------------------------------------------------- #
# Public legs (imported #183 metrics, τ=1 published map).
# --------------------------------------------------------------------------- #
def public_lcb(ctx, lam, qfl, qfu, tau=1.0) -> float:
    return LAC.metrics_at(ctx, lam, qfl, qfu, tau)["lcb_full_tps"]


def public_central(ctx, lam, qfl, qfu, tau=1.0) -> float:
    return LAC.metrics_at(ctx, lam, qfl, qfu, tau)["central_tps"]


# --------------------------------------------------------------------------- #
# The mechanism: drop_mech(λ) via #193 geometric profile + #178/#172 tree E[T].
# --------------------------------------------------------------------------- #
def _spines(ep, lam, beta, qfl, qfu, r_d) -> tuple[list[float], list[float]]:
    """Public spine under #193's geometric recovery profile, and the adverse spine
    (public scaled per-rung by the #176 domain ratio r_d, clipped to a probability)."""
    H = ep["horizon"]
    prof = D178.geometric_lambda(H, lam, beta)
    pub = D178.spine_from_profile(ep, prof, qfl, qfu)
    adv = [min(1.0, max(1e-9, pub[d] * _at(r_d, d))) for d in range(H)]
    return pub, adv


def drop_mech(ep, lam, beta, qfl, qfu, r_d) -> float:
    """Mechanism tree-drop at recovery λ: 1 − E_T(adverse)/E_T(public) on the card's tree."""
    pub, adv = _spines(ep, lam, beta, qfl, qfu, r_d)
    et_pub = D178.et_of_spine(ep, pub)
    et_adv = D178.et_of_spine(ep, adv)
    if et_pub <= 0:
        return float("nan")
    return 1.0 - et_adv / et_pub


def drop_coupled(ep, lam, beta, qfl, qfu, r_d, drop_176) -> float:
    """λ-coupled private drop, anchored so drop(1)=drop_176 (reproduces #176 at full recovery).
    Shape from the mechanism: drop_176 · drop_mech(λ)/drop_mech(1)."""
    m1 = drop_mech(ep, 1.0, beta, qfl, qfu, r_d)
    if not _finite(m1) or abs(m1) < 1e-12:
        return drop_176
    return drop_176 * (drop_mech(ep, lam, beta, qfl, qfu, r_d) / m1)


def private_lcb_coupled(ctx, lam, qfl, qfu, r_d, drop_176, tau_low, beta) -> float:
    """#191 forward map with the λ-coupled drop: public_LCB(λ)·(1−drop(λ))·τ_low."""
    return (public_lcb(ctx, lam, qfl, qfu, 1.0)
            * (1.0 - drop_coupled(ctx["ep"], lam, beta, qfl, qfu, r_d, drop_176))
            * tau_low)


def private_lcb_fixed(ctx, lam, qfl, qfu, drop_176, tau_low) -> float:
    """#191's ORIGINAL fixed-drop forward map (the λ-flat limit / import check)."""
    return public_lcb(ctx, lam, qfl, qfu, 1.0) * (1.0 - drop_176) * tau_low


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize(beta: float = BETA_PRIMARY) -> dict[str, Any]:
    imp = load_176()
    pub = build_public_ctx()
    ctx = pub["ctx"]
    ep = ctx["ep"]
    tau_low = imp["tau_low"]
    r_d = imp["r_d"]

    topos = {
        "descent_only": {"qfl": pub["descent_only"][0], "qfu": pub["descent_only"][1],
                         "drop176": imp["drop_descent_176"],
                         "ref_taulow": imp["ref_descent_taulow_adverse"]},
        "both_bugs": {"qfl": pub["both_bugs"][0], "qfu": pub["both_bugs"][1],
                      "drop176": imp["drop_both_176"],
                      "ref_taulow": imp["ref_both_taulow_adverse"]},
    }

    per_topo: dict[str, Any] = {}
    for lab, t in topos.items():
        qfl, qfu, drop176 = t["qfl"], t["qfu"], t["drop176"]

        # --- coupled drop at the three diagnostic λ --- #
        drop_at_floor = drop_coupled(ep, LAMBDA_FLOOR, beta, qfl, qfu, r_d, drop176)
        drop_at_bar191 = drop_coupled(ep, LAMBDA_STAR_191, beta, qfl, qfu, r_d, drop176)
        drop_at_lam1 = drop_coupled(ep, 1.0, beta, qfl, qfu, r_d, drop176)   # == drop176 (anchor)

        # --- coupled private build bar: smallest λ whose coupled private LCB clears 500 --- #
        lam_star_coupled = LAC._bisect_lambda(
            lambda l: private_lcb_coupled(ctx, l, qfl, qfu, r_d, drop176, tau_low, beta),
            TARGET_OFFICIAL)
        # --- #191 fixed-drop bar (λ-flat limit; import check) --- #
        lam_star_fixed = LAC._bisect_lambda(
            lambda l: private_lcb_fixed(ctx, l, qfl, qfu, drop176, tau_low),
            TARGET_OFFICIAL)

        coupled_reachable = _finite(lam_star_coupled)
        fixed_reachable = _finite(lam_star_fixed)

        priv_lcb_lam1_coupled = private_lcb_coupled(ctx, 1.0, qfl, qfu, r_d, drop176, tau_low, beta)
        priv_lcb_at_bar191 = private_lcb_coupled(ctx, LAMBDA_STAR_191, qfl, qfu, r_d, drop176, tau_low, beta)

        per_topo[lab] = {
            "drop_176_pct": drop176 * 100.0,
            "drop_at_lambda1_pct": drop_at_lam1 * 100.0,                 # reproduces #176
            "drop_at_lambda_floor_pct": drop_at_floor * 100.0,          # λ̂₁=0.342
            "drop_at_lambda_bar_pct": drop_at_bar191 * 100.0,           # λ at #191's 0.9780
            "drop_floor_minus_176_pp": (drop_at_floor - drop176) * 100.0,
            "drop_bar_minus_176_pp": (drop_at_bar191 - drop176) * 100.0,
            "lambda_star_lcb_private_coupled": lam_star_coupled if coupled_reachable else None,
            "lambda_star_lcb_private_fixed_191": lam_star_fixed if fixed_reachable else None,
            "coupled_bar_shift_from_fixed": (
                (lam_star_coupled - lam_star_fixed) if (coupled_reachable and fixed_reachable) else None),
            "coupled_bar_shift_from_public": (
                (lam_star_coupled - PUBLIC_BAR_BOTH) if coupled_reachable else None),
            "private_lcb_reachable_at_full_recovery": coupled_reachable,
            "private_lcb_lambda1_coupled": priv_lcb_lam1_coupled,
            "private_lcb_at_bar191": priv_lcb_at_bar191,
            "private_lcb_margin_at_bar191": priv_lcb_at_bar191 - TARGET_OFFICIAL,
        }

    bb = per_topo["both_bugs"]
    dd = per_topo["descent_only"]

    both_bugs_lambda_star_lcb_private_coupled = bb["lambda_star_lcb_private_coupled"]   # TEST
    descent_reachable = dd["private_lcb_reachable_at_full_recovery"]
    both_bugs_required_at_private_bar = not descent_reachable

    # the coupling moves the bar stricter iff coupled both-bugs bar > #191's fixed 0.9780.
    coupled_bar = both_bugs_lambda_star_lcb_private_coupled
    private_nogo_more_robust_under_coupling = bool(
        coupled_bar is not None and coupled_bar > LAMBDA_STAR_191 + 1e-9)
    bar_shift_pp = (coupled_bar - LAMBDA_STAR_191) if coupled_bar is not None else None

    # drop λ-dependence: does the coupled drop change with λ at all?
    drop_is_lambda_dependent = bool(
        abs(bb["drop_at_lambda_floor_pct"] - bb["drop_at_lambda1_pct"]) > 1e-6)

    # gap of the coupled bar to the realistic floor (λ̂₁=0.342), on the bar (constant-λ) axis:
    coupled_bar_gap_to_floor = (coupled_bar - LAMBDA_FLOOR) if coupled_bar is not None else None

    # coupling sign + interpretation (both-bugs is the binding, reachable path).
    coupling_positive = bool(bb["drop_at_lambda_floor_pct"] >= bb["drop_at_lambda1_pct"] - 1e-9)
    coupling_sign = "positive" if coupling_positive else "negative"
    # #191 used the FULL-recovery drop (drop_176, the λ=1 value). If drop(λ)≤drop_176 for
    # all λ<1 (negative coupling), #191's fixed-drop is a CONSERVATIVE upper bound.
    fixed_drop_191_is_conservative = bool(
        bb["drop_at_lambda_floor_pct"] <= bb["drop_176_pct"] + 1e-9
        and bb["drop_at_lambda_bar_pct"] <= bb["drop_176_pct"] + 1e-9)
    # near the bar (λ≈0.978) the coupled drop ≈ drop_176, so the bar barely moves.
    coupling_negligible_near_bar = bool(
        abs(bb["drop_at_lambda_bar_pct"] - bb["drop_176_pct"]) < 0.05)
    drop_range_pp = bb["drop_at_lambda1_pct"] - bb["drop_at_lambda_floor_pct"]
    # the NO-GO verdict is unchanged iff the bar shift is negligible vs the floor gap.
    nogo_verdict_unchanged = bool(
        coupled_bar is not None and abs(coupled_bar - LAMBDA_STAR_191) < 1e-3)

    if coupling_positive and private_nogo_more_robust_under_coupling:
        verdict = ("POSITIVE coupling — λ-coupled drop RISES at low λ, so the private bar is "
                   "STRICTER than #191's fixed-drop 0.9780; the realistic-floor NO-GO is MORE robust.")
    else:
        verdict = (
            "NEGATIVE/NEGLIGIBLE coupling — VALIDATES #191's fixed-drop 0.9780. The per-rung "
            "adverse draft deficit COMPOUNDS along the accepted chain, so the drop is SMALLEST at "
            "the shallow realistic-floor λ̂₁=0.342 (%.4f%%) and LARGEST at full recovery (%.4f%% = "
            "#191's value). #191 used the full-recovery drop, so its fixed-drop composition is a "
            "CONSERVATIVE upper bound; the coupled both-bugs bar is %.6f (shift %.2e in λ, "
            "NEGLIGIBLE). The hypothesised depth-AMPLIFICATION of the adverse drop is REFUTED: the "
            "mechanism compounds with depth, so low realized λ gives LESS accumulated drop, not more. "
            "Realistic-floor NO-GO unchanged (floor 0.342 misses the ~0.978 bar by %.3f in λ)."
            % (bb["drop_at_lambda_floor_pct"], bb["drop_at_lambda1_pct"],
               coupled_bar if coupled_bar is not None else float("nan"),
               bar_shift_pp if bar_shift_pp is not None else float("nan"),
               coupled_bar_gap_to_floor if coupled_bar_gap_to_floor is not None else float("nan")))

    return {
        "beta": beta,
        "imports": {k: imp[k] for k in (
            "tau_low", "tau_central", "K_cal", "step", "decode_drop_pct", "W_hard",
            "adverse_axis", "q_pub", "q_adv", "r_d", "delta_d",
            "drop_descent_176", "drop_both_176",
            "ref_descent_taulow_adverse", "ref_both_taulow_adverse")},
        "constants": {
            "target_official": TARGET_OFFICIAL,
            "disqualify_gate_pct": DISQUALIFY_GATE_PCT,
            "public_bar_both_bugs": PUBLIC_BAR_BOTH,
            "lambda_star_191_fixed_drop": LAMBDA_STAR_191,
            "lambda_floor_liveprobe": LAMBDA_FLOOR,
            "beta_primary": BETA_PRIMARY,
            "beta_range": list(BETA_RANGE),
            "beta_crit_depth1_sufficient": BETA_CRIT,
            "tau_corner_low": tau_low,
            "resid_tol_tps": RESID_TOL_TPS,
        },
        "mechanism_spec": {
            "delta_d": "1 - q_adv[d]/q_pub[d]  (#176 per-rung adverse deficit; +shallow / -deep)",
            "depth_profile": "lambda_d = lambda * beta^(d-1)  (#193 geometric staleness, beta=%.6f)" % beta,
            "adverse_spine": "a_adv[d] = a_pub[d]*(1-delta_d)  on #178's q_floor/q_full interpolation",
            "drop_mech": "1 - E_T(a_adv)/E_T(a_pub)  via #178 et_of_spine (#172 tree DP)",
            "anchor": "drop(lambda) = drop_176 * drop_mech(lambda)/drop_mech(1)  -> drop(1)=drop_176",
            "forward_map": "private_LCB(lambda) = public_LCB(lambda)*(1-drop(lambda))*tau_low  (#191)",
            "quality_component": "drop_176 (depth-flat-delta baseline = full-recovery drop; lambda-INDEPENDENT)",
            "acceptance_component": ("drop_176*(drop_mech(lambda)/drop_mech(1) - 1) (depth-variation of the "
                                     "per-rung deficit). SIGN is a model OUTPUT: here NEGATIVE (the shallow "
                                     "deficits delta_0..2>0 COMPOUND along the accepted chain, so deeper trees "
                                     "carry more drop -> drop FALLS at low lambda). Refutes the hypothesised "
                                     "depth-amplification (which predicted drop RISES at low lambda)."),
        },
        "per_topology": per_topo,
        "headline": {
            "both_bugs_lambda_star_lcb_private_coupled": both_bugs_lambda_star_lcb_private_coupled,
            "lambda_star_191_fixed_drop": LAMBDA_STAR_191,
            "coupled_bar_shift_from_191_pp": bar_shift_pp,
            "private_nogo_more_robust_under_coupling": private_nogo_more_robust_under_coupling,
            "drop_is_lambda_dependent": drop_is_lambda_dependent,
            "both_bugs_drop_at_lambda_floor_pct": bb["drop_at_lambda_floor_pct"],
            "both_bugs_drop_at_lambda_bar_pct": bb["drop_at_lambda_bar_pct"],
            "both_bugs_drop_fixed_191_pct": bb["drop_176_pct"],
            "descent_only_lambda_star_coupled": dd["lambda_star_lcb_private_coupled"],
            "both_bugs_required_at_private_bar": both_bugs_required_at_private_bar,
            "coupled_bar_gap_to_realistic_floor": coupled_bar_gap_to_floor,
            "coupling_sign": coupling_sign,
            "fixed_drop_191_is_conservative": fixed_drop_191_is_conservative,
            "coupling_negligible_near_bar": coupling_negligible_near_bar,
            "drop_range_floor_to_full_pp": drop_range_pp,
            "nogo_verdict_unchanged": nogo_verdict_unchanged,
            "verdict": verdict,
        },
    }


def _selftests(syn: dict, beta: float) -> dict[str, Any]:
    imp = build_public_ctx()
    ctx = imp["ctx"]
    ep = ctx["ep"]
    d176 = load_176()
    tau_low = d176["tau_low"]
    r_d = d176["r_d"]
    topos = {
        "descent_only": (imp["descent_only"][0], imp["descent_only"][1], d176["drop_descent_176"]),
        "both_bugs": (imp["both_bugs"][0], imp["both_bugs"][1], d176["drop_both_176"]),
    }

    # (a) λ-flat limit (acceptance component -> 0) reproduces #191's 0.9780 exactly.
    qfl_b, qfu_b, drop_b = topos["both_bugs"]
    lam_flat = LAC._bisect_lambda(
        lambda l: private_lcb_fixed(ctx, l, qfl_b, qfu_b, drop_b, tau_low), TARGET_OFFICIAL)
    cond_a = bool(_finite(lam_flat) and abs(lam_flat - LAMBDA_STAR_191) < 1e-6)

    # (b) coupled drop at λ=1 reproduces #176's drop 2.35%/2.30% (the anchor).
    cond_b = True
    b_detail = {}
    for lab, (qfl, qfu, d176v) in topos.items():
        got = drop_coupled(ep, 1.0, beta, qfl, qfu, r_d, d176v)
        b_detail[lab] = {"got_pct": got * 100.0, "ref_pct": d176v * 100.0,
                         "resid_pp": abs(got - d176v) * 100.0}
        if abs(got - d176v) > 1e-9:
            cond_b = False
    cond_b = bool(cond_b)

    # (c) public leg reproduces #183/#191 import points within tol.
    import_points = {
        "both_bugs": [(0.342, 404.1), (0.838, 486.2), (0.9052, 500.0), (1.0, 520.95)],
        "descent_only": [(1.0, 505.53)],
    }
    cond_c = True
    c_detail = {}
    for lab, pts in import_points.items():
        qfl, qfu, _ = topos[lab]
        rows = []
        for lam, ref in pts:
            got = public_lcb(ctx, lam, qfl, qfu, 1.0)
            resid = abs(got - ref)
            rows.append({"lambda": lam, "ref": ref, "got": got, "resid": resid})
            if resid >= RESID_TOL_TPS:
                cond_c = False
        c_detail[lab] = rows
    cond_c = bool(cond_c)

    # (d) ordering + mechanism consistency. The coupling sign is an OUTPUT of the
    # mechanism, not an assumption. drop(λ) must be MONOTONE in λ (well-posed), and a
    # NEGATIVE coupling (drop SMALLER at lower λ — i.e. adverse drafts relatively
    # better when recovery is low) must be MECHANISM-BACKED, else flagged. The mechanism
    # here is depth-COMPOUNDING: the shallow per-rung deficits (δ₀,δ₁,δ₂ > 0) stack
    # multiplicatively along the accepted chain, so a deeper tree (higher λ) accumulates
    # MORE total drop. We PROVE this is the driver (not the deep-rung gains δ_d<0) by
    # re-checking with the deep gains removed (r clipped to ≤1): the negative sign must
    # PERSIST. FAIL only on non-monotonicity or an unexplained negative coupling.
    grid = [i / 40.0 for i in range(2, 41)]   # 0.05..1.0 (avoid λ=0 degeneracy)
    cond_d = True
    d_detail = {}
    r_no_deep_gains = [min(1.0, x) for x in r_d]   # zero the multilingual deep-rung gains
    for lab, (qfl, qfu, d176v) in topos.items():
        # monotonicity (either direction) over the grid:
        vals = [drop_coupled(ep, l, beta, qfl, qfu, r_d, d176v) for l in grid]
        nondec = all(vals[i + 1] >= vals[i] - 1e-12 for i in range(len(vals) - 1))
        noninc = all(vals[i + 1] <= vals[i] + 1e-12 for i in range(len(vals) - 1))
        monotone = bool(nondec or noninc)
        drop_floor = drop_coupled(ep, LAMBDA_FLOOR, beta, qfl, qfu, r_d, d176v)
        drop_one = drop_coupled(ep, 1.0, beta, qfl, qfu, r_d, d176v)
        coupling_positive = bool(drop_floor >= drop_one - 1e-12)   # drop rises as λ falls
        # mechanism proof for the negative branch: persists w/o deep gains → compounding.
        df_nodeep = drop_mech(ep, LAMBDA_FLOOR, beta, qfl, qfu, r_no_deep_gains)
        d1_nodeep = drop_mech(ep, 1.0, beta, qfl, qfu, r_no_deep_gains)
        neg_persists_without_deep_gains = bool(df_nodeep < d1_nodeep - 1e-12)
        mechanism_backed = bool(coupling_positive or neg_persists_without_deep_gains)
        ok = bool(monotone and mechanism_backed)
        d_detail[lab] = {
            "monotone": monotone,
            "coupling_sign": "positive(drop↑ as λ↓)" if coupling_positive else "negative(drop↓ as λ↓)",
            "drop_at_floor_pct": drop_floor * 100.0,
            "drop_at_lambda1_pct": drop_one * 100.0,
            "negative_coupling_persists_without_deep_gains": neg_persists_without_deep_gains,
            "mechanism_backed": mechanism_backed,
            "ok": ok,
        }
        cond_d = cond_d and ok
    cond_d = bool(cond_d)

    # monotone increasing private_lcb_coupled (forward map well-posed for bisection)
    mono_fwd = True
    for lab, (qfl, qfu, d176v) in topos.items():
        prev = None
        for i in range(0, 51):
            l = i / 50.0
            v = private_lcb_coupled(ctx, l, qfl, qfu, r_d, d176v, tau_low, beta)
            if prev is not None and v < prev - 1e-9:
                mono_fwd = False
            prev = v

    return {
        "conditions": {
            "a_flat_limit_reproduces_191_0p9780": cond_a,
            "b_coupled_drop_at_lambda1_reproduces_176": cond_b,
            "c_public_leg_reproduces_183_import_points": cond_c,
            "d_drop_monotone_and_coupling_mechanism_backed": cond_d,
            # e (NaN-clean) filled in main() after walking the full payload.
        },
        "flat_limit_lambda_star": lam_flat if _finite(lam_flat) else None,
        "b_anchor_detail": b_detail,
        "c_import_point_detail": c_detail,
        "d_ordering_detail": d_detail,
        "forward_map_monotone_increasing": bool(mono_fwd),
        "partial_passes_a_to_d": bool(cond_a and cond_b and cond_c and cond_d),
    }


def beta_sweep() -> list[dict[str, Any]]:
    """Coupled both-bugs bar across #193's β construction range + flat (β=1)."""
    betas = sorted({BETA_RANGE[0], BETA_PRIMARY, BETA_RANGE[1], BETA_CRIT, 1.0})
    pub = build_public_ctx()
    ctx = pub["ctx"]
    ep = ctx["ep"]
    d176 = load_176()
    tau_low = d176["tau_low"]
    r_d = d176["r_d"]
    qfl, qfu = pub["both_bugs"]
    drop176 = d176["drop_both_176"]
    rows = []
    for b in betas:
        lam_star = LAC._bisect_lambda(
            lambda l: private_lcb_coupled(ctx, l, qfl, qfu, r_d, drop176, tau_low, b),
            TARGET_OFFICIAL)
        rows.append({
            "beta": b,
            "both_bugs_lambda_star_coupled": lam_star if _finite(lam_star) else None,
            "shift_from_191_pp": (lam_star - LAMBDA_STAR_191) if _finite(lam_star) else None,
            "drop_at_floor_pct": drop_coupled(ep, LAMBDA_FLOOR, b, qfl, qfu, r_d, drop176) * 100.0,
            "is_flat_beta1": bool(abs(b - 1.0) < 1e-12),
        })
    return rows


# --------------------------------------------------------------------------- #
# NaN-clean walk.
# --------------------------------------------------------------------------- #
def _nan_paths(node: Any, p: str = "result") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            bad += _nan_paths(v, f"{p}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            bad += _nan_paths(v, f"{p}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        bad.append(p)
    return bad


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def _print_report(syn: dict, st: dict) -> None:
    h = syn["headline"]
    print("\n" + "=" * 94, flush=True)
    print("λ-DEPENDENT PRIVATE DROP (PR #198) — #176 drop × #193 depth mechanism × #191 forward map",
          flush=True)
    print("=" * 94, flush=True)
    print(f"  drop(λ) = drop_176·drop_mech(λ)/drop_mech(1)   [β={syn['beta']:.6f}, τ_low={syn['constants']['tau_corner_low']:.10f}]",
          flush=True)
    print("-" * 94, flush=True)
    for lab in ("both_bugs", "descent_only"):
        t = syn["per_topology"][lab]
        ls = t["lambda_star_lcb_private_coupled"]
        ls_s = f"{ls:.6f}" if ls is not None else "UNREACHABLE(λ=1 LCB<500)"
        print(f"  {lab:<13} drop176={t['drop_176_pct']:.4f}%  "
              f"drop(λ̂₁=0.342)={t['drop_at_lambda_floor_pct']:.4f}%  "
              f"drop(0.9780)={t['drop_at_lambda_bar_pct']:.4f}%", flush=True)
        print(f"  {'':<13} λ*_coupled={ls_s}  "
              f"(#191 fixed={t['lambda_star_lcb_private_fixed_191']})  "
              f"shift={t['coupled_bar_shift_from_fixed']}", flush=True)
    print("-" * 94, flush=True)
    print("  HEADLINE:", flush=True)
    print(f"    both_bugs λ*_coupled              = {h['both_bugs_lambda_star_lcb_private_coupled']}",
          flush=True)
    print(f"    #191 fixed-drop bar              = {h['lambda_star_191_fixed_drop']:.6f}", flush=True)
    print(f"    coupled bar shift from #191      = {h['coupled_bar_shift_from_191_pp']} (λ units)", flush=True)
    print(f"    drop_is_lambda_dependent         = {h['drop_is_lambda_dependent']}", flush=True)
    print(f"    private_nogo_more_robust         = {h['private_nogo_more_robust_under_coupling']}", flush=True)
    print(f"    descent-only λ*_coupled          = {h['descent_only_lambda_star_coupled']}", flush=True)
    print(f"    both_bugs_required_at_private_bar = {h['both_bugs_required_at_private_bar']}", flush=True)
    print(f"    coupled bar gap to floor(0.342)  = {h['coupled_bar_gap_to_realistic_floor']}", flush=True)
    print("-" * 94, flush=True)
    print("  SELF-TEST conditions:", flush=True)
    for k, v in st["conditions"].items():
        print(f"     - {k}: {v}", flush=True)
    for lab in ("both_bugs", "descent_only"):
        dd = st["d_ordering_detail"][lab]
        print(f"     · {lab} coupling: {dd['coupling_sign']} | "
              f"neg_persists_no_deep_gains={dd['negative_coupling_persists_without_deep_gains']} | "
              f"mechanism_backed={dd['mechanism_backed']}", flush=True)
    print("-" * 94, flush=True)
    print(f"  VERDICT: {h['verdict']}", flush=True)
    print("=" * 94, flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors #191; never fatal).
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
        print(f"[lambda-private-drop] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    h = syn["headline"]
    bb = syn["per_topology"]["both_bugs"]
    run = init_wandb_run(
        job_type="lambda-private-drop",
        agent="stark",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["lambda-private-drop", "validity-gate", "private-drop", "mechanism-coupling",
              "salvage-staleness", "composition"],
        config={
            "target_official": TARGET_OFFICIAL, "public_bar_both_bugs": PUBLIC_BAR_BOTH,
            "lambda_star_191_fixed_drop": LAMBDA_STAR_191, "beta_primary": BETA_PRIMARY,
            "beta_range": list(BETA_RANGE), "lambda_floor_liveprobe": LAMBDA_FLOOR,
            "tau_corner_low": syn["constants"]["tau_corner_low"],
            "imports": "stark#176 per-rung drop × denken#193 λ_d=λ̂₁·β^(d−1) × stark#191 forward map × denken#183 card",
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[lambda-private-drop] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    cbar = h["both_bugs_lambda_star_lcb_private_coupled"]
    summary: dict[str, Any] = {
        "lambda_private_drop_self_test_passes": int(bool(payload["self_test_passes"])),
        "both_bugs_lambda_star_lcb_private_coupled": cbar,
        "lambda_star_191_fixed_drop": LAMBDA_STAR_191,
        "coupled_bar_shift_from_191_pp": h["coupled_bar_shift_from_191_pp"],
        "private_nogo_more_robust_under_coupling": int(bool(h["private_nogo_more_robust_under_coupling"])),
        "drop_is_lambda_dependent": int(bool(h["drop_is_lambda_dependent"])),
        "coupling_is_positive": int(h["coupling_sign"] == "positive"),
        "fixed_drop_191_is_conservative": int(bool(h["fixed_drop_191_is_conservative"])),
        "coupling_negligible_near_bar": int(bool(h["coupling_negligible_near_bar"])),
        "drop_range_floor_to_full_pp": h["drop_range_floor_to_full_pp"],
        "nogo_verdict_unchanged": int(bool(h["nogo_verdict_unchanged"])),
        "both_bugs_drop_at_lambda_floor_pct": h["both_bugs_drop_at_lambda_floor_pct"],
        "both_bugs_drop_at_lambda_bar_pct": h["both_bugs_drop_at_lambda_bar_pct"],
        "both_bugs_drop_fixed_191_pct": h["both_bugs_drop_fixed_191_pct"],
        "both_bugs_required_at_private_bar": int(bool(h["both_bugs_required_at_private_bar"])),
        "descent_only_lambda_star_coupled_unreachable":
            int(h["descent_only_lambda_star_coupled"] is None),
        "coupled_bar_gap_to_realistic_floor": h["coupled_bar_gap_to_realistic_floor"],
        "both_bugs_coupled_bar_shift_from_fixed": bb["coupled_bar_shift_from_fixed"],
        "nan_clean": int(bool(payload["nan_clean"])),
        "beta": syn["beta"],
        **{f"selftest_{k}": int(bool(v))
           for k, v in syn["self_test"]["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="lambda_private_drop_result", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[lambda-private-drop] wandb logged: {summary}", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--beta", type=float, default=BETA_PRIMARY, help="depth-decay β (#193)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="lambda-dependent-private-drop")
    args = ap.parse_args(argv)

    syn = synthesize(beta=args.beta)
    st = _selftests(syn, beta=args.beta)
    syn["self_test"] = st
    syn["beta_sweep"] = beta_sweep()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "created_at": created_at,
        "pr": 198,
        "agent": "stark",
        "kind": "lambda-private-drop",
        "synthesis": syn,
    }

    # (e) NaN-clean over the full payload.
    nan_bad = _nan_paths(payload)
    payload["nan_clean"] = not nan_bad
    if nan_bad:
        print(f"[lambda-private-drop] WARNING non-finite values at: {nan_bad}", flush=True)
    syn["self_test"]["conditions"]["e_nan_clean"] = bool(payload["nan_clean"])

    cond = syn["self_test"]["conditions"]
    self_test_passes = bool(all(cond.values()))
    payload["self_test_passes"] = self_test_passes
    syn["self_test"]["lambda_private_drop_self_test_passes"] = self_test_passes

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload["peak_mem_mib"] = round(peak_kib / 1024.0, 3)

    _print_report(syn, st)
    print(f"  PRIMARY lambda_private_drop_self_test_passes = {self_test_passes}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[lambda-private-drop] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = self_test_passes and payload["nan_clean"]
        print(f"[lambda-private-drop] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
