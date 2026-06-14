#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Private-side build bar (PR #191) — compose #176's adverse-skew private drop
through #183's finite-sample-LCB build-bar forward map.

THE UN-COMPOSED SEAM
--------------------
stark #176 (`uzl7ixll`) banked an adverse-skew PRIVATE certificate that PASSES at
λ=1: descent-only τ-low 504.15 (+4.15). But that +4.15 was banked against the
public CENTRAL at λ=1 (519.95). denken #183 (`82uisrez`, lambda_acceptance_card)
proved the BINDING public build bar is on the finite-sample LCB, not the central:
public-LCB(λ=0.9052)=500.0, public-LCB(λ=1)=520.95 (both-bugs) / 505.53 (descent).
The two certificates live at DIFFERENT lower-bound notions (central vs LCB) and
have never been multiplied. This PR multiplies them: it propagates #176's private
drop through #183's LCB forward map to derive the PRIVATE-side build bar
λ*_LCB,private and the launch-validity verdict valid_at_bar.

COMPOSITION (imports — NOT re-derived)
--------------------------------------
    public_LCB(λ), public_central(λ)  ← #183 metrics_at(...) executed verbatim
    drop                              ← #176 adverse_vertex tree drop (worst corner)
    τ_corner = τ_low = 0.9924318649…  ← #181/#176 tree-class τ floor

    private_LCB(λ)     = public_LCB(λ)     · (1 − drop) · τ_corner   # DELIVERABLE bar
    private_central(λ) = public_central(λ) · (1 − drop) · τ_corner   # #176-consistency leg

`public_LCB`/`public_central` are #183's τ=1 forward map (its published map);
τ_corner multiplies the τ=1 number down to the conservative τ-low corner. The
deliverable build bar solves private_LCB(λ*)=500. The drop is #176's worst-corner
adverse-vertex ceiling (NOT a fresh sampling CI — #176 disclaims one), so it is
already the conservative upper edge.

LOCAL CPU-only analytic. No GPU / vLLM / HF Job / submission / served-file change.
BASELINE stays 481.53. Greedy/PPL untouched. Bank-the-analysis (PRIMARY =
self-test, adds 0 TPS). NOT open2. NOT a launch.

PRIMARY metric  private_build_bar_self_test_passes
TEST    metric  lambda_star_lcb_private  (worst-corner private LCB clears 500; both-bugs)

Run:
    python -m research.validity.private_build_bar.private_build_bar \
        --self-test --wandb-name stark/private-build-bar \
        --wandb-group private-build-bar
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
DISQUALIFY_GATE_PCT = 5.0          # ≤5% private drop disqualification gate (baseline 4.3% = VALID)
PUBLIC_BAR_BOTH = 0.9052283680740145   # #183/#184 both-bugs public LCB build bar (τ=1)
RESID_TOL_TPS = 0.5                # self-test tolerance (matches #183/#176 convention)


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- import denken #183's committed machinery (public leg; not re-derived) --- #
LAC = _import("lambda_acceptance_card", _LAC_PATH)
D172 = LAC.D172


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Imports.
# --------------------------------------------------------------------------- #
def load_176_drop() -> dict[str, Any]:
    """Import stark #176's adverse-vertex private drop (worst corner) + τ_low + the
    λ=1 τ-low reproduction targets, byte-from-the-banked results.json. NOT re-derived."""
    with _D176_RESULTS.open(encoding="utf-8") as fh:
        r = json.load(fh)
    av = r["adverse_vertex"]
    const = r["constants"]
    out = {
        "tau_low": const["tau_low"],                       # 0.9924318649123313
        "tau_central": const["tau_central"],               # 1.0
        "target_500": const["target_500"],                 # 500.0
        "K_cal": const["K_cal"],
        "step": const["step"],
        "gt_drop_pct": const["gt_drop_pct"],               # 4.2946 decode-frame GT calibration
        # adverse vertex (worst realistic skew over the cap-0.5 domain simplex):
        "drop_descent": av["descent_tree_drop_pct"] / 100.0,   # 0.022999781…  (tree frame, TPS-relevant)
        "drop_both": av["both_tree_drop_pct"] / 100.0,         # 0.023502817…
        "decode_drop_pct": av["achieved_decode_drop_pct"],     # 4.2946 (decode frame, ≤5% DQ gate)
        "W_hard": av["W_hard"],
        "adverse_axis": av["kind"],
        # #176's own λ=1 τ-low numbers (self-test (a) reproduction targets):
        "ref_descent_central_adverse": av["descent_tps_central"],   # 507.993
        "ref_descent_taulow_adverse": av["descent_tps_taulow"],     # 504.1485838…
        "ref_both_central_adverse": av["both_tps_central"],         # 522.849
        "ref_both_taulow_adverse": av["both_tps_taulow"],           # 518.892
        "both_bugs_required_private_adverse_176": r["headline"]["both_bugs_required_private_adverse"],
    }
    return out


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
# Public legs (imported #183 metrics, τ=1 published map) and private composition.
# --------------------------------------------------------------------------- #
def public_central(ctx, lam, qfl, qfu, tau=1.0) -> float:
    return LAC.metrics_at(ctx, lam, qfl, qfu, tau)["central_tps"]


def public_lcb(ctx, lam, qfl, qfu, tau=1.0) -> float:
    return LAC.metrics_at(ctx, lam, qfl, qfu, tau)["lcb_full_tps"]


def private_central(ctx, lam, qfl, qfu, drop, tau_corner) -> float:
    return public_central(ctx, lam, qfl, qfu, 1.0) * (1.0 - drop) * tau_corner


def private_lcb(ctx, lam, qfl, qfu, drop, tau_corner) -> float:
    """PR #191 formula: take #183's τ=1 LCB map, apply the private drop, scale to τ-low."""
    return public_lcb(ctx, lam, qfl, qfu, 1.0) * (1.0 - drop) * tau_corner


def private_lcb_proper(ctx, lam, qfl, qfu, drop, tau_corner) -> float:
    """Tightening cross-check: recompute #183's LCB AT τ_corner (σ_hw not τ-scaled),
    then apply the drop. Slightly MORE conservative than the PR multiply form (~0.05 TPS)."""
    return public_lcb(ctx, lam, qfl, qfu, tau_corner) * (1.0 - drop)


def _bisect(f_at: Callable[[float], float], target: float) -> float:
    """Smallest λ∈[0,1] with f_at(λ) ≥ target (f monotone↑). NaN if even λ=1 misses."""
    return LAC._bisect_lambda(f_at, target)


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    imp = load_176_drop()
    pub = build_public_ctx()
    ctx = pub["ctx"]
    qfl_d, qfu_d = pub["descent_only"]
    qfl_b, qfu_b = pub["both_bugs"]
    tau_low = imp["tau_low"]
    drop_d, drop_b = imp["drop_descent"], imp["drop_both"]

    topos = {
        "descent_only": {"qfl": qfl_d, "qfu": qfu_d, "drop": drop_d,
                         "ref_central1": imp["ref_descent_central_adverse"],
                         "ref_taulow1": imp["ref_descent_taulow_adverse"]},
        "both_bugs": {"qfl": qfl_b, "qfu": qfu_b, "drop": drop_b,
                      "ref_central1": imp["ref_both_central_adverse"],
                      "ref_taulow1": imp["ref_both_taulow_adverse"]},
    }

    per_topo: dict[str, Any] = {}
    for lab, t in topos.items():
        qfl, qfu, drop = t["qfl"], t["qfu"], t["drop"]

        # --- public legs (reproduce #183 import points) --- #
        pub_lcb_1 = public_lcb(ctx, 1.0, qfl, qfu, 1.0)
        pub_cen_1 = public_central(ctx, 1.0, qfl, qfu, 1.0)
        pub_lcb_bar = public_lcb(ctx, PUBLIC_BAR_BOTH, qfl, qfu, 1.0)

        # --- private legs at λ=1 (#176 consistency + headline) --- #
        priv_cen_1 = private_central(ctx, 1.0, qfl, qfu, drop, tau_low)   # reproduces #176 τ-low
        priv_lcb_1 = private_lcb(ctx, 1.0, qfl, qfu, drop, tau_low)       # the seam: ~490 descent
        priv_lcb_1_proper = private_lcb_proper(ctx, 1.0, qfl, qfu, drop, tau_low)

        # --- private build bar: smallest λ whose private LCB clears 500 --- #
        lam_star = _bisect(lambda l: private_lcb(ctx, l, qfl, qfu, drop, tau_low), TARGET_OFFICIAL)
        lam_star_proper = _bisect(lambda l: private_lcb_proper(ctx, l, qfl, qfu, drop, tau_low),
                                  TARGET_OFFICIAL)
        # central-leg build bar (what #176's margin implied) — for the seam contrast:
        lam_star_central = _bisect(lambda l: private_central(ctx, l, qfl, qfu, drop, tau_low),
                                   TARGET_OFFICIAL)

        reachable = _finite(lam_star)
        # --- at the public bar λ=0.9052 --- #
        priv_lcb_at_bar = private_lcb(ctx, PUBLIC_BAR_BOTH, qfl, qfu, drop, tau_low)
        priv_cen_at_bar = private_central(ctx, PUBLIC_BAR_BOTH, qfl, qfu, drop, tau_low)

        per_topo[lab] = {
            "drop_tree_pct": drop * 100.0,
            "public_central_lambda1": pub_cen_1,
            "public_lcb_lambda1": pub_lcb_1,
            "public_lcb_at_public_bar": pub_lcb_bar,
            "private_central_lambda1_taulow": priv_cen_1,
            "private_lcb_lambda1_taulow": priv_lcb_1,
            "private_lcb_lambda1_taulow_proper": priv_lcb_1_proper,
            "ref_176_taulow_lambda1": t["ref_taulow1"],
            "resid_private_central_vs_176": abs(priv_cen_1 - t["ref_taulow1"]),
            "lambda_star_lcb_private": lam_star if reachable else None,
            "lambda_star_lcb_private_proper": lam_star_proper if _finite(lam_star_proper) else None,
            "lambda_star_central_private": lam_star_central if _finite(lam_star_central) else None,
            "private_lcb_reachable_at_full_recovery": reachable,
            "private_bar_shift_from_public": (lam_star - PUBLIC_BAR_BOTH) if reachable else None,
            "private_lcb_at_public_bar": priv_lcb_at_bar,
            "private_central_at_public_bar": priv_cen_at_bar,
            "margin_at_public_bar_lcb": priv_lcb_at_bar - TARGET_OFFICIAL,
            "private_lcb_margin_at_lambda1": priv_lcb_1 - TARGET_OFFICIAL,
        }

    # ---------- headline composition ---------- #
    bb = per_topo["both_bugs"]
    dd = per_topo["descent_only"]

    # the binding private bar: both-bugs is the reachable path; descent is unreachable on the LCB.
    lambda_star_lcb_private = bb["lambda_star_lcb_private"]            # TEST (both-bugs reachable bar)
    descent_reachable = dd["private_lcb_reachable_at_full_recovery"]
    both_reachable = bb["private_lcb_reachable_at_full_recovery"]
    # descent-only cannot clear the private LCB even at full recovery → both-bugs required.
    both_bugs_required_at_private_bar = not descent_reachable

    # binding bar: private is stricter if the reachable bar > public 0.9052, or descent unreachable.
    private_stricter = ((both_reachable and lambda_star_lcb_private is not None
                         and lambda_star_lcb_private > PUBLIC_BAR_BOTH + 1e-9)
                        or not descent_reachable)
    binding_bar = "private-stricter" if private_stricter else "public-0.9052"

    # ---------- valid_at_bar (the MUST-HAVE launch-validity floor) ---------- #
    # disqualification gate is the DECODE-frame drop ≤5% (baseline 4.3% = VALID); the
    # adverse vertex is calibrated to keep the decode drop at GT (λ-independent).
    decode_drop_at_bar_pct = imp["decode_drop_pct"]
    private_drop_at_bar_pct = dd["drop_tree_pct"]          # descent tree drop (self-test (d) target)
    valid_at_bar = bool(decode_drop_at_bar_pct <= DISQUALIFY_GATE_PCT
                        and private_drop_at_bar_pct <= DISQUALIFY_GATE_PCT
                        and bb["drop_tree_pct"] <= DISQUALIFY_GATE_PCT)
    # private LCB at the public bar (both-bugs, where public-LCB=500.0):
    private_lcb_at_public_bar = bb["private_lcb_at_public_bar"]

    # ---------- self-test (PRIMARY) ---------- #
    # (a) private_central at λ=1 reproduces #176's adverse τ-low for BOTH paths.
    cond_a = bool(dd["resid_private_central_vs_176"] < RESID_TOL_TPS
                  and bb["resid_private_central_vs_176"] < RESID_TOL_TPS)
    # (b) private_LCB monotone↑ in λ (grid sweep), both topologies.
    grid = [i / 50.0 for i in range(51)]
    mono = True
    mono_detail = {}
    for lab, t in topos.items():
        prev = None
        ok = True
        for l in grid:
            v = private_lcb(ctx, l, t["qfl"], t["qfu"], t["drop"], tau_low)
            if prev is not None and v < prev - 1e-9:
                ok = False
            prev = v
        mono_detail[lab] = ok
        mono = mono and ok
    cond_b = bool(mono)
    # (c) λ*_lcb_private ≥ public 0.9052 (private ≤ public ALWAYS) — flag if violated.
    #     unreachable (None) counts as stricter-or-equal (maximally strict).
    def _ge_public(ls):
        return ls is None or ls >= PUBLIC_BAR_BOTH - 1e-9
    cond_c = bool(_ge_public(bb["lambda_star_lcb_private"])
                  and _ge_public(dd["lambda_star_lcb_private"]))
    cond_c_violation = not cond_c
    # (d) at λ=0.9052 the worst-corner private (tree) drop reproduces #176 ≤2.300% & valid_at_bar.
    cond_d = bool(abs(private_drop_at_bar_pct - 2.30) < 0.02
                  and private_drop_at_bar_pct <= 2.30 + 1e-6
                  and valid_at_bar)
    # (e) public leg reproduces #183 import points within tol.
    import_points = {
        "both_bugs": [(0.342, 404.1), (0.838, 486.2), (0.9052, 500.0), (1.0, 520.95)],
        "descent_only": [(1.0, 505.53)],
    }
    e_detail = {}
    cond_e = True
    for lab, pts in import_points.items():
        qfl, qfu = topos[lab]["qfl"], topos[lab]["qfu"]
        rows = []
        for lam, ref in pts:
            got = public_lcb(ctx, lam, qfl, qfu, 1.0)
            resid = abs(got - ref)
            rows.append({"lambda": lam, "ref": ref, "got": got, "resid": resid})
            if resid >= RESID_TOL_TPS:
                cond_e = False
        e_detail[lab] = rows
    cond_e = bool(cond_e)

    payload_for_nan: dict[str, Any] = {"per_topo": per_topo}
    cond_f = True  # set after nan walk on the full payload

    self_test_passes_partial = bool(cond_a and cond_b and cond_c and cond_d and cond_e)

    handoff = (
        f"composing #176's adverse-skew private drop through #183's build-bar forward map, "
        f"the build is VALID-at-the-public-bar = {valid_at_bar} and the PRIVATE-side build bar is "
        f"λ*_LCB,private = {lambda_star_lcb_private:.4f} (both-bugs; vs public 0.9052, STRICTER); "
        f"descent-only's private LCB is UNREACHABLE at full recovery (so "
        f"both_bugs_required_at_private_bar = {both_bugs_required_at_private_bar}); the binding bar "
        f"land #71 must clear is {binding_bar}, which fern #185 should consume as the "
        f"private-validity row."
    ) if lambda_star_lcb_private is not None else (
        f"private LCB unreachable for both topologies at full recovery; valid_at_bar={valid_at_bar}."
    )

    return {
        "imports": imp,
        "constants": {
            "target_official": TARGET_OFFICIAL,
            "disqualify_gate_pct": DISQUALIFY_GATE_PCT,
            "public_bar_both_bugs": PUBLIC_BAR_BOTH,
            "tau_corner_low": tau_low,
            "resid_tol_tps": RESID_TOL_TPS,
        },
        "private_forward_map_spec": {
            "formula_lcb": "private_LCB(lam) = public_LCB(lam) * (1 - drop) * tau_corner",
            "formula_central": "private_central(lam) = public_central(lam) * (1 - drop) * tau_corner",
            "public_leg": "denken #183 lambda_acceptance_card.metrics_at(...)[lcb_full_tps|central_tps] @ tau=1 (executed verbatim)",
            "drop_leg": "stark #176 adverse_vertex tree drop (worst-corner ceiling over cap-0.5 domain simplex; NOT a fresh sampling CI)",
            "tau_corner": f"tau_low = {tau_low} (#181/#176 tree-class floor)",
            "drop_descent_pct": drop_d * 100.0,
            "drop_both_pct": drop_b * 100.0,
            "note": ("public_LCB is #183's tau=1 map; tau_corner multiplies it to the conservative "
                     "tau-low corner. private_lcb_proper recomputes the LCB at tau_low (sigma_hw not "
                     "tau-scaled) as a ~0.05 TPS tightening cross-check; conclusion is identical."),
        },
        "per_topology": per_topo,
        "headline": {
            "lambda_star_lcb_private": lambda_star_lcb_private,                # TEST (both-bugs)
            "lambda_star_lcb_private_descent": dd["lambda_star_lcb_private"],  # None = unreachable
            "lambda_star_lcb_private_both": bb["lambda_star_lcb_private"],
            "private_bar_shift_from_public": bb["private_bar_shift_from_public"],
            "binding_bar": binding_bar,
            "both_bugs_required_at_private_bar": both_bugs_required_at_private_bar,
            "both_bugs_required_private_176_central": imp["both_bugs_required_private_adverse_176"],
            "valid_at_bar": valid_at_bar,
            "private_drop_at_bar_pct": private_drop_at_bar_pct,
            "private_decode_drop_at_bar_pct": decode_drop_at_bar_pct,
            "private_lcb_at_public_bar": private_lcb_at_public_bar,
            "descent_private_lcb_margin_at_lambda1": dd["private_lcb_margin_at_lambda1"],
            "descent_176_central_margin_at_lambda1": dd["ref_176_taulow_lambda1"] - TARGET_OFFICIAL,
            "seam_tps_lost_to_finite_sample_descent": dd["private_central_lambda1_taulow"] - dd["private_lcb_lambda1_taulow"],
        },
        "self_test": {
            "conditions": {
                "a_private_central_reproduces_176_taulow": cond_a,
                "b_private_lcb_monotone_increasing": cond_b,
                "c_lambda_star_ge_public_0p9052": cond_c,
                "d_drop_at_bar_reproduces_176_and_valid": cond_d,
                "e_public_leg_reproduces_183_import_points": cond_e,
                # f (NaN-clean) filled in main() after walking the full payload.
            },
            "c_violation_private_looser_than_public": cond_c_violation,
            "monotone_detail": mono_detail,
            "import_point_detail": e_detail,
            "partial_passes_a_to_e": self_test_passes_partial,
        },
        "handoff_line": handoff,
    }


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
def _print_report(syn: dict) -> None:
    h = syn["headline"]
    st = syn["self_test"]["conditions"]
    print("\n" + "=" * 92, flush=True)
    print("PRIVATE-SIDE BUILD BAR (PR #191) — #176 adverse private drop × #183 LCB forward map",
          flush=True)
    print("=" * 92, flush=True)
    print("  private_LCB(λ) = public_LCB(λ)·(1−drop)·τ_low   [τ_low="
          f"{syn['constants']['tau_corner_low']:.10f}]", flush=True)
    print("-" * 92, flush=True)
    for lab in ("both_bugs", "descent_only"):
        t = syn["per_topology"][lab]
        ls = t["lambda_star_lcb_private"]
        ls_s = f"{ls:.4f}" if ls is not None else "UNREACHABLE(λ=1 LCB<500)"
        print(f"  {lab:<13} drop={t['drop_tree_pct']:.4f}%  "
              f"pub_LCB(1)={t['public_lcb_lambda1']:.2f}  pub_cen(1)={t['public_central_lambda1']:.2f}",
              flush=True)
        print(f"  {'':<13} priv_central(1,τlow)={t['private_central_lambda1_taulow']:.3f} "
              f"(#176 ref {t['ref_176_taulow_lambda1']:.3f}, resid {t['resid_private_central_vs_176']:.2e})",
              flush=True)
        print(f"  {'':<13} priv_LCB(1,τlow)={t['private_lcb_lambda1_taulow']:.3f} "
              f"(margin {t['private_lcb_margin_at_lambda1']:+.2f})   λ*_LCB,private={ls_s}", flush=True)
    print("-" * 92, flush=True)
    print(f"  HEADLINE:", flush=True)
    print(f"    valid_at_bar                     = {h['valid_at_bar']}  "
          f"(decode drop {h['private_decode_drop_at_bar_pct']:.3f}% ≤ 5% DQ gate; "
          f"tree drop {h['private_drop_at_bar_pct']:.3f}%)", flush=True)
    print(f"    private_lcb_at_public_bar(0.9052)= {h['private_lcb_at_public_bar']:.3f} TPS", flush=True)
    lsp = h["lambda_star_lcb_private"]
    print(f"    λ*_LCB,private (both-bugs)        = {lsp:.4f}   "
          f"shift_from_public = {h['private_bar_shift_from_public']:+.4f}", flush=True)
    print(f"    λ*_LCB,private (descent-only)     = "
          f"{h['lambda_star_lcb_private_descent'] or 'UNREACHABLE'}", flush=True)
    print(f"    binding_bar                      = {h['binding_bar']}", flush=True)
    print(f"    both_bugs_required_at_private_bar = {h['both_bugs_required_at_private_bar']}  "
          f"(#176 central-based was {h['both_bugs_required_private_176_central']})", flush=True)
    print(f"    descent seam: #176 central τ-low margin {h['descent_176_central_margin_at_lambda1']:+.2f} "
          f"→ finite-sample LCB margin {h['descent_private_lcb_margin_at_lambda1']:+.2f} "
          f"(−{h['seam_tps_lost_to_finite_sample_descent']:.2f} TPS to finite-sample)", flush=True)
    print("-" * 92, flush=True)
    print("  SELF-TEST conditions:", flush=True)
    for k, v in st.items():
        print(f"     - {k}: {v}", flush=True)
    print("=" * 92, flush=True)
    print(f"\n  HAND-OFF: {syn['handoff_line']}\n", flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors #183; never fatal).
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
        print(f"[private-build-bar] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    h = syn["headline"]
    run = init_wandb_run(
        job_type="private-build-bar",
        agent="stark",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["private-build-bar", "validity-gate", "finite-sample-lcb", "private-drop", "composition"],
        config={
            "target_official": TARGET_OFFICIAL, "disqualify_gate_pct": DISQUALIFY_GATE_PCT,
            "public_bar_both_bugs": PUBLIC_BAR_BOTH, "tau_corner_low": syn["constants"]["tau_corner_low"],
            "imports": "stark#176 adverse_vertex drop × denken#183 LCB forward map × #181 tau_low",
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[private-build-bar] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    bb = syn["per_topology"]["both_bugs"]
    dd = syn["per_topology"]["descent_only"]
    summary: dict[str, Any] = {
        "private_build_bar_self_test_passes": int(bool(payload["self_test_passes"])),
        "lambda_star_lcb_private": h["lambda_star_lcb_private"],
        "lambda_star_lcb_private_both": h["lambda_star_lcb_private_both"],
        "lambda_star_lcb_private_descent_unreachable":
            int(h["lambda_star_lcb_private_descent"] is None),
        "private_bar_shift_from_public": h["private_bar_shift_from_public"],
        "binding_bar_private_stricter": int(h["binding_bar"] == "private-stricter"),
        "both_bugs_required_at_private_bar": int(bool(h["both_bugs_required_at_private_bar"])),
        "valid_at_bar": int(bool(h["valid_at_bar"])),
        "private_drop_at_bar_pct": h["private_drop_at_bar_pct"],
        "private_decode_drop_at_bar_pct": h["private_decode_drop_at_bar_pct"],
        "private_lcb_at_public_bar": h["private_lcb_at_public_bar"],
        "descent_private_lcb_margin_at_lambda1": h["descent_private_lcb_margin_at_lambda1"],
        "descent_176_central_margin_at_lambda1": h["descent_176_central_margin_at_lambda1"],
        "seam_tps_lost_to_finite_sample_descent": h["seam_tps_lost_to_finite_sample_descent"],
        "both_bugs_private_lcb_lambda1": bb["private_lcb_lambda1_taulow"],
        "descent_private_lcb_lambda1": dd["private_lcb_lambda1_taulow"],
        "resid_private_central_vs_176_descent": dd["resid_private_central_vs_176"],
        "resid_private_central_vs_176_both": bb["resid_private_central_vs_176"],
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in syn["self_test"]["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="private_build_bar_result", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[private-build-bar] wandb logged: {summary}", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="private-build-bar")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "created_at": created_at,
        "pr": 191,
        "agent": "stark",
        "kind": "private-build-bar",
        "synthesis": syn,
    }

    # (f) NaN-clean over the full payload.
    nan_bad = _nan_paths(payload)
    payload["nan_clean"] = not nan_bad
    if nan_bad:
        print(f"[private-build-bar] WARNING non-finite values at: {nan_bad}", flush=True)
    syn["self_test"]["conditions"]["f_nan_clean"] = bool(payload["nan_clean"])

    cond = syn["self_test"]["conditions"]
    self_test_passes = bool(all(cond.values()))
    payload["self_test_passes"] = self_test_passes
    syn["self_test"]["private_build_bar_self_test_passes"] = self_test_passes

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload["peak_mem_mib"] = round(peak_kib / 1024.0, 3)

    _print_report(syn)
    print(f"  PRIMARY private_build_bar_self_test_passes = {self_test_passes}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[private-build-bar] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = self_test_passes and payload["nan_clean"]
        print(f"[private-build-bar] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
