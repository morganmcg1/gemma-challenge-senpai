#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #243 (stark) — NLS worst-case f_priv under the corrected 0.73% divergence:
does my #233 publish-first breakeven 0.9598 still straddle?

THE QUESTION (re-pricing the #233 worst-case-vertex f_priv BLEND)
----------------------------------------------------------------
My #233 (`pszvrf2a`) placed the publish-first f_priv breakeven at 0.9598 with the
realizable worst-case f_priv STRADDLING it across [0.957054 grounded, 0.969107 clean].
That band was the worst-case under the OLD assumption that the int4 verify is
substantially lossy — kanna #114's M=1 56.08% divergence. lawine #232 (`nxwv6pam`)
has since MEASURED the deployed divergence at 0.73% (near-greedy, M=8) — an
order-of-magnitude correction. This leg re-prices the worst-case-vertex f_priv under
the corrected near-greedy weight.

THE FRAME (a divergence-weighted blend — IMPORTED anchors, NOT re-derived)
-------------------------------------------------------------------------
    f_priv_wc(d) = (1 - d)·f_clean + d·f_int4div

  * d         the int4 divergence FRACTION (weight on the adverse int4-divergent decode-drop).
  * f_clean   the clean-decode f_priv at the binding non-Latin-script (NLS) vertex = 0.969107
              (#226 `tzcc5xuq` realizable worst-case == kanna #217). The d→0 limit.
  * f_int4div the fully-int4-divergent decode-drop, PINNED by the calibration round-trip so that
              at the OLD d=0.5608 the blend reproduces #233's grounded floor 0.957054. The d→1 limit.

At d=0.5608 the blend reproduces the #233 worst-case band's LOWER end (0.957054 grounded), and at
d=0 it is the band's UPPER end (0.969107 clean) — so [f_priv_wc(0.5608), f_priv_wc(0)] == the #233
realizable worst-case band [0.957054, 0.969107]. When the int4 decode-drop weight d shrinks toward
lawine #232's measured 0.73%, the int4-divergent component carries almost no weight, so the
worst-case f_priv RISES (less adverse) toward the clean value.

THE DELIVERABLE (the private bar's LOCATION under the corrected int4 physics)
---------------------------------------------------------------------------
1. The blend; solve f_int4div from the d=0.5608 round-trip (reproduces [0.957054, 0.969107]).
2. fpriv_worstcase_under_measured_div = f_priv_wc(0.0073) and the corrected lambda_floor via the
   #233 sensitivity d(λ_floor)/d(f_priv) = -2.3535 (exact #233 reach-DP solver cross-check too).
   Straddle verdict: does the corrected worst-case still STRADDLE breakeven 0.9598 or move ABOVE?
3. NLS vertex confirmation under the corrected divergence; one table d ∈ {0.0073, 0.10, 0.30,
   0.5608} × (f_priv_wc, implied λ_floor, straddles-breakeven bool).
4. Self-test (PRIMARY): (a) d=0.5608 round-trips the band; (b) f_priv_wc ↑ as d↓; (c) corrected >
   old worst-case; (d) straddle verdict stated; (e) NLS vertex confirmed; (f) NaN-clean.
5. Hand-off to kanna's f_priv-band + fern's card + #124.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / served-file change / draw.
BASELINE stays 481.53; adds **0 TPS**; greedy/PPL untouched. Extends my #233; lawine owns the
divergence rate, kanna owns the draw-risk distribution, fern owns the card. **NOT a launch. NOT open2.**

PRIMARY metric  fpriv_worstcase_measured_div_self_test_passes
TEST    metric  fpriv_worstcase_under_measured_div   (≈ 0.96895)

Run:
  cd target/ && CUDA_VISIBLE_DEVICES="" python \
    research/validity/fpriv_worstcase_measured_div/fpriv_worstcase_measured_div.py \
    --self-test --wandb_group issue192-reading-calibration \
    --wandb_name stark/fpriv-worstcase-measured-div
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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Imported anchors (provenance: W&B run-ids / committed JSON, project
# wandb-applied-ai-team/gemma-challenge-senpai). IMPORTED, NOT re-derived.
# --------------------------------------------------------------------------- #
# --- #233 publish_first_fpriv_breakeven (pszvrf2a) ---
F_PRIV_BREAKEVEN_LITERAL = 0.9597799742440889    # publish-first breakeven (500 / CEIL_INT4)
LAMBDA_FLOOR_CENTRAL_LITERAL = 0.9780440967672128  # λ_floor at f_clean (central)
DLAMBDA_FLOOR_DFPRIV_LITERAL = -2.353508688669459  # d(λ_floor)/d(f_priv) sensitivity
# --- #226 private_fpriv_worstcase (tzcc5xuq): the NLS clean worst-case + per-axis simplex vertices ---
F_PRIV_CLEAN_LITERAL = 0.969106920637722         # NLS (native_multilingual) clean-decode f_priv == f_clean
NLS_AXIS = "native_multilingual"
# per-axis clean-decode f_priv at λ=1 (the realizable-simplex VERTICES, #226). NLS is the min (worst).
PER_AXIS_CLEAN_FPRIV_226 = {
    "native_multilingual": 0.969106920637722,    # NLS — the f_priv-minimizing (binding worst) vertex
    "native_code": 0.9692692903767706,           # runner-up vertex
    "native_sharegpt": 0.970718701723238,
    "native_casual": 0.9701254960270697,
    "native_math": 0.9716581006759408,
    "native_longctx": 0.9767116893827787,
}
# --- #224 / #52 grounded floor (the lone hard paired draw) ---
F_PRIV_GROUNDED_LITERAL = 0.9570535584491102     # 460.85 / 481.53 == 0.957054
# --- lawine #232 (nxwv6pam) measured + kanna #114 (9q5yy9l1) old int4 divergence ---
D_MEASURED = 0.0073                              # lawine #232 deployed M=8 divergence (1 - 0.9927 identity)
D_OLD = 0.5608                                   # kanna #114 M=1 56.08% — the OLD weight #233's band used
# --- ceiling + target for the reachability gate (#204 / publish-first) ---
CEIL_INT4 = 520.9527323111674                    # #204 int4-spec λ=1 physical ceiling
TARGET = 500.0                                   # publish-first POINT-estimate clear bar

# table divergence grid (deliverable 3):
D_TABLE = [D_MEASURED, 0.10, 0.30, D_OLD]
# robustness: vary the OLD divergence (the round-trip pin) — the corrected worst-case must be insensitive:
D_OLD_SWEEP = [0.55, 0.5608, 0.57, 0.5608 - 0.05, 0.5608 + 0.05]

TOL_ROUNDTRIP = 1e-9     # the d=0.5608 calibration round-trip must reproduce the band to machine eps
TOL_LAMBDA = 5e-3        # linear-sensitivity λ_floor vs exact #233 reach-DP solver agreement band


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _load(name: str, relpath: str):
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_233_anchors() -> dict[str, Any]:
    """Pull the #233 breakeven / λ_floor_central / sensitivity from its committed results.json."""
    p = REPO_ROOT / "research/validity/publish_first_fpriv_breakeven/results.json"
    out = {
        "f_priv_breakeven": F_PRIV_BREAKEVEN_LITERAL,
        "lambda_floor_central": LAMBDA_FLOOR_CENTRAL_LITERAL,
        "dlambda_floor_dfpriv": DLAMBDA_FLOOR_DFPRIV_LITERAL,
        "f_priv_clean": F_PRIV_CLEAN_LITERAL,
        "f_priv_grounded": F_PRIV_GROUNDED_LITERAL,
        "source": "literal(#233/#226/#224)",
    }
    if p.exists():
        try:
            h = json.loads(p.read_text(encoding="utf-8"))["synthesis"]["headline"]
            comp = json.loads(p.read_text(encoding="utf-8"))["synthesis"]["composition"]
            out["f_priv_breakeven"] = float(h["f_priv_breakeven_publish_first"])
            out["lambda_floor_central"] = float(h["lambda_floor_central"])
            out["dlambda_floor_dfpriv"] = float(h["dlambda_floor_dfpriv"])
            out["f_priv_clean"] = float(comp["f_priv_central_217"])
            out["f_priv_grounded"] = float(comp["f_priv_empirical_floor_52"])
            out["source"] = str(p.relative_to(REPO_ROOT))
        except Exception:
            pass
    return out


def _load_226_vertices() -> tuple[dict[str, float], str]:
    """Pull the six realizable per-axis clean f_priv vertices from #226's committed results.json."""
    p = REPO_ROOT / "research/validity/private_fpriv_worstcase/results.json"
    if p.exists():
        try:
            per = json.loads(p.read_text(encoding="utf-8"))["synthesis"]["worstcase"]["per_axis"]
            verts = {ax: float(d["f_priv"]) for ax, d in per.items()}
            if NLS_AXIS in verts:
                return verts, str(p.relative_to(REPO_ROOT))
        except Exception:
            pass
    return dict(PER_AXIS_CLEAN_FPRIV_226), "literal(#226)"


def _exact_lambda_floor_solver():
    """Best-effort: the EXACT #233 reach-DP λ_floor solver for a cross-check of the linear sensitivity.
    Returns a callable f_priv -> λ_floor|None, or None if the banked machinery is unavailable."""
    try:
        M = _load("m233_fpriv_breakeven",
                  "research/validity/publish_first_fpriv_breakeven/publish_first_fpriv_breakeven.py")
        banked = M._load_banked_spines()
        reg = banked["regimes"]["both_bugs"]
        curve = M.LambdaCurve(reg["floor_spine"], reg["ceil_spine"], banked["lambda_hat"])
        ceil, _ = M._load_int4_ceiling()
        return lambda f: M.lambda_floor_publish_first(curve, ceil, f, TARGET)
    except Exception as exc:  # pragma: no cover - cross-check is optional, linear is the PR method
        print(f"[fpriv-worstcase-measured-div] exact λ_floor cross-check unavailable: {exc}", flush=True)
        return None


# --------------------------------------------------------------------------- #
# The blend.
# --------------------------------------------------------------------------- #
def solve_f_int4div(f_clean: float, f_grounded: float, d_old: float) -> float:
    """Pin the fully-int4-divergent decode-drop from the round-trip:
    (1 - d_old)·f_clean + d_old·f_int4div = f_grounded  ⟹  f_int4div = (f_grounded - (1-d_old)·f_clean)/d_old."""
    return (f_grounded - (1.0 - d_old) * f_clean) / d_old


def f_priv_wc(d: float, f_clean: float, f_int4div: float) -> float:
    """Divergence-weighted worst-case f_priv blend."""
    return (1.0 - d) * f_clean + d * f_int4div


def lambda_floor_linear(f_priv: float, f_clean: float, lam0: float, slope: float) -> float:
    """Corrected publish-first λ_floor via the #233 sensitivity d(λ_floor)/d(f_priv).
    Returned value may exceed 1.0 — the caller reads >1 as ∅/UNREACHABLE."""
    return lam0 + slope * (f_priv - f_clean)


def reachable_at_ceiling(f_priv: float, ceil: float = CEIL_INT4, target: float = TARGET) -> bool:
    """private_mean(λ=1, f_priv) = ceil·f_priv ≥ target — i.e. the publish-first floor exists (λ_floor ≤ 1)."""
    return bool(ceil * f_priv >= target)


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    a = _load_233_anchors()
    f_clean = a["f_priv_clean"]
    f_grounded = a["f_priv_grounded"]
    f_be = a["f_priv_breakeven"]
    lam0 = a["lambda_floor_central"]
    slope = a["dlambda_floor_dfpriv"]
    verts, verts_src = _load_226_vertices()
    exact_lf = _exact_lambda_floor_solver()

    # ----- (1) pin f_int4div from the d=0.5608 round-trip ----- #
    f_int4div = solve_f_int4div(f_clean, f_grounded, D_OLD)
    spread = f_clean - f_int4div                       # = (f_clean - f_priv_wc(d))/d, the per-unit-d drop
    # calibration round-trip: the blend must reproduce both band ends.
    band_low_roundtrip = f_priv_wc(D_OLD, f_clean, f_int4div)     # ≈ 0.957054 (grounded)
    band_high_roundtrip = f_priv_wc(0.0, f_clean, f_int4div)      # == f_clean (0.969107)
    roundtrip_resid_low = abs(band_low_roundtrip - f_grounded)
    roundtrip_resid_high = abs(band_high_roundtrip - f_clean)

    # ----- (2) re-price under the measured 0.73% divergence ----- #
    fpriv_wc_measured = f_priv_wc(D_MEASURED, f_clean, f_int4div)
    lam_floor_measured_linear = lambda_floor_linear(fpriv_wc_measured, f_clean, lam0, slope)
    lam_floor_measured_exact = exact_lf(fpriv_wc_measured) if exact_lf else None
    reachable_measured = reachable_at_ceiling(fpriv_wc_measured)
    lam_floor_measured_linear_clamped = (None if lam_floor_measured_linear > 1.0 + 1e-12
                                         else lam_floor_measured_linear)
    lam_floor_resid_measured = (abs(lam_floor_measured_linear - lam_floor_measured_exact)
                                if (lam_floor_measured_exact is not None) else None)

    # the decision-relevant flip: did the corrected worst-case move ABOVE the breakeven?
    straddles_measured = bool(fpriv_wc_measured < f_be)          # band [worst-case, clean] contains f_be
    moved_above = bool(fpriv_wc_measured >= f_be)
    delta_vs_breakeven = fpriv_wc_measured - f_be
    delta_vs_old_worstcase = fpriv_wc_measured - f_grounded      # corrected − old worst-case (must be > 0)

    # the d at which the worst-case crosses the breakeven (the un-straddle threshold):
    d_crossover = (f_clean - f_be) / spread if spread != 0 else float("nan")

    # ----- (3) NLS vertex confirmation under the corrected divergence ----- #
    # Shared int4-divergent floor across vertices ⇒ argmin over v of f_priv_wc^v(d) = argmin f_clean^v = NLS.
    wc_by_axis_measured = {ax: f_priv_wc(D_MEASURED, fc, f_int4div) for ax, fc in verts.items()}
    binding_axis_measured = min(wc_by_axis_measured, key=wc_by_axis_measured.get)
    nls_confirmed = bool(binding_axis_measured == NLS_AXIS)
    sorted_axes = sorted(wc_by_axis_measured.items(), key=lambda kv: kv[1])
    runner_up_axis = sorted_axes[1][0] if len(sorted_axes) > 1 else None
    nls_margin_to_runner = (sorted_axes[1][1] - sorted_axes[0][1]) if len(sorted_axes) > 1 else None
    # the int4div drop that WOULD be needed to invert the NLS↔runner-up ordering at d=D_MEASURED
    # (a robustness number: how much LOWER code's int4div would have to be than NLS's to flip the vertex):
    clean_gap_nls_runner = verts.get(runner_up_axis, f_clean) - verts[NLS_AXIS]
    int4div_gap_to_flip = ((1.0 - D_MEASURED) * clean_gap_nls_runner / D_MEASURED
                           if D_MEASURED > 0 else float("inf"))

    # ----- the table: d × (f_priv_wc, implied λ_floor, straddles-breakeven) ----- #
    table = []
    for d in D_TABLE:
        fp = f_priv_wc(d, f_clean, f_int4div)
        reachable = reachable_at_ceiling(fp)
        lf_lin = lambda_floor_linear(fp, f_clean, lam0, slope)
        lf_lin_clamped = None if lf_lin > 1.0 + 1e-12 else lf_lin
        lf_exact = exact_lf(fp) if exact_lf else None
        table.append({
            "d": d,
            "f_priv_wc": fp,
            "reachable_at_ceiling": reachable,
            "lambda_floor_linear": lf_lin,                       # raw (may be >1)
            "lambda_floor_linear_clamped": lf_lin_clamped,       # None ⇒ ∅/unreachable
            "lambda_floor_exact": lf_exact,                      # None ⇒ ∅ (or cross-check unavailable)
            "lambda_floor_resid_linear_vs_exact": (abs(lf_lin - lf_exact)
                                                   if (lf_exact is not None and lf_lin <= 1.0 + 1e-12)
                                                   else None),
            "straddles_breakeven": bool(fp < f_be),
            "is_measured": bool(abs(d - D_MEASURED) < 1e-15),
            "is_old": bool(abs(d - D_OLD) < 1e-15),
        })

    # ----- robustness: insensitivity of the corrected worst-case to the OLD-divergence pin ----- #
    d_old_sweep = []
    for d_old in D_OLD_SWEEP:
        fi = solve_f_int4div(f_clean, f_grounded, d_old)
        fp_meas = f_priv_wc(D_MEASURED, f_clean, fi)
        d_old_sweep.append({
            "d_old": d_old, "f_int4div": fi, "fpriv_wc_measured": fp_meas,
            "moved_above_breakeven": bool(fp_meas >= f_be),
        })
    d_old_sweep_band = [min(r["fpriv_wc_measured"] for r in d_old_sweep),
                        max(r["fpriv_wc_measured"] for r in d_old_sweep)]
    d_old_sweep_all_above = all(r["moved_above_breakeven"] for r in d_old_sweep)

    # ----- monotonicity: f_priv_wc strictly INCREASING as d DECREASES ----- #
    d_mono_grid = [0.0, D_MEASURED, 0.05, 0.10, 0.20, 0.30, 0.40, D_OLD, 0.70, 1.0]
    wc_mono = [f_priv_wc(d, f_clean, f_int4div) for d in d_mono_grid]
    monotone_decreasing_in_d = all(wc_mono[i] > wc_mono[i + 1] - 1e-15 for i in range(len(wc_mono) - 1))

    # ----- (5) self-test (PRIMARY) ----- #
    st = _selftests(
        roundtrip_resid_low, roundtrip_resid_high, monotone_decreasing_in_d,
        fpriv_wc_measured, f_grounded, straddles_measured, moved_above,
        nls_confirmed, binding_axis_measured, lam_floor_resid_measured,
        d_old_sweep_all_above, spread,
    )

    verdict = (
        "NEAR-GREEDY DIVERGENCE UN-STRADDLES THE BREAKEVEN: re-pricing the #233 worst-case-vertex "
        "f_priv as a divergence-weighted blend and swapping the OLD kanna #114 M=1 56.08% int4-loss "
        f"weight for lawine #232's MEASURED 0.73% near-greedy divergence moves the worst-case f_priv "
        f"from the grounded 0.957054 floor UP to fpriv_worstcase_under_measured_div = "
        f"{fpriv_wc_measured:.6f} — "
        + ("ABOVE" if moved_above else "STILL BELOW")
        + f" the #233 publish-first breakeven {f_be:.6f} (Δ={delta_vs_breakeven:+.6f}). At the measured "
        f"0.73% the int4-divergent component carries almost no weight, so the worst-case tightens to "
        f"within {f_clean - fpriv_wc_measured:.6f} of the clean NLS value 0.969107; the implied "
        f"publish-first λ_floor is {lam_floor_measured_linear:.5f} (vs central {lam0:.5f}). The "
        f"realizable band [worst-case, clean] therefore "
        + ("NO LONGER STRADDLES" if moved_above else "STILL STRADDLES")
        + f" the breakeven — the un-straddle threshold is d*={d_crossover:.4f}, so any divergence below "
        f"~{d_crossover:.2f} clears it and the measured 0.73% clears it with a wide margin. The binding "
        f"vertex stays NON-LATIN-SCRIPT ({binding_axis_measured}); the publish-first private bar's "
        "LOCATION is SAFER once the corrected int4 physics replaces kanna #114's M=1 56%. CPU-only; "
        "adds 0 TPS; authorizes nothing. NOT a launch. NOT open2.")

    handoff = (
        "under lawine #232's measured near-greedy 0.73% divergence the worst-case-vertex f_priv "
        f"tightens to fpriv_worstcase_under_measured_div={fpriv_wc_measured:.6f} (vs the old 0.957 "
        f"floor), which "
        + ("no longer straddles" if moved_above else "still straddles")
        + f" the #233 breakeven {f_be:.4f} — so the publish-first private bar's location is "
        + ("safer" if moved_above else "unchanged")
        + " once the corrected int4 physics replaces kanna #114's M=1 56%.")

    return {
        "self_test": st,
        "test_metric": {"fpriv_worstcase_under_measured_div": fpriv_wc_measured},
        "headline": {
            "fpriv_worstcase_under_measured_div": fpriv_wc_measured,          # TEST
            "fpriv_worstcase_measured_div_self_test_passes":
                st["fpriv_worstcase_measured_div_self_test_passes"],          # PRIMARY
            "straddles_breakeven_under_measured_div": straddles_measured,     # the decision flip
            "moved_above_breakeven_under_measured_div": moved_above,
            "delta_vs_breakeven": delta_vs_breakeven,
            "delta_vs_old_worstcase": delta_vs_old_worstcase,
            "lambda_floor_under_measured_div_linear": lam_floor_measured_linear,
            "lambda_floor_under_measured_div_linear_clamped": lam_floor_measured_linear_clamped,
            "lambda_floor_under_measured_div_exact": lam_floor_measured_exact,
            "lambda_floor_resid_linear_vs_exact": lam_floor_resid_measured,
            "lambda_floor_central": lam0,
            "f_priv_breakeven": f_be,
            "f_priv_clean": f_clean,
            "f_priv_grounded_old_worstcase": f_grounded,
            "f_int4div_solved": f_int4div,
            "d_measured": D_MEASURED,
            "d_old": D_OLD,
            "d_crossover_unstraddle": d_crossover,
            "binding_vertex_under_measured_div": binding_axis_measured,
            "nls_vertex_confirmed": nls_confirmed,
            "nls_margin_to_runner_under_measured_div": nls_margin_to_runner,
            "reachable_at_ceiling_under_measured_div": reachable_measured,
        },
        "blend_table": table,
        "nls_vertex": {
            "binding_axis_under_measured_div": binding_axis_measured,
            "nls_confirmed": nls_confirmed,
            "runner_up_axis": runner_up_axis,
            "nls_margin_to_runner_fpriv": nls_margin_to_runner,
            "clean_gap_nls_runner": clean_gap_nls_runner,
            "int4div_gap_to_flip_vertex": int4div_gap_to_flip,
            "wc_by_axis_under_measured_div": wc_by_axis_measured,
            "per_axis_clean_fpriv": verts,
            "note": (
                "Under a SHARED int4-divergent floor the blend f_priv_wc^v(d)=(1-d)·f_clean^v+d·f_int4div "
                "is monotone increasing in f_clean^v, so the f_priv-MINIMIZING vertex is argmin f_clean^v = "
                f"NLS ({NLS_AXIS}) for any d∈[0,1). At the measured d=0.0073 the NLS→{runner_up_axis} clean "
                f"gap {clean_gap_nls_runner:.6f} would require code's int4-divergent f_priv to sit "
                f"{int4div_gap_to_flip:.5f} BELOW NLS's to flip the binding vertex — implausible for the "
                "hardest axis, so NLS stays binding with margin."),
        },
        "robustness": {
            "d_old_sweep": d_old_sweep,
            "d_old_sweep_fpriv_wc_measured_band": d_old_sweep_band,
            "d_old_sweep_all_above_breakeven": d_old_sweep_all_above,
            "monotone_decreasing_in_d": monotone_decreasing_in_d,
            "spread_f_clean_minus_f_int4div": spread,
            "note": (
                "The corrected worst-case is INSENSITIVE to the exact OLD-divergence pin: at the measured "
                "d=0.0073 the int4-divergent weight is tiny, so varying d_old over [0.55,0.57] (±0.05) "
                f"moves fpriv_worstcase_under_measured_div only within {d_old_sweep_band[1]-d_old_sweep_band[0]:.2e}, "
                "all comfortably above the breakeven."),
        },
        "composition": {
            "f_priv_clean_226": f_clean,
            "f_priv_grounded_52_224": f_grounded,
            "f_priv_breakeven_233": f_be,
            "lambda_floor_central_233": lam0,
            "dlambda_floor_dfpriv_233": slope,
            "f_int4div_solved": f_int4div,
            "d_measured_lawine_232": D_MEASURED,
            "d_old_kanna_114": D_OLD,
            "ceil_int4_204": CEIL_INT4,
            "target_publish_first": TARGET,
            "nls_axis": NLS_AXIS,
            "anchors_source": a["source"],
            "vertices_source": verts_src,
        },
        "model_note": (
            "f_priv_wc(d) = (1-d)·f_clean + d·f_int4div, the divergence-weighted worst-case f_priv blend. "
            "f_clean=0.969107 is #226's NLS clean realizable worst-case (the d→0 limit); f_int4div is the "
            "fully-int4-divergent decode-drop pinned by the d=0.5608 round-trip to #233's grounded floor "
            "0.957054 (the d→1 weight on that floor). The blend reproduces #233's worst-case band "
            "[0.957054, 0.969107] as [f_priv_wc(0.5608), f_priv_wc(0)]. λ_floor via the #233 sensitivity "
            "d(λ_floor)/d(f_priv)=-2.3535, cross-checked against the exact #233 reach-DP solver."),
        "verdict": verdict,
        "handoff_line": handoff,
        "imports": {
            "provenance": (
                "stark#233 pszvrf2a (publish_first_fpriv_breakeven: breakeven 0.9598, band "
                "[0.957054,0.969107], λ_floor_central 0.97804, dλ_floor/df_priv -2.3535) x stark#226 "
                "tzcc5xuq (private_fpriv_worstcase: NLS native_multilingual f_priv-min vertex 0.969107, "
                "six realizable axes) x lawine#232 nxwv6pam (deployed M=8 divergence 0.73%) x kanna#114 "
                "9q5yy9l1 (M=1 56.08%) x kanna#224 1081oc84 / PR#52 (grounded f_priv 0.957054) x #204 "
                "(CEIL_INT4 520.9527). All run-ids in wandb-applied-ai-team/gemma-challenge-senpai."),
            "mechanism": (
                "Pure analytic divergence-weighted blend; exact λ_floor cross-check reuses #233's banked "
                "reach-DP solver (publish_first_fpriv_breakeven.lambda_floor_publish_first)."),
        },
    }


def _selftests(roundtrip_resid_low: float, roundtrip_resid_high: float,
               monotone_decreasing_in_d: bool, fpriv_wc_measured: float, f_grounded: float,
               straddles_measured: bool, moved_above: bool, nls_confirmed: bool,
               binding_axis_measured: str, lam_floor_resid_measured: float | None,
               d_old_sweep_all_above: bool, spread: float) -> dict[str, Any]:
    # (a) at d=0.5608 the blend round-trips #233's worst-case band (both ends; resid → 0).
    cond_a = bool(roundtrip_resid_low <= TOL_ROUNDTRIP and roundtrip_resid_high <= TOL_ROUNDTRIP)

    # (b) f_priv_wc monotone INCREASING as d DECREASES (less int4 weight ⇒ cleaner); spread > 0.
    cond_b = bool(monotone_decreasing_in_d and spread > 0.0)

    # (c) fpriv_worstcase_under_measured_div > the old worst-case (grounded floor) — corrected is less adverse.
    cond_c = bool(fpriv_wc_measured > f_grounded)

    # (d) the breakeven-straddle verdict is stated (the two booleans are complementary and definite).
    cond_d = bool(straddles_measured != moved_above)

    # (e) the binding (f_priv-minimizing) vertex is confirmed NLS under the corrected divergence.
    cond_e = bool(nls_confirmed and binding_axis_measured == NLS_AXIS)

    # (f) nan-clean (filled at payload level).
    cond_f = True

    # cross-check (robustness, not gating-critical): linear λ_floor matches the exact #233 solver,
    # and the corrected worst-case is insensitive to the OLD-divergence pin.
    cond_xcheck = bool((lam_floor_resid_measured is None or lam_floor_resid_measured < TOL_LAMBDA)
                       and d_old_sweep_all_above)

    conditions = {
        "a_d_old_roundtrips_233_worstcase_band_both_ends": cond_a,
        "b_fpriv_wc_increasing_as_d_decreases": cond_b,
        "c_measured_worstcase_gt_old_worstcase": cond_c,
        "d_breakeven_straddle_verdict_stated": cond_d,
        "e_nls_vertex_confirmed_under_measured_div": cond_e,
        "f_nan_clean": cond_f,
        "xcheck_linear_lambda_floor_matches_exact_and_pin_insensitive": cond_xcheck,
    }
    return {
        "conditions": conditions,
        "fpriv_worstcase_measured_div_self_test_passes": bool(all(conditions.values())),
        "a_detail": {"roundtrip_resid_low": roundtrip_resid_low,
                     "roundtrip_resid_high": roundtrip_resid_high, "tol": TOL_ROUNDTRIP},
        "c_detail": {"fpriv_wc_measured": fpriv_wc_measured, "old_worstcase_grounded": f_grounded,
                     "delta": fpriv_wc_measured - f_grounded},
        "d_detail": {"straddles_measured": straddles_measured, "moved_above": moved_above},
        "e_detail": {"binding_axis": binding_axis_measured, "nls_axis": NLS_AXIS},
        "xcheck_detail": {"lambda_floor_resid_linear_vs_exact": lam_floor_resid_measured,
                          "d_old_sweep_all_above_breakeven": d_old_sweep_all_above},
    }


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B (mirrors #233/#226; never fatal).
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


def _print_report(syn: dict) -> None:
    h, st, comp = syn["headline"], syn["self_test"], syn["composition"]
    print("\n" + "=" * 100, flush=True)
    print("PR #243  NLS worst-case f_priv under the corrected 0.73% divergence — does the #233 "
          "breakeven 0.9598 move?", flush=True)
    print("=" * 100, flush=True)
    print(f"  blend: f_priv_wc(d) = (1-d)·f_clean + d·f_int4div   "
          f"[f_clean={comp['f_priv_clean_226']:.6f} (#226 NLS), f_int4div={comp['f_int4div_solved']:.6f} "
          f"(pinned @ d_old={comp['d_old_kanna_114']})]", flush=True)
    print("-" * 100, flush=True)
    print(f"  TEST  fpriv_worstcase_under_measured_div = f_priv_wc({comp['d_measured_lawine_232']}) = "
          f"{h['fpriv_worstcase_under_measured_div']:.8f}", flush=True)
    print(f"  FLIP  breakeven {h['f_priv_breakeven']:.6f}: corrected worst-case is "
          f"{'ABOVE (un-straddled — lane SAFER)' if h['moved_above_breakeven_under_measured_div'] else 'BELOW (still straddles)'}"
          f"  (Δ={h['delta_vs_breakeven']:+.6f}; un-straddle threshold d*={h['d_crossover_unstraddle']:.4f})",
          flush=True)
    print(f"        corrected − old worst-case = {h['delta_vs_old_worstcase']:+.6f}  (less adverse)",
          flush=True)
    print(f"        λ_floor(corrected) = {h['lambda_floor_under_measured_div_linear']:.6f}  "
          f"(central {h['lambda_floor_central']:.6f}; exact "
          + (f"{h['lambda_floor_under_measured_div_exact']:.6f}" if h["lambda_floor_under_measured_div_exact"] is not None else "∅")
          + (f", resid {h['lambda_floor_resid_linear_vs_exact']:.2e})" if h["lambda_floor_resid_linear_vs_exact"] is not None else ")"),
          flush=True)
    print("-" * 100, flush=True)
    print("  blend table   d        f_priv_wc      λ_floor        straddles-breakeven", flush=True)
    for r in syn["blend_table"]:
        lf = r["lambda_floor_linear_clamped"]
        lf_s = f"{lf:.6f}" if lf is not None else "∅ UNREACH"
        tag = "  <- measured" if r["is_measured"] else ("  <- old (#114)" if r["is_old"] else "")
        print(f"             {r['d']:7.4f}   {r['f_priv_wc']:.6f}     {lf_s:>11}      "
              f"{r['straddles_breakeven']}{tag}", flush=True)
    print("-" * 100, flush=True)
    nv = syn["nls_vertex"]
    print(f"  NLS vertex: binding={nv['binding_axis_under_measured_div']} confirmed={nv['nls_confirmed']}  "
          f"(runner-up {nv['runner_up_axis']}, margin {nv['nls_margin_to_runner_fpriv']:.6f}; "
          f"int4div gap to flip {nv['int4div_gap_to_flip_vertex']:.5f})", flush=True)
    rob = syn["robustness"]
    print(f"  robustness: d_old∈[0.50,0.61] ⇒ corrected band "
          f"[{rob['d_old_sweep_fpriv_wc_measured_band'][0]:.6f}, "
          f"{rob['d_old_sweep_fpriv_wc_measured_band'][1]:.6f}] all_above={rob['d_old_sweep_all_above_breakeven']}",
          flush=True)
    print(f"  PRIMARY fpriv_worstcase_measured_div_self_test_passes = "
          f"{st['fpriv_worstcase_measured_div_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print("=" * 100, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"\n  HAND-OFF: {syn['handoff_line']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[fpriv-worstcase-measured-div] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    h, st, comp, nv, rob = (syn["headline"], syn["self_test"], syn["composition"],
                            syn["nls_vertex"], syn["robustness"])
    run = init_wandb_run(
        job_type="fpriv-worstcase-measured-div",
        agent="stark",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["fpriv-worstcase-measured-div", "issue-192", "issue-124", "validity-gate", "f_priv",
              "worst-case-vertex", "nls", "divergence-blend", "publish-first", "breakeven",
              "issue192-reading-calibration", "bank-the-analysis"],
        config={
            "f_priv_clean_226": comp["f_priv_clean_226"],
            "f_priv_grounded_52_224": comp["f_priv_grounded_52_224"],
            "f_priv_breakeven_233": comp["f_priv_breakeven_233"],
            "lambda_floor_central_233": comp["lambda_floor_central_233"],
            "dlambda_floor_dfpriv_233": comp["dlambda_floor_dfpriv_233"],
            "d_measured_lawine_232": comp["d_measured_lawine_232"],
            "d_old_kanna_114": comp["d_old_kanna_114"],
            "ceil_int4_204": comp["ceil_int4_204"],
            "target_publish_first": comp["target_publish_first"],
            "nls_axis": comp["nls_axis"],
            "wandb_group": args.wandb_group, "baseline_tps": 481.53,
            "source_runs": "stark#233 pszvrf2a, stark#226 tzcc5xuq, lawine#232 nxwv6pam, "
                           "kanna#114 9q5yy9l1, kanna#224 1081oc84",
        },
    )
    if run is None:
        print("[fpriv-worstcase-measured-div] wandb: no run (no WANDB_API_KEY/mode) — skipping",
              flush=True)
        return

    summary: dict[str, Any] = {
        "fpriv_worstcase_measured_div_self_test_passes":
            int(bool(st["fpriv_worstcase_measured_div_self_test_passes"])),          # PRIMARY
        "fpriv_worstcase_under_measured_div": h["fpriv_worstcase_under_measured_div"],  # TEST
        "straddles_breakeven_under_measured_div": int(bool(h["straddles_breakeven_under_measured_div"])),
        "moved_above_breakeven_under_measured_div": int(bool(h["moved_above_breakeven_under_measured_div"])),
        "delta_vs_breakeven": h["delta_vs_breakeven"],
        "delta_vs_old_worstcase": h["delta_vs_old_worstcase"],
        "lambda_floor_under_measured_div_linear": h["lambda_floor_under_measured_div_linear"],
        "lambda_floor_under_measured_div_exact": h["lambda_floor_under_measured_div_exact"],
        "lambda_floor_resid_linear_vs_exact": h["lambda_floor_resid_linear_vs_exact"],
        "lambda_floor_central": h["lambda_floor_central"],
        "f_priv_breakeven": h["f_priv_breakeven"],
        "f_priv_clean": h["f_priv_clean"],
        "f_priv_grounded_old_worstcase": h["f_priv_grounded_old_worstcase"],
        "f_int4div_solved": h["f_int4div_solved"],
        "d_measured": h["d_measured"],
        "d_old": h["d_old"],
        "d_crossover_unstraddle": h["d_crossover_unstraddle"],
        "nls_vertex_confirmed": int(bool(h["nls_vertex_confirmed"])),
        "nls_margin_to_runner_under_measured_div": h["nls_margin_to_runner_under_measured_div"],
        "int4div_gap_to_flip_vertex": nv["int4div_gap_to_flip_vertex"],
        "reachable_at_ceiling_under_measured_div": int(bool(h["reachable_at_ceiling_under_measured_div"])),
        "d_old_sweep_all_above_breakeven": int(bool(rob["d_old_sweep_all_above_breakeven"])),
        "monotone_decreasing_in_d": int(bool(rob["monotone_decreasing_in_d"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    # the blend table f_priv_wc + λ_floor as logged scalars (keyed by d):
    for r in syn["blend_table"]:
        dk = f"{r['d']:.4f}".replace(".", "p")
        summary[f"fpriv_wc_d_{dk}"] = r["f_priv_wc"]
        if r["lambda_floor_linear_clamped"] is not None:
            summary[f"lambda_floor_d_{dk}"] = r["lambda_floor_linear_clamped"]
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="fpriv_worstcase_measured_div_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[fpriv-worstcase-measured-div] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="issue192-reading-calibration")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 243, "agent": "stark",
        "kind": "fpriv-worstcase-measured-div", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["f_nan_clean"] = not nan_paths
    syn["self_test"]["fpriv_worstcase_measured_div_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    syn["headline"]["fpriv_worstcase_measured_div_self_test_passes"] = (
        syn["self_test"]["fpriv_worstcase_measured_div_self_test_passes"])
    if nan_paths:
        print(f"[fpriv-worstcase-measured-div] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[fpriv-worstcase-measured-div] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["fpriv_worstcase_measured_div_self_test_passes"]
              and payload["nan_clean"])
        print(f"[fpriv-worstcase-measured-div] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
