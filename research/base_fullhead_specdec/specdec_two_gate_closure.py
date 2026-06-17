#!/usr/bin/env python3
"""Spec-dec two-gate CLOSURE (PR #583).

Reconcile the four banked base_fullhead spec-dec legs into a single definitive
verdict on the ONE axis the per-token decode ceiling does not bound:

  GATE 1 (SPEED):    served TPS > 375.857 (official ship)
  GATE 2 (IDENTITY): #319 strict byte-exact greedy-token-identity vs no-spec
                     base_fullhead.

This is a CPU-only reconciliation. We do NOT re-measure the legs; we ingest their
banked numbers and resolve the one open flag `any_drafter_at_k_clears_ship`
ANALYTICALLY from the measured cost-acceptance structure.

LOCAL only: analysis_only=true, official_tps=0, NO HF Job, NO /v1/jobs:run,
NO --launch, NO submission, NO served-file change. NO FIRE.

The crux
--------
A ship-clearing drafter needs BOTH cheap verify AND high acceptance. We have two
measured Pareto corners that bracket the achievable plane:

  * ngram  (#573 tkapaz90): cheap effective cost (free draft), LOW acceptance.
  * MTP K=7 (#572 wndiyzxk): expensive effective cost (full draft head), HIGH
    acceptance.

The break-even acceptance to clear a TPS bar B is, frame-invariantly,
    A_breakeven(B) = c * (B / anchor)          [the A/c speedup model]
                   = A_achieved * (B / TPS_achieved)
where c = t_v / t_1 is the same-pod verify-step cost ratio (hardware-portable)
and TPS_achieved is that leg's projected served TPS. Equivalently a config
clears B iff its speedup A/c >= B/anchor.

Both measured corners sit BELOW the break-even line A_ship(c) = c * ship/anchor,
and because the break-even line rises faster (slope = ship/anchor = 1.437) than
the achievable acceptance envelope (slope ~0.80 between the corners, and
saturating beyond per #575/#577), the gap only widens -> the achievable envelope
lies entirely below A_ship(c). #575 separately pins the VERIFY-cost floor as
cheap, proving the bottleneck is the draft-cost<->quality tradeoff, not verify.

So `any_drafter_at_k_clears_ship = False` on the achievable Pareto. The single
named escalation is a hypothetical cheap-verify (ngram-cost) drafter with
acceptance >= A_ship_at_ngram_cost (~2.68) -- a +17% acceptance lift over ngram
at equal verify cost -- and even THAT is IDENTITY-blocked.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # .../target (holds scripts/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
# Banked canonical inputs (provenance: the four legs named in PR #583).        #
# These are RECONCILED, not re-measured.                                       #
# --------------------------------------------------------------------------- #
SHIP_TPS = 375.857          # official ship
FLOOR_TPS = 311.25          # magically-free capstone floor (lawine #554)
GATE_TPS = 500.0            # clear-gate
# base_fullhead no-spec anchors (same config, two frames):
ANCHOR_OFFICIAL = 261.59369723354945  # this-pod no-spec scaled to official frame (#573)
ANCHOR_WIRBEL = 252.69                # advisor directly-measured no-spec (wirbel #553)
ANCHOR = ANCHOR_OFFICIAL              # headline frame (the one A_ship=2.6806 uses)
SHIP_SPEEDUP_REQUIRED = SHIP_TPS / ANCHOR       # 1.43680
FLOOR_SPEEDUP_REQUIRED = FLOOR_TPS / ANCHOR
GATE_SPEEDUP_REQUIRED = GATE_TPS / ANCHOR

# --- #573 ngram corner (tkapaz90) -- cheap-draft / low-acceptance ----------- #
NGRAM = {
    "run_id": "tkapaz90",
    "label": "ngram K=7",
    "acceptance": 2.286525040263824,          # offline-sim e_accept (exact verify)
    "c_overhead": 1.865677364279604,          # t_v/t_1, 100%-coverage effective cost
    "proj_tps": 285.0913706631914,            # anchor_official * blended on-pod speedup
    "speedup_blended": 1.089825074832225,     # measured (coverage-included) speedup over ref
    "coverage": 0.27886733537224373,          # fraction of decode positions ngram drafts
    "A_ship_banked": 2.6805993589363384,      # c_overhead * ship_speedup_required (banked)
    "identity_seq_exact": 0.1640625,          # served greedy seq-byte-exact vs no-spec base
    "identity_per_step": 0.4656524658203125,  # served per-step token identity
}
# ngram K-sweep (cheap-end frontier: lower K -> lower acceptance -> lower speedup)
NGRAM_KSWEEP = [
    {"k": 3, "acceptance": 2.0084214781378424, "c_overhead": 1.742346146010073},
    {"k": 7, "acceptance": 2.286525040263824, "c_overhead": 1.865677364279604},
]

# --- #572 MTP K=7 corner (wndiyzxk) -- expensive-draft / high-acceptance ----- #
MTP = {
    "run_id": "wndiyzxk",
    "label": "MTP K=7",
    "acceptance": 3.8442646633920834,         # e_accept_exact
    "proj_tps": 262.9360241121258,            # official_projected_tps
    "num_speculative_tokens": 7,
    "identity_seq_exact": 0.15625,            # served greedy seq-byte-exact vs no-spec base
    "identity_per_step": 0.4776611328125,     # served per-step argmax identity
}

# --- #575 verify-cost asymptote (qgyqilcm) -- pins the verify-cost FLOOR ----- #
VERIFY = {
    "run_id": "qgyqilcm",
    "true_C1_ms": 11.496036290062799,         # M=1 (~no-spec) step cost
    "verify_cost_k7_ms": 13.13689941411235,   # C(M=8): verify forward over K=7+1 positions
    "c_compute_ms_per_pos": 0.08462453266874165,
    "implied_mtp_mean_acceptance": 2.9495050404810774,
    "specdec_ceiling_exceeds_ship_unphysical": True,  # 1/C(inf) bound IF acceptance kept up
    "fit_r2": 0.9272188783994963,
}

# --- #577 worst-case dispersion (q5631wt0) -- best workload still misses ----- #
DISPERSION = {
    "run_id": "q5631wt0",
    "mtp_strata": {  # net official-frame TPS by workload difficulty (MTP, the strong drafter)
        "easy": {"accept_rate": 0.25662668203288586, "mean_accept_len": 2.796386774230201, "net_tps": 185.12136658366114},
        "mix":  {"accept_rate": 0.40038377555746707, "mean_accept_len": 3.8026864289022697, "net_tps": 250.30245560549952},
        "hard": {"accept_rate": 0.5092937785886148, "mean_accept_len": 4.565056450120304, "net_tps": 309.9261922170867},
    },
    "self_det": 0.875,
}

# --- identity precedent: fern #566 candidate-verify (the BEST spec-dec id) --- #
CANDIDATE_VERIFY = {
    "run_id": "ix0oap4a",
    "per_step": 0.994,     # K=1 candidate-verify per-step identity (best spec-dec)
    "seq_exact": 0.14,     # still NOT byte-exact
}


def _breakeven_acceptance(achieved_acc: float, bar_tps: float, achieved_tps: float) -> float:
    """Frame-invariant break-even: acceptance needed to reach bar_tps at this
    config's verify cost. TPS scales linearly with acceptance (speedup=A/c, c
    fixed), so A_be = achieved_acc * bar/achieved_tps."""
    return achieved_acc * bar_tps / achieved_tps


def _effective_cost(achieved_acc: float, achieved_tps: float, anchor: float) -> float:
    """c = A / speedup = A / (proj/anchor) = A*anchor/proj (100%-coverage effective
    verify cost in no-spec-token units)."""
    return achieved_acc * anchor / achieved_tps


def build_synthesis() -> dict:
    # ---- effective costs (100%-coverage A/c frame) ------------------------- #
    ngram_c = NGRAM["c_overhead"]
    mtp_c = _effective_cost(MTP["acceptance"], MTP["proj_tps"], ANCHOR)

    # ---- break-even acceptances at each corner's cost ---------------------- #
    A_ship_at_ngram_cost = ngram_c * SHIP_SPEEDUP_REQUIRED            # banked ~2.6806
    A_ship_at_mtp_cost = mtp_c * SHIP_SPEEDUP_REQUIRED                # ~5.495
    mtp_breakeven = _breakeven_acceptance(MTP["acceptance"], SHIP_TPS, MTP["proj_tps"])

    # ---- speedups (A/c) and shortfalls ------------------------------------- #
    ngram_speedup_ideal = NGRAM["acceptance"] / ngram_c              # 100%-coverage
    ngram_speedup_real = NGRAM["proj_tps"] / ANCHOR                  # coverage-included
    mtp_speedup = MTP["proj_tps"] / ANCHOR
    max_measured_speedup = max(ngram_speedup_ideal, ngram_speedup_real, mtp_speedup)

    # ---- envelope geometry: line slope vs achievable-envelope slope -------- #
    # break-even line  A_ship(c) = SHIP_SPEEDUP_REQUIRED * c  (slope 1.4368)
    line_slope = SHIP_SPEEDUP_REQUIRED
    # achievable envelope chord through the two measured corners:
    env_slope = (MTP["acceptance"] - NGRAM["acceptance"]) / (mtp_c - ngram_c)
    env_intercept = NGRAM["acceptance"] - env_slope * ngram_c
    # the line outruns the envelope iff line_slope > env_slope (gap widens with c)
    line_outruns_envelope = line_slope > env_slope
    # naive (unphysical) linear-extrapolation crossing on the cheap side:
    #   line_slope*c = env_slope*c + env_intercept  ->  c* = env_intercept/(line_slope-env_slope)
    if abs(line_slope - env_slope) > 1e-9:
        c_cross_naive = env_intercept / (line_slope - env_slope)
        a_cross_naive = line_slope * c_cross_naive
    else:
        c_cross_naive = float("nan")
        a_cross_naive = float("nan")
    # physical: ngram is the free-draft cheap floor; lowering K LOWERS speedup
    # (K=3 speedup < K=7 speedup), so no cheaper ngram config clears, and the
    # naive crossing at c*~1.25 requires acceptance the measured trend never
    # reaches at that cost -> unphysical.
    ngram_k3 = NGRAM_KSWEEP[0]
    ngram_k3_speedup = ngram_k3["acceptance"] / ngram_k3["c_overhead"]
    cheap_end_monotone_down = ngram_k3_speedup < ngram_speedup_ideal  # lower K -> lower speedup
    crossing_is_physical = (not cheap_end_monotone_down)  # False: cheap end only gets worse

    # ---- GATE 1: does any ACHIEVABLE drafter clear ship? ------------------- #
    ngram_clears = ngram_speedup_real >= SHIP_SPEEDUP_REQUIRED
    mtp_clears = mtp_speedup >= SHIP_SPEEDUP_REQUIRED
    worstcase_best_stratum_tps = DISPERSION["mtp_strata"]["hard"]["net_tps"]
    worstcase_clears = worstcase_best_stratum_tps >= SHIP_TPS
    # achievable envelope lies entirely below A_ship(c): both corners below AND
    # line outruns envelope AND cheap-end is monotone-down (no physical crossing)
    envelope_below_everywhere = (
        (NGRAM["acceptance"] < A_ship_at_ngram_cost)
        and (MTP["acceptance"] < A_ship_at_mtp_cost)
        and line_outruns_envelope
        and (not crossing_is_physical)
    )
    any_drafter_at_k_clears_ship = not envelope_below_everywhere  # -> False
    specdec_speed_clears_ship = bool(ngram_clears or mtp_clears or worstcase_clears)  # -> False

    # the single named escalation candidate (hypothetical, off the measured frontier)
    clearing_drafter_required_acceptance_at_ngram_cost = A_ship_at_ngram_cost
    escalation_acceptance_lift_over_ngram = (
        clearing_drafter_required_acceptance_at_ngram_cost - NGRAM["acceptance"]
    )
    escalation_acceptance_lift_pct = 100.0 * escalation_acceptance_lift_over_ngram / NGRAM["acceptance"]

    # ---- GATE 2: identity ------------------------------------------------- #
    # #319 strict byte-exact requires per-step == 1.0 AND seq == 1.0.
    ngram_byte_exact = (NGRAM["identity_per_step"] >= 1.0 and NGRAM["identity_seq_exact"] >= 1.0)
    mtp_byte_exact = (MTP["identity_per_step"] >= 1.0 and MTP["identity_seq_exact"] >= 1.0)
    best_specdec_identity_per_step = CANDIDATE_VERIFY["per_step"]  # 0.994, still < 1.0
    specdec_identity_fire_eligible = bool(ngram_byte_exact or mtp_byte_exact
                                          or best_specdec_identity_per_step >= 1.0)  # -> False

    # ---- THE VERDICT ------------------------------------------------------ #
    specdec_two_gate_closed = bool(
        (not any_drafter_at_k_clears_ship) and (not specdec_identity_fire_eligible)
    )

    syn = {
        # frame / anchors
        "analysis_only": True,
        "official_tps": 0,
        "ship_tps": SHIP_TPS,
        "capstone_floor_tps": FLOOR_TPS,
        "gate_tps": GATE_TPS,
        "anchor_base_fullhead_nospec_official": ANCHOR_OFFICIAL,
        "anchor_base_fullhead_nospec_wirbel553": ANCHOR_WIRBEL,
        "ship_speedup_required": SHIP_SPEEDUP_REQUIRED,
        "floor_speedup_required": FLOOR_SPEEDUP_REQUIRED,
        "gate_speedup_required": GATE_SPEEDUP_REQUIRED,
        # ngram corner
        "ngram_run_id": NGRAM["run_id"],
        "ngram_acceptance": NGRAM["acceptance"],
        "ngram_effective_cost": ngram_c,
        "ngram_proj_tps": NGRAM["proj_tps"],
        "ngram_speedup_ideal_100pct_cov": ngram_speedup_ideal,
        "ngram_speedup_real_coverage": ngram_speedup_real,
        "ngram_coverage": NGRAM["coverage"],
        # MTP corner
        "mtp_run_id": MTP["run_id"],
        "mtp_acceptance": MTP["acceptance"],
        "mtp_effective_cost": mtp_c,
        "mtp_proj_tps": MTP["proj_tps"],
        "mtp_speedup": mtp_speedup,
        # break-evens
        "A_ship_at_ngram_cost": A_ship_at_ngram_cost,
        "A_ship_at_mtp_cost": A_ship_at_mtp_cost,
        "mtp_k7_break_even_acceptance": mtp_breakeven,
        # envelope geometry
        "breakeven_line_slope": line_slope,
        "achievable_envelope_slope": env_slope,
        "line_outruns_envelope": line_outruns_envelope,
        "cheap_end_monotone_down": cheap_end_monotone_down,
        "naive_crossing_cost": c_cross_naive,
        "naive_crossing_acceptance": a_cross_naive,
        "naive_crossing_is_physical": crossing_is_physical,
        "envelope_below_a_ship_everywhere": envelope_below_everywhere,
        "max_measured_speedup": max_measured_speedup,
        "speedup_shortfall_vs_ship": SHIP_SPEEDUP_REQUIRED - max_measured_speedup,
        # #575 verify floor
        "verify_cost_k7_ms": VERIFY["verify_cost_k7_ms"],
        "true_C1_ms": VERIFY["true_C1_ms"],
        "verify_only_cost_ratio": VERIFY["verify_cost_k7_ms"] / VERIFY["true_C1_ms"],
        "implied_mtp_mean_acceptance_575": VERIFY["implied_mtp_mean_acceptance"],
        # #577 dispersion
        "dispersion_worstcase_best_stratum_tps": worstcase_best_stratum_tps,
        "dispersion_worstcase_clears_ship": worstcase_clears,
        # identity
        "ngram_identity_per_step": NGRAM["identity_per_step"],
        "ngram_identity_seq_exact": NGRAM["identity_seq_exact"],
        "mtp_identity_per_step": MTP["identity_per_step"],
        "mtp_identity_seq_exact": MTP["identity_seq_exact"],
        "best_specdec_identity_per_step_candidate_verify": best_specdec_identity_per_step,
        # ----- KEY OUTPUTS (PR #583) ----- #
        "any_drafter_at_k_clears_ship": any_drafter_at_k_clears_ship,
        "clearing_drafter_required_acceptance_at_ngram_cost": clearing_drafter_required_acceptance_at_ngram_cost,
        "escalation_acceptance_lift_over_ngram": escalation_acceptance_lift_over_ngram,
        "escalation_acceptance_lift_pct": escalation_acceptance_lift_pct,
        "ngram_acceptance_gap": A_ship_at_ngram_cost - NGRAM["acceptance"],
        "mtp_k7_break_even_acceptance_out": mtp_breakeven,
        "specdec_identity_fire_eligible": specdec_identity_fire_eligible,
        "specdec_speed_clears_ship": specdec_speed_clears_ship,
        "specdec_two_gate_closed": specdec_two_gate_closed,
    }
    return syn


def make_figure(syn: dict, out_png: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        print(f"[closure] matplotlib unavailable ({exc}); skipping figure.", flush=True)
        return False

    ngram_c = syn["ngram_effective_cost"]
    mtp_c = syn["mtp_effective_cost"]
    env_slope = syn["achievable_envelope_slope"]
    env_b = syn["ngram_acceptance"] - env_slope * ngram_c
    cs = np.linspace(1.0, max(mtp_c, 4.5) * 1.05, 200)
    a_ship = syn["ship_speedup_required"] * cs
    a_floor = syn["floor_speedup_required"] * cs

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    # break-even lines
    ax.plot(cs, a_ship, "-", color="#c0392b", lw=2.2,
            label=f"A_ship(c) = {syn['ship_speedup_required']:.3f}·c  (clear 375.857)")
    ax.plot(cs, a_floor, "--", color="#e67e22", lw=1.3,
            label=f"A_floor(c)  (clear {syn['capstone_floor_tps']:.1f})")
    # achievable envelope: SOLID-dotted only over the measured range [ngram, mtp];
    # faint outside (extrapolation). Beyond mtp it SATURATES (acceptance-limited)
    # so the true envelope is even lower than the chord -> draw a saturating cap.
    cs_meas = np.linspace(ngram_c, mtp_c, 60)
    ax.plot(cs_meas, env_slope * cs_meas + env_b, ":", color="#2c3e50", lw=2.0,
            label=f"achievable envelope (measured, slope {env_slope:.3f})")
    cs_lo = np.linspace(1.0, ngram_c, 40)
    cs_hi = np.linspace(mtp_c, cs[-1], 40)
    ax.plot(cs_lo, env_slope * cs_lo + env_b, ":", color="#95a5a6", lw=1.1, alpha=0.8)
    ax.plot(cs_hi, env_slope * cs_hi + env_b, ":", color="#95a5a6", lw=1.1, alpha=0.8,
            label="envelope extrapolation (unphysical: cheap end caps, rich end saturates)")
    # mark the naive (unphysical) cheap-side crossing
    cx, ay = syn["naive_crossing_cost"], syn["naive_crossing_acceptance"]
    if cx == cx and 1.0 <= cx <= cs[-1]:  # not NaN and in range
        ax.scatter([cx], [ay], s=60, facecolor="none", edgecolor="#7f8c8d", lw=1.4, zorder=4)
        ax.annotate("naive crossing — UNPHYSICAL\n(ngram is the free-draft floor;\nlower K → lower speedup)",
                    xy=(cx, ay), xytext=(1.05, 1.15), fontsize=7.2, color="#7f8c8d",
                    arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=0.9))
    # measured corners
    ax.scatter([ngram_c], [syn["ngram_acceptance"]], s=130, color="#2980b9",
               zorder=5, edgecolor="k",
               label=f"ngram K=7  (A={syn['ngram_acceptance']:.3f}, miss {syn['ngram_acceptance_gap']:.3f})")
    ax.scatter([mtp_c], [syn["mtp_acceptance"]], s=130, color="#8e44ad",
               marker="D", zorder=5, edgecolor="k",
               label=f"MTP K=7  (A={syn['mtp_acceptance']:.3f}, needs {syn['mtp_k7_break_even_acceptance']:.3f})")
    # break-even markers at each corner
    ax.scatter([ngram_c], [syn["A_ship_at_ngram_cost"]], s=70, color="#c0392b", marker="x", zorder=6)
    ax.scatter([mtp_c], [syn["A_ship_at_mtp_cost"]], s=70, color="#c0392b", marker="x", zorder=6)
    # the named escalation region (upper-left): cheap cost, acceptance >= A_ship_at_ngram_cost
    ax.annotate(
        f"escalation: cheap-verify drafter\nneeds A≥{syn['A_ship_at_ngram_cost']:.3f} at c≈{ngram_c:.2f}\n(+{syn['escalation_acceptance_lift_pct']:.0f}% over ngram) — identity-blocked",
        xy=(ngram_c, syn["A_ship_at_ngram_cost"]), xytext=(1.35, 4.6),
        fontsize=8, color="#7f8c8d",
        arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.0))
    # gap arrows
    for c, a_have, a_need, col in [
        (ngram_c, syn["ngram_acceptance"], syn["A_ship_at_ngram_cost"], "#2980b9"),
        (mtp_c, syn["mtp_acceptance"], syn["A_ship_at_mtp_cost"], "#8e44ad"),
    ]:
        ax.annotate("", xy=(c, a_need), xytext=(c, a_have),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.4, alpha=0.7))

    ax.set_xlabel("effective verify cost  c = t_v / t_1  (no-spec-token units)")
    ax.set_ylabel("mean acceptance length  A")
    ax.set_title("Spec-dec two-gate closure: achievable envelope lies BELOW A_ship(c)\n"
                 "GATE-1 SPEED closed — both measured corners miss; line outruns envelope")
    ax.legend(loc="upper left", fontsize=7.6, framealpha=0.92)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(1.0, max(mtp_c, 4.5) * 1.05)
    ax.set_ylim(0.8, max(syn["A_ship_at_mtp_cost"], 6.0) * 1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return True


def print_table(syn: dict) -> None:
    rows = [
        ("ship TPS / required speedup", f"{syn['ship_tps']:.3f} / {syn['ship_speedup_required']:.4f}"),
        ("anchor (official / wirbel)", f"{syn['anchor_base_fullhead_nospec_official']:.2f} / {syn['anchor_base_fullhead_nospec_wirbel553']:.2f}"),
        ("--- ngram corner (cheap draft) ---", ""),
        ("  acceptance / eff.cost c", f"{syn['ngram_acceptance']:.4f} / {syn['ngram_effective_cost']:.4f}"),
        ("  A_ship needed @ this cost", f"{syn['A_ship_at_ngram_cost']:.4f}"),
        ("  acceptance gap (miss by)", f"{syn['ngram_acceptance_gap']:.4f}"),
        ("  speedup ideal / real(cov)", f"{syn['ngram_speedup_ideal_100pct_cov']:.4f} / {syn['ngram_speedup_real_coverage']:.4f}"),
        ("--- MTP K=7 corner (rich draft) ---", ""),
        ("  acceptance / eff.cost c", f"{syn['mtp_acceptance']:.4f} / {syn['mtp_effective_cost']:.4f}"),
        ("  A_ship needed @ this cost", f"{syn['A_ship_at_mtp_cost']:.4f}"),
        ("  break-even acceptance", f"{syn['mtp_k7_break_even_acceptance']:.4f}"),
        ("  speedup", f"{syn['mtp_speedup']:.4f}"),
        ("--- #575 verify floor ---", ""),
        ("  verify-only cost ratio (M8/M1)", f"{syn['verify_only_cost_ratio']:.4f}"),
        ("--- #577 dispersion ---", ""),
        ("  best-workload(hard) MTP net TPS", f"{syn['dispersion_worstcase_best_stratum_tps']:.2f}  (< ship)"),
        ("--- envelope geometry ---", ""),
        ("  line slope / envelope slope", f"{syn['breakeven_line_slope']:.4f} / {syn['achievable_envelope_slope']:.4f}"),
        ("  line outruns envelope", str(syn["line_outruns_envelope"])),
        ("  max measured speedup", f"{syn['max_measured_speedup']:.4f}  (need {syn['ship_speedup_required']:.4f})"),
        ("--- identity (GATE 2) ---", ""),
        ("  ngram per-step / seq", f"{syn['ngram_identity_per_step']:.4f} / {syn['ngram_identity_seq_exact']:.4f}"),
        ("  MTP per-step / seq", f"{syn['mtp_identity_per_step']:.4f} / {syn['mtp_identity_seq_exact']:.4f}"),
        ("  best spec-dec id (cand-verify)", f"{syn['best_specdec_identity_per_step_candidate_verify']:.4f}  (< 1.0)"),
        ("=== KEY OUTPUTS ===", ""),
        ("any_drafter_at_k_clears_ship", str(syn["any_drafter_at_k_clears_ship"])),
        ("clearing_drafter_req_acc @ ngram cost", f"{syn['clearing_drafter_required_acceptance_at_ngram_cost']:.4f}"),
        ("ngram_acceptance_gap", f"{syn['ngram_acceptance_gap']:.4f}"),
        ("mtp_k7_break_even_acceptance", f"{syn['mtp_k7_break_even_acceptance']:.4f}"),
        ("specdec_identity_fire_eligible", str(syn["specdec_identity_fire_eligible"])),
        ("specdec_speed_clears_ship", str(syn["specdec_speed_clears_ship"])),
        ("specdec_two_gate_closed", str(syn["specdec_two_gate_closed"])),
    ]
    width = max(len(k) for k, _ in rows)
    print("\n" + "=" * 72)
    print("SPEC-DEC TWO-GATE CLOSURE  (PR #583, analysis_only)")
    print("=" * 72)
    for k, v in rows:
        print(f"{k.ljust(width)}  {v}")
    print("=" * 72 + "\n")


def nan_clean(obj):
    if isinstance(obj, dict):
        return {k: nan_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_clean(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def log_wandb(syn: dict, args, fig_path: Path | None) -> str | None:
    if args.no_wandb:
        return None
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_file_artifact, log_json_artifact,
                                           log_summary)
    except Exception as exc:
        print(f"[closure] wandb_logging import failed: {exc}; skipping W&B.", flush=True)
        return None
    run = init_wandb_run(
        job_type="analysis",
        agent="fern",
        name=args.wandb_name or "fern/specdec-two-gate-closure",
        group=args.wandb_group or "base-fullhead-specdec-ceiling",
        notes="PR #583: reconcile the four banked spec-dec legs into the "
              "specdec_two_gate_closed verdict (analysis_only, no fire).",
        tags=["specdec", "closure", "analysis_only", "base_fullhead"],
        config={
            "pr": 583, "analysis_only": True, "official_tps": 0,
            "legs": {"ngram_573": "tkapaz90", "mtp_572": "wndiyzxk",
                     "verifycost_575": "qgyqilcm", "dispersion_577": "q5631wt0"},
        },
    )
    if run is None:
        print("[closure] W&B not initialised (no key/mode); report saved locally.", flush=True)
        return None
    log_summary(run, nan_clean(syn), step=0, run_prefix="")
    log_json_artifact(run, name="specdec_two_gate_closure", artifact_type="analysis",
                      data=nan_clean(syn))
    if fig_path is not None and fig_path.exists():
        try:
            import wandb
            run.log({"specdec_two_gate_closure/figure": wandb.Image(str(fig_path)),
                     "global_step": 0})
        except Exception as exc:
            print(f"[closure] figure log failed: {exc}", flush=True)
        log_file_artifact(run, path=fig_path, name="specdec_two_gate_closure_fig",
                          artifact_type="figure")
    rid = run.id
    finish_wandb(run)
    return rid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "specdec_two_gate_closure.json")
    ap.add_argument("--fig", type=Path, default=HERE / "specdec_two_gate_closure.png")
    args = ap.parse_args(argv)

    syn = build_synthesis()
    print_table(syn)

    fig_ok = make_figure(syn, args.fig)
    report = {
        "pr": 583,
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "analysis_only": True,
        "official_tps": 0,
        "legs": {"ngram_573": NGRAM["run_id"], "mtp_572": MTP["run_id"],
                 "verifycost_575": VERIFY["run_id"], "dispersion_577": DISPERSION["run_id"]},
        "inputs": {"ngram": NGRAM, "ngram_ksweep": NGRAM_KSWEEP, "mtp": MTP,
                   "verify": VERIFY, "dispersion": DISPERSION,
                   "candidate_verify_identity": CANDIDATE_VERIFY},
        "synthesis": syn,
    }
    args.out.write_text(json.dumps(nan_clean(report), indent=2))
    print(f"[closure] wrote {args.out}")
    if fig_ok:
        print(f"[closure] wrote {args.fig}")

    rid = log_wandb(syn, args, args.fig if fig_ok else None)
    if rid:
        print(f"[closure] W&B run id: {rid}")

    # epitaph
    print("\nEPITAPH (spec-dec axis):")
    print("  The one lever the per-token decode ceiling did not bound is shut on BOTH gates.")
    print(f"  SPEED: the cheap-verify corner (ngram, A={syn['ngram_acceptance']:.3f}) misses the "
          f"{syn['A_ship_at_ngram_cost']:.3f} bar by {syn['ngram_acceptance_gap']:.3f}; the "
          f"high-acceptance corner (MTP, A={syn['mtp_acceptance']:.3f}) pays a verify cost that "
          f"lifts its bar to {syn['mtp_k7_break_even_acceptance']:.3f}.")
    print(f"  The achievable acceptance envelope (slope {syn['achievable_envelope_slope']:.2f}) "
          f"is outrun by the break-even line (slope {syn['breakeven_line_slope']:.2f}) -> no "
          f"achievable drafter clears 375.857.")
    print("  IDENTITY: every served spec-dec config is 15-16% seq-exact (best per-step 0.994, "
          "candidate-verify) — none is #319 byte-exact, because multi-position verify reorders "
          "bf16 ties.")
    print(f"  VERDICT specdec_two_gate_closed = {syn['specdec_two_gate_closed']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
