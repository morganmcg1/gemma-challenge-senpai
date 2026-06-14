"""Local-``wall_tps`` -> official-TPS projection calibration (PR #99).

Turns the team's *implicit ~constant* local->official multiplier into a **pinned
number with an honest CI**, and makes land #71's tree-verify build a **zero-lag
>=500 decision**: once the build lands, the #82 paired-A/B runner measures its
local ``wall_tps`` and this module maps it to a projected-official band in one
shot (no new HF launch, no re-derivation).

Why this is well-posed
----------------------
``wall_tps = num_completion_tokens / decode_duration_s`` is *definitionally* the
official ``output_throughput`` (PR #72/#82). So the multiplier is **not** a
metric-conversion factor -- it is a pure **hardware/environment transfer factor**
between the local AWS A10G (SM clock pinned 1710 MHz) and the HF-Jobs
``a10g-small`` instance (driver/clock/thermal/harness-warmup differences). A
transfer factor that depends only on the box -- not on the submission -- is the
load-bearing assumption that lets us project the tree from its local measurement.

The anchors (all COMMITTED in this repo; no new run needed)
----------------------------------------------------------
* OFFICIAL: PR #52 ``fa2sw_precache_kenyan`` (linear MTP K=7 + 3D split-KV),
  **481.53** official ``a10g-small`` TPS, private-VERIFIED 2026-06-13
  (460.85 private, dlt 4.3% <= 5%, PPL 2.3772/2.3777, 128/128). The ONLY
  private-verified official anchor on the spec frontier.
* LOCAL: the SAME deployed K=7 stack measured as ``wall_tps`` (median-of-N=3,
  the #72/#82 protocol) across independent sessions -- see
  ``LOCAL_DEPLOYED_SESSIONS``. CV across sessions ~0.02% (rock-stable
  denominator).

What the CI is (and is not)
---------------------------
* The LOCAL denominator is pinned to ~0.02% CV (many sessions, the #72 0.035%
  per-run floor averaged down) -> the multiplier's local-side CI is +/-0.03%.
* The OFFICIAL numerator is a **single anchor** (one public run of this exact
  submission). Its run-to-run variance is UNMEASURED -- this is the dominant,
  honestly-unquantified term. We therefore also report a **sensitivity envelope**
  parameterised by an assumed official per-run CV (``official_cv_assumed_pct``),
  so the >=500 decision can be shown robust to it rather than resting on a point.

Config-stability
----------------
Within the spec-frontier config family the multiplier *denominator* is provably
stable (K-sweep 5..9 and MBT-sweep 512..8192 all read the same 454.x wall_tps for
the deployed point; off-point configs move wall_tps by a measured, signed amount
but the deployed reference does not drift). Cross-*precision* invariance (int4 vs
bf16 rungs) is NOT cleanly testable from the repo: those older rungs were metered
with the 16-prompt ``local_prevalidate`` steady meter, not the 128-prompt
``wall_tps`` protocol, so their official/local ratios conflate config-drift with
meter-drift. The tree extends the SAME int4/split-KV/lm_head precision as the
anchor (it only widens the drafter M=8 -> M=32, a bandwidth-bound / flat-in-M
regime per #30/#85), so the precision axis is not the binding one for the tree.

CPU-only, no GPU, no network. Importable by ``paired_tps_ab.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.tps_noise_floor.analyze_noise_floor import (  # noqa: E402
    Z_DETECT,
    bootstrap_stat_cv,
)

# ---------------------------------------------------------------------------
# Committed anchors
# ---------------------------------------------------------------------------
# The official frontier anchor (PR #52). Numerator of the multiplier.
OFFICIAL_ANCHOR = {
    "tps": 481.53,
    "config": "fa2sw_precache_kenyan (linear MTP K=7 + 3D split-KV)",
    "job": "6a2dce05871c005b5352c0b9",
    "run_prefix": "results/senpai/fa2sw-precache-kenyan-20260613T213911Z",
    "date": "2026-06-13",
    "private_verified_tps": 460.85,
    "private_gap_pct": 4.3,
    "ppl": 2.3772,
    "completed": 128,
    "source": "BASELINE.md PR #52; private-VERIFIED cmpatino-verifier 2026-06-13 23:04Z",
}

# Local ``wall_tps`` of the SAME deployed K=7 stack. Each entry is one independent
# measurement *session*; ``median`` is the #72/#82 protocol metric (median-of-N),
# ``runs`` are the raw per-run wall_tps (for a structure-aware bootstrap).
LOCAL_DEPLOYED_SESSIONS = [
    {
        "label": "pr82_selfnull_baseline",
        "median": 454.08539709184896,
        "runs": [454.1376352465194, 454.0775516411879, 454.08539709184896],
        "source": "research/walltps_ab/selfnull/paired_ab.json arms.baseline (PR #82, N=3)",
    },
    {
        "label": "pr82_selfnull_candidate",
        "median": 454.22098715299325,
        "runs": [453.9614145812932, 454.22098715299325, 454.24516899032216],
        "source": "research/walltps_ab/selfnull/paired_ab.json arms.candidate (PR #82, N=3)",
    },
    {
        "label": "pr90_ksweep_baseline",
        "median": 454.33835724012107,
        "runs": [454.3336404857965, 454.3428803397955, 454.33835724012107],
        "source": "research/walltps_ab/mtp_k*/paired_ab.json arms.baseline (PR #90 K=7, N=3) -- LOCKED reference 454.338",
    },
]

# Corroborating sessions cited from BASELINE.md text (full per-run records not in a
# committed paired_ab.json). Used only as a cross-check, not in the primary fit.
LOCAL_DEPLOYED_COROBORATING = [
    {"label": "pr72_fresh_n12", "median": 454.12,
     "source": "BASELINE.md PR #82 entry: '#72's N=12 454.12'"},
    {"label": "pr43_wallclock", "median": 454.25,
     "source": "BASELINE.md PR #43 entry: tps_local_splitkv_wallclock=454.25"},
]

# The LOCKED linear-chain reference (PR #90). The harness self-check must reproduce
# this within MDE, and the multiplier must map it back to ~481.53 official.
LINEAR_REFERENCE_WALL_TPS = 454.33835724012107

# 500-TPS gate target (human-theykk; BASELINE.md "Next target: 500 TPS").
OFFICIAL_TARGET_TPS = 500.0

# Operative #72/#82 MDE bar for the median-of-N=3 wall_tps A/B (the linear-chain
# self-check tolerance: the harness "reproduces 454.338" iff within this band).
SELF_CHECK_MDE_PCT = 0.10

# ---------------------------------------------------------------------------
# land #71 analytical tree spec (Step 3, pre-build projection)
# ---------------------------------------------------------------------------
# Central net LOCAL gain of the M=32 max-branch-3 tree over the linear K=7 chain.
# wirbel #83 drafter-aware re-price (+18.2%); denken #85 net-after-overhead +19.82%
# gross / ~17.9% verify-side-refined; fern #92 E[T] independence-gap DE-RISKED.
TREE_SPEC = {
    "E_T": 5.207,
    "net_local_gain_pct": 18.2,           # wirbel #83 drafter-aware central
    "net_local_gain_range_pct": [17.9, 19.82],  # denken #85 refined .. gross
    "modeling_band_pct": 2.3,             # fern #92 (+/-2-3% -> published 558-581)
    "topology": "M=32 depth-9 max-branch-3 (wirbel #83)",
    "fern_published_official_band": [558.0, 581.0],
    "sources": "wirbel #83 (+18.2%), denken #85 (+19.82% gross/overhead audit), "
               "fern #92 (E[T]=5.208, independence-gap +0.025%, band 558-581)",
}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
@dataclass
class Calibration:
    multiplier: float
    local_wall_tps: float
    local_cv_pct: float
    official_tps: float
    # local-anchored (measured) CI on the multiplier
    mult_ci_local_lo: float
    mult_ci_local_hi: float
    mult_se_local: float
    # conservative envelope adding an ASSUMED official per-run CV
    official_cv_assumed_pct: float
    mult_ci_env_lo: float
    mult_ci_env_hi: float
    n_sessions: int
    n_runs: int
    per_session: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "local_to_official_multiplier": self.multiplier,
            "local_wall_tps_deployed": self.local_wall_tps,
            "local_cv_pct_across_runs": self.local_cv_pct,
            "official_tps_anchor": self.official_tps,
            "multiplier_se_local": self.mult_se_local,
            "multiplier_ci95_local": [self.mult_ci_local_lo, self.mult_ci_local_hi],
            "official_cv_assumed_pct": self.official_cv_assumed_pct,
            "multiplier_ci95_envelope": [self.mult_ci_env_lo, self.mult_ci_env_hi],
            "n_sessions": self.n_sessions,
            "n_runs_total": self.n_runs,
            "per_session_multiplier": self.per_session,
            "notes": self.notes,
        }


def _all_runs() -> list[float]:
    out: list[float] = []
    for s in LOCAL_DEPLOYED_SESSIONS:
        out.extend(float(v) for v in s.get("runs", []))
    return out


def calibrate(official_cv_assumed_pct: float = 1.0,
              reps: int = 8000) -> Calibration:
    """Pin the local->official multiplier with CI from the committed anchors.

    Local denominator = mean of all committed per-run ``wall_tps`` of the deployed
    K=7 stack. Local-side CI from a bootstrap SE of that mean. The official
    numerator is a single anchor; ``official_cv_assumed_pct`` injects an assumed
    official per-run CV to produce a *conservative sensitivity envelope* on top of
    the measured local-side CI (clearly labelled -- it is a sensitivity knob, not a
    measured number)."""
    runs = _all_runs()
    n_runs = len(runs)
    local_mean = statistics.fmean(runs)
    local_sd = statistics.stdev(runs) if n_runs > 1 else 0.0
    local_cv = 100.0 * local_sd / local_mean if local_mean else float("nan")
    official = float(OFFICIAL_ANCHOR["tps"])
    multiplier = official / local_mean

    # Local-side SE of the mean denominator -> SE of the multiplier (numerator fixed).
    boot = bootstrap_stat_cv(runs, n_runs, stat="mean", reps=reps)
    se_local_mean = boot["se"] if boot else (local_sd / math.sqrt(n_runs) if n_runs else 0.0)
    # d(mult)/d(local) = -official/local^2 ; |.|*se gives the multiplier SE.
    mult_se_local = official / (local_mean ** 2) * se_local_mean
    ci_local_lo = multiplier - Z_DETECT * mult_se_local
    ci_local_hi = multiplier + Z_DETECT * mult_se_local

    # Conservative envelope: add (in quadrature) an assumed official-numerator SE.
    off_se = official * official_cv_assumed_pct / 100.0  # single-sample SE under assumed CV
    mult_se_env = math.hypot(mult_se_local,
                             off_se / local_mean)  # d(mult)/d(official) = 1/local
    ci_env_lo = multiplier - Z_DETECT * mult_se_env
    ci_env_hi = multiplier + Z_DETECT * mult_se_env

    per_session = []
    for s in LOCAL_DEPLOYED_SESSIONS + LOCAL_DEPLOYED_COROBORATING:
        med = float(s["median"])
        per_session.append({
            "label": s["label"],
            "local_median_wall_tps": med,
            "multiplier": official / med,
            "source": s["source"],
            "primary_fit": s in LOCAL_DEPLOYED_SESSIONS,
        })

    mult_min = min(p["multiplier"] for p in per_session)
    mult_max = max(p["multiplier"] for p in per_session)
    notes = [
        f"Multiplier = {official:.2f} (official, single anchor PR#52) / "
        f"{local_mean:.3f} (local mean wall_tps, n_runs={n_runs} over "
        f"{len(LOCAL_DEPLOYED_SESSIONS)} committed sessions) = {multiplier:.5f}.",
        f"Per-session multiplier spans [{mult_min:.5f}, {mult_max:.5f}] "
        f"(incl. corroborating sessions) -- denominator is config-stable to "
        f"+/-{100.0*(mult_max-mult_min)/2/multiplier:.3f}%.",
        "Local-side CI is tight (many sessions). DOMINANT honest uncertainty is the "
        "single official anchor: its run-to-run variance is UNMEASURED; the envelope "
        f"assumes official per-run CV={official_cv_assumed_pct:.2f}% as a sensitivity knob.",
        "Config-stability holds within the spec-frontier family (K/MBT sweeps); "
        "cross-precision invariance is untested (older rungs used a different local "
        "meter) but is not the binding axis for the tree (same precision as anchor).",
    ]
    return Calibration(
        multiplier=multiplier, local_wall_tps=local_mean, local_cv_pct=local_cv,
        official_tps=official, mult_ci_local_lo=ci_local_lo, mult_ci_local_hi=ci_local_hi,
        mult_se_local=mult_se_local, official_cv_assumed_pct=official_cv_assumed_pct,
        mult_ci_env_lo=ci_env_lo, mult_ci_env_hi=ci_env_hi,
        n_sessions=len(LOCAL_DEPLOYED_SESSIONS), n_runs=n_runs,
        per_session=per_session, notes=notes,
    )


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def project_official(local_wall_tps: float, *, calib: Calibration | None = None,
                     modeling_band_pct: float = 0.0,
                     use_envelope: bool = True) -> dict[str, Any]:
    """Map a MEASURED local ``wall_tps`` to a projected-official band.

    ``modeling_band_pct`` is an OPTIONAL extra band (>=0) for the case where the
    local input is itself a projection (e.g. the analytical tree before the build
    lands). For a *measured* arm it is 0 -- the only band is the multiplier CI.

    The band combines (in quadrature) the multiplier relative CI with the modeling
    band. ``use_envelope`` selects the conservative official-CV envelope (default)
    vs the tight local-only CI."""
    if calib is None:
        calib = calibrate()
    central = local_wall_tps * calib.multiplier
    if use_envelope:
        rel_mult = Z_DETECT * (math.hypot(calib.mult_se_local,
                               (calib.official_cv_assumed_pct / 100.0) * calib.official_tps
                               / calib.local_wall_tps)) / calib.multiplier
    else:
        rel_mult = Z_DETECT * calib.mult_se_local / calib.multiplier
    rel_model = modeling_band_pct / 100.0
    rel = math.hypot(rel_mult, rel_model)
    lo = central * (1.0 - rel)
    hi = central * (1.0 + rel)
    return {
        "local_wall_tps": local_wall_tps,
        "multiplier": calib.multiplier,
        "projected_official": central,
        "projected_official_lo": lo,
        "projected_official_hi": hi,
        "band_rel_pct": 100.0 * rel,
        "band_from_multiplier_pct": 100.0 * rel_mult,
        "band_from_modeling_pct": 100.0 * rel_model,
        "clears_500": lo >= OFFICIAL_TARGET_TPS,
        "margin_to_500_pct_at_lo": 100.0 * (lo - OFFICIAL_TARGET_TPS) / OFFICIAL_TARGET_TPS,
        "margin_to_500_pct_at_central": 100.0 * (central - OFFICIAL_TARGET_TPS) / OFFICIAL_TARGET_TPS,
    }


def project_tree(calib: Calibration | None = None,
                 net_local_gain_pct: float | None = None,
                 modeling_band_pct: float | None = None) -> dict[str, Any]:
    """Step-3 analytical projection of land #71's tree (pre-build).

    Applies the net LOCAL gain to the LOCKED linear reference, then the multiplier.
    The band carries fern #92's modeling uncertainty AND the multiplier envelope."""
    if calib is None:
        calib = calibrate()
    gain = TREE_SPEC["net_local_gain_pct"] if net_local_gain_pct is None else net_local_gain_pct
    band = TREE_SPEC["modeling_band_pct"] if modeling_band_pct is None else modeling_band_pct
    local_tree = LINEAR_REFERENCE_WALL_TPS * (1.0 + gain / 100.0)
    proj = project_official(local_tree, calib=calib, modeling_band_pct=band, use_envelope=True)
    proj.update({
        "linear_reference_wall_tps": LINEAR_REFERENCE_WALL_TPS,
        "net_local_gain_pct": gain,
        "local_tree_wall_tps": local_tree,
        "E_T": TREE_SPEC["E_T"],
        "topology": TREE_SPEC["topology"],
        "fern_published_official_band": TREE_SPEC["fern_published_official_band"],
        "projected_official_clears_500_bool": proj["clears_500"],
    })
    return proj


def gate_verdict(calib: Calibration, tree_proj: dict[str, Any], *,
                 margin_green_pct: float = 5.0,
                 spread_amber_pct: float = 0.5,
                 spread_red_pct: float = 2.0) -> dict[str, Any]:
    """Step-3 GREEN/AMBER/RED gate on land #71's projected tree.

    * RED   -- multiplier UNSTABLE (per-session spread > spread_red_pct) OR the
               projected band straddles 500 (lo < 500 < hi, decision inconclusive)
               OR the band is wholly below 500 (clear fail).
    * AMBER -- config-DRIFT (spread in [spread_amber_pct, spread_red_pct]) OR the
               low band edge clears 500 by less than margin_green_pct (thin margin).
    * GREEN -- multiplier config-stable (spread < spread_amber_pct) AND the band
               clears 500 with >= margin_green_pct at the low edge.

    The band read here is the CONSERVATIVE envelope band (tree_proj already folds in
    the assumed-official-CV multiplier envelope + fern #92 modeling band), so GREEN
    is robust to the single-official-anchor uncertainty, not just the point."""
    lo = tree_proj["projected_official_lo"]
    hi = tree_proj["projected_official_hi"]
    margin_lo = tree_proj["margin_to_500_pct_at_lo"]
    mults = [p["multiplier"] for p in calib.per_session]
    spread_pct = 100.0 * (max(mults) - min(mults)) / calib.multiplier

    straddles = lo < OFFICIAL_TARGET_TPS < hi
    wholly_below = hi <= OFFICIAL_TARGET_TPS
    unstable = spread_pct > spread_red_pct
    config_drift = spread_amber_pct <= spread_pct <= spread_red_pct
    thin_margin = (lo >= OFFICIAL_TARGET_TPS) and (margin_lo < margin_green_pct)

    reasons: list[str] = []
    if unstable:
        reasons.append(f"multiplier UNSTABLE: per-session spread {spread_pct:.3f}% > {spread_red_pct:.1f}%")
    if straddles:
        reasons.append(f"band straddles 500: [{lo:.1f}, {hi:.1f}]")
    if wholly_below:
        reasons.append(f"band wholly below 500: hi={hi:.1f}")
    if config_drift:
        reasons.append(f"config-drift: per-session spread {spread_pct:.3f}% in "
                       f"[{spread_amber_pct:.1f}, {spread_red_pct:.1f}]%")
    if thin_margin:
        reasons.append(f"thin margin: low edge clears 500 by only {margin_lo:.2f}% "
                       f"(< {margin_green_pct:.1f}%)")

    if unstable or straddles or wholly_below:
        verdict = "RED"
    elif config_drift or thin_margin:
        verdict = "AMBER"
    else:
        verdict = "GREEN"
        reasons.append(
            f"config-stable (spread {spread_pct:.3f}% < {spread_amber_pct:.1f}%) AND "
            f"band [{lo:.1f}, {hi:.1f}] clears 500 by {margin_lo:.2f}% (>= {margin_green_pct:.1f}%)")

    return {
        "verdict": verdict,
        "per_session_spread_pct": spread_pct,
        "config_stable": spread_pct < spread_amber_pct,
        "band_lo": lo, "band_hi": hi,
        "band_straddles_500": straddles,
        "band_wholly_below_500": wholly_below,
        "margin_to_500_pct_at_lo": margin_lo,
        "clears_500_with_margin": (lo >= OFFICIAL_TARGET_TPS and margin_lo >= margin_green_pct),
        "margin_green_pct": margin_green_pct,
        "reasons": reasons,
    }


def self_check(calib: Calibration | None = None) -> dict[str, Any]:
    """Closed loop: project the LOCKED linear reference and confirm it maps back to
    the official anchor (~481.53) WITHIN the self-check MDE.

    This is a noisy-meter consistency check, not a bit-exact identity. The multiplier
    denominator is the pooled-mean local wall_tps (454.194); the value projected here
    is the #90 LOCKED session reference (454.338). Their difference (~0.03%) is the
    session-to-session spread -- itself sub-MDE -- so the closed loop recovers the
    anchor to within that spread. PASS iff |recovered - anchor| <= SELF_CHECK_MDE_PCT.
    A bit-exact loop would be circular (project the exact value that built the
    denominator); the honest claim is recovery within the measurement noise floor."""
    if calib is None:
        calib = calibrate()
    proj = project_official(LINEAR_REFERENCE_WALL_TPS, calib=calib,
                            modeling_band_pct=0.0, use_envelope=True)
    recovered = proj["projected_official"]
    target = float(OFFICIAL_ANCHOR["tps"])
    abs_err = abs(recovered - target)
    rel_err_pct = 100.0 * abs_err / target
    proj["recovered_official"] = recovered
    proj["official_anchor"] = target
    proj["abs_err_vs_anchor_tps"] = abs_err
    proj["rel_err_vs_anchor_pct"] = rel_err_pct
    proj["self_check_mde_pct"] = SELF_CHECK_MDE_PCT
    proj["recovers_official_anchor"] = rel_err_pct <= SELF_CHECK_MDE_PCT
    return proj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _fmt(v, p=4):
    return f"{v:.{p}f}" if isinstance(v, (int, float)) and v == v else "—"


def _print_calibration(c: Calibration) -> None:
    print("\n===== LOCAL -> OFFICIAL MULTIPLIER CALIBRATION (PR #99) =====")
    print(f"  official anchor (PR#52, private-VERIFIED): {c.official_tps:.2f} TPS")
    print(f"  local deployed wall_tps (mean of {c.n_runs} runs / {c.n_sessions} sessions): "
          f"{c.local_wall_tps:.3f}  (per-run CV {c.local_cv_pct:.4f}%)")
    print(f"  >>> multiplier = {c.multiplier:.5f}")
    print(f"      local-side 95% CI : [{c.mult_ci_local_lo:.5f}, {c.mult_ci_local_hi:.5f}] "
          f"(+/-{100.0*Z_DETECT*c.mult_se_local/c.multiplier:.3f}%)  [measured]")
    print(f"      envelope  95% CI : [{c.mult_ci_env_lo:.5f}, {c.mult_ci_env_hi:.5f}] "
          f"(assumes official per-run CV={c.official_cv_assumed_pct:.2f}%)  [sensitivity]")
    print("  per-session multiplier:")
    for p in c.per_session:
        tag = "fit" if p["primary_fit"] else "corrob"
        print(f"     {p['label']:28s} local={p['local_median_wall_tps']:.3f} "
              f"mult={p['multiplier']:.5f}  [{tag}]")
    for n in c.notes:
        print(f"  - {n}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true", help="print the calibration report")
    ap.add_argument("--self-check", action="store_true",
                    help="closed-loop: project the locked linear reference -> official anchor")
    ap.add_argument("--tree", action="store_true",
                    help="Step-3 analytical projection of land #71's tree + 500-gate")
    ap.add_argument("--project-wall-tps", type=float, default=None,
                    help="project a measured local wall_tps to an official band")
    ap.add_argument("--modeling-band-pct", type=float, default=0.0)
    ap.add_argument("--official-cv-assumed-pct", type=float, default=1.0,
                    help="conservative official per-run CV for the envelope (sensitivity knob)")
    ap.add_argument("--out", type=Path, default=None, help="write the full report JSON")
    args = ap.parse_args(argv)

    calib = calibrate(official_cv_assumed_pct=args.official_cv_assumed_pct)
    report: dict[str, Any] = {"calibration": calib.as_dict(),
                              "official_anchor": OFFICIAL_ANCHOR}

    if args.calibrate or not (args.self_check or args.tree or args.project_wall_tps):
        _print_calibration(calib)

    if args.self_check or not (args.tree or args.project_wall_tps):
        sc = self_check(calib)
        report["self_check"] = sc
        ok = "PASS" if sc["recovers_official_anchor"] else "FAIL"
        print(f"\n----- closed-loop self-check -----")
        print(f"  project({LINEAR_REFERENCE_WALL_TPS:.3f}) = {sc['recovered_official']:.2f} "
              f"official  (anchor {sc['official_anchor']:.2f}, "
              f"residual {sc['rel_err_vs_anchor_pct']:.3f}% vs MDE {sc['self_check_mde_pct']:.2f}%) "
              f"-> {ok}")
        print(f"  band [{sc['projected_official_lo']:.1f}, {sc['projected_official_hi']:.1f}] "
              f"(+/-{sc['band_rel_pct']:.2f}%)")

    if args.project_wall_tps is not None:
        pj = project_official(args.project_wall_tps, calib=calib,
                              modeling_band_pct=args.modeling_band_pct)
        report["projection"] = pj
        print(f"\n----- projection of measured local wall_tps={args.project_wall_tps:.3f} -----")
        print(f"  projected official = {pj['projected_official']:.2f} "
              f"[{pj['projected_official_lo']:.1f}, {pj['projected_official_hi']:.1f}] "
              f"(+/-{pj['band_rel_pct']:.2f}%)")
        print(f"  clears 500: {pj['clears_500']}  "
              f"(margin@lo {pj['margin_to_500_pct_at_lo']:+.1f}%, "
              f"margin@central {pj['margin_to_500_pct_at_central']:+.1f}%)")

    if args.tree:
        tr = project_tree(calib)
        gate = gate_verdict(calib, tr)
        report["tree_projection"] = tr
        report["gate"] = gate
        print(f"\n----- land #71 tree projection (analytical, pre-build) -----")
        print(f"  topology: {tr['topology']}  E[T]={tr['E_T']}")
        print(f"  linear ref {tr['linear_reference_wall_tps']:.3f} x (1+{tr['net_local_gain_pct']:.1f}%) "
              f"= local tree {tr['local_tree_wall_tps']:.2f} wall_tps")
        print(f"  x multiplier {tr['multiplier']:.5f} = projected official {tr['projected_official']:.1f}")
        print(f"  band [{tr['projected_official_lo']:.1f}, {tr['projected_official_hi']:.1f}] "
              f"(+/-{tr['band_rel_pct']:.2f}%; fern published {tr['fern_published_official_band']})")
        print(f"  >>> projected_official_clears_500_bool = {tr['projected_official_clears_500_bool']} "
              f"(margin@lo {tr['margin_to_500_pct_at_lo']:+.1f}%, "
              f"margin@central {tr['margin_to_500_pct_at_central']:+.1f}%)")
        print(f"  >>> GATE = {gate['verdict']}")
        for r in gate["reasons"]:
            print(f"        - {r}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n[projection] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
