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

# denken #105's tree-free slice model is the SINGLE SOURCE OF TRUTH for the decode
# budget + lever composition. We consume it (not re-derive it) so the two harnesses
# cannot drift. It lives beside this file in scripts/profiler/.
PROFILER_DIR = Path(__file__).resolve().parent
if str(PROFILER_DIR) not in sys.path:
    sys.path.insert(0, str(PROFILER_DIR))

import tree_free_500_ceiling as tf  # noqa: E402  (denken #105 slice model)

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
# Tree-free 500-path instrument (PR #112)
# ---------------------------------------------------------------------------
# This is the PRIMARY 500-path after Cycle 40->41 (tree demoted to UPSIDE/insurance,
# denken #105 GREEN). It maps a MEASURED SplitK% (ubel #108) + the small additive
# levers (LK #95, wirbel #110 palette) -> projected-official band vs 500, through
# denken #105's slice composition `official = K_cal*(E[T]/step)*tau`, K_cal=125.268.
#
# How the two harnesses share ONE anchor (no double-count of the transfer factor):
# denken's K_cal = 481.53/3.844 FOLDS the deployed local->official multiplier (1.0599)
# into the 481.53 official anchor. So tf.compose(...) already returns a number on the
# OFFICIAL scale. We carry THIS module's measured multiplier CI [1.05999, 1.06038] as
# a RELATIVE rescale `mult/multiplier_central` (=1.0 at the central anchor, +/-0.018%
# at the CI edges) so the central stays bit-exact on 481.53 while the band honestly
# widens. tau is denken's residual realization factor (Step 2 bounds it from local data).

# tau (local->official realization factor) band carried through the projection. Central
# 1.00 == the deployed multiplier folded into K_cal; the [0.96, 1.00] denken default is
# the GENERIC config-change floor. Step 2 (bound_tau_local) tightens it for the SplitK
# kernel-swap axis specifically.
TAU_BAND_DEFAULT = dict(tf.TAU)  # {"low":0.96,"central":1.00,"high":1.00}

# wirbel #110 palette (LUT scale-of-scales) -- the byte lever that replaces the
# info-theoretically-KILLed #104 double-quant (BASELINE merge-history). It is the SAME
# verify-GEMM byte-count factor denken modelled as double-quant, so it converts to an
# f_dq through tf.dq_tps_to_fdq identically. UNREALIZED today (wirbel #110 WIP) -> we
# bank central=0 (do NOT credit the unbuilt lever) and carry denken's old byte-lever
# magnitude as UPSIDE only. This is the one place this instrument deviates from denken's
# literal central (which still carried the now-KILLed double-quant central +0.5%).
PALETTE_TPS = {"low": 0.0, "central": 0.0, "high": tf.DQ_TPS["high"]}

# ubel #108 SplitK prior window (verify-GEMM bandwidth-utilisation speedup s, closing
# the 77.1%->100% HBM gap). The MEASURED central + CI are supplied at call time; these
# committed defaults (denken #105 SPLITK_UBEL) seed the band before ubel reports.
SPLITK_PRIOR = dict(tf.SPLITK_UBEL)  # {"low":0.05,"central":0.085,"high":0.12}


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
# Tree-free 500-path map (PR #112): SplitK% + levers -> projected official
# ---------------------------------------------------------------------------
def tree_free_official(splitk_s: float, *, mult: float, tau: float,
                       lk_mult: float = 1.0, palette_tps_pct: float = 0.0,
                       fp32_m8: float = 0.0, persist_reclaim: float = 0.0,
                       mult_central: float | None = None) -> dict[str, Any]:
    """Map a SplitK speedup ``splitk_s`` + additive levers to projected-official TPS.

    Runs denken #105's slice composition (the single source of truth for the decode
    budget) then applies THIS module's multiplier CI as a relative rescale
    ``mult/mult_central``. At ``mult == mult_central`` the rescale is 1.0, so the map
    is bit-exact on denken's anchor; at the CI edges it widens by +/-0.018%. ``tau`` is
    denken's residual realization factor; ``palette_tps_pct`` (wirbel #110 byte lever)
    converts to a verify-GEMM byte reduction exactly like denken's double-quant."""
    if mult_central is None:
        mult_central = calibrate().multiplier
    f_dq = tf.dq_tps_to_fdq(palette_tps_pct / 100.0) if palette_tps_pct else 0.0
    p = {"lk_mult": lk_mult, "f_dq": f_dq, "fp32_m8": fp32_m8,
         "persist_reclaim": persist_reclaim, "tau": tau}
    comp = tf.compose(splitk_s, p)
    scale = mult / mult_central
    official = comp["official_tps"] * scale
    return {
        "splitk_s": splitk_s,
        "lk_mult": lk_mult,
        "palette_tps_pct": palette_tps_pct,
        "f_dq": f_dq,
        "tau": tau,
        "multiplier": mult,
        "multiplier_central": mult_central,
        "multiplier_rel_scale": scale,
        "E_T": comp["E_T"],
        "step_time": comp["step_time"],
        "verify_gemm_slice": comp["verify_gemm_slice"],
        "official_tps": official,
        "clears_500": official >= OFFICIAL_TARGET_TPS,
        "margin_to_500_pct": 100.0 * (official - OFFICIAL_TARGET_TPS) / OFFICIAL_TARGET_TPS,
    }


def tree_free_threshold(*, mult: float, tau: float, lk_mult: float = 1.0,
                        palette_tps_pct: float = 0.0, fp32_m8: float = 0.0,
                        persist_reclaim: float = 0.0,
                        mult_central: float | None = None) -> float | None:
    """Minimum SplitK speedup ``s`` that clears 500 under ``(mult, tau, levers)``.

    The #99 multiplier rescale ``scale = mult/mult_central`` multiplies the composed
    official TPS, so clearing ``official*scale >= 500`` is exactly denken's inverter
    with an effective ``tau*scale``. We therefore REUSE ``tf.splitk_threshold_for_500``
    (single source of truth for the inversion algebra) with that scaled tau. Returns
    0.0 if ``s=0`` already clears, ``float('inf')`` if the gap ceiling cannot reach 500."""
    if mult_central is None:
        mult_central = calibrate().multiplier
    scale = mult / mult_central
    p = {
        "lk_mult": lk_mult,
        "f_dq": tf.dq_tps_to_fdq(palette_tps_pct / 100.0) if palette_tps_pct else 0.0,
        "fp32_m8": fp32_m8,
        "persist_reclaim": persist_reclaim,
        "tau": tau * scale,  # multiplier rescale folds into tau in the linear inverter
    }
    return tf.splitk_threshold_for_500(p)


def project_tree_free(splitk_s: float, *,
                      splitk_lo: float | None = None,
                      splitk_hi: float | None = None,
                      lk_mult_central: float | None = None,
                      palette_tps_pct_central: float | None = None,
                      tau_band: dict[str, float] | None = None,
                      calib: Calibration | None = None) -> dict[str, Any]:
    """SINGLE-command tree-free projection: a measured SplitK% (ubel #108) + the small
    additive levers (LK #95, wirbel #110 palette) -> projected-official band vs 500.

    Three corners, all on denken #105's slice model x the #99 multiplier CI:
      * central      -- multiplier-central x tau-central(1.00) x measured SplitK,
                        LK central (projected 1.010), palette central (0, unrealized).
      * conservative -- multiplier-LOW x tau-LOW(0.96) x SplitK-LOW, LK off (1.0),
                        palette off (0), fp32 M=8 haircut high. THIS is the >=500
                        decision corner the PR names (multiplier-low x tau-low).
      * optimistic   -- multiplier-HIGH x tau-high(1.00) x SplitK-HIGH, LK high,
                        palette high, no haircut.

    The gate reads >=500 off the conservative corner so the decision is robust to all
    three CIs (multiplier, tau, SplitK) simultaneously."""
    if calib is None:
        calib = calibrate()
    tb = TAU_BAND_DEFAULT if tau_band is None else tau_band
    mc = calib.multiplier
    lk_c = tf.LK_MULT["central"] if lk_mult_central is None else lk_mult_central
    pal_c = PALETTE_TPS["central"] if palette_tps_pct_central is None else palette_tps_pct_central
    # SplitK CI: explicit edges from ubel #108 when supplied, else collapse to the point.
    s_lo = splitk_s if splitk_lo is None else splitk_lo
    s_hi = splitk_s if splitk_hi is None else splitk_hi

    central = tree_free_official(
        splitk_s, mult=mc, tau=tb["central"], lk_mult=lk_c,
        palette_tps_pct=pal_c, fp32_m8=tf.FP32_M8["central"],
        persist_reclaim=tf.PERSIST_RECLAIM["central"], mult_central=mc)
    conservative = tree_free_official(
        s_lo, mult=calib.mult_ci_local_lo, tau=tb["low"], lk_mult=tf.LK_MULT["low"],
        palette_tps_pct=PALETTE_TPS["low"], fp32_m8=tf.FP32_M8["high"],
        persist_reclaim=tf.PERSIST_RECLAIM["low"], mult_central=mc)
    optimistic = tree_free_official(
        s_hi, mult=calib.mult_ci_local_hi, tau=tb["high"], lk_mult=tf.LK_MULT["high"],
        palette_tps_pct=PALETTE_TPS["high"], fp32_m8=tf.FP32_M8["low"],
        persist_reclaim=tf.PERSIST_RECLAIM["high"], mult_central=mc)

    cen, lo, hi = central["official_tps"], conservative["official_tps"], optimistic["official_tps"]
    return {
        "splitk_s_central": splitk_s,
        "splitk_s_lo": s_lo,
        "splitk_s_hi": s_hi,
        "central": central,
        "conservative_corner": conservative,
        "optimistic_corner": optimistic,
        "projected_official": cen,
        "projected_official_lo": lo,
        "projected_official_hi": hi,
        "band_rel_pct": 100.0 * (hi - lo) / (2.0 * cen) if cen else float("nan"),
        "clears_500_central": cen >= OFFICIAL_TARGET_TPS,
        "clears_500_conservative": lo >= OFFICIAL_TARGET_TPS,
        "margin_to_500_pct_at_conservative": 100.0 * (lo - OFFICIAL_TARGET_TPS) / OFFICIAL_TARGET_TPS,
        "margin_to_500_pct_at_central": 100.0 * (cen - OFFICIAL_TARGET_TPS) / OFFICIAL_TARGET_TPS,
        "tau_band": tb,
        "multiplier_ci": [calib.mult_ci_local_lo, calib.mult_ci_local_hi],
    }


def tree_free_self_check(calib: Calibration | None = None) -> dict[str, Any]:
    """Closed loop: the NULL-lever tree-free point (s=0, LK=1, palette=0, tau=1,
    mult=central) must reproduce the 481.53 official anchor BIT-EXACT.

    Unlike the linear-reference self-check (a noisy-meter recovery), this one is an
    EXACT identity: ``tree_free_official(0, mult=central, tau=1, ...) ==
    K_cal*E_T_linear*1.0*1.0 == 481.53``. It pins that denken #105's K_cal and this
    module's multiplier agree on the same anchor with no double-count."""
    if calib is None:
        calib = calibrate()
    mc = calib.multiplier
    null = tree_free_official(0.0, mult=mc, tau=1.0, lk_mult=1.0,
                              palette_tps_pct=0.0, fp32_m8=0.0,
                              persist_reclaim=0.0, mult_central=mc)
    recovered = null["official_tps"]
    target = float(OFFICIAL_ANCHOR["tps"])
    abs_err = abs(recovered - target)
    rel_err_pct = 100.0 * abs_err / target
    return {
        "recovered_official": recovered,
        "official_anchor": target,
        "abs_err_vs_anchor_tps": abs_err,
        "rel_err_vs_anchor_pct": rel_err_pct,
        "self_check_mde_pct": SELF_CHECK_MDE_PCT,
        "reproduces_anchor_bit_exact": abs_err < 1e-9,
        "reproduces_official_anchor": rel_err_pct <= SELF_CHECK_MDE_PCT,
        "k_cal": tf.K_CAL,
        "e_t_linear": tf.E_T_LINEAR,
        "note": "NULL-lever tree-free point == K_cal*E_T_linear == 481.53 by construction; "
                "the bit-exact consistency anchor tying denken #105's K_cal to #99's multiplier.",
    }


# ---------------------------------------------------------------------------
# Step 2: bound tau from committed local data
# ---------------------------------------------------------------------------
# The three committed local meters of the IDENTICAL deployed K=7 stack -- their ratio
# to the single official anchor is the "multiplier" each meter would imply. The spread
# across meters is the METER CONFOUND that swamps any cross-precision config signal.
METER_WITNESS = [
    {"meter": "steady (16-prompt local_prevalidate, FRAGILE/retired)", "local_tps": 428.37,
     "source": "BASELINE.md PR#43 tps_local_splitkv_steady; retired by #72 (BASELINE.md:178)"},
    {"meter": "wall_tps (128-prompt #72/#82 protocol, CANONICAL)", "local_tps": 454.09,
     "source": "BASELINE.md re-baseline (median N=3, CV 0.007%)"},
    {"meter": "windowed-steady (drop W=3 cold intervals)", "local_tps": 459.83,
     "source": "research/EXPERIMENTS_LOG.md:498 (robust interval-meter variant)"},
]


def bound_tau_local(calib: Calibration | None = None) -> dict[str, Any]:
    """Step 2: bound ``tau`` (the local->official realization factor for a config that
    DIFFERS from the deployed anchor) from COMMITTED cross-config local data, as the
    empirical proxy for whether a SplitK kernel swap moves the transfer.

    The honest finding -- there is exactly ONE matched (official, local) pair on the
    frontier (the deployed anchor, which DEFINES tau=1.00). No committed config gives a
    second matched pair in the SAME meter:
      * config CHANGES that have a committed LOCAL wall_tps (K-sweep 5..9, MBT-sweep
        512..8192) all read the same ~454.x denominator -> the wall_tps meter is
        config-stable to <0.1% (the per-session spread below), but those configs have
        NO official counterpart, so they bound LOCAL stability, not tau.
      * configs with BOTH an official and a local number (older precision rungs) were
        metered with the 16-prompt steady meter, not the 128-prompt wall_tps protocol,
        so their official/local ratio conflates config-drift with the ~7% METER spread
        (steady 428 / wall_tps 454 / windowed 460 for the IDENTICAL deployed stack).
    So tau for a kernel swap is NOT directly measurable from committed data. The band is
    a MECHANISM inference + a physical ceiling, with an explicit re-anchor recommendation
    for denken #109."""
    if calib is None:
        calib = calibrate()
    official = float(OFFICIAL_ANCHOR["tps"])

    # (a) meter-confound witness: the multiplier each committed meter would imply.
    meter_rows = []
    for m in METER_WITNESS:
        meter_rows.append({**m, "implied_multiplier": official / m["local_tps"]})
    meter_mults = [r["implied_multiplier"] for r in meter_rows]
    meter_spread_pct = 100.0 * (max(meter_mults) - min(meter_mults)) / statistics.fmean(meter_mults)

    # (b) within-meter config-stability witness: per-session wall_tps multiplier spread
    # (committed K-sweep / MBT sessions). This is the meter-MATCHED signal -- tiny.
    sess_mults = [p["multiplier"] for p in calib.per_session]
    sess_spread_pct = 100.0 * (max(sess_mults) - min(sess_mults)) / calib.multiplier

    # (c) the band. Upper bound tau<=1.00 is a physical CEILING: a bandwidth-utilisation
    # lever (SplitK) realized locally cannot OVER-realize officially (both A10G boxes
    # share GDDR6 ~600 GB/s sm_86 bandwidth; the verify-GEMM is BW-bound so the speedup
    # is SM-clock-insensitive). Lower bound: the mechanism says the fractional speedup
    # transfers ~1:1 -> 0.99; denken's GENERIC config floor is 0.96.
    tau_mechanism_low = 0.99
    tau_generic_low = float(tf.TAU["low"])  # 0.96
    tau_high = float(tf.TAU["high"])        # 1.00
    tau_band_local = [tau_mechanism_low, tau_high]

    # (d) does the choice of floor change the >=500 decision? Threshold at the
    # CONSERVATIVE corner (mult-low, LK off, palette off, fp32 high) across tau.
    mc = calib.multiplier
    def cons_threshold(tau: float) -> float | None:
        return tree_free_threshold(
            mult=calib.mult_ci_local_lo, tau=tau, lk_mult=tf.LK_MULT["low"],
            palette_tps_pct=PALETTE_TPS["low"], fp32_m8=tf.FP32_M8["high"],
            persist_reclaim=tf.PERSIST_RECLAIM["low"], mult_central=mc)
    thr_at_tau = {f"{t:.2f}": cons_threshold(t)
                  for t in (0.96, 0.97, 0.98, 0.99, 1.00)}
    ubel_high = float(tf.SPLITK_UBEL["high"])
    ceiling = float(tf.SPLITK_CEILING)
    thr_mech = cons_threshold(tau_mechanism_low)
    thr_generic = cons_threshold(tau_generic_low)

    def reachable(s):  # within ubel's nominal-high deliverable
        return s is not None and s != float("inf") and s <= ubel_high
    decides_at_mech = reachable(thr_mech)
    decides_at_generic = reachable(thr_generic)

    # (e) recommendation to denken #109.
    if decides_at_generic:
        recommendation = "SHIP_ON_LOCAL_CAL"
        rec_detail = ("No official re-anchor needed: even at denken's GENERIC tau floor "
                      f"0.96 the conservative-corner SplitK threshold ({_pct(thr_generic)}) "
                      f"is within ubel's nominal-high deliverable ({ubel_high*100:.0f}%) -> "
                      "ubel #108's SplitK number alone decides 500.")
    elif decides_at_mech:
        recommendation = "ONE_OFFICIAL_SPLITK_ANCHOR"
        rec_detail = ("Config-sensitive at the floor: at the MECHANISM tau floor 0.99 the "
                      f"conservative threshold ({_pct(thr_mech)}) is within ubel's "
                      f"deliverable ({ubel_high*100:.0f}%), but at the generic 0.96 floor "
                      f"it rises to {_pct(thr_generic)} (> deliverable). denken #109 should "
                      "take ONE official SplitK anchor to confirm tau>=0.99 (the mechanism "
                      "predicts it: SplitK is a bandwidth lever, bandwidth is identical "
                      "across both A10G boxes) -> converts AMBER to GREEN.")
    else:
        recommendation = "TRANSFER_UNTRUSTWORTHY"
        rec_detail = ("Even at the mechanism floor 0.99 the conservative threshold "
                      f"({_pct(thr_mech)}) exceeds ubel's deliverable -> the tree-free "
                      "local projection cannot decide 500 on its own.")

    return {
        "test_metric_name": "tau_band_local",
        "tau_band_local": tau_band_local,
        "tau_mechanism_low": tau_mechanism_low,
        "tau_generic_low": tau_generic_low,
        "tau_high_physical_ceiling": tau_high,
        "meter_confound": {
            "meters": meter_rows,
            "implied_multiplier_spread_pct": meter_spread_pct,
            "interpretation": ("meter choice alone swings the implied multiplier by "
                               f"{meter_spread_pct:.1f}% -- this swamps any cross-precision "
                               "config signal, so precision rungs cannot bound tau."),
        },
        "within_meter_config_stability": {
            "per_session_multiplier_spread_pct": sess_spread_pct,
            "interpretation": ("meter-MATCHED config changes (K-sweep, MBT) move the local "
                               f"wall_tps denominator by only {sess_spread_pct:.3f}% -- the "
                               "transfer is NOT config-sensitive within a matched meter."),
        },
        "conservative_threshold_at_tau": thr_at_tau,
        "ubel_nominal_high": ubel_high,
        "splitk_ceiling": ceiling,
        "decides_at_mechanism_floor": decides_at_mech,
        "decides_at_generic_floor": decides_at_generic,
        "recommendation": recommendation,
        "recommendation_detail": rec_detail,
        "mechanism": ("SplitK closes the verify-GEMM HBM-bandwidth gap; both the local AWS "
                      "A10G and the HF-Jobs a10g-small are sm_86 / GDDR6 ~600 GB/s and the "
                      "verify-GEMM is bandwidth-bound (SM-clock-insensitive), so the "
                      "fractional speedup transfers ~1:1 -> tau in [0.99, 1.00]."),
    }


# ---------------------------------------------------------------------------
# Step 3: tree-free 500-path gate
# ---------------------------------------------------------------------------
def tree_free_gate(calib: Calibration | None = None, *,
                   tau_bound: dict[str, Any] | None = None,
                   spread_red_pct: float = 2.0) -> dict[str, Any]:
    """Step-3 gate: is the tree-free->official instrument ARMED, and is tau tight enough
    that ubel #108's SplitK number ALONE decides 500 at the conservative corner?

    * GREEN -- armed (self-check reproduces the anchor) AND the conservative-corner
               SplitK threshold is within ubel's deliverable even at denken's GENERIC
               tau floor (0.96) -> SplitK alone decides, tau is irrelevant to the call.
    * AMBER -- armed, but the decision turns on the tau floor: it clears at the
               MECHANISM floor (0.99) yet not at the generic 0.96 -> name ONE official
               SplitK anchor for denken #109 to confirm tau>=0.99 (OR an aggressive
               SplitK above ubel's nominal, still within the bandwidth-gap ceiling).
    * RED   -- transfer config-UNSTABLE (per-session multiplier spread > spread_red_pct)
               OR the threshold is unreachable within the bandwidth-gap ceiling even at
               the mechanism floor -> the local projection can't be trusted.

    PRIMARY metric ``tree_free_projection_armed`` (bool). TEST metric ``tau_band_local``."""
    if calib is None:
        calib = calibrate()
    if tau_bound is None:
        tau_bound = bound_tau_local(calib)

    sc = tree_free_self_check(calib)
    armed = bool(sc["reproduces_anchor_bit_exact"])

    sess_mults = [p["multiplier"] for p in calib.per_session]
    spread_pct = 100.0 * (max(sess_mults) - min(sess_mults)) / calib.multiplier
    unstable = spread_pct > spread_red_pct

    thr_mech = tau_bound["conservative_threshold_at_tau"]["0.99"]
    ceiling = tau_bound["splitk_ceiling"]
    unreachable = (thr_mech is None) or (thr_mech == float("inf")) or (thr_mech > ceiling)
    decides_at_generic = bool(tau_bound["decides_at_generic_floor"])
    decides_at_mech = bool(tau_bound["decides_at_mechanism_floor"])

    reasons: list[str] = []
    if not armed:
        reasons.append(f"NOT ARMED: self-check residual {sc['rel_err_vs_anchor_pct']:.4f}% "
                       "(null-lever point must reproduce 481.53 bit-exact)")
    if unstable:
        reasons.append(f"transfer UNSTABLE: per-session multiplier spread {spread_pct:.3f}% "
                       f"> {spread_red_pct:.1f}%")
    if unreachable:
        reasons.append("threshold unreachable within the bandwidth-gap ceiling even at the "
                       "mechanism tau floor 0.99")

    if (not armed) or unstable or unreachable:
        verdict = "RED"
        verdict_label = ("tree-free local projection cannot be trusted "
                         "(unarmed / config-unstable / unreachable)")
    elif decides_at_generic:
        verdict = "GREEN"
        verdict_label = ("instrument ARMED + tau tight enough that ubel #108's SplitK number "
                         "ALONE decides 500 at the conservative corner (clears even at the "
                         "generic 0.96 floor)")
        reasons.append("conservative-corner threshold within ubel's deliverable at every tau "
                       "in [0.96, 1.00] -> SplitK alone decides 500")
    else:
        verdict = "AMBER"
        verdict_label = ("instrument ARMED; the >=500 call turns on the tau floor -> denken "
                         "#109 needs ONE official SplitK anchor to confirm tau>=0.99")
        reasons.append(tau_bound["recommendation_detail"])

    return {
        "primary_metric_name": "tree_free_projection_armed",
        "tree_free_projection_armed": armed,
        "test_metric_name": "tau_band_local",
        "tau_band_local": tau_bound["tau_band_local"],
        "verdict": verdict,
        "verdict_label": verdict_label,
        "self_check": sc,
        "per_session_spread_pct": spread_pct,
        "transfer_stable": not unstable,
        "decides_at_mechanism_floor": decides_at_mech,
        "decides_at_generic_floor": decides_at_generic,
        "recommendation": tau_bound["recommendation"],
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _pct(s) -> str:
    if s is None:
        return "clears@s=0"
    if s == float("inf"):
        return ">ceiling"
    if s == 0.0:
        return "clears@s=0"
    return f"{s*100:.2f}%"


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


def _log_tree_free_wandb(args, calib: Calibration, tau_bound: dict[str, Any],
                         gate: dict[str, Any], projection: dict[str, Any] | None) -> None:
    """Rich W&B log of the PR#112 tree-free instrument (group tree-free-projection-harden)."""
    import wandb

    def jnum(x):  # inf/None -> finite sentinels for numeric panels
        if x is None:
            return -1.0
        return 9.99 if x == float("inf") else float(x)

    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        name=args.wandb_name, group=args.wandb_group, job_type="analysis",
        config={
            "instrument": "tree-free-500-projection (PR#112)",
            "method": "denken#105 slice-compose x #99 multiplier-CI rescale, CPU-analytic",
            "official_anchor_tps": calib.official_tps,
            "multiplier_central": calib.multiplier,
            "multiplier_ci_local": [calib.mult_ci_local_lo, calib.mult_ci_local_hi],
            "k_cal": tf.K_CAL, "e_t_linear": tf.E_T_LINEAR, "budget": tf.BUDGET,
            "tau_band_default": TAU_BAND_DEFAULT, "palette_tps": PALETTE_TPS,
            "splitk_prior": SPLITK_PRIOR, "splitk_ceiling": tf.SPLITK_CEILING,
            "target_official": OFFICIAL_TARGET_TPS,
        })
    s = wandb.summary
    sc = gate["self_check"]
    s["tree_free_projection_armed"] = gate["tree_free_projection_armed"]
    s["self_check_recovered_official"] = sc["recovered_official"]
    s["self_check_residual_pct"] = sc["rel_err_vs_anchor_pct"]
    s["tau_band_local_low"] = tau_bound["tau_band_local"][0]
    s["tau_band_local_high"] = tau_bound["tau_band_local"][1]
    s["tau_mechanism_low"] = tau_bound["tau_mechanism_low"]
    s["tau_generic_low"] = tau_bound["tau_generic_low"]
    s["meter_confound_spread_pct"] = tau_bound["meter_confound"]["implied_multiplier_spread_pct"]
    s["within_meter_config_spread_pct"] = (
        tau_bound["within_meter_config_stability"]["per_session_multiplier_spread_pct"])
    s["decides_at_mechanism_floor"] = tau_bound["decides_at_mechanism_floor"]
    s["decides_at_generic_floor"] = tau_bound["decides_at_generic_floor"]
    s["recommendation"] = tau_bound["recommendation"]
    s["verdict"] = gate["verdict"]
    s["verdict_label"] = gate["verdict_label"]
    s["per_session_spread_pct"] = gate["per_session_spread_pct"]

    # tau -> conservative-corner SplitK threshold sweep
    tt = wandb.Table(columns=["tau", "conservative_splitk_threshold",
                              "within_ubel_deliverable"])
    ubel_high = tau_bound["ubel_nominal_high"]
    for t, thr in tau_bound["conservative_threshold_at_tau"].items():
        within = (thr is not None and thr != float("inf") and thr <= ubel_high)
        tt.add_data(float(t), jnum(thr), within)
        wandb.log({"tau_threshold/tau": float(t),
                   "tau_threshold/conservative_splitk": jnum(thr),
                   "tau_threshold/ubel_nominal_high": ubel_high,
                   "tau_threshold/splitk_ceiling": tau_bound["splitk_ceiling"]})
    wandb.log({"tau_vs_conservative_splitk_threshold": tt})

    # meter-confound witness
    mt = wandb.Table(columns=["meter", "local_tps", "implied_multiplier"])
    for r in tau_bound["meter_confound"]["meters"]:
        mt.add_data(r["meter"], r["local_tps"], r["implied_multiplier"])
    wandb.log({"meter_confound": mt})

    if projection is not None:
        s["projection_central_official"] = projection["projected_official"]
        s["projection_conservative_official"] = projection["projected_official_lo"]
        s["projection_optimistic_official"] = projection["projected_official_hi"]
        s["projection_clears_500_conservative"] = projection["clears_500_conservative"]

    print(f"\nW&B run: {run.id}  ({run.url})")
    wandb.finish()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true", help="print the calibration report")
    ap.add_argument("--self-check", action="store_true",
                    help="closed-loop: project the locked linear reference -> official anchor")
    ap.add_argument("--tree", action="store_true",
                    help="Step-3 analytical projection of land #71's tree + 500-gate")
    ap.add_argument("--tree-free", action="store_true",
                    help="PR#112: tree-free 500-path instrument (tau bound + Step-3 gate)")
    ap.add_argument("--splitk-frac", type=float, default=None,
                    help="project a measured SplitK speedup s (fraction, e.g. 0.085) -> official band")
    ap.add_argument("--splitk-lo", type=float, default=None, help="SplitK CI low edge (fraction)")
    ap.add_argument("--splitk-hi", type=float, default=None, help="SplitK CI high edge (fraction)")
    ap.add_argument("--lk-mult", type=float, default=None,
                    help="LK #95 E[T] multiplier central override (default denken 1.010)")
    ap.add_argument("--palette-tps-pct", type=float, default=None,
                    help="wirbel #110 palette central TPS%% gain (default 0, unrealized)")
    ap.add_argument("--project-wall-tps", type=float, default=None,
                    help="project a measured local wall_tps to an official band")
    ap.add_argument("--modeling-band-pct", type=float, default=0.0)
    ap.add_argument("--official-cv-assumed-pct", type=float, default=1.0,
                    help="conservative official per-run CV for the envelope (sensitivity knob)")
    ap.add_argument("--out", type=Path, default=None, help="write the full report JSON")
    ap.add_argument("--wandb", action="store_true", help="log the tree-free gate to W&B")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="lawine/tree-free-projection-harden")
    ap.add_argument("--wandb-group", default="tree-free-projection-harden")
    args = ap.parse_args(argv)

    calib = calibrate(official_cv_assumed_pct=args.official_cv_assumed_pct)
    report: dict[str, Any] = {"calibration": calib.as_dict(),
                              "official_anchor": OFFICIAL_ANCHOR}

    explicit = bool(args.self_check or args.tree or args.tree_free
                    or args.project_wall_tps is not None or args.splitk_frac is not None)

    if args.calibrate or not explicit:
        _print_calibration(calib)

    if args.self_check or not explicit:
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

    if args.splitk_frac is not None:
        pj = project_tree_free(
            args.splitk_frac, splitk_lo=args.splitk_lo, splitk_hi=args.splitk_hi,
            lk_mult_central=args.lk_mult, palette_tps_pct_central=args.palette_tps_pct,
            calib=calib)
        report["tree_free_projection"] = pj
        print(f"\n----- tree-free projection of SplitK s={args.splitk_frac:.4f} "
              f"(LK={pj['central']['lk_mult']:.3f}, palette={pj['central']['palette_tps_pct']:.2f}%) -----")
        print(f"  central      = {pj['projected_official']:.1f} official "
              f"(margin@central {pj['margin_to_500_pct_at_central']:+.1f}%)")
        print(f"  conservative = {pj['projected_official_lo']:.1f}  (mult-low x tau-low x SplitK-low; "
              f"margin {pj['margin_to_500_pct_at_conservative']:+.1f}%)")
        print(f"  optimistic   = {pj['projected_official_hi']:.1f}")
        print(f"  >>> clears 500 at conservative corner: {pj['clears_500_conservative']} "
              f"(central: {pj['clears_500_central']})")

    if args.tree_free:
        tau_bound = bound_tau_local(calib)
        gate = tree_free_gate(calib, tau_bound=tau_bound)
        report["tree_free_tau_bound"] = tau_bound
        report["tree_free_gate"] = gate
        sc = gate["self_check"]
        print(f"\n===== TREE-FREE 500-PATH INSTRUMENT (PR #112) =====")
        print(f"  [arm] null-lever self-check: {sc['recovered_official']:.6f} official "
              f"(anchor {sc['official_anchor']:.2f}, residual {sc['rel_err_vs_anchor_pct']:.2e}%) "
              f"-> {'BIT-EXACT' if sc['reproduces_anchor_bit_exact'] else 'FAIL'}")
        print(f"\n  [Step 2] bound tau from committed local data:")
        mc = tau_bound["meter_confound"]
        print(f"    meter confound (same deployed stack, 3 committed meters):")
        for r in mc["meters"]:
            print(f"       {r['local_tps']:7.2f} tok/s -> implied mult {r['implied_multiplier']:.4f}  "
                  f"[{r['meter'].split('(')[0].strip()}]")
        print(f"       -> meter choice alone swings the multiplier by "
              f"{mc['implied_multiplier_spread_pct']:.1f}% (swamps cross-precision config signal)")
        wm = tau_bound["within_meter_config_stability"]
        print(f"    within-meter (matched wall_tps) config spread: "
              f"{wm['per_session_multiplier_spread_pct']:.3f}% (K/MBT sweeps stable)")
        print(f"    >>> tau_band_local = [{tau_bound['tau_band_local'][0]:.2f}, "
              f"{tau_bound['tau_band_local'][1]:.2f}]  "
              f"(mechanism floor 0.99; physical ceiling 1.00; generic floor 0.96)")
        print(f"    conservative-corner SplitK threshold vs tau:")
        for t, s in tau_bound["conservative_threshold_at_tau"].items():
            flag = "<= ubel-high" if (s is not None and s != float("inf")
                                      and s <= tau_bound["ubel_nominal_high"]) else "> ubel-high"
            print(f"       tau={t} -> SplitK {_pct(s):>10s}  ({flag})")
        print(f"    re-anchor recommendation: {tau_bound['recommendation']}")
        print(f"       {tau_bound['recommendation_detail']}")
        print(f"\n  [Step 3] GATE = {gate['verdict']} -- {gate['verdict_label']}")
        print(f"    primary  tree_free_projection_armed = {gate['tree_free_projection_armed']}")
        print(f"    test     tau_band_local             = {gate['tau_band_local']}")
        for r in gate["reasons"]:
            print(f"       - {r}")

        if args.wandb:
            _log_tree_free_wandb(args, calib, tau_bound, gate, report.get("tree_free_projection"))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n[projection] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
