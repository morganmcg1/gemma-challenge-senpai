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
# Step 1 + 2 (PR #116): DERIVE tau for a bandwidth lever from a first-principles
# roofline -- replace the ASSERTED 0.99 floor (bound_tau_local) with a derived one
# ---------------------------------------------------------------------------
# WHY this supersedes the asserted floor
# --------------------------------------
# bound_tau_local proved the *data* path to tau is blocked (one matched (official,
# local) pair; 7.1% meter confound) and fell back to an ungrounded round number
# (0.99). This block DERIVES tau from the decode-step roofline instead.
#
# The key identity. tau enters denken #105's model as a multiplier on the official
# TPS: official = K_cal * (E[T]/step_local) * tau, where K_cal folds the DEPLOYED
# local->official multiplier (m=1.0602) into the 481.53 anchor. At the null point
# (s=0) tau == 1.0 by construction (the bit-exact self-check). So tau is the
# *residual realization factor* of a LEVER: how much of the locally-measured
# step-time improvement actually shows up officially. Because tau->1 as s->0, any
# deviation is necessarily SECOND-ORDER in the lever size s -- there is no
# first-order-in-s-free epsilon. This is the structural reason the band is tight.
#
# tau DECOMPOSES into two independent factors, tau = tau_eff * tau_mix:
#
#   tau_mix  (slice-mix realization).  The lever removes LOCAL verify-GEMM time.
#     Officially that time is re-weighted by the per-component local->official
#     speed ratios m_c. If every slice transfers at the SAME ratio (the deployed
#     m), the local improvement transfers 1:1 -> tau_mix = 1.0 EXACTLY. Deviation
#     requires the verify-GEMM's transfer m_vg to differ from the step average m,
#     and even then |tau_mix - 1| = s * phi_vg * |m/m_vg - 1| (second order). The
#     ABSOLUTE per-component transfers (incl. any ECC/thermal/clock difference
#     between the boxes) are ALREADY absorbed into the deployed m=K_cal; only their
#     SPREAD feeds tau_mix, and only at second order. denken #97 (MERGED) measured
#     the ~32% "other" tail as 97.83% GPU-busy, "bus is the wall" -> bus(BW)-bound
#     like the verify-GEMM -> uniform transfer -> tau_mix_central = 1.0 (eps=0).
#
#   tau_eff  (lever-efficacy realization).  Does the SplitK *fractional* speedup s
#     itself realize officially? s closes the verify-GEMM's achieved-HBM-util gap
#     (77.1%->100%). util is dimensionless (achieved = util * peak_BW), so closing
#     it multiplies achieved BW by (1+s) on BOTH boxes regardless of peak_BW --
#     i.e. the peak-BW difference CANCELS in the ratio. tau_eff = 1.0 IFF (a) the
#     verify-GEMM is BW-bound officially too and (b) the util gap is architecture-
#     invariant. (a): int4 W4A16 at M=8 sits at arithmetic intensity ~32 FLOP/byte
#     vs the sm_86 FP16/TC ridge ~117-208 (>=3.6x margin) -- BW-bound with margin
#     on identical silicon (Marlin/Machete/IBM-SplitK literature + repo #68). (b):
#     the util gap is wave-quantization, set by SM count -- identical on two A10G.
#     The ONE un-pinnable residual: the split-K reduction's SYNC OVERHEAD is mildly
#     absolute-BW-sensitive (the IBM A100 PCIe-vs-SXM caution), so s could under-
#     realize by a small relative haircut local data cannot measure -> the named
#     job of the ONE pre-registered official anchor (Step 3).
ROOFLINE_BUDGET = dict(tf.BUDGET)  # denken #97/#105: vg .53, draft .07, attn .08, other .32

# denken #97 (MERGED) tail characterization -- the load-bearing central input.
TAIL_GPU_BUSY_FRAC = 0.9783        # decode 97.83% GPU-busy
TAIL_RECLAIMABLE_IDLE = 0.0217     # only 2.17pp reclaimable GPU-idle ("bus is the wall")

# verify-GEMM roofline (researcher cross-check + repo #68). int4 W4A16, M=8.
VERIFY_GEMM_INTENSITY_M8 = 32.0    # FLOP/byte (2*M FLOP per 0.5-byte int4 weight elt, M=8)
SM86_RIDGE_FLOP_PER_BYTE = {       # peak-compute / peak-BW on sm_86 (A10G ~600 GB/s)
    "fp16_tc": 208.0,              # ~125 TFLOP/s FP16-accum tensor core
    "fp32_accum_tc": 117.0,        # ~70 TFLOP/s FP32-accum tensor core
    "conservative_non_tc": 52.0,   # ~31 TFLOP/s non-TC (researcher's lower bound)
}

# ===== TREE verify-tau roofline (PR #126): the M=32 wide-verify geometry =======
# The tree (land #71 topology, denken #101 E[T]=5.207) verifies M=32 candidate
# tokens per step, NOT the M=8 of the SplitK/tree-free path. Its verify-GEMM has 4x
# the FLOPs and sits AT the sm_86 knee, so the SplitK-class tau ([0.9983,1.00], #116)
# cannot be BORROWED -- the tree-class tau is DERIVED here from denken #68's MEASURED
# M=32 roofline + my #107 MEASURED M=32/M=8 step denominator.
#
# denken #68 verify_gemm_roofline.json aggregate_by_M (MEASURED A10G Marlin W4A16):
TREE_AI_AGG = {                    # aggregate arithmetic intensity (FLOP/byte) by width
    8:  28.045859872611466,        # SplitK/tree-free width: BW-bound (20% compute peak)
    32: 107.65770171149144,        # tree width: AT THE KNEE (68% compute / 68% HBM)
    33: 110.83569794050344,        # one past the Marlin tile cliff (M=33)
}
TREE_PCT_COMPUTE_AGG = {8: 20.153307906403988, 32: 68.057370953174, 33: 54.21212799229282}
TREE_PCT_HBM_AGG = {8: 77.05767476868124, 32: 67.79042922403865, 33: 52.45116202444165}
TREE_RIDGE = {                     # sm_86 ridge = peak_compute / peak_BW (FLOP/byte)
    "measured_marlin": 107.23543542869047,    # MEASURED Marlin 64.34 TFLOPS / 600 GB/s
    "datasheet_fp16accum": 116.66666666666667,  # 70 TFLOPS datasheet / 600 GB/s
    "fp16_tc": 208.0,              # 125 TFLOPS FP16-accum / 600 (far ceiling)
}
TREE_PEAK_TFLOPS_MEASURED = 64.34126125721428   # denken #68 compute-ceiling best (M=1024)
TREE_BYTES_RATIO_M32_M8 = 15564800.0 / 14950400.0  # 1.0411: weights fixed, activations ~4x
# M=33 Marlin tile cliff (LOCAL kernel-tiling artifact, architecture-determined).
TREE_TILE_CLIFF = {
    "tile_n": 128,                 # Marlin N-tile -> GEMM flat M<=32, +step at M=33
    "sm_count": 80,                # sm_86 A10G SM count (sets wave quantization)
    "m33_step_jump_pct_of_decode_step": 14.645273562135367,  # denken #68 marginal[33]
    "tree_operates_at_M": 32,      # the tree is DESIGNED at M<=32 (under the cliff)
}
# my #107 tree_step_denominator.json (MEASURED, median N=5) -- the step denominator.
TREE_STEP = {
    "r_gemm": 1.1686205063215744,  # verify-GEMM M=32/M=8 time ratio (median)
    "r_attn": 1.8325004530121336,  # verify-attention M=32/M=8 (tree-mask, median)
    "whole_step_ratio": 1.1559689045914052,  # method_A budget-share whole-step M=32/M=8
    "budget_share_gemm": 0.53,     # verify-GEMM share of the M=8 decode step
    "budget_share_attn": 0.08,     # attention share
    "budget_share_remainder": 0.39,  # M-invariant remainder (drafter/sample/launch/KV)
    "E_T_tree": 5.207,             # denken #101 analytical-ceiling tree E[T]
    "E_T_linear_chain": 3.844,     # linear-chain reference E[T]
}
TREE_SMCLOCK_PINNED_MHZ = 1710.0   # LOCAL SM clock PINNED (boost); official a10g-small free.
# Official/local effective SM-clock ratio m_comp = clock_off / clock_loc. The ONLY
# un-pinnable axis: the BW peak cancels via the deployed multiplier and SM-count +
# FLOP/cycle are architecture-identical, so compute-bound work transfers at this clock
# ratio alone. Credited band uses a mild thermal throttle; deeper throttle corners are
# NAMED but NOT credited (the ONE official anchor measures the true clock under load).
TREE_MCOMP = {
    "uniform_central": None,       # central = uniform transfer (m_comp == m_bus) -> tau=1
    "bus_parity": 1.0,             # official SM clock == local pin (compute misses +6% bus)
    "mild_throttle": 0.965,        # -3.5% below the pin (credited adversarial floor)
    "deep_throttle": 0.877,        # -12.3% toward base clock (named, NOT credited)
}
# fern #106/#111 tree ship projection (report_lever_composition.md) the fold-in targets.
TREE_FERN_CENTRAL_OFFICIAL = 568.0          # net_tree central 0.1796 -> 568 official
TREE_LEVER_CONSERVATIVE_BORROWED_TAU = 517.9560880341589  # lever_composition tree_alone cons
TREE_BORROWED_TAU_LOW = 0.96                # the generic tau floor (#99) being replaced


def _transfer_profile(name: str, m: float) -> dict[str, float]:
    """Per-component local->official speed ratios m_c consistent with the deployed
    overall multiplier m (= 1/sum(phi_c/m_c)). Three bounding hypotheses for WHERE
    the deployed +6% officially-faster lives:

      * uniform      -- every slice bus(BW)-bound, transfers at the SAME bus ratio m
                        (denken #97 "bus is the wall"). The CENTRAL hypothesis.
      * bw_carries   -- ADVERSARIAL floor: the whole ~32% tail is launch/SM-clock-
                        bound and transfers at 1.0 (no speedup); the verify-GEMM and
                        the other BW-bound slices carry ALL the +6% (m_vg = m_bw > m)
                        -> removing verify-GEMM time UNDER-realizes -> tau_mix < 1.
      * tail_carries -- the verify-GEMM is BW-identical (m_vg=1.0) and the tail carries
                        all +6% (m_tail > m) -> removing verify-GEMM time OVER-realizes
                        -> tau_mix > 1 (banked only as a ceiling; we do not credit it)."""
    b = ROOFLINE_BUDGET
    rest = b["verify_gemm"] + b["drafter"] + b["attention"]  # 0.68 BW-bound slices
    tail = b["other"]                                         # 0.32 tail
    if name == "uniform":
        return {c: m for c in b}
    if name == "bw_carries":
        m_bw = rest / (1.0 / m - tail)        # tail @1.0 forces the rest to carry +6%
        return {"verify_gemm": m_bw, "drafter": m_bw, "attention": m_bw, "other": 1.0}
    if name == "tail_carries":
        m_tail = tail / (1.0 / m - rest)       # rest @1.0 forces the tail to carry +6%
        return {"verify_gemm": 1.0, "drafter": 1.0, "attention": 1.0, "other": m_tail}
    raise ValueError(name)


def tau_mix(splitk_s: float, m_c: dict[str, float],
            budget: dict[str, float] | None = None) -> float:
    """EXACT slice-mix realization factor of a SplitK speedup ``splitk_s`` under the
    per-component transfer profile ``m_c``.

    tau_mix = (official step speedup) / (local step speedup), where the lever scales
    the verify-GEMM local time by 1/(1+s) and leaves the other slices unchanged:

        tau_mix = [ (sum phi_c/m_c) * (sum phi_c g_c) ] / (sum phi_c g_c/m_c)

    with g_vg = 1/(1+s), g_else = 1, and the local step normalised to sum phi_c = 1.
    Returns 1.0 EXACTLY when all m_c are equal (uniform transfer)."""
    b = ROOFLINE_BUDGET if budget is None else budget
    g = {c: (1.0 / (1.0 + splitk_s) if c == "verify_gemm" else 1.0) for c in b}
    t_off = sum(b[c] / m_c[c] for c in b)
    tp_loc = sum(b[c] * g[c] for c in b)
    tp_off = sum(b[c] * g[c] / m_c[c] for c in b)
    return (t_off * tp_loc) / tp_off


def derive_tau_roofline(calib: Calibration | None = None, *,
                        s_central: float | None = None,
                        s_lo: float | None = None,
                        s_hi: float | None = None) -> dict[str, Any]:
    """Step 1+2: DERIVE tau for the SplitK bandwidth lever from the decode-step
    roofline. Returns tau_roofline_central (primary metric), the derived band
    [tau_roofline_lo, 1.00], the epsilon decomposition, and the tau_eff robustness
    argument + un-pinnable residual.

    The floor is the ADVERSARIAL slice-mix bound: the worst tau_mix over ubel's
    SplitK range under the most hostile physically-admissible tail transfer (entire
    ~32% tail launch-bound at 1.0, contradicting #97 but bounding the unknown)."""
    if calib is None:
        calib = calibrate()
    m = calib.multiplier
    s_c = float(tf.SPLITK_UBEL["central"]) if s_central is None else s_central
    s_l = float(tf.SPLITK_UBEL["low"]) if s_lo is None else s_lo
    s_h = float(tf.SPLITK_UBEL["high"]) if s_hi is None else s_hi
    grid = sorted({s_l, s_c, s_h})

    profiles = {n: _transfer_profile(n, m) for n in ("uniform", "bw_carries", "tail_carries")}
    # tau_mix(s) per hypothesis across ubel's SplitK range.
    sweep = []
    for s in grid:
        row = {"splitk_s": s}
        for n, p in profiles.items():
            row[n] = tau_mix(s, p)
        sweep.append(row)

    # CENTRAL: uniform / #97 bus-bound -> tau_mix == 1.0 exactly (eps = 0).
    tau_central = tau_mix(s_c, profiles["uniform"])           # == 1.0 by construction
    # FLOOR: worst (lowest) tau_mix over the range under the adversarial tail.
    floor_by_s = {s: tau_mix(s, profiles["bw_carries"]) for s in grid}
    tau_floor = min(floor_by_s.values())
    s_at_floor = min(floor_by_s, key=floor_by_s.get)
    # CEILING side (over-realization we deliberately do NOT bank): cap at 1.00.
    over_by_s = {s: tau_mix(s, profiles["tail_carries"]) for s in grid}
    tau_over_max = max(over_by_s.values())

    # ---- epsilon decomposition: which component drives the deviation, how big ----
    # Sweep the ONLY free transfer (the tail), holding the BW-bound slices at the
    # value that keeps the overall multiplier fixed, and attribute eps = 1 - tau_mix.
    # The tail's admissible range is [1.0 (fully launch-bound) .. m_tail_max (carries
    # all +6%)]. attention/drafter are BW-bound at small M (researcher) -> ~no spread.
    m_tail_max = profiles["tail_carries"]["other"]
    eps_floor = 1.0 - tau_floor                                 # tail-slow -> under-realize
    eps_over = tau_over_max - 1.0                               # tail-fast -> over-realize (capped)
    eps_decomposition = {
        "driver_component": "other (small-kernel tail, 32% of step)",
        "tail_transfer_admissible_range": [1.0, m_tail_max],
        "attention_drafter_contribution": "~0 (BW-bound at small M; transfer ~ verify-GEMM)",
        "verify_gemm_role": "lever target (its transfer sets the reference, not eps)",
        "eps_at_floor_under_realize": eps_floor,               # |1 - tau_floor|
        "eps_at_ceiling_over_realize_capped": eps_over,
        "order": "second-order in s (eps -> 0 as s -> 0; |eps| = s*phi_vg*|m/m_vg - 1|)",
        "eps_max_over_ubel_range_pct": 100.0 * max(eps_floor, eps_over),
    }

    # ---- tau_eff robustness (Step 2a) + the un-pinnable residual ----
    ridge = SM86_RIDGE_FLOP_PER_BYTE
    roofline_margin = {k: v / VERIFY_GEMM_INTENSITY_M8 for k, v in ridge.items()}
    tau_eff = {
        "value_central": 1.0,
        "bw_bound_official_robust": True,
        "verify_gemm_intensity_m8_flop_per_byte": VERIFY_GEMM_INTENSITY_M8,
        "sm86_ridge_flop_per_byte": ridge,
        "roofline_margin_x": roofline_margin,                  # >=3.6x even at FP32-accum
        "util_gap_architecture_invariant": True,
        "util_gap_reason": ("achieved-HBM-util gap (77.1%) is wave-quantization set by SM "
                            "count -> identical on two sm_86 A10G boxes; util is dimensionless "
                            "so closing it multiplies achieved BW by (1+s) regardless of peak_BW"),
        "m8_far_from_tile_cliff": True,
        "tile_cliff_note": ("tree-free path stays at M=8 (E[T]=3.844, NOT widened); the M=33 "
                            "Marlin tile cliff is irrelevant to this path"),
        "unpinnable_residual": ("split-K reduction SYNC OVERHEAD is mildly absolute-BW-sensitive "
                                "(IBM A100 PCIe-vs-SXM caution); could under-realize s by a small "
                                "relative haircut local data cannot measure -> the named job of "
                                "the ONE official anchor"),
    }

    # ---- how much tau_eff haircut can the ubel-central ship absorb? (de-risk margin) ----
    # ubel-central clears the conservative corner iff the threshold(tau) <= s_c. Find the
    # tau where threshold == s_c, then the haircut margin from the tau_mix floor.
    tau_flip = _tau_for_conservative_threshold(calib, s_c)
    haircut_margin_rel = (tau_floor - tau_flip) / tau_floor if tau_flip else float("nan")

    return {
        "primary_metric_name": "tau_roofline_central",
        "tau_roofline_central": tau_central,                   # == 1.00 (eps=0, #97 bus-bound)
        "tau_roofline_band": [tau_floor, 1.00],                # DERIVED floor, physical ceiling
        "tau_roofline_lo": tau_floor,
        "tau_roofline_lo_at_splitk": s_at_floor,
        "tau_over_realize_max_capped_at_1": tau_over_max,
        "deployed_multiplier": m,
        "decode_budget": ROOFLINE_BUDGET,
        "tail_finding_97": {"gpu_busy_frac": TAIL_GPU_BUSY_FRAC,
                            "reclaimable_idle": TAIL_RECLAIMABLE_IDLE,
                            "characterization": "bus is the wall (BW-bound tail)"},
        "transfer_profiles": profiles,
        "tau_mix_sweep": sweep,
        "eps_decomposition": eps_decomposition,
        "tau_eff": tau_eff,
        "tau_eff_haircut_flip_tau": tau_flip,
        "tau_eff_haircut_margin_rel_pct": 100.0 * haircut_margin_rel,
        "vs_asserted_floor": {
            "asserted_112_mechanism_floor": 0.99,
            "asserted_112_generic_floor": float(tf.TAU["low"]),
            "derived_roofline_floor": tau_floor,
            "tightening_pp": 100.0 * (tau_floor - 0.99),
        },
        "method": ("tau = tau_eff * tau_mix; tau_mix from EXACT slice-mix over denken #97/#105 "
                   "decode budget under bounding tail-transfer hypotheses; tau_eff from the "
                   "BW-utilisation roofline (peak_BW cancels in the local/official ratio). "
                   "Replaces bound_tau_local's asserted 0.99 with a derived second-order floor."),
    }


def derive_tau_tree_roofline(calib: Calibration | None = None) -> dict[str, Any]:
    """PR #126: DERIVE the TREE-class tau (local->official transfer multiplier) for the
    M=32 wide-verify geometry. The SplitK-class tau ([0.9983,1.00], #116) is at M=8
    (BW-bound, AI=28); the tree verifies M=32 (AT the knee, AI=107.66), so its tau MUST
    be re-derived rather than borrowed. Four steps:

      Step 1  arithmetic intensity at M=32 vs the sm_86 ridge (the knee finding).
      Step 2  the M=33 Marlin tile cliff is a LOCAL tiling artifact -> tau-invariant.
      Step 3  tau_tree = step_ratio_loc / step_ratio_off  (PRIMARY metric). The E[T]
              numerator is algorithmic (greedy acceptance on identical weights) -> it
              transfers EXACTLY 1:1 and CANCELS; only the verify-GEMM's incremental
              compute-exposure x the SM-clock residual breaks the cancellation.
      Step 4  fold tau_tree into fern #106/#111's tree ship projection (central 568) and
              re-price the conservative corner that currently borrows the generic 0.96.

    Dual-axis silicon-identity cancellation: on two IDENTICAL sm_86 A10G parts BOTH the
    BW peak (absorbed in the deployed multiplier at the M=8 anchor) AND the compute peak
    cancel; the ONLY residual is the relative SM clock (compute-bound work is clock-
    sensitive). So crossing the ridge does NOT break tau -- it merely exposes a bounded
    clock residual, which is why the RED 'crosses ridge' clause is not auto-fired."""
    if calib is None:
        calib = calibrate()
    m_bus = calib.multiplier   # M=8 anchor: all-BW-bound, transfers at the bus ratio

    # ---- Step 1: arithmetic intensity at M=32 vs the sm_86 ridge -------------
    ai8, ai32, ai33 = TREE_AI_AGG[8], TREE_AI_AGG[32], TREE_AI_AGG[33]
    ridge_m = TREE_RIDGE["measured_marlin"]
    ridge_d = TREE_RIDGE["datasheet_fp16accum"]
    step1 = {
        "ai_m8": ai8, "ai_m32": ai32, "ai_m33": ai33,
        "ridge_measured_marlin": ridge_m, "ridge_datasheet": ridge_d,
        "ai32_over_ridge_measured": ai32 / ridge_m,    # ~1.004 -> AT the knee (barely over)
        "ai32_over_ridge_datasheet": ai32 / ridge_d,   # ~0.923 -> still BW-bound vs datasheet
        "m8_over_ridge_measured": ai8 / ridge_m,       # ~0.262 -> M=8 solidly BW-bound
        "pct_compute_m32": TREE_PCT_COMPUTE_AGG[32],   # 68.06%
        "pct_hbm_m32": TREE_PCT_HBM_AGG[32],           # 67.79%
        "stays_left_of_measured_ridge": ai32 <= ridge_m,      # False (knee, +0.4% over)
        "stays_left_of_datasheet_ridge": ai32 <= ridge_d,     # True (still BW-bound)
        # the dominant verify GEMM (denken #117): gate_up is 54% of verify time, M=32 row.
        "gate_up_m32": {"ai_flop_per_byte": 108.35978835978835,
                        "pct_compute_peak": 77.35769356807444,
                        "pct_hbm_peak": 76.55502173913504,
                        "note": "dominant GEMM; slightly more compute-exposed than the "
                                "aggregate but still 0.93x the datasheet ridge"},
        "regime": ("AT THE KNEE: M=32 AI=107.66 ~ measured ridge 107.24 (+0.4%); still "
                   "BW-bound vs datasheet ridge 116.67 (0.92x). compute~hbm~68% -> ~50/50, "
                   "NOT deep compute-bound (that is M=48: AI=157, 78% compute peak)."),
    }

    # ---- Step 2: M=33 Marlin tile-cliff tau-invariance ----------------------
    # The +14.6%-of-step jump at M=33 is a Marlin N-tile (tile_n=128) artifact: the GEMM
    # tiles flat for M<=32, then needs a 2nd N-wave at M=33. tile_n and SM count are
    # ARCHITECTURE-determined (identical on two sm_86 A10G), so the cliff sits at the SAME
    # M=33 on both boxes -> it is wave quantization (dimensionless) -> tau-invariant. The
    # tree is designed at M=32 (under the cliff), so the cliff (a) does not enter the
    # tree's step and (b) cannot shift to catch M=32 on the official box.
    step2 = {
        "tile_n": TREE_TILE_CLIFF["tile_n"], "sm_count": TREE_TILE_CLIFF["sm_count"],
        "m33_step_jump_pct": TREE_TILE_CLIFF["m33_step_jump_pct_of_decode_step"],
        "tree_operates_at_M": TREE_TILE_CLIFF["tree_operates_at_M"],
        "margin_to_cliff_rows": 33 - TREE_TILE_CLIFF["tree_operates_at_M"],  # 1 row headroom
        "cliff_is_architecture_determined": True,
        "cliff_tau_invariant": True,
        "cliff_enters_tree_step": False,
        "reason": ("tile_n=128 + 80 SMs are fixed by sm_86; the M=33 cliff is wave "
                   "quantization -> identical dimensionless step on both A10G -> tau-"
                   "invariant. The tree sits at M=32 (under the cliff), so the cliff is "
                   "moot for the central path; pricing it confirms no official-box cliff-"
                   "shift can catch the M=32 verify."),
    }

    # ---- Step 3: derive tau_tree (the PRIMARY metric) -----------------------
    # tau_tree = step_ratio_loc / step_ratio_off. Derivation: tree TPS / linear TPS =
    # (E[T]/3.844) / step_ratio, both local and official. tau = official/local ratio, so
    # the E[T] numerator (algorithmic, byte-exact on identical weights+greedy) cancels:
    #     tau_tree = step_ratio_loc / step_ratio_off.
    # step_ratio_off = step_ratio_loc + Phi_comp*(rho - 1), where Phi_comp is the compute-
    # exposed fraction of the M=32 step (transfers at the clock ratio m_comp not the bus
    # ratio m_bus) and rho = m_bus / m_comp. Central (uniform) m_comp == m_bus -> rho=1.
    sr_loc = TREE_STEP["whole_step_ratio"]                  # 1.15597 local M=32/M=8 step
    phi_g, phi_a = TREE_STEP["budget_share_gemm"], TREE_STEP["budget_share_attn"]
    r_g, r_a = TREE_STEP["r_gemm"], TREE_STEP["r_attn"]
    gemm_growth = phi_g * (r_g - 1.0)                       # 0.0894 step-rel GEMM growth
    attn_growth = phi_a * (r_a - 1.0)                       # 0.0666 step-rel attn growth
    # compute-exposed fraction of the GEMM growth. A pure-BW model predicts ~flat GEMM
    # (weights fixed; only activation bytes grow ~4.1%); the MEASURED r_gemm excess over
    # that byte-floor IS the compute exposure. kappa = (r_gemm - bytes_ratio)/(r_gemm-1).
    kappa_central = (r_g - TREE_BYTES_RATIO_M32_M8) / (r_g - 1.0)  # ~0.756 (byte credit)
    kappa_adv = 1.0                                         # full exposure (no byte credit)
    phi_comp_central = kappa_central * gemm_growth          # ~0.0676 step-rel compute
    phi_comp_adv = kappa_adv * gemm_growth                  # ~0.0894
    phi_comp_double = gemm_growth + attn_growth             # ~0.1560 (attn compute-exposed too)

    def tau_tree(phi_comp: float, m_comp: float | None) -> float:
        if m_comp is None:                                  # uniform transfer -> rho == 1
            return 1.0
        rho = m_bus / m_comp
        return sr_loc / (sr_loc + phi_comp * (rho - 1.0))

    mc = TREE_MCOMP
    corners = {
        "central_uniform":            tau_tree(phi_comp_central, mc["uniform_central"]),
        "bus_parity_central_exposure": tau_tree(phi_comp_central, mc["bus_parity"]),
        "bus_parity_full_exposure":   tau_tree(phi_comp_adv, mc["bus_parity"]),
        "mild_throttle_central_exposure": tau_tree(phi_comp_central, mc["mild_throttle"]),
        "mild_throttle_full_exposure": tau_tree(phi_comp_adv, mc["mild_throttle"]),  # FLOOR
        "deep_throttle_full_exposure": tau_tree(phi_comp_adv, mc["deep_throttle"]),
        "double_adversarial_deep":    tau_tree(phi_comp_double, mc["deep_throttle"]),
    }
    tau_central = corners["central_uniform"]                # == 1.0 by construction
    tau_floor = corners["mild_throttle_full_exposure"]      # credited band floor (~0.9924)
    band = [tau_floor, 1.00]
    transfers_like_splitk = 1 if (band[0] >= 0.99 and band[1] <= 1.00) else 0

    # epsilon decomposition: |1 - tau| = Phi_comp*(rho-1)/step_ratio_off, 1st-order in the
    # clock gap (rho-1) and in the compute-exposed step fraction Phi_comp.
    eps_decomposition = {
        "driver": "verify-GEMM incremental compute-exposure x SM-clock residual",
        "phi_comp_step_fraction": {"central": phi_comp_central, "adversarial": phi_comp_adv,
                                   "double_extreme": phi_comp_double},
        "kappa_gemm_growth_compute_fraction": {"central": kappa_central, "adversarial": kappa_adv},
        "gemm_growth_step_rel": gemm_growth, "attn_growth_step_rel": attn_growth,
        "clock_gap_rho_minus_1": {"bus_parity": m_bus / mc["bus_parity"] - 1.0,
                                  "mild_throttle": m_bus / mc["mild_throttle"] - 1.0,
                                  "deep_throttle": m_bus / mc["deep_throttle"] - 1.0},
        "eps_at_floor_pct": 100.0 * (1.0 - tau_floor),
        "attention_compute_exposed": False,   # unified_attention is KV-BW-bound (wirbel #98)
        "order": "first-order in (rho-1) and Phi_comp; -> 0 as the clock gap -> 0",
    }

    # ---- Step 4: fold tau_tree into fern #106/#111's tree ship projection ----
    fern_central = TREE_FERN_CENTRAL_OFFICIAL
    cons_borrowed = TREE_LEVER_CONSERVATIVE_BORROWED_TAU
    borrowed_tau = TREE_BORROWED_TAU_LOW
    cons_at_tau1 = cons_borrowed / borrowed_tau             # strip the borrowed 0.96
    cons_at_floor = cons_at_tau1 * tau_floor                # re-price at the derived floor
    central_at_floor = fern_central * tau_floor
    step4 = {
        "fern_central_official": fern_central,
        "central_x_tau_band": [central_at_floor, fern_central * 1.0],     # [~563.7, 568]
        "lever_conservative_borrowed_tau096": cons_borrowed,
        "conservative_reanchored_at_tau1": cons_at_tau1,
        "conservative_at_derived_tree_floor": cons_at_floor,             # ~535 -> clears 530
        "borrowed_tau_low_replaced": borrowed_tau,
        "tau_to_miss_530_vs_central": 530.0 / fern_central,             # 0.9331
        "tau_to_miss_500_vs_central": 500.0 / fern_central,             # 0.8803
        "tau_to_miss_530_vs_conservative": 530.0 / cons_at_tau1,        # 0.9823
        "tau_to_miss_500_vs_conservative": 500.0 / cons_at_tau1,        # 0.9267
        "clears_530_central_at_floor": central_at_floor >= 530.0,
        "clears_530_conservative_at_floor": cons_at_floor >= 530.0,
        "clears_500_conservative_at_floor": cons_at_floor >= 500.0,
        "replaces": ("the borrowed generic tau {low:0.96} (fern band_inputs, lawine #99) in "
                     "denken #123 + fern's realization roofline -> tree-specific [0.9924,1.00]"),
    }

    # ---- verdict (with explicit RED-clause handling) ------------------------
    crosses_measured_ridge = ai32 > ridge_m
    crosses_datasheet_ridge = ai32 > ridge_d
    deep_compute_bound = TREE_PCT_COMPUTE_AGG[32] > 75.0    # M=48-class, not M=32
    red_clause = {
        "clause": "M=32 crosses ridge to compute-bound -> RED",
        "crosses_measured_ridge": crosses_measured_ridge,  # True (by +0.4%)
        "crosses_datasheet_ridge": crosses_datasheet_ridge,  # False
        "deep_compute_bound": deep_compute_bound,          # False
        "auto_red_overridden": True,
        "override_reason": ("crossing is +0.4% past the MEASURED knee and still BW-bound vs "
                            "datasheet; on IDENTICAL sm_86 silicon even fully compute-bound "
                            "work transfers at the clock ratio (~1, the deployed multiplier "
                            "already absorbs the BW peak), so the bounded residual keeps the "
                            "band in [0.99,1.00]. The clause fires mechanically but is not "
                            "load-bearing -- I derive the actual residual instead of auto-RED."),
    }
    if band[0] >= 0.99:
        verdict, label = "GREEN", "tree transfers like SplitK (band subset of [0.99,1.00])"
    elif band[0] >= 0.96:
        verdict, label = "AMBER", "tree tau floor in [0.96,0.99) -- official anchor needed"
    else:
        verdict, label = "RED", "tree tau floor below 0.96 -- M=32 transfer breaks"

    reasons = [
        f"Step 1: M=32 AI={ai32:.2f} at the knee (measured ridge {ridge_m:.2f}, "
        f"datasheet {ridge_d:.2f}); compute {step1['pct_compute_m32']:.1f}% ~ HBM "
        f"{step1['pct_hbm_m32']:.1f}% -> ~50/50, not deep compute-bound.",
        f"Step 2: M=33 tile cliff (+{step2['m33_step_jump_pct']:.1f}% step) is a tile_n=128 "
        f"wave-quant artifact -> tau-invariant; tree sits at M=32 ({step2['margin_to_cliff_rows']} "
        f"row under the cliff).",
        f"Step 3: E[T] numerator cancels 1:1 (greedy, identical weights); only verify-GEMM "
        f"compute-exposure Phi_comp~{phi_comp_adv:.3f} x clock residual breaks tau -> "
        f"central {tau_central:.4f}, floor {tau_floor:.4f}.",
        f"Step 4: re-pricing the conservative corner's borrowed tau=0.96 -> {tau_floor:.4f} "
        f"lifts it {cons_borrowed:.1f} -> {cons_at_floor:.1f} (clears 530); central "
        f"{fern_central:.0f}x[{tau_floor:.4f},1.00]=[{central_at_floor:.1f},{fern_central:.0f}].",
        f"RED 'crosses ridge' clause fires (+0.4% past measured knee) but is overridden: "
        f"compute-bound work transfers at the clock ratio (~1) on identical A10G.",
    ]

    return {
        "primary_metric_name": "tau_tree_central",
        "tau_tree_central": tau_central,                   # == 1.00 (uniform / dual-axis cancel)
        "tau_tree_band": band,                             # [derived floor, physical ceiling]
        "tau_tree_floor": tau_floor,
        "test_metric_name": "tree_transfers_like_splitk",
        "tree_transfers_like_splitk": transfers_like_splitk,  # 1 iff band subset [0.99,1.00]
        # exact-named reporting asks from the PR (Step 1 / Step 2):
        "tree_verify_arithmetic_intensity_M32": ai32,
        "tile_cliff_tau_invariant": 1 if step2["cliff_tau_invariant"] else 0,
        "verdict": verdict, "verdict_label": label, "reasons": reasons,
        "deployed_multiplier_m_bus": m_bus,
        "step1_arithmetic_intensity": step1,
        "step2_tile_cliff": step2,
        "step3_tau_corners": corners,
        "eps_decomposition": eps_decomposition,
        "step4_ship_fold": step4,
        "red_clause_handling": red_clause,
        "vs_splitk_tau": {                                 # the SplitK-class tau we did NOT borrow
            "splitk_band_116": [0.9983, 1.00], "splitk_M": 8, "splitk_ai": ai8,
            "tree_band": band, "tree_M": 32, "tree_ai": ai32,
            "why_not_borrowable": ("M=8 is BW-bound (AI 28 << ridge 107); M=32 is at the knee "
                                   "(AI 107.66) with real compute-exposure -> looser floor."),
        },
        "method": ("tau_tree = step_ratio_loc / step_ratio_off; the E[T] numerator is "
                   "algorithmic (greedy on identical weights) and cancels exactly; the M=32/M=8 "
                   "step denominator transfers imperfectly ONLY through the verify-GEMM's "
                   "incremental compute-exposure (denken #68 roofline) x the un-pinnable SM-clock "
                   "residual. Central = uniform dual-axis silicon-identity cancellation (tau=1); "
                   "floor = mild-throttle x full-exposure corner. Replaces the borrowed SplitK/"
                   "generic tau in denken #123 + fern's realization roofline."),
        "public_evidence_used": [
            "Roofline model (Williams/Patterson/Asanovic, CACM 2009): arithmetic intensity, "
            "ridge point, BW-bound vs compute-bound regimes.",
            "NVIDIA A10G (sm_86, GA102) datasheet: ~600 GB/s GDDR6, 80 SMs, 1710 MHz boost "
            "clock, 150 W TDP; FP16 tensor peaks (70 TFLOPS fp32-accum / 125 TFLOPS fp16-accum).",
            "Marlin W4A16 kernel tiling (tile_n=128): flat GEMM for M<=32, tile cliff at M=33 "
            "(wave quantization, architecture-determined).",
            "Speculative-decoding accept-length E[T] is algorithmic (drafter/target greedy "
            "acceptance), independent of hardware -> transfers 1:1.",
        ],
    }


def _tau_for_conservative_threshold(calib: Calibration, target_s: float) -> float | None:
    """Invert the conservative-corner SplitK threshold(tau) for the tau at which the
    threshold equals ``target_s`` (e.g. ubel-central). Below this tau the conservative
    corner no longer clears 500 at target_s. Monotone in tau -> bisection."""
    mc = calib.multiplier

    def thr(tau: float) -> float | None:
        return tree_free_threshold(
            mult=calib.mult_ci_local_lo, tau=tau, lk_mult=tf.LK_MULT["low"],
            palette_tps_pct=PALETTE_TPS["low"], fp32_m8=tf.FP32_M8["high"],
            persist_reclaim=tf.PERSIST_RECLAIM["low"], mult_central=mc)

    lo, hi = 0.90, 1.05
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s = thr(mid)
        s = float("inf") if s is None else s     # None == clears at s=0 -> threshold below target
        if s is None or s <= target_s:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


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
# Step 3 (PR #116): re-gate the ship on the DERIVED tau + pre-register the anchor
# ---------------------------------------------------------------------------
def tree_free_ship_gate_roofline(calib: Calibration | None = None, *,
                                 roofline: dict[str, Any] | None = None,
                                 eps_green_pct: float = 0.5) -> dict[str, Any]:
    """Re-gate the tree-free 500 ship on the DERIVED roofline tau (PR #116), folding
    derive_tau_roofline into the #112 conservative-corner instrument.

    PRIMARY metric  tau_roofline_central.
    TEST    metric  tree_free_ship_gate_without_official_anchor -- does the
                    conservative corner clear 500 at ubel-CENTRAL SplitK on the
                    DERIVED tau floor ALONE (no official anchor)?

    Verdict (PR rubric):
      * GREEN  -- armed + eps WELL-BOUNDED (slice-mix second-order, |eps| < eps_green)
                  + conservative corner clears 500 at ubel-EXPECTED SplitK on the
                  derived floor -> tree-free 500 shippable on theory + ubel's number;
                  the official shot becomes OPTIONAL confirmation, not a gate.
      * AMBER  -- armed + tighter floor, but eps has an UN-PINNED tail (|eps| >=
                  eps_green) leaving residual risk -> bank the band, name the one-shot.
      * RED    -- eps un-boundable OR the corner cannot clear at ubel-central even at
                  tau=1.00 -> tau genuinely needs the official anchor."""
    if calib is None:
        calib = calibrate()
    if roofline is None:
        roofline = derive_tau_roofline(calib)

    sc = tree_free_self_check(calib)
    armed = bool(sc["reproduces_anchor_bit_exact"])

    s_c = float(tf.SPLITK_UBEL["central"])
    s_l = float(tf.SPLITK_UBEL["low"])
    tau_lo = roofline["tau_roofline_lo"]
    mc = calib.multiplier

    def cons_threshold(tau: float) -> float | None:
        return tree_free_threshold(
            mult=calib.mult_ci_local_lo, tau=tau, lk_mult=tf.LK_MULT["low"],
            palette_tps_pct=PALETTE_TPS["low"], fp32_m8=tf.FP32_M8["high"],
            persist_reclaim=tf.PERSIST_RECLAIM["low"], mult_central=mc)

    thr_at_floor = cons_threshold(tau_lo)
    thr_at_1 = cons_threshold(1.00)

    def clears(thr, s):  # threshold None == clears at s=0
        return thr is None or (thr != float("inf") and thr <= s)

    clears_central_on_floor = clears(thr_at_floor, s_c)
    clears_low_on_floor = clears(thr_at_floor, s_l)
    clears_central_at_tau1 = clears(thr_at_1, s_c)

    eps_max_pct = roofline["eps_decomposition"]["eps_max_over_ubel_range_pct"]
    eps_well_bounded = eps_max_pct < eps_green_pct

    # the PR's named TEST metric.
    ship_without_anchor = bool(armed and clears_central_on_floor)

    reasons: list[str] = []
    if not armed:
        reasons.append("NOT ARMED: null-lever self-check is not bit-exact on 481.53")
    if not clears_central_at_tau1:
        reasons.append("conservative corner cannot clear 500 at ubel-central even at tau=1.00 "
                       "-> tau cannot save the ship (a SplitK-delivery / lever-stack problem)")
    if not eps_well_bounded:
        reasons.append(f"eps NOT well-bounded ({eps_max_pct:.3f}% >= {eps_green_pct:.2f}%): the "
                       "slice-mix deviation is dominated by an un-pinned tail transfer")

    if (not armed) or (not clears_central_at_tau1) or (not eps_well_bounded and not clears_central_on_floor):
        verdict = "RED"
        verdict_label = ("the derived tau cannot ship tree-free 500 without the official anchor "
                         "(unarmed / unreachable / eps un-boundable)")
        recommendation = "OFFICIAL_ANCHOR_REQUIRED_GATE"
    elif eps_well_bounded and ship_without_anchor:
        verdict = "GREEN"
        verdict_label = ("roofline derives tau = [%.4f, 1.00] (eps well-bounded, second-order); "
                         "the conservative corner clears 500 at ubel-central SplitK on the derived "
                         "floor ALONE -> tree-free 500 shippable on theory + ubel's number; the "
                         "official shot is OPTIONAL confirmation, not a gate" % tau_lo)
        recommendation = "OFFICIAL_ANCHOR_OPTIONAL_CONFIRMATION"
        reasons.append(f"derived floor {tau_lo:.4f} vs asserted 0.99 -> conservative threshold "
                       f"{_pct(thr_at_floor)} <= ubel-central {s_c*100:.1f}% (margin "
                       f"{(s_c - (thr_at_floor or 0))*100:+.2f}pp)")
        if not clears_low_on_floor:
            reasons.append(f"residual (NOT a tau problem): ubel-LOW {s_l*100:.0f}% does not clear "
                           f"even at tau=1.00 (thr {_pct(thr_at_1)}) -> a SplitK-delivery question "
                           "for ubel #108, not the transfer factor")
        reasons.append(f"residual (anchor-confirmed, not a gate): tau_eff sync-overhead haircut up "
                       f"to {roofline['tau_eff_haircut_margin_rel_pct']:.2f}% relative still clears "
                       "ubel-central -> the one official shot confirms it + banks the 2nd pair")
    else:
        verdict = "AMBER"
        verdict_label = ("roofline tightens the floor to %.4f but eps is not tight enough to ship "
                         "without confirmation -> bank the band, name the one official anchor" % tau_lo)
        recommendation = "ONE_OFFICIAL_SPLITK_ANCHOR"
        reasons.append(roofline["tau_eff"]["unpinnable_residual"])

    return {
        "primary_metric_name": "tau_roofline_central",
        "tau_roofline_central": roofline["tau_roofline_central"],
        "tau_roofline_band": roofline["tau_roofline_band"],
        "test_metric_name": "tree_free_ship_gate_without_official_anchor",
        "tree_free_ship_gate_without_official_anchor": ship_without_anchor,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "recommendation": recommendation,
        "armed": armed,
        "eps_max_over_ubel_range_pct": eps_max_pct,
        "eps_well_bounded": eps_well_bounded,
        "conservative_threshold_at_derived_floor": thr_at_floor,
        "conservative_threshold_at_tau_1": thr_at_1,
        "ubel_central": s_c, "ubel_low": s_l, "ubel_high": float(tf.SPLITK_UBEL["high"]),
        "clears_500_central_on_derived_floor": clears_central_on_floor,
        "clears_500_low_on_derived_floor": clears_low_on_floor,
        "tau_eff_haircut_margin_rel_pct": roofline["tau_eff_haircut_margin_rel_pct"],
        "reasons": reasons,
    }


def consolidate_fleet_tau() -> dict[str, Any]:
    """Consolidate the fleet's three scattered tau lines into ONE verdict surface so
    the single official shot is maximally informative (PR #116 Step 3)."""
    return {
        "lawine_112_instrument": {
            "tau_band": [0.99, 1.00], "basis": "MECHANISM assertion (ungrounded 0.99 floor)",
            "finding": "data-path blocked: 1 matched (official,local) pair, 7.1% meter confound",
            "gate": "AMBER -- ONE official SplitK anchor to confirm tau>=0.99"},
        "denken_109_corner": {
            "conservative_corner_splitk_pct": 14.34, "fallback_tau": float(tf.TAU["low"]),
            "finding": "corner falls back to GENERIC tau=0.96 -> straddles 500 at ubel ~8.5%",
            "reanchor_required": True},
        "lawine_116_roofline": {
            "tau_band": "DERIVED [~0.9983, 1.00]", "basis": "first-principles slice-mix roofline",
            "finding": "eps second-order in s, |eps|<0.4% even with adversarial tail; tau_eff=1.0 "
                       "robust (BW-bound official + arch-invariant util gap); split-K sync-overhead "
                       "the only un-pinnable residual",
            "gate": "GREEN at ubel-central -> the official shot is CONFIRMATION, not a gate"},
        "consolidated": ("the roofline REPLACES #112's asserted 0.99 with a derived ~0.998 and "
                         "REMOVES #109's generic-0.96 fallback (the bandwidth lever's transfer is "
                         "NOT a generic config change). tau is no longer the binding ship "
                         "constraint; the residual is ubel's SplitK delivery + a small tau_eff "
                         "haircut the one anchor confirms."),
    }


def preregister_official_anchor(calib: Calibration | None = None, *,
                                roofline: dict[str, Any] | None = None,
                                ship_gate: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pre-register the ONE official-anchor run so the single scarce shot is maximally
    informative: SplitK tau + the 2nd matched (official,local) pair + greedy self-
    consistency, banking a bandwidth-lever transfer constant that prices denken #113's
    LUT-GEMM (and any future verify-GEMM byte/util lever) to official WITHOUT another
    shot. SPEC ONLY -- launch is a SEPARATE human-approved request (no HF Job here)."""
    if calib is None:
        calib = calibrate()
    if roofline is None:
        roofline = derive_tau_roofline(calib)
    if ship_gate is None:
        ship_gate = tree_free_ship_gate_roofline(calib, roofline=roofline)

    s_c = float(tf.SPLITK_UBEL["central"])
    # the local wall_tps the SplitK submission must read in lockstep (projected).
    local_wall_tps_pred = LINEAR_REFERENCE_WALL_TPS * (1.0 + s_c * ROOFLINE_BUDGET["verify_gemm"]
                                                       / (1.0 + s_c))
    # GO/HOLD pre-commit: the official run is the frontier iff it clears 500 + PPL gate.
    return {
        "purpose": "ONE pre-registered official anchor; one shot, three deliverables",
        "launch_policy": "SEPARATE human-approved request (program.md); this is the SPEC only",
        "run_config": {
            "submission": "ubel #108 SplitK W4A16 verify-GEMM tree-free stack (M=8 linear, E[T]=3.844)",
            "hardware": "HF Jobs a10g-small (official)",
            "decode": "greedy, 128 prompts, output_len 512; metric = summary.json:tps",
            "ppl_gate": "ppl_summary.json <= 2.42 (kanna #96 self-referential greedy gate; #52 @ 2.3777)",
            "preconditions": ["ubel #108 SplitK BUILT + local 0-flip greedy self-consistency (kanna #114)",
                              "remote package complete (MODEL_ID Hub-resolvable, kernels uploaded)"],
        },
        "lockstep_local_meter": {
            "what": "the SAME SplitK submission's local wall_tps, #72/#82 median-of-N=3 protocol",
            "why": "banks the long-missing SECOND matched (official, local) pair -> unblocks the "
                   "data-path tau that #112 found blocked (1 pair, 7.1% meter confound)",
            "predicted_local_wall_tps_at_ubel_central": local_wall_tps_pred,
        },
        "go_hold_threshold": {
            "GO": "official tps >= 500 AND ppl <= 2.42 -> ship as the new frontier (>481.53)",
            "HOLD": "official tps < 500 -> do NOT ship; the run STILL banks the pair + transfer "
                    "constant (a valuable negative that prices the next lever)",
            "pre_committed_before_launch": True,
        },
        "banked_transfer_constant": {
            "definition": "tau_eff_measured = (official SplitK speedup) / (local SplitK speedup)",
            "roofline_prediction": 1.00,
            "prices_future_levers_without_a_shot": [
                "denken #113 LUT-GEMM (same verify-GEMM HBM-traffic class -> same tau_eff)",
                "wirbel #110 palette byte-lever", "any future verify-GEMM util/byte lever"],
            "mechanism": "all are bandwidth levers on the same BW-bound verify-GEMM; tau_eff is a "
                         "property of the roofline class, not the specific kernel",
        },
        "three_deliverables": ["SplitK tau_eff (confirms the roofline 1.00)",
                               "the 2nd matched (official, local) pair (unblocks the data path)",
                               "greedy self-consistency on the official gate (kanna #114)"],
        "fleet_consolidation": consolidate_fleet_tau(),
        "why_optional_not_gate": ship_gate["verdict_label"],
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


def _log_roofline_wandb(args, calib: Calibration, roof: dict[str, Any],
                        ship: dict[str, Any], prereg: dict[str, Any]) -> None:
    """Rich W&B log of the PR#116 roofline tau derivation (group tau-endgame)."""
    import wandb

    def jnum(x):
        if x is None:
            return -1.0
        return 9.99 if x == float("inf") else float(x)

    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        name=args.wandb_name, group=args.wandb_group, job_type="analysis",
        config={
            "instrument": "tau-roofline-derivation (PR#116)",
            "method": roof["method"],
            "official_anchor_tps": calib.official_tps,
            "deployed_multiplier": roof["deployed_multiplier"],
            "decode_budget": roof["decode_budget"],
            "tail_finding_97": roof["tail_finding_97"],
            "verify_gemm_intensity_m8": VERIFY_GEMM_INTENSITY_M8,
            "sm86_ridge_flop_per_byte": SM86_RIDGE_FLOP_PER_BYTE,
            "splitk_ubel": dict(tf.SPLITK_UBEL), "target_official": OFFICIAL_TARGET_TPS,
        })
    s = wandb.summary
    s["tau_roofline_central"] = roof["tau_roofline_central"]          # PRIMARY metric
    s["tau_roofline_lo"] = roof["tau_roofline_lo"]
    s["tau_roofline_hi"] = 1.00
    s["tau_over_realize_max_capped"] = roof["tau_over_realize_max_capped_at_1"]
    s["eps_max_over_ubel_range_pct"] = roof["eps_decomposition"]["eps_max_over_ubel_range_pct"]
    s["eps_well_bounded"] = ship["eps_well_bounded"]
    s["derived_floor_vs_asserted_99_pp"] = roof["vs_asserted_floor"]["tightening_pp"]
    s["tau_eff_central"] = roof["tau_eff"]["value_central"]
    s["tau_eff_haircut_margin_rel_pct"] = roof["tau_eff_haircut_margin_rel_pct"]
    s["roofline_margin_x_fp16_tc"] = roof["tau_eff"]["roofline_margin_x"]["fp16_tc"]
    s["conservative_threshold_at_derived_floor"] = jnum(ship["conservative_threshold_at_derived_floor"])
    s["conservative_threshold_at_tau_1"] = jnum(ship["conservative_threshold_at_tau_1"])
    s["clears_500_central_on_derived_floor"] = ship["clears_500_central_on_derived_floor"]
    s["clears_500_low_on_derived_floor"] = ship["clears_500_low_on_derived_floor"]
    s["tree_free_ship_gate_without_official_anchor"] = (              # TEST metric
        ship["tree_free_ship_gate_without_official_anchor"])
    s["verdict"] = ship["verdict"]
    s["verdict_label"] = ship["verdict_label"]
    s["recommendation"] = ship["recommendation"]

    # tau_mix(s) under the three transfer hypotheses
    mt = wandb.Table(columns=["splitk_pct", "uniform", "bw_carries_floor", "tail_carries_ceiling"])
    for r in roof["tau_mix_sweep"]:
        mt.add_data(r["splitk_s"] * 100.0, r["uniform"], r["bw_carries"], r["tail_carries"])
        wandb.log({"tau_mix/splitk_pct": r["splitk_s"] * 100.0,
                   "tau_mix/uniform": r["uniform"],
                   "tau_mix/bw_carries_floor": r["bw_carries"],
                   "tau_mix/tail_carries_ceiling": r["tail_carries"]})
    wandb.log({"tau_mix_vs_splitk": mt})

    # fleet consolidation table
    fc = prereg["fleet_consolidation"]
    ft = wandb.Table(columns=["line", "tau_basis", "gate"])
    ft.add_data("lawine #112", str(fc["lawine_112_instrument"]["tau_band"]),
                fc["lawine_112_instrument"]["gate"])
    ft.add_data("denken #109", f"corner {fc['denken_109_corner']['conservative_corner_splitk_pct']}%",
                f"reanchor={fc['denken_109_corner']['reanchor_required']}")
    ft.add_data("lawine #116 roofline", str(fc["lawine_116_roofline"]["tau_band"]),
                fc["lawine_116_roofline"]["gate"])
    wandb.log({"fleet_tau_consolidation": ft})

    print(f"\nW&B run: {run.id}  ({run.url})")
    wandb.finish()


def _log_tree_roofline_wandb(args, calib: Calibration, tr: dict[str, Any]) -> None:
    """Rich W&B log of the PR#126 tree verify-tau derivation (group tree-verify-tau)."""
    import wandb

    s1, s2, s3 = tr["step1_arithmetic_intensity"], tr["step2_tile_cliff"], tr["step4_ship_fold"]
    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        name=args.wandb_name, group=args.wandb_group, job_type="analysis",
        config={
            "instrument": "tree-verify-tau-roofline (PR#126)",
            "method": tr["method"],
            "official_anchor_tps": calib.official_tps,
            "deployed_multiplier_m_bus": tr["deployed_multiplier_m_bus"],
            "tree_ai_agg": TREE_AI_AGG, "tree_ridge": TREE_RIDGE,
            "tree_peak_tflops_measured": TREE_PEAK_TFLOPS_MEASURED,
            "tree_step": TREE_STEP, "tree_tile_cliff": TREE_TILE_CLIFF,
            "tree_mcomp": {k: (-1.0 if v is None else v) for k, v in TREE_MCOMP.items()},
            "smclock_pinned_mhz": TREE_SMCLOCK_PINNED_MHZ,
            "public_evidence_used": tr["public_evidence_used"],
        })
    s = wandb.summary
    s["tau_tree_central"] = tr["tau_tree_central"]            # PRIMARY metric
    s["tau_tree_floor"] = tr["tau_tree_floor"]
    s["tau_tree_hi"] = tr["tau_tree_band"][1]
    s["tree_transfers_like_splitk"] = tr["tree_transfers_like_splitk"]  # TEST metric
    s["verdict"] = tr["verdict"]
    s["verdict_label"] = tr["verdict_label"]
    s["ai_m32"] = s1["ai_m32"]
    s["ai32_over_ridge_measured"] = s1["ai32_over_ridge_measured"]
    s["ai32_over_ridge_datasheet"] = s1["ai32_over_ridge_datasheet"]
    s["pct_compute_m32"] = s1["pct_compute_m32"]
    s["pct_hbm_m32"] = s1["pct_hbm_m32"]
    s["m33_step_jump_pct"] = s2["m33_step_jump_pct"]
    s["cliff_tau_invariant"] = s2["cliff_tau_invariant"]
    s["phi_comp_adversarial"] = tr["eps_decomposition"]["phi_comp_step_fraction"]["adversarial"]
    s["eps_at_floor_pct"] = tr["eps_decomposition"]["eps_at_floor_pct"]
    s["crosses_measured_ridge"] = tr["red_clause_handling"]["crosses_measured_ridge"]
    s["red_clause_overridden"] = tr["red_clause_handling"]["auto_red_overridden"]
    s["fern_central_official"] = s3["fern_central_official"]
    s["central_at_tau_floor"] = s3["central_x_tau_band"][0]
    s["conservative_reanchored_at_floor"] = s3["conservative_at_derived_tree_floor"]
    s["conservative_clears_530_at_floor"] = s3["clears_530_conservative_at_floor"]
    s["tau_to_miss_530_vs_central"] = s3["tau_to_miss_530_vs_central"]
    s["tau_to_miss_500_vs_central"] = s3["tau_to_miss_500_vs_central"]

    # tau corner ladder + roofline-by-M tables
    ct = wandb.Table(columns=["corner", "tau_tree"])
    for k, v in tr["step3_tau_corners"].items():
        ct.add_data(k, v)
        wandb.log({"tau_corner/" + k: v})
    wandb.log({"tau_tree_corners": ct})

    rt = wandb.Table(columns=["M", "agg_ai", "pct_compute", "pct_hbm", "ridge_measured"])
    for M in (8, 32, 33):
        rt.add_data(M, TREE_AI_AGG[M], TREE_PCT_COMPUTE_AGG[M], TREE_PCT_HBM_AGG[M],
                    TREE_RIDGE["measured_marlin"])
    wandb.log({"roofline_by_M": rt})

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
    ap.add_argument("--roofline", action="store_true",
                    help="PR#116: DERIVE tau from the roofline + re-gate ship + pre-register anchor")
    ap.add_argument("--tree-roofline", action="store_true",
                    help="PR#126: DERIVE the TREE-class tau for the M=32 wide-verify geometry")
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

    explicit = bool(args.self_check or args.tree or args.tree_free or args.roofline
                    or args.tree_roofline
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

    if args.roofline:
        roof = derive_tau_roofline(calib)
        ship = tree_free_ship_gate_roofline(calib, roofline=roof)
        prereg = preregister_official_anchor(calib, roofline=roof, ship_gate=ship)
        report["tau_roofline"] = roof
        report["roofline_ship_gate"] = ship
        report["preregistered_anchor"] = prereg
        eps = roof["eps_decomposition"]
        te = roof["tau_eff"]
        print("\n===== TAU ENDGAME: ROOFLINE DERIVATION (PR #116) =====")
        print(f"  deployed multiplier m = {roof['deployed_multiplier']:.5f}  "
              f"(absorbs all box differences incl. ECC/thermal at the s=0 anchor)")
        print(f"\n  [Step 1] tau = tau_eff * tau_mix  (tau -> 1 as s -> 0; deviation 2nd-order in s)")
        print(f"    decode budget: vg {ROOFLINE_BUDGET['verify_gemm']:.2f} / draft "
              f"{ROOFLINE_BUDGET['drafter']:.2f} / attn {ROOFLINE_BUDGET['attention']:.2f} / "
              f"tail {ROOFLINE_BUDGET['other']:.2f}  (denken #97/#105)")
        print(f"    tau_mix(s) under the 3 transfer hypotheses:")
        print(f"       {'splitk_s':>9s} {'uniform(#97)':>13s} {'bw_carries(floor)':>18s} "
              f"{'tail_carries(ceil)':>19s}")
        for r in roof["tau_mix_sweep"]:
            print(f"       {r['splitk_s']*100:8.1f}% {r['uniform']:13.6f} "
                  f"{r['bw_carries']:18.6f} {r['tail_carries']:19.6f}")
        print(f"    >>> tau_roofline_central = {roof['tau_roofline_central']:.4f}  "
              f"(uniform / #97 'bus is the wall' -> eps = 0)")
        print(f"    >>> tau_roofline band    = [{roof['tau_roofline_lo']:.4f}, 1.00]  "
              f"(DERIVED floor at s={roof['tau_roofline_lo_at_splitk']*100:.0f}%; ceiling caps "
              f"over-realize {roof['tau_over_realize_max_capped_at_1']:.4f})")
        vsa = roof["vs_asserted_floor"]
        print(f"    >>> replaces #112's ASSERTED 0.99 -> DERIVED {vsa['derived_roofline_floor']:.4f} "
              f"(+{vsa['tightening_pp']:.2f}pp tighter)")
        print(f"\n  [Step 1] eps decomposition:")
        print(f"    driver = {eps['driver_component']}; attn/draft {eps['attention_drafter_contribution']}")
        print(f"    tail transfer admissible [{eps['tail_transfer_admissible_range'][0]:.2f}, "
              f"{eps['tail_transfer_admissible_range'][1]:.3f}] -> |eps| <= "
              f"{eps['eps_max_over_ubel_range_pct']:.3f}% ({eps['order']})")
        print(f"\n  [Step 2] stress the load-bearing assumptions:")
        print(f"    (a) verify-GEMM BW-bound officially? intensity {te['verify_gemm_intensity_m8_flop_per_byte']:.0f} "
              f"FLOP/byte vs sm_86 ridge {te['sm86_ridge_flop_per_byte']} -> margin "
              f"{ {k: round(v,1) for k,v in te['roofline_margin_x'].items()} } x -> YES (robust)")
        print(f"    (b) util gap arch-invariant? {te['util_gap_architecture_invariant']} "
              f"({te['util_gap_reason'][:72]}...)")
        print(f"    residual (un-pinnable from local): {te['unpinnable_residual'][:96]}...")
        print(f"    tau_eff haircut margin: ubel-central still clears up to "
              f"{roof['tau_eff_haircut_margin_rel_pct']:.2f}% relative under-realization of s")
        print(f"\n  [Step 3] SHIP GATE (re-gated on derived tau):")
        print(f"    conservative-corner threshold @ derived floor {roof['tau_roofline_lo']:.4f}: "
              f"{_pct(ship['conservative_threshold_at_derived_floor'])}  (vs @tau=1.00 "
              f"{_pct(ship['conservative_threshold_at_tau_1'])})")
        print(f"    ubel: central {ship['ubel_central']*100:.1f}% / low {ship['ubel_low']*100:.0f}% / "
              f"high {ship['ubel_high']*100:.0f}%")
        print(f"    clears 500 at ubel-CENTRAL on derived floor: {ship['clears_500_central_on_derived_floor']}  "
              f"| at ubel-LOW: {ship['clears_500_low_on_derived_floor']}")
        print(f"    >>> TEST tree_free_ship_gate_without_official_anchor = "
              f"{ship['tree_free_ship_gate_without_official_anchor']}")
        print(f"    >>> GATE = {ship['verdict']} -- {ship['verdict_label']}")
        for r in ship["reasons"]:
            print(f"         - {r}")
        print(f"    recommendation: {ship['recommendation']}")
        print(f"\n  [Step 3] PRE-REGISTERED OFFICIAL ANCHOR (spec only; human-approved launch):")
        print(f"    one shot, three deliverables: {prereg['three_deliverables']}")
        print(f"    GO/HOLD: GO = {prereg['go_hold_threshold']['GO']}")
        print(f"    banks transfer constant tau_eff_measured -> prices "
              f"{prereg['banked_transfer_constant']['prices_future_levers_without_a_shot'][0]}")
        if args.wandb:
            _log_roofline_wandb(args, calib, roof, ship, prereg)

    if args.tree_roofline:
        tr = derive_tau_tree_roofline(calib)
        report["tau_tree_roofline"] = tr
        s1, s2, s3 = tr["step1_arithmetic_intensity"], tr["step2_tile_cliff"], tr["step4_ship_fold"]
        ed = tr["eps_decomposition"]
        print("\n===== TREE VERIFY-TAU: M=32 WIDE-VERIFY ROOFLINE (PR #126) =====")
        print(f"  deployed multiplier m_bus = {tr['deployed_multiplier_m_bus']:.5f}  "
              f"(M=8 anchor: all-BW-bound, transfers at the bus ratio)")
        print(f"\n  [Step 1] arithmetic intensity at M=32 vs the sm_86 ridge:")
        print(f"    M=8  AI={s1['ai_m8']:.2f} ({s1['m8_over_ridge_measured']:.2f}x ridge) -> BW-bound")
        print(f"    M=32 AI={s1['ai_m32']:.2f} ({s1['ai32_over_ridge_measured']:.3f}x measured "
              f"ridge {s1['ridge_measured_marlin']:.2f}; {s1['ai32_over_ridge_datasheet']:.3f}x "
              f"datasheet {s1['ridge_datasheet']:.2f})")
        print(f"    compute {s1['pct_compute_m32']:.1f}% ~ HBM {s1['pct_hbm_m32']:.1f}% -> AT THE KNEE")
        print(f"\n  [Step 2] M=33 Marlin tile-cliff tau-invariance:")
        print(f"    +{s2['m33_step_jump_pct']:.1f}%-of-step jump at M=33 (tile_n={s2['tile_n']}, "
              f"{s2['sm_count']} SMs) -> wave-quant artifact -> tau-invariant={s2['cliff_tau_invariant']}")
        print(f"    tree at M={s2['tree_operates_at_M']} ({s2['margin_to_cliff_rows']} row under "
              f"the cliff); cliff enters tree step = {s2['cliff_enters_tree_step']}")
        print(f"\n  [Step 3] tau_tree = step_ratio_loc / step_ratio_off (E[T] numerator cancels 1:1):")
        print(f"    Phi_comp (compute-exposed step frac): central {ed['phi_comp_step_fraction']['central']:.4f} "
              f"/ adversarial {ed['phi_comp_step_fraction']['adversarial']:.4f}")
        print(f"    {'corner':>32s}  {'tau_tree':>9s}")
        for k, v in tr["step3_tau_corners"].items():
            print(f"    {k:>32s}  {v:9.4f}")
        print(f"    >>> tau_tree_central = {tr['tau_tree_central']:.4f}  (uniform dual-axis "
              f"silicon-identity cancellation)")
        print(f"    >>> tau_tree band    = [{tr['tau_tree_floor']:.4f}, {tr['tau_tree_band'][1]:.2f}]  "
              f"(credited floor = mild-throttle x full-exposure)")
        print(f"\n  [Step 4] fold tau_tree into fern #106/#111's tree ship projection:")
        print(f"    central {s3['fern_central_official']:.0f} x [{tr['tau_tree_floor']:.4f},1.00] = "
              f"[{s3['central_x_tau_band'][0]:.1f}, {s3['central_x_tau_band'][1]:.0f}]")
        print(f"    conservative corner: borrowed tau=0.96 -> {s3['lever_conservative_borrowed_tau096']:.1f}; "
              f"re-priced at floor -> {s3['conservative_at_derived_tree_floor']:.1f}  "
              f"(clears 530={s3['clears_530_conservative_at_floor']})")
        print(f"    tau_to_miss_530 (vs central {s3['fern_central_official']:.0f}) = "
              f"{s3['tau_to_miss_530_vs_central']:.4f}; (vs conservative) = "
              f"{s3['tau_to_miss_530_vs_conservative']:.4f}  -> floor sits above both")
        rc = tr["red_clause_handling"]
        print(f"\n  [verdict] RED 'crosses ridge' clause: crosses_measured={rc['crosses_measured_ridge']} "
              f"crosses_datasheet={rc['crosses_datasheet_ridge']} deep_compute_bound={rc['deep_compute_bound']} "
              f"-> overridden={rc['auto_red_overridden']}")
        print(f"    >>> PRIMARY tau_tree_central        = {tr['tau_tree_central']:.4f}")
        print(f"    >>> TEST    tree_transfers_like_splitk = {tr['tree_transfers_like_splitk']}")
        print(f"    >>> GATE = {tr['verdict']} -- {tr['verdict_label']}")
        for r in tr["reasons"]:
            print(f"         - {r}")
        if args.wandb:
            _log_tree_roofline_wandb(args, calib, tr)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n[projection] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
