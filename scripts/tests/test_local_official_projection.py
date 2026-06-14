#!/usr/bin/env python3
"""CPU-only correctness tests for scripts/profiler/local_official_projection.py (PR #99).

The local->official projection turns the team's implicit ~constant multiplier into a
pinned number + CI and makes land #71's tree a one-shot >=500 decision. These tests
pin the load-bearing properties so a future build's measured wall_tps can be trusted
through this map:

* the multiplier IS official_anchor / pooled-mean-local (exact identity);
* the closed-loop self-check recovers the official anchor within the self-check MDE;
* per-session multipliers bracket the pooled estimate (config-stability witness);
* the projection band combines the multiplier CI and any modeling band in quadrature
  and is monotone in the input wall_tps;
* the envelope CI (assumed official CV) strictly contains the local-only CI;
* the analytical tree projection clears 500 and its band overlaps fern #92's
  published [558, 581] official band.

Run: python -m pytest scripts/tests/test_local_official_projection.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

PROFILER_DIR = Path(__file__).resolve().parents[1] / "profiler"
sys.path.insert(0, str(PROFILER_DIR))
import local_official_projection as lop  # noqa: E402


def test_multiplier_is_official_over_pooled_local_mean():
    """multiplier == official_anchor / mean(all committed per-run wall_tps), exactly."""
    c = lop.calibrate()
    runs = lop._all_runs()
    assert c.n_runs == len(runs) == 9, c.n_runs
    pooled_mean = sum(runs) / len(runs)
    assert math.isclose(c.local_wall_tps, pooled_mean, rel_tol=0, abs_tol=1e-9)
    expected = float(lop.OFFICIAL_ANCHOR["tps"]) / pooled_mean
    assert math.isclose(c.multiplier, expected, rel_tol=0, abs_tol=1e-12)
    # Sanity: the team's implicit ~1.06 transfer factor.
    assert 1.055 < c.multiplier < 1.065, c.multiplier


def test_project_pooled_mean_recovers_anchor_exactly():
    """Projecting the EXACT denominator returns the official anchor bit-exactly
    (the non-circular consistency anchor of the whole map)."""
    c = lop.calibrate()
    pj = lop.project_official(c.local_wall_tps, calib=c)
    assert math.isclose(pj["projected_official"], c.official_tps, rel_tol=0, abs_tol=1e-9)


def test_self_check_recovers_anchor_within_mde():
    """The closed loop projects the LOCKED #90 reference (454.338) and recovers the
    481.53 anchor within the self-check MDE -- residual is the session-vs-mean spread."""
    sc = lop.self_check()
    assert sc["recovers_official_anchor"] is True, sc
    assert sc["rel_err_vs_anchor_pct"] <= lop.SELF_CHECK_MDE_PCT
    # The residual is exactly (locked_ref - pooled_mean)/pooled_mean -> tiny, sub-MDE.
    c = lop.calibrate()
    expected_resid = 100.0 * abs(
        lop.LINEAR_REFERENCE_WALL_TPS - c.local_wall_tps) / c.local_wall_tps
    assert math.isclose(sc["rel_err_vs_anchor_pct"], expected_resid, rel_tol=1e-6)


def test_per_session_multipliers_bracket_pooled():
    """Config-stability witness: the pooled multiplier lies within the per-session
    spread, and the spread itself is tight (<0.1%)."""
    c = lop.calibrate()
    mults = [p["multiplier"] for p in c.per_session]
    assert min(mults) <= c.multiplier <= max(mults)
    spread_pct = 100.0 * (max(mults) - min(mults)) / c.multiplier
    assert spread_pct < 0.1, spread_pct


def test_envelope_ci_contains_local_ci():
    """The conservative envelope (assumed official CV) must be strictly wider than the
    measured local-only CI -- it adds an unmeasured term, never removes one."""
    c = lop.calibrate(official_cv_assumed_pct=1.0)
    assert c.mult_ci_env_lo < c.mult_ci_local_lo
    assert c.mult_ci_env_hi > c.mult_ci_local_hi
    # Zero assumed official CV collapses the envelope onto the local-only CI.
    c0 = lop.calibrate(official_cv_assumed_pct=0.0)
    assert math.isclose(c0.mult_ci_env_lo, c0.mult_ci_local_lo, rel_tol=1e-9)
    assert math.isclose(c0.mult_ci_env_hi, c0.mult_ci_local_hi, rel_tol=1e-9)


def test_projection_band_quadrature_and_monotone():
    """Band combines multiplier-rel and modeling-rel in quadrature; central scales
    linearly and monotonically with input wall_tps."""
    c = lop.calibrate()
    base = lop.project_official(500.0, calib=c, modeling_band_pct=0.0)
    withmodel = lop.project_official(500.0, calib=c, modeling_band_pct=3.0)
    # quadrature: total^2 == mult^2 + model^2
    tot = withmodel["band_rel_pct"]
    rm = withmodel["band_from_multiplier_pct"]
    rmod = withmodel["band_from_modeling_pct"]
    assert math.isclose(tot, math.hypot(rm, rmod), rel_tol=1e-9)
    assert withmodel["band_rel_pct"] > base["band_rel_pct"]
    # monotone + linear central
    lo = lop.project_official(400.0, calib=c)["projected_official"]
    hi = lop.project_official(600.0, calib=c)["projected_official"]
    assert lo < base["projected_official"] < hi
    assert math.isclose(hi / lo, 600.0 / 400.0, rel_tol=1e-9)


def test_clears_500_threshold_logic():
    """A clearly-low wall_tps must NOT clear 500; a clearly-high one must, using the
    lower band edge as the gate."""
    c = lop.calibrate()
    low = lop.project_official(450.0, calib=c)   # ~477 official -> below 500
    high = lop.project_official(520.0, calib=c)  # ~551 official -> above 500
    assert low["clears_500"] is False
    assert high["clears_500"] is True
    assert high["projected_official_lo"] >= lop.OFFICIAL_TARGET_TPS


def test_tree_projection_clears_500_and_overlaps_published_band():
    """Step-3: the analytical tree projects above 500 (lower edge clears) and its band
    overlaps fern #92's published official band [558, 581]."""
    tr = lop.project_tree()
    assert tr["projected_official_clears_500_bool"] is True
    assert tr["margin_to_500_pct_at_lo"] > 0.0
    pub_lo, pub_hi = lop.TREE_SPEC["fern_published_official_band"]
    # band overlap (our band must intersect the published band)
    assert tr["projected_official_hi"] >= pub_lo
    assert tr["projected_official_lo"] <= pub_hi
    # central within a couple % of fern's midpoint
    mid = 0.5 * (pub_lo + pub_hi)
    assert abs(tr["projected_official"] - mid) / mid < 0.05


def test_tree_gain_applied_to_locked_linear_reference():
    """local_tree == LINEAR_REFERENCE * (1 + gain); projected == local_tree * mult."""
    c = lop.calibrate()
    tr = lop.project_tree(c)
    gain = lop.TREE_SPEC["net_local_gain_pct"]
    expect_local = lop.LINEAR_REFERENCE_WALL_TPS * (1.0 + gain / 100.0)
    assert math.isclose(tr["local_tree_wall_tps"], expect_local, rel_tol=1e-9)
    assert math.isclose(tr["projected_official"], expect_local * c.multiplier, rel_tol=1e-9)


def test_gate_green_on_current_spec_tree():
    """The committed-anchor calibration + current-spec tree must read GREEN: the
    multiplier is config-stable and the conservative band clears 500 with margin."""
    c = lop.calibrate()
    tr = lop.project_tree(c)
    g = lop.gate_verdict(c, tr)
    assert g["verdict"] == "GREEN", g
    assert g["config_stable"] is True
    assert g["band_straddles_500"] is False
    assert g["clears_500_with_margin"] is True
    assert g["margin_to_500_pct_at_lo"] >= g["margin_green_pct"]


def test_gate_red_when_band_straddles_500():
    """A weak gain whose projected band straddles 500 must read RED (inconclusive)."""
    c = lop.calibrate()
    # gain chosen so central ~ just above 500 but lo < 500: 500/(454.338*1.0602) - 1
    weak_gain = 100.0 * (500.0 / (lop.LINEAR_REFERENCE_WALL_TPS * c.multiplier) - 1.0) + 0.3
    tr = lop.project_tree(c, net_local_gain_pct=weak_gain, modeling_band_pct=2.3)
    g = lop.gate_verdict(c, tr)
    assert g["verdict"] == "RED", (weak_gain, g)
    assert g["band_straddles_500"] is True


def test_gate_amber_on_thin_margin():
    """A gain that clears 500 only by a thread (lo just above 500, < margin_green)
    must read AMBER, not GREEN."""
    c = lop.calibrate()
    # pick gain so the LOW edge lands ~2% above 500 (below the 5% GREEN bar)
    target_lo = 500.0 * 1.02
    # invert: lo = central*(1-rel); approximate with modeling band 0 for a tight handle
    tr0 = lop.project_tree(c, net_local_gain_pct=0.0, modeling_band_pct=0.0)
    rel = tr0["band_rel_pct"] / 100.0
    needed_central = target_lo / (1.0 - rel)
    gain = 100.0 * (needed_central / (lop.LINEAR_REFERENCE_WALL_TPS * c.multiplier) - 1.0)
    tr = lop.project_tree(c, net_local_gain_pct=gain, modeling_band_pct=0.0)
    g = lop.gate_verdict(c, tr)
    assert g["verdict"] == "AMBER", (gain, g)
    assert 0 < g["margin_to_500_pct_at_lo"] < g["margin_green_pct"]
