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


# ---------------------------------------------------------------------------
# PR #112 tree-free 500-path instrument
# ---------------------------------------------------------------------------
def test_tree_free_self_check_reproduces_anchor_bit_exact():
    """The NULL-lever tree-free point (s=0, LK=1, palette=0, tau=1, mult=central) is
    the bit-exact consistency anchor: it must equal K_cal*E_T_linear == 481.53 with
    zero residual (this is what ties denken #105's K_cal to #99's multiplier)."""
    sc = lop.tree_free_self_check()
    assert sc["reproduces_anchor_bit_exact"] is True, sc
    assert math.isclose(sc["recovered_official"], 481.53, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(lop.tf.K_CAL * lop.tf.E_T_LINEAR, 481.53, rel_tol=0, abs_tol=1e-9)


def test_tree_free_multiplier_rescale_keeps_anchor_and_is_monotone():
    """The #99 multiplier CI enters as a RELATIVE rescale: at mult==central the map is
    exact on the anchor; the CI edges shift it by only the multiplier's ~0.018% and
    monotonically (low edge below central, high edge above)."""
    c = lop.calibrate()
    mc = c.multiplier
    at_c = lop.tree_free_official(0.0, mult=mc, tau=1.0, mult_central=mc)["official_tps"]
    at_lo = lop.tree_free_official(0.0, mult=c.mult_ci_local_lo, tau=1.0, mult_central=mc)["official_tps"]
    at_hi = lop.tree_free_official(0.0, mult=c.mult_ci_local_hi, tau=1.0, mult_central=mc)["official_tps"]
    assert math.isclose(at_c, 481.53, rel_tol=0, abs_tol=1e-9)
    assert at_lo < at_c < at_hi
    # the rescale band is exactly the multiplier's relative CI half-width (<0.1%).
    assert abs(at_hi - at_lo) / at_c < 0.001


def test_tree_free_threshold_round_trips_compose():
    """At the inverted threshold s, the forward map must land exactly on 500 (the
    inverter and the composer agree -- single source of truth via denken's algebra)."""
    c = lop.calibrate()
    mc = c.multiplier
    corner = dict(mult=c.mult_ci_local_lo, tau=0.96, lk_mult=lop.tf.LK_MULT["low"],
                  palette_tps_pct=lop.PALETTE_TPS["low"], fp32_m8=lop.tf.FP32_M8["high"],
                  persist_reclaim=lop.tf.PERSIST_RECLAIM["low"], mult_central=mc)
    s = lop.tree_free_threshold(**corner)
    assert s not in (None, float("inf")) and s > 0
    back = lop.tree_free_official(s, **corner)["official_tps"]
    assert math.isclose(back, lop.OFFICIAL_TARGET_TPS, rel_tol=0, abs_tol=1e-6)


def test_tree_free_threshold_monotone_in_tau():
    """A lower tau (worse realization) needs MORE SplitK to clear 500: the
    conservative-corner threshold strictly decreases as tau rises 0.96 -> 1.00."""
    c = lop.calibrate()
    mc = c.multiplier
    def thr(tau):
        return lop.tree_free_threshold(
            mult=c.mult_ci_local_lo, tau=tau, lk_mult=lop.tf.LK_MULT["low"],
            palette_tps_pct=lop.PALETTE_TPS["low"], fp32_m8=lop.tf.FP32_M8["high"],
            persist_reclaim=lop.tf.PERSIST_RECLAIM["low"], mult_central=mc)
    series = [thr(t) for t in (0.96, 0.97, 0.98, 0.99, 1.00)]
    assert all(a > b for a, b in zip(series, series[1:])), series


def test_project_tree_free_band_ordering_and_conservative_corner():
    """The single-command projection orders conservative <= central <= optimistic, and
    the conservative corner is exactly (multiplier-low x tau-low) as the PR names it."""
    c = lop.calibrate()
    pj = lop.project_tree_free(0.12, calib=c)  # ubel nominal-high
    assert pj["projected_official_lo"] <= pj["projected_official"] <= pj["projected_official_hi"]
    cc = pj["conservative_corner"]
    assert math.isclose(cc["multiplier"], c.mult_ci_local_lo, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(cc["tau"], lop.TAU_BAND_DEFAULT["low"], rel_tol=0, abs_tol=1e-12)
    # central uses the central multiplier and tau=1.00.
    assert math.isclose(pj["central"]["multiplier"], c.multiplier, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(pj["central"]["tau"], 1.00, rel_tol=0, abs_tol=1e-12)


def test_bound_tau_local_band_ceiling_and_meter_confound():
    """tau_band_local is the mechanism band [0.99, 1.00], strictly inside denken's
    generic [0.96, 1.00]; the cross-meter spread dwarfs the within-meter config spread
    (that asymmetry is WHY precision rungs can't bound tau)."""
    tb = lop.bound_tau_local()
    assert tb["tau_band_local"] == [0.99, 1.00]
    assert tb["tau_mechanism_low"] >= tb["tau_generic_low"]
    assert lop.tf.TAU["low"] <= tb["tau_band_local"][0] and tb["tau_band_local"][1] <= lop.tf.TAU["high"]
    meter = tb["meter_confound"]["implied_multiplier_spread_pct"]
    within = tb["within_meter_config_stability"]["per_session_multiplier_spread_pct"]
    assert meter > 10.0 * within, (meter, within)
    assert tb["recommendation"] in (
        "SHIP_ON_LOCAL_CAL", "ONE_OFFICIAL_SPLITK_ANCHOR", "TRANSFER_UNTRUSTWORTHY")


def test_tree_free_gate_amber_landing():
    """On the committed inputs the gate reads AMBER: the instrument is ARMED but the
    >=500 call turns on the tau floor -> one official SplitK anchor named for #109."""
    c = lop.calibrate()
    g = lop.tree_free_gate(c)
    assert g["verdict"] == "AMBER", g
    assert g["tree_free_projection_armed"] is True
    assert g["tau_band_local"] == [0.99, 1.00]
    assert g["recommendation"] == "ONE_OFFICIAL_SPLITK_ANCHOR"
    assert g["transfer_stable"] is True


def test_tree_free_gate_green_when_splitk_decides_at_generic_floor():
    """If the conservative threshold clears even at the GENERIC 0.96 floor, SplitK alone
    decides 500 regardless of tau -> GREEN."""
    c = lop.calibrate()
    tb = lop.bound_tau_local(c)
    tb = dict(tb); tb["decides_at_generic_floor"] = True
    g = lop.tree_free_gate(c, tau_bound=tb)
    assert g["verdict"] == "GREEN", g
    assert g["tree_free_projection_armed"] is True


def test_tree_free_gate_red_when_threshold_unreachable():
    """If even the mechanism-floor threshold exceeds the bandwidth-gap ceiling, the
    local projection can't decide 500 -> RED."""
    c = lop.calibrate()
    tb = lop.bound_tau_local(c)
    tb = dict(tb)
    tb["conservative_threshold_at_tau"] = dict(tb["conservative_threshold_at_tau"])
    tb["conservative_threshold_at_tau"]["0.99"] = float("inf")
    g = lop.tree_free_gate(c, tau_bound=tb)
    assert g["verdict"] == "RED", g
