#!/usr/bin/env python3
"""Unit tests for the measured-500-gate CONFIDENCE ENVELOPE (PR #146).

Pins the pure, CPU-only statistics that wrap fern #142's point gate:

  * ``wilson_ci`` — binomial score interval (correct near p~0.4, finite n),
    against a textbook value and clamped at the [0,1] extremes.
  * ``clt_ci`` — normal/CLT half-width = z * sd / sqrt(n) (scipy-free z table).
  * ``bootstrap_ci`` — nonparametric resample CI brackets the mean, agrees with
    the CLT half-width on a moderate sample, and is degenerate on a constant one.
  * ``required_n_to_separate`` — the verify-step count that lifts the branch-hit
    Wilson lower bound above the 0.033 chain-reject floor.
  * ``required_n_for_verdict`` — the THREE-WAY required-N: GREEN above the worst
    corner, RED below the best corner, and **unbounded inside the systematic
    dead-band** [bar_best, bar_worst] (a point there straddles 500 for the
    step+tau band alone, so no finite N resolves it — the guard that keeps the
    required-N curve consistent with the robust_verdict self-test).
  * the per-step reconstruction (``pmf_from_cumulative`` sums to 1,
    ``samples_from_ladder`` mean-pins) and ``compose_tps_ci`` / ``robust_verdict``
    classification of the RED / GREEN / INDETERMINATE anchors.
  * end-to-end: the shipped script self-test passes and emits NaN-clean JSON.

No GPU, no model load, no W&B, no HF Job. Run:
    python -m pytest scripts/tests/test_gate_confidence_envelope.py -v
  or: python scripts/tests/test_gate_confidence_envelope.py    (no-pytest fallback)
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = REPO_ROOT / "scripts" / "profiler" / "m16_gate_confidence_envelope.py"


def _load_env():
    spec = importlib.util.spec_from_file_location("m16_gate_confidence_envelope", _ENV_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


env = _load_env()


# --------------------------------------------------------------------------- #
# Wilson score interval
# --------------------------------------------------------------------------- #
def test_wilson_ci_textbook_value():
    """k=10, n=20 (p=0.5), 95% Wilson interval is [0.299, 0.701], centred at 0.5."""
    ci = env.wilson_ci(10, 20, 95)
    assert abs(ci["center"] - 0.5) < 1e-9
    assert abs(ci["lo"] - 0.2992) < 2e-3, ci["lo"]
    assert abs(ci["hi"] - 0.7008) < 2e-3, ci["hi"]


def test_wilson_ci_clamped_in_unit_interval():
    """The Wilson bounds stay inside [0,1] even at k=0 and k=n (where a naive
    normal interval would spill out)."""
    for k in (0, 1, 1023, 1024):
        ci = env.wilson_ci(k, 1024, 99)
        assert ci["lo"] >= 0.0 - 1e-12, (k, ci["lo"])
        assert ci["hi"] <= 1.0 + 1e-12, (k, ci["hi"])


def test_wilson_99_contains_95():
    a = env.wilson_ci(427, 1024, 95)
    b = env.wilson_ci(427, 1024, 99)
    assert b["lo"] <= a["lo"] and b["hi"] >= a["hi"]


def test_branch_hit_wilson_excludes_chain_reject_floor():
    """The measured rho2=0.4165 over the 1024-step oracle budget must sit well
    above the 0.033 chain-reject floor (the discriminator that the walk DESCENDS)."""
    k = int(round(env.RHO2_BRANCH_HIT * 1024))
    ci = env.wilson_ci(k, 1024, 99)
    assert ci["lo"] > env.CHAIN_REJECT_FLOOR


# --------------------------------------------------------------------------- #
# CLT / normal interval
# --------------------------------------------------------------------------- #
def test_clt_ci_closed_form():
    """mean=5, sd=3, n=900, 95% -> half-width = 1.96*3/30 = 0.196."""
    ci = env.clt_ci(5.0, 3.0, 900, 95)
    assert abs(ci["half_width"] - 0.19600) < 1e-4, ci
    assert abs(ci["lo"] - 4.804) < 1e-3 and abs(ci["hi"] - 5.196) < 1e-3


def test_clt_99_wider_than_95():
    a = env.clt_ci(5.0, 3.0, 900, 95)
    b = env.clt_ci(5.0, 3.0, 900, 99)
    assert b["half_width"] > a["half_width"]


# --------------------------------------------------------------------------- #
# Bootstrap interval
# --------------------------------------------------------------------------- #
def test_bootstrap_brackets_mean_and_matches_clt():
    """On a moderate skewed sample the bootstrap mean-CI brackets the sample mean
    and its half-width agrees with the CLT half-width to ~20% (both estimate the
    same standard error of the mean)."""
    rng = np.random.default_rng(0)
    s = env.samples_from_ladder(env.ORACLE_CUM_LADDER, 4000, env.ORACLE_E_T)
    boot = env.bootstrap_ci(s, 95, 8000, rng)
    clt = env.clt_ci(float(s.mean()), float(s.std(ddof=1)), len(s), 95)
    assert boot["lo"] < s.mean() < boot["hi"]
    assert abs(boot["half_width"] - clt["half_width"]) / clt["half_width"] < 0.20


def test_bootstrap_99_wider_than_95():
    rng = np.random.default_rng(1)
    s = env.samples_from_ladder(env.ORACLE_CUM_LADDER, 2000, env.ORACLE_E_T)
    a = env.bootstrap_ci(s, 95, 6000, rng)
    b = env.bootstrap_ci(s, 99, 6000, rng)
    assert b["half_width"] >= a["half_width"]


def test_bootstrap_constant_sample_is_degenerate():
    rng = np.random.default_rng(2)
    s = np.full(512, 6.0)
    ci = env.bootstrap_ci(s, 99, 4000, rng)
    assert abs(ci["lo"] - 6.0) < 1e-9 and abs(ci["hi"] - 6.0) < 1e-9


# --------------------------------------------------------------------------- #
# required-N to separate the branch-hit from the chain-reject floor
# --------------------------------------------------------------------------- #
def test_required_n_to_separate_small_and_ordered():
    n95 = env.required_n_to_separate(env.RHO2_BRANCH_HIT, env.CHAIN_REJECT_FLOOR, 95)
    n99 = env.required_n_to_separate(env.RHO2_BRANCH_HIT, env.CHAIN_REJECT_FLOOR, 99)
    assert n95 is not None and n99 is not None
    assert 1 <= n95 <= n99 <= 50  # 0.4165 vs 0.033 separates in a handful of steps


def test_required_n_to_separate_none_when_below_floor():
    """A proportion at/under the floor can never separate -> None (not a crash)."""
    assert env.required_n_to_separate(0.03, 0.033, 99) is None


# --------------------------------------------------------------------------- #
# required_n_for_verdict — the three-way (GREEN / RED / dead-band) logic
# --------------------------------------------------------------------------- #
def _bars():
    return env.effective_clear500_bar(env.STEP_MEASURED_DEPTH9, 0.005)


def test_effective_bar_ordering():
    """bar_best < bar_central < bar_worst, and the central bar is the point where
    official(.,step,1.0)=500 (== fern's accept_length_for_official)."""
    b = _bars()
    assert b["bar_best"] < b["bar_central"] < b["bar_worst"]
    ref = env.accept_length_for_official(env.TARGET_OFFICIAL, env.STEP_MEASURED_DEPTH9, env.TAU["central"])
    assert abs(b["bar_central"] - ref) < 1e-9


def test_required_n_verdict_green_side_finite():
    b = _bars()
    et = b["bar_worst"] + 0.30  # clearly above the worst corner
    r = env.required_n_for_verdict(et, 3.0, b["bar_worst"], b["bar_best"], 99)
    assert r["side"] == "GREEN" and r["feasible"] and r["required_n"] >= 1


def test_required_n_verdict_red_side_finite():
    b = _bars()
    et = b["bar_best"] - 2.0  # clearly below the best corner (the oracle regime)
    r = env.required_n_for_verdict(et, 1.8, b["bar_worst"], b["bar_best"], 99)
    assert r["side"] == "RED" and r["feasible"] and r["required_n"] >= 1


def test_required_n_verdict_dead_band_is_unbounded():
    """A point INSIDE the systematic dead-band [bar_best, bar_worst] is permanently
    INDETERMINATE — no finite N resolves it. Regression guard: an earlier single-bar
    version reported a finite (large) N here, contradicting the robust_verdict
    self-test that classifies the same point INDETERMINATE."""
    b = _bars()
    et = b["bar_central"]  # strictly between bar_best and bar_worst
    for conf in (95, 99):
        r = env.required_n_for_verdict(et, 3.0, b["bar_worst"], b["bar_best"], conf)
        assert r["required_n"] is None
        assert r["unbounded"] is True
        assert r.get("in_systematic_dead_band") is True
        assert r["side"] == "INDETERMINATE"


def test_required_n_verdict_monotone_on_green_side():
    """Further above the worst-corner bar -> fewer verify steps needed."""
    b = _bars()
    ns = [env.required_n_for_verdict(et, 3.0, b["bar_worst"], b["bar_best"], 99)["required_n"]
          for et in (b["bar_worst"] + 0.05, b["bar_worst"] + 0.20, b["bar_worst"] + 0.40)]
    assert ns[0] > ns[1] > ns[2] >= 1


# --------------------------------------------------------------------------- #
# per-step reconstruction
# --------------------------------------------------------------------------- #
def test_pmf_from_cumulative_sums_to_one():
    pmf = env.pmf_from_cumulative(env.ORACLE_CUM_LADDER)
    assert abs(float(pmf.sum()) - 1.0) < 1e-12
    assert (pmf >= 0.0).all()


def test_pmf_mean_matches_one_plus_sum_ladder():
    """E[T] = 1 + sum(C): the ladder's marginal PMF mean of T=1+D reproduces the
    spine-only accept length before the salvage residual is folded in."""
    pmf = env.pmf_from_cumulative(env.ORACLE_CUM_LADDER)
    depths = np.arange(len(pmf))
    e_t = 1.0 + float((depths * pmf).sum())
    assert abs(e_t - (1.0 + sum(env.ORACLE_CUM_LADDER))) < 1e-9


def test_samples_from_ladder_mean_pinned():
    s = env.samples_from_ladder(env.ORACLE_CUM_LADDER, 4096, env.ORACLE_E_T)
    assert abs(float(s.mean()) - env.ORACLE_E_T) < 1e-6
    assert len(s) == 4096


# --------------------------------------------------------------------------- #
# compose_tps_ci + robust_verdict classification of the three anchors
# --------------------------------------------------------------------------- #
def test_robust_verdict_thresholds():
    assert env.robust_verdict({"tps_lo": 510.0, "tps_hi": 560.0}) == "robust-GREEN"
    assert env.robust_verdict({"tps_lo": 250.0, "tps_hi": 290.0}) == "robust-RED"
    assert env.robust_verdict({"tps_lo": 480.0, "tps_hi": 520.0}) == "INDETERMINATE"


def _analyse(point_et, sample, n_boot=3000, seed=3):
    rng = np.random.default_rng(seed)
    k = int(round(env.RHO2_BRANCH_HIT * len(sample)))
    return env.analyse_sample("t", sample, k, len(sample), env.STEP_MEASURED_DEPTH9,
                              0.005, n_boot, rng, point_et=point_et)


def test_anchor_oracle_is_robust_red():
    s = env.samples_from_ladder(env.ORACLE_CUM_LADDER, 1024, env.ORACLE_E_T)
    r = _analyse(env.ORACLE_E_T, s)
    assert r["robust_verdict_99"] == "robust-RED"
    assert r["tps_ci_bootstrap"]["99"]["tps_hi"] < env.TARGET_OFFICIAL


def test_anchor_high_is_robust_green():
    s = np.full(1024, 6.0)  # E[T]=6.0 clears 500 on every corner
    r = _analyse(6.0, s)
    assert r["robust_verdict_99"] == "robust-GREEN"
    assert r["tps_ci_bootstrap"]["99"]["tps_lo"] >= env.TARGET_OFFICIAL


def test_anchor_boundary_is_indeterminate():
    """At the exact clear-500 E[T] (central TPS == 500) the step+tau band straddles
    500 -> INDETERMINATE at any N, and the required-N is unbounded (dead-band)."""
    b = _bars()
    s = np.full(1024, b["bar_central"])
    r = _analyse(b["bar_central"], s)
    assert r["robust_verdict_99"] == "INDETERMINATE"
    assert r["required_n_for_verdict"]["99"]["required_n"] is None


# --------------------------------------------------------------------------- #
# end-to-end: the shipped script self-test passes and emits NaN-clean JSON
# --------------------------------------------------------------------------- #
def _walk_nonfinite(o, path=""):
    bad = []
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            bad.append((path, o))
    elif isinstance(o, dict):
        for k, v in o.items():
            bad += _walk_nonfinite(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            bad += _walk_nonfinite(v, f"{path}[{i}]")
    return bad


def test_end_to_end_self_test_passes_and_nan_clean():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "envelope.json"
        cmd = [sys.executable, "scripts/profiler/m16_gate_confidence_envelope.py",
               "--no-wandb", "--n-boot", "3000", "--out", str(out)]
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-2000:]
        d = json.loads(out.read_text())
        # primary metric
        assert d["gate_ci_self_test_passes"] == 1
        assert d["self_test"]["passes"] is True
        assert d["self_test"]["red_anchor_robust_RED"] is True
        assert d["self_test"]["green_anchor_robust_GREEN"] is True
        assert d["self_test"]["borderline_INDETERMINATE"] is True
        assert d["self_test"]["fern142_point_anchors_reproduced"] is True
        # test metric: oracle point is decisively RED -> tiny required-N (MOOT)
        rn = d["required_n_for_robust_500_verdict"]
        assert isinstance(rn, int) and 1 <= rn <= 50
        # the dead-band boundary row in the hand-off curve must read unbounded
        row0 = d["handoff_land71"]["required_n_curve"][0]
        assert row0["required_n_99"]["required_n"] is None
        assert row0["required_n_99"].get("in_systematic_dead_band") is True
        # NaN-cleanliness across the whole artifact
        assert _walk_nonfinite(d) == []
        assert "NaN" not in out.read_text() and "Infinity" not in out.read_text()


# --------------------------------------------------------------------------- #
# no-pytest fallback runner
# --------------------------------------------------------------------------- #
def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface setup errors too
            failed += 1
            print(f"ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
