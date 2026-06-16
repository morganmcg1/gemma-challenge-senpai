"""PR #508 -- Surgical-357 private-outcome dossier + floor-lock portfolio price.

analysis_only=true, official_tps=0, CPU-only. NO serve, NO HF job, NO --launch,
NO submission, NO served-file change.

This is a pure *composition* PR: it reuses my own #504 and #478 OUTPUTS as
inputs (it re-derives nothing) and folds them into one decision-grade
private-outcome dossier for the shipped surgical-357, plus the
floor-lock-vs-surgical portfolio price for the challenge reopen.

Inputs (all my own merged work, validated against the cited W&B runs):
  - kanna #504 (0urxqwob): realized propagation factor 0.9999 (linear) ->
    surgical-357 private breach 4.295% off the 357.22 public anchor.
  - kanna #478 (mssuss3f, "single-draw-risk-474"): sigma_hw one-shot ~1.00%
    fractional (between-allocation 0.9623%/4.864 TPS dominated 13.9x; within
    0.0726%), plus the floor-lock single-official-draw risk model.
  - the ship: surgical-357 (PR #499, j7qao5e9; public 375.857 official / 357.22
    local; spec-alive MTP K=7; operative-1.0).
  - the fallback: floor-lock 166.23 (stark #485 pavotwci,
    submissions/fa2sw_strict_m1ar_int4; literal-1.0; private-SAFE; zero breach).

The composition (PR formula):
    private_TPS = ship_public * (1 - breach_frac) * (1 +- sigma_hw_frac)
with breach_frac the #504 LINEAR breach and sigma_hw_frac the #478 one-shot.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import wandb  # noqa: F401  (import first to win over any ./wandb shadow dir)

from scripts import wandb_logging
from scripts.common import ROOT

# --------------------------------------------------------------------------- #
# INPUTS (cited baselines -- my own merged work; validated vs the W&B runs)    #
# --------------------------------------------------------------------------- #
# Ship being priced: surgical-357 (stark #499, j7qao5e9), spec-alive MTP K=7.
SHIP_PUBLIC_TPS = 357.22          # kanna #504 0urxqwob (recert l0attso0); PR formula writes "357"
SHIP_PUBLIC_TPS_ROUND = 357.0     # PR-literal robustness check

# Linear breach from kanna #504 (0urxqwob): realized propagation factor 0.9999
# (deployed 481.53->460.85 = 4.295% TPS gap / denken dalpha 4.295% -> PF 1.00).
# alpha-invariant C_step => the FRACTIONAL breach transfers 481-stack -> 357-ship.
BREACH_FRAC = 0.04294644155088977
BREACH_FRAC_ROUND = 0.043         # PR-literal "1 - 0.043"
REALIZED_PROPAGATION_FACTOR = 0.9999171490311938

# denken #489 worst-case contrast (refuted by #504 as the *expected* outcome:
# needs PF~5.6 / deep-block survival, far outside the natural envelope [0.74, 2.24]).
DENKEN_WORSTCASE_BREACH = 0.24

# sigma_hw from kanna #478 (mssuss3f): one-shot ~1.00% fractional, between-dominated.
SIGMA_HW_FRAC = 0.01              # one-shot convention, VINDICATED for a single official draw
SIGMA_HW_BETWEEN_FRAC = 0.009623  # cross-allocation leg (the dominant one)
SIGMA_HW_WITHIN_FRAC = 0.000726   # same-device leg (negligible for a one-shot draw)
SIGMA_HW_ONESHOT_FRAC_MEASURED = 0.010128  # sqrt(within^2+between^2) measured
SIGMA_HW_TPS_AT_481 = 4.864       # PR's "sigma_hw 4.864" == between-leg absolute @481.53
SIGMA_HW_REF_TPS = 481.53         # the op point at which 1% == 4.8153

# Floor-lock fallback: literal-1.0, private-SAFE, ZERO breach (M=1 AR, no spec).
FLOORLOCK_TPS = 166.23            # stark #485 pavotwci, submissions/fa2sw_strict_m1ar_int4
FLOORLOCK_TPS_478 = 161.70        # prior #474-era value (mssuss3f); re-measured up to 166.23

# Invalidate-on-breach speed threshold (the validity bar the org could draw).
SPEED_THRESHOLD_FRAC = 0.95       # private < 0.95 x public => "breach"

Z95 = 1.959963984540054

# Reuse #504's persisted OUTPUT to assert exact reproduction (no re-derivation).
RECONCILE_504_JSON = ROOT / "research/private_draw_breach_reconcile/reconcile_breach.json"


# --------------------------------------------------------------------------- #
# Normal helpers (no scipy dependency)                                         #
# --------------------------------------------------------------------------- #
def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def trunc_normal_mean_above(mu: float, sigma: float, thresh: float) -> float:
    """E[X | X >= thresh] for X ~ N(mu, sigma^2) (inverse-Mills ratio)."""
    if sigma <= 0:
        return mu
    z = (thresh - mu) / sigma
    tail = 1.0 - norm_cdf(z)
    if tail <= 0.0:
        return mu  # threshold far in the lower tail; conditioning is ~no-op
    return mu + sigma * norm_pdf(z) / tail


# --------------------------------------------------------------------------- #
# Band composition                                                            #
# --------------------------------------------------------------------------- #
def compose_breach_band(public: float, breach_frac: float, sigma_frac: float) -> dict[str, Any]:
    """Surgical band: mean = public*(1-breach); sigma_hw is fractional on the mean."""
    mean = public * (1.0 - breach_frac)
    sigma_tps = mean * sigma_frac
    threshold = SPEED_THRESHOLD_FRAC * public
    p_below = norm_cdf((threshold - mean) / sigma_tps) if sigma_tps > 0 else float("nan")
    return {
        "public_tps": public,
        "breach_frac": breach_frac,
        "breach_pct": 100.0 * breach_frac,
        "sigma_hw_frac": sigma_frac,
        "private_tps_mean": mean,
        "private_tps_sigma": sigma_tps,
        "private_tps_band_68": [mean - sigma_tps, mean + sigma_tps],
        "private_tps_band_95": [mean - Z95 * sigma_tps, mean + Z95 * sigma_tps],
        "threshold_095_public": threshold,
        "p_private_below_95pct_public": p_below,
    }


def compose_floorlock_band(tps: float, sigma_frac: float) -> dict[str, Any]:
    """Floor-lock band: literal-1.0, ZERO breach => private mean == public.
    Still carries the universal fractional sigma_hw (the SAME hardware draw)."""
    mean = tps
    sigma_tps = mean * sigma_frac
    threshold = SPEED_THRESHOLD_FRAC * tps
    p_below = norm_cdf((threshold - mean) / sigma_tps) if sigma_tps > 0 else float("nan")
    return {
        "public_tps": tps,
        "breach_frac": 0.0,
        "sigma_hw_frac": sigma_frac,
        "private_tps_mean": mean,
        "private_tps_sigma": sigma_tps,
        "private_tps_band_68": [mean - sigma_tps, mean + sigma_tps],
        "private_tps_band_95": [mean - Z95 * sigma_tps, mean + Z95 * sigma_tps],
        "threshold_095_public": threshold,
        "p_private_below_95pct_public": p_below,
        # absolute-sigma sensitivity (#478 raised the abs model; ~3x wider for a 166-TPS sub).
        "abs_sigma_tps_481conv": SIGMA_HW_BETWEEN_FRAC * SIGMA_HW_REF_TPS,
        "abs_model_band95_lo": mean - Z95 * (SIGMA_HW_BETWEEN_FRAC * SIGMA_HW_REF_TPS),
    }


# --------------------------------------------------------------------------- #
# Portfolio price (the floor-lock vs surgical decision)                        #
# --------------------------------------------------------------------------- #
def portfolio_price(surg: dict[str, Any], floor: dict[str, Any]) -> dict[str, Any]:
    surg_mean = surg["private_tps_mean"]
    surg_sig = surg["private_tps_sigma"]
    surg_95lo = surg["private_tps_band_95"][0]
    floor_mean = floor["private_tps_mean"]
    floor_sig = floor["private_tps_sigma"]
    floor_95hi = floor["private_tps_band_95"][1]

    # --- raw-TPS gaps ---
    expected_gap_tps = surg_mean - floor_mean
    expected_gap_pct = 100.0 * expected_gap_tps / floor_mean
    downside_gap_vs_floor_mean = surg_95lo - floor_mean       # surg worst vs floor mean
    downside_gap_vs_floor_95hi = surg_95lo - floor_95hi       # surg worst vs floor BEST

    # surgical even at the refuted 24% worst-case breach (still spec-alive band).
    wc = compose_breach_band(SHIP_PUBLIC_TPS, DENKEN_WORSTCASE_BREACH, SIGMA_HW_FRAC)
    worstcase24_mean = wc["private_tps_mean"]
    worstcase24_95lo = wc["private_tps_band_95"][0]
    worstcase24_downside_gap_vs_floor_95hi = worstcase24_95lo - floor_95hi

    # --- P(surgical private < floor-lock) ---
    # point: floor-lock at its literal mean (PR's "P(private < floor-lock 166.23)").
    p_surg_below_floorlock_point = norm_cdf((floor_mean - surg_mean) / surg_sig) if surg_sig > 0 else float("nan")
    # convolution: both draws random & independent on the SAME official allocation.
    diff_sigma = math.sqrt(surg_sig ** 2 + floor_sig ** 2)
    p_surg_below_floorlock_conv = norm_cdf((0.0 - expected_gap_tps) / diff_sigma) if diff_sigma > 0 else float("nan")

    # --- RULE (a): penalize-private-breach (score == realized private TPS) ---
    # You keep your realized private speed; the breach only lowers your number.
    rule_penalize = {
        "surgical_score_mean": surg_mean,
        "surgical_score_95lo": surg_95lo,
        "surgical_score_worstcase24_95lo": worstcase24_95lo,
        "floorlock_score": floor_mean,
        "winner": "surgical",
        "margin_expected_tps": expected_gap_tps,
        "margin_worstcase24_tps": worstcase24_95lo - floor_mean,
        "note": ("surgical dominates by +%.1f TPS (+%.0f%%) in expectation and across the "
                 "entire band; even the refuted 24%% worst-case 95%%-downside (%.1f) beats "
                 "floor-lock (%.2f)." % (expected_gap_tps, expected_gap_pct,
                                         worstcase24_95lo, floor_mean)),
    }

    # --- RULE (b): invalidate-on-breach (a breach zeroes the submission) ---
    # (b1) speed-threshold invalidation: invalid iff private < 0.95 x public.
    p_invalidate = surg["p_private_below_95pct_public"]
    threshold = surg["threshold_095_public"]
    e_surg_given_valid = trunc_normal_mean_above(surg_mean, surg_sig, threshold)
    e_surg_score_speed = (1.0 - p_invalidate) * e_surg_given_valid
    # floor-lock is essentially never invalidated under a speed rule (no breach).
    p_invalidate_floor = floor["p_private_below_95pct_public"]
    rule_invalidate_speed = {
        "threshold_tps": threshold,
        "P_surgical_invalidate": p_invalidate,
        "E_surgical_given_valid_tps": e_surg_given_valid,
        "E_surgical_score_tps": e_surg_score_speed,
        "P_floorlock_invalidate": p_invalidate_floor,
        "floorlock_score": floor_mean,
        "winner_expected_value": "surgical" if e_surg_score_speed > floor_mean else "floor-lock",
        "winner_maximin": "floor-lock",  # surgical worst outcome is 0 (invalidated) < floor 166.23
        "note": ("risk-neutral picks surgical (E=%.1f > %.2f); maximin/guaranteed-floor picks "
                 "floor-lock (surgical carries a %.1f%% chance of a 0 wipe-out; floor-lock's "
                 "%.2f is guaranteed)." % (e_surg_score_speed, floor_mean,
                                           100.0 * p_invalidate, floor_mean)),
    }

    # (b2) literal-greedy-identity invalidation: ANY private greedy divergence zeroes.
    # surgical is operative-1.0 (spec-alive: the stack diverges from plain M=1 AR greedy on
    # off-distribution/private prompts by construction); floor-lock is literal-1.0.
    rule_invalidate_literal = {
        "surgical_classification": "operative-1.0 (spec-alive; diverges from plain greedy off-public)",
        "surgical_score": 0.0,
        "floorlock_classification": "literal-1.0 (M=1 AR; greedy-identical by construction)",
        "floorlock_score": floor_mean,
        "winner": "floor-lock",
        "margin_tps": floor_mean - 0.0,
        "note": ("under a LITERAL private-identity rule surgical is invalidated -> 0; "
                 "floor-lock is the only guaranteed-valid option -> floor-lock dominates outright."),
    }

    # --- bracketed verdict ---
    portfolio_verdict = "bracketed"
    portfolio_verdict_long = (
        "BRACKETED. On raw TPS surgical-357 dominates floor-lock under every plausible "
        "private draw (P(surgical < floor-lock) ~ 0; even the 24%% worst-case 95%%-downside "
        "%.0f >> floor-lock best-case %.0f) -- floor-lock is NEVER a speed case, only an "
        "invalidation-insurance case. So the decision forks on the organizer's private "
        "VALIDITY rule: (a) penalize-breach OR (b1) speed-threshold-invalidate with an "
        "expected-value objective -> SHIP surgical-357 (+%.0f TPS / +%.0f%% expected; "
        "E=%.0f even after the 23%% speed-invalidate tail). (b2) literal-greedy-identity "
        "private rule OR a maximin/guaranteed-floor objective -> KEEP floor-lock-166.23 "
        "(surgical is operative-1.0 and risks a 0; floor-lock's %.0f is guaranteed)."
        % (worstcase24_95lo, floor_95hi, expected_gap_tps, expected_gap_pct,
           e_surg_score_speed, floor_mean)
    )

    return {
        "expected_private_gap_tps": expected_gap_tps,
        "expected_private_gap_pct": expected_gap_pct,
        "downside_gap_surg95lo_vs_floor_mean_tps": downside_gap_vs_floor_mean,
        "downside_gap_surg95lo_vs_floor_95hi_tps": downside_gap_vs_floor_95hi,
        "worstcase24_mean_tps": worstcase24_mean,
        "worstcase24_95lo_tps": worstcase24_95lo,
        "worstcase24_downside_gap_vs_floor_95hi_tps": worstcase24_downside_gap_vs_floor_95hi,
        "P_surgical_below_floorlock_point": p_surg_below_floorlock_point,
        "P_surgical_below_floorlock_conv": p_surg_below_floorlock_conv,
        "band_worstcase24": wc,
        "rule_penalize_breach": rule_penalize,
        "rule_invalidate_speed_threshold": rule_invalidate_speed,
        "rule_invalidate_literal_identity": rule_invalidate_literal,
        "portfolio_verdict": portfolio_verdict,
        "portfolio_verdict_long": portfolio_verdict_long,
    }


# --------------------------------------------------------------------------- #
# Build + self-test                                                           #
# --------------------------------------------------------------------------- #
def build_results() -> dict[str, Any]:
    surg = compose_breach_band(SHIP_PUBLIC_TPS, BREACH_FRAC, SIGMA_HW_FRAC)
    surg_round = compose_breach_band(SHIP_PUBLIC_TPS_ROUND, BREACH_FRAC_ROUND, SIGMA_HW_FRAC)
    floor = compose_floorlock_band(FLOORLOCK_TPS, SIGMA_HW_FRAC)
    port = portfolio_price(surg, floor)

    one_line = (
        "surgical-357 private TPS ~= %.1f [95%% %.1f-%.1f, +-sigma_hw 1%%], breach %.2f%% "
        "(linear, PF 1.00 #504); P(private<0.95xpublic)=%.3f; P(private<floor-lock %.2f)~%.0e "
        "-> surgical DOMINATES floor-lock on raw TPS by +%.0f (+%.0f%%) under every draw "
        "(even 24%% worst-case 95%%-downside %.0f >> %.2f). VERDICT BRACKETED: ship surgical "
        "under penalize/speed-invalidate(E-value); keep floor-lock-166.23 under a literal "
        "private-identity rule or maximin objective (surgical operative-1.0 risks a 0)."
        % (surg["private_tps_mean"], surg["private_tps_band_95"][0], surg["private_tps_band_95"][1],
           surg["breach_pct"], surg["p_private_below_95pct_public"], FLOORLOCK_TPS,
           port["P_surgical_below_floorlock_point"], port["expected_private_gap_tps"],
           port["expected_private_gap_pct"], port["worstcase24_95lo_tps"], floor["private_tps_mean"]))

    dossier_verdict = (
        "SHIP surgical-357 as primary for the reopen; hold floor-lock-166.23 (%s) as the "
        "pre-staged fallback to swap in IFF the organizer's reopen rules invalidate on "
        "literal private greedy identity (or the objective is maximin/guaranteed-floor). "
        "Expected private outcome %.1f TPS [95%% %.1f-%.1f]; mild 4.3%% breach; zero realistic "
        "risk of dropping below the floor-lock fallback on raw speed (P ~ %.0e)."
        % ("submissions/fa2sw_strict_m1ar_int4, literal-1.0", surg["private_tps_mean"],
           surg["private_tps_band_95"][0], surg["private_tps_band_95"][1],
           port["P_surgical_below_floorlock_point"]))

    results = {
        "pr": 508,
        "agent": "kanna",
        "analysis_only": True,
        "official_tps": 0,
        "no_serve": True,
        "no_hf_job": True,
        "no_launch": True,
        "no_submission": True,
        "no_served_file_change": True,
        "lane_discipline": ("pure composition of my own #504 (propagation/breach) and #478 "
                            "(sigma_hw + floor-lock single-draw risk) outputs; re-derives "
                            "nothing -- recomputes the #504 surgical band only to assert exact "
                            "reproduction."),
        "inputs": {
            "ship_public_tps": SHIP_PUBLIC_TPS,
            "breach_frac": BREACH_FRAC,
            "realized_propagation_factor": REALIZED_PROPAGATION_FACTOR,
            "sigma_hw_frac": SIGMA_HW_FRAC,
            "sigma_hw_between_frac": SIGMA_HW_BETWEEN_FRAC,
            "sigma_hw_within_frac": SIGMA_HW_WITHIN_FRAC,
            "sigma_hw_oneshot_frac_measured": SIGMA_HW_ONESHOT_FRAC_MEASURED,
            "sigma_hw_between_tps_at_481": SIGMA_HW_TPS_AT_481,
            "floorlock_tps": FLOORLOCK_TPS,
            "floorlock_tps_478_prior": FLOORLOCK_TPS_478,
            "denken_worstcase_breach_frac": DENKEN_WORSTCASE_BREACH,
            "sigma_hw_provenance_note": (
                "composition uses the scale-invariant 1.00%% fractional one-shot convention "
                "(#478 mssuss3f, between-allocation dominated 13.9x). PR's 'sigma_hw 4.864' is "
                "the directly-measured between-leg in TPS (frantic-penguin 3 official draws, "
                "measured on a ~505-mean pool: 4.864/0.009623~505), kept as provenance only -- "
                "not composed with, since fractional sigma scales with the operating point "
                "(1%% of the 341.9 surgical private mean = 3.42 TPS, NOT 4.864)."),
            "source_runs": {
                "kanna_504_propagation": "0urxqwob",
                "kanna_478_sigma_hw_and_floorlock_risk": "mssuss3f",
                "ship_surgical357": "j7qao5e9",
                "floorlock_stark485": "pavotwci",
            },
        },
        # --- composition: the surgical-357 private band (PR item 1) ---
        "surgical357_private_mean": surg["private_tps_mean"],
        "surgical357_private_band_68": surg["private_tps_band_68"],
        "surgical357_private_95band": surg["private_tps_band_95"],
        "P_private_below_95pct": surg["p_private_below_95pct_public"],
        "P_surgical_below_floorlock": port["P_surgical_below_floorlock_point"],
        "band_surgical": surg,
        "band_surgical_round357": surg_round,
        "band_floorlock": floor,
        # --- portfolio price (PR item 2) ---
        "portfolio": port,
        "portfolio_verdict": port["portfolio_verdict"],
        # --- one-page dossier verdict (PR item 3) ---
        "one_line_summary": one_line,
        "dossier_verdict": dossier_verdict,
    }
    results["self_test"] = self_test(results)
    return results


def self_test(r: dict[str, Any]) -> dict[str, Any]:
    surg, floor, port = r["band_surgical"], r["band_floorlock"], r["portfolio"]
    checks: dict[str, bool] = {}

    # (1) reproduce #504 exactly (the surgical band is reused, not re-derived).
    if RECONCILE_504_JSON.exists():
        ref = json.loads(RECONCILE_504_JSON.read_text())
        checks["reproduces_504_mean"] = abs(surg["private_tps_mean"] - ref["surgical357_private_tps_mean"]) < 1e-6
        checks["reproduces_504_band95_lo"] = abs(surg["private_tps_band_95"][0] - ref["surgical357_private_tps_band_95"][0]) < 1e-6
        checks["reproduces_504_band95_hi"] = abs(surg["private_tps_band_95"][1] - ref["surgical357_private_tps_band_95"][1]) < 1e-6
        checks["reproduces_504_p_below"] = abs(surg["p_private_below_95pct_public"] - ref["P_private_below_95pct_public"]) < 1e-9
    else:
        checks["reproduces_504_present"] = False

    # (2) band ordering, both submissions.
    checks["surg_band_ordered"] = surg["private_tps_band_95"][0] < surg["private_tps_mean"] < surg["private_tps_band_95"][1]
    checks["floor_band_ordered"] = floor["private_tps_band_95"][0] < floor["private_tps_mean"] < floor["private_tps_band_95"][1]
    checks["surg_mean_near_342"] = abs(surg["private_tps_mean"] - 342.0) < 3.0
    checks["floor_mean_is_166"] = abs(floor["private_tps_mean"] - FLOORLOCK_TPS) < 1e-9
    checks["floor_zero_breach"] = floor["breach_frac"] == 0.0

    # (3) probabilities are valid probabilities.
    checks["P_private_below_95pct_valid"] = 0.0 <= surg["p_private_below_95pct_public"] <= 1.0
    checks["P_private_below_95pct_near_0p23"] = abs(surg["p_private_below_95pct_public"] - 0.2306) < 0.01
    checks["P_surgical_below_floorlock_valid"] = 0.0 <= port["P_surgical_below_floorlock_point"] <= 1.0
    checks["P_surgical_below_floorlock_approx_zero"] = port["P_surgical_below_floorlock_point"] < 1e-6

    # (4) raw-TPS dominance: surgical's worst case beats floor-lock's BEST case,
    #     even at the refuted 24% worst-case breach.
    checks["surg95lo_beats_floor95hi"] = surg["private_tps_band_95"][0] > floor["private_tps_band_95"][1]
    checks["worstcase24_95lo_beats_floor95hi"] = port["worstcase24_95lo_tps"] > floor["private_tps_band_95"][1]

    # (5) scoring-rule logic.
    rs = port["rule_invalidate_speed_threshold"]
    checks["penalize_winner_surgical"] = port["rule_penalize_breach"]["winner"] == "surgical"
    checks["speed_rule_p_invalidate_matches_p_below"] = abs(rs["P_surgical_invalidate"] - surg["p_private_below_95pct_public"]) < 1e-12
    checks["speed_rule_E_surgical_above_floor"] = rs["E_surgical_score_tps"] > floor["private_tps_mean"]
    # conditioning above a sub-mean threshold raises the conditional mean above the band mean;
    # the invalidation-discounted *score* must sit strictly between 0 and that mean.
    checks["speed_rule_E_given_valid_ge_mean"] = rs["E_surgical_given_valid_tps"] >= surg["private_tps_mean"] - 1e-9
    checks["speed_rule_E_score_discounted"] = (
        0.0 < rs["E_surgical_score_tps"] < surg["private_tps_mean"]
        and abs(rs["E_surgical_score_tps"]
                - (1.0 - rs["P_surgical_invalidate"]) * rs["E_surgical_given_valid_tps"]) < 1e-6)
    checks["literal_rule_winner_floorlock"] = port["rule_invalidate_literal_identity"]["winner"] == "floor-lock"
    checks["verdict_bracketed"] = r["portfolio_verdict"] == "bracketed"

    # (6) sigma_hw provenance roundtrips. The composition uses the scale-invariant 1%
    #     fractional convention; 0.01*481.53 == 4.8153 (the convention) and the measured
    #     one-shot 0.010128*481.53 == 4.877 both hold cleanly. (The PR's "4.864" between-leg
    #     TPS was measured on a ~505-mean pool, so 4.864/0.009623 ~ 505 != 481.53; it is
    #     provenance context, not a quantity composed with -- so not asserted as an identity.)
    checks["sigma_hw_convention_roundtrip"] = abs(SIGMA_HW_FRAC * SIGMA_HW_REF_TPS - 4.8153) < 0.01
    checks["sigma_hw_oneshot_tps_roundtrip"] = abs(SIGMA_HW_ONESHOT_FRAC_MEASURED * SIGMA_HW_REF_TPS - 4.877) < 0.01

    # (7) NaN-clean over every numeric leaf.
    checks["nan_clean"] = _all_finite(r)

    return {"checks": checks, "passes": all(checks.values())}


def _all_finite(obj: Any) -> bool:
    if isinstance(obj, bool):
        return True
    if isinstance(obj, (int, float)):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(_all_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_all_finite(v) for v in obj)
    return True  # strings / None


# --------------------------------------------------------------------------- #
# Pretty-print + W&B                                                          #
# --------------------------------------------------------------------------- #
def _print(r: dict[str, Any]) -> None:
    surg, floor, port = r["band_surgical"], r["band_floorlock"], r["portfolio"]
    print("\n[dossier] ===== SURGICAL-357 PRIVATE-OUTCOME DOSSIER + FLOOR-LOCK PORTFOLIO (PR #508) =====", flush=True)
    print("  COMPOSITION: private = public*(1-breach)*(1+-sigma_hw); breach=#504 LINEAR, sigma_hw=#478 one-shot", flush=True)
    print("  -- (1) surgical-357 private band (public=%.2f, breach=%.3f%%, sigma_hw=1%% frac) --" % (
        surg["public_tps"], surg["breach_pct"]), flush=True)
    print("     private mean %.1f | 68%% [%.1f, %.1f] | 95%% [%.1f, %.1f]" % (
        surg["private_tps_mean"], surg["private_tps_band_68"][0], surg["private_tps_band_68"][1],
        surg["private_tps_band_95"][0], surg["private_tps_band_95"][1]), flush=True)
    print("     P(private<0.95xpublic=%.1f) = %.4f ; P(private<floor-lock %.2f) = %.2e" % (
        surg["threshold_095_public"], surg["p_private_below_95pct_public"],
        FLOORLOCK_TPS, port["P_surgical_below_floorlock_point"]), flush=True)
    print("  -- floor-lock band (literal-1.0, ZERO breach; same sigma_hw) --", flush=True)
    print("     private mean %.2f | 95%% [%.2f, %.2f]" % (
        floor["private_tps_mean"], floor["private_tps_band_95"][0], floor["private_tps_band_95"][1]), flush=True)
    print("  -- (2) portfolio price --", flush=True)
    print("     expected private gap surgical-floorlock = +%.1f TPS (+%.0f%%)" % (
        port["expected_private_gap_tps"], port["expected_private_gap_pct"]), flush=True)
    print("     downside: surg 95%%-lo (%.1f) vs floor 95%%-hi (%.1f) = +%.1f (surg worst > floor best)" % (
        surg["private_tps_band_95"][0], floor["private_tps_band_95"][1],
        port["downside_gap_surg95lo_vs_floor_95hi_tps"]), flush=True)
    print("     even 24%% worst-case 95%%-lo (%.1f) > floor 95%%-hi (%.1f)" % (
        port["worstcase24_95lo_tps"], floor["private_tps_band_95"][1]), flush=True)
    rp, rs, rl = (port["rule_penalize_breach"], port["rule_invalidate_speed_threshold"],
                  port["rule_invalidate_literal_identity"])
    print("     RULE (a) penalize-breach        -> winner=%s (+%.1f TPS)" % (rp["winner"], rp["margin_expected_tps"]), flush=True)
    print("     RULE (b1) invalidate@speed-0.95 -> E[surgical]=%.1f (P_inval=%.1f%%) vs floor=%.2f ; E-value=%s, maximin=%s" % (
        rs["E_surgical_score_tps"], 100.0 * rs["P_surgical_invalidate"], rs["floorlock_score"],
        rs["winner_expected_value"], rs["winner_maximin"]), flush=True)
    print("     RULE (b2) invalidate@literal-id -> surgical=0 (operative-1.0) vs floor=%.2f -> winner=%s" % (
        rl["floorlock_score"], rl["winner"]), flush=True)
    print("  -- (3) VERDICT: portfolio_verdict = %s --" % r["portfolio_verdict"], flush=True)
    print("     %s" % port["portfolio_verdict_long"], flush=True)
    print("  SELF-TEST passes = %s" % r["self_test"]["passes"], flush=True)
    if not r["self_test"]["passes"]:
        for k, v in r["self_test"]["checks"].items():
            if not v:
                print("    FAILED: %s" % k, flush=True)
    print("\n  ONE-LINE: %s" % r["one_line_summary"], flush=True)
    print("  DOSSIER VERDICT: %s" % r["dossier_verdict"], flush=True)


def _flat_summary(r: dict[str, Any]) -> dict[str, float | int]:
    surg, floor, port = r["band_surgical"], r["band_floorlock"], r["portfolio"]
    rs = port["rule_invalidate_speed_threshold"]
    flat = {
        # KEY OUTPUTS
        "surgical357_private_mean": r["surgical357_private_mean"],
        "surgical357_private_95band_lo": r["surgical357_private_95band"][0],
        "surgical357_private_95band_hi": r["surgical357_private_95band"][1],
        "surgical357_private_68band_lo": surg["private_tps_band_68"][0],
        "surgical357_private_68band_hi": surg["private_tps_band_68"][1],
        "P_private_below_95pct": r["P_private_below_95pct"],
        "P_surgical_below_floorlock": r["P_surgical_below_floorlock"],
        "P_surgical_below_floorlock_conv": port["P_surgical_below_floorlock_conv"],
        # portfolio
        "expected_private_gap_tps": port["expected_private_gap_tps"],
        "expected_private_gap_pct": port["expected_private_gap_pct"],
        "downside_gap_surg95lo_vs_floor_mean_tps": port["downside_gap_surg95lo_vs_floor_mean_tps"],
        "downside_gap_surg95lo_vs_floor_95hi_tps": port["downside_gap_surg95lo_vs_floor_95hi_tps"],
        "worstcase24_mean_tps": port["worstcase24_mean_tps"],
        "worstcase24_95lo_tps": port["worstcase24_95lo_tps"],
        "worstcase24_downside_gap_vs_floor_95hi_tps": port["worstcase24_downside_gap_vs_floor_95hi_tps"],
        "floorlock_private_mean": floor["private_tps_mean"],
        "floorlock_private_95band_lo": floor["private_tps_band_95"][0],
        "floorlock_private_95band_hi": floor["private_tps_band_95"][1],
        # rule outcomes (numeric flags for filtering)
        "E_surgical_score_invalidate_speed": rs["E_surgical_score_tps"],
        "P_surgical_invalidate_speed": rs["P_surgical_invalidate"],
        "penalize_winner_surgical": int(port["rule_penalize_breach"]["winner"] == "surgical"),
        "speed_rule_winner_expected_value_surgical": int(rs["winner_expected_value"] == "surgical"),
        "speed_rule_winner_maximin_floorlock": int(rs["winner_maximin"] == "floor-lock"),
        "literal_rule_winner_floorlock": int(port["rule_invalidate_literal_identity"]["winner"] == "floor-lock"),
        "portfolio_verdict_bracketed": int(r["portfolio_verdict"] == "bracketed"),
        # inputs / provenance
        "ship_public_tps": SHIP_PUBLIC_TPS,
        "breach_frac": BREACH_FRAC,
        "breach_pct": 100.0 * BREACH_FRAC,
        "realized_propagation_factor": REALIZED_PROPAGATION_FACTOR,
        "sigma_hw_frac": SIGMA_HW_FRAC,
        "sigma_hw_between_frac": SIGMA_HW_BETWEEN_FRAC,
        "floorlock_tps": FLOORLOCK_TPS,
        "self_test_passes": int(r["self_test"]["passes"]),
    }
    return {k: v for k, v in flat.items()
            if isinstance(v, (int, float)) and math.isfinite(v)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="kanna/ship-private-dossier")
    ap.add_argument("--group", default="ship-private-dossier")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    r = build_results()
    _print(r)

    out_path = Path(__file__).resolve().parent / "dossier.json"
    out_path.write_text(json.dumps(r, indent=2))
    print("\n[dossier] artifacts -> %s" % out_path, flush=True)

    if not r["self_test"]["passes"]:
        print("[dossier] SELF-TEST FAILED -- not logging to W&B", flush=True)
        return 1
    if args.no_wandb:
        return 0

    run = wandb_logging.init_wandb_run(
        job_type="private-outcome-dossier", agent="kanna",
        name=args.name, group=args.group,
        tags=["ship-private-dossier", "private-outcome", "portfolio", "surgical357",
              "floor-lock", "sigma-hw-composition", "breach-composition", "reopen-decision",
              "analysis-only"],
        notes="Surgical-357 private-outcome band + floor-lock-vs-surgical portfolio price (reopen).",
        config={
            "pr": 508,
            "ship_public_tps": SHIP_PUBLIC_TPS,
            "breach_frac": BREACH_FRAC,
            "sigma_hw_frac": SIGMA_HW_FRAC,
            "floorlock_tps": FLOORLOCK_TPS,
            "denken_worstcase_breach_frac": DENKEN_WORSTCASE_BREACH,
            "analysis_only": True, "official_tps": 0,
            "source_runs": ["0urxqwob", "mssuss3f", "j7qao5e9", "pavotwci"],
        },
    )
    if run is None:
        print("[dossier] wandb disabled (no API key); skipping", flush=True)
        return 0
    wandb_logging.log_summary(run, _flat_summary(r), step=0)
    wandb_logging.log_json_artifact(
        run, name="ship_private_dossier", artifact_type="private-outcome-dossier", data=r)
    wandb_logging.finish_wandb(run)
    print("[dossier] wandb_run_id=%s" % getattr(run, "id", None), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
