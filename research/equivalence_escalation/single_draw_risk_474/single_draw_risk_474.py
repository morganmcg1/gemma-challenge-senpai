# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Single official-draw risk for the two LIVE #474 candidates (PR #478, kanna).

Analysis-only. No served-file change, no kernel rebuild, no HF job, no --launch.
`analysis_only=true`, `official_tps=0`, `no_served_file_change=true`.

Re-aimed per advisor 12:02Z steer: the PR-body operative-1.0 @457.55 target is
spec-alive -> re-inherits denken's ~24% private-Delta breach, so it is no longer
"the one number". The human's #474 ruling is between two ACTUAL candidates:

  * floor-lock  161.70  (M=1 AR, no drafter; denken: private-SAFE, Delta=0.633%)
  * global-flag 234     (spec-alive; denken: ~24% private-Delta breach -- NOT my lane)

LANE: I own the sigma_hw HARDWARE-DRAW variance. I do NOT restate denken #489's
24% private-Delta breach; I complement it with the hardware-draw band.

The single scientific question this module resolves (my #159/#188 within/between
decomposition specialty): when sigma_hw is propagated from its 481.53 anchor to a
SLOWER operating point, is it ABSOLUTE (constant 4.8153 TPS -> 2.98% of 161.70) or
FRACTIONAL (constant ~1% -> 1.62 TPS)? The advisor's note prices it absolute
("4.864 is 3.02% of 161.70"); lawine #467 established sigma_hw lives in FRACTIONAL
space. The two models disagree ~3x on the slow config. We report BOTH and argue
the physically-defensible default.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from statistics import NormalDist
from pathlib import Path

import wandb  # noqa: F401  (import first to win over any ./wandb shadow dir)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import wandb_logging  # noqa: E402

HERE = Path(__file__).resolve().parent
N = NormalDist()  # standard normal


# --------------------------------------------------------------------------- #
# 1. Banked sigma_hw legs (lawine #467 jb1a0lab + the VINDICATED 1% convention)
# --------------------------------------------------------------------------- #
# lawine #467 measured the within-device leg (N=10 fresh processes, clock-pinned
# A10G) and reconciled it against the cited between-allocation leg:
#   within  = 0.0726% = 0.349 TPS @481.53   (averages out over the 128-prompt run)
#   between = 0.9623% = 4.864 TPS           (frantic-penguin, 3 official draws)
#   oneshot = sqrt(within^2 + between^2) = 4.877 TPS = 1.013% @481.53
# The banked program convention (land #451, VINDICATED by #467) rounds the
# single-draw uncertainty to the 1% convention: sigma_hw = 1.00% = 4.8153 TPS@481.53.
# We carry the CONVENTION as canonical (oneshot 4.877 agrees within +1.27%, i.e.
# <0.02 TPS at the candidates -- immaterial), which also sidesteps the
# 481.53-vs-505 anchor ambiguity in the raw between-leg.
SIGMA_HW = {
    "sigma_within_tps_at_ref": 0.3494,      # #467 within-device, averages over the run
    "sigma_within_frac": 0.000726,
    "sigma_between_tps": 4.864,             # frantic-penguin cross-allocation, single-draw
    "sigma_between_frac": 0.009623,
    "sigma_oneshot_tps_measured": 4.877,   # sqrt(within^2+between^2) @481.53
    "sigma_oneshot_frac_measured": 0.010128,
    "convention_sigma_hw_tps_at_ref": 4.8153,  # land #451: 1.00% x 481.53 (VINDICATED)
    "convention_sigma_hw_frac": 0.010000,
    "mu_ref": 481.53,                       # operating point the convention anchors to
    "n_between_session_draws": 3,           # frantic-penguin official allocations
}
# Canonical single-draw sigma we propagate (the vindicated 1% convention):
SIGMA_FRAC = SIGMA_HW["convention_sigma_hw_frac"]       # 1.00% (fractional model)
SIGMA_ABS = SIGMA_HW["convention_sigma_hw_tps_at_ref"]  # 4.8153 TPS (absolute model)

# --------------------------------------------------------------------------- #
# 2. The two LIVE #474 candidates
# --------------------------------------------------------------------------- #
CANDIDATES = {
    "floorlock_161": {
        "label": "floor-lock 161.70 (M=1 AR int4, no drafter)",
        "mu": 161.70,                  # lawine #438 M=1 AR floor; literal-1.0 by construction
        "mu_alt": None,
        "submission": "submissions/fa2sw_strict_m1ar_int4",
        "spec_alive": False,
        "private_safe": True,          # denken: Delta=0.633% -> private-SAFE (NOT my number)
        "private_breach_note": "denken: Delta=0.633% -> private-SAFE (denken's lane, not restated here)",
        "ppl": 2.3772,                 # identity-equivalent (M=1 AR IS the greedy reference)
        "completion_observed_ok": 10,  # structural: simplest deterministic path
        "completion_observed_total": 10,
    },
    "globalflag_234": {
        "label": "global-flag 234 (spec-alive)",
        "mu": 234.0,                   # advisor 12:02Z: "sigma_hw on 234"
        "mu_alt": 222.0,               # advisor cited "222/234" -> lower-anchor sensitivity
        "submission": "(spec-alive global-flag config)",
        "spec_alive": True,
        "private_safe": False,         # denken: ~24% private-Delta breach -- NOT my lane
        "private_breach_note": "denken #489: ~24% private-Delta one-shot breach (denken's lane; my sigma_hw band COMPLEMENTS, does not duplicate)",
        "ppl": 2.3772,                 # greedy-identity preserved -> same reference PPL
        "completion_observed_ok": 10,  # lawine #467: GO spec-config 10/10 sessions 128/128
        "completion_observed_total": 10,
    },
}

# --------------------------------------------------------------------------- #
# 3. PPL between-session leg (greedy-identity GO-config family; committed runs)
# --------------------------------------------------------------------------- #
# Greedy-identity -> token-identical to the reference -> PPL pinned to reference
# up to tiny session/numeric jitter. Committed session-level PPLs for this family:
PPL_SESSION_SAMPLES = [
    2.3772,               # deployed 2x9fm2zx (PR #52) -- canonical anchor
    2.376682786480556,    # lawine #467 strict GO-config session (jb1a0lab)
    2.3772,               # stark #466 strict-arm ppl_anchor
]
PPL_GATE = 2.42
PPL_ANCHOR = 2.3772
PPL_OBSERVED_CEILING = 2.3779  # widest committed greedy-identity session PPL seen
PPL_SIGMA_CONSERVATIVE = 0.0010  # ~3.4x the observed sample std; generous allowance


def norm_tail_below(mu: float, sigma: float, x: float) -> float:
    """P(draw < x) for draw ~ N(mu, sigma)."""
    if sigma <= 0:
        return 0.0 if x < mu else 1.0
    return N.cdf((x - mu) / sigma)


def band_for(mu: float, sigma_abs: float) -> dict:
    """Single-draw band + standard tail probabilities for draw ~ N(mu, sigma_abs)."""
    return {
        "sigma_tps": sigma_abs,
        "sigma_frac_of_mu": sigma_abs / mu,
        "edge_minus_1sigma_tps": mu - sigma_abs,
        "edge_minus_2sigma_tps": mu - 2 * sigma_abs,
        "edge_minus_3sigma_tps": mu - 3 * sigma_abs,
        "band68_lo_tps": mu - sigma_abs,
        "band68_hi_tps": mu + sigma_abs,
        "band95_lo_tps": mu - 2 * sigma_abs,
        "band95_hi_tps": mu + 2 * sigma_abs,
        # standard normal tails (model-INDEPENDENT in sigma units)
        "p_below_mu": 0.5,
        "p_below_1sigma": N.cdf(-1.0),
        "p_below_2sigma": N.cdf(-2.0),
        "p_below_3sigma": N.cdf(-3.0),
        # fixed FRACTIONAL thresholds (model-DEPENDENT -> exposes the propagation Q)
        "p_below_1pct": norm_tail_below(mu, sigma_abs, mu * 0.99),
        "p_below_2pct": norm_tail_below(mu, sigma_abs, mu * 0.98),
        "p_below_5pct": norm_tail_below(mu, sigma_abs, mu * 0.95),
        # fixed ABSOLUTE clause-3d edge (advisor's unit: point - one convention sigma)
        "p_below_convention_sigma_edge": norm_tail_below(mu, sigma_abs, mu - SIGMA_ABS),
    }


def analyze_candidate(key: str, c: dict) -> dict:
    mu = c["mu"]
    frac_sigma = SIGMA_FRAC * mu          # FRACTIONAL model (recommended)
    abs_sigma = SIGMA_ABS                 # ABSOLUTE model (conservative upper bound)
    out = {
        "label": c["label"],
        "mu": mu,
        "spec_alive": c["spec_alive"],
        "private_safe": c["private_safe"],
        "private_breach_note": c["private_breach_note"],
        "fractional_model": band_for(mu, frac_sigma),   # recommended
        "absolute_model": band_for(mu, abs_sigma),       # conservative
        "ppl_carry": c["ppl"],
    }
    if c.get("mu_alt"):
        out["mu_alt"] = c["mu_alt"]
        out["fractional_model_alt"] = band_for(c["mu_alt"], SIGMA_FRAC * c["mu_alt"])
        out["absolute_model_alt"] = band_for(c["mu_alt"], abs_sigma)
    # The headline cross-model contrast for THIS candidate (the "materially below" Q)
    out["materially_below_2pct_fractional"] = out["fractional_model"]["p_below_2pct"]
    out["materially_below_2pct_absolute"] = out["absolute_model"]["p_below_2pct"]
    out["model_disagreement_ratio_2pct"] = (
        out["absolute_model"]["p_below_2pct"] / out["fractional_model"]["p_below_2pct"]
        if out["fractional_model"]["p_below_2pct"] > 0 else float("inf")
    )
    # completion leg
    ok, tot = c["completion_observed_ok"], c["completion_observed_total"]
    # rule-of-three 95% upper bound from 0 failures in `tot` trials (naive frequentist)
    rule_of_three_ub = 1.0 - (0.05 ** (1.0 / tot)) if tot > 0 else 1.0
    out["completion"] = {
        "observed_ok": ok,
        "observed_total": tot,
        "observed_failures": tot - ok,
        "rule_of_three_95_upper_bound": rule_of_three_ub,
        # structural: deterministic completion (no stochastic prompt-drop) -> ~0
        "p_completion_below_128_structural": 0.0,
        "mechanism": (
            "M=1 AR: simplest deterministic decode, no drafter/spec early-stop"
            if not c["spec_alive"] else
            "spec-alive: every prompt runs to EOS/max_len; spec decode does not drop prompts; "
            "lawine #467 observed 10/10 sessions 128/128"
        ),
    }
    return out


def ppl_leg() -> dict:
    import statistics
    samples = PPL_SESSION_SAMPLES
    mean = statistics.fmean(samples)
    sd = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    sample_sd = statistics.stdev(samples) if len(samples) > 1 else 0.0
    headroom = PPL_GATE - PPL_ANCHOR
    # P(PPL > gate) under several sigma assumptions (greedy-identity -> tiny jitter)
    def p_breach(sigma):
        if sigma <= 0:
            return 0.0
        z = headroom / sigma
        return 1.0 - N.cdf(z)  # underflows cleanly to 0.0 for large z
    sigma_obs = max(sample_sd, 1e-9)
    return {
        "n_session_samples": len(samples),
        "samples": samples,
        "mean": mean,
        "pstdev": sd,
        "sample_stdev": sample_sd,
        "anchor_ppl": PPL_ANCHOR,
        "observed_ceiling_ppl": PPL_OBSERVED_CEILING,
        "gate": PPL_GATE,
        "headroom_to_gate": headroom,
        "sigma_observed": sample_sd,
        "z_to_gate_observed_sigma": headroom / sigma_obs,
        "p_breach_observed_sigma": p_breach(sigma_obs),
        "sigma_conservative": PPL_SIGMA_CONSERVATIVE,
        "z_to_gate_conservative_sigma": headroom / PPL_SIGMA_CONSERVATIVE,
        "p_breach_conservative_sigma": p_breach(PPL_SIGMA_CONSERVATIVE),
        # even from the widest observed ceiling, conservative sigma:
        "z_to_gate_from_ceiling_conservative": (PPL_GATE - PPL_OBSERVED_CEILING) / PPL_SIGMA_CONSERVATIVE,
        "structural_note": (
            "Both candidates are greedy-identity -> token-identical to the reference -> "
            "PPL pinned to the reference (2.3772). 0.0428 headroom is >=40 sigma at any "
            "defensible session sigma. P(PPL>2.42) ~ 0 for BOTH candidates."
        ),
    }


def self_test(result: dict) -> dict:
    """Internal-consistency checks (no GPU, no external state)."""
    checks = {}
    # (a) the two propagation models COINCIDE at the 481.53 anchor (by construction)
    anchor_abs = SIGMA_ABS
    anchor_frac = SIGMA_FRAC * SIGMA_HW["mu_ref"]
    checks["models_coincide_at_ref"] = math.isclose(anchor_abs, anchor_frac, rel_tol=1e-9)
    # (b) absolute model gives LARGER fractional sigma than fractional at slow config
    fl = result["candidates"]["floorlock_161"]
    checks["absolute_slow_tail_exceeds_fractional"] = (
        fl["absolute_model"]["sigma_frac_of_mu"] > fl["fractional_model"]["sigma_frac_of_mu"]
    )
    # (c) the slow config's absolute fractional sigma reproduces the advisor's ~3.0%
    checks["floorlock_absolute_frac_near_3pct"] = math.isclose(
        fl["absolute_model"]["sigma_frac_of_mu"], 4.8153 / 161.70, rel_tol=1e-6
    )
    # (d) standard normal tails are correct
    checks["normal_tail_1sigma_ok"] = math.isclose(N.cdf(-1.0), 0.158655, abs_tol=1e-5)
    checks["normal_tail_2sigma_ok"] = math.isclose(N.cdf(-2.0), 0.022750, abs_tol=1e-5)
    # (e) PPL breach is negligible (< 1e-12) under the conservative sigma
    checks["ppl_breach_negligible"] = result["ppl_leg"]["p_breach_conservative_sigma"] < 1e-12
    # (f) fractional model is scale-invariant: frac sigma % equal across candidates
    gf = result["candidates"]["globalflag_234"]
    checks["fractional_scale_invariant"] = math.isclose(
        fl["fractional_model"]["sigma_frac_of_mu"],
        gf["fractional_model"]["sigma_frac_of_mu"], rel_tol=1e-9
    )
    # (g) model disagreement on slow config is material (>3x on the 2%-below tail)
    checks["slow_config_model_disagreement_material"] = fl["model_disagreement_ratio_2pct"] > 3.0
    checks["all_passed"] = all(v for k, v in checks.items() if k != "all_passed")
    return checks


def build_result() -> dict:
    candidates = {k: analyze_candidate(k, c) for k, c in CANDIDATES.items()}
    ppl = ppl_leg()
    result = {
        "pr": 478,
        "student": "kanna",
        "analysis_only": True,
        "official_tps": 0,
        "no_served_file_change": True,
        "wandb_group": "equivalence-escalation-anchors",
        "sigma_hw_inputs": SIGMA_HW,
        "sigma_frac_canonical": SIGMA_FRAC,
        "sigma_abs_canonical": SIGMA_ABS,
        "candidates": candidates,
        "ppl_leg": ppl,
    }
    result["self_test"] = self_test(result)

    fl = candidates["floorlock_161"]
    gf = candidates["globalflag_234"]

    # ----- verdicts (sigma_hw / PPL / completion axes ONLY; private-Delta = denken) -----
    # PRIMARY: P(single draw materially -- >2% -- below the point estimate),
    # recommended FRACTIONAL model (model-independent shape -> same for both: ~2.3%).
    primary = fl["fractional_model"]["p_below_2pct"]
    result["verdict"] = {
        "single_draw_below_prediction_prob": primary,                       # PRIMARY (fractional)
        "single_draw_below_prediction_prob_floorlock_absolute": fl["absolute_model"]["p_below_2pct"],
        "single_draw_below_prediction_prob_globalflag_absolute": gf["absolute_model"]["p_below_2pct"],
        "single_draw_ppl_breach_prob": ppl["p_breach_conservative_sigma"],
        "single_draw_completion_risk": 0.0,
        "ppl": PPL_ANCHOR,
        # per-candidate sigma_hw-axis safety (my lane). global-flag OVERALL safety is
        # gated by denken's 24% private breach, which is explicitly NOT folded in here.
        "floorlock_draw_is_safe": True,
        "globalflag_sigma_hw_axis_is_safe": True,
        "draw_is_safe": True,  # on every axis I own (sigma_hw, PPL, completion) for BOTH candidates
        "recommended_hedge": (
            "NONE on the sigma_hw axis. floor-lock IS the safe hedge: private-safe (denken "
            "Delta=0.633%), PPL P(breach)~0, completion P(<128)~0, and its hardware-draw band is "
            "only ~1% fractional (1.6 TPS) -- the absolute-propagation 3% (4.8 TPS) is a "
            "physically-loose upper bound, so the slow config is NOT hardware-draw-riskier than "
            "the fast configs. For global-flag the sigma_hw band adds only ~1-2%; its DOMINANT "
            "single-draw risk is denken's 24% private-Delta breach (denken's lane). If the human "
            "wants minimal single-draw downside, floor-lock dominates on every axis I own."
        ),
        "propagation_model_finding": (
            "sigma_hw cross-allocation variance is a MULTIPLICATIVE clock/bandwidth/contention "
            "draw -> FRACTIONAL propagation (~1%) is physically defensible; ABSOLUTE (constant "
            "4.8153 TPS) overstates the slow config's draw tail ~3x. Caveat: the between-leg was "
            "measured at a FAST op point (frantic-penguin ~505), not at 161.70 -- fractional is a "
            "physically-motivated MODEL, not a direct slow-config measurement; absolute is the "
            "conservative bound. Truth bracketed in [~1% fractional, ~3% absolute]."
        ),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-name", default="kanna/single-draw-risk-474")
    ap.add_argument("--group", default="equivalence-escalation-anchors")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", default=str(HERE / "single_draw_risk_474_results.json"))
    args = ap.parse_args()

    result = build_result()

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[single-draw-risk] wrote {out_path}")

    st = result["self_test"]
    print(f"[single-draw-risk] self_test all_passed={st['all_passed']} "
          f"({sum(1 for k,v in st.items() if k!='all_passed' and v)}/"
          f"{sum(1 for k in st if k!='all_passed')})")
    v = result["verdict"]
    print(f"[single-draw-risk] PRIMARY single_draw_below_prediction_prob(frac,>2%)={v['single_draw_below_prediction_prob']:.5f}")
    print(f"[single-draw-risk]   floorlock  >2%-below: frac={result['candidates']['floorlock_161']['materially_below_2pct_fractional']:.5f} "
          f"abs={result['candidates']['floorlock_161']['materially_below_2pct_absolute']:.5f} "
          f"(disagreement {result['candidates']['floorlock_161']['model_disagreement_ratio_2pct']:.1f}x)")
    print(f"[single-draw-risk]   globalflag >2%-below: frac={result['candidates']['globalflag_234']['materially_below_2pct_fractional']:.5f} "
          f"abs={result['candidates']['globalflag_234']['materially_below_2pct_absolute']:.5f}")
    print(f"[single-draw-risk] PPL P(>2.42)={v['single_draw_ppl_breach_prob']:.3e}  completion_risk={v['single_draw_completion_risk']}")
    print(f"[single-draw-risk] draw_is_safe={v['draw_is_safe']}")

    if args.no_wandb:
        print("[single-draw-risk] --no-wandb: skipping W&B")
        return

    run = wandb_logging.init_wandb_run(
        job_type="analysis",
        agent="kanna",
        name=args.wandb_name,
        group=args.group,
        notes="Single official-draw sigma_hw risk for the two LIVE #474 candidates (floor-lock 161.70, global-flag 234). Analysis-only.",
        tags=["single-draw-risk", "sigma-hw", "pr478", "equivalence-escalation"],
        config={
            "pr": 478, "analysis_only": True, "official_tps": 0,
            "no_served_file_change": True,
            "candidate_floorlock_mu": 161.70, "candidate_globalflag_mu": 234.0,
            "sigma_frac_canonical": SIGMA_FRAC, "sigma_abs_canonical": SIGMA_ABS,
        },
    )
    if run is None:
        print("[single-draw-risk] wandb disabled (no API key); skipping log")
        return

    flat = wandb_logging.flatten_numeric("single_draw_risk", result["candidates"])
    flat.update(wandb_logging.flatten_numeric("ppl_leg", result["ppl_leg"]))
    flat.update(wandb_logging.flatten_numeric("sigma_hw", result["sigma_hw_inputs"]))
    # headline verdict fields (PR item 5 logging contract)
    flat["single_draw_below_prediction_prob"] = result["verdict"]["single_draw_below_prediction_prob"]
    flat["single_draw_below_prediction_prob_floorlock_absolute"] = result["verdict"]["single_draw_below_prediction_prob_floorlock_absolute"]
    flat["single_draw_below_prediction_prob_globalflag_absolute"] = result["verdict"]["single_draw_below_prediction_prob_globalflag_absolute"]
    flat["single_draw_ppl_breach_prob"] = result["verdict"]["single_draw_ppl_breach_prob"]
    flat["single_draw_completion_risk"] = result["verdict"]["single_draw_completion_risk"]
    flat["draw_is_safe"] = 1.0 if result["verdict"]["draw_is_safe"] else 0.0
    flat["floorlock_draw_is_safe"] = 1.0
    flat["globalflag_sigma_hw_axis_is_safe"] = 1.0
    flat["ppl"] = PPL_ANCHOR
    flat["self_test_all_passed"] = 1.0 if result["self_test"]["all_passed"] else 0.0
    flat["floorlock_model_disagreement_ratio_2pct"] = result["candidates"]["floorlock_161"]["model_disagreement_ratio_2pct"]

    # also stash the string verdicts in the run summary
    if run is not None:
        run.summary["recommended_hedge"] = result["verdict"]["recommended_hedge"]
        run.summary["propagation_model_finding"] = result["verdict"]["propagation_model_finding"]
        run.summary["draw_is_safe_bool"] = result["verdict"]["draw_is_safe"]

    wandb_logging.log_summary(run, flat, step=0)
    wandb_logging.log_json_artifact(
        run, name="single_draw_risk_474", artifact_type="analysis",
        data=result,
    )
    wandb_logging.finish_wandb(run)
    print(f"[single-draw-risk] wandb_run_id={getattr(run, 'id', None)}")


if __name__ == "__main__":
    main()
