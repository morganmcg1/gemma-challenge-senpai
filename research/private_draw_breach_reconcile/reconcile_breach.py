"""PR #504 -- Reconcile the private-TPS-breach magnitude (4.3% vs 24%) on surgical-357.

analysis_only=true, official_tps=0, CPU-only. NO serve, NO HF job, NO --launch,
NO submission, NO served-file change. Same discipline as #478/#497.

================================================================================
THE TENSION (from the PR body)
================================================================================
The surgical-357 ship is spec-alive (keeps the MTP K=7 drafter), so its private
leaderboard TPS inherits the drafter's public->private acceptance shift.
  * denken #489 (q1ivw9tt) MEASURED that acceptance shift: dalpha/alpha = 4.295%,
    and MODELED it into a ~24% TPS-drift breach.
  * The deployed baseline PR #52 (2x9fm2zx) is the SAME spec-alive MTP K=7 stack and
    reports public 481.53 / private 460.85 = a 4.29% public->private TPS gap -- almost
    exactly the 4.295% acceptance delta, ~5.6x SMALLER than the modeled 24%.
Does a 4.295% acceptance drop propagate to a ~4.3% TPS breach (linear) or a ~24%
breach (amplified)? The answer sets the ship's private outcome at ~342 (linear) or
~271 (amplified).

================================================================================
THE STRUCTURAL CRUX (this module's novel contribution: the TPS-propagation leg)
================================================================================
In MTP K=7 speculative decoding, each verify step ALWAYS drafts K tokens (one MTP
forward) and ALWAYS verifies K positions (one target forward). Acceptance changes
only the YIELD per step (how many drafted tokens survive), NOT the per-step wall
cost. So the per-step cost C_step = C_draft + C_verify is INVARIANT to acceptance:

    TPS = E[T] / C_step              E[T] = mean tokens emitted per verify step
    dTPS/TPS = dE[T]/E[T]            (C_step cancels -- it does not depend on alpha)
    PF := (dTPS/TPS) / (dalpha/alpha) = elasticity of E[T] w.r.t. the acceptance
                                        quantity that was perturbed.

So the 4.3%-vs-24% question reduces to: WHICH acceptance elasticity does the data
realize? denken owns the acceptance leg (dalpha=4.295%, taken here as an INPUT, NOT
re-derived). We own the TPS leg: measure/derive dTPS/TPS on the same axis and divide.

Three legs, none requiring the serve stack to come up:
  (1b) ANALYTIC: elasticity of E[T] w.r.t. acceptance under the natural acceptance
       definitions, evaluated at the deployed op point (E[T]=3.8512, K=7, per-position
       a_k banked in lawine #282 / kanna #289). Bounds the structural PF.
  (anchor) DEPLOYED EMPIRICAL ARBITER: PR #52 public 481.53 / private 460.85 = 4.29%
       realized TPS gap vs denken dalpha=4.295% -> realized PF ~= 1.0 (linear). This is
       the strongest evidence: the real private leaderboard, the exact stack, the exact
       acceptance quantity denken measured.
  (xcheck) WITHIN-RUN GPU-MEASURED: kanna #478 sigma_hw run jb1a0lab on the EXACT
       deployed fa2sw_precache_kenyan MTP K=7 stack banked paired (E[T], TPS) across 10
       runs; regress to estimate the structural elasticity dTPS/dE[T] INDEPENDENTLY of
       the anchor. (Substitutes for the blocked GPU bracket leg -- see GPU_BLOCKER.)

  (3) COMPOSITION: surgical-357 private band = ship_public x (1 - breach) x (1 +- sigma_hw),
      sigma_hw ~= 1.00% fractional (kanna #478, between-allocation-dominated one-shot).

Run under the repo .venv (CPU, ~1s, no GPU)::

    .venv/bin/python -m research.private_draw_breach_reconcile.reconcile_breach
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import wandb  # noqa: F401  (import first to win over any ./wandb shadow dir)

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.append(str(ROOT))

from scripts import wandb_logging  # noqa: E402

# --------------------------------------------------------------------------- #
# INPUTS (cited baselines -- all on advisor branch / my own merged work)       #
# --------------------------------------------------------------------------- #
K_SPEC = 7  # MTP num_speculative_tokens (both manifests: fa2sw_precache_kenyan / surgical357)

# Deployed op point: E[T] and per-position conditional acceptance a_k.
# Banked whole-run in lawine #282 (2j0e8xgg); decomposed in kanna #289 (fi34s269).
E_T_DEPLOYED = 3.8512
A_PER_POSITION = [0.729, 0.760, 0.793, 0.823, 0.835, 0.836, 0.846]  # a_1..a_7 (#289)

# denken #489 acceptance leg (q1ivw9tt) -- INPUT, NOT re-derived (lane discipline).
DELTA_ALPHA_FRAC = 0.04295          # 4.295% public->private acceptance drop (measured)
DENKEN_MODELED_BREACH_FRAC = 0.24   # ~24% modeled TPS-drift breach (the model under test)

# Deployed anchor (the empirical arbiter): PR #52 (2x9fm2zx), spec-alive MTP K=7.
DEPLOYED_PUBLIC_TPS = 481.53
DEPLOYED_PRIVATE_TPS = 460.85       # observed private leaderboard TPS

# Ship being priced: surgical-357 (firing on stark #499), spec-alive (MTP K=7 kept).
# recert l0attso0 = 357.22 (lawine #488 / stark #494 42qroec1); PR formula writes "357".
SHIP_PUBLIC_TPS = 357.22
SHIP_PUBLIC_TPS_ROUND = 357.0

# sigma_hw (kanna #478 mssuss3f, MERGED): one-shot ~= 1.00% fractional, between-dominated
# (within-leg jb1a0lab 0.0726%; between-leg #159/#188 0.9623%; convention 4.8153 @481.53).
SIGMA_HW_FRAC = 0.01                 # 1.00% fractional one-shot convention (#478-vindicated)
SIGMA_HW_FRAC_RECONSTRUCTED = None   # filled from the #478 json if available

# Banked within-run paired (E[T], TPS) source (kanna #478, run jb1a0lab).
SIGMA_HW_JSON = ROOT / "research/empirical_sigma_hw/fresh_n10/sigma_hw.json"

# Why the primary GPU bracket leg fell back to banked data + analytic.
GPU_BLOCKER = (
    "Full production serve stack not brought up on pod-A10G: vLLM not importable in "
    "base env; custom cu129 wheel (vllm-0.22.1rc1.dev307+g3e8afdf78.cu129) vs pod CUDA "
    "13.2 (driver 580.159.04); 40KB serve.py + 52KB sitecustomize + 8 custom-kernel "
    "monkeypatches (FUSED_SPARSE_ARGMAX, ONEGRAPH/LOOPGRAPH, splitkv-verify, PLE-fold, "
    "precache, surgical_attn) + multi-bucket artifact downloads (osoi5-v0-baked int4, "
    "kenyan-duma drafter ft-v1-epoch_001, dixie lmhead12k keepset, qat-assistant); none "
    "cached. Disproportionate to a cross-check that cannot beat the deployed private "
    "leaderboard anchor. SUBSTITUTED: my own #478 banked (E[T],TPS) pairs from the EXACT "
    "deployed fa2sw_precache_kenyan stack (run jb1a0lab, 10 GPU runs). PR sanctions the "
    "CPU path: 'the deployed anchor alone is enough to reconcile.'"
)
# The #497 bracket splits carry IDENTITY-flip rates (attention argmax flip), NOT drafter
# acceptance or TPS, so they cannot yield a bracket-resolved propagation factor; they are
# ubel's identity lane, not the acceptance/TPS lane. Used only as qualitative off-dist
# context (private flip ratio 0.50x easy / 0.37x hard < 1 => stack is BOUNDED, not
# collapsing, off-distribution -- consistent with a mild/linear rather than catastrophic
# private regime).
BRACKET_PRIVATE_FLIP_RATIO_EASY = 0.49674479166666663  # ubel #497 wdaje3eh
BRACKET_PRIVATE_FLIP_RATIO_HARD = 0.37255859375        # ubel #497 nq0mslrb


# --------------------------------------------------------------------------- #
# Spec-dec E[T] models and their acceptance elasticities                       #
# --------------------------------------------------------------------------- #
def e_t_from_per_position(a: list[float]) -> tuple[float, list[float]]:
    """E[T] = 1 + sum_m G(m), G(m) = prod_{k<=m} a_k (survival of accepted-draft count)."""
    g, surv = 1.0, []
    for ak in a:
        g *= ak
        surv.append(g)
    return 1.0 + sum(surv), surv


def e_t_geometric(alpha: float, k: int) -> float:
    """Leviathan 2023: E[T] = (1 - alpha^(k+1)) / (1 - alpha) for per-token accept alpha."""
    if abs(1.0 - alpha) < 1e-12:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def solve_geometric_alpha(target_et: float, k: int) -> float:
    """Invert E[T]=(1-a^(k+1))/(1-a) for a in (0,1) by bisection."""
    lo, hi = 1e-6, 1.0 - 1e-9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if e_t_geometric(mid, k) < target_et:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def elasticity_geometric(alpha: float, k: int) -> float:
    """d ln E[T] / d ln alpha for the geometric model, closed form."""
    num = -(k + 1) * alpha**k * (1.0 - alpha) + (1.0 - alpha ** (k + 1))
    denom = (1.0 - alpha) * (1.0 - alpha ** (k + 1))
    return alpha * num / denom


def central_log_elasticity(f, x0: float, rel: float = 1e-4) -> float:
    """Numerical d ln f / d ln x at x0 (central difference) -- cross-checks closed forms."""
    xh, xl = x0 * (1 + rel), x0 * (1 - rel)
    return (math.log(f(xh)) - math.log(f(xl))) / (math.log(xh) - math.log(xl))


# --------------------------------------------------------------------------- #
# Normal CDF (no scipy dependency)                                             #
# --------------------------------------------------------------------------- #
def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# --------------------------------------------------------------------------- #
# Leg 1b: analytic elasticities at the deployed op point                       #
# --------------------------------------------------------------------------- #
def analytic_elasticities() -> dict[str, Any]:
    et_pp, surv = e_t_from_per_position(A_PER_POSITION)

    # (A) alpha = mean acceptance_rate = (E[T]-1)/K  (the vLLM accept-rate metric;
    #     E[T] = 1 + K*alpha_A so dE[T]/dalpha_A = K and elasticity = (E[T]-1)/E[T]).
    alpha_A = (E_T_DEPLOYED - 1.0) / K_SPEC
    eps_A = (E_T_DEPLOYED - 1.0) / E_T_DEPLOYED

    # (B) alpha = per-token geometric acceptance (Leviathan); solve a for E[T]=deployed.
    alpha_B = solve_geometric_alpha(E_T_DEPLOYED, K_SPEC)
    eps_B = elasticity_geometric(alpha_B, K_SPEC)
    eps_B_num = central_log_elasticity(lambda a: e_t_geometric(a, K_SPEC), alpha_B)

    # (C) alpha = each per-position conditional a_k drops by dalpha relatively (compounding).
    a_shift = [ak * (1.0 - DELTA_ALPHA_FRAC) for ak in A_PER_POSITION]
    et_shift, _ = e_t_from_per_position(a_shift)
    breach_C = (et_pp - et_shift) / et_pp
    eps_C = breach_C / DELTA_ALPHA_FRAC

    # (D) worst-case: alpha enters throughput as deep-block survival G(K)=prod a_k ~ alpha^p.
    #     Elasticity of alpha^p w.r.t. alpha is p; the breach for a dalpha drop is ~p*dalpha.
    #     The 24% model implies an effective power p_needed = breach/ dalpha.
    pf_needed_for_24 = DENKEN_MODELED_BREACH_FRAC / DELTA_ALPHA_FRAC
    breach_full_block = K_SPEC * DELTA_ALPHA_FRAC  # G(K)=alpha^K elasticity = K = 7

    return {
        "et_from_per_position": et_pp,
        "survival_G": surv,
        # A
        "alpha_A_mean_accept_rate": alpha_A,
        "elasticity_A_mean_accept": eps_A,
        "breach_A_pct": 100.0 * eps_A * DELTA_ALPHA_FRAC,
        # B
        "alpha_B_geometric": alpha_B,
        "elasticity_B_geometric": eps_B,
        "elasticity_B_geometric_numcheck": eps_B_num,
        "breach_B_pct": 100.0 * eps_B * DELTA_ALPHA_FRAC,
        # C
        "elasticity_C_per_position_compound": eps_C,
        "breach_C_pct": 100.0 * breach_C,
        # D / worst-case
        "pf_needed_for_24pct": pf_needed_for_24,
        "elasticity_D_full_block_alpha_pow_K": float(K_SPEC),
        "breach_D_full_block_pct": 100.0 * breach_full_block,
        # the natural-elasticity envelope (A..C) the structure admits
        "natural_elasticity_lo": min(eps_A, eps_B, eps_C),
        "natural_elasticity_hi": max(eps_A, eps_B, eps_C),
    }


# --------------------------------------------------------------------------- #
# Deployed anchor: realized propagation factor on the real private leaderboard  #
# --------------------------------------------------------------------------- #
def deployed_anchor() -> dict[str, Any]:
    realized_gap = (DEPLOYED_PUBLIC_TPS - DEPLOYED_PRIVATE_TPS) / DEPLOYED_PUBLIC_TPS
    pf_realized = realized_gap / DELTA_ALPHA_FRAC
    return {
        "deployed_public_tps": DEPLOYED_PUBLIC_TPS,
        "deployed_private_tps": DEPLOYED_PRIVATE_TPS,
        "deployed_realized_tps_gap_pct": 100.0 * realized_gap,
        "denken_delta_alpha_pct": 100.0 * DELTA_ALPHA_FRAC,
        "realized_propagation_factor": pf_realized,
    }


# --------------------------------------------------------------------------- #
# Within-run GPU-measured cross-check (banked #478, run jb1a0lab)              #
# --------------------------------------------------------------------------- #
def within_run_elasticity() -> dict[str, Any]:
    if not SIGMA_HW_JSON.exists():
        return {"available": False, "reason": f"missing {SIGMA_HW_JSON}"}
    data = json.loads(SIGMA_HW_JSON.read_text())
    recs = data.get("records", [])
    et = np.array([r["e_accept_exact"] for r in recs], dtype=float)
    steady = np.array([r["steady_gen_tps_mean"] for r in recs], dtype=float)
    wall = np.array([r["wall_tps"] for r in recs], dtype=float)
    n = len(et)
    out: dict[str, Any] = {"available": True, "n_runs": n,
                           "source_run": data["analysis"].get("wandb_run_id"),
                           "submission": data.get("submission")}

    def reg_elasticity(y: np.ndarray) -> dict[str, float]:
        # OLS slope of y on E[T]; elasticity = slope * mean(E[T])/mean(y).
        # If TPS = E[T]/C_step (C_step alpha-invariant), elasticity == 1 exactly.
        sx, sy = et.std(), y.std()
        if sx == 0 or sy == 0:
            return {"slope": float("nan"), "r": float("nan"), "elasticity": float("nan")}
        slope = float(np.polyfit(et, y, 1)[0])
        r = float(np.corrcoef(et, y)[0, 1])
        elasticity = slope * float(et.mean()) / float(y.mean())
        return {"slope": slope, "r": r, "elasticity": elasticity}

    def bootstrap_elasticity_ci(y: np.ndarray, n_boot: int = 5000) -> list[float]:
        # Resample (E[T], y) pairs to bound the noise-limited slope-elasticity.
        rng = np.random.default_rng(504)
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            xb, yb = et[idx], y[idx]
            if xb.std() == 0 or yb.std() == 0:
                continue
            slope = float(np.polyfit(xb, yb, 1)[0])
            vals.append(slope * float(xb.mean()) / float(yb.mean()))
        if not vals:
            return [float("nan"), float("nan")]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    out["et_mean"] = float(et.mean())
    out["et_cv_pct"] = float(100.0 * et.std(ddof=1) / et.mean())
    out["steady_gen_tps"] = reg_elasticity(steady)
    out["wall_tps"] = reg_elasticity(wall)
    out["steady_gen_tps"]["cv_pct"] = float(100.0 * steady.std(ddof=1) / steady.mean())
    out["wall_tps"]["cv_pct"] = float(100.0 * wall.std(ddof=1) / wall.mean())
    out["steady_gen_tps"]["elasticity_ci95"] = bootstrap_elasticity_ci(steady)
    # reconstructed one-shot sigma_hw fraction from the #478 analysis (for provenance).
    out["sigma_hw_within_frac_pct"] = data["analysis"].get("empirical_sigma_hw_frac_pct")
    out["sigma_hw_convention_frac"] = data["analysis"].get("convention_sigma_hw_frac")
    return out


# --------------------------------------------------------------------------- #
# Leg 3: surgical-357 private single-draw band (breach (X) sigma_hw)            #
# --------------------------------------------------------------------------- #
def compose_band(ship_public: float, breach_frac: float, sigma_frac: float) -> dict[str, Any]:
    mean = ship_public * (1.0 - breach_frac)
    sigma_tps = mean * sigma_frac
    z95 = 1.959963984540054
    band68 = (mean - sigma_tps, mean + sigma_tps)
    band95 = (mean - z95 * sigma_tps, mean + z95 * sigma_tps)
    threshold = 0.95 * ship_public
    # P(private draw materially below public x 0.95) under the fractional sigma_hw draw.
    p_below = norm_cdf((threshold - mean) / sigma_tps) if sigma_tps > 0 else float("nan")
    return {
        "ship_public_tps": ship_public,
        "breach_frac": breach_frac,
        "breach_pct": 100.0 * breach_frac,
        "sigma_hw_frac": sigma_frac,
        "private_tps_mean": mean,
        "private_tps_sigma": sigma_tps,
        "private_tps_band_68": list(band68),
        "private_tps_band_95": list(band95),
        "threshold_095_public": threshold,
        "p_private_below_95pct_public": p_below,
    }


# --------------------------------------------------------------------------- #
# Verdict + self-test                                                          #
# --------------------------------------------------------------------------- #
def build_results() -> dict[str, Any]:
    ana = analytic_elasticities()
    anc = deployed_anchor()
    xr = within_run_elasticity()

    pf_realized = anc["realized_propagation_factor"]
    breach_reconciled_frac = (DEPLOYED_PUBLIC_TPS - DEPLOYED_PRIVATE_TPS) / DEPLOYED_PUBLIC_TPS

    # Verdict: which model does the data support?
    # linear if realized PF and the natural-elasticity envelope are << the 5.6 needed for 24%.
    pf_needed = ana["pf_needed_for_24pct"]
    linear_supported = pf_realized < 0.5 * pf_needed and ana["natural_elasticity_hi"] < 0.5 * pf_needed
    breach_model_verdict = "linear ~4.3%" if linear_supported else "amplified ~24%"

    # Compose the ship band under the reconciled (linear) breach, plus the amplified contrast.
    band_linear = compose_band(SHIP_PUBLIC_TPS, breach_reconciled_frac, SIGMA_HW_FRAC)
    band_linear_round = compose_band(SHIP_PUBLIC_TPS_ROUND, breach_reconciled_frac, SIGMA_HW_FRAC)
    band_amplified = compose_band(SHIP_PUBLIC_TPS, DENKEN_MODELED_BREACH_FRAC, SIGMA_HW_FRAC)

    results = {
        "pr": 504,
        "agent": "kanna",
        "analysis_only": True,
        "official_tps": 0,
        "no_serve": True,
        "no_hf_job": True,
        "no_launch": True,
        "no_submission": True,
        "no_served_file_change": True,
        "k_spec": K_SPEC,
        "e_t_deployed": E_T_DEPLOYED,
        "gpu_blocker": GPU_BLOCKER,
        "lane_discipline": ("denken #489 dalpha=4.295% is an INPUT (acceptance leg, NOT "
                            "re-derived); novel contribution is the TPS-propagation leg + "
                            "sigma_hw composition"),
        "analytic": ana,
        "deployed_anchor": anc,
        "within_run_xcheck": xr,
        "bracket_context": {
            "private_flip_ratio_easy": BRACKET_PRIVATE_FLIP_RATIO_EASY,
            "private_flip_ratio_hard": BRACKET_PRIVATE_FLIP_RATIO_HARD,
            "note": ("#497 splits carry identity-flip rates only (not acceptance/TPS); "
                     "private flip ratio <1 => stack bounded off-distribution, qualitatively "
                     "consistent with a mild/linear private regime, NOT a catastrophic one"),
        },
        # --- the reconciliation verdict ---
        "realized_propagation_factor": pf_realized,
        "private_tps_breach_pct_reconciled": 100.0 * breach_reconciled_frac,
        "pf_needed_for_24pct": pf_needed,
        "breach_model_verdict": breach_model_verdict,
        "denken_24pct_is": ("worst-case-only (requires PF~5.6 / deep-block-survival accept "
                            "model far outside the natural elasticity envelope [%.2f, %.2f] "
                            "and the realized PF~%.2f)" % (ana["natural_elasticity_lo"],
                                                           ana["natural_elasticity_hi"], pf_realized)),
        # --- composition: the number the team needs ---
        "surgical357_private_tps_mean": band_linear["private_tps_mean"],
        "surgical357_private_tps_band_68": band_linear["private_tps_band_68"],
        "surgical357_private_tps_band_95": band_linear["private_tps_band_95"],
        "P_private_below_95pct_public": band_linear["p_private_below_95pct_public"],
        "band_linear": band_linear,
        "band_linear_round357": band_linear_round,
        "band_amplified_contrast": band_amplified,
        "sigma_hw_frac": SIGMA_HW_FRAC,
    }

    # one-line verdict
    results["verdict_one_line"] = (
        "deployed anchor (481.53->460.85 = %.2f%% TPS gap) / denken dalpha=4.295%% realize a "
        "propagation factor of %.2f (linear; natural-elasticity envelope [%.2f, %.2f], all "
        "<<5.6) -> surgical-357 private TPS ~= %.1f [95%% %.1f-%.1f, +-sigma_hw 1%%], breach "
        "%.2f%% -> denken #489's 24%% is WORST-CASE-ONLY (needs PF~5.6 / deep-block-survival "
        "accept model), data-refuted as the expected outcome." % (
            anc["deployed_realized_tps_gap_pct"], pf_realized,
            ana["natural_elasticity_lo"], ana["natural_elasticity_hi"],
            band_linear["private_tps_mean"], band_linear["private_tps_band_95"][0],
            band_linear["private_tps_band_95"][1], 100.0 * breach_reconciled_frac))

    results["self_test"] = self_test(results)
    return results


def self_test(r: dict[str, Any]) -> dict[str, Any]:
    ana, anc, band = r["analytic"], r["deployed_anchor"], r["band_linear"]
    checks: dict[str, bool] = {}

    # E[T] from per-position a_k reconstructs the banked deployed E[T].
    checks["et_reconstructs_deployed"] = abs(ana["et_from_per_position"] - E_T_DEPLOYED) < 0.01
    # mean-accept elasticity equals the closed form (E[T]-1)/E[T].
    checks["elasticity_A_closed_form"] = abs(
        ana["elasticity_A_mean_accept"] - (E_T_DEPLOYED - 1.0) / E_T_DEPLOYED) < 1e-9
    # geometric elasticity closed form matches the numerical central-difference.
    checks["elasticity_B_numcheck"] = abs(
        ana["elasticity_B_geometric"] - ana["elasticity_B_geometric_numcheck"]) < 1e-3
    # natural elasticity envelope sits an order of magnitude below the 24% requirement.
    checks["natural_envelope_below_24_requirement"] = ana["natural_elasticity_hi"] < 0.5 * ana["pf_needed_for_24pct"]
    # realized PF on the deployed leaderboard is ~1 (linear, not amplified).
    checks["realized_pf_is_linear"] = 0.7 < anc["realized_propagation_factor"] < 1.3
    # pf needed for 24% reproduces ~5.6.
    checks["pf_needed_for_24_is_5p6"] = abs(ana["pf_needed_for_24pct"] - 5.59) < 0.2
    # ship band ordering + linear/amplified separation matches PR (342 vs 271).
    checks["band_ordered"] = (band["private_tps_band_95"][0] < band["private_tps_mean"]
                              < band["private_tps_band_95"][1])
    checks["linear_mean_near_342"] = abs(band["private_tps_mean"] - 342.0) < 3.0
    checks["amplified_mean_near_271"] = abs(r["band_amplified_contrast"]["private_tps_mean"] - 271.0) < 3.0
    # probability is a valid probability.
    checks["p_below_valid"] = 0.0 <= band["p_private_below_95pct_public"] <= 1.0
    # sigma_hw convention roundtrip (1% of 481.53 = 4.8153).
    checks["sigma_hw_roundtrip"] = abs(SIGMA_HW_FRAC * DEPLOYED_PUBLIC_TPS - 4.8153) < 0.01
    # within-run GPU xcheck (if available): elasticity 95% CI upper bound excludes the 5.6
    # required for the 24% model -> the GPU-measured leg also refutes amplification.
    xr = r["within_run_xcheck"]
    if xr.get("available"):
        ci = xr["steady_gen_tps"].get("elasticity_ci95", [float("nan"), float("nan")])
        checks["within_run_ci_excludes_24_requirement"] = (
            math.isfinite(ci[1]) and ci[1] < ana["pf_needed_for_24pct"])
    # NaN-clean over every numeric leaf.
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
# Pretty-print + W&B                                                           #
# --------------------------------------------------------------------------- #
def _print(r: dict[str, Any]) -> None:
    ana, anc, xr, b = r["analytic"], r["deployed_anchor"], r["within_run_xcheck"], r["band_linear"]
    print("\n[reconcile] ===== PRIVATE-TPS-BREACH RECONCILIATION (PR #504) =====", flush=True)
    print("  STRUCTURE: TPS = E[T]/C_step, C_step alpha-invariant => PF = elasticity(E[T], alpha)", flush=True)
    print("  -- leg 1b: analytic elasticities at deployed op point (E[T]=%.4f, K=%d) --" % (E_T_DEPLOYED, K_SPEC), flush=True)
    print("    (A) mean accept_rate : eps=%.3f -> breach %.2f%%  [+1 bonus dilutes, sub-linear]" % (
        ana["elasticity_A_mean_accept"], ana["breach_A_pct"]), flush=True)
    print("    (B) geometric alpha  : alpha=%.3f eps=%.3f -> breach %.2f%%" % (
        ana["alpha_B_geometric"], ana["elasticity_B_geometric"], ana["breach_B_pct"]), flush=True)
    print("    (C) per-pos compound : eps=%.3f -> breach %.2f%%" % (
        ana["elasticity_C_per_position_compound"], ana["breach_C_pct"]), flush=True)
    print("    natural envelope eps in [%.3f, %.3f]" % (
        ana["natural_elasticity_lo"], ana["natural_elasticity_hi"]), flush=True)
    print("    (D) 24%% needs PF=%.2f (~deep-block alpha^%.1f survival) -- worst-case only" % (
        ana["pf_needed_for_24pct"], ana["pf_needed_for_24pct"]), flush=True)
    print("  -- DEPLOYED ANCHOR (empirical arbiter, real private leaderboard) --", flush=True)
    print("    public %.2f / private %.2f = %.3f%% TPS gap / dalpha 4.295%% => realized PF = %.3f" % (
        anc["deployed_public_tps"], anc["deployed_private_tps"],
        anc["deployed_realized_tps_gap_pct"], anc["realized_propagation_factor"]), flush=True)
    if xr.get("available"):
        s = xr["steady_gen_tps"]
        ci = s.get("elasticity_ci95", [float("nan"), float("nan")])
        print("  -- WITHIN-RUN GPU XCHECK (#478 jb1a0lab, n=%d, deployed stack) --" % xr["n_runs"], flush=True)
        print("    E[T] cv %.3f%%; steady_gen_tps elasticity=%.2f (r=%.2f, 95%% CI [%.2f, %.2f], cv %.2f%%); wall elasticity=%.2f (overhead-pinned)" % (
            xr["et_cv_pct"], s["elasticity"], s["r"], ci[0], ci[1], s["cv_pct"], xr["wall_tps"]["elasticity"]), flush=True)
    print("  -- VERDICT --", flush=True)
    print("    realized_propagation_factor = %.3f -> breach_model_verdict = %s" % (
        r["realized_propagation_factor"], r["breach_model_verdict"]), flush=True)
    print("    private_tps_breach_pct_reconciled = %.2f%%" % r["private_tps_breach_pct_reconciled"], flush=True)
    print("  -- COMPOSITION (surgical-357, ship_public=%.2f, sigma_hw 1%% fractional) --" % b["ship_public_tps"], flush=True)
    print("    private mean %.1f | 68%% [%.1f, %.1f] | 95%% [%.1f, %.1f]" % (
        b["private_tps_mean"], b["private_tps_band_68"][0], b["private_tps_band_68"][1],
        b["private_tps_band_95"][0], b["private_tps_band_95"][1]), flush=True)
    print("    threshold 0.95xpublic=%.1f -> P(private < 0.95xpublic) = %.3f" % (
        b["threshold_095_public"], b["p_private_below_95pct_public"]), flush=True)
    print("    [amplified contrast] private mean %.1f, P=%.3f" % (
        r["band_amplified_contrast"]["private_tps_mean"],
        r["band_amplified_contrast"]["p_private_below_95pct_public"]), flush=True)
    print("  SELF-TEST passes = %s" % r["self_test"]["passes"], flush=True)
    if not r["self_test"]["passes"]:
        for k, v in r["self_test"]["checks"].items():
            if not v:
                print("    FAILED: %s" % k, flush=True)
    print("\n  VERDICT: %s" % r["verdict_one_line"], flush=True)


def _flat_summary(r: dict[str, Any]) -> dict[str, float | int]:
    """Flatten the decision-relevant scalars for the W&B summary (finite only)."""
    ana, anc, b, ba = r["analytic"], r["deployed_anchor"], r["band_linear"], r["band_amplified_contrast"]
    flat = {
        "realized_propagation_factor": r["realized_propagation_factor"],
        "private_tps_breach_pct_reconciled": r["private_tps_breach_pct_reconciled"],
        "pf_needed_for_24pct": r["pf_needed_for_24pct"],
        "surgical357_private_tps_mean": r["surgical357_private_tps_mean"],
        "surgical357_private_tps_band_95_lo": r["surgical357_private_tps_band_95"][0],
        "surgical357_private_tps_band_95_hi": r["surgical357_private_tps_band_95"][1],
        "surgical357_private_tps_band_68_lo": b["private_tps_band_68"][0],
        "surgical357_private_tps_band_68_hi": b["private_tps_band_68"][1],
        "P_private_below_95pct_public": r["P_private_below_95pct_public"],
        "elasticity_A_mean_accept": ana["elasticity_A_mean_accept"],
        "elasticity_B_geometric": ana["elasticity_B_geometric"],
        "elasticity_C_per_position": ana["elasticity_C_per_position_compound"],
        "natural_elasticity_lo": ana["natural_elasticity_lo"],
        "natural_elasticity_hi": ana["natural_elasticity_hi"],
        "breach_A_pct": ana["breach_A_pct"],
        "breach_B_pct": ana["breach_B_pct"],
        "breach_C_pct": ana["breach_C_pct"],
        "deployed_realized_tps_gap_pct": anc["deployed_realized_tps_gap_pct"],
        "amplified_private_tps_mean": ba["private_tps_mean"],
        "amplified_P_below_95pct": ba["p_private_below_95pct_public"],
        "ship_public_tps": SHIP_PUBLIC_TPS,
        "sigma_hw_frac": SIGMA_HW_FRAC,
        "self_test_passes": int(r["self_test"]["passes"]),
        "breach_verdict_linear": int(r["breach_model_verdict"].startswith("linear")),
    }
    xr = r["within_run_xcheck"]
    if xr.get("available"):
        flat["within_run_steady_elasticity"] = xr["steady_gen_tps"]["elasticity"]
        flat["within_run_steady_r"] = xr["steady_gen_tps"]["r"]
        flat["within_run_wall_elasticity"] = xr["wall_tps"]["elasticity"]
        ci = xr["steady_gen_tps"].get("elasticity_ci95", [None, None])
        if ci[0] is not None and math.isfinite(ci[0]):
            flat["within_run_steady_elasticity_ci_lo"] = ci[0]
            flat["within_run_steady_elasticity_ci_hi"] = ci[1]
    return {k: v for k, v in flat.items()
            if isinstance(v, (int, float)) and math.isfinite(v)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="kanna/private-draw-breach-reconcile")
    ap.add_argument("--group", default="private-draw-breach-reconcile")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    r = build_results()
    _print(r)

    out_path = Path(__file__).resolve().parent / "reconcile_breach.json"
    out_path.write_text(json.dumps(r, indent=2))
    print("\n[reconcile] artifacts -> %s" % out_path, flush=True)

    if not r["self_test"]["passes"]:
        print("[reconcile] SELF-TEST FAILED -- not logging to W&B", flush=True)
        return 1
    if args.no_wandb:
        return 0

    run = wandb_logging.init_wandb_run(
        job_type="private-draw-breach-reconcile", agent="kanna",
        name=args.name, group=args.group,
        tags=["private-draw-breach-reconcile", "propagation", "spec-dec", "mtp-k7",
              "surgical357", "sigma-hw-composition", "fa2sw_precache_kenyan", "analysis-only"],
        notes="TPS-propagation reconciliation: 4.3% (linear, deployed-anchored) vs 24% (worst-case).",
        config={
            "k_spec": K_SPEC, "e_t_deployed": E_T_DEPLOYED,
            "delta_alpha_pct": 100.0 * DELTA_ALPHA_FRAC,
            "denken_modeled_breach_pct": 100.0 * DENKEN_MODELED_BREACH_FRAC,
            "deployed_public_tps": DEPLOYED_PUBLIC_TPS,
            "deployed_private_tps": DEPLOYED_PRIVATE_TPS,
            "ship_public_tps": SHIP_PUBLIC_TPS, "sigma_hw_frac": SIGMA_HW_FRAC,
            "analysis_only": True, "official_tps": 0,
            "source_runs": ["2x9fm2zx", "q1ivw9tt", "mssuss3f", "jb1a0lab", "fi34s269"],
        },
    )
    if run is None:
        print("[reconcile] wandb disabled (no API key); skipping", flush=True)
        return 0
    wandb_logging.log_summary(run, _flat_summary(r), step=0)
    wandb_logging.log_json_artifact(
        run, name="private_draw_breach_reconcile", artifact_type="breach-reconcile", data=r)
    wandb_logging.finish_wandb(run)
    print("[reconcile] wandb_run_id=%s" % getattr(run, "id", None), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
