#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Private-safe TPS threshold: is a fast byte-exact (spec-alive) rung gate-safe? (PR #489, denken).

CPU-ONLY ANALYSIS. NO kernel re-measure, NO served-file change, NO HF Job, NO
submission, NO --launch. analysis_only=true, official_tps=0, no_served_file_change=true.

THE FORWARD QUESTION
--------------------
My #486 (`shcdordv`) established that a SPEC-ALIVE byte-exact config carries a FIXED
systematic private delta Delta ~= 4.295% (3.661% drafter-acceptance + 0.633% ctxlen),
essentially independent of TPS -- it RE-INHERITS the acceptance bucket because it keeps
the drafter. The PR asks: is there a CROSSOVER public-TPS above which a spec-alive
byte-exact config flips from gate-risky to gate-safe? That decides whether lawine #488's
surgical-attention-only rung (~457, spec-alive -> same Delta 4.295%) is a fast private-safe
next rung, or whether private-safety still forces the slow 161.70 AR floor.

THE TWO sigma MODELS (and why they give opposite crossovers)
------------------------------------------------------------
The single official private re-run draws TPS ~ N(mu_priv, sigma), mu_priv = public*(1-Delta).
The one-shot breach is P(private < 0.95*public) = Phi((0.95*public - mu_priv)/sigma). The
SHAPE of breach(TPS) depends entirely on whether sigma is ABSOLUTE or FRACTIONAL:

  (a) ABSOLUTE sigma_oneshot = 4.876 TPS (the PR's conservative worst-case). breach(TPS) =
      Phi(-(0.05-Delta)*TPS / sigma_abs) DECREASES with TPS (the fixed noise is a smaller
      fraction of the larger systematic headroom). A crossover EXISTS.
  (b) FRACTIONAL sigma = sigma_frac*TPS, sigma_frac = 4.876/481.53 = 1.013% (PHYSICAL:
      clock/thermal/contention noise is multiplicative on wall-time -> fractional on TPS;
      this is the model #486 headlined). breach(TPS) = Phi(-(0.05-Delta)/sigma_frac) is
      SCALE-INVARIANT -- both the systematic gap (Delta*TPS) and the noise (sigma_frac*TPS)
      scale with TPS, so their ratio, hence the breach, does NOT depend on TPS at all.

THE LOAD-BEARING RESULT
-----------------------
For a SPEC-ALIVE config (Delta = 4.295%):
  * FRACTIONAL (physical): breach = 24.31% at EVERY TPS. NO crossover. Speed buys NOTHING.
  * ABSOLUTE: breach decreases but crosses 5% only at ~1137 TPS and 1% at ~1608 TPS --
    ABOVE the entire realizable range AND above the deployed 481.53 itself (which carries
    24.3% one-shot breach in this frame; it "passed" because its single draw landed at the
    mean, 460.85 > gate 457.45, not because it sat comfortably clear).
So surgical-457 (spec-alive) is NOT above any in-range crossover under EITHER model.

THE CROSSOVER THE PR INTUITS LIVES IN THE Delta=0 WORLD
-------------------------------------------------------
The "deployed 481 = 4.95 sigma safe, global-flag 234 = 2.41 sigma" framing is the BYTE-EXACT
PREMISE (Delta=0): IF the config truly had zero systematic gap, its only risk is sigma_hw,
and under ABSOLUTE sigma the sigma-distance from the gate = 0.05*TPS/sigma_abs grows linearly
with TPS -> crossover at ~160 TPS (5%) / ~227 TPS (1%). THAT is the "between 234 and 481"
intuition. But #486 proved spec-alive configs are NOT Delta=0 -- they carry Delta=4.295%, and
then the crossover blows up to 1137 TPS (abs) or vanishes entirely (frac). Private-safety is
a property of Delta (shed the drafter -> 161.70 AR floor, Delta=0.633% -> safe), NOT of speed.

Reproduce: cd target/ && .venv/bin/python \
  research/validity/private_safe_tps_threshold/private_safe_tps_threshold.py \
  --wandb_group private-safe-tps-threshold --wandb_name denken/private-safe-tps-threshold
"""
from __future__ import annotations

import argparse
import json
import math
import os

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, "..", "..", ".."))

# ---- banked inputs (all read-only; this card measures nothing on the GPU) -----------
GAP_JSON = os.path.join(
    _root, "research/validity/public_private_gap_decomposition/public_private_gap_decomposition_results.json")
SIGMA_RECON = os.path.join(_root, "research/empirical_sigma_hw/fresh_n10/reconciliation.json")
SIGMA_JSON = os.path.join(_root, "research/empirical_sigma_hw/fresh_n10/sigma_hw.json")
LITERAL_JSON = os.path.join(
    _root, "research/validity/literal_1p0_config_reachable/literal_1p0_config_reachable_results.json")
CROSSCHECK_JSON = os.path.join(
    _root, "research/validity/strict_frontier_realize_crosscheck/strict_frontier_realize_crosscheck_report.json")

# Validity gates (BASELINE.md / program.md).
DELTA_GATE = 0.05            # public<->private TPS reproduction gate (private >= 95% of public)
PPL_GATE = 2.42
PPL_ANCHOR = 2.3772          # deployed public PPL; byte-exact greedy configs reproduce the base PPL

# Deployed ground-truth public/private pair (organizer cmpatino-verifier, BASELINE.md).
DEPLOYED_PUBLIC_TPS = 481.53
DEPLOYED_PRIVATE_TPS = 460.85

# Sweep + candidate config (the surgical-457 rung is CONDITIONAL on lawine #488 realizing).
SWEEP_LO, SWEEP_HI, SWEEP_STEP = 150.0, 500.0, 5.0
SURGICAL457_TPS = 457.0      # PR-stated ~457 byte-exact rung (lawine #488, in flight)
BREACH_BARS = [0.05, 0.01]   # gate-safe bars: breach < 5% and breach < 1%


def _phi(z: float) -> float:
    """Standard-normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _phinv(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation; |err| < 1.2e-9)."""
    if not (0.0 < p < 1.0):
        return float("nan")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=os.path.join(_here, "private_safe_tps_threshold_results.json"))
    ap.add_argument("--surgical457_tps", type=float, default=SURGICAL457_TPS,
                    help="PR-stated ~457 byte-exact rung TPS (CONDITIONAL on lawine #488).")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="private-safe-tps-threshold")
    ap.add_argument("--wandb_name", default="denken/private-safe-tps-threshold")
    ap.add_argument("--job_type", default="analysis")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    gap = json.load(open(GAP_JSON))
    recon = json.load(open(SIGMA_RECON))["reconciliation"]
    sigma_an = json.load(open(SIGMA_JSON))["analysis"]
    literal = json.load(open(LITERAL_JSON))
    cross = json.load(open(CROSSCHECK_JSON))

    # ---- deployed public->private gap decomposition (ubel #379) ----
    deployed_gap = float(gap["decomposition_central"]["total_gap_frac"])               # 0.042946
    accept_bucket = float(gap["decomposition_central"]["bucket_acceptance_abs_pct"]) / 100.0  # 0.03661 (DRAFTER)
    ctxlen_bucket = float(gap["decomposition_central"]["bucket_ctxlen_abs_pct"]) / 100.0       # 0.00633 (global KV)
    deployed_gap_recon = (DEPLOYED_PUBLIC_TPS - DEPLOYED_PRIVATE_TPS) / DEPLOYED_PUBLIC_TPS

    # ---- sigma_hw anchors (lawine #467 reconciliation) ----
    sigma_within = float(recon["sigma_within_measured_tps_at_481"])                    # 0.349 (same-session)
    sigma_between = float(recon["sigma_between_cited_tps"])                             # 4.864 (between-session)
    sigma_oneshot = float(recon["sigma_oneshot_reconstructed_tps"])                    # 4.8765 (single official draw)
    sigma_convention = float(sigma_an["convention_sigma_hw"])                          # 4.8153 (1% convention)
    # FRACTIONAL model: the relative sigma the noise carries at ANY TPS (calibrated at the deployed anchor).
    sigma_oneshot_frac = sigma_oneshot / DEPLOYED_PUBLIC_TPS                            # 1.013%

    # ---- candidate public TPS (the REAL fire configs + the conditional surgical rung) ----
    floorlock_tps = float(literal["m1_floor_tps"])                                     # 161.70 (M=1 AR floor)
    globalflag_tps = float(cross["realized_strict_tps_bi_pin"])                        # 234.4667 (blanket BI)
    globalflag_eaccept = float(cross["e_accept_under_bi"])                             # 3.8695 (spec ALIVE)
    floorlock_ppl = float(literal["ppl"])                                              # 2.3772 (base greedy)
    globalflag_ppl = float(cross["ppl"])                                               # 2.3770 (byte-exact spec)
    surgical_tps = float(args.surgical457_tps)                                         # ~457 (CONDITIONAL on #488)

    # The two systematic-delta frames:
    #   spec-alive (the HONEST frame for any drafter-keeping config): re-inherits the full deployed gap.
    #   byte-exact PREMISE (Delta=0): the PR's mental model -- "byte-exact output => ~0 systematic gap".
    DELTA_SPEC_ALIVE = deployed_gap          # 0.042946 (acceptance 3.661% + ctxlen 0.633%)
    DELTA_BYTEEXACT = 0.0                     # the (false-for-spec) premise
    DELTA_FLOORLOCK = ctxlen_bucket           # 0.00633 (non-spec: acceptance bucket GONE, ctxlen only)

    # ---- core breach functions: P(private < 0.95*public) under N(mu_priv, sigma) ----
    def predict(tps, delta):
        return tps * (1.0 - delta)

    def threshold(tps):
        return (1.0 - DELTA_GATE) * tps

    def breach_abs(tps, delta, sigma_abs=sigma_oneshot):
        """Absolute-sigma one-shot breach (fixed sigma in TPS)."""
        return _phi((threshold(tps) - predict(tps, delta)) / sigma_abs)

    def breach_frac(tps, delta, sigma_frac=sigma_oneshot_frac):
        """Fractional-sigma one-shot breach (sigma scales with TPS) -- the PHYSICAL model."""
        return _phi((threshold(tps) - predict(tps, delta)) / (sigma_frac * tps))

    # ---- (1) CROSSOVER SWEEP over [150,500], spec-alive Delta=4.295% (+ byte-exact contrast) ----
    # Closed forms:
    #   abs : breach = Phi(-(0.05-Delta)*TPS/sigma_abs) -> = bar  =>  TPS = -Phi^-1(bar)*sigma_abs/(0.05-Delta)
    #   frac: breach = Phi(-(0.05-Delta)/sigma_frac)    -> SCALE-INVARIANT (no TPS dependence) -> crossover
    #         exists only if that constant < bar, else None.
    head_spec = DELTA_GATE - DELTA_SPEC_ALIVE     # 0.007054 systematic headroom fraction (>0 since Delta<gate)
    head_byte = DELTA_GATE - DELTA_BYTEEXACT      # 0.05

    def crossover_abs(head, bar):
        """Smallest public TPS at which absolute-sigma breach drops below `bar` (breach falls with TPS)."""
        if head <= 0:
            return float("inf")                   # mu_priv below gate -> breach never beats the bar
        return -_phinv(bar) * sigma_oneshot / head

    def crossover_frac(head, bar):
        """Fractional-sigma breach is scale-invariant; crossover exists only if the constant < bar."""
        const = _phi(-head / sigma_oneshot_frac)
        return SWEEP_LO if const < bar else None  # if already safe, every TPS in range qualifies

    crossover = {}
    for frame, head in [("spec_alive", head_spec), ("byteexact_premise", head_byte)]:
        crossover[frame] = {}
        const_frac = _phi(-head / sigma_oneshot_frac)
        for bar in BREACH_BARS:
            tag = f"breach_lt_{int(round(bar*100))}pct"
            xa = crossover_abs(head, bar)
            xf = crossover_frac(head, bar)
            crossover[frame][tag] = {
                "abs_crossover_tps": xa,
                "abs_in_sweep_range": bool(SWEEP_LO <= xa <= SWEEP_HI),
                "abs_below_deployed_481": bool(xa <= DEPLOYED_PUBLIC_TPS),
                "frac_crossover_tps": xf,                # None => no crossover (scale-invariant breach > bar)
                "frac_scale_invariant_breach": const_frac,
                "frac_safe_at_all_tps": bool(xf is not None),
            }

    # sweep grid (for the W&B table / record) -- breach at each TPS under both sigma and both Delta frames.
    sweep_rows = []
    t = SWEEP_LO
    while t <= SWEEP_HI + 1e-9:
        sweep_rows.append({
            "public_tps": round(t, 3),
            "spec_alive_breach_abs": breach_abs(t, DELTA_SPEC_ALIVE),
            "spec_alive_breach_frac": breach_frac(t, DELTA_SPEC_ALIVE),
            "byteexact_breach_abs": breach_abs(t, DELTA_BYTEEXACT),
            "byteexact_breach_frac": breach_frac(t, DELTA_BYTEEXACT),
        })
        t += SWEEP_STEP

    # ---- (2) CANDIDATE TABLE: four real/candidate points, breach under abs+frac ----
    def make_candidate(label, tps, delta, is_spec, conditional=False, **extra):
        mu = predict(tps, delta)
        return {
            "label": label,
            "public_tps": tps,
            "is_speculative": is_spec,
            "conditional_on_488": conditional,
            "systematic_delta_pct": 100.0 * delta,
            "predicted_private_tps": mu,
            "threshold_tps": threshold(tps),
            "headroom_pp": 100.0 * (DELTA_GATE - delta),
            "mean_clears_gate": bool(mu >= threshold(tps)),
            "sigma_hw_frac_of_tps_pct": 100.0 * sigma_oneshot / tps,
            "breach_prob_abs": breach_abs(tps, delta),
            "breach_prob_frac": breach_frac(tps, delta),
            **extra,
        }

    candidates = {
        "floor_lock_161p70": make_candidate(
            "floor-lock 161.70 (M=1 AR, NON-spec, literal-1.0; acceptance bucket GONE)",
            floorlock_tps, DELTA_FLOORLOCK, is_spec=False, ppl=floorlock_ppl,
            note="systematic = ctxlen-only (no drafter)"),
        "global_flag_234p47": make_candidate(
            "global-flag 234.47 (blanket BI, SPEC alive E_accept~3.87; byte-exact OUTPUT)",
            globalflag_tps, DELTA_SPEC_ALIVE, is_spec=True, e_accept=globalflag_eaccept, ppl=globalflag_ppl,
            note="re-inherits full 4.295% (acceptance + ctxlen)"),
        "surgical_457": make_candidate(
            "surgical-457 (~457, SPEC alive, byte-exact; CONDITIONAL on lawine #488 realizing)",
            surgical_tps, DELTA_SPEC_ALIVE, is_spec=True, conditional=True,
            note="keeps the drafter -> SAME 4.295% as deployed"),
        "deployed_481p53": make_candidate(
            "deployed 481.53 (organizer-VALID spec-alive anchor; private 460.85 measured)",
            DEPLOYED_PUBLIC_TPS, DELTA_SPEC_ALIVE, is_spec=True,
            ppl=PPL_ANCHOR, measured_private_tps=DEPLOYED_PRIVATE_TPS,
            note="passed its single draw (mean 460.85 > gate 457.45, +0.70 sigma_oneshot)"),
    }
    # deployed reconciliation: the model's predicted private mean must reproduce the measured 460.85.
    deployed_pred = candidates["deployed_481p53"]["predicted_private_tps"]

    # ---- (3) KEY OUTPUT: surgical457_private_safe + realized-TPS floor for a spec-alive config ----
    surg = candidates["surgical_457"]
    surgical457_breach_abs = surg["breach_prob_abs"]
    surgical457_breach_frac = surg["breach_prob_frac"]
    surgical457_private_safe = bool(surgical457_breach_abs < 0.05 and surgical457_breach_frac < 0.05)
    # realized-TPS floor for a SPEC-ALIVE (Delta=4.295%) config to clear each bar:
    spec_floor = {}
    for bar in BREACH_BARS:
        tag = f"breach_lt_{int(round(bar*100))}pct"
        xa = crossover_abs(head_spec, bar)
        xf = crossover_frac(head_spec, bar)
        spec_floor[tag] = {
            "abs_sigma_tps_floor": xa,
            "abs_reachable_in_range": bool(SWEEP_LO <= xa <= SWEEP_HI),
            "abs_reachable_below_deployed": bool(xa <= DEPLOYED_PUBLIC_TPS),
            "frac_sigma_tps_floor": xf,                # None -> unreachable at any TPS (physical model)
            "frac_reachable": bool(xf is not None),
        }

    # ---- (4) FLOOR-LOCK 161.70 reconfirm under this unified framework ----
    fl = candidates["floor_lock_161p70"]
    floorlock_reconfirm_safe = bool(fl["systematic_delta_pct"] < 100.0 * DELTA_GATE
                                    and fl["breach_prob_frac"] < 0.01)
    floorlock_abs_sigma_flag = bool(fl["breach_prob_abs"] >= 0.05)   # the conservative absolute-tail caveat

    # ---- (5) COEXISTENCE VERDICT ----
    # Does ANY realized spec-alive config below 481 clear the gate with comfortable margin?
    # Comfortable == breach < 5% under BOTH sigma models.
    spec_alive_in_range_points = [globalflag_tps, surgical_tps]
    any_spec_alive_safe_in_range = any(
        breach_abs(t, DELTA_SPEC_ALIVE) < 0.05 and breach_frac(t, DELTA_SPEC_ALIVE) < 0.05
        for t in spec_alive_in_range_points)
    # frac model: spec-alive breach is the SAME 24.3% at every TPS -> speed is irrelevant.
    spec_alive_frac_breach = breach_frac(DEPLOYED_PUBLIC_TPS, DELTA_SPEC_ALIVE)   # scale-invariant constant
    fast_byteexact_privatesafe_coexists = bool(any_spec_alive_safe_in_range)

    # ---- self-tests ----
    st = {}
    st["deployed_gap_reconstructs_4p295"] = bool(abs(deployed_gap - deployed_gap_recon) < 1e-4)
    st["accept_plus_ctxlen_equals_deployed_gap"] = bool(abs((accept_bucket + ctxlen_bucket) - deployed_gap) < 1e-4)
    st["sigma_oneshot_reconstructs_between_within"] = bool(
        abs(sigma_oneshot - math.sqrt(sigma_between**2 + sigma_within**2)) < 0.05)
    st["phinv_inverts_phi_at_5pct"] = bool(abs(_phi(_phinv(0.05)) - 0.05) < 1e-6)
    st["phinv_inverts_phi_at_1pct"] = bool(abs(_phi(_phinv(0.01)) - 0.01) < 1e-6)
    # spec-alive FRACTIONAL breach is scale-invariant (same at 234, 457, 481 within numerical noise).
    st["spec_alive_frac_breach_scale_invariant"] = bool(
        abs(breach_frac(globalflag_tps, DELTA_SPEC_ALIVE) - breach_frac(surgical_tps, DELTA_SPEC_ALIVE)) < 1e-9
        and abs(breach_frac(surgical_tps, DELTA_SPEC_ALIVE)
                - breach_frac(DEPLOYED_PUBLIC_TPS, DELTA_SPEC_ALIVE)) < 1e-9)
    st["spec_alive_frac_breach_is_material"] = bool(spec_alive_frac_breach > 0.20)   # ~24.3%
    # spec-alive ABSOLUTE breach is monotone DECREASING in TPS.
    st["spec_alive_abs_breach_monotone_decreasing"] = bool(
        breach_abs(floorlock_tps, DELTA_SPEC_ALIVE) > breach_abs(globalflag_tps, DELTA_SPEC_ALIVE)
        > breach_abs(surgical_tps, DELTA_SPEC_ALIVE) > breach_abs(DEPLOYED_PUBLIC_TPS, DELTA_SPEC_ALIVE))
    # spec-alive crossover (5%) is OUT of range AND above deployed-481 (abs); frac has NO crossover.
    xa5 = crossover_abs(head_spec, 0.05)
    xa1 = crossover_abs(head_spec, 0.01)
    st["spec_alive_abs_crossover5_out_of_range"] = bool(xa5 > SWEEP_HI)
    st["spec_alive_abs_crossover5_above_deployed"] = bool(xa5 > DEPLOYED_PUBLIC_TPS)
    st["spec_alive_abs_crossover_monotone_in_bar"] = bool(xa1 > xa5)   # tighter bar -> higher TPS floor
    st["spec_alive_frac_no_crossover"] = bool(crossover_frac(head_spec, 0.05) is None)
    # abs and frac coincide exactly at the deployed anchor (sigma_frac calibrated there).
    st["abs_frac_coincide_at_deployed"] = bool(
        abs(breach_abs(DEPLOYED_PUBLIC_TPS, DELTA_SPEC_ALIVE)
            - breach_frac(DEPLOYED_PUBLIC_TPS, DELTA_SPEC_ALIVE)) < 1e-6)
    # surgical-457: NOT safe under either model.
    st["surgical457_breach_above_gate_both"] = bool(surgical457_breach_abs > 0.05 and surgical457_breach_frac > 0.05)
    st["surgical457_private_safe_is_false"] = bool(not surgical457_private_safe)
    st["surgical457_inherits_spec_alive_delta"] = bool(abs(surg["systematic_delta_pct"] - 100.0 * deployed_gap) < 1e-6)
    # byte-exact PREMISE reproduces the PR's crossover + sigma-distance intuition.
    xb5 = crossover_abs(head_byte, 0.05)
    xb1 = crossover_abs(head_byte, 0.01)
    st["byteexact_premise_crossover5_in_sweep_range"] = bool(SWEEP_LO <= xb5 <= SWEEP_HI)        # ~160 in [150,500]
    st["byteexact_premise_crossover1_in_sweep_range"] = bool(SWEEP_LO <= xb1 <= SWEEP_HI)        # ~227 in [150,500]
    st["byteexact_premise_crossovers_below_deployed"] = bool(xb5 < DEPLOYED_PUBLIC_TPS and xb1 < DEPLOYED_PUBLIC_TPS)
    st["byteexact_premise_234_clears_1pct"] = bool(breach_abs(globalflag_tps, DELTA_BYTEEXACT) < 0.01)  # 234 above ~227 floor
    st["byteexact_premise_frac_safe_all_tps"] = bool(crossover_frac(head_byte, 0.01) is not None)
    st["byteexact_premise_sigma_dist_481_near_4p95"] = bool(
        abs((DEPLOYED_PUBLIC_TPS - threshold(DEPLOYED_PUBLIC_TPS)) / sigma_oneshot - 4.94) < 0.05)
    st["byteexact_premise_sigma_dist_234_near_2p41"] = bool(
        abs((globalflag_tps - threshold(globalflag_tps)) / sigma_oneshot - 2.40) < 0.05)
    # floor-lock reconfirm.
    st["floorlock_nonspec_ctxlen_only"] = bool(abs(DELTA_FLOORLOCK - ctxlen_bucket) < 1e-9)
    st["floorlock_frac_safe"] = bool(fl["breach_prob_frac"] < 0.01)
    st["floorlock_abs_tail_flag"] = bool(0.05 <= fl["breach_prob_abs"] < 0.10)
    st["floorlock_delta_below_globalflag"] = bool(fl["systematic_delta_pct"] < candidates["global_flag_234p47"]["systematic_delta_pct"])
    st["floorlock_reconfirm_safe_true"] = bool(floorlock_reconfirm_safe)
    # deployed reconciliation + "deployed can be safe".
    st["deployed_pred_reproduces_460p85"] = bool(abs(deployed_pred - DEPLOYED_PRIVATE_TPS) < 0.05)
    st["deployed_mean_clears_gate"] = bool(candidates["deployed_481p53"]["mean_clears_gate"])
    st["deployed_specalive_breach_is_24pct"] = bool(0.20 < candidates["deployed_481p53"]["breach_prob_frac"] < 0.30)
    # coexistence.
    st["no_in_range_spec_alive_safe"] = bool(not any_spec_alive_safe_in_range)
    st["coexistence_verdict_false"] = bool(not fast_byteexact_privatesafe_coexists)
    # ppl gates.
    st["ppl_clears_both_real_candidates"] = bool(floorlock_ppl <= PPL_GATE and globalflag_ppl <= PPL_GATE)
    st["threshold_is_95pct_of_public"] = bool(abs(threshold(surgical_tps) - 0.95 * surgical_tps) < 1e-9)
    finite = [surgical457_breach_abs, surgical457_breach_frac, spec_alive_frac_breach, xa5, xa1, xb5, xb1,
              deployed_pred, fl["breach_prob_abs"], fl["breach_prob_frac"], sigma_oneshot, sigma_oneshot_frac]
    st["nan_clean"] = all(math.isfinite(x) for x in finite)
    self_test_passes = all(st.values())

    verdict = {
        # ---- (1) crossover ----
        "spec_alive_crossover_abs_breach_lt_5pct_tps": crossover["spec_alive"]["breach_lt_5pct"]["abs_crossover_tps"],
        "spec_alive_crossover_abs_breach_lt_1pct_tps": crossover["spec_alive"]["breach_lt_1pct"]["abs_crossover_tps"],
        "spec_alive_crossover_abs_5pct_in_range": crossover["spec_alive"]["breach_lt_5pct"]["abs_in_sweep_range"],
        "spec_alive_crossover_abs_5pct_above_deployed": crossover["spec_alive"]["breach_lt_5pct"]["abs_below_deployed_481"] is False,
        "spec_alive_crossover_frac_breach_lt_5pct_tps": crossover["spec_alive"]["breach_lt_5pct"]["frac_crossover_tps"],
        "spec_alive_crossover_frac_breach_lt_1pct_tps": crossover["spec_alive"]["breach_lt_1pct"]["frac_crossover_tps"],
        "spec_alive_frac_scale_invariant_breach_pct": 100.0 * spec_alive_frac_breach,
        "byteexact_premise_crossover_abs_breach_lt_5pct_tps": crossover["byteexact_premise"]["breach_lt_5pct"]["abs_crossover_tps"],
        "byteexact_premise_crossover_abs_breach_lt_1pct_tps": crossover["byteexact_premise"]["breach_lt_1pct"]["abs_crossover_tps"],
        # ---- (2/3) candidate + surgical-457 ----
        "floorlock_breach_abs_pct": 100.0 * fl["breach_prob_abs"],
        "floorlock_breach_frac_pct": 100.0 * fl["breach_prob_frac"],
        "globalflag_breach_abs_pct": 100.0 * candidates["global_flag_234p47"]["breach_prob_abs"],
        "globalflag_breach_frac_pct": 100.0 * candidates["global_flag_234p47"]["breach_prob_frac"],
        "surgical457_breach_abs_pct": 100.0 * surgical457_breach_abs,
        "surgical457_breach_frac_pct": 100.0 * surgical457_breach_frac,
        "surgical457_private_safe": surgical457_private_safe,
        "surgical457_spec_alive_tps_floor_breach_lt_5pct_abs": spec_floor["breach_lt_5pct"]["abs_sigma_tps_floor"],
        "surgical457_spec_alive_tps_floor_breach_lt_1pct_abs": spec_floor["breach_lt_1pct"]["abs_sigma_tps_floor"],
        "surgical457_spec_alive_tps_floor_frac_reachable": spec_floor["breach_lt_5pct"]["frac_reachable"],
        "deployed_breach_specalive_pct": 100.0 * candidates["deployed_481p53"]["breach_prob_frac"],
        "deployed_predicted_private_tps": deployed_pred,
        "deployed_mean_clears_gate": candidates["deployed_481p53"]["mean_clears_gate"],
        # ---- (4) floor-lock reconfirm ----
        "floorlock_reconfirm_safe": floorlock_reconfirm_safe,
        "floorlock_abs_sigma_flag": floorlock_abs_sigma_flag,
        # ---- (5) coexistence ----
        "fast_byteexact_privatesafe_coexists": fast_byteexact_privatesafe_coexists,
        # ---- shared / config ----
        "delta_spec_alive_pct": 100.0 * DELTA_SPEC_ALIVE,
        "delta_floorlock_pct": 100.0 * DELTA_FLOORLOCK,
        "deployed_gap_pct": 100.0 * deployed_gap,
        "deployed_acceptance_bucket_pct": 100.0 * accept_bucket,
        "deployed_ctxlen_bucket_pct": 100.0 * ctxlen_bucket,
        "sigma_oneshot_tps": sigma_oneshot,
        "sigma_oneshot_frac_pct": 100.0 * sigma_oneshot_frac,
        "delta_gate": DELTA_GATE, "ppl_gate": PPL_GATE,
        "analysis_only": True, "official_tps": 0, "no_served_file_change": True,
        "no_kernel_rebuild": True, "no_hf_job": True, "no_launch": True, "no_submission": True,
        "self_test_passes": self_test_passes,
    }

    reconcile = (
        f"Private-safe TPS threshold for a SPEC-ALIVE byte-exact rung. The crossover the PR intuits "
        f"(speed flips a config gate-safe) is REAL only under the BYTE-EXACT PREMISE Delta=0: abs-sigma "
        f"crossover {crossover['byteexact_premise']['breach_lt_5pct']['abs_crossover_tps']:.0f} TPS (<5%) / "
        f"{crossover['byteexact_premise']['breach_lt_1pct']['abs_crossover_tps']:.0f} TPS (<1%), reproducing the "
        f"PR's 2.41 sigma @234 / 4.95 sigma @481. But #486 proved a drafter-keeping config is NOT Delta=0 -- it "
        f"re-inherits Delta={100*deployed_gap:.3f}% ({100*accept_bucket:.3f}% acceptance + {100*ctxlen_bucket:.3f}% "
        f"ctxlen). For SPEC-ALIVE Delta={100*deployed_gap:.3f}%: FRACTIONAL (physical) breach is SCALE-INVARIANT "
        f"{100*spec_alive_frac_breach:.1f}% at EVERY TPS (no crossover; speed buys nothing), and ABSOLUTE-sigma "
        f"breach crosses 5% only at {crossover['spec_alive']['breach_lt_5pct']['abs_crossover_tps']:.0f} TPS / 1% at "
        f"{crossover['spec_alive']['breach_lt_1pct']['abs_crossover_tps']:.0f} TPS -- both ABOVE the realizable range "
        f"AND above the deployed 481.53 itself. surgical-457 (spec-alive, CONDITIONAL on #488): breach "
        f"{100*surgical457_breach_abs:.1f}% abs / {100*surgical457_breach_frac:.1f}% frac -> surgical457_private_safe="
        f"{surgical457_private_safe}. FLOOR-LOCK 161.70 (Delta={100*DELTA_FLOORLOCK:.3f}%, no drafter): breach "
        f"{100*fl['breach_prob_frac']:.4f}% frac -> reconfirm_safe={floorlock_reconfirm_safe} (only flag: "
        f"{100*fl['breach_prob_abs']:.2f}% absolute-sigma tail). VERDICT fast_byteexact_privatesafe_coexists="
        f"{fast_byteexact_privatesafe_coexists}: NO realized spec-alive config below (or at) 481 clears the gate with "
        f"comfortable margin. Private-safety is a property of Delta (shed the drafter -> 161.70 AR floor), NOT of TPS. "
        f"The deployed-481 'safety' is the Delta=0 counterfactual; spec-honest, deployed carried the SAME "
        f"{100*spec_alive_frac_breach:.1f}% one-shot breach and won its single draw (mean {deployed_pred:.2f} > gate "
        f"{threshold(DEPLOYED_PUBLIC_TPS):.2f}). The only fast private-safe ship is the deployed-481 class; the only "
        f"strict private-safe ship is floor-lock 161.70.")
    verdict["reconcile_line"] = reconcile

    payload = {
        "pr": 489,
        "issue": 488,
        "author": "denken",
        "leg": "private-safe TPS threshold: crossover for a spec-alive byte-exact rung; is surgical-457 gate-safe?",
        "config": {
            "delta_gate": DELTA_GATE, "ppl_gate": PPL_GATE,
            "deployed_public_tps": DEPLOYED_PUBLIC_TPS, "deployed_private_tps": DEPLOYED_PRIVATE_TPS,
            "delta_spec_alive": DELTA_SPEC_ALIVE, "delta_floorlock": DELTA_FLOORLOCK,
            "floorlock_public_tps": floorlock_tps, "globalflag_public_tps": globalflag_tps,
            "surgical457_public_tps": surgical_tps,
            "sigma_oneshot_tps": sigma_oneshot, "sigma_oneshot_frac": sigma_oneshot_frac,
            "sigma_within_tps": sigma_within, "sigma_between_tps": sigma_between,
            "sigma_convention_tps": sigma_convention,
            "sweep_lo": SWEEP_LO, "sweep_hi": SWEEP_HI, "sweep_step": SWEEP_STEP, "breach_bars": BREACH_BARS,
            "imports": {
                "deployed_gap_decomp": os.path.relpath(GAP_JSON, _root),
                "sigma_reconciliation": os.path.relpath(SIGMA_RECON, _root),
                "sigma_fresh_n10": os.path.relpath(SIGMA_JSON, _root),
                "floorlock_m1_floor": os.path.relpath(LITERAL_JSON, _root),
                "globalflag_bi_pin": os.path.relpath(CROSSCHECK_JSON, _root),
            },
            "note": "Extends denken #486 (private_validity_real_candidates). CPU analysis only; no kernel "
                    "re-measure, no served change, no HF Job, no launch, no submission. surgical-457 is "
                    "CONDITIONAL on lawine #488 realizing ~457 byte-exact; its TPS is a PR-stated placeholder.",
        },
        "crossover": crossover,
        "candidates": candidates,
        "surgical457_spec_alive_tps_floor": spec_floor,
        "sweep": sweep_rows,
        "verdict": verdict,
        "self_test_conditions": st,
        "public_evidence_used": (
            "denken #486 (shcdordv) private_validity_real_candidates: a spec-alive byte-exact config carries the "
            "fixed deployed Delta 4.295% (3.661% acceptance + 0.633% ctxlen) because it re-inherits the drafter "
            "acceptance bucket; floor-lock 161.70 (no drafter) keeps only the 0.633% ctxlen term. ubel #379 "
            "(5kpb73tb) public->private gap decomposition is the 4.295% anchor. lawine #467 sigma_hw "
            "reconciliation: sigma_within 0.349, sigma_between 4.864, sigma_oneshot 4.876. denken #476 "
            "literal_1p0_config_reachable: literal-1.0 reachable only at the M=1 AR floor 161.70. ubel #470 "
            "(ugqnytji) strict-frontier BI-pin: blanket VLLM_BATCH_INVARIANT=1 realizes 234.47, spec ALIVE "
            "E_accept~3.87, PPL 2.3770<=2.42. lawine #488 (in flight): surgical-attention-only rung targeting "
            "~457 byte-exact; if realized it is spec-alive -> same 4.295% Delta. Deployed pair 481.53->460.85 "
            "(Delta 4.3%) organizer cmpatino-verifier (BASELINE.md, PR #52, 2x9fm2zx)."),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(payload, open(args.output, "w"), indent=2,
              default=lambda o: float(o) if isinstance(o, (int, float)) else str(o))

    print(f"[pst] spec-alive Delta={100*deployed_gap:.3f}%: FRAC breach SCALE-INVARIANT "
          f"{100*spec_alive_frac_breach:.1f}% (no crossover) | ABS crossover "
          f"breach<5% @ {crossover['spec_alive']['breach_lt_5pct']['abs_crossover_tps']:.0f} TPS, "
          f"breach<1% @ {crossover['spec_alive']['breach_lt_1pct']['abs_crossover_tps']:.0f} TPS "
          f"(both > 500 and > deployed 481)", flush=True)
    print(f"[pst] byteexact-PREMISE Delta=0: ABS crossover breach<5% @ "
          f"{crossover['byteexact_premise']['breach_lt_5pct']['abs_crossover_tps']:.0f} TPS / breach<1% @ "
          f"{crossover['byteexact_premise']['breach_lt_1pct']['abs_crossover_tps']:.0f} TPS "
          f"(this is where the PR's '2.41 sigma @234 / 4.95 sigma @481' crossover lives)", flush=True)
    print(f"[pst] CANDIDATES (breach abs/frac): floor-lock 161.70 "
          f"{100*fl['breach_prob_abs']:.2f}%/{100*fl['breach_prob_frac']:.4f}% | global-flag 234.47 "
          f"{100*candidates['global_flag_234p47']['breach_prob_abs']:.1f}%/"
          f"{100*candidates['global_flag_234p47']['breach_prob_frac']:.1f}% | surgical-457 "
          f"{100*surgical457_breach_abs:.1f}%/{100*surgical457_breach_frac:.1f}% | deployed 481.53 "
          f"{100*candidates['deployed_481p53']['breach_prob_abs']:.1f}%/"
          f"{100*candidates['deployed_481p53']['breach_prob_frac']:.1f}%", flush=True)
    print(f"[pst] surgical457_private_safe={surgical457_private_safe} | floorlock_reconfirm_safe="
          f"{floorlock_reconfirm_safe} (abs-tail flag={floorlock_abs_sigma_flag}) | "
          f"fast_byteexact_privatesafe_coexists={fast_byteexact_privatesafe_coexists}", flush=True)
    print(f"[pst] self_test={self_test_passes}", flush=True)
    print(f"[pst] {reconcile}", flush=True)

    if not args.no_wandb:
        _log_wandb(args, payload)
    return 0 if self_test_passes else 1


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type=args.job_type, config=payload.get("config", {}))
    vd = payload["verdict"]
    run.summary.update({k: v for k, v in vd.items() if isinstance(v, (int, float, bool, str))})

    # candidate table
    ct = wandb.Table(columns=["candidate", "public_tps", "is_speculative", "systematic_delta_pct",
                              "predicted_private_tps", "headroom_pp", "mean_clears_gate",
                              "sigma_hw_frac_pct", "breach_abs_pct", "breach_frac_pct"])
    for key in ["floor_lock_161p70", "global_flag_234p47", "surgical_457", "deployed_481p53"]:
        c = payload["candidates"][key]
        ct.add_data(key, c["public_tps"], c["is_speculative"], c["systematic_delta_pct"],
                    c["predicted_private_tps"], c["headroom_pp"], c["mean_clears_gate"],
                    c["sigma_hw_frac_of_tps_pct"], 100.0 * c["breach_prob_abs"], 100.0 * c["breach_prob_frac"])
    run.log({"candidate_breach": ct})

    # sweep table (breach vs TPS under both sigma + both Delta frames)
    sw = wandb.Table(columns=["public_tps", "spec_alive_breach_abs_pct", "spec_alive_breach_frac_pct",
                              "byteexact_breach_abs_pct", "byteexact_breach_frac_pct"])
    for r in payload["sweep"]:
        sw.add_data(r["public_tps"], 100.0 * r["spec_alive_breach_abs"], 100.0 * r["spec_alive_breach_frac"],
                    100.0 * r["byteexact_breach_abs"], 100.0 * r["byteexact_breach_frac"])
    run.log({"breach_sweep": sw})

    run.finish()
    print(f"[pst] logged W&B run {args.wandb_entity}/{args.wandb_project} "
          f"name={args.wandb_name} group={args.wandb_group}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
