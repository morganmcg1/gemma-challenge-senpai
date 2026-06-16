#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Private re-run validity: will the strict #474 submission clear the 5% delta gate? (PR #480, denken).

CPU-ONLY ANALYSIS. NO kernel re-measure, NO served-file change, NO HF Job, NO
submission, NO --launch. analysis_only=true, official_tps=0, no_served_file_change=true.

THE QUESTION
------------
When #474 fires, the strict submission becomes a leaderboard entry and faces the
organizer's PRIVATE re-run validity gate: Delta(public<->private TPS) <= 5% AND
PPL <= 2.42 on the held-out private prompts. We own a measured public/private pair
for the DEPLOYED config -- 481.53 public -> 460.85 private (Delta 4.295%, organizer
`cmpatino-verifier`). The risk this card prices: the strict GO config pins the 7
sliding-window local reductions to FULL attention for byte-identity, so it carries
a STEEPER per-L attention tax than the deployed (windowed) path. On the longer
private KV trajectory that steeper tax could push strict's public->private Delta
ABOVE the deployed 4.295% and toward the 5% gate -- a *systematic* tail that the
session-noise model (sigma_hw) does NOT capture.

THE MODEL (the load-bearing cancellation)
-----------------------------------------
Write the per-verify-step time as T(L) = B + A_dep(L) + added(L):
  B        = body GEMM + lm_head + framework/sampler (L-independent),
  A_dep(L) = deployed attention (global layers grow with L; local layers WINDOWED
             -> flat beyond the window),
  added(L) = the STRICT-only extra tax = the 7 pinned local reductions run FULL
             instead of windowed (stark #472/#475 `whole_delta_us(L)`, grows ~O(L)).
TPS = E[T]*1e6 / T(L). The deployed and strict stacks share B, A_dep, the MTP-K7
drafter, and therefore the acceptance behaviour AND the global-layer ctxlen growth.
They differ ONLY in added(L). Propagating the public->private trajectory shift:

  strict_gap = 1 - (1 - deployed_gap)*(1 - local_pinned)
  local_pinned = 1 - kvw_strict(private_traj)/kvw_strict(public_traj)

where kvw_strict is stark #475's token-weighted-harmonic strict TPS over the KV
trajectory, and the shift moves every served prompt length by +ΔP tokens. The
deployed ctxlen bucket (global layers) and the acceptance bucket (drafter) CANCEL
out of `local_pinned` -- they are already inside `deployed_gap`. So strict simply
inherits the full deployed 4.295% gap PLUS the extra L-sensitivity from pinning the
local reductions to full attention. (Derivation reproduced in the self-tests:
local_pinned(ΔP=0) == 0 exactly, and kvw_strict(public_traj) reproduces the banked
461.8049.)

CALIBRATION (instruction 1)
---------------------------
stark #475's strict TPS is NOT a raw local number: kvw_strict(L) =
DEPLOYED_OFFICIAL(481.53) * CYC/(CYC+added(L)). The local pod enters ONLY through
the hardware-INVARIANT ratio CYC/(CYC+added) (both local-us, the clock cancels);
the absolute scale is the OFFICIAL deployed anchor. So 461.80 is already on the
official-public scale -- `local_to_public_bias` does NOT re-multiply it (doing so
would double-count). We report the deployed bias (0.9660 = local 465.14 / official
481.53, lawine #467 / systems transfer) for context and bound the residual
tax-fraction-transfer risk by tau_lo's stability (0.13%).

NOISE (instruction 4, cross-ref kanna #478)
-------------------------------------------
The single private re-run draws TPS ~ N(mu_priv, sigma_hw). lawine #467 measured the
between-run served-TPS sigma EMPIRICALLY at 0.349 TPS (0.073%) over 10 fresh runs
(clock-locked 1710/6251 MHz) -- the "1% convention" (4.8153 TPS) overstates it
13.8x. The official re-run is cross-SESSION/cross-node (unmeasured), so we bound the
noise on [empirical 0.349, convention 4.8153] and report the breach-prob band. Our
SYSTEMATIC shift STACKS with (does not double-count) kanna #478's session-noise band:
mu_priv carries the systematic (deployed_gap + local_pinned); sigma_hw is the fresh
single-draw noise on top.

Reproduce: cd target/ && .venv/bin/python \
  research/validity/private_rerun_validity/private_rerun_validity.py \
  --wandb_group equivalence-escalation-anchors --wandb_name denken/private-rerun-validity-474
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, "..", "..", ".."))

# ---- banked inputs (all read-only; this card measures nothing on the GPU) -----------
KVW_JSON = os.path.join(_root, "research/speed/kv_weighted_strict_tps/kv_weighted_strict_tps.json")
SWEEP_JSON = os.path.join(_root, "research/speed/strict_wholecycle_ab/strict_wholecycle_ab.json")
GAP_JSON = os.path.join(
    _root, "research/validity/public_private_gap_decomposition/public_private_gap_decomposition_results.json")
XFER_JSON = os.path.join(_root, "research/systems/local_official_tps_transfer/report.json")
SIGMA_JSON = os.path.join(_root, "research/empirical_sigma_hw/fresh_n10/sigma_hw.json")

# Validity gates (BASELINE.md / program.md).
DELTA_GATE = 0.05            # public<->private TPS reproduction gate
PPL_GATE = 2.42
PPL_ANCHOR = 2.3772         # deployed public PPL; greedy-identity invariant (denken #471/#476)
PPL_PRIVATE_DEPLOYED = 2.3777  # organizer private re-run PPL for the deployed stack

# Deployed ground-truth public/private pair (organizer cmpatino-verifier).
DEPLOYED_PUBLIC_TPS = 481.53
DEPLOYED_PRIVATE_TPS = 460.85

# Prompt-shift corners imported from ubel #379 (private prompt-length shift, tokens).
DP_BANKED = 0.0             # pure-rho convention: whole gap is acceptance, no traj shift
DP_CENTRAL = 50.0          # modest held-out shift (#379 central)
DP_PESSIMISTIC = 130.0     # public high-decile (#379 pessimistic)


def _phi(z: float) -> float:
    """Standard-normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def piecewise_linear(xs, ys):
    """Piecewise-linear interpolant, edge-slope extrapolation (identical to stark #475)."""
    pts = sorted(zip(xs, ys))

    def f(x):
        if x <= pts[0][0]:
            (x0, y0), (x1, y1) = pts[0], pts[1]
        elif x >= pts[-1][0]:
            (x0, y0), (x1, y1) = pts[-2], pts[-1]
        else:
            for k in range(len(pts) - 1):
                if pts[k][0] <= x <= pts[k + 1][0]:
                    (x0, y0), (x1, y1) = pts[k], pts[k + 1]
                    break
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=os.path.join(_here, "private_rerun_validity_results.json"))
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="equivalence-escalation-anchors")
    ap.add_argument("--wandb_name", default="denken/private-rerun-validity-474")
    ap.add_argument("--job_type", default="analysis")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    kvw = json.load(open(KVW_JSON))
    sweep = json.load(open(SWEEP_JSON))["verdict"]
    gap = json.load(open(GAP_JSON))
    xfer = json.load(open(XFER_JSON))
    sigma = json.load(open(SIGMA_JSON))["analysis"]

    # ---- (1) strict per-L model + trajectory (imported EXACT from stark #472/#475) ----
    DEPLOYED_TPS = float(sweep["deployed_tps"])         # 481.53 official anchor
    CYC = float(sweep["cycle_perm_us"])                 # 7666.83 deployed permissive cycle us
    added_at = {int(k): float(v) for k, v in kvw["config"]["added_us_at_L"].items()}
    Ls = sorted(added_at)
    added_pw = piecewise_linear(Ls, [added_at[L] for L in Ls])
    P = list(kvw["trajectory"]["served_prompt_lengths_sorted"])  # 128 served prompt lengths
    OUT = int(kvw["config"]["output_len"])              # 512
    strict_pub_banked = float(kvw["verdict"]["kv_weighted_strict_tps"])  # 461.8049
    public_mean_L = float(kvw["verdict"]["kv_trajectory_mean_L"])        # 527.656

    def tps_from_added(a):
        return DEPLOYED_TPS * CYC / (CYC + a)

    def kvw_strict(shift):
        """Token-weighted harmonic strict TPS over the KV trajectory shifted by +`shift` tokens."""
        inv = 0.0
        n = 0
        for p in P:
            base = p + shift
            for i in range(OUT):
                inv += 1.0 / tps_from_added(added_pw(base + i))
                n += 1
        return n / inv

    strict_pub = kvw_strict(0.0)                        # reproduces 461.8049

    def local_pinned(dp):
        """Strict-only extra L-tax fraction from a +dp-token private trajectory shift."""
        return 1.0 - kvw_strict(dp) / strict_pub

    # ---- (1) calibration: local->official-public bias (deployed); strict already anchored ----
    local_anchor = float(xfer["transfer"]["local_anchor_tps"])      # 465.14
    official_anchor = float(xfer["transfer"]["official_anchor_tps"])  # 481.53
    tau_lo = float(xfer["tau_lo"])                                   # 1.03524
    local_to_public_bias = local_anchor / official_anchor           # 0.9660
    tau_lo_spread_pct = float(xfer["tau_lo_stability"]["tau_lo_spread_pct"])  # 0.135 residual

    # ---- (2) deployed public->private decomposition (imported from ubel #379) ----
    deployed_gap = float(gap["decomposition_central"]["total_gap_frac"])   # 0.042946
    acceptance_pct = float(gap["bucket_acceptance_abs_pct"])               # 3.661 (drafter, SHARED)
    ctxlen_pct = float(gap["bucket_ctxlen_abs_pct"])                       # 0.633 (KV-traj, global)
    deployed_gap_recon = (DEPLOYED_PUBLIC_TPS - DEPLOYED_PRIVATE_TPS) / DEPLOYED_PUBLIC_TPS
    public_mean_prompt = statistics.mean(P)
    private_mean_prompt_central = public_mean_prompt + DP_CENTRAL
    private_mean_L_central = public_mean_L + DP_CENTRAL                    # ~577.7
    private_trajectory_delta_pct = ctxlen_pct        # the deployed KV-trajectory component

    # ---- (3) propagate to the strict config ----
    def strict_gap_of(dp):
        return 1.0 - (1.0 - deployed_gap) * (1.0 - local_pinned(dp))

    def strict_priv_tps(dp):
        return strict_pub * (1.0 - strict_gap_of(dp))

    corners = {}
    for tag, dp in [("banked", DP_BANKED), ("central", DP_CENTRAL), ("pessimistic", DP_PESSIMISTIC)]:
        lp = local_pinned(dp)
        sg = strict_gap_of(dp)
        corners[tag] = {
            "delta_p_tokens": dp,
            "local_pinned_pct": 100.0 * lp,
            "strict_gap_pct": 100.0 * sg,
            "strict_private_tps": strict_priv_tps(dp),
            "exceeds_5pct": bool(sg > DELTA_GATE),
        }

    strict_gap_central = strict_gap_of(DP_CENTRAL)
    strict_private_delta_pct = 100.0 * strict_gap_central          # PRIMARY
    strict_private_predicted_tps = strict_priv_tps(DP_CENTRAL)

    # strict systematic breakeven: the +dp that drives strict_gap to exactly 5%
    lo, hi = 0.0, 600.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if strict_gap_of(mid) < DELTA_GATE:
            lo = mid
        else:
            hi = mid
    strict_breakeven_dp = (lo + hi) / 2.0

    # ---- (4) P(breach) = P(strict_gap_systematic + session_noise > 5%) ----
    # threshold: breach iff private draw < 0.95 * strict_public.
    threshold_tps = (1.0 - DELTA_GATE) * strict_pub
    sigma_empirical = float(sigma["empirical_sigma_hw_tps"])       # 0.349 (lawine #467, measured)
    sigma_convention = float(sigma["convention_sigma_hw"])         # 4.8153 (1% convention, refuted 13.8x)

    def breach_prob(dp, sig):
        mu = strict_priv_tps(dp)
        # P(draw < threshold) under N(mu, sig)
        return _phi((threshold_tps - mu) / sig)

    breach_grid = {}
    for tag, dp in [("central", DP_CENTRAL), ("pessimistic", DP_PESSIMISTIC)]:
        breach_grid[tag] = {
            "empirical_sigma": breach_prob(dp, sigma_empirical),
            "convention_sigma": breach_prob(dp, sigma_convention),
        }
    # headline breach prob: realistic (empirical sigma) at the central shift; the
    # conservative band-top (convention sigma) is logged alongside.
    private_validity_breach_prob = breach_grid["central"]["empirical_sigma"]
    private_validity_breach_prob_conservative = breach_grid["central"]["convention_sigma"]

    # deployed analog (for the comparison the gate cares about): how much margin
    # the deployed stack itself had, and its breach prob on a fresh draw.
    deployed_threshold = (1.0 - DELTA_GATE) * DEPLOYED_PUBLIC_TPS
    deployed_breach_emp = _phi((deployed_threshold - DEPLOYED_PRIVATE_TPS) / sigma_empirical)
    deployed_breach_conv = _phi((deployed_threshold - DEPLOYED_PRIVATE_TPS) / sigma_convention)
    deployed_margin_pct = 100.0 * (DELTA_GATE - deployed_gap)
    strict_margin_pct = 100.0 * (DELTA_GATE - strict_gap_central)

    # ---- (3) PPL on private: greedy-identity invariant at the locus ----
    ppl_private_predicted = PPL_ANCHOR    # carry; strict is byte-exact (denken #471/#476)
    ppl_margin = PPL_GATE - ppl_private_predicted
    ppl_clears = bool(ppl_private_predicted <= PPL_GATE)

    # ---- (4) verdict ----
    # SAFE iff: central strict_gap clears 5%, realistic-noise breach prob is small,
    # PPL clears with margin, and the systematic breakeven sits above the #379 central shift.
    central_clears = bool(strict_gap_central < DELTA_GATE)
    realistic_breach_small = bool(private_validity_breach_prob < 0.05)
    breakeven_above_central = bool(strict_breakeven_dp > DP_CENTRAL)
    private_validity_safe = bool(central_clears and realistic_breach_small and ppl_clears)
    # FLAG: thinner margin than deployed, and pessimistic-shift corner breaches.
    thin_margin_flag = bool(strict_margin_pct < deployed_margin_pct)
    pessimistic_breaches = bool(corners["pessimistic"]["exceeds_5pct"])

    # ---- self-tests ----
    st = {}
    st["kvw_reproduces_banked_pub"] = bool(abs(strict_pub - strict_pub_banked) < 0.01)
    st["local_pinned_zero_at_no_shift"] = bool(abs(local_pinned(0.0)) < 1e-9)
    st["local_pinned_monotone_in_shift"] = bool(
        local_pinned(0.0) < local_pinned(50.0) < local_pinned(130.0) < local_pinned(253.0))
    st["deployed_gap_reconstructs_4p295"] = bool(abs(deployed_gap - deployed_gap_recon) < 1e-4)
    st["strict_gap_ge_deployed_gap"] = bool(strict_gap_central >= deployed_gap - 1e-12)
    st["strict_gap_central_clears_5pct"] = bool(strict_gap_central < DELTA_GATE)
    st["strict_gap_banked_equals_deployed"] = bool(
        abs(corners["banked"]["strict_gap_pct"] / 100.0 - deployed_gap) < 1e-9)
    st["breakeven_between_central_and_pessimistic"] = bool(DP_CENTRAL < strict_breakeven_dp < DP_PESSIMISTIC)
    st["pessimistic_corner_breaches"] = bool(corners["pessimistic"]["exceeds_5pct"])
    st["strict_pub_below_openevolve_max_verifiable"] = bool(strict_pub < 495.7)  # honest, not a mirage
    st["calibration_bias_in_unit_band"] = bool(0.90 < local_to_public_bias < 1.0)
    st["bias_does_not_remultiply_strict"] = bool(  # strict already anchored to official 481.53
        abs(tps_from_added(0.0) - DEPLOYED_PUBLIC_TPS) < 1e-6)
    st["ppl_clears_gate_with_margin"] = bool(ppl_clears and ppl_margin > 0.03)
    st["breach_prob_monotone_in_sigma"] = bool(
        breach_grid["central"]["empirical_sigma"] <= breach_grid["central"]["convention_sigma"])
    st["breach_prob_monotone_in_shift"] = bool(
        breach_grid["central"]["empirical_sigma"] <= breach_grid["pessimistic"]["empirical_sigma"])
    st["threshold_is_95pct_of_pub"] = bool(abs(threshold_tps - 0.95 * strict_pub) < 1e-9)
    st["sigma_empirical_below_convention"] = bool(sigma_empirical < sigma_convention)
    finite = [strict_gap_central, strict_private_predicted_tps, private_validity_breach_prob,
              local_to_public_bias, private_trajectory_delta_pct, strict_breakeven_dp,
              ppl_private_predicted, private_validity_breach_prob_conservative]
    st["nan_clean"] = all(math.isfinite(x) for x in finite)
    self_test_passes = all(st.values())

    verdict = {
        # ---- PR-required logged metrics ----
        "strict_private_delta_pct": strict_private_delta_pct,                 # PRIMARY (central)
        "strict_private_predicted_tps": strict_private_predicted_tps,
        "private_validity_breach_prob": private_validity_breach_prob,         # P(Delta>5%), realistic
        "private_validity_breach_prob_conservative": private_validity_breach_prob_conservative,
        "local_to_public_bias": local_to_public_bias,
        "private_trajectory_delta_pct": private_trajectory_delta_pct,
        "private_validity_safe": private_validity_safe,
        "ppl": PPL_ANCHOR,
        # ---- supporting numbers ----
        "deployed_gap_pct": 100.0 * deployed_gap,
        "deployed_acceptance_pct": acceptance_pct,
        "deployed_ctxlen_pct": ctxlen_pct,
        "strict_gap_central_pct": 100.0 * strict_gap_central,
        "strict_local_pinned_central_pct": corners["central"]["local_pinned_pct"],
        "strict_breakeven_dp_tokens": strict_breakeven_dp,
        "strict_margin_pct": strict_margin_pct,
        "deployed_margin_pct": deployed_margin_pct,
        "thin_margin_flag": thin_margin_flag,
        "pessimistic_corner_breaches": pessimistic_breaches,
        "public_mean_prompt_tokens": public_mean_prompt,
        "private_mean_prompt_tokens_central": private_mean_prompt_central,
        "public_mean_L": public_mean_L,
        "private_mean_L_central": private_mean_L_central,
        "strict_public_tps": strict_pub,
        "sigma_hw_empirical_tps": sigma_empirical,
        "sigma_hw_convention_tps": sigma_convention,
        "deployed_breach_prob_empirical": deployed_breach_emp,
        "deployed_breach_prob_convention": deployed_breach_conv,
        "tau_lo": tau_lo,
        "tau_lo_residual_spread_pct": tau_lo_spread_pct,
        "ppl_private_predicted": ppl_private_predicted,
        "ppl_private_deployed_measured": PPL_PRIVATE_DEPLOYED,
        "ppl_margin": ppl_margin,
        "ppl_clears": ppl_clears,
        "delta_gate": DELTA_GATE,
        "ppl_gate": PPL_GATE,
        "analysis_only": True, "official_tps": 0, "no_served_file_change": True,
        "no_kernel_rebuild": True, "no_hf_job": True, "no_launch": True, "no_submission": True,
        "self_test_passes": self_test_passes,
    }

    reconcile = (
        f"Strict #474 private re-run validity. Strict shares the MTP-K7 drafter, body B, "
        f"global-layer attention, and acceptance with the deployed stack; it differs ONLY by "
        f"pinning the 7 local reductions to FULL attention (stark #472/#475 added_us(L)). So "
        f"strict_gap = deployed_gap({100*deployed_gap:.3f}%) + local_pinned (the extra L-tax over "
        f"the private trajectory shift); the deployed ctxlen + acceptance buckets CANCEL out of "
        f"local_pinned. Central (DP=+{DP_CENTRAL:.0f} tok, ubel #379): local_pinned="
        f"{corners['central']['local_pinned_pct']:.3f}% -> strict_gap={strict_private_delta_pct:.3f}% "
        f"(predicted private {strict_private_predicted_tps:.2f} TPS), CLEARS 5% with "
        f"{strict_margin_pct:.3f}pp vs deployed's {deployed_margin_pct:.3f}pp. Systematic breakeven "
        f"at DP=+{strict_breakeven_dp:.1f} tok (between #379 central +50 and pessimistic +130); "
        f"pessimistic +130 corner breaches ({corners['pessimistic']['strict_gap_pct']:.2f}%). "
        f"P(breach) = P(systematic + session-noise > 5%): {100*private_validity_breach_prob:.3f}% "
        f"(empirical sigma_hw {sigma_empirical:.3f} TPS, lawine #467) .. "
        f"{100*private_validity_breach_prob_conservative:.1f}% (1% convention {sigma_convention:.3f}, "
        f"refuted 13.8x by #467) -- STACKS with kanna #478's noise band, no double-count. "
        f"Strict 461.80 is already official-anchored (kvw uses the 481.53 deployed anchor), so "
        f"local_to_public_bias={local_to_public_bias:.4f} does NOT re-multiply it. PPL {PPL_ANCHOR} "
        f"<= 2.42 (margin {ppl_margin:.4f}; greedy-identity invariant, denken #471/#476). "
        f"VERDICT: private_validity_safe={private_validity_safe} (central, realistic noise) with a "
        f"FLAG -- margin is thinner than deployed and the +130 shift corner breaches.")
    verdict["reconcile_line"] = reconcile

    payload = {
        "pr": 480,
        "issue": 474,
        "author": "denken",
        "leg": "private re-run SPEED-validity gate (5% Delta) for the strict #474 submission",
        "config": {
            "delta_gate": DELTA_GATE, "ppl_gate": PPL_GATE,
            "deployed_public_tps": DEPLOYED_PUBLIC_TPS, "deployed_private_tps": DEPLOYED_PRIVATE_TPS,
            "strict_public_tps": strict_pub,
            "dp_corners_tokens": {"banked": DP_BANKED, "central": DP_CENTRAL, "pessimistic": DP_PESSIMISTIC},
            "imports": {
                "strict_perL_tax": os.path.relpath(KVW_JSON, _root),
                "strict_wholecycle": os.path.relpath(SWEEP_JSON, _root),
                "deployed_gap_decomp": os.path.relpath(GAP_JSON, _root),
                "local_official_transfer": os.path.relpath(XFER_JSON, _root),
                "empirical_sigma_hw": os.path.relpath(SIGMA_JSON, _root),
            },
            "note": "Private re-run validity for strict #474. CPU analysis only; no kernel "
                    "re-measure, no served change, no HF Job, no launch, no submission.",
        },
        "corners": corners,
        "breach_grid": breach_grid,
        "verdict": verdict,
        "self_test_conditions": st,
        "public_evidence_used": (
            "openevolve board finding 20260616-062754-273 (mentions @senpai): the 5% Delta gate is "
            "actively INVALIDATING high-public entries (w256+precache 508.04->470.95 Delta 7.3% INVALID; "
            "verified 489.66 -> private ~470 Delta 3.9%); 'honest private decode ~470'; max verifiable "
            "public ~= private/0.95 ~= 495.7. Strict's 461.80 public is BELOW that mirage line (no "
            "sliding-window/precache public-only boost), so it sits in the honest regime -- bracketing "
            "strict_gap from BELOW (openevolve: removing the window mirage shrinks Delta) while this "
            "card's full-attention per-L model is the conservative UPPER bound. Deployed pair from "
            "BASELINE.md / cmpatino-verifier (20260613-230441-229)."),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(payload, open(args.output, "w"), indent=2,
              default=lambda o: float(o) if isinstance(o, (int, float)) else str(o))

    print(f"[prv] strict_private_delta={strict_private_delta_pct:.3f}% "
          f"(predicted private {strict_private_predicted_tps:.2f} TPS) | clears 5%: {central_clears} "
          f"| margin {strict_margin_pct:.3f}pp (deployed {deployed_margin_pct:.3f}pp)", flush=True)
    print(f"[prv] local_pinned central=+{corners['central']['local_pinned_pct']:.3f}% | "
          f"breakeven DP=+{strict_breakeven_dp:.1f} tok | pessimistic corner breaches: "
          f"{pessimistic_breaches}", flush=True)
    print(f"[prv] P(breach)={100*private_validity_breach_prob:.3f}% (empirical sigma) .. "
          f"{100*private_validity_breach_prob_conservative:.1f}% (1% convention) | "
          f"safe={private_validity_safe} thin_flag={thin_margin_flag}", flush=True)
    print(f"[prv] local_to_public_bias={local_to_public_bias:.4f} (deployed; strict already "
          f"official-anchored) | ppl {PPL_ANCHOR}<=2.42 margin {ppl_margin:.4f}", flush=True)
    print(f"[prv] self_test={self_test_passes} | {reconcile}", flush=True)

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

    ct = wandb.Table(columns=["corner", "delta_p_tokens", "local_pinned_pct",
                              "strict_gap_pct", "strict_private_tps", "exceeds_5pct"])
    for tag, c in payload["corners"].items():
        ct.add_data(tag, c["delta_p_tokens"], c["local_pinned_pct"],
                    c["strict_gap_pct"], c["strict_private_tps"], c["exceeds_5pct"])
    run.log({"strict_private_corners": ct})

    bt = wandb.Table(columns=["shift_corner", "sigma_model", "breach_prob"])
    for tag, g in payload["breach_grid"].items():
        for sm, pv in g.items():
            bt.add_data(tag, sm, pv)
    run.log({"breach_prob_grid": bt})
    run.finish()
    print(f"[prv] logged W&B run {args.wandb_entity}/{args.wandb_project} "
          f"name={args.wandb_name} group={args.wandb_group}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
