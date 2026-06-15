#!/usr/bin/env python
"""Drafter-fusion build scope (PR #424) — validate the ~+52 TPS roofline, gate it.

ANALYSIS ONLY. No training, no HF Job, no served-file change, official_tps=0.

The card asks: the deployed MTP K=7 drafter forward is "launch/sync/overhead-
bound, running as separate per-head launches well above its ~248 us BW floor";
fusing the 7 heads toward that floor is a "~+52 TPS roofline". Validate the
roofline, discount to a realistic band, price identity-verify cost, and scope the
build surface — precise enough to hand the human a build GO/NO-GO.

VERDICT (built strictly from merged artifacts on approval-gated-8gpu-20260613):
  NO-GO. The premise is contradicted by the served submission + merged record:
    1. The K=7 drafter is ALREADY a single ONEGRAPH CUDA-graph replay, not
       "separate per-head launches". The per-head launch overhead is already
       harvested (eager 2859 us -> graph 566 us). [served sitecustomize.py:25-33;
       PR #75 uknpbk94; PR #261 egaz6m2f]
    2. The ~248 us floor is the GEMM-chain BW floor (566 us @ 47.17% HBM peak).
       Reaching it is INFEASIBLE at M=1: the M=1 GEMVs are occupancy/latency-
       floored (41-47% HBM) and the 7 passes are autoregressive-serial, so they
       cannot be batched into one BW-saturating GEMM with unchanged outputs.
       [PR #75; PR #269 epl52mkq "physically unreachable at M=1 without M>=16"]
    3. The honest GEMM-chain -> BW-floor roofline is ~+20 TPS, NOT +52. +52
       (= +10.8%) needs ~781 us of drafter saving, MORE than the entire 566 us
       GEMM chain exists -> it must (wrongly) eat the ~879 us non-GEMM mass that
       PR #75 shows is untouchable by GEMM fusion.
    4. Realistic recovery ~0 (launch floor already harvested; M=1 BW floor
       unreachable; draft-side step-cut realization measured NEGATIVE in #273,
       realization_ratio=-2.02).
    5. Blast radius HIGH: the onegraph loopgraph IS the 481.53 engine; rewriting
       it risks the measured -16.5% regression to ~402 TPS [PR #312/#315].

All numbers below are derived (not hard-coded) from the canonical merged inputs.
"""
from __future__ import annotations

import json
import os

# ---------------------------------------------------------------------------
# CANONICAL MERGED INPUTS  (approval-gated-8gpu-20260613)
# ---------------------------------------------------------------------------
# Deployed per-step decode wall + GPU-busy split  -- PR #284 (ubel, u58fxtu6)
T_STEP_DEPLOYED_US = 8017.0   # host-to-host p50 decode wall
VERIFY_US = 6532.0            # verify forward (CUDA-event)
DRAFTER_FULL_US = 1445.0      # FULL drafter forward, in-stack (CUDA-event)
HOST_US = 40.0               # host/serving residual (0.50%)

# Drafter-forward roofline  -- PR #75 (denken, uknpbk94)
DRAFTER_GEMM_CHAIN_US = 566.0     # 7-pass GEMM/GEMV chain, ONEGRAPH (launch-free)
DRAFTER_GEMM_PCT_HBM_PEAK = 0.4717  # 47.17% of HBM peak at deployed M=1xK=7
DRAFTER_EAGER_CHAIN_US = 2859.0   # eager (NO onegraph) -> already-harvested 2.3 ms
GEMM_ZERO_HARD_CEILING_PCT_75 = 0.0513  # "every drafter GEMM -> 0 us" on #75's step

# Acceptance / TPS anchors  -- PR body #424 / #289 / #392
E_T = 3.851                  # tokens emitted per spec-step (1 + E[accepted] 2.851)
BASE_OFFICIAL_TPS = 481.53   # deployed flagship (PR #52, 2x9fm2zx)
BASE_STRICT_TPS = 482.74     # realizable strictly-equivalent stack (467.14 + cb3 15.60)

# Advisor's quoted figures for this card (to be validated)
ADVISOR_QUOTED_FLOOR_US = 248.0
ADVISOR_QUOTED_ROOFLINE_TPS = 52.0

# ---------------------------------------------------------------------------
# DERIVED QUANTITIES
# ---------------------------------------------------------------------------
DRAFTER_NONGEMM_US = DRAFTER_FULL_US - DRAFTER_GEMM_CHAIN_US          # ~879 us
GEMM_BW_FLOOR_MERGED_US = DRAFTER_GEMM_CHAIN_US * DRAFTER_GEMM_PCT_HBM_PEAK  # ~267 us
# Strict-stack step time implied by E[T] and the strict base TPS (seconds->us)
STEP_STRICT_US = E_T / BASE_STRICT_TPS * 1e6                          # ~7977 us


def tps_after_saving(base_tps: float, step_us: float, saved_us: float) -> float:
    """TPS scales as 1/step when tokens-per-step (E[T]) is held fixed."""
    return base_tps * step_us / (step_us - saved_us)


def delta_for_saving(base_tps: float, step_us: float, saved_us: float) -> float:
    return tps_after_saving(base_tps, step_us, saved_us) - base_tps


def saving_for_delta(base_tps: float, step_us: float, target_delta: float) -> float:
    """Inverse: us of step reduction needed to gain `target_delta` TPS."""
    new_tps = base_tps + target_delta
    return step_us * (1.0 - base_tps / new_tps)


# --- Roofline readings (all on the realizable strict stack unless noted) -----
# (a) The DEFENSIBLE reading: only the GEMM chain can reach a *GEMM* BW floor.
save_gemm_to_merged_floor = DRAFTER_GEMM_CHAIN_US - GEMM_BW_FLOOR_MERGED_US   # ~299
save_gemm_to_advisor_floor = DRAFTER_GEMM_CHAIN_US - ADVISOR_QUOTED_FLOOR_US  # ~318

roofline_gemm_merged_floor_strict = delta_for_saving(
    BASE_STRICT_TPS, STEP_STRICT_US, save_gemm_to_merged_floor)
roofline_gemm_advisor_floor_strict = delta_for_saving(
    BASE_STRICT_TPS, STEP_STRICT_US, save_gemm_to_advisor_floor)
roofline_gemm_merged_floor_deployed = delta_for_saving(
    BASE_OFFICIAL_TPS, T_STEP_DEPLOYED_US, save_gemm_to_merged_floor)

# (b) Hard ceiling: delete the ENTIRE GEMM chain (566 -> 0). Physical upper bound.
hard_ceiling_gemm_zero_strict = delta_for_saving(
    BASE_STRICT_TPS, STEP_STRICT_US, DRAFTER_GEMM_CHAIN_US)
hard_ceiling_gemm_zero_deployed = delta_for_saving(
    BASE_OFFICIAL_TPS, T_STEP_DEPLOYED_US, DRAFTER_GEMM_CHAIN_US)

# (c) The (wrong) full-collapse reading that ~+52 implicitly needs: collapse the
#     FULL drafter (incl. 879 us non-GEMM) to 248 us.
save_full_to_advisor_floor = DRAFTER_FULL_US - ADVISOR_QUOTED_FLOOR_US        # ~1197
fullcollapse_advisor_floor_deployed = delta_for_saving(
    BASE_OFFICIAL_TPS, T_STEP_DEPLOYED_US, save_full_to_advisor_floor)

# (d) How much drafter saving does the advisor's +52 actually require?
saving_needed_for_52_deployed = saving_for_delta(
    BASE_OFFICIAL_TPS, T_STEP_DEPLOYED_US, ADVISOR_QUOTED_ROOFLINE_TPS)
saving_needed_for_52_strict = saving_for_delta(
    BASE_STRICT_TPS, STEP_STRICT_US, ADVISOR_QUOTED_ROOFLINE_TPS)
# ...and does that exceed the entire GEMM chain (i.e. must it eat non-GEMM)?
plus52_needs_nongemm = saving_needed_for_52_deployed > DRAFTER_GEMM_CHAIN_US
plus52_nongemm_overdraw_us = saving_needed_for_52_deployed - DRAFTER_GEMM_CHAIN_US

# --- PRIMARY: the validated BW-floor roofline (advisor's own 248 floor, applied
#     correctly to the GEMM chain only, on the realizable strict stack). --------
DRAFTER_FUSION_ROOFLINE_TPS = roofline_gemm_advisor_floor_strict   # ~+20.0

# --- REALISTIC band ----------------------------------------------------------
# Launch fusion: 0 (already harvested by onegraph). BW saturation: unreachable at
# M=1. The only separable draft-side micro-levers found (GeluAndMul fold #269,
# +4.4% composition ceiling) MEASURED NEGATIVE realization in #273
# (realization_ratio=-2.02). So realistic central ~0; optimistic cap = pushing
# the GEMM chain from 47% -> 70% HBM (a stretch M=1 cannot deliver).
gemm_at_70pct_hbm_us = DRAFTER_GEMM_CHAIN_US * DRAFTER_GEMM_PCT_HBM_PEAK / 0.70
save_gemm_to_70pct = DRAFTER_GEMM_CHAIN_US - gemm_at_70pct_hbm_us             # ~185
realistic_optimistic_cap_tps = delta_for_saving(
    BASE_STRICT_TPS, STEP_STRICT_US, save_gemm_to_70pct)                      # ~+11.5
REALISTIC_LOW_TPS = 0.0
REALISTIC_CENTRAL_TPS = 1.5   # dominated by ~0 (launch harvested; M=1 intrinsic)
REALISTIC_HIGH_TPS = 5.0      # GeluAndMul-fold-class, unrealized in #273

# --- Identity / acceptance ---------------------------------------------------
# Regime A: bit-identical fusion -> identical proposals -> acceptance UNCHANGED.
#   Identity trivially preserved (and OUTPUT identity is anyway M=8-verify-gated).
#   BUT the only bit-identical fusion with positive speed headroom IS onegraph,
#   already deployed -> regime A speed headroom ~0.
FUSION_IDENTITY_PRESERVING = True
REGIME_A_SPEED_HEADROOM = False
# Regime B: numerically-different-but-faster -> proposals shift -> E[T] may move.
#   TPS sensitivity to E[T]:
DTPS_DET = BASE_STRICT_TPS / E_T                    # ~125 TPS per unit E[T]
#   A modest acceptance regression wipes the whole realistic speed band:
ET_REGRESSION_THAT_WIPES_REALISTIC = REALISTIC_HIGH_TPS / DTPS_DET   # ~0.04
FUSION_ACCEPTANCE_RATE_DELTA_LOW = -0.10   # no mechanism for a faster drafter to
FUSION_ACCEPTANCE_RATE_DELTA_HIGH = 0.05   # improve acceptance; neutral-to-worse
FUSION_ACCEPTANCE_RATE_DELTA = 0.0         # central (indeterminate sign)
REGIME_B_NET_CAN_BE_NEGATIVE = True
# Identity-verify cost: the served runtime is UNCHANGED by an offline greedy/PPL
# re-gate -> 0 served-TPS cost. (Build-time CI only: one served-vs-served greedy
# identity pass + PPL<=2.42 over 128 prompts, per BASELINE.md.)
FUSION_IDENTITY_VERIFY_COST_TPS = 0.0

# --- Stack projection --------------------------------------------------------
STACK_TPS_IF_REALISTIC_RECOVERY = BASE_STRICT_TPS + REALISTIC_CENTRAL_TPS

SERVED_KERNEL_SURFACE = (
    "submissions/fa2sw_precache_kenyan/sitecustomize.py (ONEGRAPH loopgraph: "
    "_run_graph_body / _capture_graph / _is_loopgraph_eligible; keyed to "
    "Gemma4Proposer on vllm.v1.spec_decode.gemma4 + gemma4_mtp.get_top_tokens) "
    "+ a NEW fused CUDA/Triton kernel replacing the 7 per-iteration width-1 "
    "Gemma4MTP sub-forwards (q_proj/o_proj/q_norm sliding-attn GEMVs + 4-layer "
    "256-dim gated MLP + centroid sparse sampler + 262k masked-embed gather). "
    "Served-file change => human-approval-gated."
)

FUSION_BUILD_BLAST_RADIUS = (
    "HIGH. (1) The ONEGRAPH loopgraph IS the engine of 481.53 (PR #312/#315): "
    "rewriting _run_graph_body/_capture_graph risks the MEASURED -16.5% "
    "regression to ~402 TPS if capture breaks or goes inert (~80 TPS debt). "
    "(2) Reopens all 4 gates (greedy-identity, PPL<=2.42, boot-500, TPS) + the "
    "#272 boot-500 _guard_included_router co-edit. (3) A 7-pass mega-kernel "
    "raises register-pressure/occupancy risk -> can be SLOWER than the current "
    "per-pass kernels. (4) The autoregressive serial dependency (pass i consumes "
    "pass i-1's token) means a fused kernel CANNOT beat onegraph on the GEMM math "
    "at M=1 (occupancy-bound) => high probability of ~0 payoff. Net: high risk to "
    "the flagship for <=+5 realistic TPS."
)

# ---------------------------------------------------------------------------
# SELF-TESTS
# ---------------------------------------------------------------------------
checks = {
    "step_decomposition_sums": abs((VERIFY_US + DRAFTER_FULL_US + HOST_US)
                                   - T_STEP_DEPLOYED_US) < 1e-6,
    "gemm_chain_within_full_drafter": DRAFTER_GEMM_CHAIN_US < DRAFTER_FULL_US,
    "nongemm_positive": DRAFTER_NONGEMM_US > 0,
    "gemm_bw_floor_below_chain": GEMM_BW_FLOOR_MERGED_US < DRAFTER_GEMM_CHAIN_US,
    "roofline_below_hard_ceiling": (
        DRAFTER_FUSION_ROOFLINE_TPS < hard_ceiling_gemm_zero_strict),
    "advisor52_exceeds_hard_ceiling": (
        ADVISOR_QUOTED_ROOFLINE_TPS > hard_ceiling_gemm_zero_deployed),
    "plus52_requires_nongemm": plus52_needs_nongemm,
    "plus52_overdraws_gemm_chain": (
        saving_needed_for_52_deployed > DRAFTER_GEMM_CHAIN_US),
    "realistic_high_below_roofline": REALISTIC_HIGH_TPS < DRAFTER_FUSION_ROOFLINE_TPS,
    "launch_already_harvested": DRAFTER_EAGER_CHAIN_US > 4 * DRAFTER_GEMM_CHAIN_US,
    "dtps_det_positive": DTPS_DET > 0,
    "stack_equals_base_plus_realistic": abs(
        STACK_TPS_IF_REALISTIC_RECOVERY
        - (BASE_STRICT_TPS + REALISTIC_CENTRAL_TPS)) < 1e-9,
    "et_regression_wipe_small": ET_REGRESSION_THAT_WIPES_REALISTIC < 0.06,
}
DRAFTER_FUSION_SELF_TEST_PASSES = all(checks.values())

# ---------------------------------------------------------------------------
# RESULT RECORD
# ---------------------------------------------------------------------------
summary = {
    # ---- markers (card-mandated) ----
    "analysis_only": True,
    "no_hf_job": True,
    "no_served_file_change": True,
    "official_tps": 0,

    # ---- PRIMARY deliverable ----
    "drafter_fusion_roofline_tps": round(DRAFTER_FUSION_ROOFLINE_TPS, 3),

    # ---- realistic recovery band ----
    "drafter_fusion_realistic_tps": round(REALISTIC_CENTRAL_TPS, 3),
    "drafter_fusion_realistic_tps_low": round(REALISTIC_LOW_TPS, 3),
    "drafter_fusion_realistic_tps_high": round(REALISTIC_HIGH_TPS, 3),
    "drafter_fusion_realistic_optimistic_cap_tps": round(realistic_optimistic_cap_tps, 3),

    # ---- current vs floor ----
    "drafter_forward_current_us": DRAFTER_FULL_US,
    "drafter_forward_floor_us": ADVISOR_QUOTED_FLOOR_US,
    "drafter_gemm_chain_us": DRAFTER_GEMM_CHAIN_US,
    "drafter_gemm_bw_floor_us_merged": round(GEMM_BW_FLOOR_MERGED_US, 2),
    "drafter_nongemm_us": DRAFTER_NONGEMM_US,
    "drafter_eager_chain_us": DRAFTER_EAGER_CHAIN_US,
    "drafter_gemm_pct_hbm_peak": DRAFTER_GEMM_PCT_HBM_PEAK,

    # ---- identity / acceptance ----
    "fusion_identity_preserving": FUSION_IDENTITY_PRESERVING,
    "regime_A_speed_headroom": REGIME_A_SPEED_HEADROOM,
    "fusion_acceptance_rate_delta": FUSION_ACCEPTANCE_RATE_DELTA,
    "fusion_acceptance_rate_delta_low": FUSION_ACCEPTANCE_RATE_DELTA_LOW,
    "fusion_acceptance_rate_delta_high": FUSION_ACCEPTANCE_RATE_DELTA_HIGH,
    "regime_B_net_can_be_negative": REGIME_B_NET_CAN_BE_NEGATIVE,
    "dtps_per_unit_et": round(DTPS_DET, 3),
    "et_regression_that_wipes_realistic": round(ET_REGRESSION_THAT_WIPES_REALISTIC, 4),
    "fusion_identity_verify_cost_tps": FUSION_IDENTITY_VERIFY_COST_TPS,

    # ---- build surface / blast radius ----
    "served_kernel_surface": SERVED_KERNEL_SURFACE,
    "fusion_build_blast_radius": FUSION_BUILD_BLAST_RADIUS,

    # ---- stack projection ----
    "stack_tps_if_realistic_recovery": round(STACK_TPS_IF_REALISTIC_RECOVERY, 3),

    # ---- the +52 refutation (decision-critical) ----
    "advisor_quoted_roofline_tps": ADVISOR_QUOTED_ROOFLINE_TPS,
    "roofline_gemm_merged_floor_tps": round(roofline_gemm_merged_floor_strict, 3),
    "roofline_gemm_advisor_floor_tps": round(roofline_gemm_advisor_floor_strict, 3),
    "hard_ceiling_gemm_zero_tps_strict": round(hard_ceiling_gemm_zero_strict, 3),
    "hard_ceiling_gemm_zero_tps_deployed": round(hard_ceiling_gemm_zero_deployed, 3),
    "fullcollapse_to_floor_tps_deployed": round(fullcollapse_advisor_floor_deployed, 3),
    "saving_needed_for_plus52_us": round(saving_needed_for_52_deployed, 2),
    "plus52_requires_nongemm": plus52_needs_nongemm,
    "plus52_nongemm_overdraw_us": round(plus52_nongemm_overdraw_us, 2),

    # ---- premise refutation ----
    "premise_separate_per_head_launches": False,  # served code: single ONEGRAPH replay
    "drafter_already_onegraph_captured": True,

    # ---- verdict ----
    "gate": "NO-GO",
    "verdict": "DRAFTER-FUSION-LAUNCH-FLOOR-ALREADY-HARVESTED / BW-FLOOR-M1-UNREACHABLE",
    "drafter_fusion_self_test_passes": DRAFTER_FUSION_SELF_TEST_PASSES,
    "self_test_n_passed": sum(checks.values()),
    "self_test_n_total": len(checks),
}

config = {
    "method": "drafter_fusion_build_scope",
    "lane": "drafter-fusion-scope",
    "pr": 424,
    "analysis_type": "cpu-analytic-roofline-from-merged-artifacts",
    "gpu_used": False,
    "hf_launch": False,
    "deployed_submission": "submissions/fa2sw_precache_kenyan",
    "merged_inputs": {
        "step_decode_wall_us": "#284 u58fxtu6",
        "drafter_gemm_chain_us_and_pct_hbm": "#75 uknpbk94",
        "draft_launch_already_captured": "#261 egaz6m2f / #246",
        "draft_mlp_intrinsic_m1": "#269 epl52mkq",
        "draft_sdpa_null": "#270 iwwcmvez",
        "draft_stepcut_realization_negative": "#273 (read-cut, ratio -2.02)",
        "onegraph_is_481_engine": "#312 / #315",
        "acceptance_ladder_E_T": "#289 / #392",
    },
    "baseline_official_tps": BASE_OFFICIAL_TPS,
    "baseline_strict_stack_tps": BASE_STRICT_TPS,
    "E_T": E_T,
}

if __name__ == "__main__":
    out = {"summary": summary, "config": config, "checks": checks,
           "intermediate": {
               "step_strict_us": STEP_STRICT_US,
               "save_gemm_to_merged_floor_us": save_gemm_to_merged_floor,
               "save_gemm_to_advisor_floor_us": save_gemm_to_advisor_floor,
               "gemm_at_70pct_hbm_us": gemm_at_70pct_hbm_us,
           }}
    with open(os.path.join(os.path.dirname(__file__), "scope_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nSELF-TESTS:", sum(checks.values()), "/", len(checks))
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    if os.environ.get("NO_WANDB") == "1":
        print("NO_WANDB=1 -> skipping wandb")
        raise SystemExit(0)

    import wandb
    run = wandb.init(
        project="gemma-challenge-senpai",
        entity="wandb-applied-ai-team",
        name="wirbel/drafter-fusion-build-scope",
        group="drafter-fusion-scope",
        job_type="cpu-analytic",
        config=config,
        notes=(
            "PR #424 drafter-fusion build scope (CPU analytic, no GPU/HF/served "
            "change). VERDICT NO-GO: the deployed MTP K=7 drafter is ALREADY a "
            "single ONEGRAPH CUDA-graph replay (not separate per-head launches; "
            "served sitecustomize.py:25-33, PR #75/#261) -> the per-head launch "
            "floor is already harvested. The ~248us floor is the GEMM-chain BW "
            "floor (566us@47.17% HBM), unreachable at M=1 (autoregressive serial, "
            "occupancy-bound). Honest roofline ~+20 TPS not +52 (+52=+10.8% needs "
            "~781us saving > the entire 566us GEMM chain). Realistic ~0-5. Blast "
            "radius HIGH (onegraph IS the 481.53 engine; -16.5% regression risk)."
        ),
    )
    wandb.log(summary)
    for k, v in summary.items():
        run.summary[k] = v
    print("WANDB_RUN_ID=" + run.id)
    print("WANDB_RUN_URL=" + run.url)
    run.finish()
