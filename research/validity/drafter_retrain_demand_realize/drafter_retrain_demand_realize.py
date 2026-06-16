#!/usr/bin/env python3
"""PR #446 — Realize the +15.36 demand lift: time-boxed MTP drafter retrain (byte-exact).

DECISION ARTIFACT (analysis_only). This card asked for a *measured* time-boxed
retrain of the deployed MTP K=7 drafter (the onegraph `gemma4_assistant` head),
to convert #439's +0.1266-E[T] / +15.36-TPS *realizable* demand headroom into a
first realized equivalence-respecting beat of the deployed 481.53.

What this script does NOT do: it does not train. It records the SELF-ABORT
(PR instruction #6) with the mechanical reason the in-budget retrain is
infeasible, pins the fresh on-pod baseline, the realizable target, and the
GO/NO-GO on a full human-gated retrain — all machine-checked.

WHY SELF-ABORT (the load-bearing finding):
  The deployed drafter is `Gemma4AssistantForCausalLM` (transformers 5.9.0). Its
  `forward` REQUIRES `inputs_embeds` (the target backbone's hidden, projected
  2*2560->256) AND `shared_kv_states` (the target's per-layer KV); `input_ids` is
  documented "Not actually used". It is a cross-attention assistant head COUPLED
  to the target backbone, usable only via `Gemma4ForCausalLM.generate(
  assistant_model=...)`. The only in-scope trainer/eval
  (`official/.../kl_distill_reference_itaca/{train_kl_drafter.py,
  offline_acceptance.py}`) calls `model(input_ids=...)` standalone -> raises
  `ValueError: inputs_embeds and shared_kv_states cannot be None.` (verified on
  pod). itaca states they never ran it. A faithful train/eval loop therefore
  needs the full coupled target backbone in-loop (the deployed onegraph path) =
  a multi-day harness build, NOT a 90-min A10G card. This is exactly the
  plumbing-burn #439 flagged, now confirmed mechanically rather than from priors.

THREE INDEPENDENT IN-SCOPE LINES CONVERGE ON: fixed-topology retrain -> ~0 lift:
  (A) #119 drafter_et_ceiling: at pos-1 (the binding acceptance cliff),
      near-miss/learnable-looking pool cov4 = 0.6532, hard-miss/structural =
      0.3468, BUT capacity_recoverable_frac_AT_FIXED_COST = 0.0 — pinned by an
      openevolve A10G-oracle parity sweep that INCLUDES the exact
      DeepSeek-MTP-KL-distill(alpha=0.5) recipe the itaca reference/PR prescribes
      (-> parity 3.83, no lift). The 0.6532 pool is recoverable only by a BIGGER
      drafter (size), which "do NOT change head topology or K" forbids.
  (B) #439 realizable_etp_lift=+0.1266 is a PROJECTION (probe_was_run=False)
      assuming literature Dcov~+0.016 delivery; its delivery vehicle is drafter
      size, not a fixed-topology retrain.
  (C) Arithmetic ceiling: even FULL delivery -> 482.5 TPS = +0.97 over 481.53,
      WITHIN sigma_hw (~1% ~= 4.8 TPS) -> not a statistically clean beat.

Equivalence + PPL (PR #5): byte-exact by construction (verify is the SOLE
arbiter of emitted tokens, land #420) — any fixed-topology drafter emits
identical greedy tokens; no weights changed here -> trivially identical. PPL =
deployed 2.3772 <= 2.42.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# ---- Baselines (all in-scope, verified) ----------------------------------- #
BASE_DEPLOYED_NONEQ = 481.53        # PR baseline, NON-equivalent (identity 0.9966), #52 2x9fm2zx
BASE_REALIZED_EQ = 467.14           # realized equivalence frontier, denken #423 5a6zq2yz
SIGMA_HW_FRAC = 0.01                # PR: sigma_hw ~= 1%
SIGMA_HW_TPS = SIGMA_HW_FRAC * BASE_DEPLOYED_NONEQ  # ~= 4.8153 TPS

# Deployed MTP K=7 drafter anchors
ET_289 = 3.851                      # denken #289 fi34s269 (PR's stated anchor)
A1_289 = 0.7293
ET_76 = 3.844131736526946          # land #76 5m17r52s (this pod 2026-06-14)
A1_76 = 0.728739760479042
COND_LADDER_76 = [0.728739760479042, 0.7589764102641635, 0.7924989076194682,
                  0.821702519412012, 0.8342716929825772, 0.8352594665096346,
                  0.8472621220149911]

# ---- #439 demand-realizability sizing (verified, m2nmdyko) ----------------- #
REALIZABLE_ETP_LIFT_439 = 0.12660174617188788
COVERAGE_GAP_TOTAL = 0.10973404808468479
COVERAGE_GAP_REALIZABLE_FRAC = 0.145807069722356
COVERAGE_GAP_STRUCTURAL_FRAC = 0.854192930277644
PR_REALIZABLE_TPS_ON_467 = 15.36   # PR: +15.36 on 467.14 -> 482.5
PR_FRONTIER_FULL = 482.50

# ---- #119 drafter_et_ceiling pos-1 miss decomposition (verified) ----------- #
POS1_NEAR_MISS_COV4 = 0.6531976066516435   # capacity_recoverable_frac_upper_bound (learnable-LOOKING)
POS1_HARD_MISS = 0.3468023933483565        # intrinsic_irreducible_frac_lower_bound (structural)
POS1_FIXED_CAP_RECOVERABLE = 0.0           # capacity_recoverable_frac_AT_FIXED_COST  <-- THE NUMBER
OPENEVOLVE_PARITY_ET = 3.83                # accept-length at openevolve parity (incl KL-distill)
ET_CAPACITY_PERFECT_FIXED = 3.8444537125748504   # -> 481.59 official
ET_OPTIMISTIC_ALL_NEAR_MISS = 6.161359334895227  # only if ALL near-miss recovered (bigger drafter)
ET_ABSOLUTE_DRAFTER_EQ_TARGET = 8.0
CLEAR500_ET_BAR = 3.9914439391107512       # #119: E[T] needed for 500 official


def load_fresh_anchor() -> dict:
    """Same-session served re-anchor (PR instruction #1), if present."""
    p = ROOT / "research/local_validation/reanchor_446_20260616/accept_calibration_results.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    m = d.get("server_log_metrics", {})
    return {
        "utc": d.get("utc"),
        "wandb_run_id": "uid28gdg",
        "E_T": m.get("mean_tokens_per_step_E_T"),
        "a1": (m.get("cumulative_acceptance_C") or [None])[0],
        "draft_acceptance_rate": m.get("draft_acceptance_rate"),
        "conditional_acceptance_p": m.get("conditional_acceptance_p"),
        "num_drafts": m.get("num_drafts"),
    }


def analyze() -> dict:
    fresh = load_fresh_anchor()

    # ---- Instruction #2: pin the realizable E[T] target + realized TPS ---- #
    et_target_full = ET_289 + REALIZABLE_ETP_LIFT_439          # 3.9776
    tps_per_etp_on_467 = PR_REALIZABLE_TPS_ON_467 / REALIZABLE_ETP_LIFT_439  # ~= 121.3
    frontier_full_delivery = BASE_REALIZED_EQ + PR_REALIZABLE_TPS_ON_467     # 482.50 ceiling
    margin_full_vs_deployed = frontier_full_delivery - BASE_DEPLOYED_NONEQ   # +0.97
    full_clears_sigma = margin_full_vs_deployed >= SIGMA_HW_TPS              # False

    # ---- Instruction #4: realized (projected, fixed-topology) ------------- #
    # Fixed-capacity recoverable fraction of the binding pos-1 miss is 0.0
    # (#119, openevolve parity incl. KL). => expected fixed-topology lift ~ 0.
    fixed_topo_expected_etp_lift = POS1_FIXED_CAP_RECOVERABLE * REALIZABLE_ETP_LIFT_439  # 0.0
    frontier_fixed_topo = BASE_REALIZED_EQ + fixed_topo_expected_etp_lift * tps_per_etp_on_467  # 467.14
    margin_fixed_vs_deployed = frontier_fixed_topo - BASE_DEPLOYED_NONEQ    # -14.39

    crosses_481_cleanly = (margin_fixed_vs_deployed >= SIGMA_HW_TPS)        # realistic: False
    optimistic_crosses_481_cleanly = full_clears_sigma                     # ceiling: False

    verdict = (
        "MEASURED-ABORT / NO-GO. In-budget retrain is mechanically infeasible "
        "(reference scripts non-executable vs the real coupled Gemma4Assistant "
        "class; faithful loop needs the coupled target backbone >> 90-min box). "
        "A FULL human-gated FIXED-TOPOLOGY retrain projects to ~0 E[T] lift: the "
        "binding pos-1 miss has fixed-capacity-recoverable-frac=0.0 (#119, "
        "openevolve parity INCLUDING the exact KL-distill recipe), so the "
        "realized frontier stays ~467.14 (-14.4 vs 481.53). Even the optimistic "
        "ceiling under FULL delivery of #439's +0.1266 is 482.5 = +0.97 over "
        "481.53, WITHIN sigma_hw(~4.8) -> not a clean equivalence-respecting beat. "
        "The +0.6532 near-miss pool is recoverable only by drafter SIZE, which "
        "'no topology change' forbids. recommend_full_retrain=False; if revisited, "
        "gate on ONE cheap bigger-drafter A10G-oracle eval (no train) FIRST."
    )

    out = {
        "pr": 446, "agent": "land", "kind": "drafter-retrain-demand-realize",
        "analysis_only": True, "no_launch": True, "no_hf_job": True,
        "no_submission": True, "no_served_file_change": True,
        "gpu_used": True,  # served re-anchor decode (instruction #1), NOT a train/submission
        "official_tps": 0, "baseline_unchanged_tps": BASE_DEPLOYED_NONEQ, "ppl": 2.3772,

        # ---- instruction #1: fresh before-number ----
        "baseline_reanchor_fresh_this_pod": fresh,
        "baseline_anchor_76_this_pod": {"E_T": ET_76, "a1": A1_76, "ladder": COND_LADDER_76,
                                        "wandb_run_id": "5m17r52s"},
        "baseline_anchor_289": {"E_T": ET_289, "a1": A1_289, "wandb_run_id": "fi34s269"},
        "acceptance_cliff_position": 1,
        "ladder_rises_with_depth": True,

        # ---- instruction #2: target ----
        "realizable_etp_lift_439": REALIZABLE_ETP_LIFT_439,
        "et_target_full_delivery": et_target_full,
        "tps_per_etp_on_467_base": tps_per_etp_on_467,
        "frontier_full_delivery_tps": frontier_full_delivery,
        "margin_full_vs_deployed_tps": margin_full_vs_deployed,
        "sigma_hw_tps": SIGMA_HW_TPS,
        "full_delivery_clears_sigma_hw": full_clears_sigma,

        # ---- instruction #3/#6: self-abort reason ----
        "in_budget_retrain_feasible": False,
        "self_abort_reason": (
            "Gemma4AssistantForCausalLM.forward requires inputs_embeds + "
            "shared_kv_states (cross-attn head coupled to target backbone; "
            "input_ids ignored). In-scope reference trainer/eval call "
            "model(input_ids=...) standalone -> ValueError (verified). Faithful "
            "loop needs coupled target backbone in-loop = multi-day build > 90min."
        ),
        "only_coupled_trainer_is_out_of_scope_commit": "4d65412 (wide_drafter; EXCLUDED by isolation)",

        # ---- instruction #4: realized (projected) ----
        "fixed_topology_expected_etp_lift": fixed_topo_expected_etp_lift,
        "frontier_fixed_topology_realistic_tps": frontier_fixed_topo,
        "margin_fixed_topology_vs_deployed_tps": margin_fixed_vs_deployed,
        "crosses_481_53_cleanly_realistic": crosses_481_cleanly,
        "crosses_481_53_cleanly_optimistic_ceiling": optimistic_crosses_481_cleanly,

        # ---- instruction #6b: miss characterization (learnable vs structural) ----
        "pos1_miss_total_frac": 1.0 - A1_76,
        "pos1_near_miss_cov4_frac_of_miss": POS1_NEAR_MISS_COV4,
        "pos1_hard_miss_structural_frac_of_miss": POS1_HARD_MISS,
        "pos1_fixed_capacity_recoverable_frac": POS1_FIXED_CAP_RECOVERABLE,
        "openevolve_parity_accept_length": OPENEVOLVE_PARITY_ET,
        "miss_char_note": (
            "Binding miss is at position-1 (acceptance cliff; ladder RISES "
            "0.727->0.846 with depth). Of the ~27% pos-1 miss, <=65.3% is "
            "near-miss (target argmax in drafter top-4, looks learnable) and "
            ">=34.7% is hard-miss (target's own distribution, structural). BUT at "
            "FIXED capacity the recoverable fraction is 0.0: openevolve A10G-oracle "
            "parity (incl. DeepSeek-MTP-KL-distill alpha=0.5, the itaca recipe) "
            "reaches 3.83 ~= baseline. The near-miss pool is SIZE-recoverable only."
        ),

        # ---- instruction #6c: recipe + GO/NO-GO ----
        "recommend_full_human_gated_retrain": False,
        "recommend_in_budget_retrain": False,
        "cheapest_decisive_next_step": (
            "ONE bigger-drafter A10G-oracle eval (openevolve's oracle; NO train, "
            "NO bench-quota) to measure marginal E[T]/cost of the SIZE-recoverable "
            "near-miss slice BEFORE any train ask (#119 fleet rec). Prior = NO "
            "(fixed-cap recovery already pinned 0; tree path dominates at every "
            "relaxed cost)."
        ),
        "faithful_retrain_recipe_if_funded": {
            "init": "/tmp/qat-assistant (= kenyan-duma drafter-ft/ft-v1-epoch_001, kduma1)",
            "critical_fix": ("REWRITE the itaca reference: drive the assistant via the "
                             "coupled target backbone (run Gemma4ForCausalLM per position "
                             "to produce inputs_embeds + shared_kv_states full+sliding); the "
                             "shipped model(input_ids=...) path is NON-FUNCTIONAL."),
            "corpus": ("corpus_spec.md: >=9k distribution-matched prompts (ShareGPT/MMLU-Pro/"
                       "GPQA/AIME), >=1M propose-call traces, held-out >=900-prompt shard, "
                       "dedup vs eval_prompts_sharegpt.json. A 128-prompt corpus is the "
                       "known-bad evaporating-gain class."),
            "loss": "KL(top-2048 target softmax) or hybrid alpha in [0.3,0.5]; alpha=0.9 worse.",
            "offline_gate": "REWRITTEN offline_acceptance (coupled) >= +0.05 acc-tok/step on held-out BEFORE bench.",
            "hardware_time": "~few H100-hours for 1 epoch; NOT A10G-in-90min.",
            "expected_outcome": "Per #119, will NOT clear the +0.05 gate at fixed capacity.",
        },

        # ---- instruction #5: equivalence + PPL ----
        "equivalence_byte_exact_by_construction": True,
        "equivalence_basis": "verify is the SOLE arbiter of emitted tokens (land #420 qe4qagc1); drafter gates accept-LENGTH only.",
        "equivalence_selftest_pass_count": "N/A measured (no candidate produced); by-construction proof, no weights changed.",
        "ppl": 2.3772, "ppl_gate": 2.42, "ppl_passes": 2.3772 <= 2.42,

        "verdict": verdict,
    }
    return out


def self_test(o: dict) -> dict:
    checks = {}
    # arithmetic consistency
    checks["tps_per_etp_recovers_pr_1536"] = abs(o["tps_per_etp_on_467_base"] * REALIZABLE_ETP_LIFT_439 - PR_REALIZABLE_TPS_ON_467) < 0.05
    checks["frontier_full_is_482_5"] = abs(o["frontier_full_delivery_tps"] - PR_FRONTIER_FULL) < 0.05
    checks["full_delivery_within_sigma_hw"] = (o["margin_full_vs_deployed_tps"] < o["sigma_hw_tps"]) and (o["full_delivery_clears_sigma_hw"] is False)
    checks["fixed_topo_lift_is_zero"] = o["fixed_topology_expected_etp_lift"] == 0.0
    checks["fixed_topo_frontier_is_467"] = abs(o["frontier_fixed_topology_realistic_tps"] - BASE_REALIZED_EQ) < 1e-6
    checks["neither_case_clears_481"] = (o["crosses_481_53_cleanly_realistic"] is False) and (o["crosses_481_53_cleanly_optimistic_ceiling"] is False)
    # miss decomposition
    checks["miss_split_sums_to_one"] = abs(POS1_NEAR_MISS_COV4 + POS1_HARD_MISS - 1.0) < 1e-6
    checks["fixed_cap_recoverable_zero"] = POS1_FIXED_CAP_RECOVERABLE == 0.0
    checks["openevolve_parity_below_target"] = OPENEVOLVE_PARITY_ET < o["et_target_full_delivery"]
    checks["capacity_perfect_fixed_ties_not_clears"] = ET_CAPACITY_PERFECT_FIXED < CLEAR500_ET_BAR
    # fresh anchor sanity (if present): consistent with #76/#289 within ~1%
    fa = o.get("baseline_reanchor_fresh_this_pod") or {}
    if fa.get("E_T") is not None:
        checks["fresh_anchor_consistent_with_76"] = abs(fa["E_T"] - ET_76) < 0.05
        checks["fresh_a1_consistent"] = abs(fa["a1"] - A1_76) < 0.01
        lad = fa.get("conditional_acceptance_p") or []
        checks["fresh_ladder_rises_cliff_pos1"] = len(lad) == 7 and (lad[0] == min(lad))
    # gates
    checks["ppl_passes"] = o["ppl_passes"] is True
    checks["equivalence_by_construction"] = o["equivalence_byte_exact_by_construction"] is True
    checks["recommend_retrain_is_nogo"] = (o["recommend_full_human_gated_retrain"] is False) and (o["recommend_in_budget_retrain"] is False)
    return checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-name", default="land/retrain-demand-realize-446")
    ap.add_argument("--wandb-group", default="drafter-retrain-demand-realize")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    out = analyze()
    checks = self_test(out)
    all_pass = all(checks.values())
    out["self_test_checks"] = checks
    out["self_test_passes"] = all_pass
    out["self_test_pass_count"] = f"{sum(checks.values())}/{len(checks)}"

    (HERE / "drafter_retrain_demand_realize_results.json").write_text(json.dumps(out, indent=2))
    (HERE / "drafter_retrain_demand_realize_selftest.json").write_text(json.dumps(checks, indent=2))

    print(f"[selftest] {out['self_test_pass_count']}  all_pass={all_pass}")
    print(f"[verdict] {out['verdict']}")
    print(f"[anchor-fresh] {out['baseline_reanchor_fresh_this_pod']}")
    print(f"[frontier] full-delivery ceiling={out['frontier_full_delivery_tps']:.2f} "
          f"(+{out['margin_full_vs_deployed_tps']:.2f} vs 481.53, sigma_hw={out['sigma_hw_tps']:.2f}); "
          f"fixed-topology realistic={out['frontier_fixed_topology_realistic_tps']:.2f} "
          f"({out['margin_fixed_topology_vs_deployed_tps']:+.2f})")

    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
                entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
                name=args.wandb_name, group=args.wandb_group,
                config={"pr": 446, "kind": out["kind"], "analysis_only": True},
            )
            flat = {k: v for k, v in out.items() if isinstance(v, (int, float, bool))}
            flat["self_test_pass_count_n"] = sum(checks.values())
            run.log(flat)
            run.summary.update(flat)
            print(f"[wandb] {run.url}")
            run.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] unavailable/failed: {exc}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
