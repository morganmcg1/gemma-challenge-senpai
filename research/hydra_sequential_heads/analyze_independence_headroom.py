#!/usr/bin/env python
"""PR #115 -- Hydra sequential-MTP-head headroom analysis (CPU-only, no GPU).

Question (advisor): the M=8 linear-MTP drafter is at E[T]=3.844. Does
Hydra-style *sequential head-conditioning* (Cai et al. 2024) -- where each draft
head conditions on the previously-drafted token instead of predicting
independently from the base hidden -- have enough headroom to lift E[T] past the
#106 milestones (4.45 beat-linear / 4.62 clear-500 / 4.7 tree-overtakes)?

PRIMARY metric  : independence_attributable_reject_frac
TEST metric     : et_ceiling_sequential_conditioning
Gate            : frac < 0.10 -> KILL ; frac > 0.25 -> green-light light-GPU prototype.

================================  VERDICT  ================================
The premise is architecturally void. The deployed drafter
(google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant) is NOT a set of
independent heads -- vLLM resolves it to `Gemma4MTP`, a single recurrent 4-layer
MTP module run once per draft step, and the proposer feeds the
**previously-drafted token** + the fed-back backbone hidden into every step. That
IS Hydra-style sequential conditioning, already shipped. So a Hydra head recovers
nothing the deployed head misses:

    independence_attributable_reject_frac = 0.0  ->  gate KILL (<0.10)
    et_ceiling_sequential_conditioning   = E[T] = 3.844  (no movement)
    -> does NOT reach 4.45 / 4.62 / 4.7 ; tree lane stays AMBER on this lever.

This is a no-GPU, definitional/architectural result -- *stronger* than an offline
sim, which the public oracle (openevolve, 2026-06-14) showed OVER-reports for
trained drafters. The genuine ceiling is model uncertainty + the small drafter's
capacity (256-d, 4-layer, KV-shared Q-only), not an independence assumption.

Run:  python research/hydra_sequential_heads/analyze_independence_headroom.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ACCEPT_JSON = ROOT / "research/accept_calibration/accept_calibration_results.json"
OUT_JSON = ROOT / "research/hydra_sequential_heads/independence_headroom_results.json"

# #106 / crossover E[T] milestones the headroom is judged against.
MILESTONES = {
    "beat_linear_481p53": 4.45,
    "clear_500": 4.62,
    "tree_overtakes_treefree": 4.7,
    "stretch_563": 5.207,
}

# Architectural evidence: file:line citations proving the deployed MTP drafter is
# a recurrent sequential proposer (each draft token conditions on the previously
# drafted token), i.e. the Hydra mechanism is already implemented.
ARCH_EVIDENCE = {
    "drafter_model": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
    "vllm_class": "Gemma4MTP (model_executor/models/gemma4_mtp.py) "
    "served by Gemma4Proposer (v1/spec_decode/gemma4.py)",
    "single_recurrent_module": "Gemma4MultiTokenPredictor: num_mtp_layers == "
    "text_config.num_hidden_layers == 4 decoder layers, run ONCE PER DRAFT STEP "
    "(proposer docstring: 'runs all decoder layers per draft step (producing one "
    "token)'). Not K independent heads.",
    "conditions_on_prev_token_embed": "gemma4_mtp.py:463  combined = "
    "torch.cat([inputs_embeds, hidden_states], dim=-1)  -- input is "
    "cat(embed(input_token), prev_hidden).",
    "feeds_back_prev_drafted_token": "llm_base_proposer.py:574  input_ids = "
    "draft_token_ids_list[-1].int()  -- the token DRAFTED at step k-1 is the "
    "input_ids for step k.",
    "feeds_back_recurrent_hidden": "llm_base_proposer.py:597,613-614  "
    "model_kwargs['hidden_states'] = prev step's backbone_hidden (gemma4_mtp.py:"
    "476-477 returns it).",
    "greedy_verify_accept_rule": "serve.py:7-10  temperature=0 -> vLLM rejection "
    "sampler short-circuits to target-argmax; accept iff draft_token == target "
    "argmax, so every ACCEPTED prefix token is target-correct.",
    "medusa_is_the_independent_lane": "medusa.py:propose  draft_tokens = stack("
    "[logit.argmax(-1) for logit in self.model.compute_logits(self.model("
    "target_hidden_states))]) -- K heads from the SAME target_hidden_states, NO "
    "drafted-token feedback. THIS is the independent-heads arch Hydra fixes; the "
    "deployed submission does NOT use it.",
}

# Public evidence (challenge board) that independently confirms the ceiling.
PUBLIC_EVIDENCE = {
    "openevolve_2026-06-14T02:38Z": "message_board/20260614-023846-928_openevolve.md "
    "-- A10G-oracle: EVERY retrained drafter (CE, recipe sweeps, faithful "
    "vLLM-hidden capture, and itaca's DeepSeek-MTP KL-distillation a in {0.5,0.9}) "
    "lands at PARITY ~3.83 accept_length; 'e1 appears to be at the architecture's "
    "acceptance ceiling for this workload'. Offline/HF accept screens OVER-report "
    "for trained drafters (anti-correlated with oracle off-baseline).",
    "leaderboard_2026-06-14": "top public ~489 TPS (frantic-penguin "
    "lmhead12k-fa2sw-precache-skv64); frontier-to-beat PR#52 fa2sw_precache_kenyan "
    "= 481.53 official.",
    "crossover_106": "tree_vs_treefree_crossover -> AMBER: tree overtakes tree-free "
    "only if realized E[T] >= 4.791 (alone)/4.583 (with splitk+lk); headroom is in "
    "the tree-verify build, not drafter conditioning.",
}


def compute_et(cumulative_C: list[float]) -> float:
    """E[T] = 1 + sum_k C[k], C[k] = P(first k draft tokens all accepted)."""
    return 1.0 + sum(cumulative_C)


def reject_decomposition(q_cond: list[float], cum_C: list[float]) -> dict:
    """First-rejection-by-position over the K=7 linear chain.

    For the chain to REACH draft position k, tokens 1..k-1 were accepted, hence
    target-correct (greedy verify). The deployed recurrent drafter is fed exactly
    that accepted (correct) prefix token at step k -- the SAME conditioning a
    Hydra head would get. So no rejection is 'independence-attributable':
    independence_attributable_reject_frac == 0 by construction.
    """
    K = len(q_cond)
    C_prev = [1.0] + list(cum_C)  # C_prev[k] = P(reach position k+1) = C[k]
    first_reject_at = []
    for k in range(K):
        reach = C_prev[k]          # P(reach position k+1)
        first_reject_at.append(reach * (1.0 - q_cond[k]))
    total_reject = sum(first_reject_at)
    full_accept = cum_C[-1]        # all K accepted, no reject in window

    # Every chain rejection occurs AFTER an accepted (==correct) prefix, so the
    # conditioning input at the reject position is already the correct token.
    # Position 1 is the strongest illustration: it is fed the real VERIFIED token
    # + real target hidden (oracle conditioning) yet misses 1-q0 of the time.
    pos1_share = first_reject_at[0] / total_reject if total_reject else 0.0
    return {
        "first_reject_prob_by_position": first_reject_at,
        "total_reject_prob": total_reject,
        "full_window_accept_prob": full_accept,
        "sanity_sum": total_reject + full_accept,
        "pos1_oracle_conditioned_reject_share": pos1_share,
        "q0_first_position_acceptance": q_cond[0],
        "pos1_miss_is_genuine_uncertainty": 1.0 - q_cond[0],
        "independence_attributable_reject_frac": 0.0,
        "rationale": "Every accepted prefix token == target argmax (greedy "
        "verify), and the deployed recurrent MTP feeds that accepted token "
        "(embed + recurrent hidden) into the next step. A Hydra head conditioning "
        "on 'head-(k-1)'s accepted token' sees the identical correct token, so it "
        "recovers no rejection the deployed head misses.",
    }


def main() -> None:
    accept = json.loads(ACCEPT_JSON.read_text())
    head = accept["headline"]
    q_cond = head["conditional_acceptance_p"]
    cum_C = head["cumulative_acceptance_C"]
    K = accept["server_log_metrics"]["num_speculative_tokens"]

    et = compute_et(cum_C)
    et_reported = accept["primary_metric"]["value"]

    decomp = reject_decomposition(q_cond, cum_C)
    frac = decomp["independence_attributable_reject_frac"]

    # Perfect recovery of an empty (measure-zero) set of rejections does not move
    # E[T]; the ceiling under sequential conditioning is the current E[T].
    et_ceiling = et_reported

    milestone_reach = {
        name: {"target": tgt, "reached": et_ceiling >= tgt}
        for name, tgt in MILESTONES.items()
    }

    if frac < 0.10:
        gate = "KILL"
        gate_reason = (
            "independence_attributable_reject_frac=0.0 < 0.10. The deployed "
            "Gemma4MTP drafter is ALREADY a recurrent sequential proposer "
            "(conditions each draft token on the previously-drafted token); the "
            "Hydra lever is already implemented, so it has zero headroom. No "
            "light-GPU prototype is warranted."
        )
    elif frac > 0.25:
        gate = "GREEN_LIGHT_PROTOTYPE"
        gate_reason = "frac > 0.25 -- real headroom."
    else:
        gate = "AMBER"
        gate_reason = "0.10 <= frac <= 0.25 -- ambiguous."

    results = {
        "pr": 115,
        "hypothesis": "Hydra sequential MTP head-conditioning to break E[T]=3.844.",
        "method": "CPU-only architectural + definitional analysis of the deployed "
        "vLLM Gemma4MTP proposer, anchored to the server-log per-position "
        "acceptance ladder (accept_calibration). No GPU, no served change, greedy "
        "identity untouched.",
        "inputs": {
            "num_speculative_tokens_K": K,
            "conditional_acceptance_p": q_cond,
            "cumulative_acceptance_C": cum_C,
            "E_T_recomputed_from_C": et,
            "E_T_reported": et_reported,
        },
        "architecture_evidence": ARCH_EVIDENCE,
        "public_evidence": PUBLIC_EVIDENCE,
        "reject_decomposition": decomp,
        "milestones": MILESTONES,
        "verdict": {
            "primary_metric_name": "independence_attributable_reject_frac",
            "independence_attributable_reject_frac": frac,
            "test_metric_name": "et_ceiling_sequential_conditioning",
            "et_ceiling_sequential_conditioning": et_ceiling,
            "milestone_reach": milestone_reach,
            "reaches_4p7_tree_overtakes": et_ceiling >= MILESTONES["tree_overtakes_treefree"],
            "gate": gate,
            "gate_reason": gate_reason,
            "tree_lane_status_on_this_lever": "AMBER (unchanged) -- upper bound "
            "3.844 < 4.7; sequential conditioning cannot move E[T].",
            "true_binding_constraint": "Drafter capacity + genuine model "
            "uncertainty. Even at draft position 1 (oracle conditioning: real "
            "verified token + real target hidden) the drafter accepts only "
            f"q0={q_cond[0]:.4f}; the {1.0 - q_cond[0]:.4f} miss is immune to "
            "sequential conditioning. Headroom is in tree-verify (#106), not Hydra.",
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"[hydra] wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"[hydra] E[T] recomputed={et:.6f} (reported {et_reported:.6f})")
    print(f"[hydra] PRIMARY independence_attributable_reject_frac = {frac}")
    print(f"[hydra] TEST    et_ceiling_sequential_conditioning   = {et_ceiling:.6f}")
    print(f"[hydra] reaches 4.7 (tree-overtakes)? "
          f"{results['verdict']['reaches_4p7_tree_overtakes']}")
    print(f"[hydra] GATE = {gate}")

    # ---- W&B logging (rich audit trail) ----
    try:
        from scripts.wandb_logging import (
            finish_wandb,
            init_wandb_run,
            log_event,
            log_json_artifact,
        )

        run = init_wandb_run(
            job_type="analysis",
            agent="fern",
            name="fern/hydra-sequential-heads-headroom",
            group="hydra-sequential-heads",
            notes="PR#115 Hydra sequential-MTP headroom: KILL (drafter already "
            "recurrent/sequential; independence_attributable_reject_frac=0).",
            tags=["pr115", "drafter", "mtp", "hydra", "headroom", "kill"],
            config={
                "pr": 115,
                "num_speculative_tokens_K": K,
                "q0_first_position_acceptance": q_cond[0],
                "milestones": MILESTONES,
            },
        )
        if run is not None:
            v = results["verdict"]
            log_event(
                run,
                "headroom_verdict",
                step=0,
                metrics={
                    "metric/independence_attributable_reject_frac": frac,
                    "metric/et_ceiling_sequential_conditioning": et_ceiling,
                    "metric/E_T_current": et_reported,
                    "metric/q0_first_position_acceptance": q_cond[0],
                    "metric/pos1_oracle_reject_share": decomp[
                        "pos1_oracle_conditioned_reject_share"
                    ],
                    "metric/reaches_4p7": int(v["reaches_4p7_tree_overtakes"]),
                    "metric/gate_kill": int(v["gate"] == "KILL"),
                },
                data={"gate": v["gate"], "gate_reason": v["gate_reason"]},
            )
            for name, tgt in MILESTONES.items():
                run.summary[f"milestone/{name}_target"] = tgt
                run.summary[f"milestone/{name}_reached"] = et_ceiling >= tgt
            run.summary["verdict/gate"] = v["gate"]
            run.summary["verdict/independence_attributable_reject_frac"] = frac
            run.summary["verdict/et_ceiling_sequential_conditioning"] = et_ceiling
            log_json_artifact(
                run,
                name="hydra-independence-headroom",
                artifact_type="analysis",
                data=results,
            )
            print(f"[hydra] wandb run: {run.id}")
            results["wandb_run_id"] = run.id
            OUT_JSON.write_text(json.dumps(results, indent=2))
            finish_wandb(run)
    except Exception as exc:  # noqa: BLE001 -- logging must never fail the analysis
        print(f"[hydra] wandb logging skipped: {exc}")


if __name__ == "__main__":
    main()
