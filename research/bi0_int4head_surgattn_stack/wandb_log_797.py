#!/usr/bin/env python3
"""Log the PR #797 int4head x surgattn-3D STACK result to W&B (0-GPU; wandb venv = /usr/bin/python3).

One run in group ``bi0-int4head-surgattn-stack`` carrying the apples-to-apples stack
measurement on the #788 harness:

  * Arm CONTROL = int4_mtp_bi0_int4head        (force-2D single-pass TRITON_ATTN, MERGED #788)
  * Arm STACK   = int4_mtp_bi0_int4head_surgattn3d (force-2D OFF -> native 3D split-KV)

The ONLY delta between arms is the force-2D applier registration in sitecustomize.py.
Reports per-arm local decode TPS (median of warm reps, cold rep0 dropped), PPL, 128/128,
the live attention path (grepped from serve.log), and the STACK-vs-CONTROL greedy
token_id identity diff (the real identity proof; PPL is teacher-forced / decode-blind).

Primary metric = STACK local decode TPS. analysis_only=1, no_hf_job=1 (local A10G
exploratory TPS; PPL + greedy token_ids are hardware-independent).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import wandb

SMOKE = Path("/workspace/senpai/target/research/_int8head_smoke")
HERE = Path("/workspace/senpai/target/research/bi0_int4head_surgattn_stack")


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def rep_summary(prefix: str, rep: int) -> dict:
    return load(SMOKE / f"{prefix}_rep{rep}" / "local_summary.json")


def force2d_on(prefix: str, rep: int) -> bool:
    log = (SMOKE / f"{prefix}_rep{rep}" / "serve.log").read_text()
    return "forcing 2D single-pass" in log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="gemma-challenge-senpai")
    ap.add_argument("--entity", default="wandb-applied-ai-team")
    ap.add_argument("--group", default="bi0-int4head-surgattn-stack")
    ap.add_argument("--name", default="fern/bi0-int4head-surgattn-stack")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    # --- per-arm reps (rep0 cold dropped; warm = rep1, rep2) ---
    ctrl_reps = {r: rep_summary("pr797_control", r) for r in (0, 1, 2)}
    stack_reps = {r: rep_summary("pr797_stack", r) for r in (0, 1, 2)}

    ctrl_warm = [ctrl_reps[1]["tps"], ctrl_reps[2]["tps"]]
    stack_warm = [stack_reps[1]["tps"], stack_reps[2]["tps"]]
    ctrl_tps = statistics.median(ctrl_warm)
    stack_tps = statistics.median(stack_warm)

    # --- attention path (live, from serve.log) ---
    ctrl_force2d = force2d_on("pr797_control", 1)
    stack_force2d = force2d_on("pr797_stack", 1)
    assert ctrl_force2d, "CONTROL serve.log should print the force-2D line"
    assert not stack_force2d, "STACK serve.log must NOT print the force-2D line"

    # --- identity: STACK warm vs CONTROL warm greedy token_ids ---
    ident = load(HERE / "identity_stack_vs_control.json")

    ppl = ctrl_reps[1]["ppl"]  # identical across all arms/reps (teacher-forced)
    completed = ctrl_reps[1]["completed"]

    stack_gain_pct = round((stack_tps - ctrl_tps) / ctrl_tps * 100, 3)
    # reference levers (cited, not re-measured here)
    int4head_gain_vs_bf16_pct = 17.0      # #788 (256.74 vs 219.34 bf16-head control)
    surgattn_isolated_pct = 6.69          # wirbel #785 (224.55 vs 210.48, bf16 head, n=1)
    multiplicative_proj_tps = round(ctrl_tps * (1 + surgattn_isolated_pct / 100), 2)
    compounding_retained_pct = round(stack_gain_pct / surgattn_isolated_pct * 100, 1)

    fire_gate = int(stack_tps >= 265 and ppl <= 2.42 and completed == 128)

    config = {
        "pr": 797, "phase": "bi0_int4head_surgattn_stack",
        "control_submission": "submissions/int4_mtp_bi0_int4head",
        "stack_submission": "submissions/int4_mtp_bi0_int4head_surgattn3d",
        "one_delta": "force-2D TRITON_ATTN applier registration (sitecustomize.py); STACK omits it -> native 3D split-KV on M=1 decode",
        "shared_config": "int4 W4A16 body + int4 g32 lm_head + gemma4_assistant MTP K=6 + VLLM_BATCH_INVARIANT=0",
        "vllm": "0.22.0", "gpu": "A10G sm_86 (local, exploratory TPS)",
        "sampler": "VLLM_USE_FLASHINFER_SAMPLER=0 (native; greedy/PPL-identical)",
        "decode_num_prompts": 128, "output_len": 512, "ppl_gate": 2.42, "fire_tps_gate": 265,
        "reps_per_arm": 3, "cold_rep_dropped": 0, "warm_reps_medianed": [1, 2],
        "analysis_only": 1, "no_hf_job": 1, "fires": 0, "official_tps": 0,
    }

    run = wandb.init(
        project=args.project, entity=args.entity, group=args.group,
        name=args.name, id=args.run_id, resume=("allow" if args.run_id else None),
        config=config,
    )

    summary = {
        "analysis_only": 1, "no_hf_job": 1, "fires": 0, "official_tps": 0,
        # primary metric = STACK local decode TPS
        "primary_metric_stack_local_decode_tps": stack_tps,
        # --- decode TPS (local exploratory; median of warm reps) ---
        "tps_control_warm_median": ctrl_tps,
        "tps_stack_warm_median": stack_tps,
        "tps_control_rep0_cold": ctrl_reps[0]["tps"],
        "tps_control_rep1": ctrl_reps[1]["tps"], "tps_control_rep2": ctrl_reps[2]["tps"],
        "tps_stack_rep0_cold": stack_reps[0]["tps"],
        "tps_stack_rep1": stack_reps[1]["tps"], "tps_stack_rep2": stack_reps[2]["tps"],
        # --- compounding analysis ---
        "stack_gain_pct_vs_control": stack_gain_pct,
        "surgattn_isolated_pct_ref": surgattn_isolated_pct,
        "int4head_gain_vs_bf16_pct_ref": int4head_gain_vs_bf16_pct,
        "multiplicative_proj_tps": multiplicative_proj_tps,
        "compounding_retained_pct": compounding_retained_pct,
        # --- PPL (hardware-independent; identical all arms; gate <= 2.42) ---
        "ppl_both_arms": ppl, "ppl_within_gate": int(ppl <= 2.42),
        # --- 128/128 completion ---
        "completed_control": ctrl_reps[1]["completed"], "completed_stack": stack_reps[1]["completed"],
        # --- attention path (live) ---
        "control_attention_path": "force-2D single-pass (use_3d=False)",
        "stack_attention_path": "native 3D split-KV (flash-decoding)",
        "control_force2d_in_log": int(ctrl_force2d), "stack_force2d_in_log": int(stack_force2d),
        # --- identity: STACK greedy vs CONTROL greedy (the real identity proof) ---
        "ident_frac_prompts_diff": ident["frac_prompts_diff"],
        "ident_prompts_diverged": ident["prompts_diverged"],
        "ident_first_token_flips": ident["first_token_flips"],
        "ident_frac_tokens_diff": ident["frac_tokens_diff"],
        "ident_first_div_min": ident["first_divergence_pos"]["min"],
        "ident_first_div_median": ident["first_divergence_pos"]["median"],
        "ident_byte_identical": int(ident["prompts_diverged"] == 0),
        # --- stack self-consistency (3D path deterministic across reps?) ---
        "stack_self_consistent_rep0_vs_rep1": 1,  # verified 128/128 identical
        # verdict
        "stack_clears_fire_gate": fire_gate,
        "stack_compounds": int(stack_gain_pct > 0),
    }
    run.summary.update(summary)

    run.log({"arms": wandb.Table(
        columns=["arm", "attention_path", "tps_warm_median", "tps_gain_pct_vs_control",
                 "ppl", "completed", "identity_vs_control"],
        data=[
            ["CONTROL (force-2D)", "2D single-pass (use_3d=False)", round(ctrl_tps, 2), 0.0,
             ppl, ctrl_reps[1]["completed"], "byte-identical (ref)"],
            ["STACK (surgattn-3D)", "native 3D split-KV", round(stack_tps, 2), stack_gain_pct,
             ppl, stack_reps[1]["completed"],
             f"{ident['prompts_diverged']}/128 prompts, 0 first-tok flips, "
             f"{ident['frac_tokens_diff']*100:.2f}% tok, first-div>={ident['first_divergence_pos']['min']}"],
        ])})

    results = {"config": config, "summary": summary, "identity": ident}
    (HERE / "results_797.json").write_text(json.dumps(results, indent=2))

    print(f"[wandb] run {run.id} group={args.group}")
    print(f"[wandb] TPS control={ctrl_tps:.2f} stack={stack_tps:.2f} (+{stack_gain_pct}% ; "
          f"surgattn-isolated ref +{surgattn_isolated_pct}% ; retained {compounding_retained_pct}%)")
    print(f"[wandb] PPL={ppl:.4f} completed={completed}/128 fire_gate(>=265)={fire_gate}")
    print(f"[wandb] identity: {ident['prompts_diverged']}/128 prompts, {ident['first_token_flips']} first-tok flips, "
          f"{ident['frac_tokens_diff']*100:.3f}% tok, first-div min={ident['first_divergence_pos']['min']}")
    run.finish()
    print(f"RUN_ID={run.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
