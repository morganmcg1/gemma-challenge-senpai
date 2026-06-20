#!/usr/bin/env python
"""Log the PR #785 surgattn 2D-vs-3D overhead comparison to W&B.

Single run in group ``bi0-surgattn-overhead`` capturing both arms (control =
surgattn ON / force-2D; variant = surgattn OFF / kernel gate picks 3D split-KV on
M=1 forwards), the greedy-identity comparison, PPL, and the spec-decode
acceptance stats.
"""
from __future__ import annotations

import json
from pathlib import Path

import wandb

BASE = Path("research/validity/bi_detax_surgical_attn/surgattn_overhead_785")


def jload(p):
    return json.loads((BASE / p).read_text())


def main() -> None:
    ctrl = jload("control/local_summary.json")
    var = jload("variant/local_summary.json")
    gi = jload("greedy_identity.json")
    acc = jload("accept_stats.json")

    c_tps, v_tps = ctrl["tps"], var["tps"]
    tps_delta_pct = 100.0 * (v_tps - c_tps) / c_tps

    run = wandb.init(
        entity="wandb-applied-ai-team",
        project="gemma-challenge-senpai",
        name="wirbel/surgattn-overhead-2d-vs-3d",
        group="bi0-surgattn-overhead",
        job_type="local-prevalidate",
        config={
            "pr": 785,
            "submission": "int4_mtp_bi0_surgattn",
            "hypothesis": "does force-2D (surgattn) cost TPS vs the 3D split-KV path on bi0?",
            "control": "surgattn ON (VLLM_SURGATTN unset -> force-2D, use_3d=False everywhere)",
            "variant": "surgattn OFF (VLLM_SURGATTN=0 -> kernel gate picks 3D split-KV on M=1)",
            "vllm_batch_invariant": 0,
            "num_speculative_tokens": 6,
            "decode_num_prompts": 32,
            "decode_output_len": 512,
            "ppl_records": 128,
            "hardware": "local A10G (sm_86), exploratory single-stream proxy (NOT official a10g)",
            "use_3d_fires_on_M1": True,
            "use_3d_on_M_gt_1": False,
            "seq_threshold_3D": 7,
            "local_env_fix": "symlinked nvidia-cu13 curand*.h into /usr/local/cuda/include for flashinfer JIT (submission untouched)",
        },
    )

    summary = {
        # speed (local proxy)
        "control_tps": c_tps,
        "variant_tps": v_tps,
        "tps_delta_pct": tps_delta_pct,
        "control_duration_s": ctrl.get("decode_duration_s"),
        "variant_duration_s": var.get("decode_duration_s"),
        # quality / gate
        "control_ppl": ctrl["ppl"],
        "variant_ppl": var["ppl"],
        "ppl_gate_threshold": 2.42,
        "control_completed": ctrl["completed"],
        "variant_completed": var["completed"],
        # greedy identity
        "greedy_byte_exact": gi["byte_exact_greedy_identity"],
        "greedy_identical_records": gi["identical_records"],
        "greedy_diverging_records": gi["diverging_records"],
        "greedy_n_common": gi["n_common"],
        "greedy_record_divergence_rate": gi["record_divergence_rate"],
        "greedy_token_divergence_rate": gi["token_divergence_rate"],
        "greedy_total_token_diffs": gi["total_token_diffs"],
        # acceptance
        "control_mean_accept_len": acc["control"]["mean_accept_len_avg"],
        "variant_mean_accept_len": acc["variant"]["mean_accept_len_avg"],
        "control_overall_draft_accept_pct": acc["control"]["overall_draft_accept_rate_pct"],
        "variant_overall_draft_accept_pct": acc["variant"]["overall_draft_accept_rate_pct"],
        # gates (PR #785)
        "gate_tps_variant_gt_control": v_tps > c_tps,
        "gate_greedy_identity_pass": gi["byte_exact_greedy_identity"],
        "gate_ppl_pass": var["ppl"] <= 2.42,
        # verdict
        "verdict": "surgattn LOAD-BEARING for greedy identity; 3D fires on M=1 and breaks byte-exact greedy identity; +TPS local not free (gated on downstream-quality re-validation)",
    }
    run.summary.update(summary)

    # Per-arm comparison table
    t = wandb.Table(columns=[
        "arm", "surgattn", "m1_attn_path", "tps", "ppl", "completed",
        "duration_s", "mean_accept_len", "overall_draft_accept_pct",
    ])
    t.add_data("control", "ON (force-2D)", "2D single-pass", c_tps, ctrl["ppl"],
               ctrl["completed"], ctrl.get("decode_duration_s"),
               acc["control"]["mean_accept_len_avg"], acc["control"]["overall_draft_accept_rate_pct"])
    t.add_data("variant", "OFF", "3D split-KV", v_tps, var["ppl"],
               var["completed"], var.get("decode_duration_s"),
               acc["variant"]["mean_accept_len_avg"], acc["variant"]["overall_draft_accept_rate_pct"])
    run.log({"arm_comparison": t})

    # Diverging-records detail table
    dd = wandb.Table(columns=["id", "first_divergence_pos", "len_control", "len_variant", "n_token_diffs"])
    for r in gi["diverging_detail"]:
        dd.add_data(r["id"], r["first_divergence_pos"], r["len_control"], r["len_variant"], r["n_token_diffs"])
    run.log({"diverging_records": dd})

    print("WANDB_RUN_ID=" + run.id)
    print("WANDB_RUN_URL=" + run.url)
    run.finish()


if __name__ == "__main__":
    main()
