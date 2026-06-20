"""PR #794 — log microbench + e2e results to W&B (group bi0-surgattn-attrib).

The microbench and end2end harnesses run in the vllm022 serve venv (no wandb);
this is the separate uv-env W&B step that picks up their JSON and leaves a rich
record: mechanism findings, per-layer numerics, timing, and the 2D-vs-3D e2e
greedy-compare + TPS. Verdict scalars summarize the clean null.

    uv run --no-sync python research/bi0_surgattn_attrib_combine/log_results_wandb.py
"""
from __future__ import annotations

import json
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
MICRO = json.loads((HERE / "microbench_results.json").read_text())
E2E = json.loads((HERE / "e2e" / "results.json").read_text())
_DETERM_PATH = HERE / "e2e" / "determinism_control.json"
DETERM = json.loads(_DETERM_PATH.read_text()) if _DETERM_PATH.exists() else None


def decode_tps(arm: dict) -> float:
    d = arm["decode_summary"]
    return d["num_completion_tokens"] / d["duration_s"]


def main() -> None:
    a2, a3 = E2E["arms"]["2d"], E2E["arms"]["3d"]
    wall_2d, wall_3d = decode_tps(a2), decode_tps(a3)
    gc, td = E2E["greedy_compare"], E2E["tps_delta"]

    run = wandb.init(
        entity="wandb-applied-ai-team",
        project="gemma-challenge-senpai",
        group="bi0-surgattn-attrib",
        name="stark/surgattn-combine-results",
        job_type="analysis",
        config={
            "pr": 794,
            "base_submission": "int4_mtp_bi0_surgattn",
            "vllm": "0.22.0",
            "device": MICRO["device"],
            "seq_threshold_3D": MICRO["seq_threshold_3D"],
            "num_par_softmax_segments": MICRO["num_par_softmax_segments"],
            # mechanism
            "finding_partials_dtype": "float32 (config-a structural no-op)",
            "finding_divergence_source": "reduction reassociation (cross-segment combine + TILE_DECODE 16 vs TILE_PREFILL 32)",
            "config_a_fp32_accumulator": "no-op (already fp32; bf16 partials diverge MORE)",
            "config_b_deterministic_order": "= serialize combine = remove split = remove occupancy win",
            "verdict": "CLEAN_NULL: +speedup inseparable from reassociation; official greedy gate is byte-exact (no tie-tolerance)",
            # baselines
            "bi0_official_tps": 218.02,
            "bi0_official_ppl": 2.0058,
            "wirbel_785_3d_tps": 224.55,
            "wirbel_785_tok_divergence": 0.0176,
            "wirbel_785_prompt_divergence": 0.0625,
        },
    )

    # ---- microbench numerics table (per layer/ctx) ----
    num_cols = ["layer_type", "ctx", "head_dim",
                "frac_differ_3dfp32_bf16", "frac_differ_3dbf16_bf16",
                "frac_differ_3dfp32_fp32", "max_abs_3dfp32_bf16",
                "mean_abs_err_2d_vs_sdpa", "mean_abs_err_3dfp32_vs_sdpa"]
    num_tbl = wandb.Table(columns=num_cols)
    bf16_worse = 0
    fp32_full_reassoc = 0
    sdpa_3d_le_2d = 0
    for n in MICRO["numerics"]:
        f_fp32 = n["d_2d_vs_3dfp32_bf16"]["frac_differ"]
        f_bf16 = n["d_2d_vs_3dbf16_bf16"]["frac_differ"]
        f_fp32fp32 = n["d_2d_vs_3dfp32_fp32"]["frac_differ"]
        e2d = n["err_2d_vs_sdpa"]["mean_abs"]
        e3d = n["err_3dfp32_vs_sdpa"]["mean_abs"]
        bf16_worse += int(f_bf16 > f_fp32)
        fp32_full_reassoc += int(f_fp32fp32 >= 0.999)
        sdpa_3d_le_2d += int(e3d <= e2d)
        num_tbl.add_data(n["layer_type"], n["ctx"], n["head_dim"],
                         f_fp32, f_bf16, f_fp32fp32,
                         n["d_2d_vs_3dfp32_bf16"]["max_abs"], e2d, e3d)

    # ---- microbench timing table (per layer/ctx) ----
    tim_cols = ["layer_type", "ctx", "us_2d", "us_3d", "speedup_2d_over_3d",
                "used_3d", "gbps_2d", "gbps_3d"]
    tim_tbl = wandb.Table(columns=tim_cols)
    all_used_3d = True
    for t in MICRO["timing"]:
        all_used_3d &= bool(t["used_3d"])
        tim_tbl.add_data(t["layer_type"], t["ctx"], t["us_2d"], t["us_3d"],
                         t["speedup_2d_over_3d"], t["used_3d"],
                         t["gbps_2d"], t["gbps_3d"])

    wandb.log({
        "microbench/numerics": num_tbl,
        "microbench/timing": tim_tbl,
        # mechanism confirmations (counts over the 8 layer/ctx configs)
        "microbench/n_configs": len(MICRO["numerics"]),
        "microbench/bf16_partials_worse_than_fp32_count": bf16_worse,
        "microbench/fp32_output_full_reassociation_count": fp32_full_reassoc,
        "microbench/3dfp32_as_accurate_as_2d_vs_sdpa_count": sdpa_3d_le_2d,
        "microbench/all_configs_used_3d_split_kv": int(all_used_3d),

        # ---- e2e scalars ----
        # probe TPS (256-tok single stream, N=6 reps rep0 discarded)
        "e2e/tps_probe_2d_median": a2["tps_median"],
        "e2e/tps_probe_3d_median": a3["tps_median"],
        "e2e/tps_probe_2d_cv": a2["tps_cv"],
        "e2e/tps_probe_3d_cv": a3["tps_cv"],
        "e2e/tps_probe_delta_pct": td["pct"],
        "e2e/tps_probe_delta_in_sigma": td["delta_in_sigma"],
        # realistic 128-prompt decode wall throughput (official-comparable)
        "e2e/decode_wall_tps_2d": wall_2d,
        "e2e/decode_wall_tps_3d": wall_3d,
        "e2e/decode_wall_tps_delta_pct": 100.0 * (wall_3d - wall_2d) / wall_2d,
        # ppl (prefill teacher-forcing; never hits M=1 -> bit-identical)
        "e2e/ppl_2d": a2["ppl_summary"]["ppl"],
        "e2e/ppl_3d": a3["ppl_summary"]["ppl"],
        "e2e/ppl_bit_identical": int(
            a2["ppl_summary"]["neg_log_likelihood"]
            == a3["ppl_summary"]["neg_log_likelihood"]),
        # greedy identity (3d candidate vs 2d control)
        "e2e/greedy_verdict_3d_vs_2d": gc["verdict"],
        "e2e/num_prompts": E2E["num_prompts"],
        "e2e/num_identical": gc["num_identical"],
        "e2e/num_divergent": gc["num_divergent"],
        "e2e/prompt_divergence_frac": gc["num_divergent"] / E2E["num_prompts"],
        "e2e/onset_min": gc["onset_min"],
        "e2e/onset_median": gc["onset_median"],
        "e2e/onset_max": gc["onset_max"],
        "e2e/onset_min_frac_of_outlen": gc["onset_min"] / gc["output_len"],
        "e2e/output_len": gc["output_len"],
        # toggle audit
        "e2e/arm2d_force2d_wrapped": int(a2["log_force2d_wrapped"]),
        "e2e/arm3d_surgattn_disabled": int(a3["log_surgattn_disabled"]),
    })

    # onset table for the divergent prompts
    onset_tbl = wandb.Table(columns=["divergent_prompt_rank", "first_divergence_tok_idx",
                                     "frac_of_output_len"])
    for i, o in enumerate(gc["onsets"]):
        onset_tbl.add_data(i, o, o / gc["output_len"])
    wandb.log({"e2e/divergence_onsets": onset_tbl})

    # ---- determinism control: is the 4/128 the 3D effect or cross-session noise? ----
    # 2d_a-vs-2d_b and 3d_a-vs-3d_b are the same-config cross-session noise floors;
    # 2d-vs-3d pairs are the cross-config effect. If the noise floors are 0 and the
    # cross-config divergence is identical across all four pairings, the 4/128 is
    # PURELY the 2D-vs-3D reassociation, not run-to-run nondeterminism.
    if DETERM is not None:
        det_tbl = wandb.Table(columns=["pair", "verdict", "num_divergent",
                                       "total_divergent_tokens", "onsets"])
        by_pair = {}
        for p in DETERM["pairs"]:
            by_pair[p["pair"]] = p
            det_tbl.add_data(p["pair"], p["verdict"], p["num_divergent"],
                             p.get("total_divergent_tokens"), str(p.get("onsets")))
        noise_2d = by_pair.get("2d_a_vs_2d_b", {}).get("num_divergent")
        noise_3d = by_pair.get("3d_a_vs_3d_b", {}).get("num_divergent")
        xcfg = [by_pair[k]["num_divergent"] for k in
                ("2d_a_vs_3d_a", "2d_a_vs_3d_b", "2d_b_vs_3d_a", "2d_b_vs_3d_b")
                if k in by_pair]
        wandb.log({
            "determ/control": det_tbl,
            "determ/run_to_run_2d_divergent": noise_2d,
            "determ/run_to_run_3d_divergent": noise_3d,
            "determ/noise_band_is_zero": int(noise_2d == 0 and noise_3d == 0),
            "determ/cross_config_2d_vs_3d_divergent_min": min(xcfg) if xcfg else None,
            "determ/cross_config_2d_vs_3d_divergent_max": max(xcfg) if xcfg else None,
            # the clincher: cross-config divergence is identical & reproducible across
            # all 4 pairings, while same-config run-to-run divergence is exactly 0.
            "determ/4of128_is_pure_3d_effect_not_noise": int(
                noise_2d == 0 and noise_3d == 0 and xcfg and min(xcfg) == max(xcfg)),
        })

    print(f"[results] W&B run: {run.url}  id={run.id}")
    wandb.finish()
    print("[results] done")


if __name__ == "__main__":
    main()
