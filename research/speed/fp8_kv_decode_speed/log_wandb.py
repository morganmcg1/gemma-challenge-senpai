#!/usr/bin/env python
"""Log the PR #729 fp8/int8-KV decode-lever FEASIBILITY verdict to W&B.

project = gemma-challenge-senpai, entity = wandb-applied-ai-team,
group = denken-fp8-kv-decode. analysis_only -- logs the feasibility census,
the even-if-runnable roofline, and the auto(bf16-KV) baseline validation.
No official TPS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

HERE = Path("research/speed/fp8_kv_decode_speed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict", default=str(HERE / "feasibility_verdict.json"))
    ap.add_argument("--auto", default=str(HERE / "arm_auto.json"))
    ap.add_argument("--roofline", default=str(HERE / "roofline_int4head.json"))
    ap.add_argument("--name", default="denken/fp8-kv-decode-speed")
    args = ap.parse_args()

    v = json.loads(Path(args.verdict).read_text())
    auto = json.loads(Path(args.auto).read_text())
    roof = json.loads(Path(args.roofline).read_text())
    feas = v["feasibility"]
    base = v["baseline_validation"]
    comp = base["comparator_self_test_auto_vs_auto"]

    run = wandb.init(
        entity="wandb-applied-ai-team", project="gemma-challenge-senpai",
        group="denken-fp8-kv-decode", name=args.name,
        config={
            "pr": 729, "lever": "fp8_int8_kv_cache_decode", "analysis_only": True,
            "official_tps": 0, "no_served_file_change": True, "no_hf_job": True,
            "anchor_submission": "int4_g128_lmhead", "anchor_official_tps": 126.378,
            "anchor_ppl": 2.019, "anchor_wandb": "905tbujn",
            "scored_output_len": 512, "deployed_max_model_len": 4096,
            "gpu": "A10G_sm86_ampere", "vllm": "0.22.0",
            "attn_backend": "TRITON_ATTN_forced",
            "int8_kv_supported_vllm_0_22": False,
            "fp8_e4m3_runnable_on_sm86": False,
            "fp8_e5m2_runnable_with_quant_ckpt": False,
            "n_text_layers": 42, "n_sliding": 35, "n_full": 7, "sliding_window": 512,
            "n_kv_heads": 2, "head_dim_local": 256, "head_dim_global": 512,
            "decode_weight_read_gib": roof["weight_read_gib"],
            "verdict": v["verdict"],
        },
    )

    summary = {
        # feasibility census
        "feasibility/fp8_e4m3_runnable": int(feas["fp8_e4m3"]["runnable"]),
        "feasibility/fp8_e4m3_fails_compiled": int(feas["fp8_e4m3"]["compiled_fails"]),
        "feasibility/fp8_e4m3_fails_eager": int(feas["fp8_e4m3"]["eager_fails"]),
        "feasibility/fp8_e5m2_runnable": int(feas["fp8_e5m2"]["runnable"]),
        "feasibility/fp8_e5m2_init_blocked": int(feas["fp8_e5m2"]["init_fails"]),
        "feasibility/int8_runnable": int(feas["int8"]["runnable"]),
        "feasibility/auto_bf16_runnable": int(feas["auto_bf16"]["runnable"]),
        "feasibility/any_kv_dtype_reduction_runnable": int(
            feas["any_kv_dtype_reduction_runnable_on_stack"]),
        # even-if-runnable roofline (optimistic upper bound)
        "roofline/fp8_uplift_pct_at_512_optimistic": roof["fp8_uplift_pct_at_512_optimistic"],
        "roofline/fp8_uplift_pct_at_2048_optimistic": roof["fp8_uplift_pct_at_2048_optimistic"],
        "roofline/fp8_uplift_pct_at_8192_optimistic": roof["fp8_uplift_pct_at_8192_optimistic"],
        "roofline/kv_read_frac_at_512": roof["kv_read_frac_at_512"],
        "roofline/kv_material_crossover_10pct_position": roof["kv_material_crossover_10pct_position"],
        "roofline/crossover_beyond_deployed_cap": int(roof["crossover_beyond_deployed_cap"]),
        "roofline/material_even_optimistic_at_512": int(v["material_even_optimistic_at_512"]),
        # auto(bf16-KV) baseline validation
        "baseline/auto_ppl": auto["ppl"]["ppl"],
        "baseline/auto_ppl_safe": int(base["auto_bf16_kv"]["ppl_safe"]),
        "baseline/auto_tps_512": auto["tps"]["512"]["output_tps"],
        "baseline/auto_tps_2048": auto["tps"]["2048"]["output_tps"],
        "baseline/auto_tps_8192": auto["tps"]["8192"]["output_tps"],
        "baseline/peak_gib": auto.get("peak_gib"),
        "baseline/load_s": auto.get("load_s"),
        "baseline/ppl_reproduces_anchor": int(base["ppl_reproduces_anchor"]),
        "baseline/tps_512_near_official": int(base["tps_512_near_official"]),
        "baseline/sliding_window_pin": int(base["sliding_window_pin"]),
        # comparator self-test (auto-vs-auto degenerate identity)
        "comparator/greedy_gate_identical": int(comp["greedy_gate"] == "GREEDY_IDENTICAL"),
        "comparator/flips": comp["flips"],
        "comparator/delta_pct_at_512": comp["delta_pct_at_512"],
        "comparator/self_test_passes": int(comp["self_test_passes"]),
        # verdict
        "verdict/kv_lever_is_green_for_official": int(v["kv_lever_is_green_for_official"]),
        "verdict/infeasible_and_immaterial": int(v["verdict"] == "INFEASIBLE_AND_IMMATERIAL"),
        "verdict/crossover_reachable_in_deployed_cap": int(v["crossover_reachable_in_deployed_cap"]),
        "self_test_passes": int(v["self_test"]["self_test_passes"]),
    }

    # roofline TPS-vs-length table (the even-if-runnable optimistic bound)
    rtbl = wandb.Table(columns=["output_len", "kv_bytes_MB", "kv_read_frac",
                                "fp8_uplift_pct_optimistic", "reachable_deployed"])
    for L in sorted(int(k) for k in roof["rows"]):
        r = roof["rows"][str(L)]
        rtbl.add_data(L, r["kv_bytes_bf16_MB"], r["kv_read_frac_bf16"],
                      r["fp8_uplift_pct_roofline_optimistic"],
                      int(r["reachable_in_deployed_cap"]))

    # measured auto(bf16-KV) TPS table
    atbl = wandb.Table(columns=["output_len", "auto_bf16_tps", "n_prompts",
                                "reachable_deployed"])
    for L in sorted(int(k) for k in auto["tps"]):
        t = auto["tps"][str(L)]
        atbl.add_data(L, t["output_tps"], t["n_prompts"], int((L + 300) <= 4096))

    # feasibility census table
    ftbl = wandb.Table(columns=["kv_dtype", "runnable", "blocking_layer", "reason"])
    for key in ("fp8_e4m3", "fp8_e5m2", "int8", "auto_bf16"):
        d = feas[key]
        ftbl.add_data(key, int(d["runnable"]), d.get("layer", ""), d["reason"])

    run.log({"roofline_vs_length": rtbl, "auto_tps_vs_length": atbl,
             "feasibility_census": ftbl, **summary})
    for k, val in summary.items():
        run.summary[k] = val
    print("WANDB_RUN_ID", run.id)
    run.finish()


if __name__ == "__main__":
    main()
