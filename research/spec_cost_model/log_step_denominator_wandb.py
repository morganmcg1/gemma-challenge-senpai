"""Log the PR #154 step-denominator-reduction audit (LOCAL analysis, no training run)
to W&B under --wandb_group step-denominator-reduction, so the audit leaves a rich,
queryable record alongside the JSON artifacts. No HF Job / no served-file change.
"""
import json
import os

import wandb

ROOT = "/workspace/senpai/target/research/spec_cost_model"
AUDIT = os.path.join(ROOT, "step_denominator_reduction_audit.json")
LAUNCH = os.path.join(ROOT, "launch_overhead_graph_leg.json")


def main():
    with open(AUDIT) as f:
        a = json.load(f)
    with open(LAUNCH) as f:
        lg = json.load(f)

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY"),
        group="step-denominator-reduction",
        name="ubel/step-denominator-reduction-audit",
        job_type="analysis",
        config={
            "pr": 154,
            "scope": a["scope"],
            "lever": a["lever"],
            "official_tps_baseline": a["projection_constants"]["official_tps"],
            "ppl_baseline": a["projection_constants"]["ppl"],
            "K_cal": a["projection_constants"]["K_cal"],
            "step_dimensionless": a["projection_constants"]["step_dimensionless_136"],
            "clear_500_bar": a["projection_constants"]["clear_500_bar"],
            "tau": a["projection_constants"]["tau"],
            "no_hf_job": True,
            "no_served_file_change": True,
            "baseline_unchanged_481_53": True,
        },
    )

    bar = a["projection_constants"]["clear_500_bar"]
    cons_bar = next(r for r in a["propagation"]["conservative_M32_tree"]["rows"] if r["E_T"] == bar)
    real_bar = next(r for r in a["propagation"]["realistic_M32_tree"]["rows"] if r["E_T"] == bar)

    wandb.summary.update({
        # PRIMARY / TEST as named by the PR
        "step_reduction_audit_self_test_passes": a["step_reduction_audit_self_test_passes"],
        "recoverable_step_pct": a["recoverable_step_pct"],
        "recoverable_step_pct_realistic": a["recoverable_step_pct_realistic"],
        # avoidable us/step
        "avoidable_gross_us_M8": a["avoidable_us"]["gross_M8"],
        "avoidable_gross_us_M32": a["avoidable_us"]["gross_M32"],
        "avoidable_net_conservative_us_M32": a["avoidable_us"]["net_conservative_M32"],
        "avoidable_net_realistic_us_M32": a["avoidable_us"]["net_realistic_M32"],
        # TPS + bar effect at the clear-500 bar (E[T]=4.862, tree M=32)
        "dtps_conservative_at_bar": cons_bar["dtps"],
        "dtps_realistic_at_bar": real_bar["dtps"],
        "clear_500_bar_new_conservative": cons_bar["clear_500_bar_new"],
        "clear_500_bar_new_realistic": real_bar["clear_500_bar_new"],
        "clear_500_bar_drop_realistic": real_bar["clear_500_bar_drop"],
        # denken #144 anchors reproduced
        "anchor_gemm_only_us": a["anchors_denken_144"]["gemm_only_us"],
        "anchor_scatter_only_us": a["anchors_denken_144"]["scatter_only_us"],
        "anchor_compute_logits_full_us_M8": a["anchors_denken_144"]["compute_logits_full_us_M8"],
        "measured_scatter_us_M8": a["measurement"]["per_M"]["8"]["scatter_us"],
        "gemm_bw_floor_us": a["measurement"]["gemm_bw_floor_us"],
        "hbm_copy_gbps": a["measurement"]["hbm_copy_gbps"],
        # Leg 2 (CUDA-graph) — already closed
        "launch_per_launch_overhead_us": lg["measurement"]["per_launch_overhead_us"],
        "launch_total_eager": lg["model"]["launch_counts"]["total_eager"],
        "launch_captured_by_deployed_graph": lg["model"]["launch_counts"]["captured_by_deployed_graph"],
        "launch_residual_uncaptured": lg["model"]["launch_counts"]["residual_uncaptured"],
        "launch_residual_headroom_pct_upper": lg["model"]["residual_headroom_pct_of_bar_step_upper"],
        "launch_residual_headroom_pct_80pct_overlap": lg["model"]["residual_headroom_pct_illustrative_80pct_overlap"],
        # health
        "metrics_nan_clean": a["metrics_nan_clean"],
        "self_test_n_passed": a["self_test"]["n_passed"],
        "self_test_n_checks": a["self_test"]["n_checks"],
        "greedy_safety_softcap_monotone_ok_all_M": a["greedy_safety"]["softcap_argmax_monotone_ok_all_M"],
    })

    # propagation sweep table (all four bound x E[T])
    cols = ["bound", "E_T", "step_abs_us", "recoverable_step_pct",
            "official_old_tps", "official_new_tps", "dtps", "clear_500_bar_new", "clear_500_bar_drop"]
    rows = []
    for key, block in a["propagation"].items():
        for r in block["rows"]:
            rows.append([block["label"], r["E_T"], r["step_abs_us"], r["recoverable_step_pct"],
                         r["official_old_tps"], r["official_new_tps"], r["dtps"],
                         r["clear_500_bar_new"], r["clear_500_bar_drop"]])
    wandb.log({"propagation_sweep": wandb.Table(columns=cols, data=rows)})

    # self-test table
    st_cols = ["name", "passes", "detail"]
    st_rows = [[c["name"], c["passes"], c["detail"]] for c in a["self_test"]["checks"]]
    wandb.log({"self_test_checks": wandb.Table(columns=st_cols, data=st_rows)})

    # avoidable-by-M table
    m_cols = ["M", "scatter_us", "lp_served_us", "gross_avoidable_us",
              "net_conservative_us", "net_realistic_us", "softcap_argmax_monotone_ok", "nan_clean"]
    m_rows = []
    for M, d in a["measurement"]["per_M"].items():
        m_rows.append([int(M), d["scatter_us"], d["lp_served_us"], d["gross_avoidable_us"],
                       d["net_conservative_us"], d["net_realistic_us"],
                       d["softcap_argmax_monotone_ok"], d["nan_clean"]])
    wandb.log({"avoidable_by_M": wandb.Table(columns=m_cols, data=m_rows)})

    run_id = run.id
    run_url = run.url
    run.finish()
    print("WANDB_RUN_ID", run_id)
    print("WANDB_RUN_URL", run_url)


if __name__ == "__main__":
    main()
