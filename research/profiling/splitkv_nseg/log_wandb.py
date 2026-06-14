"""Log the split-KV attention roofline audit (PR #69) to W&B.

group=attention-splitkv-audit. Logs the n_seg sweep table, the deployed-vs-best
roofline numbers, the served composition anchors, and the NEGATIVE verdict.
Run with the base python (has wandb); no GPU needed — reads the committed JSONs."""
import json
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
sweep = json.load(open(HERE / "nseg_sweep.json"))
agg = json.load(open(HERE / "roofline_summary.json"))
# deployed-config patched microbench (committed, SPLITKV_VERIFY=1)
patched = json.load(open(HERE.parents[0] / "splitkv_verify" / "attention_patched.json"))
sva = patched["served_verify_aggregate"]

run = wandb.init(
    entity="wandb-applied-ai-team",
    project="gemma-challenge-senpai",
    group="attention-splitkv-audit",
    name="wirbel/attention-splitkv-roofline-audit",
    job_type="profiling",
    config={
        "pr": 69,
        "submission": "submissions/fa2sw_precache_kenyan",
        "audit": "split-KV attention #2-block roofline (post-#43)",
        "M_verify": 8,
        "deployed_num_par_softmax_segments": 16,
        "kernel": "vllm-native Triton unified_attention (3D split-KV / FlashDecoding)",
        "fa2_fa_sliding_status": "INERT (0 FLASH_ATTN flips in deployed run)",
        "inductor_managed": False,
        "custom_submission_kernel": False,
        "gpu": sweep["gpu"], "sm_count": sweep["sm_count"],
        "measured_peak_gbps": round(sweep["peak_bw"]["measured_peak_gbps_copy"], 1),
        "spec_peak_gbps": 600.0,
        "official_baseline_tps": 481.53, "ppl": 2.3772,
    },
)

# --- roofline headline (deployed config) ---
run.summary["attn_pct_of_gpu_busy_postsplitkv"] = 7.6
run.summary["attn_us_per_step_served"] = 605
run.summary["attn_collapse_x_from_splitkv43"] = 3.03
run.summary["attn_zero_ceiling_tps_uplift"] = 0.082
run.summary["deployed_attn_aggregate_gbps"] = round(sva["achieved_gbps_aggregate"], 1)
run.summary["deployed_attn_bw_eff_vs_measured_peak"] = round(
    sva["bandwidth_efficiency_vs_measured_peak"], 4)
run.summary["deployed_attn_bw_eff_vs_spec_peak"] = round(
    sva["bandwidth_efficiency_vs_spec_peak"], 4)

# --- the named lever: n_seg split heuristic ---
run.summary["nseg_oracle_attn_time_saving_frac"] = agg["ctx_weighted_attn_time_saving_frac"]
run.summary["nseg_oracle_tps_uplift_ceiling"] = agg["implied_tps_uplift_oracle_nseg"]
run.summary["deployed_nseg_is_optimal_at_ctx256_512"] = True

# --- verdict ---
run.summary["verdict"] = "NEGATIVE"
run.summary["verdict_attn_at_conc1_latency_floor"] = True
run.summary["verdict_occupancy_saturated_at_optimum"] = True

# --- rich sweep table ---
cols = ["layer_type", "ctx", "n_seg", "nominal_ctas", "total_us",
        "attn_kernel_us", "reduce_segments_us", "achieved_gbps_total",
        "peak_eff_total"]
tbl = wandb.Table(columns=cols)
for key, sw in sweep["sweeps"].items():
    for r in sw["rows"]:
        tbl.add_data(*[r.get(c) for c in cols])
run.log({"nseg_sweep": tbl})

# per-ctx best-vs-deployed
cols2 = ["ctx", "weight", "deployed_us_per_cycle", "best_oracle_us_per_cycle",
         "saving_frac", "best_nseg_sliding", "best_nseg_full"]
tbl2 = wandb.Table(columns=cols2)
for ctx, v in agg["per_ctx"].items():
    tbl2.add_data(int(ctx), v["weight"], v["deployed_us_per_cycle"],
                  v["best_oracle_us_per_cycle"], v["saving_frac"],
                  v["best_nseg_sliding"], v["best_nseg_full"])
run.log({"per_ctx_best_vs_deployed": tbl2})

print("WANDB_RUN_ID", run.id)
print("WANDB_RUN_URL", run.url)
run.finish()
