"""PR #648 served recompute-fire census -> W&B group `served-recompute-fire-census-land`.

Reads fire_census_result.json (no server, no recompute). Logs the per-position
served fire-fraction (overall + per K) with bootstrap CI, the pre/post-divergence
decomposition, clustering stats, and the wall-TPS tax cross-check vs stark #636's
139.20 projection.  analysis_only=true, official_tps=0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts import wandb_logging  # noqa: E402

RESULT = HERE / "fire_census_result.json"

PR = 648
N632_DIVERGENCE_PROMPT_FRAC = 0.844  # 108/128 prompt-level (#632)
STARK_TF_FIRE = 0.0780
STARK_PROJ_TPS = 139.20
OPTIONB_BASE_TPS = 172.74
LOCKED_TPS = 126.378


def main() -> int:
    res = json.loads(RESULT.read_text())
    hk = res["headline_k"]
    head = res["per_k"][f"k{hk}"]

    run = wandb_logging.init_wandb_run(
        job_type="fire_census",
        agent="land",
        name="land/served-recompute-fire-census",
        group="served-recompute-fire-census-land",
        notes=("PR#648: per-POSITION census of stark #636's tau=0.5 M=1-recompute flag over the "
               "#632 served Option-B BI=1 spec stream (all 128 prompts, full 512-token length, not "
               "just root forks). served_fire_frac is the COST-side input that cross-checks stark "
               "#642's wall-TPS de-projection. Extends #645 (root-fork-only) to all positions."),
        config={
            "pr": PR, "analysis_only": True, "official_tps": 0,
            "vllm": "0.22.0", "drafter": "/tmp/qat-assistant", "batch_invariant": 1,
            "tau_nat": res["tau"], "num_prompts": head["n_prompts"], "output_len": 512,
            "n_logprobs": 20, "served_reference": "served_spec_off_M1_AR (BASELINE.md L10)",
            "ks_censused": sorted(int(k[1:]) for k in res["per_k"]),
            "headline_k": hk, "stark_636_tf_fire": STARK_TF_FIRE,
            "optionb_base_tps": OPTIONB_BASE_TPS, "stark_proj_tps": STARK_PROJ_TPS,
            "locked_tps": LOCKED_TPS,
        },
        tags=["optionb", "batch_invariant", "pr648", "fire_census", "served", "cost_model"],
    )
    if run is None:
        print("WANDB disabled/unavailable; dumping summary only", flush=True)

    summary = {
        "census/headline_k": hk,
        "census/served_fire_frac": res["served_fire_frac_overall"],
        "census/ci95_lo": res["served_fire_frac_ci95"][0],
        "census/ci95_hi": res["served_fire_frac_ci95"][1],
        "census/stark_tf_fire_frac": STARK_TF_FIRE,
        "census/served_over_tf_ratio": res["served_fire_frac_overall"] / STARK_TF_FIRE,
        "census/total_positions": head["total_positions"],
        "census/total_fires": head["total_fires"],
        "decomp/pre_div_fire_frac": head["pre_div_fire_frac"],
        "decomp/post_div_fire_frac": head["post_div_fire_frac"],
        "decomp/post_over_pre_amplification": res["post_over_pre_amplification"],
        "cluster/fires_per_prompt_mean": head["fires_per_prompt_mean"],
        "cluster/fires_per_prompt_median": head["fires_per_prompt_median"],
        "cluster/fires_per_prompt_max": head["fires_per_prompt_max"],
        "cluster/n_prompts_zero_fire": head["n_prompts_zero_fire"],
        "cluster/mean_inter_fire_gap": head["mean_inter_fire_gap"],
        "kdep/k_spread_pp": res["k_spread_pp"],
        "kdep/k_independent": int(bool(res["k_independent"])),
        "tax/stark_implied_overhead_ratio_r": head["tax_crosscheck"]["stark_implied_overhead_ratio_r"],
        "tax/crosscheck_wall_tps": res["crosscheck_wall_tps"],
        "tax/delta_vs_stark_proj": head["tax_crosscheck"]["delta_vs_stark_proj"],
        "tax/stays_above_locked_126p378": int(bool(res["stays_above_locked"])),
        "tax/fire_frac_breakeven_to_locked": head["tax_crosscheck"]["fire_frac_breakeven_to_locked"],
        "decision/verdict": res["verdict"],
    }

    if run is not None:
        import wandb
        # per-K table
        cols = ["K", "total_positions", "total_fires", "served_fire_frac",
                "ci95_lo", "ci95_hi", "pre_div_fire_frac", "post_div_fire_frac",
                "fires_per_prompt_mean", "mean_inter_fire_gap", "n_sha_ok", "crosscheck_wall_tps"]
        tbl = wandb.Table(columns=cols)
        for k in sorted(int(kk[1:]) for kk in res["per_k"]):
            r = res["per_k"][f"k{k}"]
            tbl.add_data(k, r["total_positions"], r["total_fires"], r["served_fire_frac"],
                         r["ci95_boot"][0], r["ci95_boot"][1], r["pre_div_fire_frac"],
                         r["post_div_fire_frac"], r["fires_per_prompt_mean"],
                         r["mean_inter_fire_gap"], r["n_sha_ok"],
                         r["tax_crosscheck"]["crosscheck_wall_tps"])
        run.log({"per_k_fire_census": tbl})
        # fires-per-prompt histogram (headline K)
        run.log({"hist/fires_per_prompt_headline": wandb.Histogram(
            head["fires_per_prompt_hist"], num_bins=40)})

        wandb_logging.log_summary(run, summary, step=PR)
        wandb_logging.log_json_artifact(
            run, name="served_fire_census_648", artifact_type="analysis",
            data={"summary": summary, "result": res})
        url = getattr(run, "url", "")
        rid = getattr(run, "id", "")
        wandb_logging.finish_wandb(run)
        print(f"[wandb] fire census id={rid} url={url}", flush=True)

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
