"""PR #645 served-divergence margin census -> W&B group `served-margin-census-land`.

Reads the local analysis (margin_census_result.json + margin_per_fork.jsonl); no
server, no recompute. Logs the margin histogram (both the PR-instruction AR-token
margin and the literal M=8 top1-top2 gap), the per-fork table, and the verdict on
whether stark #636's tau=0.5 recompute flag covers the 84% served divergences.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts import wandb_logging  # noqa: E402

RESULT = HERE / "margin_census_result.json"
PER_FORK = HERE / "margin_per_fork.jsonl"

# context from PR #645 body / #632 closeout (advisor-provided, not re-derived)
PR = 645
N632_DIVERGENCE_PROMPT_FRAC = 0.844  # 108/128
STARK_TF_MIN_TAU = 0.5               # stark #636 teacher-forced min_tau
STARK_FLAG_FIRE_FRAC = 0.0780        # stark #636 flag-fires on 7.80% of positions


def main() -> int:
    res = json.loads(RESULT.read_text())
    forks = [json.loads(l) for l in PER_FORK.read_text().splitlines() if l.strip()]

    run = wandb_logging.init_wandb_run(
        job_type="margin_census",
        agent="land",
        name="land/served-margin-census",
        group="served-margin-census-land",
        notes=("PR#645: served-trajectory M=8 verify-margin census at the ROOT fork of every "
               "Option-B BI=1 spec vs served-M=1-AR divergence (#632's 84%). Tests whether stark "
               "#636's tau=0.5 recompute flag covers all served root forks or leaves a hole."),
        config={
            "pr": PR, "analysis_only": True, "official_tps": 0,
            "vllm": "0.22.0", "submission": "submissions/int4_mtp_batchinv",
            "drafter": "/tmp/qat-assistant", "batch_invariant": 1,
            "K": 7, "M_verify": 8, "num_prompts": 128, "output_len": 512, "seed": 1,
            "served_reference": "served_spec_off_M1_AR (BASELINE.md L10)",
            "k7_anchor_run_632": "8sfauo3i", "n_logprobs": 20,
            "stark_636_min_tau": STARK_TF_MIN_TAU,
        },
        tags=["optionb", "batch_invariant", "pr645", "margin_census", "served", "identity"],
    )
    if run is None:
        print("WANDB disabled/unavailable; dumping summary only", flush=True)

    summary = {
        "census/n_prompts_diverged": res["n_prompts_diverged"],
        "census/n_root_forks": res["n_root_forks"],
        "census/divergence_prompt_frac_632": N632_DIVERGENCE_PROMPT_FRAC,
        # AR-token margin (PR instruction #2)
        "margin_AB/frac_sub_0p5": res["hist_AB_pr_instruction"]["frac_sub_0p5"],
        "margin_AB/frac_0p5_to_1p0": res["hist_AB_pr_instruction"]["frac_0p5_to_1p0"],
        "margin_AB/frac_ge_1p0": res["hist_AB_pr_instruction"]["frac_ge_1p0"],
        "margin_AB/min": res["hist_AB_pr_instruction"]["min"],
        "margin_AB/median": res["hist_AB_pr_instruction"]["median"],
        "margin_AB/p95": res["hist_AB_pr_instruction"]["p95"],
        "margin_AB/max": res["hist_AB_pr_instruction"]["max"],
        "margin_AB/mean": res["hist_AB_pr_instruction"]["mean"],
        # literal M=8 top1-top2 gap (what a gap acceptor thresholds on)
        "margin_AC/frac_sub_0p5": res["hist_AC_top1_top2_gap"]["frac_sub_0p5"],
        "margin_AC/frac_0p5_to_1p0": res["hist_AC_top1_top2_gap"]["frac_0p5_to_1p0"],
        "margin_AC/frac_ge_1p0": res["hist_AC_top1_top2_gap"]["frac_ge_1p0"],
        "margin_AC/min": res["hist_AC_top1_top2_gap"]["min"],
        "margin_AC/median": res["hist_AC_top1_top2_gap"]["median"],
        "margin_AC/p95": res["hist_AC_top1_top2_gap"]["p95"],
        "margin_AC/max": res["hist_AC_top1_top2_gap"]["max"],
        # cross-check + verdict
        "crosscheck/served_min_tau_for_zero_break": res["served_min_tau_for_zero_break"],
        "crosscheck/stark_teacher_forced_min_tau": STARK_TF_MIN_TAU,
        "crosscheck/transfers_to_served": int(res["transfers_to_served"]),
        "crosscheck/frac_B_eq_runner_up": res["frac_B_eq_C"],
        "crosscheck/n_B_not_found_in_top20": res["n_B_not_found_in_top20"],
        "crosscheck/n_decode_mapping_ok": res["n_decode_mapping_ok"],
        "crosscheck/stark_flag_fire_frac_tf": STARK_FLAG_FIRE_FRAC,
        "decision/verdict": res["verdict"],
        "config/peak_vram_mib": 19921,
    }

    if run is not None:
        import wandb
        # per-fork table
        cols = ["id", "root_pos", "A_str", "B_str", "C_str",
                "logp_A", "logp_B", "logp_C", "margin_AB", "margin_AC", "B_eq_C", "B_rank"]
        tbl = wandb.Table(columns=cols)
        for f in sorted(forks, key=lambda x: x["root_pos"]):
            tbl.add_data(*[f.get(c) for c in cols])
        # margin histograms (native)
        gaps_AC = [f["margin_AC"] for f in forks]
        gaps_AB = [f["margin_AB"] if f["margin_AB"] is not None else f["margin_AB_lb"] for f in forks]
        run.log({
            "per_fork_margins": tbl,
            "hist/margin_AC_top1_top2": wandb.Histogram(gaps_AC, num_bins=40),
            "hist/margin_AB_ar_token": wandb.Histogram(gaps_AB, num_bins=40),
        })
        # explicit 3-bin coverage table (the deliverable histogram)
        binp = wandb.Table(columns=["bin", "label", "margin_AB_frac", "margin_AC_frac"])
        hA, hC = res["hist_AB_pr_instruction"], res["hist_AC_top1_top2_gap"]
        binp.add_data("[0,0.5)", "caught_by_tau0.5", hA["frac_sub_0p5"], hC["frac_sub_0p5"])
        binp.add_data("[0.5,1.0)", "coverage_hole", hA["frac_0p5_to_1p0"], hC["frac_0p5_to_1p0"])
        binp.add_data(">=1.0", "coverage_hole_loud", hA["frac_ge_1p0"], hC["frac_ge_1p0"])
        run.log({"coverage_bins": binp})

        wandb_logging.log_summary(run, summary, step=PR)
        wandb_logging.log_json_artifact(
            run, name="served_margin_census_645", artifact_type="analysis",
            data={"summary": summary, "result": res, "per_fork": forks})
        url = getattr(run, "url", "")
        rid = getattr(run, "id", "")
        wandb_logging.finish_wandb(run)
        print(f"[wandb] margin census id={rid} url={url}", flush=True)

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
