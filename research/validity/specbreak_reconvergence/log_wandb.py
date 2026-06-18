#!/usr/bin/env python3
"""Log spec-break reconvergence scalars to W&B (PR #686, analysis_only).

Reads reconvergence_report.json (produced by analyze_reconvergence.py) and logs
machine-checkable scalars + per-source + per-W breakdowns, plus a detail
artifact. NO GPU, NO HF Job. official_tps=0, fires=0.
"""
import json
import os
import wandb

REPORT = "research/validity/specbreak_reconvergence/reconvergence_report.json"
HEADLINE_CELL = "suffix6"
HEADLINE_W = 16
VERDICT = "BREAK_RECONVERGENT"  # transient/partial + shifted (see PR analysis)


def flat_scalars(rep):
    """Flatten the per-cell/per-W/per-source aggregates into a flat dict."""
    out = {}
    KEYS = [
        "n_break", "assessable", "assessable_frac", "not_assessable", "filler",
        "score_relevant", "reconverge_frac", "reconverged",
        "reconverge_frac_relevant", "reconverge_frac_samepos",
        "reconverged_samepos", "ends_coupled_frac", "ends_coupled",
        "ends_coupled_frac_relevant", "permanent_divergence_frac",
        "permanent_divergence", "redivergence", "median_realign_gap",
        "mean_realign_gap", "max_realign_gap", "median_samepos_gap",
        "median_abs_shift", "max_abs_shift", "mean_coverage", "both_commit",
        "answer_flips",
    ]
    SRC_KEYS = ["n_break", "assessable", "reconverge_frac",
                "reconverge_frac_samepos", "ends_coupled_frac",
                "permanent_divergence_frac", "median_realign_gap"]
    for cell, cd in rep["cells"].items():
        for W, a in cd["W"].items():
            pre = f"{cell}/W{W}"
            for k in KEYS:
                v = a.get(k)
                if isinstance(v, (int, float)):
                    out[f"{pre}/{k}"] = v
            for src, sa in a.get("per_source", {}).items():
                for k in SRC_KEYS:
                    v = sa.get(k)
                    if isinstance(v, (int, float)):
                        out[f"{pre}/{src}/{k}"] = v
    return out


def main():
    rep = json.load(open(REPORT))
    head = rep["cells"][HEADLINE_CELL]["W"][str(HEADLINE_W)] \
        if str(HEADLINE_W) in rep["cells"][HEADLINE_CELL]["W"] \
        else rep["cells"][HEADLINE_CELL]["W"][HEADLINE_W]

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        name="kanna/specbreak-reconvergence-realignment",
        group="specbreak-reconvergence-kanna",
        job_type="analysis",
        config=dict(
            analysis_only=1, official_tps=0, fires=0,
            pr=686, verdict=VERDICT,
            headline_cell=HEADLINE_CELL, headline_W=HEADLINE_W,
            windows=rep["windows"], decode_len=rep["L"],
            n_prompts=128,
            set="57 mmlu_pro + 57 gpqa_diamond + 14 aime2026 (#678 scored set)",
            source_decode="specdec_official_dist_breakrate/_runs/confirm "
                          "(decode_{ar,suffix6,ngram5}_r0.jsonl)",
            cross_reads=["#685 sd3mbkdp", "#678 gnfgcn90", "#673 decode"],
            reconverge_def="shift-tolerant: spec[q:q+W] matches a contiguous AR "
                           "window of len W>=16 (difflib longest block on "
                           "post-divergence tails); W=16 coincidence ~ V^-16 ~1e-83",
            ends_coupled_def="a >=W block reaches AR's tail end -> AR's final W "
                             "tokens reproduced (shifted) in spec; best #682 predictor",
            samepos_def="strict shift=0: spec[q:q+W]==AR[q:q+W] (lower bound)",
            note_682="wirbel #682 NOT cross-read (launch-isolation: kanna-only "
                     "scope); prediction stated for advisor to cross-check",
        ),
    )

    # flat scalars (all cells/W/sources)
    scal = flat_scalars(rep)
    # promote headline scalars to top-level (un-namespaced) for easy reading
    scal.update({
        "analysis_only": 1, "official_tps": 0, "fires": 0,
        "reconverge_frac": head["reconverge_frac"],
        "reconverge_frac_samepos": head["reconverge_frac_samepos"],
        "ends_coupled_frac": head["ends_coupled_frac"],
        "permanent_divergence_frac": head["permanent_divergence_frac"],
        "assessable_frac": head["assessable_frac"],
        "median_realign_gap": head["median_realign_gap"],
        "median_abs_shift": head["median_abs_shift"],
        "mean_coverage": head["mean_coverage"],
        "answer_flips": head["answer_flips"],
        "both_commit": head["both_commit"],
    })
    wandb.log(scal)
    for k, v in scal.items():
        run.summary[k] = v
    run.summary["verdict"] = VERDICT

    art = wandb.Artifact("specbreak_reconvergence_detail", type="analysis")
    art.add_file(REPORT)
    art.add_file("research/validity/specbreak_reconvergence/analyze_reconvergence.py")
    run.log_artifact(art)

    print(f"WANDB_RUN_ID={run.id}")
    print(f"verdict={VERDICT}  reconverge_frac={head['reconverge_frac']:.4f}  "
          f"ends_coupled_frac={head['ends_coupled_frac']:.4f}  "
          f"assessable_frac={head['assessable_frac']:.4f}")
    run.finish()


if __name__ == "__main__":
    main()
