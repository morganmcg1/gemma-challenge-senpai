#!/usr/bin/env python3
"""Log spec-break reconvergence scalars to W&B (analysis_only).

Default mode (PR #686): reads reconvergence_report.json (analyze_reconvergence.py)
and logs reconverge/realign scalars + per-source/per-W breakdowns + detail
artifact.

--classify mode (PR #689): reads shift_edit_classification.json
(classify_shift_edits.py) and logs the shift-edit content audit -- cosmetic_edit_
frac (shifted heals) + permfork_content_divergence_frac (permanent forks), per
cell / pooled / dedup / per-source / per-class, + the fork-risk manifest artifact.

NO GPU, NO HF Job. analysis_only=1, official_tps=0, fires=0.
"""
import argparse
import json
import os
import wandb

DIR = "research/validity/specbreak_reconvergence"
REPORT = f"{DIR}/reconvergence_report.json"
CLASSIFY_REPORT = f"{DIR}/shift_edit_classification.json"
MANIFEST = f"{DIR}/fork_risk_manifest.jsonl"
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


def flat_classify_scalars(d):
    """Flatten shift_edit_classification.json into a flat scalar dict."""
    out = {}
    SHIFT_KEYS = ["n", "cosmetic", "content", "cosmetic_edit_frac", "median_excursion"]
    FORK_KEYS = ["n", "content_fork", "format_length_fork",
                 "permfork_content_divergence_frac"]
    EDIT_CLASSES = ["WHITESPACE_PUNCT", "NUMBER_FORMAT", "EQUIVALENT_TOKEN", "CONTENT"]
    FORK_CLASSES = ["CONTENT_FORK", "FORMAT_LENGTH_FORK"]
    for cell, c in d["shift_edit"].items():
        pre = f"shift_edit/{cell}"
        for k in SHIFT_KEYS:
            if isinstance(c.get(k), (int, float)):
                out[f"{pre}/{k}"] = c[k]
        for kc in EDIT_CLASSES:
            out[f"{pre}/class/{kc}"] = c.get("classes", {}).get(kc, 0)
        for src, ss in c.get("per_source", {}).items():
            out[f"{pre}/{src}/cosmetic_edit_frac"] = ss["cosmetic_edit_frac"]
            out[f"{pre}/{src}/n"] = ss["n"]
    for cell, c in d["perm_fork"].items():
        pre = f"perm_fork/{cell}"
        for k in FORK_KEYS:
            if isinstance(c.get(k), (int, float)):
                out[f"{pre}/{k}"] = c[k]
        for kc in FORK_CLASSES:
            out[f"{pre}/class/{kc}"] = c.get("classes", {}).get(kc, 0)
        for src, ss in c.get("per_source", {}).items():
            out[f"{pre}/{src}/permfork_content_divergence_frac"] = \
                ss["permfork_content_divergence_frac"]
            out[f"{pre}/{src}/n"] = ss["n"]
    for cell, c in d.get("samepos_heal", {}).items():
        out[f"samepos_heal/{cell}/cosmetic_edit_frac"] = c["cosmetic_edit_frac"]
        out[f"samepos_heal/{cell}/n"] = c["n"]
    return out


def classify_main(group):
    d = json.load(open(CLASSIFY_REPORT))
    h = d["headline"]
    verdict = h["verdict"]

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        name="kanna/shift-edit-content-audit",
        group=group,
        job_type="analysis",
        config=dict(
            analysis_only=1, official_tps=0, fires=0,
            pr=689, verdict=verdict,
            model=d["model"], headline_W=d["headline_W"],
            n_prompts=128,
            set="57 mmlu_pro + 57 gpqa_diamond + 14 aime2026 (#678 scored set)",
            source_decode="specdec_official_dist_breakrate/_runs/confirm "
                          "(decode_{ar,suffix6,ngram5}_r0.jsonl)",
            inputs="#686 reconvergence_report.json (aao6xpyn) + tokenizer decode",
            cross_reads=["#686 aao6xpyn", "#685 sd3mbkdp", "#678 gnfgcn90",
                         "#680 5iy1mhe4"],
            edit_classes=d["edit_classes"],
            cosmetic_classes=d["cosmetic_classes"],
            fork_gate=d["fork_gate"],
            cosmetic_def="edit class in {WHITESPACE_PUNCT, NUMBER_FORMAT, "
                         "EQUIVALENT_TOKEN}; CONTENT = different word/number/"
                         "operator/reasoning token (surface tokenizer-level, "
                         "hand-audited for semantic-equivalents)",
            permfork_def="CONTENT_FORK vs FORMAT_LENGTH_FORK on post-divergence "
                         "tails (capped at EOS): content-word seq-ratio & Jaccard "
                         "gate, answer-match override",
            note="surface classifier slightly over-counts CONTENT (semantic "
                 "equivalents like 'methyl group'<->'CH_3', 'n=p'<->'N=Z' read "
                 "as CONTENT); even crediting hand-verified equivalents the "
                 "content fraction stays dominant -- verdict robust",
        ),
    )

    scal = flat_classify_scalars(d)
    scal.update({
        "analysis_only": 1, "official_tps": 0, "fires": 0,
        "cosmetic_edit_frac": h["cosmetic_edit_frac_pooled"],
        "cosmetic_edit_frac_dedup": h["cosmetic_edit_frac_dedup"],
        "permfork_content_divergence_frac": h["permfork_content_divergence_frac_pooled"],
        "permfork_content_divergence_frac_dedup": h["permfork_content_divergence_frac_dedup"],
        "n_shift_pooled": h["n_shift_pooled"], "n_fork_pooled": h["n_fork_pooled"],
        "n_shift_dedup": h["n_shift_dedup"], "n_fork_dedup": h["n_fork_dedup"],
    })
    wandb.log(scal)
    for k, v in scal.items():
        run.summary[k] = v
    run.summary["verdict"] = verdict

    art = wandb.Artifact("shift_edit_classification_detail", type="analysis")
    art.add_file(CLASSIFY_REPORT)
    art.add_file(MANIFEST)
    art.add_file(f"{DIR}/classify_shift_edits.py")
    run.log_artifact(art)

    print(f"WANDB_RUN_ID={run.id}")
    print(f"verdict={verdict}  cosmetic_edit_frac={h['cosmetic_edit_frac_pooled']:.4f} "
          f"(dedup {h['cosmetic_edit_frac_dedup']:.4f})  "
          f"permfork_content_divergence_frac={h['permfork_content_divergence_frac_pooled']:.4f} "
          f"(dedup {h['permfork_content_divergence_frac_dedup']:.4f})")
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--classify", action="store_true",
                    help="log PR #689 shift-edit content audit instead of #686")
    ap.add_argument("--wandb_group", default="specbreak-reconvergence-kanna")
    args = ap.parse_args()
    if args.classify:
        classify_main(args.wandb_group)
    else:
        main()
