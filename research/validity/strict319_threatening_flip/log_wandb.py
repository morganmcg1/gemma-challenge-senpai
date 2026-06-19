"""Log the PR #694 strict-#319 threatening-flip capstone to W&B.

analysis_only: this NEVER launches an HF Job, never touches a served submission
file, never measures official TPS. It logs the end-to-end answer-flip outcome of
the #689-flagged strict-#319 spec-break subset, re-decoded at the full 6144-tok
natural-EOS budget (#694 decode_arm.py), classified by classify.py.

Reads (all produced locally, analysis_only):
  outcome_summary.json   -- classify.py summary (counts + threatening_answer_flip_frac)
  outcome_table.jsonl    -- classify.py per-prompt outcome table  (-> W&B artifact)
  flagged_subset.json    -- select_subset.py provenance            (-> W&B artifact)
  verdict.json (optional)-- {verdict, reconciliation_682, notes, gold_provenance}
  _runs/confirm/decode_{ar,suffix6,ngram5}.jsonl -- decode provenance (token/finish stats)

Required by the PR as EXPLICIT top-level summary scalars:
  analysis_only=1, official_tps=0, no_hf_job=1, fires=0,
  threatening_answer_flip_frac (PRIMARY), flip_r2w_count (TEST / load-bearing),
  flip_w2r_count, flip_benign_count, preserved_count.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # research/validity/strict319_threatening_flip -> repo root
sys.path.insert(0, str(ROOT))

from scripts.wandb_logging import (  # noqa: E402
    finish_wandb,
    init_wandb_run,
    log_file_artifact,
    log_json_artifact,
    log_summary,
)

ANCHOR_MODEL = "/workspace/gemma_build/int4_g128_lmhead"  # strict anchor int4_g128_lmhead (#4)


def _decode_provenance(runs_dir: Path) -> dict:
    """Per-arm decode hygiene: how many committed / hit length / stopped on EOS."""
    prov = {}
    for arm in ("ar", "suffix6", "ngram5"):
        p = runs_dir / f"decode_{arm}.jsonl"
        if not p.exists():
            prov[arm] = {"present": False}
            continue
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        fin = Counter(r["finish_reason"] for r in recs)
        toks = [r["num_completion_tokens"] for r in recs]
        prov[arm] = {
            "present": True,
            "n_records": len(recs),
            "finish_stop": fin.get("stop", 0),
            "finish_length": fin.get("length", 0),
            "max_completion_tokens": max(toks) if toks else 0,
            "median_completion_tokens": sorted(toks)[len(toks) // 2] if toks else 0,
            "any_hit_budget": fin.get("length", 0) > 0,
            "total_decode_s": round(sum(r.get("decode_s", 0.0) for r in recs), 1),
        }
    return prov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="kanna/strict319-threatening-flip-capstone")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="strict319-threatening-flip-capstone-kanna")
    ap.add_argument("--summary", default=str(HERE / "outcome_summary.json"))
    ap.add_argument("--table", default=str(HERE / "outcome_table.jsonl"))
    ap.add_argument("--subset", default=str(HERE / "flagged_subset.json"))
    ap.add_argument("--verdict", default=str(HERE / "verdict.json"))
    ap.add_argument("--runs-dir", default=str(HERE / "_runs" / "confirm"))
    args = ap.parse_args()

    summ = json.loads(Path(args.summary).read_text())
    subset = json.loads(Path(args.subset).read_text())
    verdict = {}
    vp = Path(args.verdict)
    if vp.exists():
        verdict = json.loads(vp.read_text())
    prov = _decode_provenance(Path(args.runs_dir))

    run = init_wandb_run(
        job_type="validity-analytic",
        agent="kanna",
        name=args.wandb_name,
        group=args.wandb_group,
        notes=("PR#694 strict-#319 threatening-flip capstone: end-to-end answer "
               "outcome of #689-flagged spec breaks at 6144-tok natural EOS. "
               "analysis_only, no HF job, no served-file change, official_tps=0."),
        tags=["strict319-threatening-flip-capstone", "validity-analytic",
              "spec-break", "answer-flip", "pr694", "analysis-only",
              "capstone-685-686-689"],
        config={
            "pr": 694,
            "lineage": "685->686->689->694",
            "anchor_model": ANCHOR_MODEL,
            "served_stack": "int4_g128 + untied int4 lm_head, VLLM_BATCH_INVARIANT=1, TRITON_ATTN",
            "decode_budget_tokens": 6144,
            "decode_eos": "natural (ignore_eos=False)",
            "spec_arms": {"suffix6": {"method": "suffix", "k": 6},
                          "ngram5": {"method": "ngram", "k": 5, "lookup": [2, 6]}},
            "n_flagged": subset["n_flagged"],
            "n_content_fork": subset["n_content_fork"],
            "n_content_heal_sharp": subset["n_content_heal_sharp"],
            "by_source": subset["by_source"],
            "subset_manifest": subset.get("manifest"),
            "wandb_group": args.wandb_group,
            "imports": {
                "n689_fork_risk_manifest": "shift_edit_classification_detail:v0 (run tyzovau1)",
                "n682_aggregate_retention": "ezvgx3et (aggregate symmetric reshuffle only)",
            },
        },
    )
    if run is None:
        print("[694-log] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return 1

    # ---- summary payload (house-style summary/* via log_summary) ----
    summary = dict(summ)
    summary.update({
        # PR-mandated boundary scalars
        "analysis_only": 1,
        "official_tps": 0,
        "no_hf_job": 1,
        "fires": 0,
        # decode hygiene
        "ar_finish_length": prov.get("ar", {}).get("finish_length"),
        "suffix6_finish_length": prov.get("suffix6", {}).get("finish_length"),
        "ngram5_finish_length": prov.get("ngram5", {}).get("finish_length"),
        "ar_median_completion_tokens": prov.get("ar", {}).get("median_completion_tokens"),
        # verdict
        "verdict": verdict.get("verdict", "PENDING"),
    })
    # reconciliation scalars (if the analysis filled them)
    for k in ("n682_aggregate_flip_frac", "reconciliation_consistent"):
        if k in verdict:
            summary[k] = verdict[k]
    summary = {k: v for k, v in summary.items() if v is not None}
    log_summary(run, summary, step=0)

    # ---- ALSO set the load-bearing scalars at TOP LEVEL (no summary/ prefix) ----
    top = {
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "threatening_answer_flip_frac": summ.get("threatening_answer_flip_frac"),
        "flip_r2w_count": summ.get("flip_r2w_count"),
        "flip_w2r_count": summ.get("flip_w2r_count"),
        "flip_benign_count": summ.get("flip_benign_count"),
        "preserved_count": summ.get("preserved_count"),
        "flip_ungraded_count": summ.get("flip_ungraded_count"),
        "no_commit_count": summ.get("no_commit_count"),
        "n_flagged": summ.get("n_flagged"),
        "n_both_commit": summ.get("n_both_commit"),
        "flip_count_both_commit": summ.get("flip_count_both_commit"),
        "verdict": verdict.get("verdict", "PENDING"),
    }
    for k, v in top.items():
        if v is not None:
            run.summary[k] = v

    # ---- artifacts ----
    log_json_artifact(run, name="strict319_threatening_flip_summary",
                      artifact_type="analysis", data={"summary": summ, "verdict": verdict,
                                                       "decode_provenance": prov})
    log_json_artifact(run, name="strict319_threatening_flip_subset",
                      artifact_type="analysis", data=subset)
    tbl = Path(args.table)
    if tbl.exists():
        log_file_artifact(run, path=tbl, name="strict319_threatening_flip_outcome_table",
                          artifact_type="analysis")

    run_id = run.id
    finish_wandb(run)
    print(f"[694-log] logged run_id={run_id} group={args.wandb_group}", flush=True)
    print(json.dumps({"wandb_run_id": run_id,
                      "threatening_answer_flip_frac": summ.get("threatening_answer_flip_frac"),
                      "flip_r2w_count": summ.get("flip_r2w_count"),
                      "verdict": verdict.get("verdict", "PENDING")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
