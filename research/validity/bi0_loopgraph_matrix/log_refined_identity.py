"""Log the REFINED greedy-identity decomposition for the bi0-loopgraph matrix (PR #771).

Companion to the raw matrix run (W&B kanna/bi0-loopgraph-matrix). The raw run scored
greedy identity rep0-inclusive (control self-divergence 106/128, variant-vs-control
7/128). This script recomputes the decomposition AFTER separating the rep0 warmup
transient from the stabilized regime, across TWO independent control server launches,
and logs the corrected picture:

  * stabilized control (rep1 == rep2) is byte-exact WITHIN and ACROSS both launches
    -> bi0 greedy decode is deterministic once warmed up; the 106/128 was a rep0
       first-batch artifact, not run-to-run nondeterminism.
  * stabilized variant diverges from the deterministic control on exactly 2/128
    prompts (onset deep, tok 262-274); both are degenerate ignore_eos tails with no
    parseable answer for control OR variant -> answer-immaterial.
  * loopgraph_only == loopgraph_fused byte-exact (128/128) -> fused sparse-argmax is
    exactly argmax-preserving.
  * PPL teacher-forced NLL is bit-identical control vs variant (128/128 per record).

Read-only over the saved JSONL; LOCAL-only; no serving.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from compare_streams import compare, load  # noqa: E402
from scripts import wandb_logging  # noqa: E402

A = HERE / "run_20260620_112858"
B = HERE / "run_20260620_122114"


def ppl_records(path: Path) -> dict:
    out = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[r["id"]] = r
    return out


def main() -> int:
    # --- token streams ---
    a_c1 = load(A / "control" / "rep1.jsonl")
    a_c2 = load(A / "control" / "rep2.jsonl")
    b_c1 = load(B / "control" / "rep1.jsonl")
    b_c2 = load(B / "control" / "rep2.jsonl")
    a_only = load(A / "loopgraph_only" / "rep0.jsonl")
    a_fused0 = load(A / "loopgraph_fused" / "rep0.jsonl")
    a_fused1 = load(A / "loopgraph_fused" / "rep1.jsonl")

    cross_launch = [
        ("A_c1_vs_A_c2", compare(a_c1, a_c2)),
        ("A_c1_vs_B_c1", compare(a_c1, b_c1)),
        ("A_c2_vs_B_c2", compare(a_c2, b_c2)),
        ("B_c1_vs_B_c2", compare(b_c1, b_c2)),
    ]
    var_vs_ctrl = compare(a_c1, a_fused1)            # stabilized variant vs stabilized ctrl
    fused_vs_only = compare(a_only, a_fused0)         # rep0 vs rep0 (same warmup state)

    stabilized_ctrl_all_identical = all(c["n_divergent"] == 0 for _, c in cross_launch)

    # --- PPL bit-identity ---
    pc = ppl_records(A / "control" / "ppl.jsonl")
    pf = ppl_records(A / "loopgraph_fused" / "ppl.jsonl")
    ppl_ident = sum(
        1 for k in set(pc) & set(pf)
        if abs(pc[k]["neg_log_likelihood"] - pf[k]["neg_log_likelihood"]) < 1e-9
    )

    summary = {
        "capture_ok": 1,
        "stabilized_ctrl_cross_launch_identical": 1 if stabilized_ctrl_all_identical else 0,
        "variant_stabilized_divergent_prompts": var_vs_ctrl["n_divergent"],
        "variant_stabilized_divergent_tokens": var_vs_ctrl["total_divergent_tokens"],
        "variant_stabilized_onset_min": var_vs_ctrl["onset_min"],
        "variant_stabilized_onset_median": var_vs_ctrl["onset_median"],
        "variant_two_flips_answer_material": 0,   # both degenerate tails, ANSWER=None for both
        "fused_vs_loopgraph_only_divergent_prompts": fused_vs_only["n_divergent"],
        "ppl_control": 2.523403987549849,
        "ppl_variant": 2.523403987549849,
        "ppl_per_record_identical": ppl_ident,
        "ppl_per_record_total": len(set(pc) & set(pf)),
        "tps_delta_loopgraph_only_pct": 0.037,
        "tps_delta_loopgraph_fused_pct": 1.99,
        "tps_fused_over_only_pct": 1.95,
        "tps_run_to_run_cv_pct": 1.13,
    }
    verdict = {
        "verdict": "MARGINAL_NULL_PRIVATE_STABLE",
        "loopgraph_capture": "WORKS on stock vllm==0.22.0 + int4-Marlin + MTP K=6 (sm86)",
        "loopgraph_only_tps": "NULL (+0.037%) — stock already PIECEWISE-graphs the draft loop",
        "fused_argmax_tps": "+1.95% (the entire lever) but ~1.7sigma over 1.13% CV — marginal",
        "greedy_identity": "126/128 byte-exact stabilized; 2 BI=0 near-tie flips (answer-immaterial); fused-argmax exactly argmax-preserving",
        "ppl": "bit-identical to bi0 control (teacher-forced NLL, 128/128 per record)",
        "recommendation": "do not fire; +2% does not clear a bar over bi0@218.02; bank as closed lever",
    }

    print(json.dumps({"summary": summary, "cross_launch": dict(cross_launch),
                      "var_vs_ctrl": var_vs_ctrl, "fused_vs_only": fused_vs_only,
                      "verdict": verdict}, indent=2, default=str))

    run = wandb_logging.init_wandb_run(
        job_type="serveconfig",
        agent="kanna",
        name="kanna/bi0-loopgraph-refined-identity",
        group="bi0-loopgraph-sparseargmax",
        tags=["bi0-loopgraph", "greedy-identity", "refined-analysis", "analysis-only"],
        notes="PR #771 refined greedy-identity: stabilized cross-launch determinism + 2-flip materiality + PPL bit-identity",
        config={
            "analysis_only": True,
            "matrix_run": "wjeykst8",
            "control_submission": "int4_mtp_bi0_surgattn",
            "submission_variant": "int4_mtp_bi0_loopgraph",
            "num_prompts": 128, "output_len": 512, "seed": 1,
        },
    )
    if run is None:
        print("[refined] wandb disabled; printed summary only")
        return 0
    step = 0
    for label, c in cross_launch:
        wandb_logging.log_event(run, "cross_launch", step=step,
                                metrics={f"identity/{label}/n_divergent": c["n_divergent"],
                                         f"identity/{label}/n_identical": c["n_identical"]})
        step += 1
    wandb_logging.log_summary(run, summary, step=step)
    wandb_logging.log_json_artifact(run, name="bi0_loopgraph_refined_identity",
                                    artifact_type="serveconfig",
                                    data={"summary": summary, "verdict": verdict,
                                          "cross_launch": dict(cross_launch),
                                          "var_vs_ctrl": var_vs_ctrl,
                                          "fused_vs_only": fused_vs_only})
    print(f"[refined] logged -> {run.url}")
    wandb_logging.finish_wandb(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
