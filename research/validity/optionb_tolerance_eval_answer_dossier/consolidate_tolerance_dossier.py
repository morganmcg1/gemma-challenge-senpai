#!/usr/bin/env python
"""PR #641 — τ=0.3-nat tolerance-#319 ANSWER dossier (consolidation / analysis-only).

The single cross-suite, answer-level statement Morgan needs to rule on the
strict-vs-tolerance #319 contract (open #481 identity question):

  Under a τ=0.3-nat tolerance contract, does Option-B (int4 body + MTP-K7 spec)
  emit the SAME graded answer as the strict int4-AR rung on every benchmark
  question, or here is the exact count + which ones differ?

This is a CONSOLIDATION card. It reuses the denken #626 (W&B `bj8d88gf`) GREEDY
matched-arm paired per-question streams that live on the advisor branch under
`research/validity/optionb_319_answer_materiality/results/` (spec_*.jsonl /
ar_*.jsonl / gaps_*.jsonl). It does NOT re-run any eval.

What it ADDS over #626:
  * #626 flagged "large-margin answer flips" at the 0.5-nat NEARTIE threshold and
    found 0. This card re-cuts the residual at the EXACT τ=0.3-nat tolerance the
    competition #481 contract would use, at TWO answer-levels:
      - extracted-CHOICE residual  : answer-divergent items whose ROOT token
                                      divergence gap > τ (or spec outside AR top-k)
      - grade-FLIP residual        : accuracy-MOVING (one-correct-one-wrong)
                                      items whose root gap > τ  [decision-critical]
  * Cross-suite headline table + the single decision sentence for Morgan.

GPQA + GSM8K are MEASURED (paired int4-AR stream exists). MMLU-Pro + AIME have NO
paired int4-AR stream on the branch and the #626 harness only builds gpqa/gsm8k
items, so producing them needs net-new evalset code + a fresh paired generation
run — scoped OUT and flagged for routing per the PR's explicit instruction.

ANALYSIS-ONLY. No GPU. analysis_only=True, official_tps=0.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "research/validity/optionb_319_answer_materiality/results"  # denken #626 paired data
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # so `from scripts import wandb_logging` resolves

# #620/#626 paired-stats backbone (same machinery denken used; on-branch).
sys.path.insert(0, str(ROOT / "research/validity/spec_distribution_preservation_matched_arm"))
from analyze_matched_arm import mcnemar, cluster_bootstrap  # noqa: E402

TAU = 0.3          # the #481 tolerance contract under test
NEARTIE = 0.5      # #616/#626 grid-tie band (cross-check)
MEASURED = ["gpqa", "gsm8k"]
SCOPED_OUT = {
    "mmlu_pro": "no paired int4-AR greedy stream on branch; #626 harness builds only "
                "gpqa/gsm8k items. Option-B-only greedy gb6144: 0.664 (0p22) / 0.664 (dev307).",
    "aime": "no paired int4-AR greedy stream on branch; long free-form maj@1 reasoning has "
            "the largest cascade surface. Option-B-only greedy maj@1 gb6144: 0.3667 (0p22) / 0.400 (dev307).",
}


def load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[str(r["id"])] = r
    return out


def _residual_at_tau(ids: list[str], gaps: dict[str, dict], tau: float) -> tuple[list[str], list[str]]:
    """Split a set of item ids into (rescued_at_tau, residual_survives_tau) by the
    root-divergence gap. Residual = gap > tau OR spec token outside AR top-k (decisive)."""
    rescued, residual = [], []
    for iid in ids:
        g = gaps.get(iid, {})
        gap = g.get("gap_ar_minus_spec")
        outside = bool(g.get("spec_outside_topk"))
        if outside or (gap is not None and float(gap) > tau):
            residual.append(iid)
        else:
            rescued.append(iid)
    return rescued, residual


def analyze(kind: str) -> dict[str, Any]:
    spec = load_jsonl(SRC / f"spec_{kind}.jsonl")
    ar = load_jsonl(SRC / f"ar_{kind}.jsonl")
    gaps = load_jsonl(SRC / f"gaps_{kind}.jsonl")
    common = sorted(set(spec) & set(ar))

    clean: list[str] = []
    spec_c: list[int] = []
    ar_c: list[int] = []
    answer_div: list[str] = []     # extracted-choice differs
    grade_disc: list[str] = []     # accuracy-moving (one correct, one wrong)
    sha_mismatch = 0
    for iid in common:
        s, a = spec[iid], ar[iid]
        ssha, asha = s.get("prompt_sha256"), a.get("prompt_sha256")
        if ssha and asha and ssha != asha:
            sha_mismatch += 1
            continue
        if s.get("error") or a.get("error"):
            continue
        clean.append(iid)
        sc, ac = int(bool(s.get("correct"))), int(bool(a.get("correct")))
        spec_c.append(sc)
        ar_c.append(ac)
        if s.get("answer") != a.get("answer"):
            answer_div.append(iid)
        if sc != ac:
            grade_disc.append(iid)

    n = len(clean)
    n_tok_div = sum(1 for i in clean if gaps.get(i, {}).get("divergent"))

    # τ=0.3 residuals at the two answer-levels.
    _, choice_resid_tau = _residual_at_tau(answer_div, gaps, TAU)
    _, grade_resid_tau = _residual_at_tau(grade_disc, gaps, TAU)
    # cross-check at the #626 0.5-nat band.
    _, choice_resid_05 = _residual_at_tau(answer_div, gaps, NEARTIE)
    _, grade_resid_05 = _residual_at_tau(grade_disc, gaps, NEARTIE)

    # net graded delta (spec - ar) with cluster bootstrap CI (each question = 1 cluster).
    cb = cluster_bootstrap(np.array(clean), np.array(spec_c), np.array(ar_c))
    mc = mcnemar(list(zip(spec_c, ar_c)))
    delta_ci = cb["delta_ci95"]
    delta_ci_contains_0 = bool(delta_ci[0] <= 0.0 <= delta_ci[1])

    # detail rows for any τ=0.3 survivor (so Morgan sees exactly which question & direction).
    def detail(iid: str) -> dict[str, Any]:
        s, a, g = spec[iid], ar[iid], gaps.get(iid, {})
        return {
            "id": iid, "gold": s.get("target"),
            "spec_answer": s.get("answer"), "ar_answer": a.get("answer"),
            "spec_correct": bool(s.get("correct")), "ar_correct": bool(a.get("correct")),
            "root_gap_nat": g.get("gap_ar_minus_spec"), "spec_outside_topk": bool(g.get("spec_outside_topk")),
            "first_div_index": g.get("first_div_index"),
            "grade_changed": bool(s.get("correct")) != bool(a.get("correct")),
        }

    return {
        "kind": kind,
        "n_questions": n,
        "n_prompt_sha_mismatch": sha_mismatch,
        "n_token_divergent_items": n_tok_div,
        "n_answer_diffs_choice_strict": len(answer_div),
        "answer_div_rate_strict": len(answer_div) / n if n else float("nan"),
        "n_grade_flips_strict": len(grade_disc),
        # τ=0.3 survivors
        "n_choice_residual_tau0p3": len(choice_resid_tau),
        "n_grade_residual_tau0p3": len(grade_resid_tau),
        # 0.5-nat cross-check (should match #626's "0 large-margin flips")
        "n_choice_residual_0p5nat": len(choice_resid_05),
        "n_grade_residual_0p5nat": len(grade_resid_05),
        # rescued counts (the tolerance does its job)
        "n_choice_rescued_tau0p3": len(answer_div) - len(choice_resid_tau),
        "n_grade_rescued_tau0p3": len(grade_disc) - len(grade_resid_tau),
        # net quality
        "spec_acc": cb["spec_acc"], "ar_acc": cb["ar_acc"],
        "net_graded_delta_spec_minus_ar": cb["delta"],
        "net_graded_delta_ci95": delta_ci,
        "net_delta_ci_contains_0": delta_ci_contains_0,
        "mcnemar_b_spec_better": mc["b"], "mcnemar_c_ar_better": mc["c"], "mcnemar_p_exact": mc["p_exact"],
        "choice_residual_tau0p3_detail": [detail(i) for i in choice_resid_tau],
        "grade_residual_tau0p3_detail": [detail(i) for i in grade_resid_tau],
        # one-word verdict for the table
        "verdict": (
            "ANSWER_IDENTICAL" if len(answer_div) == 0 else
            "TOLERANCE_RESCUES_ALL" if len(grade_resid_tau) == 0 and delta_ci_contains_0 else
            "RESIDUAL_REAL_DIFFS"
        ),
    }


def pooled(evals: dict[str, dict]) -> dict[str, Any]:
    # rebuild pooled clean arrays from source for an honest pooled bootstrap.
    cl_ids: list[str] = []
    sp: list[int] = []
    arr: list[int] = []
    for kind in MEASURED:
        spec = load_jsonl(SRC / f"spec_{kind}.jsonl")
        ar = load_jsonl(SRC / f"ar_{kind}.jsonl")
        for iid in sorted(set(spec) & set(ar)):
            s, a = spec[iid], ar[iid]
            ssha, asha = s.get("prompt_sha256"), a.get("prompt_sha256")
            if ssha and asha and ssha != asha:
                continue
            if s.get("error") or a.get("error"):
                continue
            cl_ids.append(f"{kind}:{iid}")
            sp.append(int(bool(s.get("correct"))))
            arr.append(int(bool(a.get("correct"))))
    cb = cluster_bootstrap(np.array(cl_ids), np.array(sp), np.array(arr))
    tot = lambda key: sum(evals[k][key] for k in MEASURED)  # noqa: E731
    return {
        "n_questions": tot("n_questions"),
        "n_token_divergent_items": tot("n_token_divergent_items"),
        "n_answer_diffs_choice_strict": tot("n_answer_diffs_choice_strict"),
        "n_grade_flips_strict": tot("n_grade_flips_strict"),
        "total_choice_residual_tau0p3": tot("n_choice_residual_tau0p3"),
        "total_grade_residual_tau0p3": tot("n_grade_residual_tau0p3"),
        "total_choice_residual_0p5nat": tot("n_choice_residual_0p5nat"),
        "total_grade_residual_0p5nat": tot("n_grade_residual_0p5nat"),
        "net_graded_delta_pooled": cb["delta"],
        "net_graded_delta_pooled_ci95": cb["delta_ci95"],
        "net_delta_pooled_ci_contains_0": bool(cb["delta_ci95"][0] <= 0.0 <= cb["delta_ci95"][1]),
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wandb_name", default="fern/optionb-tolerance-eval-answer-dossier")
    ap.add_argument("--wandb_group", default="optionb-tolerance-eval-answer-dossier")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    evals = {kind: analyze(kind) for kind in MEASURED}
    pl = pooled(evals)

    # decision logic: graded-outcome answer-safe iff NO accuracy-moving flip survives τ=0.3
    # on any measured benchmark AND every net-delta CI contains 0.
    graded_safe = (pl["total_grade_residual_tau0p3"] == 0
                   and all(evals[k]["net_delta_ci_contains_0"] for k in MEASURED))
    result = {
        "pr": 641,
        "analysis_only": True,
        "official_tps": 0,
        "tau_nat": TAU,
        "source": "denken #626 (W&B bj8d88gf) on-branch paired greedy streams; reused, not re-run",
        "design": "GREEDY matched-arm paired (spec=int4+MTP-K7 ON M=8, ar=int4 spec OFF M=1, "
                  "same int4 body, vllm==0.22.0, BI=1, MAX_NUM_SEQS=1 serial)",
        "measured": evals,
        "scoped_out": SCOPED_OUT,
        "pooled": pl,
        "primary_metric_total_answer_diffs_at_tau0p3": pl["total_choice_residual_tau0p3"],
        "decision_grade_residual_tau0p3": pl["total_grade_residual_tau0p3"],
        "net_graded_delta_pooled": pl["net_graded_delta_pooled"],
        "verdict_graded": "TOLERANCE_ANSWER_SAFE_GRADED" if graded_safe else "TOLERANCE_RESIDUAL_RISK",
    }
    (OUT / "dossier.json").write_text(json.dumps(result, indent=2, default=str))

    # ---- human-readable headline table ----
    L = []
    L.append("τ=0.3-nat TOLERANCE-#319 ANSWER DOSSIER (PR #641) — Option-B (int4+spec) vs strict int4-AR")
    L.append(f"source: {result['source']}")
    L.append(f"design: {result['design']}")
    L.append("")
    hdr = ("bench", "n_q", "tok_div", "ans_diffs(strict)", "grade_flips(strict)",
           "choice_resid@τ0.3", "grade_resid@τ0.3", "net_Δ(spec-ar)", "verdict")
    L.append("  ".join(f"{h:>18}" if i else f"{h:<8}" for i, h in enumerate(hdr)))
    for kind in MEASURED:
        e = evals[kind]
        ci = e["net_graded_delta_ci95"]
        row = (kind, e["n_questions"], e["n_token_divergent_items"],
               f"{e['n_answer_diffs_choice_strict']} ({e['answer_div_rate_strict']:.1%})",
               e["n_grade_flips_strict"], e["n_choice_residual_tau0p3"], e["n_grade_residual_tau0p3"],
               f"{e['net_graded_delta_spec_minus_ar']:+.4f}[{ci[0]:+.3f},{ci[1]:+.3f}]", e["verdict"])
        L.append("  ".join(f"{str(c):>18}" if i else f"{str(c):<8}" for i, c in enumerate(row)))
    ci = pl["net_graded_delta_pooled_ci95"]
    prow = ("POOLED", pl["n_questions"], pl["n_token_divergent_items"],
            f"{pl['n_answer_diffs_choice_strict']}", pl["n_grade_flips_strict"],
            pl["total_choice_residual_tau0p3"], pl["total_grade_residual_tau0p3"],
            f"{pl['net_graded_delta_pooled']:+.4f}[{ci[0]:+.3f},{ci[1]:+.3f}]",
            result["verdict_graded"])
    L.append("  ".join(f"{str(c):>18}" if i else f"{str(c):<8}" for i, c in enumerate(prow)))
    L.append("")
    L.append("SCOPED OUT (no paired int4-AR stream; needs net-new harness + fresh gen — flag for routing):")
    for k, why in SCOPED_OUT.items():
        L.append(f"  - {k}: {why}")
    L.append("")
    for kind in MEASURED:
        for d in evals[kind]["choice_residual_tau0p3_detail"]:
            L.append(f"τ=0.3 choice-residual [{kind}] {d['id']}: gold={d['gold']} "
                     f"spec={d['spec_answer']}({'✓' if d['spec_correct'] else '✗'}) "
                     f"ar={d['ar_answer']}({'✓' if d['ar_correct'] else '✗'}) "
                     f"root_gap={d['root_gap_nat']} nat  grade_changed={d['grade_changed']}")
    L.append("")
    L.append(f"PRIMARY total_answer_diffs_at_tau0p3 (extracted-choice survivors) = {result['primary_metric_total_answer_diffs_at_tau0p3']}")
    L.append(f"DECISION grade-flip survivors @τ0.3 (accuracy-moving)            = {result['decision_grade_residual_tau0p3']}")
    L.append(f"TEST net_graded_delta_pooled (spec-ar)                          = {result['net_graded_delta_pooled']:+.5f} "
             f"CI95 {pl['net_graded_delta_pooled_ci95']}")
    L.append(f"VERDICT (graded) = {result['verdict_graded']}")
    report = "\n".join(L)
    (OUT / "dossier_report.txt").write_text(report + "\n")
    print(report, flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(result)
        except Exception as exc:  # noqa: BLE001
            print(f"[dossier] WARNING wandb logging failed ({type(exc).__name__}: {exc}); "
                  f"json preserved at {OUT/'dossier.json'}", flush=True)
    return 0


def _log_wandb(result: dict[str, Any]) -> None:
    from scripts import wandb_logging as wl

    cfg = {"pr": 641, "tau_nat": TAU, "analysis_only": True, "official_tps": 0,
           "stack": "vllm==0.22.0", "design": result["design"], "source": result["source"]}
    run = wl.init_wandb_run(
        job_type="optionb-tolerance-answer-dossier", agent="fern",
        name="fern/optionb-tolerance-eval-answer-dossier",
        group="optionb-tolerance-eval-answer-dossier",
        notes="PR641 cross-suite answer-level: does a τ=0.3-nat tolerance #319 contract change any graded answer?",
        tags=["pr641", "tolerance", "tau0p3", "specdec", "answer-materiality", "option-b", "319", "consolidation"],
        config=cfg,
    )
    if run is None:
        print("[dossier] wandb not configured (no API key/mode) — skipping", flush=True)
        return
    metrics: dict[str, Any] = {}
    for kind, e in result["measured"].items():
        metrics.update(wl.flatten_numeric(kind, {k: v for k, v in e.items()
                       if isinstance(v, (int, float, bool))}))
    metrics.update(wl.flatten_numeric("pooled", {k: v for k, v in result["pooled"].items()
                   if isinstance(v, (int, float, bool))}))
    metrics["primary/total_answer_diffs_at_tau0p3"] = result["primary_metric_total_answer_diffs_at_tau0p3"]
    metrics["decision/grade_residual_tau0p3"] = result["decision_grade_residual_tau0p3"]
    metrics["test/net_graded_delta_pooled"] = result["net_graded_delta_pooled"]
    metrics["analysis_only"] = 1
    metrics["official_tps"] = 0
    wl.log_event(run, "dossier_complete", step=0, metrics=metrics)
    for k, v in metrics.items():
        run.summary[k] = v
    run.summary["verdict_graded"] = result["verdict_graded"]
    run.summary["analysis_only"] = True
    run.summary["official_tps"] = 0
    wl.log_json_artifact(run, name="pr641_tolerance_answer_dossier",
                         artifact_type="answer-materiality", data=result)
    wl.finish_wandb(run)
    print("[dossier] wandb logged", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
