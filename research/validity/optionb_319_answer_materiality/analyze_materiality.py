#!/usr/bin/env python
"""PR #626 — GREEDY answer-materiality analysis (spec vs AR, same int4 body).

Consumes the paired greedy decode jsonls written by gen_paired_greedy.py and answers
the load-bearing question: do the ~0.43% residual greedy token-flips (#616 int4-Marlin
M=8-verify grid-ties) ever change a final extracted EVAL ANSWER, and if so, does that
systematically move quality?

Three measurements (PR #626):
  (a) TOKEN-level greedy divergence — the official greedy_identity verifier path
      (compare_files) gives the canonical per-prompt divergence; we report BOTH the
      naive cascade-amplified divergent-item fraction (#607-style) AND the un-amplified
      matched-prefix per-step HAZARD rate (n_diverged / sum first-div-or-full-len),
      which is the number that corroborates #616's ~0.43% structural flip rate.
  (b) ANSWER-level divergence [HEADLINE] — frac of paired items whose extracted answer
      differs between arms, per eval.
  (c) NET quality consequence — signed McNemar (spec-wins b vs ar-wins c) + cluster
      bootstrap on the paired accuracy delta (spec-ar) with CI. Each greedy item is its
      own cluster (deterministic; no seed correlation).
  (d) FLIP CHARACTER — from the AR-side prompt_logprobs gap probe at each item's first
      cross-arm divergence: frac of flips that are <0.5-nat near-ties, the tau=0.3-nat
      relaxed-acceptor rescue fraction, and a FLAG for any LARGE-margin answer-flipping
      divergence (an answer-divergent item whose ROOT token-divergence was NOT a grid-tie).

Verdict:
  RESIDUAL_ANSWER_IMMATERIAL  -> net paired delta CI contains 0 on BOTH evals AND no
                                 large-margin answer-flipping divergence. The residual
                                 greedy flips are grid-tie coin-flips in the CoT that
                                 never systematically move a graded answer; firing option
                                 B costs ONLY the strict-#319 byte-exact greedy contract.
  RESIDUAL_FLIPS_ANSWERS      -> some eval's delta CI excludes 0, OR a large-margin
                                 structural flip changed an answer. The residual is NOT
                                 answer-immaterial.

Backbone (mcnemar / cluster_bootstrap / cluster_level_paired_diff) reused verbatim from
the #620 matched-arm analyzer so the paired statistics are identical machinery.

ANALYSIS-ONLY. No GPU. analysis_only=True, official_tps=0.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# #620 paired-stats backbone (pure numpy/math; importing only defines functions).
sys.path.insert(0, str(ROOT / "research/validity/spec_distribution_preservation_matched_arm"))
from analyze_matched_arm import (  # noqa: E402
    mcnemar,
    cluster_bootstrap,
    cluster_level_paired_diff,
)

# official greedy-identity verifier (canonical token-divergence read).
import importlib.util  # noqa: E402

_GI_PATH = (
    ROOT / "official/main_bucket/shared_resources/"
    "gemma_greedy_identity_verifier_flowian-powers/greedy_identity.py"
)
_spec = importlib.util.spec_from_file_location("greedy_identity", _GI_PATH)
gi = importlib.util.module_from_spec(_spec)
sys.modules["greedy_identity"] = gi  # dataclass needs the module registered before exec
_spec.loader.exec_module(gi)

# A flip is a near-tie (a small relaxed-acceptor tolerance could rescue it) if the
# AR-side gap between the AR token and the spec token is small (#616 NEARTIE_NATS).
NEARTIE_NATS = 0.5
TAU_RESCUE = 0.3  # #616 rescued 100% of int4-grid flips at tau=0.3


# --------------------------------------------------------------------------- IO
def load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[str(r["id"])] = r
    return out


def _arm_path(arm: str, kind: str) -> Path:
    return RES / f"{arm}_{kind}.jsonl"


# --------------------------------------------------------------------------- (a) tokens
def token_divergence(kind: str, clean_ids: set[str]) -> dict[str, Any]:
    """Canonical greedy_identity verdict (ref=ar, cand=spec) PLUS the un-cascaded
    matched-prefix hazard rate, restricted to the CLEAN paired items."""
    ar_path, spec_path = _arm_path("ar", kind), _arm_path("spec", kind)
    if not ar_path.exists() or not spec_path.exists():
        return {"available": False}
    report = gi.compare_files(ar_path, spec_path)  # ref=ar greedy, cand=spec
    # per-prompt rows restricted to clean (non-errored, both-present) items.
    rows = [p for p in report.per_prompt if p.key in clean_ids]
    n_items = len(rows)
    n_div_items = sum(1 for p in rows if not p.identical)
    tot_tokens = sum(p.num_compared for p in rows)
    tot_div_tokens = sum(p.num_divergent_tokens for p in rows)
    # matched-prefix hazard: each item contributes its first-divergence index (or full
    # compared length if identical) as the matched run, and 1 "event" iff it diverged.
    # hazard = events / matched-prefix-tokens-at-risk == per-step P(first flip), no cascade.
    risk_tokens = 0
    onsets: list[int] = []
    for p in rows:
        if p.identical:
            risk_tokens += p.num_compared
        else:
            fdi = p.first_divergence_index if p.first_divergence_index is not None else 0
            risk_tokens += fdi + 1  # the matched prefix plus the flipping step itself
            onsets.append(fdi)
    hazard = (n_div_items / risk_tokens) if risk_tokens else float("nan")
    return {
        "available": True,
        "verdict_official": report.verdict,            # expected DIVERGENT
        "n_items": n_items,
        "n_divergent_items": n_div_items,
        "frac_divergent_items_cascade": (n_div_items / n_items) if n_items else float("nan"),
        "total_tokens_compared": tot_tokens,
        "total_divergent_tokens": tot_div_tokens,
        "naive_token_div_rate_cascade": (tot_div_tokens / tot_tokens) if tot_tokens else float("nan"),
        "matched_prefix_hazard_rate": hazard,          # the #616-comparable per-step rate
        "risk_tokens": risk_tokens,
        "first_div_onset_median": int(statistics.median(onsets)) if onsets else None,
        "first_div_onset_min": min(onsets) if onsets else None,
        "n_integrity_failures": len(report.integrity_failures),
        "n_missing_in_spec": len(report.missing_in_candidate),
        "n_missing_in_ar": len(report.missing_in_reference),
    }


# --------------------------------------------------------------------------- (d) gaps
def load_gaps(kind: str) -> dict[str, dict]:
    return load_jsonl(RES / f"gaps_{kind}.jsonl")


def flip_character(kind: str, gaps: dict[str, dict]) -> dict[str, Any]:
    """Distribution of the AR-side first-divergence logit gap (ar_tok vs spec_tok).
    Small gap == int4 grid-tie (rescuable); large gap == structural M-induced flip."""
    probed = [g for g in gaps.values()
              if g.get("divergent") and g.get("gap_ar_minus_spec") is not None]
    g_vals = [float(g["gap_ar_minus_spec"]) for g in probed]
    outside = sum(1 for g in gaps.values() if g.get("spec_outside_topk"))
    n = len(g_vals)
    frac_under = (sum(1 for g in g_vals if g < NEARTIE_NATS) / n) if n else None
    frac_rescued_tau = (sum(1 for g in g_vals if g <= TAU_RESCUE) / n) if n else None
    qs = sorted(g_vals)

    def q(p):
        if not qs:
            return None
        if len(qs) == 1:
            return qs[0]
        pos = p * (len(qs) - 1)
        lo = int(math.floor(pos)); hi = min(lo + 1, len(qs) - 1)
        return qs[lo] * (1 - (pos - lo)) + qs[hi] * (pos - lo)

    return {
        "n_divergent_with_probe": n,
        "frac_flips_under_0p5nat": frac_under,
        "frac_flips_rescued_at_tau_0p3nat": frac_rescued_tau,
        "n_spec_tok_outside_topk": outside,
        "gap_median": q(0.5),
        "gap_p90": q(0.9),
        "gap_p99": q(0.99),
        "gap_max": max(g_vals) if g_vals else None,
        "gap_mean": statistics.fmean(g_vals) if g_vals else None,
    }


# --------------------------------------------------------------------------- pairing
def pair_eval(kind: str) -> dict[str, Any]:
    spec = load_jsonl(_arm_path("spec", kind))
    ar = load_jsonl(_arm_path("ar", kind))
    gaps = load_gaps(kind)
    common = sorted(set(spec) & set(ar))

    sha_mismatch: list[str] = []
    spec_err: set[str] = set()
    ar_err: set[str] = set()
    clean: list[str] = []
    answer_div_ids: list[str] = []
    cluster_ids: list[str] = []
    spec_c: list[int] = []
    ar_c: list[int] = []

    for iid in common:
        s, a = spec[iid], ar[iid]
        if s.get("error"):
            spec_err.add(iid)
        if a.get("error"):
            ar_err.add(iid)
        # prompt_sha gate — identical by construction (same tokenized prompt to both arms).
        ssha, asha = s.get("prompt_sha256"), a.get("prompt_sha256")
        if ssha is not None and asha is not None and ssha != asha:
            sha_mismatch.append(iid)
            continue
        if s.get("error") or a.get("error"):
            continue
        clean.append(iid)
        cluster_ids.append(iid)
        spec_c.append(1 if s.get("correct") else 0)
        ar_c.append(1 if a.get("correct") else 0)
        if s.get("answer") != a.get("answer"):
            answer_div_ids.append(iid)

    clean_set = set(clean)
    tok = token_divergence(kind, clean_set)
    fc = flip_character(kind, gaps)

    # (c) net quality consequence on clean pairs.
    pairs = list(zip(spec_c, ar_c))
    mc = mcnemar(pairs)
    cb = (cluster_bootstrap(np.array(cluster_ids), np.array(spec_c), np.array(ar_c))
          if spec_c else {})
    cl = (cluster_level_paired_diff(np.array(cluster_ids), np.array(spec_c), np.array(ar_c))
          if spec_c else {})

    # (b) headline answer divergence.
    n_clean = len(clean)
    answer_div_rate = (len(answer_div_ids) / n_clean) if n_clean else float("nan")

    # large-margin answer-flips: an answer-divergent item whose ROOT (first) token
    # divergence was NOT a near-tie (gap >= NEARTIE_NATS) or whose spec token fell
    # outside the AR top-k (decisive). These would be structural, not grid-tie.
    large_margin_answer_flips = []
    for iid in answer_div_ids:
        g = gaps.get(iid, {})
        gap = g.get("gap_ar_minus_spec")
        outside = g.get("spec_outside_topk")
        if outside or (gap is not None and float(gap) >= NEARTIE_NATS):
            large_margin_answer_flips.append({
                "id": iid, "gap_ar_minus_spec": gap, "spec_outside_topk": bool(outside),
                "first_div_index": g.get("first_div_index"),
                "spec_answer": spec[iid].get("answer"), "ar_answer": ar[iid].get("answer"),
                "spec_correct": spec[iid].get("correct"), "ar_correct": ar[iid].get("correct"),
            })

    delta_ci = cb.get("delta_ci95")
    delta_ci_contains_0 = bool(delta_ci and delta_ci[0] <= 0.0 <= delta_ci[1])

    return {
        "kind": kind,
        "n_common": len(common),
        "n_clean_pairs": n_clean,
        "n_prompt_sha_mismatch": len(sha_mismatch),
        "prompt_sha_gate_pass": len(sha_mismatch) == 0,
        "errors": {
            "n_spec_errored": len(spec_err),
            "n_ar_errored": len(ar_err),
            "symmetric": spec_err == ar_err,
            "errored_examples": sorted(spec_err | ar_err)[:5],
        },
        "token_divergence": tok,
        "answer_divergence": {
            "n_answer_divergent": len(answer_div_ids),
            "answer_div_rate": answer_div_rate,
            "answer_divergent_ids": answer_div_ids[:50],
        },
        "net_quality": {
            "mcnemar": mc,
            "cluster_bootstrap": cb,
            "cluster_level_paired_diff": cl,
            "net_graded_delta_spec_minus_ar": cb.get("delta"),
            "net_graded_delta_ci95": delta_ci,
            "delta_ci_contains_0": delta_ci_contains_0,
        },
        "flip_character": fc,
        "large_margin_answer_flips": large_margin_answer_flips,
        "n_large_margin_answer_flips": len(large_margin_answer_flips),
    }


def verdict_for(ev: dict[str, Any]) -> dict[str, Any]:
    """Per-eval materiality read."""
    delta0 = ev["net_quality"]["delta_ci_contains_0"]
    n_large = ev["n_large_margin_answer_flips"]
    immaterial = bool(delta0 and n_large == 0)
    return {
        "delta_ci_contains_0": delta0,
        "n_large_margin_answer_flips": n_large,
        "verdict": "RESIDUAL_ANSWER_IMMATERIAL" if immaterial else "RESIDUAL_FLIPS_ANSWERS",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evals", default="gpqa,gsm8k")
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_group", default="optionb-319-residual-answer-materiality")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    evals = [e.strip() for e in args.evals.split(",") if e.strip()]

    meta = {}
    meta_path = RES / "gen_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    result: dict[str, Any] = {
        "pr": 626,
        "analysis_only": True,
        "official_tps": 0,
        "design": "GREEDY matched-arm paired (spec=int4+MTP-K7 ON M=8, ar=int4 spec OFF M=1, "
                  "same body, MAX_NUM_SEQS=1 serial)",
        "stack": "vllm==0.22.0",
        "decoding": {"temperature": 0.0, "min_tokens": meta.get("min_tokens", 8),
                     "max_model_len": meta.get("max_model_len", 6144),
                     "max_tokens_by_kind": meta.get("max_tokens_by_kind", {}),
                     "max_num_seqs": 1, "batch_invariant": 1},
        "gen_peaks_gb": meta.get("peaks", {}),
        "evals": {},
    }

    per_eval_verdicts = {}
    for kind in evals:
        ev = pair_eval(kind)
        ev["materiality"] = verdict_for(ev)
        result["evals"][kind] = ev
        per_eval_verdicts[kind] = ev["materiality"]["verdict"]

    # overall: IMMATERIAL only if EVERY eval is immaterial.
    all_immaterial = all(v == "RESIDUAL_ANSWER_IMMATERIAL" for v in per_eval_verdicts.values())
    result["headline_verdict"] = (
        "RESIDUAL_ANSWER_IMMATERIAL" if all_immaterial else "RESIDUAL_FLIPS_ANSWERS"
    )
    result["per_eval_verdicts"] = per_eval_verdicts

    # pooled headline numbers across both evals (single-number terminal metrics).
    pool_div_items = pool_risk = pool_div_tok = pool_tok = 0
    pool_flips_under = pool_probed = pool_outside = 0
    pool_answer_div = pool_clean = 0
    pool_large_margin = 0
    for kind in evals:
        ev = result["evals"][kind]
        td = ev["token_divergence"]
        if td.get("available"):
            pool_div_items += td["n_divergent_items"]; pool_risk += td["risk_tokens"]
            pool_div_tok += td["total_divergent_tokens"]; pool_tok += td["total_tokens_compared"]
        fc = ev["flip_character"]
        n_pr = fc["n_divergent_with_probe"]
        if n_pr:
            pool_flips_under += round((fc["frac_flips_under_0p5nat"] or 0.0) * n_pr)
            pool_probed += n_pr
        pool_outside += fc["n_spec_tok_outside_topk"]
        pool_answer_div += ev["answer_divergence"]["n_answer_divergent"]
        pool_clean += ev["n_clean_pairs"]
        pool_large_margin += ev["n_large_margin_answer_flips"]
    result["pooled"] = {
        "token_div_rate_greedy_hazard": (pool_div_items / pool_risk) if pool_risk else float("nan"),
        "naive_token_div_rate_cascade": (pool_div_tok / pool_tok) if pool_tok else float("nan"),
        "frac_flips_under_0p5nat": (pool_flips_under / pool_probed) if pool_probed else None,
        "n_divergent_probed": pool_probed,
        "n_spec_tok_outside_topk": pool_outside,
        "answer_div_rate_pooled": (pool_answer_div / pool_clean) if pool_clean else float("nan"),
        "n_answer_divergent": pool_answer_div,
        "n_clean_pairs": pool_clean,
        "n_large_margin_answer_flips_total": pool_large_margin,
    }

    # terminal SENPAI metrics (PR #626): pull the per-eval + pooled headline numbers up.
    terminal: dict[str, Any] = {
        "verdict": result["headline_verdict"],
        "token_div_rate_greedy": result["pooled"]["token_div_rate_greedy_hazard"],
        "frac_flips_under_0p5nat": result["pooled"]["frac_flips_under_0p5nat"],
    }
    for kind in evals:
        ev = result["evals"][kind]
        terminal[f"token_div_rate_greedy_{kind}"] = ev["token_divergence"].get("matched_prefix_hazard_rate")
        terminal[f"answer_div_rate_{kind}"] = ev["answer_divergence"]["answer_div_rate"]
        terminal[f"net_graded_delta_spec_minus_ar_{kind}"] = ev["net_quality"]["net_graded_delta_spec_minus_ar"]
        terminal[f"net_graded_delta_ci95_{kind}"] = ev["net_quality"]["net_graded_delta_ci95"]
        terminal[f"frac_flips_under_0p5nat_{kind}"] = ev["flip_character"]["frac_flips_under_0p5nat"]
    result["terminal_metrics"] = terminal

    (RES / "materiality_analysis.json").write_text(json.dumps(result, indent=2, default=str))

    # ---- human-readable report ----
    lines = ["GREEDY ANSWER-MATERIALITY: option-B spec vs AR, same int4 body (PR #626)",
             f"stack={result['stack']}  {result['design']}"]
    for kind in evals:
        ev = result["evals"][kind]
        td = ev["token_divergence"]; ad = ev["answer_divergence"]
        nq = ev["net_quality"]; fc = ev["flip_character"]
        cb = nq["cluster_bootstrap"]
        lines.append(f"\n=== {kind.upper()}  (n_clean={ev['n_clean_pairs']} / common={ev['n_common']}) ===")
        lines.append(f"  prompt_sha gate: {'PASS' if ev['prompt_sha_gate_pass'] else 'FAIL'}"
                     f"   serving errors: spec={ev['errors']['n_spec_errored']} ar={ev['errors']['n_ar_errored']}"
                     f" (symmetric={ev['errors']['symmetric']})")
        if td.get("available"):
            lines.append(f"  (a) TOKEN div: official={td['verdict_official']}  "
                         f"divergent-items {td['n_divergent_items']}/{td['n_items']} "
                         f"(cascade {td['frac_divergent_items_cascade']:.3%})")
            lines.append(f"      per-step HAZARD (un-cascaded) = {td['matched_prefix_hazard_rate']:.4%}"
                         f"  [#616 anchor ~0.43%]  onset median={td['first_div_onset_median']}")
        lines.append(f"  (b) ANSWER div rate = {ad['answer_div_rate']:.4%} "
                     f"({ad['n_answer_divergent']}/{ev['n_clean_pairs']})   [HEADLINE]")
        if cb:
            lines.append(f"  (c) spec acc={cb['spec_acc']:.4f} ar acc={cb['ar_acc']:.4f}  "
                         f"net delta(spec-ar)={cb['delta']:+.4f} CI95={[round(x,4) for x in cb['delta_ci95']]}"
                         f"  contains0={nq['delta_ci_contains_0']}")
        lines.append(f"      McNemar: b(spec>ar)={nq['mcnemar']['b']} c(ar>spec)={nq['mcnemar']['c']} "
                     f"p_exact={nq['mcnemar']['p_exact']:.4f}")
        lines.append(f"  (d) flips<0.5nat={fc['frac_flips_under_0p5nat']} "
                     f"(rescued@tau0.3={fc['frac_flips_rescued_at_tau_0p3nat']}, "
                     f"probed={fc['n_divergent_with_probe']}, outside_topk={fc['n_spec_tok_outside_topk']})")
        lines.append(f"      gap median={fc['gap_median']} p90={fc['gap_p90']} max={fc['gap_max']}")
        if ev["n_large_margin_answer_flips"]:
            lines.append(f"  !! LARGE-MARGIN ANSWER FLIPS: {ev['n_large_margin_answer_flips']} "
                         f"-> {ev['large_margin_answer_flips'][:3]}")
        else:
            lines.append("  large-margin answer flips: NONE")
        lines.append(f"  --> {kind} verdict: {ev['materiality']['verdict']}")
    pl = result["pooled"]
    lines.append(f"\nPOOLED: token_div_rate_greedy(hazard)={pl['token_div_rate_greedy_hazard']:.4%} "
                 f"[#616 ~0.43%]  frac_flips<0.5nat={pl['frac_flips_under_0p5nat']} "
                 f"(probed={pl['n_divergent_probed']}, outside_topk={pl['n_spec_tok_outside_topk']})  "
                 f"large-margin-answer-flips={pl['n_large_margin_answer_flips_total']}")
    lines.append(f"\nHEADLINE VERDICT: {result['headline_verdict']}")
    rep = "\n".join(lines)
    (RES / "materiality_report.txt").write_text(rep + "\n")
    print(rep, flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(result, name=args.wandb_name, group=args.wandb_group)
        except Exception as exc:  # noqa: BLE001
            print(f"[analyze] WARNING: wandb logging failed ({type(exc).__name__}: {exc}); "
                  f"report preserved at {RES/'materiality_analysis.json'}", flush=True)
    return 0


def _log_wandb(result: dict[str, Any], *, name: str | None, group: str | None) -> None:
    from scripts import wandb_logging as wl

    cfg = {
        "pr": 626, "stack": result["stack"], "design": result["design"],
        "analysis_only": True, "official_tps": 0,
        **{f"decoding/{k}": v for k, v in result["decoding"].items()
           if isinstance(v, (int, float, str))},
    }
    run = wl.init_wandb_run(
        job_type="optionb-319-answer-materiality", agent="denken",
        name=name or "denken/optionb-319-residual-answer-materiality",
        group=group or "optionb-319-residual-answer-materiality",
        notes="PR626 greedy matched-arm: do residual int4-Marlin spec flips change a graded answer?",
        tags=["pr626", "specdec", "greedy-identity", "answer-materiality", "int4-mtp", "option-b"],
        config=cfg,
    )
    if run is None:
        print("[analyze] wandb not configured (no API key/mode) — skipping", flush=True)
        return
    metrics: dict[str, Any] = {}
    for kind, ev in result["evals"].items():
        metrics.update(wl.flatten_numeric(f"{kind}/token", ev["token_divergence"]))
        metrics.update(wl.flatten_numeric(f"{kind}/answer", ev["answer_divergence"]))
        metrics.update(wl.flatten_numeric(f"{kind}/net", ev["net_quality"]))
        metrics.update(wl.flatten_numeric(f"{kind}/flip", ev["flip_character"]))
        metrics[f"{kind}/n_large_margin_answer_flips"] = ev["n_large_margin_answer_flips"]
    metrics.update(wl.flatten_numeric("pooled", result["pooled"]))
    metrics.update(wl.flatten_numeric("terminal", result["terminal_metrics"]))
    metrics["analysis_only"] = 1
    metrics["official_tps"] = 0
    wl.log_event(run, "materiality_complete", step=0, metrics=metrics)
    for k, v in metrics.items():
        run.summary[k] = v
    run.summary["headline_verdict"] = result["headline_verdict"]
    run.summary["analysis_only"] = True
    run.summary["official_tps"] = 0
    wl.log_json_artifact(run, name="pr626_materiality_report", artifact_type="answer-materiality",
                         data=result)
    wl.finish_wandb(run)
    print("[analyze] wandb logged", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
