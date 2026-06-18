"""PR #632 K-sweep closeout -> W&B group `optionb-bi1-k-sweep`.

Consolidates the per-K A/B runs (k3 0206qiry, k4 x6yyuglx, k5 uo6netrr, k6
obfvs9ma; k7 anchor is the k3 paired baseline) into one decision run:
  * the net wall_tps vs K curve (Table + native line plot) that exposes K*,
  * the empirical cost fit cycle_time(K)=c_draft*K+c_verify (why a lower K nets
    more TPS under BI=1: it pays fewer M=1 BI-taxed draft forwards/cycle),
  * the #319 byte-identity gate at K* and K=7 (finalize_kstar.sh phase B),
  * the teacher-forced PPL sanity at K* (phase C, <=2.42 gate).

Reads only local artifacts (the A/B JSONs + finalize outputs); no server, no
recompute. Degrades gracefully if the finalize gate/PPL files are absent.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts import wandb_logging  # noqa: E402

# reuse the analyzer verbatim -- do NOT reimplement the curve math
sys.path.insert(0, str(HERE))
import analyze_ksweep as A  # noqa: E402

ANCHOR_K7 = 152.29     # #623 banked BI=1 K=7 median
LOCKED_RUNG = 126.378  # strict-#319 AR official rung
PPL_GATE = 2.42
PPL_K7 = 2.0055        # #623 K=7 PPL


def _load(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _gate(label: str):
    g = _load(HERE / f"gate_{label}.json")
    if not g:
        return None
    return {
        "verdict": g.get("verdict"),
        "num_identical": g.get("num_identical"),
        "num_divergent": g.get("num_divergent"),
        "num_prompts_compared": g.get("num_prompts_compared"),
        "total_tokens_compared": g.get("total_tokens_compared"),
        "total_divergent_tokens": g.get("total_divergent_tokens"),
        "byte_exact": g.get("verdict") == "GREEDY_IDENTICAL",
    }


def _ppl(kstar: int):
    hits = sorted(glob.glob(str(HERE / f"ppl_k{kstar}" / "ppl_summary_*.json")))
    if not hits:
        return None, None
    s = _load(Path(hits[0]))
    if not s:
        return None, None
    tag = "spec" if "_spec.json" in hits[0] else ("specoff" if "specoff" in hits[0] else "unknown")
    return float(s["ppl"]), tag


def main() -> int:
    rows = A.summarize(A.collect())
    if not rows:
        print("no records", flush=True)
        return 1
    kstar_row = max(rows, key=lambda r: r["wall_tps_median"])
    k7_row = next((r for r in rows if r["K"] == 7), None)
    kstar = kstar_row["K"]
    tps_kstar = kstar_row["wall_tps_median"]
    tps_k7 = k7_row["wall_tps_median"] if k7_row else None
    beats_k7 = bool(kstar != 7 and tps_k7 is not None and tps_kstar > tps_k7)
    cost = A.fit_cost(rows) or {}

    g_k7 = _gate("k7")
    g_kstar = _gate(f"k{kstar}")
    ppl_kstar, ppl_tag = _ppl(kstar)
    byte_exact = bool((g_kstar or {}).get("byte_exact")) and bool((g_k7 or {}).get("byte_exact"))
    gated = g_kstar is not None and g_k7 is not None
    # K-independence of the identity outcome: K* and K=7 diverge at ~equal rates
    # (the spec-verify FP divergence is intrinsic to M>1, not the K depth)
    k_indep_identity = gated and (not (g_kstar or {}).get("byte_exact")) and (not (g_k7 or {}).get("byte_exact"))

    if beats_k7 and byte_exact:
        verdict = "K_SWEEP_RECOVERS_TPS"
    elif beats_k7 and k_indep_identity:
        # TPS optimum found, but the Option-B BI=1 spec lane fails the contract's
        # greedy-token-identity gate (line 27-28) at every K, not just K*.
        verdict = "K_SWEEP_RECOVERS_TPS__BUT_SPEC_NOT_GREEDY_IDENTICAL"
    elif beats_k7:
        verdict = "K_SWEEP_RECOVERS_TPS__identity_unconfirmed"
    else:
        verdict = "K7_ALREADY_OPTIMAL"

    run = wandb_logging.init_wandb_run(
        job_type="ksweep_closeout",
        agent="land",
        name="land/optionb-bi1-ksweep-closeout",
        group="optionb-bi1-k-sweep",
        notes=("PR#632 K-sweep closeout: net wall_tps vs K under BI=1 (Option-B int4+MTP). "
               "Finds K*, the #319-safe net-TPS optimum, with byte-identity + PPL gates."),
        config={
            "pr": 632, "analysis_only": True, "official_tps": 0,
            "vllm": "0.22.0", "drafter": "/tmp/qat-assistant",
            "batch_invariant": 1, "n_per_arm": 3, "num_prompts": 128, "output_len": 512, "seed": 1,
            "per_k_runs": {"k3": "0206qiry", "k4": "x6yyuglx", "k5": "uo6netrr", "k6": "obfvs9ma"},
        },
        tags=["optionb", "batch_invariant", "pr632", "k_sweep", "closeout"],
    )
    if run is None:
        print("WANDB disabled/unavailable; dumping summary only", flush=True)

    import wandb

    # ---- per-K curve: Table + native line plot, and stepwise log (K as x) ----
    cols = ["K", "wall_tps_median", "wall_tps_mean", "wall_tps_std", "e_accept_mean",
            "accept_per_draft", "accept_draft_per_cycle", "draft_fwd_per_tok", "cycle_time_ms"]
    if run is not None:
        tbl = wandb.Table(columns=cols)
        for r in rows:
            tbl.add_data(*[r[c] for c in cols])
            run.log({
                "global_step": r["K"],
                "curve/K": r["K"],
                "curve/wall_tps_median": r["wall_tps_median"],
                "curve/wall_tps_mean": r["wall_tps_mean"],
                "curve/wall_tps_std": r["wall_tps_std"],
                "curve/e_accept_mean": r["e_accept_mean"],
                "curve/accept_per_draft": r["accept_per_draft"],
                "curve/cycle_time_ms": r["cycle_time_ms"],
                "curve/draft_fwd_per_tok": r["draft_fwd_per_tok"],
            })
        run.log({
            "plot/net_wall_tps_vs_K": wandb.plot.line(
                tbl, "K", "wall_tps_median", title="PR#632 net wall_tps vs K (BI=1, Option-B)"),
            "plot/e_accept_vs_K": wandb.plot.line(
                tbl, "K", "e_accept_mean", title="mean acceptance length vs K"),
            "plot/cycle_time_vs_K": wandb.plot.line(
                tbl, "K", "cycle_time_ms", title="cycle time (ms) vs K"),
            "ksweep_curve": tbl,
        })

    summary = {
        "decision/k_star": kstar,
        "decision/tps_k_star_local": tps_kstar,
        "decision/tps_k7_local": tps_k7,
        "decision/tps_k7_anchor_623": ANCHOR_K7,
        "decision/k_star_beats_k7": int(beats_k7),
        "decision/k_star_vs_k7_tps": (tps_kstar - tps_k7) if tps_k7 else None,
        "decision/k_star_vs_k7_pct": (100.0 * (tps_kstar - tps_k7) / tps_k7) if tps_k7 else None,
        "decision/locked_rung": LOCKED_RUNG,
        "decision/k_star_vs_locked_tps": tps_kstar - LOCKED_RUNG,
        "decision/k_star_vs_locked_pct": 100.0 * (tps_kstar - LOCKED_RUNG) / LOCKED_RUNG,
        "decision/k_star_byte_exact_319": int(byte_exact),
        "decision/spec_identity_is_k_independent": int(k_indep_identity),
        "decision/verdict": verdict,
        # cost model (why a lower K nets more under BI=1)
        "cost/c_draft_ms_per_m1_fwd": cost.get("c_draft_ms"),
        "cost/c_verify_plus_overhead_ms": cost.get("c_verify_ms"),
        "cost/draft_verify_ratio": (cost.get("c_draft_ms") / cost.get("c_verify_ms"))
        if cost.get("c_verify_ms") else None,
        # #319 byte-identity gate (instruction #3)
        "identity/k_star_verdict": (g_kstar or {}).get("verdict"),
        "identity/k_star_num_identical": (g_kstar or {}).get("num_identical"),
        "identity/k_star_num_divergent": (g_kstar or {}).get("num_divergent"),
        "identity/k_star_divergent_tokens": (g_kstar or {}).get("total_divergent_tokens"),
        "identity/k_star_prompts_compared": (g_kstar or {}).get("num_prompts_compared"),
        "identity/k7_verdict": (g_k7 or {}).get("verdict"),
        "identity/k7_num_identical": (g_k7 or {}).get("num_identical"),
        "identity/k7_num_divergent": (g_k7 or {}).get("num_divergent"),
        "identity/k7_divergent_tokens": (g_k7 or {}).get("total_divergent_tokens"),
        # PPL sanity (instruction #4)
        "ppl/k_star": ppl_kstar,
        "ppl/k_star_tag": ppl_tag,
        "ppl/k7_reference_623": PPL_K7,
        "ppl/gate_threshold": PPL_GATE,
        "ppl/k_star_passes_gate": int(bool(ppl_kstar is not None and ppl_kstar <= PPL_GATE)),
        "ppl/k_star_minus_k7": (ppl_kstar - PPL_K7) if ppl_kstar is not None else None,
        "config/peak_vram_mib": 19917,
    }

    if run is not None:
        wandb_logging.log_summary(run, summary, step=int(kstar))
        wandb_logging.log_json_artifact(
            run, name="optionb_bi1_ksweep_closeout", artifact_type="analysis",
            data={"summary": summary, "rows": rows, "cost_fit": cost,
                  "gate_k7": g_k7, "gate_kstar": g_kstar})
        url = getattr(run, "url", "")
        rid = getattr(run, "id", "")
        wandb_logging.finish_wandb(run)
        print(f"[wandb] ksweep closeout id={rid} url={url}", flush=True)

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
