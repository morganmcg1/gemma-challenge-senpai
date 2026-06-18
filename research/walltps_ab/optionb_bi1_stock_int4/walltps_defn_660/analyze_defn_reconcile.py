"""PR #660 wall_tps-definition reconcile analysis -> W&B `walltps-defn-reconcile-land`.

Reads walltps_defn_capture.json (from capture_defn_660.py) and prices the
un-rescued served wall-TPS under EVERY plausible definition, then tests the
advisor's hypothesis that the 9.4% gap to stark #642's un-rescued K6 (155.58) is
a full_e2e-vs-steady-state DEFINITIONAL artifact.

Definitions priced per K:
  * full_e2e_nonstream  -- canonical num_completion_tokens / duration_s (the #632
    headline; total wall incl. every per-prompt prefill + the 1 cold-start ramp).
  * steady_gen_meter    -- vLLM "Avg generation throughput" interval meter mean
    (generation-phase only; excludes prefill/warmup ramp). THE PR's "steady".
  * stream_full_e2e     -- streaming: sum(tok)/sum(per-request wall) [TTFT incl].
  * stream_steady       -- streaming: sum(tok-1)/sum(decode window) [prefill excl].
  * cold_job_wall       -- boot-inclusive: tok / (duration_s + server_ready_s).

The PR-named axis is (full_e2e vs steady). gap_residual_after_def_match = the
smallest |residual vs 155.58| achievable across the DECODE-WINDOW definitions
(full_e2e, steady_gen_meter, stream_*). If that residual stays ~9% then aligning
the named definitions does NOT close the gap -> the 9.4% is NOT that artifact.

Rescued re-price: rescued = 1/(1/U + f/A), f = per-K position-fire (#648), A =
77.962 AR rung (#658). Reported under full_e2e and steady U.

Cross-val target 155.58 (+ stark AR 77.89) is read from THIS agent's own #658
xval artifact (summary/xval/unrescued_k6_stark); stark's branch is NOT inspected.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import wandb_logging  # noqa: E402

# --- cross-val + rescue constants (all from THIS agent's #658 / #648 artifacts) ---
STARK_UNRESCUED_K6 = 155.58       # #658 summary/xval/unrescued_k6_stark (my logged xval)
STARK_AR_REF = 77.89              # #658 summary/xval/ar_ref_local_stark
LAND_AR_REF_A = 77.962            # #658 A_local_ar_rung_tps (rescue denominator A)
PPL_UNCHANGED = 2.0055            # #632/#623 spec-lane PPL (identity-preserving, K-indep)
FIRE = {6: 0.072823, 5: 0.072739, 7: 0.072906, 3: 0.072861, 4: 0.0728}   # #658 per-K τ=0.5 fire
LAND_632_UNRESCUED = {6: 170.209, 5: 172.742}   # #658 per-K un-rescued (full_e2e) reference
LAND_658_RESCUED = {6: 146.86, 5: 148.766}      # #658 rescued_local headline


def reprice(U: float | None, f: float, A: float) -> float | None:
    if U is None or U <= 0:
        return None
    return 1.0 / (1.0 / U + f / A)


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def analyze_row(row: dict[str, Any]) -> dict[str, Any]:
    K = int(row["K"])
    f = FIRE.get(K, 0.0728)
    gm = row.get("gen_meter", {}) or {}
    sp = row.get("pass2_stream", {}) or {}

    defs: dict[str, float | None] = {
        "full_e2e_nonstream": row.get("pass1_full_e2e_wall_tps"),
        "steady_gen_meter": gm.get("steady_gen_tps_mean"),
        "stream_full_e2e": sp.get("stream_full_e2e_wall_tps"),
        "stream_steady": sp.get("stream_steady_wall_tps"),
        "cold_job_wall": row.get("cold_job_wall_tps"),
    }
    decode_window_defs = ["full_e2e_nonstream", "steady_gen_meter",
                          "stream_full_e2e", "stream_steady"]

    full = defs["full_e2e_nonstream"]
    steady = defs["steady_gen_meter"]
    # PR-named axis: full_e2e vs steady-state
    named_axis_delta_pct = pct(steady, full) if (full and steady) else None

    residual_vs_stark = {
        name: (pct(v, STARK_UNRESCUED_K6) if v else None) for name, v in defs.items()
    }
    # best-case definitional alignment (decode-window only, boot excluded)
    dw_residuals = [abs(residual_vs_stark[n]) for n in decode_window_defs
                    if residual_vs_stark.get(n) is not None]
    gap_residual_after_def_match_pct = min(dw_residuals) if dw_residuals else None

    rescued_full = reprice(full, f, LAND_AR_REF_A)
    rescued_steady = reprice(steady, f, LAND_AR_REF_A)

    return {
        "K": K, "fire_rate": f,
        "server_ready_s": row.get("server_ready_s"),
        "num_completion_tokens": row.get("pass1_num_completion_tokens"),
        "duration_s": row.get("pass1_duration_s"),
        "mean_ttft_s": sp.get("mean_ttft_s"),
        "defs": defs,
        "named_axis_delta_pct": named_axis_delta_pct,
        "residual_vs_stark_pct": residual_vs_stark,
        "gap_residual_after_def_match_pct": gap_residual_after_def_match_pct,
        "rescued_full_e2e": rescued_full,
        "rescued_steady": rescued_steady,
        "reproduces_632_full_e2e": (abs(pct(full, LAND_632_UNRESCUED[K])) < 2.0
                                    if (full and K in LAND_632_UNRESCUED) else None),
        "gen_meter_n": gm.get("steady_gen_tps_n"),
        "espec_mean": gm.get("espec_mean"),
    }


def decide(k6: dict[str, Any]) -> dict[str, Any]:
    named_delta = abs(k6["named_axis_delta_pct"]) if k6["named_axis_delta_pct"] is not None else None
    rv = k6["residual_vs_stark_pct"]
    dw = ["full_e2e_nonstream", "steady_gen_meter", "stream_full_e2e", "stream_steady"]
    any_decode_matches = any(rv.get(n) is not None and abs(rv[n]) < 1.5 for n in dw)
    cold = rv.get("cold_job_wall")
    boot_explains = (cold is not None and abs(cold) < 3.0)

    # The named hypothesis: gap = full_e2e (mine) vs steady (stark). It REQUIRES the
    # full-vs-steady spread to be ~9% AND a decode-window def to land at 155.58.
    named_axis_is_the_gap = (named_delta is not None and named_delta >= 5.0) and any_decode_matches
    if named_axis_is_the_gap:
        verdict = "GAP_IS_DEFINITION"
    elif any_decode_matches:
        verdict = "GAP_IS_DEFINITION"          # some decode-window def reproduces 155.58
    else:
        verdict = "GAP_IS_REAL"                # no decode-window def reaches 155.58
    return {
        "verdict": verdict,
        "named_axis_full_vs_steady_delta_pct": named_delta,
        "named_axis_below_2pct": (named_delta is not None and named_delta < 2.0),
        "any_decode_window_def_matches_stark": any_decode_matches,
        "gap_residual_after_def_match_pct": k6["gap_residual_after_def_match_pct"],
        "boot_inclusive_explains_gap": boot_explains,
        "cold_job_wall_residual_vs_stark_pct": cold,
        "primary_rescued_k6_reconciled": k6["rescued_full_e2e"],
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default=str(HERE / "walltps_defn_capture.json"))
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.capture).read_text())
    rows = [analyze_row(r) for r in data["rows"]]
    by_k = {r["K"]: r for r in rows}
    k6 = by_k.get(6)
    decision = decide(k6) if k6 else {"verdict": "INCOMPLETE", "note": "no K=6 row"}

    out = {
        "pr": 660, "analysis_only": True, "official_tps": 0,
        "cross_val": {
            "stark_unrescued_k6": STARK_UNRESCUED_K6, "stark_ar_ref": STARK_AR_REF,
            "land_ar_ref_A": LAND_AR_REF_A, "source": "#658 xval artifact (own)",
        },
        "ppl_unchanged": PPL_UNCHANGED,
        "rows": rows, "decision": decision,
    }
    (HERE / "defn_reconcile.json").write_text(json.dumps(out, indent=2, default=str))

    # ---- console table ----
    print("\n=== PR#660 wall_tps definition reconcile ===")
    print(f"stark un-rescued K6 (xval target) = {STARK_UNRESCUED_K6}   A(AR rung)={LAND_AR_REF_A}")
    for r in rows:
        d = r["defs"]
        def s(x): return f"{x:7.2f}" if isinstance(x, (int, float)) else "   n/a "
        print(f"\nK={r['K']}  boot={r['server_ready_s']}  ttft={r['mean_ttft_s']}")
        print(f"  full_e2e      {s(d['full_e2e_nonstream'])}  (repro #632: {r['reproduces_632_full_e2e']})")
        print(f"  steady_meter  {s(d['steady_gen_meter'])}  (n={r['gen_meter_n']})")
        print(f"  stream_full   {s(d['stream_full_e2e'])}")
        print(f"  stream_steady {s(d['stream_steady'])}")
        print(f"  cold_job_wall {s(d['cold_job_wall'])}")
        print(f"  named-axis full-vs-steady Δ = {r['named_axis_delta_pct']}%")
        print(f"  gap_residual_after_def_match = {r['gap_residual_after_def_match_pct']}%")
        print(f"  rescued: full={r['rescued_full_e2e']}  steady={r['rescued_steady']}  (#658={LAND_658_RESCUED.get(r['K'])})")
    print(f"\nDECISION: {json.dumps(decision, indent=2, default=str)}")

    if args.no_wandb:
        return 0

    # ---- W&B ----
    run = wandb_logging.init_wandb_run(
        job_type="walltps_defn_reconcile", agent="land",
        name="land/walltps-defn-reconcile",
        group="walltps-defn-reconcile-land",
        notes=("PR#660: does full_e2e-vs-steady wall_tps definition explain the 9.4% un-rescued "
               "K6 gap to stark #642 (155.58)? Fresh instrumented K=5,6 capture on the #632 stack; "
               "prices wall_tps under 5 definitions + re-prices rescued. Cross-val 155.58 from own "
               "#658 xval artifact (stark branch NOT read)."),
        config={
            "pr": 660, "analysis_only": True, "official_tps": 0,
            "stack": "int4_mtp_batchinv", "drafter": data.get("drafter"),
            "batch_invariant": 1, "max_num_seqs": 1, "greedy": True,
            "num_prompts": data.get("config", {}).get("num_prompts"),
            "output_len": data.get("config", {}).get("output_len"), "seed": 1,
            "vllm": "0.22.0",
            "stark_unrescued_k6": STARK_UNRESCUED_K6, "land_ar_ref_A": LAND_AR_REF_A,
            "ppl_unchanged": PPL_UNCHANGED,
        },
        tags=["optionb", "batch_invariant", "pr660", "walltps_defn", "reconcile", "served"],
    )
    if run is not None:
        import wandb
        cols = ["K", "full_e2e_nonstream", "steady_gen_meter", "stream_full_e2e",
                "stream_steady", "cold_job_wall", "named_axis_delta_pct",
                "gap_residual_after_def_match_pct", "rescued_full_e2e", "rescued_steady",
                "mean_ttft_s", "server_ready_s"]
        tbl = wandb.Table(columns=cols)
        for r in rows:
            d = r["defs"]
            tbl.add_data(r["K"], d["full_e2e_nonstream"], d["steady_gen_meter"],
                         d["stream_full_e2e"], d["stream_steady"], d["cold_job_wall"],
                         r["named_axis_delta_pct"], r["gap_residual_after_def_match_pct"],
                         r["rescued_full_e2e"], r["rescued_steady"], r["mean_ttft_s"],
                         r["server_ready_s"])
            run.log({
                "global_step": r["K"], "curve/K": r["K"],
                "curve/full_e2e": d["full_e2e_nonstream"],
                "curve/steady_gen_meter": d["steady_gen_meter"],
                "curve/stream_full_e2e": d["stream_full_e2e"],
                "curve/stream_steady": d["stream_steady"],
                "curve/cold_job_wall": d["cold_job_wall"],
                "curve/named_axis_delta_pct": r["named_axis_delta_pct"],
                "curve/rescued_full_e2e": r["rescued_full_e2e"],
                "curve/rescued_steady": r["rescued_steady"],
            })
        run.log({"defn_curve": tbl})

        summary = {
            "decision/verdict": decision["verdict"],
            "decision/named_axis_full_vs_steady_delta_pct": decision.get("named_axis_full_vs_steady_delta_pct"),
            "decision/named_axis_below_2pct": int(bool(decision.get("named_axis_below_2pct"))),
            "decision/any_decode_window_def_matches_stark": int(bool(decision.get("any_decode_window_def_matches_stark"))),
            "decision/gap_residual_after_def_match_pct": decision.get("gap_residual_after_def_match_pct"),
            "decision/boot_inclusive_explains_gap": int(bool(decision.get("boot_inclusive_explains_gap"))),
            "decision/cold_job_wall_residual_vs_stark_pct": decision.get("cold_job_wall_residual_vs_stark_pct"),
            "decision/primary_rescued_k6_reconciled": decision.get("primary_rescued_k6_reconciled"),
            "config/stark_unrescued_k6": STARK_UNRESCUED_K6,
            "config/land_ar_ref_A": LAND_AR_REF_A,
            "config/ppl_unchanged": PPL_UNCHANGED,
        }
        for r in rows:
            K = r["K"]; d = r["defs"]
            summary[f"perK/K{K}_full_e2e"] = d["full_e2e_nonstream"]
            summary[f"perK/K{K}_steady_gen_meter"] = d["steady_gen_meter"]
            summary[f"perK/K{K}_stream_full_e2e"] = d["stream_full_e2e"]
            summary[f"perK/K{K}_stream_steady"] = d["stream_steady"]
            summary[f"perK/K{K}_cold_job_wall"] = d["cold_job_wall"]
            summary[f"perK/K{K}_named_axis_delta_pct"] = r["named_axis_delta_pct"]
            summary[f"perK/K{K}_rescued_full_e2e"] = r["rescued_full_e2e"]
            summary[f"perK/K{K}_rescued_steady"] = r["rescued_steady"]
            summary[f"perK/K{K}_server_ready_s"] = r["server_ready_s"]
            summary[f"perK/K{K}_mean_ttft_s"] = r["mean_ttft_s"]
        wandb_logging.log_summary(run, summary, step=6)
        wandb_logging.log_json_artifact(run, name="walltps_defn_reconcile_660",
                                        artifact_type="analysis", data=out)
        url = getattr(run, "url", ""); rid = getattr(run, "id", "")
        wandb_logging.finish_wandb(run)
        print(f"[wandb] walltps-defn-reconcile id={rid} url={url}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
