#!/usr/bin/env python3
"""PR #811: log the int4head serve-config byte-exact TPS sweep to W&B.

LOCAL A10G sweep (NOT an HF Job, NOT a submission): local single-stream decode TPS
proxy at conc=1 / output_len=512 over the 128 ShareGPT speed-benchmark prompts.
Reads research/serve_config_sweep/results.jsonl and creates, in group
``serve-config-tps-sweep``:
  * one run per config label (comparable sweep points), and
  * one summary run holding the master per-knob delta table + verdict.

Run with a wandb-capable interpreter from a NON-root cwd so the local ./wandb
dir / package shadowing gotcha is avoided, e.g.:
    cd research/serve_config_sweep && /usr/bin/python3 wandb_log.py
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "gemma-challenge-senpai"
GROUP = "serve-config-tps-sweep"
RESULTS = Path("/workspace/senpai/target/research/serve_config_sweep/results.jsonl")

# Graph-shape-altering knobs force a cold recompile / CUDA-graph re-capture (long
# startup ~155s vs ~95s warm). MEASURED FINDING: the served int4-Marlin + K6-MTP-
# spec-decode + VLLM_BATCH_INVARIANT=0 stack is NOT boot-deterministic at the token
# level -- a recompiled boot can land on a different (still coherent, valid greedy)
# FP-reduction attractor, diverging from control on 75-83% of prompts (greedy forks
# at the first near-tie logit, then cascades). Same config, different boots => different
# streams: maxlen3072 gave A(warm) and B(cold); batched2048 gave two distinct divergent
# streams on BOTH reps (so warm reuse does NOT reliably reconverge to control). Non-graph
# knobs (gpu_mem) reuse the warm cache and stayed byte-exact. So byte-exact parity here is
# a property of the compile/capture cache state, NOT guaranteed by a "config-only" change.
GRAPH_ALTERING = {"MAX_MODEL_LEN", "MAX_NUM_BATCHED_TOKENS", "CUDAGRAPH_CAPTURE_SIZES"}


def warm_tps(runs: list[dict]) -> list[float]:
    runs = sorted(runs, key=lambda r: r["rep"])
    warm = runs[1:] if len(runs) > 1 else runs  # drop cold rep0 when >1 rep
    return [r["tps"] for r in warm if r.get("tps")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--dry-run", action="store_true", help="print, do not log")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.results.read_text().splitlines() if l.strip()]
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)

    ctrl_runs = sorted(by["control"], key=lambda r: r["rep"])
    ctrl_ok = [r for r in ctrl_runs if r.get("ok")]
    ctrl_ref_hash = ctrl_ok[0]["parity_hash"]
    ctrl_tps = warm_tps(ctrl_ok)
    ctrl_med = statistics.median(ctrl_tps)
    ctrl_spread_pct = (max(ctrl_tps) - min(ctrl_tps)) / ctrl_med * 100 if len(ctrl_tps) > 1 else 0.0

    # Per-label summary records.
    summary_rows = []
    for label in sorted(by):
        runs = sorted(by[label], key=lambda r: r["rep"])
        ok = [r for r in runs if r.get("ok")]
        knob = runs[0].get("config", {})
        knob_keys = set(knob)
        graph_altering = bool(knob_keys & GRAPH_ALTERING)
        if ok:
            tps_list = warm_tps(ok)
            med = statistics.median(tps_list) if tps_list else ok[-1].get("tps", 0.0)
            d = med - ctrl_med
            dp = d / ctrl_med * 100
            steady = ok[-1]["parity_hash"]              # cache-warm (highest successful rep)
            steady_byte_exact = steady == ctrl_ref_hash
            cold = ok[0]["parity_hash"]
            cold_rep0_diverged = (cold != ctrl_ref_hash)
            any_rep_byte_exact = all(r["parity_hash"] == ctrl_ref_hash for r in ok)
            win = bool(label != "control" and steady_byte_exact and dp >= 1.0)
            rec = {
                "label": label, "knob": json.dumps(knob), "n_reps": len(runs), "n_ok": len(ok),
                "valid": True, "graph_altering": int(graph_altering),
                "median_warm_tps": round(med, 4),
                "delta_tps_vs_control": round(d, 4), "delta_pct_vs_control": round(dp, 4),
                "steady_state_byte_exact": int(steady_byte_exact),
                "cold_rep0_diverged": int(cold_rep0_diverged),
                "all_reps_byte_exact": int(any_rep_byte_exact),
                "steady_parity_hash": steady, "cold_parity_hash": cold,
                "startup_s_cold": ok[0].get("startup_s"), "startup_s_warm": ok[-1].get("startup_s"),
                "win": int(win), "error": None,
                "all_reps_tps": [r.get("tps") for r in ok],
            }
        else:
            # invalid config (e.g. MAX_MODEL_LEN that truncates real prompts -> HTTP 400)
            rec = {
                "label": label, "knob": json.dumps(knob), "n_reps": len(runs), "n_ok": 0,
                "valid": False, "graph_altering": int(graph_altering),
                "median_warm_tps": None, "delta_tps_vs_control": None, "delta_pct_vs_control": None,
                "steady_state_byte_exact": None, "cold_rep0_diverged": None,
                "all_reps_byte_exact": None, "steady_parity_hash": None, "cold_parity_hash": None,
                "startup_s_cold": runs[0].get("startup_s"), "startup_s_warm": None,
                "win": 0, "error": runs[0].get("error"), "all_reps_tps": [],
            }
        summary_rows.append(rec)

    any_win = any(r["win"] for r in summary_rows)
    verdict = ("CONFIG_WIN" if any_win
               else "NULL_CONFIG_ALREADY_OPTIMAL")

    if args.dry_run:
        print(f"control: median_tps={ctrl_med:.4f} spread={ctrl_spread_pct:.3f}% ref={ctrl_ref_hash[:12]}")
        for r in summary_rows:
            print(json.dumps(r))
        print(f"VERDICT={verdict} any_win={any_win}")
        return 0

    base_cfg = {
        "pr": 811, "phase": "serve_config_tps_sweep",
        "model_id": "/workspace/gemma_build/bi0_int4head_g32 (int4head)",
        "served_model_name": "gemma-4-e4b-it",
        "engine": "vllm-0.22.0 v1 api_server (online, CUDA graphs)",
        "spec_method": "mtp", "num_speculative_tokens": 6, "batch_invariant": 0,
        "workload": "single-stream conc=1 / MAX_NUM_SEQS=1, output_len=512, 128 ShareGPT prompts (seed=1)",
        "tps_kind": "LOCAL_PROXY_decode_completion_tokens_over_duration",
        "official_tps": 0, "no_hf_job": 1, "no_launch": 1, "no_submission": 1, "local_serve": 1,
        "control_median_tps": round(ctrl_med, 4), "control_ref_parity_hash": ctrl_ref_hash,
        "control_warmrep_spread_pct": round(ctrl_spread_pct, 4),
        "win_criterion": "byte_exact(steady) AND delta_pct>=+1.0 (#784 cap)",
        "prompt_len_max_input_tok": 2427, "max_model_len_floor": 2939,
    }

    run_ids: dict[str, str] = {}
    # one comparable run per config label
    for r in summary_rows:
        knob = json.loads(r["knob"])
        cfg = dict(base_cfg)
        cfg.update({"config_label": r["label"], "knob": r["knob"], **{f"knob_{k}": v for k, v in knob.items()}})
        run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                         name=f"fern/sweep-{r['label']}", job_type="serve-config-cell",
                         config=cfg, reinit=True)
        run.summary.update({k: v for k, v in r.items() if k not in ("all_reps_tps", "knob")})
        run.summary.update({"verdict": verdict, "control_median_tps": round(ctrl_med, 4)})
        if r["all_reps_tps"]:
            run.log({"reps": wandb.Table(columns=["rep_idx", "tps"],
                                         data=[[i, t] for i, t in enumerate(r["all_reps_tps"])])})
        run_ids[r["label"]] = run.id
        run.finish()

    # summary run with the master per-knob delta table
    srun = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                      name="fern/serve-config-tps-sweep-summary", job_type="summary",
                      config={**base_cfg, "verdict": verdict, "n_configs": len(summary_rows)},
                      reinit=True)
    cols = ["label", "knob", "valid", "graph_altering", "median_warm_tps",
            "delta_tps_vs_control", "delta_pct_vs_control", "steady_state_byte_exact",
            "cold_rep0_diverged", "n_reps", "startup_s_cold", "startup_s_warm", "win", "error"]
    tbl = wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in summary_rows])
    srun.log({"serve_config_sweep": tbl})
    srun.summary.update({
        "verdict": verdict, "any_win": int(any_win),
        "control_median_tps": round(ctrl_med, 4),
        "control_warmrep_spread_pct": round(ctrl_spread_pct, 4),
        "best_byteexact_delta_pct": max(
            (r["delta_pct_vs_control"] for r in summary_rows
             if r["valid"] and r["steady_state_byte_exact"] and r["label"] != "control"),
            default=None),
        "official_tps": 0, "no_hf_job": 1,
    })
    run_ids["summary"] = srun.id
    srun.finish()

    print("VERDICT=", verdict)
    for label, rid in run_ids.items():
        print(f"RUN_ID {label}={rid}")
    print("WANDB_RUN_IDS_JSON=" + json.dumps(run_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
