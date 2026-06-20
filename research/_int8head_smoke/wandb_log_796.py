#!/usr/bin/env python3
"""Log PR #796 (int4 lm_head byte-floor: channelwise + MSE-optimal scales on bi0) to W&B.

One run in group ``bi0-lmhead-bytefloor``. The ONE delta vs merged #788 (int4 g32
minmax lm_head) is the lm_head quant: channelwise (group_size=-1) drops the per-group
scale bytes, and MSE/clip-aware scales replace minmax. Body int4 / drafter / surgattn /
MTP K=6 / env are byte-identical across all arms.

Arms (each = a serve of submissions/int4_mtp_bi0_lmhead_bytefloor with a different
MODEL_ID build; local A10G exploratory TPS, hardware-independent PPL + GSM8K greedy):
  control  int4 g32   minmax  (merged #788 lm_head; reference)
  A        int4 chan  minmax
  B        int4 chan  mse
  C        int4 g32   mse      (optional; logged if artifacts present)

Reads prevalidate local_summary.json (tps/ppl/completed) + gsm8k <label>_greedy.json
(accuracy). Builder quant diagnostics (weight rel_err, clip-ratio, lm_head bytes) are
stable build facts -> carried in ARMS below. analysis_only=1, no_hf_job=1, fires=0,
official_tps=0. Primary metric = best byte-floor arm local decode TPS; test metric =
its GSM8K greedy accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent

# Screen / stacking gates (from PR #796 body).
PPL_GATE = 2.42
GSM8K_FLOOR = 0.824          # within 5% of bi0 anchor (~0.867)
TPS_788_CITED = 256.74       # merged #788 local decode TPS (stacking-win threshold)

# --- per-arm static build facts + artifact locations ---
# preval = research/_int8head_smoke/<preval>/local_summary.json
# gsm8k  = research/_int8head_smoke/gsm8k/<gsm>_greedy.json
ARMS = {
    "control_g32_mm": {
        "group_size": 32, "observer": "minmax",
        "rel_err": 0.06743, "clip_mean": None, "frac_clipped": None,
        "lmhead_gb": 0.3775, "reduction_x": 3.56,
        "preval": "prevalidate_int4_candidate",   # may be overridden by --ctrl-preval
        "gsm": "ctrl_g32_mm",
    },
    "A_chan_mm": {
        "group_size": -1, "observer": "minmax",
        "rel_err": 0.17425, "clip_mean": 1.0, "frac_clipped": 0.0,
        "lmhead_gb": 0.3361, "reduction_x": 3.99,
        "preval": "prevalA_chan_mm", "gsm": "armA_chan_mm",
    },
    "B_chan_mse": {
        "group_size": -1, "observer": "mse",
        "rel_err": 0.13644, "clip_mean": 0.8001, "frac_clipped": 1.0,
        "lmhead_gb": 0.3361, "reduction_x": 3.99,
        "preval": "prevalB_chan_mse", "gsm": "armB_chan_mse",
    },
    "C_g32_mse": {
        "group_size": 32, "observer": "mse",
        "rel_err": 0.04463, "clip_mean": 0.9684, "frac_clipped": 0.6310,
        "lmhead_gb": 0.3775, "reduction_x": 3.56,
        "preval": "prevalC_g32_mse", "gsm": "armC_g32_mse",
    },
}


def load_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return None


def collect(arm: str, meta: dict, ctrl_preval: str | None) -> dict | None:
    preval_sub = ctrl_preval if (arm == "control_g32_mm" and ctrl_preval) else meta["preval"]
    pv = load_json(HERE / preval_sub / "local_summary.json")
    if pv is None:
        return None
    gs = load_json(HERE / "gsm8k" / f"{meta['gsm']}_greedy.json")
    row = {
        "arm": arm,
        "group_size": meta["group_size"], "observer": meta["observer"],
        "weight_rel_err": meta["rel_err"], "clip_ratio_mean": meta["clip_mean"],
        "frac_clipped": meta["frac_clipped"],
        "lmhead_bytes_gb": meta["lmhead_gb"], "byte_reduction_x": meta["reduction_x"],
        "tps": round(pv["tps"], 4), "ppl": round(pv["ppl"], 4),
        "completed": pv["completed"],
        "decode_tokens": pv.get("decode_num_completion_tokens"),
        "ppl_within_gate": int(pv["ppl"] <= PPL_GATE),
        "gsm8k_acc": (round(gs["accuracy"], 4) if gs else None),
        "gsm8k_n_correct": (gs["n_correct"] if gs else None),
        "gsm8k_n": (gs["n_problems"] if gs else None),
        "gsm8k_strict_rate": (gs.get("strict_rate") if gs else None),
        "gsm8k_trunc_rate": (gs.get("truncation_rate") if gs else None),
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="gemma-challenge-senpai")
    ap.add_argument("--entity", default="wandb-applied-ai-team")
    ap.add_argument("--group", default="bi0-lmhead-bytefloor")
    ap.add_argument("--name", default="lawine/bi0-lmhead-bytefloor")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--ctrl-preval", default=None,
                    help="prevalidate subdir for the control (if re-measured this session)")
    ap.add_argument("--dry-run", action="store_true", help="print rows, do not init wandb")
    args = ap.parse_args()

    rows = {}
    for arm, meta in ARMS.items():
        r = collect(arm, meta, args.ctrl_preval)
        if r is not None:
            rows[arm] = r
        else:
            print(f"[skip] {arm}: no prevalidate artifact")

    ctrl = rows.get("control_g32_mm")
    ctrl_tps = ctrl["tps"] if ctrl else None
    ctrl_gsm = ctrl["gsm8k_acc"] if ctrl else None

    def verdict(r: dict) -> dict:
        within_ppl = r["ppl"] <= PPL_GATE
        within_tps_session = (ctrl_tps is not None and r["tps"] >= ctrl_tps)  # vs same-session g32
        clears_tps_cited = r["tps"] >= TPS_788_CITED
        q = r["gsm8k_acc"]
        within_gsm_floor = (q is not None and q >= GSM8K_FLOOR)
        within_gsm_ctrl = (q is not None and ctrl_gsm and q >= 0.95 * ctrl_gsm)
        return {
            "within_ppl_gate": int(within_ppl),
            "ge_g32_tps_session": int(within_tps_session),
            "ge_788_cited_tps": int(clears_tps_cited),
            "gsm8k_ge_floor_0824": int(bool(within_gsm_floor)),
            "gsm8k_within5pct_ctrl": int(bool(within_gsm_ctrl)),
            # stacking win needs a REAL TPS gain (>= cited #788) AND quality held
            "stacking_win": int(bool(within_ppl and clears_tps_cited and within_gsm_floor)),
        }

    for arm, r in rows.items():
        r.update(verdict(r))

    config = {
        "pr": 796, "phase": "bi0_lmhead_bytefloor",
        "base_submission": "submissions/int4_mtp_bi0_lmhead_bytefloor",
        "delta_vs_788": "lm_head quant only (channelwise scale-floor + MSE/clip scales); body int4 byte-identical",
        "spec_method": "mtp (gemma4_assistant drafter)", "num_speculative_tokens": 6,
        "vllm": "0.22.0", "transformers": "5.9.0", "gpu": "A10G sm_86 (local, exploratory TPS)",
        "kernel": "Marlin W4A16 (MARLIN_SUPPORTED_GROUP_SIZES=[-1,32,64,128]; chan g=-1 first-class)",
        "sampler": "VLLM_USE_FLASHINFER_SAMPLER=0 (native; greedy/PPL-identical)",
        "decode_num_prompts": 128, "output_len": 512,
        "ppl_gate": PPL_GATE, "gsm8k_floor": GSM8K_FLOOR, "tps_788_cited": TPS_788_CITED,
        "gsm8k_n": 200, "gsm8k_seed": 1234, "gsm8k_nshot": 8, "gsm8k_regime": "greedy",
        "clip_ratio_grid": [1.0, 0.95, 0.90, 0.85, 0.80],
        "analysis_only": 1, "no_hf_job": 1, "fires": 0, "official_tps": 0,
    }

    cols = ["arm", "group_size", "observer", "weight_rel_err", "clip_ratio_mean",
            "lmhead_bytes_gb", "byte_reduction_x", "tps", "ppl", "completed",
            "gsm8k_acc", "ppl_within_gate", "ge_788_cited_tps", "gsm8k_ge_floor_0824",
            "stacking_win"]
    table_data = [[rows[a].get(c) for c in cols] for a in rows]

    if args.dry_run:
        print(json.dumps({"config": config, "rows": rows}, indent=2))
        return 0

    run = wandb.init(
        project=args.project, entity=args.entity, group=args.group,
        name=args.name, id=args.run_id, resume=("allow" if args.run_id else None),
        config=config,
    )

    best = max((r for r in rows.values() if r["arm"] != "control_g32_mm"),
               key=lambda r: r["tps"], default=ctrl)
    summary = {
        "analysis_only": 1, "no_hf_job": 1, "fires": 0, "official_tps": 0,
        "primary_metric_best_bytefloor_tps": best["tps"] if best else None,
        "test_metric_best_bytefloor_gsm8k": best["gsm8k_acc"] if best else None,
        "any_candidate_stacking_win": int(any(
            r["stacking_win"] for a, r in rows.items() if a != "control_g32_mm")),
        "ctrl_g32_tps": ctrl_tps, "ctrl_g32_gsm8k": ctrl_gsm,
    }
    for arm, r in rows.items():
        for k in ("tps", "ppl", "completed", "gsm8k_acc", "weight_rel_err",
                  "lmhead_bytes_gb", "stacking_win", "ge_788_cited_tps", "gsm8k_ge_floor_0824"):
            summary[f"{arm}__{k}"] = r.get(k)
    run.summary.update(summary)
    run.log({"arms": wandb.Table(columns=cols, data=table_data)})

    (HERE / "results_796.json").write_text(json.dumps({"config": config, "rows": rows}, indent=2))
    print(f"[wandb] run {run.id} group={args.group}")
    for a, r in rows.items():
        print(f"  {a:16s} tps={r['tps']:.2f} ppl={r['ppl']:.4f} "
              f"gsm8k={r['gsm8k_acc']} stacking_win={r['stacking_win']}")
    run.finish()
    print(f"RUN_ID={run.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
