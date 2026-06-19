#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Log the PR #764 independent fire-literal-identity cross-validation to W&B.

Reads runs/analysis.json (analyze_xcheck.py) + the arm summaries and creates ONE run in group
``fire_literal_identity_xcheck``. Emits the deliverable: fire_literal_greedy_identity (primary,
N/128 byte-exact spec_on vs my independent spec_off M=1 AR ref), xcheck_consistent_with_751 (test),
the divergence distribution (frac diverging + first-div histogram = the int4 near-tie cascade
signature), and the AR-vs-AR determinism floor. Required card flags: analysis_only=1, official_tps=0,
no_hf_job=1, fires=0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, default=HERE / "runs" / "analysis.json")
    ap.add_argument("--project", default="gemma-challenge-senpai")
    ap.add_argument("--entity", default="wandb-applied-ai-team")
    ap.add_argument("--group", default="fire_literal_identity_xcheck")
    ap.add_argument("--name", default="land/fire-literal-identity-xcheck")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    a = json.loads(args.analysis.read_text())
    arms = a["arms"]
    spec_on = arms["spec_on"]
    spec_off = arms["spec_off_ref"]
    pm = a["primary_metric"]
    xc = a["xcheck"]
    dv = a["divergence"]
    det = a.get("determinism_control")

    run = wandb.init(
        project=args.project, entity=args.entity, group=args.group,
        name=args.name, id=args.run_id, resume=("allow" if args.run_id else None),
        config={
            "pr": 764, "phase": "fire_literal_identity_xcheck",
            "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
            "served_via": "submissions/int4_mtp_batchinv/serve.py (the merged fire submission)",
            "attn_backend": "TRITON_ATTN (vLLM auto-forces for Gemma4 heterogeneous head dims)",
            "engine": "vllm-0.22.0 v1 api_server (online, CUDA graphs)",
            "spec_method": "gemma4_assistant MTP drafter (Gemma4MTPModel)",
            "num_speculative_tokens": 6,
            "batch_invariant": 1,
            "reference_construction": "fire serve.py + SENPAI_REFERENCE_MODE=1 (M=1 AR, drafter OFF)",
            "independent_of": "wirbel #751 harness (not read); reuses land #748 client+identity",
            "n_prompts": spec_on.get("n_prompts"), "output_len": spec_on.get("output_len"),
            "sampling": "greedy temp=0 (strict-#319 identity protocol; NOT generation_config.json)",
            "wirbel_751_identical": 20, "wirbel_751_frac": 0.156,
            "tolerance_identical": xc["tolerance_identical"],
            "locked_tps_anchor": 126.378,
            "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        },
    )

    flat = {
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "verdict": a["verdict"],
        # PRIMARY + TEST
        "fire_literal_greedy_identity": pm["value"],
        "fire_literal_greedy_identity_n_match": pm["n_match"],
        "fire_literal_greedy_identity_n_total": pm["n_total"],
        "fire_literal_greedy_identity_n_diverge": pm["n_diverge"],
        "xcheck_consistent_with_751": a["test_metric"]["value"],
        # xcheck reconciliation
        "n_identical": xc["n_identical"],
        "wirbel_751_identical": xc["wirbel_751_identical"],
        "abs_diff_identical": xc["abs_diff_identical"],
        "tolerance_identical": xc["tolerance_identical"],
        "consistency_band_lo": xc["consistency_band_identical"][0],
        "consistency_band_hi": xc["consistency_band_identical"][1],
        "my_frac": xc["my_frac"], "wirbel_751_frac": xc["wirbel_751_frac"],
        # divergence distribution (mechanism evidence)
        "frac_diverging": dv["frac_diverging"],
        "n_diverging": dv["n_diverging"],
        "wirbel_751_diverge_frac": dv["wirbel_751_diverge_frac"],
        "per_token_flip_hazard": dv["per_token_flip_hazard"],
        "first_div_min": dv["first_div_pos_histogram"]["min"],
        "first_div_median": dv["first_div_pos_histogram"]["median"],
        "first_div_max": dv["first_div_pos_histogram"]["max"],
        "frac_first_div_after_tok16": dv["first_div_pos_histogram"]["frac_first_div_after_tok16"],
        # local TPS probes (non-transferable)
        "tps_spec_on_local": spec_on.get("output_tps"),
        "tps_spec_off_ref_local": spec_off.get("output_tps"),
        "spec_on_peak_gpu_mem_mib": spec_on.get("peak_gpu_mem_mib"),
        "honest_read": a["honest_read"],
    }
    if det is not None:
        di = det["identity"]
        flat.update({
            "determinism_floor_n_match": di["n_match"],
            "determinism_floor_n_total": di["n_total"],
            "determinism_floor_frac": di["frac"],
            "determinism_floor_per_token_flip_hazard": di.get("per_token_flip_hazard"),
            "stack_deterministic_within_config": int(det["deterministic"]),
            "determinism_n_shared_prompts": det["n_shared_prompts"],
        })
    run.summary.update(flat)

    # arms table
    run.log({"arms": wandb.Table(
        columns=["arm", "kind", "num_spec", "output_tps", "n_prompts", "output_len",
                 "peak_mem_mib", "boot_s"],
        data=[[k, v.get("kind"), v.get("num_speculative_tokens"), v.get("output_tps"),
               v.get("n_prompts"), v.get("output_len"), v.get("peak_gpu_mem_mib"),
               v.get("boot_s")] for k, v in arms.items()])})

    # first-divergence histogram (the spread-not-root-clustered near-tie signature)
    hist = dv["first_div_pos_histogram"]["bins"]
    run.log({"first_div_pos_histogram": wandb.Table(
        columns=["token_pos_bin", "n_prompts"], data=[[k, v] for k, v in hist.items()])})

    print(f"[wandb] run {run.id}  group={args.group}")
    print(f"[wandb] VERDICT={a['verdict']}  fire_literal_greedy_identity="
          f"{pm['n_match']}/{pm['n_total']} ({pm['value']:.4f})  "
          f"xcheck_consistent_with_751={a['test_metric']['value']}")
    print(f"[wandb] frac_diverging={dv['frac_diverging']}  "
          f"determinism_floor={(det or {}).get('floor_frac')}")
    run.finish()
    print(f"RUN_ID={run.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
