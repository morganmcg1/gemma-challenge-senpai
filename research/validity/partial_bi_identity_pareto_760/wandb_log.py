#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Log the PR #760 partial-BI identity-coverage Pareto to W&B (0-GPU, wandb-capable venv).

Reads runs/pareto.json (from pareto.py) and creates ONE run in group ``fire_bi_tax_750``
(same group as land#748 / fern#750 / lawine#755 so the cards compare directly). Emits the
identity-vs-coverage Pareto: identity (k/128) + anchored TPS per realizable rung, the
verdict (full_bi_necessary, min_strict_bi_tps), and the realizability findings. Required PR
flags: analysis_only=1, official_tps=0, no_hf_job=1, fires=0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pareto", type=Path, default=HERE / "runs" / "pareto.json")
    ap.add_argument("--project", default="gemma-challenge-senpai")
    ap.add_argument("--entity", default="wandb-applied-ai-team")
    ap.add_argument("--group", default="fire_bi_tax_750")
    ap.add_argument("--name", default="land/partial-bi-identity-pareto")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    a = json.loads(args.pareto.read_text())
    anc = a["anchoring"]
    v = a["verdict"]
    rungs = a["rungs_realizable"]
    given = a["rungs_given_anchors"]

    run = wandb.init(
        project=args.project, entity=args.entity, group=args.group,
        name=args.name, id=args.run_id, resume=("allow" if args.run_id else None),
        config={
            "pr": 760, "phase": "partial_bi_identity_pareto",
            "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
            "attn_backend": "TRITON_ATTN (architecturally forced for Gemma-4)",
            "engine": "vllm-0.22.0 v1 api_server",
            "spec_method": "ngram", "num_speculative_tokens": 6,
            "n_prompts": 128, "output_len": 512,
            "reuses": "land#748 strict_clean_served_byteexact harness (merged)",
            "anchoring_method": "fern#750 official-anchoring (merged compute_projection.py)",
            "locked_tps_anchor": 126.378,
            "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        },
    )

    flat = {
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        # ---- VERDICT (primary deliverables) ----
        "full_bi_necessary": v["full_bi_necessary"],
        "min_strict_bi_tps": v["min_strict_bi_tps"],
        "strict_rung": v["strict_rung"],
        "n_realizable_rungs_hitting_128": v["n_realizable_rungs_hitting_128"],
        "n_deployable_realizable_hitting_128": v["n_deployable_realizable_hitting_128"],
        "max_realizable_identity": v["max_realizable_identity"],
        "max_deployable_realizable_identity": v["max_deployable_realizable_identity"],
        # ---- anchoring provenance ----
        "anchor_official_int4qat": anc["ANCHOR_OFFICIAL_int4qat"],
        "local_int4qat_nospec_ourmeter": anc["local_int4_qat_nospec_ourmeter"],
        "R_int4_pod_to_official": anc["R_int4_pod_to_official"],
        "meter_agreement_pct_vs_fern": anc["meter_agreement_pct"],
        "fire_stack_scale": anc["fire_stack_scale"],
        "fern_bi0_anchored": anc["FERN_BI0_ANCHORED"],
        "fern_bi1_anchored": anc["FERN_BI1_ANCHORED"],
        # ---- controls ----
        "determinism_floor_k": a["determinism_floor"]["k"],
        "determinism_floor_n": a["determinism_floor"]["n"],
        "determinism_floor_ok": int(a["determinism_floor"]["ok"]),
        "ar_only_bi_xcheck_k": a["ar_only_bi_xcheck"]["k"],
        "ar_only_bi_xcheck_n": a["ar_only_bi_xcheck"]["n"],
        # ---- realizability findings ----
        "bi_flag_monolithic_no_per_family_toggle": 1,
        "gemma4_forces_triton_attn": 1,
        "triton_attn_does_not_pin_attention_split_under_bi": 1,
        "attention_split_coverage_realizable": 0,
        # ---- key realizable rung identities (deployable cudagraph) ----
        "rung_R0_zeroBI_identity_k": rungs[0]["identity_k"],
        "rung_R1_gemmReductionBI_identity_k": rungs[1]["identity_k"],
        "rung_R2_eagerProbe_identity_k": rungs[2]["identity_k"],
        "rung_R0_anchored_tps_fire": rungs[0]["anchored_tps_fire"],
        "rung_R1_anchored_tps_fire": rungs[1]["anchored_tps_fire"],
        "rung_R2_anchored_tps_fire": rungs[2]["anchored_tps_fire"],
        "rung_R1_decode_tax_pct_vs_R0": round(
            (rungs[0]["local_tps_spec"] - rungs[1]["local_tps_spec"])
            / rungs[0]["local_tps_spec"] * 100, 3),
    }
    run.summary.update(flat)

    # ---- the Pareto table (realizable + given anchors) ----
    cols = ["rung", "identity_k", "identity_n", "identity_frac", "attn_split_pinned",
            "capture_symmetric", "deployable", "realizable", "local_tps_spec",
            "anchored_tps_fire", "anchored_tps_literal_int4qat", "source"]
    data = []
    for r in rungs + given:
        data.append([r["rung"], r["identity_k"], r["identity_n"], r["identity_frac"],
                     int(r["attention_split_pinned"]), int(r["cudagraph_capture_symmetric"]),
                     int(r["deployable"]), int(r["realizable_with_current_toggles"]),
                     r["local_tps_spec"], r["anchored_tps_fire"],
                     r["anchored_tps_literal_int4qat"], r["source"]])
    run.log({"identity_coverage_pareto": wandb.Table(columns=cols, data=data)})

    # ---- Pareto scatter (identity frac vs anchored fire TPS), for the W&B chart ----
    scat_cols = ["rung", "anchored_tps_fire", "identity_frac", "deployable"]
    scat = [[r["rung"], r["anchored_tps_fire"], r["identity_frac"], int(r["deployable"])]
            for r in rungs + given if r["anchored_tps_fire"] is not None]
    run.log({"pareto_scatter": wandb.Table(columns=scat_cols, data=scat)})

    print(f"[wandb] run {run.id}  group={args.group}")
    print(f"[wandb] full_bi_necessary={v['full_bi_necessary']}  "
          f"min_strict_bi_tps={v['min_strict_bi_tps']}  "
          f"max_realizable={v['max_realizable_identity']}")
    run.finish()
    print(f"RUN_ID={run.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
