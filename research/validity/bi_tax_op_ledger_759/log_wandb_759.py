#!/usr/bin/env python3
"""PR #759: log the batch-invariance-tax per-kernel-family ledger to W&B.

One run, grouped under ``fire_bi_tax_750`` (so it sits beside #750's BI1/BI0 TPS
cards in W&B). Reads runs/ledger.json (+ the two arm summaries). 0-GPU; run on
/usr/bin/python3 (the only interpreter here with wandb) from a cwd WITHOUT a
./wandb dir, per the logging gotcha.

Headline logged: bi_tax_top_op_family + bi_tax_top_op_share (largest family's
share of the BI slowdown), the per-family ledger table, and the #750-anchored
BI-tax constants this decomposes (tps_BI0=229.847, tps_BI1=156.949, ~73 TPS).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "gemma-challenge-senpai"
ENTITY = "wandb-applied-ai-team"

# #750 official-anchored constants this card decomposes (from PR #750 RESULTS.json)
TPS_BI0 = 229.847
TPS_BI1 = 156.949
BI_TAX_PCT = 0.3172
BI_TAX_TOTAL_TPS = round(TPS_BI0 - TPS_BI1, 3)  # ~72.9 ~ 73


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--bi0-summary", type=Path, required=True)
    ap.add_argument("--bi1-summary", type=Path, required=True)
    ap.add_argument("--group", default="fire_bi_tax_750")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    L = json.loads(args.ledger.read_text())
    s0 = json.loads(args.bi0_summary.read_text())
    s1 = json.loads(args.bi1_summary.read_text())

    config = {
        "pr": 759,
        "phase": "fire_bi_tax_op_ledger",
        "axis": "cost_attribution_per_kernel",
        "VLLM_BATCH_INVARIANT_arms": [0, 1],
        "spec_on": True,
        "num_speculative_tokens": 6,
        "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
        "drafter_model": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
        "attention_backend": "TRITON_ATTN",
        "device": "A10G(SM86)",
        "tensor_parallel": 1,
        "vllm": "0.22.0",
        "profiler": "torch (vLLM --profiler-config) device-trace, record_shapes",
        "profile_protocol": (
            "prefix-cache-isolated pure-decode window: warmup x3 primes prefix "
            "cache + JIT, then /start_profile -> 256-tok greedy ignore_eos gen "
            "(prefill=cache hit) -> /stop_profile; device kernels summed per family"
        ),
        "anchor_tps_BI0": TPS_BI0,
        "anchor_tps_BI1": TPS_BI1,
        "anchor_bi_tax_pct": BI_TAX_PCT,
        "bi_tax_total_tps": BI_TAX_TOTAL_TPS,
        "official_tps": 0, "no_hf_job": 1, "fires": 0,  # analysis card, per PR
        "prompt_tokens_bi0": s0.get("prompt_tokens"),
        "prompt_tokens_bi1": s1.get("prompt_tokens"),
        "completion_tokens_bi0": s0.get("completion_tokens"),
        "completion_tokens_bi1": s1.get("completion_tokens"),
    }

    run = wandb.init(project=PROJECT, entity=ENTITY, group=args.group,
                     name="fern/bi-tax-op-ledger", id=args.run_id,
                     resume=("allow" if args.run_id else None), config=config)

    summ = {
        "bi_tax_top_op_family": L["bi_tax_top_op_family"],
        "bi_tax_top_op_share": L["bi_tax_top_op_share"],
        "profiled_device_bi_tax_pct": L["profiled_device_bi_tax_pct"],
        "total_device_ms_per_tok_bi0": L["total_device_ms_per_tok_bi0"],
        "total_device_ms_per_tok_bi1": L["total_device_ms_per_tok_bi1"],
        "total_device_ms_per_tok_added": L["total_device_ms_per_tok_added"],
        "total_added_ms_per_tok_positive": L["total_added_ms_per_tok_positive"],
        "bi_tax_total_tps": BI_TAX_TOTAL_TPS,
        "decode_tps_proxy_bi0": L.get("decode_tps_proxy_bi0"),
        "decode_tps_proxy_bi1": L.get("decode_tps_proxy_bi1"),
        # int4-body vs bf16 GEMM control: proves the GEMM tax is the bf16 swap
        # (lm_head + drafter), NOT the int4 Marlin body (a CUDA op BI never touches).
        "gemm_int4_body_added_ms_per_tok": L.get("gemm_int4_body_added_ms_per_tok"),
        "gemm_bf16_added_ms_per_tok": L.get("gemm_bf16_added_ms_per_tok"),
    }
    for r in L["ledger"]:
        f = r["family"]
        summ[f"family/{f}/bi0_ms_per_tok"] = r["bi0_ms_per_tok"]
        summ[f"family/{f}/bi1_ms_per_tok"] = r["bi1_ms_per_tok"]
        summ[f"family/{f}/added_ms_per_tok"] = r["added_ms_per_tok"]
        summ[f"family/{f}/share_of_total_added"] = r["share_of_total_added"]
    run.summary.update({k: v for k, v in summ.items() if v is not None})

    # per-family ledger table
    cols = ["family", "bi0_ms_per_tok", "bi1_ms_per_tok", "added_ms_per_tok",
            "share_of_total_added"]
    run.log({"bi_tax_ledger": wandb.Table(
        columns=cols, data=[[r[c] for c in cols] for r in L["ledger"]])})

    # top kernels per family per arm (rich record for future analysis)
    for arm, key in (("bi0", "top_kernels_bi0"), ("bi1", "top_kernels_bi1")):
        rows = []
        for fam, lst in L.get(key, {}).items():
            for name, ms in lst:
                rows.append([fam, name, ms])
        if rows:
            run.log({f"top_kernels_{arm}": wandb.Table(
                columns=["family", "kernel", "ms_total"], data=rows)})

    print(f"[wandb] run={run.id} top_family={L['bi_tax_top_op_family']} "
          f"share={L['bi_tax_top_op_share']}")
    rid = run.id
    run.finish()
    print(f"RUN_ID={rid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
