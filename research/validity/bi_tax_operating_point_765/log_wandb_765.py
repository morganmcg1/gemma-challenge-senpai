#!/usr/bin/env python3
"""PR #765: log the BI-tax operating-point robustness ledger to W&B.

One run, grouped under ``bi_tax_operating_point`` (per the PR's reproduce note).
Reads runs/sweep_ledger.json (+ the two arm summaries). 0-GPU analysis card.

Per the wandb logging gotcha: run on /usr/bin/python3 (the only interpreter here
with wandb) from a cwd WITHOUT a ./wandb dir; pass absolute paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "gemma-challenge-senpai"
ENTITY = "wandb-applied-ai-team"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--bi0-summary", type=Path, required=True)
    ap.add_argument("--bi1-summary", type=Path, required=True)
    ap.add_argument("--group", default="bi_tax_operating_point")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    L = json.loads(args.ledger.read_text())
    s0 = json.loads(args.bi0_summary.read_text())
    s1 = json.loads(args.bi1_summary.read_text())

    pre0 = L.get("prefill_bi0") or {}
    pre1 = L.get("prefill_bi1") or {}

    config = {
        "pr": 765,
        "phase": "fire_bi_tax_operating_point",
        "axis": "operating_point_robustness_of_31.72pct_tax",
        "calibrates": "fern #759 BI_TAX_GEMM_DOMINATED (9hf7gvzd) + fern #750 ~157",
        "VLLM_BATCH_INVARIANT_arms": [0, 1],
        "spec_on": True,
        "num_speculative_tokens": 6,
        "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
        "drafter_model": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
        "attention_backend": "TRITON_ATTN",
        "sliding_window": 512,
        "layer_pattern": "5:1 sliding:full (35 sliding / 7 full of 42)",
        "device": "A10G(SM86)",
        "tensor_parallel": 1,
        "vllm": "0.22.0",
        "engine": "local /tmp/senpai-venvs/20f658587e8a6643 (fire config verbatim, "
                  "submissions/int4_mtp_batchinv) + --profiler-config torch device-trace",
        "profiler": "torch (vLLM --profiler-config) device-trace, record_shapes",
        "profile_protocol": (
            "one server boot per BI arm; repeated /start_profile+/stop_profile. "
            "DECODE: warmup x4 primes prefix cache + JIT, throwaway window absorbs "
            "CUPTI init, then per-GEN pure-decode windows (warm prompt prefill = "
            "cache hit). PREFILL: fresh unique-nonce prompt (cold prefill) max_tokens=1."
        ),
        "gen_sweep": L.get("gens"),
        "anchor_gen": L.get("anchor_gen"),
        "tps_official_anchor": L.get("tps_official_anchor"),
        "bi_tax_anchor_pct_759": L.get("bi_tax_anchor_pct_759"),
        "tolerance_rel": L.get("tolerance_rel"),
        "tolerance_abs_pp": L.get("tolerance_abs_pp"),
        # analysis card, per PR
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "prompt_tokens_bi0_prefill": pre0.get("prompt_tokens"),
        "prompt_tokens_bi1_prefill": pre1.get("prompt_tokens"),
    }

    run = wandb.init(project=PROJECT, entity=ENTITY, group=args.group,
                     name="fern/bi-tax-operating-point", id=args.run_id,
                     resume=("allow" if args.run_id else None), config=config)

    summ = {
        # primary metric + verdict
        "bi_tax_decode_ms_per_tok": L.get("bi_tax_decode_ms_per_tok"),
        "bi_tax_operating_point_robust": L.get("bi_tax_operating_point_robust"),
        "decode_bi_tax_pct_at_anchor": L.get("decode_bi_tax_pct_at_anchor"),
        # gen-length flatness
        "decode_tax_pct_min": L.get("decode_tax_pct_min"),
        "decode_tax_pct_max": L.get("decode_tax_pct_max"),
        "decode_tax_pct_mean": L.get("decode_tax_pct_mean"),
        "decode_tax_pct_spread": L.get("decode_tax_pct_spread"),
        "decode_tax_max_dev_from_anchor": L.get("decode_tax_max_dev_from_anchor"),
        # prefill split
        "prefill_bi_tax_ms_total": L.get("prefill_bi_tax_ms_total"),
        "prefill_bi_tax_pct": L.get("prefill_bi_tax_pct"),
        "prefill_ms_total_bi0": pre0.get("prefill_ms_total"),
        "prefill_ms_total_bi1": pre1.get("prefill_ms_total"),
        "prefill_ms_per_prompt_tok_bi0": pre0.get("prefill_ms_per_prompt_tok"),
        "prefill_ms_per_prompt_tok_bi1": pre1.get("prefill_ms_per_prompt_tok"),
        # prediction band
        "bi1_band_lo_tps": L.get("bi1_band_lo_tps"),
        "bi1_band_hi_tps": L.get("bi1_band_hi_tps"),
    }
    # scale factors + per-arm asymptote
    for k, v in (L.get("scale_factor") or {}).items():
        summ[f"scale_factor/bi{k}"] = v
    band = L.get("prediction_band") or {}
    for key, val in band.items():
        if isinstance(key, str) and key.startswith("asymptote"):
            for kk, vv in val.items():
                summ[f"band/{kk}"] = vv
    run.summary.update({k: v for k, v in summ.items() if v is not None})

    # gen-length curve table
    gl = L.get("gen_length_curve") or []
    if gl:
        cols = ["gen", "bi0_decode_ms_per_tok", "bi1_decode_ms_per_tok",
                "added_decode_ms_per_tok", "decode_bi_tax_pct",
                "bi0_decode_tps_proxy", "bi1_decode_tps_proxy"]
        run.log({"gen_length_curve": wandb.Table(
            columns=cols, data=[[r.get(c) for c in cols] for r in gl])})

    # prediction band table (numeric GEN keys only)
    rows = []
    for g in (L.get("gens") or []):
        b = band.get(str(g)) or band.get(g)
        if b:
            rows.append([g, b.get("bi1_implied_official_tps"),
                         b.get("bi0_implied_official_tps"), b.get("op_point_tax_pct")])
    if rows:
        run.log({"prediction_band": wandb.Table(
            columns=["gen", "bi1_implied_official_tps", "bi0_implied_official_tps",
                     "op_point_tax_pct"], data=rows)})

    # anchor (GEN=512) per-family ledger table
    afl = L.get("anchor_family_ledger") or []
    if afl:
        cols = ["family", "bi0_ms_per_tok", "bi1_ms_per_tok", "added_ms_per_tok",
                "share_of_total_added"]
        run.log({"anchor_family_ledger_gen512": wandb.Table(
            columns=cols, data=[[r[c] for c in cols] for r in afl])})

    print(f"[wandb] run={run.id} robust={L.get('bi_tax_operating_point_robust')} "
          f"decode_ms_per_tok={L.get('bi_tax_decode_ms_per_tok')} "
          f"band=[{L.get('bi1_band_lo_tps')},{L.get('bi1_band_hi_tps')}]")
    rid = run.id
    run.finish()
    print(f"RUN_ID={rid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
