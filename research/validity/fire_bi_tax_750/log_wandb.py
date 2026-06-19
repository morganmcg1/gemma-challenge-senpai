#!/usr/bin/env python3
"""Log PR #750 fire-config BI=1 vs BI=0 served-TPS + strict-#319 identity to W&B.

Two runs, grouped under ``fire_bi_tax_predict`` (one per arm). Reads the
accumulated RESULTS.json (all scalars) + the two compare_identity summaries
(diverged-example tables). 0-GPU; run from a cwd WITHOUT a ./wandb dir, on
/usr/bin/python3 (the only interpreter here with wandb) per the logging gotcha.

Deliverables logged: tps_BI1, tps_BI0, bi_tax_pct, identity_BI1, identity_BI0,
tps_BI1_official_anchored (int4_qat same-checkpoint anchor), clears_126378.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "gemma-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
BAR = 126.378            # int4_g128_lmhead official a10g-small tps (the rung to clear)
ANCHOR_OFFICIAL = 95.463  # int4_qat official a10g-small tps (SAME checkpoint as fire)


def _num_flat(prefix: str, d: dict, out: dict) -> dict:
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            out[key] = int(v)
        elif isinstance(v, (int, float)):
            out[key] = v
        elif isinstance(v, str):
            out[key] = v
    return out


def _log_arm(arm: str, R: dict, ident: dict | None, group: str, run_id: str | None):
    """One W&B run for arm in {'bi1','bi0'}."""
    bi = 1 if arm == "bi1" else 0
    tps = R.get(f"tps_BI{bi}")
    tps_anchored = R.get(f"tps_BI{bi}_official_anchored")
    config = {
        "pr": 750,
        "phase": "fire_bi_tax_predict",
        "arm": arm,
        "VLLM_BATCH_INVARIANT": bi,
        "spec_on": True,
        "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
        "drafter_model": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
        "num_speculative_tokens": 6,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.90,
        "max_num_batched_tokens": 512,
        "max_num_seqs": 1,
        "vllm": "0.22.0",
        "transformers": "5.9.0",
        "tps_protocol": "sglang.bench_serving vllm-chat, 128 prompts, out_len 512, conc 1, rate inf, warmup 4, seed 1, ignore_eos, temp 0",
        "identity_protocol": "official decode_outputs.py /v1/completions int-token greedy max_tokens 512 ignore_eos return_token_ids; candidate=spec-ON vs reference=spec-OFF same engine/BI",
        "anchor_submission": "int4_qat",
        "anchor_checkpoint_same_as_fire": True,
        "anchor_official_tps": ANCHOR_OFFICIAL,
        "bar_submission": "int4_g128_lmhead",
        "bar_official_tps": BAR,
    }
    run = wandb.init(project=PROJECT, entity=ENTITY, group=group,
                     name=f"fern/fire-bi-tax-{arm}", id=run_id,
                     resume=("allow" if run_id else None), config=config,
                     reinit=True)
    flat: dict = {}
    # headline + projection scalars from RESULTS.json (all keys for this arm + shared)
    _num_flat("", R, flat)
    # identity summary for this arm
    if ident is not None:
        flat[f"{arm}/identity_n_exact"] = ident.get("n_exact")
        flat[f"{arm}/identity_n_total"] = ident.get("n_total")
        flat[f"{arm}/identity_rate"] = (
            ident.get("n_exact", 0) / ident.get("n_total", 1) if ident.get("n_total") else None)
        flat[f"{arm}/identity_n_diverged"] = ident.get("n_diverged")
        flat[f"{arm}/identity_n_len_mismatch"] = ident.get("n_len_mismatch")
        flat[f"{arm}/identity_first_div_min"] = ident.get("first_div_min")
        flat[f"{arm}/identity_first_div_mean"] = ident.get("first_div_mean")
        flat[f"{arm}/identity_first_div_max"] = ident.get("first_div_max")
    run.summary.update({k: v for k, v in flat.items() if v is not None})

    # diverged-examples table (rich record for BI=0 break)
    if ident is not None and ident.get("diverged_examples"):
        cols = ["key", "id", "first_div_index", "len_cand", "len_ref", "cand_tok", "ref_tok"]
        data = [[d.get(c) for c in cols] for d in ident["diverged_examples"]]
        run.log({f"{arm}/diverged_examples": wandb.Table(columns=cols, data=data)})

    print(f"[wandb] arm={arm} run={run.id}  tps=BI{bi}:{tps}  "
          f"anchored={tps_anchored}  identity={ident.get('identity') if ident else 'NA'}")
    rid = run.id
    run.finish()
    return rid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--identity-bi1", type=Path, default=None)
    ap.add_argument("--identity-bi0", type=Path, default=None)
    ap.add_argument("--group", default="fire_bi_tax_predict")
    ap.add_argument("--run-id-bi1", default=None)
    ap.add_argument("--run-id-bi0", default=None)
    args = ap.parse_args()

    R = json.loads(args.results.read_text())
    ident1 = json.loads(args.identity_bi1.read_text()) if args.identity_bi1 and args.identity_bi1.exists() else None
    ident0 = json.loads(args.identity_bi0.read_text()) if args.identity_bi0 and args.identity_bi0.exists() else None

    rid1 = _log_arm("bi1", R, ident1, args.group, args.run_id_bi1)
    rid0 = _log_arm("bi0", R, ident0, args.group, args.run_id_bi0)
    print(f"RUN_ID_BI1={rid1}")
    print(f"RUN_ID_BI0={rid0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
