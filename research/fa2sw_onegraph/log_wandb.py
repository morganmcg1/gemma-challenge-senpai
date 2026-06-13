#!/usr/bin/env python
"""Log the fa2sw + onegraph ablation (PR #7) to W&B as a durable record.

Reads the authoritative offline-gate summaries (runs_official/*/summary.json) and
the official greedy-verifier verdicts (constants below, transcribed from
runs_official/gate.log) plus the served-base validation, and logs one run with a
comparison table + scalar summary metrics. No GPU work — this only publishes the
already-measured artifacts so the negative result has a W&B home.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import wandb

RES = Path(__file__).resolve().parent
RUNS = RES / "runs_official"

# Greedy verdicts vs base, official verifier (check_greedy_identity.py), from gate.log.
VERDICTS = {
    "base": {"verdict": "REFERENCE", "identical": 128, "divergent_prompts": 0, "divergent_tokens": 0},
    "fa2sw": {"verdict": "DIVERGENT", "identical": 46, "divergent_prompts": 82, "divergent_tokens": 12075},
    "onegraph": {"verdict": "DIVERGENT", "identical": 127, "divergent_prompts": 1, "divergent_tokens": 59},
    "both": {"verdict": "DIVERGENT", "identical": 46, "divergent_prompts": 82, "divergent_tokens": 11767},
}
# Served base (serve_runs/base + base_modality_recheck), levers OFF — what ships.
SERVED_BASE = {
    "ppl": 2.005477944404151,
    "ppl_cap": 2.42,
    "completed_decode": 128,
    "output_len": 512,
    "modalities_all_ok": True,
    "modality_text": True,
    "modality_image": True,
    "modality_audio": True,
}


def main() -> int:
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "senpai-v1"),
        name="denken/fa2sw-onegraph",
        group="fa2sw_onegraph",
        job_type="ablation",
        tags=["gemma-8gpu-progress-20260613", "pr-7", "denken", "negative-result",
              "fa2sw", "onegraph", "greedy-identity", "conc1"],
        config={
            "pr": 7,
            "model": "google/gemma-4-E4B-it-qat-w4a16-ct",
            "regime": "M=1 AR (sequential, prefix-cache OFF) — int4-deterministic",
            "hardware": "AWS A10G (local, exploratory)",
            "tps_tokens": 256,
            "tps_repeats": 3,
            "greedy_gate_prompts": 128,
            "greedy_gate_gen_tokens": 256,
            "verifier": "gemma_greedy_identity_verifier_flowian-powers/check_greedy_identity.py",
            "levers_shipped": "OFF (FA2SW=0, ONEGRAPH=0)",
            "official_hf_job": "none (local-only PR; not approved/run)",
        },
        notes="Negative result: fa2sw -4.9% TPS + greedy-DIVERGENT; onegraph TPS-parity + "
              "greedy-DIVERGENT (1 near-tie flip); both DIVERGENT. Decode is bandwidth-bound at "
              "conc=1. Ships levers OFF (verified int4 base, PPL 2.005, all modalities).",
    )

    cols = ["variant", "tps_mean", "tps_std", "tps_delta_pct_vs_base", "peak_mem_gb",
            "backend_map", "greedy_verdict", "divergent_prompts", "divergent_tokens"]
    table = wandb.Table(columns=cols)

    base_tps = json.loads((RUNS / "base/summary.json").read_text())["tps"]["tps_mean"]
    per_variant = {}
    for v in ["base", "fa2sw", "onegraph", "both"]:
        d = json.loads((RUNS / v / "summary.json").read_text())
        tps = d["tps"]
        delta = 100.0 * (tps["tps_mean"] - base_tps) / base_tps
        ver = VERDICTS[v]
        table.add_data(
            v, round(tps["tps_mean"], 3), round(tps["tps_std"], 4), round(delta, 2),
            round(d["peak_mem_gb"], 2), json.dumps(d["backend_summary"]),
            ver["verdict"], ver["divergent_prompts"], ver["divergent_tokens"],
        )
        per_variant[v] = {"tps": tps["tps_mean"], "tps_delta_pct": delta, **ver}
        # per-variant scalar metrics
        wandb.summary[f"{v}/tps"] = tps["tps_mean"]
        wandb.summary[f"{v}/tps_delta_pct_vs_base"] = delta
        wandb.summary[f"{v}/greedy_divergent_prompts"] = ver["divergent_prompts"]
        wandb.summary[f"{v}/greedy_valid"] = (ver["verdict"] in ("REFERENCE",))

    wandb.log({"ablation_matrix": table})
    wandb.summary.update({
        "primary_metric_name": "tps_local_conc1_base",
        "primary_metric_value": base_tps,
        "test_metric_name": "served_base_ppl",
        "test_metric_value": SERVED_BASE["ppl"],
        "fa2sw_tps_delta_pct": per_variant["fa2sw"]["tps_delta_pct"],
        "onegraph_tps_delta_pct": per_variant["onegraph"]["tps_delta_pct"],
        "any_lever_valid": False,
        "served_base": SERVED_BASE,
        "verdict": "NEGATIVE — both levers greedy-DIVERGENT (invalid); no valid TPS win",
    })

    print(f"WANDB_RUN_ID={run.id}")
    print(f"WANDB_RUN_URL={run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
