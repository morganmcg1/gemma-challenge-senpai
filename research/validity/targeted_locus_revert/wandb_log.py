#!/usr/bin/env python
"""PR #776 — log the targeted-locus-revert served-determinism verdict to W&B.

Reads served_determinism_verdict.json (produced by analyze_served_determinism.py)
and logs the per-arm self-determinism + wall_tps + verdict as numeric summary
metrics. Run under the repo .venv (has wandb). Project/entity default to
wandb-applied-ai-team/gemma-challenge-senpai (scripts/wandb_logging).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
BASE = Path(__file__).resolve().parent

from scripts.wandb_logging import (finish_wandb, init_wandb_run,  # noqa: E402
                                   log_file_artifact, log_summary)

VERDICT_CODE = {"CONFIRMED_SERVED": 1, "INCONCLUSIVE": 0, "REFUTED_SERVED": -1}


def main() -> int:
    rep_path = BASE / "served_determinism_verdict.json"
    rep = json.loads(rep_path.read_text())
    arms = rep["arms"]

    # Official self-referential greedy gate (greedy_gate.compare) on the served
    # surgical captures — the divergence_count the PR's SENPAI-RESULT asks for.
    gate_path = BASE / "served_interlock_gate.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    sc = gate.get("self_consistency_gate", {})

    def g(arm, key):
        return arms.get(arm, {}).get(key)

    summary = {
        "verdict": rep["verdict"],
        "verdict_code": VERDICT_CODE.get(rep["verdict"], -9),
        # eager (force-2D, no CUDA graphs) — the #761 byte-exact path
        "eager_surgattn_self_determ_frac": g("eager_surgattn", "min_byte_identical_frac"),
        "eager_surgattn_self_deterministic": int(bool(g("eager_surgattn", "self_deterministic"))),
        "eager_surgattn_wall_tps": g("eager_surgattn", "median_wall_tps"),
        # served (force-2D, CUDA graphs) — the production path
        "served_surgattn_self_determ_frac": g("served_surgattn", "min_byte_identical_frac"),
        "served_surgattn_self_deterministic": int(bool(g("served_surgattn", "self_deterministic"))),
        "served_surgattn_wall_tps": g("served_surgattn", "median_wall_tps"),
        "served_surgattn_specoff_self_determ_frac": g("served_surgattn_specoff", "min_byte_identical_frac"),
        "served_surgattn_specoff_wall_tps": g("served_surgattn_specoff", "median_wall_tps"),
        # served batchinv (BI=1) — the PR's claimed byte-exact 157-TPS baseline
        "served_batchinv_self_determ_frac": g("served_batchinv", "min_byte_identical_frac"),
        "served_batchinv_self_deterministic": int(bool(g("served_batchinv", "self_deterministic")))
        if g("served_batchinv", "self_deterministic") is not None else None,
        "served_batchinv_wall_tps": g("served_batchinv", "median_wall_tps"),
        # headline comparison
        "surgical_vs_batchinv_tps_speedup_pct": (
            round(100.0 * (g("served_surgattn", "median_wall_tps") - g("served_batchinv", "median_wall_tps"))
                  / g("served_batchinv", "median_wall_tps"), 2)
            if g("served_surgattn", "median_wall_tps") and g("served_batchinv", "median_wall_tps") else None
        ),
        "eager_self_deterministic": int(bool(rep.get("eager_self_deterministic"))),
        "served_self_deterministic": int(bool(rep.get("served_self_deterministic"))),
        # OFFICIAL gate (greedy_gate.compare) — the PR's divergence_count test metric
        "served_gate_verdict": gate.get("verdict"),
        "divergence_count": gate.get("batch_invariant_self_divergence_tokens"),
        "served_gate_all_greedy_identical": int(bool(sc.get("all_greedy_identical")))
        if sc.get("all_greedy_identical") is not None else None,
        "served_gate_num_divergent_runs": sc.get("num_divergent_runs"),
        "served_gate_wall_tps": gate.get("batch_invariant_wall_tps"),
        "served_gate_tps_cost_pct_vs_official_ref": gate.get("batch_invariant_tps_cost_pct"),
        "output_tps": g("served_surgattn", "median_wall_tps"),  # primary metric
    }

    run = init_wandb_run(
        job_type="served-determinism-locus", agent="lawine",
        name="lawine/targeted-locus-revert",
        group="targeted-locus-revert",
        tags=["targeted-locus-revert", "byte-exact", "served-determinism", "pr776"],
        config={"pr": 776, "submission_fast": "int4_mtp_bi0_surgattn",
                "submission_strict": "int4_mtp_batchinv",
                "num_prompts": 32, "output_len": 512, "max_num_seqs": 1},
    )
    if run is None:
        print("[wandb] run not created (no creds); verdict JSON is the record", flush=True)
        return 0
    log_summary(run, summary, step=0)
    log_file_artifact(run, path=rep_path, name="served_determinism_verdict",
                      artifact_type="served-determinism-verdict")
    print(f"[wandb] run id={run.id} name={run.name} project={run.project} entity={run.entity}", flush=True)
    print(f"[wandb] verdict={rep['verdict']}", flush=True)
    finish_wandb(run)
    # emit the id for the SENPAI-RESULT line
    print(f"WANDB_RUN_ID={run.id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
