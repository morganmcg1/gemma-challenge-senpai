#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #588 (wirbel) -- W&B logger for the operative-#319 identity formalization.

Reads operative_319_remeasure.json (the served-M=1 base_fullhead re-measurement under the
canonical OPERATIVE predicate) and logs it to W&B with the census re-stamp summary. LOCAL
analysis_only, NO FIRE: official_tps=0, no HF job, no served-file change.

The canonical bar measured here is the OPERATIVE (near-tie-tolerant) free-running greedy
sequence match -- because literal byte-identity is unsatisfiable for the int4 served stack
run-to-run (the GEMV reduction order resolves <=2-ULP top-2 ties either way). base_fullhead
passes iff every run-to-run first-divergence is a near-tie (m1_self_gap<=eps_star=0.125),
i.e. zero semantic flips. This is the program's standing tie-tolerant predicate (fullserve
census operative_identity_rate / benchmark_config det_diffs_all_near_tie; #429 operative=1.0).
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research/validity/operative_identity"
# Append ROOT so the real `wandb` package wins over any wandb/ output dir at ROOT front.
os.environ.setdefault("WANDB_DIR", str(HERE))
sys.path.append(str(ROOT))
from scripts.wandb_logging import (  # noqa: E402
    init_wandb_run, log_event, log_summary, log_json_artifact, finish_wandb,
)

RESULT = HERE / "operative_319_remeasure.json"          # raw driver data (a-referenced legs)
CANONICAL = HERE / "operative_319_canonical.json"        # corrected canonical verdict
result = json.loads(RESULT.read_text()) if RESULT.exists() else {}
canon = json.loads(CANONICAL.read_text()) if CANONICAL.exists() else {}
margin = result.get("margin", {})
ct = canon.get("cold_start_transient", {})
steady = canon.get("steady_state_literal", {})

run = init_wandb_run(
    job_type="analysis",
    agent="wirbel",
    name="wirbel/operative-identity-formalize",
    group="operative-identity-formalize",
    notes="PR #588: pin ONE canonical, measurable operative-#319 greedy-identity predicate, "
          "show base_fullhead passes it at the served M=1 geometry, and re-stamp every banked "
          "census verdict (#556 head, #562 attn, #571 body, #583/#584 spec-dec) onto it. The "
          "full 128x512x3 census CORRECTED the 3-prompt smoke: at served M=1, base_fullhead is "
          "LITERALLY byte-identical run-to-run at WARM steady state (b_vs_c GREEDY_IDENTICAL "
          "128/128, 0/65536 divergent tokens). The only run-to-run nondeterminism is a one-time "
          "first-pass COLD-START transient (lazy Triton-JIT / prefix-cache settling), uniformly "
          "bounded at <=4 ULP (PPL-neutral near-ties). Canonical bar = literal leaderboard bar "
          "vs the int4 self-reference (R1, #585); near-tie tolerance (eps_star=0.25 nat=4 ULP, "
          "MEASURED) is a cold-start robustness margin, not a steady-state necessity. "
          "analysis_only, NO FIRE.",
    tags=["pr-588", "analysis-only", "no-fire", "operative-319", "greedy-identity",
          "census-restamp", "cold-start", "warm-steady-state", "served-m1", "literal-byte-exact"],
    config={
        "pr": 588,
        "agent": "wirbel",
        "analysis_only": True,
        "official_tps": 0,
        "no_hf_job": True,
        "no_served_file_change": True,
        "no_submission": True,
        "wandb_group": "operative-identity-formalize",
        "vllm_build": "vllm-0.22.1rc1.dev307+g3e8afdf78",
        "checkpoint": "google/gemma-4-E4B-it-qat-w4a16-ct (stock int4, native 262k head)",
        "arm": "base_fullhead (FA_SLIDING + SURGICAL_ATTN_USE_3D_OFF 2D attn + PLE fold)",
        "serve_geometry": "MAX_NUM_SEQS=1 (deployed), spec-OFF, greedy temp=0",
        "canonical_predicate": "LITERAL byte-identity (official check_greedy_identity.py "
                               "GREEDY_IDENTICAL, zero tolerance) of the WARM steady-state "
                               "free-running greedy decode to the same int4 checkpoint's plain "
                               "greedy AR decode (R1). Near-tie tolerance (eps_star=0.25 nat=4 "
                               "ULP) is a cold-start robustness envelope, not the steady-state bar.",
        "eps_star": 0.25,                # MEASURED int4-M=1 cold-start near-tie envelope (4 ULP)
        "eps_star_ulps": 4.0,
        "ulp_nat": 0.0625,
        "num_prompts": result.get("num_prompts"),
        "output_len": result.get("output_len"),
        "r_passes": result.get("r_passes"),
        "official_decode_harness": "speed_benchmark/scripts/decode_outputs.py (serial, 1 req/prompt)",
        "official_verifier": "gemma_greedy_identity_verifier_flowian-powers/check_greedy_identity.py",
        "cold_start_mechanism": "first-pass lazy Triton-JIT/FlashInfer-autotune + prefix-cache "
                                "cold-prefill numerics; confined to pass 1; <=4 ULP",
        # census re-stamp anchors (all NO-FIRE; none flips under the canonical predicate)
        "census_arms_restamped": ["#556 head", "#562 attention", "#571 body",
                                  "#583 spec-dec(fern)", "#584 spec-dec(lawine)"],
        # banked confirmations
        "cites": ["#585 2u44yaa1 (int4 not bf16-byte-exact -> R1)",
                  "#429 (operative_identity=1.0; literal 0.9989 reflects cold-start/batched)",
                  "benchmark_config census (determinism_served=0.1875 at BATCHED geometry)",
                  "#564 cilb (selfdet batched geometry)"],
    },
)
if run is None:
    print("[wandb] init returned None (no key/disabled)")
    raise SystemExit(0)

(HERE / "wandb_run_id.txt").write_text(run.id)
log_event(run, "started", step=0, metrics={"status_code": 3})

# Headline scalar metrics (the CANONICAL, corrected verdict the card rides on).
ladder = ct.get("gap_ladder_ulps_per_pair", {})
metrics = {
    # canonical verdict
    "operative/base_fullhead_passes_operative_319": int(bool(canon.get("base_fullhead_passes_operative_319"))),
    "operative/passes_literal_warm_steady_state": int(bool(canon.get("passes_literal_warm_steady_state"))),
    "operative/passes_operative_4ulp_including_cold_start": int(bool(canon.get("passes_operative_4ulp_including_cold_start"))),
    "operative/eps_star_cold_start_envelope_ulps": canon.get("eps_star_cold_start_envelope_ulps"),
    "operative/eps_star_cold_start_envelope_nat": canon.get("eps_star_cold_start_envelope_nat"),
    # warm/warm steady-state self-determinism (the literal leg, b_vs_c)
    "warm/byte_identical": int(steady.get("verdict") == "GREEDY_IDENTICAL"),
    "warm/num_identical_prompts": steady.get("num_identical"),
    "warm/total_divergent_tokens": steady.get("total_divergent_tokens"),
    "warm/self_determinism_token_rate": steady.get("self_determinism_token_rate"),
    # cold-start transient (a vs warm) characterization
    "coldstart/max_first_div_gap_ulps": ct.get("max_first_div_gap_ulps"),
    "coldstart/n_divergent_prompts_per_pair": ct.get("n_divergent_prompts_vs_warm_per_pair"),
    "coldstart/all_first_div_within_4ulp": int(bool(ct.get("all_first_div_within_4ulp"))),
    "coldstart/pairs_physically_identical": int(bool(ct.get("cold_start_pairs_physically_identical"))),
    "coldstart/gap_ladder_0ulp": ladder.get("0"),
    "coldstart/gap_ladder_2ulp": ladder.get("2"),
    "coldstart/gap_ladder_4ulp": ladder.get("4"),
    "coldstart/literal_token_rate_cold_vs_warm": margin.get("literal_self_determinism_token_rate"),
    # census re-stamp
    "census/census_stable_under_canonical_operative": int(bool(canon.get("census_stable_under_canonical_operative"))),
    "serve/peak_gpu_gb": canon.get("peak_gpu_gb") or result.get("peak_gpu_gb"),
    "serve/server_startup_s": canon.get("server_startup_s") or result.get("server_startup_s"),
}
metrics = {k: v for k, v in metrics.items() if v is not None}
log_event(run, "operative_canonical", step=1, metrics=metrics)
log_summary(run, canon, step=1)
for k, v in metrics.items():
    run.summary[k] = v
run.summary["census_stable_under_canonical_operative"] = bool(canon.get("census_stable_under_canonical_operative"))
# both artifacts: canonical verdict + raw driver legs (provenance)
log_json_artifact(run, name="operative_319_canonical", artifact_type="measurement", data=canon)
log_json_artifact(run, name="operative_319_remeasure_raw", artifact_type="measurement", data=result)

print(f"[wandb] run id={run.id} group=operative-identity-formalize "
      f"passes_operative_319={canon.get('base_fullhead_passes_operative_319')} "
      f"warm_byte_identical={steady.get('verdict')} "
      f"coldstart_max_ulp={ct.get('max_first_div_gap_ulps')} "
      f"census_stable={canon.get('census_stable_under_canonical_operative')}")
finish_wandb(run)
