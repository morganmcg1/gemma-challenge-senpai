#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #599 (wirbel) -- W&B logger for the cold-start C1 disambiguation card.

Reads coldstart_poff.json (Q2 prefix-OFF re-measure) + operative_319_canonical.json (#588
stock prefix-ON baseline) and logs the combined Q1 (harness-warms) + Q2 (C1-localization)
verdicts. LOCAL analysis_only, NO FIRE: official_tps=0, no HF job, no served-file change.
group=operative-identity-coldstart.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research/validity/operative_identity"
os.environ.setdefault("WANDB_DIR", str(HERE))
sys.path.append(str(ROOT))
from scripts.wandb_logging import (  # noqa: E402
    init_wandb_run, log_event, log_summary, log_json_artifact, finish_wandb,
)

POFF = HERE / "coldstart_poff.json"             # Q2 prefix-OFF re-measure (this card)
CANON = HERE / "operative_319_canonical.json"   # #588 stock prefix-ON baseline
poff = json.loads(POFF.read_text()) if POFF.exists() else {}
canon = json.loads(CANON.read_text()) if CANON.exists() else {}

pv = poff.get("verdicts", {})
plit = poff.get("pairwise_literal", {})
# #588 stock baseline (prefix ON): warm/warm byte-identical; cold pass dissents <=4 ULP.
stock_warm = canon.get("steady_state_literal", {})
stock_cold = canon.get("cold_start_transient", {})

# Q1 verdict is an analytic code-read (no run needed): the official harness scores a WARM
# server -- run_decode_capture (the scored greedy-identity candidate) runs AFTER run_benchmark
# (sglang.bench_serving, --warmup-requests 4 + full 128x512 over the same suite).
Q1_HARNESS_WARMS = True

# Synthesis: the operative-#319 bar is warm-scored iff the harness warms (Q1) AND the warm
# steady state is byte-identical (#588 b_vs_c). Either mitigation collapsing C1 (Q2) is an
# orthogonal confirmation that even a cold server is cheaply made byte-deterministic.
prefix_off_collapses = bool(pv.get("prefix_caching_off_collapses_C1"))
warmup_collapses = bool(pv.get("warmup_pass_collapses_C1"))
served_byte_det_from_pass1 = bool(prefix_off_collapses or warmup_collapses)
operative_319_warm_scored = bool(Q1_HARNESS_WARMS and stock_warm.get("verdict") == "GREEDY_IDENTICAL")

run = init_wandb_run(
    job_type="analysis",
    agent="wirbel",
    name="wirbel/operative-identity-coldstart",
    group="operative-identity-coldstart",
    notes="PR #599: cold-start C1 disambiguation -- is the operative-#319 bar warm-scored? "
          "Q1 (analytic code-read of hf_bucket_single_job.py): the official harness scores a "
          "WARM server -- the scored greedy-identity candidate (run_decode_capture) is generated "
          "AFTER run_benchmark (sglang.bench_serving, --warmup-requests 4 + full 128x512 over the "
          "identical sharegpt suite), so the one-time first-pass cold-start transient C1 is "
          "excluded from scoring. Q2 (re-measure): re-ran the #588 R=3 census with "
          "enable_prefix_caching=False to isolate the prefix-cache leg of C1, vs #588 stock "
          "(prefix ON) as the 2-config factorial baseline. Synthesis: the literal-WARM "
          "operative-#319 bar (base_fullhead b_vs_c GREEDY_IDENTICAL 128/128, 0/65536) is the "
          "operative contract with ZERO margin consumed. analysis_only, NO FIRE.",
    tags=["pr-599", "analysis-only", "no-fire", "operative-319", "greedy-identity",
          "cold-start", "C1-disambiguation", "warm-scored", "served-m1", "prefix-caching"],
    config={
        "pr": 599,
        "agent": "wirbel",
        "analysis_only": True,
        "official_tps": 0,
        "no_hf_job": True,
        "no_served_file_change": True,
        "no_submission": True,
        "wandb_group": "operative-identity-coldstart",
        "vllm_build": poff.get("build", "vllm-0.22.1rc1.dev307+g3e8afdf78"),
        "checkpoint": "google/gemma-4-E4B-it-qat-w4a16-ct (stock int4, native 262k head)",
        "arm": "base_fullhead (FA_SLIDING + SURGICAL_ATTN_USE_3D_OFF 2D attn + PLE fold)",
        "serve_geometry": "MAX_NUM_SEQS=1 (deployed), spec-OFF, greedy temp=0",
        "diagnostic_toggle": "enable_prefix_caching=False (Q2 leg-isolation instrument; NOT a serve change)",
        "official_orchestrator": "speed_benchmark/scripts/hf_bucket_single_job.py",
        "official_decode_harness": "speed_benchmark/scripts/decode_outputs.py (serial, 1 req/prompt)",
        "official_verifier": "gemma_greedy_identity_verifier_flowian-powers/check_greedy_identity.py",
        "num_prompts": poff.get("num_prompts"),
        "output_len": poff.get("output_len"),
        "r_passes": poff.get("r_passes"),
        "eps_star_nat": poff.get("eps_star_nat", 0.25),
        "ulp_nat": poff.get("ulp_nat", 0.0625),
        "cites": ["#588 n32yblfs (canonical operative-#319 bar; b_vs_c GREEDY_IDENTICAL warm)",
                  "#585 2u44yaa1 (int4 not bf16-byte-exact -> R1 self-reference)",
                  "shipped int4_g128_lmhead @126.38 905tbujn (passed official greedy-identity)",
                  "fern #597 int4_g128+MTP (live fire candidate; MTP draft cold-start is a SEPARATE risk)"],
    },
)
if run is None:
    print("[wandb] init returned None (no key/disabled)")
    raise SystemExit(0)

(HERE / "wandb_run_id.txt").write_text(run.id)
log_event(run, "started", step=0, metrics={"status_code": 3})

def _rate(d):
    return d.get("self_determinism_token_rate")

metrics = {
    # --- Q1: official harness warms before the scored greedy-identity pass (analytic) ---
    "q1/official_harness_warms_before_greedy_identity": int(Q1_HARNESS_WARMS),
    "q1/warmup_requests": 4,
    "q1/decode_capture_runs_after_benchmark": 1,
    # --- Q2: cold-start leg localization (re-measured, prefix OFF) ---
    "q2/prefix_caching_off_collapses_C1": int(prefix_off_collapses),
    "q2/warmup_pass_collapses_C1": int(warmup_collapses),
    "q2/served_stack_byte_deterministic_from_pass1": int(served_byte_det_from_pass1),
    "q2/poff_all_three_passes_byte_identical": int(bool(pv.get("all_three_passes_byte_identical"))),
    # prefix-OFF pairwise (this run)
    "poff/a_vs_b_verdict_identical": int(plit.get("a_vs_b", {}).get("verdict") == "GREEDY_IDENTICAL"),
    "poff/a_vs_c_verdict_identical": int(plit.get("a_vs_c", {}).get("verdict") == "GREEDY_IDENTICAL"),
    "poff/b_vs_c_verdict_identical": int(plit.get("b_vs_c", {}).get("verdict") == "GREEDY_IDENTICAL"),
    "poff/a_vs_b_divergent_tokens": plit.get("a_vs_b", {}).get("total_divergent_tokens"),
    "poff/b_vs_c_divergent_tokens": plit.get("b_vs_c", {}).get("total_divergent_tokens"),
    "poff/a_vs_b_token_rate": _rate(plit.get("a_vs_b", {})),
    "poff/peak_gpu_gb": poff.get("peak_gpu_gb"),
    "poff/server_startup_s": poff.get("server_startup_s"),
    # --- #588 stock (prefix ON) baseline, for the 2-config factorial ---
    "stock/warm_warm_byte_identical": int(stock_warm.get("verdict") == "GREEDY_IDENTICAL"),
    "stock/warm_warm_divergent_tokens": stock_warm.get("total_divergent_tokens"),
    "stock/cold_warm_n_divergent_prompts": stock_cold.get("n_divergent_prompts_vs_warm_per_pair"),
    "stock/cold_warm_max_first_div_gap_ulps": stock_cold.get("max_first_div_gap_ulps"),
    # --- Synthesis ---
    "synthesis/operative_319_bar_is_warm_scored": int(operative_319_warm_scored),
    "synthesis/literal_warm_bar_zero_margin_consumed": int(operative_319_warm_scored),
}
metrics = {k: v for k, v in metrics.items() if v is not None}
log_event(run, "coldstart_disambiguation", step=1, metrics=metrics)
for k, v in metrics.items():
    run.summary[k] = v
log_summary(run, poff, step=1)
log_json_artifact(run, name="coldstart_poff", artifact_type="measurement", data=poff)

print(f"[wandb] run id={run.id} group=operative-identity-coldstart "
      f"Q1_warms={Q1_HARNESS_WARMS} prefix_off_collapses_C1={prefix_off_collapses} "
      f"warmup_collapses_C1={warmup_collapses} operative_319_warm_scored={operative_319_warm_scored}")
finish_wandb(run)
