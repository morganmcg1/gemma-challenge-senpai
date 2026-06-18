#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #687 (stark) -- Phase-0 feasibility gate for the K=5 WRITE-BACK correcting rescue.

This is an ANALYSIS-ONLY driver (no GPU serve, no HF Job, no submission, served file
untouched). It records the determination of whether a *real* write-back correcting
rescue -- one that recovers the target-verified token id at a flagged verify gap and
SUBSTITUTES it into the emitted stream while keeping the loop's KV/state consistent --
is implementable through the available vLLM-0.22.0 patch/API surface, and logs the
verdict + the carried-forward #669 composition baseline as explicit W&B scalars.

The feasibility conclusion is grounded in direct reads of the served stack
(``submissions/int4_mtp_batchinv/vllm_recompute_acceptor_patch.py`` +
``vllm==0.22.0`` ``GPUModelRunner``); the citations live in ``FINDINGS`` below.
Either feasibility outcome is terminal & decision-grade per the PR:
  * INFEASIBLE -> the local live measurement is impossible; the gated official
    benchmark is the sole speed resolver (the local speed-leg analysis is closed).
  * FEASIBLE   -> Phase 1/2 implement + measure (NOT run by this driver).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# #669 carried-forward measured baseline (committed matched_k5* JSON; analysis only).
# These are the composition this PR set out to convert into a live measurement.
# ---------------------------------------------------------------------------
BASE_669 = {
    "ar_rung_local": 126.75,
    "ar_rung_official_locked": 126.378,          # LOCKED #319
    "plus10_bar_official": 136.378,
    # speed (matched_k5_speed/matched_speed.json)
    "unrescued_k5_ceiling_local": 159.300,
    "pre_fork_flag_rate_per_emit_tau027": 0.04190,   # matched_identity verdict
    "correcting_rescue_tps_local": 140.264,          # composition @ pre-fork rate
    "correcting_rescue_official_equiv": 139.852,
    "correcting_clears_plus10_by": 3.474,            # 139.852 - 136.378
    "dummy_tau_gate_tps_local": 124.975,            # literal over-firing gate
    "dummy_tau_gate_official_equiv": 124.610,       # MISSES +10
    "dummy_tau_realized_fire_rate_per_emit": 0.11251,  # 2.7x the pre-fork rate
    "per_fire_cost_ms": 19.903,
    # identity (matched_k5/matched_identity.json) -- the "nothing to correct" facts
    "ar_vs_ar_floor_pos0": 0,
    "spec_vs_ar_pos0": 2,                # both measured exact-tie prefill artifacts
    "draft_break_count_at_tau027": 0,    # 0 draft-position breaks over 25,154 positions
    "prefork_draft_positions": 25154,
    "confident_genuine_pos0_flips": 0,
    "strict_literal_holds": 0,           # 2/128 raw token diffs (zero-tolerance read)
}

# ---------------------------------------------------------------------------
# Phase-0 feasibility findings (grounded code citations). Each entry: the claim,
# the file:line evidence, and what it blocks.
# ---------------------------------------------------------------------------
FINDINGS: list[dict[str, str]] = [
    {
        "id": "dummy_run_is_a_profiling_stub",
        "claim": "_fire()'s _dummy_run(1,...) cannot recover a real verified token: it "
                 "runs the model on shared DUMMY input_ids/positions with dummy attention "
                 "metadata and slot_mapping filled with -1 (no real KV read/write), and "
                 "returns hidden_states for those garbage inputs.",
        "evidence": "vllm/v1/worker/gpu_model_runner.py:5701-5705 (slot_mapping.fill_(-1), "
                    "'Dummy runs have no real slot assignments'); :5777 input_ids = "
                    "self.input_ids.gpu[:n] (shared dummy buffer); :5785 dummy positions; "
                    ":5907 returns hidden_states for the dummy batch. _fire discards it: "
                    "vllm_recompute_acceptor_patch.py:731-745.",
        "blocks": "Recovering the target-verified token id (part a) via the as-implemented "
                  "fire path is structurally impossible -- it is a cost/capture stub.",
    },
    {
        "id": "no_single_position_real_recompute_primitive",
        "claim": "There is no exposed primitive to run a REAL width-1 target forward at a "
                 "specific flagged sequence position (real token id + real per-request KV "
                 "context + real attention metadata) mid-sample_tokens. The real forward "
                 "path is execute_model(scheduler_output) -> _prepare_inputs, which is "
                 "scheduler-driven over the whole step batch; recomputing one ad-hoc "
                 "position requires hand-reconstructing CommonAttentionMetadata + slot "
                 "mapping + forward_context, i.e. reimplementing a slice of _prepare_inputs.",
        "evidence": "vllm/v1/worker/gpu_model_runner.py:3955 execute_model / :4298 sets "
                    "execute_model_state; _prepare_inputs builds the batch tensors; no "
                    "public recompute_one(req_id,pos)->logits method exists on GPUModelRunner.",
        "blocks": "Sourcing the M=1 correction token through the available surface.",
    },
    {
        "id": "post_hook_is_post_commit",
        "claim": "The patch wraps the WHOLE sample_tokens and runs its logic AFTER "
                 "orig_sample_tokens, which has already (1) run the rejection sampler "
                 "(_sample -> emitted tokens), (2) committed state via "
                 "_update_states_after_model_execute, (3) run the drafter to propose the "
                 "NEXT step's drafts on those emitted tokens, and (4) committed the tokens "
                 "to CPU structures in _bookkeeping_sync (input_batch.token_ids_cpu, "
                 "num_tokens_no_spec, req_state.output_token_ids.extend). Substituting a "
                 "token at the post-hook would require rolling back all four; the wrap "
                 "surface exposes no such rollback, and mutating the GPU sampled_token_ids "
                 "tensor post-hoc is vacuous because the CPU commit already happened.",
        "evidence": "vllm/v1/worker/gpu_model_runner.py:4364 _sample; :4366 "
                    "_update_states_after_model_execute; :4386-4399/4428 "
                    "propose_draft_token_ids; :4489 _bookkeeping_sync -> :3629 token_ids_cpu "
                    "write, :3631 num_tokens_no_spec write, :3635 output_token_ids.extend. "
                    "Patch post-hook: vllm_recompute_acceptor_patch.py:761 output = "
                    "orig_sample_tokens(...) then :816-819 _fire AFTER.",
        "blocks": "Substituting the token into the emitted stream while keeping KV/state "
                  "consistent (part b + consistency).",
    },
    {
        "id": "writeback_is_vacuous_under_BI1",
        "claim": "Even setting implementability aside, under VLLM_BATCH_INVARIANT=1 the M=1 "
                 "recompute argmax provably EQUALS the width-M verify argmax already emitted "
                 "by the greedy rejection sampler: #669 measured draft_break=0 over 25,154 "
                 "pre-fork draft positions at tau=0.27, and the only 2 residual divergences "
                 "are prefill pos-0 exact-tie artifacts that have NO draft row and are "
                 "structurally outside the acceptor (a write-back cannot reach them). So a "
                 "correcting write-back would be a measurement of a no-op at every flaggable "
                 "position.",
        "evidence": "The greedy rejection sampler already STORES the target argmax at the "
                    "first-reject position (vllm/v1/worker/gpu/spec_decode/"
                    "rejection_sampler_utils.py:96-107, temp==0 greedy path uses "
                    "target_local_argmax) -- so the emitted token IS the width-M verify "
                    "argmax. Under VLLM_BATCH_INVARIANT=1 the M=1 recompute returns the "
                    "identical argmax. Empirical confirmation in matched_identity.json: "
                    "draft_break_count_at_tau(0.27)=0, ar_vs_ar_floor pos0=0, spec_vs_ar "
                    "pos0=2 (both prefill exact-tie), confident_genuine_pos0_flips=0.",
        "blocks": "The premise of a measurable correction: there is nothing to correct at "
                  "draft positions, so the 140.2 composition cannot be split from the "
                  "un-rescued ceiling by a local write-back serve.",
    },
]

# Verdict is set from analysis (overridable for record-keeping).
DEFAULT_VERDICT = "WRITEBACK_INFEASIBLE_LOCAL"


def build_result(verdict: str) -> dict[str, Any]:
    feasible = verdict != "WRITEBACK_INFEASIBLE_LOCAL"
    return {
        "pr": 687,
        "leg": "writeback_feasibility_gate",
        "phase": "phase0_feasibility",
        "analysis_only": True,
        "official_tps": 0,
        "no_hf_job": True,
        "fires": False,
        "submission": "int4_mtp_batchinv",
        "vllm_version": "0.22.0",
        "writeback_feasible_local": feasible,
        "verdict": verdict,
        "findings": FINDINGS,
        "baseline_669": BASE_669,
        # boundary discipline: this closes the LOCAL live-measurement question only; the
        # official-equiv assumed-equal-tax conversion is resolved ONLY by the gated
        # official benchmark (denken #677 prices it razor-thin). Do not over-claim official.
        "boundary": ("LOCAL basis only. official-equiv conversion (x126.378/126.75) is the "
                     "assumed-equal-tax map; only the gated official benchmark resolves it. "
                     "A measured-or-projected local HOLDS is a SURFACE + approval trigger, "
                     "not a fire; fire stays quality-blocked by int4-body AIME (ubel/fern)."),
    }


def log_wandb(result: dict[str, Any], name: str, group: str) -> str | None:
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] import failed: {exc!r}; JSON only", flush=True)
        return None
    run = init_wandb_run(
        job_type="local_analysis", agent="stark", name=name, group=group,
        notes="PR#687 Phase-0 feasibility gate for the K=5 write-back correcting rescue "
              "(analysis_only; no GPU serve, no HF Job). Verdict + #669 carried baseline.",
        config={"pr": 687, "phase": "phase0_feasibility", "submission": "int4_mtp_batchinv",
                "vllm_version": "0.22.0", "analysis_only": True, "official_tps": 0,
                "no_hf_job": True, "fires": False, "verdict": result["verdict"]},
    )
    if run is None:
        print("[wandb] disabled; JSON only", flush=True)
        return None
    b = result["baseline_669"]
    summary: dict[str, Any] = {
        # ---- guardrails (carry #669 standard; verified-clean explicit scalars) ----
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        # ---- the feasibility determination ----
        "writeback_feasible_local": int(bool(result["writeback_feasible_local"])),
        "verdict": result["verdict"],
        "n_blocking_findings": len(FINDINGS),
        # ---- carried-forward #669 measured composition (best-available, unchanged) ----
        "baseline669/correcting_rescue_official_equiv": b["correcting_rescue_official_equiv"],
        "baseline669/correcting_clears_plus10_by": b["correcting_clears_plus10_by"],
        "baseline669/dummy_tau_gate_official_equiv": b["dummy_tau_gate_official_equiv"],
        "baseline669/unrescued_k5_ceiling_local": b["unrescued_k5_ceiling_local"],
        "baseline669/draft_break_count_at_tau027": b["draft_break_count_at_tau027"],
        "baseline669/prefork_draft_positions": b["prefork_draft_positions"],
        "baseline669/confident_genuine_pos0_flips": b["confident_genuine_pos0_flips"],
        "baseline669/spec_vs_ar_pos0": b["spec_vs_ar_pos0"],
        "baseline669/plus10_bar_official": b["plus10_bar_official"],
        # ---- terminal-marker primary/test metrics ----
        # primary: the official-equiv this PR sought to MEASURE -> remains the #669
        # projection (now proven un-measurable locally). NOT a write-back measurement.
        "writeback_official_equiv_tps": b["correcting_rescue_official_equiv"],
        # test: full-stream confident flips of the (un-built) correcting serve == the
        # #669 matched-basis residual, unchanged (write-back cannot touch the prefill flips).
        "writeback_full_stream_confident_flips": b["confident_genuine_pos0_flips"],
    }
    for k, v in summary.items():
        if v is not None:
            run.summary[k] = v
    run.summary["findings_json"] = json.dumps(FINDINGS)
    finish_wandb(run)
    print(f"[wandb] logged feasibility run {run.id} verdict={result['verdict']}", flush=True)
    return run.id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdict", default=DEFAULT_VERDICT,
                    choices=["WRITEBACK_INFEASIBLE_LOCAL", "WRITEBACK_FEASIBLE_LOCAL"])
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "research/validity/optionb_rescue_k5_acceptor/writeback_k5")
    ap.add_argument("--wandb-name", default="stark/k5-writeback-feasibility")
    ap.add_argument("--wandb-group", default="optionb-livecert-k5-stark")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    out_dir = a.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_result(a.verdict)
    result["ts"] = time.time()

    print(f"[feasibility] verdict={result['verdict']} "
          f"writeback_feasible_local={result['writeback_feasible_local']} "
          f"n_findings={len(FINDINGS)}", flush=True)
    for f in FINDINGS:
        print(f"  - [{f['id']}] {f['claim'][:88]}...", flush=True)

    run_id = None
    if not a.no_wandb:
        run_id = log_wandb(result, a.wandb_name, a.wandb_group)
    result["wandb_run_id"] = run_id
    (out_dir / "writeback_feasibility.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[feasibility] -> {out_dir / 'writeback_feasibility.json'} (run_id={run_id})",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
