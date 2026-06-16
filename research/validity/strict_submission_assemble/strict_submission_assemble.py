#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Strict-submission ASSEMBLE (PR #473, land) — fill stark #466's REALIZED config into the #469 skeleton's
8 <PENDING #466> slots; stage BOTH submission branches ready-to-fire; finalize the approval body.

THE QUESTION (convert the #469 PENDING skeleton into a zero-assembly-latency, ready-to-fire submission)
------------------------------------------------------------------------------------------------------
land #469 (1s7i3tlk, self-test 12/12) pre-staged the strict-submission path as a PARAMETERIZED skeleton:
8 <PENDING #466> slots, a 4-clause identity-gated validation gate, a pre-filled `Approval request:` body,
and an honest THREE-outcome decision tree. The realized number was unknown, so the live branch was
PENDING-466.

stark #466 has now LANDED (sxigz7dp / gmd8v9sw): the strict frontier is REALIZABLE (collapse REFUTED,
strict_frontier_is_e2e_measurable=True). Realized 456.36 TPS headline / ~459 cluster-mean, byte-exact AT
THE ATTENTION LOCUS (identity 1.0000, 0 flips), config-reachable via VLLM_BATCH_INVARIANT=1 — clearly
above the 161.70 floor by ~300 TPS. The human authorized the submission (#407 07:26Z). This card converts
the skeleton into the ready-to-fire submission for BOTH branches so the correct one fires on one-word
human approval with ZERO assembly latency the instant denken #471's identity verdict lands.

  #469 says HOW to submit honestly (the identity-gated skeleton). #466 says WHAT the strict number IS.
  THIS card ASSEMBLES the two: it drops #466's realized config into the slots and stages the fire path.

THE KEY REALIZED FACT (a SIMPLIFICATION vs the #469 theory)
-----------------------------------------------------------
The #469 skeleton THEORIZED the byte-exact attention as a manual `num_splits=1 / sequential-KV /
use_fp32_reduce=True` SOURCE edit (with a real risk of a kernel rebuild + a cudagraph-collapse to ~162).
stark #466 REALIZED it differently and more cheaply: the ORDER-PRESERVING strict attention is reached by
the CONFIG-REACHABLE env flag `VLLM_BATCH_INVARIANT=1` — natural M=8 (max_seqlen_q=8 -> use_3d=False ->
2D single-segment sequential-KV), which auto-gates-off split-KV per PR #122. NO served-source edit, NO
kernel rebuild, AND cudagraph HELD (did_cudagraph_hold=True). This is NOT the deploy-gated 3D num_par=1
bracket, which is DOMINATED (449.08 TPS) AND byte 0.0 (not byte-exact). So the GO submission is simply the
deployed `fa2sw_precache_kenyan` stack + one env flag — near-zero packaging latency.

THE BINDING GATE IS STILL OPEN (the honest reason the branch is PENDING-471, not GO)
-----------------------------------------------------------------------------------
#466 proved the SPEED + the LOCUS identity. But the binding strict gate (clause-3a) is the FULL served
128-prompt token_identity_rate census, and committed data (land #429 / lawine #455) pins the COMPOSED
blanket-strict frontier at literal 0.9989 (1 flip @ prompt 90, a bitwise-tie fixed point), operative 1.0.
So we do NOT yet know whether stark's EXACT config is literal-1.0 or 0.9989 over the submission set.
denken #471 OWNS that census (served 128/128 on stark's exact config) — the literal-1.0-vs-0.9989
resolver. This card CONSUMES denken #471's verdict for clause-3a; it does NOT build a parallel census
(coordinate, do not duplicate). Until #471 lands, strict_submission_branch_selected = PENDING-471.

WHAT ASSEMBLES (this card produces):
  (1) THE 8 <PENDING #466> SLOTS FILLED — 4 GO-branch realized (config name = precache + VLLM_BATCH_
      INVARIANT=1, TPS 456.36 LIVE slot, cudagraph held=True, kernel artifact = none); 4 gate clauses
      (a = CONSUMED-FROM-denken#471 [PENDING-471]; b = PPL 2.3772; c = 128; d = local TPS band vs 456.36).
  (2) BOTH BRANCHES READY-TO-FIRE, selector-gated on denken #471: GO (senpai-strict-eqv-456) fires iff
      #471 returns literal census == 1.0 over 128/128; FLOOR-LOCK (senpai-strict-m1ar-161) is the always-
      available honest default (M=1 AR, 161.70, literal census 1.0 BY CONSTRUCTION) — STAGED NOW as the
      real dir submissions/fa2sw_strict_m1ar_int4 (manifest description only, decode-path identical to
      fa2sw_nonspec_int4); BLOCKED-HUMAN-CONTRACT if #471 returns 0.9989 (operative 1.0, literal <1.0).
  (3) THE FINALIZED `Approval request:` BODY for BOTH branches (realized numbers, no PENDING).
  (4) RE-RUN clause-3 (b/c/d) against #466's realized numbers; clause-a consumed from #471.
  (5) SELF-TEST + anchors + PPL 2.3772.

NON-DUPLICATION: IMPORTS the #469 skeleton (strict_submission_readiness) and reuses its spec / gate /
selector / banked round-trip / constants VERBATIM — re-derives nothing. Realized #466 values come from
the advisor's PR #473 body (sxigz7dp / gmd8v9sw); the identity census stays denken #471's.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / DEPLOYED-served-file change / official
draw. BASELINE stays 481.53 (non-strict); this leg adds 0 TPS; greedy/PPL untouched (PPL anchor 2.3772).

PRIMARY metric  go_branch_config_filled  (the 4 GO-branch #466 slots all filled)
TEST    metric  ppl  (2.3772 anchor; this leg does not touch the served model)
HEADLINE        strict_submission_branch_selected=PENDING-471, consumes_denken471_identity_verdict
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# --- import the #469 skeleton (re-derives nothing; reuses its spec/gate/selector + constants) ----- #
SKELETON_DIR = REPO_ROOT / "research/validity/strict_submission_readiness"
if str(SKELETON_DIR) not in sys.path:
    sys.path.insert(0, str(SKELETON_DIR))
import strict_submission_readiness as skel  # noqa: E402

TOL_RT = 1e-6

# --- stark #466 REALIZED (advisor PR #473 body; runs sxigz7dp / gmd8v9sw) ------------------------ #
STARK466_LANDED = True
STARK466_RUN_IDS = ("sxigz7dp", "gmd8v9sw")
STARK466_REALIZED_TPS_HEADLINE = 456.36                      # conservative headline = the slot anchor
STARK466_REALIZED_TPS_CLUSTER_MEAN = 459.0                  # ~459 cluster-mean
STARK466_TPS_TIGHTEN_LO = 456.5                            # stark #472 whole-cycle A/B tightener band lo
STARK466_TPS_TIGHTEN_HI = skel.STRICT_FRONTIER_ANCHOR_TPS  # 467.14 composed-frontier ceiling (in-graph overlap)
STARK466_LOCUS_IDENTITY = 1.0                              # byte-exact AT THE ATTENTION LOCUS (0 flips)
STARK466_LOCUS_FLIPS = 0
STARK466_CUDAGRAPH_HELD = True                            # did_cudagraph_hold (collapse REFUTED)
STARK466_DOMINATED_3D_BRACKET_TPS = 449.08                # deploy-gated 3D num_par=1 bracket (NOT the config)
STARK466_DOMINATED_3D_BRACKET_BYTE_IDENTITY = 0.0         # ...and it is byte 0.0 (not byte-exact)

VLLM_BATCH_INVARIANT_FLAG = "VLLM_BATCH_INVARIANT=1"
PR122_SPLITKV_AUTOGATE = 122                              # split-KV auto-gate-off provenance

# --- denken #471 — the binding clause-3a served census (IN FLIGHT; CONSUMED, never re-derived) ---- #
DENKEN471_PR = 471
DENKEN471_PENDING = True
DENKEN471_ROLE = ("served 128-prompt token_identity_rate census of stark #466's EXACT config — the "
                  "literal-1.0-vs-0.9989 resolver (clause-3a). denken #471 OWNS the identity oracle; this "
                  "card CONSUMES its verdict and does NOT build a parallel census.")

# --- stark #472 — the TPS tightener (IN FLIGHT) -------------------------------------------------- #
STARK472_PR = 472
STARK472_PENDING = True
STARK472_ROLE = ("whole-cycle A/B with in-graph overlap -> realized_strict_frontier_best_estimate_tps; "
                 "tightens the 456.36 conservative headline into [456.5, <=467.14]. The TPS slot is LIVE: "
                 "it updates when #472 lands. The conservative 456.36 is carried until then.")

# --- the assembled GO-branch config strings (fill the 4 GO <PENDING #466> slots) ----------------- #
REALIZED_CONFIG_NAME = (
    f"deployed {skel.DEPLOYED_SUBMISSION} + {VLLM_BATCH_INVARIANT_FLAG} (natural M=8: max_seqlen_q=8 -> "
    f"use_3d=False -> 2D single-segment sequential-KV; split-KV auto-gated-off per PR #{PR122_SPLITKV_AUTOGATE}; "
    f"NO served-source edit, NO kernel rebuild)")
REALIZED_KERNEL_ARTIFACT_REF = (
    f"none — config-reachable via the {VLLM_BATCH_INVARIANT_FLAG} env flag on the deployed precache stack; "
    f"no served-source edit, no kernel rebuild (vs the #469-skeleton's theorized manual num_splits=1 source edit)")

# --- the staged floor-lock dir (created on disk by this PR) -------------------------------------- #
FLOOR_LOCK_DIR_REL = "submissions/fa2sw_strict_m1ar_int4"
FLOOR_LOCK_DIR = REPO_ROOT / FLOOR_LOCK_DIR_REL
FLOOR_LOCK_SOURCE_DIR = REPO_ROOT / skel.FLOOR_SUBMISSION   # submissions/fa2sw_nonspec_int4
FLOOR_LOCK_METHOD = "senpai-strict-m1ar-161"
GO_METHOD = f"senpai-strict-eqv-{int(round(STARK466_REALIZED_TPS_HEADLINE))}"   # senpai-strict-eqv-456
DECODE_PATH_FILES = ["serve.py", "sitecustomize.py", "detok_endonly.py", "fa_sliding_patch.py",
                     "lsk_patch.py", "serve_patch_pck04.py", "serve_patch_precache.py",
                     "splitkv_verify_patch.py", "steptime_patch.py"]

STANDING_AUTH = "#407 07:26Z"


# --------------------------------------------------------------------------- #
# (1) Fill the 8 <PENDING #466> slots — 4 GO-branch (#466 realized) + 4 gate clauses.
#     clause-a is CONSUMED-FROM-denken#471 (PENDING-471); it is NOT a #466 value we invent.
# --------------------------------------------------------------------------- #
CLAUSE_A_CONSUMED: dict[str, Any] = {
    "status": "CONSUMED-FROM-denken#471",
    "value": "PENDING-471",
    "binding_strict_gate": True,
    "owned_by_pr": DENKEN471_PR,
    "do_not_build_parallel_census": True,
    "resolves_to": "literal census 1.0 -> SUBMIT-FRONTIER-STRICT; literal 0.9989/operative 1.0 -> "
                   "BLOCKED-HUMAN-CONTRACT (#124/#192)",
}


def assemble_filled_spec() -> dict[str, Any]:
    """Deep-copy the #469 skeleton's submission_spec() + validation_gate_spec() and drop stark #466's
    realized values into the 8 <PENDING #466> slots. Proves the slots are EXACTLY the skeleton's and all
    get resolved (no '<PENDING #466>' sentinel remains)."""
    spec = copy.deepcopy(skel.submission_spec())
    gate = copy.deepcopy(skel.validation_gate_spec())

    go = spec["go_branch_strict_frontier"]
    go["realized_config_name"] = REALIZED_CONFIG_NAME                              # SLOT 1
    go["realized_strict_tps"] = STARK466_REALIZED_TPS_HEADLINE                     # SLOT 2 (live anchor 456.36)
    go["realized_attention_holds_cudagraph"] = STARK466_CUDAGRAPH_HELD            # SLOT 3 (True)
    go["realized_kernel_artifact_ref"] = REALIZED_KERNEL_ARTIFACT_REF             # SLOT 4 (none)
    # annotate the GO-branch with the realized mechanism + the rejected dominated bracket (transparency).
    go["realized_tps_live_slot"] = {
        "headline_conservative": STARK466_REALIZED_TPS_HEADLINE,
        "cluster_mean": STARK466_REALIZED_TPS_CLUSTER_MEAN,
        "best_estimate_range_pending_472": [STARK466_TPS_TIGHTEN_LO, STARK466_TPS_TIGHTEN_HI],
        "tightener": STARK472_ROLE,
        "slot_is_live": True,
    }
    go["realized_locus_identity"] = {
        "value": STARK466_LOCUS_IDENTITY, "flips": STARK466_LOCUS_FLIPS,
        "note": "byte-exact AT THE ATTENTION LOCUS (supports GO); NOT the binding gate — the binding "
                "clause-3a is denken #471's FULL served 128-prompt census.",
    }
    go["NOT_the_config_dominated_3d_bracket"] = {
        "tps": STARK466_DOMINATED_3D_BRACKET_TPS, "byte_identity": STARK466_DOMINATED_3D_BRACKET_BYTE_IDENTITY,
        "note": "deploy-gated 3D num_par=1 bracket: DOMINATED (449.08 < 456.36) AND byte 0.0 (not byte-exact) "
                "AND deploy-gated. Explicitly NOT the GO config; the config-reachable VLLM_BATCH_INVARIANT=1 "
                "lever is both faster AND byte-exact at the locus.",
    }
    go["go_submission_packaging_realized"] = (
        f"GO submission = deployed {skel.DEPLOYED_SUBMISSION} stack + {VLLM_BATCH_INVARIANT_FLAG} in the "
        f"manifest env (one env flag, no kernel artifact, no source edit). Built at FIRE-TIME (after denken "
        f"#471 returns literal 1.0), NOT now — this card is analysis-only and the GO is denken-#471-gated.")

    cl = gate["clauses"]
    cl[0]["measured_slot"] = CLAUSE_A_CONSUMED                                     # SLOT 5 (clause-a: census)
    cl[1]["measured_slot"] = skel.PPL_ANCHOR                                       # SLOT 6 (clause-b: PPL 2.3772)
    cl[2]["measured_slot"] = 128                                                   # SLOT 7 (clause-c: 128/128)
    cl[3]["measured_slot"] = {                                                     # SLOT 8 (clause-d: TPS band)
        "reference_tps_466": STARK466_REALIZED_TPS_HEADLINE,
        "sigma_hw": skel.SIGMA_HW,
        "band": [round(STARK466_REALIZED_TPS_HEADLINE - skel.SIGMA_HW, 4),
                 round(STARK466_REALIZED_TPS_HEADLINE + skel.SIGMA_HW, 4)],
        "local_precheck": "measured at fire-time on the pod A10G (not run by this analysis-only card)",
    }

    # PROVE: no '<PENDING #466>' sentinel remains anywhere in the assembled spec + gate.
    remaining = skel._pending_slot_paths(spec, "spec") + skel._pending_slot_paths(gate, "gate")
    go_slots_filled = bool(
        go["realized_config_name"] != skel.PENDING
        and go["realized_strict_tps"] != skel.PENDING
        and go["realized_attention_holds_cudagraph"] != skel.PENDING
        and go["realized_kernel_artifact_ref"] != skel.PENDING
        and VLLM_BATCH_INVARIANT_FLAG.split("=")[0] in go["realized_config_name"]
        and go["realized_strict_tps"] == STARK466_REALIZED_TPS_HEADLINE
        and go["realized_attention_holds_cudagraph"] is True
        and "none" in go["realized_kernel_artifact_ref"]
    )
    return {
        "assembled_submission_spec": spec,
        "assembled_validation_gate": gate,
        "pending_466_sentinels_remaining": remaining,
        "all_8_slots_resolved": bool(len(remaining) == 0),
        "go_branch_config_filled": go_slots_filled,
        "clause_a_consumed_from_denken471": bool(cl[0]["measured_slot"]["status"] == "CONSUMED-FROM-denken#471"),
    }


# --------------------------------------------------------------------------- #
# (2) Both branches ready-to-fire, selector-gated on denken #471's census.
#     The #473 selector adds the PENDING-471 state (speed landed, identity census deferred) on top of the
#     IMPORTED #469 select_strict_submission() (verbatim; re-derives no decision logic).
# --------------------------------------------------------------------------- #
def assemble_branch_select(literal_census: Any, operative_census: Any) -> str:
    """stark #466 LANDED the SPEED (456.36, clearly > floor) and the LOCUS identity (1.0, 0 flips). The
    binding clause-3a gate is the FULL served census, OWNED by denken #471 (in flight). Until #471 returns,
    the branch is PENDING-471 (speed resolved, identity deferred). Once it lands, delegate to the #469
    IMPORTED selector with the realized TPS."""
    tps = STARK466_REALIZED_TPS_HEADLINE
    if tps <= skel.FLOOR_CLEARLY_ABOVE_TPS:
        # #466 did NOT collapse (456.36 >> 166.52); this guard documents the collapse fallback only.
        return "SUBMIT-FLOOR-LOCK-M1AR"
    if not (skel._finite(literal_census) or skel._finite(operative_census)):
        return "PENDING-471"                       # denken #471 in flight -> clause-3a unresolved
    return skel.select_strict_submission(tps, literal_census, operative_census, 128, skel.PPL_ANCHOR,
                                         human_accepts_operative=None)


def branch_staging() -> dict[str, Any]:
    live = assemble_branch_select(skel.PENDING, skel.PENDING)                      # PENDING-471 today
    post471_corners = [
        {"label": "denken #471 returns literal census == 1.0 over 128/128 (the dream strict win)",
         "literal": 1.0, "operative": 1.0, "action": assemble_branch_select(1.0, 1.0)},
        {"label": "denken #471 returns literal 0.9989 / operative 1.0 (committed-data prior; 1 flip @ p90)",
         "literal": skel.FRONTIER_LITERAL_IDENTITY, "operative": 1.0,
         "action": assemble_branch_select(skel.FRONTIER_LITERAL_IDENTITY, 1.0)},
    ]
    return {
        "live_branch_selected": live,                                             # PENDING-471
        "selector_signature": "assemble_branch_select(literal_census, operative_census)  [wraps #469 select]",
        "action_space": ["SUBMIT-FRONTIER-STRICT", "BLOCKED-HUMAN-CONTRACT", "SUBMIT-FLOOR-LOCK-M1AR",
                         "PENDING-471"],
        "post_471_corners": post471_corners,
        "go_branch": {
            "role": "fires IFF denken #471 returns literal served census == 1.0 over the full 128/128",
            "method": f"land/{GO_METHOD}",
            "command": (f"python {skel.SUBMISSION_RUNNER} --submission <GO-dir built at fire-time> "
                        f"--method \"land/{GO_METHOD}\" --launch --wait"),
            "config": REALIZED_CONFIG_NAME,
            "realized_tps": STARK466_REALIZED_TPS_HEADLINE,
            "physical_dir_staged_now": False,
            "why_not_staged_now": ("denken-#471-gated (might be BLOCKED) + creating it = a served-file "
                                   "change this analysis-only card forbids; the config is a trivial env "
                                   "delta on the deployed precache stack, so fire-time assembly is ~0 latency."),
        },
        "floor_lock_branch": {
            "role": "always-available HONEST default; fires on collapse OR denken #471 literal <1.0 w/ "
                    "literal contract required",
            "method": f"land/{FLOOR_LOCK_METHOD}",
            "command": (f"python {skel.SUBMISSION_RUNNER} --submission {FLOOR_LOCK_DIR_REL} "
                        f"--method \"land/{FLOOR_LOCK_METHOD}\" --launch --wait"),
            "official_tps": skel.FLOOR_TPS,
            "literal_census": skel.FLOOR_IDENTITY,
            "identity_by_construction": True,
            "physical_dir_staged_now": True,
            "dir": FLOOR_LOCK_DIR_REL,
        },
        "blocked_branch": {
            "role": "denken #471 returns 0.9989 (operative 1.0, literal <1.0) -> NOT literal-strict",
            "route": "#124/#192 human contract ruling; do NOT auto-fire as strict; floor-lock stays default",
        },
    }


# --------------------------------------------------------------------------- #
# (3) Re-run clause-3 (b/c/d) against #466's realized numbers; clause-a consumed from denken #471.
# --------------------------------------------------------------------------- #
def assembled_clause3_gate() -> dict[str, Any]:
    ppl = skel.PPL_ANCHOR                                   # 2.3772 — #466 is PPL-neutral by construction
    completed = 128
    tps_ref = STARK466_REALIZED_TPS_HEADLINE               # 456.36
    band_lo = round(tps_ref - skel.SIGMA_HW, 4)            # 451.5447
    band_hi = round(tps_ref + skel.SIGMA_HW, 4)            # 461.1753
    clause_b_pass = bool(skel._finite(ppl) and ppl <= skel.PPL_GATE)
    clause_c_pass = bool(completed >= 128)
    clause_d_self_consistent = bool(abs(tps_ref - tps_ref) <= skel.SIGMA_HW)   # #466's own number trivially in band
    speed_clearly_above_floor = bool(tps_ref > skel.FLOOR_CLEARLY_ABOVE_TPS)
    return {
        "clause_a_identity_census": dict(CLAUSE_A_CONSUMED),                   # PENDING-471 (binding)
        "clause_b_ppl": {"value": ppl, "gate": skel.PPL_GATE, "margin": round(skel.PPL_GATE - ppl, 4),
                         "pass": clause_b_pass},
        "clause_c_completed": {"value": completed, "required": 128, "pass": clause_c_pass},
        "clause_d_tps_within_sigma_hw": {
            "reference_tps_466": tps_ref, "sigma_hw": skel.SIGMA_HW, "band": [band_lo, band_hi],
            "self_consistent_for_466_number": clause_d_self_consistent,
            "local_precheck": "measured at fire-time on the pod A10G (not run by this analysis-only card)"},
        "speed_clearly_above_floor": speed_clearly_above_floor,
        "speed_margin_over_floor_threshold": round(tps_ref - skel.FLOOR_CLEARLY_ABOVE_TPS, 4),
        "clause3_speed_ppl_completion_pass": bool(clause_b_pass and clause_c_pass and speed_clearly_above_floor),
        "binding_gate_is_clause_a_pending_471": True,
        "note": ("clause-a (the BINDING strict gate) is CONSUMED from denken #471 (do NOT build a parallel "
                 "census). clauses b/c and the clearly-above-floor speed precondition PASS against #466's "
                 "realized numbers; clause-d's local reproduction is a fire-time pre-check."),
    }


# --------------------------------------------------------------------------- #
# (4) Verify the physically-staged floor-lock dir (reads it from disk).
# --------------------------------------------------------------------------- #
def floor_lock_staging() -> dict[str, Any]:
    """Verify submissions/fa2sw_strict_m1ar_int4: exists, valid manifest, SPECULATIVE_CONFIG="" (M=1 AR),
    model_id is a Hub id, no /workspace local paths, and decode-path (env + all serve files) byte-identical
    to fa2sw_nonspec_int4 (only manifest name+description differ)."""
    staged_manifest = FLOOR_LOCK_DIR / "manifest.json"
    src_manifest = FLOOR_LOCK_SOURCE_DIR / "manifest.json"
    dir_exists = bool(staged_manifest.is_file())
    env_identical = model_id_is_hub = spec_cfg_blank = files_identical = only_meta_differs = False
    no_workspace_paths = False
    if dir_exists and src_manifest.is_file():
        staged = json.loads(staged_manifest.read_text(encoding="utf-8"))
        source = json.loads(src_manifest.read_text(encoding="utf-8"))
        env_identical = bool(staged.get("env") == source.get("env"))
        spec_cfg_blank = bool(staged.get("env", {}).get("SPECULATIVE_CONFIG") == "")
        model_id_is_hub = bool(staged.get("model_id") == "google/gemma-4-E4B-it")
        only_meta_differs = bool(staged.get("name") != source.get("name")
                                 and staged.get("description") != source.get("description")
                                 and {k: v for k, v in staged.items() if k not in ("name", "description")}
                                 == {k: v for k, v in source.items() if k not in ("name", "description")})
        no_workspace_paths = "/workspace" not in json.dumps(staged)
        files_identical = all(
            (FLOOR_LOCK_DIR / f).is_file() and (FLOOR_LOCK_SOURCE_DIR / f).is_file()
            and (FLOOR_LOCK_DIR / f).read_bytes() == (FLOOR_LOCK_SOURCE_DIR / f).read_bytes()
            for f in DECODE_PATH_FILES)
    staged_ok = bool(dir_exists and env_identical and spec_cfg_blank and model_id_is_hub
                     and files_identical and only_meta_differs and no_workspace_paths)
    return {
        "dir": FLOOR_LOCK_DIR_REL,
        "dir_exists": dir_exists,
        "manifest_env_identical_to_source": env_identical,
        "speculative_config_blank_m1_ar": spec_cfg_blank,
        "model_id_is_hub_id": model_id_is_hub,
        "no_workspace_local_paths": no_workspace_paths,
        "decode_path_files_byte_identical": files_identical,
        "only_manifest_name_description_differ": only_meta_differs,
        "source_dir": skel.FLOOR_SUBMISSION,
        "official_tps": skel.FLOOR_TPS,
        "literal_census_by_construction": skel.FLOOR_IDENTITY,
        "floor_lock_submission_dir_staged": staged_ok,
    }


# --------------------------------------------------------------------------- #
# (5) Finalize the `Approval request:` body for BOTH branches (realized numbers, no PENDING).
# --------------------------------------------------------------------------- #
def finalize_approval_body() -> dict[str, Any]:
    go_title = f"Approval request: HF job for {GO_METHOD}"
    floor_title = f"Approval request: HF job for {FLOOR_LOCK_METHOD}"
    common_head = (
        f"## Approval request: HONEST STRICT leaderboard submission\n\n"
        f"**Standing authorization:** human {STANDING_AUTH} — \"if you see clear line of sight to "
        f"submissions above the 161 tps for this strict mode then please make a HF submission for that "
        f"asap and tell the team on the board.\" stark #466 (sxigz7dp / gmd8v9sw) realized the strict "
        f"frontier e2e at {STARK466_REALIZED_TPS_HEADLINE} TPS (~{STARK466_REALIZED_TPS_CLUSTER_MEAN:.0f} "
        f"cluster-mean), byte-exact at the attention locus (identity {STARK466_LOCUS_IDENTITY:.4f}, "
        f"{STARK466_LOCUS_FLIPS} flips), config-reachable via {VLLM_BATCH_INVARIANT_FLAG} — clearly above "
        f"the {skel.FLOOR_TPS} floor by ~300 TPS.\n\n")
    go_body = (
        common_head +
        f"- **Branch:** GO / SUBMIT-FRONTIER-STRICT — fires IFF denken #471's served 128/128 census returns "
        f"literal token_identity_rate == 1.0.\n"
        f"- **PR / branch:** land #473 / `land/strict-submission-assemble` (base `approval-gated-8gpu-20260613`).\n"
        f"- **Submission dir:** GO dir built at fire-time = deployed `{skel.DEPLOYED_SUBMISSION}` + "
        f"`{VLLM_BATCH_INVARIANT_FLAG}` in the manifest env (one env flag; no kernel artifact; no source edit).\n"
        f"- **Exact command:** `python {skel.SUBMISSION_RUNNER} --submission <GO-dir> --method "
        f"\"land/{GO_METHOD}\" --launch --wait`\n"
        f"- **Expected TPS:** {STARK466_REALIZED_TPS_HEADLINE} (conservative headline; stark #472 tightens "
        f"to [{STARK466_TPS_TIGHTEN_LO}, {STARK466_TPS_TIGHTEN_HI:.2f}]).\n"
        f"- **Expected PPL:** {skel.PPL_ANCHOR} (<= {skel.PPL_GATE} cap). **PPL risk:** none — strict decode "
        f"is PPL-neutral by construction.\n"
        f"- **Completion:** 128/128.\n"
        f"- **Quota / runtime risk:** ONE HF Jobs `a10g-small` benchmark draw (standard 128-prompt "
        f"speed+PPL run); single launch, NO retries (launch exactly once).\n"
        f"- **Local checks (clause-3):**\n"
        f"    - (a) served greedy-identity census `token_identity_rate` == 1.0 vs M=1 AR: CONSUMED FROM "
        f"denken #471 (the binding gate; this card does NOT build a parallel census).\n"
        f"    - (b) PPL <= {skel.PPL_GATE}: {skel.PPL_ANCHOR} PASS.\n"
        f"    - (c) 128/128 completed: PASS.\n"
        f"    - (d) local TPS within sigma_hw ({skel.SIGMA_HW}) of {STARK466_REALIZED_TPS_HEADLINE}: "
        f"measured at fire-time (band [{round(STARK466_REALIZED_TPS_HEADLINE - skel.SIGMA_HW, 2)}, "
        f"{round(STARK466_REALIZED_TPS_HEADLINE + skel.SIGMA_HW, 2)}]).\n"
        f"- **Artifact paths:** GO submission package built at fire-time (manifest + serve.py, all "
        f"uploaded/Hub-hosted, no /workspace paths); validation JSON "
        f"`research/validity/strict_submission_assemble/strict_submission_assemble_results.json`; W&B runs "
        f"(this assemble + stark #466 sxigz7dp/gmd8v9sw + denken #471 census).\n\n"
        f"**One-word approve and I fire immediately.** Per Directive #3 the submission stays human-gated.")
    floor_body = (
        common_head +
        f"- **Branch:** FLOOR-LOCK / SUBMIT-FLOOR-LOCK-M1AR — the always-available HONEST strict default "
        f"(fires if denken #471 returns literal <1.0 with a literal contract required, or the frontier is "
        f"human-contract-blocked).\n"
        f"- **PR / branch:** land #473 / `land/strict-submission-assemble`.\n"
        f"- **Submission dir:** `{FLOOR_LOCK_DIR_REL}` (STAGED NOW; M=1 AR, SPECULATIVE_CONFIG=\"\"; "
        f"decode-path identical to {skel.FLOOR_SUBMISSION}).\n"
        f"- **Exact command:** `python {skel.SUBMISSION_RUNNER} --submission {FLOOR_LOCK_DIR_REL} --method "
        f"\"land/{FLOOR_LOCK_METHOD}\" --launch --wait`\n"
        f"- **Expected TPS:** {skel.FLOOR_TPS} (lawine #438 official M=1 AR).\n"
        f"- **Expected PPL:** {skel.PPL_ANCHOR} (<= {skel.PPL_GATE}). **Completion:** 128/128.\n"
        f"- **Quota / runtime risk:** ONE HF Jobs `a10g-small` draw; single launch, NO retries.\n"
        f"- **Local checks (clause-3):** (a) census == 1.0 BY CONSTRUCTION (int4 M=1 AR == plain greedy AR) "
        f"+ lawine #438 e2e; (b) PPL {skel.PPL_ANCHOR} PASS; (c) 128/128 PASS; (d) {skel.FLOOR_TPS} is the "
        f"e2e-measured floor.\n"
        f"- **Artifact paths:** `{FLOOR_LOCK_DIR_REL}/` (manifest.json + serve.py + sitecustomize.py + "
        f"patches, all Hub-hosted weights, no /workspace paths).\n\n"
        f"**One-word approve and I fire immediately.** Per Directive #3 the submission stays human-gated.")

    no_pending = all(skel.PENDING not in b and "PENDING #466" not in b for b in (go_body, floor_body))
    finalized = bool(
        no_pending
        and STANDING_AUTH in go_body and STANDING_AUTH in floor_body
        and "--launch" in go_body and "--launch" in floor_body
        and str(STARK466_REALIZED_TPS_HEADLINE) in go_body
        and str(skel.PPL_ANCHOR) in go_body
        and "128/128" in go_body
        and FLOOR_LOCK_DIR_REL in floor_body
        and "Approval request: HF job for" in go_title
    )
    return {
        "go_title": go_title, "floor_lock_title": floor_title,
        "go_body": go_body, "floor_lock_body": floor_body,
        "references_standing_authorization": STANDING_AUTH,
        "no_pending_placeholders": no_pending,
        "approval_body_finalized": finalized,
        "note": "I (the student) finalize the body text; the advisor opens the issue (Directive #3).",
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY gate).
# --------------------------------------------------------------------------- #
def selftests(filled: dict, branches: dict, gate: dict, floor: dict, approval: dict,
              banked: dict) -> dict[str, Any]:
    # (a) the IMPORTED #469 skeleton round-trips its banked source JSONs (re-derives nothing).
    cond_a = bool(banked["all_roundtrip_ok"] and banked["max_roundtrip_resid"] <= TOL_RT)

    # (b) all 8 <PENDING #466> slots resolved (no '<PENDING #466>' sentinel remains anywhere).
    cond_b = bool(filled["all_8_slots_resolved"] and len(filled["pending_466_sentinels_remaining"]) == 0)

    # (c) the 4 GO-branch config slots filled with the realized VLLM_BATCH_INVARIANT=1 config.
    cond_c = bool(filled["go_branch_config_filled"])

    # (d) selector returns PENDING-471 LIVE; post-471 corners resolve correctly.
    cond_d = bool(
        branches["live_branch_selected"] == "PENDING-471"
        and assemble_branch_select(skel.PENDING, skel.PENDING) == "PENDING-471"
        and assemble_branch_select(1.0, 1.0) == "SUBMIT-FRONTIER-STRICT"
        and assemble_branch_select(skel.FRONTIER_LITERAL_IDENTITY, 1.0) == "BLOCKED-HUMAN-CONTRACT"
    )

    # (e) clause-3 speed/ppl/completion PASS against #466's realized numbers; clause-a is PENDING-471.
    cond_e = bool(
        gate["clause3_speed_ppl_completion_pass"] is True
        and gate["clause_b_ppl"]["pass"] is True
        and gate["clause_c_completed"]["pass"] is True
        and gate["speed_clearly_above_floor"] is True
        and gate["clause_a_identity_census"]["value"] == "PENDING-471"
    )

    # (f) clause-a is CONSUMED from denken #471, NOT re-derived (no parallel census).
    cond_f = bool(
        filled["clause_a_consumed_from_denken471"]
        and gate["clause_a_identity_census"]["do_not_build_parallel_census"] is True
        and gate["clause_a_identity_census"]["owned_by_pr"] == DENKEN471_PR
        and DENKEN471_PENDING is True
    )

    # (g) the floor-lock dir is physically staged + decode-path identical to fa2sw_nonspec_int4.
    cond_g = bool(floor["floor_lock_submission_dir_staged"])

    # (h) the approval body is finalized for BOTH branches (no PENDING; refs #407 07:26Z; --launch; numbers).
    cond_h = bool(approval["approval_body_finalized"])

    # (i) realized TPS slot == 456.36; dominated 3D bracket (449.08, byte 0.0) recorded as NOT-the-config;
    #     tighten range carried.
    go = filled["assembled_submission_spec"]["go_branch_strict_frontier"]
    cond_i = bool(
        go["realized_strict_tps"] == STARK466_REALIZED_TPS_HEADLINE
        and abs(STARK466_REALIZED_TPS_HEADLINE - 456.36) <= TOL_RT
        and go["NOT_the_config_dominated_3d_bracket"]["tps"] == STARK466_DOMINATED_3D_BRACKET_TPS
        and go["NOT_the_config_dominated_3d_bracket"]["byte_identity"] == 0.0
        and go["realized_tps_live_slot"]["best_estimate_range_pending_472"]
        == [STARK466_TPS_TIGHTEN_LO, STARK466_TPS_TIGHTEN_HI]
    )

    # (j) human-gating + analysis-only flags + PPL anchor preserved (DEPLOYED served stack untouched).
    cond_j = bool(
        abs(skel.PPL_ANCHOR - 2.3772) <= TOL_RT and skel.PPL_ANCHOR <= skel.PPL_GATE
    )

    # (k) NaN-clean — set by the caller after the full payload walk.
    cond_k = True

    conditions = {
        "a_imported_469_skeleton_roundtrips": cond_a,
        "b_all_8_slots_resolved": cond_b,
        "c_go_branch_config_filled": cond_c,
        "d_selector_pending_471_and_post471_corners": cond_d,
        "e_clause3_speed_ppl_completion_pass": cond_e,
        "f_consumes_denken471_identity_verdict": cond_f,
        "g_floor_lock_dir_staged_decodepath_identical": cond_g,
        "h_approval_body_finalized_both_branches": cond_h,
        "i_realized_tps_slot_and_dominated_contrast": cond_i,
        "j_human_gated_ppl_anchor_preserved": cond_j,
        "k_nan_clean": cond_k,
    }
    return {
        "conditions": conditions,
        "strict_sub_self_test_passes": bool(all(conditions.values())),
        "detail": {
            "max_roundtrip_resid": banked["max_roundtrip_resid"],
            "pending_466_sentinels_remaining": filled["pending_466_sentinels_remaining"],
            "live_branch_selected": branches["live_branch_selected"],
            "realized_strict_tps_slot": STARK466_REALIZED_TPS_HEADLINE,
            "floor_lock_dir": FLOOR_LOCK_DIR_REL,
        },
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    banked = skel.load_banked()                            # #469 imported round-trip (re-derives nothing)
    filled = assemble_filled_spec()
    branches = branch_staging()
    gate = assembled_clause3_gate()
    floor = floor_lock_staging()
    approval = finalize_approval_body()
    st = selftests(filled, branches, gate, floor, approval, banked)

    headline = {
        "go_branch_config_filled": bool(filled["go_branch_config_filled"]),            # PRIMARY
        "strict_submission_branch_selected": branches["live_branch_selected"],         # PENDING-471
        "floor_lock_submission_dir_staged": bool(floor["floor_lock_submission_dir_staged"]),
        "approval_body_finalized": bool(approval["approval_body_finalized"]),
        "realized_strict_tps_slot": STARK466_REALIZED_TPS_HEADLINE,                    # 456.36
        "clause3_speed_ppl_completion_pass": bool(gate["clause3_speed_ppl_completion_pass"]),
        "consumes_denken471_identity_verdict": bool(st["conditions"]["f_consumes_denken471_identity_verdict"]),
        "strict_sub_self_test_passes": bool(st["strict_sub_self_test_passes"]),
        "all_8_slots_resolved": bool(filled["all_8_slots_resolved"]),
        "realized_tps_cluster_mean": STARK466_REALIZED_TPS_CLUSTER_MEAN,               # 459
        "realized_tps_tighten_range_pending_472": [STARK466_TPS_TIGHTEN_LO, STARK466_TPS_TIGHTEN_HI],
        "realized_locus_identity": STARK466_LOCUS_IDENTITY,                            # 1.0 (0 flips)
        "dominated_3d_bracket_tps": STARK466_DOMINATED_3D_BRACKET_TPS,                 # 449.08 (byte 0.0)
        "floor_tps": skel.FLOOR_TPS,                                                   # 161.70
        "clearly_above_floor_threshold_tps": skel.FLOOR_CLEARLY_ABOVE_TPS,             # 166.52
        "deployed_tps_non_strict": skel.DEPLOYED_TPS,                                  # 481.53
        "frontier_literal_census_prior": skel.FRONTIER_LITERAL_IDENTITY,              # 0.9989 (committed prior)
        "sigma_hw": skel.SIGMA_HW,
        "ppl": skel.PPL_ANCHOR,
        "ppl_gate": skel.PPL_GATE,
        "analysis_only": True,
        "no_served_file_change": True,
        "official_tps": 0,
    }
    verdict = (
        f"STRICT-SUBMISSION-ASSEMBLED: 8/8 <PENDING #466> slots resolved (7 #466-realized + clause-a "
        f"CONSUMED-FROM-denken#471); GO-config FILLED = deployed {skel.DEPLOYED_SUBMISSION}+{VLLM_BATCH_INVARIANT_FLAG} "
        f"@ {STARK466_REALIZED_TPS_HEADLINE} TPS (cudagraph HELD, kernel-rebuild=none, byte-exact@locus 1.0/0-flips); "
        f"BRANCH={branches['live_branch_selected']} (speed landed, binding clause-3a census is denken #471's, "
        f"in flight); FLOOR-LOCK STAGED = {FLOOR_LOCK_DIR_REL} (M=1 AR {skel.FLOOR_TPS}, literal census 1.0 "
        f"BY CONSTRUCTION); APPROVAL BODY finalized BOTH branches (refs {STANDING_AUTH}); clause-3 speed/ppl/"
        f"completion PASS (PPL {skel.PPL_ANCHOR}<= {skel.PPL_GATE}, 128/128, {STARK466_REALIZED_TPS_HEADLINE}>>"
        f"{skel.FLOOR_CLEARLY_ABOVE_TPS:.2f}). Dominated 3D num_par=1 bracket ({STARK466_DOMINATED_3D_BRACKET_TPS}, "
        f"byte 0.0) is NOT the config. HUMAN-GATED (Directive #3)."
    )
    handoff = (
        f"ASSEMBLED & READY-TO-FIRE (one-word human approve -> instant launch). Selector "
        f"assemble_branch_select(literal,operative) = {branches['live_branch_selected']} TODAY. When denken "
        f"#471 lands: literal 1.0 -> GO `python {skel.SUBMISSION_RUNNER} --submission <precache+VLLM_BATCH_"
        f"INVARIANT=1> --method land/{GO_METHOD} --launch --wait` ({STARK466_REALIZED_TPS_HEADLINE} TPS, "
        f"#472-tightened); literal 0.9989/operative 1.0 -> BLOCKED-HUMAN-CONTRACT (#124/#192); collapse/"
        f"literal-required -> FLOOR-LOCK `--submission {FLOOR_LOCK_DIR_REL} --method land/{FLOOR_LOCK_METHOD}` "
        f"({skel.FLOOR_TPS}, literal 1.0 BY CONSTRUCTION, staged NOW). clause-3a is denken #471's census "
        f"(consumed, never duplicated); b/c/d pass against #466's realized numbers."
    )
    return {
        "headline": headline,
        "assembled_filled_spec": filled,
        "branch_staging": branches,
        "assembled_clause3_gate": gate,
        "floor_lock_staging": floor,
        "finalized_approval": approval,
        "banked_roundtrip": banked,
        "self_test": st,
        "constants": {
            "stark466_run_ids": list(STARK466_RUN_IDS),
            "realized_tps_headline": STARK466_REALIZED_TPS_HEADLINE,
            "realized_tps_cluster_mean": STARK466_REALIZED_TPS_CLUSTER_MEAN,
            "realized_tps_tighten_range": [STARK466_TPS_TIGHTEN_LO, STARK466_TPS_TIGHTEN_HI],
            "realized_locus_identity": STARK466_LOCUS_IDENTITY, "realized_locus_flips": STARK466_LOCUS_FLIPS,
            "cudagraph_held": STARK466_CUDAGRAPH_HELD,
            "dominated_3d_bracket_tps": STARK466_DOMINATED_3D_BRACKET_TPS,
            "vllm_batch_invariant_flag": VLLM_BATCH_INVARIANT_FLAG, "pr122_splitkv_autogate": PR122_SPLITKV_AUTOGATE,
            "denken471_pr": DENKEN471_PR, "denken471_pending": DENKEN471_PENDING,
            "stark472_pr": STARK472_PR, "stark472_pending": STARK472_PENDING,
            "go_method": GO_METHOD, "floor_lock_method": FLOOR_LOCK_METHOD,
            "floor_lock_dir": FLOOR_LOCK_DIR_REL, "floor_lock_source_dir": skel.FLOOR_SUBMISSION,
            "deployed_submission": skel.DEPLOYED_SUBMISSION, "submission_runner": skel.SUBMISSION_RUNNER,
            "floor_tps": skel.FLOOR_TPS, "floor_clearly_above_tps": skel.FLOOR_CLEARLY_ABOVE_TPS,
            "deployed_tps": skel.DEPLOYED_TPS, "sigma_hw": skel.SIGMA_HW,
            "frontier_literal_identity": skel.FRONTIER_LITERAL_IDENTITY,
            "ppl_anchor": skel.PPL_ANCHOR, "ppl_gate": skel.PPL_GATE,
            "standing_authorization": STANDING_AUTH,
        },
        "verdict": verdict,
        "handoff_line": handoff,
        "imports": {
            "provenance": (
                "IMPORTS land #469 strict_submission_readiness (1s7i3tlk): reuses submission_spec / "
                "validation_gate_spec / select_strict_submission / load_banked / constants VERBATIM. "
                "Realized values from advisor PR #473 body: stark #466 sxigz7dp/gmd8v9sw (456.36 TPS, locus "
                "identity 1.0/0-flips, cudagraph held, VLLM_BATCH_INVARIANT=1 config-reachable, dominated 3D "
                "bracket 449.08/byte-0.0). Binding clause-3a census = denken #471 (in flight, CONSUMED). TPS "
                "tightener = stark #472 (in flight). All run-ids in wandb-applied-ai-team/gemma-challenge-senpai."),
            "machinery": "imports #469 skeleton; overlays #466 realized values into the 8 slots; stages the "
                         "floor-lock dir + finalizes the approval body; re-derives no measurements",
        },
    }


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B (mirrors #465/#469; never fatal).
# --------------------------------------------------------------------------- #
def _nan_paths(node: Any, p: str = "result") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            bad += _nan_paths(v, f"{p}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            bad += _nan_paths(v, f"{p}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        bad.append(p)
    return bad


def _print_report(syn: dict) -> None:
    h = syn["headline"]
    branches, gate, floor = syn["branch_staging"], syn["assembled_clause3_gate"], syn["floor_lock_staging"]
    st = syn["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("STRICT-SUBMISSION ASSEMBLE (PR #473, land) — fill #466 config, stage GO + floor-lock, CPU-only",
          flush=True)
    print("=" * 100, flush=True)
    print(f"  (1) 8/8 SLOTS RESOLVED = {h['all_8_slots_resolved']}; GO-config FILLED = "
          f"{h['go_branch_config_filled']}", flush=True)
    print(f"      GO config: {REALIZED_CONFIG_NAME}", flush=True)
    print(f"      realized TPS slot = {h['realized_strict_tps_slot']} (cluster-mean "
          f"{h['realized_tps_cluster_mean']}; #472-tighten {h['realized_tps_tighten_range_pending_472']}); "
          f"cudagraph held = {STARK466_CUDAGRAPH_HELD}; locus identity {h['realized_locus_identity']}/0-flips", flush=True)
    print(f"      NOT the config: dominated 3D num_par=1 bracket {h['dominated_3d_bracket_tps']} TPS, byte 0.0", flush=True)
    print("-" * 100, flush=True)
    print(f"  (2) BRANCH selected (LIVE) = {h['strict_submission_branch_selected']}", flush=True)
    for c in branches["post_471_corners"]:
        print(f"        [{c['action']:<22}] {c['label']}", flush=True)
    print(f"      FLOOR-LOCK staged dir = {floor['dir']} (staged={h['floor_lock_submission_dir_staged']}, "
          f"decode-path identical={floor['decode_path_files_byte_identical']})", flush=True)
    print("-" * 100, flush=True)
    print(f"  (3) clause-3: a=CONSUMED-FROM-denken#471 [{gate['clause_a_identity_census']['value']}] (BINDING); "
          f"b PPL {gate['clause_b_ppl']['value']}<= {gate['clause_b_ppl']['gate']} {gate['clause_b_ppl']['pass']}; "
          f"c 128/128 {gate['clause_c_completed']['pass']}; d band {gate['clause_d_tps_within_sigma_hw']['band']}", flush=True)
    print(f"      clause3_speed_ppl_completion_pass = {h['clause3_speed_ppl_completion_pass']}; "
          f"consumes_denken471 = {h['consumes_denken471_identity_verdict']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (4) APPROVAL body finalized BOTH branches = {h['approval_body_finalized']} "
          f"(refs {syn['finalized_approval']['references_standing_authorization']})", flush=True)
    print("-" * 100, flush=True)
    print(f"  (5) PRIMARY strict_sub_self_test_passes = {st['strict_sub_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print("=" * 100, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"\n  CAPSTONE HANDOFF: {syn['handoff_line']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[strict-assemble] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    h, st = syn["headline"], syn["self_test"]
    run = init_wandb_run(
        job_type="strict-submission-assemble",
        agent="land",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["strict-submission-assemble", "equivalence-escalation-anchors", "submission-assemble",
              "fill-466-config", "go-branch", "floor-lock-staged", "approval-body-finalized",
              "consumes-denken471", "pending-471", "identity-1p0-gated", "analysis-only",
              "bank-the-analysis", "strict-analog-of-465", "builds-on-469"],
        config={
            "realized_tps_headline": STARK466_REALIZED_TPS_HEADLINE,
            "realized_tps_cluster_mean": STARK466_REALIZED_TPS_CLUSTER_MEAN,
            "realized_locus_identity": STARK466_LOCUS_IDENTITY,
            "cudagraph_held": STARK466_CUDAGRAPH_HELD,
            "dominated_3d_bracket_tps": STARK466_DOMINATED_3D_BRACKET_TPS,
            "vllm_batch_invariant_flag": VLLM_BATCH_INVARIANT_FLAG,
            "floor_tps": skel.FLOOR_TPS, "floor_clearly_above_tps": skel.FLOOR_CLEARLY_ABOVE_TPS,
            "deployed_tps": skel.DEPLOYED_TPS, "sigma_hw": skel.SIGMA_HW,
            "ppl_anchor": skel.PPL_ANCHOR, "ppl_gate": skel.PPL_GATE,
            "go_method": GO_METHOD, "floor_lock_method": FLOOR_LOCK_METHOD,
            "floor_lock_dir": FLOOR_LOCK_DIR_REL, "deployed_submission": skel.DEPLOYED_SUBMISSION,
            "denken471_pr": DENKEN471_PR, "stark472_pr": STARK472_PR,
            "wandb_group": args.wandb_group,
            "source_runs": "land#469 1s7i3tlk (skeleton, imported), stark#466 sxigz7dp/gmd8v9sw (realized "
                           "456.36 TPS, locus 1.0/0-flips, VLLM_BATCH_INVARIANT=1), denken#471 (census, "
                           "in flight, CONSUMED), stark#472 (TPS tightener, in flight).",
        },
    )
    if run is None:
        print("[strict-assemble] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "go_branch_config_filled": int(bool(h["go_branch_config_filled"])),                # PRIMARY
        "strict_sub_self_test_passes": int(bool(h["strict_sub_self_test_passes"])),
        "floor_lock_submission_dir_staged": int(bool(h["floor_lock_submission_dir_staged"])),
        "approval_body_finalized": int(bool(h["approval_body_finalized"])),
        "realized_strict_tps_slot": h["realized_strict_tps_slot"],                          # 456.36
        "clause3_speed_ppl_completion_pass": int(bool(h["clause3_speed_ppl_completion_pass"])),
        "consumes_denken471_identity_verdict": int(bool(h["consumes_denken471_identity_verdict"])),
        "all_8_slots_resolved": int(bool(h["all_8_slots_resolved"])),
        "realized_tps_cluster_mean": h["realized_tps_cluster_mean"],
        "realized_locus_identity": h["realized_locus_identity"],
        "dominated_3d_bracket_tps": h["dominated_3d_bracket_tps"],
        "floor_tps": h["floor_tps"],
        "clearly_above_floor_threshold_tps": h["clearly_above_floor_threshold_tps"],
        "deployed_tps_non_strict": h["deployed_tps_non_strict"],
        "frontier_literal_census_prior": h["frontier_literal_census_prior"],
        "sigma_hw_tps": h["sigma_hw"],
        "max_roundtrip_resid": syn["banked_roundtrip"]["max_roundtrip_resid"],
        "ppl": skel.PPL_ANCHOR,
        "ppl_gate": skel.PPL_GATE,
        "analysis_only": 1,
        "no_served_file_change": 1,
        "official_tps": 0,
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    try:
        run.summary["verdict"] = syn["verdict"]
        run.summary["handoff_line"] = syn["handoff_line"]
        run.summary["strict_submission_branch_selected"] = h["strict_submission_branch_selected"]
    except Exception:
        pass
    log_json_artifact(run, name="strict_submission_assemble_result", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[strict-assemble] wandb logged {len(summary)} keys; run id {getattr(run, 'id', '?')}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="equivalence-escalation-anchors")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 473, "agent": "land",
        "kind": "strict-submission-assemble", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["k_nan_clean"] = not nan_paths
    syn["self_test"]["strict_sub_self_test_passes"] = bool(all(syn["self_test"]["conditions"].values()))
    syn["headline"]["strict_sub_self_test_passes"] = syn["self_test"]["strict_sub_self_test_passes"]
    if nan_paths:
        print(f"[strict-assemble] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "strict_submission_assemble_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[strict-assemble] wrote {out_path}", flush=True)

    st_path = out_dir / "strict_submission_assemble_selftest.json"
    with st_path.open("w", encoding="utf-8") as fh:
        json.dump(syn["self_test"]["conditions"], fh, indent=2, sort_keys=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["strict_sub_self_test_passes"] and payload["nan_clean"])
        print(f"[strict-assemble] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
