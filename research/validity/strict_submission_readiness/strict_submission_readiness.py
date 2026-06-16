#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Strict-submission readiness skeleton (PR #469, land) — CPU-only spec-authoring / pre-stage.

THE QUESTION (turn the human "submit anything strict clearly above 161 asap" GO into a zero-latency,
fully-specified, identity-gated leaderboard submission)
--------------------------------------------------------------------------------------------------
The human authorized a strict leaderboard submission (#407, 2026-06-16 07:26Z): "if you see clear line
of sight to submissions above the 161 tps for this strict mode then please make a HF submission for that
asap and tell the team on the board" + (07:30Z) "rally the other collaborators to adhere to the strict
decoding definition." Right now there is NO pre-staged strict submission. The instant stark #466 lands a
strict TPS clearly above the 161.70 floor, we want to fire with ZERO latency — not scramble to assemble
the submission-runner command, the served config, the identity-validation gate, the board announcement,
and the approval issue under time pressure.

This is the STRICT analog of land #465's relax-execution skeleton (now CLOSED — the relax config returned
a bar-invariant ROLLBACK). Same execution-readiness discipline, live target:

  #466 says WHAT the strict number IS (the e2e realization).  THIS card says HOW to SUBMIT it HONESTLY
  (the identity-gated submission skeleton) the instant it lands.

It is a SPEC / SKELETON ONLY — NO HF job, NO `train.py --launch`, NO submission, NO served-file change.
The submission itself stays HUMAN-GATED (operator Directive #3 + the `Approval request:` issue). stark
#466's realized config drops into the empty <PENDING #466> slots.

STATUS (2026-06-16): stark #466 (strict-frontier-realize) is IN FLIGHT on the pod, not yet reported
(advisor #407 07:41Z: "hours away ... I've pre-staged the submission so we fire the instant it lands ...
land #469 is pre-staging the submission now"). So `strict_submission_branch_selected = PENDING-466`.
The parameterized PENDING form is the DELIVERABLE: ready either way.

THE HONEST DECISION TREE (the PR's two SPEED branches, sharpened by an orthogonal IDENTITY gate that
committed data already pins) — THREE outcomes, all pre-specced so we are ready for whichever lands:
  (A) #466 holds strict TPS clearly > 161.70 (e.g. ~460) AND literal census == 1.0  -> SUBMIT-FRONTIER.
      The dream outcome: a +300-over-floor honest strict win. Fill the slots, fire.
  (B) #466 holds ~460 BUT literal census == 0.9989 (operative 1.0) -> BLOCKED-HUMAN-CONTRACT. This is
      what committed data PREDICTS: the blanket-strict frontier (the config #466 realizes) measures
      LITERAL identity 0.9989 (1 flip @ prompt 90, a bitwise-tie fixed point), operative 1.0 (land #429
      4u/gghmgtk9-class census; lawine #455 0r0ounl8 re-anchor pinned arm). Whether "operative 1.0"
      satisfies the byte-exact strict contract is a HUMAN CONTRACT DECISION (open issues #124/#192). Under
      a LITERAL reading of clause-3(a), that config is NOT census-1.0 and must NOT be submitted as strict.
  (C) #466 collapses toward ~162 (forcing num_splits=1/sequential-KV kills cudagraph/ONEGRAPH -> serial,
      exactly as the relax-prize collapsed e2e to 466.20) -> SUBMIT-FLOOR-LOCK. The best HONEST strict
      submission is M=1 AR (lawine #438, 161.70, literal census 1.0 BY CONSTRUCTION: int4 M=1 AR == plain
      greedy AR of the submitted int4 checkpoint). submissions/fa2sw_nonspec_int4 (SPECULATIVE_CONFIG="").

So the SPEED gate (clause-d, TPS within sigma_hw of #466) might pass at ~460, but the binding STRICT gate
is clause-3(a) (literal census == 1.0), and committed data already shows the frontier sits at 0.9989
literal. The ONLY config PROVABLY at literal census 1.0 today is M=1 AR (161.70). This mirrors #465's
finding (PPL alone would have passed the relax config; TPS+KIND caught it) — here, SPEED alone may pass
the frontier; the IDENTITY census is what gates.

SIX PRODUCTS (this card produces):
  (1) THE EXACT SUBMISSION SPEC, parameterized on #466 (the submission-runner command + the served config
      + every realized value as a <PENDING #466> slot). NO launch.
  (2) THE HONEST 3-OUTCOME DECISION TREE on #466 (both PR speed branches + the identity-census branch).
  (3) THE PRE-SUBMISSION VALIDATION GATE — 4 clauses, each mapped to the metric that CERTIFIES it; (a) is
      the binding strict gate (literal census == 1.0). Identity-1.0-GATED: no census 1.0 -> no submission.
  (4) THE STRICT-LABELED LEADERBOARD NAME + the team/board announcement (Directive B — rally strict).
  (5) THE PRE-FILLED `Approval request: HF job for <name>` ISSUE BODY (Directive #3 audit trail).
  (6) SELF-TEST + PPL anchor 2.3772.

NON-DUPLICATION: round-trips the committed lawine #455 / land #429 / wirbel #378 result JSONs; re-derives
no measurements. The realized strict TPS / census / completion / PPL stay #466 PARAMETERIZED SLOTS.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM / HF Job / submission / served-file change / official draw.
BASELINE stays 481.53 (non-strict); this leg adds 0 TPS; greedy/PPL untouched (PPL anchor 2.3772).

PRIMARY metric  strict_submission_pending_slots  (count of <PENDING #466> slots in spec + gate)
TEST    metric  ppl  (2.3772 anchor; this leg does not touch the served model)
HEADLINE        strict_submission_is_identity_1p0_gated, submission_name_is_strict_labeled
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# --- committed source JSONs round-tripped here (re-derives nothing) ----------------------------- #
REANCHOR_JSON = (
    REPO_ROOT / "research/validity/strict_frontier_reanchor/strict_frontier_reanchor_results.json"
)  # lawine #455 0r0ounl8 — re-anchor: strict frontier 466.02, M=1 AR floor 161.70, frontier literal 0.9989
OPERATIVE_JSON = (
    REPO_ROOT
    / "research/validity/blanket_strict_operative_identity/blanket_strict_operative_identity_results.json"
)  # land #429 — blanket-strict literal 0.9989 (1 flip @ 90 bitwise-tie) vs operative 1.0; human-contract
DEPLOYABLE_JSON = (
    REPO_ROOT / "research/validity/deployable_strict_served_tps/deployable_strict_served_tps_results.json"
)  # wirbel #378 gghmgtk9 — strict byte-exact confirmed; strict_floor_196 165.44; deployed off-strict 481.53

# --- anchors (single source of truth; round-tripped against the committed JSONs in load_banked) -- #
DEPLOYED_TPS = 481.53                  # PR #52 2x9fm2zx — NON-equivalent (identity 0.9966, 3 flips), NOT strict
DEPLOYED_IDENTITY = 0.9965986394557823  # deployed served literal identity (3 flips {11,18,118})
DEPLOYED_FLIPS = 3
SIGMA_HW = 4.8153                      # canonical empirical hw-noise envelope (#455 rounds to 4.8)
PPL_ANCHOR = 2.3772                    # PR #52 served PPL (this leg does not move it)
PPL_GATE = 2.42                        # reference + 5% (program.md validity cap)

STRICT_FRONTIER_ANCHOR_TPS = 467.1400155438763    # denken #423/#412 COMPOSED frontier (attention-tax isolation)
STRICT_FRONTIER_REANCHOR_TPS = 466.0177160736458  # lawine #455 independent re-anchor (composed, not e2e)
STRICT_FRONTIER_REANCHOR_SIGMA = 0.21862139078766127
EQUIVALENCE_TAX_TPS = 15.5122839263542            # lawine #455 — the byte-exact attention tax (3.2x sigma_hw)

FLOOR_TPS = 161.70                     # M=1 AR strict floor (lawine #438 official; #455 m1_ar 161.6996)
FLOOR_CLEARLY_ABOVE_TPS = FLOOR_TPS + SIGMA_HW    # "clearly above 161" (#407) = floor + hw-noise envelope (166.52)
FLOOR_OFFICIAL_TPS_455 = 161.6995796731182        # the committed re-anchor value (rounds to 161.70)
FLOOR_LOCAL_TPS_455 = 156.1959145793974           # #455 local A10G corroboration
FLOOR_IDENTITY = 1.0                   # int4 M=1 AR == plain greedy AR of the int4 checkpoint, BY CONSTRUCTION
M196_STRICT_FLOOR_SERVED = 165.44      # wirbel #378 strict_floor_196 — nearby fully-stacked nonspec served-strict

FRONTIER_LITERAL_IDENTITY = 0.9988662131519275    # land #429 / #455 pinned arm — 1 flip @ prompt 90
FRONTIER_OPERATIVE_IDENTITY = 1.0                 # land #429 — the lone flip is a bitwise-tie fixed point
FRONTIER_FLIPS = 1
FRONTIER_FLIP_PROMPT = 90
FRONTIER_FLIP_EMITTED_TOKEN = 102643
FRONTIER_FLIP_M1_REF_TOKEN = 22355

PENDING = "<PENDING #466>"             # the sentinel for every value stark #466 fills
PENDING_TPS = "<PENDING #466 strict TPS>"
TOL_RT = 1e-6

# served submission anchors (the config the spec parameterizes; NOT edited) ---------------------- #
DEPLOYED_SUBMISSION = "submissions/fa2sw_precache_kenyan"     # deployed NON-strict 481.53 (MTP K=7 spec)
FLOOR_SUBMISSION = "submissions/fa2sw_nonspec_int4"           # M=1 AR (SPECULATIVE_CONFIG="") — the floor-lock
SERVED_VLLM_WHEEL = "vllm-0.22.1rc1.dev307+g3e8afdf78 (manifest.dependencies[0], version-pinned)"

# the submission-runner (train.py) command that produces a leaderboard entry (documented flags) ---- #
SUBMISSION_RUNNER = "train.py"        # compatibility wrapper for the challenge benchmark/submission runner


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Load + round-trip the committed source JSONs (#455 / #429 / #378). Re-derives nothing.
# --------------------------------------------------------------------------- #
def load_banked() -> dict[str, Any]:
    ra = json.loads(REANCHOR_JSON.read_text(encoding="utf-8"))           # lawine #455
    op = json.loads(OPERATIVE_JSON.read_text(encoding="utf-8"))          # land #429
    dp = json.loads(DEPLOYABLE_JSON.read_text(encoding="utf-8"))         # wirbel #378

    rt = {
        # --- lawine #455 re-anchor (the frontier/floor/identity numbers this card cites) ---
        "ra_frontier_anchor_resid": abs(ra["frontier_anchor_412"] - STRICT_FRONTIER_ANCHOR_TPS),
        "ra_frontier_reanchor_resid": abs(ra["reanchored_strict_frontier_tps"] - STRICT_FRONTIER_REANCHOR_TPS),
        "ra_floor_official_resid": abs(ra["m1_ar_strict_equiv_official_tps_438"] - FLOOR_OFFICIAL_TPS_455),
        "ra_floor_local_resid": abs(ra["m1_ar_strict_equiv_local_tps_438"] - FLOOR_LOCAL_TPS_455),
        "ra_frontier_literal_id_resid": abs(ra["strict_identity_fraction"] - FRONTIER_LITERAL_IDENTITY),
        "ra_deployed_id_resid": abs(ra["deployed_identity_fraction"] - DEPLOYED_IDENTITY),
        "ra_tax_resid": abs(ra["equivalence_tax_tps"] - EQUIVALENCE_TAX_TPS),
        "ra_ppl_resid": abs(ra["ppl"] - PPL_ANCHOR),
        "ra_ppl_gate_resid": abs(ra["ppl_gate"] - PPL_GATE),
        # --- land #429 operative-identity (the literal-vs-operative census this card hinges on) ---
        "op_literal_id_resid": abs(op["blanket_strict_literal_identity"] - FRONTIER_LITERAL_IDENTITY),
        "op_operative_id_resid": abs(op["blanket_strict_operative_identity"] - FRONTIER_OPERATIVE_IDENTITY),
        "op_ppl_resid": abs(op["ppl"] - PPL_ANCHOR),
        # --- wirbel #378 deployable-strict (strict byte-exact confirmed; floor_196; deployed-off-strict) ---
        "dp_floor196_resid": abs(dp["ladder_constants"]["strict_floor_196"] - M196_STRICT_FLOOR_SERVED),
        "dp_deployed_resid": abs(dp["ladder_constants"]["official_tps"] - DEPLOYED_TPS),
    }
    max_resid = max(rt.values())
    return {
        "roundtrip_resid": rt,
        "max_roundtrip_resid": max_resid,
        "all_roundtrip_ok": bool(max_resid <= TOL_RT),
        # banked witnesses (the qualitative facts this spec stands on) -------------------------- #
        "frontier_is_composed_not_e2e_serve": bool(ra["frontier_is_composed_not_e2e_serve"]),
        "only_e2e_measurable_strict_config": str(ra["only_e2e_measurable_strict_equiv_config"]),
        "frontier_flips": int(ra["strict_token_flips"]),
        "frontier_flip_prompts": list(ra["strict_flip_prompts"]),
        "op_literal_green": bool(op["literal_green"]),          # False — literal census interlock would flag
        "op_operative_green": bool(op["operative_green"]),      # True — operative (fixed-point) census passes
        "op_go_conjunct_ii_resolution": str(op["go_conjunct_ii_resolution"]),  # "human_contract_decision"
        "op_prompt90_is_bitwise_tie": bool(op["prompt90_is_bitwise_tie"]),
        "op_n_changes_confident_argmax_forbidden": int(op["n_changes_confident_argmax_FORBIDDEN"]),  # 0
        "dp_is_strict_byte_exact": bool(dp["is_strict_byte_exact"]),
        "reanchor_run_id": str(ra.get("wandb_run_id", "0r0ounl8")),
        "deployable_run_id": str(dp.get("wandb_run_id", "gghmgtk9")),
    }


# --------------------------------------------------------------------------- #
# (1) The exact submission spec — parameterized on #466, NO launch.
# --------------------------------------------------------------------------- #
def submission_spec() -> dict[str, Any]:
    """The submission-runner command + the served config for BOTH branches. The GO-branch (strict
    frontier) carries every #466-realized value as a <PENDING #466> slot. The FLOOR-LOCK branch (M=1 AR)
    is FULLY specified (no slots) — it is already e2e-measured (lawine #438, 161.70, literal census 1.0
    by construction)."""
    return {
        "submission_runner_command": {
            "entrypoint": SUBMISSION_RUNNER,
            "note": "train.py is the challenge benchmark/submission-runner wrapper: it uploads the selected "
                    "submission to the senpai HF scratch bucket, launches the org-credit benchmark, polls, "
                    "and prints SENPAI-RESULT. --launch is the HUMAN-GATED action (Directive #3).",
            "go_branch_command": (
                f"python {SUBMISSION_RUNNER} --submission <GO-submission-dir> "
                f"--method \"land/{PENDING}\" --launch --wait"),
            "floor_lock_command": (
                f"python {SUBMISSION_RUNNER} --submission {FLOOR_SUBMISSION} "
                f"--method \"land/senpai-strict-m1ar-161\" --launch --wait"),
            "documented_flags": ["--submission", "--method", "--launch", "--wait", "--agent",
                                 "--name", "--run-prefix", "--interval-s", "--timeout-s"],
            "human_gated": "the --launch (and any --submission upload that feeds a leaderboard entry) is "
                           "HUMAN-GATED (Directive #3 + the Approval request: issue). This card SPECIFIES "
                           "the command; it does NOT run it. analysis_only=true, no_served_file_change=true.",
        },
        "go_branch_strict_frontier": {
            "role": "THE SUBMISSION if #466 holds strict TPS clearly > 161.70 AND literal census == 1.0",
            "derives_from": DEPLOYED_SUBMISSION,
            "served_delta_vs_deployed": (
                "force the ORDER-PRESERVING (byte-exact) attention reduction on top of the deployed MTP "
                "K=7 spec config: num_splits=1 / sequential-KV combine + use_fp32_reduce=True. This is what "
                "makes M=8 verify byte-exact to M=1 AR (kills the 3 free-running deployed flips) — at the "
                "cost of the equivalence tax (%.2f TPS, lawine #455) and the RISK that num_splits=1 kills "
                "the cudagraph/ONEGRAPH capture and collapses the serve toward the ~162 floor." % EQUIVALENCE_TAX_TPS),
            "order_preserving_attention_setting": {
                "num_splits": 1,
                "kv_combine": "sequential (no split-K cross-combine reassociation)",
                "use_fp32_reduce": True,
                "provenance": "PR #469 clause-1 spec; the byte-exact attention config denken #423 composed "
                              "and stark #466 is realizing e2e.",
            },
            "realized_config_name": PENDING,                  # SLOT 1: #466's confirmed config name
            "realized_strict_tps": PENDING,                   # SLOT 2: the e2e strict TPS #466 measures
            "realized_attention_holds_cudagraph": PENDING,    # SLOT 3: did order-preserving attn hold (or serial-collapse)?
            "realized_kernel_artifact_ref": PENDING,          # SLOT 4: built artifact ref if a kernel build was needed (else "none")
            "submission_packaging": (
                "if GO: package the strict-frontier env as a NEW submissions/<name>/ (manifest + serve.py + "
                "any kernel artifact), upload it, then the human-gated --launch. Every referenced wheel / "
                "kernel / config must be uploaded with the submission or Hub-hosted (no /workspace local paths)."),
            "literal_census_prior": (
                "WARNING (committed data): the blanket-strict frontier config measures LITERAL census %.4f "
                "(1 flip @ prompt %d, a bitwise-tie fixed point), operative 1.0 (land #429). Under clause-3(a) "
                "LITERAL reading this is NOT census-1.0 -> see decision-tree branch (B): BLOCKED-HUMAN-CONTRACT."
                % (FRONTIER_LITERAL_IDENTITY, FRONTIER_FLIP_PROMPT)),
        },
        "floor_lock_m1_ar": {
            "role": "THE HONEST STRICT SUBMISSION if #466 collapses toward ~162 OR the frontier's literal "
                    "census != 1.0 and the human requires literal byte-identity",
            "submission_dir": FLOOR_SUBMISSION,
            "served_config": "deployed int4 stack with SPECULATIVE_CONFIG=\"\" (MTP drafter disabled, "
                             "K_spec 7->0) -> every decode step is plain int4 M=1 AR.",
            "official_tps": FLOOR_TPS,                        # 161.70 (lawine #438) — FULLY measured, no slot
            "literal_census": FLOOR_IDENTITY,                 # 1.0 BY CONSTRUCTION (int4 M=1 AR == plain greedy AR)
            "identity_is_by_construction": True,
            "ppl": PPL_ANCHOR,
            "nearby_corroboration": ("wirbel #378 strict_floor_196 = %.2f served (the fully-stacked nonspec "
                                     "int4) — a nearby served-strict number; the PR's canonical floor is the "
                                     "lawine #438 M=1 AR official %.2f." % (M196_STRICT_FLOOR_SERVED, FLOOR_TPS)),
            "packaging_caveat": ("submissions/fa2sw_nonspec_int4 is currently LABELLED a lawine #196 "
                                 "THROWAWAY (local serve-profiling, 'NOT a leaderboard submission'). The "
                                 "floor-lock would RE-LABEL it as a strict leaderboard submission (manifest "
                                 "description only; the served config is unchanged and already #192-compliant). "
                                 "No code change to the decode path."),
            "fully_specified_no_pending_slots": True,
        },
        "summary": (
            f"GO-branch = deployed {DEPLOYED_SUBMISSION} + order-preserving attention (num_splits=1/seq-KV + "
            f"use_fp32_reduce=True), realized values are {PENDING} slots (committed prior: literal census "
            f"{FRONTIER_LITERAL_IDENTITY:.4f} not 1.0). FLOOR-LOCK = {FLOOR_SUBMISSION} (M=1 AR, {FLOOR_TPS} "
            f"official, literal census 1.0 BY CONSTRUCTION) — fully specified, zero slots. Submission-runner "
            f"= `python {SUBMISSION_RUNNER} --submission <dir> --method land/<name> --launch --wait` "
            f"(--launch HUMAN-GATED, Directive #3)."),
    }


# --------------------------------------------------------------------------- #
# (2) The honest 3-outcome decision tree on #466.
# --------------------------------------------------------------------------- #
def select_strict_submission(realized_tps: Any, literal_census: Any, operative_census: Any,
                             completed: Any, ppl: Any, human_accepts_operative: Any) -> str:
    """Map stark #466's realized measurement -> the SUBMISSION ACTION. The strict gate is a conjunction
    (clause-3): a config may be submitted AS STRICT only if literal census == 1.0 AND ppl <= gate AND
    completed == 128 AND tps clearly > floor. The literal-vs-operative census distinction (land #429) adds
    the human-contract branch.

      realized_tps is PENDING                         -> PENDING-466 (nothing landed yet; this is LIVE today)
      ppl > gate  OR  completed < 128                 -> NO-SUBMIT (invalid/quality fail; cannot happen for
                                                          a strict config but kept for completeness)
      tps NOT clearly above floor (<= floor+sigma_hw) -> SUBMIT-FLOOR-LOCK-M1AR (honest floor; literal 1.0).
                                                          "clearly above 161" (#407) means by MORE than the
                                                          hw-noise envelope: a config within sigma_hw of the
                                                          floor is statistically == M=1 AR, so the honest pick
                                                          is the provably-literal-1.0 floor-lock, NOT a fragile
                                                          frontier config that happened to collapse to ~162.
      tps clearly above floor AND literal census == 1.0 -> SUBMIT-FRONTIER-STRICT (the dream: ~460 honest win)
      tps clearly above floor AND literal < 1.0 AND operative 1.0:
          human_accepts_operative is True             -> SUBMIT-FRONTIER-STRICT (operative path, human-ruled)
          human_accepts_operative is False            -> SUBMIT-FLOOR-LOCK-M1AR (literal contract -> floor)
          human_accepts_operative is None/undecided   -> BLOCKED-HUMAN-CONTRACT (#124/#192 ruling pending)
      tps clearly above floor AND operative < 1.0     -> SUBMIT-FLOOR-LOCK-M1AR (not even operatively strict)
    """
    if not _finite(realized_tps):
        return "PENDING-466"
    if not (_finite(ppl) and ppl <= PPL_GATE) or not (_finite(completed) and completed >= 128):
        return "NO-SUBMIT"
    if realized_tps <= FLOOR_CLEARLY_ABOVE_TPS:
        return "SUBMIT-FLOOR-LOCK-M1AR"
    # realized_tps clearly above the floor (by more than sigma_hw) -> the IDENTITY census decides
    if _finite(literal_census) and literal_census >= 1.0:
        return "SUBMIT-FRONTIER-STRICT"
    if _finite(operative_census) and operative_census >= 1.0:
        if human_accepts_operative is True:
            return "SUBMIT-FRONTIER-STRICT"
        if human_accepts_operative is False:
            return "SUBMIT-FLOOR-LOCK-M1AR"
        return "BLOCKED-HUMAN-CONTRACT"
    return "SUBMIT-FLOOR-LOCK-M1AR"


def decision_tree() -> dict[str, Any]:
    # the committed-data prior for the frontier config (#429 / #455): literal 0.9989, operative 1.0.
    corners = [
        {"label": "#466 holds 460, literal census 1.0, ppl 2.3772, 128/128 (the dream win)",
         "tps": 460.0, "literal": 1.0, "operative": 1.0, "completed": 128, "ppl": PPL_ANCHOR,
         "human_op": None,
         "action": select_strict_submission(460.0, 1.0, 1.0, 128, PPL_ANCHOR, None)},
        {"label": "#466 holds 460, literal 0.9989, operative 1.0, human ACCEPTS operative (contract ruled)",
         "tps": 460.0, "literal": FRONTIER_LITERAL_IDENTITY, "operative": 1.0, "completed": 128,
         "ppl": PPL_ANCHOR, "human_op": True,
         "action": select_strict_submission(460.0, FRONTIER_LITERAL_IDENTITY, 1.0, 128, PPL_ANCHOR, True)},
        {"label": "#466 holds 460, literal 0.9989, operative 1.0, human REQUIRES literal byte-identity",
         "tps": 460.0, "literal": FRONTIER_LITERAL_IDENTITY, "operative": 1.0, "completed": 128,
         "ppl": PPL_ANCHOR, "human_op": False,
         "action": select_strict_submission(460.0, FRONTIER_LITERAL_IDENTITY, 1.0, 128, PPL_ANCHOR, False)},
        {"label": "#466 holds 460, literal 0.9989, operative 1.0, contract UNDECIDED (committed-data prior)",
         "tps": 460.0, "literal": FRONTIER_LITERAL_IDENTITY, "operative": 1.0, "completed": 128,
         "ppl": PPL_ANCHOR, "human_op": None,
         "action": select_strict_submission(460.0, FRONTIER_LITERAL_IDENTITY, 1.0, 128, PPL_ANCHOR, None)},
        {"label": ("#466 collapses to 162 (num_splits=1 kills cudagraph -> serial); within sigma_hw of the "
                   "161.70 floor = M=1 AR territory (NOT clearly above 161)"),
         "tps": 162.0, "literal": 1.0, "operative": 1.0, "completed": 128, "ppl": PPL_ANCHOR,
         "human_op": None,
         "action": select_strict_submission(162.0, 1.0, 1.0, 128, PPL_ANCHOR, None)},
        {"label": "LIVE TODAY: #466 in flight, nothing reported (advisor #407 07:41Z)",
         "tps": PENDING, "literal": PENDING, "operative": PENDING, "completed": PENDING, "ppl": PENDING,
         "human_op": None,
         "action": select_strict_submission(PENDING, PENDING, PENDING, PENDING, PENDING, None)},
    ]
    return {
        "selector_signature": ("select_strict_submission(realized_tps, literal_census, operative_census, "
                               "completed, ppl, human_accepts_operative)"),
        "action_space": ["SUBMIT-FRONTIER-STRICT", "BLOCKED-HUMAN-CONTRACT", "SUBMIT-FLOOR-LOCK-M1AR",
                         "NO-SUBMIT", "PENDING-466"],
        "branch_A_hold_and_literal_1p0": "SUBMIT-FRONTIER-STRICT (the dream: ~460 honest strict win, +300 over floor)",
        "branch_B_hold_but_operative_only": ("BLOCKED-HUMAN-CONTRACT -> SUBMIT-FRONTIER (if human accepts "
                                             "operative 1.0) or SUBMIT-FLOOR-LOCK (if literal required). This "
                                             "is what committed data PREDICTS: frontier literal 0.9989."),
        "branch_C_collapse_to_floor": ("SUBMIT-FLOOR-LOCK-M1AR (161.70, literal census 1.0 by construction). "
                                       "Triggers when realized TPS is NOT clearly above the floor, i.e. "
                                       "<= floor + sigma_hw (%.2f) — within the hw-noise envelope of M=1 AR."
                                       % FLOOR_CLEARLY_ABOVE_TPS),
        "clearly_above_floor_threshold_tps": FLOOR_CLEARLY_ABOVE_TPS,
        "worked_corners": corners,
        "live_action": select_strict_submission(PENDING, PENDING, PENDING, PENDING, PENDING, None),  # PENDING-466
        "committed_data_prior": (
            "frontier config literal census %.4f (land #429 / lawine #455 pinned arm; 1 flip @ prompt %d, "
            "emitted %d vs m1-ref %d, a measured bitwise tie), operative 1.0; n_changes_confident_argmax_"
            "FORBIDDEN=0. So even if #466 holds ~460 TPS, the LITERAL strict gate (clause-3a) is not met and "
            "the outcome routes to branch (B) — the human contract decision (#124/#192), not an automatic win."
            % (FRONTIER_LITERAL_IDENTITY, FRONTIER_FLIP_PROMPT, FRONTIER_FLIP_EMITTED_TOKEN,
               FRONTIER_FLIP_M1_REF_TOKEN)),
    }


# --------------------------------------------------------------------------- #
# (3) The pre-submission validation gate — 4 clauses; each -> the metric that CERTIFIES it.
# --------------------------------------------------------------------------- #
def validation_gate_spec() -> dict[str, Any]:
    clauses = [
        {
            "id": "a_served_greedy_identity_census_eq_1p0",
            "check": ("served greedy-identity census == 1.0: token_identity_rate EXACTLY 1.0 vs the M=1 AR "
                      "reference over the served prompt set (the strict contract #319; census per denken "
                      "#460 — NOT PPL, NOT a-priori). THE BINDING STRICT GATE."),
            "certifying_metric": "token_identity_rate (LITERAL, vs AR reference; #319 contract)",
            "is_binding_strict_gate": True,
            "finding_if_not_1p0": ("if census != 1.0 the config is NOT strict and must NOT be submitted as "
                                   "strict — THAT ITSELF IS A FINDING (PR clause-3). Committed data: the "
                                   "frontier config is literal %.4f, operative 1.0 (land #429) -> the "
                                   "literal gate FAILS; routes to decision-tree branch (B)." % FRONTIER_LITERAL_IDENTITY),
            "measured_slot": PENDING,                          # SLOT 5
        },
        {
            "id": "b_ppl_le_gate",
            "check": f"measured served PPL <= {PPL_GATE} (program.md validity cap = reference + 5%)",
            "certifying_metric": "ppl (anchor 2.3772; margin %.4f to gate)" % (PPL_GATE - PPL_ANCHOR),
            "is_binding_strict_gate": False,
            "measured_slot": PENDING,                          # SLOT 6
        },
        {
            "id": "c_completed_128_of_128",
            "check": "128/128 public prompts completed (benchmark validity precondition; program.md)",
            "certifying_metric": "completed (== 128)",
            "is_binding_strict_gate": False,
            "measured_slot": PENDING,                          # SLOT 7
        },
        {
            "id": "d_measured_tps_within_sigma_hw_of_466",
            "check": (f"the LOCAL pre-check measured TPS reproduces #466's reported strict TPS within "
                      f"sigma_hw ({SIGMA_HW}): |local_tps - reported_466_tps| <= {SIGMA_HW} "
                      f"(reproducibility / no-regression-from-the-claimed-number gate)"),
            "certifying_metric": "tps (local served pre-check vs #466 reported)",
            "is_binding_strict_gate": False,
            "measured_slot": PENDING,                          # SLOT 8
        },
    ]
    return {
        "clauses": clauses,
        "n_clauses": len(clauses),
        "gate_is_identity_1p0_gated": True,
        "all_pass_means": "SUBMIT-READY (the config may be proposed as a strict leaderboard submission)",
        "any_fail_means": ("NOT strict-submittable. If (a) fails -> the config is not strict (finding); "
                           "route to the floor-lock or the human-contract branch. (b)/(c)/(d) are validity/"
                           "reproducibility gates."),
        "measured_locally_on_pod": ("clauses (a)-(d) are measured LOCALLY on the pod A10G once #466 lands "
                                    "(the candidate config exists); the FLOOR-LOCK's (a)=1.0 is already "
                                    "certified BY CONSTRUCTION + e2e-measured (lawine #438) and round-tripped "
                                    "here. No served job is launched by THIS card."),
        "summary": (f"4-clause gate: (a) literal census == 1.0 [token_identity_rate] = THE binding strict "
                    f"gate; (b) PPL <= {PPL_GATE} [ppl]; (c) 128/128 [completed]; (d) local TPS within "
                    f"{SIGMA_HW} of #466 [tps]. No census 1.0 -> NO strict submission."),
    }


# --------------------------------------------------------------------------- #
# (4) Strict-labeled leaderboard name + the team/board announcement (rally strict).
# --------------------------------------------------------------------------- #
def submission_naming() -> dict[str, Any]:
    return {
        "go_branch_name_template": f"senpai-strict-eqv-{PENDING_TPS}",   # e.g. senpai-strict-eqv-460
        "floor_lock_name": "senpai-strict-m1ar-161",
        "naming_rule": ("the name must SIGNAL byte-exact greedy-strict decoding on the public board: the "
                        "`strict` token + `eqv` (equivalence) + the realized TPS. This is how the board "
                        "reader distinguishes an honest token_identity_rate=1.0 entry from the non-strict "
                        "precache/sliding-window leaders (489-508 TPS) that ride the Delta<=5% public/private "
                        "tolerance (openevolve 2026-06-16 06:27Z: those are 'mirages' that fail private verify)."),
        "submission_name_is_strict_labeled": True,
    }


def board_announcement() -> dict[str, Any]:
    body = (
        f"**senpai: an HONEST byte-exact greedy-STRICT leaderboard entry — `token_identity_rate=1.0` vs "
        f"plain M=1 AR, PPL {PPL_ANCHOR} <= {PPL_GATE}.**\n\n"
        f"This entry is byte-exact greedy-strict: every emitted token is token-identical to plain greedy "
        f"autoregressive decode of the submitted int4 checkpoint (`token_identity_rate=1.0`, the #319 "
        f"contract), at PPL {PPL_ANCHOR} (<= the {PPL_GATE} cap), 128/128 completed. It is a DIFFERENT, "
        f"honest category from the {DEPLOYED_TPS}-class precache/sliding-window leaders: those run at "
        f"identity ~0.9966 (non-equivalent) and, per openevolve's 06:27Z map, the public 500+ scores are "
        f"precache mirages that fail private verification (honest private decode ~470, the verified board "
        f"floor ~489.66 is itself the same trick kept under the Delta<=5% tolerance). Strict decoding is the "
        f"only definition under which this end model is eventually useful (human directive, #407 07:30Z) — "
        f"we'd like other collaborators to report a `token_identity_rate` census beside their TPS so the "
        f"board can compare honest strict numbers, not just Delta-gated public TPS.\n\n"
        f"Realized strict TPS: {PENDING_TPS} (stark #466 e2e). The only e2e-measured strict floor today is "
        f"M=1 AR at {FLOOR_TPS} TPS; the {STRICT_FRONTIER_ANCHOR_TPS:.2f} frontier everyone quotes is a "
        f"composed attention-tax isolation, now being realized e2e. We submit the honest strict number we "
        f"can actually stand behind."
    )
    return {
        "channel": "message_board (scripts/post_message.py bucket mode; authorship via scratch bucket)",
        "when": "AFTER the human-approved --launch returns a valid strict summary.json (NOT before)",
        "body": body,
        "rallies_strict": True,
        "cites_public_evidence": ["#407 07:26Z + 07:30Z human directives",
                                  "openevolve 2026-06-16 06:27Z 'int4 floor / dead levers' map"],
    }


# --------------------------------------------------------------------------- #
# (5) The pre-filled `Approval request:` issue body (Directive #3 audit trail).
# --------------------------------------------------------------------------- #
def approval_request_issue() -> dict[str, Any]:
    title = f"Approval request: HF job for senpai-strict-eqv-{PENDING_TPS}"
    title_floor = "Approval request: HF job for senpai-strict-m1ar-161"
    body = (
        f"## Approval request: HONEST STRICT leaderboard submission\n\n"
        f"**Standing authorization:** human #407 07:26Z — \"if you see clear line of sight to submissions "
        f"above the 161 tps for this strict mode then please make a HF submission for that asap.\" This is "
        f"that submission, gated on stark #466's e2e strict realization + the clause-3 local validity gate.\n\n"
        f"- **PR / branch:** land #469 / `land/strict-submission-readiness` (base `approval-gated-8gpu-20260613`).\n"
        f"- **Submission dir:** GO-branch `submissions/<strict-frontier-name>` (or FLOOR-LOCK "
        f"`{FLOOR_SUBMISSION}` if #466 collapses / literal census != 1.0).\n"
        f"- **Exact command:** `python {SUBMISSION_RUNNER} --submission <dir> --method "
        f"\"land/senpai-strict-eqv-{PENDING_TPS}\" --launch --wait`\n"
        f"- **Expected TPS:** {PENDING_TPS} (stark #466 e2e; floor-lock fallback {FLOOR_TPS}).\n"
        f"- **Expected PPL:** {PPL_ANCHOR} (<= {PPL_GATE} cap). **PPL risk:** none — strict decode is "
        f"PPL-neutral by construction (no quality change vs deployed).\n"
        f"- **Quota / runtime risk:** one HF Jobs `a10g-small` benchmark draw (~the standard 128-prompt "
        f"speed+PPL run); single launch, no retries (Directive: launch exactly once).\n"
        f"- **Local checks performed (clause-3, measured on the pod):**\n"
        f"    - (a) served greedy-identity census `token_identity_rate` == 1.0 vs M=1 AR: {PENDING} "
        f"(floor-lock = 1.0 BY CONSTRUCTION + lawine #438 e2e; frontier literal = "
        f"{FRONTIER_LITERAL_IDENTITY:.4f} per land #429 -> human-contract gate).\n"
        f"    - (b) PPL <= {PPL_GATE}: {PENDING} (anchor {PPL_ANCHOR}).\n"
        f"    - (c) 128/128 completed: {PENDING}.\n"
        f"    - (d) local TPS within sigma_hw ({SIGMA_HW}) of #466 reported: {PENDING}.\n"
        f"- **Artifact paths:** submission package `submissions/<name>/` (manifest.json + serve.py + any "
        f"kernel artifact, all uploaded/Hub-hosted — no /workspace local paths); validation JSON "
        f"`research/validity/strict_submission_readiness/strict_submission_readiness_results.json`; "
        f"W&B run (this pre-stage) + the #466 realization run.\n\n"
        f"**One-word approve and I fire immediately.** Per Directive #3 the submission stays human-gated "
        f"until you approve here."
    )
    return {
        "title": title,
        "title_floor_lock": title_floor,
        "body": body,
        "approval_issue_prefilled": True,
        "references_standing_authorization": "#407 07:26Z",
    }


# --------------------------------------------------------------------------- #
# Slot accounting — count the <PENDING #466> sentinel leaves (PRIMARY metric).
# --------------------------------------------------------------------------- #
def _pending_slot_paths(node: Any, p: str = "spec") -> list[str]:
    paths: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            paths += _pending_slot_paths(v, f"{p}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            paths += _pending_slot_paths(v, f"{p}[{i}]")
    elif isinstance(node, str) and node == PENDING:
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
# (6) Self-test (PRIMARY gate).
# --------------------------------------------------------------------------- #
def selftests(banked: dict, spec: dict, tree: dict, gate: dict, naming: dict,
              announce: dict, approval: dict, slot_paths: list[str]) -> dict[str, Any]:
    # (a) every banked source number round-trips its committed JSON within tol.
    cond_a = bool(banked["all_roundtrip_ok"])

    # (b) the banked qualitative facts this spec stands on: frontier is composed (not e2e), the only
    #     e2e-measurable strict config is M=1 AR, strict is byte-exact, and the operative-vs-literal split
    #     resolves to a HUMAN CONTRACT DECISION with literal_green False / operative_green True.
    cond_b = bool(
        banked["frontier_is_composed_not_e2e_serve"] is True
        and "M=1 AR" in banked["only_e2e_measurable_strict_config"]
        and banked["dp_is_strict_byte_exact"] is True
        and banked["op_literal_green"] is False
        and banked["op_operative_green"] is True
        and banked["op_go_conjunct_ii_resolution"] == "human_contract_decision"
        and banked["op_n_changes_confident_argmax_forbidden"] == 0
    )

    # (c) the FLOOR-LOCK branch is FULLY specified (no PENDING slots): M=1 AR, 161.70, literal census 1.0
    #     by construction, points at fa2sw_nonspec_int4.
    fl = spec["floor_lock_m1_ar"]
    cond_c = bool(
        fl["fully_specified_no_pending_slots"] is True
        and abs(fl["official_tps"] - FLOOR_TPS) <= TOL_RT
        and fl["literal_census"] == 1.0
        and fl["identity_is_by_construction"] is True
        and fl["submission_dir"] == FLOOR_SUBMISSION
        and len(_pending_slot_paths(fl, "fl")) == 0
    )

    # (d) the GO-branch carries the order-preserving attention SPEC (num_splits=1 / seq-KV / fp32-reduce)
    #     and exactly the enumerated realized <PENDING #466> slots.
    go = spec["go_branch_strict_frontier"]
    ops = go["order_preserving_attention_setting"]
    cond_d = bool(
        ops["num_splits"] == 1
        and ops["use_fp32_reduce"] is True
        and "sequential" in ops["kv_combine"]
        and go["derives_from"] == DEPLOYED_SUBMISSION
        and go["realized_config_name"] == PENDING
        and go["realized_strict_tps"] == PENDING
    )

    # (e) PRIMARY: exactly the enumerated <PENDING #466> slots (4 submission-spec GO + 4 gate clauses),
    #     count > 0 and == len(enumerated). The floor-lock + naming/announce/approval carry none.
    expected_slots = {
        "spec.go_branch_strict_frontier.realized_config_name",
        "spec.go_branch_strict_frontier.realized_strict_tps",
        "spec.go_branch_strict_frontier.realized_attention_holds_cudagraph",
        "spec.go_branch_strict_frontier.realized_kernel_artifact_ref",
        "gate.clauses[0].measured_slot",
        "gate.clauses[1].measured_slot",
        "gate.clauses[2].measured_slot",
        "gate.clauses[3].measured_slot",
    }
    got_slots = set(slot_paths)
    n_slots = len(slot_paths)
    cond_e = bool(n_slots == len(expected_slots) and got_slots == expected_slots and n_slots > 0)

    # (f) the gate is exactly 4 clauses, identity-1.0-gated, and clause-a is the binding strict gate mapped
    #     to token_identity_rate.
    cl = gate["clauses"]
    cond_f = bool(
        gate["n_clauses"] == 4
        and gate["gate_is_identity_1p0_gated"] is True
        and cl[0]["is_binding_strict_gate"] is True
        and "token_identity_rate" in cl[0]["certifying_metric"]
        and cl[1]["certifying_metric"].startswith("ppl")
        and cl[2]["certifying_metric"].startswith("completed")
        and cl[3]["certifying_metric"].startswith("tps")
    )

    # (g) the decision selector behaves correctly on the worked corners: dream -> SUBMIT-FRONTIER;
    #     operative+accept -> SUBMIT-FRONTIER; operative+require-literal -> FLOOR-LOCK; operative+undecided
    #     -> BLOCKED; collapse -> FLOOR-LOCK; PENDING -> PENDING-466.
    cond_g = bool(
        select_strict_submission(460.0, 1.0, 1.0, 128, PPL_ANCHOR, None) == "SUBMIT-FRONTIER-STRICT"
        and select_strict_submission(460.0, FRONTIER_LITERAL_IDENTITY, 1.0, 128, PPL_ANCHOR, True) == "SUBMIT-FRONTIER-STRICT"
        and select_strict_submission(460.0, FRONTIER_LITERAL_IDENTITY, 1.0, 128, PPL_ANCHOR, False) == "SUBMIT-FLOOR-LOCK-M1AR"
        and select_strict_submission(460.0, FRONTIER_LITERAL_IDENTITY, 1.0, 128, PPL_ANCHOR, None) == "BLOCKED-HUMAN-CONTRACT"
        and select_strict_submission(162.0, 1.0, 1.0, 128, PPL_ANCHOR, None) == "SUBMIT-FLOOR-LOCK-M1AR"
        and select_strict_submission(PENDING, PENDING, PENDING, PENDING, PENDING, None) == "PENDING-466"
        and select_strict_submission(460.0, 1.0, 1.0, 128, 2.50, None) == "NO-SUBMIT"
        and select_strict_submission(460.0, 1.0, 1.0, 120, PPL_ANCHOR, None) == "NO-SUBMIT"
    )

    # (h) LIVE state today: nothing landed -> the selector returns PENDING-466 (the honest reported state).
    cond_h = bool(tree["live_action"] == "PENDING-466")

    # (i) naming is strict-labeled (go template + floor name both carry the strict token).
    cond_i = bool(
        naming["submission_name_is_strict_labeled"] is True
        and "strict" in naming["go_branch_name_template"]
        and "strict" in naming["floor_lock_name"]
    )

    # (j) the board announcement rallies strict (token_identity_rate=1.0 claim) and the approval issue is
    #     pre-filled with the standing #407 07:26Z authorization + the exact submission-runner command.
    cond_j = bool(
        announce["rallies_strict"] is True
        and "token_identity_rate=1.0" in announce["body"]
        and approval["approval_issue_prefilled"] is True
        and "#407 07:26Z" == approval["references_standing_authorization"]
        and "--launch" in approval["body"]
        and "Approval request: HF job for" in approval["title"]
    )

    # (k) human-gating + analysis-only flags + PPL anchor preserved (no served-file change, no launch).
    cond_k = bool(
        "HUMAN-GATED" in spec["submission_runner_command"]["human_gated"]
        and "Directive #3" in spec["submission_runner_command"]["human_gated"]
        and abs(PPL_ANCHOR - 2.3772) <= TOL_RT and PPL_ANCHOR <= PPL_GATE
    )

    # (l) NaN-clean — set by the caller after the full payload walk.
    cond_l = True

    conditions = {
        "a_all_banked_numbers_roundtrip": cond_a,
        "b_banked_qualitative_facts_hold": cond_b,
        "c_floor_lock_fully_specified_no_slots": cond_c,
        "d_go_branch_carries_orderpreserving_spec_and_slots": cond_d,
        "e_pending_slots_exactly_enumerated": cond_e,
        "f_gate_4_clauses_identity_gated": cond_f,
        "g_selector_correct_on_corners": cond_g,
        "h_live_state_is_pending_466": cond_h,
        "i_naming_is_strict_labeled": cond_i,
        "j_announcement_and_approval_prefilled": cond_j,
        "k_human_gated_directive3_ppl_anchor": cond_k,
        "l_nan_clean": cond_l,
    }
    return {
        "conditions": conditions,
        "strict_sub_self_test_passes": bool(all(conditions.values())),
        "detail": {
            "max_roundtrip_resid": banked["max_roundtrip_resid"],
            "strict_submission_pending_slots": n_slots,
            "pending_slot_paths": sorted(slot_paths),
            "validation_gate_clauses": gate["n_clauses"],
            "live_action": tree["live_action"],
            "frontier_literal_census": FRONTIER_LITERAL_IDENTITY,
            "floor_tps": FLOOR_TPS,
        },
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    banked = load_banked()
    spec = submission_spec()
    tree = decision_tree()
    gate = validation_gate_spec()
    naming = submission_naming()
    announce = board_announcement()
    approval = approval_request_issue()

    # count <PENDING #466> slots over the spec + gate (the two structures that carry realized slots).
    slot_paths = _pending_slot_paths(spec, "spec") + _pending_slot_paths(gate, "gate")
    st = selftests(banked, spec, tree, gate, naming, announce, approval, slot_paths)

    n_slots = len(slot_paths)
    headline = {
        "strict_sub_self_test_passes": bool(st["strict_sub_self_test_passes"]),       # PRIMARY gate
        "strict_submission_pending_slots": n_slots,                                   # PRIMARY 8
        "strict_config_validation_clauses": gate["n_clauses"],                        # 4
        "strict_submission_is_identity_1p0_gated": bool(gate["gate_is_identity_1p0_gated"]),  # True
        "fallback_floor_tps": FLOOR_TPS,                                              # 161.70
        "clearly_above_floor_threshold_tps": FLOOR_CLEARLY_ABOVE_TPS,                 # 166.52 (floor + sigma_hw)
        "submission_name_is_strict_labeled": bool(naming["submission_name_is_strict_labeled"]),  # True
        "approval_issue_prefilled": bool(approval["approval_issue_prefilled"]),       # True
        "strict_submission_branch_selected": tree["live_action"],                     # "PENDING-466"
        "deployed_tps_non_strict": DEPLOYED_TPS,                                      # 481.53 (identity 0.9966)
        "strict_frontier_anchor_tps": STRICT_FRONTIER_ANCHOR_TPS,                     # 467.14 (composed)
        "strict_frontier_reanchor_tps": STRICT_FRONTIER_REANCHOR_TPS,                 # 466.02 (composed)
        "frontier_literal_census": FRONTIER_LITERAL_IDENTITY,                         # 0.9989 (not 1.0!)
        "frontier_operative_census": FRONTIER_OPERATIVE_IDENTITY,                     # 1.0 (fixed-point tie)
        "equivalence_tax_tps": EQUIVALENCE_TAX_TPS,                                   # 15.51 (3.2x sigma_hw)
        "sigma_hw": SIGMA_HW,                                                         # 4.8153
        "ppl": PPL_ANCHOR,
        "ppl_gate": PPL_GATE,
        "analysis_only": True,
        "no_served_file_change": True,
        "official_tps": 0,
    }
    verdict = (
        f"STRICT-SUBMISSION-READINESS-{n_slots}-PENDING466-SLOTS-IDENTITY-1.0-GATED-"
        f"{gate['n_clauses']}-CLAUSE-VALIDATION-GATE-HUMAN-GATED-D3-BRANCH-{tree['live_action']}"
        f" || FLOOR-LOCK=M1AR-{FLOOR_TPS}-LITERAL-CENSUS-1.0-BY-CONSTRUCTION (fa2sw_nonspec_int4);"
        f" FRONTIER-LITERAL-CENSUS-{FRONTIER_LITERAL_IDENTITY:.4f}-NOT-1.0-(operative-1.0,HUMAN-CONTRACT-#124/#192)"
    )
    handoff = (
        f"STRICT-SUBMISSION SKELETON (ready the instant stark #466 lands): submission-runner = "
        f"`python {SUBMISSION_RUNNER} --submission <dir> --method land/<name> --launch --wait` (--launch "
        f"HUMAN-GATED, Directive #3). {n_slots} <PENDING #466> slots. GATE (identity-1.0-gated): (a) literal "
        f"token_identity_rate == 1.0 [BINDING strict gate]; (b) PPL <= {PPL_GATE}; (c) 128/128; (d) local TPS "
        f"within {SIGMA_HW} of #466. DECISION TREE: (A) #466 holds ~460 + literal 1.0 -> SUBMIT-FRONTIER "
        f"(senpai-strict-eqv-<tps>); (B) holds ~460 but literal {FRONTIER_LITERAL_IDENTITY:.4f}/operative 1.0 "
        f"-> BLOCKED-HUMAN-CONTRACT (#124/#192); (C) collapses to ~162 -> SUBMIT-FLOOR-LOCK (senpai-strict-"
        f"m1ar-161, {FLOOR_SUBMISSION}, literal census 1.0 BY CONSTRUCTION). LIVE TODAY: #466 in flight "
        f"(advisor #407 07:41Z) -> branch = {tree['live_action']}. The binding gate is IDENTITY census, not "
        f"speed: committed data already shows the frontier is literal {FRONTIER_LITERAL_IDENTITY:.4f} (1 flip "
        f"@ prompt {FRONTIER_FLIP_PROMPT}, bitwise tie), so 'fast' alone does not make it strict-submittable."
    )
    return {
        "headline": headline,
        "submission_spec": spec,
        "decision_tree": tree,
        "validation_gate_spec": gate,
        "submission_naming": naming,
        "board_announcement": announce,
        "approval_request_issue": approval,
        "banked_roundtrip": banked,
        "self_test": st,
        "constants": {
            "deployed_tps": DEPLOYED_TPS, "deployed_identity": DEPLOYED_IDENTITY, "deployed_flips": DEPLOYED_FLIPS,
            "sigma_hw": SIGMA_HW, "ppl_anchor": PPL_ANCHOR, "ppl_gate": PPL_GATE,
            "strict_frontier_anchor_tps": STRICT_FRONTIER_ANCHOR_TPS,
            "strict_frontier_reanchor_tps": STRICT_FRONTIER_REANCHOR_TPS,
            "strict_frontier_reanchor_sigma": STRICT_FRONTIER_REANCHOR_SIGMA,
            "equivalence_tax_tps": EQUIVALENCE_TAX_TPS,
            "floor_tps": FLOOR_TPS, "floor_clearly_above_tps": FLOOR_CLEARLY_ABOVE_TPS,
            "floor_official_tps_455": FLOOR_OFFICIAL_TPS_455,
            "floor_local_tps_455": FLOOR_LOCAL_TPS_455, "floor_identity": FLOOR_IDENTITY,
            "m196_strict_floor_served": M196_STRICT_FLOOR_SERVED,
            "frontier_literal_identity": FRONTIER_LITERAL_IDENTITY,
            "frontier_operative_identity": FRONTIER_OPERATIVE_IDENTITY,
            "frontier_flips": FRONTIER_FLIPS, "frontier_flip_prompt": FRONTIER_FLIP_PROMPT,
            "deployed_submission": DEPLOYED_SUBMISSION, "floor_submission": FLOOR_SUBMISSION,
            "submission_runner": SUBMISSION_RUNNER,
        },
        "verdict": verdict,
        "handoff_line": handoff,
        "imports": {
            "provenance": (
                "lawine #455 0r0ounl8 (re-anchor: strict frontier 466.02 composed/not-e2e, M=1 AR floor "
                "161.70, frontier literal 0.9989, deployed 0.9966, equivalence tax 15.51) x land #429 "
                "(blanket-strict literal 0.9989 [1 flip @ prompt 90 bitwise tie] vs operative 1.0; "
                "go_conjunct_ii = human_contract_decision; literal_green False / operative_green True) x "
                "wirbel #378 gghmgtk9 (strict byte-exact confirmed; strict_floor_196 165.44; deployed "
                "off-strict 481.53). Realized #466 strict TPS/census/PPL/completion stay PENDING slots. "
                "All run-ids in wandb-applied-ai-team/gemma-challenge-senpai."),
            "machinery": "round-trips committed #455/#429/#378 JSONs; authors the submission/gate/naming/"
                         "announcement/approval-issue spec; re-derives no measurements",
        },
    }


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B (mirrors #465; never fatal).
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
    spec, tree, gate = syn["submission_spec"], syn["decision_tree"], syn["validation_gate_spec"]
    naming, st = syn["submission_naming"], syn["self_test"]
    h = syn["headline"]
    print("\n" + "=" * 100, flush=True)
    print("STRICT-SUBMISSION READINESS SKELETON (PR #469, land) — strict analog of #465, CPU-only",
          flush=True)
    print("=" * 100, flush=True)
    print("  (1) SUBMISSION SPEC (parameterized on #466; NO launch):", flush=True)
    print(f"      runner: python {SUBMISSION_RUNNER} --submission <dir> --method land/<name> --launch --wait "
          f"(--launch HUMAN-GATED)", flush=True)
    print(f"      GO-branch: deployed {DEPLOYED_SUBMISSION} + order-preserving attn (num_splits=1/seq-KV + "
          f"fp32-reduce); realized = {PENDING}", flush=True)
    print(f"      FLOOR-LOCK: {FLOOR_SUBMISSION} (M=1 AR, {FLOOR_TPS} TPS, literal census 1.0 BY "
          f"CONSTRUCTION) — fully specified", flush=True)
    print("-" * 100, flush=True)
    print("  (2) DECISION TREE on #466:", flush=True)
    for cn in tree["worked_corners"]:
        print(f"        [{cn['action']:<22}] {cn['label']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (3) VALIDATION GATE ({gate['n_clauses']} clauses; identity-1.0-gated="
          f"{gate['gate_is_identity_1p0_gated']}):", flush=True)
    for c in gate["clauses"]:
        binding = "  <== BINDING STRICT GATE" if c["is_binding_strict_gate"] else ""
        print(f"        [{c['id']:<38}] {c['certifying_metric'][:34]}{binding}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (4) NAME: go=`{naming['go_branch_name_template']}` floor=`{naming['floor_lock_name']}` "
          f"(strict-labeled={naming['submission_name_is_strict_labeled']})", flush=True)
    print(f"  (5) APPROVAL ISSUE pre-filled = {syn['approval_request_issue']['approval_issue_prefilled']} "
          f"(refs {syn['approval_request_issue']['references_standing_authorization']})", flush=True)
    print("-" * 100, flush=True)
    print(f"  (R) COMMITTED-DATA PRIOR: frontier literal census {FRONTIER_LITERAL_IDENTITY:.4f} (operative "
          f"1.0) -> even a fast #466 routes to branch (B) human-contract (#124/#192). Only M=1 AR is "
          f"literal-1.0.", flush=True)
    print(f"      LIVE branch_selected = {h['strict_submission_branch_selected']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (6) PRIMARY strict_sub_self_test_passes = {st['strict_sub_self_test_passes']}; "
          f"pending_slots = {h['strict_submission_pending_slots']}; identity_1p0_gated = "
          f"{h['strict_submission_is_identity_1p0_gated']}", flush=True)
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
        print(f"[strict-readiness] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    h, gate = syn["headline"], syn["validation_gate_spec"]
    st = syn["self_test"]
    run = init_wandb_run(
        job_type="strict-submission-readiness",
        agent="land",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["strict-submission-readiness", "equivalence-escalation-anchors", "submission-pre-stage",
              "submission-spec", "identity-1p0-gated", "decision-tree", "floor-lock-m1ar",
              "board-announcement", "approval-issue-prefilled", "rally-strict", "analysis-only",
              "bank-the-analysis", "pending-466", "strict-analog-of-465"],
        config={
            "deployed_tps": DEPLOYED_TPS, "sigma_hw": SIGMA_HW, "ppl_anchor": PPL_ANCHOR,
            "ppl_gate": PPL_GATE,
            "strict_frontier_anchor_tps": STRICT_FRONTIER_ANCHOR_TPS,
            "strict_frontier_reanchor_tps": STRICT_FRONTIER_REANCHOR_TPS,
            "floor_tps": FLOOR_TPS, "floor_clearly_above_tps": FLOOR_CLEARLY_ABOVE_TPS,
            "floor_submission": FLOOR_SUBMISSION,
            "deployed_submission": DEPLOYED_SUBMISSION,
            "frontier_literal_identity": FRONTIER_LITERAL_IDENTITY,
            "frontier_operative_identity": FRONTIER_OPERATIVE_IDENTITY,
            "submission_runner": SUBMISSION_RUNNER,
            "wandb_group": args.wandb_group,
            "source_runs": "lawine#455 0r0ounl8 (re-anchor: frontier 466.02 composed, floor 161.70, "
                           "literal 0.9989), land#429 (operative-identity: literal 0.9989 vs operative 1.0, "
                           "human_contract_decision), wirbel#378 gghmgtk9 (strict byte-exact, floor_196 "
                           "165.44, deployed-off-strict 481.53). stark#466 PENDING (in flight).",
        },
    )
    if run is None:
        print("[strict-readiness] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "strict_sub_self_test_passes": int(bool(st["strict_sub_self_test_passes"])),       # PRIMARY gate
        "strict_submission_pending_slots": h["strict_submission_pending_slots"],           # PRIMARY
        "strict_config_validation_clauses": h["strict_config_validation_clauses"],         # 4
        "strict_submission_is_identity_1p0_gated": int(bool(h["strict_submission_is_identity_1p0_gated"])),
        "fallback_floor_tps": h["fallback_floor_tps"],                                      # 161.70
        "clearly_above_floor_threshold_tps": h["clearly_above_floor_threshold_tps"],        # 166.52
        "submission_name_is_strict_labeled": int(bool(h["submission_name_is_strict_labeled"])),
        "approval_issue_prefilled": int(bool(h["approval_issue_prefilled"])),
        "strict_submission_branch_selected": h["strict_submission_branch_selected"],       # "PENDING-466"
        "deployed_tps_non_strict": DEPLOYED_TPS,
        "strict_frontier_anchor_tps": STRICT_FRONTIER_ANCHOR_TPS,
        "strict_frontier_reanchor_tps": STRICT_FRONTIER_REANCHOR_TPS,
        "frontier_literal_census": FRONTIER_LITERAL_IDENTITY,
        "frontier_operative_census": FRONTIER_OPERATIVE_IDENTITY,
        "equivalence_tax_tps": EQUIVALENCE_TAX_TPS,
        "sigma_hw_tps": SIGMA_HW,
        "max_roundtrip_resid": syn["banked_roundtrip"]["max_roundtrip_resid"],
        "ppl": PPL_ANCHOR,
        "ppl_gate": PPL_GATE,
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
    log_json_artifact(run, name="strict_submission_readiness_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[strict-readiness] wandb logged {len(summary)} keys; run id {getattr(run, 'id', '?')}",
          flush=True)


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
        "created_at": created_at, "pr": 469, "agent": "land",
        "kind": "strict-submission-readiness", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["l_nan_clean"] = not nan_paths
    syn["self_test"]["strict_sub_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    syn["headline"]["strict_sub_self_test_passes"] = syn["self_test"]["strict_sub_self_test_passes"]
    if nan_paths:
        print(f"[strict-readiness] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "strict_submission_readiness_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[strict-readiness] wrote {out_path}", flush=True)

    st_path = out_dir / "strict_submission_readiness_selftest.json"
    with st_path.open("w", encoding="utf-8") as fh:
        json.dump(syn["self_test"]["conditions"], fh, indent=2, sort_keys=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["strict_sub_self_test_passes"] and payload["nan_clean"])
        print(f"[strict-readiness] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
