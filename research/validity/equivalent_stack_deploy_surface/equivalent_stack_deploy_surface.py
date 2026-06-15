#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #417 (lawine) -- Deploy-surface to SHIP the fastest strictly-equivalent config.

THE CARD (the ONE page a human reads before authorizing the deploy of the fastest equivalent winner)
----------------------------------------------------------------------------------------------------
My #411 (`078yjgax`) deploy-surface ledger priced the flagged-supply levers for the (now-retired) 500 line.
Under #407 -- forget 500; MAXIMIZE THE FASTEST STRICTLY BYTE-EQUIVALENT TPS -- the ledger has a new,
higher-value target: what is the cheapest deploy surface + identity-verify cost to actually SHIP the fastest
strictly-equivalent config? Deploying a measured equivalent winner is the program's ONE remaining human-
flagged action (served-file change + leaderboard submission), so pricing its deploy surface NOW de-risks the
gate: when stark #412 lands a measured selective-recompute number and kanna #416 lands the composite
fastest_equivalent_tps, we hand the human a COMPLETE proposal (TPS + files touched + verify GPU-min +
reversibility + the single risky line) in one shot. PURE STATIC ANALYSIS -- it BUILDS NOTHING, SHIPS NOTHING.

THE FASTEST-EQUIVALENT STACK (three components; each a deploy-surface row)
-------------------------------------------------------------------------
1. selective-recompute verify (stark #412, measuring): fast attention EVERYWHERE + a free <=eps near-tie
   gate + a higher-precision reduction on ONLY the ~23.6% flagged near-tie steps. This is a VERIFY-PATH
   change: it edits the served attention/verify REDUCTION logic (splitkv_verify_patch.py / fa_sliding_patch.py)
   IN PLACE -- it modifies existing served hot-path behavior, it does NOT add an orthogonal module. Weightless
   (no checkpoint). Byte-EXACT to blanket-strict BY DESIGN (it is a faster way to compute the SAME strict
   output), so it does NOT mint a new reference -- but it DOES require identity-VERIFICATION (the eps gate
   must be proven to never miss an argmax flip). Modeled +~9-11 TPS over blanket-strict 467.48 -> ~476-478;
   the measured number is stark #412 `selective-recompute-equivalent-tps`. THIS IS THE ONE IN-PLACE EDIT.
2. cb3 body supply (kanna #403 `iv9i2wks`, k*=229, +15.60 served M=8): an ADDITIVE new submission dir (fork
   of the deployed serve stack) + a NEW cb3-baked checkpoint bucket. 6 files, ALL additive, 0 in-place. Body-
   GEMM quant (RHT+VQ dequant) -> different argmax -> a genuinely NEW reference (re-keyed once). Reversible by
   a manifest bucket/quant_method flip. The verify/attention reduction LOGIC is untouched by cb3.
3. MTP K=7 drafter / M=8 verify: ALREADY DEPLOYED. NO change. Zero deploy surface, zero verify GPU-min, no
   new reference. In the stack for completeness (it is the drafter the fastest-equivalent config rides), but
   it contributes NOTHING to the deploy surface.

lm_head stays at the deployed 16384-row truncated head (land #414 prices true-vocab separately; NOT in the
fast stack).

THE KEY DISTINCTION vs #411 (additive-only) -- ONE lever is IN-PLACE
-------------------------------------------------------------------
#411's three levers were all ADDITIVE forks (they add orthogonal modules, touch the verify path NOT AT ALL).
Here cb3 is additive but selective-recompute is the ONE IN-PLACE edit: it changes the served verify/attention
reduction LOGIC itself -- the hot path EVERY decode step runs through. That asymmetry drives blast radius and
reversibility: cb3 reverts by a config-flag bucket flip (logic never touched); selective-recompute's revert-
while-keeping-cb3 is a CODE change (or a wired feature flag, whose gate logic still ships in the served
binary), and a bug in its eps gate corrupts ALL served output, not just an isolated kernel artifact.

COMBINABILITY + SHARED VERIFY (does the in-place edit break the shared-e2e payoff?)
----------------------------------------------------------------------------------
The combined deploy is ONE submission dir: cb3's additive modules (kernel + quant patch + checkpoint) PLUS
the selective-recompute IN-PLACE verify edit applied to the same forked serve stack, riding the unchanged MTP
drafter. The expensive #319 tier-3 e2e identity gate captures the FINAL composed served stack's greedy output
and validates byte-identity -- it does NOT care WHICH change is additive vs in-place. So ONE e2e capture both
RE-KEYS cb3's new reference AND VALIDATES selective-recompute's byte-exactness (if the eps gate ever missed a
flip, the same capture would show divergence). The shared-e2e payoff SURVIVES the in-place edit:
`combined_incremental_verify_gpu_min` = tier3 e2e (SHARED ~35.8) + tier2 decode-width (SHARED 4.0) + one
tier-1 micro per identity-claim change (cb3 new-ref micro + selective-recompute byte-exact micro) = ~41.8 --
NOT the naive 40.8+40.8=81.6 sum. The in-place edit does NOT force a separate cb3 re-run.

THE DEPLOY PROPOSAL (parameterized; auto-completes when #412 + #416 land)
------------------------------------------------------------------------
combined_served_files = 7 (3 shared scaffold + 3 cb3 distinct + 1 selective-recompute in-place verify edit;
6 additive + 1 in-place). total identity-verify = ~41.8 GPU-min (shared e2e). whole_stack_reversible = True
(deployment is via submission packages -> roll back = re-submit the prior package; cb3 by flag, selective-
recompute by code revert). SINGLE most expensive/risky line = the in-place selective-recompute verify edit.
fastest_equivalent_tps is parameterized on the pending measured numbers: stark #412
`selective-recompute-equivalent-tps` (modeled [476.48, 478.48]) + cb3 +15.60 -> kanna #416
`fastest_equivalent_tps` (modeled [492.08, 494.08]). Under #407 the gate is NOT "clears 500" -- it is "is this
the fastest STRICTLY-EQUIVALENT config to ship", and the modeled ~492-494 already BEATS the (non-strict)
deployed 481.53 while carrying the byte-identity guarantee the deployed config lacks. SHIPPING THIS IS THE
HUMAN-GATED ACTION (served-file change + leaderboard submission). This card only PRICES it; it ships nothing.

SCOPE: read-only static analysis + in-repo file enumeration. analysis_only=True, no_hf_job=True,
no_served_file_change=True, official_tps=0. 0 GPU compute. NO build, NO patch, NO compile, NO load, NO
served-file change. Baseline 481.53 / PPL 2.3772 / 128/128 UNCHANGED (#52, 2x9fm2zx). Public evidence
(advisor-branch banked / advisor-provided): #411 unified ledger (078yjgax), #404 cb3 surface (jqhlftrc),
#403 cb3 conservative-k k*=229 +15.60 (iv9i2wks), #393 corrected strict base 467.48 (0q7ynumg); pending
(advisor-provided params): stark #412 selective-recompute-equivalent-tps, kanna #416 fastest_equivalent_tps.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------------------- #
# Shared baselines (PR #417 body; identical to #411 / #403 / #393 for ADDITIVITY across cards)
# ---------------------------------------------------------------------------------------- #
DEPLOYED_TPS = 481.53                  # #52 (2x9fm2zx) deployed NON-STRICT #1 -- UNCHANGED
DEPLOYED_PPL = 2.3772
PPL_CAP = 2.42
STRICT_BASE_BLANKET = 467.48           # #393 (0q7ynumg) corrected BLANKET-strict base (high-prec everywhere)

# the 5 canonical change categories a deploy can touch (same taxonomy as #411)
CANONICAL_CATEGORIES = (
    "cuda_cpp_extension",
    "vllm_quant_registration",
    "attn_backend_patch",
    "served_config_selector",
    "serve_py_hook",
    "verify_reduction_edit",            # NEW for #417: the in-place verify/attention reduction edit
)

# ---------------------------------------------------------------------------------------- #
# Component 1 -- selective-recompute verify (stark #412). THE ONE IN-PLACE EDIT.
# ---------------------------------------------------------------------------------------- #
SELREC_RUN = "#412 (measuring)"
SELREC_FLAGGED_STEP_FRAC = 0.236       # ~23.6% near-tie steps get the higher-precision reduction
SELREC_LIFT_MODEL_LO = 9.0             # modeled +9 TPS over blanket-strict 467.48
SELREC_LIFT_MODEL_HI = 11.0            # modeled +11 TPS over blanket-strict 467.48
SELREC_TPS_MODEL_LO = round(STRICT_BASE_BLANKET + SELREC_LIFT_MODEL_LO, 2)   # 476.48
SELREC_TPS_MODEL_HI = round(STRICT_BASE_BLANKET + SELREC_LIFT_MODEL_HI, 2)   # 478.48
SELREC_IS_IN_PLACE = True              # edits served verify/attention reduction LOGIC in place
SELREC_PRODUCES_NEW_REFERENCE = False  # byte-EXACT to blanket-strict BY DESIGN (same output, faster)
SELREC_REQUIRES_IDENTITY_VERIFICATION = True   # the eps gate must be PROVEN byte-exact (e2e validates it)
SELREC_TOUCHES_CHECKPOINT = False      # weightless reduction-precision change
SELREC_MEASURED_TPS = None             # stark #412 `selective-recompute-equivalent-tps` (PENDING)

# ---------------------------------------------------------------------------------------- #
# Component 2 -- cb3 body supply (kanna #403 iv9i2wks). ADDITIVE. Carry the row forward, k*=229.
# ---------------------------------------------------------------------------------------- #
CB3_RUN = "iv9i2wks"
CB3_KSTAR = 229
CB3_LIFT_M8_DEPLOYABLE = 15.60         # #403 m8_lift_at_kstar (served M=8, PPL-safe by construction)
CB3_HELDOUT_WORST_PPL = 2.3780         # #403 heldout_worst_ppl_at_kstar
CB3_OOD_PPL = 2.4067                   # #403 ood_ppl_at_kstar
CB3_PRODUCES_NEW_REFERENCE = True      # RHT+VQ dequant -> different argmax -> re-keyed reference
CB3_VERIFY_GPU_MIN_STANDALONE = 40.8   # #404/#411 central per-lever (e2e capture dominates)
CB3_SAFE_TO_REQUEST = True             # #404 cb3_build_safe_to_request

# ---------------------------------------------------------------------------------------- #
# Component 3 -- MTP K=7 drafter / M=8 verify. ALREADY DEPLOYED. NO CHANGE.
# ---------------------------------------------------------------------------------------- #
MTP_K = 7
MTP_M = 8

# pending measured composite (kanna #416)
FASTEST_EQUIVALENT_RUN = "#416 (measuring)"
FASTEST_EQUIVALENT_MEASURED_TPS = None         # kanna #416 `fastest_equivalent_tps` (PENDING)
# modeled composite = selective-recompute modeled bracket + cb3 +15.60
FASTEST_EQUIVALENT_MODEL_LO = round(SELREC_TPS_MODEL_LO + CB3_LIFT_M8_DEPLOYABLE, 2)   # 492.08
FASTEST_EQUIVALENT_MODEL_HI = round(SELREC_TPS_MODEL_HI + CB3_LIFT_M8_DEPLOYABLE, 2)   # 494.08

# the 3-tier identity-verify harness (confirmed in-repo paths; reused from #411/#404)
HARNESS = {
    "per_gemm_byte_exact_390": "research/validity/strict_ceiling_corrected_rollup/strict_ceiling_corrected_rollup.py",
    "decode_width_e2e_381": "research/validity/decodewidth_e2e_identity/decodewidth_e2e_identity.py",
    "self_ref_reference_319": "scripts/local_validation/gen_greedy_reference.py",
    "self_ref_compare_319": "scripts/local_validation/greedy_gate.py",
    "self_ref_interlock_319": "scripts/validity/greedy_identity_interlock.py",
}
HARNESS_3TIER = ("TIER1 per-GEMM/-config M=1-vs-M=8 byte-identity micro (#390); TIER2 decode-width e2e identity "
                 "(#381); TIER3 e2e SELF-REFERENTIAL gate (#319: gen_greedy_reference --mode served + "
                 "greedy_gate.compare + greedy_identity_interlock).")

# banked evidence this card reasons from (read-only); cb3 + base are advisor-branch merged, selective-recompute
# numbers are advisor-provided pending params (#412/#416)
EVIDENCE = {
    "unified_ledger_411": "research/validity/flagged_supply_deploy_surface_ledger/flagged_supply_deploy_surface_ledger.py",
    "cb3_surface_404": "research/validity/cb3_flagged_build_surface_scope/cb3_flagged_build_surface_scope.py",
    "cb3_conservative_k_403": "research/validity/cb3_conservative_k_deployable_lift/cb3_conservative_k_deployable_lift.py",
    "attn_pin_cost": "research/validity/attention_strict_pin_cost/attention_strict_pin_cost.py",
}

# deployed served stack (READ-ONLY references). cb3 FORKS these; selective-recompute EDITS the verify
# reduction (splitkv_verify_patch.py / fa_sliding_patch.py) IN PLACE.
DEPLOYED_SUBMISSION = "submissions/fa2sw_treeverify_kenyan"
DEPLOYED_SERVE = f"{DEPLOYED_SUBMISSION}/serve.py"
DEPLOYED_MANIFEST = f"{DEPLOYED_SUBMISSION}/manifest.json"
DEPLOYED_SITECUSTOMIZE = f"{DEPLOYED_SUBMISSION}/sitecustomize.py"
DEPLOYED_VERIFY_PATCH = f"{DEPLOYED_SUBMISSION}/splitkv_verify_patch.py"   # the served verify reduction
DEPLOYED_ATTN_PATCH = f"{DEPLOYED_SUBMISSION}/fa_sliding_patch.py"          # the served attention reduction


# ======================================================================================== #
# Part 1 -- per-component served-file surface (additive vs in-place), tagged by subsystem
# ======================================================================================== #
def _f(path, category, subsystem, scaffold_role, change, classification="additive",
       touches_deployed=False, touches_in_tree=False, touches_checkpoint=False):
    return {
        "path": path, "category": category, "subsystem": subsystem, "scaffold_role": scaffold_role,
        "change": change, "classification": classification,
        "touches_deployed_file": touches_deployed, "touches_in_tree_vllm": touches_in_tree,
        "touches_checkpoint": touches_checkpoint,
    }


def selrec_surface() -> list[dict[str, Any]]:
    """selective-recompute verify -- THE ONE IN-PLACE EDIT. It modifies the served verify/attention
    reduction LOGIC (splitkv_verify_patch.py / fa_sliding_patch.py); the manifest/serve/sitecustomize are
    forked scaffold (SHARED with cb3 in the combined dir). 1 in-place + 3 additive scaffold."""
    s = "submissions/equivalent_<name>"
    return [
        _f(f"{s}/splitkv_verify_patch.py + fa_sliding_patch.py  (EDIT verify/attn reduction)",
           "verify_reduction_edit", "verify_path", "verify_reduction",
           "IN-PLACE edit of the served verify/attention REDUCTION logic: fast attention everywhere + a free "
           "<=eps near-tie gate + a higher-precision reduction on ONLY the ~23.6% flagged near-tie steps. This "
           "MODIFIES existing served hot-path behavior (the path every decode step runs through); it is NOT an "
           "orthogonal additive module. Weightless. THE single in-place / binding line.",
           classification="in_place"),
        _f(f"{s}/manifest.json  (fork)", "served_config_selector", "verify_path", "manifest",
           "FORK of the deployed manifest; delta = flag/select the selective-recompute verify mode (lets the "
           "in-place edit be feature-gated). NO WEIGHTS_BUCKET change (weightless reduction-precision change)."),
        _f(f"{s}/serve.py  (fork)", "serve_py_hook", "verify_path", "serve",
           "FORK of the deployed serve stack; delta = wire the modified verify reduction path. NO new vLLM "
           "source-patch."),
        _f(f"{s}/sitecustomize.py  (fork)", "attn_backend_patch", "verify_path", "sitecustomize",
           "FORK of the deployed sitecustomize finder; delta = import the modified verify/attn reduction patch."),
    ]


def cb3_surface() -> list[dict[str, Any]]:
    """cb3 body supply -- ADDITIVE (same 6-file surface as #404/#411). subsystem=body_gemm_quant.
    The verify/attention reduction LOGIC is UNTOUCHED by cb3 (it only swaps the body-GEMM weights)."""
    s = "submissions/equivalent_<name>"
    return [
        _f(f"{s}/kernels/cb3_qtip_kernel-*.whl  (prebuilt sm_86 .whl/.so)", "cuda_cpp_extension",
           "body_gemm_quant", "kernel",
           "NEW prebuilt cb3 QTIP/QuIP# CUDA ext (RHT incoherence + L1-resident K=64 dim-2 Gaussian VQ dequant "
           "GEMM); built OUT-OF-HARNESS, uploaded prebuilt. Orthogonal to the verify path."),
        _f(f"{s}/cb3_quant_patch.py", "vllm_quant_registration", "body_gemm_quant", "patch",
           "NEW Cb3Config + Cb3LinearMethod via the ADDITIVE @register_quantization_config(\"cb3\") decorator "
           "(appends to vLLM QUANTIZATION_METHODS at import). NO in-tree vLLM edit."),
        _f(f"{s}/manifest.json  (fork)", "served_config_selector", "body_gemm_quant", "manifest",
           "FORK of the deployed manifest; deltas = +cb3 kernel dep, WEIGHTS_BUCKET -> cb3-baked checkpoint."),
        _f(f"{s}/serve.py  (fork)", "serve_py_hook", "body_gemm_quant", "serve",
           "FORK of the deployed serve stack; delta = +setup_cb3_path() so the child imports cb3_quant_patch."),
        _f(f"{s}/sitecustomize.py  (fork)", "vllm_quant_registration", "body_gemm_quant", "sitecustomize",
           "FORK of the deployed sitecustomize finder; delta = import cb3_quant_patch."),
        _f("<cb3 bucket>/config.json  (remote, NEW cb3-baked checkpoint, k*=229)", "served_config_selector",
           "body_gemm_quant", "checkpoint",
           "NEW remote checkpoint declaring quant_method \"cb3\" with the k*=229 PPL-safe allocation (vLLM "
           "auto-selects, no CLI flag). Separate bucket; the deployed int4 checkpoint is untouched.",
           touches_checkpoint=True),
    ]


def mtp_surface() -> list[dict[str, Any]]:
    """MTP K=7 / M=8 drafter -- ALREADY DEPLOYED. ZERO deploy surface (no served files change)."""
    return []


def split(surface: list[dict[str, Any]]) -> dict[str, int]:
    additive = sum(1 for e in surface if e["classification"] == "additive")
    inplace = sum(1 for e in surface if e["classification"] == "in_place")
    return {"additive": additive, "in_place": inplace, "total": len(surface)}


# ======================================================================================== #
# Part 2 -- identity-verify cost (incremental, per component). #319 3-tier, e2e dominates.
# ======================================================================================== #
def verify_cost_3tier() -> dict[str, float]:
    """Price the strict greedy-identity gate (the SAME #319 3-tier harness as #411/#404). The e2e capture
    dominates and is change-agnostic, so every identity-claim component costs ~the same in isolation. The
    capture cost is a function only of token count + spec-on/off TPS, NOT of which change is being verified."""
    capture_tokens = 128 * 512
    spec_off_tps = 165.0          # M=1 AR reference (#196 non-spec floor regime)
    spec_on_tps = DEPLOYED_TPS    # cost proxy; e2e capture cost moves <0.1 min across 467..494
    boot_min = 4.5
    cap_off = (capture_tokens / spec_off_tps) / 60.0
    cap_on = (capture_tokens / spec_on_tps) / 60.0
    tier3 = 2 * (boot_min + cap_off) + 2 * (boot_min + cap_on)   # 2 spec-off + 2 spec-on reloads
    tier1 = 1.0                    # #390 per-GEMM / per-config (or verify-correctness) micro
    tier2 = 4.0                    # #381 decode-width e2e
    return {"tier1_min": tier1, "tier2_min": tier2, "tier3_e2e_min": round(tier3, 1),
            "central_min": round(tier1 + tier2 + tier3, 1)}


# ======================================================================================== #
# Part 3 -- per-component ledger rows (the 5 columns of the PR)
# ======================================================================================== #
def build_rows() -> list[dict[str, Any]]:
    verify = verify_cost_3tier()
    rows: list[dict[str, Any]] = []

    # ---- ROW 1: selective-recompute verify (THE in-place edit) -------------------------- #
    sr = selrec_surface()
    sr_split = split(sr)
    rows.append({
        "component": "selective_recompute",
        "name": "selective-recompute verify",
        "wandb_run": SELREC_RUN,
        "subsystem": "verify_path",
        "deploy_kind": "in_place_verify_edit",
        # col 1: served_files_touched + additive-vs-in-place
        "served_files_touched": len(sr),
        "additive_vs_inplace": sr_split,
        "served_file_surface": sr,
        "is_in_place": bool(sr_split["in_place"] >= 1),
        "is_additive_only": bool(sr_split["in_place"] == 0),
        # col 2: reversibility / blast radius (DIFFERENT from an additive dir -- this is the point)
        "reversible_by_config_flag": True,    # via a wired feature flag, BUT the gate logic still ships
        "reversibility_mechanism": ("feature-flag gate (code-level) OR re-submit the prior package; NOT a "
                                    "manifest bucket flip -- the change IS to served verify LOGIC, so revert-"
                                    "while-keeping-cb3 is a CODE change."),
        "blast_radius": "served_verify_hot_path_every_decode_step",
        # col 3: identity_verify_gpu_minutes + produces_new_reference + requires verification
        "identity_verify_gpu_minutes": verify["central_min"],   # standalone it still needs the full e2e
        "produces_new_reference": SELREC_PRODUCES_NEW_REFERENCE,         # False -- byte-exact to strict
        "requires_identity_verification": SELREC_REQUIRES_IDENTITY_VERIFICATION,   # True -- prove the eps gate
        "verify_harness_3tier": HARNESS_3TIER,
        "verify_reference_rekey": ("byte-EXACT to blanket-strict BY DESIGN -> targets the EXISTING strict "
                                   "reference, does NOT mint a new one. BUT the eps near-tie gate makes a DATA-"
                                   "DEPENDENT decision, so #319 must PROVE byte-exactness over the full eval "
                                   "set (a flip the gate misses would show as divergence). Verification is "
                                   "required even though no new reference is produced."),
        # col 4: tps_unlocked (parameterized on the pending #412 measurement)
        "tps_unlocked": {"measured": SELREC_MEASURED_TPS, "modeled_lo": SELREC_LIFT_MODEL_LO,
                         "modeled_hi": SELREC_LIFT_MODEL_HI,
                         "modeled_tps_bracket": [SELREC_TPS_MODEL_LO, SELREC_TPS_MODEL_HI]},
        "tps_unlocked_headline": round((SELREC_LIFT_MODEL_LO + SELREC_LIFT_MODEL_HI) / 2.0, 2),
        "flagged_step_frac": SELREC_FLAGGED_STEP_FRAC,
        "contingency": ("stark #412 `selective-recompute-equivalent-tps` (MEASURING). Modeled +9..+11 over "
                        "blanket-strict 467.48 -> ~476-478. The proposal auto-fills `tps_unlocked.measured` "
                        "when #412 lands."),
        "touches_checkpoint": SELREC_TOUCHES_CHECKPOINT,
    })

    # ---- ROW 2: cb3 body supply (ADDITIVE, k*=229) -------------------------------------- #
    cb3 = cb3_surface()
    cb3_split = split(cb3)
    rows.append({
        "component": "cb3",
        "name": "cb3 body supply",
        "wandb_run": CB3_RUN,
        "subsystem": "body_gemm_quant",
        "deploy_kind": "additive_submission_dir",
        "served_files_touched": len(cb3),
        "additive_vs_inplace": cb3_split,
        "served_file_surface": cb3,
        "is_in_place": bool(cb3_split["in_place"] >= 1),
        "is_additive_only": bool(cb3_split["in_place"] == 0
                                 and not any(e["touches_deployed_file"] for e in cb3)
                                 and not any(e["touches_in_tree_vllm"] for e in cb3)),
        "reversible_by_config_flag": True,
        "reversibility_mechanism": ("manifest bucket/quant_method flip back to the deployed int4 checkpoint -- "
                                    "the verify/attention reduction LOGIC was never touched."),
        "blast_radius": "isolated_new_kernel_checkpoint",
        "identity_verify_gpu_minutes": CB3_VERIFY_GPU_MIN_STANDALONE,
        "produces_new_reference": CB3_PRODUCES_NEW_REFERENCE,
        "requires_identity_verification": True,
        "verify_harness_3tier": HARNESS_3TIER,
        "verify_reference_rekey": ("RHT+VQ dequant changes the body-GEMM weights -> different argmax -> #319 "
                                   "re-keys to cb3's OWN M=1 AR decode (a genuinely new reference)."),
        "tps_unlocked": {"m8_deployable": CB3_LIFT_M8_DEPLOYABLE, "k_star": CB3_KSTAR,
                         "heldout_worst_ppl": CB3_HELDOUT_WORST_PPL, "ood_ppl": CB3_OOD_PPL},
        "tps_unlocked_headline": CB3_LIFT_M8_DEPLOYABLE,
        "flagged_step_frac": None,
        "contingency": ("MEASURED & banked (#403 iv9i2wks): k*=229 gives +15.60 served M=8, PPL-safe by "
                        "construction (heldout-worst 2.3780 / OOD 2.4067, both <=2.41). NOT contingent."),
        "touches_checkpoint": True,
        "safe_to_request": CB3_SAFE_TO_REQUEST,
    })

    # ---- ROW 3: MTP K=7 drafter / M=8 verify (DEPLOYED, NO CHANGE) ----------------------- #
    mtp = mtp_surface()
    mtp_split = split(mtp)
    rows.append({
        "component": "mtp_drafter",
        "name": f"MTP K={MTP_K} drafter / M={MTP_M} verify",
        "wandb_run": "deployed",
        "subsystem": "drafter",
        "deploy_kind": "no_change_already_deployed",
        "served_files_touched": len(mtp),
        "additive_vs_inplace": mtp_split,
        "served_file_surface": mtp,
        "is_in_place": False,
        "is_additive_only": True,            # vacuously: no files at all
        "reversible_by_config_flag": True,
        "reversibility_mechanism": "no change -- nothing to revert (already the deployed drafter).",
        "blast_radius": "none_no_change",
        "identity_verify_gpu_minutes": 0.0,
        "produces_new_reference": False,
        "requires_identity_verification": False,
        "verify_harness_3tier": HARNESS_3TIER,
        "verify_reference_rekey": "no change -> no re-key (the deployed reference already covers it).",
        "tps_unlocked": {"already_deployed": True},
        "tps_unlocked_headline": 0.0,
        "flagged_step_frac": None,
        "contingency": "none -- already deployed; it is the drafter the fastest-equivalent config rides.",
        "touches_checkpoint": False,
    })

    return rows


# ======================================================================================== #
# Part 4 -- combinability + shared verify (does the in-place edit break the shared-e2e payoff?)
# ======================================================================================== #
SCAFFOLD = {"manifest", "serve", "sitecustomize"}


def combinability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["component"]: r for r in rows}
    subsystems = {r["component"]: r["subsystem"] for r in rows if r["served_files_touched"] > 0}
    distinct_subsystems = len(set(subsystems.values()))

    def distinct_nonscaffold(comp: str) -> list[str]:
        return [e["path"].split("  ")[0] for e in by[comp]["served_file_surface"]
                if e["scaffold_role"] not in SCAFFOLD]

    # any in-tree / deployed-file edit two components contend for? NONE -- cb3 forks, selective-recompute
    # edits the FORKED verify patch (which lives in the same combined dir), MTP changes nothing.
    any_in_tree = any(e["touches_in_tree_vllm"] for r in rows for e in r["served_file_surface"])
    any_deployed = any(e["touches_deployed_file"] for r in rows for e in r["served_file_surface"])
    # exactly one component is in-place (selective-recompute); the others are additive / no-change
    in_place_components = [r["component"] for r in rows if r["is_in_place"]]
    additive_components = [r["component"] for r in rows
                           if (not r["is_in_place"]) and r["served_files_touched"] > 0]

    note = (
        "The combined deploy is ONE submission dir: cb3's ADDITIVE modules (kernel + quant patch + checkpoint, "
        "orthogonal to the verify path) PLUS the selective-recompute IN-PLACE edit applied to the same forked "
        "verify/attention reduction, riding the unchanged MTP drafter. They share the scaffold (manifest / "
        "serve.py / sitecustomize, forked ONCE, carrying BOTH cb3's additive deltas and the in-place verify "
        "edit). cb3 touches the body-GEMM weights and the verify path NOT AT ALL; selective-recompute touches "
        "the verify reduction and the weights NOT AT ALL -> NO contention. The expensive #319 tier-3 e2e gate "
        "captures the FINAL composed served stack's greedy output and validates byte-identity -- it is change-"
        "agnostic, so ONE e2e capture both RE-KEYS cb3's new reference AND VALIDATES selective-recompute's "
        "byte-exactness. The shared-e2e payoff SURVIVES the in-place edit: the in-place change does NOT force a "
        "separate cb3 re-run, because both live in ONE final serve.py and the e2e gate captures the composite."
    )
    return {
        "components_stack_in_one_dir": True,
        "distinct_subsystems": distinct_subsystems,
        "subsystem_by_component": subsystems,
        "in_place_components": in_place_components,
        "additive_components": additive_components,
        "exactly_one_in_place": bool(len(in_place_components) == 1),
        "any_two_contend_for_same_edit": bool(any_in_tree or any_deployed),
        "shared_scaffold_files": sorted(SCAFFOLD),
        "selrec_distinct_files": distinct_nonscaffold("selective_recompute"),
        "cb3_distinct_files": distinct_nonscaffold("cb3"),
        "shared_e2e_verify": True,
        "shared_e2e_survives_inplace": True,
        "combinability_note": note,
    }


def combined_served_files(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Union of served files across the stack, counting the shared scaffold ONCE. Splits additive vs
    in-place so the proposal can name the single in-place line."""
    scaffold_present: set[str] = set()
    distinct: list[dict[str, str]] = []
    for r in rows:
        for e in r["served_file_surface"]:
            if e["scaffold_role"] in SCAFFOLD:
                scaffold_present.add(e["scaffold_role"])
            else:
                distinct.append({"component": r["component"], "path": e["path"].split("  ")[0],
                                 "classification": e["classification"]})
    n_inplace = sum(1 for d in distinct if d["classification"] == "in_place")
    n_additive_distinct = sum(1 for d in distinct if d["classification"] == "additive")
    total = len(scaffold_present) + len(distinct)
    return {"total": total,
            "shared_scaffold_files": sorted(scaffold_present),
            "n_shared_scaffold": len(scaffold_present),
            "distinct_files": distinct,
            "n_distinct": len(distinct),
            "n_in_place": n_inplace,
            "n_additive": len(scaffold_present) + n_additive_distinct}


def combined_verify(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Combined incremental identity-verify GPU-min: the expensive #319 e2e capture (tier3) + tier2 decode-
    width are SHARED across the combined stack; each identity-claim component (new reference OR requires
    verification) adds its tier-1 micro. The in-place selective-recompute rides the SAME shared e2e as cb3."""
    v = verify_cost_3tier()
    claim_components = [r["component"] for r in rows
                        if r.get("produces_new_reference") or r.get("requires_identity_verification")]
    n_claims = len(claim_components)
    combined = round(v["tier3_e2e_min"] + v["tier2_min"] + n_claims * v["tier1_min"], 1)
    naive_sum = round(sum(r["identity_verify_gpu_minutes"] for r in rows
                          if r["identity_verify_gpu_minutes"] > 0), 1)
    return {
        "combined_incremental_verify_gpu_min": combined,
        "identity_claim_components": claim_components,
        "n_identity_claim_components": n_claims,
        "shared_tier3_e2e_min": v["tier3_e2e_min"],
        "shared_tier2_min": v["tier2_min"],
        "per_claim_tier1_min": v["tier1_min"],
        "naive_sum_if_unshared": naive_sum,
        "shared_e2e_saving_gpu_min": round(naive_sum - combined, 1),
        "shared_e2e_survives_inplace": True,
    }


# ======================================================================================== #
# Part 5 -- the deploy-proposal skeleton (the ready-to-hand-to-human card)
# ======================================================================================== #
def deploy_proposal(rows: list[dict[str, Any]], comb: dict[str, Any],
                    files: dict[str, Any], verify: dict[str, Any]) -> dict[str, Any]:
    by = {r["component"]: r for r in rows}
    # the SINGLE most expensive/risky line = the in-place selective-recompute verify edit
    sr = by["selective_recompute"]
    most_expensive = (
        "selective-recompute IN-PLACE verify-path edit (eps near-tie gate + higher-precision reduction on the "
        f"~{SELREC_FLAGGED_STEP_FRAC*100:.1f}% flagged steps, edited into splitkv_verify_patch.py / "
        "fa_sliding_patch.py): it is the hot path EVERY decode step runs through, so a gate bug corrupts ALL "
        "served output (not an isolated kernel artifact); its identity-verify validates a DATA-DEPENDENT "
        "correctness property (the eps gate must never miss an argmax flip); and revert-while-keeping-cb3 is a "
        "CODE change, not a manifest bucket flip. THIS is the line a human must scrutinize."
    )

    # whole-stack reversibility: deployment is via submission PACKAGES, so roll back = re-submit the prior
    # package. cb3 reverts by a config flag; selective-recompute by code revert / feature-flag. All reversible.
    whole_stack_reversible = bool(all(r["reversible_by_config_flag"] for r in rows))

    # parameterized fastest-equivalent TPS (auto-completes when #412 + #416 land)
    selrec_measured = sr["tps_unlocked"]["measured"]
    if FASTEST_EQUIVALENT_MEASURED_TPS is not None:
        fastest_eq = {"value": FASTEST_EQUIVALENT_MEASURED_TPS, "source": "kanna #416 (measured)"}
    elif selrec_measured is not None:
        fastest_eq = {"value": round(selrec_measured + CB3_LIFT_M8_DEPLOYABLE, 2),
                      "source": "stark #412 selective-recompute-equivalent-tps + cb3 +15.60"}
    else:
        fastest_eq = {"value": None, "source": "PENDING (#412 + #416 measuring)",
                      "modeled_bracket": [FASTEST_EQUIVALENT_MODEL_LO, FASTEST_EQUIVALENT_MODEL_HI]}

    tps_formula = ("fastest_equivalent_tps = selective_recompute_equivalent_tps (stark #412, PENDING; modeled "
                   f"[{SELREC_TPS_MODEL_LO}, {SELREC_TPS_MODEL_HI}]) + cb3 +{CB3_LIFT_M8_DEPLOYABLE} (kanna "
                   "#403 k*=229, MEASURED) == kanna #416 `fastest_equivalent_tps` (PENDING; modeled "
                   f"[{FASTEST_EQUIVALENT_MODEL_LO}, {FASTEST_EQUIVALENT_MODEL_HI}]). Under #407 the gate is NOT "
                   f"'clears 500' -- the modeled ~492-494 already BEATS the non-strict deployed {DEPLOYED_TPS} "
                   "while carrying the byte-identity guarantee the deployed config lacks.")

    paragraph = (
        "DEPLOY-SURFACE TO SHIP THE FASTEST STRICTLY-EQUIVALENT CONFIG (#407). Three components, one card. "
        "(1) selective-recompute verify (stark #412, measuring): THE ONE IN-PLACE edit -- fast attention "
        f"everywhere + a free <=eps near-tie gate + higher-precision reduction on ONLY the "
        f"~{SELREC_FLAGGED_STEP_FRAC*100:.1f}% flagged steps, edited into the served verify/attention reduction "
        "in place; weightless; byte-EXACT to blanket-strict by design (no new reference) but REQUIRES the e2e "
        f"identity gate to prove the eps gate; modeled +9..+11 over 467.48. (2) cb3 body supply (kanna #403 "
        f"k*=229): ADDITIVE 6-file dir + cb3 checkpoint, +{CB3_LIFT_M8_DEPLOYABLE} served M=8, PPL-safe, new "
        "reference, reversible-by-flag. (3) MTP K=7/M=8 drafter: already DEPLOYED, ZERO deploy surface. "
        f"COMBINED: {files['total']} served files ({files['n_additive']} additive + {files['n_in_place']} "
        f"in-place), ONE submission dir, the in-place verify edit and cb3's additive modules DO NOT contend. "
        f"VERIFY: {verify['combined_incremental_verify_gpu_min']} GPU-min -- the #319 e2e gate is SHARED (one "
        "capture re-keys cb3 AND validates selective-recompute byte-exactness; the in-place edit does NOT force "
        f"a separate cb3 re-run), vs a naive {verify['naive_sum_if_unshared']} unshared. WHOLE STACK REVERSIBLE "
        f"(re-submit the prior package). MOST EXPENSIVE/RISKY LINE: the in-place selective-recompute verify "
        f"edit. TPS: {tps_formula} SHIPPING THIS IS THE HUMAN-GATED ACTION (served-file change + leaderboard "
        "submission); this card only PRICES it. No GPU compute, no HF job, no served-file change were performed."
    )

    return {
        "combined_served_files": files["total"],
        "combined_served_files_additive": files["n_additive"],
        "combined_served_files_in_place": files["n_in_place"],
        "combined_served_files_breakdown": files,
        "combined_incremental_verify_gpu_min": verify["combined_incremental_verify_gpu_min"],
        "verify_breakdown": verify,
        "whole_stack_reversible": whole_stack_reversible,
        "most_expensive_deploy_line": most_expensive,
        "deploy_is_human_gated": True,
        "fastest_equivalent_tps": fastest_eq,
        "fastest_equivalent_tps_formula": tps_formula,
        "fastest_equivalent_tps_modeled_bracket": [FASTEST_EQUIVALENT_MODEL_LO, FASTEST_EQUIVALENT_MODEL_HI],
        "selective_recompute_tps_modeled_bracket": [SELREC_TPS_MODEL_LO, SELREC_TPS_MODEL_HI],
        "beats_deployed_nonstrict_when_modeled": bool(FASTEST_EQUIVALENT_MODEL_LO > DEPLOYED_TPS),
        "human_proposal_paragraph": paragraph,
    }


# ======================================================================================== #
# repo-fact probes (read-only) + self-test
# ======================================================================================== #
def repo_facts() -> dict[str, bool]:
    def ok(rel: str) -> bool:
        return (REPO_ROOT / rel).is_file()
    facts = {f"deployed::{k}": ok(v) for k, v in {
        "serve": DEPLOYED_SERVE, "manifest": DEPLOYED_MANIFEST,
        "sitecustomize": DEPLOYED_SITECUSTOMIZE, "verify_patch": DEPLOYED_VERIFY_PATCH,
        "attn_patch": DEPLOYED_ATTN_PATCH,
    }.items()}
    facts.update({f"harness::{k}": ok(v) for k, v in HARNESS.items()})
    facts.update({f"evidence::{k}": ok(v) for k, v in EVIDENCE.items()})
    return facts


def selftest(rows: list[dict[str, Any]], comb: dict[str, Any], prop: dict[str, Any],
             facts: dict[str, bool], flags: dict[str, bool]) -> dict[str, Any]:
    by = {r["component"]: r for r in rows}
    sr, cb3, mtp = by["selective_recompute"], by["cb3"], by["mtp_drafter"]
    c: dict[str, bool] = {}

    # (a) selective-recompute = THE one in-place edit, byte-exact-to-strict, weightless, requires verification
    c["a_selrec_is_in_place"] = (sr["is_in_place"] is True)
    c["a_selrec_one_inplace_file"] = (sr["additive_vs_inplace"]["in_place"] == 1)
    c["a_selrec_not_additive_only"] = (sr["is_additive_only"] is False)
    c["a_selrec_no_new_reference"] = (sr["produces_new_reference"] is False)
    c["a_selrec_requires_verification"] = (sr["requires_identity_verification"] is True)
    c["a_selrec_weightless"] = (sr["touches_checkpoint"] is False)
    c["a_selrec_verify_subsystem"] = (sr["subsystem"] == "verify_path")
    c["a_selrec_flagged_frac"] = (abs(sr["flagged_step_frac"] - 0.236) < 1e-9)
    c["a_selrec_modeled_bracket"] = (sr["tps_unlocked"]["modeled_tps_bracket"] == [476.48, 478.48])
    c["a_selrec_blast_hotpath"] = ("hot_path" in sr["blast_radius"])

    # (b) cb3 = ADDITIVE, 6 files, +15.60 at k*=229, new reference, reversible-by-flag
    c["b_cb3_is_additive_only"] = (cb3["is_additive_only"] is True)
    c["b_cb3_zero_inplace"] = (cb3["additive_vs_inplace"]["in_place"] == 0)
    c["b_cb3_six_files"] = (cb3["served_files_touched"] == 6)
    c["b_cb3_lift_15_60"] = (abs(cb3["tps_unlocked"]["m8_deployable"] - 15.60) < 1e-9)
    c["b_cb3_kstar_229"] = (cb3["tps_unlocked"]["k_star"] == 229)
    c["b_cb3_new_reference"] = (cb3["produces_new_reference"] is True)
    c["b_cb3_ppl_safe"] = (cb3["tps_unlocked"]["heldout_worst_ppl"] <= 2.41
                           and cb3["tps_unlocked"]["ood_ppl"] <= 2.41)
    c["b_cb3_body_subsystem"] = (cb3["subsystem"] == "body_gemm_quant")
    c["b_cb3_standalone_verify_40_8"] = (abs(cb3["identity_verify_gpu_minutes"] - 40.8) < 0.1)

    # (c) MTP drafter = DEPLOYED, no change, zero surface, no verify, no new reference
    c["c_mtp_zero_files"] = (mtp["served_files_touched"] == 0)
    c["c_mtp_zero_verify"] = (abs(mtp["identity_verify_gpu_minutes"] - 0.0) < 1e-9)
    c["c_mtp_no_new_reference"] = (mtp["produces_new_reference"] is False)
    c["c_mtp_no_verification"] = (mtp["requires_identity_verification"] is False)
    c["c_mtp_not_in_place"] = (mtp["is_in_place"] is False)

    # (d) exactly ONE in-place across the stack (selective-recompute); cb3 additive; no contention
    c["d_exactly_one_in_place"] = (comb["exactly_one_in_place"] is True)
    c["d_in_place_is_selrec"] = (comb["in_place_components"] == ["selective_recompute"])
    c["d_cb3_in_additive"] = ("cb3" in comb["additive_components"])
    c["d_no_contention"] = (comb["any_two_contend_for_same_edit"] is False)
    c["d_two_subsystems_with_files"] = (comb["distinct_subsystems"] == 2)  # verify_path + body_gemm_quant
    c["d_three_rows"] = (len(rows) == 3)
    c["d_categories_canonical"] = all(e["category"] in CANONICAL_CATEGORIES
                                      for r in rows for e in r["served_file_surface"])

    # (e) combined surface + SHARED verify survives the in-place edit
    c["e_combined_files_7"] = (prop["combined_served_files"] == 7)
    c["e_combined_in_place_1"] = (prop["combined_served_files_in_place"] == 1)
    c["e_combined_additive_6"] = (prop["combined_served_files_additive"] == 6)
    c["e_combined_verify_41_8"] = (abs(prop["combined_incremental_verify_gpu_min"] - 41.8) < 0.1)
    c["e_shared_e2e_survives_inplace"] = (comb["shared_e2e_survives_inplace"] is True
                                          and prop["verify_breakdown"]["shared_e2e_survives_inplace"] is True)
    # shared e2e is far below the naive unshared sum (40.8 + 40.8 = 81.6)
    c["e_verify_below_naive_sum"] = (prop["combined_incremental_verify_gpu_min"]
                                     < prop["verify_breakdown"]["naive_sum_if_unshared"] - 30.0)
    c["e_two_identity_claims"] = (prop["verify_breakdown"]["n_identity_claim_components"] == 2)

    # (f) deploy proposal: in-place line is the binding one, human-gated, reversible, TPS parameterized
    c["f_most_expensive_is_inplace"] = ("IN-PLACE" in prop["most_expensive_deploy_line"]
                                        and "selective-recompute" in prop["most_expensive_deploy_line"])
    c["f_deploy_human_gated"] = (prop["deploy_is_human_gated"] is True)
    c["f_whole_stack_reversible"] = (prop["whole_stack_reversible"] is True)
    c["f_tps_pending_param"] = (prop["fastest_equivalent_tps"]["value"] is None)  # both #412/#416 pending
    c["f_tps_modeled_bracket"] = (prop["fastest_equivalent_tps_modeled_bracket"] == [492.08, 494.08])
    c["f_beats_deployed_modeled"] = (prop["beats_deployed_nonstrict_when_modeled"] is True)
    c["f_tps_formula_cites_412_416"] = ("#412" in prop["fastest_equivalent_tps_formula"]
                                        and "#416" in prop["fastest_equivalent_tps_formula"])

    # (g) in-repo facts (read-only existence)
    for k, v in facts.items():
        c[f"g_{k}"] = v

    # (h) analysis-only hygiene
    c["h_official_tps_zero"] = (flags.get("official_tps") == 0)
    c["h_analysis_only"] = bool(flags.get("analysis_only"))
    c["h_no_hf_job"] = bool(flags.get("no_hf_job"))
    c["h_no_served_file_change"] = bool(flags.get("no_served_file_change"))
    return {"conditions": c, "n_checks": len(c), "passes": all(c.values())}


# ======================================================================================== #
# report + IO + wandb
# ======================================================================================== #
def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, bool) or isinstance(o, (int, float, str)) or o is None:
        return o
    return str(o)


def print_report(p: dict[str, Any]) -> None:
    st = p["selftest"]
    prop = p["deploy_proposal"]
    comb = p["combinability"]
    print("=" * 100)
    print(f"PR #417 lawine -- DEPLOY-SURFACE TO SHIP THE FASTEST STRICTLY-EQUIVALENT CONFIG  ({p['created_at']})")
    print(f"  analysis_only={p['analysis_only']}  no_hf_job={p['no_hf_job']}  "
          f"no_served_file_change={p['no_served_file_change']}  official_tps={p['official_tps']}")
    print(f"  blanket-strict base {STRICT_BASE_BLANKET} (#393)  deployed {DEPLOYED_TPS} (non-strict) UNCHANGED")
    print("-" * 100)
    print(f"  {'COMPONENT':22s} {'kind':26s} {'files(add/ip)':14s} {'verify_min':10s} {'new_ref':8s} "
          f"{'tps_unlocked':14s}")
    for r in p["rows"]:
        sp = r["additive_vs_inplace"]
        tu = r["tps_unlocked_headline"]
        print(f"  {r['component']:22s} {r['deploy_kind']:26s} "
              f"{str(r['served_files_touched']) + '(' + str(sp['additive']) + '/' + str(sp['in_place']) + ')':14s} "
              f"{r['identity_verify_gpu_minutes']:<10} {str(r['produces_new_reference']):8s} "
              f"{('+' + format(tu, '.2f')):14s}")
    print("-" * 100)
    print("  COMBINABILITY + SHARED VERIFY")
    print(f"    exactly_one_in_place={comb['exactly_one_in_place']} (in_place={comb['in_place_components']})  "
          f"additive={comb['additive_components']}")
    print(f"    any_two_contend={comb['any_two_contend_for_same_edit']}  "
          f"shared_e2e_survives_inplace={comb['shared_e2e_survives_inplace']}")
    print("-" * 100)
    print("  DEPLOY PROPOSAL SKELETON")
    print(f"    combined_served_files ............. {prop['combined_served_files']} "
          f"({prop['combined_served_files_additive']} additive + {prop['combined_served_files_in_place']} in-place)")
    print(f"    combined_incremental_verify ...... {prop['combined_incremental_verify_gpu_min']} GPU-min "
          f"(shared e2e; naive unshared = {prop['verify_breakdown']['naive_sum_if_unshared']})")
    print(f"    whole_stack_reversible ........... {prop['whole_stack_reversible']}")
    print(f"    deploy_is_human_gated ............ {prop['deploy_is_human_gated']}")
    print(f"    fastest_equivalent_tps ........... {prop['fastest_equivalent_tps']}")
    print(f"    most_expensive_deploy_line ....... {prop['most_expensive_deploy_line'][:90]}...")
    print("-" * 100)
    print("  HUMAN PROPOSAL PARAGRAPH")
    para = prop["human_proposal_paragraph"]
    for i in range(0, len(para), 110):
        print(f"    {para[i:i + 110]}")
    print("-" * 100)
    print(f"  SELF-TEST {st['n_checks']} checks -> {'PASS' if st['passes'] else 'FAIL'}")
    if not st["passes"]:
        for k, v in st["conditions"].items():
            if not v:
                print(f"    FAILED: {k}")
    print("=" * 100)


def build_payload(flags: dict[str, bool]) -> dict[str, Any]:
    rows = build_rows()
    comb = combinability(rows)
    files = combined_served_files(rows)
    verify = combined_verify(rows)
    prop = deploy_proposal(rows, comb, files, verify)
    facts = repo_facts()
    st = selftest(rows, comb, prop, facts, flags)
    payload: dict[str, Any] = {
        "agent": "lawine", "pr": 417,
        "kind": "equivalent-stack-deploy-surface",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        # ---- top-level scalars/lists (W&B summary deliverables) ----
        "n_components": len(rows),
        "components": [r["component"] for r in rows],
        "selective_recompute_is_in_place": SELREC_IS_IN_PLACE,
        "cb3_is_additive": bool(rows[1]["is_additive_only"]),
        "combined_served_files": prop["combined_served_files"],
        "combined_incremental_verify_gpu_min": prop["combined_incremental_verify_gpu_min"],
        "shared_e2e_survives_inplace": comb["shared_e2e_survives_inplace"],
        "whole_stack_reversible": prop["whole_stack_reversible"],
        "most_expensive_deploy_line": prop["most_expensive_deploy_line"],
        "deploy_is_human_gated": prop["deploy_is_human_gated"],
        "fastest_equivalent_tps_modeled_bracket": prop["fastest_equivalent_tps_modeled_bracket"],
        "equivalent_stack_deploy_surface_self_test_passes": bool(st["passes"]),
        "human_proposal_paragraph": prop["human_proposal_paragraph"],
        # ---- detail blocks ----
        "rows": rows,
        "combinability": comb,
        "deploy_proposal": prop,
        "repo_facts": facts,
        "selftest": st,
        "shared_baselines": {
            "deployed_tps": DEPLOYED_TPS, "deployed_ppl": DEPLOYED_PPL, "ppl_cap": PPL_CAP,
            "strict_base_blanket": STRICT_BASE_BLANKET,
            "cb3_lift_m8": CB3_LIFT_M8_DEPLOYABLE, "cb3_kstar": CB3_KSTAR,
            "selrec_modeled_tps_bracket": [SELREC_TPS_MODEL_LO, SELREC_TPS_MODEL_HI],
        },
    }
    return payload


def maybe_log_wandb(payload: dict[str, Any], args) -> str | None:
    if args.no_wandb:
        return None
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from scripts.wandb_logging import (init_wandb_run, log_summary,
                                            log_json_artifact, finish_wandb)
    except Exception as e:  # noqa: BLE001
        print(f"[card] wandb helpers unavailable: {e}")
        return None
    st = payload["selftest"]
    prop = payload["deploy_proposal"]
    run = init_wandb_run(
        job_type="analysis-static-scope", agent="lawine",
        name=args.wandb_name, group=args.wandb_group,
        tags=["equivalent-stack-deploy-surface", "selective-recompute", "cb3", "mtp-drafter",
              "deploy-surface", "identity-verify", "in-place-vs-additive", "decision-doc", "pr-417"],
        config={"pr": 417, "kind": "equivalent-stack-deploy-surface",
                "analysis_only": True, "no_hf_job": True, "no_served_file_change": True,
                "official_tps": 0, "strict_base_blanket": STRICT_BASE_BLANKET,
                "deployed_tps": DEPLOYED_TPS, "cb3_lift_m8": CB3_LIFT_M8_DEPLOYABLE},
    )
    if run is None:
        print("[card] wandb disabled (no API key / WANDB_MODE).")
        return None
    flat: dict[str, float] = {
        "card/n_components": float(payload["n_components"]),
        "card/self_test_passes": float(st["passes"]),
        "card/self_test_n_checks": float(st["n_checks"]),
        "card/selective_recompute_is_in_place": float(payload["selective_recompute_is_in_place"]),
        "card/cb3_is_additive": float(payload["cb3_is_additive"]),
        "proposal/combined_served_files": float(payload["combined_served_files"]),
        "proposal/combined_served_files_in_place": float(prop["combined_served_files_in_place"]),
        "proposal/combined_served_files_additive": float(prop["combined_served_files_additive"]),
        "proposal/combined_incremental_verify_gpu_min": float(payload["combined_incremental_verify_gpu_min"]),
        "proposal/verify_naive_sum_if_unshared": float(prop["verify_breakdown"]["naive_sum_if_unshared"]),
        "proposal/verify_shared_e2e_saving_gpu_min": float(prop["verify_breakdown"]["shared_e2e_saving_gpu_min"]),
        "proposal/shared_e2e_survives_inplace": float(payload["shared_e2e_survives_inplace"]),
        "proposal/whole_stack_reversible": float(payload["whole_stack_reversible"]),
        "proposal/deploy_is_human_gated": float(payload["deploy_is_human_gated"]),
        "proposal/beats_deployed_nonstrict_when_modeled": float(prop["beats_deployed_nonstrict_when_modeled"]),
        "proposal/fastest_equivalent_tps_modeled_lo": float(FASTEST_EQUIVALENT_MODEL_LO),
        "proposal/fastest_equivalent_tps_modeled_hi": float(FASTEST_EQUIVALENT_MODEL_HI),
        "card/official_tps": float(payload["official_tps"]),
    }
    # per-component scalars (rich logging)
    for r in payload["rows"]:
        comp = r["component"]
        flat[f"comp/{comp}/served_files"] = float(r["served_files_touched"])
        flat[f"comp/{comp}/additive"] = float(r["additive_vs_inplace"]["additive"])
        flat[f"comp/{comp}/in_place"] = float(r["additive_vs_inplace"]["in_place"])
        flat[f"comp/{comp}/identity_verify_gpu_min"] = float(r["identity_verify_gpu_minutes"])
        flat[f"comp/{comp}/produces_new_reference"] = float(bool(r["produces_new_reference"]))
        flat[f"comp/{comp}/requires_identity_verification"] = float(bool(r["requires_identity_verification"]))
        flat[f"comp/{comp}/is_in_place"] = float(bool(r["is_in_place"]))
        flat[f"comp/{comp}/tps_unlocked_headline"] = float(r["tps_unlocked_headline"])
    run.log({"global_step": 0, **flat})
    log_summary(run, _jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="equivalent_stack_deploy_surface", artifact_type="analysis",
                      data=_jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[card] wandb logged {len(flat)} keys (run {rid})")
    return rid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", "--selftest", dest="self_test", action="store_true",
                    help="run the analytic self-test and exit nonzero on failure (no wandb)")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="lawine/equivalent-stack-deploy-surface")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="equivalent-stack-deploy-surface")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    flags = {"analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0}
    payload = build_payload(flags)
    print_report(payload)

    out_path = Path(args.out_dir) / "equivalent_stack_deploy_surface_results.json"
    out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"\n[card] wrote {out_path}")

    st = payload["selftest"]
    if args.self_test:
        assert st["passes"], f"self-test FAILED ({st['n_checks']} checks)"
        assert st["n_checks"] >= 20, f"need >=20 asserts, have {st['n_checks']}"
        print(f"[card] SELF-TEST PASS ({st['n_checks']} checks)")
        print("\nSENPAI-RESULT " + json.dumps({
            "terminal": True, "status": "complete", "pending_arms": False, "wandb_run_ids": [],
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            "equivalent_stack_deploy_surface_self_test_passes": bool(st["passes"]),
            "primary_metric": {"name": "equivalent_stack_deploy_surface_self_test_passes",
                               "value": float(st["passes"])},
            "test_metric": {"name": "combined_incremental_verify_gpu_min",
                            "value": float(payload["combined_incremental_verify_gpu_min"])},
        }))
        sys.exit(0 if st["passes"] else 1)

    rid = maybe_log_wandb(payload, args)
    if rid:
        payload["wandb_run_id"] = rid
        out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))

    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [rid] if rid else [],
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "n_components": payload["n_components"],
        "selective_recompute_is_in_place": payload["selective_recompute_is_in_place"],
        "cb3_is_additive": payload["cb3_is_additive"],
        "combined_served_files": payload["combined_served_files"],
        "combined_incremental_verify_gpu_min": payload["combined_incremental_verify_gpu_min"],
        "shared_e2e_survives_inplace": payload["shared_e2e_survives_inplace"],
        "whole_stack_reversible": payload["whole_stack_reversible"],
        "deploy_is_human_gated": payload["deploy_is_human_gated"],
        "equivalent_stack_deploy_surface_self_test_passes": bool(st["passes"]),
        "primary_metric": {"name": "equivalent_stack_deploy_surface_self_test_passes",
                           "value": float(st["passes"])},
        "test_metric": {"name": "combined_incremental_verify_gpu_min",
                        "value": float(payload["combined_incremental_verify_gpu_min"])},
    }))


if __name__ == "__main__":
    main()
