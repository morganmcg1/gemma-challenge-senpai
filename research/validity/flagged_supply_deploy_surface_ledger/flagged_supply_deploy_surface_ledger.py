#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #411 (lawine) -- Unified flagged-supply-lever deploy-surface ledger.

THE DECISION DOC (the ONE page a human reads before authorizing any >500 build)
-------------------------------------------------------------------------------
My #404 (`jqhlftrc`) priced ONE flagged supply lever (cb3 body-read shrink) to the floor: 6 additive
files, 0 in-place edits, ~41 GPU-min identity-verify, reversible-by-config, cb3_build_safe_to_request=True.
But the program has THREE flagged supply levers, and the >500 authorization decision needs ALL of them on
one page -- so a human can authorize the CHEAPEST SUFFICIENT COMBINATION the moment the numbers settle.
This card carries the cb3 row forward and adds the other two, same rigor, same schema, then rolls up the
cheapest sufficient combination and its SINGLE binding contingency. PURE STATIC ANALYSIS -- it BUILDS NOTHING.

THE THREE FLAGGED SUPPLY LEVERS (each a row; columns per the #404 schema)
------------------------------------------------------------------------
1. cb3 body-read shrink (#404 `jqhlftrc`): NEW submission dir (fork of fa2sw_treeverify_kenyan) + NEW
   cb3-baked checkpoint bucket. 6 files, ALL additive, 0 in-place. Body-GEMM quant subsystem. New greedy
   reference (RHT+VQ dequant -> different argmax). +33 honest / +38 realistic M=1 TPS, contingent on
   kanna #403 PPL-safe-k (measured floor only +15.67). ~40.8 GPU-min identity-verify. safe_to_request=True.
2. pinned-K attention rebuild (#400 `o7yhpkej`): a fixed 64-CTA split-reduce is M-invariant byte-exact
   FEASIBLE, but the served varlen FA2 kernel REJECTS num_splits>1 (NotImplementedError) -> a kernel
   REBUILD is the only way to reach it, and the multi-split reduction order != the deployed serial un-pack
   -> pinnedk_produces_new_reference=True (re-capture = flagged served-file change). 5 files, ALL additive,
   0 in-place, NO checkpoint change (weightless reduction-order change). Attention-kernel subsystem.
   Buys at most +14.29 TPS and CAPS at 481.53 ALONE (attention-free deployed strict < deployed non-strict)
   -> USEFUL ONLY STACKED, cannot clear 500 by itself.
3. lm_head read-reduction / truncation (#398 `dzgbnsrp` / #385 `a30iri8i`): the deployed 21 MB head is a
   LOSSY 16384-row truncation ALREADY in the base; best_loadable_read_shrink_frac=0.0 -> there is NO
   loadable identity-safe shrink below the int4-Marlin full-vocab floor (336 MB). The ONLY further read-
   shrink is MORE lossy truncation (lmhead12k 12288-row), which forfeits full-256k identity and whose
   provability is land #406 (in flight). lm_head's marginal identity-safe tps_unlocked = 0.

COMBINABILITY (the key new deliverable beyond #404)
---------------------------------------------------
The three levers touch THREE ORTHOGONAL subsystems -- body-GEMM quant (cb3), attention kernel (pinned-K),
lm_head rows (truncation). They STACK ADDITIVELY at the served-file level: ONE combined new submission dir
holds one serve.py fork with three DISTINCT, non-colliding additions (setup_cb3_path + pinned-K attn load +
lm_head prune keepset), one manifest (+both kernel deps), one sitecustomize (import both patches). cb3's
cb3-baked checkpoint and lm_head's further-pruned checkpoint COMBINE into ONE bucket (cb3 is the body GEMM,
lm_head is the head -- orthogonal weight regions); pinned-K needs NO checkpoint change. NO two levers
contend for the same in-tree edit (there are NO in-tree edits -- all additive). And the expensive e2e
identity-verify (#319 tier-3) is SHARED: one combined served stack -> one e2e self-referential capture
re-keys ALL changes at once, so the combined verify cost is single-lever + per-lever tier-1 micro, NOT the
sum.

ROLL-UP VERDICT (the cheapest sufficient combination + its single binding contingency)
--------------------------------------------------------------------------------------
Corrected strict base 467.48 (#393), gap_to_500 = 32.52. No single lever clears 500 except cb3, and cb3
only IF #403 returns a PPL-safe-k preserving its lift. Two sharp thresholds on the cb3 lift L:
  * L >= 32.52  -> {cb3} ALONE clears 500           (6 additive files, ~40.8 GPU-min).
  * 18.23 <= L < 32.52 -> {cb3, pinned-K} clears 500 (8 additive files, ~41.8 GPU-min, shared e2e verify).
  * L < 18.23  -> NO combination of all three clears 500 (max stack 467.48 + L + 14.29 + 0 < 500;
                  lm_head adds 0). The current measured FLOOR L=+15.67 -> max stack 497.44 MISSES 500.
So the WHOLE >500 decision rides on a SINGLE binding contingency: kanna #403's PPL-safe-k LIFT MAGNITUDE,
with the binding threshold +18.23 TPS. pinned-K caps 481.53 alone and lm_head's identity-safe shrink is 0,
so neither rescues a sub-18.23 cb3 lift. The combination stays fully additive + reversible regardless.

SCOPE: read-only static analysis + in-repo file enumeration. analysis_only=True, no_hf_job=True,
no_served_file_change=True, official_tps=0. 0 GPU compute. NO build, NO patch, NO compile, NO load, NO
served-file change. Baseline 481.53 UNCHANGED. Public evidence (advisor-branch banked): #404 cb3 surface
(jqhlftrc), #400 pinned-K headroom (o7yhpkej), #398 loadable-lmhead floor (dzgbnsrp), #385 lmhead12k
decomposition (a30iri8i), #393 corrected strict base (0q7ynumg), #394 cb3 PPL-block, #403 PPL-safe-k (in
flight). QTIP 2406.11235 / QuIP# 2402.04396; vLLM 0.22 out-of-tree quant registration API.
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
# Shared baselines (PR #411 body; identical to #404 / #393 for ADDITIVITY across the ledger)
# ---------------------------------------------------------------------------------------- #
DEPLOYED_TPS = 481.53                   # #52 (2x9fm2zx) deployed non-strict #1 -- UNCHANGED
DEPLOYED_PPL = 2.3772
PPL_CAP = 2.42
TARGET_TPS = 500.0
STRICT_BASE_CORRECTED = 467.48          # #393 (0q7ynumg) corrected strict base
STRICT_GAP_TO_500 = round(TARGET_TPS - STRICT_BASE_CORRECTED, 2)   # 32.52

# the 5 canonical change categories a flagged supply-lever deploy can touch
CANONICAL_CATEGORIES = (
    "cuda_cpp_extension",
    "vllm_quant_registration",
    "attn_backend_patch",
    "served_config_selector",
    "serve_py_hook",
)

# ---------------------------------------------------------------------------------------- #
# Lever 1 -- cb3 body-read shrink (#404 jqhlftrc). Carry the row forward EXACTLY.
# ---------------------------------------------------------------------------------------- #
CB3_RUN = "jqhlftrc"
CB3_VERIFY_GPU_MIN = 40.8               # #404 identity_verify_gpu_minutes (bracket [18, 45])
CB3_PRODUCES_NEW_REFERENCE = True       # RHT+VQ dequant -> different argmax -> re-keyed reference
CB3_LIFT_HONEST = 33.0                  # #392 honest M=1
CB3_LIFT_REALISTIC = 38.02             # #392/#388 realistic M=1
CB3_LIFT_FLOOR = 15.67                 # #391 (3udzpoq8) measured-floor at HBM eff 0.256
CB3_SAFE_TO_REQUEST = True             # #404 cb3_build_safe_to_request

# ---------------------------------------------------------------------------------------- #
# Lever 2 -- pinned-K attention rebuild (#400 o7yhpkej). Reproduce its banked facts.
# ---------------------------------------------------------------------------------------- #
PINNEDK_RUN = "o7yhpkej"
PINNEDK_LIFT_CAP = 14.29                # #400 attn_lever_max_tps_gain_deployed (14.2894...)
PINNEDK_CAPS_TPS = 481.53              # #400 attn_free_deployed_strict_tps_roofline (caps at deployed)
PINNEDK_PRODUCES_NEW_REFERENCE = True  # #400 pinnedk_produces_new_reference (multi-split != serial bytes)
PINNEDK_M_INVARIANT_FEASIBLE = True    # #400 pinnedk_m_invariant_byte_exact_feasible (Marlin-grounded)
PINNEDK_CLEARS_500_ALONE = False       # #400 attn_alone_clears_500_realistic
PINNEDK_N_REBUILDS = 1                 # #400 n_distinct_kernel_rebuilds_attn_free
PINNEDK_VERIFY_GPU_MIN = 40.8          # same 3-tier re-key (e2e capture dominates; kernel-agnostic)

# ---------------------------------------------------------------------------------------- #
# Lever 3 -- lm_head read-reduction / truncation (#398 dzgbnsrp / #385 a30iri8i).
# ---------------------------------------------------------------------------------------- #
LMHEAD_RUN = "dzgbnsrp"
LMHEAD_BEST_LOADABLE_SHRINK_FRAC = 0.0      # #398 best_loadable_read_shrink_frac (THE key fact)
LMHEAD_LOADABLE_LEVER_EXISTS = False        # #398 loadable_lmhead_lever_exists
LMHEAD_DEPLOYED_READ_MB = 21.0              # #398 lmhead_read_mb_deployed (lossy 16384-row truncation)
LMHEAD_INT4_FLOOR_MB = 336.07               # #398 loadable_identity_safe_floor_read_mb (channel int4)
LMHEAD_FULLVOCAB_IDENTITY_TPS_COST = 54.10  # #398 |full_vocab_identity_tps_cost_vs_deployed| (roofline)
LMHEAD_TPS_UNLOCKED_IDENTITY_SAFE = 0.0     # loadable identity-safe marginal lift (deployed already banks it)
LMHEAD_LOSSY_PROVABILITY_PR = "#406 (in flight)"   # further-pruning provability

# the 3-tier identity-verify harness (confirmed in-repo paths; reused across all new-reference levers)
HARNESS = {
    "per_gemm_byte_exact_390": "research/validity/strict_ceiling_corrected_rollup/strict_ceiling_corrected_rollup.py",
    "decode_width_e2e_381": "research/validity/decodewidth_e2e_identity/decodewidth_e2e_identity.py",
    "self_ref_reference_319": "scripts/local_validation/gen_greedy_reference.py",
    "self_ref_compare_319": "scripts/local_validation/greedy_gate.py",
    "self_ref_interlock_319": "scripts/validity/greedy_identity_interlock.py",
}
HARNESS_3TIER = ("TIER1 per-GEMM/-config M=1-vs-M=8 byte-identity (#390); TIER2 decode-width e2e identity "
                 "(#381); TIER3 e2e SELF-REFERENTIAL gate (#319: gen_greedy_reference --mode served + "
                 "greedy_gate.compare + greedy_identity_interlock).")
# banked evidence this card reasons from (read-only)
EVIDENCE = {
    "cb3_surface_404": "research/validity/cb3_flagged_build_surface_scope/cb3_flagged_build_surface_scope.py",
    "pinnedk_headroom_400": "research/validity/attention_strict_pin_cost/attention_strict_pin_cost.py",
    "loadable_lmhead_398": "research/validity/loadable_lmhead_readreduction/loadable_lmhead_readreduction.py",
}

# deployed served stack (READ-ONLY references; every lever FORKS these, edits NONE in place)
DEPLOYED_SUBMISSION = "submissions/fa2sw_treeverify_kenyan"
DEPLOYED_SERVE = f"{DEPLOYED_SUBMISSION}/serve.py"
DEPLOYED_MANIFEST = f"{DEPLOYED_SUBMISSION}/manifest.json"
DEPLOYED_SITECUSTOMIZE = f"{DEPLOYED_SUBMISSION}/sitecustomize.py"
DEPLOYED_LMHEAD_PRUNE = f"{DEPLOYED_SUBMISSION}/serve_patch_pck04.py"   # the deployed PCK-04 lm_head prune


# ======================================================================================== #
# Part 1 -- per-lever served-file surface (additive vs in-place), tagged by subsystem + scaffold role
# ======================================================================================== #
def _f(path, category, subsystem, scaffold_role, change, classification="additive",
       touches_deployed=False, touches_in_tree=False, touches_checkpoint=False):
    return {
        "path": path, "category": category, "subsystem": subsystem, "scaffold_role": scaffold_role,
        "change": change, "classification": classification,
        "touches_deployed_file": touches_deployed, "touches_in_tree_vllm": touches_in_tree,
        "touches_checkpoint": touches_checkpoint,
    }


def cb3_surface() -> list[dict[str, Any]]:
    """Reproduce the #404 cb3 6-file surface exactly. subsystem=body_gemm_quant."""
    s = "submissions/cb3_<name>"
    return [
        _f(f"{s}/kernels/cb3_qtip_kernel-*.whl  (prebuilt sm_86 .whl/.so)", "cuda_cpp_extension",
           "body_gemm_quant", "kernel",
           "NEW prebuilt cb3 QTIP/QuIP# CUDA ext (RHT incoherence + L1-resident K=64 dim-2 Gaussian VQ "
           "dequant GEMM); built OUT-OF-HARNESS, uploaded prebuilt."),
        _f(f"{s}/cb3_quant_patch.py", "vllm_quant_registration", "body_gemm_quant", "patch",
           "NEW Cb3Config + Cb3LinearMethod via the ADDITIVE @register_quantization_config(\"cb3\") "
           "decorator (appends to vLLM QUANTIZATION_METHODS at import). NO in-tree vLLM edit."),
        _f(f"{s}/manifest.json  (fork)", "served_config_selector", "body_gemm_quant", "manifest",
           "FORK of the deployed manifest; deltas = +cb3 kernel dep, WEIGHTS_BUCKET -> cb3-baked checkpoint."),
        _f(f"{s}/serve.py  (fork)", "serve_py_hook", "body_gemm_quant", "serve",
           "FORK of the deployed serve stack; delta = +setup_cb3_path() so the child imports cb3_quant_patch. "
           "NO new vLLM source-patch."),
        _f(f"{s}/sitecustomize.py  (fork)", "vllm_quant_registration", "body_gemm_quant", "sitecustomize",
           "FORK of the deployed sitecustomize finder; delta = import cb3_quant_patch."),
        _f("<cb3 bucket>/config.json  (remote, NEW cb3-baked checkpoint)", "served_config_selector",
           "body_gemm_quant", "checkpoint",
           "NEW remote checkpoint config declaring quant_method \"cb3\" (vLLM auto-selects, no CLI flag). "
           "Separate bucket; the deployed int4 checkpoint is untouched.", touches_checkpoint=True),
    ]


def pinnedk_surface() -> list[dict[str, Any]]:
    """Pinned-K attention rebuild: 5 additive files, NO checkpoint (weightless reduction-order change)."""
    s = "submissions/pinnedk_<name>"
    return [
        _f(f"{s}/kernels/fa2_pinnedk-*.whl  (prebuilt sm_86 .whl/.so)", "cuda_cpp_extension",
           "attention_kernel", "kernel",
           "NEW prebuilt rebuilt-FA2 varlen kernel supporting a FIXED 64-CTA num_splits>1 deterministic "
           "split-reduce (the served kernel rejects num_splits>1 with NotImplementedError). Built "
           "OUT-OF-HARNESS, uploaded prebuilt. #400 n_distinct_kernel_rebuilds=1."),
        _f(f"{s}/pinnedk_attn_patch.py", "attn_backend_patch", "attention_kernel", "patch",
           "NEW patch routing the served attention to the rebuilt fixed-64-CTA FA2 (deterministic reduction "
           "order). Imported via sitecustomize. NO in-tree vLLM edit; the shipping FA2 path is untouched."),
        _f(f"{s}/manifest.json  (fork)", "served_config_selector", "attention_kernel", "manifest",
           "FORK of the deployed manifest; delta = +pinned-K attn kernel dep. NO WEIGHTS_BUCKET change "
           "(weights identical; only the attention reduction order changes)."),
        _f(f"{s}/serve.py  (fork)", "serve_py_hook", "attention_kernel", "serve",
           "FORK of the deployed serve stack; delta = +load the pinned-K attn kernel / select the backend."),
        _f(f"{s}/sitecustomize.py  (fork)", "attn_backend_patch", "attention_kernel", "sitecustomize",
           "FORK of the deployed sitecustomize finder; delta = import pinnedk_attn_patch."),
    ]


def lmhead_surface() -> list[dict[str, Any]]:
    """lm_head FURTHER-pruning (the ONLY read-shrink route, since best_loadable_read_shrink_frac=0.0).
    5 additive files. LOSSY (forfeits full-256k identity) + #406-gated; loadable identity-safe shrink = 0."""
    s = "submissions/lmhead_<name>"
    return [
        _f(f"{s}/serve_patch_pck04.py  (fork: smaller keepset)", "serve_py_hook", "lmhead_rows", "patch",
           "FORK of the deployed PCK-04 _prune_lm_head_rows patch with a SMALLER keepset (e.g. 16384->12288 "
           "lmhead12k). Meta-tensor scatter hook (already in the deployed stack). LOSSY: cannot emit the "
           "dropped vocab rows -> byte-exact only vs its OWN pruned checkpoint, NOT full-256k AR."),
        _f(f"{s}/manifest.json  (fork)", "served_config_selector", "lmhead_rows", "manifest",
           "FORK of the deployed manifest; delta = WEIGHTS_BUCKET -> further-pruned checkpoint."),
        _f(f"{s}/serve.py  (fork)", "serve_py_hook", "lmhead_rows", "serve",
           "FORK of the deployed serve stack; imports the (smaller-keepset) prune patch (same mechanism as "
           "the deployed head, smaller keepset)."),
        _f(f"{s}/sitecustomize.py  (fork)", "served_config_selector", "lmhead_rows", "sitecustomize",
           "FORK of the deployed sitecustomize finder; delta = import the prune patch."),
        _f("<pruned bucket>/config.json  (remote, NEW further-pruned checkpoint)", "served_config_selector",
           "lmhead_rows", "checkpoint",
           "NEW remote checkpoint baked with the smaller lm_head keepset. Provability of identity-safety "
           "for the further prune is land #406 (in flight). The deployed checkpoint is untouched.",
           touches_checkpoint=True),
    ]


def split(surface: list[dict[str, Any]]) -> dict[str, int]:
    additive = sum(1 for e in surface if e["classification"] == "additive")
    inplace = sum(1 for e in surface if e["classification"] == "in_place")
    return {"additive": additive, "in_place": inplace, "total": len(surface)}


# ======================================================================================== #
# Part 2 -- identity-verify cost (incremental, per lever). New-reference levers re-key #319.
# ======================================================================================== #
def verify_cost_3tier() -> dict[str, float]:
    """Price the strict greedy-identity re-key (the SAME #319 3-tier harness as #404, the e2e capture
    dominates and is change-agnostic, so every new-reference lever costs ~the same in isolation)."""
    capture_tokens = 128 * 512
    spec_off_tps = 165.0          # M=1 AR reference (#196 non-spec floor regime)
    spec_on_tps = DEPLOYED_TPS
    boot_min = 4.5
    cap_off = (capture_tokens / spec_off_tps) / 60.0
    cap_on = (capture_tokens / spec_on_tps) / 60.0
    tier3 = 2 * (boot_min + cap_off) + 2 * (boot_min + cap_on)   # 2 spec-off + 2 spec-on reloads
    tier1 = 1.0                    # #390 per-GEMM / per-config M=1-vs-M=8 micro
    tier2 = 4.0                    # #381 decode-width e2e
    return {"tier1_min": tier1, "tier2_min": tier2, "tier3_e2e_min": round(tier3, 1),
            "central_min": round(tier1 + tier2 + tier3, 1)}


# ======================================================================================== #
# Part 3 -- per-lever ledger rows (the 5 columns of the PR)
# ======================================================================================== #
def build_rows() -> list[dict[str, Any]]:
    verify = verify_cost_3tier()
    rows: list[dict[str, Any]] = []

    # ---- ROW 1: cb3 (carry #404 forward exactly) ---------------------------------------- #
    cb3 = cb3_surface()
    cb3_split = split(cb3)
    rows.append({
        "lever": "cb3",
        "name": "cb3 body-read shrink",
        "wandb_run": CB3_RUN,
        "subsystem": "body_gemm_quant",
        # col 1: served_files_touched + additive-vs-in-place
        "served_files_touched": len(cb3),
        "additive_vs_inplace": cb3_split,
        "served_file_surface": cb3,
        # col 2: additive / reversible / blast radius
        "is_additive_not_inplace": bool(cb3_split["in_place"] == 0
                                        and not any(e["touches_deployed_file"] for e in cb3)
                                        and not any(e["touches_in_tree_vllm"] for e in cb3)),
        "reversible_by_config_flag": True,
        "blast_radius": "isolated-new-kernel",
        # col 3: identity_verify_gpu_minutes + produces_new_reference + 3-tier harness
        "identity_verify_gpu_minutes": CB3_VERIFY_GPU_MIN,
        "produces_new_reference": CB3_PRODUCES_NEW_REFERENCE,
        "verify_harness_3tier": HARNESS_3TIER,
        "verify_reference_rekey": ("RHT+VQ dequant changes the weights -> different argmax -> #319 re-keys "
                                   "to cb3's OWN M=1 AR decode (never inherited from int4)."),
        # col 4: tps_unlocked + contingency
        "tps_unlocked": {"honest_m1": CB3_LIFT_HONEST, "realistic": CB3_LIFT_REALISTIC,
                         "measured_floor": CB3_LIFT_FLOOR},
        "tps_unlocked_headline": CB3_LIFT_HONEST,
        "contingency": "kanna #403 PPL-safe-k (headline-k PPL-blocked per #394; measured floor +15.67 "
                       "-> 483.15 MISSES 500). cb3 is necessary-not-sufficient.",
        "clears_500_alone": bool(STRICT_BASE_CORRECTED + CB3_LIFT_HONEST >= TARGET_TPS),
        "safe_to_request": CB3_SAFE_TO_REQUEST,
        "touches_checkpoint": True,
    })

    # ---- ROW 2: pinned-K attention rebuild (#400) --------------------------------------- #
    pk = pinnedk_surface()
    pk_split = split(pk)
    rows.append({
        "lever": "pinnedk_attn",
        "name": "pinned-K attention rebuild",
        "wandb_run": PINNEDK_RUN,
        "subsystem": "attention_kernel",
        "served_files_touched": len(pk),
        "additive_vs_inplace": pk_split,
        "served_file_surface": pk,
        "is_additive_not_inplace": bool(pk_split["in_place"] == 0
                                        and not any(e["touches_deployed_file"] for e in pk)
                                        and not any(e["touches_in_tree_vllm"] for e in pk)),
        "reversible_by_config_flag": True,
        "blast_radius": "isolated-new-kernel",
        "identity_verify_gpu_minutes": PINNEDK_VERIFY_GPU_MIN,
        "produces_new_reference": PINNEDK_PRODUCES_NEW_REFERENCE,
        "verify_harness_3tier": HARNESS_3TIER,
        "verify_reference_rekey": ("the fixed 64-CTA multi-split reduction order != the deployed num_splits=1 "
                                   "serial un-pack -> NEW bytes (#400 empirical new-reference probe) -> #319 "
                                   "re-keys to the rebuilt-attn stack's OWN M=1 AR. M-invariant byte-exact "
                                   "FEASIBLE (Marlin-grounded)."),
        "tps_unlocked": {"cap": PINNEDK_LIFT_CAP, "caps_at_tps": PINNEDK_CAPS_TPS},
        "tps_unlocked_headline": PINNEDK_LIFT_CAP,
        "contingency": ("CAPS 481.53 ALONE (attention-free deployed strict < deployed non-strict) -> "
                        "USEFUL ONLY STACKED; cannot clear 500 by itself. The +14.29 itself is MEASURED "
                        "(not contingent)."),
        "clears_500_alone": PINNEDK_CLEARS_500_ALONE,
        "safe_to_request": True,           # additive + reversible + identity-verifiable
        "touches_checkpoint": False,       # weightless reduction-order change
        "n_kernel_rebuilds": PINNEDK_N_REBUILDS,
        "m_invariant_byte_exact_feasible": PINNEDK_M_INVARIANT_FEASIBLE,
    })

    # ---- ROW 3: lm_head read-reduction / truncation (#398/#385) ------------------------- #
    lh = lmhead_surface()
    lh_split = split(lh)
    rows.append({
        "lever": "lmhead",
        "name": "lm_head read-reduction / truncation",
        "wandb_run": LMHEAD_RUN,
        "subsystem": "lmhead_rows",
        "served_files_touched": len(lh),
        "additive_vs_inplace": lh_split,
        "served_file_surface": lh,
        "is_additive_not_inplace": bool(lh_split["in_place"] == 0
                                        and not any(e["touches_deployed_file"] for e in lh)
                                        and not any(e["touches_in_tree_vllm"] for e in lh)),
        "reversible_by_config_flag": True,
        "blast_radius": "isolated-new-checkpoint",
        # the loadable identity-safe shrink path requires NO new reference (it is 0 shrink); only the LOSSY
        # truncation route changes bytes, and a truncation DOES change the argmax support -> self-referential-
        # only reference (NOT full-256k identity-safe).
        "identity_verify_gpu_minutes": 0.0,    # loadable identity-safe path: no change -> no re-key
        "identity_verify_gpu_minutes_lossy_route": verify["central_min"],   # if the lossy prune is taken
        "produces_new_reference": False,        # loadable identity-safe: shrink=0 -> no change -> no new ref
        "produces_new_reference_lossy_route": True,   # truncation changes the argmax support (self-ref-only)
        "verify_harness_3tier": HARNESS_3TIER,
        "verify_reference_rekey": ("best_loadable_read_shrink_frac=0.0 -> NO loadable identity-safe shrink "
                                   "changes anything (no new reference). The ONLY read-shrink is MORE "
                                   "truncation, which changes the argmax SUPPORT -> a self-referential-only "
                                   "(NOT full-256k) reference; its provability is land #406 (in flight)."),
        "tps_unlocked": {"loadable_identity_safe": LMHEAD_TPS_UNLOCKED_IDENTITY_SAFE,
                         "best_loadable_read_shrink_frac": LMHEAD_BEST_LOADABLE_SHRINK_FRAC},
        "tps_unlocked_headline": LMHEAD_TPS_UNLOCKED_IDENTITY_SAFE,
        "contingency": (f"best_loadable_read_shrink_frac=0.0 -> the deployed 21 MB head is a LOSSY 16384-row "
                        f"truncation ALREADY in the 467.48 base; the loadable identity-safe floor is "
                        f"int4-Marlin {LMHEAD_INT4_FLOOR_MB:.0f} MB (full-vocab identity COSTS "
                        f"~{LMHEAD_FULLVOCAB_IDENTITY_TPS_COST:.0f} roofline-TPS). The only further shrink is "
                        f"MORE lossy truncation; provability = land {LMHEAD_LOSSY_PROVABILITY_PR}."),
        "clears_500_alone": False,
        "safe_to_request": False,          # no loadable identity-safe lever; lossy route #406-gated
        "loadable_lmhead_lever_exists": LMHEAD_LOADABLE_LEVER_EXISTS,
        "best_loadable_read_shrink_frac": LMHEAD_BEST_LOADABLE_SHRINK_FRAC,
        "deployed_read_mb": LMHEAD_DEPLOYED_READ_MB,
        "int4_floor_mb": LMHEAD_INT4_FLOOR_MB,
        "touches_checkpoint": True,        # the lossy route bakes a further-pruned checkpoint
    })

    return rows


# ======================================================================================== #
# Part 4 -- combinability (the key new deliverable beyond #404)
# ======================================================================================== #
def combinability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Do the three levers stack additively in ONE submission dir, or do any two contend?
    They touch THREE orthogonal subsystems; they share the scaffold (manifest/serve/sitecustomize) but
    each delta is distinct. The combined verify shares the expensive e2e capture (#319 tier-3)."""
    by = {r["lever"]: r for r in rows}
    subsystems = {r["lever"]: r["subsystem"] for r in rows}
    distinct_subsystems = len(set(subsystems.values()))
    # scaffold files every lever forks (held ONCE in a combined dir, with all deltas merged)
    SCAFFOLD = {"manifest", "serve", "sitecustomize"}

    def distinct_nonscaffold(lever: str) -> list[str]:
        return [e["path"].split("  ")[0] for e in by[lever]["served_file_surface"]
                if e["scaffold_role"] not in SCAFFOLD]

    # checkpoint contention: cb3 (body GEMM) + lm_head (head rows) -> ONE combined checkpoint (orthogonal
    # weight regions); pinned-K -> NO checkpoint. So at most ONE combined bucket, no two contend.
    checkpoint_levers = [r["lever"] for r in rows if r.get("touches_checkpoint")]
    checkpoint_regions = {"cb3": "body_gemm_weights", "lmhead": "lm_head_rows"}
    checkpoint_collision = False   # body GEMM vs lm_head rows are disjoint weight regions

    # any in-tree / shared-file edit that two levers contend for? NONE -- all surfaces are additive forks.
    any_in_place = any(r["additive_vs_inplace"]["in_place"] > 0 for r in rows)
    any_in_tree = any(e["touches_in_tree_vllm"] for r in rows for e in r["served_file_surface"])

    note = (
        "The three levers touch THREE ORTHOGONAL subsystems -- cb3=body-GEMM quant, pinned-K=attention "
        "kernel, lm_head=lm_head rows -- so they STACK ADDITIVELY at the served-file level. ONE combined "
        "submission dir holds a single serve.py fork with three DISTINCT non-colliding additions "
        "(setup_cb3_path + pinned-K attn load + lm_head prune keepset), one manifest (+both kernel deps + "
        "bucket repoint), one sitecustomize (import all patches). cb3's body-quant checkpoint and lm_head's "
        "further-pruned checkpoint COMBINE into ONE bucket (disjoint weight regions: body GEMM vs head "
        "rows); pinned-K needs NO checkpoint. NO two levers contend for the same in-tree edit (there are "
        "NO in-tree edits). The expensive e2e identity-verify (#319 tier-3) is SHARED: one combined served "
        "stack -> one e2e self-referential capture re-keys ALL changes at once, so combined verify = "
        "single-lever + per-lever tier-1 micro, NOT the sum."
    )
    return {
        "levers_stack_additively": bool(not any_in_place and not any_in_tree),
        "distinct_subsystems": distinct_subsystems,
        "subsystem_by_lever": subsystems,
        "any_two_contend_for_same_in_tree_edit": bool(any_in_tree),
        "checkpoint_levers": checkpoint_levers,
        "checkpoint_regions": checkpoint_regions,
        "checkpoint_collision": checkpoint_collision,
        "combined_checkpoint_buckets": 1,     # cb3-body + lm_head-rows in ONE bucket
        "shared_scaffold_files": sorted(SCAFFOLD),
        "cb3_distinct_files": distinct_nonscaffold("cb3"),
        "pinnedk_distinct_files": distinct_nonscaffold("pinnedk_attn"),
        "lmhead_distinct_files": distinct_nonscaffold("lmhead"),
        "shared_e2e_verify": True,
        "combinability_note": note,
    }


def combined_surface_count(rows: list[dict[str, Any]], levers: list[str]) -> dict[str, Any]:
    """Union of served files for a combination, counting the shared scaffold (manifest/serve/sitecustomize)
    ONCE. Returns the total additive file count + the shared/distinct breakdown."""
    by = {r["lever"]: r for r in rows}
    SCAFFOLD = {"manifest", "serve", "sitecustomize"}
    scaffold_present = set()
    distinct = []
    for lv in levers:
        for e in by[lv]["served_file_surface"]:
            if e["scaffold_role"] in SCAFFOLD:
                scaffold_present.add(e["scaffold_role"])
            else:
                distinct.append(f"{lv}:{e['path'].split('  ')[0]}")
    total = len(scaffold_present) + len(distinct)
    return {"levers": levers, "total_additive_files": total,
            "shared_scaffold_files": sorted(scaffold_present), "distinct_files": distinct}


def combined_verify_min(rows: list[dict[str, Any]], levers: list[str]) -> float:
    """Combined incremental identity-verify GPU-min: the expensive #319 e2e capture (tier3) is SHARED
    across all new-reference levers in one combined stack; each new-reference lever adds its tier-1 micro;
    tier-2 decode-width is shared. lm_head's loadable identity-safe path adds 0 (no new reference)."""
    v = verify_cost_3tier()
    by = {r["lever"]: r for r in rows}
    n_new_ref = sum(1 for lv in levers if by[lv].get("produces_new_reference"))
    if n_new_ref == 0:
        return 0.0
    return round(v["tier3_e2e_min"] + v["tier2_min"] + n_new_ref * v["tier1_min"], 1)


# ======================================================================================== #
# Part 5 -- roll-up verdict: cheapest sufficient combination + single binding contingency
# ======================================================================================== #
def rollup(rows: list[dict[str, Any]], comb: dict[str, Any]) -> dict[str, Any]:
    base = STRICT_BASE_CORRECTED
    # thresholds on the cb3 lift L (the only contingent, sufficient-magnitude lever)
    cb3_alone_threshold = round(TARGET_TPS - base, 2)                       # L >= 32.52 -> cb3 alone
    cb3_pinnedk_threshold = round(TARGET_TPS - base - PINNEDK_LIFT_CAP, 2)  # L >= 18.23 -> cb3 + pinned-K
    # current measured-floor cb3 lift (headline-k PPL-blocked per #394; #403 in flight)
    L_floor = CB3_LIFT_FLOOR
    max_stack_floor = round(base + L_floor + PINNEDK_LIFT_CAP + LMHEAD_TPS_UNLOCKED_IDENTITY_SAFE, 2)
    clears_under_current_floor = bool(max_stack_floor >= TARGET_TPS)

    # the cheapest sufficient combination as a function of the (contingent) cb3 lift
    def cheapest_for_lift(L: float) -> dict[str, Any]:
        if base + L >= TARGET_TPS:
            sc = combined_surface_count(rows, ["cb3"])
            return {"set": ["cb3"], "lifted_tps": round(base + L, 2), "clears_500": True,
                    "total_additive_files": sc["total_additive_files"],
                    "total_verify_gpu_min": combined_verify_min(rows, ["cb3"])}
        if base + L + PINNEDK_LIFT_CAP >= TARGET_TPS:
            sc = combined_surface_count(rows, ["cb3", "pinnedk_attn"])
            return {"set": ["cb3", "pinnedk_attn"], "lifted_tps": round(base + L + PINNEDK_LIFT_CAP, 2),
                    "clears_500": True, "total_additive_files": sc["total_additive_files"],
                    "total_verify_gpu_min": combined_verify_min(rows, ["cb3", "pinnedk_attn"])}
        # even cb3 + pinned-K + lm_head cannot clear (lm_head adds 0)
        sc = combined_surface_count(rows, ["cb3", "pinnedk_attn", "lmhead"])
        return {"set": [], "lifted_tps": round(base + L + PINNEDK_LIFT_CAP, 2), "clears_500": False,
                "max_stack_tps": round(base + L + PINNEDK_LIFT_CAP + LMHEAD_TPS_UNLOCKED_IDENTITY_SAFE, 2),
                "total_additive_files": sc["total_additive_files"],
                "total_verify_gpu_min": combined_verify_min(rows, ["cb3", "pinnedk_attn", "lmhead"])}

    ladder = {
        "cb3_realistic_38.02": cheapest_for_lift(CB3_LIFT_REALISTIC),
        "cb3_honest_33.0": cheapest_for_lift(CB3_LIFT_HONEST),
        "cb3_floor_15.67": cheapest_for_lift(CB3_LIFT_FLOOR),
    }
    # the headline cheapest sufficient combination, evaluated at the HONEST cb3 lift (the lift #403 must
    # preserve): {cb3} alone clears at +33 -> 6 files, ~40.8 GPU-min.
    cheapest = cheapest_for_lift(CB3_LIFT_HONEST)

    whole_additive_reversible = bool(
        comb["levers_stack_additively"]
        and all(r["is_additive_not_inplace"] for r in rows)
        and all(r["reversible_by_config_flag"] for r in rows))

    binding = (
        f"kanna #403 PPL-safe-k LIFT MAGNITUDE. The whole >500 decision rides on a SINGLE contingency with a "
        f"sharp threshold L>=+{cb3_pinnedk_threshold} TPS: below it, NO additive combination of the three "
        f"flagged levers clears 500 (pinned-K caps {PINNEDK_CAPS_TPS} alone, lm_head's loadable identity-safe "
        f"shrink is 0). At the CURRENT measured floor L=+{L_floor} the MAX stack "
        f"({base}+{L_floor}+{PINNEDK_LIFT_CAP}+0) = {max_stack_floor} MISSES 500 by "
        f"{round(TARGET_TPS - max_stack_floor, 2)}. If #403 returns L in [{cb3_pinnedk_threshold}, "
        f"{cb3_alone_threshold}) the cheapest sufficient combination is {{cb3, pinned-K}}; if L>="
        f"{cb3_alone_threshold}, {{cb3}} alone."
    )
    paragraph = (
        "UNIFIED FLAGGED-SUPPLY-LEVER DEPLOY-SURFACE LEDGER. Three flagged supply levers, one page. (1) cb3 "
        f"body-read shrink: 6 additive files / 0 in-place, ~{CB3_VERIFY_GPU_MIN} GPU-min verify, new "
        f"reference, +{CB3_LIFT_HONEST:.0f}/+{CB3_LIFT_REALISTIC:.0f} TPS contingent on #403, safe_to_request. "
        f"(2) pinned-K attention rebuild: 5 additive files / 0 in-place, NO checkpoint, new reference, but "
        f"buys at most +{PINNEDK_LIFT_CAP} and CAPS {PINNEDK_CAPS_TPS} ALONE -> useful only STACKED. (3) "
        f"lm_head: best_loadable_read_shrink_frac=0.0 -> the deployed 21 MB head is a lossy 16384-row "
        f"truncation already in the base; no loadable identity-safe shrink exists, only more lossy "
        f"truncation (#406-gated) -> marginal identity-safe lift 0. COMBINABILITY: the three touch three "
        f"orthogonal subsystems and STACK ADDITIVELY in ONE submission dir + ONE combined checkpoint; no two "
        f"contend (no in-tree edits); the expensive #319 e2e verify is SHARED. ROLL-UP: no single lever "
        f"clears 500 except cb3 (contingent). The cheapest sufficient combination is {{cb3}} if #403 returns "
        f"L>=+{cb3_alone_threshold}, else {{cb3, pinned-K}} if L>=+{cb3_pinnedk_threshold}; below "
        f"+{cb3_pinnedk_threshold} NOTHING clears 500 (current floor +{L_floor} -> max stack "
        f"{max_stack_floor}). The whole combination stays additive + reversible. SINGLE BINDING CONTINGENCY: "
        f"#403's PPL-safe-k lift magnitude (threshold +{cb3_pinnedk_threshold} TPS). No GPU compute, no HF "
        f"job, no served-file change were performed for this ledger."
    )
    return {
        "strict_base_corrected": base,
        "strict_gap_to_500": STRICT_GAP_TO_500,
        "cb3_alone_clears_500_threshold_tps": cb3_alone_threshold,
        "cb3_plus_pinnedk_clears_500_threshold_tps": cb3_pinnedk_threshold,
        "current_floor_cb3_lift": L_floor,
        "max_stack_tps_under_current_floor": max_stack_floor,
        "cheapest_sufficient_combination_clears_500_under_current_floor": clears_under_current_floor,
        "cheapest_sufficient_combination": cheapest["set"],
        "cheapest_sufficient_combination_lifted_tps": cheapest["lifted_tps"],
        "cheapest_sufficient_combination_clears_500": cheapest["clears_500"],
        "cheapest_sufficient_combination_total_additive_files": cheapest["total_additive_files"],
        "cheapest_sufficient_combination_total_verify_gpu_min": cheapest["total_verify_gpu_min"],
        "combination_ladder_by_cb3_lift": ladder,
        "whole_combination_additive_and_reversible": whole_additive_reversible,
        "single_binding_contingency": binding,
        "human_ledger_paragraph": paragraph,
    }


# ======================================================================================== #
# repo-fact probes (read-only) + self-test
# ======================================================================================== #
def repo_facts() -> dict[str, bool]:
    def ok(rel: str) -> bool:
        return (REPO_ROOT / rel).is_file()
    facts = {f"deployed::{k}": ok(v) for k, v in {
        "serve": DEPLOYED_SERVE, "manifest": DEPLOYED_MANIFEST,
        "sitecustomize": DEPLOYED_SITECUSTOMIZE, "lmhead_prune": DEPLOYED_LMHEAD_PRUNE,
    }.items()}
    facts.update({f"harness::{k}": ok(v) for k, v in HARNESS.items()})
    facts.update({f"evidence::{k}": ok(v) for k, v in EVIDENCE.items()})
    return facts


def selftest(rows: list[dict[str, Any]], comb: dict[str, Any], roll: dict[str, Any],
             facts: dict[str, bool], flags: dict[str, bool]) -> dict[str, Any]:
    by = {r["lever"]: r for r in rows}
    cb3, pk, lh = by["cb3"], by["pinnedk_attn"], by["lmhead"]
    c: dict[str, bool] = {}

    # (a) cb3 row reproduces #404 EXACTLY ({6,0}, 40.8 GPU-min, safe_to_request=True, new reference)
    c["a_cb3_six_files"] = (cb3["served_files_touched"] == 6)
    c["a_cb3_six_additive"] = (cb3["additive_vs_inplace"]["additive"] == 6)
    c["a_cb3_zero_inplace"] = (cb3["additive_vs_inplace"]["in_place"] == 0)
    c["a_cb3_verify_40_8"] = (abs(cb3["identity_verify_gpu_minutes"] - 40.8) < 1e-9)
    c["a_cb3_new_reference"] = (cb3["produces_new_reference"] is True)
    c["a_cb3_safe_to_request"] = (cb3["safe_to_request"] is True)
    c["a_cb3_additive_not_inplace"] = (cb3["is_additive_not_inplace"] is True)

    # (b) pinned-K row reproduces #400 (new reference + +14.29 cap + caps 481.53 + clears-500-alone False)
    c["b_pinnedk_new_reference"] = (pk["produces_new_reference"] is True)
    c["b_pinnedk_cap_14_29"] = (abs(pk["tps_unlocked"]["cap"] - 14.29) < 1e-9)
    c["b_pinnedk_caps_481_53"] = (abs(pk["tps_unlocked"]["caps_at_tps"] - 481.53) < 1e-9)
    c["b_pinnedk_no_clear_alone"] = (pk["clears_500_alone"] is False)
    c["b_pinnedk_no_checkpoint"] = (pk["touches_checkpoint"] is False)
    c["b_pinnedk_five_files"] = (pk["served_files_touched"] == 5)
    c["b_pinnedk_one_rebuild"] = (pk["n_kernel_rebuilds"] == 1)
    c["b_pinnedk_feasible"] = (pk["m_invariant_byte_exact_feasible"] is True)

    # (c) lm_head row reproduces #398 (best_loadable_read_shrink_frac=0.0, lever_exists False, 0 lift)
    c["c_lmhead_shrink_zero"] = (abs(lh["best_loadable_read_shrink_frac"] - 0.0) < 1e-9)
    c["c_lmhead_lever_absent"] = (lh["loadable_lmhead_lever_exists"] is False)
    c["c_lmhead_tps_zero"] = (abs(lh["tps_unlocked"]["loadable_identity_safe"] - 0.0) < 1e-9)
    c["c_lmhead_deployed_21mb"] = (abs(lh["deployed_read_mb"] - 21.0) < 0.1)
    c["c_lmhead_floor_336mb"] = (abs(lh["int4_floor_mb"] - 336.07) < 0.5)
    c["c_lmhead_loadable_no_new_ref"] = (lh["produces_new_reference"] is False)
    c["c_lmhead_lossy_new_ref"] = (lh["produces_new_reference_lossy_route"] is True)

    # (d) every lever additive (0 in-place, no deployed/in-tree edit) + reversible
    c["d_all_zero_inplace"] = all(r["additive_vs_inplace"]["in_place"] == 0 for r in rows)
    c["d_no_deployed_touched"] = all(not e["touches_deployed_file"]
                                     for r in rows for e in r["served_file_surface"])
    c["d_no_in_tree_touched"] = all(not e["touches_in_tree_vllm"]
                                    for r in rows for e in r["served_file_surface"])
    c["d_all_reversible"] = all(r["reversible_by_config_flag"] for r in rows)
    c["d_categories_canonical"] = all(e["category"] in CANONICAL_CATEGORIES
                                      for r in rows for e in r["served_file_surface"])
    c["d_three_rows"] = (len(rows) == 3)

    # (e) combinability: three orthogonal subsystems, stack additively, no contention, shared e2e
    c["e_three_subsystems"] = (comb["distinct_subsystems"] == 3)
    c["e_stack_additively"] = (comb["levers_stack_additively"] is True)
    c["e_no_in_tree_contention"] = (comb["any_two_contend_for_same_in_tree_edit"] is False)
    c["e_one_combined_bucket"] = (comb["combined_checkpoint_buckets"] == 1)
    c["e_no_checkpoint_collision"] = (comb["checkpoint_collision"] is False)
    c["e_shared_e2e_verify"] = (comb["shared_e2e_verify"] is True)

    # (f) roll-up arithmetic: thresholds, current floor misses, cheapest combo, shared verify < sum
    c["f_cb3_alone_threshold"] = (abs(roll["cb3_alone_clears_500_threshold_tps"] - 32.52) < 0.05)
    c["f_cb3_pinnedk_threshold"] = (abs(roll["cb3_plus_pinnedk_clears_500_threshold_tps"] - 18.23) < 0.05)
    c["f_floor_misses_500"] = (roll["cheapest_sufficient_combination_clears_500_under_current_floor"] is False)
    c["f_max_stack_floor_below_500"] = (roll["max_stack_tps_under_current_floor"] < TARGET_TPS)
    c["f_cheapest_is_cb3_at_honest"] = (roll["cheapest_sufficient_combination"] == ["cb3"])
    c["f_cheapest_clears_at_honest"] = (roll["cheapest_sufficient_combination_clears_500"] is True)
    c["f_cheapest_verify_40_8"] = (abs(roll["cheapest_sufficient_combination_total_verify_gpu_min"] - 40.8) < 0.1)
    c["f_cheapest_six_files"] = (roll["cheapest_sufficient_combination_total_additive_files"] == 6)
    # combined {cb3, pinned-K} shared e2e verify (~41.8) is far below the naive sum (40.8 + 40.8 = 81.6)
    combined_pk = combined_verify_min(rows, ["cb3", "pinnedk_attn"])
    c["f_combined_verify_shared"] = bool(combined_pk < (CB3_VERIFY_GPU_MIN + PINNEDK_VERIFY_GPU_MIN) - 30.0)
    c["f_whole_additive_reversible"] = (roll["whole_combination_additive_and_reversible"] is True)
    c["f_binding_is_403"] = ("#403" in roll["single_binding_contingency"])

    # (g) in-repo facts (read-only existence) + analysis-only hygiene
    for k, v in facts.items():
        c[f"g_{k}"] = v
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
    roll = p["rollup"]
    print("=" * 100)
    print(f"PR #411 lawine -- UNIFIED FLAGGED-SUPPLY-LEVER DEPLOY-SURFACE LEDGER  ({p['created_at']})")
    print(f"  analysis_only={p['analysis_only']}  no_hf_job={p['no_hf_job']}  "
          f"no_served_file_change={p['no_served_file_change']}  official_tps={p['official_tps']}")
    print(f"  strict base {STRICT_BASE_CORRECTED} (#393)  gap_to_500={STRICT_GAP_TO_500}  "
          f"deployed {DEPLOYED_TPS} UNCHANGED")
    print("-" * 100)
    print(f"  {'LEVER':14s} {'files(add/ip)':14s} {'verify_min':10s} {'new_ref':8s} {'tps_unlocked':18s} "
          f"{'clears500_alone':15s} subsystem")
    for r in p["rows"]:
        sp = r["additive_vs_inplace"]
        tu = r["tps_unlocked_headline"]
        print(f"  {r['lever']:14s} {str(r['served_files_touched']) + '(' + str(sp['additive']) + '/' + str(sp['in_place']) + ')':14s} "
              f"{r['identity_verify_gpu_minutes']:<10} {str(r['produces_new_reference']):8s} "
              f"{('+' + format(tu, '.2f')):18s} {str(r['clears_500_alone']):15s} {r['subsystem']}")
    print("-" * 100)
    print("  CONTINGENCIES")
    for r in p["rows"]:
        print(f"    {r['lever']:14s} {r['contingency'][:110]}")
    print("-" * 100)
    print("  COMBINABILITY")
    cb = p["combinability"]
    print(f"    stack_additively={cb['levers_stack_additively']}  distinct_subsystems={cb['distinct_subsystems']}  "
          f"in_tree_contention={cb['any_two_contend_for_same_in_tree_edit']}  "
          f"combined_buckets={cb['combined_checkpoint_buckets']}  shared_e2e_verify={cb['shared_e2e_verify']}")
    print("-" * 100)
    print("  ROLL-UP VERDICT")
    print(f"    cb3-alone clears-500 threshold ....... L >= +{roll['cb3_alone_clears_500_threshold_tps']}")
    print(f"    cb3+pinned-K clears-500 threshold .... L >= +{roll['cb3_plus_pinnedk_clears_500_threshold_tps']}")
    print(f"    current floor cb3 lift ............... +{roll['current_floor_cb3_lift']} -> max stack "
          f"{roll['max_stack_tps_under_current_floor']} (clears_500="
          f"{roll['cheapest_sufficient_combination_clears_500_under_current_floor']})")
    print(f"    cheapest sufficient combination ...... {roll['cheapest_sufficient_combination']} "
          f"(lifted {roll['cheapest_sufficient_combination_lifted_tps']}, "
          f"{roll['cheapest_sufficient_combination_total_additive_files']} files, "
          f"{roll['cheapest_sufficient_combination_total_verify_gpu_min']} GPU-min)")
    print(f"    whole combination additive+reversible  {roll['whole_combination_additive_and_reversible']}")
    print("    combination ladder by cb3 lift:")
    for k, v in roll["combination_ladder_by_cb3_lift"].items():
        print(f"      {k:22s} -> set={v['set']} lifted={v['lifted_tps']} clears_500={v['clears_500']} "
              f"files={v['total_additive_files']} verify={v['total_verify_gpu_min']}")
    print("-" * 100)
    print("  HUMAN LEDGER PARAGRAPH")
    para = roll["human_ledger_paragraph"]
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
    roll = rollup(rows, comb)
    facts = repo_facts()
    st = selftest(rows, comb, roll, facts, flags)
    payload: dict[str, Any] = {
        "agent": "lawine", "pr": 411,
        "kind": "flagged-supply-deploy-surface-ledger",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        # ---- top-level scalars/lists (W&B summary deliverables) ----
        "n_levers": len(rows),
        "levers": [r["lever"] for r in rows],
        "total_served_files_all_levers": sum(r["served_files_touched"] for r in rows),
        "ledger_self_test_passes": bool(st["passes"]),
        "cheapest_sufficient_combination": roll["cheapest_sufficient_combination"],
        "cheapest_sufficient_combination_total_additive_files":
            roll["cheapest_sufficient_combination_total_additive_files"],
        "cheapest_sufficient_combination_total_verify_gpu_min":
            roll["cheapest_sufficient_combination_total_verify_gpu_min"],
        "cheapest_sufficient_combination_clears_500": roll["cheapest_sufficient_combination_clears_500"],
        "clears_500_under_current_floor":
            roll["cheapest_sufficient_combination_clears_500_under_current_floor"],
        "cb3_alone_clears_500_threshold_tps": roll["cb3_alone_clears_500_threshold_tps"],
        "cb3_plus_pinnedk_clears_500_threshold_tps": roll["cb3_plus_pinnedk_clears_500_threshold_tps"],
        "max_stack_tps_under_current_floor": roll["max_stack_tps_under_current_floor"],
        "whole_combination_additive_and_reversible": roll["whole_combination_additive_and_reversible"],
        "levers_stack_additively": comb["levers_stack_additively"],
        "single_binding_contingency": roll["single_binding_contingency"],
        "human_ledger_paragraph": roll["human_ledger_paragraph"],
        # ---- detail blocks ----
        "rows": rows,
        "combinability": comb,
        "rollup": roll,
        "repo_facts": facts,
        "selftest": st,
        "shared_baselines": {
            "deployed_tps": DEPLOYED_TPS, "deployed_ppl": DEPLOYED_PPL, "ppl_cap": PPL_CAP,
            "target_tps": TARGET_TPS, "strict_base_corrected": STRICT_BASE_CORRECTED,
            "strict_gap_to_500": STRICT_GAP_TO_500,
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
        print(f"[ledger] wandb helpers unavailable: {e}")
        return None
    st = payload["selftest"]
    roll = payload["rollup"]
    run = init_wandb_run(
        job_type="analysis-static-scope", agent="lawine",
        name=args.wandb_name, group=args.wandb_group,
        tags=["flagged-supply-deploy-surface-ledger", "cb3", "pinnedk-attn", "lmhead",
              "deploy-surface", "identity-verify", "decision-doc", "pr-411"],
        config={"pr": 411, "kind": "flagged-supply-deploy-surface-ledger",
                "analysis_only": True, "no_hf_job": True, "no_served_file_change": True,
                "official_tps": 0, "strict_base_corrected": STRICT_BASE_CORRECTED,
                "deployed_tps": DEPLOYED_TPS},
    )
    if run is None:
        print("[ledger] wandb disabled (no API key / WANDB_MODE).")
        return None
    flat: dict[str, float] = {
        "ledger/n_levers": float(payload["n_levers"]),
        "ledger/total_served_files_all_levers": float(payload["total_served_files_all_levers"]),
        "ledger/self_test_passes": float(st["passes"]),
        "ledger/self_test_n_checks": float(st["n_checks"]),
        "rollup/cheapest_combo_total_additive_files":
            float(payload["cheapest_sufficient_combination_total_additive_files"]),
        "rollup/cheapest_combo_total_verify_gpu_min":
            float(payload["cheapest_sufficient_combination_total_verify_gpu_min"]),
        "rollup/cheapest_combo_clears_500": float(payload["cheapest_sufficient_combination_clears_500"]),
        "rollup/clears_500_under_current_floor": float(payload["clears_500_under_current_floor"]),
        "rollup/cb3_alone_threshold_tps": float(payload["cb3_alone_clears_500_threshold_tps"]),
        "rollup/cb3_plus_pinnedk_threshold_tps": float(payload["cb3_plus_pinnedk_clears_500_threshold_tps"]),
        "rollup/max_stack_tps_under_current_floor": float(payload["max_stack_tps_under_current_floor"]),
        "rollup/whole_combination_additive_and_reversible":
            float(payload["whole_combination_additive_and_reversible"]),
        "rollup/levers_stack_additively": float(payload["levers_stack_additively"]),
        "ledger/official_tps": float(payload["official_tps"]),
    }
    # per-lever scalars (rich logging)
    for r in payload["rows"]:
        lv = r["lever"]
        flat[f"lever/{lv}/served_files"] = float(r["served_files_touched"])
        flat[f"lever/{lv}/additive"] = float(r["additive_vs_inplace"]["additive"])
        flat[f"lever/{lv}/in_place"] = float(r["additive_vs_inplace"]["in_place"])
        flat[f"lever/{lv}/identity_verify_gpu_min"] = float(r["identity_verify_gpu_minutes"])
        flat[f"lever/{lv}/produces_new_reference"] = float(bool(r["produces_new_reference"]))
        flat[f"lever/{lv}/tps_unlocked_headline"] = float(r["tps_unlocked_headline"])
        flat[f"lever/{lv}/clears_500_alone"] = float(bool(r["clears_500_alone"]))
    run.log({"global_step": 0, **flat})
    log_summary(run, _jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="flagged_supply_deploy_surface_ledger", artifact_type="analysis",
                      data=_jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[ledger] wandb logged {len(flat)} keys (run {rid})")
    return rid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levers", default="cb3,pinnedk_attn,lmhead",
                    help="comma list (informational; the ledger always builds all three rows)")
    ap.add_argument("--self-test", "--selftest", dest="self_test", action="store_true",
                    help="run the analytic self-test and exit nonzero on failure (no wandb)")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="lawine/flagged-supply-deploy-surface-ledger")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="flagged-supply-deploy-surface-ledger")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    flags = {"analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0}
    payload = build_payload(flags)
    print_report(payload)

    out_path = Path(args.out_dir) / "flagged_supply_deploy_surface_ledger_results.json"
    out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"\n[ledger] wrote {out_path}")

    st = payload["selftest"]
    if args.self_test:
        assert st["passes"], f"self-test FAILED ({st['n_checks']} checks)"
        assert st["n_checks"] >= 20, f"need >=20 asserts, have {st['n_checks']}"
        print(f"[ledger] SELF-TEST PASS ({st['n_checks']} checks)")
        print("\nSENPAI-RESULT " + json.dumps({
            "terminal": True, "status": "complete", "pending_arms": False, "wandb_run_ids": [],
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            "ledger_self_test_passes": bool(st["passes"]),
            "primary_metric": {"name": "ledger_self_test_passes", "value": float(st["passes"])},
            "test_metric": {"name": "cheapest_sufficient_combination_total_verify_gpu_min",
                            "value": float(payload["cheapest_sufficient_combination_total_verify_gpu_min"])},
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
        "n_levers": payload["n_levers"],
        "cheapest_sufficient_combination": payload["cheapest_sufficient_combination"],
        "cheapest_sufficient_combination_total_additive_files":
            payload["cheapest_sufficient_combination_total_additive_files"],
        "cheapest_sufficient_combination_total_verify_gpu_min":
            payload["cheapest_sufficient_combination_total_verify_gpu_min"],
        "clears_500_under_current_floor": payload["clears_500_under_current_floor"],
        "whole_combination_additive_and_reversible": payload["whole_combination_additive_and_reversible"],
        "ledger_self_test_passes": bool(st["passes"]),
        "primary_metric": {"name": "ledger_self_test_passes", "value": float(st["passes"])},
        "test_metric": {"name": "cheapest_sufficient_combination_total_verify_gpu_min",
                        "value": float(payload["cheapest_sufficient_combination_total_verify_gpu_min"])},
    }))


if __name__ == "__main__":
    main()
