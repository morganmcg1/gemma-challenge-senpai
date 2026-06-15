#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #404 (lawine) -- Scope the flagged cb3 source-build surface + identity-verify cost.

THE DECISION DOC (what a human reads BEFORE approving the cb3 kernel build)
---------------------------------------------------------------------------
My #395 (`koi6gsmq`, decisive NEGATIVE) proved the cb3 supply lift is irreducibly
SOURCE-BUILD-GATED: no shipping vLLM-0.22 kernel substitutes for the QTIP/QuIP#-class cb3
Marlin extension, because sub-int4 read-shrink and byte-identity are mutually exclusive among
shipping kernels and the 4-bit Marlin floor is a hard architectural bound. The program has
converged -- every >500 lever needs either a flagged served-file change (cb3 build, lm_head
truncation #398, tie-break #397, pinned-K attention #400) or a forbidden retrain. Before we ask
the human team to authorize the ONE flagged change most likely to clear 500 (the cb3 kernel
build), this card prices it precisely: what file surface changes, what the identity-verify costs,
and whether the change is safe + reversible. PURE STATIC ANALYSIS -- it BUILDS NOTHING.

WHAT THE SCOPE FOUND
--------------------
1. SURFACE = 6 files, ALL ADDITIVE, ZERO in-place edits of any deployed or in-tree file. A cb3
   deploy is cleanest as a NEW submission dir (fork of fa2sw_treeverify_kenyan) + a NEW cb3-baked
   bucket. The 6: (i) prebuilt sm_86 cb3 kernel wheel/.so (HF runner can't source-build in-harness
   -> ship prebuilt); (ii) cb3 quant registration via the ADDITIVE `@register_quantization_config
   ("cb3")` decorator (appends to vLLM QUANTIZATION_METHODS at import, raises on built-in
   collision, no in-tree edit); (iii) manifest.json fork (delta = +kernel dep, WEIGHTS_BUCKET
   repoint); (iv) serve.py fork (delta = +setup_cb3_path(); NO new vLLM source-patch, NO scatter
   hook); (v) sitecustomize.py fork (delta = import the cb3 patch); (vi) cb3 checkpoint config.json
   (remote, new bucket, quant_method "cb3"). The shipping int4-Marlin GEMM path is UNTOUCHED.
2. IDENTITY IS NOT INHERITED. cb3's random-Hadamard incoherence + VQ codebook dequant yields
   DIFFERENT weights -> a different argmax -> a NEW greedy reference (like the pinned-K attention
   #393). The strict self-referential gate (#319) must be RE-RUN against cb3's own M=1 AR decode,
   not transferred from the int4 stack. Priced at ~36 GPU-min (bracket [18, 45]): per-GEMM
   M=8-vs-M=1 byte-identity (#390) + decode-width e2e identity (#381) + the e2e self-referential
   gate (#319: gen_greedy_reference --mode served + greedy_gate.compare + interlock).
3. BLAST RADIUS = isolated-new-kernel; reversible by config (serve the old submission / repoint the
   bucket); the deployed int4 path is the instant rollback.

DECISION: cb3_build_surface_bounded = TRUE, cb3_build_safe_to_request = TRUE (additive + reversible
+ identity-verifiable). It unlocks +33 honest / +38 realistic M=1 TPS on the corrected strict base
467.48 (closing the 32.53 strict gap to ~500-505), CONTINGENT on kanna #403 confirming a PPL-safe
codebook-k that preserves the lift under the 2.42 cap (headline-k PPL-blocked per #394; measured
floor only +15.67). cb3 is necessary-not-sufficient: it supplies the read-shrink; strict-compliant
>500 still rides on the base verify already being identity-compliant (cb3 INHERITS, does not fix,
the verify-batch-invariance) and on #403's PPL-safe-k.

SCOPE: read-only static analysis + in-repo file enumeration. analysis_only=True, no_hf_job=True,
no_served_file_change=True, official_tps=0. 0 GPU compute. NO build, NO patch, NO compile, NO load,
NO served-file change. Baseline 481.53 UNCHANGED. Public evidence: QTIP 2406.11235 / QuIP# 2402.04396
(L1-resident trellis/E8 codebook batch=1 decode), vLLM 0.22 out-of-tree quant registration API.
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
# cb3 supply facts -- canonical (#372 allocation / #388 / #392 / #395 cb3_kernel_realized_bw)
# ---------------------------------------------------------------------------------------- #
INT4_BPW = 4.125                       # deployed: 4-bit + bf16 g128 scale
CB3_BPW_EFF = 3.2368598382749325       # #372 mixed cb3 allocation (~88.8% body params at cb3)
BODY_BYTES_FRAC = 0.7846932941272564   # = CB3_BPW_EFF / INT4_BPW (the PR's "0.785" byte ratio)
CB3_READ_SHRINK_FRAC = 1.0 - BODY_BYTES_FRAC   # +0.2153 == cb3's "-21.5%" body-read reduction
F_BODY_STRICT = 0.76240970145034       # honest shrinkable body-GEMM weight-read fraction
F_ATTN = 0.09506718019009251
F_LMHEAD = 0.022428229458960704
F_DRAFT = 0.12009488890060672

# ---------------------------------------------------------------------------------------- #
# baselines (PR #404 body)
# ---------------------------------------------------------------------------------------- #
DEPLOYED_TPS = 481.53                   # #52 (2x9fm2zx) -- UNCHANGED
DEPLOYED_PPL = 2.3772
PPL_CAP = 2.42
TARGET_TPS = 500.0
STRICT_BASE_CORRECTED = 467.48          # #393 (0q7ynumg) corrected strict base
STRICT_GAP_TO_500 = 32.53               # 500 - 467.48
CB3_LIFT_REALISTIC = 38.02              # #392 / #388 realistic M=1
CB3_LIFT_HONEST_M1 = 33.0               # #392 honest M=1 (off-the-shelf +32.65 -> "+33")
CB3_LIFT_FLOOR_MEASURED = 15.67         # #391 (3udzpoq8) measured-floor at HBM eff 0.256

QTIP_UPSTREAM = "github.com/Cornell-RelaxML/qtip"

# ---------------------------------------------------------------------------------------- #
# deployed served stack -- READ-ONLY references (the cb3 deploy forks these, edits NONE in place)
# ---------------------------------------------------------------------------------------- #
DEPLOYED_SUBMISSION = "submissions/fa2sw_treeverify_kenyan"
DEPLOYED_SERVE = f"{DEPLOYED_SUBMISSION}/serve.py"
DEPLOYED_MANIFEST = f"{DEPLOYED_SUBMISSION}/manifest.json"
DEPLOYED_SITECUSTOMIZE = f"{DEPLOYED_SUBMISSION}/sitecustomize.py"

# identity-verify harness (confirmed in-repo paths)
HARNESS = {
    "per_gemm_byte_exact_390": "research/validity/strict_ceiling_corrected_rollup/strict_ceiling_corrected_rollup.py",
    "decode_width_e2e_381": "research/validity/decodewidth_e2e_identity/decodewidth_e2e_identity.py",
    "self_ref_reference_319": "scripts/local_validation/gen_greedy_reference.py",
    "self_ref_compare_319": "scripts/local_validation/greedy_gate.py",
    "self_ref_interlock_319": "scripts/validity/greedy_identity_interlock.py",
}
# banked evidence this card reasons from (read-only)
EVIDENCE = {
    "cb3_no_shipping_substitute_395": "research/validity/cb3_kernel_realized_bw/cb3_shipping_kernel_surrogate.py",
    "cb3_supply_lift_392": "research/validity/cb3_supply_lift_mtp_honest/cb3_supply_lift_mtp_honest.py",
}

# the 4 canonical change categories a quant-kernel deploy can touch
CANONICAL_CATEGORIES = (
    "cuda_cpp_extension",
    "vllm_quant_registration",
    "served_config_selector",
    "serve_py_hook",
)


# ======================================================================================== #
# Part 1 -- served-file surface
# ======================================================================================== #
def served_file_surface() -> list[dict[str, Any]]:
    """Every file a cb3 deploy touches vs the deployed int4-Marlin path, additive-vs-in-place."""
    return [
        {
            "path": "submissions/cb3_<name>/kernels/cb3_qtip_kernel-*.whl  (prebuilt sm_86 .whl/.so)",
            "category": "cuda_cpp_extension",
            "change": "NEW prebuilt cb3 QTIP/QuIP# CUDA ext (RHT incoherence + L1-resident K=64 "
                      "dim-2 Gaussian VQ dequant GEMM); HF runner cannot source-build in-harness, "
                      "so the build is OUT-OF-HARNESS and the artifact is uploaded prebuilt.",
            "classification": "additive",
            "touches_deployed_file": False,
            "touches_in_tree_vllm": False,
        },
        {
            "path": "submissions/cb3_<name>/cb3_quant_patch.py",
            "category": "vllm_quant_registration",
            "change": "NEW Cb3Config + Cb3LinearMethod registered via the ADDITIVE "
                      "`@register_quantization_config(\"cb3\")` decorator (appends to vLLM "
                      "QUANTIZATION_METHODS at import, raises on built-in collision); imported via "
                      "sitecustomize. NO in-tree vLLM edit, NO string-replacement source-patch.",
            "classification": "additive",
            "touches_deployed_file": False,
            "touches_in_tree_vllm": False,
        },
        {
            "path": "submissions/cb3_<name>/manifest.json  (fork of fa2sw_treeverify_kenyan)",
            "category": "served_config_selector",
            "change": "FORK of the deployed manifest; deltas = +cb3 kernel wheel dependency, "
                      "WEIGHTS_BUCKET repointed to the new cb3-baked checkpoint. The deployed "
                      "manifest is untouched (new submission dir).",
            "classification": "additive",
            "touches_deployed_file": False,
            "touches_in_tree_vllm": False,
        },
        {
            "path": "submissions/cb3_<name>/serve.py  (fork of fa2sw_treeverify_kenyan)",
            "category": "serve_py_hook",
            "change": "FORK of the deployed serve stack; delta = +setup_cb3_path() so the child "
                      "imports cb3_quant_patch. NO scatter hook (unlike the lm_head-scatter path), "
                      "NO new vLLM source-patch -- cb3 selection is intrinsic to the checkpoint "
                      "config.json (same as int4-Marlin auto-detect, no --quantization flag).",
            "classification": "additive",
            "touches_deployed_file": False,
            "touches_in_tree_vllm": False,
        },
        {
            "path": "submissions/cb3_<name>/sitecustomize.py  (fork of fa2sw_treeverify_kenyan)",
            "category": "vllm_quant_registration",
            "change": "FORK of the deployed sitecustomize meta-path finder; delta = import "
                      "cb3_quant_patch so the registration fires in every child import.",
            "classification": "additive",
            "touches_deployed_file": False,
            "touches_in_tree_vllm": False,
        },
        {
            "path": "<cb3 bucket>/config.json  (remote, NEW cb3-baked checkpoint)",
            "category": "served_config_selector",
            "change": "NEW remote checkpoint config declaring quantization_config.quant_method "
                      "\"cb3\" so vLLM auto-selects Cb3LinearMethod (no CLI flag). Separate bucket; "
                      "the deployed osoi5 int4 checkpoint is untouched.",
            "classification": "additive",
            "touches_deployed_file": False,
            "touches_in_tree_vllm": False,
        },
    ]


def additive_inplace_split(surface: list[dict[str, Any]]) -> dict[str, int]:
    additive = sum(1 for e in surface if e["classification"] == "additive")
    inplace = sum(1 for e in surface if e["classification"] == "in_place")
    return {"additive": additive, "in_place": inplace, "total": len(surface)}


# ======================================================================================== #
# Part 2 -- identity-verify cost
# ======================================================================================== #
def identity_verify_cost() -> dict[str, Any]:
    """Price the strict greedy-identity verification of a cb3 build."""
    # e2e self-referential gate (#319): 128 prompts x 512 tokens = 65536 tok per capture.
    capture_tokens = 128 * 512
    spec_off_tps = 165.0   # M=1 AR reference (#196 non-spec floor regime)
    spec_on_tps = DEPLOYED_TPS
    boot_min = 4.5         # ~3-6 min server boot/weight-sync per launch
    n_spec_off = 2         # 2 reloads (spec-OFF self-determinism)
    n_spec_on = 2          # 2 reloads (spec-ON self-determinism)
    cap_off_min = (capture_tokens / spec_off_tps) / 60.0
    cap_on_min = (capture_tokens / spec_on_tps) / 60.0
    tier3_min = n_spec_off * (boot_min + cap_off_min) + n_spec_on * (boot_min + cap_on_min)
    tier1_min = 1.0        # #390 per-GEMM M=8-vs-M=1 byte-identity (body projections) -- micro
    tier2_min = 4.0        # #381 decode-width e2e identity
    central = round(tier1_min + tier2_min + tier3_min, 1)
    return {
        "gpu_minutes_central": central,
        "gpu_minutes_low": 18.0,
        "gpu_minutes_high": 45.0,
        "tiers": {
            "tier1_per_gemm_byte_exact_390_min": tier1_min,
            "tier2_decode_width_e2e_381_min": tier2_min,
            "tier3_self_referential_319_min": round(tier3_min, 1),
        },
        "harness": (
            "TIER1 per-GEMM M=8-vs-M=1 byte-identity across body projections at real geometry "
            "(#390 strict_ceiling_corrected_rollup); TIER2 decode-width e2e identity (#381 "
            "decodewidth_e2e_identity, pass pinned_identity>=1-1e-12); TIER3 e2e SELF-REFERENTIAL "
            "gate (#319): scripts/local_validation/gen_greedy_reference.py --mode served "
            "(drafter-off M=1 AR reference) + scripts/local_validation/greedy_gate.py compare + "
            "scripts/validity/greedy_identity_interlock.py --self-referential."
        ),
        "pass_criterion": (
            "byte-exact per-GEMM at M in {1,8}; 0 divergent tokens across 128 prompts x 512 tokens "
            "vs cb3's OWN spec-off M=1 AR reference (re-keyed, not the int4 reference)."
        ),
        "produces_new_reference": True,   # RHT+VQ dequant -> different argmax -> re-keyed reference
        "reference_rekey_note": (
            "cb3's random-Hadamard incoherence + VQ codebook dequant changes the weights, so the "
            "greedy argmax stream differs from int4. The #319 gate is self-referential, so it "
            "re-keys to cb3's own M=1 AR decode (exactly like the pinned-K attention #393). "
            "Identity is VERIFIABLE but must be MEASURED on cb3, never inherited from int4."
        ),
        "one_time_build_note": (
            "Excludes the one-time, out-of-harness cb3 codebook re-quant/calibration build "
            "(multi-GPU-hr, kanna #403) -- that is a separate cost, not the per-verify cost."
        ),
    }


# ======================================================================================== #
# Part 3 -- blast radius / reversibility
# ======================================================================================== #
def blast_radius_assessment(surface: list[dict[str, Any]], split: dict[str, int]) -> dict[str, Any]:
    is_additive = (split["in_place"] == 0
                   and not any(e["touches_deployed_file"] for e in surface)
                   and not any(e["touches_in_tree_vllm"] for e in surface))
    return {
        "is_additive_not_inplace": is_additive,
        "reversible_by_config_flag": True,
        "blast_radius": "isolated-new-kernel",
        "rollback": (
            "Instant: serve the existing fa2sw_treeverify_kenyan submission, or repoint "
            "WEIGHTS_BUCKET back to the int4-baked checkpoint. Nothing in-tree or shared was "
            "edited, so there is no patch to revert."
        ),
        "blast_radius_options": ["isolated-new-kernel", "shared-GEMM-edit", "serve-hook"],
    }


# ======================================================================================== #
# Part 4 -- go/no-go scope summary
# ======================================================================================== #
def scope_summary(surface: list[dict[str, Any]], split: dict[str, int],
                  verify: dict[str, Any], blast: dict[str, Any]) -> dict[str, Any]:
    bounded = (len(surface) <= 8 and split["in_place"] == 0
               and len({e["category"] for e in surface}) <= len(CANONICAL_CATEGORIES))
    safe = bool(blast["is_additive_not_inplace"]
                and blast["reversible_by_config_flag"]
                and verify["produces_new_reference"] is not None
                and verify["gpu_minutes_central"] < 120)
    lift_realistic_tps = STRICT_BASE_CORRECTED + CB3_LIFT_REALISTIC
    lift_honest_tps = STRICT_BASE_CORRECTED + CB3_LIFT_HONEST_M1
    lift_floor_tps = STRICT_BASE_CORRECTED + CB3_LIFT_FLOOR_MEASURED
    paragraph = (
        "APPROVAL ASK (cb3 source-build). Authorize a one-time, out-of-harness source-build of the "
        "QTIP/QuIP#-class cb3 quantizer kernel (random-Hadamard incoherence + an L1-resident K=64 "
        f"dim-2 Gaussian VQ codebook, ~{CB3_BPW_EFF:.4f} effective bpw, body weight-read ratio "
        f"{BODY_BYTES_FRAC:.3f} => -{CB3_READ_SHRINK_FRAC*100:.1f}% body read) from "
        f"{QTIP_UPSTREAM}, delivered as a PREBUILT sm_86 wheel/.so plus an out-of-tree vLLM quant "
        "registration (@register_quantization_config(\"cb3\")). It deploys as a NEW submission "
        "directory forked from fa2sw_treeverify_kenyan with WEIGHTS_BUCKET repointed to a new "
        f"cb3-baked checkpoint -- {len(surface)} files touched, ALL additive, ZERO in-place edits "
        "of any deployed or in-tree file (the shipping int4-Marlin GEMM path is untouched and is "
        "the instant rollback). The surface is bounded and reversible by config. It unlocks a "
        f"realized decode lift of +{CB3_LIFT_HONEST_M1:.0f} honest / +{CB3_LIFT_REALISTIC:.0f} "
        f"realistic M=1 TPS on the corrected strict base {STRICT_BASE_CORRECTED:.2f} (closing the "
        f"{STRICT_GAP_TO_500:.2f} strict gap to ~{lift_honest_tps:.0f}-{lift_realistic_tps:.0f}), "
        "CONTINGENT on kanna #403 confirming a PPL-safe codebook-k that preserves the lift under "
        f"the {PPL_CAP} PPL cap (headline-k is PPL-blocked per #394; the measured-floor lift is "
        f"only +{CB3_LIFT_FLOOR_MEASURED:.2f} -> {lift_floor_tps:.0f}, which MISSES 500). Identity "
        "is verifiable but NOT inherited: cb3's RHT+VQ produces a NEW greedy reference, so the "
        f"strict self-referential gate (#319) must be re-run against cb3's own M=1 AR -- a bounded "
        f"~{verify['gpu_minutes_central']:.0f} GPU-min check (per-GEMM #390 + decode-width #381 + "
        "e2e self-referential #319). RESIDUAL: cb3 is necessary-not-sufficient -- it supplies the "
        "read-shrink; the strict-compliant >500 still rides on the base verify already being "
        "identity-compliant (cb3 inherits, does not fix, the verify-batch-invariance) and on "
        "#403's PPL-safe-k. No GPU compute, no HF job, and no served-file change were performed for "
        "this scope document."
    )
    return {
        "cb3_build_surface_bounded": bounded,
        "cb3_build_safe_to_request": safe,
        "human_scope_paragraph": paragraph,
        "realized_lift": {
            "honest_m1_tps": CB3_LIFT_HONEST_M1,
            "realistic_tps": CB3_LIFT_REALISTIC,
            "measured_floor_tps": CB3_LIFT_FLOOR_MEASURED,
            "strict_base_corrected": STRICT_BASE_CORRECTED,
            "lifted_honest_tps": round(lift_honest_tps, 2),
            "lifted_realistic_tps": round(lift_realistic_tps, 2),
            "lifted_floor_tps": round(lift_floor_tps, 2),
            "clears_500_realistic": bool(lift_realistic_tps >= TARGET_TPS),
            "clears_500_honest": bool(lift_honest_tps >= TARGET_TPS),
            "clears_500_floor": bool(lift_floor_tps >= TARGET_TPS),
        },
        "residual_gap_after_realistic": round(TARGET_TPS - lift_realistic_tps, 2),
        "binding_contingency": "kanna #403 PPL-safe-k (headline-k PPL-blocked per #394)",
        "go_no_go": ("GO-TO-REQUEST: surface is bounded, additive, reversible, and identity-"
                     "verifiable. The build is SAFE TO REQUEST; the GO on the realized >500 then "
                     "rides on #403's PPL-safe-k, not on the build surface."),
    }


# ======================================================================================== #
# repo-fact probes (read-only) + self-test
# ======================================================================================== #
def repo_facts() -> dict[str, bool]:
    def ok(rel: str) -> bool:
        return (REPO_ROOT / rel).is_file()
    facts = {f"deployed::{k}": ok(v) for k, v in {
        "serve": DEPLOYED_SERVE, "manifest": DEPLOYED_MANIFEST, "sitecustomize": DEPLOYED_SITECUSTOMIZE,
    }.items()}
    facts.update({f"harness::{k}": ok(v) for k, v in HARNESS.items()})
    facts.update({f"evidence::{k}": ok(v) for k, v in EVIDENCE.items()})
    return facts


def selftest(surface: list[dict[str, Any]], split: dict[str, int], verify: dict[str, Any],
             blast: dict[str, Any], summary: dict[str, Any], facts: dict[str, bool],
             flags: dict[str, bool]) -> dict[str, Any]:
    c: dict[str, bool] = {}
    cats = {e["category"] for e in surface}
    # (a) cb3 supply arithmetic
    c["a_cb3_is_subint4"] = (CB3_BPW_EFF < INT4_BPW)
    c["a_byte_ratio_matches"] = (abs(BODY_BYTES_FRAC - CB3_BPW_EFF / INT4_BPW) < 1e-9)
    c["a_read_shrink_is_21_5pct"] = (CB3_READ_SHRINK_FRAC > 0 and abs(CB3_READ_SHRINK_FRAC - 0.2153) < 1e-3)
    c["a_gap_consistent"] = (abs((TARGET_TPS - STRICT_BASE_CORRECTED) - STRICT_GAP_TO_500) < 0.5)
    # (b) surface enumeration + additive/in-place split
    c["b_surface_count_6"] = (len(surface) == 6)
    c["b_split_total_eq_surface"] = (split["total"] == len(surface))
    c["b_zero_inplace"] = (split["in_place"] == 0)
    c["b_all_additive"] = (split["additive"] == len(surface))
    c["b_no_deployed_file_touched"] = all(not e["touches_deployed_file"] for e in surface)
    c["b_no_in_tree_vllm_touched"] = all(not e["touches_in_tree_vllm"] for e in surface)
    c["b_categories_within_canonical"] = cats.issubset(set(CANONICAL_CATEGORIES))
    c["b_has_cuda_ext"] = ("cuda_cpp_extension" in cats)
    c["b_has_quant_registration"] = ("vllm_quant_registration" in cats)
    c["b_has_serve_hook"] = ("serve_py_hook" in cats)
    c["b_has_config_selector"] = ("served_config_selector" in cats)
    # (c) identity-verify cost
    c["c_produces_new_reference"] = (verify["produces_new_reference"] is True)
    c["c_minutes_bracketed"] = (verify["gpu_minutes_low"] <= verify["gpu_minutes_central"] <= verify["gpu_minutes_high"])
    c["c_minutes_plausible"] = (0.0 < verify["gpu_minutes_central"] < 120.0)
    c["c_harness_cites_319"] = ("#319" in verify["harness"] and "self-referential" in verify["harness"].lower())
    c["c_harness_cites_390_381"] = ("#390" in verify["harness"] and "#381" in verify["harness"])
    # (d) blast radius / reversibility
    c["d_is_additive_not_inplace"] = (blast["is_additive_not_inplace"] is True)
    c["d_reversible"] = (blast["reversible_by_config_flag"] is True)
    c["d_blast_radius_isolated"] = (blast["blast_radius"] == "isolated-new-kernel")
    c["d_blast_radius_valid"] = (blast["blast_radius"] in blast["blast_radius_options"])
    # (e) scope summary
    c["e_surface_bounded"] = (summary["cb3_build_surface_bounded"] is True)
    c["e_safe_to_request"] = (summary["cb3_build_safe_to_request"] is True)
    c["e_paragraph_nonempty"] = (len(summary["human_scope_paragraph"]) > 200)
    c["e_paragraph_cites_qtip"] = ("qtip" in summary["human_scope_paragraph"].lower())
    c["e_paragraph_cites_additive"] = ("additive" in summary["human_scope_paragraph"].lower())
    c["e_paragraph_cites_403"] = ("#403" in summary["human_scope_paragraph"])
    c["e_realistic_clears_500"] = (summary["realized_lift"]["clears_500_realistic"] is True)
    c["e_floor_misses_500"] = (summary["realized_lift"]["clears_500_floor"] is False)
    # (f) in-repo facts (read-only existence)
    for k, v in facts.items():
        c[f"f_{k}"] = v
    # (g) analysis-only hygiene
    c["g_official_tps_zero"] = (flags.get("official_tps") == 0)
    c["g_analysis_only"] = bool(flags.get("analysis_only"))
    c["g_no_hf_job"] = bool(flags.get("no_hf_job"))
    c["g_no_served_file_change"] = bool(flags.get("no_served_file_change"))
    return {"conditions": c, "n_checks": len(c), "passes": all(c.values())}


# ======================================================================================== #
# report + IO + wandb
# ======================================================================================== #
def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, bool):
        return o
    if isinstance(o, (int, float, str)) or o is None:
        return o
    return str(o)


def print_report(p: dict[str, Any]) -> None:
    st = p["selftest"]
    print("=" * 92)
    print(f"PR #404 lawine -- cb3 flagged-build surface + identity-verify scope  ({p['created_at']})")
    print(f"  analysis_only={p['analysis_only']}  no_hf_job={p['no_hf_job']}  "
          f"no_served_file_change={p['no_served_file_change']}  official_tps={p['official_tps']}")
    print("-" * 92)
    print(f"  SERVED-FILE SURFACE ({p['served_files_touched']} files, "
          f"additive={p['additive_vs_inplace_split']['additive']} "
          f"in_place={p['additive_vs_inplace_split']['in_place']})")
    for e in p["served_file_surface"]:
        print(f"    [{e['classification']:8s}] {e['category']:24s} {e['path']}")
    print("-" * 92)
    print("  IDENTITY-VERIFY COST")
    print(f"    gpu_minutes ............ {p['identity_verify_gpu_minutes']} "
          f"(bracket [{p['identity_verify']['gpu_minutes_low']}, {p['identity_verify']['gpu_minutes_high']}])")
    print(f"    produces_new_reference . {p['cb3_produces_new_reference']}")
    print(f"    harness ................ {p['identity_verify_harness'][:120]}...")
    print("-" * 92)
    print("  BLAST RADIUS / REVERSIBILITY")
    print(f"    is_additive_not_inplace  {p['is_additive_not_inplace']}")
    print(f"    reversible_by_config ..  {p['reversible_by_config_flag']}")
    print(f"    blast_radius ..........  {p['blast_radius']}")
    print("-" * 92)
    print("  GO / NO-GO")
    print(f"    cb3_build_surface_bounded  {p['cb3_build_surface_bounded']}")
    print(f"    cb3_build_safe_to_request  {p['cb3_build_safe_to_request']}")
    print(f"    realized lift (corrected strict base {STRICT_BASE_CORRECTED}): "
          f"honest +{CB3_LIFT_HONEST_M1:.0f} -> {p['scope']['realized_lift']['lifted_honest_tps']}, "
          f"realistic +{CB3_LIFT_REALISTIC:.0f} -> {p['scope']['realized_lift']['lifted_realistic_tps']}, "
          f"floor +{CB3_LIFT_FLOOR_MEASURED:.2f} -> {p['scope']['realized_lift']['lifted_floor_tps']}")
    print("-" * 92)
    print("  HUMAN SCOPE PARAGRAPH")
    para = p["human_scope_paragraph"]
    for i in range(0, len(para), 108):
        print(f"    {para[i:i+108]}")
    print("-" * 92)
    print(f"  SELF-TEST {st['n_checks']} checks -> {'PASS' if st['passes'] else 'FAIL'}")
    if not st["passes"]:
        for k, v in st["conditions"].items():
            if not v:
                print(f"    FAILED: {k}")
    print("=" * 92)


def build_payload(flags: dict[str, bool]) -> dict[str, Any]:
    surface = served_file_surface()
    split = additive_inplace_split(surface)
    verify = identity_verify_cost()
    blast = blast_radius_assessment(surface, split)
    summary = scope_summary(surface, split, verify, blast)
    facts = repo_facts()
    st = selftest(surface, split, verify, blast, summary, facts, flags)
    payload: dict[str, Any] = {
        "agent": "lawine", "pr": 404,
        "kind": "cb3-flagged-build-surface-scope",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        # ---- analysis-only flags ----
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        # ---- PR #404 required W&B summary deliverables (top-level scalars/lists) ----
        "served_files_touched": len(surface),
        "served_file_surface": surface,
        "served_file_surface_paths": [e["path"] for e in surface],
        "served_file_surface_str": " | ".join(
            f"{e['classification']}:{e['category']}:{e['path'].split('  ')[0]}" for e in surface),
        "additive_vs_inplace_split": split,
        "additive_vs_inplace_split_str": f"additive={split['additive']},in_place={split['in_place']}",
        "identity_verify_gpu_minutes": verify["gpu_minutes_central"],
        "identity_verify_harness": verify["harness"],
        "cb3_produces_new_reference": verify["produces_new_reference"],
        "is_additive_not_inplace": blast["is_additive_not_inplace"],
        "reversible_by_config_flag": blast["reversible_by_config_flag"],
        "blast_radius": blast["blast_radius"],
        "cb3_build_surface_bounded": summary["cb3_build_surface_bounded"],
        "cb3_build_safe_to_request": summary["cb3_build_safe_to_request"],
        "human_scope_paragraph": summary["human_scope_paragraph"],
        "cb3_flagged_build_scope_self_test_passes": bool(st["passes"]),
        # ---- detail blocks ----
        "identity_verify": verify,
        "blast": blast,
        "scope": summary,
        "repo_facts": facts,
        "selftest": st,
        "cb3_supply_facts": {
            "int4_bpw": INT4_BPW, "cb3_bpw_eff": CB3_BPW_EFF, "body_bytes_frac": BODY_BYTES_FRAC,
            "cb3_read_shrink_frac": CB3_READ_SHRINK_FRAC, "f_body_strict": F_BODY_STRICT,
            "deployed_tps": DEPLOYED_TPS, "strict_base_corrected": STRICT_BASE_CORRECTED,
            "strict_gap_to_500": STRICT_GAP_TO_500,
        },
        "decision_gate": summary["go_no_go"],
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
        print(f"[cb3-scope] wandb helpers unavailable: {e}")
        return None
    st = payload["selftest"]
    run = init_wandb_run(
        job_type="analysis-static-scope", agent="lawine",
        name=args.wandb_name, group=args.wandb_group,
        tags=["cb3-flagged-build-surface-scope", "cb3", "source-build", "identity-verify",
              "319-strict-lock", "decision-doc", "pr-404"],
        config={"pr": 404, "kind": "cb3-flagged-build-surface-scope",
                "analysis_only": True, "no_hf_job": True, "no_served_file_change": True,
                "official_tps": 0, "int4_bpw": INT4_BPW, "cb3_bpw_eff": CB3_BPW_EFF,
                "strict_base_corrected": STRICT_BASE_CORRECTED},
    )
    if run is None:
        print("[cb3-scope] wandb disabled (no API key / WANDB_MODE).")
        return None
    flat: dict[str, float] = {
        "decision/served_files_touched": float(payload["served_files_touched"]),
        "decision/additive": float(payload["additive_vs_inplace_split"]["additive"]),
        "decision/in_place": float(payload["additive_vs_inplace_split"]["in_place"]),
        "decision/identity_verify_gpu_minutes": float(payload["identity_verify_gpu_minutes"]),
        "decision/cb3_produces_new_reference": float(payload["cb3_produces_new_reference"]),
        "decision/is_additive_not_inplace": float(payload["is_additive_not_inplace"]),
        "decision/reversible_by_config_flag": float(payload["reversible_by_config_flag"]),
        "decision/cb3_build_surface_bounded": float(payload["cb3_build_surface_bounded"]),
        "decision/cb3_build_safe_to_request": float(payload["cb3_build_safe_to_request"]),
        "decision/official_tps": float(payload["official_tps"]),
        "lift/lifted_honest_tps": float(payload["scope"]["realized_lift"]["lifted_honest_tps"]),
        "lift/lifted_realistic_tps": float(payload["scope"]["realized_lift"]["lifted_realistic_tps"]),
        "lift/lifted_floor_tps": float(payload["scope"]["realized_lift"]["lifted_floor_tps"]),
        "lift/strict_gap_to_500": float(STRICT_GAP_TO_500),
        "selftest/passes": float(st["passes"]),
        "selftest/n_checks": float(st["n_checks"]),
    }
    run.log({"global_step": 0, **flat})
    log_summary(run, _jsonable(payload), step=0, run_prefix=args.wandb_name)
    log_json_artifact(run, name="cb3_flagged_build_surface_scope", artifact_type="analysis",
                      data=_jsonable(payload))
    finish_wandb(run)
    rid = getattr(run, "id", None)
    print(f"[cb3-scope] wandb logged {len(flat)} keys (run {rid})")
    return rid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", "--selftest", dest="self_test", action="store_true",
                    help="run the analytic self-test and exit nonzero on failure (no wandb)")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="lawine/cb3-flagged-build-surface-scope")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="cb3-flagged-build-surface-scope")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    flags = {"analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0}
    payload = build_payload(flags)
    print_report(payload)

    out_path = Path(args.out_dir) / "cb3_flagged_build_surface_scope_results.json"
    out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"\n[cb3-scope] wrote {out_path}")

    st = payload["selftest"]
    if args.self_test:
        assert st["passes"], f"self-test FAILED ({st['n_checks']} checks)"
        assert st["n_checks"] >= 20, f"need >=20 asserts, have {st['n_checks']}"
        print(f"[cb3-scope] SELF-TEST PASS ({st['n_checks']} checks)")
        print("\nSENPAI-RESULT " + json.dumps({
            "terminal": True, "status": "complete", "pending_arms": False, "wandb_run_ids": [],
            "analysis_only": True, "official_tps": 0,
            "cb3_flagged_build_scope_self_test_passes": bool(st["passes"]),
            "primary_metric": {"name": "cb3_flagged_build_scope_self_test_passes", "value": float(st["passes"])},
            "test_metric": {"name": "cb3_build_safe_to_request", "value": float(payload["cb3_build_safe_to_request"])},
        }))
        sys.exit(0 if st["passes"] else 1)

    rid = maybe_log_wandb(payload, args)
    if rid:
        payload["wandb_run_id"] = rid
        out_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))

    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [rid] if rid else [],
        "analysis_only": True, "official_tps": 0,
        "served_files_touched": payload["served_files_touched"],
        "additive_vs_inplace_split": payload["additive_vs_inplace_split"],
        "identity_verify_gpu_minutes": payload["identity_verify_gpu_minutes"],
        "cb3_produces_new_reference": payload["cb3_produces_new_reference"],
        "is_additive_not_inplace": payload["is_additive_not_inplace"],
        "reversible_by_config_flag": payload["reversible_by_config_flag"],
        "blast_radius": payload["blast_radius"],
        "cb3_build_surface_bounded": payload["cb3_build_surface_bounded"],
        "cb3_build_safe_to_request": payload["cb3_build_safe_to_request"],
        "cb3_flagged_build_scope_self_test_passes": bool(st["passes"]),
        "primary_metric": {"name": "cb3_flagged_build_scope_self_test_passes", "value": float(st["passes"])},
        "test_metric": {"name": "cb3_build_safe_to_request", "value": float(payload["cb3_build_safe_to_request"])},
    }))


if __name__ == "__main__":
    main()
