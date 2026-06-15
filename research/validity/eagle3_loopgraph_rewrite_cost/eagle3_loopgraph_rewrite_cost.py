#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Price the served-file loopgraph rewrite EAGLE-3 requires (PR #312).

THE QUESTION
------------
wirbel #307 (run 88eh8twv, MERGED) found the EAGLE-3 served swap is NOT a config
drop-in: 3 served-file touchpoints (T5/T6/T7) on the deployed
`submissions/fa2sw_precache_kenyan/sitecustomize.py` go INERT under method=='eagle3'
because the onegraph loopgraph that PRODUCES 481.53 is MTP-specific (keyed to
Gemma4Proposer / gemma4_mtp.get_top_tokens; its "width-1 is exact" correctness rests
on the MTP drafter being Q-only / KV-shared -- FALSE for EAGLE-3's own-KV Llama draft
layer). The GO/NO-GO packet was MISSING this DEPLOYMENT-COST axis. This card prices it.

WHAT THIS IS (and is NOT)
-------------------------
READ-ONLY STATIC SCOPING from the real `sitecustomize.py` source. NO served-file edit,
NO build, NO boot test, NO HF Job, NO submission change. 0 GPU, 0 TPS. BASELINE 481.53
unchanged. This is an ENGINEERING-COMPLEXITY estimate + a BANKED-decomposition floor,
not an implementation and not a measured fallback TPS.

METHOD
------
1. LOCATE each of T5/T6/T7 in `sitecustomize.py` by re-reading a (file, substring)
   anchor from the real repo -- the touchpoint inventory is FALSIFIABLE, not asserted.
2. For each, encode the concrete EAGLE-3 delta and classify its REWRITE complexity
   into exactly one of {mechanical-config, moderate-rewrite, correctness-rederivation},
   then list the re-validation gates it reopens (subset of
   {greedy-identity, PPL<=2.42, boot-500, TPS}).
3. EAGLE-3-on-EAGER fallback floor: if the loopgraph rewrite is DEFERRED, the served
   path is the stock un-onegraphed eager EagleProposer. Estimate its TPS floor from the
   BANKED onegraph->eager drafter-chain decomposition (the loopgraph's contribution to
   481.53 vs the eager baseline), re-read from repo artifacts. Bound it; flag ESTIMATE.
4. Bundle the Issue #272 directive: T6/T7 reopen the unguarded `sitecustomize.py`
   (0 `_guard_included_router` matches), so port the sibling's guard as a co-edit.
5. Self-test: PRIMARY bool loopgraph_rewrite_cost_self_test_passes; TEST metrics
   eager_fallback_tps_floor (float, ESTIMATE) + rewrite_reopens_greedy_identity (bool).

DELIVERABLE
-----------
Per-touchpoint complexity + reopened gates, the eager-fallback TPS floor (banked
decomposition, bounded estimate), and the #272 co-edit flag -- the missing
"deployment cost" axis on the EAGLE-3 GO/NO-GO packet. 0 TPS.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

# Repo root: research/validity/eagle3_loopgraph_rewrite_cost/<this> -> 3 parents up.
ROOT = Path(__file__).resolve().parents[3]

PRECACHE = "submissions/fa2sw_precache_kenyan"
TREEVERIFY = "submissions/fa2sw_treeverify_kenyan"
SITE = f"{PRECACHE}/sitecustomize.py"

OFFICIAL_BASELINE = 481.53          # PR #52, served public #1 (PPL 2.3772, 128/128).
PRIVATE_VERIFIED = 460.85           # organizer re-run, Delta 4.3% <= 5%.
EAGLE3_TARGET_LAYERS = (2, 21, 39)  # default [low, mid, high] for E4B 42-layer body.
EAGLE3_FUSED_IN = 7680              # arch_notes PR #16: fc[7680->2560] (3 x 2560 aux cat).
MTP_HIDDEN = 2560                   # the single [1,2560] static hidden buffer onegraph assumes.

# Rewrite-complexity taxonomy (mutually exclusive, exhaustive) for the served-file
# touchpoints. NOTE: #307 already classified T5/T6/T7 as `served-file-change`; this
# card SUB-classifies their ENGINEERING complexity (the rewrite/re-validation surface).
MECHANICAL = "mechanical-config"            # re-point a key / flag, no logic change.
MODERATE = "moderate-rewrite"               # re-target + rewrite a body against a new API.
REDERIVE = "correctness-rederivation"       # the correctness invariant itself is false.
COMPLEXITY = (MECHANICAL, MODERATE, REDERIVE)

# Re-validation gate universe (the program.md validity gates + boot-500 from #272).
G_GREEDY = "greedy-identity"
G_PPL = "PPL<=2.42"
G_BOOT = "boot-500"
G_TPS = "TPS"
GATES = (G_GREEDY, G_PPL, G_BOOT, G_TPS)


# --------------------------------------------------------------------------- #
# The three served-file touchpoints (T5/T6/T7) the EAGLE-3 swap must rewrite.
# Each: id, name, mtp_anchor (file, substring re-read + confirmed present), the
# concrete eagle3_delta, the rewrite complexity, the reopened re-validation gates,
# and an honest caveat.
# --------------------------------------------------------------------------- #
TOUCHPOINTS: list[dict[str, Any]] = [
    {
        "id": "T5",
        "name": "Fused sparse-argmax patch (drafter head)",
        "mtp_anchor": [
            (SITE, 'TOP_TOKEN_TARGET = "vllm.model_executor.models.gemma4_mtp"'),
            (SITE, "embedder_cls = module.Gemma4MTPMaskedEmbedder"),
            (SITE, "self.model.get_top_tokens"),
        ],
        "mtp_assumption": (
            "The fused Triton sparse-argmax kernel (_apply_fused_top_token_patch) wraps "
            "Gemma4MTPMaskedEmbedder.get_top_tokens and is built on MTP-only structure: "
            "self.centroids, self.centroid_intermediate_top_k, self.token_ordering, "
            "self.num_centroids, self.vocab_size_per_centroid, self.num_selected "
            "(clustered top-12k sparse verify). It scores hidden over a sparse candidate "
            "set, not a dense lm_head."
        ),
        "eagle3_delta": (
            "EAGLE-3's head is Eagle3LlamaForCausalLM: it has NO get_top_tokens and NO "
            "centroid clustering -- it does a dense compute_logits over its (d2t-reduced) "
            "draft vocab. So TOP_TOKEN_TARGET=gemma4_mtp matches no EAGLE module and the "
            "patch goes INERT. Retaining a fused drafter argmax means re-targeting the "
            "meta-path finder to the EAGLE head and wrapping compute_logits argmax -- but "
            "the centroid-sparse kernel has NO EAGLE analogue (no token_ordering), so the "
            "fused-sparse bandwidth saving is LOST unless re-derived from scratch; the "
            "drop-in is a dense argmax over the EAGLE draft head."
        ),
        "complexity": MODERATE,
        "revalidation_gates": [G_GREEDY, G_TPS],
        "caveat": (
            "Inert-not-crash. Drafter argmax is emit-neutral in principle (greedy spec "
            "decode emits the TARGET argmax), so PPL is not directly moved; but the "
            "changed drafter argmax shifts acceptance -> verify batch M -> batch-variant "
            "int4-Marlin near-tie flips, so greedy-identity must be re-measured and the "
            "lost sparse kernel re-priced for TPS."
        ),
    },
    {
        "id": "T6",
        "name": "onegraph loopgraph capture (proposer key + body)",
        "mtp_anchor": [
            (SITE, 'LOOPGRAPH_TARGET = "vllm.v1.spec_decode.gemma4"'),
            (SITE, "proposer_cls = module.Gemma4Proposer"),
            (SITE, "def propose_onegraph"),
        ],
        "mtp_assumption": (
            "_apply_loopgraph_patch is keyed to LOOPGRAPH_TARGET=vllm.v1.spec_decode.gemma4 "
            "and rebinds Gemma4Proposer.propose to propose_onegraph. The whole body is "
            "written against the Gemma4Proposer helper API: set_inputs_first_pass, "
            "build_per_group_and_layer_attn_metadata, _determine_batch_execution_and_padding, "
            "_sample_draft_tokens, self.model.get_top_tokens, self.hidden_states/input_ids "
            "static buffers."
        ),
        "eagle3_delta": (
            "Under method=='eagle3' vLLM instantiates EagleProposer (vllm.v1.spec_decode."
            "eagle), NOT Gemma4Proposer -- so the onegraph K=7 single-replay (the engine "
            "that produces 481.53) is NEVER invoked. Retaining it requires re-pointing "
            "LOOPGRAPH_TARGET -> 'vllm.v1.spec_decode.eagle', proposer_cls -> "
            "module.EagleProposer, AND rewriting the propose_onegraph body against "
            "EagleProposer's interface (its own propose signature + draft loop; the named "
            "Gemma4Proposer helpers do not exist on it). The CUDA-graph capture scaffolding "
            "(_build_static_buffers / _capture_graph / ping-pong slots) is structurally "
            "reusable, but the per-iteration forward + token plumbing must be re-bodied."
        ),
        "complexity": MODERATE,
        "revalidation_gates": [G_GREEDY, G_BOOT, G_TPS],
        "caveat": (
            "Re-pointing the key is trivial; rewriting the body against the EagleProposer "
            "API is the real work. A fresh capture path can fail at boot "
            "(LOOPGRAPH_REQUIRE_CAPTURE=1 raises), so boot-500 reopens; acceptance/TPS "
            "and the greedy pre-flight reopen with any drafter-loop change."
        ),
    },
    {
        "id": "T7",
        "name": "onegraph correctness invariant + static buffers / KV-bearing chain",
        "mtp_anchor": [
            (SITE, "Q-only and KV-shared"),
            (SITE, "Width-1 is exact"),
            (SITE, "def _build_static_buffers"),
            (SITE, "def _capture_graph"),
            (SITE, "self.hidden_states[:1].copy_(target_hidden_states[sample_index])"),
        ],
        "mtp_assumption": (
            "The onegraph correctness invariant (sitecustomize.py:25-32): the Gemma4 MTP "
            "drafter is Q-only and KV-shared -- it never writes KV and has no "
            "cross-position dependencies, so width-1 is exact. _build_static_buffers "
            "allocates a SINGLE [1,2560] hidden buffer; the width-1 body seeds it from one "
            "target hidden state (target_hidden_states[sample_index]) and recycles "
            "backbone_hidden across the K iterations; per-layer attn metadata is built once "
            "(build_for_drafting, draft_index=1) over the SHARED target KV pool."
        ),
        "eagle3_delta": (
            "The invariant is FALSE for EAGLE-3: its draft layer is a full Llama decoder "
            "with its own k_proj/v_proj -- it WRITES its own KV at each draft position and "
            "iteration i+1 reads the KV it wrote at positions 0..i (cross-position deps). "
            "So (a) the seed is a fused [1,7680] aux input (cat of layers 2/21/39), not a "
            "single [1,2560] target hidden; (b) the static buffers must hold a GROWING "
            "draft KV and advance seq_len/slot_mapping per iteration INSIDE the captured "
            "graph; (c) 'width-1 is exact' must be re-derived for a KV-bearing chain. This "
            "is a correctness re-derivation, not a re-pointing."
        ),
        "complexity": REDERIVE,
        "revalidation_gates": [G_GREEDY, G_PPL, G_BOOT, G_TPS],
        "caveat": (
            "The deepest and most coupled touchpoint. A wrong KV-bearing chain changes "
            "accepted lengths (-> verify batch M -> batch-variant argmax -> served stream "
            "-> greedy-identity AND same-path PPL must be re-measured), the re-sized "
            "capture can crash boot, and TPS is the whole point. ALL four gates reopen."
        ),
    },
]

# Issue #272 boot-500 guard: the deployed frontier is MISSING the guard the sibling
# carries. T6/T7 reopen this exact file, so porting the guard is a near-zero-marginal
# co-edit (the rewrite already edits sitecustomize.py).
BOOT_FRAGILE_TOKEN = "_guard_included_router"

# Banked artifacts for the eager-fallback floor (re-read from repo -> falsifiable).
ROOFLINE_JSON = "research/spec_cost_model/drafter_forward_roofline.json"
LAUNCH_LEG_JSON = "research/spec_cost_model/launch_overhead_graph_leg.json"
EAGLE3_STEP_JSON = "research/validity/eagle3_step_overhead/eagle3_step_overhead_results.json"


def _read(rel: str) -> str | None:
    p = ROOT / rel
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8", errors="replace")


def _read_json(rel: str) -> dict[str, Any] | None:
    txt = _read(rel)
    if txt is None:
        return None
    try:
        return json.loads(txt)
    except Exception:
        return None


def locate_touchpoints() -> dict[str, Any]:
    """Re-read every cited (file, substring) anchor and confirm each touchpoint is
    LOCATED in the real source (self-test condition (a))."""
    checks: list[dict[str, Any]] = []
    located: dict[str, bool] = {}
    for tp in TOUCHPOINTS:
        tp_ok = True
        for rel, substr in tp["mtp_anchor"]:
            text = _read(rel)
            present = bool(text) and (substr in text)
            tp_ok = tp_ok and present
            checks.append(
                {"touchpoint": tp["id"], "file": rel, "substring": substr, "present": present}
            )
        located[tp["id"]] = tp_ok
    return {
        "all_located": all(located.values()),
        "located": located,
        "n_located": sum(1 for v in located.values() if v),
        "checks": checks,
    }


def classify_complexity() -> dict[str, Any]:
    """Sub-classify each served-file touchpoint's rewrite complexity + reopened gates
    (self-test condition (b))."""
    counts = {c: 0 for c in COMPLEXITY}
    classified_with_gates = True
    reopened_union: set[str] = set()
    per_tp: list[dict[str, Any]] = []
    for tp in TOUCHPOINTS:
        c = tp["complexity"]
        gset = tp["revalidation_gates"]
        valid = (c in COMPLEXITY) and bool(gset) and all(g in GATES for g in gset)
        classified_with_gates = classified_with_gates and valid
        if c in counts:
            counts[c] += 1
        reopened_union |= set(gset)
        per_tp.append({"id": tp["id"], "complexity": c, "gates": list(gset), "valid": valid})
    return {
        "counts": counts,
        "classified_with_gates": classified_with_gates,
        "reopened_gates_union": sorted(reopened_union),
        "has_correctness_rederivation": counts[REDERIVE] > 0,
        "rewrite_reopens_greedy_identity": G_GREEDY in reopened_union,
        "rewrite_reopens_ppl": G_PPL in reopened_union,
        "rewrite_reopens_boot": G_BOOT in reopened_union,
        "per_touchpoint": per_tp,
    }


def eager_fallback_floor() -> dict[str, Any]:
    """Estimate the TPS floor a config-only swap delivers (stock un-onegraphed eager
    EagleProposer) from the BANKED onegraph->eager drafter-chain decomposition.

    PRIMARY anchor (direct measurement, research/spec_cost_model/drafter_forward_roofline.json):
      the deployed MTP drafter K=7 loop measures graph=566.49us vs eager=2859.34us within
      an 11.6ms decode step. Removing the onegraph drafter capture inflates the step by the
      measured eager penalty; iso-E[T] TPS scales by step_graph/step_eager.

    The MTP-equivalent floor is an UPPER bound on the true EAGLE-3 eager floor (EAGLE's
    own-KV Llama drafter is heavier than MTP's Q-only layer). The launch-count model
    (#154) gives a milder cross-check that UNDERCOUNTS drafter launches. All flagged
    ESTIMATE pending a real eager-path run.
    """
    method_parts: list[str] = []
    notes: list[str] = []

    # --- PRIMARY: direct drafter-chain graph-vs-eager measurement -------------- #
    roof = _read_json(ROOFLINE_JSON)
    floor_mtp_equiv = None
    step_us = drafter_graph_us = drafter_eager_us = step_eager_mtp = None
    if roof is not None:
        try:
            step_us = float(roof["config"]["decode_step_ms"]) * 1000.0          # 11600 us
            ch = roof["chain"]
            drafter_graph_us = float(ch["decode_step_chain_us_graph"])          # 566.49
            drafter_eager_us = float(ch["decode_step_chain_us_eager"])          # 2859.34
            step_eager_mtp = step_us + (drafter_eager_us - drafter_graph_us)
            floor_mtp_equiv = OFFICIAL_BASELINE * step_us / step_eager_mtp
            method_parts.append(
                f"drafter-chain direct measurement ({ROOFLINE_JSON}): step {step_us:.0f}us, "
                f"drafter loop graph {drafter_graph_us:.1f}us -> eager {drafter_eager_us:.1f}us, "
                f"iso-E[T] floor = {OFFICIAL_BASELINE} * {step_us:.0f}/{step_eager_mtp:.0f}"
            )
        except Exception as exc:  # pragma: no cover - artifact-shape guard
            notes.append(f"roofline parse failed ({exc!r}); MTP-equivalent floor unavailable")
    else:
        notes.append(f"{ROOFLINE_JSON} missing; MTP-equivalent floor unavailable")

    # --- CROSS-CHECK (upper): launch-count model (#154) ------------------------ #
    leg = _read_json(LAUNCH_LEG_JSON)
    floor_launchleg = None
    n_draft = plo = None
    if leg is not None and step_us is not None:
        try:
            n_draft = float(leg["model"]["launch_counts"]["drafter_propose"])    # 35
            plo = float(leg["model"]["per_launch_overhead_us"])                  # 7.42
            tax = n_draft * plo                                                  # 259.6 (zero-overlap)
            floor_launchleg = OFFICIAL_BASELINE * step_us / (step_us + tax)
            method_parts.append(
                f"launch-count cross-check ({LAUNCH_LEG_JSON}): drafter_propose={n_draft:.0f} "
                f"launches x {plo:.2f}us zero-overlap = {tax:.0f}us tax (UNDERCOUNTS drafter "
                f"kernels ~5x vs the measured chain; milder upper bound)"
            )
        except Exception as exc:  # pragma: no cover
            notes.append(f"launch-leg parse failed ({exc!r})")

    # --- LOWER bound: EAGLE-3 heavier own-KV drafter (#293) --------------------- #
    d293 = _read_json(EAGLE3_STEP_JSON)
    floor_eagle_lower = None
    eagle_mtp_draft_ratio = None
    if d293 is not None and step_us is not None and drafter_eager_us is not None:
        try:
            prim = d293["synthesis"]["primary"]
            mtp_draft = float(prim["linear_draft_us"])                           # 149.84
            eagle_draft = float(prim["by_m"]["3"]["eagle3_draft_us"])            # 449.52 (L_fuse=3)
            eagle_mtp_draft_ratio = eagle_draft / mtp_draft                      # ~3.0
            # Crude lower bound: scale the MEASURED drafter eager penalty by the EAGLE/MTP
            # draft-compute ratio (EAGLE's own-KV chain costs more eager). Conservative:
            # the eager penalty is launch-dominated and scales sub-linearly with compute,
            # so this OVER-states the loss -> a hard floor, not a central estimate.
            penalty = drafter_eager_us - float(drafter_graph_us)
            step_eager_eagle = step_us + eagle_mtp_draft_ratio * penalty
            floor_eagle_lower = OFFICIAL_BASELINE * step_us / step_eager_eagle
            method_parts.append(
                f"EAGLE-heavier hard-lower ({EAGLE3_STEP_JSON}): draft ratio "
                f"{eagle_mtp_draft_ratio:.2f}x x measured penalty {penalty:.0f}us"
            )
        except Exception as exc:  # pragma: no cover
            notes.append(f"eagle3 #293 parse failed ({exc!r})")

    # Fallbacks if any artifact is absent (documented constants from the reports above).
    if floor_mtp_equiv is None:
        floor_mtp_equiv = 402.0
        notes.append("MTP-equivalent floor fell back to documented 402.0 (roofline 566->2859us)")
    if floor_launchleg is None:
        floor_launchleg = 471.0
        notes.append("launch-leg cross-check fell back to documented ~471")
    if floor_eagle_lower is None:
        floor_eagle_lower = 302.0
        notes.append("EAGLE-heavier lower fell back to documented ~302")

    headline = round(float(floor_mtp_equiv), 1)   # the directly-measured MTP-equiv onegraph->eager floor
    band_lo = round(float(floor_eagle_lower), 1)
    band_hi = round(float(floor_launchleg), 1)

    method = (
        "Banked onegraph->eager decomposition. HEADLINE = MTP-equivalent eager floor "
        "(directly measured drafter loop graph-vs-eager within the 11.6ms step, iso-E[T]); "
        "it is an UPPER bound on the true EAGLE-3 eager floor because EAGLE's own-KV Llama "
        "drafter is heavier. BAND = [EAGLE-heavier hard-lower, launch-count upper]. "
        "ESTIMATE -- no eager-path TPS was measured. | " + " || ".join(method_parts)
    )
    return {
        "eager_fallback_tps_floor": headline,                 # TEST metric (float, ESTIMATE)
        "floor_band_lo": band_lo,
        "floor_band_hi": band_hi,
        "floor_below_baseline": headline < OFFICIAL_BASELINE,
        "official_baseline": OFFICIAL_BASELINE,
        "regression_pct_at_headline": round((OFFICIAL_BASELINE - headline) / OFFICIAL_BASELINE * 100.0, 2),
        "inputs": {
            "decode_step_us": step_us,
            "drafter_loop_graph_us": drafter_graph_us,
            "drafter_loop_eager_us": drafter_eager_us,
            "step_eager_mtp_us": step_eager_mtp,
            "drafter_propose_launches": n_draft,
            "per_launch_overhead_us": plo,
            "eagle_mtp_draft_ratio": eagle_mtp_draft_ratio,
        },
        "method": method,
        "notes": notes,
        "is_estimate": True,
    }


def guard_272_coedit() -> dict[str, Any]:
    """Re-grep the boot-500 guard: missing on the frontier (the #272 gap), present on the
    sibling. T6/T7 reopen this file, so the guard port is a required near-zero co-edit
    (self-test condition (d))."""
    precache = _read(SITE) or ""
    treeverify = _read(f"{TREEVERIFY}/sitecustomize.py") or ""
    precache_guard = len(re.findall(BOOT_FRAGILE_TOKEN, precache))
    treeverify_guard = len(re.findall(BOOT_FRAGILE_TOKEN, treeverify))
    # Sibling guard line numbers (def / call) -- located for the porting hand-off.
    tv_lines = [
        i + 1
        for i, line in enumerate(treeverify.splitlines())
        if BOOT_FRAGILE_TOKEN in line
    ]
    return {
        "precache_guard_count": precache_guard,        # expect 0 (the #272 gap)
        "treeverify_guard_count": treeverify_guard,    # expect 2 (def + call)
        "sibling_guard_lines": tv_lines,               # expect [2680, 2696]
        "precache_missing_guard": precache_guard == 0,
        "sibling_has_guard": treeverify_guard > 0,
        # The rewrite already edits sitecustomize.py for T6/T7, so the marginal cost of
        # also porting the guard is ~0 -> flagged required co-edit.
        "co_edit_required": (precache_guard == 0 and treeverify_guard > 0),
        "co_edit_marginal_cost": "near-zero (file already reopened for T6/T7)",
    }


CAVEATS: list[str] = [
    "STATIC scoping from source -- an engineering-complexity estimate, NOT an "
    "implementation and NOT a measured fallback TPS.",
    "The eager-fallback floor is an ESTIMATE from a BANKED onegraph->eager decomposition; "
    "no eager-path run was launched. 0 TPS.",
    "The MTP-equivalent floor (headline) is an UPPER bound on the true EAGLE-3 eager floor "
    "because EAGLE-3's own-KV Llama drafter is heavier than MTP's Q-only layer.",
    "The launch-count model (#154) UNDERCOUNTS drafter launches (~5/pass vs ~25 real "
    "kernels/pass) and assumes overlap, so it over-states the floor -- kept only as a "
    "milder upper cross-check; the direct drafter-chain measurement supersedes it.",
    "EAGLE-3's E[T] (acceptance) is held at the MTP value (iso-acceptance) for the floor; "
    "any EAGLE-3 E[T] gain is a SEPARATE numerator axis (#293/#295/#304), not credited here.",
    "The EAGLE-3 checkpoint does not exist yet (training-gated); this prices the DEPLOY "
    "rewrite, not the train.",
]


def synthesize() -> dict[str, Any]:
    loc = locate_touchpoints()
    cls = classify_complexity()
    floor = eager_fallback_floor()
    guard = guard_272_coedit()

    metrics = {
        "official_baseline": OFFICIAL_BASELINE,
        "private_verified": PRIVATE_VERIFIED,
        "tps_added_by_this_card": 0.0,
        "eager_fallback_tps_floor": floor["eager_fallback_tps_floor"],
        "floor_band_lo": floor["floor_band_lo"],
        "floor_band_hi": floor["floor_band_hi"],
        "regression_pct_at_headline": floor["regression_pct_at_headline"],
        "n_touchpoints": float(len(TOUCHPOINTS)),
        "n_located": float(loc["n_located"]),
        "n_mechanical_config": float(cls["counts"][MECHANICAL]),
        "n_moderate_rewrite": float(cls["counts"][MODERATE]),
        "n_correctness_rederivation": float(cls["counts"][REDERIVE]),
        "precache_guard_count": float(guard["precache_guard_count"]),
        "treeverify_guard_count": float(guard["treeverify_guard_count"]),
    }
    nan_clean = all(
        isinstance(v, (int, float)) and math.isfinite(float(v)) for v in metrics.values()
    )

    conditions = {
        # (a) all 3 touchpoints located in source.
        "all_touchpoints_located": loc["all_located"],
        "three_touchpoints": len(TOUCHPOINTS) == 3,
        # (b) each classified + re-validation gates listed.
        "all_classified_with_gates": cls["classified_with_gates"],
        "has_correctness_rederivation": cls["has_correctness_rederivation"],
        # (c) eager-fallback floor bounded with method stated.
        "floor_finite": math.isfinite(float(floor["eager_fallback_tps_floor"])),
        "floor_bounded": floor["floor_band_lo"] < floor["eager_fallback_tps_floor"] < OFFICIAL_BASELINE
        and floor["floor_band_lo"] > 0.0
        and floor["floor_band_hi"] <= OFFICIAL_BASELINE,
        "floor_method_stated": bool(floor["method"]) and floor["is_estimate"],
        "floor_below_baseline": floor["floor_below_baseline"],
        # (d) #272 co-edit flagged.
        "guard_272_coedit_flagged": guard["co_edit_required"],
        "precache_missing_guard": guard["precache_missing_guard"],
        "sibling_has_guard": guard["sibling_has_guard"],
        # (e) NaN-clean.
        "nan_clean": nan_clean,
        # (f) caveats carried.
        "caveats_carried": len(CAVEATS) >= 5,
        # the deliverable booleans must be self-consistent.
        "rewrite_reopens_greedy_identity_true": cls["rewrite_reopens_greedy_identity"],
    }
    self_test_passes = all(bool(v) for v in conditions.values())

    return {
        "loopgraph_rewrite_cost_self_test_passes": bool(self_test_passes),
        "eager_fallback_tps_floor": floor["eager_fallback_tps_floor"],
        "rewrite_reopens_greedy_identity": bool(cls["rewrite_reopens_greedy_identity"]),
        "nan_clean": bool(nan_clean),
        "conditions": conditions,
        "locate": loc,
        "classify": cls,
        "floor": floor,
        "guard_272": guard,
        "metrics": metrics,
        "caveats": CAVEATS,
        "touchpoints": TOUCHPOINTS,
    }


def build_summary(syn: dict[str, Any]) -> dict[str, Any]:
    m = syn["metrics"]
    cls = syn["classify"]
    summary = {
        # PRIMARY + TEST.
        "loopgraph_rewrite_cost_self_test_passes": int(syn["loopgraph_rewrite_cost_self_test_passes"]),
        "eager_fallback_tps_floor": float(syn["eager_fallback_tps_floor"]),
        "rewrite_reopens_greedy_identity": int(bool(syn["rewrite_reopens_greedy_identity"])),
        # floor band + framing.
        "floor_band_lo": float(m["floor_band_lo"]),
        "floor_band_hi": float(m["floor_band_hi"]),
        "floor_regression_pct": float(m["regression_pct_at_headline"]),
        "floor_is_estimate": 1,
        # touchpoint complexity partition.
        "n_touchpoints": int(m["n_touchpoints"]),
        "n_located": int(m["n_located"]),
        "n_mechanical_config": int(m["n_mechanical_config"]),
        "n_moderate_rewrite": int(m["n_moderate_rewrite"]),
        "n_correctness_rederivation": int(m["n_correctness_rederivation"]),
        "rewrite_reopens_ppl": int(bool(cls["rewrite_reopens_ppl"])),
        "rewrite_reopens_boot": int(bool(cls["rewrite_reopens_boot"])),
        # #272 co-edit cross-check.
        "precache_guard_count": int(m["precache_guard_count"]),
        "treeverify_guard_count": int(m["treeverify_guard_count"]),
        "guard_272_coedit_required": int(bool(syn["guard_272"]["co_edit_required"])),
        # honest 0-TPS framing.
        "official_baseline": float(m["official_baseline"]),
        "private_verified": float(m["private_verified"]),
        "tps_added_by_this_card": float(m["tps_added_by_this_card"]),
        "nan_clean": int(bool(syn["nan_clean"])),
    }
    summary.update({f"selftest_{k}": int(bool(v)) for k, v in syn["conditions"].items()})
    return summary


def log_wandb(syn: dict[str, Any], args: argparse.Namespace) -> None:
    try:
        import wandb as _wb  # noqa: F401
        if not hasattr(_wb, "init"):
            raise ImportError("resolved a stub wandb with no .init")
        sys.path.insert(0, str(ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[eagle3-rewrite-cost] wandb logging skipped (analysis unaffected): {exc}", flush=True)
        return

    try:
        run = init_wandb_run(
            job_type="validity-gate",
            agent="wirbel",
            name=args.wandb_name or "wirbel/eagle3-rewrite-cost",
            group=args.wandb_group,
            tags=[
                "eagle3-rewrite-cost", "deployment-cost", "loopgraph-rewrite",
                "eager-fallback-floor", "onegraph-mtp-specific", "boot-272-coedit",
                "pr-312", "zero-tps",
            ],
            config={
                "official_baseline": OFFICIAL_BASELINE,
                "private_verified": PRIVATE_VERIFIED,
                "eagle3_target_layers": list(EAGLE3_TARGET_LAYERS),
                "eagle3_fused_in": EAGLE3_FUSED_IN,
                "mtp_hidden": MTP_HIDDEN,
                "deployed_method": "mtp",
                "target_method": "eagle3",
                "deployed_submission": PRECACHE,
                "sibling_with_guard": TREEVERIFY,
                "wandb_group": args.wandb_group,
                "imports": (
                    "wirbel#307(88eh8twv: swap_is_config_only=0, T5/T6/T7 served-file, "
                    "sitecustomize:18/20/25-32/274, #272 guard 0/2) x drafter_forward_roofline"
                    "(PR#154: drafter loop graph 566us->eager 2859us in 11.6ms step) x "
                    "launch_overhead_graph_leg(PR#154: drafter_propose=35 launches x 7.42us) x "
                    "wirbel#293(eagle3 draft ~3x MTP at L_fuse=3) x Issue#272(boot-500 guard)"
                ),
            },
        )
    except Exception as exc:
        print(f"[eagle3-rewrite-cost] wandb init failed (analysis unaffected): {exc}", flush=True)
        return
    if run is None:
        print("[eagle3-rewrite-cost] wandb: no run (no API key / disabled) -- skipping", flush=True)
        return

    summary = build_summary(syn)
    log_summary(run, summary, step=0)
    try:
        log_json_artifact(
            run,
            name="eagle3_loopgraph_rewrite_cost",
            artifact_type="validity-analysis",
            data={
                "verdict": {
                    "loopgraph_rewrite_cost_self_test_passes": syn["loopgraph_rewrite_cost_self_test_passes"],
                    "eager_fallback_tps_floor": syn["eager_fallback_tps_floor"],
                    "rewrite_reopens_greedy_identity": syn["rewrite_reopens_greedy_identity"],
                },
                "classify": syn["classify"],
                "floor": syn["floor"],
                "guard_272": syn["guard_272"],
                "conditions": syn["conditions"],
                "caveats": syn["caveats"],
                "touchpoints": syn["touchpoints"],
                "locate": syn["locate"],
            },
        )
    except Exception as exc:
        print(f"[eagle3-rewrite-cost] wandb artifact skipped: {exc}", flush=True)
    finish_wandb(run)
    print(f"[eagle3-rewrite-cost] wandb run logged: {getattr(run, 'id', '?')}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--no-wandb", action="store_true", help="skip W&B logging")
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument(
        "--wandb-group", "--wandb_group", dest="wandb_group", default="eagle3-rewrite-cost"
    )
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args(argv)

    syn = synthesize()
    payload = {
        "pr": 312,
        "question": "Price the served-file loopgraph rewrite EAGLE-3 requires.",
        "verdict": {
            "loopgraph_rewrite_cost_self_test_passes": syn["loopgraph_rewrite_cost_self_test_passes"],
            "eager_fallback_tps_floor": syn["eager_fallback_tps_floor"],
            "rewrite_reopens_greedy_identity": syn["rewrite_reopens_greedy_identity"],
        },
        "nan_clean": syn["nan_clean"],
        "summary": build_summary(syn),
        "conditions": syn["conditions"],
        "classify": syn["classify"],
        "floor": syn["floor"],
        "guard_272": syn["guard_272"],
        "locate": syn["locate"],
        "caveats": syn["caveats"],
        "touchpoints": syn["touchpoints"],
        "honest_framing": (
            "STATIC scoping from the real sitecustomize.py source -- engineering-complexity "
            "estimate + a banked onegraph->eager floor. NO served-file edit, NO build, NO "
            "boot test, NO HF Job, NO submission change. 0 GPU, 0 TPS. BASELINE 481.53 "
            "unchanged. The eager-fallback floor is an ESTIMATE pending a real eager-path run."
        ),
        "handoff_sentence": (
            "DEPLOYMENT-COST axis for the EAGLE-3 GO/NO-GO packet: the config-only swap is "
            "NOT free -- it deploys the stock eager EagleProposer at an estimated "
            f"~{syn['eager_fallback_tps_floor']:.0f} TPS floor (band "
            f"[{syn['floor']['floor_band_lo']:.0f}, {syn['floor']['floor_band_hi']:.0f}], a "
            f"{syn['floor']['regression_pct_at_headline']:.0f}% regression vs 481.53), because "
            "the MTP-specific onegraph loopgraph goes inert. Recovering frontier runtime needs "
            "2 moderate-rewrite touchpoints (T5/T6) + 1 correctness-rederivation (T7, the "
            "KV-bearing draft chain), reopening greedy-identity / PPL<=2.42 / boot-500 / TPS, "
            "plus a near-zero #272 guard co-edit on the same file."
        ),
    }

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_loopgraph_rewrite_cost_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[eagle3-rewrite-cost] wrote {out_path}", flush=True)

    v = payload["verdict"]
    f = syn["floor"]
    print(
        "[eagle3-rewrite-cost] VERDICT: "
        f"self_test_passes={v['loopgraph_rewrite_cost_self_test_passes']} "
        f"eager_fallback_tps_floor={v['eager_fallback_tps_floor']} "
        f"(band [{f['floor_band_lo']}, {f['floor_band_hi']}], "
        f"-{f['regression_pct_at_headline']}% vs 481.53) "
        f"rewrite_reopens_greedy_identity={v['rewrite_reopens_greedy_identity']}",
        flush=True,
    )
    print(
        "[eagle3-rewrite-cost] complexity partition: "
        f"mechanical={syn['classify']['counts'][MECHANICAL]} "
        f"moderate={syn['classify']['counts'][MODERATE]} "
        f"rederive={syn['classify']['counts'][REDERIVE]} | "
        f"reopened gates={syn['classify']['reopened_gates_union']} | "
        f"#272 precache_guard={syn['guard_272']['precache_guard_count']} "
        f"treeverify_guard={syn['guard_272']['treeverify_guard_count']} "
        f"lines={syn['guard_272']['sibling_guard_lines']}",
        flush=True,
    )
    print(
        f"[eagle3-rewrite-cost] self_test_passes={syn['loopgraph_rewrite_cost_self_test_passes']} "
        f"nan_clean={syn['nan_clean']}",
        flush=True,
    )

    if not args.no_wandb:
        log_wandb(syn, args)

    if args.self_test:
        if not syn["loopgraph_rewrite_cost_self_test_passes"]:
            failed = [k for k, val in syn["conditions"].items() if not val]
            print(f"[eagle3-rewrite-cost] SELF-TEST FAILED: {failed}", flush=True)
            return 1
        print("[eagle3-rewrite-cost] SELF-TEST PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
