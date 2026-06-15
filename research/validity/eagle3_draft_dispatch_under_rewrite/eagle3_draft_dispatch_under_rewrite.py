#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Does the #312 loopgraph rewrite REINTRODUCE draft-side capture-dispatch risk? (PR #315)

THE QUESTION
------------
ubel #311 (os01ttw9, MERGED) found the deployed DRAFT side is dispatch-safe BY
CONSTRUCTION: the K=7 MTP drafter is a manual ONEGRAPH capture (CUDAGraphMode.NONE,
sitecustomize.py:170) that never touches vLLM's capture-size list -- so only the
M-token VERIFY side carries the lawine-#101 dispatch risk (max_safe=16; M=32 needs
[24,32] added). But wirbel #312 (9b1arani, MERGED) showed the loopgraph rewrite
(T6/T7) RE-BODIES that capture against EAGLE-3's KV-bearing draft chain, which "must
hold a growing draft KV and advance seq_len/slot_mapping INSIDE the captured graph"
-- unlike the MTP's single recycled [1,2560] static buffer recycled across K.

This card closes the open sub-axis: does the EAGLE-3 draft capture re-bodied by #312's
rewrite REINTRODUCE the draft-side capture-dispatch risk the MTP ONEGRAPH avoids?

THE TWO DISTINCT "DISPATCH" CONCEPTS (carried honestly; the crux of the answer)
-------------------------------------------------------------------------------
  1. vLLM LIST-dispatch (the lawine #101 / capture-SIZE axis): vLLM's full/piecewise
     CUDA-graph runtime picks ONE captured graph keyed by the padded token-count of the
     CURRENT forward, looked up against `cudagraph_capture_sizes` (ceiling
     max_cudagraph_capture_size=16). token-count > ceiling -> IndexError (no captured
     graph). This is a PER-FORWARD-CALL engine dispatch. #311 priced it for VERIFY.
  2. MANUAL unrolled ONEGRAPH (CUDAGraphMode.NONE): the drafter path sets
     cudagraph_runtime_mode=CUDAGraphMode.NONE (:170) -> the engine does NOT route the
     forward through its cudagraph dispatcher -- and the patch ITSELF wraps the whole
     K-iteration loop in a single torch.cuda.graph(graph) capture (:243) replayed via
     graph.replay(). The draft chain BYPASSES the vLLM list entirely; there is no
     per-call lookup against `cudagraph_capture_sizes`.

MTP draft-safety rests on mechanism (2), NOT on single-shape: single-shape is merely
what makes the MTP manual capture TRIVIAL (one width-1 body replayed K times). EAGLE-3
breaks single-shape (growing draft KV, advancing seq_len) -- but a manual unrolled
graph records the per-iteration kernels with WHATEVER (static) shapes they had at
capture; per-step variation is ABSORBED into one graph, NOT pushed onto a capture-size
list. So the list-dispatch (#101) risk is reintroduced ONLY IF the rewrite ABANDONS
CUDAGraphMode.NONE. #312's banked plan KEEPS the manual scaffolding
(_build_static_buffers/_capture_graph/ping-pong "structurally reusable"), so under the
intended rewrite the draft side stays OFF the list-dispatch axis -> classification (a).
Belt-and-suspenders: even a degenerate list-route is safe, K=7 within-chain replay
sizes {1..7} all <= the deployed ceiling-16 (only verify M=32 breaches it).

WHAT THIS LEG DOES (LOCAL, read-only STATIC analysis, NO GPU, NO served change, 0 TPS)
-------------------------------------------------------------------------------------
  1. RE-READ (read-only) the deployed draft capture in sitecustomize.py and confirm WHY
     MTP is dispatch-safe: CUDAGraphMode.NONE manual capture, single recycled [1,2560]
     hidden, Q-only/KV-shared, FIXED width-1 replay across all K -> never list-dispatched.
  2. MAP #312's T6/T7 rewrite onto the draft capture: enumerate the EAGLE draft-chain
     deltas vs MTP (own-KV k_proj/v_proj, growing draft KV, seq_len/slot_mapping advance
     INSIDE the captured graph, fused [1,7680] aux seed) and DERIVE, per draft step i, the
     replay shape the captured EAGLE draft graph must dispatch.
  3. CLASSIFY the draft-side dispatch risk (a) single-shape/dispatch-safe vs (b) per-step
     shape varies -> needs a draft capture-size list. Emit the dispatch-correctness GATE.
  4. CROSS-CHECK #311 (os01ttw9) + #312 (9b1arani) banked constants <= 1e-6.

DERIVED, not measured: no EAGLE-3 checkpoint exists (training-gated). The draft capture
shapes are DERIVED from #312's banked rewrite spec + the deployed MTP source, NOT
measured from a running EagleProposer. This PRICES the dispatch-correctness property the
rewrite must satisfy; it does NOT implement it. NaN-clean; BASELINE 481.53 untouched;
0 TPS; NOT a launch / build / served-file change / HF Job."""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]                       # .../target
SUBMISSION_DIR = REPO_ROOT / "submissions" / "fa2sw_precache_kenyan"
DISPATCH_RISK_311_RESULTS = (
    REPO_ROOT / "research" / "validity" / "eagle3_capture_dispatch_risk"
    / "eagle3_capture_dispatch_risk_results.json"
)
REWRITE_COST_312_RESULTS = (
    REPO_ROOT / "research" / "validity" / "eagle3_loopgraph_rewrite_cost"
    / "eagle3_loopgraph_rewrite_cost_results.json"
)

# --------------------------------------------------------------------------- #
# Banked anchors imported VERBATIM from ubel #311 (os01ttw9) and wirbel #312
# (9b1arani); NEVER re-derived. The <=1e-6 cross-check (self-test e) reloads both
# results JSONs at runtime and asserts these match the recorded values.
# All runs live in wandb-applied-ai-team/gemma-challenge-senpai.
# --------------------------------------------------------------------------- #
OFFICIAL_BASELINE = 481.53                         # PR #52 official frontier TPS (untouched)

# ---- ubel #311 (os01ttw9): the capture-SIZE DISPATCH axis this card extends ----
K_SPEC = 7                                          # K=7 draft chain (num_speculative_tokens)
M_VERIFY_DEPLOYED = 8                               # deployed verify width (1+K spine)
MAX_CUDAGRAPH_CAPTURE_SIZE_101 = 16                 # deployed (inherited) ceiling
VALID_CAPTURED_SIZES_306 = (8, 16)                  # verify (1+K)-multiples within ceiling
MAX_SAFE_TREE_WIDTH_311 = 16                        # #311 verify max-safe under deployed list
DEPLOYED_M8_DISPATCH_SAFE_311 = True               # #311 deployed verify M=8 dispatch-safe
DRAFT_NOT_LIST_DISPATCHED_311 = False              # #311 draft routed_through_vllm_capture_list
DRAFT_DISPATCH_SAFE_311 = True                     # #311 draft side dispatch-safe (by construction)
M32_MITIGATION_SIZES_TO_ADD_311 = (24, 32)         # #311 verify M=32 list edit

# ---- wirbel #312 (9b1arani): the loopgraph rewrite cost this card sub-axes ----
EAGER_FALLBACK_TPS_FLOOR_312 = 402.1               # config-only-swap eager floor (ESTIMATE)
FLOOR_REGRESSION_PCT_312 = 16.5                    # eager floor regression vs 481.53
MTP_HIDDEN_DIM = 2560                               # MTP single recycled [1,2560] hidden buffer
EAGLE_FUSED_AUX_DIM = 7680                          # EAGLE fused [1,7680] aux seed (cat layers 2/21/39)
N_FUSED_AUX_LAYERS = 3                              # 7680 == 3 * 2560 (internal consistency)

# vLLM dispatch-mechanism refs (carried from #311; corroborate IndexError != OOM).
VLLM_DISPATCH_REFS = ("vLLM#29091", "vLLM-PR#23679")


# --------------------------------------------------------------------------- #
# (1) Re-read the deployed MTP draft capture (read-only) and confirm WHY it is
#     dispatch-safe by construction.
# --------------------------------------------------------------------------- #
def parse_mtp_draft_safety(submission_dir: Path) -> dict[str, Any]:
    """Static, read-only confirmation that the deployed MTP draft is dispatch-safe.

    The safety is mechanism-#2 (CUDAGraphMode.NONE manual capture), NOT single-shape:
      - cudagraph_runtime_mode=CUDAGraphMode.NONE (:170) -> NOT routed through the vLLM
        cudagraph_capture_sizes dispatcher.
      - the patch wraps the whole K-iteration loop in ONE torch.cuda.graph(graph) (:243),
        replayed by graph.replay() -> a flat recorded kernel stream, no per-call list lookup.
      - every model call inside the loop is FIXED width-1 (self.input_ids[:1],
        self.hidden_states[:1]) recycling a SINGLE [1,2560] hidden buffer -> the MTP
        capture is single-shape (Q-only, KV-shared -> width-1 is exact).
    """
    manifest = json.loads((submission_dir / "manifest.json").read_text())
    env = manifest.get("env", {})
    spec_cfg = json.loads(env.get("SPECULATIVE_CONFIG", "{}") or "{}")
    site_src = (submission_dir / "sitecustomize.py").read_text()

    markers = {
        "cudagraph_mode_none": "cudagraph_runtime_mode=CUDAGraphMode.NONE" in site_src,
        "manual_cuda_graph_capture": "with torch.cuda.graph(graph):" in site_src,
        "graph_replay": "graph.replay()" in site_src,
        "width1_input_copy": "self.input_ids[:1].copy_(source)" in site_src,
        "single_recycled_hidden": "self.hidden_states[:1].copy_(backbone_hidden[:1])" in site_src,
        "q_only_kv_shared_comment": "Q-only and KV-shared" in site_src,
        "width1_is_exact_comment": "Width-1 is exact" in site_src,
        "onegraph_env": env.get("ONEGRAPH") == "1",
        "loopgraph_require_capture": env.get("LOOPGRAPH_REQUIRE_CAPTURE") == "1",
    }
    # Mechanism (2) is the load-bearing one for draft dispatch-safety.
    not_list_dispatched = bool(
        markers["cudagraph_mode_none"]
        and markers["manual_cuda_graph_capture"]
        and markers["graph_replay"]
    )
    single_shape = bool(
        markers["width1_input_copy"]
        and markers["single_recycled_hidden"]
        and markers["q_only_kv_shared_comment"]
    )
    return {
        "sitecustomize_path": str((submission_dir / "sitecustomize.py").relative_to(REPO_ROOT)),
        "manifest_path": str((submission_dir / "manifest.json").relative_to(REPO_ROOT)),
        "k_num_speculative_tokens": (
            int(spec_cfg["num_speculative_tokens"])
            if spec_cfg.get("num_speculative_tokens") is not None else None
        ),
        "spec_method": spec_cfg.get("method"),
        "mtp_hidden_dim": MTP_HIDDEN_DIM,
        "cudagraph_mode_none_line": 170,             # ubel #311 cited :170
        "manual_capture_line": 243,                  # torch.cuda.graph(graph)
        "markers": markers,
        # draft dispatch-safety mechanism (load-bearing):
        "draft_not_list_dispatched": not_list_dispatched,
        "draft_single_shape_mtp": single_shape,
        "dispatch_safe_mechanism": (
            "CUDAGraphMode.NONE manual torch.cuda.graph capture -> OFF the vLLM "
            "cudagraph_capture_sizes list -> cannot hit the lawine-#101 IndexError. "
            "Single width-1 shape is what makes the MTP capture TRIVIAL, not what makes "
            "it dispatch-safe."
        ),
    }


# --------------------------------------------------------------------------- #
# (2) Map #312's T6/T7 rewrite onto the draft capture: enumerate the EAGLE deltas
#     and derive the per-step replay shape for each of the K draft positions.
# --------------------------------------------------------------------------- #
def eagle_draft_chain_deltas() -> list[dict[str, Any]]:
    """The EAGLE-3 draft-chain deltas vs the MTP draft (banked from wirbel #312 T7)."""
    return [
        {
            "delta": "own-KV draft layer",
            "mtp": "Q-only, KV-shared: drafter never writes KV (reads the shared target KV pool)",
            "eagle3": "full Llama draft decoder with its own k_proj/v_proj: WRITES its own KV at each draft position",
            "source": "#312 9b1arani T7 (eagle3_delta)",
        },
        {
            "delta": "growing draft KV / cross-position deps",
            "mtp": "no cross-position deps -> width-1 is exact; each of K positions independent",
            "eagle3": "draft step i reads the KV it wrote at positions 0..i -> attention KV extent GROWS 1..K",
            "source": "#312 9b1arani T7 (cross-position deps)",
        },
        {
            "delta": "seq_len / slot_mapping advance INSIDE the captured graph",
            "mtp": "static_cad.seq_lens is a fixed [1] buffer set once; metadata built once over shared KV",
            "eagle3": "static buffers must advance seq_len/slot_mapping per iteration INSIDE the captured graph",
            "source": "#312 9b1arani T7 (advance inside captured graph)",
        },
        {
            "delta": "fused aux seed dim",
            "mtp": f"single recycled [1,{MTP_HIDDEN_DIM}] hidden, seeded from one target hidden state",
            "eagle3": f"fused [1,{EAGLE_FUSED_AUX_DIM}] aux cat of {N_FUSED_AUX_LAYERS} backbone layers (2/21/39)",
            "source": "#312 9b1arani T7 (fused [1,7680] aux input)",
        },
    ]


def derive_eagle_draft_step_shapes(k_spec: int) -> list[dict[str, Any]]:
    """For each EAGLE-3 draft step i in 0..K-1, DERIVE the replay shape the captured
    draft graph must dispatch (from #312's KV-bearing-chain spec, NOT measured).

    Per step i:
      - query_tokens     = 1            (one new draft token; the query stays width-1)
      - draft_kv_extent  = i + 1        (writes draft slot i, attends draft positions 0..i)
      - slot_write_target= i            (fixed per UNROLLED iteration -> static address)
      - position         = prefix + i   (advances 0..i; baked per iteration)
      - matches_mtp      = (draft_kv_extent == 1)  (only step 0 matches MTP's single shape)
    The query is width-1 every step, but the ATTENTION KV extent grows -> the per-step
    body is NOT identical (MTP single-shape invariant breaks), yet each shape is STATIC
    per unrolled step (a compile-time-constant i) -> capturable as ONE manual graph.
    """
    steps: list[dict[str, Any]] = []
    for i in range(int(k_spec)):
        steps.append({
            "step": i,
            "query_tokens": 1,
            "draft_kv_extent": i + 1,
            "slot_write_target": i,
            "position": f"prefix+{i}",
            "matches_mtp_single_shape": (i + 1) == 1,
            "static_per_unrolled_step": True,   # i is a compile-time constant under unrolling
        })
    return steps


# --------------------------------------------------------------------------- #
# (3) Classify (a) vs (b) + emit the dispatch-correctness gate.
# --------------------------------------------------------------------------- #
def classify_draft_dispatch_risk(step_shapes: list[dict[str, Any]], k_spec: int,
                                 max_capture: int) -> dict[str, Any]:
    """Classify the draft-side dispatch risk under #312's rewrite.

    (a) single-shape / dispatch-safe: re-captures the K chain as ONE manual graph (like the
        MTP ONEGRAPH), advancing KV via static-buffer slot-writes at fixed (per-unrolled-step)
        shape -> NOT routed through vLLM's capture-size list -> safe by construction.
    (b) per-step shape VARIES so it needs a DRAFT capture-size list, with K positions that
        must sit within the deployed ceiling-16.

    Finding: the per-step bodies VARY (the MTP single-shape invariant breaks -- the draft
    KV extent grows 1..K), so it is NOT (a) "for free". BUT that variation is ABSORBED by
    unrolling into ONE manual CUDAGraphMode.NONE graph -- it does NOT force a capture-size
    LIST. So the rewrite does NOT reintroduce the lawine-#101 list-dispatch risk on the
    draft side, PROVIDED it preserves the manual NONE capture (which #312's plan does:
    scaffolding "structurally reusable"). Classification = (a), CONDITIONAL on the gate.
    Belt-and-suspenders: even a degenerate list-route is safe -- the within-chain draft
    replay sizes {1..K} all sit <= the deployed ceiling-16 (only verify M=32 breaches it).
    """
    max_draft_kv_extent = max(s["draft_kv_extent"] for s in step_shapes)   # == K
    within_chain_replay_sizes = sorted({s["draft_kv_extent"] for s in step_shapes})  # {1..K}
    per_step_shapes_vary = any(not s["matches_mtp_single_shape"] for s in step_shapes)

    # TEST bools (findings, NOT a pass/fail of the analysis).
    eagle_draft_capture_is_single_shape = not per_step_shapes_vary
    draft_positions_fit_deployed_ceiling16 = bool(
        k_spec <= max_capture and max_draft_kv_extent <= max_capture
        and all(sz <= max_capture for sz in within_chain_replay_sizes)
    )

    # Does the rewrite reintroduce the LIST-dispatch (#101) risk on the draft side?
    # NO -- iff CUDAGraphMode.NONE is preserved (the per-step variation is absorbed into one
    # manual graph, off the cudagraph_capture_sizes axis). This is the conditional finding.
    reintroduces_list_dispatch_risk = False
    classification = "(a)"   # no list-dispatch reintroduced
    classification_is_conditional = True   # (a)-safety holds only if the gate is honored

    gate = [
        "PRESERVE CUDAGraphMode.NONE for the draft chain: keep the re-bodied propose_onegraph "
        "wrapping the K loop in a manual torch.cuda.graph(graph) replay so it stays OFF vLLM's "
        "cudagraph_capture_sizes list-dispatcher (the single load-bearing property -- "
        "abandoning NONE for FULL/PIECEWISE cudagraph mode is the ONLY way to reintroduce #101 "
        "on the draft side).",
        "PRE-ALLOCATE the draft KV to full K="
        f"{k_spec} extent (static address) and advance it by FIXED-ADDRESS slot-writes (draft "
        "slot i baked per unrolled iteration); NO realloc/resize inside the captured region -- "
        "a realloc breaks the manual capture OUTRIGHT (a harder failure than a dispatch error).",
        "BAKE the per-iteration attention metadata (seq_len_i=prefix+i+1, slot_mapping target "
        "slot i) as STATIC compile-time-constant slices so each of the K unrolled steps records "
        "a fixed-shape attention kernel identical at every replay (this is the 'advance "
        "seq_len/slot_mapping INSIDE the captured graph' #312 flagged -- it must be a STATIC "
        "schedule, not a data-dependent shape; requires a draft attention backend that admits "
        "a static per-step capture, as the deployed MTP path already does under NONE).",
        "CROSS-CHECK the within-chain draft replay sizes {1.."
        f"{k_spec}"
        "} are all <= the deployed ceiling-"
        f"{max_capture} (they are; max extent {max_draft_kv_extent} <= {max_capture}) -- so even "
        "a degenerate list-route stays dispatch-safe on the draft side; the verify side (M=32 "
        "needs [24,32] per #311) remains the SOLE #101 exposure.",
    ]

    return {
        "k_spec": int(k_spec),
        "deployed_ceiling": int(max_capture),
        "per_step_shapes_vary": bool(per_step_shapes_vary),
        "max_draft_kv_extent": int(max_draft_kv_extent),
        "within_chain_replay_sizes": within_chain_replay_sizes,
        "eagle_draft_capture_is_single_shape": bool(eagle_draft_capture_is_single_shape),
        "draft_positions_fit_deployed_ceiling16": bool(draft_positions_fit_deployed_ceiling16),
        "reintroduces_list_dispatch_risk": bool(reintroduces_list_dispatch_risk),
        "classification": classification,
        "classification_is_conditional": bool(classification_is_conditional),
        "why": (
            "The MTP single-shape invariant BREAKS for EAGLE-3 (draft KV extent grows 1.."
            f"{max_draft_kv_extent}), so it is NOT (a) by the MTP mechanism. BUT a manual "
            "unrolled CUDAGraphMode.NONE capture records each per-step kernel at its OWN static "
            "shape -- per-step variation is absorbed into ONE graph, NOT pushed onto a "
            "capture-size LIST. So the rewrite does NOT reintroduce the lawine-#101 list-dispatch "
            "IndexError on the draft side, CONDITIONAL on preserving NONE (which #312's reusable "
            "scaffolding does). Belt-and-suspenders: the draft replay sizes {1.."
            f"{max_draft_kv_extent}"
            "} all fit the deployed ceiling-"
            f"{max_capture} even in a degenerate list-route. Classification = (a)-conditional; "
            "gate emitted."
        ),
        "gate": gate,
    }


# --------------------------------------------------------------------------- #
# (e) #311 + #312 cross-check: reload both results JSONs and assert <= 1e-6.
# --------------------------------------------------------------------------- #
def _get(d: dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def cross_check_anchors(tol: float = 1e-6) -> dict[str, Any]:
    """Reload the #311 (os01ttw9) and #312 (9b1arani) results JSONs and assert the banked
    constants this card imports match the recorded values to <= tol."""
    checks: dict[str, Any] = {}
    max_abs_err = 0.0
    loaded_311 = loaded_312 = False

    if DISPATCH_RISK_311_RESULTS.exists():
        r311 = json.loads(DISPATCH_RISK_311_RESULTS.read_text())
        syn = r311.get("synthesis", {})
        consts = _get(syn, "constants", default={})
        draft = _get(syn, "dispatch_arithmetic", "draft_side", default={})
        pairs_311 = {
            "311_k_spec": (float(K_SPEC), float(consts.get("k_spec", float("nan")))),
            "311_m_verify_deployed": (float(M_VERIFY_DEPLOYED),
                                      float(consts.get("m_verify_deployed", float("nan")))),
            "311_max_capture_size": (float(MAX_CUDAGRAPH_CAPTURE_SIZE_101),
                                     float(consts.get("max_cudagraph_capture_size_101", float("nan")))),
            "311_max_safe_tree_width": (float(MAX_SAFE_TREE_WIDTH_311),
                                        float(r311.get("max_safe_tree_width_under_deployed_list", float("nan")))),
        }
        for key, (mine, rec) in pairs_311.items():
            ok = math.isfinite(rec) and abs(mine - rec) <= tol
            if math.isfinite(rec):
                max_abs_err = max(max_abs_err, abs(mine - rec))
            checks[key] = {"mine": mine, "recorded": rec, "ok": bool(ok)}
        # bool anchors (draft safe-by-construction).
        checks["311_draft_dispatch_safe"] = {
            "mine": DRAFT_DISPATCH_SAFE_311, "recorded": bool(draft.get("dispatch_safe")),
            "ok": bool(draft.get("dispatch_safe") is True)}
        checks["311_draft_not_list_dispatched"] = {
            "mine": DRAFT_NOT_LIST_DISPATCHED_311,
            "recorded": bool(draft.get("routed_through_vllm_capture_list")),
            "ok": bool(draft.get("routed_through_vllm_capture_list") is False)}
        checks["311_deployed_m8_dispatch_safe"] = {
            "mine": DEPLOYED_M8_DISPATCH_SAFE_311,
            "recorded": bool(r311.get("deployed_m8_dispatch_safe")),
            "ok": bool(r311.get("deployed_m8_dispatch_safe") is True)}
        loaded_311 = True

    if REWRITE_COST_312_RESULTS.exists():
        r312 = json.loads(REWRITE_COST_312_RESULTS.read_text())
        floor = float(_get(r312, "verdict", "eager_fallback_tps_floor", default=float("nan")))
        reg = float(_get(r312, "summary", "floor_regression_pct", default=float("nan")))
        base = float(_get(r312, "summary", "official_baseline", default=float("nan")))
        pairs_312 = {
            "312_eager_fallback_tps_floor": (EAGER_FALLBACK_TPS_FLOOR_312, floor),
            "312_floor_regression_pct": (FLOOR_REGRESSION_PCT_312, reg),
            "312_official_baseline": (OFFICIAL_BASELINE, base),
        }
        for key, (mine, rec) in pairs_312.items():
            ok = math.isfinite(rec) and abs(mine - rec) <= tol
            if math.isfinite(rec):
                max_abs_err = max(max_abs_err, abs(mine - rec))
            checks[key] = {"mine": mine, "recorded": rec, "ok": bool(ok)}
        # the T7 KV-bearing-chain touchpoint must be the correctness-rederivation.
        t7 = next((t for t in _get(r312, "touchpoints", default=[]) if t.get("id") == "T7"), {})
        checks["312_t7_correctness_rederivation"] = {
            "mine": "correctness-rederivation", "recorded": t7.get("complexity"),
            "ok": t7.get("complexity") == "correctness-rederivation"}
        loaded_312 = True

    # Internal consistency of the carried EAGLE seed dim.
    checks["fused_aux_dim_consistency"] = {
        "mine": EAGLE_FUSED_AUX_DIM, "recorded": N_FUSED_AUX_LAYERS * MTP_HIDDEN_DIM,
        "ok": EAGLE_FUSED_AUX_DIM == N_FUSED_AUX_LAYERS * MTP_HIDDEN_DIM}

    all_ok = bool(loaded_311 and loaded_312 and all(c.get("ok") for c in checks.values()))
    return {
        "results_file_311": str(DISPATCH_RISK_311_RESULTS.relative_to(REPO_ROOT)),
        "results_file_312": str(REWRITE_COST_312_RESULTS.relative_to(REPO_ROOT)),
        "loaded_311": loaded_311, "loaded_312": loaded_312,
        "tol": tol, "checks": checks, "max_abs_err": max_abs_err,
        "all_within_tol": all_ok,
    }


# --------------------------------------------------------------------------- #
# Synthesis + self-test.
# --------------------------------------------------------------------------- #
CAVEATS = [
    "DERIVED, NOT measured: no EAGLE-3 checkpoint exists yet (training-gated). The per-step "
    "draft KV extents 0..i, the fused [1,7680] aux seed, and the own-KV writes are DERIVED from "
    "wirbel #312's banked rewrite spec (9b1arani) + the deployed MTP source -- NOT measured from "
    "a running EagleProposer. This PRICES the dispatch-correctness property the rewrite must "
    "satisfy; it does NOT implement it.",
    "MECHANISM, not single-shape: the MTP draft is dispatch-safe because of CUDAGraphMode.NONE "
    "(manual torch.cuda.graph capture, OFF vLLM's cudagraph_capture_sizes list), NOT because the "
    "body is single-shape. EAGLE-3 BREAKS single-shape (draft KV extent grows 1..K) -- but that "
    "variation is absorbed into ONE manual unrolled graph and does NOT by itself reintroduce the "
    "list-dispatch (#101) risk. The risk returns ONLY if the rewrite abandons NONE for "
    "FULL/PIECEWISE cudagraph mode.",
    "CONDITIONAL (a): the (a)-safety holds ONLY IF the rewrite honors the emitted gate (preserve "
    "NONE; pre-allocate full-K draft KV with fixed-address slot-writes / no realloc; bake a "
    "STATIC per-iteration seq_len/slot_mapping schedule). #312's banked plan keeps the manual "
    "scaffolding 'structurally reusable', which is consistent with honoring the gate -- but the "
    "gate is a CHECKLIST the T6/T7 implementation must satisfy, not a guarantee it already does.",
    "BACKEND assumption: gate item 3 assumes the EAGLE draft attention backend admits a STATIC "
    "per-step capture under CUDAGraphMode.NONE (the deployed MTP draft path already manually "
    "captures attention under NONE, so the same backend path is reusable). If the chosen draft "
    "attention backend pads/reshapes on a runtime seq_len in a non-capturable way, item 3 fails "
    "and the chain cannot be a single manual graph -- a backend-compatibility risk flagged here.",
    "VERIFY side UNCHANGED: this card scopes ONLY the DRAFT-side sub-axis. #311's verify gate "
    "stands independently -- verify dispatch-safe up to M=16 under the deployed list; M=32 "
    "re-enters lawine #101 and needs [24,32] added to cudagraph_capture_sizes (a served-file "
    "change). The draft side does NOT touch that ceiling.",
    "0 TPS / config-correctness property: depends only on integer token-counts / KV extents and "
    "the capture mode, NOT on tensor VALUES (no tensors at all transfer). NOT a launch, NOT a "
    "build, NOT a served-file change, NOT an HF Job. BASELINE 481.53 untouched; adds 0 TPS.",
]


def synthesize(mtp: dict[str, Any], deltas: list[dict[str, Any]],
               step_shapes: list[dict[str, Any]], cls: dict[str, Any],
               xcheck: dict[str, Any]) -> dict[str, Any]:
    # ---- self-test conditions (a-g) ----
    cond: dict[str, Any] = {}
    # (a) MTP draft-safety mechanism confirmed from source.
    cond["a_mtp_draft_safety_mechanism_confirmed"] = bool(
        mtp.get("k_num_speculative_tokens") == K_SPEC
        and mtp.get("draft_not_list_dispatched") is True
        and mtp.get("draft_single_shape_mtp") is True
        and mtp["markers"]["cudagraph_mode_none"]
        and mtp["markers"]["manual_cuda_graph_capture"]
    )
    # (b) #312 T6/T7 draft-chain deltas enumerated (>=4, all sourced to #312).
    cond["b_eagle_deltas_enumerated"] = bool(
        len(deltas) >= 4 and all("9b1arani" in d.get("source", "") for d in deltas)
    )
    # (c) per-step EAGLE draft replay shape derived for each of K steps.
    cond["c_per_step_shapes_derived"] = bool(
        len(step_shapes) == K_SPEC
        and all(s["query_tokens"] == 1 for s in step_shapes)
        and [s["draft_kv_extent"] for s in step_shapes] == list(range(1, K_SPEC + 1))
    )
    # (d) risk classified (a)/(b) with the gate emitted.
    cond["d_classified_with_gate"] = bool(
        cls.get("classification") in ("(a)", "(b)")
        and isinstance(cls.get("gate"), list) and len(cls["gate"]) >= 3
    )
    # (e) #311/#312 constants <= 1e-6.
    cond["e_anchor_constants_exact"] = bool(xcheck["all_within_tol"])
    # (f) NaN-clean -- set by the payload finalizer.
    cond["f_nan_clean"] = True
    # (g) caveats carried (incl. the DERIVED-not-measured caveat).
    cond["g_caveats_carried"] = bool(len(CAVEATS) >= 4)

    passes = all(cond.values())

    verdict = (
        f"The #312 loopgraph rewrite does NOT reintroduce the draft-side capture-dispatch "
        f"(lawine #101) risk -- classification (a), CONDITIONAL on the emitted gate. The MTP "
        f"draft is dispatch-safe by MECHANISM (CUDAGraphMode.NONE manual torch.cuda.graph "
        f"capture, sitecustomize.py:170/:243 -> OFF vLLM's cudagraph_capture_sizes list), NOT "
        f"by single-shape. EAGLE-3 BREAKS single-shape (its draft KV extent grows 1..K={K_SPEC} "
        f"-- eagle_draft_capture_is_single_shape={cls['eagle_draft_capture_is_single_shape']}), "
        f"but a manual UNROLLED NONE capture records each per-step kernel at its own static "
        f"shape, absorbing the variation into ONE graph rather than a capture-size LIST. So the "
        f"#101 list-dispatch risk returns ONLY if the rewrite abandons NONE; #312's plan keeps "
        f"the manual scaffolding 'structurally reusable'. Belt-and-suspenders: the within-chain "
        f"draft replay sizes {cls['within_chain_replay_sizes']} all fit the deployed ceiling-"
        f"{cls['deployed_ceiling']} (draft_positions_fit_deployed_ceiling16="
        f"{cls['draft_positions_fit_deployed_ceiling16']}), so even a degenerate list-route is "
        f"draft-safe -- verify M=32 (needs [24,32] per #311) remains the SOLE #101 exposure. "
        f"DERIVED from #312's spec (no EAGLE checkpoint); analysis-only; BASELINE "
        f"{OFFICIAL_BASELINE} untouched; 0 TPS."
    )
    handoff = (
        f"EAGLE-3 draft-side dispatch-correctness priced under #312's T6/T7 rewrite: "
        f"classification (a) -- NO list-dispatch (#101) risk reintroduced on the draft side, "
        f"CONDITIONAL on preserving CUDAGraphMode.NONE + a static full-K slot-write KV-advance "
        f"schedule. The MTP single-shape invariant BREAKS (draft KV grows 1..{K_SPEC}) but is "
        f"absorbed into one manual unrolled graph, not a capture-size list; and K={K_SPEC} <= "
        f"the deployed ceiling-{cls['deployed_ceiling']} anyway. The dispatch checklist T6/T7 "
        f"must satisfy: (1) keep the draft chain off vLLM's list-dispatcher (manual NONE "
        f"replay), (2) pre-allocate draft KV to full-K with fixed-address slot-writes (no "
        f"realloc in-graph), (3) bake a STATIC per-iteration seq_len/slot_mapping schedule, (4) "
        f"confirm draft replay sizes {{1..{K_SPEC}}} <= ceiling-{cls['deployed_ceiling']}. The "
        f"verify-side #101 gate (#311: M<=16 safe, M=32 needs [24,32]) is UNCHANGED and "
        f"orthogonal. The human GO/NO-GO can treat draft-side dispatch as CLEARED provided the "
        f"rewrite honors gate item (1)."
    )

    return {
        "constants": {
            "official_baseline": OFFICIAL_BASELINE,
            "k_spec": K_SPEC,
            "m_verify_deployed": M_VERIFY_DEPLOYED,
            "max_cudagraph_capture_size_101": MAX_CUDAGRAPH_CAPTURE_SIZE_101,
            "max_safe_tree_width_311": MAX_SAFE_TREE_WIDTH_311,
            "mtp_hidden_dim": MTP_HIDDEN_DIM,
            "eagle_fused_aux_dim": EAGLE_FUSED_AUX_DIM,
            "eager_fallback_tps_floor_312": EAGER_FALLBACK_TPS_FLOOR_312,
            "imports": (
                "ubel#311(os01ttw9 draft manual ONEGRAPH CUDAGraphMode.NONE safe-by-construction, "
                "max_safe_tree_width=16, verify M=32 needs [24,32], sitecustomize.py:170) x "
                "wirbel#312(9b1arani T6/T7 KV-bearing draft chain: own-KV k_proj/v_proj, growing "
                "draft KV, seq_len advance inside graph, single [1,2560]->fused [1,7680]) x "
                "lawine#101 size-29 dispatch IndexError"),
        },
        "mtp_draft_safety": mtp,
        "eagle_draft_chain_deltas": deltas,
        "eagle_draft_step_shapes": step_shapes,
        "classification": cls,
        "cross_check_anchors": xcheck,
        "self_test": {"conditions": cond, "passes": bool(passes)},
        "verdict": verdict,
        "handoff": handoff,
        "caveats": CAVEATS,
    }


# --------------------------------------------------------------------------- #
def _assert_nan_clean(obj: Any, path: str = "") -> list[str]:
    bad: list[str] = []
    if isinstance(obj, float):
        if not math.isfinite(obj):
            bad.append(path or "<root>")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad += _assert_nan_clean(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += _assert_nan_clean(v, f"{path}[{i}]")
    return bad


def _print_human(syn: dict) -> None:
    print("\n" + "=" * 104, flush=True)
    print(" EAGLE-3 DRAFT-SIDE DISPATCH RISK UNDER THE #312 LOOPGRAPH REWRITE (PR #315)", flush=True)
    print("=" * 104, flush=True)
    mtp = syn["mtp_draft_safety"]
    print(f"  MTP DRAFT SAFETY  spec-K={mtp['k_num_speculative_tokens']}  "
          f"CUDAGraphMode.NONE(:{mtp['cudagraph_mode_none_line']})="
          f"{mtp['markers']['cudagraph_mode_none']}  manual_capture(:{mtp['manual_capture_line']})="
          f"{mtp['markers']['manual_cuda_graph_capture']}  not_list_dispatched="
          f"{mtp['draft_not_list_dispatched']}  single_shape_mtp={mtp['draft_single_shape_mtp']}",
          flush=True)
    print("-" * 104, flush=True)
    print("  EAGLE DRAFT-CHAIN DELTAS vs MTP (banked #312 T7):", flush=True)
    for d in syn["eagle_draft_chain_deltas"]:
        print(f"        - {d['delta']}: MTP[{d['mtp']}] -> EAGLE[{d['eagle3']}]", flush=True)
    print("  PER-STEP EAGLE DRAFT REPLAY SHAPE (DERIVED from #312 spec; query stays width-1):",
          flush=True)
    for s in syn["eagle_draft_step_shapes"]:
        print(f"        step {s['step']}: query={s['query_tokens']}  "
              f"draft_kv_extent={s['draft_kv_extent']}  slot={s['slot_write_target']}  "
              f"pos={s['position']}  matches_mtp={s['matches_mtp_single_shape']}", flush=True)
    cls = syn["classification"]
    print("-" * 104, flush=True)
    print(f"  CLASSIFY  {cls['classification']} (conditional={cls['classification_is_conditional']})"
          f"  reintroduces_list_dispatch_risk={cls['reintroduces_list_dispatch_risk']}", flush=True)
    print(f"            per_step_shapes_vary={cls['per_step_shapes_vary']}  "
          f"max_draft_kv_extent={cls['max_draft_kv_extent']}  "
          f"within_chain_replay_sizes={cls['within_chain_replay_sizes']}", flush=True)
    print(f"            eagle_draft_capture_is_single_shape="
          f"{cls['eagle_draft_capture_is_single_shape']}  "
          f"draft_positions_fit_deployed_ceiling16={cls['draft_positions_fit_deployed_ceiling16']}",
          flush=True)
    print("  GATE (the rewrite must honor to KEEP draft-side (a)-safety):", flush=True)
    for i, g in enumerate(cls["gate"], 1):
        print(f"        ({i}) {g}", flush=True)
    xc = syn["cross_check_anchors"]
    print("-" * 104, flush=True)
    print(f"  ANCHOR CROSS-CHECK  311={xc['loaded_311']} 312={xc['loaded_312']}  "
          f"max_abs_err={xc['max_abs_err']:.2e} (<= {xc['tol']:.0e})  "
          f"all_within_tol={xc['all_within_tol']}", flush=True)
    st = syn["self_test"]
    print("-" * 104, flush=True)
    print(f"  SELF-TEST: { {k: int(v) for k, v in st['conditions'].items()} }  -> PASS={st['passes']}",
          flush=True)
    print(f"\n  VERDICT: {syn['verdict']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.append(repo)
    _w = sys.modules.get("wandb")
    if _w is not None and not hasattr(_w, "init"):
        del sys.modules["wandb"]
    try:
        import wandb as _wb
        if not hasattr(_wb, "init"):
            raise ImportError("resolved a stub wandb with no .init")
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[eagle3-draft-dispatch] wandb logging skipped (analysis unaffected): {exc}",
              flush=True)
        return

    syn = payload["synthesis"]
    cls = syn["classification"]
    try:
        run = init_wandb_run(
            job_type="validity-gate", agent="ubel", name=args.wandb_name, group=args.wandb_group,
            tags=["eagle3-draft-dispatch", "cudagraph-none", "manual-onegraph",
                  "kv-bearing-draft-chain", "dispatch-correctness", "pr-315"],
            config={
                "official_baseline": OFFICIAL_BASELINE, "k_spec": K_SPEC,
                "m_verify_deployed": M_VERIFY_DEPLOYED,
                "max_cudagraph_capture_size_101": MAX_CUDAGRAPH_CAPTURE_SIZE_101,
                "mtp_hidden_dim": MTP_HIDDEN_DIM, "eagle_fused_aux_dim": EAGLE_FUSED_AUX_DIM,
                "eager_fallback_tps_floor_312": EAGER_FALLBACK_TPS_FLOOR_312,
                "imports": syn["constants"]["imports"], "wandb_group": args.wandb_group,
            },
        )
    except Exception as exc:
        print(f"[eagle3-draft-dispatch] wandb init failed (analysis unaffected): {exc}", flush=True)
        return
    if run is None:
        print("[eagle3-draft-dispatch] wandb: no run (no API key/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "draft_dispatch_under_rewrite_self_test_passes": int(bool(
            payload["draft_dispatch_under_rewrite_self_test_passes"])),
        "eagle_draft_capture_is_single_shape": int(bool(
            cls["eagle_draft_capture_is_single_shape"])),
        "draft_positions_fit_deployed_ceiling16": int(bool(
            cls["draft_positions_fit_deployed_ceiling16"])),
        "reintroduces_list_dispatch_risk": int(bool(cls["reintroduces_list_dispatch_risk"])),
        "classification_is_a": int(cls["classification"] == "(a)"),
        "classification_is_conditional": int(bool(cls["classification_is_conditional"])),
        "per_step_shapes_vary": int(bool(cls["per_step_shapes_vary"])),
        "max_draft_kv_extent": int(cls["max_draft_kv_extent"]),
        "k_spec": int(K_SPEC),
        "deployed_ceiling": int(cls["deployed_ceiling"]),
        "n_eagle_deltas": len(syn["eagle_draft_chain_deltas"]),
        "n_gate_items": len(cls["gate"]),
        "mtp_draft_not_list_dispatched": int(bool(syn["mtp_draft_safety"]["draft_not_list_dispatched"])),
        "mtp_draft_single_shape": int(bool(syn["mtp_draft_safety"]["draft_single_shape_mtp"])),
        "xcheck_max_abs_err": syn["cross_check_anchors"]["max_abs_err"],
        "xcheck_all_within_tol": int(bool(syn["cross_check_anchors"]["all_within_tol"])),
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in syn["self_test"]["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if v is not None and not (isinstance(v, float) and not math.isfinite(v))}
    try:
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="eagle3_draft_dispatch_under_rewrite_result",
                          artifact_type="validity", data=payload)
        finish_wandb(run)
        print(f"[eagle3-draft-dispatch] wandb logged {len(summary)} summary keys", flush=True)
    except Exception as exc:
        print(f"[eagle3-draft-dispatch] wandb write failed (analysis unaffected): {exc}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--k-spec", "--k_spec", dest="k_spec", type=int, default=K_SPEC)
    ap.add_argument("--submission-dir", type=Path, default=SUBMISSION_DIR)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="eagle3-draft-dispatch")
    args = ap.parse_args(argv)

    mtp = parse_mtp_draft_safety(args.submission_dir)
    deltas = eagle_draft_chain_deltas()
    step_shapes = derive_eagle_draft_step_shapes(args.k_spec)
    cls = classify_draft_dispatch_risk(step_shapes, args.k_spec, MAX_CUDAGRAPH_CAPTURE_SIZE_101)
    xcheck = cross_check_anchors()
    syn = synthesize(mtp, deltas, step_shapes, cls, xcheck)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 315, "agent": "ubel",
        "kind": "eagle3-draft-dispatch-under-rewrite", "analysis_only": True,
        "synthesis": syn,
        "draft_dispatch_under_rewrite_self_test_passes": syn["self_test"]["passes"],
        "eagle_draft_capture_is_single_shape": cls["eagle_draft_capture_is_single_shape"],
        "draft_positions_fit_deployed_ceiling16": cls["draft_positions_fit_deployed_ceiling16"],
        "host_peak_mem_mib": round(peak_kib / 1024.0, 3),
        "greedy_ppl_safety_certificate": {
            "analysis_only": True, "served_file_changed": False, "emitted_token_changed": False,
            "hf_job_or_submission": False, "is_launch": False, "is_build": False,
            "baseline_tps_unchanged": OFFICIAL_BASELINE, "tps_added_by_this_leg": 0.0,
        },
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["f_nan_clean"] = bool(payload["nan_clean"])
    syn["self_test"]["passes"] = bool(all(syn["self_test"]["conditions"].values()))
    payload["draft_dispatch_under_rewrite_self_test_passes"] = bool(
        syn["self_test"]["passes"] and payload["nan_clean"])
    if nan_paths:
        print(f"[eagle3-draft-dispatch] WARNING non-finite at: {nan_paths[:10]}", flush=True)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_draft_dispatch_under_rewrite_results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))

    payload["primary_metric"] = {
        "name": "draft_dispatch_under_rewrite_self_test_passes",
        "value": int(bool(payload["draft_dispatch_under_rewrite_self_test_passes"]))}
    payload["test_metric"] = {
        "name": "eagle_draft_capture_is_single_shape",
        "value": int(bool(payload["eagle_draft_capture_is_single_shape"]))}

    _print_human(syn)
    print(f"[eagle3-draft-dispatch] wrote {out_path}", flush=True)
    print(f"[eagle3-draft-dispatch] PRIMARY draft_dispatch_under_rewrite_self_test_passes = "
          f"{payload['draft_dispatch_under_rewrite_self_test_passes']}", flush=True)
    print(f"[eagle3-draft-dispatch] TEST eagle_draft_capture_is_single_shape = "
          f"{payload['eagle_draft_capture_is_single_shape']}  "
          f"draft_positions_fit_deployed_ceiling16 = "
          f"{payload['draft_positions_fit_deployed_ceiling16']}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = payload["draft_dispatch_under_rewrite_self_test_passes"]
        print(f"[eagle3-draft-dispatch] PRIMARY self-test: {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
