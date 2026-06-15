#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""EAGLE-3 served-path integration readiness: drop-in swap or boot-risk change? (PR #307)

THE QUESTION
------------
The EAGLE-3 build-economics matrix has priced nearly every COST axis -- STEP cost
(wirbel #295, c334qaqu, ~2.95x linear, validates 6.1245), VRAM fit (ubel #299,
jnoss7id, <=24GiB, reuses target embed/lm_head), per-position target (kanna #289 /
denken #297), companion floor (lawine #292/#296), build cost (denken #301), read
companion (fern #302), private-bar (lawine #300, 8t5q6sr0), numerator reachability
(denken #304, dtf1ouml). EVERY one of those axes silently ASSUMES the trained
drafter can be DEPLOYED into the served runner. That assumption is unpriced.

THE UNPRICED AXIS (this leg)
----------------------------
Is wiring a {2,21,39}-fusion EAGLE-3 drafter into the deployed
`submissions/fa2sw_precache_kenyan` served path a CONFIG/weights DROP-IN, or does it
require a served-file (or vLLM-fork) code change -- and if so, does that change
re-open the Issue #272 boot-500 risk (the `_guard_included_router` / prometheus
`_IncludedRouter` fragility that produces a `/v1/models` 500 -> 0 records on a fresh
runner)? This is a DEPLOYMENT-FEASIBILITY gate on the human GO/NO-GO, orthogonal to
every cost/economics axis.

WHAT THIS IS (and is NOT)
-------------------------
READ-ONLY STATIC ANALYSIS. NO served-file edit, NO build, NO boot test, NO HF Job,
NO submission change. 0 GPU. BASELINE 481.53 unchanged; this card adds 0 TPS -- it
prices DEPLOYMENT feasibility, not speed.

METHOD
------
1. Encode the integration delta linear-MTP K=7 -> {2,21,39}-fusion EAGLE-3 as a
   touchpoint table, each classified into exactly one of
   {config-only, served-file-change, fork-code-change}.
2. Make the analysis FALSIFIABLE: every touchpoint cites a (file, substring) that
   this script re-reads from the real repo and verifies present. The verdict is
   DERIVED from the partition, not asserted.
3. Cross-check vs Issue #272: grep the deployed `fa2sw_precache_kenyan` and the
   sibling `fa2sw_treeverify_kenyan` sitecustomize for `_guard_included_router`,
   and confirm the EAGLE-3 touchpoints live in the spec-decode / model-execution
   layer, NOT the FastAPI route-registration layer the boot-500 implicates.
4. Self-test: PRIMARY bool eagle3_integration_readiness_self_test_passes; TEST
   metric swap_is_config_only. Log all key fields under summary/.

DELIVERABLE
-----------
swap_is_config_only (bool), reuses_boot_fragile_path (bool),
eagle3_integration_readiness in {drop-in-config, served-file-change, fork-change},
readiness_blocks_go (bool), and the green/yellow/red hand-off for the GO/NO-GO packet.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

# Repo root: research/validity/eagle3_integration_readiness/<this> -> 3 parents up.
ROOT = Path(__file__).resolve().parents[3]

PRECACHE = "submissions/fa2sw_precache_kenyan"
TREEVERIFY = "submissions/fa2sw_treeverify_kenyan"

# Classification universe (mutually exclusive, exhaustive).
CONFIG_ONLY = "config-only"
SERVED_FILE = "served-file-change"
FORK_CODE = "fork-code-change"
CLASSES = (CONFIG_ONLY, SERVED_FILE, FORK_CODE)

# Readiness verdict universe.
DROP_IN = "drop-in-config"
SERVED_CHANGE = "served-file-change"
FORK_CHANGE = "fork-change"

OFFICIAL_BASELINE = 481.53          # PR #52, served public #1 (PPL 2.3772, 128/128).
PRIVATE_VERIFIED = 460.85           # organizer re-run, Delta 4.3% <= 5%.
EAGLE3_TARGET_LAYERS = (2, 21, 39)  # default [low, mid, high] for E4B 42-layer body.


# --------------------------------------------------------------------------- #
# The integration delta: linear-MTP K=7  ->  {2,21,39}-fusion EAGLE-3.
# Each touchpoint: id, name, what_changes, classification, evidence list of
# (repo-relative-file, substring-that-must-be-present), and a note.
# --------------------------------------------------------------------------- #
TOUCHPOINTS: list[dict[str, Any]] = [
    {
        "id": "T1",
        "name": "Speculative method + config",
        "what_changes": (
            'SPECULATIVE_CONFIG.method "mtp" -> "eagle3"; model -> EAGLE-3 ckpt dir; '
            "num_speculative_tokens kept (7); optional eagle_aux_hidden_state_layer_ids "
            "[2,21,39] (already the default). A manifest env edit consumed by vLLM "
            "--speculative-config."
        ),
        "classification": CONFIG_ONLY,
        "evidence": [
            # manifest stores SPECULATIVE_CONFIG as a JSON-encoded STRING, so the
            # literal file bytes carry escaped quotes: \"method\":\"mtp\".
            (f"{PRECACHE}/manifest.json", r'\"method\":\"mtp\"'),
            (f"{PRECACHE}/manifest.json", r'\"num_speculative_tokens\":7'),
            (f"{PRECACHE}/serve.py", 'append_env_arg(args, "SPECULATIVE_CONFIG", "--speculative-config")'),
        ],
        "note": "vLLM selects the proposer/head from method; manifest env, no served .py edit.",
    },
    {
        "id": "T2",
        "name": "Drafter weight artifact",
        "what_changes": (
            "DRAFTER_BUCKET -> EAGLE-3 checkpoint; ensure_drafter() hf-buckets-sync + "
            "sha256 + centroid_intermediate_top_k rewrite is GENERIC (same mechanism "
            "as the deployed retrained MTP drafter)."
        ),
        "classification": CONFIG_ONLY,
        "evidence": [
            (f"{PRECACHE}/serve.py", "def ensure_drafter"),
            (f"{PRECACHE}/manifest.json", "DRAFTER_BUCKET"),
        ],
        "note": (
            "Swap MECHANISM is config-only. CAVEAT: the EAGLE-3 ckpt does not exist yet "
            "(training-gated; arch_notes 'deployment gated on kanna #5'), and "
            "ensure_drafter writes the MTP-only key centroid_intermediate_top_k into the "
            "drafter config (spurious-but-harmless for an EAGLE-3 config). vLLM-load "
            "verification is DEFERRED (arch_notes §7). The TRAINING gate is a separate "
            "axis; here only the swap mechanism is priced."
        ),
    },
    {
        "id": "T3",
        "name": "Aux-hidden capture (target exposes layers 2/21/39)",
        "what_changes": (
            "vLLM auto-sets use_aux_hidden_state_outputs for method=='eagle3' and calls "
            "set_aux_hidden_state_layers; Gemma4Model already implements SupportsEagle3, "
            "so the (hidden_states, aux_hidden_states) tuple return + multi-layer cat "
            "fusion are built-in (no model-class surgery, no forward hooks)."
        ),
        "classification": CONFIG_ONLY,
        "evidence": [
            ("research/eagle3_feasibility/feasibility_report.md", "eagle3_hiddens_accessible"),
            ("research/eagle3_feasibility/feasibility_report.md", "SupportsEagle3"),
            ("research/eagle3_drafter/arch_notes.md", "(2, 21, 39)"),
        ],
        "note": (
            "Config-only on STOCK vLLM 0.22.x (feasibility PR #15: '0 hours of vLLM "
            "work'). CAVEAT: this submission source-patches gemma4.py (PLE fast-path / "
            "scale-fold) -- orthogonal to the aux-collection return (gemma4.py:1318, "
            "1337-1356) but co-resident, so non-interference needs re-validation."
        ),
    },
    {
        "id": "T4",
        "name": "Drafter head class",
        "what_changes": (
            "MTP head Gemma4MTP (get_top_tokens, single last-hidden) -> EAGLE-3 head "
            "Eagle3LlamaForCausalLM (Llama decoder layer + fc[7680->2560] fusion + own "
            "lm_head/compute_logits; its draft layer writes its OWN KV)."
        ),
        "classification": CONFIG_ONLY,
        "evidence": [
            ("research/eagle3_drafter/arch_notes.md", "Eagle3LlamaForCausalLM"),
            ("research/eagle3_feasibility/feasibility_report.md", "llama_eagle3.py"),
        ],
        "note": (
            "The head CLASS is selected by vLLM's registry from the method, so selection "
            "is config-only -- but the structural change (own-KV Llama layer, fused "
            "[7680] input, compute_logits not get_top_tokens) is precisely what breaks "
            "the MTP-keyed served patches T5/T6/T7 below."
        ),
    },
    {
        "id": "T5",
        "name": "Fused sparse-argmax patch (drafter head)",
        "what_changes": (
            "sitecustomize _apply_fused_top_token_patch is keyed to "
            "TOP_TOKEN_TARGET=vllm.model_executor.models.gemma4_mtp and wraps the MTP "
            "head's get_top_tokens. EAGLE-3's head has no get_top_tokens (uses "
            "compute_logits), so the patch goes INERT; retaining the fused-argmax kernel "
            "requires re-targeting it to the EAGLE head."
        ),
        "classification": SERVED_FILE,
        "evidence": [
            (f"{PRECACHE}/sitecustomize.py", 'TOP_TOKEN_TARGET = "vllm.model_executor.models.gemma4_mtp"'),
            (f"{PRECACHE}/sitecustomize.py", "self.model.get_top_tokens"),
        ],
        "note": "Inert-not-crash, but a served-file edit is required to KEEP the optimization.",
    },
    {
        "id": "T6",
        "name": "onegraph loopgraph capture (proposer)",
        "what_changes": (
            "sitecustomize _apply_loopgraph_patch is keyed to "
            "LOOPGRAPH_TARGET=vllm.v1.spec_decode.gemma4 and does "
            "proposer_cls = module.Gemma4Proposer. Under method=='eagle3' vLLM "
            "instantiates EagleProposer (vllm.v1.spec_decode.eagle), so the onegraph "
            "K=7 single-replay substrate -- the source of 481.53 TPS -- is never invoked "
            "(inert). Retaining it requires REWRITING the loopgraph for the EAGLE proposer."
        ),
        "classification": SERVED_FILE,
        "evidence": [
            (f"{PRECACHE}/sitecustomize.py", 'LOOPGRAPH_TARGET = "vllm.v1.spec_decode.gemma4"'),
            (f"{PRECACHE}/sitecustomize.py", "proposer_cls = module.Gemma4Proposer"),
            (f"{PRECACHE}/sitecustomize.py", "Patch vLLM Gemma4 MTP drafting"),
        ],
        "note": (
            "This is the load-bearing finding: the engine that makes the submission the "
            "frontier is MTP-specific and goes inert for EAGLE-3."
        ),
    },
    {
        "id": "T7",
        "name": "onegraph correctness invariant + static buffers / cudagraph sizing",
        "what_changes": (
            "The onegraph 'Q-only, KV-shared, width-1 is exact' correctness invariant is "
            "FALSE for EAGLE-3: the EAGLE draft Llama layer writes its own KV and has "
            "cross-position dependencies within the draft chain. _build_static_buffers / "
            "_run_graph_body / _capture_graph assume a single [1,2560] hidden buffer and "
            "K width-1 iterations; EAGLE-3 needs a fused [1,7680] aux input and a "
            "KV-bearing chain -- the buffers and capture must be re-derived and re-sized."
        ),
        "classification": SERVED_FILE,
        "evidence": [
            (f"{PRECACHE}/sitecustomize.py", "Width-1 is exact"),
            (f"{PRECACHE}/sitecustomize.py", "def _build_static_buffers"),
            (f"{PRECACHE}/sitecustomize.py", "def _capture_graph"),
        ],
        "note": (
            "Correctness re-derivation, not just re-pointing -- the deepest served-file "
            "change and the one most coupled to greedy-identity re-validation."
        ),
    },
    {
        "id": "T8",
        "name": "Shared-base fused-accept-prep patch",
        "what_changes": (
            "sitecustomize _apply_fused_accept_proposer_patch is keyed to "
            "PROPOSER_TARGET=vllm.v1.spec_decode.llm_base_proposer and wraps "
            "SpecDecodeBaseProposer.prepare_next_token_ids_padded -- the SHARED base that "
            "both the MTP and EAGLE proposers inherit, so it carries over unchanged."
        ),
        "classification": CONFIG_ONLY,
        "evidence": [
            (f"{PRECACHE}/sitecustomize.py", 'PROPOSER_TARGET = "vllm.v1.spec_decode.llm_base_proposer"'),
            (f"{PRECACHE}/sitecustomize.py", "proposer_cls = module.SpecDecodeBaseProposer"),
        ],
        "note": (
            "No edit needed (shared base). CAVEAT: the fused-accept fast path's "
            "num_reqs==1 / accept-geometry assumptions should be re-validated for EAGLE "
            "draft chains, but no served-file CHANGE is required to keep it functioning."
        ),
    },
    {
        "id": "T9",
        "name": "PLE gemma4.py source patches (orthogonality)",
        "what_changes": (
            "serve.py patch_gemma4_source (PLE valid-token fast path, embed-scale fold, "
            "scratch reuse) edits the target gemma4.py. These touch per-layer-embeddings "
            "/ embed_scale inside the decoder, NOT the Gemma4Model.forward aux-collection "
            "return path, so they are orthogonal to EAGLE-3 aux capture; no change needed."
        ),
        "classification": CONFIG_ONLY,
        "evidence": [
            (f"{PRECACHE}/serve.py", "def patch_gemma4_source"),
            (f"{PRECACHE}/serve.py", "PLE valid-token fast path"),
        ],
        "note": "No edit needed; non-interference with the aux tuple return should be re-validated.",
    },
]

# Issue #272 boot-500 cross-check: the deployed frontier is MISSING the guard the
# sibling submission carries. The EAGLE-3 touchpoints' target modules live in the
# spec-decode / model-execution / worker layer, NOT the FastAPI route-registration
# layer where _IncludedRouter / _get_route_name throws.
BOOT_FRAGILE_TOKEN = "_guard_included_router"
ROUTE_LAYER_PREFIXES = (
    "vllm.entrypoints",          # api_server / openai routers
    "prometheus",                # the instrumentator that throws
    "vllm.renderers",            # chat-template rendering at request time
)
SPEC_EXEC_LAYER_PREFIXES = (
    "vllm.v1.spec_decode",
    "vllm.model_executor",
    "vllm.v1.worker",
)
# Module targets that the EAGLE-3 touchpoints actually move through.
EAGLE3_TOUCHED_MODULES = (
    "vllm.v1.spec_decode.gemma4",            # MTP proposer (inert under eagle3)
    "vllm.v1.spec_decode.eagle",             # EAGLE proposer (the new active path)
    "vllm.v1.spec_decode.llm_base_proposer",  # shared base
    "vllm.model_executor.models.gemma4_mtp",  # MTP head (inert)
    "vllm.model_executor.models.llama_eagle3",  # EAGLE head
    "vllm.v1.worker.gpu_model_runner",        # aux-capture wiring
)


def _read(rel: str) -> str | None:
    p = ROOT / rel
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8", errors="replace")


def verify_evidence() -> dict[str, Any]:
    """Re-read every cited (file, substring) from the real repo and confirm present."""
    results = []
    all_ok = True
    for tp in TOUCHPOINTS:
        for rel, substr in tp["evidence"]:
            text = _read(rel)
            present = bool(text) and (substr in text)
            all_ok = all_ok and present
            results.append(
                {"touchpoint": tp["id"], "file": rel, "substring": substr, "present": present}
            )
    return {"all_present": all_ok, "checks": results}


def boot_272_crosscheck() -> dict[str, Any]:
    precache = _read(f"{PRECACHE}/sitecustomize.py") or ""
    treeverify = _read(f"{TREEVERIFY}/sitecustomize.py") or ""
    precache_guard = len(re.findall(BOOT_FRAGILE_TOKEN, precache))
    treeverify_guard = len(re.findall(BOOT_FRAGILE_TOKEN, treeverify))

    # Does ANY EAGLE-3-touched module sit in the route-registration layer?
    touches_route_layer = any(
        m.startswith(ROUTE_LAYER_PREFIXES) for m in EAGLE3_TOUCHED_MODULES
    )
    all_in_spec_exec = all(
        m.startswith(SPEC_EXEC_LAYER_PREFIXES) for m in EAGLE3_TOUCHED_MODULES
    )
    # The boot-500 guard token appears nowhere in the deployed frontier sitecustomize
    # (it is exactly what #272 flags); so the served path -- MTP today, EAGLE-3
    # tomorrow -- never goes through _guard_included_router either way.
    reuses_boot_fragile_path = touches_route_layer and (precache_guard > 0)
    return {
        "precache_guard_count": precache_guard,        # expect 0  (the #272 gap)
        "treeverify_guard_count": treeverify_guard,    # expect >0 (reference impl)
        "precache_missing_guard": precache_guard == 0,
        "sibling_has_guard": treeverify_guard > 0,
        "eagle3_touches_route_layer": touches_route_layer,
        "eagle3_all_in_spec_exec_layer": all_in_spec_exec,
        "reuses_boot_fragile_path": bool(reuses_boot_fragile_path),
    }


def classify() -> dict[str, Any]:
    counts = {c: 0 for c in CLASSES}
    for tp in TOUCHPOINTS:
        counts[tp["classification"]] += 1
    total = len(TOUCHPOINTS)

    classes_valid = all(tp["classification"] in CLASSES for tp in TOUCHPOINTS)
    # Each touchpoint holds exactly one class string -> partition is mutually
    # exclusive by construction; verify exhaustiveness (counts sum to total).
    exhaustive = sum(counts.values()) == total
    mutually_exclusive = classes_valid and exhaustive

    swap_is_config_only = (counts[SERVED_FILE] == 0 and counts[FORK_CODE] == 0)

    if counts[FORK_CODE] > 0:
        readiness = FORK_CHANGE
    elif counts[SERVED_FILE] > 0:
        readiness = SERVED_CHANGE
    else:
        readiness = DROP_IN

    # Verdict must follow from the partition: config-only <=> drop-in.
    verdict_follows = (swap_is_config_only == (readiness == DROP_IN))
    return {
        "counts": counts,
        "total": total,
        "classes_valid": classes_valid,
        "exhaustive": exhaustive,
        "mutually_exclusive": mutually_exclusive,
        "swap_is_config_only": bool(swap_is_config_only),
        "eagle3_integration_readiness": readiness,
        "verdict_follows": bool(verdict_follows),
    }


def synthesize() -> dict[str, Any]:
    cls = classify()
    ev = verify_evidence()
    boot = boot_272_crosscheck()

    # readiness_blocks_go: does deployment add a blocker BEYOND the economics, i.e.
    # is EAGLE-3 worse than the already-deployed, already-private-verified linear
    # path? True iff the swap is not config-only (a served-file rewrite is required).
    readiness_blocks_go = not cls["swap_is_config_only"]

    # No vendored vLLM fork exists (stock wheels.vllm.ai wheel); the EAGLE-3 head +
    # aux export are upstream, so there is NO fork-code-change path -- the worst case
    # is a served-file change.
    no_fork_path = cls["counts"][FORK_CODE] == 0

    # Hand-off light: GREEN drop-in, YELLOW served-file-change, RED fork-change.
    light = {
        DROP_IN: "green",
        SERVED_CHANGE: "yellow",
        FORK_CHANGE: "red",
    }[cls["eagle3_integration_readiness"]]

    conditions = {
        "evidence_all_present": ev["all_present"],
        "classes_valid": cls["classes_valid"],
        "partition_exhaustive": cls["exhaustive"],
        "partition_mutually_exclusive": cls["mutually_exclusive"],
        "verdict_follows_from_partition": cls["verdict_follows"],
        "readiness_is_served_file_change": cls["eagle3_integration_readiness"] == SERVED_CHANGE,
        "swap_is_not_config_only": not cls["swap_is_config_only"],
        # #272 cross-check cites real, measurable line counts.
        "precache_missing_guard_272": boot["precache_missing_guard"],
        "sibling_has_guard_272": boot["sibling_has_guard"],
        "eagle3_not_in_route_layer": not boot["eagle3_touches_route_layer"],
        "reuses_boot_fragile_path_is_false": not boot["reuses_boot_fragile_path"],
        "no_fork_code_path": no_fork_path,
        "readiness_blocks_go_true": readiness_blocks_go,
    }
    self_test_passes = all(bool(v) for v in conditions.values())

    metrics = {
        "official_baseline": OFFICIAL_BASELINE,
        "private_verified": PRIVATE_VERIFIED,
        "tps_added_by_this_card": 0.0,
        "n_touchpoints": float(cls["total"]),
        "n_config_only": float(cls["counts"][CONFIG_ONLY]),
        "n_served_file_change": float(cls["counts"][SERVED_FILE]),
        "n_fork_code_change": float(cls["counts"][FORK_CODE]),
        "precache_guard_count": float(boot["precache_guard_count"]),
        "treeverify_guard_count": float(boot["treeverify_guard_count"]),
    }
    nan_clean = all(
        isinstance(v, (int, float)) and math.isfinite(float(v)) for v in metrics.values()
    )

    return {
        "swap_is_config_only": cls["swap_is_config_only"],
        "reuses_boot_fragile_path": boot["reuses_boot_fragile_path"],
        "eagle3_integration_readiness": cls["eagle3_integration_readiness"],
        "readiness_blocks_go": bool(readiness_blocks_go),
        "handoff_light": light,
        "eagle3_integration_readiness_self_test_passes": bool(self_test_passes),
        "nan_clean": bool(nan_clean),
        "conditions": conditions,
        "classification": cls,
        "evidence": ev,
        "boot_272": boot,
        "metrics": metrics,
        "touchpoints": TOUCHPOINTS,
    }


def build_summary(syn: dict[str, Any]) -> dict[str, Any]:
    m = syn["metrics"]
    summary = {
        # PRIMARY + TEST.
        "eagle3_integration_readiness_self_test_passes": int(syn["eagle3_integration_readiness_self_test_passes"]),
        "swap_is_config_only": int(bool(syn["swap_is_config_only"])),
        # headline verdict fields.
        "reuses_boot_fragile_path": int(bool(syn["reuses_boot_fragile_path"])),
        "eagle3_integration_readiness": syn["eagle3_integration_readiness"],
        "readiness_blocks_go": int(bool(syn["readiness_blocks_go"])),
        "handoff_light": syn["handoff_light"],
        "nan_clean": int(bool(syn["nan_clean"])),
        # touchpoint partition.
        "n_touchpoints": int(m["n_touchpoints"]),
        "n_config_only": int(m["n_config_only"]),
        "n_served_file_change": int(m["n_served_file_change"]),
        "n_fork_code_change": int(m["n_fork_code_change"]),
        # #272 cross-check.
        "precache_guard_count": int(m["precache_guard_count"]),
        "treeverify_guard_count": int(m["treeverify_guard_count"]),
        # honest 0-TPS framing.
        "official_baseline": m["official_baseline"],
        "private_verified": m["private_verified"],
        "tps_added_by_this_card": m["tps_added_by_this_card"],
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
        print(f"[eagle3-integration-readiness] wandb logging skipped (analysis unaffected): {exc}", flush=True)
        return

    try:
        run = init_wandb_run(
            job_type="validity-gate",
            agent="wirbel",
            name=args.wandb_name or "wirbel/eagle3-integration-readiness",
            group=args.wandb_group,
            tags=[
                "eagle3-integration-readiness", "deployment-feasibility", "served-path",
                "drop-in-vs-served-change", "boot-272-crosscheck", "onegraph-mtp-specific",
                "pr-307", "zero-tps",
            ],
            config={
                "official_baseline": OFFICIAL_BASELINE,
                "private_verified": PRIVATE_VERIFIED,
                "eagle3_target_layers": list(EAGLE3_TARGET_LAYERS),
                "deployed_method": "mtp",
                "target_method": "eagle3",
                "deployed_submission": PRECACHE,
                "sibling_with_guard": TREEVERIFY,
                "wandb_group": args.wandb_group,
                "imports": (
                    "feasibility PR#15(fern: eagle3_hiddens_accessible=1, SupportsEagle3 "
                    "built-in, 0h vLLM work) x arch_notes PR#16(fern: Eagle3LlamaForCausalLM "
                    "Llama-layers, fc[7680->2560], aux{2,21,39}) x wirbel#295(c334qaqu step "
                    "~2.95x) x ubel#299(jnoss7id VRAM<=24GiB) x Issue#272(boot-500 missing "
                    "_guard_included_router on the active frontier)"
                ),
            },
        )
    except Exception as exc:
        print(f"[eagle3-integration-readiness] wandb init failed (analysis unaffected): {exc}", flush=True)
        return
    if run is None:
        print("[eagle3-integration-readiness] wandb: no run (no API key / disabled) — skipping", flush=True)
        return

    summary = build_summary(syn)
    log_summary(run, summary, step=0)
    try:
        log_json_artifact(
            run,
            name="eagle3_integration_readiness",
            artifact_type="validity-analysis",
            data={
                "verdict": {
                    "swap_is_config_only": syn["swap_is_config_only"],
                    "reuses_boot_fragile_path": syn["reuses_boot_fragile_path"],
                    "eagle3_integration_readiness": syn["eagle3_integration_readiness"],
                    "readiness_blocks_go": syn["readiness_blocks_go"],
                    "handoff_light": syn["handoff_light"],
                },
                "classification": syn["classification"],
                "boot_272": syn["boot_272"],
                "conditions": syn["conditions"],
                "touchpoints": syn["touchpoints"],
                "evidence": syn["evidence"],
            },
        )
    except Exception as exc:
        print(f"[eagle3-integration-readiness] wandb artifact skipped: {exc}", flush=True)
    finish_wandb(run)
    print(f"[eagle3-integration-readiness] wandb run logged: {getattr(run, 'id', '?')}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--no-wandb", action="store_true", help="skip W&B logging")
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument(
        "--wandb-group", "--wandb_group", dest="wandb_group", default="eagle3-integration-readiness"
    )
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args(argv)

    syn = synthesize()
    payload = {
        "pr": 307,
        "question": "EAGLE-3 served-path integration: drop-in swap or boot-risk change?",
        "verdict": {
            "swap_is_config_only": syn["swap_is_config_only"],
            "reuses_boot_fragile_path": syn["reuses_boot_fragile_path"],
            "eagle3_integration_readiness": syn["eagle3_integration_readiness"],
            "readiness_blocks_go": syn["readiness_blocks_go"],
            "handoff_light": syn["handoff_light"],
        },
        "eagle3_integration_readiness_self_test_passes": syn["eagle3_integration_readiness_self_test_passes"],
        "nan_clean": syn["nan_clean"],
        "summary": build_summary(syn),
        "conditions": syn["conditions"],
        "classification": syn["classification"],
        "boot_272": syn["boot_272"],
        "evidence": syn["evidence"],
        "touchpoints": syn["touchpoints"],
        "honest_framing": (
            "Prices DEPLOYMENT feasibility ONLY. BASELINE 481.53 unchanged; this card "
            "adds 0 TPS. No served-file edit, no build, no boot test, no HF Job, no "
            "submission change, 0 GPU."
        ),
        "handoff_sentence": (
            "GO/NO-GO packet: served-path integration of a {2,21,39}-fusion EAGLE-3 "
            "drafter is a YELLOW light -- the feature-export prerequisite is green "
            "(stock vLLM aux capture is built-in) but the swap is NOT the advertised "
            "config drop-in: the MTP-specific onegraph loopgraph that produces 481.53 "
            "goes inert under the EAGLE proposer and must be REWRITTEN (a served-file "
            "change to the same sitecustomize.py that #272 flags as missing the boot "
            "guard), then greedy-identity / PPL / boot / TPS re-validated before any "
            "launch."
        ),
    }

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_integration_readiness_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[eagle3-integration-readiness] wrote {out_path}", flush=True)

    v = payload["verdict"]
    print(
        "[eagle3-integration-readiness] VERDICT: "
        f"swap_is_config_only={v['swap_is_config_only']} "
        f"reuses_boot_fragile_path={v['reuses_boot_fragile_path']} "
        f"readiness={v['eagle3_integration_readiness']} "
        f"readiness_blocks_go={v['readiness_blocks_go']} "
        f"light={v['handoff_light']}",
        flush=True,
    )
    print(
        "[eagle3-integration-readiness] partition: "
        f"config-only={syn['classification']['counts'][CONFIG_ONLY]} "
        f"served-file={syn['classification']['counts'][SERVED_FILE]} "
        f"fork-code={syn['classification']['counts'][FORK_CODE]} | "
        f"#272 precache_guard={syn['boot_272']['precache_guard_count']} "
        f"treeverify_guard={syn['boot_272']['treeverify_guard_count']}",
        flush=True,
    )
    print(
        f"[eagle3-integration-readiness] self_test_passes={syn['eagle3_integration_readiness_self_test_passes']} "
        f"nan_clean={syn['nan_clean']}",
        flush=True,
    )

    if not args.no_wandb:
        log_wandb(syn, args)

    if args.self_test:
        if not syn["eagle3_integration_readiness_self_test_passes"]:
            failed = [k for k, val in syn["conditions"].items() if not val]
            print(f"[eagle3-integration-readiness] SELF-TEST FAILED: {failed}", flush=True)
            return 1
        print("[eagle3-integration-readiness] SELF-TEST PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
