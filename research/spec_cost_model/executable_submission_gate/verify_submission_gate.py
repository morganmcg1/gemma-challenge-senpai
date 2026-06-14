#!/usr/bin/env python3
"""PR #189 — Executable fail-closed MUST-RETAIN submission gate.

Converts ubel #186's *static* MUST-RETAIN manifest (a DOCUMENT) into an
*executable, fail-closed packaging gate* (an ENFORCEMENT). The manifest proved
that ONE dropped flag — the `relocate_salvaged_kv` device-vectorization
reverting to a 37-layer host Python loop — silently costs 85% of projected
throughput (516->77 TPS) with no submit-time warning. This gate makes that
regression impossible to ship silently.

`verify_submission_gate(build_env, build_introspection) ->`
    {packaging_verdict: GO|NO-GO, failing_rows[], per_row_assertions[],
     validity_class_failures[], binding_failure, ...}

It imports the #186 manifest JSON as the SOURCE OF ROWS + COSTS (it does NOT
re-derive any cost), walks all 22 enumerated flags, asserts each of the 19
MUST-RETAIN rows is present/correct and the TRAP (`LSK_SKIP_LAYERS`) is UNSET,
and is FAIL-CLOSED: any MUST-RETAIN FAIL, a present TRAP, or missing/unparseable
introspection for a row -> NO-GO (never silent-pass). On any violation it names
the exact failing row + its banked cost-of-omission.

Scope boundary (honest): STATIC flag presence/shape assertion only.
  * No GPU / vLLM / HF Job / submission / served-file change / kernel deploy.
  * Adds 0 TPS (primary = self-test). BASELINE stays 481.53. Greedy/PPL untouched.
  * It CONSUMES #186's costs; it does NOT re-derive them.
  * It does NOT run the numerical GO/NO-GO (fern #185's lane).
  * It does NOT run the output-validity gate boot/PPL/128 (denken's lane) — it
    asserts flag *presence/shape*; denken asserts output *correctness*. The two
    merge through `validity_seam`.
  * It authorizes nothing — a human still files `Approval request: HF job`.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "research/spec_cost_model/executable_submission_gate"
MANIFEST_PATH = (
    ROOT / "research/spec_cost_model/tree_submission_manifest/tree_submission_manifest.json"
)

# ---- status vocabulary -------------------------------------------------------
PASS = "PASS"      # row satisfied
FAIL = "FAIL"      # flag absent/wrong -> blocking for must_retain
MISSING = "MISSING"  # introspection/env absent or unparseable -> fail-closed blocking
WARN = "WARN"      # non-ideal but non-blocking (free/cosmetic rows)
INFO = "INFO"      # informational only (free rows)

BLOCKING_STATUS = (FAIL, MISSING)

# Banked numbers used ONLY to assert the gate flags the right cost (NOT derived
# here — these are the #186 manifest's own imported values, asserted in the
# self-test against the loaded JSON).
ROW1_BANKED_COST_TPS = 444.92351419266527  # h163 realizable(522.38) - hostloop(77.45)
PRECACHE_BANKED_COST_TPS = 18.334174130253853  # 3.526% of proj official descent


# =============================================================================
# Introspection schema — what the gate reads from an assembled build
# =============================================================================
INTROSPECTION_SCHEMA = {
    "build_env": {
        "type": "dict[str,str]",
        "doc": "the RESOLVED served env (manifest.json 'env' merged with runtime "
        "overrides). Read directly for served-flag rows; JSON-embedded values "
        "(num_speculative_tokens, temperature) are parsed out of SPECULATIVE_CONFIG "
        "/ OVERRIDE_GENERATION_CONFIG.",
        "keys_read": [
            "PRECACHE_BENCH", "PRECACHE_REQUIRE", "SPECULATIVE_CONFIG.num_speculative_tokens",
            "CENTROID_TOP_K", "ONEGRAPH", "LOOPGRAPH_REQUIRE_CAPTURE",
            "DIXIE_FUSED_ACCEPT_PREP", "DIXIE_SLIM_GREEDY", "LM_HEAD_PRUNE", "FA_SLIDING",
            "SPLITKV_VERIFY", "PLE_FOLD_EMBED_SCALE", "PLE_ASSUME_VALID_TOKEN_IDS",
            "FEOPT_ORJSON", "FASTRENDER", "DETOK_ENDONLY", "LD_PRELOAD",
            "PYTORCH_CUDA_ALLOC_CONF", "PERFORMANCE_MODE", "DRAFTER_BUCKET", "DRAFTER_SHA256",
            "OVERRIDE_GENERATION_CONFIG.temperature", "MAX_NUM_SEQS", "MAX_MODEL_LEN",
            "MAX_NUM_BATCHED_TOKENS", "DTYPE", "WEIGHTS_BUCKET", "LOCAL_MODEL_DIR",
            "PCK04_KEEPSET", "FUSED_SPARSE_ARGMAX_BLOCK", "LSK_SKIP_LAYERS",
            "STEPTIME", "FA_SLIDING_DIAG", "PROFILER_CONFIG",
        ],
    },
    "build_introspection": {
        "type": "dict[str,dict]",
        "doc": "a STRUCTURAL probe of the assembled accept/relocate/decode path "
        "(served sitecustomize.py / serve_patch_*.py / tree-build graph). One key "
        "per code-shaped row. A missing key (or a None field) is FAIL-CLOSED "
        "(-> MISSING -> NO-GO), never a silent pass.",
        "relocate_salvaged_kv": {
            "row": "row 1 (BUILD-BLOCKER, 85.17% binding cost)",
            "fields": {
                "device_vectorized_op_present": "bool — single fused [L,W,H,D] "
                "index_select+index_copy_ by DEVICE commit-index (one launch sequence)",
                "host_layer_loop_present": "bool — a per-layer Python for-loop over the "
                "~37 decoder layers doing D2H/H2D KV copies (the 145ms/call landmine)",
                "host_sync_in_relocate": "bool — a .item()/.cpu()/.tolist() host readout "
                "inside the relocate path (re-introduces sync, falls out of the graph)",
                "n_host_layer_iterations": "int — host-side per-layer iteration count "
                "(0 == fully vectorized single launch; >0 == host-bound)",
                "op_symbol": "str (evidence) — resolved relocate symbol name",
            },
            "pass_rule": "device_vectorized_op_present AND NOT host_layer_loop_present "
            "AND NOT host_sync_in_relocate AND n_host_layer_iterations == 0",
        },
        "accept_walk": {
            "row": "row 6 (capturability rule)",
            "fields": {
                "device_argmax_accept_len_present": "bool — match-mask->cumprod->"
                "device-argmax accept length on device (vLLM-v1 RejectionSampler zero-sync)",
                "host_item_call_present": "bool — a per-node .item() host readout in the "
                "accept walk (sync-bound; breaks CUDA-graph capture)",
            },
            "pass_rule": "device_argmax_accept_len_present AND NOT host_item_call_present",
        },
        "decode_logits": {
            "row": "row 4/5 (denominator lever; DOUBLE-LOAD-BEARING validity seam)",
            "fields": {
                "decode_argmax_only": "bool — greedy token-selection uses "
                "argmax(pruned[M,12288])->kept_ids remap (no full 262144 scatter on decode)",
                "prefill_full_scatter_lp_retained": "bool — the FULL scatter[*,262144]+"
                "logprobs is RETAINED on the prompt_logprobs/prefill PPL path (the "
                "double-load-bearing seam; dropping it THERE breaks PPL)",
            },
            "pass_rule": "decode_argmax_only AND prefill_full_scatter_lp_retained",
            "validity_note": "prefill_full_scatter_lp_retained==False is a PPL-breaking "
            "validity failure (denken's output gate confirms numerically); "
            "decode_argmax_only==False alone is a speed-only revert.",
        },
    },
}

ROW1_CHECK_LOGIC = {
    "deliverable": "distinguish the device-vectorized relocate from a 37-layer host "
    "Python loop from a STATIC introspection of the assembled accept/relocate path.",
    "pass": "device_vectorized_op_present == True (fused [L,W,H,D] index_select+"
    "index_copy_ by device commit-index) AND host_layer_loop_present == False AND "
    "host_sync_in_relocate == False AND n_host_layer_iterations == 0.",
    "fail_host_loop": "host_layer_loop_present == True OR n_host_layer_iterations > 0 "
    "OR host_sync_in_relocate == True (a data-dependent Python loop over 37 layers "
    "CANNOT be CUDA-graph-captured -> pins the step host-bound ~122ms instead of the "
    "9.7ms captured target -> descent E[T]=5.04 (->522 TPS) collapses to ~77 TPS).",
    "fail_absent": "device_vectorized_op_present == False -> relocate op not assembled.",
    "missing": "no relocate_salvaged_kv introspection key, or n_host_layer_iterations "
    "is None -> MISSING -> NO-GO (fail-closed, NOT silent-pass).",
    "banked_cost_on_fail_tps": ROW1_BANKED_COST_TPS,
    "banked_cost_pct_of_official": 85.17267846343493,
    "source_leg": "#157 / #163 (imported via #186 manifest row 1)",
}

VALIDITY_SEAM = {
    "purpose": "the 5 DOUBLE-LOAD-BEARING rows break PPL/greedy/scoring-basis (not "
    "just speed) if dropped. The gate emits them in `validity_class_failures` so its "
    "STATIC packaging verdict can merge with denken's DYNAMIC output-validity preflight "
    "(boot/PPL/128/greedy) into one launch-readiness surface.",
    "boundary": "ubel #189 asserts flag PRESENCE/SHAPE only. denken asserts output "
    "CORRECTNESS (boot, PPL<=cap, completed==128, greedy-identical). This gate does "
    "NOT re-implement the PPL/greedy/128 check.",
    "double_load_bearing_rows": [
        "decode-path argmax-only logits (prefill scatter+LP seam) -> PPL",
        "OVERRIDE_GENERATION_CONFIG temperature=0.0 -> greedy token-identity (#124)",
        "MAX_NUM_SEQS=1/MAX_MODEL_LEN=4096/.../DTYPE=bfloat16 -> PPL/throughput basis",
        "WEIGHTS_BUCKET/LOCAL_MODEL_DIR/PCK04_KEEPSET -> model identity/artifact",
        "LSK_SKIP_LAYERS (TRAP, must stay UNSET) -> output (decoder layer-skip)",
    ],
    "shared_schema": {
        "packaging_gate": {
            "owner": "ubel #189 (this gate)",
            "verdict": "GO|NO-GO",
            "failing_rows": "list[{flag, status, banked_cost_tps}]",
            "validity_class_failures": "list[{flag, status, breaks, detail}]",
        },
        "output_gate": {
            "owner": "denken (NOT computed here — denken fills this)",
            "boot_ok": "bool", "ppl": "float<=2.42", "completed": "int==128",
            "greedy_identical": "bool",
        },
    },
    "combined_rule": "launch_ready_validity = packaging_gate.verdict==GO AND "
    "output_gate.{boot_ok, ppl<=cap, completed==128, greedy_identical} all-pass.",
    "feeds": "fern #185's numerical GO/NO-GO ledger as the `packaging-gate: GO` "
    "precondition row.",
}


# =============================================================================
# env helpers
# =============================================================================
def _present(env, key):
    v = env.get(key)
    return v is not None and str(v).strip() != ""


def _truthy(env, key):
    return str(env.get(key, "")).strip() in ("1", "true", "True", "TRUE", "yes")


def _eq(env, key, expected):
    return _present(env, key) and str(env.get(key)) == str(expected)


def _json_field(env, key, field):
    """Parse a JSON-string env var and pull a field. Returns (status, value)."""
    raw = env.get(key)
    if not raw:
        return MISSING, None
    try:
        obj = json.loads(raw)
    except Exception:  # noqa: BLE001
        return MISSING, None
    if not isinstance(obj, dict) or field not in obj:
        return MISSING, None
    return PASS, obj[field]


# =============================================================================
# per-row checkers  (env, introspection) -> (status, detail)
# =============================================================================
def chk_relocate(env, intro):  # row 1 — STRUCTURAL (the deliverable)
    node = intro.get("relocate_salvaged_kv")
    if not isinstance(node, dict):
        return MISSING, "no relocate_salvaged_kv introspection (fail-closed -> NO-GO)"
    n_host = node.get("n_host_layer_iterations")
    if n_host is None:
        return MISSING, "n_host_layer_iterations absent (fail-closed -> NO-GO)"
    dev = bool(node.get("device_vectorized_op_present"))
    host_loop = bool(node.get("host_layer_loop_present"))
    host_sync = bool(node.get("host_sync_in_relocate"))
    if dev and not host_loop and not host_sync and int(n_host) == 0:
        return PASS, (
            f"device-vectorized [L,W,H,D] index_select+index_copy_ "
            f"(op={node.get('op_symbol')}, n_host_iter=0)"
        )
    reasons = []
    if not dev:
        reasons.append("device-vectorized op ABSENT")
    if host_loop:
        reasons.append("37-layer host Python loop PRESENT (145ms/call landmine)")
    if host_sync:
        reasons.append(".item()/.cpu() host readout in relocate path")
    if int(n_host) != 0:
        reasons.append(f"n_host_layer_iterations={n_host}")
    return FAIL, "; ".join(reasons) + " -> host-bound, NOT CUDA-graph-capturable (516->77 TPS)"


def chk_accept_walk(env, intro):  # row 6 — STRUCTURAL
    node = intro.get("accept_walk")
    if not isinstance(node, dict):
        return MISSING, "no accept_walk introspection (fail-closed -> NO-GO)"
    if node.get("host_item_call_present") is None:
        return MISSING, "host_item_call_present absent (fail-closed -> NO-GO)"
    dev = bool(node.get("device_argmax_accept_len_present"))
    host_item = bool(node.get("host_item_call_present"))
    if dev and not host_item:
        return PASS, "sync-free device accept length (match-mask->cumprod->device-argmax, no .item())"
    reasons = []
    if not dev:
        reasons.append("device accept-length ABSENT")
    if host_item:
        reasons.append(".item() per node -> sync-bound, breaks CUDA-graph capture")
    return FAIL, "; ".join(reasons)


def chk_decode_logits(env, intro):  # row 4/5 — STRUCTURAL (double-load-bearing)
    node = intro.get("decode_logits")
    if not isinstance(node, dict):
        return MISSING, "no decode_logits introspection (fail-closed -> NO-GO)"
    am = node.get("decode_argmax_only")
    pf = node.get("prefill_full_scatter_lp_retained")
    if am is None or pf is None:
        return MISSING, "decode_argmax_only / prefill_full_scatter_lp_retained absent (fail-closed)"
    if bool(am) and bool(pf):
        return PASS, "decode argmax-only (pruned[M,12288]->kept_ids) WITH full scatter+LP retained on prefill PPL path"
    reasons = []
    if not bool(am):
        reasons.append("decode reverted to full scatter[M,262144]+LP (speed -1.11% step)")
    if not bool(pf):
        reasons.append("PREFILL full scatter+LP DROPPED -> BREAKS PPL (validity)")
    return FAIL, "; ".join(reasons)


def chk_precache(env, intro):  # row 2 — ENV
    if not _present(env, "PRECACHE_BENCH"):
        return FAIL, "PRECACHE_BENCH unset (fail-closed); prefill re-enters timed window (3.526%)"
    if not _truthy(env, "PRECACHE_BENCH"):
        return FAIL, f"PRECACHE_BENCH={env.get('PRECACHE_BENCH')} (must be 1); prefill re-enters timed window (3.526%)"
    note = "PRECACHE_BENCH=1"
    note += " + PRECACHE_REQUIRE=1 (fail-closed)" if _truthy(env, "PRECACHE_REQUIRE") else " (note: PRECACHE_REQUIRE not fail-closed)"
    return PASS, note


def chk_num_spec(env, intro):  # row 3 — ENV-JSON
    st, k = _json_field(env, "SPECULATIVE_CONFIG", "num_speculative_tokens")
    if st == MISSING:
        return MISSING, "SPECULATIVE_CONFIG absent/unparseable or num_speculative_tokens missing (fail-closed)"
    if int(k) != 7:
        return FAIL, f"num_speculative_tokens={k} (must be 7; K8/K9 -13/-16 TPS, inverted-U optimum K7)"
    return PASS, "num_speculative_tokens=7 (MTP draft sweet spot)"


def chk_centroid(env, intro):  # row — ENV
    if not _present(env, "CENTROID_TOP_K"):
        return FAIL, "CENTROID_TOP_K unset (fail-closed)"
    if str(env.get("CENTROID_TOP_K")) != "64":
        return FAIL, f"CENTROID_TOP_K={env.get('CENTROID_TOP_K')} (must be 64; topk128 -3.9 TPS, no accept gain)"
    return PASS, "CENTROID_TOP_K=64 (optimum)"


def chk_onegraph(env, intro):  # row — ENV
    miss = [k for k in ("ONEGRAPH", "LOOPGRAPH_REQUIRE_CAPTURE") if not _truthy(env, k)]
    if miss:
        return FAIL, f"{'/'.join(miss)} not =1 -> drafter propose loop falls to eager (capture-class)"
    return PASS, "ONEGRAPH=1 + LOOPGRAPH_REQUIRE_CAPTURE=1"


def chk_dixie(env, intro):  # row — ENV
    miss = [k for k in ("DIXIE_FUSED_ACCEPT_PREP", "DIXIE_SLIM_GREEDY") if not _truthy(env, k)]
    if miss:
        return FAIL, f"{'/'.join(miss)} not =1 -> accept-prep leaves the device-resident Triton kernel"
    return PASS, "DIXIE_FUSED_ACCEPT_PREP=1 + DIXIE_SLIM_GREEDY=1"


def chk_lmhead_prune(env, intro):  # row — ENV
    if not _truthy(env, "LM_HEAD_PRUNE"):
        return FAIL, "LM_HEAD_PRUNE not =1 -> 12k-head GEMM lost"
    return PASS, "LM_HEAD_PRUNE=1 (12k pruned head)"


def chk_fa_sliding(env, intro):  # row — ENV
    if not _truthy(env, "FA_SLIDING"):
        return FAIL, "FA_SLIDING not =1 -> FA2 sliding-window backend lost"
    return PASS, "FA_SLIDING=1"


def chk_splitkv(env, intro):  # row — ENV
    if not _truthy(env, "SPLITKV_VERIFY"):
        return FAIL, "SPLITKV_VERIFY not =1 -> 3D split-KV verify path lost"
    return PASS, "SPLITKV_VERIFY=1"


def chk_ple(env, intro):  # row — ENV
    miss = [k for k in ("PLE_FOLD_EMBED_SCALE", "PLE_ASSUME_VALID_TOKEN_IDS") if not _truthy(env, k)]
    if miss:
        return FAIL, f"{'/'.join(miss)} not =1 -> PLE fold/fastpath lost"
    return PASS, "PLE_FOLD_EMBED_SCALE=1 + PLE_ASSUME_VALID_TOKEN_IDS=1"


def chk_feopt(env, intro):  # row — ENV
    miss = [k for k in ("FEOPT_ORJSON", "FASTRENDER", "DETOK_ENDONLY") if not _truthy(env, k)]
    if miss:
        return FAIL, f"{'/'.join(miss)} not =1 -> front-end/detok speed lost"
    return PASS, "FEOPT_ORJSON=1 / FASTRENDER=1 / DETOK_ENDONLY=1"


def chk_runtime_alloc(env, intro):  # row — ENV
    reasons = []
    if "tcmalloc" not in str(env.get("LD_PRELOAD", "")):
        reasons.append("LD_PRELOAD missing tcmalloc")
    if not _present(env, "PYTORCH_CUDA_ALLOC_CONF"):
        reasons.append("PYTORCH_CUDA_ALLOC_CONF unset")
    if env.get("PERFORMANCE_MODE") != "interactivity":
        reasons.append(f"PERFORMANCE_MODE={env.get('PERFORMANCE_MODE')} (want interactivity)")
    if reasons:
        return FAIL, "; ".join(reasons)
    return PASS, "tcmalloc + PYTORCH_CUDA_ALLOC_CONF + PERFORMANCE_MODE=interactivity"


def chk_drafter(env, intro):  # row — ENV
    bucket = str(env.get("DRAFTER_BUCKET", ""))
    if not bucket:
        return FAIL, "DRAFTER_BUCKET unset (fail-closed) -> wrong/absent drafter -> E[accept] drops"
    if "ft-v1-epoch_001" not in bucket:
        return FAIL, f"DRAFTER_BUCKET not the ft-v1-epoch_001 drafter ({bucket})"
    sha = " (+DRAFTER_SHA256 guard)" if _present(env, "DRAFTER_SHA256") else " (note: DRAFTER_SHA256 guard absent)"
    return PASS, "DRAFTER_BUCKET=ft-v1-epoch_001" + sha


def chk_temperature(env, intro):  # row — ENV-JSON (double-load-bearing, VALIDITY)
    st, t = _json_field(env, "OVERRIDE_GENERATION_CONFIG", "temperature")
    if st == MISSING:
        return MISSING, "OVERRIDE_GENERATION_CONFIG absent/unparseable or temperature missing (fail-closed)"
    if float(t) != 0.0:
        return FAIL, f"temperature={t} (must be 0.0; breaks greedy token-identity, Issue #124)"
    return PASS, "temperature=0.0 (greedy identity)"


def chk_scoring_contract(env, intro):  # row — ENV (double-load-bearing, VALIDITY)
    reasons = []
    for key, want in (("MAX_NUM_SEQS", "1"), ("MAX_MODEL_LEN", "4096"), ("MAX_NUM_BATCHED_TOKENS", "512")):
        if str(env.get(key)) != want:
            reasons.append(f"{key}={env.get(key)} (want {want})")
    dtype = env.get("DTYPE")
    if dtype is not None and str(dtype).lower() not in ("bfloat16", "bf16", "auto"):
        reasons.append(f"DTYPE={dtype} (must be bfloat16/auto; downcast perturbs PPL)")
    if reasons:
        return FAIL, "; ".join(reasons) + " -> perturbs PPL/throughput scoring basis (validity)"
    dt = env.get("DTYPE") or "native-bf16"
    return PASS, f"MAX_NUM_SEQS=1 / MAX_MODEL_LEN=4096 / MAX_NUM_BATCHED_TOKENS=512 / DTYPE={dt}"


def chk_model_artifact(env, intro):  # row — ENV (double-load-bearing, VALIDITY)
    reasons = []
    if not _present(env, "WEIGHTS_BUCKET"):
        reasons.append("WEIGHTS_BUCKET unset")
    if not _present(env, "LOCAL_MODEL_DIR"):
        reasons.append("LOCAL_MODEL_DIR unset")
    ks = env.get("PCK04_KEEPSET")
    if not ks:
        reasons.append("PCK04_KEEPSET unset")
    elif _present(env, "LOCAL_MODEL_DIR") and not str(ks).startswith(str(env["LOCAL_MODEL_DIR"])):
        reasons.append(f"PCK04_KEEPSET ({ks}) not under LOCAL_MODEL_DIR ({env.get('LOCAL_MODEL_DIR')})")
    if reasons:
        return FAIL, "; ".join(reasons) + " -> wrong/absent model artifact (validity)"
    return PASS, "WEIGHTS_BUCKET + LOCAL_MODEL_DIR + PCK04_KEEPSET consistent (int4-pck04 baked dir)"


def chk_fused_block(env, intro):  # row — FREE (must_retain=False, non-blocking)
    v = env.get("FUSED_SPARSE_ARGMAX_BLOCK")
    if v is None:
        return INFO, "FUSED_SPARSE_ARGMAX_BLOCK unset (default; K-neutral, free)"
    if str(v) in ("16", "64"):
        return PASS, f"FUSED_SPARSE_ARGMAX_BLOCK={v} (K-neutral 16/64, free)"
    return WARN, f"FUSED_SPARSE_ARGMAX_BLOCK={v} (expected 16/64; free/cosmetic, non-blocking)"


def chk_logging(env, intro):  # row — FREE (must_retain=False, non-blocking)
    return INFO, "logging/setup flags (no served-compute effect; free)"


def chk_trap_lsk(env, intro):  # row — TRAP (must_retain=True; asserts ABSENT)
    v = env.get("LSK_SKIP_LAYERS")
    if v is None or str(v).strip() == "":
        return PASS, "LSK_SKIP_LAYERS UNSET (TRAP absent)"
    return FAIL, f"LSK_SKIP_LAYERS={v} SET -> drops decoder layers, BREAKS OUTPUT (TRAP, validity)"


def chk_diag(env, intro):  # row — FREE/INERT (must_retain=False; should stay off)
    on = []
    for k in ("STEPTIME", "FA_SLIDING_DIAG", "PROFILER_CONFIG"):
        v = env.get(k)
        if v is not None and str(v).strip() not in ("", "0"):
            on.append(f"{k}={v}")
    if on:
        return WARN, f"diagnostic probe(s) ON: {', '.join(on)} (should stay off; free/inert, non-blocking)"
    return PASS, "diagnostics off/unset"


# Registry: unique flag-substring -> (checker, check_kind). Each of the 22
# manifest flags must match EXACTLY ONE key (asserted at gate time, fail-closed).
CHECKERS = {
    "relocate_salvaged_kv": (chk_relocate, "structural"),
    "PRECACHE_BENCH=1": (chk_precache, "env"),
    "num_speculative_tokens == 7": (chk_num_spec, "env-json"),
    "decode-path argmax-only": (chk_decode_logits, "structural"),
    "CENTROID_TOP_K == 64": (chk_centroid, "env"),
    "accept-walk == sync-free": (chk_accept_walk, "structural"),
    "ONEGRAPH=1": (chk_onegraph, "env"),
    "DIXIE_FUSED_ACCEPT_PREP=1": (chk_dixie, "env"),
    "LM_HEAD_PRUNE=1": (chk_lmhead_prune, "env"),
    "FA_SLIDING=1": (chk_fa_sliding, "env"),
    "SPLITKV_VERIFY=1": (chk_splitkv, "env"),
    "PLE_FOLD_EMBED_SCALE=1": (chk_ple, "env"),
    "FEOPT_ORJSON=1": (chk_feopt, "env"),
    "LD_PRELOAD=tcmalloc": (chk_runtime_alloc, "env"),
    "DRAFTER_BUCKET=": (chk_drafter, "env"),
    "OVERRIDE_GENERATION_CONFIG temperature=0.0": (chk_temperature, "env-json"),
    "MAX_NUM_SEQS=1 / MAX_MODEL_LEN=4096": (chk_scoring_contract, "env"),
    "WEIGHTS_BUCKET/LOCAL_MODEL_DIR": (chk_model_artifact, "env"),
    "FUSED_SPARSE_ARGMAX_BLOCK": (chk_fused_block, "env"),
    "UVICORN_LOG_LEVEL": (chk_logging, "env"),
    "LSK_SKIP_LAYERS": (chk_trap_lsk, "trap"),
    "STEPTIME/FA_SLIDING_DIAG": (chk_diag, "env"),
}


def _breaks_class(flag):
    if "LSK_SKIP_LAYERS" in flag:
        return "output (decoder layer-skip)"
    if "temperature" in flag:
        return "greedy token-identity (#124)"
    if "MAX_NUM_SEQS" in flag:
        return "PPL/throughput scoring basis"
    if "WEIGHTS_BUCKET" in flag:
        return "model identity/artifact"
    if "decode-path argmax-only" in flag:
        return "PPL (prefill scatter+LP seam)"
    return "validity"


# =============================================================================
# The gate
# =============================================================================
def load_manifest(path: pathlib.Path = MANIFEST_PATH) -> dict:
    """Import the #186 MUST-RETAIN manifest (source of rows + banked costs)."""
    return json.loads(path.read_text(encoding="utf-8"))


def verify_submission_gate(build_env: dict, build_introspection: dict, manifest: dict | None = None) -> dict:
    """Fail-closed packaging gate. Walks all 22 manifest rows, asserts each
    MUST-RETAIN row present/correct + the TRAP absent, returns GO/NO-GO with the
    banked Δ-cost attached to every FAIL. Costs are IMPORTED from the manifest,
    not re-derived."""
    if manifest is None:
        manifest = load_manifest()
    man = manifest["manifest"]
    rows = list(man["must_retain_manifest"]) + list(man["served_surface_enumeration"])

    per_row, failing, validity_failures, construction_errors = [], [], [], []
    for row in rows:
        flag = row["flag"]
        must = bool(row.get("must_retain"))
        dlb = bool(row.get("double_load_bearing"))
        is_trap = str(row.get("klass", "")).startswith("TRAP")
        cost = row.get("cost_sort_tps")  # None for the 14 non-banked rows

        matched = [k for k in CHECKERS if k in flag]
        if len(matched) != 1:
            # gate-construction error -> fail-closed (never silent-pass a must_retain row)
            status, detail, kind = MISSING, (
                f"gate has {len(matched)} checkers for this row (expected 1) -> fail-closed"
            ), "unmatched"
            construction_errors.append({"flag": flag, "n_matched": len(matched)})
        else:
            checker, kind = CHECKERS[matched[0]]
            try:
                status, detail = checker(build_env, build_introspection)
            except Exception as e:  # noqa: BLE001 — any checker error is fail-closed
                status, detail = MISSING, f"checker raised {e!r} (fail-closed)"

        rowres = {
            "flag": flag,
            "klass": row.get("klass"),
            "surface": row.get("surface", "served_env"),
            "check_kind": kind,
            "must_retain": must,
            "double_load_bearing": dlb,
            "is_trap": is_trap,
            "greedy_ppl_safe": row.get("greedy_ppl_safe"),
            "status": status,
            "detail": detail,
            "banked_cost_tps": cost,
        }
        per_row.append(rowres)

        if must and status in BLOCKING_STATUS:
            failing.append(rowres)
            if dlb or is_trap:
                validity_failures.append({
                    "flag": flag,
                    "status": status,
                    "breaks": _breaks_class(flag),
                    "detail": detail,
                })

    verdict = "NO-GO" if failing else "GO"
    banked_fail = [r for r in failing if r["banked_cost_tps"] is not None]
    binding = max(banked_fail, key=lambda r: r["banked_cost_tps"]) if banked_fail else None
    total_banked_cost = sum(r["banked_cost_tps"] for r in banked_fail)

    return {
        "packaging_verdict": verdict,
        "n_rows_checked": len(per_row),
        "n_must_retain": sum(r["must_retain"] for r in per_row),
        "n_failing": len(failing),
        "failing_rows": failing,
        "validity_class_failures": validity_failures,
        "binding_failure": binding,
        "total_banked_cost_tps_at_risk": total_banked_cost,
        "construction_errors": construction_errors,
        "per_row_assertions": per_row,
    }


# =============================================================================
# Fixtures — synthetic/mutated build-env + introspection for the self-test
# =============================================================================
# The faithful shipped fa2sw_precache_kenyan served env (submissions/
# fa2sw_precache_kenyan/manifest.json 'env'). The GO baseline.
FA2SW_PRECACHE_KENYAN_ENV = {
    "WEIGHTS_BUCKET": "hf://buckets/gemma-challenge/gemma-chiku-inu/weights/osoi5-v0-baked",
    "LOCAL_MODEL_DIR": "/tmp/osoi5-v0-baked",
    "MAX_MODEL_LEN": "4096",
    "GPU_MEMORY_UTILIZATION": "0.90",
    "MAX_NUM_BATCHED_TOKENS": "512",
    "MAX_NUM_SEQS": "1",
    "PERFORMANCE_MODE": "interactivity",
    "CENTROID_TOP_K": "64",
    "SPECULATIVE_CONFIG": '{"method":"mtp","model":"/tmp/qat-assistant","num_speculative_tokens":7}',
    "GENERATION_CONFIG": "vllm",
    "OVERRIDE_GENERATION_CONFIG": '{"temperature":0.0,"top_p":1.0,"top_k":0}',
    "DISABLE_LOG_STATS": "1",
    "PLE_ASSUME_VALID_TOKEN_IDS": "1",
    "PLE_FOLD_EMBED_SCALE": "1",
    "PLE_FOLD_TARGET_MODEL": "/tmp/osoi5-v0-baked",
    "PLE_SCRATCH_REUSE": "1",
    "FUSED_SPARSE_ARGMAX": "1",
    "FUSED_SPARSE_ARGMAX_REQUIRE": "1",
    "FUSED_SPARSE_ARGMAX_BLOCK": "16",
    "LOOPGRAPH_REQUIRE_CAPTURE": "1",
    "LOOPGRAPH_WARMUP_CALLS": "20",
    "LOOPGRAPH_PINGPONG_SLOTS": "3",
    "UVICORN_LOG_LEVEL": "warning",
    "PATCH_BENCH_JINJA2": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,expandable_segments:True",
    "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4",
    "DIXIE_SLIM_GREEDY": "1",
    "DIXIE_FUSED_ACCEPT_PREP": "1",
    "DIXIE_FUSED_ACCEPT_PREP_REQUIRE": "1",
    "DIXIE_PREWARM_GREEDY_KERNEL": "1",
    "ONEGRAPH": "1",
    "PCK04_KEEPSET": "/tmp/osoi5-v0-baked/pck04_keepset.json",
    "DRAFTER_BUCKET": "hf://buckets/gemma-challenge/gemma-kenyan-duma/weights/drafter-ft/ft-v1-epoch_001",
    "DRAFTER_SHA256": "ed159e334999fd6b5f2d0dbad026346d4efac89eb7c6f55c5cdb042eca5dd18e",
    "FEOPT_ORJSON": "1",
    "FASTRENDER": "1",
    "DETOK_ENDONLY": "1",
    "LM_HEAD_PRUNE": "1",
    "LM_HEAD_PRUNE_REQUIRE": "1",
    "LM_HEAD_PRUNE_DST": "/tmp/osoi5-12k-baked",
    "LM_HEAD_KEEPSET_BUCKET": "hf://buckets/gemma-challenge/gemma-dixie-flatline/weights/int4-pck04c-12k",
    "FA_SLIDING": "1",
    "FA_SLIDING_DIAG": "0",
    "SPLITKV_VERIFY": "1",
    "SPLITKV_VERIFY_MAX_Q": "64",
    "PRECACHE_BENCH": "1",
    "PRECACHE_REQUIRE": "1",
    "PRECACHE_MAX_TOKENS": "4",
    "PRECACHE_DATASET": "/harness/data/eval_prompts_sharegpt.json",
    # LSK_SKIP_LAYERS intentionally ABSENT (the TRAP must stay unset).
}

# land #71's intended descending build spec — the three code-shaped rows assembled
# correctly (device-vectorized relocate, sync-free accept walk, argmax-only decode
# with the prefill scatter+LP seam retained).
FAITHFUL_INTROSPECTION = {
    "relocate_salvaged_kv": {
        "device_vectorized_op_present": True,
        "host_layer_loop_present": False,
        "host_sync_in_relocate": False,
        "n_host_layer_iterations": 0,
        "op_symbol": "fused_relocate_kv_device([L,W,H,D] index_select+index_copy_)",
    },
    "accept_walk": {
        "device_argmax_accept_len_present": True,
        "host_item_call_present": False,
    },
    "decode_logits": {
        "decode_argmax_only": True,
        "prefill_full_scatter_lp_retained": True,
    },
}


def faithful_build_env():
    return copy.deepcopy(FA2SW_PRECACHE_KENYAN_ENV)


def faithful_introspection():
    return copy.deepcopy(FAITHFUL_INTROSPECTION)


def _mut(base, **over):
    d = copy.deepcopy(base)
    d.update(over)
    return d


def mutate_relocate_hostloop(intro):
    i = copy.deepcopy(intro)
    i["relocate_salvaged_kv"] = {
        "device_vectorized_op_present": False,
        "host_layer_loop_present": True,
        "host_sync_in_relocate": True,
        "n_host_layer_iterations": 37,
        "op_symbol": "relocate_salvaged_kv (host Python for-loop over 37 layers, D2H/H2D)",
    }
    return i


def mutate_drop_relocate_intro(intro):
    i = copy.deepcopy(intro)
    i.pop("relocate_salvaged_kv", None)
    return i


def double_load_bearing_drops(env0, intro0):
    """The 5 double-load-bearing rows, each individually dropped/broken."""
    # decode prefill scatter+LP seam dropped -> PPL break (keep decode argmax)
    intro_decode = copy.deepcopy(intro0)
    intro_decode["decode_logits"]["prefill_full_scatter_lp_retained"] = False
    return {
        "decode_prefill_scatterLP_dropped": (
            env0, intro_decode, "decode-path argmax-only"),
        "temperature_nonzero": (
            _mut(env0, OVERRIDE_GENERATION_CONFIG='{"temperature":0.7,"top_p":1.0,"top_k":0}'),
            intro0, "OVERRIDE_GENERATION_CONFIG temperature=0.0"),
        "scoring_contract_MAX_NUM_SEQS": (
            _mut(env0, MAX_NUM_SEQS="4"), intro0, "MAX_NUM_SEQS=1 / MAX_MODEL_LEN=4096"),
        "model_artifact_WEIGHTS_BUCKET_dropped": (
            {k: v for k, v in env0.items() if k != "WEIGHTS_BUCKET"}, intro0,
            "WEIGHTS_BUCKET/LOCAL_MODEL_DIR"),
        "LSK_trap_set": (
            _mut(env0, LSK_SKIP_LAYERS="19,20"), intro0, "LSK_SKIP_LAYERS"),
    }


# =============================================================================
# Self-test (PRIMARY) + test metric
# =============================================================================
def self_test(manifest: dict) -> dict:
    checks = []
    captured = []  # gate results captured for nan-clean sweep

    def chk(name, cond, detail=""):
        checks.append({"name": name, "passes": bool(cond), "detail": str(detail)})

    env0 = faithful_build_env()
    intro0 = faithful_introspection()

    # (a) faithful build -> GO
    g = verify_submission_gate(env0, intro0, manifest); captured.append(g)
    chk("(a) faithful build -> GO", g["packaging_verdict"] == "GO", g["packaging_verdict"])
    chk("(a) zero failing rows", g["n_failing"] == 0, g["n_failing"])
    chk("(a) zero validity-class failures", len(g["validity_class_failures"]) == 0,
        len(g["validity_class_failures"]))
    chk("(a) all 22 rows checked", g["n_rows_checked"] == 22, g["n_rows_checked"])
    chk("(a) 19 must-retain rows", g["n_must_retain"] == 19, g["n_must_retain"])
    chk("(a) zero gate-construction errors", len(g["construction_errors"]) == 0, g["construction_errors"])
    mr_nonpass = [r["flag"][:34] for r in g["per_row_assertions"]
                  if r["must_retain"] and r["status"] != "PASS"]
    chk("(a) every must-retain row PASS", not mr_nonpass, mr_nonpass)

    # (b) relocate reverted to host-loop -> NO-GO flagging row 1 with -85%/444.92
    gb = verify_submission_gate(env0, mutate_relocate_hostloop(intro0), manifest); captured.append(gb)
    chk("(b) host-loop -> NO-GO", gb["packaging_verdict"] == "NO-GO", gb["packaging_verdict"])
    chk("(b) row1 relocate in failing_rows",
        any(r["flag"].startswith("relocate_salvaged_kv") for r in gb["failing_rows"]),
        [r["flag"][:24] for r in gb["failing_rows"]])
    b = gb["binding_failure"]
    chk("(b) binding failure is row1",
        bool(b) and b["flag"].startswith("relocate_salvaged_kv"), b and b["flag"][:24])
    chk("(b) row1 banked cost ~444.92 (imported, not re-derived)",
        bool(b) and abs(b["banked_cost_tps"] - ROW1_BANKED_COST_TPS) < 0.5, b and b["banked_cost_tps"])
    chk("(b) host-loop is SPEED-class, not validity (row1 greedy-safe)",
        len(gb["validity_class_failures"]) == 0, len(gb["validity_class_failures"]))

    # (c) LSK_SKIP_LAYERS set -> NO-GO flagging the TRAP (validity)
    gc = verify_submission_gate(_mut(env0, LSK_SKIP_LAYERS="19,20"), intro0, manifest); captured.append(gc)
    chk("(c) LSK set -> NO-GO", gc["packaging_verdict"] == "NO-GO", gc["packaging_verdict"])
    chk("(c) LSK trap in failing_rows",
        any("LSK_SKIP_LAYERS" in r["flag"] for r in gc["failing_rows"]), "")
    chk("(c) LSK trap in validity_class_failures",
        any("LSK_SKIP_LAYERS" in v["flag"] for v in gc["validity_class_failures"]), "")

    # (d) PRECACHE_BENCH=0 -> NO-GO flagging row 2 (-3.5%)
    gd = verify_submission_gate(_mut(env0, PRECACHE_BENCH="0"), intro0, manifest); captured.append(gd)
    chk("(d) PRECACHE=0 -> NO-GO", gd["packaging_verdict"] == "NO-GO", gd["packaging_verdict"])
    pf = [r for r in gd["failing_rows"] if r["flag"].startswith("PRECACHE_BENCH")]
    chk("(d) PRECACHE row in failing_rows", len(pf) == 1, len(pf))
    chk("(d) PRECACHE banked cost ~18.33 (3.526%)",
        bool(pf) and abs(pf[0]["banked_cost_tps"] - PRECACHE_BANKED_COST_TPS) < 0.5,
        pf and pf[0]["banked_cost_tps"])

    # (e) each of the 5 double-load-bearing rows individually dropped -> NO-GO w/ validity_class
    for name, (env_m, intro_m, expect) in double_load_bearing_drops(env0, intro0).items():
        gi = verify_submission_gate(env_m, intro_m, manifest); captured.append(gi)
        chk(f"(e) DLB drop [{name}] -> NO-GO", gi["packaging_verdict"] == "NO-GO", gi["packaging_verdict"])
        chk(f"(e) DLB drop [{name}] in validity_class_failures",
            any(expect in v["flag"] for v in gi["validity_class_failures"]),
            [v["flag"][:24] for v in gi["validity_class_failures"]])

    # (f) missing introspection for a row -> NO-GO (fail-closed, NOT silent-pass)
    gf = verify_submission_gate(env0, mutate_drop_relocate_intro(intro0), manifest); captured.append(gf)
    chk("(f) missing relocate introspection -> NO-GO", gf["packaging_verdict"] == "NO-GO",
        gf["packaging_verdict"])
    mr = [r for r in gf["failing_rows"] if r["flag"].startswith("relocate_salvaged_kv")]
    chk("(f) missing introspection -> MISSING (fail-closed, not FAIL-by-accident)",
        bool(mr) and mr[0]["status"] == "MISSING", mr and mr[0]["status"])

    # (f2) empty env + empty introspection -> fail-closed NO-GO (nothing silent-passes).
    # 18 of 19 must-retain rows fail on presence; the TRAP row (LSK_SKIP_LAYERS) PASSES
    # because its assertion is ABSENCE and an empty build has it unset.
    ge = verify_submission_gate({}, {}, manifest); captured.append(ge)
    chk("(f2) empty build -> NO-GO", ge["packaging_verdict"] == "NO-GO", ge["packaging_verdict"])
    chk("(f2) 18 presence-required must-retain rows fail (TRAP correctly passes when absent)",
        ge["n_failing"] == 18, ge["n_failing"])
    lsk = [r for r in ge["per_row_assertions"] if "LSK_SKIP_LAYERS" in r["flag"]]
    chk("(f2) TRAP row passes on empty build (absence is correct)",
        bool(lsk) and lsk[0]["status"] == "PASS", lsk and lsk[0]["status"])

    # (g) NaN/exception-clean across every captured fixture
    chk("(g) all fixtures nan-clean & exception-free", all(nan_clean(_strip(r)) for r in captured),
        f"{len(captured)} fixtures swept")

    n_pass = sum(c["passes"] for c in checks)
    n_total = len(checks)
    return {"checks": checks, "n_pass": n_pass, "n_total": n_total, "all_pass": n_pass == n_total}


def _strip(gate_result):
    """Numeric-only view for the nan-clean sweep (drop string details)."""
    return {
        "n_rows_checked": gate_result["n_rows_checked"],
        "n_must_retain": gate_result["n_must_retain"],
        "n_failing": gate_result["n_failing"],
        "total_banked_cost_tps_at_risk": gate_result["total_banked_cost_tps_at_risk"],
        "banked_costs": [r["banked_cost_tps"] for r in gate_result["per_row_assertions"]
                         if r["banked_cost_tps"] is not None],
    }


def compute_test_metric(manifest: dict) -> dict:
    """TEST gate_catches_row1_host_loop: run the gate on the host-loop fixture and
    assert it returns NO-GO, flags row 1, and attaches the imported 444.92 TPS /
    85.17% cost. This is the headline deliverable — the gate makes the silent
    85%-cost regression impossible to ship."""
    gb = verify_submission_gate(faithful_build_env(),
                                mutate_relocate_hostloop(faithful_introspection()), manifest)
    b = gb["binding_failure"]
    realizable = manifest["banked_imports"]["h163_realizable_descent_tps"]
    cost_pct = (b["banked_cost_tps"] / realizable * 100.0) if b else float("nan")
    catches = bool(
        gb["packaging_verdict"] == "NO-GO"
        and b is not None
        and b["flag"].startswith("relocate_salvaged_kv")
        and abs(b["banked_cost_tps"] - ROW1_BANKED_COST_TPS) < 0.5
    )
    return {
        "gate_catches_row1_host_loop": catches,
        "verdict_on_host_loop": gb["packaging_verdict"],
        "row1_banked_cost_tps": b["banked_cost_tps"] if b else None,
        "row1_cost_pct_of_official": cost_pct,
        "row1_detail": b["detail"] if b else None,
    }


def nan_clean(obj) -> bool:
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(nan_clean(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(nan_clean(v) for v in obj)
    return True


def _summarize(g) -> dict:
    return {
        "packaging_verdict": g["packaging_verdict"],
        "n_failing": g["n_failing"],
        "n_validity_class_failures": len(g["validity_class_failures"]),
        "failing_flags": [r["flag"] for r in g["failing_rows"]],
        "binding_failure": (
            {"flag": g["binding_failure"]["flag"],
             "banked_cost_tps": g["binding_failure"]["banked_cost_tps"],
             "detail": g["binding_failure"]["detail"]}
            if g["binding_failure"] else None),
        "validity_class_failures": g["validity_class_failures"],
    }


# =============================================================================
# W&B + main
# =============================================================================
def wandb_log(args, res: dict) -> None:
    try:
        import wandb

        run_w = wandb.init(
            project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
            group=args.wandb_group, name=args.wandb_name,
            config={
                "pr": 189,
                "manifest_pr": res["manifest_pr"],
                "manifest_wandb_run_id": res["manifest_wandb_run_id"],
                "baseline_official_tps": 481.53,
                "served_submission": "fa2sw_precache_kenyan",
                "n_rows": res["n_rows"],
            },
        )
        log = {
            "submission_gate_self_test_passes": int(res["self_test"]["all_pass"]),
            "gate_catches_row1_host_loop": int(res["test_metric"]["value"]),
            "self_test_n_pass": res["self_test"]["n_pass"],
            "self_test_n_total": res["self_test"]["n_total"],
            "n_rows": res["n_rows"],
            "n_must_retain": res["n_must_retain"],
            "n_double_load_bearing": res["n_double_load_bearing"],
            "go_example_verdict_is_go": int(res["worked_go_example"]["packaging_verdict"] == "GO"),
            "nogo_example_verdict_is_nogo": int(res["worked_nogo_example"]["packaging_verdict"] == "NO-GO"),
            "row1_banked_cost_tps": res["test_metric_detail"]["row1_banked_cost_tps"],
            "row1_cost_pct_of_official": res["test_metric_detail"]["row1_cost_pct_of_official"],
            "metrics_nan_clean": int(res["metrics_nan_clean"]),
        }
        wandb.log(log)
        run_w.summary.update(log)
        res["wandb_run_id"] = run_w.id
        wandb.finish()
        print(f"[gate] W&B run {run_w.id} (group {args.wandb_group})", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[gate] W&B logging skipped: {e!r}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wandb-group", type=str, default="executable-submission-gate")
    ap.add_argument("--wandb-name", type=str, default="ubel/executable-submission-gate")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--output", type=pathlib.Path, default=OUT_DIR / "executable_submission_gate.json")
    args = ap.parse_args(argv)

    manifest = load_manifest()
    man = manifest["manifest"]
    st = self_test(manifest)
    tm = compute_test_metric(manifest)

    go = verify_submission_gate(faithful_build_env(), faithful_introspection(), manifest)
    nogo = verify_submission_gate(faithful_build_env(),
                                  mutate_relocate_hostloop(faithful_introspection()), manifest)

    res = {
        "pr": 189,
        "lane": "executable fail-closed MUST-RETAIN submission gate (packaging/flag-presence)",
        "method": "LOCAL CPU-only static analysis + assertion. No GPU / vLLM / HF Job / "
        "submission / served-file change / kernel deploy. Adds 0 TPS (primary = self-test). "
        "BASELINE 481.53. Greedy/PPL untouched. Imports #186 costs; does NOT re-derive them.",
        "primary_metric": {"name": "submission_gate_self_test_passes", "value": int(st["all_pass"])},
        "test_metric": {"name": "gate_catches_row1_host_loop", "value": int(tm["gate_catches_row1_host_loop"])},
        "introspection_schema": INTROSPECTION_SCHEMA,
        "row1_check_logic": ROW1_CHECK_LOGIC,
        "validity_seam": VALIDITY_SEAM,
        "manifest_source": str(MANIFEST_PATH.relative_to(ROOT)),
        "manifest_pr": manifest.get("pr"),
        "manifest_wandb_run_id": manifest.get("wandb_run_id"),
        "n_rows": man["n_flags_enumerated"],
        "n_must_retain": man["n_must_retain"],
        "n_double_load_bearing": man["n_double_load_bearing"],
        "worked_go_example": _summarize(go),
        "worked_nogo_example": _summarize(nogo),
        "self_test": st,
        "test_metric_detail": tm,
        "scope": "STATIC flag/shape assertion. Consumes #186's costs (no re-derivation). "
        "Does NOT run the numerical GO/NO-GO (fern #185) or the output-validity gate "
        "(denken). Authorizes nothing — a human still files Approval request: HF job. "
        "NOT open2. NOT a launch.",
        "handoff": "run verify_submission_gate(build_env, introspection) against land #71's "
        "assembled build before any `Approval request: HF job`; a NO-GO names the exact "
        "failing flag + its banked cost — esp. row 1, the 85%-cost relocate-host-loop trap "
        "— and feeds fern #185's ledger as the `packaging-gate: GO` precondition row.",
    }
    res["metrics_nan_clean"] = nan_clean(_strip(go)) and nan_clean(_strip(nogo)) and nan_clean(
        {"row1": tm["row1_banked_cost_tps"], "pct": tm["row1_cost_pct_of_official"]}
    )

    if not args.no_wandb:
        wandb_log(args, res)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2), encoding="utf-8")
    try:
        shown = args.output.resolve().relative_to(ROOT)
    except ValueError:
        shown = args.output
    print(f"[gate] wrote {shown}", flush=True)
    print(f"[gate] PRIMARY submission_gate_self_test_passes = {int(st['all_pass'])} "
          f"({st['n_pass']}/{st['n_total']})", flush=True)
    print(f"[gate] TEST gate_catches_row1_host_loop = {int(tm['gate_catches_row1_host_loop'])} "
          f"(verdict={tm['verdict_on_host_loop']}, row1 cost {tm['row1_banked_cost_tps']:.2f} TPS "
          f"= {tm['row1_cost_pct_of_official']:.2f}% of official)", flush=True)
    print(f"[gate] GO example: {go['packaging_verdict']} ({go['n_failing']} failing) | "
          f"NO-GO example: {nogo['packaging_verdict']} "
          f"(binding={nogo['binding_failure']['flag'][:32] if nogo['binding_failure'] else None})", flush=True)
    if not st["all_pass"]:
        for c in st["checks"]:
            if not c["passes"]:
                print(f"[gate]   FAIL self-test: {c['name']} -> {c['detail']}", flush=True)
    return 0 if st["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
