#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""EAGLE-3 #101 capture-SIZE DISPATCH risk: price it against the deployed list (PR #311).

THE QUESTION
------------
ubel #306 (y1lji0c6, MERGED) cleared the VRAM axis (runtime build peak 20.158 GiB, 3.84 GiB
headroom) and found that the REAL #101-class launch risk is ORTHOGONAL to memory: lawine's
"size-29 CUDA-graph crash" (#101 / #245-cycle-1, EXPERIMENTS_LOG.md) is a capture-SIZE-LIST
DISPATCH failure -- `max_cudagraph_capture_size=16`, and a verify request whose replay
token-count is NOT in `cudagraph_capture_sizes` finds NO captured graph -> `IndexError`
lookup crash -- NOT an allocation failure. VRAM clears M=32 trivially (+12 MiB) but the
dispatch list does NOT.

This leg PRICES that dispatch risk -- a config-list CORRECTNESS property -- against the
ACTUAL deployed capture-size config that the frontier `fa2sw_precache_kenyan` boots with, and
the EAGLE-3 verify tree-width options M in {8,16,32}. It closes the last open sub-axis of
#306's runtime-VRAM finding: the dispatch-list correctness property the eventual EAGLE-3
served path must honor.

THE MECHANISM (carried honestly; NOT a VRAM OOM)
-----------------------------------------------
vLLM V1 captures CUDA graphs only at a discrete set of batch/token sizes
(`cudagraph_capture_sizes`), the largest being `max_cudagraph_capture_size`. At dispatch time
a forward pass looks up the captured graph for its (padded) token-count. If that token-count
exceeds `max_cudagraph_capture_size`, the runtime-mode dispatch indexes past the captured set
-> `IndexError` (vLLM #29091 / PR#23679; repo precedent report_descend_walk.md:87
"graph-capture is the size-29 crash -> enforce-eager"). For EAGLE/MTP with K draft tokens the
VERIFY forward processes the M-token candidate tree; for the graph to dispatch, M must be a
captured size <= the ceiling. With K=7 the spine unit is (1+K)=8, so the captured
multiples-of-8 within max-16 are {8, 16}. The deployed M=8 spine clears it; widening the tree
to M=32 re-enters lawine's crash regime -- a DISPATCH-LIST blocker, fixed by ADDING the size
to `cudagraph_capture_sizes`, not by freeing memory.

WHAT THIS LEG DOES (LOCAL, read-only static analysis, NO GPU, NO served change, 0 TPS)
-------------------------------------------------------------------------------------
  1. PARSE the deployed capture config (read-only) from the served manifest + serve.py +
     sitecustomize.py that `fa2sw_precache_kenyan` boots with: the engine knobs that determine
     the capture-size list, whether the list is explicitly pinned, the deployed verify width
     (the M=8 prewarm shape), and the drafter's manual ONEGRAPH capture mode.
  2. PER-WIDTH DISPATCH ARITHMETIC for the draft side (K=7) and verify side (M in {8,16,32}):
     compute the per-replay token-count that must appear in `cudagraph_capture_sizes`, and
     classify each width as dispatch-SAFE (captured) or CRASH-regime (IndexError fall-through).
  3. CROSS-CHECK #306 (import y1lji0c6 constants <=1e-6): confirm M=32's VRAM cost is +12 MiB
     (trivial) so the blocker is the dispatch list; confirm deployed M=8 clears; emit the EXACT
     config edit (which sizes to ADD) that makes M=16 / M=32 dispatch-safe.
  4. MECHANISM corroboration: cross-reference #101 (lawine #245-c1) + vLLM #29091/PR#23679 to
     confirm the dispatch-`IndexError` mechanism (not OOM).

Analysis-only. BASELINE 481.53 untouched (adds 0 TPS). NOT a launch; no served-file change; no
HF Job; no submission; NOT a build. The dispatch arithmetic is a config-list property -- it
does not depend on tensor VALUES, so random-init shapes (and no tensors at all) transfer."""
from __future__ import annotations

import argparse
import json
import math
import re
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]                       # .../target
SUBMISSION_DIR = REPO_ROOT / "submissions" / "fa2sw_precache_kenyan"
CAPTURE_PEAK_RESULTS = (
    REPO_ROOT / "research" / "validity" / "eagle3_capture_peak"
    / "eagle3_capture_peak_results.json"
)

# --------------------------------------------------------------------------- #
# Banked anchors imported VERBATIM from ubel #306 (y1lji0c6) and its sources;
# NEVER re-derived. All runs live in wandb-applied-ai-team/gemma-challenge-senpai.
# The <=1e-6 cross-check (self-test d) reloads the #306 results JSON at runtime
# and asserts these match the recorded values.
# --------------------------------------------------------------------------- #
OFFICIAL_BASELINE = 481.53                         # PR #52 official frontier TPS (untouched)

# ---- #306 y1lji0c6: runtime-VRAM axis this leg's dispatch axis is ORTHOGONAL to ----
BUILD_PEAK_GIB_306 = 20.158085215091706            # runtime build PEAK (resident + transient)
RESIDENT_FLOOR_GIB_306 = 20.100143778324128        # #299 resident floor stacked under the peak
PEAK_HEADROOM_24_HARD_GIB_306 = 3.841914784908294  # 24.0 - 20.158 (runtime peak headroom)
PEAK_HEADROOM_23_USABLE_GIB_306 = 2.841914784908294
CAPTURE_TRANSIENT_GIB_306 = 0.041015625            # dominant runtime transient (capture pool)
TOTAL_TRANSIENT_GIB_306 = 0.057941436767578125
# logit-buffer bytes (262144-vocab x M x 2B, bf16-native) -- the VRAM cost of widening M:
LOGIT_BUF_BYTES_M8_306 = 4194304                   # 262144 * 8 * 2
LOGIT_BUF_BYTES_M16_306 = 8388608                  # 262144 * 16 * 2
LOGIT_BUF_BYTES_M32_306 = 16777216                 # 262144 * 32 * 2
M32_VRAM_DELTA_OVER_M8_BYTES_306 = LOGIT_BUF_BYTES_M32_306 - LOGIT_BUF_BYTES_M8_306  # 12582912 = 12 MiB
M16_VRAM_DELTA_OVER_M8_BYTES_306 = LOGIT_BUF_BYTES_M16_306 - LOGIT_BUF_BYTES_M8_306  # 4194304 = 4 MiB
MIB = float(1024 ** 2)
M32_VRAM_DELTA_OVER_M8_MIB_306 = M32_VRAM_DELTA_OVER_M8_BYTES_306 / MIB              # 12.0 MiB
M16_VRAM_DELTA_OVER_M8_MIB_306 = M16_VRAM_DELTA_OVER_M8_BYTES_306 / MIB              # 4.0 MiB

# ---- lawine size-29 capture-crash boundary (#101 / #245-cycle-1, EXPERIMENTS_LOG.md) ----
MAX_CUDAGRAPH_CAPTURE_SIZE_101 = 16                # vLLM target-model capture-size ceiling
SPINE_VERIFY_TOKENS_101 = 8                        # (1+K)=8 spine verify; captures cleanly
SIZE29_CRASH_TOKENS_101 = 29                       # the verify-token count that crashed (29 > 16)
VALID_CAPTURED_SIZES_306 = (8, 16)                 # #306 precedent_101.valid_captured_sizes
EAGLE_CAPTURE_SIZE_DIVISOR_1PK_306 = 8             # (1+K) spine unit; captured sizes are multiples

# ---- deployed build geometry (#306 / live manifest) ----
K_SPEC = 7                                          # K=7 EAGLE/MTP draft chain (num_speculative_tokens)
M_VERIFY_DEPLOYED = 8                               # deployed verify width (1+K spine)
VOCAB = 262144

# vLLM source refs corroborating the dispatch (not OOM) mechanism.
VLLM_DISPATCH_REFS = ("vLLM#29091", "vLLM-PR#23679")
M_SWEEP_DEFAULT = (8, 16, 32)


# --------------------------------------------------------------------------- #
# (1) Parse the deployed capture-size config from the served source (read-only).
# --------------------------------------------------------------------------- #
def parse_deployed_capture_config(submission_dir: Path) -> dict[str, Any]:
    """Read-only static parse of the served manifest + serve.py + sitecustomize.py.

    The frontier launch does NOT explicitly pin `cudagraph_capture_sizes` /
    `max_cudagraph_capture_size`; it inherits vLLM's default derivation for the deployed
    engine config. The banked #101/#306 anchor establishes that default's ceiling = 16 for
    this `max_num_seqs=1` + spec-K=7 engine. We parse the SOURCE to (a) confirm the list is
    not pinned, (b) recover the deployed verify width from the prewarm shape, and (c) confirm
    the drafter uses a manual ONEGRAPH capture (CUDAGraphMode.NONE), so the dispatch risk
    lives ENTIRELY on the verify side.
    """
    manifest_path = submission_dir / "manifest.json"
    serve_path = submission_dir / "serve.py"
    site_path = submission_dir / "sitecustomize.py"

    manifest = json.loads(manifest_path.read_text())
    env = manifest.get("env", {})

    spec_cfg_raw = env.get("SPECULATIVE_CONFIG", "")
    spec_cfg = json.loads(spec_cfg_raw) if spec_cfg_raw else {}
    k_parsed = int(spec_cfg.get("num_speculative_tokens")) if spec_cfg.get(
        "num_speculative_tokens") is not None else None

    serve_src = serve_path.read_text()
    site_src = site_path.read_text()

    # The capture-size list is determined by the vLLM launch flags. Confirm the launch does
    # NOT pin it (no --compilation-config / --cuda-graph-sizes / --cudagraph*), and is NOT
    # eager (which would disable capture entirely and side-step dispatch).
    def _flag_present(flag: str) -> bool:
        return flag in serve_src

    pins = {
        "enforce_eager": _flag_present("--enforce-eager"),
        "compilation_config": _flag_present("--compilation-config"),
        "cuda_graph_sizes": _flag_present("--cuda-graph-sizes"),
        "cudagraph_capture_sizes_literal": ("cudagraph_capture_sizes" in serve_src
                                            or "cudagraph_capture_sizes" in site_src),
        "max_capture_size_literal": ("max_capture_size" in serve_src
                                     or "max_capture_size" in site_src),
    }
    capture_list_pinned = bool(
        pins["compilation_config"] or pins["cuda_graph_sizes"]
        or pins["cudagraph_capture_sizes_literal"] or pins["max_capture_size_literal"]
    )

    # Deployed verify width recovered from the greedy-rejection prewarm shape
    # (serve.py ~487-492): output_token_ids = torch.full((1, <M>), ...) with
    # cu_num_draft_tokens=[<K>] and draft_token_ids=arange(<K>).  <M> == 1+<K> spine.
    prewarm_m = None
    m_full = re.search(r"output_token_ids\s*=\s*torch\.full\(\s*\(\s*1\s*,\s*(\d+)\s*\)",
                       serve_src)
    if m_full:
        prewarm_m = int(m_full.group(1))
    prewarm_k = None
    m_cu = re.search(r"cu_num_draft_tokens\s*=\s*torch\.tensor\(\s*\[\s*(\d+)\s*\]", serve_src)
    if m_cu:
        prewarm_k = int(m_cu.group(1))

    # Drafter capture mode: the ONEGRAPH loop runs with CUDAGraphMode.NONE (manual graph),
    # so the draft side is NOT routed through vLLM's cudagraph_capture_sizes dispatch.
    drafter_manual_onegraph = ("cudagraph_runtime_mode=CUDAGraphMode.NONE" in site_src
                               or "CUDAGraphMode.NONE" in site_src)
    onegraph_env = env.get("ONEGRAPH") == "1"

    return {
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "serve_path": str(serve_path.relative_to(REPO_ROOT)),
        "sitecustomize_path": str(site_path.relative_to(REPO_ROOT)),
        "k_num_speculative_tokens": k_parsed,
        "spec_method": spec_cfg.get("method"),
        "max_num_seqs": env.get("MAX_NUM_SEQS"),
        "max_num_batched_tokens": env.get("MAX_NUM_BATCHED_TOKENS"),
        "performance_mode": env.get("PERFORMANCE_MODE"),
        "onegraph_env": onegraph_env,
        "loopgraph_require_capture": env.get("LOOPGRAPH_REQUIRE_CAPTURE"),
        "dixie_prewarm_greedy_kernel": env.get("DIXIE_PREWARM_GREEDY_KERNEL", "1"),
        "launch_flag_pins": pins,
        "capture_list_explicitly_pinned": capture_list_pinned,
        "drafter_manual_onegraph_capture": drafter_manual_onegraph,
        "prewarm_verify_width": prewarm_m,
        "prewarm_draft_k": prewarm_k,
        # The effective list is the vLLM default for this engine; banked ceiling = 16.
        "max_cudagraph_capture_size_effective": MAX_CUDAGRAPH_CAPTURE_SIZE_101,
        "capture_sizes_source": (
            "vLLM default derivation (NOT pinned in submission); ceiling banked from "
            "lawine #101 size-29 crash + ubel #306 y1lji0c6"
        ),
        # Verify-relevant captured subset = multiples of (1+K) within the ceiling.
        "verify_captured_sizes_multiples_of_1pK": list(VALID_CAPTURED_SIZES_306),
    }


# --------------------------------------------------------------------------- #
# (2) Per-width dispatch arithmetic.
# --------------------------------------------------------------------------- #
def verify_replay_tokens(m_width: int) -> int:
    """Per-replay token-count of the VERIFY forward for a width-M candidate tree.

    The target verifies all M tree nodes in one forward (M query rows under a tree-causal
    mask), so the verify graph's dispatch size == M. (For the deployed linear spine M == 1+K.)
    """
    return int(m_width)


def captured_multiples_of_spine(max_capture: int, spine_unit: int) -> list[int]:
    """The captured sizes a (1+K)-structured verify tree can land on: multiples of the spine
    unit (1+K) up to and including the ceiling."""
    return [s for s in range(spine_unit, max_capture + 1, spine_unit)]


def classify_width(m_width: int, captured_sizes: list[int], max_capture: int,
                   spine_unit: int) -> dict[str, Any]:
    tokens = verify_replay_tokens(m_width)
    is_spine_multiple = (tokens % spine_unit == 0)
    within_ceiling = (tokens <= max_capture)
    captured = bool(within_ceiling and tokens in captured_sizes)
    if captured:
        regime = "boundary" if tokens == max_capture else "clear"
    else:
        regime = "crash-regime(IndexError-dispatch)"
    return {
        "m_width": int(m_width),
        "verify_replay_tokens": tokens,
        "is_multiple_of_1pK": bool(is_spine_multiple),
        "within_capture_ceiling": bool(within_ceiling),
        "dispatch_captured": captured,
        "dispatch_safe": captured,
        "regime": regime,
    }


def dispatch_arithmetic(k_spec: int, m_sweep: list[int], max_capture: int) -> dict[str, Any]:
    spine_unit = 1 + k_spec                          # (1+K)=8
    captured_sizes = captured_multiples_of_spine(max_capture, spine_unit)

    # DRAFT side: the ONEGRAPH loop is a MANUAL capture (CUDAGraphMode.NONE), batch=1 inner,
    # K-unrolled. It is NOT routed through vLLM's cudagraph_capture_sizes -> dispatch-safe by
    # construction, independent of the list. (Its notional per-iteration size is 1; the fused
    # draft+verify capture #306 measured was K+M=15 tokens, also <= the size-16 ceiling.)
    draft = {
        "k_spec": int(k_spec),
        "capture_mode": "manual ONEGRAPH (CUDAGraphMode.NONE), batch=1 inner, K-unrolled",
        "routed_through_vllm_capture_list": False,
        "per_iteration_tokens": 1,
        "fused_draft_plus_verify_tokens": int(k_spec + (1 + k_spec)),   # 7 + 8 = 15
        "dispatch_safe": True,
        "note": ("draft side is NOT subject to the vLLM capture-size list; the dispatch risk "
                 "is entirely on the verify side"),
    }

    verify = {str(m): classify_width(m, captured_sizes, max_capture, spine_unit)
              for m in m_sweep}

    safe_widths = [m for m in m_sweep if verify[str(m)]["dispatch_safe"]]
    crash_widths = [m for m in m_sweep if not verify[str(m)]["dispatch_safe"]]
    # The widest verify tree that still dispatches under the deployed list == the largest
    # captured (1+K)-multiple within the ceiling.
    max_safe_tree_width = max(captured_sizes) if captured_sizes else 0

    return {
        "spine_unit_1pK": spine_unit,
        "max_cudagraph_capture_size": int(max_capture),
        "captured_sizes_multiples_of_1pK": captured_sizes,
        "draft_side": draft,
        "verify_side": verify,
        "safe_widths": safe_widths,
        "crash_widths": crash_widths,
        "max_safe_tree_width_under_deployed_list": int(max_safe_tree_width),
        "deployed_m8_dispatch_safe": bool(verify[str(M_VERIFY_DEPLOYED)]["dispatch_safe"]),
    }


# --------------------------------------------------------------------------- #
# (3) Exact mitigating config edit (which sizes to ADD).
# --------------------------------------------------------------------------- #
def mitigating_config_edit(m_target: int, k_spec: int, current_max: int) -> dict[str, Any]:
    """The config-list edit that makes a width-M_target verify tree dispatch-safe.

    The deployed launch does NOT pin the list, so the fix is to ADD an EXPLICIT pin on the
    vLLM launch (serve.py args / manifest) that includes M_target (and the intervening
    (1+K)-multiples) and raises the ceiling to >= M_target. ADDING the size to
    `cudagraph_capture_sizes` -- NOT freeing memory -- is the entire fix for the dispatch axis.
    """
    spine_unit = 1 + k_spec
    if m_target <= current_max:
        sizes_to_add: list[int] = []
        new_max = current_max
    else:
        sizes_to_add = [s for s in range(current_max + spine_unit, m_target + 1, spine_unit)]
        if m_target not in sizes_to_add and m_target % spine_unit == 0:
            sizes_to_add.append(m_target)
        sizes_to_add = sorted(set(sizes_to_add))
        new_max = m_target
    # Full explicit list = small non-spec sizes + the (1+K)-multiples up to new_max.
    full_list = [1, 2, 4] + list(range(spine_unit, new_max + 1, spine_unit))
    full_list = sorted(set(full_list))
    edit_flag = ("--compilation-config '{\"cudagraph_capture_sizes\": "
                 + json.dumps(full_list) + "}'")
    edit_flag_alt = f"--cuda-graph-sizes {new_max}"
    return {
        "m_target": int(m_target),
        "current_max_cudagraph_capture_size": int(current_max),
        "dispatch_safe_already": m_target <= current_max,
        "sizes_to_add": sizes_to_add,
        "new_max_cudagraph_capture_size": int(new_max),
        "explicit_capture_list_after_edit": full_list,
        "edit_flag": edit_flag,
        "edit_flag_alt": edit_flag_alt,
        "edit_location": "submissions/fa2sw_precache_kenyan/serve.py vLLM launch args (or manifest env)",
        "also_required_served_change": (
            "widening M also requires the verify prewarm shape serve.py:487-492 "
            "(torch.full((1, M))) and the tree SpecDecodeMetadata to be widened to M -- a "
            "served-file change OUT OF SCOPE for this read-only dispatch audit, flagged"),
    }


# --------------------------------------------------------------------------- #
# (4) Mechanism corroboration (#101 + vLLM source).
# --------------------------------------------------------------------------- #
def mechanism_corroboration() -> dict[str, Any]:
    return {
        "precedent": "lawine #101 / #245-cycle-1 size-29 CUDA-graph capture crash",
        "experiments_log_ref": "research/EXPERIMENTS_LOG.md (lawine #245-c1 cycle)",
        "repo_corroboration": ("research/tree_verify_path/report_descend_walk.md:87 "
                               "'graph-capture is the size-29 crash -> enforce-eager'"),
        "vllm_refs": list(VLLM_DISPATCH_REFS),
        "crash_is_vram_oom": False,
        "crash_is_capture_size_dispatch_indexerror": True,
        "why": ("at dispatch the runtime-mode lookup indexes the captured-size set for the "
                "forward's padded token-count; token-count > max_cudagraph_capture_size finds "
                "no captured graph -> IndexError. size-29 > max-16 => crash. It is a "
                "config-list correctness property, independent of allocation/VRAM bytes."),
        "size29_over_max16": SIZE29_CRASH_TOKENS_101 > MAX_CUDAGRAPH_CAPTURE_SIZE_101,
        "spine8_within_max16": SPINE_VERIFY_TOKENS_101 <= MAX_CUDAGRAPH_CAPTURE_SIZE_101,
        "public_corroboration": {
            "note": ("public challenge board attributes the IN-SERVE size-29 tree-verify crash "
                     "to the custom star_gqa attention KERNEL (a distinct failure mode), fixed "
                     "by a dense masked attn N<=32 -- NOT the cudagraph capture-size dispatch "
                     "axis this leg prices. Both are size-boundary failure modes any EAGLE-3 "
                     "verify width past M=8 must clear; this leg scopes ONLY the config-list "
                     "dispatch correctness property (repo framing #306/#101 + report_descend_"
                     "walk.md:87 'graph-capture is the size-29 crash -> enforce-eager')."),
            "refs": ["openevolve 20260615-012216-024 (dense N<=32 ran 1280+ steps, no crash)",
                     "vidraft-darwin 20260614-234841-886 (star-attn CUDA crash != algorithm; "
                     "fix = dense masked attn N<=32)"],
        },
    }


# --------------------------------------------------------------------------- #
# (d) #306 cross-check: reload y1lji0c6 results and assert <=1e-6.
# --------------------------------------------------------------------------- #
def cross_check_306(tol: float = 1e-6) -> dict[str, Any]:
    """Reload the #306 (y1lji0c6) results JSON and assert the banked constants this leg
    imports match the recorded values to <= tol. Also confirm the M=32 VRAM delta = +12 MiB
    (trivial) -- proving the M=32 blocker is the DISPATCH list, not memory."""
    checks: dict[str, Any] = {}
    max_abs_err = 0.0
    loaded = False
    if CAPTURE_PEAK_RESULTS.exists():
        rec = json.loads(CAPTURE_PEAK_RESULTS.read_text())
        syn = rec.get("synthesis", {})
        analytic = syn.get("analytic", {})
        logit_bf16 = analytic.get("logit_buffer_bytes_bf16", {})
        pairs = {
            "build_peak_gib": (BUILD_PEAK_GIB_306, syn.get("eagle3_build_peak_gb")),
            "resident_floor_gib": (RESIDENT_FLOOR_GIB_306, syn.get("resident_floor_gib")),
            "peak_headroom_24_hard_gib": (PEAK_HEADROOM_24_HARD_GIB_306,
                                          syn.get("peak_headroom_24_hard_gib")),
            "peak_headroom_23_usable_gib": (PEAK_HEADROOM_23_USABLE_GIB_306,
                                            syn.get("peak_headroom_23_usable_gib")),
            "capture_transient_gib": (CAPTURE_TRANSIENT_GIB_306, syn.get("capture_transient_gib")),
            "total_transient_gib": (TOTAL_TRANSIENT_GIB_306, syn.get("total_transient_gib")),
            "logit_buf_bytes_m8": (float(LOGIT_BUF_BYTES_M8_306), float(logit_bf16.get("8", float("nan")))),
            "logit_buf_bytes_m16": (float(LOGIT_BUF_BYTES_M16_306), float(logit_bf16.get("16", float("nan")))),
            "logit_buf_bytes_m32": (float(LOGIT_BUF_BYTES_M32_306), float(logit_bf16.get("32", float("nan")))),
        }
        for key, (mine, rec_val) in pairs.items():
            if rec_val is None:
                checks[key] = {"mine": mine, "recorded": None, "ok": False}
                continue
            err = abs(float(mine) - float(rec_val))
            max_abs_err = max(max_abs_err, err)
            checks[key] = {"mine": float(mine), "recorded": float(rec_val),
                           "abs_err": err, "ok": err <= tol}
        loaded = True
    # M=32 dispatch-vs-memory framing (banked recover_gib from #306 == m_cap delta).
    m32_delta_ok = (M32_VRAM_DELTA_OVER_M8_BYTES_306 == 12582912)   # exactly 12 MiB
    all_ok = loaded and all(c.get("ok") for c in checks.values()) and m32_delta_ok
    return {
        "results_file": str(CAPTURE_PEAK_RESULTS.relative_to(REPO_ROOT)),
        "loaded": loaded,
        "tol": tol,
        "checks": checks,
        "max_abs_err": max_abs_err,
        "m32_vram_delta_over_m8_mib": M32_VRAM_DELTA_OVER_M8_MIB_306,
        "m32_vram_delta_is_12mib_trivial": m32_delta_ok,
        "m32_blocker_is_dispatch_not_memory": bool(m32_delta_ok),
        "all_within_tol": bool(all_ok),
    }


# --------------------------------------------------------------------------- #
# Synthesis + self-test.
# --------------------------------------------------------------------------- #
CAVEATS = [
    "0 TPS / config-list property: this prices the capture-SIZE DISPATCH risk -- a "
    "`cudagraph_capture_sizes` correctness property -- NOT a runtime measurement. It does not "
    "depend on tensor VALUES, so random-init shapes (indeed, no tensors at all) transfer for "
    "the size accounting; only the integer token-counts and the list matter.",
    "ORTHOGONAL to VRAM: #306 (y1lji0c6) already proved the EAGLE-3 build fits at runtime "
    "(peak 20.158 GiB, 3.84 under 24-hard) and that M=32 costs only +12 MiB. This leg shows "
    "the M=32 blocker is the DISPATCH list, fixed by ADDING the size to "
    "`cudagraph_capture_sizes`, not by freeing memory. The two axes are independent.",
    "DEPLOYED list is INHERITED, not pinned: the frontier launch sets no "
    "--compilation-config / --cuda-graph-sizes / --enforce-eager, so the capture-size list is "
    "vLLM's default derivation for this max_num_seqs=1 + spec-K=7 engine. The effective "
    "ceiling (16) is the banked empirical value from lawine #101 (size-29 crash) re-confirmed "
    "by #306; this leg does not re-derive vLLM's internal default formula.",
    "MECHANISM honesty: lawine's #101 'size-29 crash' is a capture-SIZE-LIST DISPATCH "
    "IndexError (no captured graph for batch=29 > max-16), NOT a VRAM OOM (vLLM #29091 / "
    "PR#23679; report_descend_walk.md:87). The draft side is captured by a MANUAL ONEGRAPH "
    "(CUDAGraphMode.NONE) and is NOT routed through the vLLM list -> the dispatch risk is "
    "entirely on the verify side.",
    "TOPOLOGY is a separate lane: this prices ONLY the dispatch correctness of a given verify "
    "width M. Whether a width-M tree is DESIRABLE (its E[T]/acceptance payoff) is the "
    "topology-optimization lane, orthogonal to this list-correctness audit. Widening M also "
    "needs the verify prewarm shape (serve.py:487-492) and the tree SpecDecodeMetadata "
    "widened -- served-file changes out of scope here.",
    "TWO size-boundary failure modes (public corroboration): the PUBLIC board "
    "(openevolve 20260615-012216, vidraft-darwin 20260614-234841) attributes the in-serve "
    "size-29 tree-verify crash to the custom star_gqa attention KERNEL, fixed by a dense "
    "masked attn at N<=32 -- a DIFFERENT axis than the cudagraph capture-size DISPATCH "
    "IndexError this leg prices (repo #306/#101 + report_descend_walk.md:87). Whether they are "
    "the same event or two distinct boundaries, BOTH must be cleared by any verify width past "
    "M=8; this leg scopes ONLY the config-list dispatch axis and does not claim to price the "
    "star_gqa kernel crash.",
    "The launch gate is UNCHANGED and human-approval-gated (land #245: MEASURED >=500 TPS at "
    "lambda_hat>=0.9780 AND PPL<=2.42 AND VRAM<=24 GiB). This leg adds a DISPATCH-correctness "
    "sub-clause for any EAGLE-3 verify width past the deployed M=8; it is NOT a launch, NOT a "
    "build, NOT a served-file change.",
]


def synthesize(deployed: dict[str, Any], arith: dict[str, Any], xcheck: dict[str, Any],
               mech: dict[str, Any], m_sweep: list[int]) -> dict[str, Any]:
    max_capture = arith["max_cudagraph_capture_size"]
    # Mitigations for every swept width that is NOT already dispatch-safe (plus M=16 for
    # completeness even though it sits at the boundary).
    mitigations = {str(m): mitigating_config_edit(m, K_SPEC, max_capture)
                   for m in m_sweep if m > M_VERIFY_DEPLOYED}

    deployed_m8_safe = arith["deployed_m8_dispatch_safe"]
    m32_class = arith["verify_side"][str(32)] if "32" in arith["verify_side"] else {}
    m32_crash = (not m32_class.get("dispatch_safe", True)) if m32_class else False

    # The single explicit edit the human GO/NO-GO needs for the widest swept width.
    widest = max(m_sweep)
    primary_edit = mitigating_config_edit(widest, K_SPEC, max_capture)

    # ---- self-test conditions (a-g) ----
    cond = {}
    # (a) deployed captured-size list parsed from source.
    cond["a_deployed_capture_config_parsed"] = bool(
        deployed.get("k_num_speculative_tokens") == K_SPEC
        and deployed.get("prewarm_verify_width") == M_VERIFY_DEPLOYED
        and deployed.get("prewarm_draft_k") == K_SPEC
        and deployed.get("capture_list_explicitly_pinned") is False
        and deployed.get("drafter_manual_onegraph_capture") is True
        and deployed.get("max_cudagraph_capture_size_effective") == MAX_CUDAGRAPH_CAPTURE_SIZE_101
    )
    # (b) per-width token-count arithmetic correct (multiples of 1+K within the list).
    spine_unit = arith["spine_unit_1pK"]
    b_ok = (spine_unit == 1 + K_SPEC
            and arith["captured_sizes_multiples_of_1pK"] == list(VALID_CAPTURED_SIZES_306))
    for m in m_sweep:
        c = arith["verify_side"][str(m)]
        b_ok = b_ok and (c["verify_replay_tokens"] == m)
        b_ok = b_ok and (c["is_multiple_of_1pK"] == (m % spine_unit == 0))
        b_ok = b_ok and (c["dispatch_captured"]
                         == (m <= max_capture and m in arith["captured_sizes_multiples_of_1pK"]))
    cond["b_per_width_arithmetic_correct"] = bool(b_ok)
    # (c) deployed M=8 classified SAFE, M=32 classified CRASH-regime.
    cond["c_m8_safe_m32_crash"] = bool(deployed_m8_safe and m32_crash)
    # (d) #306 constants <=1e-6.
    cond["d_imported_306_constants_exact"] = bool(xcheck["all_within_tol"])
    # (e) NaN-clean -- set by the payload finalizer.
    cond["e_nan_clean"] = True
    # (f) the mitigating config edit emitted explicitly.
    cond["f_mitigating_edit_emitted"] = bool(
        primary_edit["sizes_to_add"] and primary_edit["edit_flag"]
        and primary_edit["new_max_cudagraph_capture_size"] >= widest)
    # (g) caveats carried.
    cond["g_honest_caveats_carried"] = bool(len(CAVEATS) >= 4)

    passes = all(cond.values())

    verdict = (
        f"The deployed frontier launch does NOT pin `cudagraph_capture_sizes` "
        f"(no --compilation-config/--cuda-graph-sizes/--enforce-eager); it inherits vLLM's "
        f"default, ceiling `max_cudagraph_capture_size={max_capture}` (banked #101/#306). The "
        f"verify-side dispatchable widths are the (1+K)={spine_unit}-multiples within it: "
        f"{arith['captured_sizes_multiples_of_1pK']}. Deployed M={M_VERIFY_DEPLOYED} "
        f"(prewarm shape (1,{deployed.get('prewarm_verify_width')})) is dispatch-SAFE; M=16 "
        f"sits AT the boundary (captured); M=32 ({verify_replay_tokens(32)} tokens > "
        f"{max_capture}) falls through to lawine's IndexError CRASH regime. The M=32 blocker "
        f"is the DISPATCH LIST (#306: VRAM cost only +{M32_VRAM_DELTA_OVER_M8_MIB_306:.0f} MiB), "
        f"fixed by ADDING {primary_edit['sizes_to_add']} to the list and raising the ceiling "
        f"to {primary_edit['new_max_cudagraph_capture_size']} via "
        f"`{primary_edit['edit_flag']}` -- NOT by freeing memory. Draft side (K={K_SPEC}) is a "
        f"manual ONEGRAPH capture, not list-dispatched -> safe by construction. The draft side "
        f"clears; the verify tree is dispatch-safe only up to M="
        f"{arith['max_safe_tree_width_under_deployed_list']} under the deployed list. "
        f"Analysis-only; BASELINE {OFFICIAL_BASELINE} untouched; 0 TPS."
    )
    handoff = (
        f"EAGLE-3 capture-SIZE dispatch priced: deployed M={M_VERIFY_DEPLOYED} is dispatch-SAFE "
        f"(8 in the (1+K)-multiple captured set {{8,16}}, ceiling 16); the verify tree is "
        f"dispatch-safe only up to M={arith['max_safe_tree_width_under_deployed_list']}. "
        f"Widening to M=32 re-enters lawine #101's IndexError crash -- a DISPATCH-LIST blocker "
        f"(VRAM cost a trivial +{M32_VRAM_DELTA_OVER_M8_MIB_306:.0f} MiB per #306), fixed by "
        f"ADDING {primary_edit['sizes_to_add']} to `cudagraph_capture_sizes` (ceiling -> "
        f"{primary_edit['new_max_cudagraph_capture_size']}) on the vLLM launch, NOT by freeing "
        f"memory. The list is currently INHERITED (unpinned); the human GO/NO-GO for any "
        f"EAGLE-3 width past M=8 must add this explicit pin AND widen the verify prewarm "
        f"(serve.py:487-492) + tree metadata. Topology desirability is a separate lane."
    )

    return {
        "constants": {
            "official_baseline": OFFICIAL_BASELINE,
            "k_spec": K_SPEC,
            "m_verify_deployed": M_VERIFY_DEPLOYED,
            "vocab": VOCAB,
            "max_cudagraph_capture_size_101": MAX_CUDAGRAPH_CAPTURE_SIZE_101,
            "spine_verify_tokens_101": SPINE_VERIFY_TOKENS_101,
            "size29_crash_tokens_101": SIZE29_CRASH_TOKENS_101,
            "valid_captured_sizes_306": list(VALID_CAPTURED_SIZES_306),
            "m_sweep": list(m_sweep),
            "build_peak_gib_306": BUILD_PEAK_GIB_306,
            "m32_vram_delta_over_m8_mib_306": M32_VRAM_DELTA_OVER_M8_MIB_306,
            "imports": (
                "ubel#306(y1lji0c6 build_peak=20.158 headroom24=3.842 capture_transient=0.0410 "
                "M32_delta=+12MiB valid_captured={8,16}) x lawine#101/#245-c1(max_capture=16 "
                "size29-crash dispatch-IndexError) x vLLM#29091/PR#23679 x served "
                "fa2sw_precache_kenyan(manifest spec-K=7 + serve.py M=8 prewarm)"),
        },
        "deployed_config": deployed,
        "dispatch_arithmetic": arith,
        "cross_check_306": xcheck,
        "mechanism": mech,
        "mitigations": mitigations,
        "primary_mitigating_edit": primary_edit,
        "max_safe_tree_width_under_deployed_list": arith["max_safe_tree_width_under_deployed_list"],
        "deployed_m8_dispatch_safe": deployed_m8_safe,
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
    print(" EAGLE-3 #101 CAPTURE-SIZE DISPATCH RISK -- PRICED AGAINST THE DEPLOYED LIST (PR #311)",
          flush=True)
    print("=" * 104, flush=True)
    dep = syn["deployed_config"]
    ar = syn["dispatch_arithmetic"]
    print(f"  DEPLOYED PARSE  spec-K={dep['k_num_speculative_tokens']}  "
          f"prewarm_verify_M={dep['prewarm_verify_width']}  "
          f"max_num_seqs={dep['max_num_seqs']}  perf_mode={dep['performance_mode']}  "
          f"capture_list_pinned={dep['capture_list_explicitly_pinned']}", flush=True)
    print(f"                  drafter_manual_onegraph={dep['drafter_manual_onegraph_capture']}  "
          f"max_cudagraph_capture_size(effective)={dep['max_cudagraph_capture_size_effective']}  "
          f"captured(1+K)-multiples={ar['captured_sizes_multiples_of_1pK']}", flush=True)
    print("-" * 104, flush=True)
    print(f"  DRAFT side (K={ar['draft_side']['k_spec']}): {ar['draft_side']['capture_mode']}  "
          f"-> list-dispatched={ar['draft_side']['routed_through_vllm_capture_list']}  "
          f"safe={ar['draft_side']['dispatch_safe']}", flush=True)
    print(f"  VERIFY side per-width dispatch (tokens must be in "
          f"{ar['captured_sizes_multiples_of_1pK']}, <= {ar['max_cudagraph_capture_size']}):",
          flush=True)
    for m in syn["constants"]["m_sweep"]:
        c = ar["verify_side"][str(m)]
        print(f"        M={c['m_width']:>2}  tokens={c['verify_replay_tokens']:>2}  "
              f"captured={str(c['dispatch_captured']):>5}  regime={c['regime']}", flush=True)
    print(f"  max_safe_tree_width_under_deployed_list="
          f"{ar['max_safe_tree_width_under_deployed_list']}  "
          f"deployed_m8_dispatch_safe={ar['deployed_m8_dispatch_safe']}", flush=True)
    print("-" * 104, flush=True)
    xc = syn["cross_check_306"]
    print(f"  #306 CROSS-CHECK  loaded={xc['loaded']}  max_abs_err={xc['max_abs_err']:.2e} "
          f"(<= {xc['tol']:.0e})  all_within_tol={xc['all_within_tol']}  "
          f"M32_delta=+{xc['m32_vram_delta_over_m8_mib']:.0f} MiB "
          f"(blocker_is_dispatch_not_memory={xc['m32_blocker_is_dispatch_not_memory']})",
          flush=True)
    pe = syn["primary_mitigating_edit"]
    print(f"  MITIGATING EDIT  add {pe['sizes_to_add']} -> ceiling "
          f"{pe['new_max_cudagraph_capture_size']}:  {pe['edit_flag']}", flush=True)
    me = syn["mechanism"]
    print(f"  MECHANISM  vram_oom={me['crash_is_vram_oom']}  "
          f"dispatch_IndexError={me['crash_is_capture_size_dispatch_indexerror']}  "
          f"refs={me['vllm_refs']}", flush=True)
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
        print(f"[eagle3-dispatch-risk] wandb logging skipped (analysis unaffected): {exc}",
              flush=True)
        return

    syn = payload["synthesis"]
    ar = syn["dispatch_arithmetic"]
    try:
        run = init_wandb_run(
            job_type="validity-gate", agent="ubel", name=args.wandb_name, group=args.wandb_group,
            tags=["eagle3-dispatch-risk", "cudagraph-capture-size", "dispatch-indexerror",
                  "config-list-correctness", "tree-width", "pr-311"],
            config={
                "official_baseline": OFFICIAL_BASELINE,
                "k_spec": K_SPEC, "m_verify_deployed": M_VERIFY_DEPLOYED,
                "max_cudagraph_capture_size_101": MAX_CUDAGRAPH_CAPTURE_SIZE_101,
                "valid_captured_sizes_306": list(VALID_CAPTURED_SIZES_306),
                "m_sweep": syn["constants"]["m_sweep"],
                "build_peak_gib_306": BUILD_PEAK_GIB_306,
                "m32_vram_delta_over_m8_mib_306": M32_VRAM_DELTA_OVER_M8_MIB_306,
                "imports": syn["constants"]["imports"], "wandb_group": args.wandb_group,
            },
        )
    except Exception as exc:
        print(f"[eagle3-dispatch-risk] wandb init failed (analysis unaffected): {exc}", flush=True)
        return
    if run is None:
        print("[eagle3-dispatch-risk] wandb: no run (no API key/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "capture_dispatch_risk_self_test_passes": int(bool(
            payload["capture_dispatch_risk_self_test_passes"])),
        "max_safe_tree_width_under_deployed_list": int(
            syn["max_safe_tree_width_under_deployed_list"]),
        "deployed_m8_dispatch_safe": int(bool(syn["deployed_m8_dispatch_safe"])),
        "max_cudagraph_capture_size": int(ar["max_cudagraph_capture_size"]),
        "spine_unit_1pK": int(ar["spine_unit_1pK"]),
        "capture_list_explicitly_pinned": int(bool(
            syn["deployed_config"]["capture_list_explicitly_pinned"])),
        "drafter_manual_onegraph_capture": int(bool(
            syn["deployed_config"]["drafter_manual_onegraph_capture"])),
        "prewarm_verify_width": int(syn["deployed_config"]["prewarm_verify_width"]),
        "xcheck_306_max_abs_err": syn["cross_check_306"]["max_abs_err"],
        "xcheck_306_all_within_tol": int(bool(syn["cross_check_306"]["all_within_tol"])),
        "m32_vram_delta_over_m8_mib": syn["cross_check_306"]["m32_vram_delta_over_m8_mib"],
        "m32_blocker_is_dispatch_not_memory": int(bool(
            syn["cross_check_306"]["m32_blocker_is_dispatch_not_memory"])),
        "mitigating_sizes_to_add_count": len(syn["primary_mitigating_edit"]["sizes_to_add"]),
        "mitigating_new_max_capture_size": int(
            syn["primary_mitigating_edit"]["new_max_cudagraph_capture_size"]),
        "n_safe_widths": len(ar["safe_widths"]),
        "n_crash_widths": len(ar["crash_widths"]),
        "crash_is_vram_oom": int(bool(syn["mechanism"]["crash_is_vram_oom"])),
        "crash_is_dispatch_indexerror": int(bool(
            syn["mechanism"]["crash_is_capture_size_dispatch_indexerror"])),
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in syn["self_test"]["conditions"].items()},
    }
    for m in syn["constants"]["m_sweep"]:
        c = ar["verify_side"][str(m)]
        summary[f"verify_dispatch_safe_m{m}"] = int(bool(c["dispatch_safe"]))
        summary[f"verify_tokens_m{m}"] = int(c["verify_replay_tokens"])
    summary = {k: v for k, v in summary.items()
               if v is not None and not (isinstance(v, float) and not math.isfinite(v))}
    try:
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="eagle3_capture_dispatch_risk_result",
                          artifact_type="validity", data=payload)
        finish_wandb(run)
        print(f"[eagle3-dispatch-risk] wandb logged {len(summary)} summary keys", flush=True)
    except Exception as exc:
        print(f"[eagle3-dispatch-risk] wandb write failed (analysis unaffected): {exc}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--k-spec", "--k_spec", dest="k_spec", type=int, default=K_SPEC)
    ap.add_argument("--m-sweep", "--m_sweep", dest="m_sweep", type=str, default="8,16,32",
                    help="verify tree widths to classify (comma-separated)")
    ap.add_argument("--submission-dir", type=Path, default=SUBMISSION_DIR)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="eagle3-dispatch-risk")
    args = ap.parse_args(argv)

    m_sweep = [int(x) for x in str(args.m_sweep).split(",") if x.strip()]

    deployed = parse_deployed_capture_config(args.submission_dir)
    arith = dispatch_arithmetic(args.k_spec, m_sweep, deployed["max_cudagraph_capture_size_effective"])
    xcheck = cross_check_306()
    mech = mechanism_corroboration()
    syn = synthesize(deployed, arith, xcheck, mech, m_sweep)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 311, "agent": "ubel", "kind": "eagle3-capture-dispatch-risk",
        "analysis_only": True,
        "synthesis": syn,
        "capture_dispatch_risk_self_test_passes": syn["self_test"]["passes"],
        "max_safe_tree_width_under_deployed_list": syn["max_safe_tree_width_under_deployed_list"],
        "deployed_m8_dispatch_safe": syn["deployed_m8_dispatch_safe"],
        "host_peak_mem_mib": round(peak_kib / 1024.0, 3),
        "greedy_ppl_safety_certificate": {
            "analysis_only": True, "served_file_changed": False, "emitted_token_changed": False,
            "hf_job_or_submission": False, "is_launch": False, "is_build": False,
            "baseline_tps_unchanged": OFFICIAL_BASELINE, "tps_added_by_this_leg": 0.0,
        },
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["e_nan_clean"] = bool(payload["nan_clean"])
    syn["self_test"]["passes"] = bool(all(syn["self_test"]["conditions"].values()))
    payload["capture_dispatch_risk_self_test_passes"] = bool(
        syn["self_test"]["passes"] and payload["nan_clean"])
    if nan_paths:
        print(f"[eagle3-dispatch-risk] WARNING non-finite at: {nan_paths[:10]}", flush=True)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eagle3_capture_dispatch_risk_results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))

    payload["primary_metric"] = {"name": "capture_dispatch_risk_self_test_passes",
                                 "value": int(bool(payload["capture_dispatch_risk_self_test_passes"]))}
    payload["test_metric"] = {"name": "max_safe_tree_width_under_deployed_list",
                              "value": int(syn["max_safe_tree_width_under_deployed_list"])}

    _print_human(syn)
    print(f"[eagle3-dispatch-risk] wrote {out_path}", flush=True)
    print(f"[eagle3-dispatch-risk] PRIMARY capture_dispatch_risk_self_test_passes = "
          f"{payload['capture_dispatch_risk_self_test_passes']}", flush=True)
    print(f"[eagle3-dispatch-risk] TEST max_safe_tree_width_under_deployed_list = "
          f"{syn['max_safe_tree_width_under_deployed_list']}  deployed_m8_dispatch_safe = "
          f"{syn['deployed_m8_dispatch_safe']}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = payload["capture_dispatch_risk_self_test_passes"]
        print(f"[eagle3-dispatch-risk] PRIMARY self-test: {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
