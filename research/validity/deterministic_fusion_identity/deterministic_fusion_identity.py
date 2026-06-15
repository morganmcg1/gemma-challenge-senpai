#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #374 — deterministic-fusion-identity: which inductor fusions break M=1
greedy byte-identity, and can the breakers be pinned to a fixed reduction order?

Context (the strict >500 program, #319). The byte-strict (greedy-token-identical)
non-spec int4 M=1 AR frontier sits at the official ~165.44 TPS floor (lawine #196,
``nf6kq1ya``). denken #344 measured the single-token body read at 1.697 GB = 94.3%
of step HBM, so at A10G ~600 GB/s peak a byte-exact single-token step has a BW
ceiling ~353 TPS — yet we sit at ~47% of it. The ~2x gap is launch-overhead +
un-fused eager scheduling, both *numerically identity-preserving to recover in
principle* (no quant, no reduction reorder).

  * land #371 owns the LAUNCH-OVERHEAD half: CUDA-graph capture, fusion OFF.
  * THIS card owns the FUSION half: capture conceptually held ON, fusion varied.

kanna's own #359 (``tjd4xngn``) found bundling CUDA-graph capture WITH full
inductor fusion is ~6.2x faster but BREAKS greedy identity (the fused+captured
config diverges from plain eager on ~75% of prompts) — but never isolated WHICH
fusions break it. This card builds the taxonomy.

Hypothesis. Inductor fusions split into
  (i)  identity-SAFE pointwise/elementwise epilogue fusions (silu/gelu activation,
       residual-add, scale, rotary) — no reduction dim, so bit-identical; and
  (ii) identity-BREAKING reduction-reordering fusions — fused RMSNorm and attention
       softmax (Triton persistent single-pass reductions accumulate the row in
       block/warp order, NOT the eager scalar-loop order), split multi-pass
       reductions, cross-block cooperative/atomic reductions, coordinate-descent
       tiling, and split-K GEMM (Triton matmul templates accumulate K-tiles in a
       different order). Each reorders the float summation -> different rounding ->
       can flip a greedy argmax.
If the breaking class can be PINNED to a deterministic fixed reduction order, we
recover most of the fusion speedup at ``token_identity_rate == 1.0``.

Vehicle. This is LOCAL pod-A10G profiling — NOT an HF Job, NOT a submission, NOT a
served-file change, 0 official TPS. It is a STANDALONE ``torch.compile`` micro-
harness on the real BF16 ``google/gemma-4-E4B-it`` text model doing a trivial
fixed-shape M=1 single-token greedy AR decode (StaticCache, batch=1, one token per
step — NOT a size-29 spec tree). It reuses the #359 rig directly: the byte-exact
greedy-token-identity gate (per-prompt completion-token sha256 vs the model's own
plain-eager greedy AR — the #158 contract), the per-config ablation loop, and the
analyze / self-test / 0-GPU reanalyze / wandb scaffolding.

Why standalone (not the vLLM server, like #359 served):
  * the served int4 GEMM is the Marlin custom CUDA op, OUTSIDE inductor (PR #122
    finding), so per-inductor-class ablation is impossible on the served path and
    the GEMM-fusion classes are unreachable there;
  * "NO served-file change" forbids adding inductor-knob plumbing to serve.py;
  * owning the ``torch.compile`` call is the only way to toggle each
    ``torch._inductor.config`` class one at a time.
A standalone BF16 forward keeps EVERY named fusion class in inductor's reach (incl.
the GEMMs), so the taxonomy is complete; the transfer caveat to the int4 served
path (GEMM classes unreachable there) is reported, not hidden.

CUDA-graph capture is numerically transparent (it replays the SAME kernels), so the
identity verdict of a fusion config is independent of whether capture is on — this
is the load-bearing #371 assumption, which we also state explicitly. Identity is
therefore measured compile-only (crash-proof: capture+StaticCache reuse can
segfault), and the fusion-speedup FRACTION we report is capture-invariant (capture
scales every config by ~the same factor, which cancels in the ratio).

Reproduce (full matrix, local A10G; needs a torch+transformers env — the
``scripts.local_validation`` server venv, NOT the repo .venv which lacks torch):
  CUDA_VISIBLE_DEVICES=0 <server-venv>/bin/python \
    research/validity/deterministic_fusion_identity/deterministic_fusion_identity.py \
    --measure --wandb_group deterministic-fusion-identity \
    --wandb_name kanna/deterministic-fusion-identity
Cheap 0-GPU integrity check (PRIMARY self-test on a synthetic round-trip):
  python .../deterministic_fusion_identity.py --self-test
0-GPU re-derivation from a saved measurement:
  python .../deterministic_fusion_identity.py --reanalyze <results.json>
Tiny GPU plumbing run:
  CUDA_VISIBLE_DEVICES=0 <server-venv>/bin/python .../deterministic_fusion_identity.py --smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _SCRIPT_DIR)]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
# Imported anchors — DO NOT re-derive (self-test constant guard checks these).
# --------------------------------------------------------------------------- #
TARGET_TPS = 500.0                       # official launch gate (#319).
STRICT_BASE_OFFICIAL_TPS_196 = 165.44    # strict non-spec int4 M=1 AR floor (lawine #196, nf6kq1ya).
DEPLOYED_SPEC_OFFICIAL_TPS = 481.53      # deployed spec anchor (PR #52, context only).
BW_CEILING_353 = 353.0                   # byte-exact single-token BW ceiling (denken #344, sxltbech).
BATCH_INVARIANT_357 = 357.32             # off-the-shelf VLLM_BATCH_INVARIANT=1 (wirbel #326, io4cs2ch).
LAMBDA1_CEILING = 520.953                # λ=1 ceiling.

MODEL_ID = "google/gemma-4-E4B-it"
OUT_ROOT = ROOT / "research" / "validity" / "deterministic_fusion_identity"
RESULTS_JSON = OUT_ROOT / "deterministic_fusion_identity_results.json"
PARTIAL_JSON = OUT_ROOT / "deterministic_fusion_identity_partial.json"

# Config execution order: cheap, identity-essential configs first (so a budget/timeout
# kill keeps the load-bearing taxonomy), expensive max_autotune configs last.
CONFIG_RUN_ORDER = [
    "dynamo_eager",             # ATen-kernel control: graph capture, NO Triton (expect identity=1.0)
    "aot_eager",                # ATen-exec control: AOT core-ATen decomp, NO Triton codegen
    "compile_default",          # cheap breaker — guarantees a fused-break point + denom
    "pointwise_only",           # safe-fusion ceiling
    "deterministic_pinned",     # the inductor-knob recovery config
    "no_persistent_reductions", # isolate RMSNorm/softmax persistent reduction
    "coordinate_descent",       # isolate coord-descent tiling
    "gemm_split_k",             # autotune: split-K Triton GEMM (BF16-only)
    "full_fusion",              # autotune: the everything-on headline max
]

# Full-matrix measurement defaults. With the tensor-only sliding-cache patch
# (install_compile_friendly_sliding_update) the decode step compiles to ONE graph —
# no per-position recompile — so a config is ~60-120 s compile (max_autotune longer)
# + a few seconds of fast (~25 ms/step) compiled identity/timing; plain-eager decode is
# ~80-105 ms/step. The whole 7-config sweep fits the 80-min budget with wide margin, so
# we can afford 16 prompts for a fine break-rate (k/16) and a 64-token decode (long
# enough for a single rounding flip to cascade and be caught by the per-prompt sha gate).
FULL_N_PROMPTS = 16
FULL_L_DECODE = 64           # decode tokens per prompt (long enough for a flip to cascade).
FULL_WARMUP_STEPS = 12       # decode-loop warmup steps before timing (also triggers compile).
FULL_TIME_REPEATS = 3        # timed decode-loop repeats; take the median per-step.
FULL_BUDGET_S = 4800.0       # wall-clock budget for the config loop (80 min; ~10 min margin).
# Smoke (plumbing-only) defaults.
SMOKE_N_PROMPTS = 3
SMOKE_L_DECODE = 24
SMOKE_WARMUP_STEPS = 8
SMOKE_TIME_REPEATS = 1
MAX_CACHE_LEN = 256          # StaticCache length; >= max(prompt)+L_decode.


# ========================================================================== #
# Inductor fusion-class config matrix
# ========================================================================== #
# Each spec lists the torch._inductor.config knobs to FORCE for that config (the
# rest stay at library default), the fusion CLASS it isolates, whether it is a
# deterministic-pinned config, and the prior expectation. Knobs are applied via
# _apply_inductor_knobs (top-level vs nested triton.* handled by the dotted key).
#
# Reference = plain eager greedy AR (no compile) — the #158 byte-exact contract.
# All other configs are torch.compile'd; capture is numerically transparent (#371)
# so it is held conceptually ON but measured compile-only (see module docstring).
#
# Knob facts (torch 2.11, introspected): epilogue_fusion=True default (pointwise,
# SAFE); split_reductions=True default (top-level); triton.persistent_reductions=
# True default (RMSNorm/softmax single-pass reorder); triton.cooperative_reductions=
# False default; coordinate_descent_tuning / max_autotune_gemm=False default.
def fusion_config_specs() -> dict[str, dict[str, Any]]:
    return {
        # ---- ATen-kernel identity CONTROLS (backend != inductor; no Triton codegen) ----
        # The decisive "pin": staying on ATen kernels entirely. Inductor's Triton
        # reduction templates (RMSNorm inline-warp reduction, hidden=2560) accumulate in
        # a different order than eager's ATen kernels and NO inductor knob makes them
        # bit-identical (pytorch #107498) — so the strongest reduction-order "pin" is to
        # not emit Triton at all. These localize WHERE compile diverges from eager:
        #   dynamo_eager : graph capture only, original ATen ops, NO decomposition  -> expect identity==1.0
        #   aot_eager    : + AOTAutograd core-ATen decomposition, ATen exec, NO Triton -> tests if decomp alone reorders
        # They are NOT fusion (no kernel fusion) — any speedup is launch/dispatch
        # overhead reduction (the #371 capture lane), reported separately from fusion.
        "dynamo_eager": {
            "knobs": {},
            "backend": "eager",
            "is_fusion": False,
            "fusion_class": "capture_aten_no_triton",
            "deterministic": False,
            "expected": "safe",
            "desc": "dynamo graph capture; original ATen kernels, NO Triton/decomp (identity control, overhead-only)",
        },
        "aot_eager": {
            "knobs": {},
            "backend": "aot_eager",
            "is_fusion": False,
            "fusion_class": "aot_decomp_aten_no_triton",
            "deterministic": False,
            "expected": "safe",
            "desc": "AOTAutograd core-ATen decomposition, ATen exec, NO Triton codegen (tests decomp reorder)",
        },
        # full inductor fusion: every reduction-reorder + tiling + split-K GEMM on.
        # The fast-but-(expected-)broken headline. capture conceptually ON.
        "full_fusion": {
            "knobs": {
                "epilogue_fusion": True,
                "split_reductions": True,
                "triton.persistent_reductions": True,
                "max_autotune": True,
                "max_autotune_gemm": True,
                "coordinate_descent_tuning": True,
            },
            "backend": "inductor",
            "is_fusion": True,
            "fusion_class": "full",
            "deterministic": False,
            "expected": "break",
            "desc": "all inductor fusion: persistent+split reductions, coord-descent, split-K GEMM",
        },
        # library-default torch.compile: epilogue + persistent + split reductions,
        # ATEN GEMM (no split-K). Isolates the default reduction-fusion class.
        "compile_default": {
            "knobs": {},  # library defaults
            "backend": "inductor",
            "is_fusion": True,
            "fusion_class": "reduction_default",
            "deterministic": False,
            "expected": "break",
            "desc": "torch.compile default (persistent+split reductions, ATEN GEMM, epilogue)",
        },
        # pointwise/elementwise epilogue ONLY — every reduction-reorder path off.
        # Expected SAFE; this is the identity-safe-fusion ceiling.
        "pointwise_only": {
            "knobs": {
                "epilogue_fusion": True,
                "split_reductions": False,
                "triton.persistent_reductions": False,
                "triton.cooperative_reductions": False,
                "coordinate_descent_tuning": False,
                "max_autotune": False,
                "max_autotune_gemm": False,
            },
            "backend": "inductor",
            "is_fusion": True,
            "fusion_class": "pointwise_epilogue",
            "deterministic": False,
            "expected": "safe",
            "desc": "pointwise epilogue only (silu/gelu/residual/scale); all reductions single-pass",
        },
        # isolate the persistent-reduction class (RMSNorm + softmax): default minus
        # persistent reductions. If this restores identity vs compile_default, the
        # persistent reduction is the/ a breaker.
        "no_persistent_reductions": {
            "knobs": {"triton.persistent_reductions": False},
            "backend": "inductor",
            "is_fusion": True,
            "fusion_class": "reduction_persistent",
            "deterministic": False,
            "expected": "safe",
            "desc": "default minus persistent reductions (force multi-pass RMSNorm/softmax)",
        },
        # isolate split-K GEMM (Triton matmul templates): default + GEMM autotune.
        # NB: UNREACHABLE on the deployed int4/Marlin path (custom CUDA op, #122).
        "gemm_split_k": {
            "knobs": {"max_autotune": True, "max_autotune_gemm": True},
            "backend": "inductor",
            "is_fusion": True,
            "fusion_class": "matmul_splitk",
            "deterministic": False,
            "expected": "break",
            "desc": "default + split-K Triton GEMM templates (BF16-only; int4 GEMM is Marlin, unreachable)",
        },
        # isolate coordinate-descent tiling (changes BLOCK_SIZE -> reduction bounds).
        "coordinate_descent": {
            "knobs": {"coordinate_descent_tuning": True},
            "backend": "inductor",
            "is_fusion": True,
            "fusion_class": "tiling_coorddesc",
            "deterministic": False,
            "expected": "break",
            "desc": "default + coordinate-descent tiling tuning (changes reduction tiling)",
        },
        # DETERMINISTIC PIN: pointwise epilogue + every reduction forced single-pass
        # + use_deterministic_algorithms + cuBLAS deterministic workspace + ATEN GEMM.
        # The recovery config: does pinning hold identity at fusion speed?
        "deterministic_pinned": {
            "knobs": {
                "epilogue_fusion": True,
                "split_reductions": False,
                "triton.persistent_reductions": False,
                "triton.cooperative_reductions": False,
                "coordinate_descent_tuning": False,
                "max_autotune": False,
                "max_autotune_gemm": False,
                "max_autotune_gemm_backends": "ATEN",
            },
            "backend": "inductor",
            "is_fusion": True,
            "fusion_class": "pinned",
            "deterministic": True,
            "expected": "safe",
            "desc": "pinned: pointwise epilogue + single-pass reductions + deterministic algos + ATEN GEMM",
        },
    }


# ========================================================================== #
# Identity gate (byte-exact greedy-token-identity, #158 contract)
# ========================================================================== #
def _sha_tokens(token_ids: list[int]) -> str:
    return hashlib.sha256((",".join(str(int(t)) for t in token_ids)).encode()).hexdigest()


def _per_prompt_identity(ref_sha: dict[str, str], cfg_sha: dict[str, str]) -> dict[str, Any]:
    """Per-prompt byte-exact identity of a config's greedy completions vs the eager
    reference. ``token_identity_rate`` = fraction of prompts whose ENTIRE decoded
    sequence sha-matches eager (== 1.0 iff byte-exact on every prompt). Also a
    finer ``per_token_identity_rate`` if raw token lists are available downstream."""
    ids = sorted(set(ref_sha) & set(cfg_sha))
    matched = sum(1 for i in ids if ref_sha[i] == cfg_sha[i])
    rate = matched / len(ids) if ids else float("nan")
    return {
        "token_identity_rate": rate,
        "identity_preserved": bool(ids) and matched == len(ids),
        "n_prompts_compared": len(ids),
        "n_matched": matched,
    }


def _gate_logic_self_check() -> dict[str, Any]:
    """GPU-INDEPENDENT proof the identity gate flags a known-breaking config as
    ``< 1.0`` and a known-safe config as ``== 1.0``. Synthetic fixtures: a safe map
    identical to the reference, and a breaking map with exactly one flipped token in
    one prompt. PRIMARY self-test requirement."""
    ref_tokens = {"0": [10, 20, 30, 40], "1": [11, 21, 31, 41], "2": [12, 22, 32, 42]}
    safe_tokens = {k: list(v) for k, v in ref_tokens.items()}
    break_tokens = {k: list(v) for k, v in ref_tokens.items()}
    break_tokens["1"][2] = 999  # flip exactly one token in prompt "1"
    ref_sha = {k: _sha_tokens(v) for k, v in ref_tokens.items()}
    safe = _per_prompt_identity(ref_sha, {k: _sha_tokens(v) for k, v in safe_tokens.items()})
    brk = _per_prompt_identity(ref_sha, {k: _sha_tokens(v) for k, v in break_tokens.items()})
    flags_safe = safe["token_identity_rate"] == 1.0 and safe["identity_preserved"] is True
    flags_break = brk["token_identity_rate"] < 1.0 and brk["identity_preserved"] is False
    return {
        "known_safe_rate": safe["token_identity_rate"],
        "known_break_rate": brk["token_identity_rate"],
        "gate_flags_known_safe_as_identical": bool(flags_safe),
        "gate_flags_known_break_as_divergent": bool(flags_break),
        "ok": bool(flags_safe and flags_break),
    }


# ========================================================================== #
# GPU measurement (standalone torch.compile micro-harness)
# ========================================================================== #
def _apply_inductor_knobs(knobs: dict[str, Any]) -> dict[str, Any]:
    """Force the given torch._inductor.config knobs; return the prior values so the
    caller can restore. Dotted keys (``triton.persistent_reductions``) address the
    nested config class."""
    import torch._inductor.config as ind

    prior: dict[str, Any] = {}
    for key, val in knobs.items():
        if "." in key:
            head, attr = key.split(".", 1)
            obj = getattr(ind, head)
        else:
            obj, attr = ind, key
        prior[key] = getattr(obj, attr)
        setattr(obj, attr, val)
    return prior


def _restore_inductor_knobs(prior: dict[str, Any]) -> None:
    import torch._inductor.config as ind

    for key, val in prior.items():
        if "." in key:
            head, attr = key.split(".", 1)
            setattr(getattr(ind, head), attr, val)
        else:
            setattr(ind, key, val)


def install_compile_friendly_sliding_update() -> Any:
    """Monkeypatch ``StaticSlidingWindowLayer.update`` to a TENSOR-only fast path so
    the compiled M=1 decode step does NOT recompile once per position.

    WHY. gemma4 decode otherwise compiles a fresh graph PER position: the sliding
    cache layer reads ``self.cumulative_length_int`` — a *python int* that increments
    every step — to pick its is_full / becoming-full / not-full branch (cache_utils
    ``StaticSlidingWindowLayer.update`` ~L426). torch.compile specializes on that int
    value, so each new position is a near-full recompile (smoke: 39 frames / 40 steps,
    0.01x "speedup", and a hard timeout). The FULL-attention ``StaticLayer`` already
    tracks length with a *tensor* (``cumulative_length``) and never recompiles; only the
    sliding layer regresses. This replaces the sliding update with exactly the library's
    own not-full ``index_copy_`` branch, driven by the tensor — so it never reads a python
    int inside the compiled region. Result (validated): one compile, frames==1, ~3.3x.

    IN-REGIME BIT-EXACTNESS. With ``MAX_CACHE_LEN==256`` and every prompt truncated so
    ``prefill + L_decode < 256``, the sliding cache (effective length
    ``min(sliding_window=512, 256) == 256``) NEVER fills, so the *original* update ALWAYS
    takes that same not-full branch. The patch is therefore numerically identical to plain
    eager in-regime — and ``run_measurement`` re-proves it every run via an explicit
    unpatched-vs-patched eager token-sha check before the config loop.

    The python int is no longer maintained by ``update`` (it would re-introduce the int
    read → recompiles). Instead the harness sets it = current position at mask-build time
    (eager), where the masking machinery reads it (``get_seq_length`` / ``get_mask_sizes``).
    See ``sync_sliding_cumulative_int``.
    """
    import torch
    from transformers.cache_utils import StaticSlidingWindowLayer

    if getattr(StaticSlidingWindowLayer, "_senpai_compile_patched", False):
        return StaticSlidingWindowLayer

    def _tensor_only_update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        kv_length = key_states.shape[-2]
        cache_position = torch.arange(kv_length, device=self.device) + self.cumulative_length
        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except NotImplementedError:  # pragma: no cover - MPS-only fallback
            self.keys[:, :, cache_position] = key_states
            self.values[:, :, cache_position] = value_states
        self.cumulative_length.add_(kv_length)
        return self.keys, self.values

    StaticSlidingWindowLayer._senpai_orig_update = StaticSlidingWindowLayer.update
    StaticSlidingWindowLayer.update = _tensor_only_update
    StaticSlidingWindowLayer._senpai_compile_patched = True
    return StaticSlidingWindowLayer


def sync_sliding_cumulative_int(cache: Any, position: int) -> None:
    """Set each sliding layer's python ``cumulative_length_int`` = ``position`` (the
    number of tokens cached before this step). Called in eager mask-build, where the
    masking machinery reads it; the compiled update (patched) never reads it. This keeps
    the hand-built mask bit-identical to the model's own (validated)."""
    for layer in getattr(cache, "layers", []):
        if hasattr(layer, "cumulative_length_int"):
            layer.cumulative_length_int = int(position)


def run_measurement(args: argparse.Namespace, wandb_run: Any = None) -> dict[str, Any]:
    """Load the BF16 text model once (eager). Build the plain-eager greedy AR
    identity reference. Then, per fusion config: reset dynamo, force the inductor
    knobs, torch.compile the fixed-shape M=1 decode step, warm it up, and measure
    (per-prompt token sha vs eager, steady per-step us). Reuses ONE StaticCache +
    static input buffers so the compiled step is fixed-shape and compiles once.

    If ``wandb_run`` is given (created up-front, before the compile), each config's
    gate result is streamed as a heartbeat so the run reads live during the sweep."""
    import os

    import torch

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # cuBLAS deterministic workspace — needed for the pinned config; harmless else.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_float32_matmul_precision("high")
    # Safety net for the int-position recompile footgun (see decode_step): even with
    # tensor position_ids the first few decode positions may specialize once; a larger
    # cache keeps those compiled instead of bailing to eager at the default limit of 8.
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers import StaticCache
    except Exception:  # pragma: no cover
        from transformers.cache_utils import StaticCache

    try:
        from scripts.local_validation import paths as paths_mod
        for note in paths_mod.prepare_local_gpu_env():
            print(f"[measure] {note}", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"[measure] (no local_validation paths shim: {exc!r})", flush=True)
        paths_mod = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("deterministic-fusion-identity needs a CUDA device")

    n_prompts = args.n_prompts
    l_decode = args.l_decode
    warmup_steps = args.warmup_steps
    time_repeats = args.time_repeats

    print(f"[measure] loading {MODEL_ID} (bf16, eager attn) ...", flush=True)
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()
    print(f"[measure] loaded in {time.time() - t_load:.1f}s", flush=True)

    prompts = _load_prompts(tok, n_prompts, paths_mod)
    print(f"[measure] {len(prompts)} prompts; L_decode={l_decode} warmup={warmup_steps}", flush=True)

    cache = StaticCache(
        config=model.config, max_batch_size=1, max_cache_len=MAX_CACHE_LEN,
        device=device, dtype=torch.bfloat16,
    )
    static_tok = torch.zeros(1, 1, dtype=torch.long, device=device)
    static_pos = torch.zeros(1, dtype=torch.long, device=device)

    # Build the decode causal masks OURSELVES (eagerly) and feed them into the compiled
    # forward as fixed-shape tensor args. gemma4's forward otherwise rebuilds the mask
    # each step from the cache's python-int seq-length (create_masks_for_generate), which
    # torch.compile specializes on -> a fresh graph per decode position. Building the mask
    # outside the compiled region (where the int is harmless) and passing the (1,1,1,L)
    # tensors in keeps the per-position dependence inside *tensor values* of fixed shape.
    # Validated bit-exact vs the model's own mask (smoke6: identical token sha).
    from transformers.masking_utils import (
        create_causal_mask, create_sliding_window_causal_mask,
    )
    tcfg = model.config.get_text_config()
    dummy_emb = torch.zeros(1, 1, tcfg.hidden_size, dtype=torch.bfloat16, device=device)

    def _build_masks() -> tuple["torch.Tensor", "torch.Tensor"]:
        # Keep the sliding layers' python int correct here (eager), where the masking
        # machinery reads it; the patched compiled update no longer maintains it.
        sync_sliding_cumulative_int(cache, int(static_pos.item()))
        mk = dict(config=tcfg, inputs_embeds=dummy_emb, attention_mask=None,
                  past_key_values=cache, position_ids=static_pos.view(1, -1))
        return create_causal_mask(**mk), create_sliding_window_causal_mask(**mk)

    @torch.no_grad()
    def prefill(ids: "torch.Tensor") -> "torch.Tensor":
        cpos = torch.arange(ids.shape[1], device=device)
        out = model(input_ids=ids, past_key_values=cache, cache_position=cpos,
                    position_ids=cpos.view(1, -1), use_cache=True)
        return out.logits[:, -1, :]

    @torch.no_grad()
    def decode_step(input_ids: "torch.Tensor", cache_position: "torch.Tensor",
                    full_mask: "torch.Tensor", sliding_mask: "torch.Tensor") -> "torch.Tensor":
        # position_ids AND the attention masks are passed as TENSORS so the gemma4
        # forward takes neither python-int code path (position arange+past_seen_tokens,
        # nor create_masks_for_generate's int seq-length) — both of which torch.compile
        # would specialize on, recompiling a fresh graph per decode position. The dict
        # is reconstructed here from the two tensor args (constant keys, no int).
        out = model(input_ids=input_ids, past_key_values=cache,
                    cache_position=cache_position, position_ids=cache_position.view(1, -1),
                    attention_mask={"full_attention": full_mask, "sliding_attention": sliding_mask},
                    use_cache=True)
        return out.logits[:, -1, :]

    @torch.no_grad()
    def greedy_ar(prompt_ids: "torch.Tensor", n_new: int, step_fn) -> list[int]:
        try:
            cache.reset()
        except Exception:
            pass
        plen = int(prompt_ids.shape[1])
        nxt = int(prefill(prompt_ids)[0].argmax())
        gen = [nxt]
        for i in range(n_new - 1):
            static_tok.copy_(torch.tensor([[nxt]], device=device))
            static_pos.copy_(torch.tensor([plen + i], device=device))
            fm, sm = _build_masks()
            nxt = int(step_fn(static_tok, static_pos, fm, sm)[0].argmax())
            gen.append(nxt)
        return gen

    @torch.no_grad()
    def steady_per_step_us(prompt_ids: "torch.Tensor", step_fn) -> float:
        """Median steady-state per-step us over a fixed decode loop (post-warmup)."""
        try:
            cache.reset()
        except Exception:
            pass
        plen = int(prompt_ids.shape[1])
        nxt = int(prefill(prompt_ids)[0].argmax())
        # Steady-state per-step cost is position-independent (same kernels/work), so we
        # time at ONE fixed position with ONE pre-built mask: that keeps a single compiled
        # graph hot and excludes the eager mask-build python overhead from the timer. The
        # token written is fixed; we re-run the same step to measure pure forward latency.
        static_tok.copy_(torch.tensor([[nxt]], device=device))
        static_pos.copy_(torch.tensor([plen], device=device))
        fm, sm = _build_masks()
        for _ in range(warmup_steps):  # also the compile trigger on the first call
            _ = step_fn(static_tok, static_pos, fm, sm)
        samples = []
        for _ in range(time_repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(warmup_steps):
                _ = step_fn(static_tok, static_pos, fm, sm)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) / warmup_steps * 1e6)
        return statistics.median(samples)

    prompt_tensors = [
        tok(p, return_tensors="pt").input_ids.to(device)[:, : MAX_CACHE_LEN - l_decode - 2]
        for p in prompts
    ]

    def _vram_gb() -> float:
        return torch.cuda.max_memory_allocated() / 1e9

    # ---- sliding-cache compile patch + bit-exactness proof ----
    # The compiled configs MUST use the tensor-only sliding update or they recompile once
    # per decode position (full recompile each step -> timeout). The patch is bit-exact
    # in-regime (prefill+L_decode < MAX_CACHE_LEN), but we PROVE it here every run: greedy
    # AR on prompt 0 UNPATCHED vs PATCHED must yield the identical completion-token sha.
    # If it ever diverges (regime broken), abort — the whole identity reference would be
    # invalid. The eager reference + every config then run on the patched path, so the
    # ONLY numerical difference measured across configs is the inductor fusion class.
    print("[measure] proving sliding-cache compile-patch bit-exactness ...", flush=True)
    sha_unpatched_p0 = _sha_tokens(greedy_ar(prompt_tensors[0], l_decode, decode_step))
    install_compile_friendly_sliding_update()
    sha_patched_p0 = _sha_tokens(greedy_ar(prompt_tensors[0], l_decode, decode_step))
    patch_bit_exact = sha_unpatched_p0 == sha_patched_p0
    print(f"[measure] patch bit-exact={patch_bit_exact} "
          f"(unpatched {sha_unpatched_p0} vs patched {sha_patched_p0})", flush=True)
    if not patch_bit_exact:
        raise RuntimeError(
            "sliding-cache compile patch is NOT bit-exact vs plain eager "
            f"({sha_unpatched_p0} != {sha_patched_p0}); regime broken "
            f"(prefill+L_decode must stay < MAX_CACHE_LEN={MAX_CACHE_LEN}); aborting")

    # ---- eager reference (the #158 byte-exact contract; on the patched path) ----
    print("[measure] eager reference (plain greedy AR) ...", flush=True)
    eager_sha = {str(i): _sha_tokens(greedy_ar(pt, l_decode, decode_step))
                 for i, pt in enumerate(prompt_tensors)}
    eager_us = steady_per_step_us(prompt_tensors[0], decode_step)
    eager = {
        "token_sha_by_idx": eager_sha,
        "per_step_us": eager_us,
        "decode_tps_local": 1e6 / eager_us if eager_us and math.isfinite(eager_us) else float("nan"),
    }
    print(f"[measure] eager: {eager_us:.0f} us/step  {eager['decode_tps_local']:.2f} local TPS", flush=True)

    # ---- live per-class heartbeat (no-op unless an up-front W&B run was passed) ----
    _hb_log = None
    if wandb_run is not None:
        try:
            from scripts.wandb_logging import log_event as _hb_log
        except Exception as exc:  # pragma: no cover
            print(f"[wandb] heartbeat logger unavailable: {exc!r}", flush=True)
            _hb_log = None
    _hb_state = {"step": 0}

    def _heartbeat(name: str, rec: dict[str, Any]) -> None:
        if wandb_run is None or _hb_log is None:
            return
        _hb_state["step"] += 1
        try:
            sha = rec.get("token_sha_by_idx") or {}
            rate = (_per_prompt_identity(eager_sha, sha)["token_identity_rate"]
                    if sha else float("nan"))
            metrics = {
                f"hb/{name}/identity_rate": rate,
                f"hb/{name}/tps": rec.get("decode_tps_local"),
                f"hb/{name}/is_fusion": int(bool(rec.get("is_fusion"))),
                "hb/classes_done": _hb_state["step"],
            }
            _hb_log(wandb_run, "class_gated", step=_hb_state["step"],
                    metrics={k: v for k, v in metrics.items() if v is not None},
                    data={"class": name, "fusion_class": rec.get("fusion_class")})
            print(f"[wandb] heartbeat {_hb_state['step']}: {name} "
                  f"identity_rate={rate:.3f} tps={rec.get('decode_tps_local')}", flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[wandb] heartbeat failed ({name}): {exc!r}", flush=True)

    # ---- result skeleton + incremental persistence ----
    # Each config compile is ~400 s, so we dump a partial measured-block after the
    # eager ref and after EVERY config: a wall-clock kill (or the budget guard) still
    # leaves a JSON that --reanalyze can turn into the deliverables.
    configs: dict[str, dict[str, Any]] = {}
    measured: dict[str, Any] = {
        "model_id": MODEL_ID,
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "n_prompts": len(prompts),
        "l_decode": l_decode,
        "warmup_steps": warmup_steps,
        "time_repeats": time_repeats,
        "max_cache_len": MAX_CACHE_LEN,
        "peak_vram_gb": _vram_gb(),
        "eager": eager,
        "configs": configs,
        "budget_s": float(getattr(args, "budget_s", FULL_BUDGET_S)),
        "sliding_patch_bit_exact": bool(patch_bit_exact),
        "sliding_patch_sha_unpatched": sha_unpatched_p0,
        "sliding_patch_sha_patched": sha_patched_p0,
    }

    def _dump_partial(stage: str) -> None:
        measured["peak_vram_gb"] = _vram_gb()
        measured["last_stage"] = stage
        try:
            OUT_ROOT.mkdir(parents=True, exist_ok=True)
            PARTIAL_JSON.write_text(json.dumps(
                {"measured": measured, "partial": True, "last_stage": stage,
                 "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())},
                indent=2, sort_keys=True))
        except Exception as exc:  # pragma: no cover
            print(f"[measure] (partial dump failed: {exc!r})", flush=True)

    _dump_partial("eager")

    # ---- fusion configs ----
    specs = fusion_config_specs()
    # Cheap, scientifically-essential configs first; expensive max_autotune ones
    # (gemm_split_k, full_fusion) last so the budget guard sheds THEM first.
    run_order = [n for n in CONFIG_RUN_ORDER if n in specs] + \
                [n for n in specs if n not in CONFIG_RUN_ORDER]
    if args.only:
        wanted = set(args.only.split(","))
        run_order = [n for n in run_order if n in wanted]
    t_start = time.time()
    budget_s = float(getattr(args, "budget_s", FULL_BUDGET_S))
    for name in run_order:
        spec = specs[name]
        elapsed = time.time() - t_start
        if elapsed > budget_s:
            print(f"[measure] budget {budget_s:.0f}s exceeded ({elapsed:.0f}s) — "
                  f"SKIP {name} (and any remaining)", flush=True)
            configs[name] = {
                "fusion_class": spec["fusion_class"], "deterministic": spec["deterministic"],
                "expected": spec["expected"], "desc": spec["desc"], "knobs": spec["knobs"],
                "backend": spec.get("backend", "inductor"),
                "is_fusion": bool(spec.get("is_fusion", spec.get("backend", "inductor") == "inductor")),
                "token_sha_by_idx": {}, "per_step_us": float("nan"),
                "decode_tps_local": float("nan"), "error": "skipped: budget",
            }
            _dump_partial(f"skip:{name}")
            _heartbeat(name, configs[name])
            continue
        print(f"\n[measure] ==== config {name} ==== {spec['desc']}  "
              f"(elapsed {elapsed:.0f}s / {budget_s:.0f}s)", flush=True)
        backend = spec.get("backend", "inductor")
        cfg_rec: dict[str, Any] = {
            "fusion_class": spec["fusion_class"],
            "deterministic": spec["deterministic"],
            "expected": spec["expected"],
            "desc": spec["desc"],
            "knobs": spec["knobs"],
            "backend": backend,
            "is_fusion": bool(spec.get("is_fusion", backend == "inductor")),
        }
        prior_knobs = None
        det_was = torch.are_deterministic_algorithms_enabled()
        try:
            torch._dynamo.reset()
            prior_knobs = _apply_inductor_knobs(spec["knobs"])
            if spec["deterministic"]:
                torch.use_deterministic_algorithms(True, warn_only=True)
            t_c = time.time()
            compiled = torch.compile(decode_step, backend=backend, fullgraph=False)
            sha_map = {str(i): _sha_tokens(greedy_ar(pt, l_decode, compiled))
                       for i, pt in enumerate(prompt_tensors)}
            us = steady_per_step_us(prompt_tensors[0], compiled)
            cfg_rec.update({
                "token_sha_by_idx": sha_map,
                "per_step_us": us,
                "decode_tps_local": 1e6 / us if us and math.isfinite(us) else float("nan"),
                "compile_run_s": time.time() - t_c,
                "error": None,
            })
            ident = _per_prompt_identity(eager_sha, sha_map)
            print(f"[measure] {name}: identity_rate={ident['token_identity_rate']:.3f} "
                  f"preserved={ident['identity_preserved']}  {us:.0f} us/step "
                  f"{cfg_rec['decode_tps_local']:.2f} TPS", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[measure] config {name} FAILED: {exc!r}", flush=True)
            cfg_rec.update({
                "token_sha_by_idx": {}, "per_step_us": float("nan"),
                "decode_tps_local": float("nan"), "error": repr(exc),
            })
        finally:
            if prior_knobs is not None:
                _restore_inductor_knobs(prior_knobs)
            if spec["deterministic"] and not det_was:
                torch.use_deterministic_algorithms(False)
        configs[name] = cfg_rec
        _dump_partial(name)
        _heartbeat(name, cfg_rec)

    measured["peak_vram_gb"] = _vram_gb()
    return measured


def _load_prompts(tok: Any, n: int, paths_mod: Any) -> list[str]:
    """Official sharegpt eval prompts (same set the #359 / challenge gate uses) when
    reachable; else a small varied fallback so --smoke works anywhere."""
    if paths_mod is not None:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("official_decode", str(paths_mod.DECODE_SCRIPT))
            od = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(od)
            records = od.read_sharegpt_prompts(Path(paths_mod.EVAL_PROMPTS), num_prompts=n, seed=paths_mod.SEED)
            prompts = [r["prompt_text"] for r in records]
            if prompts:
                return prompts[:n]
        except Exception as exc:  # pragma: no cover
            print(f"[measure] (sharegpt load failed, using fallback prompts: {exc!r})", flush=True)
    fallback = [
        "The capital of France is", "Explain quantum entanglement in simple terms:",
        "Write a haiku about the ocean.", "List three reasons regular exercise is good:",
        "Summarize the plot of Romeo and Juliet:", "What is the boiling point of water at sea level?",
        "Translate 'good morning' into Spanish and French:", "Describe how a rainbow forms:",
        "Give me a recipe for a simple omelette:", "Who wrote the novel Pride and Prejudice?",
        "Explain the difference between TCP and UDP:", "What causes the seasons on Earth?",
        "Name the planets of the solar system in order:", "How does a bicycle stay upright when moving?",
        "Write a short motivational quote about learning:", "What is the Pythagorean theorem?",
    ]
    return (fallback * ((n // len(fallback)) + 1))[:n]


# ========================================================================== #
# Analysis — deliverables
# ========================================================================== #
def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def analyze(measured: dict[str, Any]) -> dict[str, Any]:
    eager = measured["eager"]
    eager_sha = eager["token_sha_by_idx"]
    eager_tps = eager.get("decode_tps_local", float("nan"))
    eager_us = eager.get("per_step_us", float("nan"))

    per_config: dict[str, Any] = {}
    for name, cfg in measured["configs"].items():
        ident = _per_prompt_identity(eager_sha, cfg.get("token_sha_by_idx", {}))
        backend = cfg.get("backend", "inductor")
        per_config[name] = {
            "fusion_class": cfg["fusion_class"],
            "deterministic": cfg["deterministic"],
            "expected": cfg["expected"],
            "desc": cfg["desc"],
            "backend": backend,
            "is_fusion": bool(cfg.get("is_fusion", backend == "inductor")),
            "per_step_us": cfg.get("per_step_us", float("nan")),
            "decode_tps_local": cfg.get("decode_tps_local", float("nan")),
            "token_identity_rate": ident["token_identity_rate"],
            "identity_preserved": ident["identity_preserved"],
            "n_prompts_compared": ident["n_prompts_compared"],
            "error": cfg.get("error"),
        }

    def tps(name: str) -> float:
        return per_config.get(name, {}).get("decode_tps_local", float("nan"))

    def rate(name: str) -> float:
        return per_config.get(name, {}).get("token_identity_rate", float("nan"))

    # All measured identity-BREAKING compiled configs (the realized "fast-but-broken"
    # fusion points). The fastest of these is the speedup denominator; if the literal
    # full_fusion config ran it is normally that one, but when full_fusion is shed by
    # the budget guard we fall back to the fastest measured breaker so the headline +
    # fraction stay populated (source recorded in full_fusion_tps_source).
    breaking_meas = {
        n: v for n, v in per_config.items()
        if (not v["identity_preserved"]) and not v.get("error")
        and v["n_prompts_compared"] > 0 and _finite(v["decode_tps_local"])
    }
    fused_break_max_config = (
        max(breaking_meas, key=lambda n: breaking_meas[n]["decode_tps_local"]) if breaking_meas else None
    )
    fused_break_max_tps = (
        breaking_meas[fused_break_max_config]["decode_tps_local"] if fused_break_max_config else float("nan")
    )

    # headline "full fusion" = literal full_fusion if measured, else fastest breaker.
    if _finite(tps("full_fusion")):
        full_fusion_tps = tps("full_fusion")
        full_fusion_rate = rate("full_fusion")
        full_fusion_tps_source = "full_fusion"
    elif fused_break_max_config is not None:
        full_fusion_tps = fused_break_max_tps
        full_fusion_rate = rate(fused_break_max_config)
        full_fusion_tps_source = fused_break_max_config
    else:
        full_fusion_tps = tps("full_fusion")
        full_fusion_rate = rate("full_fusion")
        full_fusion_tps_source = None

    # Identity-preserving COMPILED configs (rate == 1.0, finite tps, no error). Split
    # into (i) real Triton FUSION configs (is_fusion) and (ii) ATen-backend CONTROLS
    # (dynamo_eager / aot_eager: capture/decomp, NO Triton -> no kernel fusion, only
    # launch/dispatch-overhead reduction = the #371 capture lane, not fusion). Step 4
    # asks for max TPS at rate==1.0 over {capture + identity-safe fusions + pinned
    # breakers}, so identity_safe_fusion_max_tps composes ALL identity-preserving
    # compiled configs; any_real_fusion_identity_safe records whether ANY actual Triton
    # fusion held identity (the strict yes/no for this card's fusion-numerics lever).
    safe_any = {
        n: v for n, v in per_config.items()
        if v["identity_preserved"] and not v.get("error") and _finite(v["decode_tps_local"])
    }
    safe_real_fusion = {n: v for n, v in safe_any.items() if v.get("is_fusion")}
    # NULL-HONEST fallback: when NO compiled config holds identity, the only byte-exact
    # "config" is plain eager (no fusion) — report it so the headline stays finite and
    # the measured null (a valid result, see build_self_test) is representable.
    if safe_any:
        identity_safe_fusion_max_config = max(safe_any, key=lambda n: safe_any[n]["decode_tps_local"])
        identity_safe_fusion_max_tps = safe_any[identity_safe_fusion_max_config]["decode_tps_local"]
        identity_safe_fusion_is_real_fusion = bool(safe_any[identity_safe_fusion_max_config].get("is_fusion"))
    else:
        identity_safe_fusion_max_config = "eager(no-identity-safe-compile)"
        identity_safe_fusion_max_tps = eager_tps
        identity_safe_fusion_is_real_fusion = False
    any_real_fusion_identity_safe = bool(safe_real_fusion)

    # deterministic-pinned (inductor-knob pin) configs that hold identity.
    pinned_safe = {
        n: v for n, v in per_config.items()
        if v["deterministic"] and v["identity_preserved"]
        and not v.get("error") and _finite(v["decode_tps_local"])
    }
    deterministic_pin_holds_identity = bool(pinned_safe)
    deterministic_pinned_max_tps = (
        max(v["decode_tps_local"] for v in pinned_safe.values())
        if pinned_safe else eager_tps  # pin failed -> only eager is byte-exact (finite null)
    )

    # backends with >= 1 identity-preserving config (eager / aot_eager / inductor).
    which_backends_preserve_identity = sorted({
        per_config[n].get("backend", "inductor") for n in safe_any
    })
    # The identity "pin" needs an ATen backend if NO inductor config holds identity but
    # some non-inductor (eager/aot_eager) one does.
    inductor_holds = any(per_config[n].get("backend", "inductor") == "inductor" for n in safe_any)
    aten_holds = any(per_config[n].get("backend", "inductor") != "inductor" for n in safe_any)
    identity_pin_requires_aten_backend = bool(aten_holds and not inductor_holds)

    # which fusion CLASSES break identity (measured, no error).
    which_break = sorted({
        v["fusion_class"] for v in per_config.values()
        if (not v["identity_preserved"]) and not v.get("error")
        and v["n_prompts_compared"] > 0
    })
    # a class is "pinnable" if it breaks in a non-pinned config but the inductor-knob
    # deterministic pin holds identity (the PR's step-3 question, answered strictly).
    breaking_pinnable = bool(which_break) and deterministic_pin_holds_identity

    # fraction of the full-fusion speedup that is identity-safe-recoverable. With the
    # null-honest fallback identity_safe_fusion_max_tps>=eager_tps is finite, so frac is
    # finite-by-construction: 0.0 when only eager is byte-exact.
    speedup_denom = full_fusion_tps - eager_tps
    if _finite(speedup_denom) and abs(speedup_denom) > 1e-9:
        frac_recovered_safe = (identity_safe_fusion_max_tps - eager_tps) / speedup_denom
        frac_recovered_pinned = (deterministic_pinned_max_tps - eager_tps) / speedup_denom
    elif _finite(full_fusion_rate) and full_fusion_rate >= 1.0:
        # full fusion itself never broke identity -> 100% recoverable (informative null).
        frac_recovered_safe = 1.0
        frac_recovered_pinned = 1.0
    else:
        frac_recovered_safe = 0.0
        frac_recovered_pinned = 0.0

    # best strict-identity TPS overall (the composable base-lift over eager).
    best_strict_tps = max(
        [t for t in (identity_safe_fusion_max_tps, deterministic_pinned_max_tps, eager_tps)
         if _finite(t)],
        default=float("nan"),
    )
    fusion_speedup_x = full_fusion_tps / eager_tps if _finite(full_fusion_tps) and _finite(eager_tps) and eager_tps > 0 else float("nan")

    return {
        "eager_byte_exact_tps": eager_tps,
        "eager_per_step_us": eager_us,
        "full_fusion_tps": full_fusion_tps,
        "full_fusion_token_identity_rate": full_fusion_rate,
        "full_fusion_tps_source": full_fusion_tps_source,
        "fused_break_max_tps": fused_break_max_tps,
        "fused_break_max_config": fused_break_max_config,
        "fusion_speedup_x": fusion_speedup_x,
        "identity_safe_fusion_max_tps": identity_safe_fusion_max_tps,
        "identity_safe_fusion_max_config": identity_safe_fusion_max_config,
        "identity_safe_fusion_is_real_fusion": identity_safe_fusion_is_real_fusion,
        "any_real_fusion_identity_safe": any_real_fusion_identity_safe,
        "deterministic_pinned_max_tps": deterministic_pinned_max_tps,
        "deterministic_pin_holds_identity": deterministic_pin_holds_identity,
        "frac_fusion_speedup_recovered_identity_safe": frac_recovered_safe,
        "frac_fusion_speedup_recovered_deterministic_pinned": frac_recovered_pinned,
        "which_fusions_break_identity": which_break,
        "breaking_fusions_pinnable_deterministic": breaking_pinnable,
        "which_backends_preserve_identity": which_backends_preserve_identity,
        "identity_pin_requires_aten_backend": identity_pin_requires_aten_backend,
        "best_strict_identity_tps_local": best_strict_tps,
        "per_config": per_config,
        # anchors carried for context
        "strict_base_official_tps_196": STRICT_BASE_OFFICIAL_TPS_196,
        "bw_ceiling_353": BW_CEILING_353,
        "batch_invariant_357": BATCH_INVARIANT_357,
        "target_tps": TARGET_TPS,
    }


# ========================================================================== #
# Self-test (PRIMARY: deterministic_fusion_self_test_passes)
# ========================================================================== #
def build_self_test(measured: dict[str, Any] | None, analysis: dict[str, Any] | None) -> dict[str, Any]:
    """PRIMARY deterministic_fusion_self_test_passes.

    A MEASUREMENT-harness integrity test (a measured null is a valid result):
    (a) the identity gate flags a known-breaking config <1.0 and a known-safe ==1.0
        (synthetic, GPU-independent);
    (b) if a measurement is present: eager reference well-formed; every config has a
        token sha map (or an honest error) and a finite/NaN per-step record; all
        headline deliverables finite; NaN-clean over headline floats;
    (c) constants imported exactly.
    """
    st: dict[str, Any] = {}
    gate = _gate_logic_self_check()
    st["gate_logic"] = gate
    st["gate_logic_ok"] = bool(gate["ok"])

    measured_ok = measured is not None and analysis is not None
    st["measured_present"] = bool(measured_ok)
    if measured_ok:
        eager = measured["eager"]
        eus = eager.get("per_step_us", float("nan"))
        st["eager_ref_ok"] = bool(
            len(eager.get("token_sha_by_idx", {})) > 0 and _finite(eus) and eus > 0
        )
        cfg_maps_ok = True
        n_measured = 0
        for name, cfg in measured["configs"].items():
            if not cfg.get("error"):
                if not cfg.get("token_sha_by_idx"):
                    cfg_maps_ok = False
                else:
                    n_measured += 1
            ps = cfg.get("per_step_us")
            if not (isinstance(ps, float)):
                cfg_maps_ok = False
        st["config_maps_ok"] = bool(cfg_maps_ok and n_measured >= 1)
        headline = [
            analysis["eager_byte_exact_tps"],
            analysis["full_fusion_tps"],
            analysis["full_fusion_token_identity_rate"],
            analysis["identity_safe_fusion_max_tps"],
            analysis["frac_fusion_speedup_recovered_identity_safe"],
            analysis["best_strict_identity_tps_local"],
        ]
        st["headline_finite_ok"] = all(isinstance(x, float) and math.isfinite(x) for x in headline)
        st["deliverables_set_ok"] = bool(
            analysis.get("which_fusions_break_identity") is not None
            and isinstance(analysis.get("breaking_fusions_pinnable_deterministic"), bool)
            and analysis.get("identity_safe_fusion_max_config") is not None
        )
        st["nan_clean_ok"] = st["headline_finite_ok"]
        # The compiled-decode infra is only legitimate if the tensor-only sliding-cache
        # patch is bit-exact vs plain eager. A synthetic measurement (no GPU) has no such
        # field and is exempt; a real GPU measurement MUST carry sliding_patch_bit_exact.
        if "sliding_patch_bit_exact" in measured:
            st["sliding_patch_bit_exact_ok"] = bool(measured["sliding_patch_bit_exact"])
        else:
            st["sliding_patch_bit_exact_ok"] = True
    else:
        st["eager_ref_ok"] = False
        st["config_maps_ok"] = False
        st["headline_finite_ok"] = False
        st["deliverables_set_ok"] = False
        st["nan_clean_ok"] = False
        st["sliding_patch_bit_exact_ok"] = False

    st["constants_ok"] = bool(
        TARGET_TPS == 500.0
        and STRICT_BASE_OFFICIAL_TPS_196 == 165.44
        and DEPLOYED_SPEC_OFFICIAL_TPS == 481.53
        and BW_CEILING_353 == 353.0
        and BATCH_INVARIANT_357 == 357.32
        and LAMBDA1_CEILING == 520.953
    )

    # When a measurement is present, require the full chain; 0-GPU integrity mode
    # (no measurement) passes on the gate-logic + constants alone.
    if measured_ok:
        st["passes"] = bool(
            st["gate_logic_ok"] and st["measured_present"] and st["eager_ref_ok"]
            and st["config_maps_ok"] and st["headline_finite_ok"]
            and st["deliverables_set_ok"] and st["nan_clean_ok"] and st["constants_ok"]
            and st["sliding_patch_bit_exact_ok"]
        )
    else:
        st["passes"] = bool(st["gate_logic_ok"] and st["constants_ok"])
    return st


# ========================================================================== #
# Report + wandb
# ========================================================================== #
def build_report(measured: dict[str, Any] | None) -> dict[str, Any]:
    analysis = analyze(measured) if measured else None
    self_test = build_self_test(measured, analysis)
    return {
        "deterministic_fusion_identity": True,
        "model_id": MODEL_ID,
        "strict_base_official_tps_196": STRICT_BASE_OFFICIAL_TPS_196,
        "target_tps": TARGET_TPS,
        "analysis": analysis,
        "self_test": self_test,
        "deterministic_fusion_self_test_passes": self_test["passes"],
        "measured": measured,
    }


def _ensure_real_wandb() -> None:
    """Force the installed wandb to win over the ./wandb run-output dir.

    This harness puts ROOT (the target repo root) on sys.path[0] so it can import
    ``scripts.*`` / ``research.*``. But wandb also writes its run data to
    ``ROOT/wandb``, which — lacking ``__init__.py`` — is importable as an empty
    *namespace package* and shadows the real wandb: ``import wandb`` then succeeds
    yet ``wandb.init`` is missing (AttributeError at run time). Drop that shadow and
    re-import the real package from site-packages, then put ROOT back (at the end,
    so ``scripts.*`` still resolves but no longer shadows installed packages).
    """
    import importlib
    import os
    cached = sys.modules.get("wandb")
    if cached is not None and getattr(cached, "__file__", None) and hasattr(cached, "init"):
        return
    for name in [n for n in list(sys.modules) if n == "wandb" or n.startswith("wandb.")]:
        del sys.modules[name]
    root_real = os.path.realpath(str(ROOT))
    saved_path = list(sys.path)
    # Strip every entry that resolves to ROOT (the explicit string, ""/cwd, ".")
    # so site-packages' real wandb wins; the import caches it in sys.modules, so
    # restoring the original path afterwards keeps the real package in force while
    # leaving scripts.*/research.* resolution exactly as the measurement saw it.
    sys.path[:] = [p for p in sys.path
                   if os.path.realpath(p if p else os.getcwd()) != root_real]
    importlib.invalidate_caches()
    try:
        importlib.import_module("wandb")
    finally:
        sys.path[:] = saved_path
        importlib.invalidate_caches()


_WANDB_TAGS = ["deterministic-fusion", "inductor-fusion-taxonomy", "greedy-identity",
               "m1-ar", "local-a10g", "pr-374"]
_WANDB_NOTES = "PR #374 inductor fusion-class greedy-identity taxonomy + deterministic pinning"


def _init_live_wandb(args: argparse.Namespace) -> Any:
    """Register the W&B run UP-FRONT — before the inductor compile — so the slot
    reads live immediately (advisor #374 ask) and per-class gate results can stream
    in as heartbeats. The final summary is logged on this same run after analysis.
    Device/torch version aren't known until the model loads, so they're filled into
    run.config later (log_wandb) and always appear in the summary.
    """
    try:
        _ensure_real_wandb()
        from scripts.wandb_logging import init_wandb_run
    except Exception as exc:  # pragma: no cover
        print(f"[wandb] unavailable: {exc}", flush=True)
        return None
    run = init_wandb_run(
        job_type="validity-profile",
        agent="kanna",
        name=args.wandb_name or "kanna/deterministic-fusion-identity",
        group=args.wandb_group or "deterministic-fusion-identity",
        tags=_WANDB_TAGS,
        notes=_WANDB_NOTES,
        config={
            "model_id": MODEL_ID,
            "n_prompts": args.n_prompts,
            "l_decode": args.l_decode,
            "warmup_steps": args.warmup_steps,
            "time_repeats": args.time_repeats,
            "budget_s": float(getattr(args, "budget_s", FULL_BUDGET_S)),
            "strict_base_official_tps_196": STRICT_BASE_OFFICIAL_TPS_196,
            "bw_ceiling_353": BW_CEILING_353,
            "batch_invariant_357": BATCH_INVARIANT_357,
            "target_tps": TARGET_TPS,
        },
    )
    if run is None:
        print("[wandb] init returned None — skipping", flush=True)
    else:
        print(f"[wandb] run LIVE before compile: id={getattr(run, 'id', None)} "
              f"group={args.wandb_group or 'deterministic-fusion-identity'}", flush=True)
    return run


def log_wandb(report: dict[str, Any], args: argparse.Namespace, run: Any = None) -> str | None:
    try:
        _ensure_real_wandb()
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_summary
    except Exception as exc:  # pragma: no cover
        print(f"[wandb] unavailable: {exc}", flush=True)
        return None
    measured = report.get("measured") or {}
    if run is None:
        run = init_wandb_run(
            job_type="validity-profile",
            agent="kanna",
            name=args.wandb_name or "kanna/deterministic-fusion-identity",
            group=args.wandb_group or "deterministic-fusion-identity",
            tags=_WANDB_TAGS,
            notes=_WANDB_NOTES,
            config={
                "model_id": MODEL_ID,
                "n_prompts": measured.get("n_prompts"),
                "l_decode": measured.get("l_decode"),
                "warmup_steps": measured.get("warmup_steps"),
                "torch_version": measured.get("torch_version"),
                "device_name": measured.get("device_name"),
                "strict_base_official_tps_196": STRICT_BASE_OFFICIAL_TPS_196,
                "bw_ceiling_353": BW_CEILING_353,
                "batch_invariant_357": BATCH_INVARIANT_357,
                "target_tps": TARGET_TPS,
            },
        )
    if run is None:
        print("[wandb] init returned None — skipping", flush=True)
        return None
    # Enrich the early-init run.config with values only known post model-load.
    try:
        run.config.update(
            {"torch_version": measured.get("torch_version"),
             "device_name": measured.get("device_name")},
            allow_val_change=True,
        )
    except Exception:  # pragma: no cover
        pass
    st = report["self_test"]
    a = report.get("analysis") or {}
    summary: dict[str, Any] = {
        "deterministic_fusion_self_test_passes": int(bool(st["passes"])),
        "self_test_gate_logic_ok": int(bool(st["gate_logic_ok"])),
        "self_test_sliding_patch_bit_exact_ok": int(bool(st.get("sliding_patch_bit_exact_ok", False))),
    }
    if "sliding_patch_bit_exact" in measured:
        summary["sliding_patch_bit_exact"] = int(bool(measured["sliding_patch_bit_exact"]))
    if a:
        summary.update({
            "eager_byte_exact_tps": a["eager_byte_exact_tps"],
            "full_fusion_tps": a["full_fusion_tps"],
            "full_fusion_token_identity_rate": a["full_fusion_token_identity_rate"],
            "fusion_speedup_x": a["fusion_speedup_x"],
            "identity_safe_fusion_max_tps": a["identity_safe_fusion_max_tps"],
            "identity_safe_fusion_is_real_fusion": int(bool(a["identity_safe_fusion_is_real_fusion"])),
            "any_real_fusion_identity_safe": int(bool(a["any_real_fusion_identity_safe"])),
            "deterministic_pinned_max_tps": a["deterministic_pinned_max_tps"],
            "deterministic_pin_holds_identity": int(bool(a["deterministic_pin_holds_identity"])),
            "frac_fusion_speedup_recovered_identity_safe": a["frac_fusion_speedup_recovered_identity_safe"],
            "frac_fusion_speedup_recovered_deterministic_pinned": a["frac_fusion_speedup_recovered_deterministic_pinned"],
            "best_strict_identity_tps_local": a["best_strict_identity_tps_local"],
            "n_fusions_break_identity": len(a["which_fusions_break_identity"]),
            "which_fusions_break_identity": ",".join(a["which_fusions_break_identity"]) or "none",
            "breaking_fusions_pinnable_deterministic": int(bool(a["breaking_fusions_pinnable_deterministic"])),
            "which_backends_preserve_identity": ",".join(a["which_backends_preserve_identity"]) or "none",
            "identity_pin_requires_aten_backend": int(bool(a["identity_pin_requires_aten_backend"])),
            "identity_safe_fusion_max_config": a["identity_safe_fusion_max_config"] or "none",
            "full_fusion_tps_source": a.get("full_fusion_tps_source") or "none",
            "fused_break_max_tps": a.get("fused_break_max_tps"),
            "fused_break_max_config": a.get("fused_break_max_config") or "none",
            "peak_vram_gb": measured.get("peak_vram_gb"),
        })
        meas_cfgs = measured.get("configs", {})
        for name, v in a["per_config"].items():
            summary[f"cfg.{name}.token_identity_rate"] = v["token_identity_rate"]
            summary[f"cfg.{name}.identity_preserved"] = int(bool(v["identity_preserved"]))
            summary[f"cfg.{name}.decode_tps_local"] = v["decode_tps_local"]
            summary[f"cfg.{name}.per_step_us"] = v["per_step_us"]
            summary[f"cfg.{name}.fusion_class"] = v.get("fusion_class") or "none"
            summary[f"cfg.{name}.backend"] = v.get("backend") or "inductor"
            summary[f"cfg.{name}.is_fusion"] = int(bool(v.get("is_fusion")))
            summary[f"cfg.{name}.expected"] = v.get("expected") or "none"
            summary[f"cfg.{name}.error"] = (v.get("error") or "none")
            cr = meas_cfgs.get(name, {}).get("compile_run_s")
            if cr is not None:
                summary[f"cfg.{name}.compile_run_s"] = cr
    summary = {k: v for k, v in summary.items() if v is not None}
    log_summary(run, summary, step=0)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    return rid


def _print_summary(report: dict[str, Any]) -> None:
    a = report.get("analysis")
    st = report["self_test"]
    line = "=" * 12 + " DETERMINISTIC FUSION IDENTITY (PR #374) " + "=" * 12
    print("\n" + line, flush=True)
    if a:
        print(f"  eager (byte-exact ref): {a['eager_per_step_us']:.0f} us/step  "
              f"{a['eager_byte_exact_tps']:.2f} local TPS", flush=True)
        print(f"  full_fusion: {a['full_fusion_tps']:.2f} TPS  "
              f"({a['fusion_speedup_x']:.2f}x over eager)  "
              f"token_identity_rate={a['full_fusion_token_identity_rate']:.3f}", flush=True)
        print("  per-config (identity_rate | TPS | backend | class):", flush=True)
        for name, v in a["per_config"].items():
            if v.get("error"):
                print(f"    {name:<26} FAILED: {v['error']}", flush=True)
                continue
            verdict = "SAFE " if v["identity_preserved"] else "BREAK"
            fus = "fuse" if v.get("is_fusion") else "ctrl"
            print(f"    {name:<26} {verdict} rate={v['token_identity_rate']:.3f} | "
                  f"{v['decode_tps_local']:>8.2f} TPS | {v.get('backend','inductor'):<10} | "
                  f"{fus} | {v['fusion_class']}", flush=True)
        print(f"  which_fusions_break_identity: {a['which_fusions_break_identity'] or 'NONE'}", flush=True)
        print(f"  any_real_fusion_identity_safe: {a['any_real_fusion_identity_safe']}  "
              f"(identity-safe winner is real fusion={a['identity_safe_fusion_is_real_fusion']})", flush=True)
        print(f"  which_backends_preserve_identity: {a['which_backends_preserve_identity'] or 'NONE'}  "
              f"(pin needs ATen backend={a['identity_pin_requires_aten_backend']})", flush=True)
        print(f"  identity_safe_fusion_max_tps: {a['identity_safe_fusion_max_tps']:.2f} TPS "
              f"(config={a['identity_safe_fusion_max_config']})", flush=True)
        print(f"  deterministic_pinned_max_tps: {a['deterministic_pinned_max_tps']:.2f} TPS  "
              f"pin_holds={a['deterministic_pin_holds_identity']} pinnable={a['breaking_fusions_pinnable_deterministic']}", flush=True)
        print(f"  frac_fusion_speedup_recovered_identity_safe: "
              f"{a['frac_fusion_speedup_recovered_identity_safe']:.3f}", flush=True)
        print(f"  best_strict_identity_tps_local: {a['best_strict_identity_tps_local']:.2f} TPS", flush=True)
    print(f"\n  SELF-TEST deterministic_fusion_self_test_passes = {st['passes']}", flush=True)
    for k in ("gate_logic_ok", "measured_present", "eager_ref_ok", "config_maps_ok",
              "headline_finite_ok", "deliverables_set_ok", "nan_clean_ok", "constants_ok"):
        print(f"    {k} = {st.get(k)}", flush=True)
    print("=" * len(line) + "\n", flush=True)


# ========================================================================== #
# Synthetic fixture (0-GPU self-test / reanalyze round-trip)
# ========================================================================== #
def _synthetic_measured() -> dict[str, Any]:
    """A tiny well-formed measured block (no GPU) so --self-test exercises the full
    analyze/self-test round-trip: eager ref + one breaking + one safe + one pinned-
    safe config. Numbers are illustrative, not measured."""
    ref = {"0": [1, 2, 3], "1": [4, 5, 6]}
    ref_sha = {k: _sha_tokens(v) for k, v in ref.items()}
    brk = {"0": [1, 2, 3], "1": [4, 9, 6]}
    safe = {k: list(v) for k, v in ref.items()}
    # Mirrors the expected real finding: every Triton FUSION config (incl. the inductor
    # deterministic pin) breaks identity on >=1 prompt, while the ATen-backend control
    # (dynamo_eager, no Triton) holds it. Exercises safe_any-non-empty + is_real_fusion
    # False + identity_pin_requires_aten_backend True. The eager-fallback null branch is
    # exercised separately by --reanalyze on a real all-break measured block.
    return {
        "model_id": MODEL_ID, "torch_version": "synthetic", "device_name": "synthetic",
        "n_prompts": 2, "l_decode": 3, "warmup_steps": 0, "time_repeats": 0,
        "max_cache_len": MAX_CACHE_LEN, "peak_vram_gb": 0.0,
        "eager": {"token_sha_by_idx": ref_sha, "per_step_us": 6000.0, "decode_tps_local": 166.67},
        "configs": {
            "dynamo_eager": {
                "fusion_class": "capture_aten_no_triton", "deterministic": False, "expected": "safe",
                "desc": "synthetic", "knobs": {}, "backend": "eager", "is_fusion": False,
                "token_sha_by_idx": {k: _sha_tokens(v) for k, v in safe.items()},
                "per_step_us": 4000.0, "decode_tps_local": 250.0, "error": None,
            },
            "full_fusion": {
                "fusion_class": "full", "deterministic": False, "expected": "break",
                "desc": "synthetic", "knobs": {}, "backend": "inductor", "is_fusion": True,
                "token_sha_by_idx": {k: _sha_tokens(v) for k, v in brk.items()},
                "per_step_us": 1000.0, "decode_tps_local": 1000.0, "error": None,
            },
            "pointwise_only": {
                "fusion_class": "pointwise_epilogue", "deterministic": False, "expected": "safe",
                "desc": "synthetic", "knobs": {}, "backend": "inductor", "is_fusion": True,
                "token_sha_by_idx": {k: _sha_tokens(v) for k, v in brk.items()},
                "per_step_us": 2000.0, "decode_tps_local": 500.0, "error": None,
            },
            "deterministic_pinned": {
                "fusion_class": "pinned", "deterministic": True, "expected": "safe",
                "desc": "synthetic", "knobs": {}, "backend": "inductor", "is_fusion": True,
                "token_sha_by_idx": {k: _sha_tokens(v) for k, v in brk.items()},
                "per_step_us": 2200.0, "decode_tps_local": 454.5, "error": None,
            },
        },
    }


# ========================================================================== #
# CLI
# ========================================================================== #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", action="store_true",
                    help="load the model + run the fusion-class identity/TPS matrix on the GPU")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny GPU plumbing run (few prompts, short decode, no wandb)")
    ap.add_argument("--reanalyze", type=Path, default=None,
                    help="0-GPU: re-derive analysis+self-test from a saved results JSON's measured block")
    ap.add_argument("--self-test", action="store_true",
                    help="exit non-zero unless deterministic_fusion_self_test_passes "
                         "(0-GPU synthetic round-trip if no measurement is provided)")
    ap.add_argument("--only", default=None, help="comma-separated config names to run")
    ap.add_argument("--n-prompts", type=int, default=FULL_N_PROMPTS)
    ap.add_argument("--l-decode", type=int, default=FULL_L_DECODE)
    ap.add_argument("--warmup-steps", type=int, default=FULL_WARMUP_STEPS)
    ap.add_argument("--time-repeats", type=int, default=FULL_TIME_REPEATS)
    ap.add_argument("--budget-s", type=float, default=FULL_BUDGET_S,
                    help="wall-clock budget for the config loop; remaining configs are "
                         "skipped (recorded as such) once exceeded so analyze+wandb still run")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke:
        args.measure = True
        args.no_wandb = True
        args.n_prompts = min(args.n_prompts, SMOKE_N_PROMPTS)
        args.l_decode = SMOKE_L_DECODE
        args.warmup_steps = SMOKE_WARMUP_STEPS
        args.time_repeats = SMOKE_TIME_REPEATS

    measured = None
    saved = None
    live_run = None
    if args.reanalyze:
        saved = json.loads(Path(args.reanalyze).read_text())
        measured = saved.get("measured")
        if not measured:
            print(f"[reanalyze] no 'measured' block in {args.reanalyze}", flush=True)
            return 1
        args.no_wandb = True
        print(f"[reanalyze] re-deriving analysis from {args.reanalyze} (0-GPU)", flush=True)
    elif args.measure:
        # Register the W&B run BEFORE the heavy inductor compile so the slot reads
        # live immediately and per-class gate results stream as heartbeats.
        if not args.no_wandb:
            live_run = _init_live_wandb(args)
        measured = run_measurement(args, wandb_run=live_run)
    elif args.self_test:
        # 0-GPU integrity: synthetic round-trip through analyze + self-test.
        measured = _synthetic_measured()
        print("[self-test] 0-GPU synthetic round-trip (no --measure given)", flush=True)

    report = build_report(measured)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if measured and args.measure and not args.no_wandb:
        report["wandb_run_id"] = log_wandb(report, args, run=live_run)
    elif args.reanalyze:
        report["wandb_run_id"] = saved.get("wandb_run_id")
        report["reanalyzed_from"] = {"path": str(args.reanalyze), "source_created_at": saved.get("created_at")}
    report["created_at"] = stamp
    # Only overwrite the canonical results file for a real or reanalyzed measurement;
    # a bare --self-test synthetic round-trip must not clobber a real result.
    if args.measure or args.reanalyze:
        RESULTS_JSON.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"[report] {RESULTS_JSON}", flush=True)
    _print_summary(report)

    if args.self_test:
        return 0 if report["deterministic_fusion_self_test_passes"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
