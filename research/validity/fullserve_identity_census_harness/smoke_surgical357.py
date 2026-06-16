#!/usr/bin/env python3
"""
Smoke-test: load surgical-357 stack as vllm.LLM(), run generate(), report key facts.
Run as: timeout 600 /tmp/senpai-venvs/5f4c623f772358a2/bin/python /workspace/senpai/target/smoke_surgical357.py
"""

from __future__ import annotations

import gc
import os
import sys
import time

# ============================================================
# 1. ENV VARS — set BEFORE any vLLM import
# ============================================================
SUBDIR = "/workspace/senpai/target/submissions/fa2sw_strict_surgical357"

# Clear sha256 check so ensure_drafter doesn't block on mismatch
os.environ.pop("DRAFTER_SHA256", None)

os.environ.update({
    # GPU device — override any stale CUDA_VISIBLE_DEVICES (physical GPU is 0)
    "CUDA_VISIBLE_DEVICES": "0",
    # Core model (already-pruned 12k lm_head)
    "LOCAL_MODEL_DIR": "/tmp/osoi5-12k-baked",
    "PCK04_KEEPSET": "/tmp/osoi5-12k-baked/pck04_keepset.json",
    # PLE patches
    "PLE_ASSUME_VALID_TOKEN_IDS": "1",
    "PLE_FOLD_EMBED_SCALE": "1",
    "PLE_FOLD_TARGET_MODEL": "/tmp/osoi5-v0-baked",
    "PLE_SCRATCH_REUSE": "1",
    # Surgical attention pin (2D order-preserving, no matmul tax)
    "SURGICAL_ATTN_USE_3D_OFF": "1",
    # Slim greedy sampler (already patched in venv, default="1")
    "DIXIE_SLIM_GREEDY": "1",
    "DIXIE_PREWARM_GREEDY_KERNEL": "1",
    # ONEGRAPH / CUDA graph
    "ONEGRAPH": "1",
    # lm_head prune (12k already baked, just needs env)
    "LM_HEAD_PRUNE": "1",
    "LM_HEAD_PRUNE_REQUIRE": "1",
    "LM_HEAD_PRUNE_DST": "/tmp/osoi5-12k-baked",
    # Fused sparse argmax (drafter top-token)
    "FUSED_SPARSE_ARGMAX": "1",
    "FUSED_SPARSE_ARGMAX_REQUIRE": "1",
    "FUSED_SPARSE_ARGMAX_BLOCK": "16",
    # LOOPGRAPH — RELAXED for smoke (no full warmup)
    "LOOPGRAPH_REQUIRE_CAPTURE": "0",
    "LOOPGRAPH_WARMUP_CALLS": "5",
    "LOOPGRAPH_PINGPONG_SLOTS": "3",
    # Fused accept — RELAXED require for smoke
    "DIXIE_FUSED_ACCEPT_PREP": "1",
    "DIXIE_FUSED_ACCEPT_PREP_REQUIRE": "0",
    # Attention plugins
    "FA_SLIDING": "1",
    "FA_SLIDING_DIAG": "0",
    "SPLITKV_VERIFY": "1",
    "SPLITKV_VERIFY_MAX_Q": "64",
    # CRITICAL: no harness dataset in library mode
    "PRECACHE_BENCH": "0",
    "PRECACHE_REQUIRE": "0",
    # Misc
    "DETOK_ENDONLY": "1",
    "FASTRENDER": "1",
    "FEOPT_ORJSON": "1",
    "GENERATION_CONFIG": "vllm",
    "OVERRIDE_GENERATION_CONFIG": '{"temperature":0.0,"top_p":1.0,"top_k":0}',
    "CENTROID_TOP_K": "64",
    "DISABLE_LOG_STATS": "1",
    # Force native (PyTorch) sampler — FlashInfer JIT build requires curand.h
    # which is absent on this system. forward_native works fine for smoke.
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    # DO NOT set VLLM_BATCH_INVARIANT (installs matmul tax)
    # DO NOT set DRAFTER_SHA256 (skip sha256 check)
})

# ============================================================
# 2. ARM SITECUSTOMIZE via sys.path (before any vLLM import)
#    sitecustomize.py auto-runs on first Python import,
#    installs pck04 meta-path finder, surgical_attn_patch, etc.
# ============================================================
sys.path.insert(0, SUBDIR)
print(f"[smoke] SUBDIR on sys.path[0]: {SUBDIR}")

# Directly install all patch meta-path finders in the parent process.
# With fork (vLLM default), the EngineCore child inherits sys.meta_path,
# so all finders (PCK04 lm_head rebuild, loopgraph, fused argmax, etc.)
# will be present when vLLM imports the patched modules in the child.
import sitecustomize  # noqa: F401  — runs all meta_path registrations in parent
print("[smoke] sitecustomize imported: all meta_path finders installed in parent")

# Also inject SUBDIR into PYTHONPATH so the vLLM V1 EngineCore subprocess
# (SyncMPClient) inherits it and auto-runs sitecustomize.py, which installs
# the PCK04 meta-path hook before gemma4.py is imported in the subprocess.
# Without this, the subprocess never gets the hook and the lm_head
# org_vocab_size assertion (262144 vs 12288) fires.
existing_pypath = os.environ.get("PYTHONPATH", "")
pypath_parts = [p for p in existing_pypath.split(os.pathsep) if p]
if SUBDIR not in pypath_parts:
    os.environ["PYTHONPATH"] = os.pathsep.join([SUBDIR] + pypath_parts)
print(f"[smoke] PYTHONPATH prefix set to: {SUBDIR}")

# ============================================================
# 3. PATCH gemma4.py SOURCE (not yet patched in venv)
# ============================================================
# Import serve from submission dir (now on sys.path)
import serve as serve_mod  # noqa: E402

print("[smoke] Calling patch_ple_sources() to patch gemma4.py ...")
serve_mod.patch_ple_sources()
print("[smoke] patch_ple_sources() done")

# ============================================================
# 4. LOAD MODEL — MTP speculative_config first attempt
# ============================================================
t0 = time.time()
from vllm import LLM, SamplingParams  # noqa: E402

SPEC_CONFIG = {
    "method": "mtp",
    "model": "/tmp/qat-assistant",
    "num_speculative_tokens": 7,
}

print("[smoke] Creating LLM with MTP speculative_config ...")
try:
    llm = LLM(
        model="/tmp/osoi5-12k-baked",
        quantization="compressed-tensors",
        dtype="bfloat16",
        max_model_len=1024,
        gpu_memory_utilization=0.85,
        max_num_seqs=1,
        enforce_eager=True,
        trust_remote_code=True,
        speculative_config=SPEC_CONFIG,
    )
    spec_on = True
    print("[smoke] MTP speculative_config: LOADED OK")
except Exception as e:
    print(f"[smoke] MTP speculative_config FAILED: {e!r}")
    print("[smoke] Retrying without speculative_config ...")
    spec_on = False
    try:
        llm = LLM(
            model="/tmp/osoi5-12k-baked",
            quantization="compressed-tensors",
            dtype="bfloat16",
            max_model_len=1024,
            gpu_memory_utilization=0.85,
            max_num_seqs=1,
            enforce_eager=True,
            trust_remote_code=True,
        )
        print("[smoke] Plain M=1 (no spec): LOADED OK")
    except Exception as e2:
        print(f"[smoke] FATAL: plain load also failed: {e2!r}")
        raise

load_time = time.time() - t0
print(f"[smoke] load time: {load_time:.1f}s  |  spec_on={spec_on}")

# ============================================================
# 5. GENERATE with logprobs=5
# ============================================================
# Small real prompt: "What is the capital of France?"
# Token IDs below are approximate BOS + text tokens for Gemma tokenizer
# We use a simple list; the exact IDs don't matter for a smoke test
PROMPT_IDS = [2, 108, 1645, 108, 3843, 603, 573, 3284, 576, 2412, 235336]

print(f"[smoke] generate() with logprobs=5, spec_on={spec_on} ...")
sp = SamplingParams(temperature=0.0, max_tokens=8, logprobs=5)
try:
    outputs = llm.generate([{"prompt_token_ids": PROMPT_IDS}], sp)
    out = outputs[0]
    token_ids_out = list(out.outputs[0].token_ids)
    logprobs_obj = out.outputs[0].logprobs
    lp_populated = (
        logprobs_obj is not None
        and len(logprobs_obj) > 0
        and logprobs_obj[0] is not None
    )
    print(f"[smoke] output token_ids: {token_ids_out}")
    print(f"[smoke] spec-on per-token logprobs populated: {lp_populated}")
    if lp_populated:
        # Show first token's logprobs keys
        first = logprobs_obj[0]
        print(f"[smoke] first token logprobs type={type(first).__name__}, len={len(first) if hasattr(first,'__len__') else 'N/A'}")
except Exception as e:
    print(f"[smoke] generate() FAILED: {e!r}")

# ============================================================
# 6. PROMPT LOGPROBS (teacher-forced per-position reads)
# ============================================================
print("[smoke] prompt_logprobs=5 test ...")
# Longer sequence: repeat pattern so we get multiple positions
SEQ_LONG = (PROMPT_IDS + [1]) * 4
sp_pl = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=5)
try:
    outputs_pl = llm.generate([{"prompt_token_ids": SEQ_LONG}], sp_pl)
    out_pl = outputs_pl[0]
    pl = out_pl.prompt_logprobs
    pl_populated = pl is not None and len(pl) > 1 and pl[1] is not None
    print(f"[smoke] prompt_logprobs populated: {pl_populated}")
    if pl_populated:
        # Count non-None positions
        non_none = sum(1 for x in pl if x is not None)
        print(f"[smoke] prompt_logprobs: {non_none}/{len(pl)} positions have data")
except Exception as e:
    print(f"[smoke] prompt_logprobs FAILED: {e!r}")

# ============================================================
# 7. PEAK GPU MEMORY
# ============================================================
import torch  # noqa: E402

peak_mb = torch.cuda.max_memory_allocated() / 1e6
print(f"[smoke] peak GPU memory: {peak_mb:.0f} MB")

# ============================================================
# 8. CLEANUP
# ============================================================
del llm
gc.collect()
torch.cuda.empty_cache()
print("[smoke] GPU freed")
print("[smoke] DONE")
