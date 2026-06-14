"""PCK-04 logits-scatter patch for vLLM.

When PCK04_KEEPSET is set to the path of a pck04_keepset.json (or keepset.json)
produced by prune_lm_head.py / build_keepset.py, this module monkey-patches
Gemma4ForCausalLM.compute_logits so that:

  1. The pruned lm_head produces [M, K] logits (K ≤ 262144).
  2. They are scattered into a full-vocab [M, 262144] buffer pre-filled with -inf
     at non-kept positions.
  3. Downstream sampler / prompt_logprobs sees full-vocab logits with original
     token IDs; non-kept tokens get probability exactly 0 (exp(-inf)).

Hook: vllm.model_executor.models.gemma4.Gemma4ForCausalLM.compute_logits
  File: vllm/model_executor/models/gemma4.py, line ~1681
  Signature: compute_logits(self, hidden_states: Tensor) -> Tensor | None

This method is called eagerly from GPUModelRunner.execute_model (not inside a
CUDA graph capture in this stack).  The onegraph sitecustomize uses
cudagraph_runtime_mode=CUDAGraphMode.NONE for the drafter loop; the main model
runner runs execute_model eagerly.  No CUDA-graph compatibility concerns here.

Patching style matches onegraph-spec7-v0/sitecustomize.py:
  - _TargetFinder / _PatchingLoader for source-patch on module load.
  - Env-var gating (PCK04_KEEPSET).
  - Fail-loud fingerprint asserts.
  - No GPU calls at import time.

Usage: include this file in PYTHONPATH (alongside or as part of sitecustomize).
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PCK04_KEEPSET_PATH = os.environ.get("PCK04_KEEPSET", "")
_TARGET_MODULE = "vllm.model_executor.models.gemma4"
_TARGET_CLASS = "Gemma4ForCausalLM"
_TARGET_METHOD = "compute_logits"

# ---------------------------------------------------------------------------
# Module-level state (allocated lazily on first device call, never at import)
# ---------------------------------------------------------------------------
_pck04_state: dict[str, Any] = {
    "keep_ids": None,       # list[int], loaded from JSON
    "full_vocab": None,     # int
    "K": None,              # int — pruned head row count
    # Per-device buffers allocated on first use:
    "device_cache": {},     # device str → {"template": Tensor, "keep_idx": Tensor}
}


def _load_keepset(path: str) -> tuple[list[int], int]:
    """Load keep_ids and full_vocab from pck04_keepset.json or keepset.json."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"[pck04] PCK04_KEEPSET={path!r} does not exist — cannot patch logits"
        )
    data = json.loads(p.read_text())
    keep_ids: list[int] = data["keep_ids"]
    # pck04_keepset.json from prune_lm_head uses "full_vocab"; keepset.json uses "vocab_size"
    full_vocab: int = int(data.get("full_vocab") or data.get("vocab_size") or 0)
    if full_vocab == 0:
        raise ValueError(
            f"[pck04] keepset JSON at {path!r} has no 'full_vocab' or 'vocab_size' key"
        )
    return keep_ids, full_vocab


def _get_device_buffers(device: Any, keep_ids: list[int], full_vocab: int) -> dict[str, Any]:
    """Allocate (once per device) the -inf template and keep_idx tensors."""
    import torch  # type: ignore

    device_str = str(device)
    cache = _pck04_state["device_cache"]
    if device_str in cache:
        return cache[device_str]

    K = len(keep_ids)
    keep_idx = torch.tensor(keep_ids, dtype=torch.long, device=device)

    # Template: a [1, full_vocab] float32 buffer filled with -inf.
    # We clone this each step to get a fresh [M, full_vocab] buffer.
    # Using float32 to avoid fp16 -inf precision issues; cast to match logits dtype later.
    template = torch.full(
        (1, full_vocab),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )

    buffers = {"template": template, "keep_idx": keep_idx, "K": K, "full_vocab": full_vocab}
    cache[device_str] = buffers
    print(
        f"[pck04] allocated scatter buffers on {device_str}: "
        f"template=[1, {full_vocab}] full_vocab, keep_idx=[{K}] (pid {os.getpid()})",
        file=sys.stderr,
        flush=True,
    )
    return buffers


def _scatter_to_full_vocab(
    pruned_logits: Any,
    keep_ids: list[int],
    full_vocab: int,
) -> Any:
    """Scatter [M, K] pruned logits into [M, full_vocab] with -inf padding.

    Strategy: persistent per-(device, dtype, M) buffer initialized to -inf
    ONCE.  Non-kept columns are never written, so they stay -inf forever;
    kept columns are fully overwritten every step by index_copy_.  Zero
    per-step allocation, zero per-step fill — only the M*K column copy.
    """
    import torch  # type: ignore

    device = pruned_logits.device
    bufs = _get_device_buffers(device, keep_ids, full_vocab)
    keep_idx = bufs["keep_idx"]
    K = bufs["K"]
    M = pruned_logits.shape[0]

    # Fingerprint: pruned head must have exactly K columns
    assert pruned_logits.shape[-1] == K, (
        f"[pck04] FINGERPRINT FAIL: expected pruned logits shape [M, {K}], "
        f"got {list(pruned_logits.shape)}.  "
        f"Check that the model was pruned with the same keepset."
    )

    # Cache ONLY decode-sized buffers (M ≤ 16, constant K_spec+1 per step) —
    # caching prefill shapes (M up to max_num_batched_tokens) retains a
    # ~0.5 GB buffer per distinct M and OOMs the prompt_logprobs stage.
    # Prefill is per-request, not per-step: transient allocation is fine.
    if M <= 16:
        out_cache = bufs.setdefault("out_cache", {})
        cache_key = (M, pruned_logits.dtype)
        out = out_cache.get(cache_key)
        if out is None:
            out = torch.full(
                (M, full_vocab),
                float("-inf"),
                dtype=pruned_logits.dtype,
                device=device,
            )
            out_cache[cache_key] = out
    else:
        out = torch.full(
            (M, full_vocab),
            float("-inf"),
            dtype=pruned_logits.dtype,
            device=device,
        )

    # out[:, keep_idx[j]] = pruned_logits[:, j] — kept columns fully
    # overwritten each call; -inf complement untouched since allocation.
    out.index_copy_(1, keep_idx, pruned_logits)

    return out


def _apply_pck04_patch(module: Any) -> None:
    """Monkey-patch Gemma4ForCausalLM.__init__ and .compute_logits on module load.

    __init__ patch: after original __init__ returns, rebuild self.lm_head with
    num_embeddings=K (the pruned row count) so the weight-loader assert
    (loaded_weight.shape[output_dim] == self.org_vocab_size) matches the
    32768-row compressed-tensors checkpoint.  ParallelLMHead is called with
    org_num_embeddings=K so that org_vocab_size is set to K (not padded size),
    matching the pack-quantized weight loader branch at line 465 of
    vocab_parallel_embedding.py which checks org_vocab_size // packed_factor.

    compute_logits patch: scatter the [M, K] pruned logits into a [M, full_vocab]
    buffer with -inf at non-kept positions so downstream samplers see the full
    vocabulary with original token IDs intact.

    LogitsProcessor constructed with vocab_size=full_vocab does:
        logits = logits[..., :self.org_vocab_size]   # line 103 logits_processor.py
    After scatter, logits are [M, full_vocab], so the slice is a no-op and safe.

    tie_word_embeddings: must be False in config (embed_tokens stays full size);
    assert below ensures we never accidentally re-tie a K-row head to it.
    """
    import torch  # type: ignore

    # --- load keepset (fail loud if env set but file missing) ---
    if not PCK04_KEEPSET_PATH:
        # PCK04_KEEPSET not set — install no-op pass-through so the module still
        # loads cleanly, but emit a warning so the operator knows this patch is
        # inactive.
        print(
            "[pck04] PCK04_KEEPSET not set — pck04 logits scatter is INACTIVE",
            file=sys.stderr,
            flush=True,
        )
        return

    keep_ids, full_vocab = _load_keepset(PCK04_KEEPSET_PATH)
    K = len(keep_ids)
    _pck04_state["keep_ids"] = keep_ids
    _pck04_state["full_vocab"] = full_vocab
    _pck04_state["K"] = K

    # --- fingerprint: class and methods must exist ---
    cls = getattr(module, _TARGET_CLASS, None)
    assert cls is not None, (
        f"[pck04] FINGERPRINT FAIL: {_TARGET_CLASS} not found in {module.__name__}"
    )
    original_init = getattr(cls, "__init__", None)
    assert original_init is not None, (
        f"[pck04] FINGERPRINT FAIL: {_TARGET_CLASS}.__init__ not found"
    )
    original_compute_logits = getattr(cls, _TARGET_METHOD, None)
    assert original_compute_logits is not None, (
        f"[pck04] FINGERPRINT FAIL: {_TARGET_CLASS}.{_TARGET_METHOD} not found"
    )

    # --- verify ParallelLMHead construction signature in the gemma4 source ---
    # Expected site (gemma4.py ~line 1621):
    #   self.lm_head = ParallelLMHead(
    #       config.vocab_size,
    #       config.hidden_size,
    #       quant_config=quant_config,
    #       prefix=maybe_prefix(prefix, "lm_head"),
    #   )
    # We replicate this exactly, substituting K for config.vocab_size and
    # passing org_num_embeddings=K so that org_vocab_size is also K (not the
    # padded num_embeddings_padded), which is what weight_loader compares
    # against the loaded checkpoint row-count.
    try:
        import inspect
        gemma4_src = inspect.getsource(module)
        assert "ParallelLMHead(" in gemma4_src, (
            "[pck04] FINGERPRINT FAIL: ParallelLMHead( not found in gemma4 source — "
            "construction site may have changed"
        )
    except (OSError, TypeError):
        # Source unavailable (e.g. compiled), skip source fingerprint.
        pass

    # Grab ParallelLMHead from the module (it's imported at module level).
    ParallelLMHead = getattr(module, "ParallelLMHead", None)
    if ParallelLMHead is None:
        # Fall back to direct import.
        from vllm.model_executor.layers.vocab_parallel_embedding import (  # type: ignore
            ParallelLMHead,
        )
    assert ParallelLMHead is not None, (
        "[pck04] FINGERPRINT FAIL: could not resolve ParallelLMHead"
    )

    maybe_prefix_fn = getattr(module, "maybe_prefix", None)
    if maybe_prefix_fn is None:
        from vllm.model_executor.models.utils import maybe_prefix as maybe_prefix_fn  # type: ignore

    import functools

    # functools.wraps is LOAD-BEARING here: vLLM's initialize_model inspects
    # the __init__ signature (inspect.signature follows __wrapped__) to decide
    # whether to pass vllm_config; a bare *args/**kwargs wrapper makes it fall
    # back to the legacy calling convention and vllm_config never arrives.
    @functools.wraps(original_init)
    def __init__pck04(self_model: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self_model, *args, **kwargs)

        # Safety: never rebuild if tie_word_embeddings is True — the head would
        # have been replaced by embed_tokens and rebuilding would break tying.
        config = getattr(self_model, "config", None)
        assert config is not None, (
            "[pck04] FINGERPRINT FAIL: Gemma4ForCausalLM instance has no .config after __init__"
        )
        assert not getattr(config, "tie_word_embeddings", False), (
            "[pck04] FINGERPRINT FAIL: config.tie_word_embeddings=True — "
            "cannot safely rebuild lm_head with K rows while embed_tokens has full vocab"
        )

        quant_config = getattr(self_model, "quant_config", None)

        # Determine prefix: inspect the existing lm_head's prefix attribute if
        # available (set by vllm on the quant layer), otherwise infer from the
        # model's own prefix kwarg or default to "lm_head".
        existing_prefix = getattr(getattr(self_model, "lm_head", None), "prefix", None)
        if existing_prefix is None:
            # Reconstruct the same prefix that the original __init__ computed:
            #   maybe_prefix(prefix, "lm_head")  where prefix came from kwargs.
            _outer_prefix = kwargs.get("prefix", "")
            existing_prefix = maybe_prefix_fn(_outer_prefix, "lm_head")

        # Rebuild lm_head with K rows.
        # Original call (gemma4.py lines 1621-1626):
        #   self.lm_head = ParallelLMHead(
        #       config.vocab_size,          → K (pruned row count)
        #       config.hidden_size,
        #       quant_config=quant_config,
        #       prefix=maybe_prefix(prefix, "lm_head"),
        #   )
        # We also pass org_num_embeddings=K so VocabParallelEmbedding sets
        # self.org_vocab_size = K, satisfying the weight_loader assert:
        #   loaded_weight.shape[output_dim] == self.org_vocab_size
        # (and for pack-quantized: loaded_weight.shape == org_vocab_size // pack_factor)
        self_model.lm_head = ParallelLMHead(
            K,                          # num_embeddings — pruned row count
            config.hidden_size,         # embedding_dim
            quant_config=quant_config,
            prefix=existing_prefix,
            org_num_embeddings=K,       # sets org_vocab_size=K in weight_loader assert
        )
        print(
            f"[pck04] rebuilt lm_head: ParallelLMHead(num_embeddings={K}, "
            f"embedding_dim={config.hidden_size}, org_num_embeddings={K}, "
            f"prefix={existing_prefix!r}) — replaced full-vocab head "
            f"(was {getattr(config, 'vocab_size', '?')} rows) "
            f"in pid {os.getpid()}",
            file=sys.stderr,
            flush=True,
        )

    cls.__init__ = __init__pck04

    # Note on CUDA graphs: this stack runs execute_model eagerly. The
    # onegraph sitecustomize uses CUDAGraphMode.NONE for the drafter and the
    # main model runner's execute_model path is also eager (not captured).
    # Verified: gpu_model_runner.py calls model.compute_logits() outside any
    # torch.cuda.graph() capture block.  No CUDA-graph compat concerns here.

    def compute_logits_pck04(self_model: Any, hidden_states: torch.Tensor) -> Any:
        # Call original — lm_head now has K rows → returns [M, K] logits.
        pruned_logits = original_compute_logits(self_model, hidden_states)
        if pruned_logits is None:
            # TP rank > 0 returns None (gather not yet complete); pass through.
            return None
        # Scatter into [M, full_vocab] with -inf at non-kept positions.
        return _scatter_to_full_vocab(pruned_logits, keep_ids, full_vocab)

    cls.compute_logits = compute_logits_pck04
    print(
        f"[pck04] patched {_TARGET_CLASS}.__init__ + {_TARGET_METHOD} in pid {os.getpid()} "
        f"(K={K}, full_vocab={full_vocab}, keepset={PCK04_KEEPSET_PATH!r})",
        file=sys.stderr,
        flush=True,
    )


# ---------------------------------------------------------------------------
# _TargetFinder / _PatchingLoader — same pattern as onegraph sitecustomize.py
# ---------------------------------------------------------------------------

class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, inner: importlib.abc.Loader, patch_fn: Any) -> None:
        self._inner = inner
        self._patch_fn = patch_fn

    def create_module(self, spec: Any) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._inner.exec_module(module)
        self._patch_fn(module)


class _TargetFinder(importlib.abc.MetaPathFinder):
    def __init__(self, target: str, patch_fn: Any) -> None:
        self._target = target
        self._patch_fn = patch_fn
        self._busy = False

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != self._target or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader, self._patch_fn)
        return spec


# Register the finder immediately on import.
sys.meta_path.insert(0, _TargetFinder(_TARGET_MODULE, _apply_pck04_patch))
