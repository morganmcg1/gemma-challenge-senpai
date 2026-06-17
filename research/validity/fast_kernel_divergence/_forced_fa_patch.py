"""POSITIVE CONTROL for PR #540 -- force the surgical-2D-attn swap the served fast
stack was *designed* to do, bypassing the guard that makes it inert on the native head.

The shipped submission patch (submissions/fa2sw_strict_m1ar_int4/fa_sliding_patch.py)
swaps eligible sliding-window (head_size=256) target layers to FlashAttention, BUT only
when `model_config.hf_config.model_type == "gemma4"`. The native int4 checkpoint
(google/gemma-4-E4B-it-qat-w4a16-ct) exposes the *text* config whose model_type is
"gemma4_text", so the guard never matches and the swap is silently inert (fa flips=0) --
this is the central as-run finding of #540.

This module is the counterfactual: identical eligibility EXCEPT the model_type guard is
relaxed to `mt.startswith("gemma4")`, so it matches gemma4_text too and the swap engages.
It exists ONLY to (a) prove the divergence probe can SEE a real kernel tax (harness
sensitivity, the analog of #529's blind-spot contrast) and (b) quantify the latent
"as-designed" tax -- what the surgical-attn swap WOULD cost on the native head if the
guard matched. It is NEVER claimed as the as-run behavior; that is fa_sliding_patch (inert).

Gated on FORCE_FA_SLIDING=1. Fail-open: any error in the decision path keeps baseline.
Eligibility mirrors fa_sliding_patch v1 exactly (excludes draft + KV-share sources 19,20).
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import re
import sys
from typing import Any

FORCE_FA_SLIDING = os.environ.get("FORCE_FA_SLIDING", "0") == "1"
EXCLUDE_IDX = {
    int(x)
    for x in os.environ.get("FA_SLIDING_EXCLUDE", "19,20").split(",")
    if x.strip()
}
DIAG_LIMIT = int(os.environ.get("FA_SLIDING_DIAG", "90"))

ATTENTION_TARGET = "vllm.model_executor.layers.attention.attention"

_stats = {"fa": 0, "diag": 0, "eligible_mt_mismatch": 0}
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _patch_attention(module: Any) -> None:
    attn_cls = module.Attention
    orig_init = attn_cls.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        try:
            head_size = args[1] if len(args) > 1 else kwargs.get("head_size")
            if head_size == 256:
                prefix = kwargs.get("prefix", "") or ""
                sw = kwargs.get("per_layer_sliding_window")
                kvshare = kwargs.get("kv_sharing_target_layer_name")
                backend = kwargs.get("attn_backend")
                m = _LAYER_RE.search(prefix)
                idx = int(m.group(1)) if m else -1
                try:
                    from vllm.config import get_current_vllm_config

                    mt = getattr(
                        get_current_vllm_config().model_config.hf_config,
                        "model_type",
                        "<none>",
                    )
                except Exception as cfg_exc:  # noqa: BLE001
                    mt = f"<err:{cfg_exc!r}>"
                if _stats["diag"] < DIAG_LIMIT:
                    _stats["diag"] += 1
                    print(
                        f"[forced-fa-diag] prefix={prefix!r} idx={idx} sw={sw} "
                        f"kvshare={kvshare!r} backend={backend} mt={mt}",
                        flush=True,
                    )
                # the SHIPPED patch's exact eligibility, with the model_type guard RELAXED
                # from `== gemma4` to `startswith(gemma4)` so it matches gemma4_text.
                base_eligible = (
                    sw is not None
                    and kvshare is None
                    and backend is None
                    and "draft" not in prefix
                    and idx >= 0
                    and idx not in EXCLUDE_IDX
                )
                if base_eligible and not str(mt).startswith("gemma4"):
                    _stats["eligible_mt_mismatch"] += 1
                if base_eligible and str(mt).startswith("gemma4"):
                    from vllm.v1.attention.backends.flash_attn import (
                        FlashAttentionBackend,
                    )

                    kwargs["attn_backend"] = FlashAttentionBackend
                    _stats["fa"] += 1
                    print(
                        f"[forced-fa] -> FLASH_ATTN for {prefix} "
                        f"(n={_stats['fa']})",
                        flush=True,
                    )
        except Exception as exc:  # noqa: BLE001 - fail-open by design
            print(f"[forced-fa] decision error, baseline kept: {exc!r}", flush=True)
        orig_init(self, *args, **kwargs)

    attn_cls.__init__ = patched_init
    print("[forced-fa] Attention.__init__ wrapper active (positive control)", flush=True)


class _ChainLoader(importlib.abc.Loader):
    def __init__(self, inner: importlib.abc.Loader, patch_fn: Any) -> None:
        self._inner = inner
        self._patch_fn = patch_fn

    def create_module(self, spec: Any) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._inner.exec_module(module)
        self._patch_fn(module)


class _ChainFinder(importlib.abc.MetaPathFinder):
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
        spec.loader = _ChainLoader(spec.loader, self._patch_fn)
        return spec


if FORCE_FA_SLIDING:
    sys.meta_path.insert(0, _ChainFinder(ATTENTION_TARGET, _patch_attention))
    print("[forced-fa] finder registered (positive control)", flush=True)
