"""agent-smith: FA2 for eligible sliding-window target layers (FA_SLIDING=1).

v1 (after fa2sw-v0 negative): v0 accidentally flipped the DRAFTER's layers
(drafter KV-sharing is wired after init, so the kv_sharing kwarg is None at
construction) and flipped zero target layers. v1:

  - excludes any prefix containing "draft";
  - excludes the KV-share SOURCE layers (default 19,20): the 16-layer shared
    tail reads their KV, and a backend/layout mismatch there costs more than
    FA2 saves (measured on the drafter in v0: +0.9ms draft, +0.9ms gap);
  - logs every head-256 Attention.__init__ clause vector ([fa-diag]) so a
    non-flip is diagnosable from one run's logs.

Eligible: target sliding layers (head 256) below the KV-shared tail, minus
share sources. Expected flips with the osoi5-baked config (37 layers,
num_kv_shared_layers=16, share sources 19/20): ~16 layers.

Fail-open: errors in the decision path keep baseline behavior.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import re
import sys
from typing import Any

FA_SLIDING = os.environ.get("FA_SLIDING", "0") == "1"
EXCLUDE_IDX = {
    int(x)
    for x in os.environ.get("FA_SLIDING_EXCLUDE", "19,20").split(",")
    if x.strip()
}
DIAG_LIMIT = int(os.environ.get("FA_SLIDING_DIAG", "90"))

ATTENTION_TARGET = "vllm.model_executor.layers.attention.attention"

_stats = {"fa": 0, "diag": 0}
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
                        f"[fa-diag] prefix={prefix!r} idx={idx} sw={sw} "
                        f"kvshare={kvshare!r} backend={backend} mt={mt}",
                        flush=True,
                    )
                if (
                    sw is not None
                    and kvshare is None
                    and backend is None
                    and mt == "gemma4"
                    and "draft" not in prefix
                    and idx >= 0
                    and idx not in EXCLUDE_IDX
                ):
                    from vllm.v1.attention.backends.flash_attn import (
                        FlashAttentionBackend,
                    )

                    kwargs["attn_backend"] = FlashAttentionBackend
                    _stats["fa"] += 1
                    print(
                        f"[fa-sliding] -> FLASH_ATTN for {prefix} "
                        f"(n={_stats['fa']})",
                        flush=True,
                    )
        except Exception as exc:  # noqa: BLE001 - fail-open by design
            print(f"[fa-sliding] decision error, baseline kept: {exc!r}",
                  flush=True)
        orig_init(self, *args, **kwargs)

    attn_cls.__init__ = patched_init
    print("[fa-sliding] Attention.__init__ wrapper active (v1)", flush=True)


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


if FA_SLIDING:
    sys.meta_path.insert(0, _ChainFinder(ATTENTION_TARGET, _patch_attention))
    print("[fa-sliding] finder registered (v1)", flush=True)
