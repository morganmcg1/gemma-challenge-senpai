"""Runtime lm_head head-WIDTH knob for the intact-body head-prune sweep (PR #547).

The intact base-int4 body (google/gemma-4-E4B-it-qat-w4a16-ct: 42 layers, int4
weight_packed transformer + BF16 tied lm_head) is served UNCHANGED; only the
OUTPUT vocabulary width is varied. Because disk is full (11 GB model, <7 GB free)
we cannot write a row-pruned checkpoint, so we do the prune at RUNTIME in VRAM.

Two modes (HEADWIDTH_MODE):

  * ``mask`` (default) -- the full BF16 [262144, 2560] head runs unchanged; we add
    an additive keepset mask to the logits (0 at kept ids, -inf elsewhere). For
    argmax (greedy) and for temperature/top-k/top-p sampling this is BIT-FAITHFUL
    to a physically row-pruned head: kept-position logits are identical, non-kept
    are -inf (probability exactly 0). This is the QUALITY arm -- it reproduces the
    exact token stream a K-row head would emit. It does NOT change lm_head FLOPs,
    so it carries FULL-head speed.

  * ``slice`` -- on first compute_logits we build a pruned weight W_K = W[keep_ids]
    ([K, 2560], a fresh tensor that BREAKS the embed_tokens tie so the full input
    embedding is untouched), then every step does only the [M, K] GEMV and scatters
    [M, K] -> [M, 262144] with -inf at non-kept ids. This is the SPEED arm -- the
    lm_head read drops from 262144*2560 to K*2560 BF16. Token output is identical to
    ``mask`` (validated A/B), so TPS measured here is the genuine pruned-head speed
    on the intact body.

Hard-reject guard (HEADWIDTH_REQUIRE=1, the default): in slice mode we assert the
built head has exactly K rows; in BOTH modes we assert the loaded full head is the
expected 262144x2560 intact head before we touch it, so a silent fallback to a
different substrate cannot masquerade as a head-width cell.

Activated only when HEADWIDTH_KEEPSET is set. No GPU work at import.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

HEADWIDTH_KEEPSET = os.environ.get("HEADWIDTH_KEEPSET", "")
HEADWIDTH_MODE = os.environ.get("HEADWIDTH_MODE", "mask").strip().lower()
HEADWIDTH_REQUIRE = os.environ.get("HEADWIDTH_REQUIRE", "1") == "1"
EXPECT_FULL_VOCAB = int(os.environ.get("HEADWIDTH_FULL_VOCAB", "262144"))
EXPECT_HIDDEN = int(os.environ.get("HEADWIDTH_HIDDEN", "2560"))

_TARGET_MODULE = "vllm.model_executor.models.gemma4"
_TARGET_CLASS = "Gemma4ForCausalLM"
_TARGET_METHOD = "compute_logits"

_state: dict[str, Any] = {
    "keep_ids": None,
    "full_vocab": None,
    "K": None,
    "device_cache": {},   # device -> buffers
    "sliced": False,
    "warned_mode": False,
}


def _load_keepset(path: str) -> tuple[list[int], int]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[headwidth] HEADWIDTH_KEEPSET={path!r} does not exist")
    d = json.loads(p.read_text())
    keep = [int(x) for x in d["keep_ids"]]
    fv = int(d.get("full_vocab") or d.get("vocab_size") or 0)
    if fv == 0:
        raise ValueError(f"[headwidth] keepset {path!r} has no full_vocab/vocab_size")
    if len(set(keep)) != len(keep):
        raise ValueError("[headwidth] keepset has duplicate ids")
    if max(keep) >= fv:
        raise ValueError(f"[headwidth] keepset id {max(keep)} >= full_vocab {fv}")
    return keep, fv


def _device_buffers(device: Any) -> dict[str, Any]:
    import torch  # type: ignore
    ds = str(device)
    cache = _state["device_cache"]
    if ds in cache:
        return cache[ds]
    keep_ids = _state["keep_ids"]
    full_vocab = _state["full_vocab"]
    K = _state["K"]
    keep_idx = torch.tensor(keep_ids, dtype=torch.long, device=device)
    # additive mask: 0 at kept ids, -inf elsewhere (float32 for safe -inf).
    add_mask = torch.full((full_vocab,), float("-inf"), dtype=torch.float32, device=device)
    add_mask.index_fill_(0, keep_idx, 0.0)
    bufs = {"keep_idx": keep_idx, "add_mask": add_mask, "K": K, "full_vocab": full_vocab}
    cache[ds] = bufs
    print(f"[headwidth] device buffers on {ds}: add_mask[{full_vocab}] keep_idx[{K}] "
          f"(pid {os.getpid()})", file=sys.stderr, flush=True)
    return bufs


def _scatter(pruned_logits: Any, bufs: dict[str, Any]) -> Any:
    import torch  # type: ignore
    keep_idx = bufs["keep_idx"]
    full_vocab = bufs["full_vocab"]
    K = bufs["K"]
    M = pruned_logits.shape[0]
    assert pruned_logits.shape[-1] == K, (
        f"[headwidth] FINGERPRINT FAIL: expected [M,{K}] got {list(pruned_logits.shape)}")
    out = torch.full((M, full_vocab), float("-inf"),
                     dtype=pruned_logits.dtype, device=pruned_logits.device)
    out.index_copy_(1, keep_idx, pruned_logits)
    return out


def _apply(module: Any) -> None:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    if not HEADWIDTH_KEEPSET:
        print("[headwidth] HEADWIDTH_KEEPSET unset -- head-width knob INACTIVE",
              file=sys.stderr, flush=True)
        return
    if HEADWIDTH_MODE not in ("mask", "slice"):
        raise ValueError(f"[headwidth] HEADWIDTH_MODE={HEADWIDTH_MODE!r} not in mask|slice")

    keep_ids, full_vocab = _load_keepset(HEADWIDTH_KEEPSET)
    _state["keep_ids"] = keep_ids
    _state["full_vocab"] = full_vocab
    _state["K"] = len(keep_ids)

    cls = getattr(module, _TARGET_CLASS, None)
    assert cls is not None, f"[headwidth] {_TARGET_CLASS} not found in {module.__name__}"
    original_compute_logits = getattr(cls, _TARGET_METHOD, None)
    assert original_compute_logits is not None, f"[headwidth] {_TARGET_METHOD} missing"

    K = _state["K"]

    def _verify_full_head(self_model: Any) -> Any:
        """Confirm the loaded head is the intact full BF16 EXPECT_FULL_VOCAB x
        EXPECT_HIDDEN head before we touch it (no silent substrate swap)."""
        lm_head = getattr(self_model, "lm_head", None)
        assert lm_head is not None, "[headwidth] model has no lm_head"
        w = getattr(lm_head, "weight", None)
        assert w is not None, "[headwidth] lm_head has no .weight"
        rows, cols = int(w.shape[0]), int(w.shape[1])
        if HEADWIDTH_REQUIRE:
            assert rows == EXPECT_FULL_VOCAB and cols == EXPECT_HIDDEN, (
                f"[headwidth] HARD-REJECT: expected intact full head "
                f"[{EXPECT_FULL_VOCAB},{EXPECT_HIDDEN}], loaded [{rows},{cols}] -- "
                f"refusing to run a head-width cell on the wrong substrate")
        assert full_vocab == EXPECT_FULL_VOCAB, (
            f"[headwidth] keepset full_vocab {full_vocab} != expected {EXPECT_FULL_VOCAB}")
        return w

    def compute_logits_mask(self_model: Any, hidden_states: "torch.Tensor") -> Any:
        logits = original_compute_logits(self_model, hidden_states)
        if logits is None:
            return None
        if not _state.get("verified"):
            _verify_full_head(self_model)
            _state["verified"] = True
            print(f"[headwidth] MASK active: full head verified, masking to K={K} "
                  f"kept ids (pid {os.getpid()})", file=sys.stderr, flush=True)
        bufs = _device_buffers(logits.device)
        # additive mask broadcast over [M, full_vocab]; -inf at non-kept.
        return logits + bufs["add_mask"].to(logits.dtype)

    def compute_logits_slice(self_model: Any, hidden_states: "torch.Tensor") -> Any:
        if not _state.get("sliced"):
            w = _verify_full_head(self_model)
            keep_idx = torch.tensor(keep_ids, dtype=torch.long, device=w.device)
            pruned_w = w.index_select(0, keep_idx).contiguous().clone()  # [K, H], breaks tie
            assert int(pruned_w.shape[0]) == K, (
                f"[headwidth] HARD-REJECT: sliced head rows {pruned_w.shape[0]} != K {K}")
            soft_cap = getattr(getattr(self_model, "config", None),
                               "final_logit_softcapping", None)
            _state["pruned_w"] = pruned_w
            _state["soft_cap"] = soft_cap
            _state["sliced"] = True
            print(f"[headwidth] SLICE active: built pruned head [{pruned_w.shape[0]},"
                  f"{pruned_w.shape[1]}] {pruned_w.dtype}, soft_cap={soft_cap} "
                  f"(pid {os.getpid()})", file=sys.stderr, flush=True)
        pruned_w = _state["pruned_w"]
        logits_K = F.linear(hidden_states, pruned_w)  # [M, K] -- genuine pruned GEMV
        soft_cap = _state["soft_cap"]
        if soft_cap is not None:
            logits_K = soft_cap * torch.tanh(logits_K / soft_cap)
        bufs = _device_buffers(hidden_states.device)
        return _scatter(logits_K, bufs)

    cls.compute_logits = compute_logits_mask if HEADWIDTH_MODE == "mask" else compute_logits_slice
    print(f"[headwidth] patched {_TARGET_CLASS}.{_TARGET_METHOD} mode={HEADWIDTH_MODE} "
          f"K={K} full_vocab={full_vocab} keepset={HEADWIDTH_KEEPSET!r} "
          f"require={HEADWIDTH_REQUIRE} (pid {os.getpid()})", file=sys.stderr, flush=True)


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, inner: importlib.abc.Loader, fn: Any) -> None:
        self._inner = inner
        self._fn = fn

    def create_module(self, spec: Any) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._inner.exec_module(module)
        self._fn(module)


class _TargetFinder(importlib.abc.MetaPathFinder):
    def __init__(self, target: str, fn: Any) -> None:
        self._target = target
        self._fn = fn
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
        spec.loader = _PatchingLoader(spec.loader, self._fn)
        return spec


sys.meta_path.insert(0, _TargetFinder(_TARGET_MODULE, _apply))
