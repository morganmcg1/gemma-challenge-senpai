"""Serve-boot attention pin + pin-active recorder for the strict-#319 served-head cert (PR #690).

This module is imported by the *copied* submission's ``sitecustomize.py`` (one
appended ``import _stark_pin`` line; the original submission is never touched). It
runs in every process of the vLLM server tree (api_server, EngineCore, worker)
because the copy dir is on ``PYTHONPATH`` exactly like the submission's own
sitecustomize boot-patch (kanna #177 precedent). ENABLE_USER_SITE is False in the
serve venv, so usercustomize is unavailable — a sitecustomize-chained import is the
only universal hook.

It does two things, both gated by env so it is a no-op outside this cert:

1. ``STARK_PIN_MODE=fixed2d`` -> force the M=1 AR decode forward onto the 2D
   single-pass kernel (matching the M>=2 verify forward) by overriding
   ``seq_threshold_3D=0`` at the ``unified_attention`` call site, the authoritative
   point where vLLM recomputes ``use_3d`` from that argument (triton_unified_attention
   lines 923-932): with threshold 0, ``num_seqs(>=1) > 0`` makes ``use_3d=False``.
   We ALSO set ``triton_attn.MIN_LAUNCH_GRID_SIZE_2D = 0`` at module import (land
   #684's lever), but that alone does NOT reach the deployed decode: the metadata
   builder derives ``seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // num_heads_kv`` and
   then, with decode CUDA graphs enabled (no ``--enforce-eager``), rounds it to the
   nearest capture size (>=1) -- so the num_seqs=1 decode never reaches threshold 0
   and stayed on 3D split-KV (PR #690 first cut: observed threshold 7, break 53.78%
   ~= the un-pinned 51.70%). The call-site override is the robust fix. NOT a kernel
   rebuild. The sole CUDA call site (triton_attn.forward) passes every argument by
   keyword, so the override is a single ``kwargs`` write.

2. Pin-active PROOF (the lawine #681 load-bearing requirement). When
   ``vllm.v1.attention.ops.triton_unified_attention`` imports we record its
   import-time ``is_batch_invariant`` (= ``envs.VLLM_BATCH_INVARIANT`` frozen at
   import) and wrap ``unified_attention`` to record, for the first few distinct
   forward shapes, the EXACT 2D-vs-3D branch the served forward takes
   (num_seqs, max_seqlen_q, seq_threshold_3D, is_batch_invariant, use_3d). This is
   a direct measurement that the pin is live inside the serving forward, not just in
   the outer env. The wrapper calls straight through; it never changes numerics.

All proof is written as small JSON to ``STARK_PIN_PROOF_DIR`` (one file per pid).
Every hook is wrapped in try/except so a recorder bug can never break serving.
"""
from __future__ import annotations

import json
import os
import sys
import time

_PROOF_DIR = os.environ.get("STARK_PIN_PROOF_DIR")
_MODE = os.environ.get("STARK_PIN_MODE", "none")
_TRITON_ATTN = "vllm.v1.attention.backends.triton_attn"
_UNIFIED = "vllm.v1.attention.ops.triton_unified_attention"
_LOGITS_PROC = "vllm.model_executor.layers.logits_processor"

# Bound the per-process branch recordings so we never spam disk: record at most one
# row per distinct (max_seqlen_q, num_seqs) shape, capped.
_seen_shapes: set = set()
_MAX_SHAPE_ROWS = 24

# ---------------------------------------------------------------------------
# PR #705 served-spec loop localization: pruned/served-head M-dependence probe.
#
# STARK_LOCALIZE_MDEP=1 wraps LogitsProcessor.forward (the lm_head projection the
# served TARGET forward runs for every decode/verify step). For each verify batch
# (M>1) it recomputes position i's logits as a SINGLETON (M=1) on the BYTE-IDENTICAL
# hidden row already in the batch, and records whether the served head's argmax at
# M=batch differs from the M=1 recompute. Because the hidden state is the same tensor
# row, any argmax difference is attributable to the lm_head/sampler M-dependence ITSELF
# (matmul tiling / reduction order), not attention or KV — this is land #684's static
# M=1-vs-M=6 head comparison, but run THROUGH the served forward so it sees the actually
# deployed head (which #690 caveat 1 says won't load into in-process LLM()).
#
# (a) head_mdep_rate high  -> the served head argmax IS M-dependent at matched KV ->
#     the served-spec break has a named, potentially-fixable head/sampler cause.
# (b) head_mdep_rate ~ 0   -> the served head argmax AGREES at M=1 vs M=batch on
#     identical hidden -> the break is NOT the head; it is loop/attention-width
#     mechanics (corroborated by the K-scaling cross-check).
#
# All recording is wrapped in try/except so a probe bug can never break serving.
# Output is a single small JSON per pid (overwritten periodically), KB-scale.
# ---------------------------------------------------------------------------
_LOCALIZE_MDEP = os.environ.get("STARK_LOCALIZE_MDEP", "0") not in ("", "0")
_POS0_CAP = 4000     # cap on per-verify position-0 recomputes (the AR-equivalent token)
_ALLPOS_CAP = 1200   # cap on all-position recomputes (characterizes position dependence)
_MARGIN_SAMPLE = 96  # cap on stored top1-top2 margin samples (knife-edge characterization)

_mdep: dict = {
    "n_logits_calls": 0,       # total LogitsProcessor.forward calls observed
    "n_verify_calls": 0,       # calls with M>1 (verify batches)
    "n_pos0_cmp": 0,           # position-0 M=batch-vs-M=1 comparisons made
    "n_pos0_flip": 0,          # position-0 comparisons where argmax differed
    "n_allpos_cmp": 0,         # all-position comparisons made (capped)
    "n_allpos_flip": 0,        # all-position comparisons where argmax differed
    "m_hist": {},              # M (rows per forward) -> count
    "vocab_hist": {},          # output vocab width -> count
    "head_org_vocab_size": None,
    "head_vocab_size": None,
    "head_quant_method": None,
    "head_soft_cap": None,
    "head_scale": None,
    "pos0_flip_margins": [],   # [top1-top2 gap] at pos0 flips (knife-edge sample)
    "pos0_nonflip_margins": [],# [top1-top2 gap] at pos0 non-flips (baseline tie density)
    "pos0_flip_examples": [],  # [argmax_Mbatch, argmax_M1, margin] capped
    "serve_env_VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT"),
}


def _proof_write(name: str, payload: dict) -> None:
    if not _PROOF_DIR:
        return
    try:
        os.makedirs(_PROOF_DIR, exist_ok=True)
        path = os.path.join(_PROOF_DIR, f"{name}_{os.getpid()}.json")
        with open(path, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass


def _proof_append(name: str, payload: dict) -> None:
    if not _PROOF_DIR:
        return
    try:
        os.makedirs(_PROOF_DIR, exist_ok=True)
        path = os.path.join(_PROOF_DIR, f"{name}_{os.getpid()}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def _patch_triton_attn(module) -> None:
    """Pin MIN_LAUNCH_GRID_SIZE_2D=0 for fixed2d; always record the live constant."""
    before = getattr(module, "MIN_LAUNCH_GRID_SIZE_2D", None)
    after = before
    try:
        if _MODE == "fixed2d":
            module.MIN_LAUNCH_GRID_SIZE_2D = 0
            after = module.MIN_LAUNCH_GRID_SIZE_2D
    except Exception:
        pass
    _proof_write(
        "pin_triton_attn",
        {
            "ts": time.time(),
            "pid": os.getpid(),
            "module": _TRITON_ATTN,
            "stark_pin_mode": _MODE,
            "min_launch_before": before,
            "min_launch_after": after,
            "serve_env_VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT"),
        },
    )


def _record_branch(module, kwargs: dict, forced: bool) -> None:
    """Record the EFFECTIVE (post-override) 2D/3D branch for one distinct shape.

    Reads straight from ``kwargs`` because the sole CUDA call site
    (triton_attn.forward, vllm 0.22.0 line 634) passes every argument by keyword.
    ``use_3d`` is an exact replica of the selector in
    triton_unified_attention.unified_attention (lines 923-932): True -> 3D split-KV,
    False -> 2D single-pass. When ``forced`` is True the ``seq_threshold_3D`` read
    here is already 0, so a num_seqs>=1 decode records as 2D_single_pass."""
    num_seqs = len(kwargs["seqused_k"])
    max_seqlen_q = int(kwargs["max_seqlen_q"])
    shape = (max_seqlen_q, num_seqs)
    if shape in _seen_shapes:
        return
    st3 = kwargs.get("seq_threshold_3D")
    nps = kwargs.get("num_par_softmax_segments")
    so = kwargs.get("softmax_segm_output")
    sm = kwargs.get("softmax_segm_max")
    se = kwargs.get("softmax_segm_expsum")
    ibi = bool(getattr(module, "is_batch_invariant", False))
    use_3d = not (
        st3 is None
        or nps is None
        or so is None
        or sm is None
        or se is None
        or max_seqlen_q > 1
        or num_seqs > st3
        or ibi
    )
    _seen_shapes.add(shape)
    _proof_append(
        "pin_branch",
        {
            "ts": time.time(),
            "pid": os.getpid(),
            "stark_pin_mode": _MODE,
            "num_seqs": num_seqs,
            "max_seqlen_q": max_seqlen_q,
            "seq_threshold_3D": st3,
            "seq_threshold_3D_forced_to_0": bool(forced),
            "is_batch_invariant": ibi,
            "use_3d_split_kv": bool(use_3d),
            "kernel": "3D_split_kv" if use_3d else "2D_single_pass",
        },
    )


def _wrap_unified_attention(module) -> None:
    """Record import-time is_batch_invariant and wrap unified_attention to (a) in
    fixed2d mode force the M=1 decode onto 2D at the call site, and (b) log the
    actual 2D/3D branch the served forward takes (direct pin-active evidence)."""
    is_bi = bool(getattr(module, "is_batch_invariant", False))
    _proof_write(
        "pin_unified_import",
        {
            "ts": time.time(),
            "pid": os.getpid(),
            "module": _UNIFIED,
            "stark_pin_mode": _MODE,
            "is_batch_invariant_at_import": is_bi,
            "serve_env_VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT"),
        },
    )

    orig = getattr(module, "unified_attention", None)
    if orig is None or getattr(orig, "_stark_wrapped", False):
        return

    def wrapper(*args, **kwargs):
        # fixed2d pin (load-bearing): unified_attention recomputes use_3d from the
        # seq_threshold_3D ARGUMENT (vllm 0.22.0 lines 923-932), so forcing it to 0
        # makes the num_seqs>=1 decode take the 2D single-pass path (use_3d=False),
        # matching the M>=2 verify path -> spec==AR byte-exact. The sole call site
        # passes every arg by keyword, so this is a single kwargs write. land #684's
        # MIN_LAUNCH_GRID_SIZE_2D=0 alone can't reach threshold 0: the builder rounds
        # it to the nearest cudagraph capture size (>=1), so the M=1 decode stayed on
        # 3D split-KV (observed threshold 7, break ~unchanged).
        forced = False
        if _MODE == "fixed2d":
            st3 = kwargs.get("seq_threshold_3D")
            if isinstance(st3, int) and st3 != 0:
                kwargs["seq_threshold_3D"] = 0
                forced = True
        if len(_seen_shapes) < _MAX_SHAPE_ROWS:
            try:
                _record_branch(module, kwargs, forced)
            except Exception:
                pass
        return orig(*args, **kwargs)

    try:
        wrapper._stark_wrapped = True  # type: ignore[attr-defined]
        module.unified_attention = wrapper
    except Exception:
        pass


def _flush_mdep() -> None:
    """Overwrite a single compact per-pid summary (KB-scale, survives SIGTERM)."""
    if not _PROOF_DIR:
        return
    try:
        n0 = _mdep["n_pos0_cmp"] or 0
        na = _mdep["n_allpos_cmp"] or 0
        payload = dict(_mdep)
        payload["pid"] = os.getpid()
        payload["ts"] = time.time()
        payload["stark_pin_mode"] = _MODE
        payload["head_mdep_rate_pos0"] = (_mdep["n_pos0_flip"] / n0) if n0 else None
        payload["head_mdep_rate_allpos"] = (_mdep["n_allpos_flip"] / na) if na else None
        os.makedirs(_PROOF_DIR, exist_ok=True)
        path = os.path.join(_PROOF_DIR, f"mdep_summary_{os.getpid()}.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception:
        pass


def _install_mdep_sigterm() -> None:
    """Flush the per-pid mdep summary on the controlled SIGTERM shutdown.

    ``LocalServer.__exit__`` sends SIGTERM to the serve process group (30s grace
    before SIGKILL), and Python's atexit does NOT run on SIGTERM, so the final
    counters would be lost without this. Best-effort: chains to any prior handler;
    if vLLM later overrides it, the eager+periodic in-loop flush still keeps the
    summary fresh (we only ever lose a sub-64 trailing bucket). Must run on the main
    thread (true at sitecustomize import); any failure falls back to the in-loop flush.
    """
    try:
        import signal

        prev = signal.getsignal(signal.SIGTERM)

        def _handler(signum, frame, _prev=prev):
            try:
                _flush_mdep()
            except Exception:
                pass
            if callable(_prev):
                _prev(signum, frame)
            else:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                os.kill(os.getpid(), signal.SIGTERM)

        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass


def _record_mdep(self, lm_head, hidden_states, out, embedding_bias, orig) -> None:
    """Compare the served head's M=batch argmax to an M=1 recompute on identical hidden.

    ``out`` is the server's real LogitsProcessor.forward output for the whole batch.
    For each verify forward (M>1) we recompute position i as a singleton via ``orig``
    on ``hidden_states[i:i+1]`` (the SAME tensor row) and check whether the argmax
    flips. A flip means the lm_head/sampler is M-dependent at matched KV (cause (a));
    agreement means the head is M-invariant and the served break is loop mechanics (b).
    """
    import torch

    if out is None or hidden_states is None or not hasattr(hidden_states, "shape"):
        return
    if hidden_states.dim() != 2 or out.dim() != 2:
        return
    m = int(hidden_states.shape[0])
    v = int(out.shape[-1])
    _mdep["n_logits_calls"] += 1
    _mdep["m_hist"][m] = _mdep["m_hist"].get(m, 0) + 1
    _mdep["vocab_hist"][v] = _mdep["vocab_hist"].get(v, 0) + 1
    if _mdep["head_quant_method"] is None:
        qm = getattr(lm_head, "quant_method", None)
        _mdep["head_quant_method"] = type(qm).__name__ if qm is not None else None
        _mdep["head_org_vocab_size"] = getattr(self, "org_vocab_size", None)
        _mdep["head_vocab_size"] = getattr(self, "vocab_size", None)
        sc = getattr(self, "soft_cap", None)
        _mdep["head_soft_cap"] = float(sc) if isinstance(sc, (int, float)) else sc
        scale = getattr(self, "scale", None)
        _mdep["head_scale"] = float(scale) if isinstance(scale, (int, float)) else scale

    if m <= 1:
        return  # singleton forward: nothing to compare against M=1
    _mdep["n_verify_calls"] += 1

    argmax_batch = out.argmax(dim=-1)  # [M] argmax the server actually uses

    # Position 0 = the AR-equivalent next token (the one an M=1 AR decode emits at
    # this same prefix). This is the apples-to-apples comparison to the served break.
    if _mdep["n_pos0_cmp"] < _POS0_CAP:
        single0 = orig(self, lm_head, hidden_states[0:1], embedding_bias)
        a_batch0 = int(argmax_batch[0].item())
        a_single0 = int(single0.argmax(dim=-1)[0].item())
        _mdep["n_pos0_cmp"] += 1
        row0 = out[0].float()
        top2 = torch.topk(row0, 2).values
        margin = float((top2[0] - top2[1]).item())
        if a_batch0 != a_single0:
            _mdep["n_pos0_flip"] += 1
            if len(_mdep["pos0_flip_margins"]) < _MARGIN_SAMPLE:
                _mdep["pos0_flip_margins"].append(margin)
            if len(_mdep["pos0_flip_examples"]) < 32:
                _mdep["pos0_flip_examples"].append([a_batch0, a_single0, margin])
        elif len(_mdep["pos0_nonflip_margins"]) < _MARGIN_SAMPLE:
            _mdep["pos0_nonflip_margins"].append(margin)

    # All-position characterization (capped): does any verify position flip?
    if _mdep["n_allpos_cmp"] < _ALLPOS_CAP:
        for i in range(m):
            if _mdep["n_allpos_cmp"] >= _ALLPOS_CAP:
                break
            si = orig(self, lm_head, hidden_states[i : i + 1], embedding_bias)
            _mdep["n_allpos_cmp"] += 1
            if int(si.argmax(dim=-1)[0].item()) != int(argmax_batch[i].item()):
                _mdep["n_allpos_flip"] += 1

    # Flush eagerly for the first verify calls (so even a tiny smoke persists a
    # summary) then periodically. SIGTERM (LocalServer's shutdown) does NOT run
    # atexit, so the in-loop flush is the primary guarantee; _install_mdep_sigterm
    # adds a best-effort final flush on top.
    nv = _mdep["n_verify_calls"]
    if nv <= 32 or nv % 64 == 0:
        _flush_mdep()


def _wrap_logits_processor(module) -> None:
    """Wrap LogitsProcessor.forward to record served-head M-dependence (no numerics change)."""
    cls = getattr(module, "LogitsProcessor", None)
    if cls is None:
        return
    orig = getattr(cls, "forward", None)
    if orig is None or getattr(orig, "_stark_mdep_wrapped", False):
        return

    def forward(self, lm_head, hidden_states, embedding_bias=None, _orig=orig):
        out = _orig(self, lm_head, hidden_states, embedding_bias)
        try:
            _record_mdep(self, lm_head, hidden_states, out, embedding_bias, _orig)
        except Exception:
            pass
        return out

    try:
        forward._stark_mdep_wrapped = True  # type: ignore[attr-defined]
        cls.forward = forward
        _proof_write(
            "mdep_installed",
            {"ts": time.time(), "pid": os.getpid(), "localize_mdep": _LOCALIZE_MDEP},
        )
    except Exception:
        pass


def _apply(module) -> None:
    try:
        name = getattr(module, "__name__", "")
        if name == _TRITON_ATTN:
            _patch_triton_attn(module)
        elif name == _UNIFIED:
            _wrap_unified_attention(module)
        elif name == _LOGITS_PROC and _LOCALIZE_MDEP:
            _wrap_logits_processor(module)
    except Exception:
        pass


def _install() -> None:
    # Patch any already-imported targets, then install a one-shot finder for the
    # rest (mirrors the submission sitecustomize's spec-decode finder pattern).
    _install_targets = [_TRITON_ATTN, _UNIFIED]
    if _LOCALIZE_MDEP:
        _install_targets.append(_LOGITS_PROC)
        _install_mdep_sigterm()
    for target in _install_targets:
        if target in sys.modules:
            _apply(sys.modules[target])

    from importlib.abc import MetaPathFinder
    from importlib.util import find_spec

    targets = set(_install_targets)

    class _StarkPinFinder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in targets:
                return None
            # Temporarily remove ourselves so the nested find_spec resolves the
            # real loader without recursing.
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            try:
                spec = find_spec(fullname)
            finally:
                if self not in sys.meta_path:
                    sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(mod, _orig=orig_exec):
                _orig(mod)
                _apply(mod)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _StarkPinFinder())


try:
    _install()
    _proof_write(
        "pin_installed",
        {"ts": time.time(), "pid": os.getpid(), "stark_pin_mode": _MODE},
    )
except Exception:
    pass
