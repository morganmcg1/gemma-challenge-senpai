"""PR #566 fern — LIVE candidate-verify head, wired into the vLLM sampler logits
path (env-gated, CV_HEAD=1). Converts #560's microbench+reprojection (cv_served_tps
_is_measured=False, projected 291.36 TPS) into a SERVED, end-to-end measurement.

Loaded into the vLLM worker through the research-dir ``sitecustomize.py`` chain
(user-site import is disabled in the worker, so ``sitecustomize`` is the channel).
Registers a meta-path finder that, after ``vllm.model_executor.models.gemma4``
loads, REPLACES ``Gemma4ForCausalLM.compute_logits`` (the 262k full-bf16-head GEMM,
1.342 GB read per verify step) with the candidate-verify path #560 measured:

  (a) int4-Marlin nominator GEMV: hidden[M,2560] @ W_int4[2560,262144] -> cand_logits.
      Reads packed uint4b8 weights + bf16 group scales = 0.346 GB (vs 1.342 GB bf16).
      This is the read-bound win (~3.88x fewer head bytes).
  (b) top-K=8 shortlist over cand_logits -> gather the K bf16 head rows -> EXACT bf16
      recompute of the K logits (the verify; matches the served bf16 head math).
  (c) scatter the K verify logits back into a [M, vocab] tensor (rest = -inf) and
      RETURN it. The server's own sampler/argmax then picks the max verify logit.

THE TIE-BREAK INVARIANT (why this is byte-exact without knowing the server's argmax
kernel): #560 stage3 measured miss_rate=0 at K=8 -> the true bf16-greedy token is
ALWAYS in the int4 top-8 shortlist. We scatter the EXACT bf16 logit at each shortlist
vocab index. The reference server argmaxes the full bf16 logits and picks token P
(its tie-break); P has the global-max bf16 value, P is in our shortlist, and we place
P's exact value at vocab index P -> the server's SAME argmax kernel over our spiked
[M,vocab] tensor picks the SAME P (any tie is resolved identically because the tied
positions, all >= the winner, are themselves in the shortlist). So identity holds for
ANY deterministic vocab-indexed argmax -- we do NOT replicate the kernel's tie-break,
we inherit it. (#560's 0.99545 failure mode was argmax over the [M,K] shortlist array,
where index==shortlist position, not vocab; scattering into vocab positions fixes it.)

Modes (env):
  CV_HEAD=1     install the replacement (default-off; unset -> module inert).
  CV_AUDIT=1    ALSO compute the bf16 oracle (base compute_logits) every step and
                compare its argmax to ours -> live per-step identity rate. SLOW
                (both heads run); use a SHORT pass. Returns the CV (spiked) logits so
                the decode follows the CV trajectory (identity is audited at the
                realized CV states). TPS from an audit pass is meaningless.
  CV_KSAFE=8    shortlist width (default 8; #560 K_safe).
  CV_GROUP=128  int4 group size.

Default-off: with CV_HEAD unset this registers nothing -> served path byte-identical.
No shipped submission file is modified.
"""
from __future__ import annotations

import atexit
import importlib.abc
import importlib.util
import os
import sys
import time
from typing import Any

_ENABLED = os.environ.get("CV_HEAD", "0") == "1"
_AUDIT = os.environ.get("CV_AUDIT", "0") == "1"
_KSAFE = int(os.environ.get("CV_KSAFE", "8"))
_GROUP = int(os.environ.get("CV_GROUP", "128"))
_WARMUP_SKIP = int(os.environ.get("CV_WARMUP_SKIP", "64"))
_REPORT_EVERY = int(os.environ.get("CV_REPORT_EVERY", "2000"))
_DECODE_M_MAX = int(os.environ.get("CV_DECODE_M_MAX", "8"))
# audit identity is accumulated in the EngineCore WORKER, which is terminated by SIGTERM
# at server teardown -> Python atexit does NOT run there. So flush the audit JSON + print
# the cumulative line every N audit calls; the last flush survives the kill. (atexit still
# runs on clean exits as a backstop.)
_AUDIT_FLUSH_EVERY = int(os.environ.get("CV_AUDIT_FLUSH_EVERY", "16"))

_TARGET = "vllm.model_executor.models.gemma4"

_state: dict[str, Any] = {
    "i": 0,
    "built": False,
    "q": None,           # (marlin_q_w, marlin_s, zp, g_idx, sort_indices, workspace)
    "rows": None,        # bf16 [vocab, hidden] for the verify gather
    "vocab": None,
    "hidden": None,
    "out_dtype": None,
    "neg": None,
    # audit
    "audit_total": 0,
    "audit_mismatch": 0,
    "audit_mismatch_examples": [],
    "tie_positions": 0,
    # head-time breakdown (speed mode, CUDA-event)
    "pending": [],       # (ev0, ev1, m, i)
    "decode_cv_ms": [0, 0.0],   # [count, sum_ms] for the CV step (m<=DECODE_M_MAX)
    "decode_base_ms": [0, 0.0],  # base head, audit mode only
    "m_hist": {},
    "reported": 0,
}


def _build(self_model: Any, hidden_states: Any, base: Any, a: tuple, kw: dict) -> None:
    """One-time: quantize the served bf16 lm_head to int4 Marlin (uint4b8, g128) and
    cache the verify rows. Learns vocab/dtype from a single base() reference call."""
    import torch

    from vllm.scalar_type import scalar_types
    import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
    import vllm.model_executor.layers.quantization.utils.marlin_utils_test as mt

    # learn the reference logits dtype/shape (one base call; also the audit oracle is
    # exactly this path, so this is faithful to what the server argmaxes).
    ref = base(self_model, hidden_states, *a, **kw)
    out_dtype = ref.dtype
    vocab = int(ref.shape[-1])
    dev = ref.device

    lm_head = getattr(self_model, "lm_head", None)
    if lm_head is None or not hasattr(lm_head, "weight"):
        raise RuntimeError(f"[cv-head] cannot find lm_head.weight on {type(self_model)}")
    W = lm_head.weight  # [vocab, hidden] bf16 (logits = hidden @ W.T)
    assert W.dim() == 2 and W.shape[0] == vocab, (tuple(W.shape), vocab)
    hidden = int(W.shape[1])

    # verify gather ALIASES the model's resident bf16 head (no second copy): rows[topk]
    # is advanced-indexed -> always a fresh contiguous [M,K,hidden], so W's own layout is
    # irrelevant to the verify. Avoids a redundant 1.342 GB persistent copy that does not
    # fit alongside the int4 nominator on the 23 GB A10G. (PLE embed-scale fold already
    # ran at load time, before this build, so W is final and safe to alias.)
    rows = W
    w_marlin = W.t().contiguous()  # [hidden, vocab] transient for marlin quantize

    t0 = time.time()
    _wref, marlin_q_w, marlin_s, g_idx, sort_indices, _r = mt.marlin_quantize(
        w_marlin, scalar_types.uint4b8, _GROUP, act_order=False
    )
    del w_marlin, _wref
    # return the [hidden,vocab] transient + dequant reference to the allocator BEFORE
    # vLLM's KV-cache memory profiling reads free VRAM (defragments the build peak).
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    zp = mu.marlin_make_empty_g_idx(dev)
    workspace = mu.marlin_make_workspace_new(torch.device(dev))
    int4_bytes = (marlin_q_w.numel() * marlin_q_w.element_size()
                  + marlin_s.numel() * marlin_s.element_size())

    _state.update(
        built=True,
        q=(marlin_q_w, marlin_s, zp, g_idx, sort_indices, workspace),
        rows=rows, vocab=vocab, hidden=hidden, out_dtype=out_dtype,
        neg=torch.finfo(out_dtype).min,
    )
    print(
        f"[cv-head] built int4 nominator: vocab={vocab} hidden={hidden} K_safe={_KSAFE} "
        f"group={_GROUP} int4_read={int4_bytes/1e9:.4f}GB (vs bf16 {W.numel()*2/1e9:.4f}GB) "
        f"quant={time.time()-t0:.1f}s out_dtype={out_dtype} audit={_AUDIT} pid={os.getpid()}",
        flush=True,
    )


def _cv_logits(hidden_states: Any) -> Any:
    """(a) int4 GEMV  (b) top-K gather + bf16 verify  (c) scatter -> [M,vocab] spiked."""
    import torch

    from vllm.scalar_type import scalar_types
    import vllm.model_executor.layers.quantization.utils.marlin_utils as mu

    marlin_q_w, marlin_s, zp, g_idx, sort_indices, workspace = _state["q"]
    vocab, hidden = _state["vocab"], _state["hidden"]
    x = hidden_states if hidden_states.dtype == torch.bfloat16 else hidden_states.to(torch.bfloat16)
    cand = mu.apply_gptq_marlin_linear(
        input=x, weight=marlin_q_w, weight_scale=marlin_s, weight_zp=zp,
        g_idx=g_idx, g_idx_sort_indices=sort_indices, workspace=workspace,
        wtype=scalar_types.uint4b8, output_size_per_partition=vocab,
        input_size_per_partition=hidden, is_k_full=True,
    )  # [M, vocab]
    topk = cand.topk(_KSAFE, dim=1).indices             # [M, K]
    rows = _state["rows"][topk]                          # [M, K, hidden] bf16
    # match the served head's numeric class: bf16 inputs, fp32 accumulate, bf16 output.
    verify = torch.einsum("mh,mkh->mk", x.float(), rows.float()).to(torch.bfloat16)  # [M, K]
    out = torch.full_like(cand, _state["neg"], dtype=_state["out_dtype"])
    out.scatter_(1, topk, verify.to(_state["out_dtype"]))
    return out


def _report() -> None:
    cv = _state["decode_cv_ms"]
    bm = _state["decode_base_ms"]
    cvm = cv[1] / cv[0] if cv[0] else float("nan")
    bmm = bm[1] / bm[0] if bm[0] else float("nan")
    line = (f"[cv-head] agg cv_calls={cv[0]} cv_head_ms_mean={cvm:.4f} "
            f"m_hist={dict(sorted(_state['m_hist'].items()))}")
    if _AUDIT:
        tot = _state["audit_total"]
        mis = _state["audit_mismatch"]
        rate = (tot - mis) / tot if tot else float("nan")
        line += (f" | audit rows={tot} mismatch={mis} identity={rate:.6f} "
                 f"base_head_ms_mean={bmm:.4f} ties={_state['tie_positions']}")
    print(line, flush=True)


def _write_audit_file() -> None:
    """Persist the audit/timing detail for the driver. GUARD: only a process that actually
    ran the head writes -- the API-server process imports this module too but never calls
    compute_logits (rows=0), so without this guard it would clobber the worker's real data."""
    path = os.environ.get("CV_AUDIT_OUT")
    if not path:
        return
    if _state["audit_total"] == 0 and _state["decode_cv_ms"][0] == 0:
        return
    try:
        import json
        cv = _state["decode_cv_ms"]
        tot = _state["audit_total"]
        json.dump({
            "audit_total": tot,
            "audit_mismatch": _state["audit_mismatch"],
            "identity_rate": ((tot - _state["audit_mismatch"]) / tot if tot else None),
            "tie_positions": _state["tie_positions"],
            "mismatch_examples": _state["audit_mismatch_examples"],
            "m_hist": {str(k): v for k, v in _state["m_hist"].items()},
            "cv_decode_calls": cv[0],
            "cv_head_ms_mean": (cv[1] / cv[0] if cv[0] else None),
            "k_safe": _KSAFE, "group": _GROUP, "audit": _AUDIT,
            "pid": os.getpid(),
        }, open(path, "w"), indent=2)
    except Exception as exc:  # pragma: no cover
        print(f"[cv-head] audit-out write failed: {exc!r}", flush=True)


def _resolve(force: bool = False) -> None:
    import torch  # noqa: F401
    pend = _state["pending"]
    while pend:
        ev0, ev1, m, i = pend[0]
        if not force and not ev1.query():
            break
        if force:
            ev1.synchronize()
        ms = ev0.elapsed_time(ev1)
        pend.pop(0)
        if i >= _WARMUP_SKIP and m <= _DECODE_M_MAX:
            _state["decode_cv_ms"][0] += 1
            _state["decode_cv_ms"][1] += ms
        done = _state["decode_cv_ms"][0]
        if done and done % _REPORT_EVERY == 0 and done != _state["reported"]:
            _state["reported"] = done
            _report()


def _wrap(module: Any) -> None:
    import torch

    cls = getattr(module, "Gemma4ForCausalLM", None)
    if cls is None:
        print("[cv-head] WARN: Gemma4ForCausalLM not found; hook inert", flush=True)
        return
    base = cls.compute_logits

    def compute_logits(self_model: Any, hidden_states: Any, *a: Any, **kw: Any) -> Any:
        if hidden_states is None or hidden_states.dim() < 1:
            return base(self_model, hidden_states, *a, **kw)
        if not _state["built"]:
            _build(self_model, hidden_states, base, a, kw)
        m = int(hidden_states.shape[0])
        _state["m_hist"][m] = _state["m_hist"].get(m, 0) + 1

        if _AUDIT:
            ref = base(self_model, hidden_states, *a, **kw)   # bf16 oracle
            out = _cv_logits(hidden_states)
            ref_tok = ref.argmax(dim=-1)
            cv_tok = out.argmax(dim=-1)
            mism = (ref_tok != cv_tok)
            n = int(mism.numel())
            nm = int(mism.sum().item())
            _state["audit_total"] += n
            _state["audit_mismatch"] += nm
            # bf16 top-1 ties (diagnostic): rows where >=2 vocab share the row-max bf16 logit
            row_max = ref.max(dim=-1, keepdim=True).values
            _state["tie_positions"] += int((ref == row_max).sum(dim=-1).gt(1).sum().item())
            if nm and len(_state["audit_mismatch_examples"]) < 32:
                idx = torch.nonzero(mism, as_tuple=False).flatten().tolist()
                for j in idx[:32]:
                    _state["audit_mismatch_examples"].append(
                        {"step": _state["i"], "row": j,
                         "ref_tok": int(ref_tok[j]), "cv_tok": int(cv_tok[j]),
                         "ref_logit_ref": float(ref[j, ref_tok[j]]),
                         "ref_logit_cv": float(ref[j, cv_tok[j]])})
            _state["i"] += 1
            # flush identity to log + file every N audit calls (survives SIGTERM teardown)
            if _state["i"] % _AUDIT_FLUSH_EVERY == 0:
                _report()
                _write_audit_file()
            return out

        # speed mode: CUDA-event-time the CV step, no base call
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        out = _cv_logits(hidden_states)
        ev1.record()
        _state["pending"].append((ev0, ev1, m, _state["i"]))
        _state["i"] += 1
        _resolve()
        return out

    cls.compute_logits = compute_logits
    print(f"[cv-head] REPLACED Gemma4ForCausalLM.compute_logits (audit={_AUDIT} "
          f"K_safe={_KSAFE}) pid={os.getpid()}", flush=True)


def _atexit() -> None:
    _resolve(force=True)
    _report()
    cv = _state["decode_cv_ms"]
    line = (f"[cv-head] FINAL cv_decode_calls={cv[0]} cv_head_ms_sum={cv[1]:.2f} "
            f"cv_head_ms_mean={(cv[1]/cv[0] if cv[0] else float('nan')):.4f}")
    if _AUDIT:
        tot, mis = _state["audit_total"], _state["audit_mismatch"]
        rate = (tot - mis) / tot if tot else float("nan")
        line += f" | AUDIT_IDENTITY rows={tot} mismatch={mis} identity_rate={rate:.6f}"
    print(line, flush=True)
    # persist audit detail for the driver (guarded: a no-op in processes that never decoded)
    _write_audit_file()
    if _state["audit_total"] or _state["decode_cv_ms"][0]:
        print(f"[cv-head] wrote audit detail -> {os.environ.get('CV_AUDIT_OUT')}", flush=True)


class _Loader(importlib.abc.Loader):
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create_module(self, spec: Any) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._inner.exec_module(module)
        _wrap(module)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self._busy = False

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != _TARGET or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader)
        return spec


if _ENABLED:
    sys.meta_path.insert(0, _Finder())
    atexit.register(_atexit)
    print(f"[cv-head] finder registered (audit={_AUDIT} K_safe={_KSAFE}) pid={os.getpid()}", flush=True)
