"""Reject-rank + verifier-entropy probe (PR #820): viability gate for top-k / FLy accept.

ENV-GATED by ``REJECTRANK_ENABLE=1``. When unset this module is never imported by
``sitecustomize`` (the hook is only installed when the env var is "1"), so the shipped
int4head serving path is byte-for-byte unchanged. Even when ENABLED the probe only
ADDS logging and calls the original ``RejectionSampler.forward`` unchanged, so the
emitted greedy tokens stay identical to production (validated by the greedy-identity
gate in the capture harness).

WHAT IT MEASURES
----------------
The shipped greedy accept criterion is strict ``draft == verifier-argmax``
(``rejection_sampler.py``: greedy path computes ``target_argmax = target_logits.argmax(-1)``
then the Triton kernel rejects where ``draft != target_argmax``). The acceptance
frontier (real E_accept=3.379 / r~0.397 vs oracle r=1.0 = +52% TPS) is the biggest
remaining lever. Two relaxations are on the table:
  (A) uniform top-k: accept draft if rank(draft) <= k everywhere.
  (B) entropy-gated top-k (FLy-inspired, arXiv:2511.22972): accept via top-k ONLY
      at high verifier-entropy positions; keep strict argmax where the verifier is
      confident (low entropy), where a non-argmax accept would be a real error.

To project the realized E_accept(k)/TPS of both BEFORE committing kernel quality
risk, we record -- for every draft position on the deployed greedy verify path --
the data both policies need. We wrap ``RejectionSampler.forward`` (the same hook
PR #86's verifier-prob probe used) and read the FULL-vocab target logits
(``logits[target_logits_indices]``) that the sampler already computed. Per draft
position we record:

  acc -- strict accept (draft == verifier argmax): the EXACT shipped criterion.
  rk  -- 1-indexed rank of the draft token in the verifier's logits, defined as
         ``(#tokens with strictly-greater logit) + 1``. rk==1 => draft is (tied for)
         the verifier argmax; ``draft in topk(k)`` <=> ``rk <= k``. (At an exact
         logit tie this is lenient -- a tied draft counts at the best rank -- which
         is exactly the semantics a top-k accept kernel would give that token.)
  H   -- verifier predictive entropy H = -sum p*log p over the full vocab (nats).
         Offline we also report the FLy-normalized h = H / log(V) in [0,1] so the
         sweep threshold maps onto FLy's theta=0.3.
  dp  -- verifier softmax prob mass at the DRAFT token (how wrong the draft is).
  ap  -- verifier top-1 (argmax) softmax prob (verifier confidence; 1-ap and H both
         measure flatness).

Records are emitted ONE JSONL line per (decode step, request-in-batch). Within a
block, positions are in draft-depth order, so the offline projector can replay each
real verify block under a relaxed criterion: vLLM computes ALL K draft-position
logits in ONE teacher-forced pass, so the logit at depth d is fixed regardless of
the accept decision at depths < d -- replaying acceptance WITHIN a block is exactly
faithful. (The only projection approximation is cross-block context drift, the same
assumption stark's TPS-vs-acceptance-length oracle curve makes.)

WHY CONTRACT-SAFE
-----------------
``logits.index_select`` makes a copy and ``softmax``/comparisons allocate new
tensors; ``logits`` is never written, and the original ``forward`` is called with
the untouched arguments and its result returned verbatim. Non-greedy batches are
skipped entirely (the challenge benchmark is greedy/temp=0). A try/except keeps any
probe bug from wedging the server (it just stops logging). Output is line-buffered
and PID-suffixed because the engine-core worker is SIGKILLed on shutdown (atexit may
not run) and several processes import this module.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
from typing import Any

_OUTPUT = os.environ.get(
    "REJECTRANK_OUTPUT",
    os.path.join(os.getcwd(), "rejectrank_records.jsonl"),
)

_PATCH_FLAG = "_int4_mtp_rejectrank_patch_applied"

_STATE: dict[str, Any] = {
    "fh": None,
    "path": None,
    "written": 0,
    "step": 0,
    "greedy_calls": 0,
    "nongreedy_calls": 0,
    "multi_req_steps": 0,
    "errors": 0,
    "vocab": None,
}


def _log(msg: str) -> None:
    print(f"[rejectrank] {msg}", file=sys.stderr, flush=True)


def _resolve_output_path() -> str:
    """Per-process output path ``{_OUTPUT}.{pid}`` (computed post-fork)."""
    return f"{_OUTPUT}.{os.getpid()}"


def _open_fh() -> Any:
    if _STATE["fh"] is None:
        path = _resolve_output_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _STATE["path"] = path
        # Line-buffered: the engine-core worker is SIGTERM/SIGKILLed on shutdown,
        # so userspace-buffered records would be lost.
        _STATE["fh"] = open(path, "w", buffering=1)
        _log(f"writing records to {path} (pid={os.getpid()})")
    return _STATE["fh"]


def _write_record(rec: dict[str, Any]) -> None:
    fh = _open_fh()
    fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    _STATE["written"] += 1


def _persist_meta(extra: dict[str, Any] | None = None) -> None:
    """Write the ``.meta.json`` sidecar. Called EAGERLY the moment vocab is first
    known (and again at exit), because the engine-core worker is SIGKILLed on
    shutdown -- atexit may not run, so a finalize-only meta would lose the vocab
    the offline FLy-normalization (h = H / log V) needs."""
    try:
        path = _STATE.get("path") or _resolve_output_path()
        rec = {"output": path, "pid": os.getpid(), "vocab": _STATE["vocab"]}
        if extra:
            rec.update(extra)
        with open(path + ".meta.json", "w") as f:
            json.dump(rec, f, indent=2)
    except Exception:  # noqa: BLE001
        pass


def _finalize() -> None:
    fh = _STATE["fh"]
    if fh is not None:
        try:
            fh.flush()
            fh.close()
        except Exception:  # noqa: BLE001
            pass
    summary = {
        k: _STATE[k]
        for k in (
            "written", "step", "greedy_calls", "nongreedy_calls",
            "multi_req_steps", "errors", "vocab",
        )
    }
    _log(f"finalize: {summary}")
    _persist_meta(summary)


def _record_forward(metadata: Any, logits: Any, sampling_metadata: Any) -> None:
    """Compute + emit per-draft-position reject-rank/entropy for one greedy verify."""
    import torch

    # Greedy-only: the challenge benchmark runs temp=0. all_greedy gates the exact
    # path whose accept criterion (draft == argmax) we are characterizing.
    if not getattr(sampling_metadata, "all_greedy", False):
        _STATE["nongreedy_calls"] += 1
        return
    _STATE["greedy_calls"] += 1

    draft = metadata.draft_token_ids                       # [num_tokens]
    idx = metadata.target_logits_indices                   # [num_tokens]
    cu = metadata.cu_num_draft_tokens                      # [batch]

    # Full-vocab verifier logits at each draft position (copy; float32 for a stable
    # softmax/entropy). Matches what the greedy sampler argmaxes over.
    tl = logits.index_select(0, idx).to(torch.float32)     # [num_tokens, vocab]
    if _STATE["vocab"] is None:
        _STATE["vocab"] = int(tl.shape[-1])
        _persist_meta()                                    # survive SIGKILL

    argmax = tl.argmax(dim=-1)                             # [num_tokens]
    draft_logit = tl.gather(1, draft.long().view(-1, 1)).squeeze(1)
    # rank = (# strictly-greater logits) + 1; rk==1 <=> draft is (tied for) argmax.
    rk = (tl > draft_logit.view(-1, 1)).sum(dim=-1) + 1    # [num_tokens] int
    acc = (draft == argmax)                               # exact shipped criterion
    probs = tl.softmax(dim=-1)
    dp = probs.gather(1, draft.long().view(-1, 1)).squeeze(1)
    ap = probs.max(dim=-1).values
    H = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)

    cu_l = cu.detach().to("cpu").tolist()
    acc_l = acc.detach().to("cpu").to(torch.int32).tolist()
    rk_l = rk.detach().to("cpu").to(torch.int32).tolist()
    H_l = H.detach().to("cpu").tolist()
    dp_l = dp.detach().to("cpu").tolist()
    ap_l = ap.detach().to("cpu").tolist()

    nreq = len(cu_l)
    if nreq > 1:
        _STATE["multi_req_steps"] += 1
    start = 0
    for i in range(nreq):
        end = int(cu_l[i])
        n = end - start
        if n > 0:
            _write_record({
                "s": _STATE["step"],
                "req": i,
                "n": n,
                "acc": acc_l[start:end],
                "rk": rk_l[start:end],
                "H": [round(x, 5) for x in H_l[start:end]],
                "dp": [round(x, 6) for x in dp_l[start:end]],
                "ap": [round(x, 6) for x in ap_l[start:end]],
            })
            _STATE["step"] += 1
        start = end


def apply(module) -> bool:
    """Wrap ``module.RejectionSampler.forward`` to log reject-rank/entropy.

    ``module`` is the imported ``vllm.v1.sample.rejection_sampler`` module. Idempotent
    across the multiple processes / repeated imports that touch the module. Returns
    True if applied, False if it was already present.
    """
    if getattr(module, _PATCH_FLAG, False):
        return False
    sampler_cls = module.RejectionSampler
    orig_forward = sampler_cls.forward

    def _forward_probe(self, metadata, draft_probs, logits, sampling_metadata):
        try:
            _record_forward(metadata, logits, sampling_metadata)
        except Exception as exc:  # noqa: BLE001
            if _STATE["errors"] < 10:
                _log(f"_record_forward failed: {exc!r}")
            _STATE["errors"] += 1
        return orig_forward(self, metadata, draft_probs, logits, sampling_metadata)

    sampler_cls.forward = _forward_probe
    setattr(module, _PATCH_FLAG, True)
    _STATE["_orig_forward"] = orig_forward
    atexit.register(_finalize)
    _log(
        f"installed RejectionSampler.forward reject-rank probe "
        f"(output={_OUTPUT}, pid={os.getpid()})"
    )
    return True
