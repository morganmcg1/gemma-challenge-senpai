"""Rank-coverage probe (PR #79): log drafter top-W ranks vs target argmax per depth.

ENV-GATED by ``RANKPROBE_ENABLE=1``. This module is dropped into a SCRATCH copy of
``submissions/fa2sw_precache_kenyan`` by ``scripts/profiler/rank_coverage.py``; the
served submission stays byte-identical. It measures

    rho_r = P(target greedy argmax == drafter rank-r token | drafter rank 1..r-1
             all missed),  conditioned on the TRUE greedy prefix (on-path)

on the deployed MTP K=7 drafter, to replace the borrowed EAGLE-3 rho=0.565 that is
the last unmeasured input to PR #76's +18.7% tree-verify TPS projection.

WHY THIS IS CONTRACT-SAFE (mirrors #76: only ADD logging, drafts byte-identical)
--------------------------------------------------------------------------------
The linear chain only ever exposes the drafter's rank-1 token. To read rank-2/3/4
we must look at the drafter's top-W candidate logits per draft depth. We do that on
a scratch copy, never on the served files, and we keep the emitted draft chain
byte-identical:

  * Force base_propose every step (``LOOPGRAPH_WARMUP_CALLS`` huge, set by the
    driver) so the onegraph CUDA graph never captures and
    ``Gemma4Proposer._greedy_sample`` runs eager in Python at every depth. The
    onegraph path is a pure execution-mode optimisation OF base_propose; the
    drafted tokens are identical either way (we are not measuring speed here).
  * Override ``_greedy_sample`` to (a) compute the drafter's top-W via the SAME
    sparse ``_select_and_score`` the deployed ``get_top_tokens`` uses, append it to
    a per-propose buffer, and (b) RETURN ``self.model.get_top_tokens(...)``
    unchanged -- i.e. the real deployed argmax -- so the draft chain is byte
    identical to production.
  * Wrap ``_dixie_fused_accept_prep`` (verify side) to pair the buffered per-depth
    top-W with the target greedy argmax + first-divergence position, emitting one
    JSONL record per decode step.

ALIGNMENT is self-checked every step: the drafter rank-1 token (top-W[:, 0]) MUST
equal the verified ``draft_token_ids``. Mismatches (e.g. stale proposals from
engine warmup/dummy runs) are counted and dropped, never trusted.

Composition: we register our own ``_TargetFinder(LOOPGRAPH_TARGET, ...)``. The
finder's ``_busy`` re-entrancy guard makes finders compose like middleware, so the
onegraph patch applies first and our ``_greedy_sample`` override layers on top.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
import threading
from collections import deque
from typing import Any

_ENABLED = os.environ.get("RANKPROBE_ENABLE") == "1"
_W = int(os.environ.get("RANKPROBE_W", "4"))
# PR #86: when set, additionally capture the drafter's top-4 softmax probabilities +
# predictive entropy per draft step, and the verifier's top-1 softmax probability +
# entropy at each target position. Off by default => byte-identical to the #79 probe.
_LOGITS = os.environ.get("RANKPROBE_LOGITS") == "1"
# entropy over the top-K sparse candidate softmax (robustness twin of the full-set H)
_HK = int(os.environ.get("RANKPROBE_HK", "64"))
_OUTPUT = os.environ.get(
    "RANKPROBE_OUTPUT",
    os.path.join(os.getcwd(), "rankprobe_records.jsonl"),
)
# Per-propose buffer of per-depth step dicts {"top":[w ids], "pd":[p1..p4]|None,
# "H":float|None, "H64":float|None}. One entry per draft depth.
_CURRENT: list[dict[str, Any]] = []
# FIFO of completed proposals (each a list of per-depth step dicts). At conc=1 the
# queue holds exactly one in-flight proposal; capped so a stuck pairing can't leak.
_QUEUE: "deque[list[dict[str, Any]]]" = deque(maxlen=64)
_LOCK = threading.Lock()
# PR #86 verifier side: flat per-target-position top-1 prob + entropy for the CURRENT
# rejection-sampler forward call. Set by the forward wrapper, read by _log_verify in
# the same (synchronous, conc=1) call. Aligned with the flat draft/target layout.
_VERIFIER: dict[str, Any] = {"vp1": None, "Hv": None}

_STATE: dict[str, Any] = {
    "fh": None,
    "path": None,
    "written": 0,
    "step": 0,
    "align_ok": 0,
    "align_bad": 0,
    "dropped_stale": 0,
    "no_proposal": 0,
    "errors": 0,
    "installed_proposer": False,
    "installed_verify": False,
}


def _log(msg: str) -> None:
    print(f"[rankprobe] {msg}", file=sys.stderr, flush=True)


def _resolve_output_path() -> str:
    """Per-process output path: ``{_OUTPUT}.{pid}``.

    vLLM runs this probe in several processes (API server, engine-core worker,
    short-lived resource probes). They all import the module and would otherwise
    open the SAME path with mode "w" and truncate/clobber each other. Only the
    worker that runs rejection sampling generates real records, but PID-suffixing
    keeps every process isolated (and supports TP>1 workers). Computed lazily so
    we pick up the post-fork child pid, not the importing parent's.
    """
    return f"{_OUTPUT}.{os.getpid()}"


def _open_fh() -> Any:
    if _STATE["fh"] is None:
        path = _resolve_output_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _STATE["path"] = path
        # LINE-buffered: each record is one '\n'-terminated line, so it is flushed
        # to the OS on every write. The engine-core worker is SIGTERM/SIGKILLed on
        # server shutdown (atexit does NOT run), so any userspace-buffered records
        # would be LOST -- which is exactly what produced the empty debug file.
        _STATE["fh"] = open(path, "w", buffering=1)
        _log(f"writing records to {path} (W={_W}, pid={os.getpid()})")
    return _STATE["fh"]


def _write_record(rec: dict[str, Any]) -> None:
    fh = _open_fh()
    fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    _STATE["written"] += 1


def _finalize() -> None:
    fh = _STATE["fh"]
    if fh is not None:
        try:
            fh.flush()
            fh.close()
        except Exception:  # noqa: BLE001
            pass
    summary = {k: _STATE[k] for k in (
        "written", "align_ok", "align_bad", "dropped_stale", "no_proposal", "errors",
    )}
    if "n_candidates" in _STATE:
        summary["n_candidates"] = _STATE["n_candidates"]
    summary["logits"] = _LOGITS
    summary["installed_verifier_prob"] = _STATE.get("installed_verifier_prob", False)
    _log(f"finalize: {summary}")
    try:
        path = _STATE.get("path") or _resolve_output_path()
        with open(path + ".meta.json", "w") as f:
            json.dump({"W": _W, "output": path, "pid": os.getpid(), **summary},
                      f, indent=2)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Proposer side: top-W per draft depth
# --------------------------------------------------------------------------- #
def _compute_proposal_step(proposer: Any, hidden_states: Any) -> list[dict[str, Any]] | None:
    """Per-row drafter step: top-W candidate token ids (+ PR #86 probs/entropy).

    ``_select_and_score`` (the submission's unsorted variant) returns
    ``(logits, selected)`` over the sparse centroid-masked candidate set, exactly
    what ``get_top_tokens`` argmaxes over. topk(W) of those logits, gathered through
    ``selected``, is the drafter's rank-1..W vocabulary tokens.

    Returns a list (one per row) of step dicts ``{"top": [w ids]}``; when
    ``RANKPROBE_LOGITS`` is set, each dict additionally carries ``pd`` (top-4 softmax
    probs p1>=p2>=p3>=p4), ``H`` (predictive entropy over the FULL sparse candidate
    set, nats), and ``H64`` (entropy over the top-K candidates). The centroid clusters
    partition the vocab, so the sparse set is duplicate-free and its softmax is the
    drafter's deployable distribution. Returns None on any failure.
    """
    try:
        model = proposer.model
        masked_emb = model.masked_embedding
        lm_head_weight = model._get_full_lm_head_weight()
        logits, selected = masked_emb._select_and_score(hidden_states, lm_head_weight)
        n_cand = int(logits.shape[-1])
        k = min(_W, n_cand)
        if not _LOGITS:
            _, topidx = logits.topk(k, dim=-1)
            top_tokens = selected.gather(-1, topidx).detach().to("cpu").tolist()
            return [{"top": t} for t in top_tokens]
        if not _STATE.get("logged_ncand"):
            _STATE["logged_ncand"] = True
            _STATE["n_candidates"] = n_cand
            _log(f"drafter sparse candidate set size n_cand={n_cand} "
                 f"(entropy ceiling log(n_cand)={__import__('math').log(n_cand):.3f} nats)")
        probs = logits.float().softmax(dim=-1)               # (rows, n_cand)
        ktop = min(max(_HK, _W, 4), n_cand)
        topp, topidx = probs.topk(ktop, dim=-1)              # sorted desc
        top_tokens = selected.gather(-1, topidx[:, :k])
        H = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)       # full sparse set
        H64 = -(topp * topp.clamp_min(1e-12).log()).sum(dim=-1)      # top-ktop
        tt = top_tokens.detach().to("cpu").tolist()
        p4 = topp[:, :4].detach().to("cpu").tolist()
        Hl = H.detach().to("cpu").tolist()
        H64l = H64.detach().to("cpu").tolist()
        out: list[dict[str, Any]] = []
        for r in range(len(tt)):
            out.append({
                "top": tt[r],
                "pd": [round(x, 6) for x in p4[r]],
                "H": round(Hl[r], 6),
                "H64": round(H64l[r], 6),
            })
        return out
    except Exception as exc:  # noqa: BLE001
        if _STATE["errors"] < 5:
            _log(f"_compute_proposal_step failed: {exc!r}")
        _STATE["errors"] += 1
        return None


def _install_proposer_patch(module: Any) -> None:
    proposer_cls = module.Gemma4Proposer
    orig_greedy_sample = proposer_cls._greedy_sample
    orig_propose = proposer_cls.propose

    def _greedy_sample_probe(self: Any, hidden_states: Any) -> Any:
        steps = _compute_proposal_step(self, hidden_states)
        if steps is not None:
            _CURRENT.extend(steps)  # one entry per row (conc=1 -> one row per depth)
        # Byte-identical deployed argmax: call the real drafter selection.
        return self.model.get_top_tokens(hidden_states)

    def _propose_probe(self: Any, *args: Any, **kwargs: Any) -> Any:
        _CURRENT.clear()
        out = orig_propose(self, *args, **kwargs)
        if _CURRENT:
            with _LOCK:
                _QUEUE.append(list(_CURRENT))
            _CURRENT.clear()
        return out

    def _noop_centroids(self: Any) -> None:
        # Skip centroids CUDA-graph capture: our _greedy_sample override never uses
        # them, and capturing would only waste load-time work.
        self._centroids_sizes = []

    proposer_cls._greedy_sample = _greedy_sample_probe
    proposer_cls.propose = _propose_probe
    proposer_cls._setup_centroids_cuda_graphs = _noop_centroids
    _STATE["installed_proposer"] = True
    _log("installed Gemma4Proposer rank-probe (_greedy_sample/propose/centroids)")
    # keep references alive
    _STATE["_orig_greedy_sample"] = orig_greedy_sample
    _STATE["_orig_propose"] = orig_propose


# --------------------------------------------------------------------------- #
# Verify side: pair buffered top-W with target argmax + first divergence
# --------------------------------------------------------------------------- #
def _to_list(t: Any) -> list[int]:
    return t.detach().to("cpu").tolist() if hasattr(t, "detach") else list(t)


def _rank_of(token: int, topw: list[int]) -> int:
    """1-indexed rank of ``token`` in ``topw`` (rank-1..W), or 0 if absent."""
    for i, tok in enumerate(topw):
        if tok == token:
            return i + 1
    return 0


def _log_verify(
    cu_num_draft_tokens: Any,
    draft_token_ids: Any,
    target_argmax: Any,
    bonus_token_ids: Any,
) -> None:
    try:
        cu = _to_list(cu_num_draft_tokens)
        draft = _to_list(draft_token_ids)
        targ = _to_list(target_argmax)
        bonus = _to_list(bonus_token_ids)
        nreq = len(cu)
        start = 0
        for i in range(nreq):
            end = int(cu[i])
            d_seg = draft[start:end]
            t_seg = targ[start:end]
            n = len(d_seg)
            start = end
            if n == 0:
                continue
            # Pop the matching proposal: its rank-1 (top[:,0]) must equal d_seg.
            proposal = _match_proposal(d_seg)
            if proposal is None:
                _STATE["no_proposal"] += 1
                continue
            topw_seg = proposal[:n]
            align = all(
                len(topw_seg[d]["top"]) > 0 and topw_seg[d]["top"][0] == d_seg[d]
                for d in range(n)
            )
            if align:
                _STATE["align_ok"] += 1
            else:
                _STATE["align_bad"] += 1
            # first divergence: first depth where draft != target argmax.
            fd = n
            for d in range(n):
                if d_seg[d] != t_seg[d]:
                    fd = d
                    break
            # hit_rank over on-path depths 0..min(fd, n-1): rank of TRUE token in
            # the drafter top-W. d < fd -> should be 1; d == fd -> the rescue rank.
            last = fd if fd < n else n - 1
            hr = [_rank_of(t_seg[d], topw_seg[d]["top"]) for d in range(last + 1)]
            rec = {
                "i": _STATE["step"],
                "req": i,
                "n": n,
                "fd": fd,
                "all_acc": fd == n,
                "hr": hr,
                "align": align,
            }
            if fd < n:
                rec["top_fd"] = topw_seg[fd]["top"]
                rec["targ_fd"] = t_seg[fd]
                rec["draft_fd"] = d_seg[fd]
                rec["rank_fd"] = hr[fd]
                if _LOGITS:
                    step = topw_seg[fd]
                    rec["pd"] = step.get("pd")        # drafter top-4 softmax probs
                    rec["Hd"] = step.get("H")         # drafter entropy (full sparse set)
                    rec["Hd64"] = step.get("H64")     # drafter entropy (top-K)
                    rec["salv2"] = 1 if hr[fd] == 2 else 0  # rank-2 salvage indicator
                    vp1, hv = _verifier_at(end - n, fd)
                    rec["vp1"] = vp1                  # verifier top-1 softmax prob
                    rec["Hv"] = hv                    # verifier entropy (full vocab)
            _write_record(rec)
            _STATE["step"] += 1
        if bonus and _STATE["step"] <= 1:
            pass  # bonus retained in signature for completeness; not needed for rho
    except Exception as exc:  # noqa: BLE001
        if _STATE["errors"] < 10:
            _log(f"_log_verify failed: {exc!r}")
        _STATE["errors"] += 1


def _match_proposal(d_seg: list[int]) -> list[dict[str, Any]] | None:
    """Pop the queued proposal whose rank-1 chain matches the verified draft.

    Discards leading stale proposals (engine warmup/dummy runs that proposed
    without a paired verify). Returns None if no match remains.
    """
    with _LOCK:
        tries = 0
        while _QUEUE and tries < len(_QUEUE) + 1:
            cand = _QUEUE.popleft()
            tries += 1
            n = len(d_seg)
            if len(cand) >= n and all(
                len(cand[d]["top"]) > 0 and cand[d]["top"][0] == d_seg[d]
                for d in range(n)
            ):
                return cand
            _STATE["dropped_stale"] += 1
        return None


def _verifier_at(seg_start: int, pos: int) -> tuple[float | None, float | None]:
    """Verifier (target) top-1 prob + full-vocab entropy at flat index seg_start+pos.

    ``_VERIFIER`` is the flat per-target-position capture from the rejection-sampler
    forward of the SAME decode step (synchronous, conc=1), indexed by
    ``target_logits_indices`` -- the identical flat layout ``_log_verify`` slices.
    """
    vp1 = _VERIFIER.get("vp1")
    if vp1 is None:
        return None, None
    hv = _VERIFIER.get("Hv")
    idx = seg_start + pos
    if 0 <= idx < len(vp1):
        h = hv[idx] if (hv is not None and idx < len(hv)) else None
        return vp1[idx], h
    return None, None


def _install_verifier_prob_patch(module: Any) -> None:
    """Wrap ``RejectionSampler.forward`` to capture per-target-position verifier top-1
    softmax prob + full-vocab entropy (PR #86).

    Reads the FULL-vocab target logits (``logits[target_logits_indices]``) BEFORE the
    SMP-02 argmax. Non-destructive: advanced-index makes a copy, softmax allocates a
    new tensor, ``logits`` is never written -- so the emitted tokens are byte-identical
    to production. Gated by ``RANKPROBE_LOGITS``; off => this finder is never installed.
    """
    sampler_cls = module.RejectionSampler
    orig_forward = sampler_cls.forward

    def _forward_probe(self: Any, metadata: Any, draft_probs: Any, logits: Any,
                       sampling_metadata: Any) -> Any:
        try:
            idx = metadata.target_logits_indices
            tl = logits.index_select(0, idx).float()       # (num_targets, vocab)
            tp = tl.softmax(dim=-1)
            vp1 = tp.max(dim=-1).values
            hv = -(tp * tp.clamp_min(1e-12).log()).sum(dim=-1)
            _VERIFIER["vp1"] = vp1.detach().to("cpu").tolist()
            _VERIFIER["Hv"] = hv.detach().to("cpu").tolist()
        except Exception as exc:  # noqa: BLE001
            _VERIFIER["vp1"] = None
            _VERIFIER["Hv"] = None
            if _STATE["errors"] < 5:
                _log(f"verifier-prob capture failed: {exc!r}")
            _STATE["errors"] += 1
        return orig_forward(self, metadata, draft_probs, logits, sampling_metadata)

    sampler_cls.forward = _forward_probe
    _STATE["installed_verifier_prob"] = True
    _STATE["_orig_forward"] = orig_forward
    _log("installed RejectionSampler.forward verifier-prob wrapper")


def _install_verify_patch() -> None:
    import sitecustomize as sc

    orig = sc._dixie_fused_accept_prep

    def _wrapped(
        output_token_ids: Any,
        cu_num_draft_tokens: Any,
        draft_token_ids: Any,
        target_argmax: Any,
        bonus_token_ids: Any,
        max_spec_len: int,
    ) -> bool:
        _log_verify(cu_num_draft_tokens, draft_token_ids, target_argmax, bonus_token_ids)
        return orig(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            max_spec_len,
        )

    sc._dixie_fused_accept_prep = _wrapped
    _STATE["installed_verify"] = True
    _log("installed _dixie_fused_accept_prep verify wrapper")


# --------------------------------------------------------------------------- #
# Install
# --------------------------------------------------------------------------- #
def _install() -> None:
    import sitecustomize as sc

    # Compose with the existing loopgraph finder (the _busy guard lets the onegraph
    # patch run first, then ours).
    sys.meta_path.insert(
        0, sc._TargetFinder(sc.LOOPGRAPH_TARGET, _install_proposer_patch)
    )
    _install_verify_patch()
    if _LOGITS:
        # PR #86: capture verifier top-1 prob/entropy by wrapping the rejection
        # sampler's forward (fires on import, after serve.py's source-level SMP-02
        # patch is already on disk, so we wrap the SMP-02-enabled forward).
        sys.meta_path.insert(
            0, sc._TargetFinder(
                "vllm.v1.sample.rejection_sampler", _install_verifier_prob_patch)
        )
    atexit.register(_finalize)
    _log(
        f"armed (W={_W}, logits={_LOGITS}, output={_OUTPUT}); "
        f"expects LOOPGRAPH_WARMUP_CALLS huge to force eager base_propose"
    )


if _ENABLED:
    try:
        _install()
    except Exception as exc:  # noqa: BLE001
        _log(f"install FAILED, probe inert: {exc!r}")
else:
    _log("RANKPROBE_ENABLE != 1 -> inert")
