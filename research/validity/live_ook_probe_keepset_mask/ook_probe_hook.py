#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""READ-ONLY live OOK probe hook for the deployed vLLM MTP proposer (PR #426, land).

This module is injected into the LOCAL served `fa2sw_precache_kenyan` run via a `.pth`
bootstrap (see live_ook_probe_keepset_mask.py). It is a PURE-LOGGING instrumentation hook:
it wraps the (already sitecustomize-patched) `Gemma4Proposer.propose` to record, per draft
position, the drafter's argmax token id and whether it falls in the 16384 keepset. It NEVER
alters the proposed or emitted tokens -> greedy identity + PPL preserved by construction
(the truncated-head verify is the sole arbiter of emitted tokens; the drafter only proposes).

WHY THE finder-wrap: the deployed submission patches `Gemma4Proposer.propose` via a claiming
`_TargetFinder` meta-path finder registered inside its `sitecustomize.py`. A second claiming
finder for the same module would never fire. So instead we install a finder for the
`sitecustomize` MODULE itself; after the real sitecustomize loads (registering its finders),
we wrap, IN PLACE, the `_patch_fn` of the finder targeting `vllm.v1.spec_decode.gemma4` so our
logging installer runs AFTER the deployed propose patch is applied. Works in every process that
loads this venv (api_server + EngineCore worker), so the proposer is captured wherever it runs.

ENV (all default-off -> inert unless explicitly enabled by the driver):
  OOK_PROBE_LOG      path to the JSONL log to append per-step records (REQUIRED to activate)
  OOK_PROBE_KEEPSET  path to pck04_keepset.json (default /tmp/osoi5-v0-baked/pck04_keepset.json)
  OOK_PROBE_TAG      a free-text tag stamped on every record (e.g. which prompt-set is running)

The per-step JSONL record (single MAX_NUM_SEQS=1 stream) carries:
  draft_ids   : list[int]   the K per-position drafter argmax ids (post FUSED_SPARSE_ARGMAX)
  n_draft     : int         len(draft_ids) == K
  n_ook       : int         how many of draft_ids fall OUTSIDE the keepset
  ook_ids     : list[int]   the actual out-of-keepset ids (for distinct-id accounting)
  num_rejected: list[int]   per-req rejected-token count from the PREVIOUS verify (ladder source)
  tag, pid    : provenance
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import sys

_LOG_PATH = os.environ.get("OOK_PROBE_LOG")

# Fully inert (not even a finder installed) unless the driver set OOK_PROBE_LOG.
if _LOG_PATH:
    _KEEPSET_PATH = os.environ.get("OOK_PROBE_KEEPSET", "/tmp/osoi5-v0-baked/pck04_keepset.json")
    _TAG = os.environ.get("OOK_PROBE_TAG", "")
    _LOOPGRAPH_TARGET = "vllm.v1.spec_decode.gemma4"

    _keepset: frozenset[int] | None = None
    _logf = None
    _log_open_failed = False
    _first_write_logged = False

    def _load_keepset() -> frozenset[int]:
        global _keepset
        if _keepset is None:
            data = json.loads(open(_KEEPSET_PATH).read())
            ids = data.get("keep_ids") or data.get("kept_ids") or []
            _keepset = frozenset(int(i) for i in ids)
        return _keepset

    def _log(rec: dict) -> None:
        global _logf, _log_open_failed, _first_write_logged
        if _logf is None:
            # mkdir-then-open so a stale/missing parent never drops records; line-buffered
            # append (single MAX_NUM_SEQS=1 writer per process). Surface an open failure ONCE
            # to stderr (serve.log) instead of silently swallowing it -- the original zero-record
            # bug was a relative OOK_PROBE_LOG opened under the served worker's cwd=submission_dir.
            try:
                parent = os.path.dirname(_LOG_PATH)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                _logf = open(_LOG_PATH, "a", buffering=1)
            except Exception as exc:
                if not _log_open_failed:
                    _log_open_failed = True
                    print(f"[ook-probe] LOG OPEN FAILED path={_LOG_PATH!r} "
                          f"cwd={os.getcwd()!r}: {exc!r}", file=sys.stderr, flush=True)
                raise
        _logf.write(json.dumps(rec) + "\n")
        if not _first_write_logged:
            _first_write_logged = True
            print(f"[ook-probe] first record written -> {_LOG_PATH} (pid {os.getpid()})",
                  file=sys.stderr, flush=True)

    def _install_propose_logger(module) -> None:
        """Wrap the (already deployed-patched) Gemma4Proposer.propose with a logger."""
        import torch

        proposer_cls = module.Gemma4Proposer
        base_propose = proposer_cls.propose
        if getattr(base_propose, "_ook_logged", False):
            return
        keep = _load_keepset()

        def logged_propose(self, *args, **kwargs):
            result = base_propose(self, *args, **kwargs)
            try:
                # result is the [num_reqs, K] int64 draft-token tensor (post fused-sparse-argmax).
                if torch.is_tensor(result):
                    ids = [int(t) for t in result.detach().to("cpu").reshape(-1).tolist()]
                    ook = [t for t in ids if t not in keep]
                    rec = {
                        "tag": _TAG,
                        "pid": os.getpid(),
                        "draft_ids": ids,
                        "n_draft": len(ids),
                        "n_ook": len(ook),
                        "ook_ids": ook,
                    }
                    nrej = kwargs.get("num_rejected_tokens_gpu")
                    if nrej is None and args:
                        # tolerate positional pass (kw in deployed runner, but be defensive)
                        for a in args:
                            if torch.is_tensor(a) and a.dtype in (torch.int32, torch.int64) and a.dim() == 1:
                                nrej = a
                    if torch.is_tensor(nrej):
                        rec["num_rejected"] = [int(x) for x in nrej.detach().to("cpu").reshape(-1).tolist()]
                    _log(rec)
            except Exception as exc:  # never break the engine on a logging error
                try:
                    _log({"tag": _TAG, "pid": os.getpid(), "error": repr(exc)})
                except Exception:
                    pass
            return result

        logged_propose._ook_logged = True  # type: ignore[attr-defined]
        proposer_cls.propose = logged_propose
        print(f"[ook-probe] installed propose logger pid {os.getpid()} "
              f"keepset={len(keep)} log={_LOG_PATH}", file=sys.stderr, flush=True)

    class _ChainLoader(importlib.abc.Loader):
        """Run the real sitecustomize, then wrap its gemma4 finder's _patch_fn."""

        def __init__(self, inner) -> None:
            self._inner = inner

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module) -> None:
            self._inner.exec_module(module)  # real sitecustomize registers its finders
            try:
                self._wrap_gemma4_finder()
            except Exception as exc:
                print(f"[ook-probe] finder-wrap failed: {exc!r}", file=sys.stderr, flush=True)

        @staticmethod
        def _wrap_gemma4_finder() -> None:
            for finder in list(sys.meta_path):
                if getattr(finder, "_target", None) != _LOOPGRAPH_TARGET:
                    continue
                orig = getattr(finder, "_patch_fn", None)
                if orig is None or getattr(orig, "_ook_chained", False):
                    continue

                def chained(mod, _orig=orig):
                    _orig(mod)                    # apply the deployed propose patch first
                    _install_propose_logger(mod)  # then layer read-only logging

                chained._ook_chained = True       # type: ignore[attr-defined]
                finder._patch_fn = chained
                print(f"[ook-probe] wrapped sitecustomize finder for {_LOOPGRAPH_TARGET} "
                      f"pid {os.getpid()}", file=sys.stderr, flush=True)

    class _SitecustomizeFinder(importlib.abc.MetaPathFinder):
        _busy = False

        def find_spec(self, fullname, path=None, target=None):
            if fullname != "sitecustomize" or _SitecustomizeFinder._busy:
                return None
            _SitecustomizeFinder._busy = True
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                _SitecustomizeFinder._busy = False
            if spec is None or spec.loader is None:
                return None
            spec.loader = _ChainLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, _SitecustomizeFinder())
    print(f"[ook-probe] armed sitecustomize finder pid {os.getpid()} (log={_LOG_PATH})",
          file=sys.stderr, flush=True)
