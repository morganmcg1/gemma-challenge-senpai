"""First-reject capture probe (PR #89): per-step MTP accept length, position-aligned.

ENV-GATED by ``FRPROBE_ENABLE=1``. Dropped into a SCRATCH copy of
``submissions/fa2sw_precache_kenyan`` by ``scripts/profiler/firstreject_capture.py``;
the served submission stays byte-identical.

WHAT IT RECORDS
---------------
One JSONL record per decode step (per request in the batch), on the DEPLOYED
spec-decode verify path:

    {"s": <global step>, "req": <req-in-batch>, "n": <num draft tokens>,
     "fd": <first-divergence depth = MTP accept length m>, "emit": [<token ids>]}

``fd`` is the first depth where the drafted token != the target greedy argmax, i.e.
the MTP chain accept length ``m`` (``fd == 0`` is the first-reject / m=0 miss that PR
#89 intersects with prompt-lookup hits). ``emit`` is the exact token sequence this
step contributed to the output (``fd`` accepted tokens + 1 correction/bonus =
``fd + 1`` tokens), reconstructed from ``target_argmax`` + ``bonus_token_ids`` so the
offline aligner can pin every step to an absolute generation position by matching the
concatenated emit-stream against the greedy completion (token-identical contract).

WHY THIS IS CONTRACT-SAFE (mirrors #79: only ADD logging, drafts byte-identical)
--------------------------------------------------------------------------------
We wrap ``sitecustomize._dixie_fused_accept_prep`` (the verify accept function),
call the original unchanged, and return its result. We do NOT force eager / disable
the onegraph drafter (unlike #79, which needed eager to expose top-W ranks): #89 only
reads ``fd`` and the emitted tokens off the verify side, which are identical whether
the drafter runs as a CUDA graph or eager. So the capture runs at full deployed speed
and is serve-faithful. No proposer override, no token mutation.

EMIT RECONSTRUCTION (matches the Triton accept kernel semantics)
----------------------------------------------------------------
The deployed ``_dixie_fused_accept_prep_kernel`` emits, per request: the accepted
draft tokens (which equal ``target_argmax`` at depths < fd) then one correction token
-- ``target_argmax[fd]`` if rejected (fd < n), else ``bonus_token_ids[req]`` (full
accept, fd == n). So:

    emit = target_argmax[0:fd] + [target_argmax[fd] if fd < n else bonus[req]]

length == fd + 1 in both cases. Using ``target_argmax`` for the accepted prefix is
exact because acceptance is defined as draft == target_argmax there.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
from typing import Any

_ENABLED = os.environ.get("FRPROBE_ENABLE") == "1"
_OUTPUT = os.environ.get(
    "FRPROBE_OUTPUT",
    os.path.join(os.getcwd(), "firstreject_records.jsonl"),
)

_STATE: dict[str, Any] = {
    "fh": None,
    "path": None,
    "written": 0,
    "step": 0,
    "multi_req_steps": 0,
    "errors": 0,
    "installed_verify": False,
}


def _log(msg: str) -> None:
    print(f"[frprobe] {msg}", file=sys.stderr, flush=True)


def _resolve_output_path() -> str:
    """Per-process output path: ``{_OUTPUT}.{pid}``.

    vLLM runs this probe in several processes (API server, engine-core worker, short
    resource probes). They all import the module; PID-suffixing keeps every process
    isolated so they cannot truncate/clobber a shared path. Only the worker running
    rejection sampling emits real records. Computed lazily to pick up the post-fork
    child pid, not the importing parent's.
    """
    return f"{_OUTPUT}.{os.getpid()}"


def _open_fh() -> Any:
    if _STATE["fh"] is None:
        path = _resolve_output_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _STATE["path"] = path
        # LINE-buffered: the engine-core worker is SIGTERM/SIGKILLed on shutdown
        # (atexit may not run), so userspace-buffered records would be lost.
        _STATE["fh"] = open(path, "w", buffering=1)
        _log(f"writing records to {path} (pid={os.getpid()})")
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
    summary = {k: _STATE[k] for k in ("written", "step", "multi_req_steps", "errors")}
    _log(f"finalize: {summary}")
    try:
        path = _STATE.get("path") or _resolve_output_path()
        with open(path + ".meta.json", "w") as f:
            json.dump({"output": path, "pid": os.getpid(), **summary}, f, indent=2)
    except Exception:  # noqa: BLE001
        pass


def _to_list(t: Any) -> list[int]:
    return t.detach().to("cpu").tolist() if hasattr(t, "detach") else list(t)


def _log_verify(
    cu_num_draft_tokens: Any,
    draft_token_ids: Any,
    target_argmax: Any,
    bonus_token_ids: Any,
) -> None:
    try:
        cu = _to_list(cu_num_draft_tokens)
        targ = _to_list(target_argmax)
        draft = _to_list(draft_token_ids)
        bonus = _to_list(bonus_token_ids)
        nreq = len(cu)
        if nreq > 1:
            _STATE["multi_req_steps"] += 1
        start = 0
        for i in range(nreq):
            end = int(cu[i])
            t_seg = targ[start:end]
            d_seg = draft[start:end]
            n = len(t_seg)
            start = end
            if n == 0:
                continue
            # first divergence: first depth where draft != target argmax (== accept len m)
            fd = n
            for d in range(n):
                if d_seg[d] != t_seg[d]:
                    fd = d
                    break
            # exact emitted tokens this step: fd accepted (= target argmax) + 1 correction
            if fd < n:
                emit = t_seg[: fd + 1]                       # ...,target_argmax[fd] (correction)
            else:
                # bonus_token_ids is 2D ([nreq, 1]); flatten to the scalar bonus id.
                raw = bonus[i] if i < len(bonus) else (bonus[0] if bonus else None)
                while isinstance(raw, list):
                    raw = raw[0] if raw else None
                emit = t_seg[:n] + ([raw] if raw is not None else [])
            _write_record({
                "s": _STATE["step"],
                "req": i,
                "n": n,
                "fd": fd,
                "emit": emit,
            })
            _STATE["step"] += 1
    except Exception as exc:  # noqa: BLE001
        if _STATE["errors"] < 10:
            _log(f"_log_verify failed: {exc!r}")
        _STATE["errors"] += 1


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
    _log("installed _dixie_fused_accept_prep verify wrapper (deployed spec-ON path)")


def _install() -> None:
    _install_verify_patch()
    atexit.register(_finalize)
    _log(f"armed (output={_OUTPUT}); deployed onegraph drafter unchanged")


if _ENABLED:
    try:
        _install()
    except Exception as exc:  # noqa: BLE001
        _log(f"install FAILED, probe inert: {exc!r}")
else:
    _log("FRPROBE_ENABLE != 1 -> inert")
