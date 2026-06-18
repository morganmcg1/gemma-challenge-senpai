"""Env-gated recompute-acceptor patch — PR #642 cost probe + PR #663 REAL acceptor.

Both modes fire width-1 ``w4a16-ct`` *target* forwards (``_dummy_run`` with
``slot_mapping=-1`` so the injected forward never writes the live KV cache) into the
served spec-decode loop AFTER each verify step. The emitted token stream is left
UNCHANGED in both modes: greedy identity of the rescued stream is certified
separately, offline, along the served M=1 AR trajectory (``served_identity_scan.py``)
— the live verify numerics are not recoverable from the API — so here we only measure
the *cost* and the *realized firing rate*. ``wall_tps = completion_tokens/decode_s``
therefore reflects ``(un-rescued spec throughput) − (recompute overhead)``.

Modes (no-op unless one env is set; the shipped serve path is byte-identical when
neither is):

  * RATE mode  — ``SENPAI_RECOMPUTE_RATE=r`` (r>0).  #642 cost probe. After each step
    fire ``floor(carry + r*emitted)`` forwards (a fractional carry makes the long-run
    total exactly ``r*total_emitted``). ``r=0`` is the un-rescued ceiling; a small
    sweep fits the additive in-loop marginal cost C.

  * TAU mode   — ``SENPAI_ACCEPTOR_TAU=t`` (t>0).  #663 REAL gap-flag acceptor. Read
    the live verify logit gaps (top1−top2) at the spec *target* positions and fire one
    width-1 recompute per position whose gap ``< t``. The firing rate is therefore
    DATA-DRIVEN — it emerges from the real gap distribution, not from an injected rate
    — which is exactly the real gap-flag M=1-recompute acceptor #663 asks for. The
    realized rate (flagged/positions) is logged so it can be checked against the
    offline scan's ``flag_trigger_rate``.

Cudagraph dispatch is RESOLVED empirically (settles #642's [eager,captured]
bracket by direct observation, not assumption):

  * ``SENPAI_RECOMPUTE_CUDAGRAPH=1`` → ``cudagraph_runtime_mode=None``: the dispatcher
    picks the runtime mode, replaying a captured graph iff a key matches (captured arm).
  * unset → ``CUDAGraphMode.NONE``: forced eager — every injected forward breaks the
    captured decode graph exactly as a real interleaved recompute would (eager floor).

On the first fire we dump ``cudagraph_dispatcher.cudagraph_keys`` and the result of
``dispatch(1, uniform_decode=True)`` — the mode ``_dummy_run(1, uniform_decode=True)``
itself resolves to — so whether a width-1 recompute *replays a captured size-1 decode
graph* in this spec stack is answered by the live dispatcher, not guessed.

A non-firing patch fails loud: it logs the realized rate periodically and (unless
``SENPAI_RECOMPUTE_REQUIRE_FIRE=0``) asserts that real forwards fired, so a silent
no-op can never again be mistaken for "zero overhead".
"""

from __future__ import annotations

import json
import os

_PATCH_FLAG = "_optionb_recompute_acceptor_patch_applied"

_RATE_ENV = "SENPAI_RECOMPUTE_RATE"
_TAU_ENV = "SENPAI_ACCEPTOR_TAU"
_TARGET_ONLY_ENV = "SENPAI_RECOMPUTE_TARGET_ONLY"
_REQUIRE_FIRE_ENV = "SENPAI_RECOMPUTE_REQUIRE_FIRE"
_CUDAGRAPH_ENV = "SENPAI_RECOMPUTE_CUDAGRAPH"
_STAT_DIR_ENV = "SENPAI_RECOMPUTE_STAT_DIR"

_LOG_EVERY = int(os.environ.get("SENPAI_RECOMPUTE_LOG_EVERY", "4096"))


def _rate() -> float:
    try:
        return float(os.environ.get(_RATE_ENV, "0") or "0")
    except (TypeError, ValueError):
        return 0.0


def _tau() -> float:
    try:
        return float(os.environ.get(_TAU_ENV, "0") or "0")
    except (TypeError, ValueError):
        return 0.0


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("", "0", "false", "no", "off")


def _write_stat(kind: str, payload: dict) -> None:
    """Atomically dump a sidecar stat JSON to ``$SENPAI_RECOMPUTE_STAT_DIR``.

    No-op when the env var is unset (the shipped serving path never writes)."""
    stat_dir = os.environ.get(_STAT_DIR_ENV, "")
    if not stat_dir:
        return
    try:
        os.makedirs(stat_dir, exist_ok=True)
        pid = os.getpid()
        path = os.path.join(stat_dir, f"recompute_{kind}_{pid}.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"pid": pid, **payload}, fh)
        os.replace(tmp, path)
    except Exception:
        pass


def _count_emitted(output, torch) -> int:
    """Per-step emitted-token count, robust to both served output shapes.

    * sync ``ModelRunnerOutput.sampled_token_ids`` is ``list[list[int]]`` (one inner
      list per request; spec decode emits 1..M accepted ids).
    * async ``AsyncGPUModelRunnerOutput`` does NOT expose a public
      ``sampled_token_ids`` until ``get_output()`` runs later; the device tensor lives
      on ``_sampled_token_ids`` ``[num_reqs, max_spec+1]`` padded with ``-1`` at
      rejected positions. Read it directly so emitted advances (and so RATE-mode
      firing, which keys off ``rate*emitted``, is not a silent no-op on async outputs).
    Returns 0 for the PP/kv-only early-return outputs (``None``/empty)."""
    stids = getattr(output, "sampled_token_ids", None)
    if stids is None:
        stids = getattr(output, "_sampled_token_ids", None)
    if stids is None:
        return 0
    if isinstance(stids, torch.Tensor):
        if stids.numel() == 0:
            return 0
        return int((stids >= 0).sum().item())
    try:
        return int(sum(len(x) for x in stids))
    except TypeError:
        return 0


def _gap_flag_count(self, ems, tau: float, torch) -> tuple[int, int]:
    """Read the live verify-logit gaps at the spec target positions and count how
    many fall below ``tau``. Returns ``(flagged, positions)``.

    ``ems`` is the unpacked ``execute_model_state`` 10-tuple set just before
    ``sample_tokens`` consumes it: ``ems[1]`` is the target ``logits``
    ``[total_scheduled, vocab]`` and ``ems[2]`` is the ``SpecDecodeMetadata`` whose
    ``target_logits_indices`` select the draft-aligned verify rows. The gap is
    ``top1 − top2`` of each verify row — the same statistic the offline identity scan
    flags on, so the realized rate here is directly comparable to the scan's
    ``flag_trigger_rate``."""
    try:
        logits = ems[1]
        sdm = ems[2]
        if logits is None or sdm is None:
            return 0, 0
        tli = getattr(sdm, "target_logits_indices", None)
        if tli is None or tli.numel() == 0:
            return 0, 0
        verify = logits.index_select(0, tli)
        top2 = torch.topk(verify, 2, dim=-1).values
        gaps = top2[:, 0] - top2[:, 1]
        positions = int(gaps.numel())
        flagged = int((gaps < tau).sum().item())
        return flagged, positions
    except Exception:
        return 0, 0


def apply(gmr) -> bool:
    """Wrap ``gmr.GPUModelRunner.sample_tokens`` to fire recompute forwards.

    ``gmr`` is the live ``vllm.v1.worker.gpu_model_runner`` module; every free symbol
    is resolved from it so we never guess import paths against a vLLM version. Returns
    True if applied, False if already present or if neither mode is enabled."""
    GPUModelRunner = gmr.GPUModelRunner
    if getattr(GPUModelRunner, _PATCH_FLAG, False):
        return False

    rate = _rate()
    tau = _tau()
    if rate <= 0.0 and tau <= 0.0:
        # No-op: shipped serve path. Do not wrap.
        return False

    torch = gmr.torch
    CUDAGraphMode = gmr.CUDAGraphMode
    logger = gmr.init_logger("int4_mtp_drafter.recompute_acceptor")

    target_only = _flag(_TARGET_ONLY_ENV, True)
    require_fire = _flag(_REQUIRE_FIRE_ENV, True)
    use_cudagraph = _flag(_CUDAGRAPH_ENV, False)
    mode_name = "TAU(real-gap)" if tau > 0.0 else "RATE(inject)"

    logger.warning(
        "[recompute-acceptor] APPLY wrapping GPUModelRunner.sample_tokens mode=%s "
        "rate=%.6f tau=%.6f target_only=%s require_fire=%s cudagraph=%s (mode=%s) "
        "log_every=%d",
        mode_name, rate, tau, target_only, require_fire, use_cudagraph,
        "None/dispatcher" if use_cudagraph else "NONE/eager", _LOG_EVERY,
    )

    orig_sample_tokens = GPUModelRunner.sample_tokens
    recompute_cg_mode = None if use_cudagraph else CUDAGraphMode.NONE

    def _resolve_dispatch(self) -> None:
        """Dump the live cudagraph dispatch for a width-1 uniform-decode forward —
        the empirical answer to whether the recompute replays a captured graph."""
        try:
            disp = self.cudagraph_dispatcher
            dmode, bd = disp.dispatch(1, uniform_decode=True)
            keys = {
                str(k): sorted(str(d) for d in v)
                for k, v in disp.cudagraph_keys.items()
            }
            cc = self.compilation_config
            payload = {
                "dispatch_1_uniform_decode_mode": str(dmode),
                "dispatch_1_uniform_decode_batch_desc": str(bd),
                "is_captured_replay": str(dmode) != str(CUDAGraphMode.NONE),
                "use_cudagraph_env": use_cudagraph,
                "cudagraph_keys": keys,
                "cudagraph_capture_sizes": list(cc.cudagraph_capture_sizes or []),
                "cudagraph_mode": str(cc.cudagraph_mode),
                "max_cudagraph_capture_size": getattr(
                    cc, "max_cudagraph_capture_size", None
                ),
                "uniform_decode_query_len": getattr(
                    self, "uniform_decode_query_len", None
                ),
                "num_spec_tokens": getattr(self, "num_spec_tokens", None),
            }
            logger.warning(
                "[recompute-acceptor] DISPATCH RESOLVE: dispatch(1,uniform_decode="
                "True) -> mode=%s captured_replay=%s | capture_sizes=%s "
                "uniform_decode_query_len=%s num_spec_tokens=%s | FULL_keys=%s",
                dmode, payload["is_captured_replay"],
                payload["cudagraph_capture_sizes"],
                payload["uniform_decode_query_len"], payload["num_spec_tokens"],
                keys.get(str(CUDAGraphMode.FULL), []),
            )
            _write_stat("dispatch_resolve", payload)
        except Exception:
            logger.exception("[recompute-acceptor] dispatch resolve failed")

    def _fire(self, n: int) -> int:
        if n <= 0:
            return 0
        fired = 0
        with torch.inference_mode():
            for _ in range(n):
                self._dummy_run(
                    1,
                    cudagraph_runtime_mode=recompute_cg_mode,
                    uniform_decode=True,
                    allow_microbatching=False,
                    skip_eplb=True,
                )
                fired += 1
        return fired

    def sample_tokens(self, grammar_output):
        # Peek the ephemeral verify state BEFORE the original consumes/clears it.
        ems = self.execute_model_state
        flagged = positions = 0
        if tau > 0.0 and ems is not None:
            flagged, positions = _gap_flag_count(self, ems, tau, torch)

        output = orig_sample_tokens(self, grammar_output)

        n_emitted = _count_emitted(output, torch)

        # First real spec step: resolve the cudagraph dispatch + first-call diag.
        if not getattr(self, "_recompute_diag_done", False) and (
            n_emitted > 0 or positions > 0
        ):
            self._recompute_diag_done = True
            _resolve_dispatch(self)
            logger.warning(
                "[recompute-acceptor] FIRST-CALL output_type=%s n_emitted=%d "
                "positions=%d flagged=%d rate=%.6f tau=%.6f",
                type(output).__name__, n_emitted, positions, flagged, rate, tau,
            )
            _write_stat(
                "firstcall",
                {
                    "output_type": type(output).__name__,
                    "n_emitted": n_emitted,
                    "positions": positions,
                    "flagged": flagged,
                    "rate": rate,
                    "tau": tau,
                },
            )

        # Decide how many recompute forwards to fire this step.
        if tau > 0.0:
            n_fire = flagged
        else:
            acc = getattr(self, "_recompute_acc", 0.0) + rate * n_emitted
            n_fire = int(acc)
            self._recompute_acc = acc - n_fire

        # Accumulators.
        self._recompute_emitted_total = (
            getattr(self, "_recompute_emitted_total", 0) + n_emitted
        )
        self._recompute_positions_total = (
            getattr(self, "_recompute_positions_total", 0) + positions
        )
        self._recompute_flagged_total = (
            getattr(self, "_recompute_flagged_total", 0) + flagged
        )

        if n_fire > 0:
            self._recompute_fired_total = getattr(
                self, "_recompute_fired_total", 0
            ) + _fire(self, n_fire)

        # Periodic progress log + sidecar stat. Gate on whichever counter advances
        # for the active mode: TAU mode reads ``positions`` every step (a robust,
        # sync-free per-step count) while ``emitted`` may lag on async outputs; RATE
        # mode keys off ``emitted``. ``max`` keeps logging alive if either is stuck.
        last = getattr(self, "_recompute_last_log", 0)
        emitted_total = self._recompute_emitted_total
        pos_total = self._recompute_positions_total
        progress_counter = max(emitted_total, pos_total)
        if progress_counter - last >= _LOG_EVERY:
            self._recompute_last_log = progress_counter
            flagged_total = self._recompute_flagged_total
            fired_total = getattr(self, "_recompute_fired_total", 0)
            realized_fire_rate = (fired_total / emitted_total) if emitted_total else 0.0
            realized_flag_rate = (flagged_total / pos_total) if pos_total else 0.0
            logger.warning(
                "[recompute-acceptor] PROGRESS mode=%s emitted=%d positions=%d "
                "flagged=%d fired=%d realized_fire_rate=%.5f realized_flag_rate=%.5f",
                mode_name, emitted_total, pos_total, flagged_total, fired_total,
                realized_fire_rate, realized_flag_rate,
            )
            _write_stat(
                "progress",
                {
                    "mode": mode_name,
                    "rate": rate,
                    "tau": tau,
                    "use_cudagraph": use_cudagraph,
                    "emitted_total": emitted_total,
                    "positions_total": pos_total,
                    "flagged_total": flagged_total,
                    "fired_total": fired_total,
                    "realized_fire_rate": realized_fire_rate,
                    "realized_flag_rate": realized_flag_rate,
                },
            )

        return output

    sample_tokens.__name__ = orig_sample_tokens.__name__
    sample_tokens.__qualname__ = getattr(
        orig_sample_tokens, "__qualname__", "GPUModelRunner.sample_tokens"
    )
    GPUModelRunner.sample_tokens = sample_tokens
    setattr(GPUModelRunner, _PATCH_FLAG, True)

    # Fail-loud guard: a require_fire run that emitted tokens but never fired is a
    # silent no-op; surface it by stashing the predicate on the class so the harness
    # / sidecar stat exposes it. (The assert lives in the harness summary, not here,
    # to avoid crashing a live server mid-stream.)
    GPUModelRunner._recompute_require_fire = require_fire
    return True
