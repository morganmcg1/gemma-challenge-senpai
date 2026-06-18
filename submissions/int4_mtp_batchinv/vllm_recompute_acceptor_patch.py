"""Env-gated recompute-acceptor SPEED probe (PR #642 — de-project #636).

#636 projected the gap-flag M=1-recompute acceptor's served throughput as an
*additive* composition ``1/(1/152.291 + ftr/126.378)`` — assuming a recompute adds
cleanly to the amortized spec loop. This patch MEASURES the real hit instead by
firing REAL width-1 target forwards into the served spec-decode loop at a
controlled rate, so the int4 GEMM weight-read + CUDA-graph-break / serialization
cost the projection ignored is paid on the wall clock.

No-op unless ``SENPAI_RECOMPUTE_RATE`` is a positive float. When set, after each
spec-decode step the patched ``GPUModelRunner.sample_tokens`` fires
``floor(carry + rate * emitted_tokens)`` extra width-1 forwards (a fractional
carry makes the long-run total exactly ``rate * total_emitted``). The drafter
dummy is suppressed during these injected forwards (``SENPAI_RECOMPUTE_TARGET_ONLY``
default on) so the injected cost is a **target-only** width-1 forward — the same
w4a16-ct M=1 recompute the acceptor would run, not target+drafter.

The emitted tokens are left UNCHANGED: greedy-identity of the rescued stream is
certified separately, offline, over the served M=1 AR trajectory (the verify
numerics are unavailable from the API), so here we only need the *cost*. Hence
``wall_tps = completion_tokens / decode_s`` correctly reflects
``(un-rescued spec throughput)`` minus ``(recompute overhead at this rate)``.

Arms driven purely by the env var (single patched build, served by the validated
``paired_tps_ab`` harness):
  * ``rate=0``                          -> un-rescued ceiling (arm c).
  * ``rate in {0.05,0.10,0.20}``        -> slope sweep -> fit per-recompute
                                           in-loop marginal cost C (de-projects
                                           the additive-composition optimism).
  * ``rate=flag_trigger_rate``          -> the de-projected acceptor (arm a).

The injected forward uses ``_dummy_run`` (KV slot_mapping = -1 so it never writes
live cache; shares pinned input buffers under ``synchronize_input_prep`` and the
next real step's ``_prepare_inputs`` overwrites any residue) in eager mode
(``CUDAGraphMode.NONE``) so each injected forward breaks the captured decode graph
exactly as a real interleaved recompute would. ``_dummy_run`` attends over a dummy
(short) context, so it captures the dominant int4 weight-read cost but slightly
under-counts long-context attention; arm (d) (served M=1 AR, full real context)
is the cross-check that bounds that gap.

Observability (PR #642 de-projection bug-fix): the original cut of this patch
parsed ``output.sampled_token_ids`` with ``if not stids`` and emitted its progress
via bare ``print``. In the live server neither held — the per-step token count
must be robust to BOTH the sync ``ModelRunnerOutput`` (``list[list[int]]``) and the
async ``AsyncGPUModelRunnerOutput`` (a padded GPU tensor) shapes, and worker-process
diagnostics must go through vLLM's captured ``init_logger`` to land in the server
log. A non-firing patch now *fails loud*: it logs the realized rate periodically and
asserts (``SENPAI_RECOMPUTE_REQUIRE_FIRE`` default on) that real forwards actually
fired, so a silent no-op can never again be mistaken for "zero overhead".
"""
from __future__ import annotations

import json
import os

# Marker so apply() is idempotent across the multiple (fork/spawn) processes that
# import gpu_model_runner.
_PATCH_FLAG = "_optionb_recompute_acceptor_patch_applied"
_RATE_ENV = "SENPAI_RECOMPUTE_RATE"
_TARGET_ONLY_ENV = "SENPAI_RECOMPUTE_TARGET_ONLY"
_REQUIRE_FIRE_ENV = "SENPAI_RECOMPUTE_REQUIRE_FIRE"
# Cudagraph mode of the injected width-1 recompute forward. The shipped served
# stack runs FULL_AND_PIECEWISE with cudagraph_capture_sizes=[1,2,4,8], so a
# width-1 uniform decode HAS a captured graph available (arm (d) M=1 AR replays
# it -> 77.89 TPS). Two faithful cost models bracket the real acceptor:
#   * unset / "0"  -> CUDAGraphMode.NONE: force eager, the WORST case (the
#     interleaved recompute breaks the captured decode graph / serializes). This
#     is the pessimistic de-projection bound.
#   * "1"          -> cudagraph_runtime_mode=None: let the cudagraph_dispatcher
#     pick the natural mode for a size-1 uniform decode (the captured graph),
#     i.e. model a recompute that reuses the stack's captured width-1 path. This
#     is the realistic/optimistic bound. (We pass None, NOT CUDAGraphMode.FULL:
#     _dummy_run asserts the passed mode == the dispatcher's, so forcing FULL
#     crashes if the dispatcher picks PIECEWISE; None defers to the dispatcher.)
_CUDAGRAPH_ENV = "SENPAI_RECOMPUTE_CUDAGRAPH"
# Capture-independent observability: the model runner lives in the EngineCore
# subprocess, whose import-time/per-step stdout is NOT reliably forwarded to the
# harness server log (only post-init vLLM log forwarding is). So in addition to
# the logger, write a sidecar JSON stat file straight to disk under this dir; the
# de-projection runner reads it to PROVE the injection fired (fired==rate*emitted).
_STAT_DIR_ENV = "SENPAI_RECOMPUTE_STAT_DIR"
# Emit a progress line to the (captured) server log every this-many emitted tokens
# so the injected recompute count is verifiable post-hoc against rate*emitted.
# Overridable (small values for smoke tests) via SENPAI_RECOMPUTE_LOG_EVERY.
_LOG_EVERY = int(os.environ.get("SENPAI_RECOMPUTE_LOG_EVERY", "4096") or "4096")


def _rate() -> float:
    try:
        return float(os.environ.get(_RATE_ENV, "") or "0")
    except (TypeError, ValueError):
        return 0.0


def _write_stat(kind: str, payload: dict) -> None:
    """Atomically dump a sidecar stat JSON to ``$SENPAI_RECOMPUTE_STAT_DIR``.
    No-op when the env var is unset (the shipped serving path never writes)."""
    stat_dir = os.environ.get(_STAT_DIR_ENV, "")
    if not stat_dir:
        return
    try:
        os.makedirs(stat_dir, exist_ok=True)
        path = os.path.join(stat_dir, f"recompute_{kind}_{os.getpid()}.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"pid": os.getpid(), **payload}, fh)
        os.replace(tmp, path)
    except Exception:
        pass


def _count_emitted(stids, torch) -> int:
    """Per-step emitted-token count, robust to both served output shapes.

    * sync ``ModelRunnerOutput.sampled_token_ids`` is ``list[list[int]]`` (one
      inner list per request; spec decode emits 1..M accepted ids).
    * async ``AsyncGPUModelRunnerOutput.sampled_token_ids`` is a GPU tensor
      ``[num_reqs, max_spec+1]`` padded with ``-1`` at rejected positions.
    Returns 0 for the PP/kv-only early-return outputs (``None``/empty)."""
    if stids is None:
        return 0
    if isinstance(stids, torch.Tensor):
        if stids.numel() == 0:
            return 0
        # valid sampled ids are >= 0; padding is -1.
        return int((stids >= 0).sum().item())
    try:
        return sum((len(t) for t in stids if t is not None), 0)
    except TypeError:
        return 0


def apply(gmr) -> bool:
    """Wrap ``gmr.GPUModelRunner.sample_tokens`` to inject rate-controlled real
    width-1 target forwards. ``gmr`` is the live ``vllm.v1.worker.gpu_model_runner``
    module; every free symbol is resolved from it so we never guess import paths
    against a vLLM version. Returns True if applied, False if already present."""
    runner_cls = gmr.GPUModelRunner
    if getattr(runner_cls, _PATCH_FLAG, False):
        return False

    torch = gmr.torch
    CUDAGraphMode = gmr.CUDAGraphMode
    logger = gmr.init_logger("int4_mtp_drafter.recompute_acceptor")
    target_only = os.environ.get(_TARGET_ONLY_ENV, "1") not in ("", "0")
    require_fire = os.environ.get(_REQUIRE_FIRE_ENV, "1") not in ("", "0")
    use_cudagraph = os.environ.get(_CUDAGRAPH_ENV, "0") not in ("", "0")
    # None -> dispatcher picks the captured width-1 graph (realistic); NONE ->
    # forced eager (worst case). Computed once; the per-step loop just reuses it.
    recompute_cg_mode = None if use_cudagraph else CUDAGraphMode.NONE
    orig_sample_tokens = runner_cls.sample_tokens

    logger.warning(
        "[recompute-acceptor] APPLY wrapping GPUModelRunner.sample_tokens "
        "rate_env=%r parsed=%.6f target_only=%s require_fire=%s cudagraph=%s "
        "(mode=%s) log_every=%d",
        os.environ.get(_RATE_ENV), _rate(), target_only, require_fire,
        use_cudagraph, "dispatcher" if use_cudagraph else "NONE/eager", _LOG_EVERY,
    )
    _write_stat("apply", {
        "rate_env": os.environ.get(_RATE_ENV), "rate": _rate(),
        "target_only": target_only, "cudagraph": use_cudagraph, "log_every": _LOG_EVERY,
    })

    def sample_tokens(self, grammar_output, _orig=orig_sample_tokens):
        output = _orig(self, grammar_output)
        r = _rate()
        if r <= 0.0:
            return output
        if output is None:
            return output

        # Resolve the per-step sampled-token tensor across BOTH served output shapes.
        # Sync path: ``ModelRunnerOutput.sampled_token_ids`` (``list[list[int]]``).
        # Async path (``use_async_scheduling`` — the served default here): the returned
        # ``AsyncGPUModelRunnerOutput`` has NO public ``sampled_token_ids`` attribute —
        # the GPU tensor lives in the PRIVATE ``_sampled_token_ids`` (the public name
        # only materializes after ``get_output()``'s blocking D2H, which runs later in
        # the output processor, not here). The original cut read only the public attr,
        # got ``None`` -> ``n_emitted==0`` every step -> silent no-op. Fall back to the
        # private GPU tensor (alive until ``get_output``; counted on-GPU via ``>= 0``).
        stids = getattr(output, "sampled_token_ids", None)
        if stids is None:
            stids = getattr(output, "_sampled_token_ids", None)
        n_emitted = _count_emitted(stids, torch)

        if not getattr(self, "_recompute_diag_done", False):
            self._recompute_diag_done = True
            logger.warning(
                "[recompute-acceptor] FIRST-FIRING-CALL output_type=%s stids_type=%s "
                "n_emitted=%d rate=%.6f",
                type(output).__name__, type(stids).__name__, n_emitted, r,
            )
            # Capture-independent firstcall proof (the worker logger is NOT reliably
            # forwarded to the server log): records which output shape was served and
            # whether extraction found tokens, so a wrong attribute can't silently no-op.
            _write_stat("firstcall", {
                "rate": r, "output_type": type(output).__name__,
                "stids_type": type(stids).__name__,
                "has_public_stids": hasattr(output, "sampled_token_ids"),
                "has_private_stids": hasattr(output, "_sampled_token_ids"),
                "n_emitted": n_emitted,
            })
        if n_emitted <= 0:
            return output

        acc = getattr(self, "_recompute_acc", 0.0) + r * n_emitted
        n_extra = int(acc)
        self._recompute_acc = acc - n_extra
        emitted_total = getattr(self, "_recompute_emitted_total", 0) + n_emitted
        self._recompute_emitted_total = emitted_total

        if n_extra > 0:
            drafter = getattr(self, "drafter", None)
            saved = None
            if target_only and drafter is not None and hasattr(drafter, "dummy_run"):
                saved = drafter.dummy_run
                drafter.dummy_run = lambda *a, **k: None
            try:
                with torch.inference_mode():
                    for _ in range(n_extra):
                        self._dummy_run(
                            1,
                            cudagraph_runtime_mode=recompute_cg_mode,
                            uniform_decode=True,
                            allow_microbatching=False,
                            skip_eplb=True,
                        )
            finally:
                if saved is not None:
                    drafter.dummy_run = saved
            fired_total = getattr(self, "_recompute_fired_total", 0) + n_extra
            self._recompute_fired_total = fired_total
        else:
            fired_total = getattr(self, "_recompute_fired_total", 0)

        last_log = getattr(self, "_recompute_last_log", 0)
        if emitted_total - last_log >= _LOG_EVERY:
            self._recompute_last_log = emitted_total
            realized = fired_total / emitted_total if emitted_total else 0.0
            logger.warning(
                "[recompute-acceptor] rate=%.4f target_only=%s emitted=%d fired=%d "
                "realized_rate=%.4f",
                r, target_only, emitted_total, fired_total, realized,
            )
            _write_stat("progress", {
                "rate": r, "target_only": target_only, "emitted": emitted_total,
                "fired": fired_total, "realized_rate": realized,
            })
            if require_fire and emitted_total >= _LOG_EVERY and fired_total == 0:
                raise RuntimeError(
                    "[recompute-acceptor] SENPAI_RECOMPUTE_RATE="
                    f"{r} set but 0 forwards fired after {emitted_total} emitted "
                    "tokens — injection is a silent no-op (set "
                    "SENPAI_RECOMPUTE_REQUIRE_FIRE=0 to bypass)."
                )
        return output

    sample_tokens.__name__ = "sample_tokens"
    sample_tokens.__qualname__ = "GPUModelRunner.sample_tokens"
    runner_cls.sample_tokens = sample_tokens
    setattr(runner_cls, _PATCH_FLAG, True)
    return True
