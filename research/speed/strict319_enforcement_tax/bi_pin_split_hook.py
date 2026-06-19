"""PR #688 — decouple the *attention-split-KV pin* from the *aten-GEMM pin*.

denken cost-decomposition instrument (analysis_only; NO HF Job, NO submission
change, NO leaderboard touch). Loaded ONLY in the spawned vLLM worker tree of the
``int4_mtp_batchinv`` K-sweep submission, and ONLY when ``BI_PIN_SPLIT=1`` — a
strict no-op otherwise.

Why this exists
---------------
``VLLM_BATCH_INVARIANT=1`` (the blanket pin my #683 ``bi_tax_ms=4.68`` priced)
turns on TWO physically distinct determinism mechanisms that share the one env
flag:

  * the **attention split-KV reduction pin** — ``flash_attn.py`` forces
    ``num_splits=1`` at every site that reads ``envs.VLLM_BATCH_INVARIANT``
    (decode ``max_num_splits`` build, encoder, cascade). land #680 (run
    ``5iy1mhe4``) identified THIS reduction as the strict-#319 break.
  * the **aten-GEMM pin** — ``enable_batch_invariant_mode()`` installs the triton
    persistent ``aten::{mm,addmm,matmul,linear,bmm}`` overrides + disables
    bf16/tf32 reduced-precision reduction. This pins the **bf16 drafter** GEMM
    (aten) but is a *no-op on the int4 target* (Marlin ``ops.marlin_gemm`` is not
    aten — land #680 / lawine #675 / globalflag #484).

To split ``bi_tax = drafter_attn_pin + drafter_gemm_pin + verify_attn_pin`` we
need the two pins on independent levers. They cannot be separated by env alone
(both read the single ``VLLM_BATCH_INVARIANT``), so we:

  * drive the **attention pin** through the real ``VLLM_BATCH_INVARIANT`` env
    (the harness sets it ``= BI_PIN_ATTN``) so every flash ``num_splits`` site
    behaves exactly as production; and
  * intercept ``init_batch_invariance`` so the **aten-GEMM override** is installed
    iff ``BI_PIN_GEMM=1`` — *regardless* of the flag — while the determinism-env
    baseline (``override_envs_for_invariance`` + ieee fp32) is held consistent
    across the attention-only / GEMM-only / blanket arms.

Arm matrix (the harness sets the three envs per arm)::

    arm        VLLM_BATCH_INVARIANT  BI_PIN_GEMM   attention   aten-GEMM override   baseline
    bi_off            0                  0         heuristic         no                no
    attn_only         1                  0         num_splits=1      no                yes
    gemm_only         0                  1         heuristic         YES               yes
    bi_full           1                  1         num_splits=1      YES               yes   (== blanket BI=1)

So ``attn_pin = T(attn_only) - T(bi_off)``, ``gemm_pin = T(gemm_only) - T(bi_off)``,
``bi_tax = T(bi_full) - T(bi_off)``, and the reconstruction residual
``T(attn_only)+T(gemm_only)-T(bi_full)-T(bi_off)`` (= baseline - cross-term) is
reported. ``bi_off`` and ``bi_full`` reproduce #683's raw / blanket arms exactly
(no determinism baseline on the floor), giving a continuity cross-check.

Injection
---------
``ENABLE_USER_SITE`` is False in this venv (so ``usercustomize`` is dead) and the
submission's own ``sitecustomize.py`` is first on ``sys.path`` (so a second
``sitecustomize`` is shadowed). The harness therefore drops a self-guarding
``.pth`` into site-packages that does
``os.environ.get("BI_PIN_SPLIT") and __import__("bi_pin_split_hook")`` — which
imports THIS module at site-init in the api_server + (spawned/forked) EngineCore +
worker, and is a strict no-op in every other process. This module then installs a
one-shot ``sys.meta_path`` finder that patches ``init_batch_invariance`` the moment
``vllm.model_executor.layers.batch_invariant`` is first imported (that import
happens at ``gpu_worker.py`` right before the call), mirroring the submission's
own attention-group patch mechanism.
"""
from __future__ import annotations

import json
import os
import sys

_TARGET = "vllm.model_executor.layers.batch_invariant"
_DONE_FLAG = "_bi_pin_split_patched"


def _truthy(name: str) -> bool:
    return os.environ.get(name, "0") not in ("", "0")


def _apply(module) -> None:
    """Replace ``init_batch_invariance`` on the freshly-imported batch_invariant
    module with an arm-aware version. Idempotent across processes / re-imports."""
    if getattr(module, _DONE_FLAG, False):
        return
    try:
        import torch

        import vllm.envs as envs

        real_enable = module.enable_batch_invariant_mode
        real_override = module.override_envs_for_invariance
        pin_gemm = _truthy("BI_PIN_GEMM")

        def _ieee_fp32_off() -> None:
            # Mirror init_batch_invariance's reduced-precision disable (lines that
            # set fp32 matmul/cudnn to ieee). bf16/int4 are untouched by tf32, but
            # we keep this in the determinism baseline so all pinned arms match.
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
            torch.backends.cudnn.rnn.fp32_precision = "ieee"

        def patched_init() -> None:
            # Attention pin is driven entirely by the real VLLM_BATCH_INVARIANT env
            # (the flash backend reads envs.VLLM_BATCH_INVARIANT directly); the
            # harness sets that env == BI_PIN_ATTN. Here we only control the
            # aten-GEMM override + the shared determinism-env baseline.
            pin_attn = _truthy("VLLM_BATCH_INVARIANT")
            if pin_gemm:
                # GEMM-only / blanket: full aten-GEMM determinism + baseline.
                real_override()
                real_enable()
                _ieee_fp32_off()
                gemm_installed = True
            elif pin_attn:
                # attention-only: baseline determinism env WITHOUT the aten-GEMM
                # override, so the only added cost over bi_off is the attention pin.
                real_override()
                _ieee_fp32_off()
                gemm_installed = False
            else:
                # bi_off floor: nothing (== plain VLLM_BATCH_INVARIANT=0).
                gemm_installed = False
            # Marker the harness greps to PROVE the patch ran in the worker and to
            # record which arm config was actually realized. Printed to the worker
            # stdout (lands in the server log) AND, if BI_PIN_SPLIT_MARKER names a
            # path, appended there as one-line JSON — robust against any worker
            # stdout redirection, so the harness has a hard correctness gate.
            realized = {
                "pin_attn": int(pin_attn),
                "pin_gemm": int(pin_gemm),
                "gemm_override_installed": int(gemm_installed),
                "batch_invariant_mode": int(getattr(module, "_batch_invariant_MODE", False)),
                "pid": os.getpid(),
            }
            print(
                "[bi_pin_split] patched_init "
                + " ".join(f"{k}={v}" for k, v in realized.items()),
                flush=True,
            )
            marker_path = os.environ.get("BI_PIN_SPLIT_MARKER")
            if marker_path:
                try:
                    with open(marker_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(realized) + "\n")
                except OSError:
                    pass

        module.init_batch_invariance = patched_init
        setattr(module, _DONE_FLAG, True)
        print(
            f"[bi_pin_split] installed patched init_batch_invariance "
            f"(BI_PIN_ATTN={os.environ.get('VLLM_BATCH_INVARIANT')}, "
            f"BI_PIN_GEMM={os.environ.get('BI_PIN_GEMM')})",
            flush=True,
        )
    except Exception:  # noqa: BLE001
        import logging

        logging.getLogger("bi_pin_split").exception("failed to patch init_batch_invariance")


def _install_finder() -> None:
    if _TARGET in sys.modules:
        _apply(sys.modules[_TARGET])
        return
    from importlib.abc import MetaPathFinder
    from importlib.util import find_spec

    class _BIPatchFinder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
            if fullname != _TARGET:
                return None
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            spec = find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec):  # noqa: ANN001
                _orig(module)
                _apply(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _BIPatchFinder())


def _arm_steptime() -> None:
    """Load the generic per-step timeline probe so the worker emits ``[steptime]
    raw`` records (drafter ``propose`` GPU + verify ``execute_model`` GPU).

    The ``int4_mtp_batchinv`` submission ships no ``steptime_patch`` (only the
    ``fa2sw_*`` lineage does), so serving it yields ZERO steptime records — that is
    exactly why #683 banked only whole-cycle ``bi_tax`` with no drafter/verify
    split. PR #688 needs the per-component split, so we import our vendored copy
    (sitting next to this module on the worker's ``sys.path``). It self-gates on
    ``STEPTIME=1`` (set by ``run_timing_pass``) and registers ONE-SHOT meta_path
    finders for ``gpu_model_runner`` + ``spec_decode.gemma4``. Those compose with
    the submission's own attn-group finder: each finder re-resolves through
    ``find_spec`` so the loaders chain regardless of registration order (verified:
    a smoke must still show the int4 num_heads patch applied)."""
    if not _truthy("STEPTIME"):
        return
    try:
        import steptime_patch  # noqa: F401  (registers finders on import)
    except Exception:  # noqa: BLE001
        import logging

        logging.getLogger("bi_pin_split").exception("failed to import steptime_patch")


# Only arm the patch when the harness explicitly requested it. The guarding .pth
# already short-circuits, but double-guard so a stray import is inert.
if _truthy("BI_PIN_SPLIT"):
    _install_finder()
    _arm_steptime()
