"""Mark the speculative-proposer scope so the force-2D patch can discriminate
drafter forwards from main-model forwards.

``vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer`` is the base
class of the Gemma4 MTP proposer (``Gemma4Proposer``). Every drafter proposer
forward runs synchronously inside ``SpecDecodeBaseProposer.propose`` (the first
pass plus the ``num_speculative_tokens - 1`` loop passes); CUDA-graph capture /
profiling for the drafter runs inside ``dummy_run``. We wrap both to raise a
thread-local re-entrancy counter (``_surgattn_arm.enter/exit_drafter``) for their
duration, so a ``unified_attention`` call can ask ``in_drafter_propose()`` and
know whether it is serving a drafter forward.

``Gemma4Proposer`` does not override ``propose`` / ``dummy_run``, so patching the
base-class methods covers it. The patch is output-neutral: it only sets/clears a
flag around the existing call; it never changes inputs, numerics, or control
flow. Wrapping ``dummy_run`` matters because the drafter's CUDA graph is captured
there — the 2D-vs-3D kernel choice is frozen at capture, so the flag must be set
during capture too, not just during the live ``propose``.
"""

from __future__ import annotations

import functools
import sys

import _surgattn_arm as _arm

_PATCH_FLAG = "_surgattn_attrib_proposer_ctx_patched"


def _wrap_method(cls, name: str) -> bool:
    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, "_surgattn_ctx_wrapper", False):
        return False

    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        _arm.enter_drafter()
        try:
            return orig(self, *args, **kwargs)
        finally:
            _arm.exit_drafter()

    wrapper._surgattn_ctx_wrapper = True
    setattr(cls, name, wrapper)
    return True


def apply(llm_base_proposer) -> bool:
    """Wrap ``SpecDecodeBaseProposer.propose`` / ``dummy_run`` to mark the drafter
    scope. ``llm_base_proposer`` is the imported
    ``vllm.v1.spec_decode.llm_base_proposer`` module. Returns ``True`` if applied.
    """
    base = llm_base_proposer.SpecDecodeBaseProposer
    if getattr(base, _PATCH_FLAG, False):
        return False
    wrapped = [name for name in ("propose", "dummy_run") if _wrap_method(base, name)]
    setattr(base, _PATCH_FLAG, True)
    print(
        f"[surgattn-attrib arm={_arm.ARM}] proposer scope marker installed on "
        f"SpecDecodeBaseProposer.{{{','.join(wrapped) or 'none'}}}",
        file=sys.stderr,
        flush=True,
    )
    return True
