"""Forward-type discriminator + arm selector for the surgattn attribution card.

This module is the single shared state between the two runtime patches in this
submission (``vllm_force2d_attn_patch`` and ``vllm_proposer_ctx_patch``). It has
no vLLM/torch import so it is safe to load in every process the
``sitecustomize`` hook runs in.

Background
----------
``int4_mtp_bi0_surgattn`` (the shipped "bi0") forces the TRITON_ATTN 2D
single-pass path on EVERY M=1 forward via ``vllm_force2d_attn_patch``. That is
load-bearing for greedy byte-identity: the kernel's launch gate only ever picks
the 3D split-KV path when ``max_seqlen_q <= 1`` (pure M=1 decode), and a 3D M=1
forward is NOT byte-identical to the M=K 2D verify forward, so an emitted
target-argmax token can flip.

But at K speculative tokens, each spec-decode step issues two *kinds* of M=1
forward:

* the **drafter proposer forwards** (the K-1 single-token passes inside
  ``SpecDecodeBaseProposer.propose``), which only produce DRAFT PROPOSALS, and
* the **main-model forward(s)** (verify / the occasional M=1 decode), which
  produce the EMITTED token.

At temperature 0 the rejection sampler emits the *target's* argmax regardless of
what the drafter proposed, so changing the numerics of the drafter's proposer
forwards CANNOT change the emitted sequence. Therefore letting the gate pick 3D
on the drafter proposer forwards ONLY is byte-identical by construction, while
still recovering whatever share of the attention speedup lives in those passes.

Arms (selected by the ``SURGATTN_ARM`` env var; default = shipped bi0)
---------------------------------------------------------------------
* ``control_2d``      force 2D on every M=1 forward (identical to shipped bi0).
                      The greedy-identity + TPS reference.
* ``drafter_only_3d`` force 2D on the main-model forwards, let the gate pick 3D
                      on the drafter proposer forwards only. Byte-identical to
                      ``control_2d`` by construction; measures the drafter share
                      of the speedup.
* ``all_3d``          never force 2D, i.e. surgattn OFF — let the gate pick 3D on
                      every M=1 forward (the identity-breaking variant, kept here
                      only as the attribution anchor).

The discriminator is a thread-local re-entrancy counter raised on entry to (and
lowered on exit from) the proposer ``propose`` / ``dummy_run`` calls. Every
drafter ``unified_attention`` call runs synchronously on the worker thread
within that scope, so the counter is > 0 exactly for drafter forwards. CUDA-graph
capture also runs the Python eagerly once inside that scope (we wrap
``dummy_run`` for that reason), so the 2D-vs-3D choice is baked into the captured
graph correctly per forward type.
"""

from __future__ import annotations

import os
import threading

CONTROL_2D = "control_2d"
DRAFTER_ONLY_3D = "drafter_only_3d"
ALL_3D = "all_3d"
_VALID = {CONTROL_2D, DRAFTER_ONLY_3D, ALL_3D}


def _read_arm() -> str:
    arm = (os.environ.get("SURGATTN_ARM") or CONTROL_2D).strip()
    if arm not in _VALID:
        # Fail safe to the shipped, byte-identical behaviour rather than silently
        # serving an unintended kernel mix.
        return CONTROL_2D
    return arm


# Resolved once at import; the env is fixed for the lifetime of a server process.
ARM = _read_arm()

_state = threading.local()


def enter_drafter() -> None:
    _state.depth = getattr(_state, "depth", 0) + 1


def exit_drafter() -> None:
    _state.depth = max(0, getattr(_state, "depth", 0) - 1)


def in_drafter_propose() -> bool:
    """True iff the current thread is inside a proposer propose/dummy_run scope."""
    return getattr(_state, "depth", 0) > 0


def should_force_2d() -> bool:
    """Whether the unified_attention call running *now* must be forced to 2D.

    * ``all_3d``          -> never force (surgattn OFF everywhere).
    * ``drafter_only_3d`` -> force 2D on the main-model path, allow 3D in drafter.
    * ``control_2d``      -> force 2D everywhere (shipped bi0).
    """
    if ARM == ALL_3D:
        return False
    if ARM == DRAFTER_ONLY_3D:
        return not in_drafter_propose()
    return True
