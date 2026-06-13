"""Greedy-safe top-k sparse verification for the LM head.

This package implements a *provably greedy-identical* vocabulary-prune lever for
the Gemma challenge target. The 262 144-token vocab is far larger than the
per-step token mass; restricting the lm_head / spec-decode verification GEMM to a
static top-k "kept set" makes it ~V/K times cheaper. The hard rule is that the
emitted greedy token must never change, so a cheap certificate decides per step
whether the full-vocab argmax is guaranteed to lie inside the kept set; when it
cannot be certified, the step falls back to the full-vocab logits.

See ``certified_argmax.SparseVerifier`` for the core lever.
"""

from .certified_argmax import SparseVerifier, VerifyStats

__all__ = ["SparseVerifier", "VerifyStats"]
