"""Soundness unit tests for the greedy-safe sparse verifier.

These are model-free: they exercise the certificate on synthetic weights and
hidden states and assert that the sparse argmax is *always* identical to the
full-vocab argmax, that certification implies the global max is genuinely in the
kept set, and that the fallback fires exactly when it must (pruned-token argmax,
intra-kept ties).

Run::

    python -m unittest scripts.sparse_verify.tests.test_certified_argmax -v
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from scripts.sparse_verify.certified_argmax import SparseVerifier  # noqa: E402


def _full_argmax(weight: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    return (hidden.to(torch.float32) @ weight.to(torch.float32).t()).argmax(dim=1)


class TestCertifiedArgmax(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.V, self.H = 4096, 128
        self.weight = torch.randn(self.V, self.H)

    def test_identity_random(self) -> None:
        """Sparse argmax == full argmax on random hidden states, certified or not."""
        kept = torch.randperm(self.V)[:512]
        ver = SparseVerifier(self.weight, kept)
        hidden = torch.randn(2000, self.H)
        toks, stats = ver.argmax(hidden)
        ref = _full_argmax(self.weight, hidden)
        self.assertTrue(torch.equal(toks, ref))
        # On isotropic random data with a small kept set, some fallback is expected.
        self.assertEqual(stats.n, 2000)
        self.assertEqual(stats.n_certified + stats.n_fallback, 2000)

    def test_certified_implies_global_max(self) -> None:
        """Every certified step's emitted token is the true full-vocab argmax."""
        kept = torch.randperm(self.V)[:1024]
        ver = SparseVerifier(self.weight, kept)
        hidden = torch.randn(3000, self.H)
        toks, _, certified = ver.argmax(hidden, return_certified=True)
        ref = _full_argmax(self.weight, hidden)
        # certified rows must match (they always do, but assert on the subset too)
        self.assertTrue(torch.equal(toks[certified], ref[certified]))
        # and the certified token must be inside the kept set
        self.assertTrue(bool(ver.kept_mask[toks[certified]].all()))

    def test_fallback_for_pruned_argmax(self) -> None:
        """A hidden aligned with a *pruned* row must fall back and emit that row."""
        kept = torch.arange(0, 2048)  # ids [0,2048) kept; high ids pruned
        ver = SparseVerifier(self.weight, kept)
        pruned_token = 3000
        # hidden parallel to W[pruned] and large => pruned token dominates.
        hidden = (self.weight[pruned_token] / self.weight[pruned_token].norm()) * 50.0
        hidden = hidden.unsqueeze(0)
        toks, stats, certified = ver.argmax(hidden, return_certified=True)
        self.assertEqual(int(toks[0]), pruned_token)
        self.assertFalse(bool(certified[0]))  # cannot be certified
        self.assertEqual(stats.n_fallback, 1)
        self.assertEqual(int(toks[0]), int(_full_argmax(self.weight, hidden)[0]))

    def test_certified_for_kept_dominant(self) -> None:
        """With favorable geometry (small-norm complement) a kept-aligned hidden certifies.

        This mirrors the real tied-embedding geometry: pruned (rare) tokens have
        smaller row norms than kept (frequent) tokens, so the complement bound
        ``||h|| * R`` is small and certification fires. Note certification can never
        be *required* for correctness — the loose-bound case simply falls back —
        but it must be *achievable* when the geometry is favorable, else the lever
        wins nothing.
        """
        weight = self.weight.clone()
        weight[2048:] *= 0.1  # pruned rows get small norms (rare-token geometry)
        kept = torch.arange(0, 2048)
        ver = SparseVerifier(weight, kept)
        kept_token = 7
        hidden = (weight[kept_token] / weight[kept_token].norm()) * 200.0
        hidden = hidden.unsqueeze(0)
        toks, stats, certified = ver.argmax(hidden, return_certified=True)
        self.assertEqual(int(toks[0]), kept_token)
        self.assertTrue(bool(certified[0]))
        self.assertEqual(stats.n_fallback, 0)

    def test_intra_kept_tie_forces_fallback(self) -> None:
        """Two identical kept rows at the max => non-unique => fallback, still correct."""
        weight = self.weight.clone()
        weight[10] = weight[11]  # exact duplicate rows, both kept
        kept = torch.arange(0, 2048)
        ver = SparseVerifier(weight, kept)
        # hidden aligned with the duplicated row so it is the max.
        hidden = (weight[10] / weight[10].norm() * 100.0).unsqueeze(0)
        toks, stats, certified = ver.argmax(hidden, return_certified=True)
        self.assertFalse(bool(certified[0]))  # tie => not certified
        # full argmax picks the lowest index of the tie under torch's reduction;
        # the fallback uses the identical op, so they must agree.
        self.assertEqual(int(toks[0]), int(_full_argmax(weight, hidden)[0]))

    def test_full_vocab_kept_set(self) -> None:
        """Kept set == whole vocab => R == 0 => always certified, always correct."""
        kept = torch.arange(0, self.V)
        ver = SparseVerifier(self.weight, kept)
        self.assertEqual(ver.R_float, 0.0)
        hidden = torch.randn(500, self.H)
        toks, stats = ver.argmax(hidden)
        self.assertEqual(stats.n_fallback, 0)
        self.assertTrue(torch.equal(toks, _full_argmax(self.weight, hidden)))

    def test_bf16_weight_fp32_compute(self) -> None:
        """Identity holds with a bf16 weight and fp32 compute dtype."""
        weight = self.weight.to(torch.bfloat16)
        kept = torch.randperm(self.V)[:1024]
        ver = SparseVerifier(weight, kept, compute_dtype=torch.float32)
        hidden = torch.randn(1500, self.H)
        toks, _ = ver.argmax(hidden)
        ref = (hidden.float() @ weight.float().t()).argmax(dim=1)
        self.assertTrue(torch.equal(toks, ref))


if __name__ == "__main__":
    unittest.main(verbosity=2)
