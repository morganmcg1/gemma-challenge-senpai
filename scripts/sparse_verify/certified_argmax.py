"""Greedy-safe top-k sparse verification of an LM head.

Given the lm_head weight ``W`` of shape ``[V, H]`` (for ``google/gemma-4-E4B-it``
this is the tied input embedding, ``V = 262144``, ``H = 2560``) and a static kept
set ``S`` of ``K`` token ids, :class:`SparseVerifier` computes the greedy argmax
over ``S`` only, then *certifies* that this kept-set winner is the true full-vocab
argmax. If it cannot certify, the step falls back to the full-vocab logits. The
emitted token is therefore **always identical** to plain full-vocab greedy argmax.

Soundness
---------
Let ``h`` be the post-final-norm hidden state that feeds the lm_head and let the
pre-softcap logit of token ``j`` be ``z_j = h . W_j``. For any pruned token
``j not in S`` Cauchy-Schwarz gives::

    z_j = h . W_j <= ||h||_2 * ||W_j||_2 <= ||h||_2 * R,
    where R = max_{j not in S} ||W_j||_2   (a single scalar, precomputed).

So if the kept-set winner ``m = max_{i in S} z_i`` satisfies ``m > ||h||_2 * R``
then **no** pruned token can equal or exceed it, hence the global argmax is in
``S``. Gemma's final-logit soft-capping ``z -> C*tanh(z/C)`` is strictly
monotonic, so it does not move the argmax; the whole certificate runs in
pre-softcap space and ignores the cap (the cap only matters for PPL, which is a
separate full-vocab path).

Bulletproofing the certificate against floating point and ties:
  * The complement bound is inflated by ``bound_margin`` so fp rounding keeps it a
    true over-estimate (never certify on a bound we might have rounded down).
  * We certify strictly (``m > bound``); equality forces fallback.
  * We require the kept-set maximiser to be *unique*; an exact tie inside the kept
    set forces fallback. Under "certified + unique" the global maximiser is unique,
    so it equals ``argmax`` of the full logits regardless of tie-break convention.
  * Fallback recomputes the full logits with the *same* weight and dtype the
    reference uses, so the fallback token is identical to the reference token by
    construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VerifyStats:
    """Aggregate accept/fallback counters for a batch of argmax calls."""

    n: int = 0
    n_certified: int = 0
    n_fallback: int = 0

    @property
    def fallback_rate(self) -> float:
        return self.n_fallback / self.n if self.n else 0.0

    @property
    def certified_rate(self) -> float:
        return self.n_certified / self.n if self.n else 0.0

    def update(self, other: "VerifyStats") -> None:
        self.n += other.n
        self.n_certified += other.n_certified
        self.n_fallback += other.n_fallback


class SparseVerifier:
    """Greedy-safe sparse verification of an LM head over a static kept set.

    Parameters
    ----------
    weight:
        ``[V, H]`` lm_head weight (the tied embedding for Gemma). Kept on its
        original device; the fallback path uses it directly.
    kept_ids:
        1-D iterable/tensor of kept vocab ids (the verified set ``S``). Deduped and
        sorted ascending internally.
    compute_dtype:
        dtype used for the kept/full GEMM and the argmax. ``float32`` by default to
        match the reference logits the model emits.
    bound_margin:
        relative inflation of the complement upper bound so it stays a true
        over-estimate under fp rounding. ``1e-4`` is ~3 orders of magnitude above
        the dot-product rounding error and costs a negligible number of extra
        fallbacks.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        kept_ids,
        *,
        compute_dtype: torch.dtype = torch.float32,
        bound_margin: float = 1e-4,
    ) -> None:
        if weight.dim() != 2:
            raise ValueError(f"weight must be [V, H], got {tuple(weight.shape)}")
        self.device = weight.device
        self.V, self.H = int(weight.shape[0]), int(weight.shape[1])
        self.compute_dtype = compute_dtype
        self.bound_margin = float(bound_margin)

        kept_ids = torch.as_tensor(kept_ids, device=self.device, dtype=torch.long)
        kept_ids = torch.unique(kept_ids)  # sorted ascending + deduped
        if kept_ids.numel() == 0:
            raise ValueError("kept_ids must be non-empty")
        if int(kept_ids[0]) < 0 or int(kept_ids[-1]) >= self.V:
            raise ValueError("kept_ids out of range for vocab size")
        self.kept_ids = kept_ids
        self.K = int(kept_ids.numel())

        kept_mask = torch.zeros(self.V, dtype=torch.bool, device=self.device)
        kept_mask[kept_ids] = True
        self.kept_mask = kept_mask

        self.weight = weight
        self.kept_weight = weight.index_select(0, kept_ids).to(compute_dtype).contiguous()

        comp_ids = torch.nonzero(~kept_mask, as_tuple=False).flatten()
        if comp_ids.numel() == 0:
            # Degenerate: kept set is the whole vocab; certificate never needed.
            self.R = torch.zeros((), dtype=torch.float64, device=self.device)
        else:
            comp_norms = weight.index_select(0, comp_ids).to(torch.float32).norm(dim=1)
            self.R = comp_norms.max().to(torch.float64)
        self.R_float = float(self.R)
        self._full_cd: torch.Tensor | None = None

    # -- diagnostics ---------------------------------------------------------
    def kept_max_norm(self) -> float:
        return float(self.kept_weight.to(torch.float32).norm(dim=1).max())

    def kept_min_norm(self) -> float:
        return float(self.kept_weight.to(torch.float32).norm(dim=1).min())

    def _full_weight_cd(self) -> torch.Tensor:
        if self._full_cd is None:
            self._full_cd = self.weight.to(self.compute_dtype).contiguous()
        return self._full_cd

    @torch.no_grad()
    def argmax(self, hidden: torch.Tensor, *, return_certified: bool = False):
        """Return greedy argmax token ids identical to full-vocab argmax.

        ``hidden`` is ``[..., H]``. Returns ``(tokens[...], VerifyStats)`` and,
        when ``return_certified`` is set, also a bool tensor marking the positions
        that were certified (i.e. did *not* fall back).
        """
        cd = self.compute_dtype
        h = hidden.to(cd)
        lead_shape = h.shape[:-1]
        flat = h.reshape(-1, self.H)
        n = int(flat.shape[0])

        kept_logits = flat @ self.kept_weight.t()  # [n, K]
        kept_max, kept_arg = kept_logits.max(dim=1)  # [n], [n]
        unique = (kept_logits == kept_max.unsqueeze(1)).sum(dim=1) == 1

        hnorm = flat.to(torch.float32).norm(dim=1).to(torch.float64)  # [n]
        bound = hnorm * self.R  # [n], fp64 upper bound on max pruned pre-softcap logit
        certified = unique & (kept_max.to(torch.float64) > bound * (1.0 + self.bound_margin))

        out = torch.empty(n, dtype=torch.long, device=self.device)
        out[certified] = self.kept_ids[kept_arg[certified]]

        fallback = ~certified
        n_fb = int(fallback.sum())
        if n_fb:
            full_logits = flat[fallback] @ self._full_weight_cd().t()  # [n_fb, V]
            out[fallback] = full_logits.argmax(dim=1)

        stats = VerifyStats(n=n, n_certified=n - n_fb, n_fallback=n_fb)
        tokens = out.reshape(lead_shape)
        if return_certified:
            return tokens, stats, certified.reshape(lead_shape)
        return tokens, stats

    @torch.no_grad()
    def full_argmax(self, hidden: torch.Tensor) -> torch.Tensor:
        """Reference full-vocab greedy argmax (same arithmetic as the fallback)."""
        cd = self.compute_dtype
        flat = hidden.to(cd).reshape(-1, self.H)
        logits = flat @ self._full_weight_cd().t()
        return logits.argmax(dim=1).reshape(hidden.shape[:-1])
