"""Custom vLLM model class for the empirical lmhead12k prune (Gemma 4).

WHY A CUSTOM CLASS (validated against the installed vLLM 0.22.0 source):

The served checkpoint's ``lm_head`` has only ``kept_size`` output rows (e.g.
12,288) instead of the full 262,144. That is the decode-step weight-bandwidth
saving. But vLLM's sampler / detokenizer / ``prompt_logprobs`` path all assume
logits are full-vocab (262,144) and index by *original* token id. A pure
``LogitsProcessor`` cannot bridge this: in vLLM V1 ``prompt_logprobs`` are read
from the model output BEFORE logits processors run. So we subclass the model and
scatter the kept-row logits back into a full 262,144-wide tensor (``-inf`` on
pruned positions) inside ``compute_logits``. Downstream the whole V1 pipeline is
unchanged and audit-correct:
  * greedy argmax over the scattered tensor == original token id (kept rows keep
    their true logit; pruned rows are ``-inf`` and can never win),
  * ``prompt_logprobs`` for any kept ground-truth token id is finite,
  * ``/v1/completions`` ``return_token_ids`` reports original ids.

ARCHITECTURE NOTE: ``google/gemma-4-E4B-it`` is a multimodal
``Gemma4ForConditionalGeneration``. Its ``compute_logits`` delegates to
``self.language_model.compute_logits`` (gemma4_mm.py), and the language model is
built via ``init_vllm_registered_model(architectures=["Gemma4ForCausalLM"])``
(gemma4_mm.py __init__). So we subclass the *text* model ``Gemma4ForCausalLM`` and
register it under the string ``"Gemma4ForCausalLM"``: the multimodal wrapper then
picks up our class for ``self.language_model`` while the vision/audio towers and
all multimodal plumbing remain the stock implementation (modalities preserved).

The parent ``Gemma4ForCausalLM.compute_logits`` is exactly
``self.logits_processor(self.lm_head, hidden_states)`` with the head and the
LogitsProcessor both sized at ``config.vocab_size`` and ``soft_cap=
final_logit_softcapping``. We rebuild both at ``kept_size`` (keeping the same
soft cap) and scatter the result. ``config.vocab_size`` MUST stay 262,144 (only
``lm_head.out_features`` shrinks) so the sampler tensor allocation is unchanged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
from vllm.model_executor.models.utils import maybe_prefix


def _load_kept_ids(vllm_config: VllmConfig) -> list[int]:
    """Locate kept_ids.json in the served checkpoint dir (or MODEL_ID env)."""
    candidates = []
    model_path = getattr(vllm_config.model_config, "model", "") or ""
    if model_path:
        candidates.append(Path(model_path) / "kept_ids.json")
    env_model = os.environ.get("MODEL_ID", "")
    if env_model:
        candidates.append(Path(env_model) / "kept_ids.json")
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text())["kept_ids"]
    raise FileNotFoundError(
        f"kept_ids.json not found next to the checkpoint (looked in {candidates}); "
        f"the pruned checkpoint must ship it so the scatter maps kept rows -> ids."
    )


class Gemma4ForCausalLMLMHead12k(Gemma4ForCausalLM):
    """Gemma4 text model with a row-pruned lm_head + scatter-to-full-vocab logits."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = self.config
        kept = _load_kept_ids(vllm_config)
        kept_size = len(kept)
        full_vocab = config.vocab_size  # stays 262144

        # Rebuild the head + logits processor at the pruned width so the
        # checkpoint's pruned lm_head loads cleanly and the GEMM only touches
        # kept_size rows. Keep the same soft cap as the parent.
        # quant_config=None forces a bf16 head: the shipped lm_head.weight is
        # bf16 (the int4 base keeps lm_head in the compressed-tensors `ignore`
        # list), and forcing None avoids depending on ignore-list prefix matching
        # surviving the multimodal wrapper + hf_to_vllm_mapper rename.
        self.lm_head = ParallelLMHead(
            kept_size,
            config.hidden_size,
            quant_config=None,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            kept_size,
            soft_cap=getattr(config, "final_logit_softcapping", None),
        )
        self.register_buffer(
            "kept_ids", torch.tensor(kept, dtype=torch.long), persistent=False
        )
        self._full_vocab = full_vocab
        self._kept_size = kept_size

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        partial = self.logits_processor(self.lm_head, hidden_states)
        if partial is None:  # non-driver TP rank
            return None
        if self.kept_ids.device != partial.device:
            self.kept_ids = self.kept_ids.to(partial.device)
        full = torch.full(
            (partial.shape[0], self._full_vocab),
            float("-inf"),
            dtype=partial.dtype,
            device=partial.device,
        )
        idx = self.kept_ids.unsqueeze(0).expand(partial.shape[0], -1)
        full.scatter_(1, idx, partial)
        return full
