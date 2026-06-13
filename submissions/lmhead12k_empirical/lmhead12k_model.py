#!/usr/bin/env python
"""Custom vLLM model class for the lmhead12k empirical prune.

DESIGN (per research pass; see PR body). The served checkpoint's lm_head has only
``kept_size`` output rows (e.g. 7584-12288) instead of 262144, which gives the
decode-step weight-bandwidth saving. But vLLM's sampler, detokenizer, and
prompt_logprobs path all assume logits are full-vocab (262144) and index by
original token id. A pure LogitsProcessor cannot bridge this: in vLLM V1
prompt_logprobs are read from the model output BEFORE logits processors run.

So we subclass the model and scatter the kept-row logits back into a full
262144-wide tensor (-inf on pruned positions) inside ``compute_logits``. After
that, the entire downstream V1 pipeline is unchanged and audit-correct:
  * greedy argmax over the scattered tensor == original token id (kept rows keep
    their true logit; pruned rows are -inf and can never win),
  * prompt_logprobs for any kept ground-truth token id is finite,
  * /v1/completions return_token_ids reports original ids.

config.json MUST keep vocab_size=262144 (only lm_head.out_features shrinks), or
vLLM's sampler tensor allocation and pad-to-multiple logic break.

!!! NEEDS GPU VALIDATION against the installed vLLM 0.22.0 Gemma3 source
(vllm/model_executor/models/gemma3.py). Class/arg names below follow the research
sketch and may need small adjustments to match the exact 0.22.0 signatures. This
module is the captured design, not yet a GPU-validated artifact.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

try:
    from vllm.model_executor.models.gemma3 import Gemma3ForCausalLM
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.models.utils import maybe_prefix
    _VLLM_AVAILABLE = True
except Exception:  # pragma: no cover - import only succeeds in the serve env
    Gemma3ForCausalLM = object  # type: ignore
    _VLLM_AVAILABLE = False


def _load_kept_ids(model_dir: str) -> list[int]:
    p = Path(model_dir) / "kept_ids.json"
    if not p.exists():
        raise FileNotFoundError(
            f"kept_ids.json not found in {model_dir}; the pruned checkpoint must "
            f"ship it so the scatter maps 12k rows -> original ids."
        )
    return json.loads(p.read_text())["kept_ids"]


class Gemma3ForCausalLMLMHead12k(Gemma3ForCausalLM):  # type: ignore[misc]
    """Gemma3 text model with a row-pruned lm_head + scatter-to-full-vocab logits."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = getattr(self, "config", None) or kwargs.get("config")
        model_dir = os.environ.get("MODEL_ID", "")
        kept = _load_kept_ids(model_dir)
        kept_size = len(kept)
        # Rebuild lm_head at the pruned width so checkpoint weights load cleanly.
        if _VLLM_AVAILABLE:
            self.lm_head = ParallelLMHead(
                kept_size,
                config.hidden_size,
                quant_config=getattr(self, "quant_config", None),
                prefix=maybe_prefix(kwargs.get("prefix", ""), "lm_head"),
            )
        # On-device buffer so the scatter index lives on the right CUDA device.
        self.register_buffer(
            "kept_ids", torch.tensor(kept, dtype=torch.long), persistent=False
        )
        self._full_vocab = config.vocab_size  # stays 262144

    def compute_logits(self, hidden_states: torch.Tensor, *args, **kwargs):
        partial = self.logits_processor(self.lm_head, hidden_states, *args, **kwargs)
        if partial is None:  # non-driver TP rank
            return None
        full = torch.full(
            (partial.shape[0], self._full_vocab),
            float("-inf"),
            dtype=partial.dtype,
            device=partial.device,
        )
        idx = self.kept_ids.unsqueeze(0).expand(partial.shape[0], -1)
        full.scatter_(1, idx, partial)
        return full
