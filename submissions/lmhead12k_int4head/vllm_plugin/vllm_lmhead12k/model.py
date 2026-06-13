"""Custom vLLM model class for the lmhead12k prune (Gemma 4), int4-head capable.

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

HEAD DTYPE (bf16 vs int4) IS CHECKPOINT-DRIVEN. We rebuild ``self.lm_head`` at the
pruned width and pass the body's ``vllm_config.quant_config`` (NOT ``None``). vLLM's
``CompressedTensorsConfig.get_quant_method`` then resolves the head's scheme from
the checkpoint's ``quantization_config``:
  * if the config TARGETS ``re:.*lm_head`` (lm_head un-ignored + a head group, the
    ``lmhead12k_int4head`` checkpoint) -> ``CompressedTensorsLinearMethod`` with a
    W4A16 Marlin scheme -> a REAL int4 GEMV on Ampere (sm_86), head ~16 MB;
  * if the config IGNORES ``lm_head`` (or there is no quant config) -> the method
    resolves to ``None`` -> an unquantized bf16 head (~63 MB).
``re:.*lm_head`` matches whether the prefix is ``lm_head`` or
``language_model.lm_head`` (multimodal wrapper), so the int4 path is robust to the
wrapper rename. ``kept_size=12288`` satisfies all Marlin shape constraints
(12288 % 64 == 0, hidden 2560 % 128 == 0) and ``pad_vocab_size(12288, 64)`` is a
no-op, so the head output width stays exactly 12288 and the scatter-back is intact.

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
import sys
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


def _served_head_report(lm_head: torch.nn.Module, kept_size: int, hidden: int) -> dict:
    """Introspect the loaded lm_head to prove its served dtype + byte footprint.

    Robust to compressed-tensors' in-place Marlin repack (``process_weights_after_
    loading`` mutates ``weight_packed``/``weight_scale`` in place rather than
    renaming). We sum every parameter/buffer byte on the head module, so an int4
    head reports ~16 MB (packed int32 + bf16 scales) and a bf16 head ~63 MB.
    """
    tensors: dict[str, torch.Tensor] = {}
    for name, t in lm_head.named_parameters(recurse=True):
        tensors[name] = t.data if hasattr(t, "data") else t
    for name, t in lm_head.named_buffers(recurse=True):
        tensors.setdefault(name, t)

    total_bytes = 0
    breakdown = {}
    for name, t in tensors.items():
        if t is None:
            continue
        nbytes = t.numel() * t.element_size()
        total_bytes += nbytes
        # keep only the non-trivial tensors in the breakdown
        if t.numel() >= 1024:
            breakdown[name] = {
                "dtype": str(t.dtype),
                "shape": list(t.shape),
                "bytes": nbytes,
            }

    quant_method = type(getattr(lm_head, "quant_method", None)).__name__
    scheme = type(getattr(lm_head, "scheme", None)).__name__
    is_int4 = any("packed" in n or "qweight" in n for n in tensors)
    bf16_ref_bytes = kept_size * hidden * 2
    report = {
        "kept_size": kept_size,
        "hidden": hidden,
        "served_head_dtype": "int4-W4A16-marlin" if is_int4 else "bf16",
        "served_head_bytes": total_bytes,
        "served_head_MB": round(total_bytes / 1e6, 3),
        "bf16_reference_bytes": bf16_ref_bytes,
        "bf16_reference_MB": round(bf16_ref_bytes / 1e6, 3),
        "byte_cut_x_vs_bf16": round(bf16_ref_bytes / max(1, total_bytes), 3),
        "quant_method": quant_method,
        "scheme": scheme,
        "tensors": breakdown,
    }
    return report


class Gemma4ForCausalLMLMHead12k(Gemma4ForCausalLM):
    """Gemma4 text model with a row-pruned lm_head + scatter-to-full-vocab logits.

    Head dtype follows the checkpoint quant config: int4 W4A16 (Marlin) when the
    config targets ``re:.*lm_head``, else bf16.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = self.config
        kept = _load_kept_ids(vllm_config)
        kept_size = len(kept)
        full_vocab = config.vocab_size  # stays 262144

        # Rebuild the head + logits processor at the pruned width so the
        # checkpoint's pruned lm_head loads cleanly and the GEMM only touches
        # kept_size rows. Pass the body's quant_config so a checkpoint that targets
        # lm_head loads the head through the SAME compressed-tensors int4 Marlin
        # path as the body (a real int4 GEMV); a checkpoint that ignores lm_head
        # resolves to an unquantized bf16 head. Keep the same soft cap as parent.
        quant_config = vllm_config.quant_config
        self.lm_head = ParallelLMHead(
            kept_size,
            config.hidden_size,
            quant_config=quant_config,
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
        self._head_logged = False

    def _log_served_head(self) -> None:
        """One-time, post-load proof of the served head dtype + bytes."""
        try:
            report = _served_head_report(
                self.lm_head, self._kept_size, self.config.hidden_size
            )
        except Exception as exc:  # never let logging break serving
            print(f"[lmhead12k_int4head] served-head introspection failed: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return
        print(f"[lmhead12k_int4head] served head: {json.dumps(report)}",
              file=sys.stderr, flush=True)
        out = os.environ.get("SENPAI_HEAD_BYTES_LOG")
        if not out:
            model_dir = os.environ.get("MODEL_ID", "")
            if model_dir and Path(model_dir).is_dir():
                out = str(Path(model_dir) / "served_head_bytes.json")
        if out:
            try:
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                Path(out).write_text(json.dumps(report, indent=2))
            except Exception:
                pass

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if not self._head_logged:
            self._head_logged = True
            self._log_served_head()
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
