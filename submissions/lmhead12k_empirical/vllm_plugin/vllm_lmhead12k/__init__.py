"""vLLM general-plugin: register the row-pruned (lmhead12k) Gemma4 text model.

vLLM calls ``load_general_plugins()`` in EVERY process (the API-server launcher
AND each V1 ``EngineCore`` subprocess). The async OpenAI server forces
multiprocessing, so an in-process ``ModelRegistry.register_model()`` in serve.py
would never reach the worker that actually builds the model. This entry-point
function runs in every process and registers the custom architecture there.

We register by STRING path (lazy import) so that importing this package does NOT
import torch/CUDA before vLLM has set up its (possibly forked) worker process.
"""
from __future__ import annotations


def register() -> None:
    """Override the built-in ``Gemma4ForCausalLM`` with the pruned-head subclass.

    The served checkpoint is a multimodal ``Gemma4ForConditionalGeneration``; its
    ``language_model`` is built via ``init_vllm_registered_model(architectures=
    ["Gemma4ForCausalLM"])`` which resolves through ``ModelRegistry``. Overriding
    that name makes the multimodal wrapper pick up our pruned text model while the
    vision/audio towers and all multimodal plumbing stay stock.
    """
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Gemma4ForCausalLM",
        "vllm_lmhead12k.model:Gemma4ForCausalLMLMHead12k",
    )
