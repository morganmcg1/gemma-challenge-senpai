"""Relax vLLM's over-broad ``fp8_e5m2`` KV-cache guard for *weight-only*
compressed-tensors checkpoints (no ``kv_cache_scheme``).

Why this patch exists
---------------------
On A10G (Ampere, sm_86) the *only* fp8 KV-cache dtype that the Triton compiler
can actually emit is ``fp8_e5m2`` (Triton's supported set on sm_86 is
``('fp8e4b15', 'fp8e5')``; ``fp8_e4m3`` maps to ``fp8e4nv``, which raises
``ValueError: type fp8e4nv not supported in this architecture`` during inductor
autotuning of the fused KV-store/RMSNorm kernel). So e5m2 is the one fp8 KV
variant that keeps the fully-compiled (cudagraph) decode path -- and therefore
the only one whose TPS is comparable to the bf16-KV bi0 baseline.

But vLLM blocks e5m2 first. ``vllm.model_executor.layers.attention.attention.
_init_kv_cache_quant`` does::

    if should_load_quant_weights(quant_method):          # True here, see below
        assert isinstance(quant_method, BaseKVCacheMethod)
        if layer.kv_cache_dtype == "fp8_e5m2":
            raise ValueError("fp8_e5m2 kv-cache is not supported with fp8 checkpoints.")

For a compressed-tensors model ``CompressedTensorsConfig.get_quant_method``
returns a ``CompressedTensorsKVCacheMethod`` for *every* ``Attention`` layer
unconditionally (compressed_tensors.py: ``if isinstance(layer, Attention):
return CompressedTensorsKVCacheMethod(self)``), so ``should_load_quant_weights``
is True even for a pure W4A16 *weight-only* checkpoint that declares **no** KV
scheme. The guard's error message ("...with fp8 checkpoints") names the case it
is actually defending against: a checkpoint that bakes in its own fp8 KV scales
(``kv_cache_scheme is not None``). Our target
(``google/gemma-4-E4B-it-qat-w4a16-ct``) has ``kv_cache_scheme is None`` --
``CompressedTensorsKVCacheMethod.validate_kv_cache_scheme(None)`` returns
immediately and ``create_weights`` initialises ``k_scale``/``v_scale`` to 1.0.
That is exactly the scale-free e5m2 storage path we want; the guard mis-fires.

What this patch does
--------------------
Wrap ``_init_kv_cache_quant`` so that, **only** when the requested dtype is
``fp8_e5m2`` *and* the checkpoint declares no ``kv_cache_scheme``, the
layer's ``kv_cache_dtype`` is temporarily presented as plain ``"fp8"`` for the
duration of the upstream call. The e5m2-specific ``raise`` is the *only*
consumer of ``layer.kv_cache_dtype`` inside that function -- ``create_weights``
reads ``quant_config.kv_cache_scheme`` and ``layer.num_kv_heads``, never the
dtype -- so the swap suppresses the false-positive raise and changes nothing
else. The real ``fp8_e5m2`` is restored before the function returns, so every
downstream consumer (the attention kernel's store/load) sees e5m2 as normal.

When the checkpoint *does* declare a kv_cache_scheme (a genuine fp8-KV
checkpoint), the dtype is left untouched and the upstream guard fires exactly as
before -- this patch is a strict no-op there, and a no-op for any non-e5m2 dtype.

Greedy-identity and PPL are validated downstream (research/validity/bi0_fp8kv/);
this patch only governs *whether the server boots*, not the numerics, which the
gate independently checks.
"""

import sys

_PATCH_FLAG = "_int4_mtp_fp8kv_e5m2_guard_relaxed"
_WRAPPER_FLAG = "_int4_mtp_e5m2_guard_wrapper"


def apply(attn_module) -> bool:
    """Wrap ``attn_module._init_kv_cache_quant`` to relax the e5m2 guard.

    ``attn_module`` is the imported
    ``vllm.model_executor.layers.attention.attention`` module. Returns ``True``
    if the patch was applied, ``False`` if it was already present or the target
    function is absent (then the guard simply behaves as upstream).
    """
    if getattr(attn_module, _PATCH_FLAG, False):
        return False

    orig = getattr(attn_module, "_init_kv_cache_quant", None)
    if orig is None:
        return False
    if getattr(orig, _WRAPPER_FLAG, False):
        setattr(attn_module, _PATCH_FLAG, True)
        return False

    def _init_kv_cache_quant(layer, quant_config, prefix):
        kv_scheme = getattr(quant_config, "kv_cache_scheme", None)
        dtype = getattr(layer, "kv_cache_dtype", None)
        if dtype == "fp8_e5m2" and kv_scheme is None:
            # Present plain "fp8" to the upstream guard so the e5m2-specific
            # raise is skipped; restore the real dtype before returning.
            layer.kv_cache_dtype = "fp8"
            try:
                return orig(layer, quant_config, prefix)
            finally:
                layer.kv_cache_dtype = dtype
        return orig(layer, quant_config, prefix)

    setattr(_init_kv_cache_quant, _WRAPPER_FLAG, True)
    attn_module._init_kv_cache_quant = _init_kv_cache_quant
    setattr(attn_module, _PATCH_FLAG, True)
    print(
        "[int4_mtp_fp8kv] _init_kv_cache_quant wrapped: relaxing the fp8_e5m2 "
        "KV guard for weight-only compressed-tensors checkpoints "
        "(kv_cache_scheme=None)",
        file=sys.stderr,
        flush=True,
    )
    return True
