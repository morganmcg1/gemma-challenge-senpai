"""Backport of the upstream vLLM attention-group ``num_heads`` dedup fix.

Upstream ``vllm-project/vllm`` ``main`` (``vllm/v1/worker/gpu_model_runner.py``,
``GPUModelRunner.initialize_attn_backend`` /
``get_attn_backends_for_group``) adds the per-rank Q-head count
(``num_heads_q``) to the attention-group dedup key, so a speculative-decode
draft model with fewer attention heads than its target is placed in its own
attention group with its own metadata builder.

Without that change, the draft model's attention layers (here the
``gemma4_assistant`` MTP drafter: 4 Q-heads) are grouped together with the int4
target's global-attention layers (8 Q-heads), because they share the same
attention backend class (``TRITON_ATTN``) and the same KV-cache spec. When the
metadata builder is created, ``get_num_attention_heads_from_layers`` in
``vllm/v1/attention/backends/utils.py`` asserts a single head count per group
and fails:

    AssertionError: All layers in one attention group must share num_heads;
    got {8, 4} for [...language_model...self_attn.attn,
                    ...draft_model.layers.0/1/2.self_attn.attn]

We pin ``vllm==0.22.0`` to match the official ``vllm/vllm-openai`` serving image
used by the benchmark harness, which predates the upstream fix. Rather than bump
the engine off the harness image, we transplant the exact upstream change as a
runtime monkeypatch on ``GPUModelRunner.initialize_attn_backend``.

The change is the upstream diff and nothing more: the dedup key becomes
``(full_cls_name, layer_kv_cache_spec, num_heads_q)`` and ``AttentionGroupKey``
gains ``num_heads_q``. For non-speculative serving the patch is a no-op -- every
attention group already has a uniform head count, so adding ``num_heads`` to the
key yields the identical grouping (and therefore identical cudagraph capture and
identical numerics). It only ever *splits* an otherwise-mixed group.

Reference: vllm-project/vllm ``main``, ``vllm/v1/worker/gpu_model_runner.py``,
``AttentionGroupKey`` / ``get_attn_backends_for_group``.
"""

from __future__ import annotations

from typing import NamedTuple

# Marker set on GPUModelRunner once patched, so apply() is idempotent across the
# multiple processes / repeated plugin loads that touch the same class object.
_PATCH_FLAG = "_int4_mtp_drafter_num_heads_patch_applied"


def apply(gmr) -> bool:
    """Patch ``gmr.GPUModelRunner.initialize_attn_backend`` in place.

    ``gmr`` is the imported ``vllm.v1.worker.gpu_model_runner`` module. All
    helper symbols are resolved from that live module so we never guess import
    paths against a specific vLLM version. Returns ``True`` if the patch was
    applied, ``False`` if it was already present.
    """
    runner_cls = gmr.GPUModelRunner
    if getattr(runner_cls, _PATCH_FLAG, False):
        return False

    # Resolve every free name from the module that owns the original method.
    AttentionLayerBase = gmr.AttentionLayerBase
    get_layers_from_vllm_config = gmr.get_layers_from_vllm_config
    defaultdict = gmr.defaultdict
    create_fast_prefill_custom_backend = gmr.create_fast_prefill_custom_backend
    UniformTypeKVCacheSpecs = gmr.UniformTypeKVCacheSpecs
    check_attention_cp_compatibility = gmr.check_attention_cp_compatibility
    AttentionGroup = gmr.AttentionGroup

    class AttentionGroupKey(NamedTuple):
        attn_backend: object
        kv_cache_spec: object
        # Splits on per-rank Q-head count in addition to backend + spec, so a
        # spec-decode draft with fewer heads than its target gets its own group.
        num_heads_q: int

    def initialize_attn_backend(self, kv_cache_config, is_profiling: bool = False):
        assert len(self.attn_groups) == 0, (
            "Attention backends are already initialized"
        )

        def get_attn_backends_for_group(kv_cache_group_spec):
            layers = get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,
                kv_cache_group_spec.layer_names,
            )
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            for layer_name in kv_cache_group_spec.layer_names:
                attn_backend = layers[layer_name].get_attn_backend()

                if layer_name in self.kv_sharing_fast_prefill_eligible_layers:
                    attn_backend = create_fast_prefill_custom_backend(
                        "FastPrefill",
                        attn_backend,
                    )

                full_cls_name = attn_backend.full_cls_name()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                # The upstream fix: layers expose ``num_heads`` (non-attention
                # layer types fall back to 0, which is fine -- they never share a
                # KV cache group with attention layers).
                num_heads_q = getattr(layers[layer_name], "num_heads", 0)
                key = (full_cls_name, layer_kv_cache_spec, num_heads_q)
                attn_backends[key] = AttentionGroupKey(
                    attn_backend, layer_kv_cache_spec, num_heads_q
                )
                attn_backend_layers[key].append(layer_name)
            return (
                {attn_backends[k]: v for k, v in attn_backend_layers.items()},
                set(group_key.attn_backend for group_key in attn_backends.values()),
            )

        def create_attn_groups(attn_backends_map, kv_cache_group_id):
            attn_groups = []
            for group_key, layer_names in attn_backends_map.items():
                attn_group = AttentionGroup(
                    group_key.attn_backend,
                    layer_names,
                    group_key.kv_cache_spec,
                    kv_cache_group_id,
                )
                attn_groups.append(attn_group)
            return attn_groups

        attention_backend_maps = []
        attention_backend_list = []
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            attn_backends = get_attn_backends_for_group(kv_cache_group_spec)
            attention_backend_maps.append(attn_backends[0])
            attention_backend_list.append(attn_backends[1])

        # Resolve cudagraph_mode before initializing metadata builders. The set
        # of backend classes per KV cache group is unchanged by the head-count
        # split (draft and target global layers share TRITON_ATTN), so this
        # behaves exactly as upstream.
        self._check_and_update_cudagraph_mode(
            attention_backend_list,
            kv_cache_config.kv_cache_groups,
            is_profiling=is_profiling,
        )

        check_attention_cp_compatibility(self.vllm_config)

        for i, attn_backend_map in enumerate(attention_backend_maps):
            self.attn_groups.append(create_attn_groups(attn_backend_map, i))

    initialize_attn_backend.__name__ = "initialize_attn_backend"
    initialize_attn_backend.__qualname__ = "GPUModelRunner.initialize_attn_backend"
    runner_cls.initialize_attn_backend = initialize_attn_backend
    setattr(runner_cls, _PATCH_FLAG, True)
    return True
