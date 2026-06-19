# PR #708 — Mixed-grid Marlin servability: the decisive source read

**Question (PR primary metric `mixed_grid_servable`):** Does vLLM 0.22.1rc1.dev307's
Marlin path read `group_size` **per-layer** (mixed-grid servable as one checkpoint)
or **model-level** (no per-module mixing without a custom config/kernel)?

**Environment:** `/tmp/senpai-venvs/5f4c623f772358a2` — vLLM `0.22.1rc1.dev307+g3e8afdf78`.
The locked submission's manifest pins `vllm==0.22.0`; the relevant plumbing
(compressed-tensors → Marlin) is identical across both (same files below).

The served model is **compressed-tensors W4A16** (`int4_g128_lmhead/serve.py` lines
2-8: vLLM auto-detects compressed-tensors from `config.json.quantization_config`
and repacks int4 → Marlin at load). So the Marlin path is reached via
`CompressedTensorsConfig` → `CompressedTensorsWNA16` → `MarlinLinearKernel`, NOT via
a `GPTQMarlinConfig` (that class does not exist in this build).

---

## Finding 1 — `group_size` is read PER-LAYER, not model-level. (servable=1)

`vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`

- `CompressedTensorsWNA16.__init__` **line 55**: `self.group_size = -1 if group_size is None else group_size` — each scheme instance carries **its own** group_size.
- `create_weights` **line 104**: `group_size=self.group_size` is passed into the per-layer `MPLinearLayerConfig`.
- **line 127**: `scales_and_zp_size = input_size // group_size` — the scale-tensor shape is computed **per layer** from that layer's group_size.
- **line 212**: `self.kernel = kernel_type(mp_linear_kernel_config, ...)` — a **per-layer** kernel instance is built with that layer's group_size.

The compressed-tensors **format** expresses this via `config_groups`: N groups, each
with its own `weights.group_size` and a `targets` list. The locked build already
uses TWO groups with DIFFERENT group_sizes — `int4_g128_lmhead/build_quant.py`
lines 161-173: `group_0` (body) at `--group-size`, `group_1` (lm_head) at
`--head-group-size`. Dispatch is per-layer:

`vllm/.../compressed_tensors/compressed_tensors.py`
- `get_scheme` **line 815** docstring (818-828): *"There can be N config_groups which each have a quantization scheme … use the quantization scheme corresponding to the matched target."*
- `get_scheme_dict` **line 864** → `find_matched_target(..., fused_mapping=self.packed_modules_mapping)` **line 885** matches each layer to its config_group.

**⇒ A per-module heterogeneous group_size (e.g. a `group_2` with the 48-module subset
at g32, `group_0` at g128) IS expressible on disk and IS dispatched per-layer at
serve time. `mixed_grid_servable = 1`.**

---

## Finding 2 — but FUSION blocks the subset's 8 attention modules. (subset_fully_servable=0)

vLLM fuses q/k/v into one `qkv_proj` and gate/up into one `gate_up_proj`:
`vllm/model_executor/models/gemma3n.py`
- **line 312** `self.qkv_proj = QKVParallelLinear(...)`, **line 242** `self.gate_up_proj = MergedColumnParallelLinear(...)`.
- `packed_modules_mapping` **lines 1094-1104**: `qkv_proj → [q_proj, k_proj, v_proj]`, `gate_up_proj → [gate_proj, up_proj]`.
- **line 485** `self.per_layer_input_gate = ReplicatedLinear(...)` — **standalone**, NOT in the packed mapping.

A **fused** layer gets exactly ONE scheme (one group_size). The fused-target
matcher does **not** enforce that the constituent shards share a scheme:
`vllm/.../compressed_tensors/utils.py`
- `_match_fused_layer` **lines 196-239**: expands `qkv_proj → [q,k,v]` paths, finds a target for each, and **line 239** `return unfused_matches[0] if all(unfused_matches) else None` — returns the **first shard's** target as long as **all** shards match *something*; it does **not** require them to match the **same** target.
- `should_ignore_layer` **lines 66-92** only raises (**line 88** *"requires all to use the same scheme"*) on the **ignore / not-ignore** boundary across shards — it does **not** check group_size consistency.

Consequence for the ubel #700 subset (3 q + 3 k + 2 v at g32, their fused partners
at g128):
1. `_match_fused_layer` would **silently** assign the whole `qkv_proj` the **first
   shard's** group_size (q's g32) — a mis-assignment, not an error.
2. Then `create_weights` builds the fused layer expecting `scales_and_zp_size =
   input_size // 32` (line 127), but the on-disk k/v `weight_scale` tensors have
   `input_size // 128` groups → **weight-load shape mismatch → load failure**. (The
   single fused qkv scale tensor cannot even be serialized with mixed per-shard
   group sizes — q,k,v concatenate along the output dim and must share one
   scale-column count.)

**⇒ The subset's q/k/v cannot be served as isolated g32 modules. Only the 40
standalone `per_layer_input_gate` (ReplicatedLinear) are independently g32-servable.
The attention members require WHOLE-qkv-block promotion (the whole layer's q+k+v →
g32), which also drags in the layer's k/v that ubel #700 did not target — so the
SERVED recovery ≠ the fake-quant 48-isolated-module recovery that ubel #702 prices
for quality.**

---

## Verdict

`mixed_grid_servable = 1` (per-layer group_size IS supported; mixed-grid servable at
**standalone-module** granularity), but `subset_fully_servable = 0`: the ubel #700
subset as-specified is **partially** servable — 40/48 modules (the standalone
per_layer_input_gate) ship cleanly; 8/48 (the fused q/k/v) do **not** ship as
isolated g32 and require whole-qkv-block promotion. This refines all three PR
verdicts: it is **not** `MIXED_GRID_UNSERVABLE` (group_size is not model-level
pinned), and it is `MIXED_GRID_SERVABLE_TAX_REALIZED` **only for the standalone
part**; the attention recovery's true ship cost is whole-qkv-block promotion (priced
in the op-bench).
