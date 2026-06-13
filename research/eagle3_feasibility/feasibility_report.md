<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# EAGLE-3 feature-export feasibility — vLLM 0.22.0 / `google/gemma-4-E4B-it`

**PR:** #15 · **Author:** fern · **Date:** 2026-06-13
**Question (binary):** Are the multi-layer intermediate hidden states that EAGLE-3
needs accessible from vLLM 0.22.0's Gemma-4 E4B forward pass?

## Verdict

**ACCESSIBLE.** Recommendation: **GO.**

vLLM 0.22.0 ships a complete, first-class EAGLE-3 feature-export path, and
`google/gemma-4-E4B-it` **already implements it**. Multi-layer intermediate
hidden states are accessible from the Gemma-4 E4B forward pass with **zero vLLM
patching, no forward hooks, and no model-class surgery**. The capture is
CUDA-graph compatible by design. The only remaining work to ship an EAGLE-3
drafter is training/sourcing the draft head and wiring the speculative config —
not feature export.

| field | value |
|---|---|
| `eagle3_hiddens_accessible` | **1** (yes) |
| access mechanism | built-in `SupportsEagle3` interface (no override) |
| model-class override effort | **0 hours** (already implemented) |
| EAGLE-3 fused aux layers (default) | **3** → `(2, 21, 39)` for the 42-layer E4B body |
| individually selectable residual points | 43 (embedding + 42 layer boundaries) |
| aux hidden state shape / dtype | `[num_tokens, 2560]` bf16, one per selected layer |
| CUDA-graph compatible | yes (capture path is aux-aware) |

All file references below are to the installed harness runtime
`vllm==0.22.0` at `/tmp/server-venv/lib/python3.12/site-packages/vllm`
(the same `vllm/vllm-openai` image the org-credit benchmark uses; manifest pins
`vllm==0.22.0`).

## 1. Which class loads for `google/gemma-4-E4B-it`?

The model's HF config (`config.json`, snapshot `fee6332c…`) declares
`architectures: ["Gemma4ForConditionalGeneration"]`, `model_type: gemma4`,
`text_config.num_hidden_layers: 42`, `text_config.hidden_size: 2560`,
`vocab_size: 262144`, and carries both a `vision_config` and an `audio_config`
(full multimodal).

vLLM registry resolves it to the **multimodal wrapper**:

- `registry.py:407` → `"Gemma4ForConditionalGeneration": ("gemma4_mm", "Gemma4ForConditionalGeneration")`
- `registry.py:118` → inner text model `"Gemma4ForCausalLM": ("gemma4", "Gemma4ForCausalLM")`

Class nesting:
`Gemma4ForConditionalGeneration` (gemma4_mm.py) → `.language_model`
= `Gemma4ForCausalLM` (gemma4.py) → `.model` = `Gemma4Model` (gemma4.py).

## 2. The EAGLE-3 interface (built into vLLM 0.22.0)

`vllm/model_executor/models/interfaces.py`:

- `EagleModelMixin` (lines 1285-1301): holds `aux_hidden_state_layers: tuple[int, ...]`,
  `_set_aux_hidden_state_layers(layers)`, and the collector
  `_maybe_add_hidden_state(aux, layer_idx, hidden_states, residual)`. The captured
  value is the **residual-stream value at the layer boundary**:
  `value = hidden_states + residual` (line 1298-1300) — i.e. post-block
  (post-MLP, post-residual-add).
- `SupportsEagle3(SupportsEagleBase, Protocol)` (lines 1334-1392): declares
  `supports_eagle3: ClassVar[Literal[True]]`, `set_aux_hidden_state_layers(layers)`,
  and `get_eagle3_default_aux_hidden_state_layers()`.
- The default-layer formula (line 1392): `(2, num_layers // 2, num_layers - 3)` —
  the canonical EAGLE-3 **[low, mid, high]** triple. For E4B's 42 layers →
  **(2, 21, 39)**.
- `supports_eagle3(model)` (lines 1403-1406) is the runtime gate used by the engine.

## 3. Gemma-4 E4B already implements the interface

- `gemma4_mm.py:917-923` — `class Gemma4ForConditionalGeneration(nn.Module,
  SupportsMultiModal, SupportsPP, SupportsLoRA, SupportsEagle3)`.
- `gemma4.py:958` — `class Gemma4Model(nn.Module, EagleModelMixin)`.
- `gemma4.py:1318,1337-1339` — inside the decoder loop, `Gemma4Model.forward`
  collects aux states at layer 0 (embedding output) and after every layer.
- `gemma4.py:1354-1356` — returns `(hidden_states, aux_hidden_states)` when aux
  layers are set, else plain `hidden_states`.
- **Tuple propagation through both wrappers:**
  - `gemma4.py:1607-1610` — `Gemma4ForCausalLM.forward` returns `self.model(...)`
    as-is.
  - `gemma4_mm.py:1487-1496` — `Gemma4ForConditionalGeneration.forward` calls the
    inner `Gemma4Model` directly and returns its result as-is. (Its `-> IntermediateTensors`
    type hint is stale; the dynamic return is the tuple, which the runner unpacks.)

So the multi-layer hiddens reach the engine through the existing return contract.

## 4. The engine consumes them end-to-end (multi-layer fusion already wired)

`vllm/v1/worker/gpu_model_runner.py`:

- Setup: `_setup_eagle3_aux_hidden_state_outputs` (5204-5223) calls
  `get_eagle3_default_aux_hidden_state_layers()` then `set_aux_hidden_state_layers()`,
  and raises if the model does not `supports_eagle3` — Gemma-4 passes. Aux layer
  indices can also be overridden from the speculative config
  (`_get_eagle3_aux_layers_from_config`, 5225+).
- Gating: `use_aux_hidden_state_outputs` is set True specifically for
  `speculative_config.method == "eagle3"` (589-594), via
  `drafter.eagle3_use_aux_hidden_state`.
- Unpack: `hidden_states, aux_hidden_states = model_output` (4240-4242).
- **Fusion (the EAGLE-3 multi-layer feature fusion):**
  `target_hidden_states = torch.cat([h[...] for h in aux_hidden_states], dim=-1)`
  (4861-4987) — the 3 aux layers are concatenated along the feature dim and fed
  to the drafter. This is exactly EAGLE-3's input.

The drafter-side EAGLE-3 head architecture also already exists:
`vllm/model_executor/models/llama_eagle3.py` (the shared EAGLE-3 draft head, not
Llama-specific) plus `vllm/v1/spec_decode/eagle.py` / `eagle3_utils.py`.

## 5. CUDA-graph compatibility (the main "tricky" concern) — resolved

`vllm/v1/worker/gpu/cudagraph_utils.py:382-395`: the capture path is
**aux-aware**. When `use_aux_hidden_state_outputs` is True it unpacks
`(hidden_states, aux_hidden_states)` from the captured model output and
pre-allocates persistent buffers (`torch.empty_like`) for each aux tensor. Because
`use_aux_hidden_state_outputs` is set at drafter init (before capture), the graph
is captured *with* the aux outputs in its signature — shapes are baked
consistently, no recapture. PIECEWISE mode (375-378) handles outputs internally.
Returning the extra aux tuple therefore does **not** break cudagraph capture.

This matches upstream history: EAGLE-3 landed in vLLM Apr 2025 (#16937), with
torch.compile (#17211) and piecewise-cudagraph (#17504) follow-ups, and Gemma-4
EAGLE-3 support formalized Aug 2025 (#22642) — all well before the vLLM 0.22.0
release (2026-05-29).

## 6. Contrast with the existing MTP drafter

The current `google/gemma-4-E4B-it-assistant` MTP drafter
(`gemma4_mtp.py:Gemma4MTP`, registry `Gemma4MTPModel` → `gemma4_mtp`, driven by
`Gemma4Proposer`) consumes a **single** `hidden_states` tensor — the target's
**last** hidden state. EAGLE-3 needs *more*: the 3 intermediate aux layers. The
same `SupportsEagle3` interface that's already on the model supplies exactly that
richer set, so the upgrade from "last hidden only" to "multi-layer fusion" is a
config/training change, not an export change.

## 7. Empirical confirmation (local GPU probe)

`research/eagle3_feasibility/probe_eagle3_export.py` (single local A10G model
load, **no HF Job**) loads the real `google/gemma-4-E4B-it` and uses
`LLM.apply_model` to exercise the interface on the instantiated module.

**Result (confirmed on the real GPU-loaded model — `research/eagle3_feasibility/probe_result.json`):**

```json
{
  "model_class": "Gemma4ForConditionalGeneration",
  "supports_eagle3": true,
  "default_aux_layers": [2, 21, 39],
  "inner_model_class": "Gemma4Model",
  "inner_is_EagleModelMixin": true,
  "num_decoder_layers": 42,
  "inner_aux_layers_after_set": [2, 21, 39],
  "hidden_size": 2560,
  "synthetic_num_aux_collected": 3,
  "synthetic_aux_shapes": [[5, 2560], [5, 2560], [5, 2560]],
  "synthetic_any_nan": false,
  "has_vision_tower": true,
  "has_audio_tower": true
}
```

Every prediction from the source audit is confirmed on the live module: the real
`google/gemma-4-E4B-it` checkpoint instantiates `Gemma4ForConditionalGeneration`,
`supports_eagle3()` is True, the inner `Gemma4Model` is an `EagleModelMixin` with
42 layers, `set_aux_hidden_state_layers((2,21,39))` mutates the inner tuple
correctly, and the collector yields exactly **3** aux tensors of shape
`[T, 2560]` with no NaN. The model loaded in **15.3 GiB** on the A10G (bf16,
enforce_eager) with both vision and audio towers present (modalities intact). No
HF Job was used.

Environment notes (both harmless, both informative):
- `apply_model` first failed with the V1 IPC `TypeError: Object of type function
  is not serializable`; resolved by `VLLM_ALLOW_INSECURE_SERIALIZATION=1`
  (cloudpickle by-value path, `serial_utils.py:228-231`).
- The optional post-probe greedy `generate` then crashed at
  `gpu_model_runner.py:4266 (sample_hidden_states = hidden_states[logits_indices])`
  with `IndexError: tuple index out of range`. This is a **probe-ordering
  artifact, not a model bug**: the probe left `aux_hidden_state_layers` set, so
  the forward returned the `(hidden_states, aux_hidden_states)` tuple, but a plain
  (non-speculative) generate has `use_aux_hidden_state_outputs=False` and does not
  unpack it. In real EAGLE-3 operation the engine sets the aux layers *and*
  `use_aux_hidden_state_outputs=True` together (`gpu_model_runner.py:586-594`,
  `5204-5223`), so it unpacks correctly. The crash therefore *corroborates* that
  `set_aux_hidden_state_layers` genuinely flips the forward's return contract to
  the multi-layer tuple.

## 8. Effort estimate & recommendation

**Feature export: 0 hours of vLLM work** — it is already shipped and wired.

To actually ship an EAGLE-3 drafter (a *separate* follow-up PR), the remaining
work is:

1. Train or source a Gemma-4 EAGLE-3 draft head (multi-layer-fusion MTP head over
   the `(2, 21, 39)` aux features). The vLLM-side drafter architecture
   (`llama_eagle3.py`) already exists; this is a *training* cost, comparable to
   land's #9 wide-corpus drafter effort, not an engine cost.
2. Wire the speculative config: `method: "eagle3"`, draft checkpoint path, and
   (optionally) `eagle_aux_hidden_state_layer_ids` if we want non-default layers.

**Coupling to the linchpin:** serving validity (greedy-identity of EAGLE-3 spec
decode) is still gated on kanna's PR #5 outcome — whether int4 batched-verify
spec decode can be greedy-bit-exact in vLLM 0.22.0. That gate is orthogonal to
*feature accessibility* (this PR) and binds the eventual training run, not the
export answer.

**Recommendation: GO.** The highest-ceiling drafter's prerequisite (multi-layer
feature export) is free on vLLM 0.22.0 for Gemma-4 E4B. Assign the EAGLE-3
drafter training run as a follow-up, coupled to the kanna #5 greedy-validity
verdict; if int4-spec proves greedy-invalid, EAGLE-3 still needs the same
workaround path the rest of the drafter ladder needs, but the feature-export
prerequisite is unconditionally satisfied.
