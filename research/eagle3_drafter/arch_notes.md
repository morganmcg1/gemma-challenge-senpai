<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# EAGLE-3 draft-head architecture notes — vLLM 0.22.0 / `google/gemma-4-E4B-it`

**PR:** #16 · **Author:** fern · **Date:** 2026-06-13
**Scope:** Step 1 of the EAGLE-3 training pipeline — read the head architecture,
answer the wiring questions, and decide how to train it. Training + offline eval
only; no serving run, no HF Job (serving validity gated on kanna #5).

All vLLM file references are to the installed harness runtime at
`/tmp/server-venv/lib/python3.12/site-packages/vllm` (`vllm==0.22.0`), the same
image the org-credit benchmark uses.

---

## TL;DR / decisions

1. **The vLLM EAGLE-3 head (`Eagle3LlamaForCausalLM` in `llama_eagle3.py`) is an
   *inference* artifact.** Its attention layer is vLLM's paged `Attention`, which
   reads KV cache + attn-metadata from a global forward context and runs
   inference-only backend kernels (no autograd). You **cannot** backprop through
   it for training. → **We train a faithful plain-PyTorch reimplementation** of
   the same architecture (`Eagle3DraftHead`), with weight names/shapes chosen so
   the checkpoint is loadable by vLLM later (deployment is gated on kanna #5).
   This mirrors how the official EAGLE repo trains (plain PyTorch), then converts
   for serving.

2. **No Gemma-4 E4B EAGLE-3 checkpoint exists.** The only public Gemma-4 EAGLE-3
   head is `thoughtworks/Gemma-4-31B-Eagle3` (hidden 5120, fc `[15360, 5120]`) —
   shape-incompatible with E4B (hidden 2560, fc `[7680, 2560]`). → **Train from
   scratch.** We *do* initialize the draft `embed_tokens` and `lm_head` from the
   target's tied embedding table for faster early convergence (see §6).

3. **The draft head is built from *Llama* decoder layers, not Gemma.** vLLM's
   `llama_eagle3.py` imports `LlamaDecoderLayer` from `models.llama`. So the head
   uses standard RoPE + RMSNorm + GQA + SwiGLU — **no** Gemma QK-norm, logit
   soft-cap, sliding-window, or per-layer-embeddings. It only *consumes* Gemma's
   hidden states. This greatly simplifies a faithful reimplementation.

4. **Feature/embedding alignment at serving lags by one** (see §5): vLLM pairs
   target feature `h_i` with the embedding of token `x_{i+1}` to predict
   `x_{i+2}` (`llm_base_proposer.py:718`, `shift_input_ids=True`). The PR's Step-3
   pseudo-code uses the simpler same-index pairing. We store the full sequence and
   make the shift a flag; default to the **vLLM-faithful** alignment so the number
   is serving-relevant and the checkpoint is deployable.

---

## 1. The EAGLE-3 head (`vllm/model_executor/models/llama_eagle3.py`)

Two classes:

### `Eagle3LlamaForCausalLM` (the head wrapper)
- `self.model` = `LlamaModel` (the EAGLE-3 draft body, below).
- `self.lm_head = ParallelLMHead(draft_vocab_size, hidden_size)` — a **separate**
  head (not weight-tied to the target by default).
- `self.draft_id_to_target_id` — `[draft_vocab_size]` long buffer mapping draft
  token ids → target token ids. Defaults `draft_vocab_size = vocab_size = 262144`
  when not set; with the buffer all-zeros the mapping is identity (full vocab).
- `combine_hidden_states(h)` — the multi-layer fusion entry point:
  optional `input_norm` (one RMSNorm over `3*H`) and/or `fc_norm` (per-chunk
  RMSNorm over each `H`), then `self.model.fc` → `[T, H]`.
- `compute_logits(h)` → `logits_processor(lm_head, h)`, then scatters draft-vocab
  logits into the full `[*, vocab_size]` via `draft_id_to_target_id` (identity for
  full vocab).

### `LlamaModel` (the draft body)
- `self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)` — the
  draft's **own** input embedding (loadable from the target's table).
- `self.fc = ReplicatedLinear(fc_input_size, hidden_size, bias=False)` where
  `fc_input_size = target_hidden_size * num_aux_hidden_states = 2560 * 3 = 7680`.
  This is the EAGLE-3 multi-layer fusion: `[T, 7680] → [T, 2560]`.
- `self.layers` = `num_hidden_layers` × EAGLE `LlamaDecoderLayer` (canonically
  **1** layer).
- `self.norm = RMSNorm(hidden_size)` — final norm.
- Optional norms: `input_norm` (RMSNorm `7680`, when `norm_before_fc`),
  `fc_norm` (3× RMSNorm `2560`, when `fc_norm`).

### EAGLE `LlamaDecoderLayer` (subclass) — the key structural twist
- **First layer (`layer_idx == 0`) takes a `2*hidden_size` QKV input**: it
  concatenates the (normed) token embedding with the (normed) fused hidden state
  along the feature dim before attention:
  ```python
  embeds = self.input_layernorm(embeds)                 # RMSNorm(H) on token embeds
  hidden_states, residual = self._norm_after_residual(hidden_states)
      # residual = fused (pre-norm);  hidden_states = self.hidden_norm(fused)
  hidden_states = torch.cat([embeds, hidden_states], -1)  # [T, 2H = 5120]
  hidden_states = self.self_attn(positions, hidden_states)  # qkv_proj in = 5120
  hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
  hidden_states = self.mlp(hidden_states)
  return hidden_states, residual
  ```
  (`qkv_input_size = 2*H if layer_idx==0 else H`, line 53.) Subsequent layers, if
  any, take the normal `H` input.
- `hidden_norm` = extra `RMSNorm(H)` applied to the incoming fused hidden state.
- `norm_before_residual` flag toggles whether `hidden_norm` is applied before or
  after the residual is captured (Gemma-style draft configs set this); default is
  `_norm_after_residual` (residual = pre-norm fused).

### Data flow for one forward (serving and our training)
```
aux = [h^(2), h^(21), h^(39)]                 # 3 × [T, 2560]  (target features)
fused = cat(aux, dim=-1)                       # [T, 7680]
h0 = combine_hidden_states(fused)              # optional norms → fc → [T, 2560]
embeds = embed_tokens(input_ids)               # [T, 2560]
hidden, _ = model.forward(input_ids, positions, h0)   # 1 decoder layer + final norm
logits = compute_logits(hidden)                # [T, vocab]
```

---

## 2. Exact input tensor shapes expected by `forward`

`Eagle3LlamaForCausalLM.forward(input_ids, positions, hidden_states, inputs_embeds=None)`
→ `LlamaModel.forward`, which asserts `hidden_states.shape[-1] == input_embeds.shape[-1]`
(line 239). So:

| tensor | shape | dtype | note |
|---|---|---|---|
| `input_ids` | `[T]` | long | draft input token ids (embedded internally) |
| `positions` | `[T]` | long | position ids (RoPE) |
| `hidden_states` | `[T, 2560]` | model dtype | **post-`fc`** fused features (NOT the raw `[T, 7680]`) |

The raw `[T, 7680]` concat is consumed only by `combine_hidden_states`, which is
called *before* `forward`. Output: `(hidden_states[T,2560], aux_output[T,2560])`;
`compute_logits(hidden_states)` → `[T, draft_vocab]`.

---

## 3. How the fused features are consumed

**Direct concat + fc, then fed as the residual stream into one attention block.**
Not cross-attention, not an MLP-only adapter. Specifically:
- The 3 aux layers are concatenated (`dim=-1`) and projected `7680→2560` by `fc`
  (this *is* the EAGLE-3 multi-layer feature fusion).
- The fc output becomes the decoder layer's `hidden_states`; the token embedding
  is concatenated to it inside layer 0 (giving the `2*H` QKV input).
- The verifier-side fusion (`gpu_model_runner.py:4861-4987`) does the same
  `torch.cat([...], dim=-1)` over the 3 aux layers — confirmed in the PR #15
  feasibility report §4.

---

## 4. The aux layers `(2, 21, 39)` ↔ HF `output_hidden_states` index mapping

vLLM collects aux states in `Gemma4Model.forward` (`gemma4.py:1318,1337-1339`) via
`EagleModelMixin._maybe_add_hidden_state(aux, layer_idx, hidden_states, residual)`
(`interfaces.py:1291-1301`), capturing the **post-block residual-stream value**
`value = hidden_states + residual`:

- index `0` → embedding output (`residual is None`).
- index `k` (k≥1) → output of decoder layer `k-1` (`hidden_states + residual`
  captured right after `layers[k-1]`).

Default triple `(2, num_layers//2, num_layers-3) = (2, 21, 39)` for 42 layers
→ outputs of decoder layers `(1, 20, 38)`.

**HF `transformers` `output_hidden_states=True`** returns a tuple of length
`num_layers+1 = 43`: index `0` = embeddings, index `i` = input to layer `i` =
output of layer `i-1` (pre-final-norm for `i < 42`). Therefore HF
`hidden_states[i]` equals vLLM's collected aux at index `i` for all
`i ∈ {2, 21, 39}` (all `< 42`, so all pre-final-norm). 

**→ Corpus generation indexes HF `out.hidden_states[2]`, `[21]`, `[39]`.** This
is the most reliable path (standard, well-tested HF feature) and is the same
residual-stream definition vLLM exports. The probe in §8 confirms shapes/no-NaN.

---

## 5. Feature/embedding alignment (serving vs. PR pseudo-code)

vLLM's drafter input prep (`llm_base_proposer.py:701-795`) sets, for EAGLE
(`shift_input_ids=True`):
- `self.input_ids[:num_tokens-1] = target_token_ids[1:]`  (draft input token ids
  shifted left by one), and the last slot = the freshly sampled `next_token_ids`;
- `self.hidden_states[:num_tokens] = target_hidden_states`  (features unshifted).

So at draft slot `i`: input `(h_i, embed(x_{i+1}))` → predict `x_{i+2}`. The
**feature lags the input-token embedding by one position.** First draft step
during serving: `(h_{n-1}, embed(x_n)) → x_{n+1}` (last prompt feature + just
sampled token). This is the canonical EAGLE setup.

The PR Step-3 pseudo-code instead pairs `(h_i, embed(x_i)) → x_{i+1}` (same index).
That number is self-consistent but slightly *easier* (the feature already encodes
`x_i`, which is also handed in as the embedding) and does **not** match serving.

**Decision:** store the full `input_ids` and full aux features in the corpus;
training/eval build `(feature, input-token, label)` triples per a `--feature_shift`
flag. Default `feature_shift=1` (vLLM-faithful) so the reported tf-acceptance
predicts serving acceptance and the checkpoint is deployable. We can also report
the PR-literal (`shift=0`) number for comparability.

---

## 6. From scratch vs. pre-init; training-methodology decisions

(From the EAGLE-3 paper arXiv 2503.01840, the SafeAILab/EAGLE reference repo, and
a literature pass — see PR body.)

- **Init:** from scratch (no compatible checkpoint). Copy the target's tied
  embedding table into the draft `embed_tokens` (frozen) and into `lm_head`
  (init); this gives the head semantically meaningful token vectors immediately
  and a target-aligned output space, which is the single biggest early-convergence
  win for a small-corpus debug run. Gemma ties embeddings, so one matrix
  (`[262144, 2560]`) seeds both.
- **Loss:** the PR specifies hard cross-entropy of head logits vs. `next_token_ids`.
  We follow it for the debug run. (Official EAGLE-3 uses *soft* CE/KL against the
  target's full distribution and drops the EAGLE-1/2 smooth-L1 feature-regression
  term; storing the full 262k-way soft target for the corpus is infeasible here —
  soft-KD is logged as a follow-up.)
- **TTT / unroll K:** the PR's debug run is single-step teacher-forced (K=1), which
  is the standard pipeline-validation setting and matches the "teacher-forced
  acceptance" metric. Official training uses K=7 multi-step unrolling for
  serving-grade acceptance — that's the follow-up full run.
- **`norm_before_fc`:** default **on** (one `RMSNorm(7680)` before fc). Gemma aux
  layers (2 vs 38) have very different activation scales; normalizing conditions
  the fc input and stabilizes early training. The setting is baked into the saved
  config so a future vLLM load matches.
- **Vocab:** full 262144 (identity `draft_id_to_target_id`); no t2d/d2t. The
  `lm_head` `[2560, 262144]` dominates memory (~1.34 GB bf16) but fits the A10G.
- **Optimizer:** AdamW lr=1e-4, cosine, 100 warmup, weight-decay 0.1, betas
  (0.9, 0.95), grad-clip 1.0 (per official ds_config).

### Realistic acceptance expectations (debug scale)
200 samples × 512 tok ≈ 100k tokens, 1000 steps, K=1 → **~0.45–0.65** teacher-forced
top-1 accuracy is plausible (severely undertrained vs. the 0.75–0.85 of fully
trained heads). The diagnostic signals that actually matter at this scale:
monotone loss decrease after the first ~50 steps, and accuracy climbing well above
chance (1/262144 ≈ 4e-6). PR interpretation bands: `<0.50` underfit/broken,
`0.50–0.70` expected (schedule full run), `≥0.70` strong.

---

## 7. Weight-name mapping (for a vLLM-loadable checkpoint)

`Eagle3LlamaForCausalLM.load_weights` (lines 413-464) expects, after its remaps:
- `midlayer.*` → `model.layers.0.*` (the single decoder layer);
- bare names (e.g. `fc.weight`, `embed_tokens.weight`, `norm.weight`,
  `input_norm.weight`) are prefixed with `model.`;
- `lm_head.*` stays as-is; `d2t` → `draft_id_to_target_id`; `t2d` is skipped.

Our checkpoint saves keys under these names plus a `config.json` carrying
`draft_vocab_size`, `norm_before_fc`, `eagle_aux_hidden_state_layer_ids=[2,21,39]`,
`target_hidden_size=2560`. Exact vLLM-load verification is deferred to the
deployment PR (gated on kanna #5); names/shapes are matched best-effort now.

---

## 8. Probe confirmation (HF, live model)

`research/eagle3_drafter/probe_hf_hiddens.py` → `probe_hf.log` (single local A10G
load, no HF Job). Confirmed on the live model:

- Model class `Gemma4ForConditionalGeneration`; the 42-layer text tower is at
  `model.language_model` (located by walking `named_modules` for the module with
  42 `layers` + an `embed_tokens`). The corpus script reuses that locator.
- `embed_tokens.weight` = `[262144, 2560]` bf16; **`lm_head` is tied to
  `embed_tokens`** (`data_ptr` match) — so one matrix seeds both draft embed +
  lm_head.
- `output_hidden_states` returns **43** tensors; `hidden_states[2]/[21]/[39]` are
  `[1, 512, 2560]` bf16, **no NaN**, with std `1.91 / 1.67 / 2.48` and absmax
  `96.5 / 65.0 / 134.0` — the differing per-layer scales justify `norm_before_fc`.
- Config knobs: `hidden_size=2560`, `num_hidden_layers=42`, `num_attention_heads=8`,
  `num_key_value_heads=2`, `head_dim=256`, `intermediate_size=10240`,
  `rms_norm_eps=1e-6`, `vocab_size=262144`. `rope_theta`/`query_pre_attn_scalar`
  are not at the text-config top level (Gemma stores rope per layer-type); since
  the draft uses *Llama* layers, we set the draft head's own `rope_theta=1e6`
  (Gemma global default; irrelevant within ≤512 positions, baked into draft config).
- **Peak GPU mem for one `1×512` forward = 16.10 GB** (≈15 GB weights + ~1 GB
  activations) on the 23 GB A10G → corpus gen runs at batch 1–4 comfortably.
