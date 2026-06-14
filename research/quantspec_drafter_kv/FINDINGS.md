# QuantSpec drafter-KV premise-check (PR #121)

**Question:** Does the deployed MTP drafter maintain a **SEPARATE** KV cache
(quantizable → QuantSpec INT4 drafter-KV lever live) or **SHARE** the verify
(target) model's KV (no separate drafter-KV → lever moot)?

**Verdict: SHARED. The MTP drafter allocates ZERO KV cache of its own.**
`drafter_kv_separate_bool = False` → **RED / moot → CLOSE the QuantSpec
drafter-KV lane.** `quantspec_drafter_kv_net_wall_tps_pct = 0.00` (lever does not
apply to our architecture).

This is a pure CPU read of the deployed serving stack
(`submissions/int4_mtp_batchinv` + pinned `vllm==0.22.0`). No model load, no GPU.

---

## Deployed stack

`submissions/int4_mtp_batchinv` serves the int4 W4A16 Gemma-4-E4B **target** with
the `gemma4_assistant` (`google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant`)
**drafter**, which vLLM resolves to `Gemma4MTPModel` and runs as a speculative
proposer (`NUM_SPECULATIVE_TOKENS=6`). serve.py itself names the drafter "a
lightweight **Q-only KV-shared** decoder" (`serve.py:6-8`). The code trace below
pins that claim against the actual `vllm==0.22.0` implementation.

## Code evidence — the drafter has no K/V projections and owns no KV cache

**1. Module contract (`vllm/model_executor/models/gemma4_mtp.py:5-8`):**
> "The Gemma4 assistant model is a lightweight decoder that **shares KV cache
> with the target (backbone) model**. All assistant decoder layers are
> KV-shared: they **only have Q projections (no K/V projections or norms), and
> read K/V from the target model's cache at runtime**."

**2. `Gemma4MTPAttention` builds Q only — there is literally no K or V to cache**
(`gemma4_mtp.py:148-232`). The constructor creates `q_proj`, `o_proj`, `q_norm`
and the `Attention` module — but **no `k_proj`, no `v_proj`, no k/v norms**. The
attention is flagged `self.is_kv_shared_layer = True` (`:221`).

**3. The drafter's attention feeds DUMMY K/V and reads the target's cache**
(`gemma4_mtp.py:234-259`). `forward()` computes only `q` from `q_proj`, then:
```python
# Attention reads K/V from the target's cache via KV sharing;
# these dummy tensors are never consumed but required by the API.
kv_dummy = torch.empty(num_tokens, self.num_kv_heads * self.head_dim, ...)
attn_output = self.attn(q, kv_dummy, kv_dummy)
```
The K/V arguments are throwaway zeros; the real K/V come from the target layer's
cache via the KV-sharing wiring.

**4. The proposer wires each draft layer to a TARGET layer's cache**
(`vllm/v1/spec_decode/gemma4.py`). Module docstring (`:5-7`): "all its attention
layers **share KV cache with the target model via cross-model KV sharing**."
`_setup_gemma4_kv_sharing()` (`:275-336`) maps every draft attention layer to a
target attention layer and sets
`attn.kv_sharing_target_layer_name = "model.layers.{target_idx}.self_attn.attn"`
(`:329`).

**5. The clincher — a layer with `kv_sharing_target_layer_name` set allocates NO
KV cache** (`vllm/v1/worker/gpu_model_runner.py:7304-7316`,
`get_kv_cache_spec`):
```python
if isinstance(attn_module, Attention) and (
    kv_tgt_layer := attn_module.kv_sharing_target_layer_name
):
    # The layer doesn't need its own KV cache and will use that of the
    # target layer. We skip creating a KVCacheSpec for it, so that KV cache
    # management logic will act as this layer does not exist, and doesn't
    # allocate KV cache for the layer.
    self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
    continue
```
Because the proposer sets `kv_sharing_target_layer_name` on **every** draft
attention layer, **every** draft layer is skipped in KV-cache-spec generation →
the drafter is allocated **0 bytes** of KV cache. There is exactly **one** KV
cache in the system: the target's.

## Why the lever's "safe by construction" claim also collapses

The hypothesis assumed quantizing the **drafter's** KV to INT4 is greedy-safe by
construction ("verify re-checks every token in full precision; quantizing the
DRAFT path changes only which tokens are proposed"). That argument requires a
**separate** draft-only cache. There is none. The bytes the drafter reads during
attention **are the target/verify KV cache** (same physical tensor). So:

- There is no "drafter KV" to quantize independently.
- Quantizing the cache the drafter reads == quantizing the **target KV cache** ==
  a *different* lever (target KV-cache dtype, listed separately under program.md
  Numerics) that **also changes the verify model's attention inputs** → real
  PPL / greedy-identity risk. The "only changes proposals" safety argument does
  **not** hold for a shared cache.

## Sizing (for completeness)

- Separate drafter-KV bytes per decode step (bf16): **0** (no allocation).
- INT4 saving on a separate drafter-KV: **0** of the ~7% drafter slice.
- The drafter's ~7% BW-bound decode slice (denken #75/#77) is dominated by its
  **weight** reads (q/o/MLP/embeddings + the centroid `lm_head`) re-read once per
  draft step × K steps, **not** a separate KV cache. KV reads the drafter issues
  hit the single shared target cache (~7.6 GiB target KV at 4k ctx), which is
  already the verify path's cache.

## Gate decision

- **RED / moot.** Shared KV → no separate drafter-KV → the QuantSpec INT4
  drafter-KV lever **does not apply** to our MTP architecture. **CLOSE the lane.**
- `drafter_kv_separate_bool = False`; `quantspec_drafter_kv_net_wall_tps_pct = 0.00`.

## Suggested follow-ups (NOT implemented — out of scope for this premise-check)

- If KV-cache quantization is still wanted, it is the **target KV-cache dtype**
  lever (single shared cache, ~7.6 GiB). It must be gated on the full verify-path
  PPL + greedy-identity contract — it is **not** a free, safe-by-construction
  drafter-only win. Distinct PR.
- The drafter's BW-bound 7% is a **weight-read** slice. Drafter weight-precision
  / read-amortization levers (already explored elsewhere) are the relevant lane,
  not KV quantization.
