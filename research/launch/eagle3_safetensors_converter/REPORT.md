# EAGLE-3 → safetensors converter (PR #333)

CPU-only, deterministic converter that turns the 15-tensor `Eagle3DraftHead.state_dict()`
(the fern #34 `gua9x68j` / `56ksyxgw` in-repo `{2,21,39}` head, saved as `model_best.pt`)
into a **vLLM-loadable two-file candidate dir** (`model.safetensors` + `config.json`),
plus a `sha256` and a human Approval-request snippet.

It closes ubel **#328** audit blockers **2** (`weight_key_namespace_mismatch`) and **3**
(`container_format`) into ONE reviewed command the human runs before the single #319
measured-read launch.

> **0 GPU. NO model forward. NO publish / NO bucket write / NO manifest change / NO HF
> job / NO submission / NO served-file change.** Publishing the artifact to `DRAFTER_BUCKET`
> and editing `manifest.json` stays HUMAN-owned (audit blocker 1 `no_published_path`). This
> card only produces the reviewed converter + a synthetic-shape dry run. Because the
> `_candidate/` dir is a regenerable ~2.9 GB local artifact it is **gitignored**; this
> REPORT embeds everything needed to review the PR from the diff alone.

## How to run

Self-test (the committed evidence; **0 GPU, no checkpoint present**, synthetic zero-tensors
at the exact #328 inventory shapes/dtypes):

```bash
cd target/ && CUDA_VISIBLE_DEVICES="" python \
  research/launch/eagle3_safetensors_converter/convert_eagle3_to_safetensors.py \
  --synthetic-shapes --self-test \
  --wandb_group eagle3-safetensors-converter --wandb_name ubel/eagle3-safetensors-converter
```

Convert the REAL head (HUMAN, after locating the local `.pt`):

```bash
python research/launch/eagle3_safetensors_converter/convert_eagle3_to_safetensors.py \
  --in research/eagle3_drafter/checkpoints/<run>/model_best.pt
```

- **Primary** metric `converter_self_test_passes` = **1** (all 19 synthetic-mode checks hold).
- **Test** metric `tensors_mapped_post_rename` = **15**.

## What it does (three mechanical, lossless steps + a loud assert)

1. **Finding-B rename** (#328 §2): strip leading `model.` from body keys; rename the single
   decoder layer `layers.0.` → the canonical EAGLE-3 `midlayer.`; keep `lm_head.*`; keep
   q/k/v + gate/up **separate** (vLLM fuses them via `stacked_params_mapping` at load).
2. **bf16 cast**: 13 body tensors fp32 → bf16; `embed_tokens`/`lm_head` are already bf16 →
   a uniform single-dtype `model.safetensors` (serving dtype; clean file, audit caveat C2).
3. **config emit**: a vLLM EAGLE-3 `config.json` (`model_type:"llama"` for AutoConfig
   survival + a nested `eagle_config` backstop), matching #328 §3 + ubel #299 arch.

Then it **ASSERTS** — by porting vLLM 0.22.1rc1's `Eagle3LlamaForCausalLM.load_weights` /
`LlamaModel.load_weights` name+shape contract — that every published tensor lands on a real
vLLM parameter/shard with an exactly matching shape. A bad export fails **loudly here, on
CPU**, not at the one HF launch.

## Rename + vLLM mapping table (all 15 land, 0 unexpected, 0 shape mismatch)

| saved key (`state_dict`) | published (renamed) | vLLM internal param | shard | shape | status |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | `embed_tokens.weight` | `model.embed_tokens.weight` | — | `[262144, 2560]` | ok |
| `model.input_norm.weight` | `input_norm.weight` | `model.input_norm.weight` | — | `[7680]` | ok |
| `model.fc.weight` | `fc.weight` | `model.fc.weight` | — | `[2560, 7680]` | ok |
| `model.layers.0.self_attn.q_proj.weight` | `midlayer.self_attn.q_proj.weight` | `model.layers.0.self_attn.qkv_proj.weight` | `q` | `[2048, 5120]` | ok |
| `model.layers.0.self_attn.k_proj.weight` | `midlayer.self_attn.k_proj.weight` | `model.layers.0.self_attn.qkv_proj.weight` | `k` | `[512, 5120]` | ok |
| `model.layers.0.self_attn.v_proj.weight` | `midlayer.self_attn.v_proj.weight` | `model.layers.0.self_attn.qkv_proj.weight` | `v` | `[512, 5120]` | ok |
| `model.layers.0.self_attn.o_proj.weight` | `midlayer.self_attn.o_proj.weight` | `model.layers.0.self_attn.o_proj.weight` | — | `[2560, 2048]` | ok |
| `model.layers.0.mlp.gate_proj.weight` | `midlayer.mlp.gate_proj.weight` | `model.layers.0.mlp.gate_up_proj.weight` | `0` | `[10240, 2560]` | ok |
| `model.layers.0.mlp.up_proj.weight` | `midlayer.mlp.up_proj.weight` | `model.layers.0.mlp.gate_up_proj.weight` | `1` | `[10240, 2560]` | ok |
| `model.layers.0.mlp.down_proj.weight` | `midlayer.mlp.down_proj.weight` | `model.layers.0.mlp.down_proj.weight` | — | `[2560, 10240]` | ok |
| `model.layers.0.input_layernorm.weight` | `midlayer.input_layernorm.weight` | `model.layers.0.input_layernorm.weight` | — | `[2560]` | ok |
| `model.layers.0.hidden_norm.weight` | `midlayer.hidden_norm.weight` | `model.layers.0.hidden_norm.weight` | — | `[2560]` | ok |
| `model.layers.0.post_attention_layernorm.weight` | `midlayer.post_attention_layernorm.weight` | `model.layers.0.post_attention_layernorm.weight` | — | `[2560]` | ok |
| `model.norm.weight` | `norm.weight` | `model.norm.weight` | — | `[2560]` | ok |
| `lm_head.weight` | `lm_head.weight` | `lm_head.weight` | — | `[262144, 2560]` | ok |

Fused-target shapes vLLM expects: `qkv_proj [3072, 5120]` (= q 2048 + k 512 + v 512),
`gate_up_proj [20480, 2560]` (= gate 10240 + up 10240); vocab `262144` is divisible by 64.

## Emitted `config.json`

```json
{
  "architectures": ["Eagle3LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 2560,
  "intermediate_size": 10240,
  "num_hidden_layers": 1,
  "num_attention_heads": 8,
  "num_key_value_heads": 2,
  "head_dim": 256,
  "vocab_size": 262144,
  "draft_vocab_size": 262144,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "max_position_embeddings": 131072,
  "norm_before_fc": true,
  "target_hidden_size": 2560,
  "num_aux_hidden_states": 3,
  "eagle_aux_hidden_state_layer_ids": [2, 21, 39],
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "eagle_config": {
    "norm_before_fc": true,
    "target_hidden_size": 2560,
    "num_aux_hidden_states": 3,
    "eagle_aux_hidden_state_layer_ids": [2, 21, 39]
  }
}
```

`model_type:"llama"` keeps the config parseable by `AutoConfig`/`LlamaConfig`; the custom
EAGLE-3 fields (and the nested `eagle_config` backstop) survive that round-trip (verified in
the self-test through the real `transformers.LlamaConfig`, audit caveat C1).

## Self-test: 19 / 19 checks pass

| check | pass |
|---|---|
| `source_has_15_tensors` | ✅ |
| `all_15_map_post_rename` | ✅ |
| `zero_unexpected_post_rename` | ✅ |
| `zero_shape_mismatch_post_rename` | ✅ |
| `covers_all_vllm_load_targets` | ✅ |
| `no_missing_targets` | ✅ |
| `raw_state_dict_not_loadable_as_is` | ✅ |
| `safetensors_round_trips_keys` | ✅ |
| `safetensors_round_trips_shapes` | ✅ |
| `uniform_bf16_after_cast` | ✅ |
| `nan_clean` | ✅ |
| `config_parses_through_autoconfig` | ✅ |
| `config_has_required_fields` | ✅ |
| `config_nested_eagle_config_present` | ✅ |
| `architectures_is_eagle3` | ✅ |
| `vocab_divisible_by_64` | ✅ |
| `fused_shard_shapes_consistent` | ✅ |
| `sha256_deterministic_across_runs` | ✅ |
| `candidate_is_two_file_dir` | ✅ |

## Strongest evidence: meta-device check vs the REAL module

The hand-written `SOURCE_INVENTORY` is not trusted on faith — it is checked against the real
`Eagle3DraftHead` (`scripts/drafter/train_eagle3.py`) instantiated on the **meta device**
(0 memory, 0 GPU). Its actual `state_dict()` exactly matches the inventory, and the rename
lands every tensor on a vLLM target:

```
REAL_N_TENSORS: 15   INVENTORY_N: 15
KEYS_EQUAL: True   ONLY_IN_REAL: []   ONLY_IN_INVENTORY: []   SHAPE_MISMATCH: {}
POST_RENAME  n_ok=15  n_unexpected=0  n_shape_mismatch=0  covers_all=True  missing=[]
```

So the contract is validated against the producer module itself, not just a transcribed shape list.

## Determinism

Sorted keys + zeroed/static safetensors metadata → a stable file hash. The synthetic dry-run
`model.safetensors` sha256 is identical across runs and processes:

```
74b864a281ec19e1b74e0536eac2fdbc825c8f3639ac118e99d8b7894a65f1dc  model.safetensors
```

(The real-weight run will print its own sha256 — that value is what the human pins into
`DRAFTER_SHA256`.)

## Candidate dir (gitignored — regenerable)

```
research/launch/eagle3_safetensors_converter/_candidate/      # gitignored, ~2.9 GB
├── model.safetensors            # uniform-bf16, 15 tensors, vLLM-loadable
├── model.safetensors.sha256
├── config.json
└── APPROVAL_REQUEST.md          # the human Approval-request snippet (below)
```

Committed in the diff: `convert_eagle3_to_safetensors.py`, `_results.json` (machine record),
`.gitignore`, and this `REPORT.md`.

## Human-owned next steps (NOT done here — audit blocker 1 `no_published_path`)

1. Re-run this converter on the REAL `model_best.pt` (`--in <path>`).
2. Publish the two-file `_candidate/` dir to
   `hf://buckets/gemma-challenge/gemma-senpai/weights/eagle3-inrepo-251head/`.
3. Set `DRAFTER_SHA256` to the printed sha256 in
   `submissions/fa2sw_precache_kenyan/manifest.json` and apply the ubel #322 §1 manifest
   delta (`method` mtp→eagle3, `SPECULATIVE_CONFIG.model` + `LOCAL_DRAFTER_DIR` +
   `DRAFTER_BUCKET` → the eagle3 dir; keep `num_speculative_tokens: 7`).
4. Local AWS A10G smoke (no HF Job): boot `serve.py`, confirm 0 missing/unexpected tensors,
   `/v1/models` 200, greedy token-identity, tiny PPL sane (ubel #322 §0 step 5).
5. Only after the smoke PASS + human approval on Issue #319: launch exactly one `a10g-small`
   measured read.

This converter adds **0 TPS** and changes **no served file**. It de-risks the #319 launch by
turning head conversion into one reviewed command; a name/shape slip fails loudly on CPU
here, and PPL/greedy risk is deferred to the human smoke (step 4).
