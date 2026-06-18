# PR#670 Arm A — `/tmp/qat-assistant` characterization (publishability + legitimacy gate)

All facts below are in-scope: own-branch artifacts, the public Google Hub drafter,
and `submissions/fa2sw_nonspec_int4/manifest.json` (my own branch). No cross-student
bucket/commit was read.

## 1. Architecture / size / dtype (qat-assistant vs stock Hub drafter)

Both drafters are the **same MTP speculative-drafter architecture** — they differ only
in (i) weights and (ii) the `centroid_intermediate_top_k` runtime knob.

| field | stock Hub (`google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant`) | `/tmp/qat-assistant` (deployed) |
|---|---|---|
| architectures | `Gemma4AssistantForCausalLM` | same |
| model_type | `gemma4_assistant` | same |
| num_hidden_layers (text_config) | 4 | 4 |
| hidden_size / backbone_hidden_size | 256 / 2560 | 256 / 2560 |
| intermediate_size | 2048 | 2048 |
| vocab_size | 262144 | 262144 |
| num_centroids | 2048 | 2048 |
| tie_word_embeddings | True | True |
| **centroid_intermediate_top_k** | **32 (native)** | **64 (deployed)** |
| param count | 78,779,908 (78.78M), 50 tensors | identical count |
| dtype | BF16 (+ I64 index buffer); unquantized | same |
| on-disk model.safetensors | 152 MiB | 152 MiB |
| finetune metadata | none | `finetune: ft-v1/epoch_001.pt` |

It is a tiny 78.78M-param head, **not** the ~4B int4 target — it cannot be a copy of
the target. It is a bona-fide speculative *proposer*.

## 2. Provenance

- **stock-topk32 == the publishable Google artifact, byte-exact.** `/tmp/stock-topk32/model.safetensors`
  sha256 `9d0e2053067590cae9a8f4fcc57eefbe20dff90599360458ebd451e5cb5c947d` **equals the Hub blob**
  (`~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-q4_0-unquantized-assistant/.../model.safetensors`,
  same sha). The Hub config ships `centroid_intermediate_top_k=32`. So **stock@topk32 is the exact
  drafter the official harness loads**, and it is freely publishable (it is Google's).
- **top_k=64 is a deployment-time runtime knob, not native to any weight set.** `top_k` is an
  inference config field, not a trained parameter. The stock Google drafter ships 32; the deployed
  local dir was set to 64. So a top_k=64 lever applies equally to the *publishable stock Google drafter*.
- **The ft-v1 weights are kenyan-duma's checkpoint.** `/tmp/qat-assistant/model.safetensors`
  sha256 `ed159e334999fd6b5f2d0dbad026346d4efac89eb7c6f55c5cdb042eca5dd18e` **exactly equals**
  `DRAFTER_SHA256` in my own `submissions/fa2sw_nonspec_int4/manifest.json`, whose
  `DRAFTER_BUCKET=hf://buckets/gemma-challenge/gemma-kenyan-duma/weights/drafter-ft/ft-v1-epoch_001`.
  Safetensors `__metadata__` = `{'finetune': 'ft-v1/epoch_001.pt'}`. So the *retrain* weights are
  **kenyan-duma's** ("kduma1") ft-v1-epoch_001.
- **Reproducibility / `/tmp` ephemerality.** `/tmp/qat-assistant` is per-pod ephemeral. The only
  reproducible source for the *ft-v1 weights* is kenyan-duma's bucket (cross-student) or the
  out-of-scope retrain recipe (commit `4d65412`, `wide_drafter`, marked "EXCLUDED by isolation").
  The *stock* and *top_k* variants ARE trivially reproducible in-scope (download Google drafter,
  edit one config field).

## 3. Legitimacy

- **Cannot change emitted tokens.** `submissions/int4_mtp_batchinv/serve.py:8-10`: at temperature=0
  vLLM's rejection sampler short-circuits to target-argmax, so decode stays token-identical to plain
  greedy AR of the int4 target *regardless of which drafter is used*. The drafter only affects
  **acceptance length (speed)**, never quality/identity. A drafter therefore cannot game the quality
  (PPL / greedy-identity) metric.
- **Not the target, not a copy.** 78.78M-param head vs ~4B int4 body.
- **Eval-training fairness flag (honest).** Because a drafter only affects *speed*, the one fairness
  risk for the speed metric is a drafter overfit to the **eval prompts** (inflated acceptance that
  would not generalize). The ft-v1 *retrain recipe is out-of-scope* so I cannot fully audit its
  training corpus from in-scope evidence. This is mitigated two ways: (a) the larger-N / subsample
  robustness check (Arm B) tests whether the edge is a few-prompt artifact; (b) if the edge is mostly
  the **top_k knob** (Arm B 2×2), the lever applies to the clean stock Google drafter and the
  eval-training concern is moot. Flagged for the verdict, not silently assumed clean.

## 4. Guards (drafter-independent #319 byte gate + PPL)

The #319 byte-exact greedy gate and PPL ≤ 2.42 are **structurally drafter-independent**:
`serve.py:34-63` — under `SENPAI_REFERENCE_MODE` the submission forces `num_speculative_tokens=0`
(drafter OFF, plain int4 M=1 AR), which is the exact-greedy reference the gate compares against.
The re-gate never loads a drafter, and at temp=0 the spec path is token-identical anyway. So
`break_rate=0` and PPL=2.019 from the locked `int4_g128_lmhead` anchor carry over unchanged for
every drafter arm. `analysis_only=true`, `official_tps=0`, `fires=false` throughout this card.
