<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Static publish + vLLM-load readiness audit — the {2,21,39} EAGLE-3 head

**PR:** #328 · **Author:** ubel · **Date:** 2026-06-15 · **W&B group:** `eagle3-ckpt-publish-readiness` (run `27y5xxce`)

**0-GPU operational-readiness card. NO build, NO HF job, NO publish action, NO
served-file change, NO GPU.** This audits whether the in-repo `gua9x68j` /
`56ksyxgw` EAGLE-3 head (fern #34) can be published and vLLM-loaded for the one
human-approved #319 Option-A measured read. It does **not** publish the checkpoint
or launch anything — the human owns the launch.

Reproduce (analytic, 0-GPU): `cd target/ && python research/launch/eagle3_ckpt_publish_readiness/audit.py`

---

## Verdict: 🟡 YELLOW — loadable with a documented, deterministic shim

`ckpt_publish_readiness_blocking_issues = 3` · `ckpt_publish_readiness_self_test_passes = 1`

The head is **not loadable as the raw `head.state_dict()`** (the saved keys are in
the `model.*` namespace and vLLM's loader double-prefixes them → hard load failure),
but **every one of its 15 tensors maps 1:1 to a vLLM `Eagle3LlamaForCausalLM`
parameter/shard with an exactly matching shape** once a deterministic, lossless
key-rename + safetensors export is applied. There is **no irreducible shape/arch
mismatch** (the train script was a faithful reimplementation with vLLM-loadable
names as the explicit goal — arch_notes §7). The three blockers are all mechanical
packaging/naming steps, each fixable with **no retrain and no architecture change**.

This turns ubel #322 §0's "vLLM-load UNVERIFIED" unknown into a concrete go/fix
checklist: the verdict is YELLOW, the shim is specified below, and the only
remaining unknowns are the four "verify-at-smoke" caveats in §5 — none of which can
be closed without the (post-publish) local A10G smoke the human runs as the last
launch gate.

---

## §1 — Artifact inventory (deliverable 1)

| field | value |
|---|---|
| **head** | `gua9x68j` (W&B eval run) / `56ksyxgw` (train run) — fern #34 benchmark-matched reasoning head |
| **provenance** | warm-started from #25 `full_20k/model_best.pt`; `[2,21,39]` aux fusion; `feature_shift=1`; best==final @ step 4500; tf_acc 0.7617, native accept/step (K=8) 0.7792 on the 240-record holdout (arch_notes §9) |
| **producer** | `scripts/drafter/train_eagle3.py` → `Eagle3DraftHead.state_dict()` → `torch.save(..., "model_best.pt")` (committed save code; authoritative) |
| **on-disk format** | torch-pickled `state_dict` (`.pt`), **mixed dtype**: `embed_tokens` + `lm_head` bf16 (explicit `.to(bfloat16)`, train lines 586-587/597-598); the other **13 body tensors are fp32** (nn.Parameter default; `.to(device)`/autocast do not change stored dtype) |
| **co-files present** | `config.json` (written by the train script with the EAGLE-3 fields), `metrics.jsonl`, `summary.json` |
| **tensor count** | **15** (matches the "15/15 tensors" warm-start load in arch_notes §9.2) |
| **published path** | **NONE.** The checkpoint dir is matched by `research/eagle3_drafter/.gitignore` (`checkpoints/`), so it is never committed and exists only as a local training artifact in a per-student workdir. **The HF `a10g` runner cannot pull a local `/senpai-run/...` or `/workspace/...` path.** This is the headline "no published path" gap. |

**The 15 saved tensors** (names/shapes/dtypes derived analytically from the committed
`Eagle3DraftHead` module — `HID=2560, VOCAB=262144, N_AUX=3, HEAD_DIM=256, N_HEADS=8,
N_KV=2, INTER=10240`):

| # | saved key | shape | dtype |
|---|---|---|---|
| 1 | `model.embed_tokens.weight` | `[262144, 2560]` | bf16 |
| 2 | `model.input_norm.weight` | `[7680]` | fp32 |
| 3 | `model.fc.weight` | `[2560, 7680]` | fp32 |
| 4 | `model.norm.weight` | `[2560]` | fp32 |
| 5 | `model.layers.0.self_attn.q_proj.weight` | `[2048, 5120]` | fp32 |
| 6 | `model.layers.0.self_attn.k_proj.weight` | `[512, 5120]` | fp32 |
| 7 | `model.layers.0.self_attn.v_proj.weight` | `[512, 5120]` | fp32 |
| 8 | `model.layers.0.self_attn.o_proj.weight` | `[2560, 2048]` | fp32 |
| 9 | `model.layers.0.mlp.gate_proj.weight` | `[10240, 2560]` | fp32 |
| 10 | `model.layers.0.mlp.up_proj.weight` | `[10240, 2560]` | fp32 |
| 11 | `model.layers.0.mlp.down_proj.weight` | `[2560, 10240]` | fp32 |
| 12 | `model.layers.0.input_layernorm.weight` | `[2560]` | fp32 |
| 13 | `model.layers.0.hidden_norm.weight` | `[2560]` | fp32 |
| 14 | `model.layers.0.post_attention_layernorm.weight` | `[2560]` | fp32 |
| 15 | `lm_head.weight` | `[262144, 2560]` | bf16 |

> **Isolation note.** This audit is derived entirely from committed sources on the
> advisor branch (`scripts/drafter/train_eagle3.py`, `scripts/drafter/eval_eagle3.py`,
> `research/eagle3_drafter/arch_notes.md`) plus the installed vLLM wheel. The local
> `.pt` binary itself was **not** loaded or read — that verification is the
> (post-publish) smoke test in §4. The analytic spec assumes the committed save code;
> §4 confirms the real bytes.

---

## §2 — vLLM-load weight-mapping audit (deliverable 2)

**vLLM read:** `0.22.1rc1.dev307+g3e8afdf78` — the exact wheel pinned in
`submissions/fa2sw_precache_kenyan/manifest.json` dependencies, i.e. the load path
the served read actually runs. (`submissions/vllm_baseline/manifest.json`'s
`vllm==0.22.0` pin is a *different* submission.) Load path:
`vllm/model_executor/models/llama_eagle3.py` →
`Eagle3LlamaForCausalLM.load_weights` (:400-451) then `LlamaModel.load_weights`
(:256-288).

### The mechanism vLLM applies to each checkpoint key

1. `Eagle3LlamaForCausalLM.load_weights` (:423-424): for every key that is **not**
   `lm_head` / `d2t` / `t2d` / `mask_hidden`, it **unconditionally prepends
   `model.`**. (`d2t`→`draft_id_to_target_id`; `t2d` dropped; `lm_head.*` untouched.)
2. `AutoWeightsLoader` dispatches the `model.*` subtree to `LlamaModel.load_weights`,
   which (:268) remaps `midlayer.` → `layers.0.`, then fuses via
   `stacked_params_mapping` (:257-264): `.q_proj/.k_proj/.v_proj` → `.qkv_proj`,
   `.gate_proj/.up_proj` → `.gate_up_proj`.
3. `AutoWeightsLoader` **raises `ValueError` on any unexpected or missing tensor** —
   a bad export fails *loudly* at load, it does not silently mis-serve.

### Finding A (BLOCKING) — the raw `state_dict()` is NOT loadable as-is

Because the saved keys are already in the `model.*` namespace, vLLM's prepend
double-prefixes them: `model.fc.weight` → `model.model.fc.weight`,
`model.layers.0.self_attn.q_proj.weight` → `model.model.layers.0.self_attn.qkv_proj.weight`,
etc. **None of these match a real parameter.** Of the 15 tensors, only
`lm_head.weight` (exempt from the prepend) lands correctly. The audit confirms
**1/15 map, 14 unexpected** → vLLM raises. *Not loadable as-is.*

### Finding B (the fix) — a deterministic, lossless rename makes all 15 map

Apply this rename to the saved keys to produce the **published** checkpoint:

- strip the leading `model.` from the body keys (`fc`, `embed_tokens`, `norm`,
  `input_norm`),
- rename the single decoder layer `layers.0.` → the canonical EAGLE-3 `midlayer.`,
- keep `lm_head.weight` unchanged,
- **keep q/k/v and gate/up SEPARATE** — vLLM fuses them itself (do not pre-fuse).

With those names, the audit confirms **15/15 map and all 12 vLLM load-targets
(incl. the q/k/v and gate/up shards) are covered, zero unexpected, zero shape
mismatch**:

| published key (in `model.safetensors`) | vLLM internal param (after prepend+remap+fuse) | shard | shape ✓ |
|---|---|---|---|
| `embed_tokens.weight` | `model.embed_tokens.weight` | — | `[262144, 2560]` |
| `input_norm.weight` | `model.input_norm.weight` | — | `[7680]` |
| `fc.weight` | `model.fc.weight` | — | `[2560, 7680]` |
| `norm.weight` | `model.norm.weight` | — | `[2560]` |
| `midlayer.self_attn.q_proj.weight` | `model.layers.0.self_attn.qkv_proj.weight` | q | `[2048, 5120]` |
| `midlayer.self_attn.k_proj.weight` | `model.layers.0.self_attn.qkv_proj.weight` | k | `[512, 5120]` |
| `midlayer.self_attn.v_proj.weight` | `model.layers.0.self_attn.qkv_proj.weight` | v | `[512, 5120]` |
| `midlayer.self_attn.o_proj.weight` | `model.layers.0.self_attn.o_proj.weight` | — | `[2560, 2048]` |
| `midlayer.mlp.gate_proj.weight` | `model.layers.0.mlp.gate_up_proj.weight` | 0 | `[10240, 2560]` |
| `midlayer.mlp.up_proj.weight` | `model.layers.0.mlp.gate_up_proj.weight` | 1 | `[10240, 2560]` |
| `midlayer.mlp.down_proj.weight` | `model.layers.0.mlp.down_proj.weight` | — | `[2560, 10240]` |
| `midlayer.input_layernorm.weight` | `model.layers.0.input_layernorm.weight` | — | `[2560]` |
| `midlayer.hidden_norm.weight` | `model.layers.0.hidden_norm.weight` | — | `[2560]` |
| `midlayer.post_attention_layernorm.weight` | `model.layers.0.post_attention_layernorm.weight` | — | `[2560]` |
| `lm_head.weight` | `lm_head.weight` | — | `[262144, 2560]` |

**Every shape matches exactly** (the fused `qkv_proj` `[3072,5120]` = q`[2048]`+k`[512]`+v`[512]`
rows; the fused `gate_up_proj` `[20480,2560]` = gate`[10240]`+up`[10240]`). `262144`
is divisible by 64, so vLLM's `VocabParallelEmbedding`/`ParallelLMHead` apply **no
vocab padding** at TP=1 → the `[262144,2560]` tables line up.

`draft_id_to_target_id` is **absent** from the checkpoint → vLLM defaults it to
identity zeros (`Eagle3LlamaForCausalLM.__init__`), and `compute_logits` scatters
the full-vocab logits 1:1 (correct full-vocab behavior). **Not** a blocker.

**Conclusion:** the head needs a **documented adapter** (rename + safetensors
export), not a retrain → **YELLOW**.

---

## §3 — Publication step enumeration (deliverable 3, runbook — do NOT execute here)

The served path consumes the drafter via `ensure_drafter()`
(`submissions/fa2sw_precache_kenyan/serve.py:720`): it `hf buckets sync
DRAFTER_BUCKET → LOCAL_DRAFTER_DIR`, then requires a file **literally named
`model.safetensors`** (`serve.py:745`) plus `config.json`, and verifies
`sha256(model.safetensors) == DRAFTER_SHA256` (`serve.py:751`). Ordered steps:

1. **Locate** the `56ksyxgw` best weights (the `gua9x68j`-evaluated
   `model_best.pt`, warm-started from #25 `full_20k`).
2. **Convert** (one short script, CPU-only): load the `state_dict`, apply the §2
   Finding-B rename, optionally cast all tensors to **bf16** (serving dtype; clean
   single-dtype file), and `safetensors.torch.save_file` → `model.safetensors`.
3. **Write `config.json`** (vLLM EAGLE-3 draft config) with:
   `architectures:["Eagle3LlamaForCausalLM"]`, `model_type:"llama"`,
   `hidden_size:2560`, `num_hidden_layers:1`, `num_attention_heads:8`,
   `num_key_value_heads:2`, `head_dim:256`, `intermediate_size:10240`,
   `vocab_size:262144`, `draft_vocab_size:262144`, `rms_norm_eps:1e-6`,
   `rope_theta:1e6`, `norm_before_fc:true`, `target_hidden_size:2560`,
   `num_aux_hidden_states:3`, `eagle_aux_hidden_state_layer_ids:[2,21,39]`,
   `tie_word_embeddings:false`. **Belt-and-suspenders:** also nest the EAGLE knobs
   under an `eagle_config:{norm_before_fc:true,
   eagle_aux_hidden_state_layer_ids:[2,21,39]}` dict (vLLM reads `eagle_config`
   first — `LlamaModel.__init__:148-157,183-185`). The train script already writes
   the top-level fields; this just adds the nested mirror.
4. **Publish** the two-file dir (`model.safetensors` + `config.json`) to a
   runner-pullable location — preferred:
   `hf://buckets/gemma-challenge/gemma-senpai/weights/eagle3-inrepo-251head/`
   (the senpai scratch bucket, same mechanism as the live `DRAFTER_BUCKET`); alt: a
   private Hub model repo. **Do not point at a local path.**
5. **Compute** `sha256(model.safetensors)` → the new `DRAFTER_SHA256`.
6. **Manifest delta** on `submissions/fa2sw_precache_kenyan/manifest.json` (per ubel
   #322 §1): `SPECULATIVE_CONFIG.method` `mtp`→`eagle3`, `SPECULATIVE_CONFIG.model`
   and `LOCAL_DRAFTER_DIR` → the same `/tmp/eagle3-inrepo-251head`, `DRAFTER_BUCKET`
   → the published bucket, `DRAFTER_SHA256` → step 5, keep `num_speculative_tokens:7`.

This is a runbook for the human/launch; **this card executes none of it.**

---

## §4 — vLLM-load + greedy-identity smoke-test spec (deliverable 4, reference — do NOT run)

A minimal **local AWS A10G** smoke (no HF Job), the last launch gate before any
spend (ubel #322 §0 step 5; `enforce_launch_gate` evidence). It must confirm:

1. **Load:** boot `serve.py` with the §3 EAGLE-3 manifest. vLLM instantiates the
   head as `Eagle3LlamaForCausalLM` with **0 missing / 0 unexpected** tensors. (This
   is exactly what catches a rename slip: `AutoWeightsLoader` raises on any
   mismatch — §2 step 3.) Confirm the config custom fields survived
   `AutoConfig(model_type=llama)` (caveat C1).
2. **Serve up:** `GET /v1/models` returns 200 with the boot-guard active (the merged
   #317 `_guard_included_router`).
3. **Greedy identity:** on a handful of prompts, the EAGLE-3 served greedy token-ids
   are **identical** to plain greedy autoregressive decode of the same target
   (`google/gemma-4-E4B-it`). Speculative decoding must not change the accepted token
   stream — this is the program.md greedy contract and the fp32→bf16 cast check
   (caveat C2).
4. **PPL sanity:** a tiny `prompt_logprobs` PPL spot-check is finite and ≈ the
   frontier (≤ 2.42, well clear), not a garbage value (would betray a silent
   weight-routing error).

Only after all four PASS is the head launch-ready. If load raises a shape/name
error the §2 rename is wrong (re-derive); if greedy identity breaks, investigate the
cast/config before spending the one HF launch.

---

## §5 — Readiness verdict + blocking issues (deliverable 5)

### Verdict: 🟡 YELLOW. `ckpt_publish_readiness_blocking_issues = 3`.

| # | id | kind | what it is | fix (deterministic) |
|---|---|---|---|---|
| 1 | `no_published_path` | publish | Artifact is a local-only `.pt` in a gitignored workdir; the HF runner cannot pull a local path. | Publish `model.safetensors`+`config.json` to `DRAFTER_BUCKET`/Hub; set `DRAFTER_SHA256` (§3). |
| 2 | `weight_key_namespace_mismatch` | load | `head.state_dict()` keys are `model.`-prefixed + `layers.0.`; vLLM prepends `model.` and remaps `midlayer.`→`layers.0.`, so raw keys double-prefix (`model.model.*`) and the loader raises (§2 Finding A). | Rename: strip `model.`, `layers.0.`→`midlayer.`, keep `lm_head.*`, keep q/k/v+gate/up separate (§2 Finding B). |
| 3 | `container_format` | publish | Checkpoint is a pickled `.pt` state_dict; `serve.py:745` + vLLM require a single `model.safetensors`. | Export the renamed tensors to `model.safetensors` (recommend uniform bf16) + `config.json` (§3). |

All three are mechanical and lossless — **no retrain, no architecture change, all 15
shapes already match** — which is exactly why this is YELLOW (loadable with a
documented shim) and not RED (irreducible mismatch).

### Verify-at-smoke caveats (NOT counted as blocking; closeable only by §4)

- **C1 — config-field survival.** `AutoConfig(model_type=llama)` must retain
  `norm_before_fc` / `target_hidden_size` / `num_aux_hidden_states` /
  `eagle_aux_hidden_state_layer_ids`. HF stores unknown kwargs as attributes, so this
  *should* hold; the nested `eagle_config` (§3 step 3) is the robust backstop.
- **C2 — dtype.** Body tensors are fp32 in the `.pt`; vLLM serves bf16 and
  `default_weight_loader` casts on `copy_`, so a mixed-dtype file loads — but a
  uniform-bf16 export is cleaner and greedy identity must be confirmed regardless.
- **C3 — d2t identity.** Absent `draft_id_to_target_id` → vLLM identity-defaults it;
  full-vocab scatter is correct. Confirmed from `compute_logits`. Not a blocker.
- **C4 — vLLM version.** Load path read from `0.22.1rc1.dev307` (the served wheel
  pin). If the org-credit harness image diverges, re-confirm the load path.

### What this unblocks

This closes ubel #322 §0's single open unknown ("vLLM-load UNVERIFIED") to the
maximum extent possible with 0 GPU: the head **is** vLLM-loadable, the adapter is
specified and self-test-verified against the installed loader, the publication and
smoke steps are enumerated, and the residual risk is the four caveats the
post-publish §4 smoke is designed to close. The #319 Option-A measured read is
**publish-ready pending the mechanical §3 build + §4 smoke** (both human-owned), not
blocked on any head-quality or architecture unknown.

---

## Public evidence used

- **ubel #322 / `eagle3_measured_read_spec.md` (`2nmem4dc`)** — the merged #319
  Option-A measured-read spec whose §0 names this exact precondition (`gua9x68j`
  local-only, vLLM-load deferred). This card operationalizes that §0.
- **fern #34 / `research/eagle3_drafter/arch_notes.md` (PR #16, #34)** — the
  `gua9x68j`/`56ksyxgw` head architecture, the §7 "best-effort" vLLM weight-name
  mapping this audit makes concrete, and the "deployment gated on kanna #5" deferral.
- **`scripts/drafter/train_eagle3.py` / `eval_eagle3.py`** — the committed
  `Eagle3DraftHead` save code (authoritative source for the 15 tensor names/shapes/dtypes).
- **vLLM `0.22.1rc1.dev307` `llama_eagle3.py`** — the `Eagle3LlamaForCausalLM` /
  `LlamaModel` load contract; the wheel pinned in `fa2sw_precache_kenyan/manifest.json`.
- **`submissions/fa2sw_precache_kenyan/serve.py:720` `ensure_drafter()`** — the
  `DRAFTER_BUCKET`→`model.safetensors`+sha256 sync mechanism that fixes the
  publication format/path requirements.
