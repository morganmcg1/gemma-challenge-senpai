# EAGLE-3 served-path integration: drop-in swap or boot-risk change? (PR #307)

**Verdict: `served-file-change` — YELLOW light. `swap_is_config_only=False`,
`reuses_boot_fragile_path=False`, `readiness_blocks_go=True`.**

Read-only static analysis. **0 GPU, 0 TPS added.** No served-file edit, no build,
no boot test, no HF Job, no submission change. BASELINE **481.53** unchanged.
W&B run: **`88eh8twv`** (group `eagle3-integration-readiness`).

---

## The question this leg prices

The EAGLE-3 build-economics matrix has priced nearly every **cost** axis — step
cost (wirbel #295 `c334qaqu`, ~2.95× linear, validates 6.1245), VRAM fit
(ubel #299, ≤24 GiB), per-position target (kanna #289 / denken #297), companion
floor (lawine #292/#296), build cost (denken #301), read-companion (fern #302),
private-bar (lawine #300), numerator reachability (denken #304). **Every one of
those axes silently assumes the trained drafter can be *deployed* into the served
runner.** That assumption was unpriced.

This card prices it: **is wiring a {2,21,39}-fusion EAGLE-3 drafter into the
deployed `submissions/fa2sw_precache_kenyan` served path a config/weights drop-in,
or does it require a served-file (or vLLM-fork) code change — and if so, does that
change re-open the Issue #272 boot-500 risk?** This is a deployment-feasibility
gate on the human GO/NO-GO, **orthogonal to every cost/economics axis**.

---

## (1) The deployed drafter-load path

The active frontier `submissions/fa2sw_precache_kenyan` runs **MTP**, not EAGLE-3.
The drafter is selected and loaded entirely through manifest env + generic sync:

| Step | Where | What it does |
|---|---|---|
| Method + topology | `manifest.json` → `SPECULATIVE_CONFIG` | `{"method":"mtp","model":"/tmp/qat-assistant","num_speculative_tokens":7}` (stored as a JSON-encoded **string**, so the file bytes carry escaped quotes `\"method\":\"mtp\"`). |
| Passed to vLLM | `serve.py` `main()` | `append_env_arg(args, "SPECULATIVE_CONFIG", "--speculative-config")` → vLLM picks the proposer + head from `method`. |
| Weights | `manifest.json` → `DRAFTER_BUCKET` / `DRAFTER_SHA256` | `hf://buckets/.../drafter-ft/ft-v1-epoch_001`. |
| Weight sync | `serve.py` `ensure_drafter()` (≈720–770) | generic hf-buckets-sync + sha256 + writes `centroid_intermediate_top_k` into the drafter config. |
| Speed engine | `sitecustomize.py` | hand-rolled **onegraph** CUDA-graph single-replay of the K=7 width-1 MTP drafter loop — *this is the engine that produces 481.53*. |

The wheel is **stock upstream** (`wheels.vllm.ai/...vllm-0.22.1rc1.dev307...`), not
a vendored fork. There is **no vLLM source tree and no `gemma4.py` model file**
checked into the repo (`find` confirms): the submission *source-patches stock vLLM
at import* via `sitecustomize.py` meta-path finders, it does not vendor a fork. So
"fork PR #15" is this repo's feasibility report and "the `gemma4.py` model in the
fork" is stock `vllm.model_executor.models.gemma4` patched at runtime — the worst
integration case is a **served-file change**, never a fork-code change
(`n_fork_code_change=0`).

**The three named sibling patch files were also traced and are drafter-agnostic**
(they stay byte-for-byte unchanged under the swap, so they are not part of the
integration *delta*):

- `serve_patch_precache.py` — kduma warmup KV-replay (targets `vllm.entrypoints.launcher`); its own docstring says *"drafter-blind concern does not apply (cache holds target-layer KV)"*. Warms the **target** prefix cache; indifferent to MTP vs EAGLE-3.
- `serve_patch_pck04.py` — PCK-04 logits-scatter on the **target** model (`vllm.model_executor.models.gemma4`); operates on target verify-logits, analogous to T9. Unchanged by the drafter swap.
- `splitkv_verify_patch.py` — routes spec-**verify** attention to split-KV FlashDecoding (`vllm.v1.attention.*`). The verify batch stays K+1=8 query-rows for a linear `num_speculative_tokens=7` swap, so it carries over. *(Caveat: would need re-validation only if EAGLE-3 were given a branching tree, which the specified swap does not.)*

---

## (2) Integration delta linear-MTP K=7 → {2,21,39}-fusion EAGLE-3

Nine touchpoints, each classified into exactly one of
`{config-only, served-file-change, fork-code-change}`. Every row cites a
`(file, substring)` that the self-test re-reads from the real repo and confirms
present — the partition is **derived, not asserted**.

| # | Touchpoint | Class | Why |
|---|---|---|---|
| T1 | Speculative method + config | **config-only** | `method "mtp"→"eagle3"`, model→EAGLE ckpt, keep `num_speculative_tokens:7`; `[2,21,39]` is already the default. Manifest env edit, no served `.py` change. |
| T2 | Drafter weight artifact | **config-only** | `DRAFTER_BUCKET`→EAGLE ckpt; `ensure_drafter()` sync is generic. *Caveat:* the ckpt **does not exist yet** (training-gated, arch_notes "deployment gated on kanna #5"); `ensure_drafter` writes the MTP-only `centroid_intermediate_top_k` key (spurious-but-harmless); vLLM-load verification deferred (arch_notes §7). |
| T3 | Aux-hidden capture (target exposes 2/21/39) | **config-only** | `Gemma4Model` already implements `SupportsEagle3`; vLLM auto-sets `use_aux_hidden_state_outputs` for `method=='eagle3'`. Feasibility PR #15: "0 hours of vLLM work". *Caveat:* PLE gemma4 source-patches are co-resident; non-interference needs re-validation. |
| T4 | Drafter head class | **config-only** | Head is selected by vLLM's registry from `method` (`Gemma4MTP`→`Eagle3LlamaForCausalLM`). Selection is config-only — but the *structural* change (own-KV Llama layer, fused `[7680]` input, `compute_logits` not `get_top_tokens`) is exactly what breaks T5/T6/T7. |
| **T5** | Fused sparse-argmax patch (head) | **served-file-change** | `sitecustomize.py:20` `TOP_TOKEN_TARGET="vllm.model_executor.models.gemma4_mtp"` wraps the MTP head's `get_top_tokens` (line 190/202). EAGLE-3's head has no `get_top_tokens` → patch goes **inert**; keeping the fused-argmax kernel needs re-targeting. |
| **T6** | onegraph loopgraph capture (proposer) | **served-file-change** | `sitecustomize.py:18` `LOOPGRAPH_TARGET="vllm.v1.spec_decode.gemma4"`, line 274 `proposer_cls = module.Gemma4Proposer`. Under `method=='eagle3'` vLLM instantiates `EagleProposer` (`vllm.v1.spec_decode.eagle`), so the onegraph K=7 single-replay — **the source of 481.53** — is never invoked. **This is the load-bearing finding.** |
| **T7** | onegraph correctness invariant + static buffers/sizing | **served-file-change** | The invariant at `sitecustomize.py:25-32` — *"the Gemma4 MTP drafter is Q-only and KV-shared … Width-1 is exact"* — is **false for EAGLE-3** (its draft Llama layer writes its own KV, has cross-position deps). `_build_static_buffers`(99) / `_capture_graph`(233) assume a single `[1,2560]` hidden buffer; EAGLE-3 needs a fused `[1,7680]` aux input + KV-bearing chain. Correctness re-derivation, not just re-pointing. |
| T8 | Shared-base fused-accept-prep patch | **config-only** | `sitecustomize.py:21` `PROPOSER_TARGET="vllm.v1.spec_decode.llm_base_proposer"`, line 1029 `proposer_cls = module.SpecDecodeBaseProposer` — the **shared base** both proposers inherit, so it carries over unchanged. *Caveat:* `num_reqs==1` accept-geometry should be re-validated, but no served *change* is required to keep it functioning. |
| T9 | PLE gemma4 source patches (orthogonality) | **config-only** | `serve.py` `patch_gemma4_source` touches per-layer-embeddings / embed-scale inside the decoder, **not** the `Gemma4Model.forward` aux-collection return path → orthogonal to EAGLE-3 aux capture; no change. Non-interference still merits re-validation. |

**Partition: 6 config-only · 3 served-file-change · 0 fork-code-change.**
→ `swap_is_config_only = (served==0 and fork==0) = **False**`.

### The load-bearing finding (T6/T7)

The feature **prerequisite** is green — stock vLLM 0.22.x exposes EAGLE-3 aux
hidden states with zero vLLM work (feasibility PR #15). But **the swap is not the
advertised config drop-in**: the MTP-specific onegraph loopgraph that produces
481.53 is keyed to `Gemma4Proposer` / `vllm.v1.spec_decode.gemma4` /
`gemma4_mtp.get_top_tokens`, and its "width-1 is exact" correctness rests on MTP
being KV-shared / Q-only — **false** for EAGLE-3's own-KV Llama draft layer. Under
`method:"eagle3"` those patches go inert and **the engine that makes this
submission the frontier is lost** unless the loopgraph is rewritten for the EAGLE
proposer (a served-file change), then greedy-identity / PPL / boot / TPS
re-validated.

---

## (3) Issue #272 boot-500 cross-check

Issue #272: vLLM 0.22.1rc1 + `prometheus_fastapi_instrumentator` `_IncludedRouter`
`_get_route_name` does `route.path` on a pathless sub-router → `AttributeError` →
`/v1/models` 500 → 0 records. Fixed by a `_guard_included_router` wrap (validated
#71/#177).

| Submission | `_guard_included_router` count | Status |
|---|---|---|
| `fa2sw_precache_kenyan` (active frontier) | **0** | **missing the guard** (the #272 gap) |
| `fa2sw_treeverify_kenyan` (sibling) | **2** (def `:2680`, call `:2696`) | carries the reference guard |

**Does EAGLE-3 re-open #272?** No — `reuses_boot_fragile_path = **False**`. Every
module the EAGLE-3 *touchpoints* move through lives in the **spec-decode /
model-execution** layer (`vllm.v1.spec_decode.{gemma4,eagle,llm_base_proposer}`,
`vllm.model_executor.models.{gemma4_mtp,llama_eagle3}`, `vllm.v1.worker.gpu_model_runner`),
**not** the FastAPI route-registration layer (`vllm.entrypoints` / `prometheus` /
`vllm.renderers`) where `_IncludedRouter` throws. The two are orthogonal: swapping
MTP→EAGLE-3 changes neither the route table nor when prometheus instruments it.

*(For completeness:* the deployed submission **does** patch the entrypoint layer —
`serve_patch_precache.py` wraps `vllm.entrypoints.launcher` for warmup KV-replay.
But that patch is (a) **pre-existing and unchanged** by the swap — it is not an
EAGLE-3 touchpoint — and (b) the **launcher**, not the `prometheus_fastapi_instrumentator`
`_IncludedRouter._get_route_name` path that #272 specifically implicates. It is
already live in the 481.53 baseline, so it adds no *new* boot surface here.*)*

**Load-bearing caveat (the real coupling):** the EAGLE-3 swap is *not free of #272
exposure* — it **requires editing the same `sitecustomize.py` that is itself
missing the boot guard** (T6/T7). The boot-500 risk is **pre-existing and
independent** of EAGLE-3, but the moment this file is reopened for the loopgraph
rewrite is the natural, low-cost moment to also port the `_guard_included_router`
fix from the sibling. *Touching this file for EAGLE-3 does not add the #272 risk —
but it should not ship without closing it either.*

---

## (4) Readiness verdict

| Field | Value |
|---|---|
| `eagle3_integration_readiness` | **`served-file-change`** |
| `swap_is_config_only` | **False** |
| `reuses_boot_fragile_path` | **False** |
| `readiness_blocks_go` | **True** |
| `handoff_light` | **yellow** |
| partition (config / served / fork) | 6 / 3 / 0 |
| #272 guard (precache / treeverify) | 0 / 2 |

`readiness_blocks_go=True` means deployment adds a blocker **beyond** the
economics: EAGLE-3 is not a like-for-like swap into the already-deployed,
already-private-verified linear path — it requires a served-file rewrite (+ full
re-validation) before it can match the frontier's runtime, and that work is
**not** reflected in any cost axis priced so far.

---

## (5) Honest 0-TPS framing

This card prices **deployment feasibility only**. It launches no run, edits no
served file, changes no submission. **BASELINE 481.53 is unchanged and this card
adds 0 TPS.** It does not claim EAGLE-3 is faster or slower — it states the
*mechanism* of integration and the *work* a GO would actually require.

---

## (6) W&B self-test

Run **`88eh8twv`** (project `gemma-challenge-senpai`, group
`eagle3-integration-readiness`):

- **PRIMARY** `summary/eagle3_integration_readiness_self_test_passes = 1`
  (all 13 conditions pass: evidence present, partition exhaustive +
  mutually-exclusive, verdict follows from partition, #272 counts real,
  EAGLE-3 not in route layer, no fork path).
- **TEST** `summary/swap_is_config_only = 0`.
- All key fields under `summary/`: `eagle3_integration_readiness`,
  `reuses_boot_fragile_path`, `readiness_blocks_go`, `handoff_light`,
  `n_{config_only,served_file_change,fork_code_change}`,
  `{precache,treeverify}_guard_count`, `official_baseline`,
  `tps_added_by_this_card=0`, plus 13 `selftest_*` booleans.

Reproduce:

```bash
cd /workspace/senpai/target
python research/validity/eagle3_integration_readiness/eagle3_integration_readiness.py \
  --self-test --wandb_group eagle3-integration-readiness
```

---

## (7) Hand-off (GO/NO-GO packet)

**YELLOW light:** served-path integration of a {2,21,39}-fusion EAGLE-3 drafter is
*not* the advertised config drop-in — the feature-export prerequisite is green
(stock vLLM aux capture is built-in), but the MTP-specific onegraph loopgraph that
produces 481.53 goes inert under the EAGLE proposer and must be **rewritten** (a
served-file change to the same `sitecustomize.py` that #272 flags as missing the
boot guard), then greedy-identity / PPL / boot / TPS re-validated before any launch.

---

## Public evidence used

- **Leaderboard** (read-only, public): top served TPS are frantic-penguin 489.63,
  need-for-speed 488.07, our 481.53. The board exposes TPS but **not** other teams'
  drafter method; our deployed frontier is verified **MTP**, and there is **no
  publicly documented EAGLE-3 served path** — so the onegraph-loopgraph rewrite the
  EAGLE swap requires has no known public precedent to borrow from.
- **Feasibility PR #15** (fern): `eagle3_hiddens_accessible=1`, `SupportsEagle3`
  built-in, "0 hours of vLLM work" — establishes the **stock-vLLM** feature export
  (green prerequisite), explicitly *not* this submission's onegraph substrate.
- **arch_notes PR #16** (fern): `Eagle3LlamaForCausalLM`, Llama decoder layer,
  `fc[7680→2560]`, aux `(2,21,39)` ↔ HF `hidden_states[2]/[21]/[39]`, own-KV draft
  layer; "deployment gated on kanna #5".
- **Issue #272**: boot-500 via `_IncludedRouter` / `prometheus_fastapi_instrumentator`
  `_get_route_name`; fix `_guard_included_router` (validated #71/#177).
- **Cost-axis context** (this launch): wirbel #295 (`c334qaqu`), ubel #299.
