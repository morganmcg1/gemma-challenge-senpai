# Price the served-file loopgraph rewrite EAGLE-3 requires (PR #312)

**Verdict:** the EAGLE-3 served swap is **NOT a free config drop-in**. The onegraph
loopgraph that *produces* the 481.53 TPS frontier is MTP-specific; under
`method=='eagle3'` all three served-file touchpoints go **inert**, the served path
falls back to the **stock un-onegraphed eager `EagleProposer`** at an estimated
**~402 TPS floor** (band **[302, 471]**, a **−16.5%** regression vs 481.53), and
recovering frontier runtime costs **2 moderate-rewrites + 1 correctness-rederivation**
that reopen **all four** validity gates — plus a near-zero **#272 boot-500 guard**
co-edit on the very same file.

This is the **DEPLOYMENT-COST axis** the EAGLE-3 GO/NO-GO packet (#305) was missing.
It is a `0`-TPS **STATIC** scoping card: read-only from the real
`submissions/fa2sw_precache_kenyan/sitecustomize.py`, no served-file edit, no build,
no boot, no HF Job, no submission change. Baseline 481.53 unchanged.

- **Self-test (PRIMARY):** `loopgraph_rewrite_cost_self_test_passes = 1` (PASS)
- **TEST:** `eager_fallback_tps_floor = 402.1` (float, ESTIMATE) ·
  `rewrite_reopens_greedy_identity = True` (bool)
- **W&B run:** `9b1arani` · group `eagle3-rewrite-cost`
- **Artifact:** `eagle3_loopgraph_rewrite_cost_results.json`

---

## Why a config swap goes inert (the MTP-specific assumptions)

`#307` (run `88eh8twv`, merged) already classified T5/T6/T7 as *served-file-change*.
This card **sub-classifies their engineering complexity** and prices the fallback. The
loopgraph rests on four MTP-only assumptions, every one of which is **false for
EAGLE-3** (own-KV Llama draft layer, fused `fc[7680→2560]` aux input from layers
2/21/39):

| MTP assumption (re-read anchor) | EAGLE-3 reality |
|---|---|
| Drafter head is `Gemma4MTPMaskedEmbedder.get_top_tokens` (centroid-sparse top-12k); patch keyed `TOP_TOKEN_TARGET=gemma4_mtp` (`:20`) | `Eagle3LlamaForCausalLM` has **no** `get_top_tokens` / **no** centroids — dense `compute_logits`. Patch matches nothing → **inert**. |
| Proposer is `Gemma4Proposer`; `LOOPGRAPH_TARGET=vllm.v1.spec_decode.gemma4` (`:18`, `:274`) | vLLM instantiates `EagleProposer` (`vllm.v1.spec_decode.eagle`). The K=7 single-replay is **never invoked**. |
| Drafter is **Q-only and KV-shared** → **width-1 is exact** (`:25–32`) | Draft layer has its own `k_proj/v_proj`: it **writes KV** and reads positions `0..i` — **cross-position deps**. Invariant is **false**. |
| Single `[1,2560]` static hidden, seeded from one target hidden, recycled across K (`_build_static_buffers` `:99`, `:490`) | Seed is a fused `[1,7680]` aux cat; buffers must hold a **growing draft KV** and advance `seq_len/slot_mapping` **inside** the captured graph. |

---

## Per-touchpoint complexity + reopened gates

| TP | What it is | Complexity | Reopened gates |
|----|------------|-----------|----------------|
| **T5** | Fused sparse-argmax drafter head (`get_top_tokens` wrap) | **moderate-rewrite** | greedy-identity, TPS |
| **T6** | onegraph loopgraph capture (proposer key + `propose_onegraph` body) | **moderate-rewrite** | greedy-identity, boot-500, TPS |
| **T7** | Correctness invariant + static buffers / **KV-bearing chain** | **correctness-rederivation** | greedy-identity, PPL≤2.42, boot-500, TPS |

Partition: **0 mechanical-config · 2 moderate-rewrite · 1 correctness-rederivation.**
Union of reopened gates: **{greedy-identity, PPL≤2.42, boot-500, TPS}** — the full set.

- **T5** is *inert-not-crash*. A drafter argmax is emit-neutral in principle (greedy
  spec decode emits the **target** argmax), so PPL is not directly moved — but the
  changed drafter shifts acceptance → verify batch `M` → batch-variant int4-Marlin
  near-tie flips, so **greedy-identity must be re-measured** and the lost
  centroid-sparse bandwidth kernel re-priced for TPS (no EAGLE analogue).
- **T6**'s capture scaffolding (`_build_static_buffers`/`_capture_graph`/ping-pong) is
  structurally reusable; the **`propose_onegraph` body must be re-bodied** against
  `EagleProposer`'s interface (the named `Gemma4Proposer` helpers don't exist on it).
  A fresh capture can raise at boot (`LOOPGRAPH_REQUIRE_CAPTURE=1`) → boot-500 reopens.
- **T7** is the deepest, most coupled edit. A wrong KV-bearing chain changes accepted
  lengths → batch-variant argmax → served stream, so **greedy-identity AND same-path
  PPL** must be re-measured; the re-sized capture can crash boot; TPS is the point.
  **All four gates reopen.**

---

## EAGLE-3-on-EAGER fallback floor (banked decomposition, ESTIMATE)

If the loopgraph rewrite is **deferred**, the config-only swap deploys the **stock
eager `EagleProposer`**. Floor from the banked onegraph→eager drafter-chain
decomposition (all inputs re-read from repo artifacts → falsifiable):

- **HEADLINE = 402.1 TPS** — MTP-equivalent eager floor, **directly measured**:
  `drafter_forward_roofline.json` measures the deployed drafter K=7 loop at
  **graph 566.49µs vs eager 2859.34µs** inside an **11.6ms** decode step.
  Removing the onegraph capture inflates the step to **13 892.85µs**; iso-E[T]
  TPS scales `481.53 × 11600/13893 = 402.1`. This is an **UPPER bound** on the true
  EAGLE-3 eager floor (EAGLE's own-KV drafter is heavier than MTP's Q-only layer).
- **Band [302.3, 471.0]:**
  - **Lower 302.3** — EAGLE-heavier hard-lower: scale the measured eager penalty by the
    `#293` EAGLE/MTP draft-compute ratio (**~3.0×** at L_fuse=3). Conservative (the
    penalty is launch-dominated, scales sub-linearly with compute) → over-states the
    loss → a hard floor, not a central estimate.
  - **Upper 471.0** — launch-count cross-check (`#154`): `drafter_propose=35` launches ×
    `7.42µs` zero-overlap. **Undercounts** drafter kernels (~5/pass modeled vs ~25 real)
    and assumes overlap → milder upper bound; the direct chain measurement supersedes it.

**ESTIMATE — no eager-path TPS was measured.** EAGLE-3's E[T] (acceptance) is held at
the MTP value (iso-acceptance); any EAGLE-3 E[T] gain is a **separate numerator axis**
(#293/#295/#304), not credited here.

---

## #272 boot-500 guard — required near-zero co-edit

The deployed frontier is **missing** the prometheus `_guard_included_router` guard
(Issue #272): `precache_guard_count = 0`. The sibling carries it
(`treeverify_guard_count = 2`, lines **2680** def / **2696** call). Because T6/T7 already
reopen `sitecustomize.py`, **porting the guard is a ~0-marginal co-edit** → flagged
`co_edit_required = True`.

---

## Honest caveats (carried in the artifact)

1. STATIC scoping from source — an **engineering-complexity estimate**, NOT an
   implementation and NOT a measured fallback TPS.
2. The eager-fallback floor is an **ESTIMATE** from a banked decomposition; no eager-path
   run was launched. **0 TPS.**
3. The headline floor is an **UPPER bound** on the true EAGLE-3 eager floor (heavier
   own-KV drafter).
4. The launch-count model (#154) **undercounts** drafter launches and assumes overlap —
   kept only as a milder upper cross-check; the direct chain measurement supersedes it.
5. EAGLE-3 acceptance is held at the MTP value (**iso-acceptance**); EAGLE E[T] gains are
   a separate numerator axis, not credited here.
6. The EAGLE-3 checkpoint does not exist yet (**training-gated**); this prices the DEPLOY
   rewrite, not the train.

---

## Reproduce

```bash
cd /workspace/senpai/target
python research/validity/eagle3_loopgraph_rewrite_cost/eagle3_loopgraph_rewrite_cost.py \
    --self-test --wandb_group eagle3-rewrite-cost --wandb_name wirbel/eagle3-rewrite-cost
# -> SELF-TEST PASSED ; writes eagle3_loopgraph_rewrite_cost_results.json
```

Public evidence imported: `#307` (88eh8twv, served-file touchpoints + #272 guard 0/2) ·
`drafter_forward_roofline.json` / `launch_overhead_graph_leg.json` (#154 banked
decomposition) · `#293` (EAGLE draft ~3× MTP at L_fuse=3) · Issue #272 (boot-500 guard).
