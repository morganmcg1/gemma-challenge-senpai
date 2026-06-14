# PR #71 — Tree-verify live integration: located seams + two decisive findings

**Status:** the descend-walk keystone (Component 4 / BUG-2) is built + validated
(`report_descend_walk.md`; re-validated GPU bit-exact on the warm env this
session). This report maps the remaining live vLLM integration to **exact
file:line seams** and reports two findings that materially change the build
plan:

- **Finding A (de-risk):** the tree-causal mask threads through the **existing**
  triton attention as a native `qq_bias` tensor — **no kernel rewrite.**
  Component 2 drops from "deep triton surgery" to tensor construction.
- **Finding B (blocker, newly surfaced):** vLLM bookkeeps acceptance as a
  **count** assuming a contiguous accepted prefix. A tree's accepted root→leaf
  path is **scattered** across the M verify rows, so count-based KV retention
  keeps the **wrong** slots. Correct **continued generation** needs explicit KV
  compaction — but a **single-step salvage measurement does not.**

Net recommendation: build the **single-step salvage probe first** (Components
1+2+3a/3b+4, defer KV-compaction) → the fleet's GO/NO-GO salvage number at zero
quota, fastest and lowest-risk. Add KV-compaction (3c) only once salvage clears.

---

## Located seams (deployed `fa2sw_precache_kenyan` + pinned vLLM 0.22.1rc1)

| Comp | what | seam (file:line) | mechanism |
|---|---|---|---|
| 1 | drafter tree-emit | `sitecustomize.py:538-564` (eager `propose_onegraph` loop); top-w via the drafter's existing sparse top-k (`top_k_token_ids`, `centroid_intermediate_top_k`) | replace the width-1 `for index in range(token_count)` chain with a topological tree-walk of width-1 forwards; at each branch node take rank-1..rank-w from the drafter top-k. Rank-1 spine forward is byte-identical to the deployed chain → spine identity by construction (BUG-1 guard). enforce-eager (ONEGRAPH=0 / LOOPGRAPH_REQUIRE_CAPTURE=0) — dodges the size-29 capture crash. |
| 2 | tree-causal mask | triton `unified_attention(..., qq_bias=None ...)` — `triton_unified_attention.py:787`, applied at `:525` `S += load_qq_bias_tile(...)` **before** the `if IS_3D` epilogue split (`:548`) | **qq_bias is an [M,M] additive query×query bias, applied in BOTH 2D and 3D paths.** `load_qq_bias_tile` (`triton_attention_helpers.py:344`) biases only the query×query block; prefix keys are unbiased. Set `qq_bias[i,j]=0` if node j ∈ ancestors(i)∪{i} else `-inf`. This is exactly `tree_spec.tree_causal_mask` (M×M bool) → `where(mask,0,-inf)`. Thread it through `splitkv_verify_patch.py` (which today passes no bias) into the redirected 3D verify. |
| 3a | tree metadata | `_calc_spec_decode_metadata` (`gpu_model_runner.py:2730`) — pure linear-chain index math | override `target_logits_indices = [parent[1..M-1]]` (local), `bonus_logits_indices = leaves`, `draft_token_ids` = node-order tree tokens, `logits_indices` = all M rows. `tree_spec.verify_index_maps` already produces these (validated == deployed metadata on the degenerate chain). |
| 3b | verify positions | the M verify rows' `positions` (RoPE) + `query_start_loc` | each node's position = `base + depth(node)` (siblings share a depth). For the linear chain depth==row-index so this is invisible today; the tree needs the explicit depth map (`tree_spec.TreeSpec.depth`). |
| 3c | **KV compaction** (DEFERRED for the probe; design PINNED) | post-sampler bookkeeping: `num_accepted = (output_token_ids != -1).sum(dim=1)` (`gpu_model_runner.py:1513`) | **the blocker (Finding B).** vLLM keeps the first `num_accepted` KV slots (contiguous-prefix assumption). The tree's accepted path is scattered → must **gather** the accepted nodes' K/V into the first `num_accepted` slots (path order) before the next step, or the next step's prefix KV is corrupt (PPL breaks). **Not needed for a single-step probe.** **MANDATORY DESIGN (ubel #157, MERGED):** the relocate MUST be a **single fused/vectorized GPU launch** — `index_select` on the `[L,W,H,D]` stack of all 37 layers' K+V → `index_copy_` scatter, with the commit-index produced **on-device** by the accept walk (lawine #147 sync-free) and consumed **without a host readout** so it stays inside the captured graph. Zero-copy ideal if paged KV (int slot-map update, ~20 µs/step). A host-bound Python loop over layers is correctness/PPL-clean but **cannot be graph-captured → ~122 ms/step → descent E[T]=5.04 collapses 522→77 TPS.** Subtlety: 37 device launches (no host trip) is already 70× faster than the host loop — it's the **host round-trip, not the layer count**; fully vectorize to ONE launch. equivalence_rate=1.0 by construction (pure bf16 permute/copy, no cast/arithmetic). |
| 4 | descend walk | replace `_dixie_fused_accept_prep` call in the injected verify block (`serve.py:429`); kernel `sitecustomize.py:920-963` | wire `tree_accept` (validated) — node-indexed `dixie_all_argmax` + node-indexed draft tokens + static children-CSR. `serve.py:416` already gathers `dixie_all_argmax` (all M rows). |

Prewarm `serve.py:487-492` hardcodes M=8; widen to the tree M (minor).

## Component 1 detail — draft-side reference validated + live shape located

`tree_spec.emit_tree` (the **draft-side twin of `descend_accept`**) is built and
CPU-validated: it reproduces the deployed linear chain on the degenerate tree
(k=7/15/31), holds **spine-identity** on M16/M32 (the rank-1 path drafts the same
tokens as a pure width-1 chain — the BUG-1 guard), and assigns rank-2/3 branch
tokens distinctly. It takes an injected `forward_fn(node, token, parent_hidden,
position)` so the topological emit order + token assignment are testable without
a GPU; the live realization supplies the real drafter forward.

Located live seams (refines the Comp-1 row):
- **top-w hook = `get_top_tokens`** (`sitecustomize.py:190`, the captured-body
  call `self.model.get_top_tokens(last_hidden[:1])`). Extend to a `top_w` variant
  via the existing `_select_and_score(hidden, lm_head_weight)` → `(logits,
  selected)` then `topk(w)` over `logits` → gather `selected`. `topk[0]` ==
  the deployed sparse argmax → rank-1 = greedy → spine identity holds.
- **parent-hidden threading.** The deployed loop carries one `self.hidden_states`
  forward (chain). A tree must store each internal node's output hidden and
  **restore the parent's hidden** before forwarding a node (node n consumes
  `hidden[parent[n]]`, not `hidden[n-1]`). A small per-node hidden cache.
- **drafter cost.** `emit_tree` forwards only internal nodes: **M16 = 13, M32 =
  25** width-1 drafter forwards (vs 7 for the chain). Verify stays one forward
  over M rows. The extra drafter latency is the tree's draft-side price; the gain
  is on E[T] (denken #85 gates the verify side, not the drafter).
- **RESOLVED (code + live): the deployed drafter is recurrent-fixed-prefix; draft
  tokens do NOT attend each other.** This settles openevolve's leg-3 concern
  ("if the tree emit re-enters the MTP head per-node WITHOUT accumulated self-KV,
  or position_ids don't increment, deep candidates degrade") **for the draft
  side**. The deployed eager draft loop (`sitecustomize.py:816-840`) proves it
  three ways: (1) `loop_metadata` is built **once** (`:811-814`) before the K-loop
  and reused every iteration; (2) `positions = self._get_positions(...)` and
  `slot_mapping = self._get_slot_mapping(...)` are called with the **same arg and
  never mutated in-loop** → constant position, constant slot → each draft step
  **OVERWRITES** the one KV slot (no cross-step self-KV accumulation, no position
  increment); (3) the **only** state carried forward is `self.hidden_states[:1] =
  hidden[:1]` (`:836`) → pure recurrence. So the deployed chain itself works
  WITHOUT accumulated self-KV / monotonic positions — they are not part of this
  drafter's mechanism. My emit (`_run_tree_emit_probe`) mirrors it exactly for
  EVERY node — `ctx = hidden_cache[parent]` restores the parent's hidden (the
  tree-correct analog of the chain's `hidden_states = hidden`), same
  `loop_metadata`/`_get_positions`/`_get_slot_mapping` — so the deep branch
  candidates (nodes 5,8,10,7) are formed by the **identical** recurrent mechanism
  as the deployed chain's tokens. openevolve's concern describes a **sequential
  self-attention** drafter; this one is recurrent, so it does not apply. Live
  spine-identity holding (Comp-1 validation) independently confirms it. Net:
  **Component 1 needs no draft-side tree mask, only the parent-hidden threading**,
  and the deep-branch risk is NOT on the draft side — the only remaining
  deep-branch question is verify-side acceptance (STAGE-2b / openevolve's free
  A10G ladder), where Component 3b already supplies monotonic `base+depth`
  positions and Component 2's qq_bias supplies the ancestor-only attention.

## Finding A detail — qq_bias makes the mask free

The verify is redirected to the **3D split-KV** path (`splitkv_verify_patch.py`
sets `max_seqlen_q=1`). The kernel is **unified**: one body with an `IS_3D`
constexpr; qq_bias is added to the score `S` at `:525`, upstream of the
`if IS_3D` output epilogue at `:548`. **So the redirected 3D verify applies
qq_bias too** — the "redirect for occupancy" and "tree mask" are compatible, no
conflict, no kernel fork. My `tree_spec.tree_causal_mask` docstring already
specified "shared KV prefix attended densely, NOT part of this intra-tree mask"
— which is exactly qq_bias semantics. The mask was designed right blind.

## Finding B detail — KV compaction is the continued-gen blocker

`num_accepted_tokens.gpu = (output_token_ids != -1).sum(dim=1)` then the next
step truncates seq_lens by the rejected count. This is a **count**, so vLLM
retains KV slots `[0, num_accepted)` in **layout order**. For the linear chain
the accepted tokens *are* the first `num_accepted` rows → correct. For a tree the
accepted path is e.g. rows `[0, 2, 5, 8]` (a salvaged rank-2 branch) → the first
`num_accepted=4` rows `[0,1,2,3]` are the WRONG slots → next-step prefix KV
corrupt → PPL break / greedy-identity loss. Fix = gather accepted path K/V →
contiguous, before bookkeeping. Costable but real surgery; defer it.
**ubel #157 (MERGED) pins HOW:** one fused/vectorized GPU launch across all 37
layers (`index_select`→`index_copy_` on the `[L,W,H,D]` K+V stack), commit-index
on-device (lawine #147 sync-free), no host readout → graph-capturable. A
host-bound per-layer Python loop is the **silent step-collapsing landmine** —
correctness/PPL-clean (passes every functional check) but ~122 ms/step → on the
0.382-salvage/step ladder it blows clear-500 to 32.6 (≫ 5.207 ceiling) and the
descent E[T]=5.04 collapses 522→77 TPS. The descent fix (Component 4) ARMS this
op on every salvage, so it is live the moment Component 3/4 lands — the relocate
must be fused **before** any benchmark/quota spend, not after.

## Recommended build order

1. **Single-step salvage PROBE (Components 1+2+3a+3b+4, NO 3c).** At a decode
   step, draft the M=16 tree, run ONE tree-masked verify forward over the M rows
   (qq_bias + tree positions) into **scratch** KV (don't mutate real state), run
   the descend walk, log salvage + tok/step + the both-halves asserts. Continue
   real generation on the untouched linear chain. → real-stack salvage number,
   zero quota, **no KV-compaction needed.** This is the GO/NO-GO the fleet waits
   on (advisor 11:35Z; fern #134 target E[T]≈5.0 with spine left at 0.679).
2. **Continued-gen integration (add 3c KV compaction — fused GPU COPY).**
   Only after the probe shows salvage ≈ ρ₂ = 0.4165. Variant DECIDED (leg-2 gate,
   below): the paged re-point is infeasible on this vLLM, so the relocate is the
   single fused/vectorized GPU **copy** (`index_select`+`index_copy_`, ubel #157/
   #163), NOT a host loop, or the measured E[T] gain never reaches wall_tps. Gives
   the wall_tps number (lawine #72, median N=3) for the eventual human-approved
   launch issue.

## STAGE-2b scratch-forward seam (mapped; the M=16 direct-E[T] forward)

The single-step probe splits into **2a** (E[T] PROJECTION from the STAGE-1 measured
rates via the validated closed form — DONE, `report_salvage_probe_stage2_etp.md`:
matched-spine **+15% M16 / +28% M32**, ~5.0 at M32@t1=0.773) and **2b** (the LIVE
M=16 tree forward that MEASURES E[T] directly + fires the both-halves assert at
M=16). Seam for 2b, mapped against installed vLLM (`gpu_model_runner.py`):

- **Forward hook:** `_run_model` (`gpu_model_runner.py:3719`) calls `self.model(...)`
  directly with **no graph dispatch** → the eager path a one-off M=16 forward needs
  (the target model is otherwise CUDA-graph-captured at `cudagraph_capture_sizes`;
  M=16 is not a capture size). Logits via `compute_logits(hidden[logits_indices])`
  (`:4315`), `logits_indices = arange(16)`.
- **positions:** `base + depth(node)` (Component 3b) into `self.model(positions=…)`.
- **qq_bias:** auto-injects through the `splitkv_verify_patch` wrapper when
  `TREE_QQ_BIAS_PROBE=1`, `TREE_QQ_BIAS_M=16`, `TREE_QQ_BIAS_PARENT=m16` and the
  verify batch presents M=16 rows (no call-site change).
- **KV — the real obstacle (corrects the naive "slot_mapping=-1" idea).** Filling
  `slot_mapping` with −1 (the `_dummy_run` pattern, `:5766`) skips the K/V write —
  but that is **WRONG for the tree verify**: the `[M,M]` qq_bias masks
  query×query attention *between* the 16 draft rows, so the deeper branch nodes
  (5,8,10 / 7) **must attend their tree-ancestors' K/V**, which therefore must be
  *written and readable*. So 2b needs a **separate scratch KV block** (1 block =
  16 slots): `block_table = [request prefix blocks (READ-ONLY) … + 1 scratch
  block]`, `slot_mapping → the scratch block's 16 slots`, `seq_lens = prefix_len+16`.
  The request's real blocks are only **read**; the scratch block is written then
  freed → real generation untouched. `compute_slot_mapping` (`:2101`) is the normal
  (mutating) path to bypass; `_build_attention_metadata` accepts a `slot_mappings`
  override (`:2205`, consumed `:2421`). This is genuine KV-cache-manager surgery
  (scratch-block alloc + combined block_table) — the one real-risk piece; it is
  the SAME machinery the continued-gen widened verify needs, so 2b's direct E[T]
  also arrives for free once the real verify widens to 16 rows.

## Build-contract relay (advisor 14:50Z — 3 legs, none redirect)

- **Leg 1 (wirbel #165, MERGED) — kernel build (Step 2), not the probe.** For the
  CONTINUED-GEN accept KERNEL the cleaner BUG-2 fix is **descent-ordered DFS node
  layout + ONE flat `target_argmax` gather** (`start_idx+pos` already correct),
  rather than my Component-3a **decoupled two maps** (`target_logits_indices` +
  `draft_token_rows`). Both are validated-equivalent; the wirbel #165 DFS-relayout
  is preferred for the kernel because PARENT_M16's node numbering is BFS-ish (node
  2 = root's rank-2 sits at flat pos 2, a sibling not a descendant) so a flat walk
  needs EITHER a remap (mine) OR a DFS relayout (#165). The single-step PROBE uses
  the CPU `descend_accept` (walks `children`, layout-agnostic) → unaffected; adopt
  the DFS layout when building the Triton descent kernel. Greedy cert for that
  kernel: wirbel #160 §5 assertion + `greedy_exact_harness.py --audit-kernel-symbol`
  (denken #158) for the GREEDY_EXACT rate-1.0 certificate before any launch.
- **Leg 2 (ubel #163, MERGED) — 3c relocate variant: GATE ANSWERED → build the
  FUSED COPY, not the paged re-point.** ubel banked two greedy-safe designs: fused
  `index_select`+`index_copy_` (35.3 µs → clear-500 bar **4.880**) and a paged
  slot-map / block-table re-point (zero-copy, 20.26 µs → bar 4.817) gated on
  *"build the paged variant **if** block-table plumbing allows the re-point with an
  on-device commit-index (no host readout)."* **I mapped the installed vLLM
  (server-venv ~0.22.1rc1) and the plumbing does NOT allow it** — the paged
  re-point is infeasible on this stack for two independent, verified reasons:
  1. **Block-table is CPU source-of-truth.** `MultiGroupBlockTable` /
     `BlockTable` (`v1/worker/block_table.py`) holds the table in a `CpuGpuBuffer`;
     every mutation writes `self.block_table.np` (CPU numpy: `append_row:118`,
     `move_row:133`, `swap_row:139`) and `commit_block_table:166-167` H2D-copies it
     (`copy_to_gpu`) **before every forward**. `slot_mapping` is then derived
     on-device FROM `block_table.gpu` by a Triton kernel (`:150-164`). So a re-point
     driven by an on-device commit-index would be **clobbered** by the next
     per-step `commit_block_table` unless we also write `block_table.np` on the
     host → a host sync that breaks CUDA-graph capture. (`gpu_input_batch.py:30,171`
     confirms the runner uses this legacy CPU-source table; the submission's
     `static_block_table` is a drafter-graph copy, `sitecustomize.py:203,222`, not
     a GPU-source verify table.)
  2. **Whole-block granularity.** `slot_ids = block_numbers*block_size +
     local_block_offsets` (`block_table.py:378`): the table maps logical→physical
     **blocks**, so a *scattered* per-token accepted path (spine + one salvaged
     branch, not block-aligned) is not expressible as a block-table edit anyway.
  **→ FUSED COPY BUILT + validated** (`scripts/profiler/tree_kv_relocate.py`,
  `report_comp3c_relocate.md`). Two corrections vs the banked spec, both forced by
  the *served* layout: (1) the 37 KV layers are **separate allocations** (one
  `torch.zeros` per `KVCacheTensor`, `gpu_model_runner.py:7012-7018`), so ubel's
  `[L,…]`-stack `index_select`/`index_copy_` has no zero-copy stack — the fused op
  reaches all layers via a **layer `data_ptr()` array Triton kernel** (gather→
  staging→scatter, 2 launches, aliasing-safe); (2) relocate width = `max_depth+1`
  (spine length, =10), not M — trailing verify rows are discarded anyway. On-device
  `commit_map`, no host readout. **Measured local (37-layer, bf16): equivalence
  1.0; SERVED cost = 8.3 µs/step (CUDA-graph replay) — UNDER the banked 35.3 µs →
  clear-500 bar ≥ 4.880.** Host-loop landmine = 12.5 ms/step (~1500×), confirming
  the 522→77 collapse. A naive *device* per-layer torch loop is 1.4 ms (74
  launches) → confirms ubel's "ONE launch" mandate is about launch count, not just
  host-residency. CPU twin for the commit-index: `tree_spec.descend_accept_path`.
- **Leg 3 (openevolve) — RESOLVED draft-side** (see Component-1 detail: recurrent
  drafter, deep candidates faithful). Verify-side deep ladder: openevolve offers a
  **free A10G per-position-ladder measurement** (hand them the built descent
  kernel's single-knob delta package) to confirm the real ladder ≥ the modeled
  E[T]=5.06 floor at zero quota — the no-risk substitute for STAGE-2b's verify-side
  scratch forward.

## Gates (unchanged)

branch-hit per first-divergence ≈ ρ₂ = 0.4165 (wirbel #83 per-position oracle) ·
verify-side ≤ 89 µs / per-op ≤ budget (denken #85) · tok/step > 3.844 toward ~5.0
(fern #134) · PPL ≤ 2.42 · greedy identity by construction · both-halves runtime
asserts (star-attn DISPATCHED via qq_bias **and** descend walk RAN on the tree
layout — chiku-inu). Decide gain on wall_tps median N=3. **No HF launch / no
oracle ping** without a human-approved issue.
