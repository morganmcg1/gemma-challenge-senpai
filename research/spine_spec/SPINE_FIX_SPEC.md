<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Depth-1 spine (BUG-1) build spec — `_dixie_fused_accept_prep_kernel`

**PR #160 · wirbel · build spec for the both-bugs (private-safe) topology.**
LOCAL CPU-only analytic spec. **No GPU / vLLM / HF Job / submission / kernel deploy.
BASELINE stays 481.53.** This is a buildable spec + a verified E[T], it does **not**
authorize a launch.

## 0. TL;DR for the builder (land #71 / spine-build seat)

- **The fix is one input contract, not kernel arithmetic.** The accept-prep kernel's
  `draft == target_argmax` greedy test is already correct. BUG-1 is that the value fed
  into `target_argmax` for the **depth-1 spine root** is gathered from the wrong logits
  row (rank-2 contaminated) by the upstream `target_logits_indices` index map.
- **Required change:** for the spine-root verify slot, `target_logits_indices` must
  index the **spine-root node's own** verifier-logits row (whose argmax is the verifier
  **rank-1** greedy token), not a drafter rank-2 sibling's row.
- **Effect:** depth-1 rank-1 acceptance `q1` rises `0.679 → 0.7287` (contamination
  fraction `f: 0.159 → 0`), lifting tree **E[T] 5.0564 → 5.2068** (the supply ceiling),
  i.e. official **~520 → ~535 (measured step) / ~538 (roofline step)**.
- **Greedy identity:** SAFE. The fix changes the accept **count** (speed), never an
  emitted token. The emit is always the verifier rank-1 argmax (Issue #124 RESOLVED).
- **Shared index map (denken #133):** the *same* corrected map plausibly fixes BUG-2
  descent too — coordinate with land #71's descending-walk build.

## 1. Root-cause recap (denken #133, MERGED 🟢)

The accept-prep consumes `target_argmax[start_idx + pos]` = the verifier's greedy
argmax at draft node `pos`. Those argmax values are gathered upstream from the
verifier's per-node logits via an index map (`target_logits_indices`). For the
**depth-1 spine root** (`pos = 0`), that index map is **rank-2 contaminated**: a
fraction `f ≈ 0.419` of root slots index the drafter's **rank-2 candidate** node's
logits row instead of the **spine root's own** (rank-1) row.

GPU-confirmed reconstruction (denken #133, run `k2dhcvbn`):

```
as_built_depth1 = (1 - f) * q_true + f * rho2
               = (1 - 0.419) * 0.728739760479042 + 0.419 * 0.4165047789261015
               = 0.598            (matches the GPU-measured deployed depth-1 accept)
```

where `q_true = 0.7287` is the verifier rank-1 acceptance of the drafter's rank-1
draft and `rho2 = 0.4165` is the rank-2 marginal (`P(draft == verifier rank-2)`).
denken #133 also GPU-ruled-out fp32 as the fix (NET ~0pp; exact-tie reshuffles) and
intrinsic/structural mismatch (0.0pp) — the 13.07pp deficit is **~96% build-plumbing**,
recoverable to `q_true`.

**Independent public corroboration (chiku-inu, board `20260614-111022-934`):** the
full-bench fp32-star run measured `depth1_accept = 0.591` over 13184 steps
(cumulative), independently bracketing denken #133's GPU anchor 0.598, and reconfirmed
the oracle's `0.679 @ 1024` shorter-sample depth-1. Both fall inside this spec's
contamination band (`f ≈ 0.42` as-built → `f ≈ 0.16` descent-only). chiku-inu also
re-confirmed fp32 is insufficient (agreeing not to spend quota on fp32-only) — so the
build-plumbing index-map fix specced here, not precision, is the depth-1 lever.

## 2. Kernel interface — the depth-1 spine path

Reference: deployed `_dixie_fused_accept_prep_kernel`,
`submissions/fa2sw_precache_kenyan/sitecustomize.py:921`. The tree accept-prep (land
#71) reuses this kernel's structure on the flat-tree draft layout. The depth-1 spine
path is `pos = 0` (the root verify node).

### Inputs (per request `req_idx`, depth-1 spine slot)

| symbol | source | spec'd meaning (the contract) |
|---|---|---|
| `draft_token_ids[start_idx + 0]` | drafter | drafter rank-1 proposed token at the spine root |
| `target_argmax[start_idx + 0]` | **verifier logits via `target_logits_indices`** | **MUST be argmax of the SPINE-ROOT node's logits row = verifier rank-1 greedy token.** This is the only field BUG-1 corrupts. |
| `cu_num_draft_tokens[req_idx]` | scheduler | unchanged (chain extent / spine length) |
| `bonus_token_ids[req_idx]` | verifier | unchanged (post-spine bonus token) |

### Outputs (depth-1 spine slot)

| symbol | spec'd meaning |
|---|---|
| `output_token_ids[row_offset + 0]` | emitted greedy token at the spine root = **verifier rank-1 argmax** (greedy-identity-bearing field) |
| `valid_counts[req_idx]` | accepted-prefix length (BUG-1 raises this in expectation) |
| `next_token_ids[req_idx]` | next decode token |

The accept test and the emit at the spine root **read the same** `target_argmax` value.
Correcting the index map fixes both simultaneously — accept rises to `q_true`, emit
stays the verifier rank-1 argmax.

## 3. Current broken behavior vs. spec'd behavior

The deployed kernel treats `pos = 0` identically to every other position — a strictly
linear break-on-mismatch chain-reject reading a flat `target_argmax` array:

```python
# CURRENT (sitecustomize.py:942-951) — pos=0 handled like any chain position
for pos in range(num_draft_tokens):
    if not rejected:
        draft_token_id   = tl.load(draft_token_ids_ptr + start_idx + pos)
        target_argmax_id = tl.load(target_argmax_ptr   + start_idx + pos)  # <-- pos=0 row
        rejected     = draft_token_id != target_argmax_id                  #     is RANK-2
        valid_count  = pos + 1                                             #     CONTAMINATED
        next_token_id = target_argmax_id
        tl.store(output_token_ids_ptr + row_offset + pos, target_argmax_id)
```

For the depth-1 spine root, `target_argmax_ptr + start_idx + 0` resolves (via the
broken `target_logits_indices`) to a drafter rank-2 sibling's logits-row argmax with
probability `f ≈ 0.419`. The `draft == target_argmax` test then compares the drafter
rank-1 draft against the rank-2 token → accept collapses `0.7287 → 0.598`.

**Spec'd depth-1 conditional** (what the kernel must be GIVEN, arithmetic unchanged):

```python
# SPEC'D — identical arithmetic; the CONTRACT on the pos=0 input changes.
# target_argmax_ptr + start_idx + 0  MUST resolve to the SPINE-ROOT node's own
# logits-row argmax (verifier rank-1). Then for pos=0:
#   target_argmax_id == verifier_rank1_argmax(spine_root)        # rank-1, not rank-2
#   accept  iff  draft_token_id == verifier_rank1_argmax(spine_root)   # q1 -> 0.7287
#   emit    =    verifier_rank1_argmax(spine_root)               # greedy-identical
```

The diff is **not** in the loop body. It is in the **index map that populates
`target_argmax`** (Section 4). The kernel optionally gains a build-time assertion
(Section 5) so the contract is validated rather than silently violated.

## 4. The exact diff — corrected `target_logits_indices` contract

The verifier-logits gather that feeds `target_argmax` must, for the spine-root verify
slot, select the spine-root node's logits row (the row whose argmax is the verifier
rank-1 greedy token under the verified prefix), **not** a drafter rank-2 candidate row.

Contract (land #71's tree-build verify path):

```
# For every request req and its spine root node r0 (depth 1):
#   target_logits_indices[slot(r0)]  ==  logits_row_of(r0)        # the node's OWN row
#   target_argmax[slot(r0)]          ==  argmax(target_logits[logits_row_of(r0)])
#                                    ==  verifier_rank1_greedy(prefix(r0))
# BROKEN (BUG-1): target_logits_indices[slot(r0)] points at a drafter rank-2
#                 sibling's row with marginal probability rho2 = 0.4165, so
#                 target_argmax[slot(r0)] is that sibling's argmax (rank-2).
```

denken #133 **shared-index-map hypothesis:** the same `target_logits_indices` class
that mis-maps the spine root also mis-maps the BUG-2 descent traversal
(realized E[T] 2.10 ≪ rho-opt 4.81). One corrected index map may close **both** bugs;
land #71 should build the spine-root rank-1 mapping and the descending-walk mapping
against a single corrected `target_logits_indices`, then assert both (Section 5).

> Note: land #71 holds the runnable tree build, so the literal source line of the
> mis-map lives there (out of this CPU-only spec's tree). This spec fixes the
> **contract** the index map must satisfy and gives the builder a self-validating
> assertion; pinpointing the exact offending line is a one-line grep in land's build.

## 5. Build-time validation hook (kernel-side assertion)

To convert the silent contract into a hard build gate, land #71 wires a debug
assertion (compiled out of the hot path) that, for each spine root, the accept-test
row equals the emit row equals the rank-1 logits row:

```python
# DEBUG build only (DIXIE_FUSED_ACCEPT_PREP_ASSERT=1): per request, at pos=0
#   assert target_logits_indices[slot(root)] == logits_row_of(root)
#   assert target_argmax[start_idx] == argmax(target_logits[logits_row_of(root)])
# Aggregate gate over the calibration prompts:
#   measured_depth1_accept  >=  0.72   (q_true band; pre-fix value is ~0.60-0.68)
#   branch_hit_rate         ~=  rho2 = 0.4165   (BUG-2 descent sanity)
```

This is the property denken #158's descent-exactness harness will later exercise on
the built kernel; the spec-level argument is in Section 6.

## 6. Greedy-identity safety (spec-level) → `spine_fix_greedy_identity_safe = TRUE`

The fix is greedy-token-identical. Argument:

1. **Greedy output = the verifier's argmax sequence.** In greedy speculative decode,
   the emitted token at each position is the verifier's greedy argmax under the
   accepted prefix. The accept walk only decides **how many** draft tokens are
   consumed before the verifier's argmax is appended — never **which** token is
   appended. (Deployed precedent: `sitecustomize.py:803` — "Verifier argmax is
   unaffected => greedy identity / PPL unchanged.")
2. **Acceptance changes count, not values.** Accepting zero drafts still emits the
   verifier rank-1 argmax at the root (1 tok/step, slow but correct); accepting more
   just emits more verifier-argmax tokens per step (fast). Output identity is invariant
   to the accept count.
3. **The fix targets the rank-1 row only.** Correcting `target_logits_indices` makes
   the spine-root accept-test row **and** emit row both equal the verifier rank-1 row.
   The deployed stack is already greedy-exact (Issue #124 RESOLVED), so the emit row is
   already rank-1; the fix aligns the **accept-test** row to the (already-correct) emit
   row. No emitted token changes.
4. **The fix only raises accept probability** (`q1: 0.679 → 0.7287`). It cannot lower
   emit fidelity — there is no path by which de-contaminating the accept-test row alters
   an emitted greedy token.

⇒ The spine fix is **speed-only**; verified greedy output is untouched. PPL guardrail
and the multimodal contract are unaffected (no served-model change).

## 7. Verified E[T] and official projection (the both-bugs ceiling)

Extending wirbel #135 / #152 E[T]-DP with the **spec'd** contamination model
`q1(f) = (1-f)·q_true + f·rho2` (not an idealized override), on the measured oracle
ladder `[0.674, 0.350, 0.203, 0.131, 0.089, 0.060, 0.037]` (board `20260614-100550-487`,
ρ₂ = 0.4165), depth-9 rho-optimal M=32/max-branch-3 topology:

| build | depth-1 q1 | contamination f | E[T] | official @ 1.2182 (τ=1) | official @ 1.2127 (τ=1) |
|---|---|---|---|---|---|
| as-built (both bugs) | 0.598 | 0.419 | 4.811 | 494.7 | 497.0 |
| descent-only (BUG-2 fixed) | 0.679 | 0.159 | **5.0564** | 519.9 | 522.3 |
| **both-bugs (spec'd fix, f→0)** | **0.7287** | **0** | **5.2068** | **535.4** | **537.9** |

- **`both_bugs_E_T_specced = 5.2070`** (TEST metric). Idealized override (wirbel #135) =
  5.2070 → **idealization gap = 0.0** (the spec'd fix `q1(0) ≡ q_true` *is* the override).
- **`spine_spec_self_test_passes = TRUE`** (PRIMARY): reproduces descent-only 5.0564,
  both-bugs 5.2068, denken #133 as-built 0.598, and denken #128 `ET_tree(0.598)=4.811`.
- MC cross-check both-bugs E[T] = 5.198 vs DP 5.207 (|Δ| = 0.009, within MC noise).
- **Clears 500 with margin:** +0.345 over the measured-step bar (4.862), +0.387 over
  ubel #154's lowered bar (4.808–4.820), at τ ∈ [0.9924, 1.0].

Private-stability (the requirement this specs toward, stark #151): descent-only's
private drop-tolerance is **5.89%** (fails 6–9% of BASELINE.md's 4–9% band); the
**both-bugs topology lifts tolerance to 9.88%**, covering the whole band — so the
spec'd depth-1 fix is what makes the tree the **private-safe launch shot**.

## 8. Hand-off

- This is the build spec for the **both-bugs (private-safe) topology** — the second
  build after land #71's descent walk, now a launch *requirement* (stark #151), not a
  522→538 margin.
- Pairs with lawine's `both-bugs-step-cost` (does the spine fix keep step ≈ 1.2182).
- Feeds fern #155's consolidator and the eventual `Approval request: HF job` go/no-go.
- denken #158's descent-exactness harness exercises the built kernel against Section 5.

Artifacts: this spec + `report_spine_spec_verify.md` + `spine_spec_results.json`
(verifier `scripts/profiler/spine_spec_verify.py`). W&B group `depth1-spine-build-spec`.
