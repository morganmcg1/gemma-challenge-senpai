# PR #71 — Tree-verify serving path: descend-walk keystone (Component 4 / BUG-2)

**Status:** the descending tree-accept walk — the decisive 500-lever per fern #134
(BUG-2 fixed alone, spine left at 0.679 → E[T]≈5.056 → official 522) — is **built
and validated on both CPU (reference) and GPU (Triton kernel)**, GPU-free salvage
logic fully de-risked. The remaining work is the heavy live vLLM integration
(tree-shaped `SpecDecodeMetadata` + tree drafter forward + star-attn mask
dispatch); this report scopes it precisely.

## Local control (paired A/B basis, lawine #72 meter)

`scripts/local_prevalidate.py` on `submissions/fa2sw_precache_kenyan`, A10G,
`CUDA_VISIBLE_DEVICES=0`, 16×512 decode:

| metric | value | gate |
|---|--:|---|
| wall_tps (completion_tokens / decode_duration) | **413.76** | local/exploratory, not official |
| PPL | **2.2347** | ≤ 2.42 ✓ |
| frontier stack engaged | splitkv-verify M=8→3D, onegraph K=7, dixie-fused-accept, fa-sliding, precache | not a degraded fallback |

Artifacts: `research/local_validation/fa2sw_kenyan_land_repro/`.

## What BUG-2 is, mechanically (localized in the deployed code)

`sitecustomize.py:920-963` `_dixie_fused_accept_prep_kernel` is a **strictly
linear break-on-mismatch** accept: `for pos in range(num_draft_tokens): rejected
= draft[pos] != target_argmax[pos]`. On a flat tree layout the rank-2/3 sibling
branches (all children of one parent) sit at later positions, so the break never
reaches them → chain-rejection → branches unreachable → the **~3% salvage**
signature every prior broken external build hit (byteshark `tree-v2`,
cheesetaco-cdx salvnodefix; chiku-inu root-cause). `serve.py:416` gathers
`dixie_target_argmax = dixie_all_argmax[target_logits_indices]` — the index map
is already the right hook; the **walk** is what was wrong.

## The fix — descending tree-accept walk (built + validated)

`scripts/profiler/tree_spec.py::descend_accept` (CPU reference) +
`scripts/profiler/tree_accept_kernel.py::_tree_accept_kernel` (GPU Triton twin).
At each accepted node the walk scans **all** children (rank-1 spine + rank-2/3
branches) for the verifier argmax and descends the first match, so a rank-2
branch salvages ~ρ₂ of first-divergences. Every emitted token is a verifier
argmax → **greedy identity preserved by construction** (the tree only changes how
many tokens commit per step, never which token the verifier authoritatively
chooses).

### Validation (all green, zero quota)

1. **Reproduces the deployed linear chain exactly** — full-accept and
   mismatch-at-pos-k cases match the linear kernel's emitted prefix + bonus.
2. **Closed-form E[T] == Monte-Carlo descend simulation** on the measured ρ
   ladder: M16 E[T]=3.974 (cf 3.975), M32 E[T]=4.553 (cf 4.554).
3. **Branch-hit == ρ₂ — the 3%-vs-41% discriminator:** simulated first-divergence
   rank-2 catch = **0.4182 (M16) / 0.4154 (M32) ≈ ρ₂ = 0.4165** (wirbel #79 local,
   byteshark 0.413 official). A correct walk salvages 41%, not 3%.
4. **GPU kernel == CPU reference bit-for-bit** over 1100 random trials
   (lin8/M16/M32), plus an explicit deterministic case proving the kernel
   **descends a spine-miss into the rank-2 branch** (node 2 → node 5, commits 3)
   where the linear kernel would stop at 1.

Measured-ρ E[T] (flat-p closed form, a structural lower-bound cross-check of the
calibrated fleet models, **not** the headline projection): linear-M8 3.396, M16
3.975 (+17.1%), M32 4.554 (+34.1%). Headline official TPS (~569, +18.2%) comes
from the calibrated models (wirbel #74/#76/#79/#83 acceptance + denken #68/#85
cost + fern #134 recovery matrix); the walk here is their serving-path realizer.

## Topology (reused, not re-derived — wirbel #83)

- M16 step-1 milestone: `parent = [-1,0,0,1,1,2,3,4,5,6,8,9,11,12,13,14]`
  (max-branch-2, depth-9). Cheapest verify (Marlin tile-1), simplest — build first.
- M32 primary: `parent = [-1,0,0,0,1,1,1,2,3,4,4,5,7,9,9,10,11,12,13,15,16,17,18,19,20,21,22,24,25,26,28,29]`
  (max-branch-3, depth-9). TPS-optimal, cliff-pinned at M=32 (denken #68: M≤32
  bandwidth-free, M=33 +53% tile cliff).

`tree_spec` reproduces `report_sequoia_dp.md` linear F exactly (2.454/2.976/
3.111/3.117 @ M4/8/16/32) and validates both #83 arrays' structure.

## Remaining: the live serving integration (the heavy vLLM patch)

The deployed proposer + verify are **chain-shaped throughout** vLLM. Four
components, each with its integration point and the both-halves runtime assert
that catches the layout bug (chiku-inu: star-attn DISPATCHED **and** the walk
RUNS on the tree layout):

| # | component | site | status | difficulty |
|---|---|---|---|---|
| 4 | descending accept walk | replace `sitecustomize.py:920-963` accept kernel | **built + validated (this PR)** | done |
| 1 | drafter tree-emit (spine-identical rank-1 + rank-2/3 branch forwards) | `sitecustomize.py` `propose_onegraph` ~427-564 | reference structure done (`tree_spec`); live emit + spine-identity guard remain | hard (eager OK; graph-capture is the size-29 crash → enforce-eager) |
| 2 | star-attn tree-causal mask **dispatched** for tree rows | thread ancestor `attn_bias` through `splitkv_verify_patch.py` (TRITON_ATTN; FlexAttention is force-overridden on gemma-4-E4B per openevolve) | mask + CSR topology built (`tree_spec`); live dispatch remains | hard (triton attention) |
| 3 | widen verify M=8→16/32 + tree `SpecDecodeMetadata` (target=parent rows, bonus=leaves, output buffer ≥ depth+1) + fix M=8 prewarm `serve.py:487-492` | vLLM GPUModelRunner spec-metadata construction | index maps built (`tree_spec.verify_index_maps`); live metadata override remains | hard (deep vLLM; chain→tree) |

Components 2+3 are the genuine surface external teams have been grinding on
(vLLM spec-decode assumes a linear chain in metadata, KV slot mapping, and
attention). The walk (Component 4) — the lever fern #134 shows clears 500 — is
the de-risked keystone; the remaining build is plumbing that must pass the
salvage gate (branch-hit ≈ 0.41) + the denken #85 cost gate (verify-side ≤ 89 µs)
on a short local run before any approval-gated launch.

**No HF launch / no oracle ping** — local only; official confirmation is a
separate human-approved issue (Issue #46 pattern).
