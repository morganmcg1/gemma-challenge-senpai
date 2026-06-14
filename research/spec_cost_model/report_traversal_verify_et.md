# Traversal Verification E[T] gate on wirbel #83's M=32 tree (PR #88)

**Verdict: RED.** On wirbel #83's DP-optimal M=32 draft tree, the leaf-to-root
"Traversal Verification" acceptance rule (NeurIPS 2025, OpenReview 8nOMhDFpkU)
realises **`traversal_et_uplift_pct = +0.0000%`** over the deployed root-to-leaf
full-tree verifier, with **`traversal_greedy_violation_count = 0`**. The uplift is
**provably zero under the challenge's greedy decode contract, for any tree and any
corpus** — confirmed empirically on wirbel's exact topology, on real #80 drafter
ranks, and by exhaustive small-tree enumeration. **Recommend land #71 keep standard
root-to-leaf verification; do not integrate traversal.**

Primary `traversal_et_uplift_pct = 0.0` (gate threshold GREEN ≥+5%, AMBER +2–5%,
RED <+2%) → **RED**. Test `traversal_greedy_violation_count = 0` (lossless — it is
trivially lossless because it equals the deployed rule).

---

## 1. The question (PR #88)

land #71 deploys tree-verify that walks the draft tree **root-to-leaf**, accepting
the longest consistent *prefix* (SpecInfer/Medusa full-tree verification: every
child of a node is a candidate, so a rank-2 sibling that matches the target argmax
at a divergence *is* accepted). Traversal Verification instead walks
**leaf-to-root**, accepting any consistent path ending at a leaf, with the stated
goal of recovering sibling-subtree mass that root-to-leaf's recursive rejection
discards — the mass wirbel #83's salvage oracle (rho2 = 0.4165) upper-bounds. PR #88
asks: on wirbel's M=32 tree, **how much E[T] does the deployable leaf-to-root rule
actually realise vs root-to-leaf, and is it 100% greedy-identical?**

## 2. The decisive result is structural (greedy ⇒ the two walks are identical)

The challenge serves **greedy** decode — the emitted sequence must be token-identical
to plain argmax autoregressive decoding (a hard validity gate; `program.md`
"Breaking greedy token identity" is *Not allowed*). Under greedy:

- The target's argmax at each position is a **single token**. At any tree node, at
  most **one** child token can equal it (siblings are distinct top-k draft tokens).
- Therefore the set of fully target-consistent tree paths is a **unique chain** — the
  prefix of the greedy target output that the tree happens to contain.
- **Root-to-leaf** descends that unique chain to its maximal end (at each node the
  matching child, when it exists, is unique, so there is no branch to mis-pick).
- **Leaf-to-root** selects the longest fully-matching root→leaf path — the **same
  unique chain**.

⇒ Under greedy, leaf-to-root accepts **exactly the same tokens** as root-to-leaf, so
`traversal_et_uplift_pct = 0` by construction and both emit the identical greedy
chain (`traversal_greedy_violation_count = 0`). This holds for **any tree and any
corpus** — the result is a property of the *decode regime × verify walk*, not of the
data, so the per-source split (aime / gpqa / mmlu_pro) is uniformly **0** without
needing a per-source capture. The independent researcher-agent read of the paper
confirms: Traversal Verification's losslessness/uplift is a **sampling-regime**
mechanism (it recovers probability mass split across multiple consistent siblings);
at temperature 0 the mechanism is **vacuous** because there is only one consistent
sibling.

**wirbel's rho2 = 0.4165 is already realised by root-to-leaf.** rho2 is the rank-2
sibling rescue ratio — the value of the **tree** (branches) over a linear chain.
Full-tree root-to-leaf verification accepts a matching rank-2 sibling exactly as it
accepts the rank-1 spine child; the rescue is in the E[T] of the tree itself. It is
**not** incremental headroom for traversal.

## 3. Empirical confirmation on wirbel's exact M=32 topology

`scripts/profiler/traversal_verify_et.py` implements **both** acceptance walks
(`walk_root_to_leaf`, `walk_leaf_to_root`) and runs four independent legs. The MTP
acceptance model reuses wirbel #83's validated `build_depth_pvecs_measured` /
`score_tree_depthrank` machinery verbatim (measured top-1 = 0.7287, rising spine
q = [0.729 … 0.847], rho_cond = [0.4165, 0.2655, 0.1908]).

| Leg | Regime | E[T] root-to-leaf | E[T] leaf-to-root | uplift % | greedy violations |
|-----|--------|-------------------|-------------------|----------|-------------------|
| **A** physical (M=32, 400k MC) | **greedy** | **5.2140** | **5.2140** | **+0.0000** | **0** |
| **B** contrast (M=32, 400k MC) | sampling-proxy | 4.4324 | 4.6348 | +4.5669 | 26 984¹ |
| **C** real #80 ranks (M=32 spine, 1868 steps) | greedy | 3.3330 | 3.3330 | +0.0000 | 0 |
| **D** exhaustive (all trees n≤6) | greedy-valid | — | — | — | 0 mismatches / 872 |

¹ Leg B's "violations" are steps where leaf-to-root accepts a *longer* path than
root-to-leaf — i.e. the regime where traversal pays. It is reported as a mechanism
check, not a greedy violation.

- **Anchor.** `score_tree_depthrank` on the M=32 topology gives analytic
  E[T] = **5.20695**, matching wirbel #83's reported **5.207** to |Δ| = 4.6e-5 —
  confirming the acceptance model is his. (Leg A's MC E[T] = 5.2140 agrees within
  Monte-Carlo noise at 400k trials.)
- **Leg A (the primary result).** On wirbel's exact tree under his measured greedy
  acceptance, the two walks accept an identical token set on **every one of 400 000
  trials**: 0 steps where traversal is longer, **0 steps with ≥2 matching siblings**
  (the greedy single-match invariant), uplift **+0.0000%**.
- **Leg B (mechanism check — the walks are NOT a trivial no-op).** Relax greedy to a
  sampling-style regime where each sibling independently carries target mass
  (>1 sibling may match). Here leaf-to-root **strictly beats** root-to-leaf:
  **+4.57%**, traversal longer on 6.75% of steps, ≥2 matching siblings on **22.06%**
  of steps. This pinpoints that the *only* condition under which traversal pays is
  "two matching siblings", which **greedy decoding forbids**.
- **Leg C (real data).** Driving the M=32 spine with #80's real per-position drafter
  hit-ranks (greedy debug-1k corpus, 20 sequences / 1868 steps) gives identical
  E[T] under both walks — uplift **0** on realised ranks. (#80's drafter is EAGLE-3;
  the uplift is drafter-independent, so this is a valid real-data anchor.)
- **Leg D (exhaustive proof).** Over all rooted trees with n ≤ 6 and all
  greedy-valid match labellings (872 cases), root-to-leaf and leaf-to-root return
  the **identical path every time** (0 mismatches). Dropping the greedy invariant,
  17/442 unrestricted labellings have leaf-to-root strictly longer — proving the two
  walk implementations genuinely differ off-greedy and Leg A's zero is real.

## 4. rho2-capture framing (what land #71 should take away)

- Full-tree **root-to-leaf already realises +24.65%** E[T] over the rank-1 spine
  alone (spine-only E[T] = 4.1773 → tree E[T] = 5.2070). **That** is where wirbel's
  rho2 rank-2 rescue is cashed in.
- Traversal's **marginal** capture of rho2 *over root-to-leaf* = **0%**. There is no
  additional losslessly-recoverable sibling mass under greedy.

## 5. Why Step 0 (CPU) was decisive and Step 1 (GPU) is unnecessary

PR #88's Step 0 envisaged replaying wirbel's **per-node** target tokens to compute
both rules offline. wirbel's capture is **aggregate-only** (q, rho_cond, topology —
no per-node target tokens), so that literal replay is not possible. But the question
does not need per-node capture: under greedy it collapses to a **structural
identity** (Section 2) that holds for every tree and corpus. The four legs confirm
that identity on wirbel's exact topology, on real ranks, and exhaustively — at
**31.7 MB peak RSS, zero GPU, ~minutes**. A Step 1 serve-faithful GPU capture of the
deployed MTP tree would only re-confirm a proven identity at real GPU cost and is
therefore **not warranted**. (The capture would be the right tool if the challenge
served *sampling*; it does not.)

## 6. Gate and recommendation

| Gate input | Value |
|---|---|
| `traversal_et_uplift_pct` (primary) | **+0.0000%** |
| `traversal_greedy_violation_count` (test) | **0** |
| GREEN ≥+5% & 0 violations / AMBER +2–5% / RED <+2% | **RED** |

**Recommend land #71 keep standard full-tree root-to-leaf verification and not
integrate leaf-to-root traversal.** The lever pays only under stochastic sampling
(Leg B: +4.57%), which the challenge's greedy contract forbids. Close the lever
cleanly. The deployed serving frontier is unchanged (no served-file edit): deployed
`fa2sw_precache_kenyan` 481.53 official TPS, PPL 2.3767, greedy 128/128.

## 7. Public-evidence note

- **wirbel #83 (MERGED):** DP-optimal M=32 max-branch-3 topology + salvage oracle
  rho2 = 0.4165 (`research/spec_cost_model/rho_optimal_topology_results.json`,
  `report_rho_optimal_topology.md`). Used here as the fixed input tree and
  acceptance model; reproduced his E[T] = 5.207 as the anchor.
- **land #71 (in-flight):** the M=16/32 tree-verify serving path (~+21.8% → ~586).
  This analysis tells land **which** acceptance rule to deploy: root-to-leaf.
- **fern #80 (MERGED):** serve-faithful native-acceptance machinery, extended here
  (an acceptance-RULE measure on the same MTP/tree, not a drafter change).
- **Traversal Verification**, NeurIPS 2025, OpenReview 8nOMhDFpkU (arXiv 2505.12398);
  **SpecInfer** arXiv:2305.09781 for the root-to-leaf baseline. The paper's
  losslessness theorem is the standard speculative-decoding argument extended to
  non-prefix paths under **sampling**; at temperature 0 it reduces to the deployed
  rule.

## 8. Reproduce

```bash
cd target/
python scripts/profiler/traversal_verify_et.py --wandb_group traversal-verify-et
# CPU-only, no GPU; ~minutes; peak RSS 31.7 MB
# writes research/spec_cost_model/traversal_verify_et_results.json
```

- **W&B run:** `yiwl2jfj` — wandb-applied-ai-team/gemma-challenge-senpai, group
  `traversal-verify-et`.
- **Peak memory:** 31.7 MB RSS (CPU; zero GPU).
- **No serving run / no `summary.json` / no HF Job** — Step 0 CPU analysis was
  decisive; Step 1 GPU capture not needed.
