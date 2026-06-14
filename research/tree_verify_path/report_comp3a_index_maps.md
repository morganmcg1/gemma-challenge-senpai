# PR #71 Component 3 — tree verify metadata (index maps + positions) + denken #133 cross-check

**Status:** built + CPU-validated (zero quota). Component 3a supplies the widened
verify index metadata (`target_logits_indices`, node order, bonus) and resolves
denken #133's **load-bearing hypothesis** with a concrete root cause: the SAME
`target_logits_indices` plumbing corrupts BOTH the depth-1 spine AND the BUG-2
descent, via one overloaded array — and one decoupling fixes both. Component 3b
supplies the per-node RoPE position/depth map.

Builders + cross-checks live in `scripts/profiler/tree_spec.py`
(`tree_verify_metadata`, `deployed_linear_metadata`, `tree_verify_positions`,
selfcheck §12–14). Verdict: `research/tree_verify_path/comp3a_index_map_verdict.json`
(all 17 checks PASS).

---

## The deployed verify-path index extraction (exact seams, pinned vLLM 0.22.1rc1)

| step | seam (file:line) | what it computes |
|---|---|---|
| build metadata | `gpu_model_runner.py:2730` `_calc_spec_decode_metadata` | linear-chain index math |
| target rows (A) | `:2772-2776` `target_logits_indices = repeat(base,nd)+arange` | `[0..nd-1]` (contiguous) |
| **draft gather (B)** | `:2798` `draft_token_ids = draft_token_ids[target_logits_indices + 1]` | **reuses (A)** |
| target argmax gather | `rejection_sampler.py:150` `dixie_all_argmax[metadata.target_logits_indices]` | per-draft target argmax |
| all-row argmax | `serve.py:410` `dixie_all_argmax = logits.argmax(-1)` | **every** verify row (node order) |
| accept (linear, BUG-2) | `sitecustomize.py:1188-1194` `draft[start+pos] != target_argmax[start+pos]` | break-on-first-mismatch |

The "do not reconstruct — use the deployed extraction" anchor:
`tree_spec.deployed_linear_metadata(7)` replays `:2747-2798` and is byte-identical
to a numpy replay of those exact ops (`tli=[0..6]`, `bonus=[7]`, draft gather
`[1..7]`), and `tree_verify_metadata(linear)` coincides with it. So the tree
builder is diffed against the REAL construction, not a parallel reimplementation.

## The finding — one array, two maps: the override-A-breaks-B trap

`target_logits_indices` is overloaded for two semantically distinct maps:

- **(A) target-row map** — which target-argmax row each draft node is checked
  against. Correct tree value: `[parent[i] for i in 1..M-1]`.
- **(B) draft-token gather** — which verify row holds each draft node's own
  token. Correct tree value: node order `[i for i in 1..M-1]`.

For the deployed **chain** these coincide (`parent[i]=i-1` ⇒ A and B differ by
exactly +1), so one array safely serves both. For a **tree** they diverge. The
obvious depth-1 fix — override (A) to the tree parent-map so each draft node is
verified against its PARENT's target row — flows through the shared line
`:2798` and **silently corrupts (B)**: the draft-token gather then reads row
`parent[i]+1` instead of node `i`'s own row for **every node off the initial
chain** (`parent[i] != i-1`): **M16 = 10/15, M32 = 30/31 draft slots corrupted.**
At the root the rank-2 child's slot collapses onto the rank-1 sibling's token —
exactly denken #133's *"the root verify-row compares against the drafter's RANK-2
instead of RANK-1 (~42% rank-2 contamination)"*. So fixing (A) alone IS the trap.

This is the concrete mechanism for denken's load-bearing hypothesis: **the same
index-map override that fixes the depth-1 spine corrupts the descent.** The fix
is to supply (A) and (B) **separately** (`tree_verify_metadata`).

## The descent sidesteps the flat gather (denken item-2 diff)

The descend walk (Component 4, `descend_accept`) does **not** consume
`target_logits_indices` at all. It consumes RAW node-order arrays + the static
children-CSR and applies the parent→child comparison from the tree topology, so
it is immune to the conflation by construction. The salvage-probe wiring contract:

| descend kernel input | source (node order, pre-gather) |
|---|---|
| `node_argmax[i]` | `serve.py:410` `dixie_all_argmax` — ALL M rows (NOT `[target_logits_indices]`) |
| `draft_token[i]` | `input_ids[logits_indices]` — node order, BEFORE the `:2798` re-gather |
| topology | `tree_spec.build_children_csr(tree)` — children in rank order |
| bonus | `tree_verify_metadata(...)['bonus_logits_indices']` — leaf rows |

**Cross-check (deterministic, M16 root width-2):** with the verifier argmax at the
root equal to the rank-2 child's token, the node-order (B) gather **salvages** the
branch (descends node 2, committed 2, salvage `[(0,2)]`); the conflated `tli+1`
gather puts node 1's token in node 2's slot → root **chain-rejects** (committed 1,
0 salvage — the 3% signature). One decoupling flips the descent from broken to
salvaging. Confirms denken's "one corrected index map may fix both."

## Component 3b — verify positions / depth map

Each node's RoPE position is `base + depth(node)` (`tree_verify_positions`);
**sibling branches at the same tree depth SHARE a position** — they are
alternative continuations of the same decode step. For the deployed linear chain
`depth == node-index`, so this reduces to the consecutive positions vLLM assigns
the K draft rows today (validated). On the trees: the rank-1 **spine sees
contiguous RoPE `0..9`** (M16 and M32 both depth-9), every ancestor's position
strictly precedes its descendants', and width-2/3 branches share `parent_depth+1`.
The contiguous-along-each-path property is what makes an accepted root→leaf path
indistinguishable from a linear chain to the attention — the RoPE half of the
BUG-1 spine-identity guard (the mask half is Component 2). Positions feed the
**verify forward** (RoPE), not the descend kernel.

## Implications for the live wiring (salvage probe, next)

1. Override `target_logits_indices := [base + parent[i]]` (A) for the dixie target
   gather **and** keep the descend kernel on the raw node-order `dixie_all_argmax`
   (`serve.py:410`) + node-order draft (`input_ids[logits_indices]`, pre-`:2798`).
   Do **not** let the (A)-override flow into the draft-token gather.
2. `logits_indices` must enumerate all M tree rows; prewarm (`serve.py:487-492`)
   hardcodes M=8 → widen to the tree M.
3. Greedy identity is preserved by construction: every emitted token is a verifier
   argmax; the rank-1 spine is the deployed chain (BUG-1 guard, Component 1).

## Gates / scope

CPU-complete (index correctness + the denken cross-check). Component 3a has no
isolated GPU behavior — its live exercise is the single-step salvage probe
(1+2+3a/3b+4). No fp32 spend (denken #133 item 3: fp32 refuted as a fix). No HF
launch / no oracle ping without a human-approved approval issue.
