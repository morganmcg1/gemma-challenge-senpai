# PR #71 — `greedy_exact` decode-step-identity cert (tree-verify accept rule)

**Scope (read this first).** This certifies **one thing**: the tree-verify accept
rule preserves greedy identity **at the decode-step level, by construction**, and
the supporting KV surgery is a bit-exact copy. It is **NOT** an end-to-end served
greedy-identity measurement of a live tree decode — **no live tree decode exists
yet** (`treeverify_served_gain_MEASURED_realized = 0.0`; the served stack is still
linear MTP K=7, the ~1,400 lines of tree code are env-gated observational probes
that never touch committed tokens — `submissions/fa2sw_treeverify_kenyan/sitecustomize.py:1590-1604`).
The end-to-end served identity measurement is the **follow-up** live-build's
deliverable, not this one's. This cert documents what is true of the *accept rule
and kernels* as built, and the inherited int4-spec token-identity gap, so an
organiser can adjudicate the latter post-hoc.

## 1. The claim: identity by construction (decode-step)

The deployed accept rule is `rejected = draft_token_id != target_argmax_id` — the
verifier's greedy argmax is authoritative and a draft token commits **only** if it
equals that argmax. The tree changes the accept rule in exactly one way: instead of
scanning a single linear chain, the descend walk scans **all children** of each
accepted node (rank-1 spine + rank-2/3 branches) for the verifier argmax and
descends the first match (`scripts/profiler/tree_spec.py::descend_accept`, GPU twin
`scripts/profiler/tree_accept_kernel.py::_tree_accept_kernel`).

Every emitted token is therefore still a verifier argmax. **The tree changes HOW
MANY tokens commit per step, never WHICH token the verifier authoritatively
chooses.** A wider candidate set cannot make the verifier accept a non-argmax token,
because the accept predicate is unchanged. This is the same greedy-identity argument
the deployed linear MTP K=7 stack already relies on — extended, not relaxed. This is
**not** the closed relaxed-accept lane (#66); the accept rule is byte-identical in
spirit.

## 2. Self-test evidence backing the claim (MEASURED, zero-quota, A10G)

All four foundation self-tests pass on the warm `/tmp/server-venv` (torch 2.11+cu130,
A10G), `CUDA_VISIBLE_DEVICES=0`:

| component | check | result |
|---|---|---|
| descend-walk accept (CPU/GPU twin) | reproduces deployed **linear** kernel on the degenerate chain (full-accept + mismatch-at-k) | exact prefix+bonus match |
| descend-walk accept | GPU kernel == CPU reference (emit + valid_count + commit_map), 1100+ trials lin8/M16/M32 | bit-for-bit |
| descend-walk accept | branch-hit (rank-2 first-divergence catch) | 0.4182 (M16) / 0.4154 (M32) ≈ ρ₂ 0.4165 — a correct walk, not the 3% broken-build signature |
| tree_spec E[T]-core | closed-form E[T] == Monte-Carlo descend-sim | M16 3.974 vs 3.975; M32 4.553 vs 4.554 |
| 3c fused KV-relocate | FUSED == reference == per-layer == stacked, M16/M32/base997 | bit-exact (rate 1.0) |
| 3c fused KV-relocate | pure bf16 permute/copy (no cast/arithmetic) | relocated K/V bit-identical to source |
| leg-1→leg-2 seam | descent `commit_map` → fused relocate, captured in ONE CUDA graph, replayed live-mutated | bit-exact, sync-free |

The relocate being a **pure bf16 copy** is the second half of the cert: compacting
the accepted scattered path into contiguous slots cannot alter any token value, so
continued-generation prefix KV stays bit-identical to the tokens the verifier chose.

Reproduce:
```
CUDA_VISIBLE_DEVICES=0 /tmp/server-venv/bin/python scripts/profiler/tree_spec.py
CUDA_VISIBLE_DEVICES=0 /tmp/server-venv/bin/python scripts/profiler/tree_accept_kernel.py
CUDA_VISIBLE_DEVICES=0 /tmp/server-venv/bin/python scripts/profiler/tree_kv_relocate.py
CUDA_VISIBLE_DEVICES=0 /tmp/server-venv/bin/python scripts/profiler/tree_seam_validate.py
```

## 3. Why byte-exact cross-run identity is NOT the instrument here

The deployed `fa2sw_precache_kenyan` (FA_SLIDING=1) is **run-to-run token-
nondeterministic** on this A10G — kernel FP-reduction noise on argmax near-ties.
Same-GPU **spec-OFF vs spec-OFF** control diverges 28/32 prompts; the plain int4
vLLM baseline (none of our kernels, no spec) diverges 29/32; `FA_SLIDING=0` restores
byte-identity (0/32) at an unmeasured TPS cost (BASELINE.md line 49;
`research/validity/served_gate_reconciliation.md`, kanna #38). A bit-exact cross-run
greedy check would therefore **false-fail** on the deployed path regardless of the
tree — so the contract-correct identity instrument is the **distributional signal
(PPL + self-consistency)**, per the PR instructions and lawine #56. The accept rule
is unchanged, so identity holds by construction; PPL is unmoved (local control PPL
2.2347 ≤ 2.42, `research/local_validation/fa2sw_kenyan_land_repro/`).

## 4. The inherited int4-spec token-identity gap (for organisers)

The **official** validity gate is **PPL + completion + modalities, NOT token-
identity** (kanna #38): the HF-Jobs harness never compares served tokens to a greedy
AR reference, which is why the entire ~420+ spec-decode frontier is leaderboard-
legal. Separately, kanna #114 / human #124 frame the deployed 481.53 int4-spec stack
as **~56% token-divergent from its own AR reference** — an int4 + speculative-decode
property of the **deployed** frontier, **inherited unchanged** by anything built on
it (the tree path adds no new divergence source: same int4 verify weights, same
accept predicate). This cert does **not** close that gap and does not claim to; it
documents it so organisers can adjudicate.

**#192 (greedy-decode-correctness "just checking") is retired as a launch *blocker*
per human #124** — decision: publish the ≥500 milestone first and let organisers
rule on the int4-spec token-identity question post-hoc (logged as accepted-risk, not
unresolved). This cert is the artifact that hands them the question with the evidence.

## 5. What this cert does and does not license

- **Does:** certify that the tree accept rule + KV relocate, as built and self-tested,
  preserve greedy identity at the decode step by construction (verifier argmax
  authoritative, unchanged predicate; relocate bit-exact).
- **Does NOT:** assert a measured end-to-end served-tree greedy-identity number — there
  is no live tree decode to measure. That measurement, on a captured valid-PPL
  representative served tree, is the **follow-up live-build's** gate, alongside the
  first **measured** `treeverify_served_gain_MEASURED_realized > 0`.
