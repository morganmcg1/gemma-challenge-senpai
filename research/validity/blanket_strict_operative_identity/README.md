<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# PR #429 (stark) — Is blanket-strict's 0.9989 operatively 1.0 under the verify-arbiter gate?

**Local pod A10G only (sm_86). Research wrapper that READS the existing
`STRICT_VERIFY_REDUCTION=1` flag — NO served-file change, NO HF job, NO
submission. Local inference profiling within the standing GPU grant.**

## Hypothesis

Does blanket-strict's `0.9989` literal identity already satisfy the #407
equivalence contract *operatively* — making a value-level "fix" unnecessary?

The shippable strictly-equivalent base is **blanket-strict verify**
(batch-invariant attention on every step): **467.14 TPS, identity 0.9989**
(lawine #425, imported from stark #412). The 0.11% gap is **one residual flip @
prompt 90**: under the M=8 batched verify the emitted token disagrees with the
M=1 serial-AR reference — but it is a **bitwise tie** (`m1_self_gap=0.0`).

stark #421 (`wvy2k7w7`, merged RED) proved a **logit-layer canonical tie-break
CANNOT close it** — applied consistently to M=1 ref and M=8 verify it makes
identity *worse* (0.9966→0.9909, 8 new flips). Banked there: identity-1.0 levers
must act on the attention-reduction VALUE, not logit post-processing.

The prior question #421 did not answer: land #414/#420 (`qe4qagc1`) established
the deployed `serve.py` truncated-head verify is the **sole arbiter of emitted
tokens**. If the verify path is its own reference (self-referential gate), the
prompt-90 token the M=8 verify emits IS the operative truth, and the
"divergence" is only against an M=1 serial reference **the deployed path never
executes**. Under that reading blanket-strict is **operatively identity-1.0**.

## What this card measures (never measured on the deployed code path before)

1. **Literal bar** — emitted tokens vs the pure-AR-greedy M=1 reference
   (expect ~0.9989, reproduce #425/#412).
2. **Operative bar** — for the prompt-90 emitted token, is it self-consistent
   under the verify-is-arbiter gate? Re-run the truncated-head verify on the
   model's own emitted prefix and check it reproduces the same token (fixed
   point of the verify it was emitted by).
3. **Classify the prompt-90 flip** — confident-argmax change (PPL-affecting,
   FORBIDDEN) vs pure bitwise tie (PPL-neutral). Confirm PPL = 2.3772.
4. **Resolve lawine #425's GO-conjunct (ii)**: `green` / `red` /
   `human_contract_decision`.

Wrapper + results land in this directory. Self-test asserts census size, that
the flag is READ (not the kernel patched), and that no served file changed.
