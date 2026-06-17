<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# The operative-#319 identity contract (PR #588, wirbel)

**Status: analysis-only diagnostic card. `analysis_only=true`, `official_tps=0`. No HF
Job, no `train.py --launch`, no `/v1/jobs:run`, no submission, no served-file change.**

This card pins **one** canonical, measurable definition of the live #319 greedy-identity
contract, shows `base_fullhead` passes it with margin at the deployed served geometry, and
re-stamps every banked census verdict onto it. It does **not** reopen a speed lever — it
only makes the bar crisp and self-consistent.

**Headline (corrected by the full 128×512×3 measurement — the 3-prompt smoke was too small
to surface it):** at the **served M=1 geometry**, `base_fullhead` is **LITERALLY byte-identical
run-to-run at warm steady-state** — two independent warm decode passes are
`GREEDY_IDENTICAL` (128/128 prompts, 0/65536 divergent tokens). The *only* run-to-run
nondeterminism is a **one-time first-pass cold-start transient** (lazy kernel-JIT /
prefix-cache settling), uniformly bounded at **≤ 4 bf16 logit ULPs (0.25 nat)** — PPL-neutral
near-ties on a clean {0, 2, 4}-ULP gap ladder. So the canonical bar is the **literal
leaderboard verifier bar against the int4 self-reference**, which `base_fullhead` passes;
the near-tie envelope is a measured cold-start *robustness margin*, not a steady-state
necessity.

This **corrects an earlier draft of this card**, which (extrapolating from a 3-prompt smoke)
claimed literal byte-identity was *unsatisfiable* for int4 even at M=1 and that the bar had
to be near-tie-tolerant at `eps_star=0.125` (2 ULP). The full census refutes both points:
warm M=1 self-determinism is byte-perfect, and the cold-start near-tie envelope is **4 ULP,
measured**, not 2 ULP assumed.

---

## 0. The two — and only two — relaxations the int4 served stack forces

The challenge's hard validity rule (program.md:27-28; official verifier README, quoted
verbatim):

> the served endpoint's greedy decode must be token-identical to **plain greedy
> autoregressive decode of the same submitted checkpoint** … Any optimization that
> changes the generated token IDs, even if TPS improves or PPL remains similar, is not
> valid for leaderboard scoring.

The reference is **"the same submitted checkpoint"** — *not* a full-precision bf16 model.
Exactly **one** relaxation is forced for the served contract, plus **one** characterized
transient that is *not* part of the steady-state contract:

| | what it is | what forces / bounds it | evidence |
| -- | ---------- | ----------------------- | -------- |
| **R1 — int4 self-reference** (not bf16) | the reference is the submitted int4 checkpoint's own plain greedy AR decode | the served int4 stack is not bf16-byte-exact, so a bf16 reference would fail the shipping base | wirbel **#585** (`2u44yaa1`): served int4 `base_fullhead` flips **6.76%** of teacher-forced argmax (4428/65536) and **0-of-128** free-running sequences vs bf16 |
| **C1 — first-pass cold-start transient** (≤ 4 ULP, *not* a steady-state relaxation) | the **first** decode pass after server load can resolve ≤4-ULP near-ties differently from the warm steady state; once warmed it is byte-deterministic forever | lazy Triton-JIT / FlashInfer-autotune kernel-config settling on first inference **and** prefix-cache cold-vs-warm chunked-prefill numerics; both confined to pass 1 | this card §3: warm/warm `b_vs_c` = `GREEDY_IDENTICAL` 128/128; cold pass `a` differs from **both** warm passes at the *identical* 67 prompts, all first-div gaps ≤ 4 ULP; server log `jit_monitor` warning during pass `a` |

**The key correction.** R2 of the earlier draft ("literal byte-identity is unsatisfiable for
int4 even at M=1, so the bar must be near-tie-tolerant") is **false at the served M=1
steady-state**. Warm `base_fullhead` is *literally* byte-identical to its own int4 AR
reference. The near-tie tolerance is needed only to absorb the **cold-start transient C1** (a
robustness margin), and — separately — the **batched quality-eval geometry** (M=16,
`--max-connections 32`), where dynamic batch composition adds a *persistent* run-to-run
nondeterminism. Neither is the served M=1 contract. So the canonical bar is **literal**, with
R1 as its only structural relaxation.

---

## 1. Enumeration of candidate operative-#319 predicates used across banked cards

Each entry states the exact measurable predicate, its pass threshold, and where it is
invoked.

### (a) Per-token argmax-class match (teacher-forced)
- **Predicate:** feed a fixed token sequence; at every position *t*,
  `argmax(candidate_logits_t) == argmax(reference_logits_t)`.
- **Statistic / threshold:** `argmax_flip_rate == 0` over the corpus.
- **Used in:** #556 head (`int4_head_argmax_flip_rate_heldout=2.86e-4`), #571 body
  (`body_argmax_flip_rate_*`), #585 (6.76% vs bf16).
- **Why it is NOT the contract:** it is *teacher-forced*. The official README is explicit
  that teacher-forced scoring (PPL-style) is **immune to which tokens the model actually
  emits** — a model can drift token IDs while every teacher-forced argmax still matches.
  (a) is the right lens for *locating* a lever's risk, not the contract itself.

### (b) Near-tie / Δlogit envelope
- **Predicate:** a first-divergence flip is *operatively benign* iff the reference's
  runner-up logit gap at that position, `m1_self_gap`, is `<= eps_star`. A flip with gap
  `> eps_star` is **SEMANTIC**.
- **Statistic / threshold (corrected, measured):** `eps_star = 0.25` nat = **4 bf16 logit
  ULPs** (`ulp_nat = 0.0625`). This is the **measured** int4-M=1 cold-start near-tie envelope
  (§3: every run-to-run first-divergence, cold or warm, has gap ≤ 4 ULP; hard ceiling, 0
  exceptions over 65536 tokens). The earlier draft's `eps_star=0.125` (2 ULP) was
  under-measured by exactly one ULP-pair.
  log-softmax is gap-preserving, so the gap is readable directly from API top-K logprobs.
- **Used in:** fullserve census (`m1_self_gap`, semantic-vs-tie split), #429 blanket-strict,
  #556 near-tie concentration.
- **Role:** the **cold-start / batched robustness operator**. It is not the steady-state
  bar; it is the classifier that certifies the cold-start transient (C1) is PPL-neutral.

### (c) Free-running greedy SEQUENCE match
- **Predicate:** the candidate's free-running emitted `completion_token_ids` are compared,
  prompt-by-prompt, to the reference's, over the official 128 prompts × 512 tokens. The
  **first** divergence index per prompt is the cascade origin (everything downstream is a
  context-change artifact — the census `det_details` methodology).
  - **(c-literal):** pass iff `total_divergent_tokens == 0`. This is exactly what the
    official `check_greedy_identity.py` emits as `GREEDY_IDENTICAL`.
  - **(c-operative):** pass iff every per-prompt first-divergence is a near-tie (b), i.e.
    `n_semantic_first_divergences == 0`.
- **Used in:** the official leaderboard verifier
  (`gemma_greedy_identity_verifier_flowian-powers`) emits the literal verdict; this card
  scores both.
- **Key finding (corrected):** **(c-literal) IS satisfiable for int4 at warm steady-state** —
  `base_fullhead`'s two warm passes are `GREEDY_IDENTICAL` (§3). (c-operative) at the measured
  4-ULP envelope additionally absorbs the cold-start pass.

### (d) Self-determinism (R/R repeat)
- **Predicate:** the same prompt produces identical output across *R* independent runs at a
  fixed serving geometry.
- **Statistic / threshold:** `R/R` sequences identical — literal or operative.
- **Used in:** #429, cilb #564 selfdet, fullserve census `determinism_M{1,8}`.
- **Role:** because `base_fullhead` is spec-OFF at M=1, its served decode **is** plain greedy
  AR decode, so (c) "served-vs-AR" collapses onto (d) "two AR runs agree." This card measures
  (d) directly: **warm/warm (d) is literal-exact; cold/warm (d) is operative-exact at 4 ULP.**

---

## 2. The canonical definition (pick + justification)

> **Operative-#319 (canonical).** A served configuration **passes operative-#319** iff its
> **warm steady-state** free-running greedy decode — `completion_token_ids` captured at the
> **deployed serving geometry MAX_NUM_SEQS=1, spec-OFF, temp=0** via `/v1/completions` with
> integer-token prompts — is **byte-identical** (official `check_greedy_identity.py` verdict
> `GREEDY_IDENTICAL`, zero tolerance) to the **same submitted int4 checkpoint's** plain
> greedy autoregressive decode, over the official public suite (128 sharegpt prompts,
> `seed=1`, `output_len=512`, `ignore_eos=True`).
>
> The only structural relaxation vs naïve literal-#319 is **R1** (the reference is the
> submitted *int4* checkpoint, not bf16). **No near-tie tolerance is required at warm steady
> state.** The first-pass cold-start transient (C1) is certified separately as a PPL-neutral
> near-tie envelope of **≤ 4 ULP (0.25 nat)** — a robustness margin, *not* part of the
> steady-state contract.

This is predicate **(c-literal)** against the **(R1)** int4 self-reference at warm steady
state, with **(b)** at the measured `eps_star = 0.25` nat used only to certify **(C1)**.

**Justification.**
1. **It is exactly the leaderboard bar.** `check_greedy_identity.py` emits `GREEDY_IDENTICAL`
   iff the two token streams are byte-for-byte equal. That is the contract a real submission
   is scored on. `base_fullhead` passes it at warm steady state (§3) — so the bar is both the
   real one *and* satisfied, with no invented tolerance.
2. **R1 is forced, not chosen.** A bf16 reference is refuted *by the shipping base itself*:
   `base_fullhead` flips 6.76% teacher-forced vs bf16 (#585). The reference must be the
   submitted int4 checkpoint. This is the README's own wording ("the same submitted
   checkpoint").
3. **No near-tie tolerance is needed for the steady-state contract.** The earlier draft's
   premise — that int4 GEMV nondeterminism makes literal identity unsatisfiable at M=1 — is
   **measurably false**: warm/warm `b_vs_c` is `GREEDY_IDENTICAL`. Tolerance is demoted to
   what it actually covers: the cold-start transient (C1) and the batched eval geometry.
4. **The cold-start envelope is PPL-neutral and bounded.** Every cold-vs-warm first-divergence
   has gap ≤ 4 ULP = 0.25 nat (§3). A ≤4-ULP swap is between two tokens the model ranks within
   a logit hair; PPL is a property of the *distribution*, not the argmax realization, so it is
   unmoved. The quality gates (MMLU-Pro ≥0.605, GPQA-D ≥0.471, GSM8K ≥0.807, AIME ≥0.090) were
   measured **on this exact stack with C1 present** and all pass.
5. **(a) is provably the wrong bar.** Teacher-forced argmax match is PPL-like and blind to
   emitted-token drift — the very thing #319 forbids. Kept only as a per-lane risk locator.

**Relation to the official literal verifier.** The canonical bar **is** the literal verifier,
restricted to warm steady state and referenced to the int4 self-decode (R1). The only place
the near-tie classifier enters is to *certify* that the cold-start pass C1 is benign — it does
not loosen the steady-state verdict.

---

## 3. `base_fullhead` under the canonical predicate (measured)

**Harness (faithful to the contract).** Serve `base_fullhead` (stock int4 native-262k head +
FA_SLIDING + SURGICAL_ATTN_USE_3D_OFF 2D order-preserving attention + PLE embed-scale fold —
the cilb #564 `base_fullhead` arm) at the **deployed served geometry MAX_NUM_SEQS=1, spec-OFF,
temp=0**, then run the **official** `decode_outputs.py` (serial, 1 request/prompt) **R=3
independent free-running passes** (a, b, c) over the official 128-prompt × 512-token suite
(`seed=1`). Score every pass-pair with the **official** `check_greedy_identity.py` (literal),
and classify each first-divergence with the top-K logprob near-tie probe (operative). Because
`base_fullhead` is spec-OFF at M=1, each pass *is* an independent plain greedy AR decode, so
pass-pairwise scoring certifies the served stack against its own int4 AR reference. Artifacts:
`operative_319_remeasure.json` (driver) + `operative_319_canonical.json` (finalizer);
reproduce with `measure_operative_319.py` then `finalize_canonical.py`. W&B run
**`n32yblfs`** (group `operative-identity-formalize`; also in `wandb_run_id.txt`).

### 3.1 Verdict

| field | value |
| ----- | ----- |
| **`base_fullhead_passes_operative_319`** | **True** (literal, at warm steady state) |
| warm/warm self-determinism (`b_vs_c`, official verifier) | **`GREEDY_IDENTICAL`** — 128/128 prompts, **0 / 65536** divergent tokens |
| passes literal warm steady-state bar | **True** |
| passes operative bar including cold-start (eps_star = 4 ULP) | **True** (max first-div gap = 4 ULP) |
| `census_stable_under_canonical_operative` | **True** |
| `eps_star` (measured cold-start envelope) | **0.25 nat = 4.0 ULP** (`ulp_nat = 0.0625`) |
| peak GPU | 19.13 GB (1× A10G) |

### 3.2 The cold-start transient (C1), characterized

The driver compared pass `a` against `b` and `c` (a = reference), which made the *raw* driver
verdict read FAIL. The missing comparison `b_vs_c` is decisive:

| pair | official verdict | identical prompts | divergent tokens |
| ---- | ---------------- | ----------------- | ---------------- |
| `a_vs_b` | DIVERGENT | 61/128 | 21815 / 65536 |
| `a_vs_c` | DIVERGENT | 61/128 | 21815 / 65536 |
| **`b_vs_c`** | **GREEDY_IDENTICAL** | **128/128** | **0 / 65536** |

- **Pass `a` is a uniform cold-start outlier.** It differs from **both** warm passes at the
  *identical* 67 prompts; at every one of the 5 first-divergences classified "semantic" at the
  old 2-ULP eps, `b == c` (the two warm passes agree; `a` is the lone dissenter). The two warm
  passes are byte-perfect with each other.
- **Bounded near-tie ladder.** Cold-vs-warm first-divergence gaps form a clean quantized
  ladder, **per pair: {0 ULP: 24, 2 ULP: 38, 4 ULP: 5}**, hard ceiling **4 ULP** — **0** of
  65536 tokens exceed it. The 5 first-divs at 4 ULP are exactly the ones a 2-ULP eps
  mislabels "semantic"; they are one more quantum of the same reduction-order noise.
- **Mechanism.** Confined to the first pass: lazy Triton-JIT / FlashInfer-autotune kernel
  settling on first inference (server log: `WARNING … jit_monitor … Triton kernel JIT
  compilation during inference: _compute_slot_mapping_kernel` at 13:54:36, during pass `a`),
  and prefix-cache cold-vs-warm chunked-prefill numerics (`enable_prefix_caching=True`,
  `enable_chunked_prefill=True`): pass `a` prefills every prompt cache-miss, passes `b`/`c`
  reuse the cached prefix — which explains the *uniform* spread across all 67 prompts and the
  `a ≠ {b,c}`, `b == c` structure. Both are warmup effects, not steady-state numerics.

### 3.3 Geometry note (load-bearing)

The served contract binds **M=1**, where warm self-determinism is **literal-exact**. The cilb
#564 selfdet probe (3/24 identical) and the fullserve census `determinism_served=0.1875` were
at the **batched quality-eval geometry** (`--max-connections 32`, `max-num-seqs 16`), where
*dynamic batch composition* adds a **persistent** run-to-run nondeterminism that does *not*
go away with warmup — that is the regime that genuinely needs near-tie tolerance, and it is
the **eval harness**, not the served M=1 contract. This card isolates the served M=1 geometry
and shows it is byte-deterministic once warm. This also refines #429's "literal 0.9989": at
warm M=1 the literal rate is **1.0**; the < 1 figures reflect cold-start / batched residual.

---

## 4. Census re-stamp onto the canonical predicate

The question for each card was *"is there a config BOTH faster than `base_fullhead` AND
operative-#319 identical to it?"* The canonical bar is **literal byte-identity at warm steady
state (R1)**, with a **4-ULP** cold-start envelope. A faster config is fire-eligible only if
its divergences from `base_fullhead` are byte-zero at warm steady state (or, at minimum, never
exceed the 4-ULP cold-start floor). The check below confirms each lane's NO-FIRE survives,
and flags the one lane where the 4-ULP envelope is genuinely more permissive.

| PR | lane | why the faster candidate fails the canonical bar (or: no candidate exists) | stable? |
| -- | ---- | -------------------------------------------------------------------------- | ------- |
| **#556** | head | Faster head = int4 (290.63 TPS vs base's bf16 262k head 252.31). Its flips vs the bf16 head are near-tie-*concentrated* (median ≈ 2.5 ULP) **but have a semantic tail: p90 = 5.85, p99 = 7.49, p100 = 7.78 ULP** — all **> the 4-ULP floor** ⇒ SEMANTIC flips exist ⇒ fails (zero-semantic). Source: `int4_head_strict_identity_results.json`. NO faster head is byte-safe. | **NO-FIRE stable** |
| **#571** | body | Faster body = int4_g128 (259.07 TPS vs g32 252.69). `body_flip_is_near_tie_concentrated=False`; int4_g32 flip-margin **median 0.7558 nat = 12.1 ULP** (ood/official median 1.482 nat = 23.7 ULP, ~11.9× flip/nonflip separation) ≫ 4 ULP, and g128 flips ≥ g32. bf16 body is exact but slower (143.99 TPS). Source: `body_strict_identity_results.json`. NO faster body is byte-safe. | **NO-FIRE stable** |
| **#562** | attention | Faster reordered kernels (seg32, tile_alt) have `bitwise_rate=0` but `argmax_rate=1.0` on a **24-draw op-probe** (`max_abs_diff ≤ 2 ULP`). Their perturbation is **within** the 4-ULP cold-start floor — so the looser envelope is genuinely more permissive here (residual, below). No FIRE is opened; the deployed order-preserving attention is byte-exact. | **NO-FIRE stable** (residual) |
| **#583** | spec-dec (fern) | `specdec_two_gate_closed=True`; binding failure is **speed** (best speedup 1.09 ngram / 1.005 mtp ≪ 1.437 needed), not identity. Spec-dec verify is identity-preserving by construction; an identity-tolerance change is **inert on a speed verdict**. | **NO-FIRE stable** |
| **#584** | spec-dec (lawine) | `any_measured_drafter_clears_ship=False`; best ngram proj 109.8 TPS, mtp realized 216/249 TPS — all speed-bound. Same: identity-preserving verify; the bar is inert on a speed NO-FIRE. | **NO-FIRE stable** |

**`census_stable_under_canonical_operative` = true** — no banked NO-FIRE verdict flips to FIRE
under the canonical predicate, at the stricter literal/warm bar *and* at the 4-ULP envelope.

**The argument is lane-specific:**
- **Precision-swap lanes (head, body):** the faster realization has flips vs `base_fullhead`
  that **exceed the 4-ULP cold-start floor** (head semantic tail p90 = 5.85 ULP; body median
  12.1 ULP) — divergences too large to be reduction-order noise, i.e. genuinely SEMANTIC ⇒
  fail. This is what keeps the lever closed.
- **Spec-dec lanes (#583/#584):** NO-FIRE is **speed**-bound; spec-dec output is
  identity-preserving by exact verify ⇒ the identity tolerance is irrelevant to the verdict.

**Operative-permissive residual — attention (#562), the one honest caveat.** The faster
reordered attention kernels' ≤2-ULP reduction-order perturbation sits **inside** the measured
4-ULP cold-start floor `base_fullhead` already tolerates run-to-run, and the 24-draw op-probe
shows `argmax_rate=1.0`. That is **suggestive but not a certification**: the canonical bar is
byte-identity at warm steady state (or, in the envelope reading, zero first-divergences over
the full 128×512 census above 4 ULP), which a 24-draw single-position probe cannot establish.
Certifying (or refuting) a faster attention kernel would require a **free-running operative
census of that kernel** — which PR #588 deliberately does **not** run, because that is a
speed-lever experiment and this card is analysis-only NO-FIRE. So: no FIRE is opened, the
deployed attention is byte-exact, and the banked NO-FIRE stands as *"no fire-eligible faster
attention realization has been certified."* Logged as a **suggested follow-up**, not a
reopened lever.

---

## 5. Scope guard — this does not reopen a speed lever

This card pins a predicate and re-stamps verdicts; it changes **no** served file, launches
**no** job, and submits **nothing** (`official_tps=0`). Every faster candidate in head / body
/ spec-dec either diverges from `base_fullhead` by **more than the 4-ULP reduction-order
floor** (head, body — genuinely semantic) or fails the **speed** bar with identity-preserving
verify (spec-dec). The single lane where the cold-start envelope is more permissive than the
strict bar that grounded the banked verdict — **attention #562** — is explicitly left as an
*uncertified residual / follow-up*, not a fire: closing it would require a free-running
operative census of a faster attention kernel, a speed-lever experiment outside this card's
analysis-only mandate. The one new artifact is a **crisp, satisfiable, leaderboard-aligned
contract** (literal byte-identity at warm steady state, int4-self-referenced, with a measured
≤4-ULP cold-start robustness envelope) so each NO-FIRE is defensible in the exact units a real
int4 submission is scored on.

**Suggested follow-ups (not implemented here):**
1. **Disambiguate / eliminate C1.** Re-run R=3 with `enable_prefix_caching=False` (and/or an
   extended autotune/CUDA-graph warmup pass before scoring). If `a == b == c` byte-identical,
   that localizes the cold-start to prefix-cache/JIT settling and shows a warmup pass makes the
   served stack byte-deterministic from pass 1.
2. **Confirm leaderboard warmup.** Check whether the official benchmark runner warms the
   endpoint before the scored greedy-identity pass; if so, C1 is excluded from scoring and the
   literal-warm bar is the operative contract with zero margin consumed.
3. **Attention #562 certification.** A free-running operative census of a faster reordered
   attention kernel (seg32 / tile_alt) over the full 128×512 suite would certify or refute the
   one operative-permissive lane — but only as a *speed-lever* experiment, out of scope here.
