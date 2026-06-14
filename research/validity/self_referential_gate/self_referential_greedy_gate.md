# PR #114 — Is the official greedy-decode gate self-referential?

**Question.** program.md (lines 27–28): *"Greedy decode must remain token-identical
to plain greedy autoregressive decode for the submitted checkpoint."* Does the
official gate compare a submission's speculative output to **its own** plain-AR
decode (self-referential, per submission), or to a **canonical** fp32/bf16
reference? If self-referential, then SplitK / tree-verify frontiers are
greedy-safe *by construction* (they only have to reproduce their own AR), and the
PR #96 interlock (composed-vs-deployed-baseline byte-identity) was over-strict.

**Verdicts.** 🟢 GREEN = gate confirmed self-referential + corrected interlock
0-divergent + decode pins named + PPL holds. 🟡 AMBER = confirmed but decode
pinning fragile. 🔴 RED = gate genuinely NOT self-referential → escalate.

---

## Step 1a — Acceptance-rule proof (the gate is self-referential by mechanism)

The submitted stack emits the **target model's argmax at every position**; the
drafter only changes *how many* positions are verified per batched forward, never
*which* token is emitted. Grounded in the served code:

**Accept kernel** — `submissions/fa2sw_precache_kenyan/sitecustomize.py:921-959`
(`_dixie_fused_accept_prep_kernel`):
- `:945` `target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos)` — the
  target's argmax at verify position `pos`.
- `:948` `rejected = draft_token_id != target_argmax_id` — reject on the first
  draft≠target mismatch.
- `:950-951` `next_token_id = target_argmax_id; tl.store(output..., target_argmax_id)`
  — **always emit the target argmax**, regardless of the draft token.
- `:953-959` if no rejection, append the bonus token (target argmax at K+1).

So the emitted stream is exactly `argmax(target_verify_logits)` at positions
`0 .. first_reject`. The draft proposals are a *scheduling* device (acceptance
length), invisible to the token identity.

**Reference mode** — `submissions/fa2sw_precache_kenyan/serve.py:977-1009`:
`SENPAI_REFERENCE_MODE=1` → `disable_speculation_for_reference_mode()` clears
`SPECULATIVE_CONFIG` → vLLM starts `speculative_config=None` → plain **M=1
autoregressive** decode on the **submission's OWN engine / kernels / quant**
(docstring `:983`: "the only removed variable is speculation"). Confirmed live in
this run's server log: `[serve] SENPAI_REFERENCE_MODE active: clearing
SPECULATIVE_CONFIG (M=1 AR greedy reference, drafter OFF)` then
`speculative_config=None`.

**The reduction.** spec-OFF emits `argmax(target_logits | M=1 sequential decode)`.
spec-ON emits `argmax(target_verify_logits | M=K+1 batched verify)`. These are
**token-identical iff the verify GEMM is batch-invariant** — i.e. the argmax is
unchanged between the M=1 decode geometry and the M=K+1 verify geometry. The only
way they differ is a near-tie position where the M=K+1 reduction order flips the
argmax vs M=1 (the batch-non-invariance mechanism characterized in PR #5 / #73;
literature: vLLM issue #41758, Thinking-Machines batch-invariant kernels). They do
**not** differ because of the drafter — Leviathan 2211.17192 / Chen 2302.01318
guarantee speculative decoding is distribution-lossless, and at temperature 0 the
accept rule above makes it argmax-lossless, *under the exact-logit assumption*.

**Therefore the gate is self-referential by construction**: it tests "does the
verify GEMM reproduce the M=1 argmax trajectory on this submission's own
weights?", never "does this submission match a canonical fp32 model?". Whether the
exact-logit assumption holds numerically on *this* stack (SPLITKV_VERIFY
engineered) is the empirical question Step 1b settles.

`self_referential_gate_confirmed` (mechanism): **yes**.

---

## Step 1c — The PR #52 anchor, CORRECTED: "128/128" is a completion count, not a greedy-identity pass

The PR framed #52 (int4-Marlin) as having "passed the official **128/128 greedy
gate**", which would be impossible against a canonical fp32 reference (int4
quant-noise ≫ near-tie ULPs) and therefore proves a self-referential reference.
**That framing rests on a misreading of what "128/128" means**, which the harness
source refutes:

- The official scored path is `speed_benchmark` (`hf_bucket_single_job.py`,
  `run_hf_bucket_benchmark.py`). It reports `completed=128` — the number of
  prompts that **returned a completion** (`result["completed"]`,
  `hf_bucket_single_job.py:347,502`). "128/128" = *128 prompts completed*, NOT 128
  prompts byte-matching a greedy reference.
- **The official harness never invokes the greedy-identity verifier.**
  `grep -rn greedy_identity official/.../speed_benchmark/` is empty: the speed run
  scores TPS + completion count; PPL is a separate `same_path_ppl` pass; modalities
  are checked separately. **No token-identity comparison runs officially.**
  (Reconfirms PRs #38 / #66: the served gate has no token-identity check.)

So #52 "passing 128/128" says nothing about the reference being self-referential —
it only means #52 completed 128 prompts. The anchor does **not** establish gate
self-referentiality; it establishes that *no official greedy-identity gate exists*.

**What IS self-referential** is the repo's own LOCAL enforcement tool, by
construction: `scripts/local_validation/gen_greedy_reference.py --mode served`
generates the reference **through the same submission's api_server with
`SENPAI_REFERENCE_MODE=1`** (drafter off → M=1 AR on its own quant/kernels), and
`greedy_gate.reference_for()` resolves that served spec-off capture as *the*
reference (`reference_kind = served_spec_off`). That tool — the one Step 2 rebuilds
— is self-referential. The OFFICIAL gate simply does not run it.

---

## Step 2 — Corrected interlock: `greedy_identity_interlock.py --self-referential` (DONE)

Replaced PR #96's over-strict composed-vs-deployed-baseline byte-identity check
with the self-referential gate:

1. **spec-ON self-determinism** (run-to-run byte-identical) — precondition (PR #38
   served wobble would make the verdict unstable; pin a `--config`).
2. **spec-OFF self-determinism** (the own-AR reference is itself stable) —
   precondition.
3. **self-consistency (THE GATE)**: every spec-ON reload is `GREEDY_IDENTICAL` to
   the spec-OFF own-AR reference, judged by the **official verifier**
   (`greedy_gate.compare`).

GREEN iff (3) identical + (1)/(2) deterministic. RED if (3) DIVERGENT (with onset
diagnostics). INCONCLUSIVE if captures missing / a precondition fails / verifier
INCOMPARABLE. `primary_metric = self_referential_divergent_runs` (0 ⇒ GREEN).

Offline validation: `scripts/validity/selftest_interlock_self_referential.py`
builds synthetic GREEN / RED / wobble / missing-reference capture trees and asserts
each verdict, plus a `--skip-capture` CLI round-trip. **All pass.**

```
[GREEN]        verdict=GREEN        confirmed=yes div_runs=0  -> OK
[RED]          verdict=RED          confirmed=no  div_runs=2 onset_min=7  -> OK
[WOBBLE]       verdict=INCONCLUSIVE spec_on_det=False  -> OK
[MISSING-AR]   verdict=INCONCLUSIVE  -> OK
[CLI skip-cap] rc=0 verdict=GREEN  -> OK
```

**Canonical run on the REAL A/B captures** (`--skip-capture` over the Step-1b tree,
`interlock_report.json`, W&B run `9q5yy9l1`) — the verdict is RED with BOTH
preconditions GREEN, so it is an unambiguous gate failure, not a precondition
artifact:

```
spec-ON  self-det : runs=2 min_byte_identical=1.0 det=True     (precondition OK)
spec-OFF self-det : runs=2 min_byte_identical=1.0 det=True     (precondition OK)
GATE              : all_greedy_identical=False div_runs=2/2 incomparable=0
  run 0: DIVERGENT tok_div_frac=0.5608 (36751/65536) onset_min=0
  run 1: DIVERGENT tok_div_frac=0.5608 (36751/65536) onset_min=0   (byte-identical to run 0)
onset             : min=0 median=120 max=496 [late/stochastic FP near-tie flips]
VERDICT: RED  (self_referential_gate_confirmed=no)
```

Both spec-ON reloads produce the *same* 36751 divergent tokens vs the spec-OFF
reference ⇒ spec-ON is self-deterministic AND the divergence is perfectly
reproducible. The interlock now correctly returns RED on the deployed stack — it
will NOT green-light a #71×#84 frontier as "greedy-safe by construction".

---

## Step 1b — Clean pinned A/B (spec-ON vs own spec-OFF M=1 AR) — **the deployed stack FAILS its own self-referential gate**

`research/validity/self_referential_gate/ab-20260614T075459Z/` — fresh spec-OFF +
spec-ON, 128×512, **identical BASE_ENV, only `SENPAI_REFERENCE_MODE` (drafter
on/off) differs.** spec-ON E_accept=3.85 (drafter genuinely accepting); spec-OFF
server log confirms `clearing SPECULATIVE_CONFIG` → `speculative_config=None` (true
M=1 AR). This removes the earlier 59.5% confound (which compared a PRECACHE_BENCH=1
+ Hub-tokenizer reference to PRECACHE_BENCH=0 + baked-tokenizer captures).

**Official verifier (`greedy_gate.compare`), spec-ON candidate vs spec-OFF own-AR reference:**

```
VERDICT                 : DIVERGENT
prompts compared        : 128
identical               : 16
divergent               : 112        (87.5% of prompts)
total tokens compared   : 65536
total divergent tokens  : 36751      (56.1% of tokens)
divergence onset (tok)  : min=0 median=121 max=496   (quartiles 0/78/121/233)
```

The divergence is **real, deterministic, and reproducible — not the env confound and not FP noise:**

- **Cross-validated** against 3 independent spec-ON captures (2 from the prior #73
  session + the fresh A/B run): all give **byte-identical** 112/128 prompts /
  36751 tokens divergent against the single fresh spec-OFF reference.
- **spec-ON is perfectly self-deterministic**: #73 spec-ON run_00 vs run_05
  (different reloads) = `GREEDY_IDENTICAL`, 0 divergent. So the 56.1% is *spec-ON
  vs spec-OFF*, never spec-ON wobble.
- **Onset signature is late/stochastic** (median 121/512; only 2/112 prompts
  diverge within the first 5 tokens) ⇒ near-tie argmax flips that **cascade**, not
  an early systematic decode-path bug. Intrinsic flip hazard ≈ 1/121 ≈ 0.8%/tok,
  consistent with PR #5's 0.33–0.72%/tok precision-independent flip rate.

**Mechanism (confirms Step 1a's "iff batch-invariant" reduction):** the M=K+1
batched-verify GEMM reduces in a different float order than the M=1 sequential
decode GEMM, so at near-tie positions `argmax(verify_logits) ≠ argmax(decode_logits)`.
The accept kernel faithfully emits `target_argmax_id` — but the *verify* argmax it
emits is computed in the wrong (batched) geometry. SPLITKV_VERIFY does **not** make
the verify batch-invariant on this A10G stack.

**spec-OFF self-determinism (CONFIRMED):** the M=1 AR reference is itself
run-to-run stable — `ab-20260614T075459Z/default__specoff` run_00 vs run_01 =
`GREEDY_IDENTICAL`, 128/128 prompts, 0/65536 divergent (official verifier). So
**both arms are individually deterministic**; the 56.1% is purely the spec-ON↔spec-OFF
structural delta, never run-to-run wobble (rules out the last #38 confound).
FA_SLIDING (the only known #38 nondeterminism source) is identical across both arms.

`self_referential_gate_confirmed`: **the reference IS self-referential (own M=1 AR,
Step 1a) — but spec-decode ≠ own-AR (56.1% divergent), so the deployed stack FAILS
the self-referential gate.**

---

## VERDICT — 🔴 RED (with a corrected escalation message)

The PR's anticipated RED was "the gate is NOT self-referential (it's vs canonical
fp32) → escalate." **The truth is more consequential:**

1. **The greedy-identity reference IS self-referential** (own M=1 AR on its own
   quant/kernels — Step 1a mechanism, airtight; the repo's `gen_greedy_reference` /
   `greedy_gate` build exactly this).
2. **The deployed speculative stack DIVERGES 56.1% of tokens (112/128 prompts)
   from its own M=1 AR** — it *fails* the self-referential gate, deterministically.
3. **Therefore "SplitK/tree are greedy-safe by construction because the gate is
   self-referential" is FALSE.** Self-referentiality of the *reference* does not
   make spec-on == own-AR; the M=K+1 verify is not batch-invariant with M=1 decode,
   and the near-tie flips cascade. A #71×#84 frontier **inherits** this — it is no
   safer "by construction" than the stack already deployed.
4. **The only reason a 56%-divergent stack is leaderboard-legal is that the
   official harness enforces NO token-identity check** (Step 1c: `speed_benchmark`
   never invokes the greedy-identity verifier; "128/128" = completion count). The
   written contract (program.md 27–28) is real but **unenforced** by the automated
   gate — reconfirming and *extending* PRs #38/#66 from "relaxed-accept divergence
   is invisible" to "even the standard greedy accept rule already diverges 56%, and
   it's invisible."

**Corrected interlock impact:** running Step 2's `--self-referential` interlock on
the deployed stack returns **RED**, not GREEN. So it cannot be used as a
"greedy-safe by construction" green-light for landing #71×#84 — it correctly flags
that the speculative frontier breaks greedy-identity vs its own AR. The honest
green-light for a frontier requires either (a) a batch-invariant verify kernel
(VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT class; Thinking-Machines) so spec-on == own
AR, or (b) an explicit human contract exception accepting that served greedy ≠ AR
(the same DECISION-2-class question PR #66 escalated).

## Step 3 — decode_path_pin_invariants

The self-referential gate is only *meaningful* when the candidate and its own-AR
reference share every decode-path knob except speculation; otherwise an env delta
(not speculation) drives the divergence. The pins, from the env that produced the
clean A/B (and the earlier confound it fixed):

| pin | why it must match candidate↔reference | evidence |
|---|---|---|
| **speculation on/off** | the variable under test (M=K+1 verify vs M=1 decode) | the only delta in the clean A/B |
| `PRECACHE_BENCH` | precache changes the served prefill path → token deltas | the 59.5%→clean-A/B confound was PRECACHE_BENCH 1 vs 0 |
| tokenizer (`/tmp/osoi5-v0-baked`) | Hub vs baked tokenizer ⇒ different token ids | same confound |
| `MAX_NUM_SEQS` / `MAX_NUM_BATCHED_TOKENS` (batch size, chunked-prefill) | batch geometry shifts reduction order → near-tie flips | prior finding: bs=1 vs bs=32 moves ~62% of greedy tokens |
| `FA_SLIDING`, `SPLITKV_VERIFY`, `VLLM_MARLIN_USE_ATOMIC_ADD` | kernel/reduction-order toggles (#73 source attribution) | #73: default self-deterministic; atomic_on breaks it |

**The load-bearing finding is that the *dominant* pin — speculation on/off — is
NOT a knob you can satisfy while keeping the optimization.** Pinning batch size,
precache, tokenizer, FA_SLIDING, etc. makes the gate *well-posed*, but even with
all of them fixed (the clean A/B did exactly that) the **M=K+1-verify-vs-M=1-decode
geometry difference remains and produces 56.1% divergence.** The self-referential
gate is satisfiable **only with speculation OFF (M=1)** — i.e. only by giving up
the speedup. So `decode_path_pin_invariants` is not a list of knobs that *rescue*
greedy-identity; it is the proof that the binding constraint is the verify batch
geometry itself, which no served-config pin removes. (#73 multi-config captures
confirm each config is self-deterministic *within* spec-on, but self-determinism ≠
identity-to-AR — the distinction this PR makes precise.)

## Step 4 — splitk_ppl_projected (moot for the verdict, but stated)

PPL is **not** the binding constraint here — greedy-identity is (the RED above).
For completeness:

- **Greedy argmax flips do not enter PPL** (PR #66). The PPL gate
  (`same_path_ppl.py`) teacher-forces a *fixed* GT corpus and scores prefill
  logprobs of the *provided* tokens; the 971 near-tie flips and the 56.1% greedy
  divergence are generation-side and never reach the PPL number. So "near-tie
  flips don't move PPL" is true **by construction**, independent of their count.
- **The kernel reduction-order perturbation** that *does* touch the prefill logits
  (genuine-SplitK emu, `mean_abs_dlogit ≈ 0.367`, `compounding.npz`) changes the
  *signed-mean* NLL — the quantity PPL depends on — by far less than the
  mean-*absolute* per-token change, because for confident tokens the perturbation
  shifts the gt-logit and the logsumexp together (cancellation). The npz stores
  only `max|δ|` per position (not signed δ at the GT token) and teacher-forces the
  *greedy* trajectory, **not** the PPL GT corpus
  (`data/ppl_ground_truth_tokens.jsonl` is **absent locally**), so a tight measured
  projection is out of local scope.
- **Empirical anchor:** the deployed stack already runs split-KV verify and
  measures PPL **2.376976** (#73 run_00 meta); #52 measured **2.3777**. The W4A16
  SplitK linear reduction is the same *class* of perturbation, and the 2.42 cap
  carries 0.0423 slack precisely to absorb reduction-order kernel choices (PPL is
  itself computed in a large-M prefill geometry that already differs from M=1).

`splitk_ppl_projected ≈ 2.378` (first-order unchanged from #52's 2.3777; **well
under the 2.42 cap**). Definitive confirmation needs a served PPL run of the actual
#84 build on the GT corpus (HF, approval-gated) — but PPL is not what fails here.

---

## Bottom line

`self_referential_gate_confirmed` = **reference self-referential: YES; spec ==
own-AR: NO.** The deployed speculative stack is **56.1% token-divergent from its
own M=1 AR greedy decode** — it fails the self-referential greedy-identity gate,
deterministically. "SplitK/tree greedy-safe by construction" is **refuted**: a
frontier inherits the same batch-non-invariant verify divergence. The stack is
leaderboard-legal only because the official harness runs **no** token-identity
check (extends #38/#66). **Verdict: 🔴 RED — escalate to advisor.**
