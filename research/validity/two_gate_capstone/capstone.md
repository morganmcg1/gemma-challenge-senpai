# Two-Gate Capstone — which constraint must relax to reopen a fire path?

> **`analysis_only=true`, `official_tps=0`. LOCAL diagnostic card — NO FIRE.**
> No HF Job, no `/v1/jobs:run`, no `train.py --launch`, no submission, no served-file change.
> CPU/light-GPU synthesis of ~15 already-banked cards. `--wandb_group two-gate-unsatisfiable-capstone`.
> Date: 2026-06-17 (post-grounding ubel #580 / land #581 / lawine #584).

---

## TL;DR — VERDICT

**`two_gate_unsatisfiable = TRUE`** under the conjunction

> **{ mandated int4 checkpoint } ∧ { #319-operative identity } ∧ { ≥90%-of-base quality } ∧ { faster-than-375.857 TPS }**

The quality-safe, identity-safe, int4 ceiling is **`base_fullhead` ≈ 252.69 TPS**, which is **≥113 TPS (≈30%) short** of the 375.857 ship. The ship reaches 375.857 only by stacking **two** levers — a 12k head-prune (kills quality) **and** MTP spec-dec (kills #319 identity) — and **each lever violates a different gate**. That is the "two-gate" interlock.

- **`n_binding_constraints = 4`** (all four are members of the irreducible unsatisfiable conjunction).
- **`reopens_fire = [int4: FALSE, identity: FALSE, quality: FALSE, target>375.857: TRUE]`** — exactly **one** single-constraint relaxation reopens a fire: **dropping the 375.857 target** (then `base_fullhead` 252.69 ships as-is, quality+identity+int4 intact).

**Robustness:** the verdict does not depend on any knife-edge. The speed gap is ≥30% and survives every contested anchor reading (see Flag #4). Even a *free* lm_head, spec-off, tops out at 328.9 TPS < 375.857 (#544).

**⚠️ This refines the PR's "Expected".** The PR expected *two* relaxations to reopen a fire without quality loss (drop-target **and** relax-int4). Measurement says **only drop-target fires**: relaxing int4 → bf16/int8 makes the body *slower* (bf16 = 143.99 TPS, int8 = 205.48 TPS; body census #571 `vct3k1vc`), moving *away* from 375.857. See the relaxation map + Flag for the refutation.

---

## The converged ledger (re-confirmed against W&B)

Three legs. Each headline number was re-checked against its cited run summary; **4 flags** are recorded below.

### Leg A — QUALITY is SATISFIED (`base_fullhead`, 4 re-anchored gates ≥90% of harness vanilla base)

| Gate (≥90% of base) | Bar | `base_fullhead` (int4) | Margin | Run / source |
|---|---|---|---|---|
| MMLU-Pro | ≥0.605 | 0.6313 | +0.026 ✓ | land #581 `qi24h8zx` (gate); arm value from #581 family |
| GPQA-Diamond | ≥0.471 | **0.4798** | **+0.009 ⚠ knife-edge** | #574 `7bi4e2ne` (paired 95/198) |
| GSM8K (pass) | ≥0.807 | ~0.85 | ~+0.04 ✓ | land #581 family |
| AIME (maj@1, min8) | ≥0.090 | 0.1167 | +0.027 ✓ | ubel #580 `yokbmy9i` |

- Denominators grounded to **harness-measured** vanilla base (MMLU 0.6727 / GPQA 0.5236 / GSM8K 0.8967 / AIME 0.100), not literature cites (land #581 `qi24h8zx`).
- The "AIME collapse vs ~0.400" was a **protocol artifact**: served harness vanilla-base AIME = **0.100 (6/60)**, Wilson95 [0.047, 0.201] (`yokbmy9i`); the 0.400 cite is extended-thinking / large-token-budget (72% truncation at the 3072 cap), not our greedy maj@1 min8 harness. int4 quant tax ≈ 0 (−0.017). `aime_gate_achievable_by_any_int4 = true`.

### Leg B — SPEED is the SOLE WALL (`base_fullhead` ≈ 252.69 < ship 375.857)

- **Quality-safe ceiling**: `base_fullhead` = full int4_g32 QAT body + full native 262k bf16 lm_head, spec-OFF, greedy, MAX_NUM_SEQS=1, min_tokens=8 (#541). **252.69 TPS / PPL 2.0057** (wirbel #553 `83jiwjr9`; corroborated 252.306 by decomposition #544).
- **Decomposition (#544):** the gap is the **262k bf16 head matmul (82.2% of the exec gap)**, not argmax. Quantizing the *full* head (int4) recovers ≤38.3 TPS → **optimized quality-safe ceiling ≈ 292.1 TPS**. Even a **free** head (spec-off) ≤ **328.9 TPS**. Both still < 375.857.
- **Internal precision levers all closed** (`any_strict_safe_speed_lever_anywhere = FALSE`): HEAD #556 (252.31, lever FALSE), ATTENTION #562 (FALSE), BODY #571 `vct3k1vc` (int4_g32=252.69 is the fastest non-byte-exact body; every other precision is slower or higher-flip). Kernels dead (stark #582). Decode-overhead floor 311.27 TPS, 99.41% GPU-bound (#569).
- **Spec-dec closed on BOTH gates** (see Leg C for the identity half). Speed half: `any_measured_drafter_clears_ship = FALSE`. Honest break-even acceptance = **4.95**; best measured point **MTP K=7 e_accept=3.844 → 262.9 official TPS, gap −113.0** (lawine #584 `gd5s78ze`, empirical; corroborates fern #583 `xmdeo3dj` analytic `specdec_two_gate_closed=True`).

### Leg C — IDENTITY is OPERATIVE / #407-int4-referenced

- The live #319 contract is operationally the **self-referential greedy identity** (token-identical to the checkpoint's own int4 greedy serving path), **not** literal bf16 byte-identity. **No int4 config is literal-bf16 byte-exact** (wirbel #585 `2u44yaa1`). `base_fullhead` is self-det 8/8 steady-state.
- Spec-dec **fails** this operative identity by construction: MTP sequence-exact ≈ 15.6–18.8%, per-step ≈ 0.996; cascade over 512 positions → most sequences diverge. `root_cause = genuine_precision` (bf16 reduction-order tie reorder at M=8 verify), `specdec_identity_fire_eligible = FALSE` (denken #576 `g7yob0yg`; fern #583).
- **Context (issue #124 ruling):** the *official scorer* runs TPS + PPL(≤2.42) + 128/128 only — **no token-identity check**. So #319-operative identity is an **internal contract**, not an official gate. (Honest caveat: a strict batch-1 audit would find the entire int4-spec leaderboard divergent.) This matters for the relaxation map: relaxing identity is relaxing a *self-imposed* constraint — and it *still* does not reopen a fire (Leg B: 262.9 < 375.857).

---

## W&B re-confirmation — FLAGS (instruction #1)

17/17 spot-checked headline numbers reproduce against their run summaries, **except** these 4 caveats. None overturns the verdict.

1. **PPL provenance.** `83jiwjr9` (wirbel #553) reproduces TPS (252.688) but logs **no PPL key** — the 2.0057 PPL is sourced from the #544 decomposition card, not this run. Corroborated < 2.42 elsewhere; non-load-bearing for the verdict.
2. **GPQA knife-edge is unresolved in `qi24h8zx`.** #581's run logs the *denominators* but **not** the int4 `base_fullhead` GPQA/MMLU-Pro arm values. The GPQA pass rests on #574 `7bi4e2ne` = **95/198 = 0.4798** (paired), which clears ≥0.471 by only **+0.009**. A second measurement of the *same* config (`intact_body_headwidth`, seed 12345) gives **93/198 = 0.4697**, which **fails** by −0.001. The promised kanna **#579 confirmation is unlanded.** → GPQA "pass" carries a one-seed asterisk (does **not** change the verdict — speed wall is independent).
3. **Project-path / attribution.** `xmdeo3dj` lives in `wandb-applied-ai-team/gemma-challenge-senpai` (PR body links `morganmcg1/gemma-challenge`); `gd5s78ze` (#584) logs the empirical Pareto; MTP-K7 `e_accept=3.844` is common to both.
4. **⚠ ANCHOR-FRAME inconsistency on 252.69 (material — flag prominently).** PR #587 body + #571/#544/#553 treat **252.69 as spec-OFF `base_fullhead`**. But lawine **#584 (merged after #583)** asserts **`anchor_252_is_mtp_not_nospec = True`** — i.e. 252.69 is *already MTP-K7-served* — and on that basis corrected fern #583's intermediate numbers (break-even 4.95 not 2.68; ngram proj 109.79 not 285; "previous inflated best 320.24 → 109.79"). **These two readings are mutually exclusive and the ledger has not reconciled them.** **Verdict is robust either way:** if 252.69 is spec-off, the achievable ceiling is 252.69 (+spec ≤262.9); if 252.69 is already MTP-served, spec-off `base_fullhead` is *even slower*. Every reading puts the achievable ceiling **≤ 262.9 ≪ 375.857**. (Authoritative spec-dec figures going forward = #584's corrected ones, which *supersede* my own #583 intermediate numbers while agreeing on the closure verdict.)

---

## The relaxation map (decision deliverable)

For each binding constraint: relax it **and only it** → what opens, nearest known config, does a fire reopen, cost/risk.

| # | Binding constraint | What opens if it ALONE relaxes | Nearest known config | `reopens_fire` | Cost / risk |
|---|---|---|---|---|---|
| 1 | **mandated int4 ckpt** | bf16 / int8 body | bf16 body+full head = **143.99 TPS**; int8 = **205.48 TPS** (body census #571) | **FALSE** | **Refutes PR-expected.** Larger dtype ⇒ more weight bytes ⇒ *slower*: −109 TPS (bf16) / −47 TPS (int8) vs 252.69. The only *faster* body is sub-int4, which kills quality (a different gate). bf16 *does* enable literal-bf16 byte-exact identity — but at 143.99 TPS, irrelevant to a 375.857 race. |
| 2 | **#319-operative identity** | spec-dec / MTP (non-self-identical decode) | MTP K=7 full-head = **262.9 TPS** (best measured drafter) | **FALSE** | Even with identity dropped, best measured drafter projects 262.9 (gap −113); `any_measured_drafter_clears_ship=False`, break-even 4.95 unreachable (#583/#584). To *also* clear 375.857 you must stack a head-prune ⇒ kills quality (the OTHER gate). Identity is internal-only (#124), so this relax has no official-scorer cost — but still no fire. |
| 3 | **≥90% quality** | 12k head-prune (dixie-flatline) | 12k/free-head, spec-off ≈ **325–329 TPS** (free-head ≤328.9, #544) | **FALSE** | Even with quality dropped, head-prune spec-off tops out ≈328.9 (gap −47). To *also* clear 375.857 you must stack spec-dec ⇒ kills identity (the OTHER gate). PPL/accuracy collapse (12k prune drops 250k vocab rows) — catastrophic, not a marginal dip. |
| 4 | **faster-than-375.857 target** | drop the speed bar | **`base_fullhead` 252.69 TPS** | **TRUE** ✅ | **None** on quality/identity/int4 — `base_fullhead` is int4 ∧ #319-operative-identity ∧ quality-passing, ships as-is. Sole asterisk: GPQA-D knife-edge (Flag #2). |

**Reading the map.** Only **row 4** fires. Rows 2 and 3 expose the interlock: the ship (375.857) needs **both** levers (head-prune **and** spec-dec) because each alone falls short (328.9 and 262.9 respectively), and **each lever destroys a different gate** — so no single relaxation among {identity, quality} reopens a fire. Row 1 fails for an independent reason: every non-int4 body is *slower*. Hence `{int4, identity, quality}` are **jointly** (not individually) binding; the **375.857 target is the only individually-relaxable-to-fire constraint.**

---

## What the human decision needs

1. **If the contract stands as written → NO-FIRE is correct and robust.** No int4, identity-safe, ≥90%-quality config reaches 375.857; the gap is ≥30% and survives every contested anchor reading.
2. **The single cheapest way to ship today is to relax the *target*, not a quality/identity/precision knob.** `base_fullhead` (252.69 TPS, quality-safe, identity-safe, int4) is a *clean* submission the moment the bar is below ~252 TPS. The only open item on that path is confirming the GPQA-D knife-edge (kanna **#579**, unlanded).
3. **Relaxing int4 does NOT buy speed** (contra the PR's expected) — it buys literal-bf16 *identity* at a 109-TPS speed *cost*. If the human wants byte-exact identity rather than speed, that is the lever; if they want speed, it is the wrong lever.
4. **Two genuine speed unlocks both require breaking a gate the contract forbids:** head-prune (forfeit quality) or spec-dec (forfeit operative identity). Each alone is *still* short of 375.857; only stacked do they reach it — and stacking forfeits *both* gates.

### Public-evidence sanity (external, non-launch context)
The quality-first conclusion aligns with the broader serving literature: int4 QAT is the practical accuracy floor for this model class (sub-int4 destroys quality), and lossless speedups on a memory-bound 262k-vocab head are dominated by the dense weight read — consistent with our 82.2% head-tax decomposition. No external result contradicts the NO-FIRE.

---

## Provenance (run IDs)
`83jiwjr9` (#553 TPS anchor) · `7bi4e2ne` (#574 GPQA paired) · `yokbmy9i` (#580 AIME) · `qi24h8zx` (#581 gates) · `vct3k1vc` (#571 body census) · `2u44yaa1` (#585 identity-operative) · `g7yob0yg` (#576 identity mechanism) · `xmdeo3dj` (#583 spec-dec analytic) · `gd5s78ze` (#584 spec-dec empirical). Decomposition: `research/validity/base_fullhead_tps_ceiling/decomposition.json` (#544). Closure: `research/base_fullhead_specdec/specdec_two_gate_closure.json` (#583).
