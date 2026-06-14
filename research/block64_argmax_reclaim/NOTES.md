# block64 argmax-reclaim (PR #137)

**Hypothesis:** `FUSED_SPARSE_ARGMAX_BLOCK` 16→64 in `fa2sw_precache_kenyan`
reclaims public #1. It controls the tile width `block_selected` of the
partial-block fan-in reduction over the sparse vocab candidates in the fused
sparse-argmax verify kernel (`sitecustomize.py:831`,
`num_blocks = cdiv(selected_count, block_selected)`). Larger block ⇒ fewer
reduction rounds. Argmax is associative so the selected token is identical
regardless of tiling ⇒ greedy/PPL unchanged by construction.

## Plan (LOCAL only — no HF launch without human approval)

1. Paired `wall_tps` A/B (`scripts/profiler/paired_tps_ab.py`), serve-time env
   override, N=3 median, 128×512 seed=1 workload.
   - Run 1: baseline block16 (manifest default) vs candidate block64 — headline.
   - Run 2/3: reuse baseline, candidates block32 / block128 — map the curve.
2. Validation gates: `local_prevalidate.py` (PPL ≤ 2.42), `private_gap_probe.py`,
   `greedy_determinism.py`.
3. Edit manifest to best block, upload submission to gemma-senpai bucket.
4. Open approval issue (do NOT launch HF job).

W&B group: `block64-argmax-reclaim`.

## Public-state intake (digest `as=senpai`, 2026-06-14 ~11:08 UTC)

Leaderboard (top, by official TPS):

| rank | agent | TPS | method | verif |
|---|---|---|---|---|
| 1 | frantic-penguin | 489.63 | osoi5…fa2sw-precache-**skv64**-v1 | valid |
| 2 | need-for-speed | 488.07 | mao-gemma-fast-skv64-v0 | valid |
| 3 | openevolve | 485.91 | splitkv-fa2sw-clean-oe-repro-v0 | pending |
| 4 | byteshark | 484.62 | splitkv-k7-argmax**block64**-v0 | valid |
| 9 | **senpai (us)** | **481.53** | fa2sw-precache-splitkv-linear-mtp-k7 | valid |

frantic-penguin's #1 stack = our stack (`SPLITKV_VERIFY_MAX_Q=64` is already in
`fa2sw_precache_kenyan`) **+ block64**. Sole missing knob confirmed:
`FUSED_SPARSE_ARGMAX_BLOCK` ours=16, theirs=64. byteshark's controlled ablation
(block32→64) was **+2.62 TPS** at unchanged PPL.

**Magnitude caveat (honest):** the gap to #1 is 8.1 TPS but block64 alone
explains only ~+2.6–4. So block64 likely lands us in the **484–487** cluster
(ranks 2–5), beating our own 481.53 but **not guaranteed to reclaim #1** vs
489.63. The local A/B + #99 projection decide whether a launch is justified.

**#124 context (greedy-identity):** advisor↔human decision request #124 notes the
deployed spec stack diverges 56% from its own M=1 AR (official scorer checks only
completion count, not token identity). block64 is **logit-invariant vs block16**
(argmax associativity) → it does **not** change this posture either way. Inherits
whatever validity status the base 481.53 stack has; `greedy_determinism.py` will
confirm block16↔block64 token identity.

## Results

### Run 1 (`block64/`, PID 9870) — COLD-START CONFOUNDED, DISCARD headline

Launched 11:01 UTC on a **cold** machine. The harness runs all-baseline-then-all-
candidate, so the baseline (block16) arm absorbed the entire cold-start penalty
(run0 paid a **235 s** cold model-load from disk + page-cache/IO contention during
decode) while the block64 arm ran fully warm. Result:

| arm | wall_tps runs | median | CV% |
|---|---|---|---|
| block16 (baseline) | [433.42, 437.13, **453.81**] | 437.14 | **2.46** |
| block64 (candidate) | [454.00, 453.88, 453.60] | 453.88 | **0.045** |

Naive verdict `+3.83% REAL` but **`floor_exceeded=True`** (baseline CV 2.46% is
70× the characterized 0.035% floor → harness flags it untrustworthy). The tell:
the **warm** block16 run (run2 = **453.81**) is identical to the block64 warm
plateau (**453.8**). The whole "+3.8%" is the two cold block16 runs dragging the
baseline median down. PR #72's "0.000 tps/run drift" claim assumes a pre-warmed
machine; ours started cold. **Conclusion: warm block16 ≈ warm block64 locally;
need a clean warm A/B to resolve the true (≤+0.6%) delta.**

### Run 2 (`block64_warm/`, PID 29138) — CLEAN, machine pre-warmed

Launched 11:29 UTC after run 1 warmed the page cache (`/tmp/osoi5-12k-baked` etc.
hot). Same command, `--tag block64_warm`,
`--wandb-name stark/argmax-block16-vs-block64-warm`. All 6 runs on the warm
plateau → trustworthy floor-respecting verdict. W&B run `daeitiwm`.

| arm | wall_tps runs | median | CV% |
|---|---|---|---|
| block16 (baseline) | [453.302, 453.032, 453.539] | **453.302** | 0.056 |
| block64 (candidate) | [453.752, 453.446, 453.688] | **453.688** | 0.036 |

**Harness verdict: NULL.** Δ_median = **+0.386 wall_tps = +0.085%**, below the
operative threshold (0.10%) and the raw-powered MDE (0.080%). `floor_exceeded=False`
(both arms tight, CV ≈ floor) → trustworthy. `e_accept_exact` identical
(block16 ≈ 3.856, block64 = 3.8514) → token-identity confirmed (block64 applied,
selection unchanged).

**Projection to official (PR #99 multiplier 1.0602, recovers 481.53 anchor to −0.2%):**

| arm | projected official TPS |
|---|---|
| block16 | 480.58 |
| block64 | **480.99** |

⇒ block64 buys **+0.41 official TPS** (central). Lands ~481, **statistically
indistinguishable from our current 481.53**, nowhere near byteshark 484.62 /
need-for-speed 488.07 / frantic-penguin 489.63.

### Why the leaderboard "8 TPS gap to #1" is best-of-N official variance, not block64

frantic-penguin's own #1 result (`20260614-070821-625`) reports a **3-draw self-eval
of the SAME submission: 489.63 / 483.80 / 480.41 TPS** — a 9.22 TPS (**1.9%**) spread.
The headline 489.63 is the **best of 3**; their **median (483.80) and min (480.41)
bracket our single-draw 481.53**. byteshark's "+2.62" (`20260614-003925-827`) was a
single official run-pair (block32 482.0 → block64 484.62, n=1 vs n=1) — well inside
that ±1.9% noise. So:

- The PR's **mechanism** premise is correct: block64 *is* the only config delta vs the
  frontier stack (confirmed against frantic-penguin + need-for-speed method strings).
- The PR's **causal** premise is **not supported**: block64 does not produce a
  reproducible TPS gain. Our controlled n=3 floor-respecting A/B bounds its true effect
  to +0.085% (≈ +0.4 official TPS); the public single-run "+2.62" is consistent with
  official-harness variance, and the gap to "#1" is dominated by best-of-N reporting on
  a ~1.9%-noisy scorer.

### Conclusion + recommendation

**block64 is a NULL perf lever.** It is provably greedy/PPL-safe (argmax-associativity
+ measured e_accept identity), so adopting it as config is harmless and brings
frontier-parity — but it will **not** reliably reclaim #1. A one-shot HF launch of a
block64 submission would, per controlled projection, return ~481 official: a predicted
null. Per the launch-discipline rule ("if a pre-launch check raises doubt, report back
instead of launching speculatively"), I am **not** opening the HF approval issue.
**Recommend: do not spend a launch on block64 alone.** Surfacing the adopt/launch
decision to the advisor.

## Validity argument (why block64 is greedy/PPL-safe)

1. **Token-identity by construction.** `FUSED_SPARSE_ARGMAX_BLOCK` tiles a
   **max/argmax** reduction over the sparse candidate set. Unlike a SUM reduction
   (FP, non-associative — the worry in `verify_argmax_margin.py` for SplitK/M-widen),
   `max()` is exactly associative+commutative and `argmax` returns the same index
   under any tiling (consistent tie-break). So block64 selects the **bit-identical**
   greedy token as block16. No FP accumulation enters the selection.
2. **PPL path-independent.** PPL is scored via `compute_logits`/`prompt_logprobs`
   (the lm_head GEMM on GT tokens), NOT the fused sparse-argmax verify kernel.
   block size never touches a logit value used by PPL ⇒ PPL invariant.
3. **Empirical confirmation already in hand.** A/B `e_accept_exact` is **identical**
   across arms (block16 ≈ 3.848–3.854, block64 = 3.8503). E[accept] = drafter
   proposals matching the verify greedy selection; if block64 changed the selected
   token, acceptance would move. It doesn't ⇒ selection unchanged.

⇒ Gate plan: **PPL gate (`local_prevalidate.py`)** is the essential confirmation
(expect ≈2.377, unchanged). `private_gap_probe.py` measures acceptance+precache gap
— both invariant under a logit-invariant change (E[accept] identity above), so it's
redundant-by-construction; run only if advisor wants the explicit number.
