# PR #659 — AIME precision-allocation lower bound (experiment ledger)

Cross-window state for an autonomous, resumable multi-window experiment. Source of
truth for cell completion = `results/summary_<body>_<kind>.json` + `results/<body>_<kind>.jsonl`.
Harness: `build_mixed.py` (CPU build) + `eval_mixed.py` (idempotent per-item, soft-cap 82min).

## Question
Uniform precision is the wrong axis (#646: uniform int8 AIME 0.4167 < 0.420 bar). Is the
int4 AIME deficit (0.400 vs bf16 0.4667) LOCALIZED in a few decoder layers? If so, the
MIN layer-upgrade that clears AIME 0.420 and its TPS price = the precision-allocation LB.

## Banked anchors (no re-run)
- N=0 pure int4 AIME **0.400** (dh0tbwpp); GPQA ~0.50 (clears).
- N=all uniform int8 AIME **0.4167** (#646 jz3ojbio) — base caveat: built from plain-bf16
  base, not qat_unq. Cite as the uniform anchor.
- Full bf16 AIME **0.4667** (zoszxnb0) = recovery ceiling.
- Bars: AIME 0.420 (90%); GPQA 0.4409.
- Decoder = 42 layers (0-41). lm_head locked int4 (never touched).

## INTERPRETER MAP (load-bearing — window-4 build failed on this)
- **BUILD** (`build_mixed.py`): use the **vLLM venv** python `/tmp/senpai-venvs/20f658587e8a6643/bin/python`
  — it has `compressed_tensors`+torch. `/usr/bin/python3` has torch but NOT compressed_tensors → build crashes.
- **EVAL** (`eval_mixed.py`): use `/usr/bin/python3` (has wandb; venv does NOT). Run eval from cwd=/tmp
  (wandb local-dir shadow gotcha). Eval is an HTTP client to the served vLLM, needs no compressed_tensors.

## HARNESS VERIFIED (2026-06-18 ~12:00, window-2) — all green, trust the deltas
- **build_mixed.py**: int4 modules copied byte-identical (in-script assert l.256-259); bf16
  upgrade POPS the int4 packed tensors + writes plain `.weight` -> a healthy serve PROVES bf16
  load (no int4 tensor exists to silently fall back to). A1 serves coherent AIME at ~34 tok/s.
- **int8 routing PROVEN on CPU** (`_verify_int8_routing.py`, vLLM 0.22.0 real resolver
  `CompressedTensorsConfig.get_scheme_dict`): upgraded-layer fused names (qkv/o/gate_up/down)
  -> num_bits=8; non-upgraded L2/L5/L41 -> num_bits=4 ("Linear"); lm_head -> num_bits=4
  (group_1); regex `(0|1)` excludes L10 (no alternation leakage). Precedence: find_matched_target
  exhausts stage-1 layer-name/regex over ALL target keys before stage-2 "Linear" substring, so
  the int8 regex wins regardless of group order. **No silent-int4-fallback risk for Phase B.**
- **eval_mixed.py**: idempotent per-item jsonl resume, banked byte-identical prompts +
  evalsets.score_item (matched instrument), TPS proxy (mean s/item + tok/s from t_req_s),
  wandb logs ONLY on complete cells and carries prior_rid across windows (no dup runs).
- **bank**: 60 AIME items w/ gold + scorer present.

## RESUME PROTOCOL (fresh-window decision tree — source of truth = results/ files)
For the active cell C (`results/<C>_aime.jsonl`):
1. `wc -l results/<C>_aime.jsonl`; eval PID alive? (`ps aux | grep eval_mixed`)
2. PID alive + jsonl growing  -> DON'T relaunch (port 8000 busy); ScheduleWakeup ~20-25min.
3. PID dead + lines < 60      -> SOFT-CAP/crash: relaunch SAME eval cmd (resumes skip-done).
4. PID dead + lines == 60     -> cell DONE: read summary_<C>_aime.json, record acc/tok/s below,
   `rm -rf /workspace/gemma_build/<C-body>` (free disk), advance to next cell.
Only ONE vLLM server at a time (1 GPU). Build next body (CPU) only after deleting the prior.

## Cell order (critical path = localize, then descend)
A1 bf16 L0-13 -> A2 bf16 L14-27 -> A3 bf16 L28-41  (rank thirds = locus)
-> N=0 int4 tok/s baseline — **acc ALREADY BANKED 0.400 (dh0tbwpp, ubel#650), do NOT re-run n=60**;
   only need int4 decode tok/s on THIS harness/server for apples-to-apples pricing → SHORT run
   (~3-5 items, kill after tok/s stabilizes) [slot in anytime GPU free]
-> Phase B int8 on winning third, then a smaller sub-block (find min_N)
-> N=all int8-from-qat uniform anchor (distinct from #646's plain-base int8)
-> Phase C: GPQA n=198 guard on candidate + 5-seed sampled AIME IF a small/med-N cell crosses 0.420.
DISK: 29G free; pinned int4(9.7)+qat(15)=24.7G; ONE active body(~12G) + room to build next. Delete-after-eval.

## Phase A — bf16 locus probe (3 thirds, MAX signal to rank locus)
| cell | layers | prec | status | AIME acc | s/item | tok/s | note |
|------|--------|------|--------|----------|--------|-------|------|
| A1 mix_bf16_L0-13  | 0-13  | bf16 | **DONE** n=48 (soft-capped; body deleted, frozen — no-lift needs no n=60) | **0.3542** (17/48) | 105.3 | 34.98 | first third; Wilson90 [0.251,0.473], 8 trunc@len. **NOT the locus** — point est BELOW int4 0.400, no lift toward bf16 0.4667. Body freed for disk. |
| A2 mix_bf16_L14-27 | 14-27 | bf16 | **DONE** n=60 (clean; body deleted) | **0.4333** (26/60) | 106.4 | 34.0 | middle third; Wilson90 [0.316,0.559], 13.3% trunc, wandb **u936qrqz**. **Candidate locus** — above int4 0.400, clears 0.420, near bf16 ceiling 0.4667. (Soft-cap n=44 read 0.4545 regressed to 0.4333 at n=60; +2/60 over int4 is noise-dominated, lean on slope not point.) |
| A3 mix_bf16_L28-41 | 28-41 | bf16 | RUNNING (window-5, eval PID 1890553, serve PID 1890558, launched 14:43 via chainer) | — | — | — | last third; body /workspace/gemma_build/mix_bf16_L28-41 (pre-built 14:36) |
Rank thirds by ΔAIME vs 0.400. Best third = locus; its bf16 acc = localized ceiling.

### Decision after Phase A
- best third bf16 >= ~0.45 (near 0.4667) -> localized & recoverable -> Phase B (int8 descent).
- best third bf16 ~0.42-0.44 (partial) -> test bf16 on best-2-thirds, then descend.
- ALL thirds bf16 <= ~0.41 -> delocalized -> recovery needs many layers -> EXPENSIVE.

## Phase B — int8 descent on the winning locus (find min N, the cheap LB)
- B1: int8 on winning third (does CHEAP int8, not bf16, on the locus clear 0.420?).
- B2: int8 on a smaller sub-block (find the knee / min_N_to_clear_0p420).
- **SMOKE FIRST (window-7 harness review):** mixed 4/8 GPU serving is NEW (build_mixed int8
  route verified on CPU resolver only via _verify_int8_routing.py; #646 proved uniform int8
  serves but not mixed-4/8). Before the first int8 n=60: build the body, serve it, gen 1-2
  AIME items, eyeball coherent greedy output (no garble/repeat) → THEN run full n=60. Cheap
  insurance vs an 80-min wasted window on a misbuilt int8 body.
- **TPS proxy is tok/s, NOT s/item** (s/item scales with output length; AIME gens 2.3-3.4k tok).
  bf16-thirds ran ~34 tok/s. int4 should be fastest, int8-on-N mid, bf16-on-N slowest (HBM read).

## ⚠️ WINDOW-10 CORRECTION (16:25Z) — decisive cell is int8-on-LOCUS, NOT uniform int8
The window-8 "monotone-ceiling" shortcut below is **UNSOUND** and is hereby superseded.
It assumed AIME(int8-on-subset) <= AIME(uniform int8) via monotonicity in #upgraded-layers.
But the bf16 thirds **disprove** that monotonicity: middle third LIFTS (0.4333) while first
(0.354) and last (<=0.417) sit AT/BELOW int4 0.400 — so adding int8 OUTSIDE the locus is NOT
guaranteed to help, hence uniform int8 is NOT a valid upper bound for int8-on-the-locus.
**The clean cheap-recovery test is int8 ON the locus (middle third L14-27)**, directly comparable
to bf16-on-locus 0.4333. Built `mix_int8_L14-27` (118 mods->int8, rel_err 0.0063, 11.0GB, clean).
Uniform int8 (`mix_int8_all`) is retained ONLY as the PR-requested N=all sanity anchor, run after.

A3 ARITHMETIC LOCK: A3 @17 correct/45, 8 items left -> final in [0.283, 0.417], CANNOT reach 0.420.
=> middle third is the LONE third clearing the bar => locus = L14-27, confirmed. A1/A3 both below int4.

## WINDOW-11 (2026-06-18 ~17:11Z) — int8-on-locus LIVE state (decisive cell mid-flight)
mix_int8_L14-27: eval PID 1928427 + driver PID 1927771 alive, **44/60** scored, acc **0.500**
(regressing 0.564@39 -> 0.5116@43 -> 0.500@44, the expected noise settle; HOLD read to n=60 ~17:23Z,
finishes FIRST-PASS, no soft-cap). Serve verified: loaded mix_int8_L14-27, compressed-tensors, BI
kernel registered, 10.5GiB. Smoke item1 (2024-II-4) -> coherent log-algebra -> \boxed{33}=gold.
**TPS-PRICING KEY FINDING:** int8-on-locus decodes **84.6 tok/s** (consistent: median 84.3, min 77,
max 93) vs bf16-on-locus **34.0 tok/s** — and completion LENGTHS are comparable (int8 mean 3494 tok
vs bf16 3614 tok), so the 2.5x s/item gap is PURE per-token decode speed, not shorter gens. HBM model
(A10G M=1 memory-bound): bf16-on-14 reads ~2x int4 bytes -> ~0.5x tok/s; int8-on-14 ~1.33x -> ~0.75x.
=> **bf16 recovery is TPS-EXPENSIVE (34 tok/s); int8 recovery is TPS-CHEAP (~85 tok/s) IF it clears
0.420.** Need the pure-int4 tok/s run (task #2) for the apples-to-apples denominator. 4 len-trunc@6144,
all-wrong (truncated reasoning never reaches boxed answer) — same ~9-13% trunc band as bf16 thirds.
Decisive logic stands: int8-on-locus n=60 >=0.420 clean -> CHEAP -> GPQA+5-seed confirm -> SURFACE.

## WINDOW-12 (2026-06-18 ~17:45Z) — Phase B greedy DONE; confirmation chain LAUNCHED
**int8-on-locus mix_int8_L14-27 DONE clean n=60:** AIME **0.45 (27/60)**, Wilson95 [0.331,0.575],
tok/s **81.79**, peak 18.9GB, W&B **nmjvtfov**. Clears 0.420. ABOVE bf16-on-locus 0.4333 (noise).

**TPS PARETO (local A10G proxy, conc1 M=1, this harness — NOT official TPS):**
| body | N | tok/s | cost vs int4 | AIME greedy |
|------|---|-------|--------------|-------------|
| int4 N=0 (priceprobe) | 0 | **97.73** | — | 0.400 (banked dh0tbwpp) |
| **int8-on-locus L14-27** | 14 | **81.79** | **−16.3%** | **0.45 (clears)** |
| bf16-on-locus L14-27 | 14 | 33.97 | −65.2% | 0.4333 (clears) |
| bf16 full | 42 | (slower) | — | 0.4667 (ceiling) |
=> **int8-on-the-locus recovers AIME at ~16% local-TPS cost vs ~65% for bf16 — the CHEAP Pareto point.**
(Official cost needs an HF job = NOT run; report local-proxy ratio, official-measurement ruling theirs.)

**Honest caveat:** greedy +3/60 over int4, Wilson95 overlaps BOTH int4 0.400 AND the bar → noise-dominated.
Card's Phase-2 mandate fires. **Confirmation chain `/tmp/confirm_chain.sh` LAUNCHED (driver PID logged
to /tmp/confirm_chain.driverpid; log = results/_confirm_chain.log):**
1. int4 priceprobe — DONE (97.73 tok/s). 2. **5-seed sampled AIME** s{12345,23456,34567,45678,56789},
   bodies mix_int8_L14-27_s<seed> (same checkpoint), pooled via `pool_sampled.py` → tight Wilson.
3. GPQA n=198 greedy guard on mix_int8_L14-27 (expected trivial; int8 ⊃ int4 prec). 4. N=all mix_int8_all anchor.

**METHODOLOGY (load-bearing for the writeup):** the AIME endpoints (bf16 0.4667 zoszxnb0 / int4 0.400
dh0tbwpp) are **GREEDY** per the PR baseline — there is NO banked *sampled* AIME reference. So the
0.420 bar is greedy-derived, and the 5-seed sampled cell is a **CONSERVATIVE one-sided test**: sampled
acc ≤ greedy (sampling penalty + temp-1.0 truncation deflation), so **sampled-clears-0.420 ⟹ robustly
recovered**; sampled-fails-0.420 is AMBIGUOUS (could be the sampling/trunc penalty, not precision). IF
the candidate sampled result is borderline/below, run an **int4 N=0 5-seed sampled paired reference**
(same seeds) — the paired diff (int8-locus − int4, McNemar) cancels both confounds. Don't pre-spend
those 5 runs; gate on the candidate read.

**Verdict in progress:** AIME_RECOVERABLE_CHEAP-leaning (greedy clears cheaply), PENDING the sampled
confirm. CHEAP ⇒ SURFACE-to-human (int4-QAT-mandate ruling theirs), NOT autonomous fire.

## WINDOW-13 (2026-06-18 ~18:13Z) — pickup; confirmation-read REFINEMENT
Chain healthy: `bash /tmp/confirm_chain.sh` PID 1950529 alive, cell 2 (s12345) eval PID 1950545
at 38/60, serve PID 1950552. Queue intact: s12345→s23456→s34567→s45678→s56789→GPQA n=198→N=all.
**Load-bearing read-correction:** the plan's "conservative one-sided" test (int8 SAMPLED pooled
CI-lo ≥ 0.420) is **almost surely UNACHIEVABLE** — sampled ≤ greedy 0.45, and a 300-sample Wilson
half-width is ~0.047, so CI-lo ≥ 0.420 needs pooled ≈ 0.47 > greedy ceiling. So the standalone
sampled CI will land *below* the greedy-derived bar **regardless of whether recovery is real** →
"ambiguous" by construction. ⇒ The DECISIVE confirmation is the **PAIRED** comparison: int8-locus
SAMPLED vs int4-N=0 SAMPLED on the SAME 5 seeds (300 paired item×seed outcomes), McNemar/paired-
bootstrap on (int8−int4). That cancels the sampling-penalty confound and directly tests "does the
int8 upgrade on L14-27 lift AIME, noise-robustly?" — the actual scientific question.
- **Budget gate honored:** do NOT pre-spend the int4 seeds. Let the 5 int8 seeds pool first; if
  pooled int8 sampled CI-lo ≥ 0.420 (unlikely) → robustly CHEAP, skip int4 ref. Else → launch int4
  N=0 sampled ref (5 seeds) for the paired diff = the load-bearing recovery test.
- pool_sampled.py reports `CI_lo_above_int4` but vs int4 GREEDY 0.400 (confounded); the paired McNemar
  is the clean version. Extend pooling for the paired diff when int4 seeds land.
- Path: int8 seeds → pool → [gate] int4 N=0 sampled ref + paired McNemar → GPQA n=198 guard → N=all
  anchor → terminal SENPAI-RESULT. GPQA+N=all are card-mandated but verdict-trivial (run after the
  decisive paired read). Tasks TaskList #1-6.

## Phase B REFINEMENT (window-8, 15:40Z) — [SUPERSEDED by window-10, see above] the monotone-ceiling argument
**Key structural fact:** an int8-on-N-layers body interpolates between N=0 (all int4, AIME 0.400)
and N=all (uniform int8). So **int8-on-ANY-subset is bounded above by uniform int8.** #646's
uniform int8 = 0.4167 (< 0.420 bar) — but that was a PLAIN-bf16 base, not this ladder's qat_unq
base, so it is NOT a valid ceiling for THIS ladder. Therefore the single most decisive Phase B
cell is **N=all int8-from-qat** (uniform int8 on qat_unq):
  - If N=all int8-from-qat **< 0.420** (likely): NO int8 subset can clear → the CHEAP (int8) lever
    is CLOSED by construction. min_N_to_clear via int8 = ∞. Recovery then requires **bf16** on a
    locus; Phase A already priced that (bf16-on-a-third ≈ 0.43-0.45, marginal, at full bf16 HBM
    cost ~34 tok/s). → verdict **AIME_RECOVERABLE_EXPENSIVE** (cheap-int8 axis closes; only bf16
    recovers, expensively). SKIP the int8 third-descent (B1/B2) — it cannot beat its own ceiling.
  - If N=all int8-from-qat **>= 0.420** (surprise — qat base beats plain base): THEN run B1/B2 int8
    descent on the best third to find the true min-N cheap LB.
**Critical-path Phase B = build qat_unq uniform int8 → SMOKE 1-2 items → AIME n=60.** This one cell
replaces the speculative int8-third guesses. Still need the short int4 tok/s run for pricing.
CAVEAT for the writeup: all three bf16-thirds CIs (A1 [.23,.50], A2 [.32,.56], A3 pending) overlap
int4 0.400 AND bf16 0.4667 — the 4-item int4→bf16 gap is below n=60 noise (±~8 items), so "ranked
locus" is a weak/slope read, NOT a crisp single-layer localization. Report honestly.

## Phase C — guards + confirmation
- GPQA n=198 greedy on the shipped candidate cell (guard >= 0.4409).
- Phase-2 5-seed sampled AIME on candidate IF it crosses 0.420 at small/medium N (Wilson CI).
- **BUDGET GATE:** 5-seed sampled AIME = 5x n=60 ~= 6.7 GPU-hr. Fire ONLY on a CLEAN Phase-B
  greedy crossing (a small/med-N int8 cell whose acc sits clearly >=0.420 AND fits a monotone
  N-slope, not a lone +1-item blip). If Phase-B greedy stays flat/within-noise of int4 0.400,
  SKIP the 5-seed and conclude EXPENSIVE/IRRECOVERABLE from the slope — don't burn 6.7 GPU-hr
  chasing a noise-floor crossing.

## Verdict map
- AIME_RECOVERABLE_CHEAP: small-N int8 clears 0.420 at tolerable TPS -> SURFACE to human.
- AIME_RECOVERABLE_EXPENSIVE: needs near-bf16/large-N -> quality-recovery axis CLOSES.
- AIME_IRRECOVERABLE: even bf16-on-locus fails -> re-open locus.

## WINDOW-9 (2026-06-18 16:05Z) — live state on pickup
- A3 mix_bf16_L28-41: eval PID 1890553 alive @82min soft-cap, **44/60** jsonl, drive log last @40/60.
  Self-healing finish-driver PID 1903387 ALIVE (waits on A3 PID, relaunches skip-done -> n=60).
  DO NOT relaunch manually (port 8000 busy). ETA n=60 ~16:35Z. Source of truth = summary json @n=60.
- mix_int8_all (decisive int8 ceiling cell): **BUILT + clean** (build log: 342 modules int8,
  rel_err mean 0.0061, lm_head int4, 12.34GB). NOT auto-chained — launch MANUALLY after A3 frees GPU.
  Smoke = watch item 1 of the full run (~4min boot+gen); kill only if garbled (jsonl idempotent).
- Critical path remaining: (1) A3->n=60 [autonomous], (2) int8 ceiling AIME n=60 [decisive],
  (3) short int4 tok/s pricing run, (4) GPQA guard + 5-seed ONLY if int8 cell clears 0.420 cleanly,
  (5) terminal SENPAI-RESULT + submit.
- Decision tree: int8 ceiling <0.420 => int8 lever CLOSED by construction => EXPENSIVE (only bf16
  recovers; bf16-on-a-third ~0.43 borderline at full HBM cost). int8 ceiling >=0.420 => descend int8
  thirds for true min-N. Plan via ScheduleWakeup (no fragile multi-stage chain; finish-driver = the
  only running auto-loop). Tasks tracked in TaskList #1-5.

## Ops constraints
- analysis_only=true, official_tps=0, NO HF job, NO submission. Group aime-precision-allocation-lb-fern.
- DISK (rechecked window-6 ~14:50): **59GB free on /** (much healthier than the ~28GB noted
  earlier). Constraint effectively relaxed — can hold the active body + build next + pinned
  without deleting first (lets Phase B pipeline build-while-eval). Still tidy up each mixed body
  after its eval+guard finish. Do NOT touch /tmp/senpai-venvs (29GB, holds the live vLLM server
  python). int4_g128_lmhead(9.7G)+qat_unq(15G) pinned; mix_bf16_L28-41(12G) active (A3).
- 90-min hard / 82-min soft per window. ~65-80 min per AIME n=60 cell (~65s/item, ~2336 tok).
- Build cmd: `python build_mixed.py --out /workspace/gemma_build/<body> --upgrade-layers <spec> --upgrade-precision <int8|bf16>`
- Eval cmd: `VLLM_BATCH_INVARIANT=1 /usr/bin/python3 eval_mixed.py --body-name <body> --body-path /workspace/gemma_build/<body> --evals aime --mode full --decode greedy --upgrade-layers <spec> --upgrade-precision <prec> --soft-cap-min 82 --wandb-group aime-precision-allocation-lb-fern`
