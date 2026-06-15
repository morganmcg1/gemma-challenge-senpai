<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# EAGLE-3 measured-read RUNBOOK — single-approval HF-job block for the Issue #319 read (STRICT world)

**PR:** #350 · **Author:** ubel · **Date:** 2026-06-15 · **W&B group:** `eagle3-read-runbook`

**0-GPU launch-prep PACKAGING. 0 TPS added. BASELINE 481.53 UNCHANGED. `no_hf_job`, `no_launch`,
`no_served_file_change` — this card PACKAGES the read; the human SUBMITS it.** This is the
mechanically-ready single-approval block for the Issue #319 EAGLE-3 measured read, assembled so the
human can fire **one** `a10g-small` job with **zero further advisor round-trips**. Do not file
`/v1/jobs:run`, do not run `train.py --launch` — those happen only after the four pre-launch gates in
§6 PASS **and** a human approves on Issue #319.

> **STRICT-IDENTITY WORLD (the live contract).** The human reversed Issue #124 on Issue #319 at
> **2026-06-15T10:56:17Z**: *"No, ignore #124, we want to ensure we stick with the strict greedy
> token matching."* That re-instates Issue **#192** as the binding contract. This runbook is built in
> the strict world (advisor Option A): the coverage bar reverts to the **identity bar 0.9213**, and
> **served greedy-token-identity == 1.0 becomes a co-equal HARD gate** (not report-only).

> **This read is a DIAGNOSTIC, NOT a >500 build GO/NO-GO gate.** Under strict, **three independent**
> results — wirbel #343 (`kklof4wr`) + denken #332 + fern #349 (`u8vmtji0`, FlashInfer-BI
> `restoration_ceiling = 473.53`) — cap the deployed substrate at **473.5 TPS < 500 for every
> realizable deterministic schedule, regardless of coverage** — so no coverage retrain on this lane
> reaches 500. The read's honest value is (a) served **greedy-identity** confirmation, (b) **C2**
> closure end-to-end, and (c) exact **gap-to-0.9213 sizing** on the existing head. See §1, §5 and §7.

---

## §0 — The single-approval GO, in one block

| field | value |
|---|---|
| **Approval issue title** | `Approval request: HF job for eagle3-read-strict` |
| **Submission** | `submissions/fa2sw_precache_kenyan` (merged frontier `fa2sw_precache_kenyan` + merged #317 boot-guard) |
| **Method tag** | `ubel/eagle3-read-strict` (free-form W&B metadata; execution driven by the manifest) |
| **Hardware** | HF Jobs `a10g-small` (sm_86), vLLM `0.22.1rc1.dev307+g3e8afdf78` |
| **Input head** | fern #34 `gua9x68j` (train `56ksyxgw`) → convert with `convert_eagle3_to_safetensors.py` (ubel #333, `quzi85y0`) → publish vLLM-loadable |
| **Read kind** | `method:"eagle3"` linear **K=7 eager-path single-stream** read; per ubel #322 (`2nmem4dc`) this needs **no** T6/T7 onegraph rewrite (the MTP-keyed patches go inert under `method:"eagle3"`) |
| **Cost** | exactly **one** `a10g-small` org-credit job, well under the 40-min (2400 s) wall; ~1 GPU-hr |
| **Expected outcome** | **NO-GO (gap-sized):** existing head ≈ 0.8903 < 0.9213, P(clear) ≈ 0.06 → confirms a retrain is needed, with the exact starting gap measured |

The copy-pasteable command is §2. **DO NOT RUN it until the four §6 gates PASS and a human approves on
Issue #319.**

---

## §1 — Why STRICT, and what this read can / cannot be

**The reversal.** Issue **#124** (accepted-risk / PPL-only, greedy-rate report-only) was the premise
this PR was originally assigned under. The human reversed it on Issue #319 at **10:56:17Z**
("…ignore #124… stick with the strict greedy token matching"), re-instating Issue **#192** as the
live contract. Per `CLAUDE.md` a direct human directive takes priority, so the bars flip:

- coverage bar reverts to the **identity bar `0.9213` (`0.9213011665456927`)** — **not** the PPL-only
  `c*=0.9089`;
- **served greedy-token-identity == 1.0** becomes a **co-equal HARD gate**, not a report-only line.

**What this read is NOT.** It is **diagnostic, NOT a >500 build GO/NO-GO** gate. In the strict world
the deployed substrate is **supply-capped at 473.5 TPS < 500** — wirbel #343 (`kklof4wr`) proved
`strict_500_reachable = False`: even at *perfect* coverage (the central λ-ceiling 520.95) the strict
ceiling is `520.95 × (1 − 0.09103) = 473.53` (denken #332 geometric-φ supply floor 0.09103), and the
worst-anchor strict ceiling is `447.999`. **fern #349 (`u8vmtji0`) independently confirms this** — a
FlashInfer batch-invariant `restoration_ceiling = 473.53` — so the cap is a **three-way** result
(wirbel #343 + denken #332 + fern #349), not a single estimate. The gap to 500 is **26.47 TPS** that
no coverage retrain on this lane can close. So a clearing coverage read here would **not** unlock >500;
the EAGLE-3 head alone inherits the ~0.73% M=8 divergence and is supply-capped.

**What this read IS** (§5): cheap (~1 GPU-hr) diagnostic insurance — served greedy-identity
confirmation (the now-binding gate), end-to-end C2 closure, and exact gap-to-0.9213 sizing on the
existing head before any GPU-weeks retrain.

---

## §2 — The exact single-approval HF-job command + the metrics it returns

```bash
cd /workspace/senpai/target
# (after the 4 pre-launch gates in §6 PASS + human approval on Issue #319)
python train.py \
  --submission submissions/fa2sw_precache_kenyan \
  --method "ubel/eagle3-read-strict" \
  --launch --wait
```

`--method` is free-form W&B metadata (`train.py:67`); execution is driven entirely by the manifest +
the staged probe (the ubel #322 §1 EAGLE-3 manifest delta: `method:"mtp"→"eagle3"`,
`model`/`LOCAL_DRAFTER_DIR`/`DRAFTER_BUCKET`/`DRAFTER_SHA256` → the published `gua9x68j` head,
`num_speculative_tokens:7`, aux `[2,21,39]` baked in the ckpt config). `--launch --wait` →
`scripts/run_hf_job.py:launch_job()` → `POST /v1/jobs:run`; the `enforce_launch_gate(...)` pre-flight
blocks the call unless the §6 smoke evidence is PASS. **Single launch only; the human runs this, not
the advisor.**

**What the one job returns** (single-stream, 128 × 512, greedy decode):

| metric | bar | kind | what it resolves |
|---|---|---|---|
| `uncond_top4` (coverage on the official 128 ShareGPT prompts) | **≥ 0.9213** (`0.9213011665456927`) | HARD | exact gap-to-identity-bar on the existing head |
| `served_greedy_identity_rate` | **== 1.0** | HARD (the now-binding #192 gate) | end-to-end C2 closure: served greedy-token-identity |
| `ppl` | **≤ 2.42** (deployed `2.3772`) | HARD | quality contract |
| `completed` | **== 128** | HARD | full public set |
| `alpha_per_depth` `[a1 … a7]` (per-depth α) | report | DIAGNOSTIC | sizes E[T] and the coverage gap |
| `vram_peak_gib` | ≤ 24.0 (ubel #299/#306: 20.158 peak / 3.84 headroom) | HARD | memory feasibility |

The harness is the ubel #322 read protocol: pass (b) `scripts/profiler/rank_coverage.py`
(`RANKPROBE_W=4` → `uncond_top4`), pass (a) `scripts/profiler/accept_calibration.py` (per-depth α →
served greedy-identity rate + E[T]), both read-only on the single `gua9x68j` load.

---

## §3 — STRICT GO/NO-GO bars (the two flips vs the PPL-only world)

| # | gate | quantity | STRICT bar | kind |
|---|---|---|---|---|
| G1 | **coverage** | `uncond_top4` (unconditional top-4 root coverage, official 128 ShareGPT) | **≥ 0.9213** (`0.9213011665456927`, the identity bar — **not** `c*=0.9089`) | HARD |
| G2 | **greedy identity** | served greedy-token-id rate vs plain greedy AR | **== 1.0** (co-equal **HARD** gate, #192 — **not** report-only) | HARD |
| G3 | **quality** | `summary.json:ppl` | **≤ 2.42** (deployed `2.3772`, margin 0.0428) | HARD |
| G4 | **completion** | `completed` | **== 128** | HARD |

The two strict flips vs the lifted (#124) world: G1's bar is the identity `0.9213011665456927`
(was `c*=0.9089`), and G2 is a HARD co-equal gate (was report-only). NaN-clean echo of the
load-bearing constants: identity bar `0.9213011665456927`; strict ceiling `473.5295953446407`
(< 500, supply-capped); worst-anchor strict ceiling `447.99898136862197`.

**Worst-500 secondary readout (carried):** the strict worst-anchor ceiling is `447.999` and the
PPL-only worst-500 coverage bar is `c*=0.9256` (`0.925603648491971`) — both below the strict
substrate cap, so the worst corner reinforces that this is a diagnostic, not a >500 gate.

---

## §4 — Pre-flight gate: only C2 is GPU-residual (ubel #338, `y4jj278b`)

ubel #338 (PR #338, `y4jj278b`) closed **3 of the 4** Issue #319 load caveats at 0 GPU. The pre-flight
is **green** for the cheap caveats; only **C2** needs the GPU smoke this read provides:

| caveat | status | closed at 0-GPU |
|---|---|---|
| **C1** — config-field survival + class registration (15/15 param-manifest 1:1, AutoConfig survival) | **CLOSED-AT-0-GPU** | ✅ green |
| **C2** — fp32→bf16 inference numerics **+ served greedy-identity == 1.0** | **REQUIRES-GPU-SMOKE** | ⛔ the one this read closes |
| **C3** — absent-`d2t` → identity-map default (`draft_vocab == target_vocab == 262144`) | **CLOSED-AT-0-GPU** | ✅ green |
| **C4** — vLLM-fork version / schema pin (`0.22.1rc1.dev307+g3e8afdf78`) | **CLOSED-AT-0-GPU** | ✅ green |

C1/C3/C4 are **green**; spend the GPU only because **C2** (the bf16-numerics + served greedy-identity
caveat) is irreducible — a CPU static audit cannot exercise a live forward. This read is exactly the
minimal spend that closes C2.

---

## §5 — The read's honest VALUE (and the expected outcome)

Three things this ~1 GPU-hr read delivers — none of which is a >500 unlock:

1. **Served greedy-identity confirmation.** Under strict, greedy-token-identity == 1.0 is the
   now-binding gate; only a live served draw measures it. This read confirms the served
   greedy-identity rate directly.
2. **C2 closure end-to-end.** It exercises the fp32→bf16 inference numerics on a real vLLM load and
   the served greedy-identity together — closing the one GPU-residual caveat (C2) from §4.
3. **Exact gap-to-0.9213 sizing.** It measures the existing head's live coverage so we know the
   precise retrain delta. lawine #330 (`hfrscdai`) puts the existing head at **0.8903**, **0.031**
   below the bar, with **P(clear) ≈ 0.06** → the honest **expected outcome is a NO-GO** with a
   measured gap, not a hoped-for clear. The read tightens the CI; it does not move the central.

**Why sizing the gap is worth it:** lawine #339 (`0aq16szh`) shows a 4-lever retrain clears the
0.9213 identity bar with **P ≈ 0.843** (independent) / 0.794 (+0.5-correlated) — ROI **JUSTIFIED**.
Knowing the exact starting gap (this read) de-risks the GPU-weeks retrain before it is authorized:
cheap insurance before an expensive, likely-to-clear retrain.

---

## §6 — §0 checkpoint-publish blocker + the four pre-launch gates

The `gua9x68j` head is a local plain-PyTorch artifact with **no Hub/bucket path** (arch_notes §7);
local `/workspace/...` paths are invisible to the HF runner. Before the one launch it must be
**published** vLLM-loadable and **smoke**-tested on the local AWS A10G (no HF Job):

1. **Convert + publish.** Run `convert_eagle3_to_safetensors.py` (ubel #333) on the `56ksyxgw` best
   checkpoint → a vLLM `Eagle3LlamaForCausalLM` directory (`config.json` with
   `architectures:["Eagle3LlamaForCausalLM"]`, `draft_vocab_size:262144`, `norm_before_fc:true`,
   `eagle_aux_hidden_state_layer_ids:[2,21,39]`); publish to
   `hf://buckets/gemma-challenge/gemma-senpai/weights/eagle3-inrepo-251head/`; compute
   `sha256(model.safetensors)` → the new `DRAFTER_SHA256`.
2. **Local A10G smoke (no HF Job).** Boot `serve.py` with the EAGLE-3 manifest; confirm vLLM loads
   the head as `Eagle3LlamaForCausalLM` (0 missing/unexpected; 15/15 per ubel #338), `/v1/models`
   returns 200 (boot-guard active), greedy decode is token-identical on a handful of prompts, and a
   tiny PPL spot-check is sane. This writes the `enforce_launch_gate` PASS evidence.
3. **Apply the manifest delta** (ubel #322 §1) and upload the submission to
   `hf://buckets/gemma-challenge/gemma-senpai/submissions/senpai/fa2sw-precache-kenyan`.
4. **Human approval on Issue #319** via the §10 `Approval request: HF job for eagle3-read-strict`
   issue. Only then run §2.

**If the checkpoint cannot be made vLLM-loadable, the read is BLOCKED** (not a head verdict) → fix
the export or revert to a known-loadable retrain. Do not burn the launch on an unverified checkpoint.

---

## §7 — The strict >500 reality + where the live search moved (honest add)

**Strict >500 needs two things together, and this read is neither.** Per ubel #192, a strict-compliant
>500 requires the **reduction/batch-invariant verify kernel** (restores greedy identity) **plus** an
**E[T]/ceiling lever** (speed) — together. The EAGLE-3 coverage retrain is at best the E[T] half, and
even at perfect coverage it is supply-capped at 473.5 < 500 (wirbel #343 + denken #332 + fern #349
`u8vmtji0` `restoration_ceiling = 473.53`, three-way). So the EAGLE-3 coverage lane is
**measured-dead under strict**; the read prices the head, it does not unlock the target.

**Where the live strict >500 search actually moved** (advisor pointers, 2026-06-15 11:29:49Z — cited
as handed off; this card does **not** inspect those branches). The strict-compliant **frontier
ladder** is:

| TPS | lane | ref |
|---|---|---|
| **165.44** | non-spec | lawine #196 |
| **357.32** | off-shelf batch-invariant spec | #326 |
| **≤ 481.53** | custom reduction-invariant kernel | wirbel #354 |

and the genuinely-new levers now in flight are **wirbel #354** (custom-kernel compliance ceiling),
**lawine #355** (sub-int4 body), **fern #357** (composite reachability), and **kanna #359**
(identity-preserving step-shave). **This reframed read is the diagnostic that confirms the EAGLE-3
substrate's strict-identity status underneath all of that** — it tells the >500 search whether the
EAGLE-3 head is a usable draft substrate (served greedy-identity == 1.0, C2 closed) at its measured
coverage, before any of those levers leans on it.

**Alternative for the human (their call).** Because the EAGLE-3 coverage lane cannot reach >500 under
strict, the single Issue #319 approval might be better spent on a **strict ceiling-lift** measurement
that attacks the 473.5 cap at its root — e.g. denken's sub-int4 body-quant (lawine #355) or stark's
sub-saturation verify screens (this cycle). This runbook is packaged ready-to-fire **as the strict
read**, but the human may prefer to point the one approval at a ceiling-lift screen instead.

---

## §8 — PPL-only CONTINGENCY (carried ONLY for "if #192 is ever lifted")

**This block is NOT the live contract** — it applies **only if #192 is ever lifted** (the #124
PPL-only world). It is carried so the package is complete if the contract changes again; the bars in
§3 (strict) are the live ones. Imported verbatim from wirbel #343 (`kklof4wr`):

| world | coverage target for 500 | lift from 0.8903 | within +0.031 budget? |
|---|---|---|---|
| **STRICT (live, #192)** | identity bar **0.9213** (and greedy == 1.0) | +0.031 | n/a — supply-capped at 473.5 < 500 regardless |
| PPL-only central (contingency, #124) | **c\*=0.9089** (`0.9089363308345582`) | +0.0186 | yes (within +0.031) |
| PPL-only worst (contingency, #124) | **c\*=0.9256** (`0.925603648491971`) | +0.0353 | no (marginally over) |

In the PPL-only world the >500 lane becomes a feasible coverage retrain (supply tax = 0); in the
strict world it does not. PPL stays `2.3772 ≤ 2.42` in both (wirbel #324 PPL-decoupling). **Live bars
= §3.**

---

## §9 — Banked anchors (assemble, cite, reuse exact — do NOT re-derive)

| anchor | W&B | what is reused |
|---|---|---|
| ubel #322 read spec/protocol | `2nmem4dc` | the read protocol, manifest delta, `rank_coverage.py`/`accept_calibration.py` harness, eager-path no-rewrite finding |
| ubel #333 converter | `quzi85y0` | `convert_eagle3_to_safetensors.py` (15/15 tensors, bf16, nan-clean) |
| ubel #338 vLLM load dry-run | `y4jj278b` | C1/C3/C4 CLOSED-AT-0-GPU; C2 the GPU-residual; 15/15 param-manifest |
| wirbel #343 strict envelope | `kklof4wr` | strict ceiling `473.5295953446407`, `strict_500_reachable=False`, PPL-only `c*` contingency |
| fern #349 FlashInfer-BI (advisor pointer) | `u8vmtji0` | `restoration_ceiling = 473.53` — independent third confirmation of the 473.5 strict cap |
| lawine #330 coverage prior | `hfrscdai` | existing head `0.8903`, P(clear 0.9213) ≈ 0.06, gap `0.031035`, LIKELY-MISSES |
| lawine #339 retrain clear prob | `0aq16szh` | retrain clears 0.9213 with P ≈ 0.843 (indep) → ROI JUSTIFIED |
| fern #34 head | `gua9x68j` (train `56ksyxgw`) | the only trained {2,21,39} fusion head the read loads |
| PR #52 frontier | `2x9fm2zx` | baseline 481.53 / PPL 2.3772 / 128/128 (unchanged; 0 TPS) |

**Scope:** `no_hf_job`, `no_launch`, `no_served_file_change`, no GPU, no submission. **BASELINE 481.53
UNCHANGED**, **0 TPS** added, greedy/PPL untouched. This card authorizes nothing; it PACKAGES the
strict read so the human can fire **one** `a10g-small` job with zero further design work.

---

## §10 — Approval-request issue body (file this to launch)

> **Title:** `Approval request: HF job for eagle3-read-strict`
>
> **PR / branch:** #350 / `ubel/eagle3-read-runbook` (targets `approval-gated-8gpu-20260613`).
>
> **What:** ONE `a10g-small` org-credit HF Job that loads the trained {2,21,39} fusion EAGLE-3 head
> `gua9x68j` (fern #34, published vLLM-loadable per §6) and runs a `method:"eagle3"` K=7 eager-path
> single-stream read returning: `uncond_top4` coverage, **served greedy-identity rate**, per-depth α,
> `ppl`, `completed`.
>
> **STRICT world:** coverage bar **0.9213** (identity), **greedy-identity == 1.0 HARD gate** (#192),
> `ppl ≤ 2.42`, `128/128`. This read is a **diagnostic** (greedy-identity confirmation + C2 closure +
> gap-to-0.9213 sizing), **NOT a >500 GO/NO-GO**: wirbel #343 caps the strict substrate at 473.5 < 500.
>
> **Expected outcome:** NO-GO, gap-sized — existing head ≈ 0.8903 < 0.9213, P(clear) ≈ 0.06.
>
> **Greedy/PPL risk:** the read measures served greedy-identity (it does not change emission); PPL is
> the deployed path's `2.3772 ≤ 2.42`. BASELINE 481.53 unchanged, 0 TPS.
>
> **VRAM / runtime / quota:** 20.158 GiB peak / 3.84 GiB headroom (ubel #299/#306); exactly ONE
> `a10g-small` job, well under the 40-min wall; ~1 GPU-hr. Launch exactly once.
>
> **Pre-launch gates done:** §6 checkpoint published vLLM-loadable + local A10G smoke PASS (0
> missing/unexpected, `/v1/models` 200, greedy-identical, PPL sane); manifest delta applied;
> submission uploaded. C1/C3/C4 green (ubel #338); only C2 is GPU-residual. Requesting human approval
> to fire the single job.
>
> **Honest alternative:** the EAGLE-3 coverage lane is supply-capped < 500 under strict, so you may
> prefer to point this approval at a strict ceiling-lift screen (denken sub-int4 body quant / stark
> sub-saturation verify) instead — your call (§7).

---

## Public evidence used

- **Issue #319** (advisor, 8gpu launch) + the human's **10:56:17Z** #124 reversal (strict greedy
  token matching) and the advisor's Option-A steer — the contract this runbook is built to.
- **wirbel #343** (`kklof4wr`): strict ceiling `473.5295953446407`, `strict_500_reachable=False`
  (denken #332 supply floor 0.09103), and the PPL-only `c*` contingency.
- **fern #349** (`u8vmtji0`, advisor pointer): FlashInfer batch-invariant `restoration_ceiling = 473.53`
  — an independent third confirmation that the strict substrate is capped below 500.
- **lawine #330** (`hfrscdai`): existing-head coverage prior `0.8903`, P(clear 0.9213) ≈ 0.06, gap
  `0.031035`, verdict LIKELY-MISSES — the expected NO-GO + the gap this read sizes.
- **lawine #339** (`0aq16szh`): retrain clears 0.9213 with P ≈ 0.843 — why the gap-sizing read is
  worth ~1 GPU-hr before a GPU-weeks retrain.
- **ubel #322 / #333 / #338**: the read protocol (`2nmem4dc`), the converter (`quzi85y0`,
  `convert_eagle3_to_safetensors.py`), and the load dry-run (`y4jj278b`, C1/C3/C4 green).
- **fern #34** (`gua9x68j` / train `56ksyxgw`): the one trained {2,21,39} fusion head the read loads.
- **Public leaderboard** (read-only): no publicly documented strict-compliant EAGLE-3 >500 served path
  exists for this model; the read prices the existing head's exact gap, the cheapest pre-retrain step.
