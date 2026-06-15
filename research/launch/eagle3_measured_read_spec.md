<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# EAGLE-3 measured-read launch spec + GO/NO-GO thresholds (Issue #319 Option A)

**PR:** #322 · **Author:** ubel · **Date:** 2026-06-15 · **W&B group:** `eagle3-measured-read-spec`

**0-GPU launch-prep card. 0 TPS added. BASELINE 481.53 UNCHANGED. THIS WRITES THE
SPEC — IT DOES NOT RUN IT.** A single `a10g-small` launch of this read requires the
human approval pending on **Issue #319**. Do not file `/v1/jobs:run`, do not run
`train.py --launch`. This card exists so that the moment #319 is approved the job
ships with zero further design work.

**Purpose:** convert the **GREEN-pending-build** EAGLE-3 verdict (Issue #319: step-side
definitively closed at the free ceiling 487.7 < 500, so an E[T]-raise via the
{2,21,39} fusion drafter is the only remaining >500 lever) into a **measured GO/NO-GO**
by taking ONE single-stream `a10g-small` read of the **already-trained in-repo
{2,21,39} EAGLE-3 head** (fern #34, W&B `gua9x68j` / train `56ksyxgw`). This is
the decisive, cheapest gate — it is run *before* any fusion retrain (Issue #319
Option B).

---

## HEADLINE

The measured read is a **linear K=7 EAGLE-3 eager-path single-stream run** on top of
the merged frontier `submissions/fa2sw_precache_kenyan` (+ the now-merged #317
`_guard_included_router` boot-guard). It measures the **substrate-independent
head-quality numbers** the whole GREEN-pending-build case rests on — native step-1
**a1** (vs the 0.7731 salvage bar), accepted-tokens-per-verify **E[T]** (vs the
3.9914 honest-500 floor), **PPL ≤ 2.42**, **128/128**, greedy token-identity — plus
the **eager-path TPS floor** (the empirical answer to wirbel #314). It does **not**
require the T6/T7 onegraph-loopgraph rewrite: under `method:"eagle3"` vLLM runs its
own `EagleProposer`, the MTP-keyed onegraph patches go **inert** (not wrong, just
dormant), so the read lands on the stock eager substrate (~402 TPS at iso-MTP
acceptance, *rising* with the head's higher E[T]).

**The single gating dependency is checkpoint publication.** The `gua9x68j` head is a
**local training artifact with no Hub/bucket path** ("deployment gated on kanna #5";
vLLM-load verification deferred, arch_notes §7). It MUST be packaged as a
vLLM-loadable `Eagle3LlamaForCausalLM`, published to a Hub/bucket repo, and
**smoke-tested for vLLM-load + greedy identity on the local AWS A10G** before the one
HF launch. This is §0 below and the top launch gate.

**Decisive thresholds (single-stream, fill-on-read):**

| axis | measured quantity | GO bar | source |
|---|---|---|---|
| acceptance | native step-1 **a1** | **≥ 0.7731** | salvage bar, #309 `7tkn4d9x` |
| numerator | **E[T]** accepted-tok/verify | **≥ 3.9914** (floor); target **≈ 6.11** | #274 floor; #295 sizing |
| TPS | single-stream `summary.json:tps` | floor **≥ 481.53** (no frontier regression); GO target **≥ 500.0** | #52 frontier; #319 target |
| quality | `summary.json:ppl` | **≤ 2.42** | program.md cap |
| completion | `completed` | **== 128** | program.md |
| identity | greedy token-id vs plain greedy AR | report (diagnostic; strict-compliance = separate BI-verify lane, HUMAN DIRECTIVE #192) | program.md |

`go_threshold_single_stream_tps = 500.0` (the headline >500 target; the 481.53
frontier is the no-regression floor).

---

## §0 — CRITICAL PRE-LAUNCH BLOCKER: publish the `gua9x68j` checkpoint

The head exists and is measured (offline native accept/step `K=8` = **0.7792**,
native step-1 top-1 = **0.7714**, tf_acc 0.7617, loss 1.1702 on the 240-record
benchmark-matched holdout, `56ksyxgw` best==final @ step 4500). But it is a **local
plain-PyTorch artifact**, saved with vLLM-loadable weight names "best-effort" only
(arch_notes §7). **Local `/workspace/...` paths are not visible on the HF runner.**
Before the one launch:

1. **Locate** the `56ksyxgw` best checkpoint (the `gua9x68j`-evaluated weights;
   warm-started from #25 `full_20k/model_best.pt`).
2. **Package** as a vLLM `Eagle3LlamaForCausalLM` directory:
   - `config.json` with `architectures:["Eagle3LlamaForCausalLM"]`,
     `draft_vocab_size:262144`, `norm_before_fc:true`,
     `eagle_aux_hidden_state_layer_ids:[2,21,39]`, `target_hidden_size:2560`,
     `hidden_size:2560`, `num_hidden_layers:1`, `num_attention_heads:8`,
     `num_key_value_heads:2`, `head_dim:256`, `intermediate_size:10240`,
     `rms_norm_eps:1e-6`, `rope_theta:1e6`, identity `draft_id_to_target_id`.
   - `model.safetensors` with the §7 name remap (`midlayer.*`→`model.layers.0.*`;
     bare `fc/embed_tokens/norm/input_norm`→`model.*`; `lm_head.*` as-is).
3. **Publish** to a writable repo the HF runner can reach — preferred:
   `hf://buckets/gemma-challenge/gemma-senpai/weights/eagle3-inrepo-251head/` (the
   senpai scratch bucket, same mechanism as the live `DRAFTER_BUCKET`); alt: a
   private Hub model repo. Do **not** point `MODEL_ID`/drafter at a local path.
4. **Compute** `sha256(model.safetensors)` → the new `DRAFTER_SHA256`.
5. **Local AWS A10G smoke (no HF Job):** boot `serve.py` with the EAGLE-3 manifest,
   confirm (a) vLLM loads the head as `Eagle3LlamaForCausalLM` with 0 missing/unexpected
   tensors, (b) `/v1/models` returns 200 (boot-guard active), (c) greedy decode on a
   handful of prompts is token-identical to plain greedy AR, (d) a tiny PPL spot-check
   is sane. Only then is the head launch-ready.

**If the local checkpoint cannot be made vLLM-loadable (shape/name mismatch that the
§7 best-effort mapping missed), the measured read is BLOCKED and the path reverts to
Option B (retrain into a known-loadable export).** This is the single most likely
launch-blocker and is why the smoke in step 5 is mandatory before spend.

---

## §1 — Config (exact served-file basis + manifest delta)

**Basis:** `submissions/fa2sw_precache_kenyan/` at the current branch HEAD — i.e.
**including the merged #317 boot-guard** (`sitecustomize.py:1308` `_guard_included_router`,
`:1324` call; closes the Issue #272 `_IncludedRouter`/`prometheus_fastapi_instrumentator`
startup-500 gap that the pre-#317 frontier was missing). vLLM
`0.22.1rc1.dev307+g3e8afdf78` (stock upstream wheel, CUDA 12.9), `a10g-small`
(sm_86). Model `google/gemma-4-E4B-it`, all modalities loaded (the drafter swap
touches only text speculative decode; image/audio/video pathways unchanged).

**Drafter wiring — `manifest.json` env delta (the ONLY required edits for the eager read):**

| key | current (MTP) | EAGLE-3 read |
|---|---|---|
| `SPECULATIVE_CONFIG` | `{"method":"mtp","model":"/tmp/qat-assistant","num_speculative_tokens":7}` | `{"method":"eagle3","model":"/tmp/eagle3-inrepo-251head","num_speculative_tokens":7}` |
| `LOCAL_DRAFTER_DIR` | *(unset → default `/tmp/qat-assistant`)* | `/tmp/eagle3-inrepo-251head` |
| `DRAFTER_BUCKET` | `hf://buckets/gemma-challenge/gemma-kenyan-duma/weights/drafter-ft/ft-v1-epoch_001` | `hf://buckets/gemma-challenge/gemma-senpai/weights/eagle3-inrepo-251head` |
| `DRAFTER_SHA256` | `ed159e33…5dd18e` | `<sha256 of published model.safetensors>` |

**Wiring invariant:** `ensure_drafter()` (`serve.py:720`) syncs `DRAFTER_BUCKET` →
`LOCAL_DRAFTER_DIR` (`serve.py:51`, default `/tmp/qat-assistant`), so
`SPECULATIVE_CONFIG.model` **must equal `LOCAL_DRAFTER_DIR`**. Set both to
`/tmp/eagle3-inrepo-251head` (or leave both at the `/tmp/qat-assistant` default — the
local dir name is cosmetic, but the two must match or vLLM finds no head at `model`).

- `method "mtp"→"eagle3"` selects vLLM's `EagleProposer` + `Eagle3LlamaForCausalLM`
  head from the registry (config-only; integration card T1/T4). `num_speculative_tokens:7`
  is kept (linear chain, **no branching tree** → `splitkv_verify_patch.py` carries over,
  its K+1=8 verify-row shape unchanged; integration card T8 caveat `num_reqs==1`
  holds since `MAX_NUM_SEQS=1`).
- `[2,21,39]` aux layers are the vLLM default for a 42-layer target **and** are baked
  into the published ckpt `config.json` — no `SPECULATIVE_CONFIG` field needed.
- `Gemma4Model` already implements `SupportsEagle3`; vLLM auto-sets
  `use_aux_hidden_state_outputs` (integration card T3, "0 hours of vLLM work").
- `serve.py:1052` `append_env_arg(...,"SPECULATIVE_CONFIG","--speculative-config")`
  passes the JSON through unchanged; `ensure_drafter()` (`serve.py:~720`) syncs
  `DRAFTER_BUCKET`→`/tmp/eagle3-inrepo-251head` generically (it spuriously writes the
  MTP-only `centroid_intermediate_top_k` key into the drafter config — harmless,
  integration card T2).
- **No other served file is edited for the eager read.** The MTP-keyed onegraph
  patches (`sitecustomize.py` T5 fused-argmax, T6 `Gemma4Proposer` loopgraph, T7
  width-1-exact buffers) go **inert** under `EagleProposer` — confirmed dormant, not
  invoked (integration card T6, the load-bearing finding). This is *why* the read is
  cheap and *why* its TPS is the eager floor, not the deployed 481.53 substrate.

**Two served-path variants (the human picks one on #319; the head-quality gates are
identical for both — only the TPS interpretation differs):**

- **A.eager (RECOMMENDED, this spec's default):** the manifest delta above, nothing
  else. One run, minimal served change, measures real a1/E[T]/PPL/identity + the
  eager TPS floor. Directly answers wirbel #314 empirically.
- **A.loopgraph (higher-fidelity, more work):** additionally rewrite T6/T7 for
  `EagleProposer` so measured TPS reflects the eventual *deployed* substrate. ubel
  #315 cleared the draft-side capture-dispatch risk of this rewrite (the EAGLE
  KV-bearing draft chain is dispatch-safe within the ceiling-16 size-list), so it is
  feasible — but it is **not** "minimal cost" and is not needed to flip the decisive
  head-quality gates. Defer unless the human wants deployed-substrate TPS measured
  directly in the same run.

---

## §2 — Eval protocol

Single-stream, fixed by the official harness
`official/main_bucket/shared_resources/speed_benchmark/scripts/hf_bucket_single_job.py`
(constants are hard-coded, not submission-tunable):

- `NUM_PROMPTS = 128`, `OUTPUT_LEN = 512`, `MAX_CONCURRENCY = 1`,
  `REQUEST_RATE = "inf"`, `WARMUP_REQUESTS = 4`, `SEED = 1`.
- Single-stream enforced twice: harness `MAX_CONCURRENCY=1` **and** manifest
  `MAX_NUM_SEQS=1`. Client = `sglang.bench_serving --backend vllm-chat` on the public
  `eval_prompts_sharegpt.json` (the same file the submission precaches via
  `PRECACHE_DATASET=/harness/data/eval_prompts_sharegpt.json`).
- Greedy decode: `OVERRIDE_GENERATION_CONFIG={"temperature":0.0,"top_p":1.0,"top_k":0}`.

**Stages & artifacts** (`/state/`): speed → decode-capture → PPL.
- `summary.json`: `tps`(==`output_throughput`, **PRIMARY**), `output_tps`, `total_tps`,
  `completed` (gate ==128), `duration_s`, latency fields, `num_prompts`/`output_len`,
  then merged `ppl`, `ppl_num_tokens`, `decode_*`. (Note: `run_prefix` is the HF-Jobs
  system id, not a `summary.json` field.)
- `ppl_summary.json`: `ppl` (gate ≤ 2.42), `num_tokens`.
- decode capture writes `return_token_ids` output for the organizer greedy audit.

**Validity gates for THIS read:** `ppl ≤ 2.42` · `completed == 128` · all modalities
loaded · greedy token-identity reported (diagnostic; see §5 on the strict-compliance
lane). The local `enforce_launch_gate("fa2sw-precache-kenyan")`
(`scripts/run_hf_job.py:35`) blocks the API call if its evidence file is not PASS —
the §0 smoke must populate it.

---

## §3 — Cost (one `a10g-small` run)

- **Quota:** exactly **one** `a10g-small` org-credit HF Job (`/v1/jobs:run`). Launch
  exactly once; if §0 smoke or a transient error raises doubt, report back — do not
  retry speculatively.
- **Wall-clock:** `a10g-small` hard limit **40 min (2400 s)**. Budget:
  - startup: target weights from `WEIGHTS_BUCKET` + EAGLE-3 head sync (lm_head
    `[262144,2560]` ≈ 1.34 GB bf16 dominates the head; total head ≈ 1.5 GB) ≈ 5–12 min
    (cache-dependent).
  - benchmark: 128 × 512 = 65,536 output tokens; at the eager substrate (~402–500
    TPS) ≈ **130–165 s**; conservative upper bound ~200 s.
  - decode-capture + PPL: a few min.
  - **Total comfortably < 40 min**, single-stream, one job. (Reference: the 481.53
    frontier benchmark phase ≈ 136 s; the 44-TPS bf16 smoke
    `vllm-baseline-20260612T193622Z` took `duration_s=1488.8` and ran out of wall
    before PPL — the EAGLE-3 read is far faster and completes all stages.)
- **VRAM:** ubel #299 — {2,21,39}-fusion drafter + hidden-state retention fits the
  24 GB lane at **20.10 GB resident / 3.90 GiB headroom** (at rest); ubel #306 runtime
  peak **20.158 GiB / 3.84 GiB headroom**. extra_kv dominates (0.719 GiB); drafter
  weights negligible (0.037 GiB); net +0.80 GB. Memory is **not** the constraint.

---

## §4 — GO/NO-GO thresholds (what the read must clear to flip GREEN-pending-build → measured-GO)

The **decisive** gates are the two substrate-independent head-quality numbers
(`a1`, `E[T]`) plus the two hard validity gates (`ppl`, `completed`). TPS is read as a
*floor* (eager substrate) interpreted against wirbel #314, not as the primary GO/NO-GO.

| # | gate | quantity | GO bar | rationale |
|---|---|---|---|---|
| G1 | **acceptance** | native step-1 `a1` | **≥ 0.7731** | #309 salvage bar (M=8 tree relaxes #304's 0.9213 by 0.1482; `7tkn4d9x` 12/12). Modeled in-repo `a1=0.7714` (#308 `5axqa6oa`, +0.0017 inside the 0.0097 native-vs-tf spread). The read confirms it on a **real vLLM load**. |
| G2 | **numerator** | `E[T]` accepted-tok/verify | floor **≥ 3.9914**; target **≈ 6.11** | #274 honest-500 E[T] floor; #295 central sizing 6.11 (bracket [5.36, 6.86], `c334qaqu`). Deployed linear MTP sits at 3.8445 (< floor) — the read shows where the EAGLE head lands. |
| G3 | **TPS** | `summary.json:tps` | floor **≥ 481.53** (no frontier regression); GO **≥ 500.0** | #52 frontier 481.53; #319 target >500 (+3.835%). On A.eager the floor is the eager substrate (~402 rising with E[T]); clearing 500 *eagerly* is an outright GO with no rewrite (wirbel #314). |
| G4 | **quality** | `summary.json:ppl` | **≤ 2.42** | program.md cap (ref 2.30 + 5%). Frontier 2.3772. Hard gate. |
| G5 | **completion** | `completed` | **== 128** | program.md. Hard gate. |
| G6 | **identity** | greedy token-id vs plain greedy AR | report; near-identity expected | program.md greedy contract; strict bitwise compliance is the separate BI-verify lane (#192). |

**Measured-GO (build proceeds):** G1 `a1 ≥ 0.7731` **and** G2 `E[T] ≥ 3.9914`
(ideally ≈ 6.11) **and** G4 `ppl ≤ 2.42` **and** G5 `128/128`. The existing head is
deployable; proceed to the A.loopgraph deployment (or, if A.eager already cleared
G3 ≥ 500, ship the eager path) and de-risk any retrain.

---

## §5 — Failure branches

Decision tree on the measured outcome:

- **NO-GO — kills the {2,21,39} build:** G4 `ppl > 2.42` (quality contract broken) **or**
  G5 `completed < 128` (incomplete run) **or** G1 `a1` so far below 0.7731 that even
  the M=8 tree-salvage cannot lift effective acceptance to the E[T] floor (i.e. the
  measured head is fundamentally below bar on real hardware — the in-repo head is not
  the lever). A clean PPL break or a hard a1 collapse retires the existing-head lane.
- **Triggers Option B (full fusion retrain, Issue #319 B):** G1 `a1` is *close but
  below* 0.7731 (e.g. measured ≈ 0.74–0.77, the head is promising but the existing
  checkpoint isn't quite deployable) **or** G2 `E[T]` is above the 3.9914 floor but
  below the 6.11 sizing target — the head-quality is real but the existing K=1-trained
  artifact under-delivers; schedule a fresh {2,21,39} fusion drafter (HASS-style TTT /
  on-policy distillation) targeting `a1 ≥ 0.7731`, now with the served-path economics
  measured and confirmed.
- **Blocked (not a head verdict):** §0 vLLM-load smoke fails (shape/name mismatch) →
  the read never launches; fix the export or fall through to Option B's known-loadable
  retrain. Do **not** burn the HF launch on an unverified checkpoint.
- **Eager-only shortfall (NOT a NO-GO):** G1/G2/G4/G5 pass but A.eager `tps` lands in
  the ~402–487 eager band (< 500). This is *expected* on the inert-loopgraph substrate
  and **confirms the build, it does not kill it** — it means the deployed >500 needs
  the A.loopgraph rewrite (ubel #315 dispatch-cleared) and/or the strict-compliant
  BI-verify kernel (#192), exactly as modeled. Report the measured eager TPS against
  wirbel #314's eager-path break-even.

---

## §6 — Pre-launch checklist (each closure → the spec line it de-risks)

**Merged analytic closures (GREEN; named in PR #322 / Issue #319):**

| closure | W&B | what it proved | de-risks spec line |
|---|---|---|---|
| **#308 trainability** | `5axqa6oa` | in-repo head native a1=0.7714 ≈ 0.7731 bar (+0.0017, inside 0.0097 spread), salvage_net_positive; self-test 47/47 | **§4 G1** — the a1 the read confirms on a real load |
| **#310 PRIVATE-500** | `2u3kcnv5` | per-position model clears PRIVATE 500 @ 586.08 TPS (+17.2%) @ E[T]=6.11; #305's ×0.804 was a double-count; break-even ρ_priv=0.8038; self-test 8/8 | **§4 G2/G3** — the E[T]→private-TPS the GO target rests on |
| **#315 dispatch** (ubel) | — | #312 T6/T7 loopgraph rewrite does **not** reintroduce draft-side capture-dispatch risk (EAGLE KV-bearing chain dispatch-safe within ceiling-16 size-list; parent #311 axis-k CLOSED) | **§1 A.loopgraph** — clears the deployment-path rewrite |
| **#295 step-profile** | `c334qaqu` | measured draft-step cost validates 6.1245 (regime [5.36,6.86], central 6.11, ~2.95× linear); step axis (a) CLOSED | **§4 G2** — the E[T] sizing target & step denominator |
| **#299 VRAM** (+#306) | — / `y1lji0c6` | {2,21,39}-fusion fits 24 GB at 20.10 GB resident / 3.90 GiB headroom (runtime peak 20.158 GiB / 3.84 GiB) | **§3 VRAM** — the read is memory-feasible |
| **#317 boot-guard** (denken) | `bjtwr9jn` | `_guard_included_router` ported byte-identical into precache (128/128 identical, PPL byte-identical, +0.02%) | **§1 basis** — the served path won't boot-500 |
| step ceiling (#291) | — | free wall-clock step ceiling 487.7 < 500 → E[T]-raise is the only >500 lever | **HEADLINE / §4 G3** — why this read matters |
| frontier (#52) | `2x9fm2zx` | 481.53 official / PPL 2.3772 / 128/128; private 460.85 (Δ4.3%); target >500 | **§4 G3/G4** — the floors & target |

**In-flight YELLOW-closers (closing analytically THIS cycle, no GPU; named in PR #322 —
mapped from their stated scope, branches not inspected per launch isolation):**

| closer | scope (per PR #322 / Issue #319) | tightens spec line |
|---|---|---|
| **#314 eager-breakeven** (wirbel) | eager-path break-even E[T] for 481.53/500 vs the 6.11 bracket; prices the rewrite-skip gap | **§1 A.eager / §4 G3 / §5** — how to read the eager TPS floor |
| **#316 rank-coverage→TPS** (lawine) | max fusion `frac_true_beyond_top4` / min rank-coverage still clearing E[T]=6.11; the rank-coverage→TPS the GO decides on | **§4 G1/G2** — the salvage-coverage behind the a1 bar |
| **#318 deep-private-tax** (fern) | worst-case fusion lower bound on ρ_priv_e3 (the 0.9421 was modeled on the linear spine, not the fusion head) vs break-even 0.8038 | **§4 G2/G3** — robustness of the private-500 projection |
| **#320 step-regime** | tightens the #295 step-cost regime bracket ([5.36,6.86]→central) | **§4 G2** — the E[T] sizing/denominator confidence |
| **#321 realization** | whether the modeled ceiling REALIZES on the host-to-host wall (vs over-crediting, cf. #298) | **§3 / §4 G3** — model→measured TPS realization gap |

**Launch gates (ALL required before the #319 issue is approved & this ships):**
1. **§0 checkpoint published** (vLLM-loadable `Eagle3LlamaForCausalLM`, Hub/bucket
   path, `DRAFTER_SHA256` set).
2. **§0 local AWS A10G smoke PASS** — vLLM-load (0 missing/unexpected), `/v1/models`
   200, greedy token-identity, tiny PPL sane; `enforce_launch_gate` evidence written.
3. **Manifest delta applied** (§1 table) and the submission uploaded to
   `hf://buckets/gemma-challenge/gemma-senpai/submissions/senpai/fa2sw-precache-kenyan`.
4. **Human approval on Issue #319** (Option A pre-authorized).

---

## §7 — Exact launch command (copy-pasteable; DO NOT RUN until §6 gates 1–4 PASS)

```bash
cd /workspace/senpai/target
# (after §0 publish + §1 manifest delta + §0 smoke PASS + #319 human approval)
python train.py \
  --submission submissions/fa2sw_precache_kenyan \
  --method "ubel/eagle3-inrepo-251head-measured-read" \
  --launch --wait
```

`--method` is free-form W&B metadata (`train.py:67`); execution is driven entirely by
the manifest. `--launch --wait` → `scripts/run_hf_job.py:launch_job()` →
`POST https://gemma-challenge-gemma-bucket-sync.hf.space/v1/jobs:run` with
`agent_id="senpai"`, `submission_prefix="submissions/senpai/fa2sw-precache-kenyan"`,
`run_prefix="results/senpai/fa2sw-precache-kenyan-<UTCstamp>"`. The
`enforce_launch_gate("fa2sw-precache-kenyan")` pre-flight (`run_hf_job.py:35`) blocks
the API call unless the §0 smoke evidence is PASS. Single launch only.

On completion, harvest from `summary.json`: `tps`, `ppl`, `completed`, `output_tps`,
`total_tps`, latency, plus the offline-vs-served `a1`/`E[T]` comparison, then post the
structured result + a board message and decide GO / Option-B / NO-GO per §5.

---

## §8 — Spec self-test (`measured_read_spec_complete_self_test_passes`)

All conditions hold for this card → **1**:

1. All 6 required sections present (§1 config · §2 eval protocol · §3 cost · §4
   GO/NO-GO · §5 failure branches · §6 pre-launch checklist) — ✔
2. §1 names the exact served basis (`fa2sw_precache_kenyan` + merged #317 guard), the
   method/drafter flag delta (mtp→eagle3, model/DRAFTER_BUCKET/DRAFTER_SHA256,
   num_speculative_tokens=7, aux [2,21,39] in ckpt config), vLLM 0.22.1rc1 / a10g-small
   sm_86 — ✔
3. §2 states single-stream (MAX_NUM_SEQS=1 + MAX_CONCURRENCY=1), output_len 512, 128
   prompts, ppl≤2.42 + 128/128 — ✔
4. §3 states one a10g-small run, <40 min wall, benchmark ~130–200 s, VRAM fit — ✔
5. §4 has all five+1 thresholds with numbers (a1≥0.7731, E[T]≥3.9914/≈6.11, TPS
   floor 481.53 / GO 500.0, ppl≤2.42, 128/128, identity) — ✔
6. §5 defines NO-GO (kill) vs Option-B (retrain) vs eager-shortfall (not NO-GO) vs
   blocked — ✔
7. §6 maps every merged closure (#308/#310/#315/#295/#299 + #317 guard + step-ceiling
   + frontier) and every named YELLOW-closer (#314/#316/#318/#320/#321) to a spec
   line — ✔
8. §0 surfaces the critical checkpoint-publish + vLLM-load-smoke blocker; §7 gives the
   exact copy-pasteable launch command behind the DO-NOT-LAUNCH gate — ✔

`go_threshold_single_stream_tps = 500.0`.

---

## Public evidence used

- **Issue #319** (advisor, 8gpu launch): the GREEN-pending-build escalation and the
  Option A (measured read) / Option B (retrain) framing this card operationalizes.
- **fern #34 / arch_notes (PR #16, #34)**: the `gua9x68j` / `56ksyxgw` in-repo
  {2,21,39} head — architecture, the vLLM-loadable weight-name mapping (§7), and the
  "deployment gated on kanna #5" / deferred load verification that §0 closes.
- **EAGLE-3 integration-readiness card (PR #307, `88eh8twv`)**: the 6/3/0
  config/served/fork partition and the load-bearing T6/T7 onegraph-inert finding that
  makes A.eager the cheap read and defines the A.loopgraph rewrite.
- **Merged closures**: #308 (`5axqa6oa`), #310 (`2u3kcnv5`), #295 (`c334qaqu`), #299/#306
  (`y1lji0c6`), #315 (ubel dispatch), #317 (`bjtwr9jn`), #291 (free ceiling), #52
  (`2x9fm2zx` frontier) — see §6.
- **Public leaderboard** (read-only): no publicly documented EAGLE-3 served path exists
  for this model — the onegraph-loopgraph rewrite the deployed swap needs has no public
  precedent to borrow; the measured read is the cheapest way to price it.
