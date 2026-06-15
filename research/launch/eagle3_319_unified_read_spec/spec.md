<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# EAGLE-3 #319 unified single-approval read — per-depth private-α + ShareGPT coverage in ONE job

**PR:** #329 · **Author:** fern · **Date:** 2026-06-15 · **W&B group:** `eagle3-319-unified-read`

**0-GPU launch-prep card. 0 TPS added. BASELINE 481.53 UNCHANGED. THIS WRITES THE SPEC — IT DOES
NOT RUN IT.** No build, no HF Job, no training, no served-file change, no submission. A single
`a10g-small` launch of this read requires the human approval pending on **Issue #319**. Do not file
`/v1/jobs:run`, do not run `train.py --launch`. This card pre-registers the unified protocol so that
the moment #319 is approved the job ships with zero further design work — and **one approval flips
two merged verdicts**.

This mirrors the **ubel #322 measured-read-spec pattern** (`research/launch/eagle3_measured_read_spec.md`):
a spec doc + a CPU-only executable self-test, banked under
`research/launch/eagle3_319_unified_read_spec/{spec.md, eagle3_319_unified_read_spec.py, *_results.json}`.

---

## HEADLINE — one model-load, two read-only passes, two verdict-flips

The three EAGLE-3 integrators just merged converge on the **same two binding unknowns**, both
checkpoint-gated under Issue #319, and both are **read-only forward passes over the SAME trained
{2,21,39} fusion head** (`gua9x68j`, fern #34):

| flip | merged card | current state | the read that flips it | bar |
|---|---|---|---|---|
| **fern #325** (`xk1pghy4`, joint compliant-500 envelope) | YELLOW | central 520.95 clears (+4.19%, kernel-capped) but worst-case private tax floors it at **492.87 (−1.43%)** | **pass (a)** per-depth private-α — measures `ρ_priv_e3` directly AND supplies the a₁/deep split that unblocks the 0-GPU tree-credit | `ρ_priv_e3 ≥ 0.8038` (and `f_deep ≥ 0.9163`) |
| **lawine #323** (`ceddxj20`, coverage achievability) | MARGINAL | trained head straddles the unconditional-top-4 build bar (aime 0.9570 clears, aggregate 0.8903 misses) | **pass (b)** RANKPROBE_W=4 unconditional-top-4 read on the official 128 ShareGPT prompts | `uncond_top4 ≥ 0.9213` |

Because both passes load `gua9x68j` **once** (read-only scratch-copy; NO served-file change, NO
submission) and run on the **same `a10g-small` single-stream substrate**, bundling them into ONE
staged job **amortizes the single model-load** and yields **BOTH verdict-flips from a SINGLE human
approval** — maximizing the value of the one #319-gated spend. The single load is doubly efficient:
**pass (b)'s public ShareGPT read also supplies the public-side per-depth α** (`a₁`, `deep_pub`) that
pass (a)'s `ρ_priv_e3 = E[T]_priv / E[T]_pub` ratio needs in its denominator.

**Decisive joint thresholds (single-stream, read-only):**

| axis | measured quantity | flip bar | source |
|---|---|---|---|
| private tax | `ρ_priv_e3` (aggregate τ-ratio E[T]_priv/E[T]_pub) | **≥ 0.8038** | fern #325/#318/#310 break-even = 500/622.080888 |
| deep retention | `f_deep` (a₁ held by tree, deep-conditional retention) | **≥ 0.9163** | fern #318 break-even decomposition |
| coverage | `uncond_top4` (unconditional top-4 root coverage, ShareGPT) | **≥ 0.9213** | lawine #316/#304 build-uniform E[T]=6.11 target |
| quality | both passes greedy/PPL untouched (read-only logit/rank probes) | n/a (no emission change) | program.md / Issue #192 |

---

## §0 — CRITICAL PRE-LAUNCH BLOCKER: publish the `gua9x68j` checkpoint (shared by BOTH passes)

Identical to the ubel #322 §0 blocker, and **amortized**: both passes load the SAME head, so this
single publication+smoke unblocks both reads at once. The `gua9x68j` head (train run `56ksyxgw`,
warm-started from fern #25 `full_20k/model_best.pt`) is a **local plain-PyTorch artifact with no
Hub/bucket path** (arch_notes §7, "deployment gated on kanna #5"; vLLM-load verification deferred).
**Local `/workspace/...` paths are not visible on the HF runner.** Before the one launch:

1. **Locate** the `56ksyxgw` best checkpoint (the `gua9x68j`-evaluated weights).
2. **Package** as a vLLM-loadable `Eagle3LlamaForCausalLM` directory:
   - `config.json` with `architectures:["Eagle3LlamaForCausalLM"]`, `draft_vocab_size:262144`,
     `norm_before_fc:true`, `eagle_aux_hidden_state_layer_ids:[2,21,39]`, `target_hidden_size:2560`,
     `hidden_size:2560`, `num_hidden_layers:1`, `num_attention_heads:8`, `num_key_value_heads:2`,
     `head_dim:256`, `intermediate_size:10240`, `rms_norm_eps:1e-6`, `rope_theta:1e6`, identity
     `draft_id_to_target_id`.
   - `model.safetensors` with the arch_notes §7 name remap (`midlayer.*`→`model.layers.0.*`; bare
     `fc/embed_tokens/norm/input_norm`→`model.*`; `lm_head.*` as-is).
3. **Publish** to a writable repo the HF runner can reach — preferred:
   `hf://buckets/gemma-challenge/gemma-senpai/weights/eagle3-inrepo-251head/` (the senpai scratch
   bucket, same mechanism as the live `DRAFTER_BUCKET`); alt: a private Hub model repo. Do **not**
   point `MODEL_ID`/drafter at a local path.
4. **Compute** `sha256(model.safetensors)` → the new `DRAFTER_SHA256`.
5. **Local AWS A10G smoke (no HF Job):** boot `serve.py` with the EAGLE-3 manifest, confirm (a) vLLM
   loads the head as `Eagle3LlamaForCausalLM` with 0 missing/unexpected tensors, (b) `/v1/models`
   returns 200 (boot-guard active), (c) greedy decode on a handful of prompts is token-identical to
   plain greedy AR, (d) a tiny PPL spot-check is sane. Only then is the head launch-ready, and the
   `enforce_launch_gate` evidence file is PASS.

**If the local checkpoint cannot be made vLLM-loadable (shape/name mismatch the §7 best-effort
mapping missed), BOTH reads are BLOCKED and the path reverts to Option B (retrain into a
known-loadable export).** This single smoke gates both passes; it is the most likely launch-blocker.

---

## §1 — The single staged job (loads `gua9x68j` ONCE; two read-only passes) — *PR deliverable 1*

**Basis:** `submissions/fa2sw_precache_kenyan/` at the current branch HEAD (including the merged #317
`_guard_included_router` boot-guard), with the EAGLE-3 manifest delta from ubel #322 §1
(`method:"mtp"→"eagle3"`, `model`/`LOCAL_DRAFTER_DIR`/`DRAFTER_BUCKET`/`DRAFTER_SHA256`→ the published
`gua9x68j` head, `num_speculative_tokens:7`, aux `[2,21,39]` baked into the ckpt config). vLLM
`0.22.1rc1.dev307+g3e8afdf78` (stock upstream wheel, CUDA 12.9), `a10g-small` (sm_86). Model
`google/gemma-4-E4B-it`, all modalities loaded (the drafter swap touches only text speculative decode).

**Single-stream, single model-load, two read-only passes — staged in one process:**

```
load gua9x68j ONCE (read-only scratch-copy of submissions/fa2sw_precache_kenyan; served files byte-identical)
  ├─ PASS (b)  public ShareGPT  → uncond_top4  AND  public per-depth α (a₁_pub, deep_pub)   [RANKPROBE_W=4]
  └─ PASS (a)  private/OOD set   → private per-depth α (a₁_priv, deep_priv) → ρ_priv_e3, f_deep
```

### §1.2 — Pass (a): per-depth private-α table (flips fern #325)

On the **PRIVATE/OOD eval set**, read per-position acceptance `α_d` for `d = 1..K` (K=7 spine; root
`a₁` + deep spine), native on-path (not teacher-forced — measures native `ρ_priv_e3`; the tf→native
root gap is the banked 0.0097). Outputs:

- **`ρ_priv_e3`** = `E[T]_priv / E[T]_pub` — the aggregate acceptance-length τ-ratio, the exact
  quantity fern #318 could only *bound* at the worst-case 0.7923 and #310 *modeled* at 0.9421. The
  read **measures** it on the real fusion head, replacing the cross-DOMAIN literature bound.
- **the per-depth α vector** `[α_1, …, α_7]` (private), and
- **the a₁-vs-deep split**: `a₁_priv` (root) and `deep_priv` (the mean conditional over d=2..7), which
  — with the M=8 tree's organizer-verified `c₁=1.0` root recovery — yield the **incremental fusion
  deep retention** `f_deep = (deep_priv / deep_pub) / c_deep_lin` (`c_deep_lin = 0.97135`, lawine
  #300). This is the very per-depth table whose absence **blocks the 0-GPU tree-credit** (fern #325
  §4A: the most-generous analytic credit gives `implied_f_deep = 0.9083 < 0.9163`, missing by ~7 TPS).

The public-side denominator `E[T]_pub` (and `a₁_pub`, `deep_pub`) comes **free** from pass (b)'s
ShareGPT run on the same loaded head, so pass (a) only adds the private read.

### §1.3 — Pass (b): ShareGPT unconditional top-4 coverage (flips lawine #323)

On the **official 128 ShareGPT eval prompts** (`eval_prompts_sharegpt.json`, the deployment
distribution — free-form, NO MCQ-letter tokens), run the **wirbel #79 RANKPROBE_W=4** read-only
scratch-probe (the same one lawine #313 pre-registered), native on-path. At each draft position,
following the verifier's true greedy continuation, record the rank at which the true greedy token
appears in the drafter's top-4. Output:

- **`uncond_top4`** = unconditional top-4 root coverage = `a₁ + (1−a₁)·cov4_cond` (the salvage
  identity `c1_eff`), the regime-invariant build-bar quantity. fern #34's reasoning-holdout aggregate
  was 0.8903 (MISS) but free-form aime was 0.9570 (CLEAR); this read lands the **deployment-distribution**
  number, the binding unknown lawine #323 left open.

**No other served file is edited for either pass.** Both are read-only `logits.topk` / spec-decode
counter reads on the inert-loopgraph eager substrate; `align_bad=0` (wirbel #79) proves the emitted
draft chain is byte-identical to production, so greedy identity is preserved (§4).

---

## §2 — Exact harness (scripts, flags, eval sets, JSON schema) — *PR deliverable 2*

### Pass (b) — ShareGPT unconditional top-4 (`scripts/profiler/rank_coverage.py`)

The wirbel #79 probe, unchanged except the served head is now `gua9x68j` (via the §1 manifest delta):

```bash
python scripts/profiler/rank_coverage.py \
  --submission submissions/fa2sw_precache_kenyan \
  --num-prompts 128 --output-len 512 --seed 1 \
  --out-dir research/launch/eagle3_319_unified_read_spec/runs/<UTCstamp>/passb \
  --wandb-group eagle3-319-unified-read --wandb-name fern/unified-read-passb
#  serves a SCRATCH COPY (served files byte-identical), env set by the script:
#  RANKPROBE_ENABLE=1  RANKPROBE_W=4  LOOPGRAPH_WARMUP_CALLS=1e9  VLLM_USE_FLASHINFER_SAMPLER=0
#  client: decode_outputs over the 128 public eval_prompts_sharegpt.json, conc=1.
```

Reads `cov₂/cov₃/cov₄` and per-depth `q[d]`; `uncond_top4 = a₁ + (1−a₁)·cov4_cond`.

### Pass (a) — per-depth private-α (`scripts/profiler/accept_calibration.py`, run twice)

The PR #76 per-position acceptance reader, which reads vLLM's own
`vllm:spec_decode_num_accepted_tokens_per_pos{position=k}` counters (cumulative `C[k]`) and
`E[T] = 1 + num_accepted/num_drafts`. Run once per distribution on the **same loaded head**:

```bash
# private/OOD read (the load-bearing pass-(a) measurement):
python scripts/profiler/accept_calibration.py \
  --submission submissions/fa2sw_precache_kenyan \
  --prompts <PRIVATE_OOD_PROMPTS> --output-len 512 --seed 1 \
  --out-dir research/launch/eagle3_319_unified_read_spec/runs/<UTCstamp>/passa_priv \
  --wandb-group eagle3-319-unified-read --wandb-name fern/unified-read-passa-priv
# public read (the ρ denominator) is the SAME accept_calibration pass on eval_prompts_sharegpt.json,
# co-located with pass (b) on the single load; per-depth α_d = C[d]/C[d-1] (conditional).
```

- **Eval sets.** Pass (b) and the public side of pass (a): the **official 128 ShareGPT eval prompts**
  `eval_prompts_sharegpt.json` (the submission precaches them via
  `PRECACHE_DATASET=/harness/data/eval_prompts_sharegpt.json`). Pass (a) private side
  `<PRIVATE_OOD_PROMPTS>`: the **organizer's held-out private/OOD eval prompt set** (the same set
  behind the 460.85 private-verified TPS). If the organizer's private prompts are not exposed to the
  `a10g` runner, the **designated OOD proxy** is the benchmark-matched reasoning holdout
  (mmlu_pro/gpqa/aime, 240-rec) fern #34 was held-out on, native on-path — flagged in §5 as the one
  design choice the human should confirm at approval.
- **Per-depth α from cumulative counters.** `α_1 = C[1]`; `α_d = C[d]/C[d−1]` for `d ≥ 2`
  (conditional acceptance at depth d). `a₁ = α_1`; `deep = geomean(α_2..α_7)` or the survival-matched
  flat `deep` reproducing `E[T]` (kanna #289 survival sum, ubel #322 §2).

### Unified JSON output schema (every key named)

The staged job writes one `eagle3_319_unified_read_results.json` with:

```json
{
  "created_at": "<UTCstamp>", "run_prefix": "<HF-jobs run id>", "pr": 329, "agent": "fern",
  "head": {"wandb": "gua9x68j", "train_run": "56ksyxgw", "arch": "Eagle3LlamaForCausalLM",
           "aux_layers": [2, 21, 39], "drafter_sha256": "<sha256>"},
  "pass_a_private_alpha": {
    "eval_set": "<private_ood|ood_proxy_reasoning_holdout>", "n_prompts": 128, "output_len": 512,
    "alpha_per_depth_private": [a1, a2, a3, a4, a5, a6, a7],
    "alpha_per_depth_public":  [a1, a2, a3, a4, a5, a6, a7],
    "a1_private": 0.0, "a1_public": 0.0,
    "deep_private": 0.0, "deep_public": 0.0,
    "et_private": 0.0, "et_public": 0.0,
    "rho_priv_e3": 0.0,
    "f_deep": 0.0,
    "a1_vs_deep_split": {"a1_recovered_by_tree": true, "c1": 1.0, "c_deep_lin": 0.9713472759982902}
  },
  "pass_b_sharegpt_coverage": {
    "eval_set": "eval_prompts_sharegpt.json", "n_prompts": 128, "output_len": 512,
    "a1": 0.0, "cov2_cond": 0.0, "cov3_cond": 0.0, "cov4_cond": 0.0,
    "uncond_top4": 0.0, "align_bad": 0, "n_div": 0
  },
  "validity": {"ppl": 0.0, "completed": 128, "greedy_token_identity": true, "align_bad": 0,
               "vram_peak_gib": 0.0},
  "decision": {
    "rho_priv_e3": 0.0, "rho_breakeven": 0.8037539966988988, "pass_a_rho_flips": false,
    "f_deep": 0.0, "f_deep_breakeven": 0.9163111901482197, "pass_a_fdeep_flips": false,
    "fern325_flips_green": false,
    "uncond_top4": 0.0, "uncond_top4_bar": 0.9213011665456927, "lawine323_flips_go": false,
    "joint_verdict": "GREEN|YELLOW|RED"
  }
}
```

---

## §3 — GO/NO-GO decision rules + joint truth table — *PR deliverable 3*

### Pass (a) → fern #325 YELLOW → GREEN

`fern325_flips_green` iff **`ρ_priv_e3 ≥ 0.8038`** AND **`f_deep ≥ 0.9163`**. These are the **same
condition expressed two ways** — `ρ_priv_e3` is strictly monotone in `f_deep`, and by construction
`ρ_priv_e3(f_deep = 0.9163111901482197) = 0.8037539966988988` (break-even). The per-depth read
supplies BOTH: the aggregate τ-ratio `ρ_priv_e3` directly, and the a₁/deep split → `f_deep`. The
read replaces the worst-case literature bound 0.7923 (which sat −0.0115 under break-even, −7.13 TPS)
with the measured fusion number; the within-task analogue is 0.957 (~5× inside break-even), so the
measured fusion `ρ` very likely clears. **NO-GO branch:** `ρ_priv_e3 < 0.8038` (equivalently
`f_deep < 0.9163`) → the fusion head's deep positions overfit public worse than the linear spine;
#325 stays YELLOW (private-tax-bound) and the >500 compliant path needs a deep-spine retrain.

### Pass (b) → lawine #323 MARGINAL → GO

`lawine323_flips_go` iff **`uncond_top4 ≥ 0.9213`** on the official 128 ShareGPT prompts. The bar
`0.9213011665456927` is the build-uniform per-position target solving `1 + Σ_{j=1..7} T^j = 6.11`
(E[T]=6.11 free build target, denken #304/lawine #316) — **regime-invariant**: every conditional-frac
bar (0.2907 @ a₁=0.72925, 0.3468 @ 0.7731) maps to the SAME unconditional top-4 = 0.9213. **NO-GO
branch:** `uncond_top4 < 0.9213` → even the free-form deployment distribution lacks the root coverage
to build E[T]=6.11; coverage is the binding build gate and the head needs more root training / soft-KD
top-k calibration.

### Joint GREEN / YELLOW / RED truth table

A deployed compliant-500 EAGLE-3 build needs BOTH: the build must be **achievable** (coverage ≥ bar,
#323) AND the private projection must **survive** (private tax ≥ break-even, #325).

| pass (a) `ρ_priv_e3 ≥ 0.8038` | pass (b) `uncond_top4 ≥ 0.9213` | joint verdict | meaning |
|:---:|:---:|:---:|---|
| ✅ | ✅ | **🟢 GREEN — full GO** | #325 → GREEN (compliant-500 robust to the private tax) **and** #323 → GO (E[T]=6.11 build achievable). Proceed to the {2,21,39} deployment build. |
| ✅ | ❌ | **🟡 YELLOW — coverage-bound** | private tax survives, but the head can't reach E[T]=6.11 (coverage < bar). Build is the limiter → root-coverage retrain (soft-KD / more root training); #325's compliant ceiling is moot until #323 clears. |
| ❌ | ✅ | **🟡 YELLOW — private-tax-bound** | build is achievable, but the measured private tax sits under break-even → #325 stays YELLOW. Deep-spine retain / OOD-robust distillation needed before >500 is bankable. |
| ❌ | ❌ | **🔴 RED — NO-GO** | both gates fail: the existing head neither covers E[T]=6.11 nor survives the private tax → the in-repo `gua9x68j` head is not the lever; full fusion retrain (Issue #319 Option B). |

The two passes are **orthogonal axes** (private-tax robustness vs build-achievability), so all four
cells are reachable; the single job resolves both coordinates at once.

---

## §4 — greedy/PPL-untouched proof + VRAM budget + approval-issue text — *PR deliverable 4*

### Greedy/PPL-untouched proof

Both passes are **read-only logit/rank probes — no emission-path change, no sampling change, no
submission**:

- **Pass (b)** reads `_select_and_score` logits + `logits.topk` on a scratch copy with
  `LOOPGRAPH_WARMUP_CALLS=1e9` forcing eager `base_propose` so per-depth selection runs in Python; it
  only **ADDS** logging. wirbel #79 measured `align_bad = 0` over 16,524 records (100% byte-identity:
  unbiased `logits.topk` rank-1 == deployed fused argmax on every record) — the emitted draft chain is
  byte-identical to production, greedy identity preserved.
- **Pass (a)** reads vLLM's own per-step spec-decode counters
  (`vllm:spec_decode_num_accepted_tokens_per_pos`), which the V1 scheduler computes from the ACTUAL
  emitted tokens independent of the fused-accept kernel (PR #76). The only env override is
  `DISABLE_LOG_STATS=0` (re-register stat loggers): a handful of host-side counter increments per step
  (<0.1 ms, no GPU compute) that do NOT change which tokens are accepted or emitted.
- **No emission, sampler, or served-file change**, so PPL is byte-unaffected; greedy token-identity is
  the deployed path's, untouched. This card touches neither emission nor PPL (Issue #192 HARD gate is
  not engaged — this is a read, not a build).

### VRAM budget (must fit the 24 GB `a10g-small`)

The two passes load the SAME head once, so the resident footprint is the single-head footprint — no
additive VRAM from bundling. ubel #299: the {2,21,39}-fusion drafter + hidden-state retention fits the
24 GB lane at **20.10 GB resident / 3.90 GiB headroom** (at rest); ubel #306 runtime **peak 20.158
GiB / 3.84 GiB headroom**. extra_kv dominates (0.719 GiB); drafter weights negligible (0.037 GiB);
net +0.80 GB over the deployed stack. Memory is **not** the constraint, and the read-only probes
stream records to disk with negligible host/GPU overhead. **Peak 20.158 GiB < 24 GiB; 3.84 GiB
headroom.**

### Exact `Approval request` issue body text (file this to launch)

> **Title:** `Approval request: HF job for eagle3-319-unified-read`
>
> **PR / branch:** #329 / `fern/eagle3-319-unified-read-spec` (targets `approval-gated-8gpu-20260613`).
>
> **What:** ONE `a10g-small` org-credit HF Job that loads the trained {2,21,39} fusion EAGLE-3 head
> `gua9x68j` (fern #34, published vLLM-loadable per §0) **once** and runs **two read-only eval passes**
> on the single load:
> - **(a)** per-depth private-α on the private/OOD eval set → `ρ_priv_e3`, per-depth α vector, a₁/deep
>   split (`f_deep`). Flips **fern #325** YELLOW→GREEN iff `ρ_priv_e3 ≥ 0.8038` and `f_deep ≥ 0.9163`.
> - **(b)** RANKPROBE_W=4 unconditional-top-4 coverage on the official 128 ShareGPT prompts →
>   `uncond_top4`. Flips **lawine #323** MARGINAL→GO iff `uncond_top4 ≥ 0.9213`.
>
> **Why one job:** both passes are read-only forward passes over the SAME head; bundling amortizes the
> single model-load and pass (b)'s public read supplies pass (a)'s ρ-denominator → **two merged-verdict
> flips from one approval**.
>
> **Greedy/PPL risk:** NONE. Both are read-only logit/rank probes (no emission/sampler/served-file
> change, no submission); wirbel #79 measured `align_bad=0` (byte-identical draft chain). PPL/greedy
> identity = the deployed path's, untouched. BASELINE 481.53 unchanged, 0 TPS added.
>
> **VRAM:** 20.158 GiB peak / 3.84 GiB headroom on the 24 GB lane (ubel #299/#306). Not the constraint.
>
> **Runtime / quota:** exactly ONE `a10g-small` job, well under the 40-min (2400 s) wall (§5);
> single-stream, two passes on one load. Launch exactly once.
>
> **Pre-launch gates done:** §0 checkpoint published vLLM-loadable + local A10G smoke PASS (0
> missing/unexpected, `/v1/models` 200, greedy-identical, PPL sane); §1 manifest delta applied;
> submission uploaded. Requesting human approval to fire the single job.

---

## §5 — Cost (one job, both passes) + failure branches

### Cost

- **Quota:** exactly **one** `a10g-small` org-credit HF Job (`/v1/jobs:run`). Launch exactly once; if
  §0 smoke or a transient error raises doubt, report back — do not retry speculatively.
- **Wall-clock:** `a10g-small` hard limit **40 min (2400 s)**. Budget: startup (target weights +
  `gua9x68j` head sync, lm_head `[262144,2560]` ≈ 1.34 GB bf16 dominates) ≈ 5–12 min; pass (b) over
  128×512 ShareGPT ≈ 130–200 s; pass (a) private read (per-depth counters, same output_len) ≈ 130–200
  s; both share the single load. **Total comfortably < 40 min**, single-stream, one job. (Reference:
  the 481.53 frontier benchmark phase ≈ 136 s.)
- **VRAM:** §4 — 20.158 GiB peak / 3.84 GiB headroom; both passes on one load add no resident VRAM.

### Failure branches

- **NO-GO (kills the existing-head lane):** both gates fail (`ρ_priv_e3 < 0.8038` AND `uncond_top4 <
  0.9213`) → RED; `gua9x68j` is not the lever → Issue #319 Option B (full fusion retrain).
- **Coverage-bound YELLOW:** pass (a) flips but pass (b) misses → root-coverage retrain (soft-KD /
  more root training) before #325's ceiling matters.
- **Private-tax-bound YELLOW:** pass (b) flips but pass (a) misses → deep-spine OOD-robust distillation
  before >500 is bankable.
- **Blocked (not a verdict):** §0 vLLM-load smoke fails (shape/name mismatch) → neither read launches;
  fix the export or fall through to Option B. Do **not** burn the HF launch on an unverified checkpoint.
- **Private-set ambiguity (design-choice flag):** if the organizer's private prompts are not exposed
  to the runner, pass (a) runs on the OOD reasoning-holdout proxy (§2); the human should confirm the
  intended private set at approval. The proxy is the conservative corner (reasoning holdout is more OOD
  to ShareGPT than the challenge's same-task held-out set), so a clearing `ρ_priv_e3` on the proxy is a
  fortiori a clear on the real private set.

---

## §6 — Pre-launch checklist (each closure → the spec line it de-risks)

**Merged analytic closures (GREEN; the two verdicts this read flips + their inputs):**

| closure | W&B | what it proved | de-risks spec line |
|---|---|---|---|
| **fern #325** (joint envelope) | `xk1pghy4` | compliant-500 YELLOW: central 520.95 (capped), worst 492.87 (−1.43%); the #319 per-depth read is the single cheapest YELLOW→GREEN flip | **HEADLINE / §3 pass (a)** — the verdict this read flips |
| **lawine #323** (coverage achievability) | `ceddxj20` | MARGINAL: fern #34 straddles the 0.9213 bar (aime 0.9570 clears, aggregate 0.8903 misses); the RANKPROBE_W=4 ShareGPT read is the single flip | **HEADLINE / §3 pass (b)** — the verdict this read flips |
| **fern #318** (deep-private-tax) | `xe8ff7hq` | `ρ_priv_e3` break-even 0.8038, `f_deep` break-even 0.9163, worst-case 0.7923, implied_f_deep 0.9083 | **§3 pass (a) bars** — the thresholds the read clears |
| **fern #310** (private per-position) | `2u3kcnv5` | honest_public(6.11)=622.08, break-even ρ=0.8038, private central 586.08; ×0.804 double-count settled | **§3 / HEADLINE** — the ρ→TPS mapping |
| **denken #308** (a₁-cliff) | `5axqa6oa` | in-repo head native a₁=0.7714 ≈ bar; native≈tf root (gap 0.0097) | **§1.2 pass (a)** — native-vs-tf justification |
| **fern #34** (the head) | `gua9x68j` | the ONLY trained {2,21,39} fusion head; tf top1/4=0.7617/0.8903, aime top4 0.9570; train `56ksyxgw` | **§0 / §1** — the head both passes load |
| **wirbel #79** (RANKPROBE) | `z6wi4z4v` | RANKPROBE_W=4 read-only scratch-probe, `align_bad=0` (byte-identity); cov₄ 0.6532 on linear | **§1.3 / §2 / §4** — the pass-(b) harness + greedy proof |
| **ubel #299/#306** (VRAM) | — / `y1lji0c6` | {2,21,39}-fusion fits 24 GB: 20.10 GB resident / 3.90 GiB headroom; runtime peak 20.158 GiB / 3.84 GiB | **§4 VRAM** — the read is memory-feasible |
| **ubel #322** (measured-read-spec) | — | the 0-GPU launch-spec pattern + the §0 checkpoint-publish blocker this card mirrors and shares | **whole card** — the pattern |

**Launch gates (ALL required before the #319 issue is approved & this ships):**
1. **§0 checkpoint published** (vLLM-loadable `Eagle3LlamaForCausalLM`, Hub/bucket path,
   `DRAFTER_SHA256` set) — shared by both passes.
2. **§0 local AWS A10G smoke PASS** — vLLM-load (0 missing/unexpected), `/v1/models` 200,
   greedy token-identity, tiny PPL sane; `enforce_launch_gate` evidence written.
3. **Manifest delta applied** (ubel #322 §1 table) and the submission uploaded to
   `hf://buckets/gemma-challenge/gemma-senpai/submissions/senpai/fa2sw-precache-kenyan`.
4. **Human approval on Issue #319** via the §4 `Approval request: HF job for eagle3-319-unified-read`
   issue (single job, both passes).

---

## §7 — Exact launch command (copy-pasteable; DO NOT RUN until §6 gates 1–4 PASS)

```bash
cd /workspace/senpai/target
# (after §0 publish + manifest delta + §0 smoke PASS + #319 human approval)
python train.py \
  --submission submissions/fa2sw_precache_kenyan \
  --method "fern/eagle3-319-unified-read" \
  --launch --wait
```

`--method` is free-form W&B metadata (`train.py:67`); execution is driven by the manifest + the staged
probe scripts (§2). `--launch --wait` → `scripts/run_hf_job.py:launch_job()` → `POST /v1/jobs:run`
(`agent_id="senpai"`, `submission_prefix="submissions/senpai/fa2sw-precache-kenyan"`,
`run_prefix="results/senpai/eagle3-319-unified-read-<UTCstamp>"`). The
`enforce_launch_gate("fa2sw-precache-kenyan")` pre-flight blocks the API call unless the §0 smoke
evidence is PASS. Single launch only; on completion harvest both passes into the §2 JSON schema and
decide per the §3 truth table.

---

## §8 — Spec self-test (`unified_read_spec_self_test_passes`)

The companion `eagle3_319_unified_read_spec.py` is CPU-only (0 GPU) and does **two** things:

1. **Decision-arithmetic reproduction (the load-bearing half, PR deliverable 5):** independently
   recomputes the merged #325/#323 thresholds from first principles + banked constants and verifies
   each to **≤ 1e-6**:
   - `ρ_breakeven = 500 / 622.080888 = 0.8037539966988988` (fern #310).
   - `f_deep_breakeven = 0.9163111901482197` via bisection on `ρ(f_deep) = ρ_breakeven`, where
     `ρ(f_deep) = E[T](a₁=0.72925, deep=0.91443·0.97135·f_deep) / 4.966` (fern #318 survival sum).
   - round-trips: `ρ(f_deep_breakeven) = ρ_breakeven`; `ρ(implied_f_deep_worst=0.9083) = ρ_worst =
     0.7923` (the 0-GPU credit cannot flip).
   - `uncond_top4 bar = 0.9213011665456927` **reconstructed** by solving `1 + Σ_{j=1..7} T^j = 6.11`
     (the build-uniform E[T] target) — a genuine derivation, not just an import — plus the
     conditional-frac bar `(1−T)/(1−a₁) = 0.2907` @ deployed a₁ and the regime-invariance identity
     `a₁ + (1−a₁)(1 − (1−T)/(1−a₁)) = T` for every a₁.
   - the joint truth table: evaluates all four ✅/❌ corners and asserts the GREEN/YELLOW/RED labels.
2. **Spec completeness (mirrors ubel #322):** machine-verifies every required section (§0–§7) and
   every load-bearing number/flag/script is present in this file.

`unified_read_spec_self_test_passes = 1` iff all arithmetic reproductions land ≤ 1e-6 AND all
completeness conditions hold AND the result is NaN-clean.

`rho_breakeven = 0.8037539966988988` · `f_deep_breakeven = 0.9163111901482197` ·
`uncond_top4_bar = 0.9213011665456927`.

---

## Public evidence used

- **Issue #319** (advisor, 8gpu launch): the single human-approval the unified read is scoped to; this
  card pre-registers the protocol so one approval flips both #325 and #323.
- **fern #325** (`xk1pghy4`, joint compliant-500 envelope, MERGED #325): the YELLOW verdict + its §4
  finding that the #319 per-depth read is the single cheapest YELLOW→GREEN flip (the 0-GPU credit
  cannot flip alone — `implied_f_deep 0.9083 < 0.9163`). This card operationalizes pass (a).
- **lawine #323** (`ceddxj20`, coverage achievability, MERGED #323): the MARGINAL verdict + its
  "what flips it" = the wirbel #79 RANKPROBE_W=4 read on the official 128 ShareGPT prompts. This card
  operationalizes pass (b).
- **fern #318** (`xe8ff7hq`), **fern #310** (`2u3kcnv5`), **lawine #316/#304**: the break-even
  thresholds (ρ 0.8038, f_deep 0.9163, uncond_top4 0.9213) the read clears, reproduced in §8.
- **fern #34** (`gua9x68j` / train `56ksyxgw`): the trained {2,21,39} head both passes load, and its
  arch_notes §7 vLLM-loadable mapping / deferred-load blocker §0 closes.
- **wirbel #79** (`z6wi4z4v`): the RANKPROBE_W=4 read-only scratch-probe (`align_bad=0` greedy-identity
  proof) — the pass-(b) harness and the §4 greedy proof.
- **ubel #299/#306** (`y1lji0c6`): the 24 GB VRAM fit (20.158 GiB peak / 3.84 GiB headroom).
- **ubel #322** (`research/launch/eagle3_measured_read_spec.md`): the 0-GPU measured-read-spec pattern
  this card mirrors, and the shared §0 checkpoint-publish blocker.
- **Public leaderboard** (read-only): no publicly documented EAGLE-3 served path exists for this model;
  the per-depth private-α table and the deployment-distribution top-4 coverage are both unmeasured —
  the unified read is the cheapest way to resolve both binding unknowns.
