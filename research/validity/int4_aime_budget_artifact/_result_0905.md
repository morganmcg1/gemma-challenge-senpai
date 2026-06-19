STUDENT kanna:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["lzbqp28p","jqecrucm"],"primary_metric":{"name":"aime_int4_pct_of_base_at_12288","value":0.7537},"test_metric":{"name":"aime_truncation_rate_delta_int4_vs_base","value":0.0333}}

## VERDICT: `AIME_REAL_PRECISION_LOSS` — the int4 AIME gap is intrinsic precision damage, not an eval-budget artifact

Greedy-basis budget axis per your 05:00Z Option-A directive. W&B **[`lzbqp28p`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/lzbqp28p)** (group `int4-aime-budget-artifact-kanna`, `analysis_only=1`/`official_tps=0`/`no_hf_job=1`/`fires=0`).

**Bottom line:** the int4/base greedy accuracy ratio is **budget-FLAT at ~0.75**, and the 0.117 gap is mechanically budget-immune — int4 mostly emits *finished, wrong* answers, not truncated chains. The budget lever is dead; the recipe/precision axis (ubel #702) is the load-bearing fix.

### Correction to my 08:24Z liveness — the cc=1 fix hypothesis is REFUTED
At 08:24Z I root-caused the non-reconciliation to a cc=16 batch-width artifact and predicted cc=1 (batch width 1, the banked int4 config) would reconcile. **The cc=1 smoke landed and refuted that.** On `.venvs/vllm022`, int4 greedy @6144 at **cc=1, eager, BI=1** still derails: `2024-II-4` (gold 33 — the banked engine solves it coherently → 33 ✓) comes back as **repetition gibberish → no extractable answer**; `2024-II-12` → wrong. So the divergence is **not** batch-width — int4 greedy corrupts on the substitute even with no batch to corrupt.

### The literal local greedy 2×2 is UNMEASURABLE on the surviving engine
The banked anchors were produced on `/tmp/vllm0220-srv` (gone). The only surviving engine is `.venvs/vllm022`. Across **every** config tried, neither body reconciles:

| body | engine/config | acc@6144 greedy | banked anchor |
|---|---|---|---|
| base bf16 | `.venvs/vllm022` compile-ON cc16 | **0.1333** | 0.4667 |
| int4 | `.venvs/vllm022` BI=1 eager cc16 | **0.0667** | 0.350 |
| int4 | `.venvs/vllm022` BI=0 compile cc16 | 0.10 | 0.350 |
| int4 | `.venvs/vllm022` eager **cc=1** | **~0** (gibberish) | 0.350 |
| int4 | `.venvs/vllm022` compile cc16 @**12288** | 0.10 | — |

Key infra finding: **"vLLM 0.22.0" is not one numeric substrate.** The banked-anchor 0.22.0 (`/tmp/vllm0220-srv`, gone) and the surviving 0.22.0 (`.venvs/vllm022`, torch2.11/cu130) **diverge on greedy AIME** — both bodies collapse on the survivor (int4 worst, into `qlql…`/`umiv…` repetition that runs to the cap). This is the greedy-side analogue of the sampled degeneration I banked at 04:52Z (`jqecrucm`). It means your 05:00Z premise "greedy is clean on 0.22.0" holds for the *gone* 0.22.0, not the survivor — this pod currently has **no** clean greedy int4 AIME engine, only banked artifacts. A naive survivor measurement could falsely return `AIME_BUDGET_ARTIFACT` (int4's gibberish-to-cap reads as truncation against a banked base), so I discard the survivor's cells and take the verdict from the clean banked anchors.

### The decisive evidence (clean banked greedy data — no survivor numbers)
The verdict doesn't need greedy@12288: the banked greedy@6144 data already carries the mechanism. Decomposing the **9-problem gap** (base-right & int4-wrong):

| gap bucket | n | budget can help? |
|---|---|---|
| int4 emitted a **wrong answer with natural EOS** | **6** | **No — generation already finished** |
| int4 **truncated** (hit length cap) | 3 | only in principle |

- **67% of the gap (6/9) is budget-immune by construction** — the model finished and was wrong; more tokens cannot change a completed generation.
- The 3 truncated gap problems run to a **median 12,860 chars ≈ 2.9× int4's own correct-solve median (4,433)** — the degenerate-loop signature, not "almost done, needs a few more tokens."
- **Absolute ceiling:** even if all 3 truncated gap problems flipped to correct, int4 → 0.40, ratio → **0.857 < 0.90**. Budget *cannot* lift int4/base across the artifact bar.

**Two budget-flat ratios bracket the axis** (you flagged these at 05:00Z):

| budget | basis | int4/base | source |
|---|---|---|---|
| 6144 | greedy | **0.750** | banked anchors (21/60 vs 28/60) |
| 12288 | sampled | **0.754** | lawine #693 (`6brpvz9x`) |

Δratio(12288−6144) = **+0.004** — flat. Doubling the budget does not move the ratio.

**Why REAL and not PARTIAL:** the *theoretical* budget-attributable ceiling is ≤3/9 of the gap, but the *realized* effect is ~0 — lawine's empirical 12288 doubling moved the ratio +0.004, and the 3 candidate problems are degenerate loops, not productive chains. So the budget lever is effectively dead → REAL, not PARTIAL.

### Metrics (PR-required)
- **Primary `aime_int4_pct_of_base_at_12288` = 0.754** — nearest 12288 datum (lawine sampled; greedy@12288 unmeasurable on any *valid* engine, but the budget-flat structure ⇒ greedy@12288 ≈ 0.75). vs the BUDGET_ARTIFACT bar 0.90: **fails by 0.15**.
- **Test `aime_truncation_rate_delta_int4_vs_base` = +0.033** (banked greedy@6144: int4 trunc 0.167 / base 0.133) — int4 caps on just 2 more of 60 problems; far too small to source an 11.7-pt gap.
- Per-cell (banked, clean): int4 **0.350** (21/60, trunc 0.167) / base **0.4667** (28/60, trunc 0.133). Test reference: lawine #693 sampled anchor int4 **0.3467** / base 0.4600 (greedy≈sampled at the int4-body level, as you noted).

### Reconciliation / command
Greedy protocol = banked anchor verbatim (k=1, T=0, top_p=1, top_k=−1, min_tokens=8, `--no-thinking`, seed=1234, years=2024,2025-I,2025-II, n=60, BI=1):
```
python research/downstream_quality_aime/aime_eval.py \
  --base-url http://127.0.0.1:8000 --model gemma-4-e4b-it \
  --years 2024,2025-I,2025-II --k 1 --seed 1234 \
  --temperature 0.0 --top-p 1.0 --top-k -1 \
  --max-tokens {6144|12288} --min-tokens 8 --no-thinking \
  --client-concurrency 1 --save-text --out <body>_greedy_<budget>.json
```
Served via `serve_body.sh` (`.venvs/vllm022`, BI=1, MML=13312, MNS=16, enforce-eager). **The 6144 cells do not reconcile on the survivor** (table above) — which is exactly why the literal 2×2 is unmeasurable and the verdict comes from banked data.

- **W&B:** `lzbqp28p` (verdict) + `jqecrucm` (04:52Z sampled-degeneration diagnostic).
- **Peak mem:** 19.1 GB (int4 @12288, 16-way) on the A10G.
- **No HF job, no fire, served file untouched.**

### What happened — honest analysis
I couldn't produce the literal greedy@12288 cell because the only surviving engine corrupts greedy AIME decode for both bodies (int4 into repetition gibberish, even at cc=1 — refuting my own 08:24Z batch-width theory). But the clean banked greedy@6144 data is *more* decisive than the 12288 ratio would have been: it exposes the **mechanism**. The int4 gap is dominated by **confident wrong answers** (6/9 gap problems finish with EOS), not truncated reasoning; only 3/9 truncate, and those are degenerate loops (~2.9× the length of int4's successful solves), so even the absolute budget ceiling (0.857) can't reach 0.90. With the two budget-flat ratios (0.750 greedy@6144 ≈ 0.754 sampled@12288), the budget lever is dead → **`AIME_REAL_PRECISION_LOSS`**.

This composes with lawine #693 (sampling axis closed: compliant ≤ greedy) into a **clean two-axis closure** — the int4 AIME deficit has no eval-protocol escape hatch (neither sampling nor budget rescues it). What must fix it is the recipe/precision axis: ubel #702's selective-grid recovery (`nqk9izab`), the load-bearing sibling.

### Suggested follow-ups (not implemented — flagging per process)
1. **Literal greedy@12288 cell:** if you want the actual number rather than the budget-flat inference, point me at a greedy int4 engine that reconciles both anchors at 6144 (base→0.4667 **and** int4→0.350); I'll run the 2×2 in one window. No such engine survives on this pod (dev307 invalid per #606; `/tmp/vllm0220-srv` gone; `.venvs/vllm022` corrupts greedy).
2. **base@cc=1 control (~50 min):** I have base collapsing at cc=16 but no base@cc=1 greedy point; running it would isolate whether the survivor's greedy corruption is int4-specific or engine-wide-at-batch — useful for the engine bug report, not needed for this verdict.
3. **Engine bug (standalone infra issue):** the survivor `.venvs/vllm022` corrupts *batched greedy AIME* for the bf16 base too (0.4667→0.1333 at cc16). Worth banking separately — it means there's currently no clean greedy AIME engine on this pod.
