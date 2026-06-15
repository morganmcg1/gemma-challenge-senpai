STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["i1e5054m"],"primary_metric":{"name":"ppl_local_official_transfer_self_test_passes","value":1},"test_metric":{"name":"tau_ppl","value":1.000217619920667}}

## Results — τ_ppl, the third local→official transfer leg (PPL)

**`ppl_local_official_transfer_self_test_passes = True` (PRIMARY).** Analysis-only, **0 TPS, BASELINE stays 481.53**. No served-file change, no HF Job, no submission. Not a launch, not open2.

### Headline

| quantity | value | note |
|---|---|---|
| `local_ppl_int4_deployed` | **2.376682786480556** | fa2sw_precache_kenyan, **OFFICIAL corpus**, OFFICIAL `ppl_endpoint.py` |
| official PPL (anchor) | 2.3772 (leaderboard) / 2.3777 (private) | same corpus, same method |
| **`tau_ppl`** (multiplicative, stable) | **1.000217619920667** | `official = local · tau_ppl`; TEST metric |
| additive Δ (equivalent) | +0.000517 (lb) / +0.001017 (priv) | interchangeable with τ_ppl to 9.4e‑06 over [2.377, 2.42] |
| `tau_ppl_residual` | 0.000210 (mult.) ≈ 0.0005 PPL | = official's own leaderboard↔private spread |
| **`safe_local_ppl_bar`** | **2.4185** (2.41846) | worst-case transfer; a build here → official **exactly 2.42** |
| **`gate_meaningful_local_ppl_headroom`** | **0.04177** | **97.6 %** of the official 0.0428 headroom |
| transfer eats | 0.00103 PPL | the cost of crossing local→official |

**Hand-off (fern #287):** *The deployed int4 config measures local PPL 2.376683 (SAME corpus as the official 2.3772, locally mirrored — not a proxy), and the local→official PPL transfer is multiplicative `tau_ppl=1.000218` (±0.000210; additive Δ=+0.000517), so the safe LOCAL PPL bar for the official ≤2.42 gate is **2.4185** — giving fern #287's read-reduction Pareto **0.04177** of gate-meaningful local-PPL budget (97.6 % of the official 0.0428 headroom), and closing the third (PPL) transfer leg alongside τ_lo and τ_acc.* A read-reduction holding **local PPL ≤ 2.4185** is official-gate-safe (official PPL ≤ 2.42) under the measured τ_ppl + jitter.

### 1. Local PPL harness + reproduction (model-load = smoke test)

The hypothesis feared a corpus-proxy situation. **It does not arise:** the official PPL corpus (`ppl_ground_truth_tokens.jsonl`, 128 records / 61,797 target tokens) **and** the official PPL method (`ppl_endpoint.py`, micro-averaged `exp(Σ NLL / Σ tokens)`, `prompt_logprobs`, `add_special_tokens=false`) are **both locally mirrored**. So the local harness scores the **same corpus by the same method** as the official board — a same-corpus, same-method, cross-hardware reproduction, **not a proxy**.

Fresh re-serve of the deployed `fa2sw_precache_kenyan` config (cached server venv `vllm 0.22.1rc1.dev307`, the exact manifest wheel; full multimodal int4 body + bf16 lm_head + MTP drafter + ONEGRAPH) on the local A10G reproduced the anchor **bit-for-bit**:

```
ppl = 2.376682786480556   NLL = 53498.016889621984   128/128 records   61797 tokens
```

This is **identical to all 16 digits** across three independent runs (2026‑06‑13 21:10, 21:37 and the fresh 2026‑06‑15 03:43), so the int4 body is bit-exact (confirms kanna/my #276) and the local PPL carries **zero run-to-run variance**. The deployed manifest already sets all three PPL-headroom keys (`MAX_NUM_BATCHED_TOKENS=512`, `GPU_MEMORY_UTILIZATION=0.90`, `PYTORCH_CUDA_ALLOC_CONF`), so the harness injected **no** overrides — this is the config exactly as deployed.

The local 2.376683 lands **2.3772 − 2.376683 = +0.000517** below the official leaderboard anchor (0.022 %).

### 2. τ_ppl characterization (additive vs multiplicative; which is STABLE)

- **Multiplicative is the physically-stable form** and the one I report as `tau_ppl`. `PPL = exp(mean-NLL)`; a constant per-token mean-NLL shift (here +0.000218 nats/token) is a constant **multiplicative** PPL factor → regime-invariant (the PPL analog of the multiplicative hardware/clock ratio that made τ_lo stable). The additive Δ is reported alongside and is **numerically interchangeable** here: over the whole gate range [2.377, 2.42] the two models disagree by at most **9.4e‑06 PPL** (the offset is only ~0.02 % of PPL, unlike τ_lo's 3.5 % gap where the distinction mattered).
- **Offset telescoping — corpus / harness / body are structurally ZERO:**
  - corpus offset = 0 (official corpus mirrored locally; same 128/61797),
  - harness/tokenization offset = 0 (the local harness *is* `ppl_endpoint.py`),
  - body-numeric offset = 0 (int4 Marlin body bit-exact, #276; local PPL reproduces bit-for-bit).
  - The **entire** residual (+0.000517) is cross-hardware FP (local vs official A10G, on the bf16 lm_head/attention accumulation) + official-side measurement jitter.
- **The residual is jitter, not a systematic local↔official bias:** the local↔official offset (0.000517) is **1.03×** the official's *own* leaderboard↔private spread (2.3772 vs 2.3777 = 0.0005). I.e. local PPL predicts official PPL **as well as the official's own re-measurement does**. `tau_ppl_residual = 0.000210` (mult.) is the uncharacterized band, the PPL analog of my ±0.0075 λ̂ jitter in #276.

### 3. Safe local PPL bar (worst-case transfer)

Conservative (protect the gate): use the higher (private-verified) official anchor as the central worst case **plus** one more official-jitter band for re-measure uncertainty.

```
additive  : offset_worst = (2.3777 − 2.376683) + 0.0005 = 0.001517  → bar = 2.42 − 0.001517 = 2.41848
multiplic.: tau_worst    = 2.3777/2.376683 + 0.0005/2.376683 = 1.000638 → bar = 2.42 / tau_worst = 2.41846
safe_local_ppl_bar = min = 2.41846   (a build sitting on the bar maps to official PPL = 2.42 exactly under worst case)
gate_meaningful_local_ppl_headroom = 2.41846 − 2.376683 = 0.04177  (97.6 % of the official 0.0428)
```

The transfer eats only **0.00103 PPL** of headroom — because corpus + harness + body offsets are zero, local PPL screening retains essentially the full official budget.

### 4. Self-test (PRIMARY) — all green

(a) sane local PPL (matches 2.3772 same-corpus to <0.01 **and** `exp(NLL/tokens)` reconstructs the anchor, resid 0.0); (b) τ_ppl round-trips both forms (resid 0.0); (c) `safe_local_ppl_bar` < gate and maps to ≤2.42 under worst case, positive headroom; (d) NaN-clean; (e) constants imported EXACT (2.3772 / 2.3777 / 2.42 / 0.0428 / 481.53 / E[T]=3.844 / step=1218.2µs / K_cal=125.268 / τ_lo=1.03524 / τ_acc=1.0); (f) leg carries the 0‑TPS + corpus-proxy caveat + int4-body-bit-exact note. `passes = True`.

### Reproduce / provenance

```bash
# analytic core + self-test (PRIMARY, GPU-free), logs W&B:
cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  research/validity/ppl_local_official_transfer/ppl_local_official_transfer.py \
  --self-test --wandb_group ppl-local-official-transfer \
  --wandb_name lawine/ppl-local-official-transfer
# fresh GPU re-serve + reproduce local PPL (model-load smoke test):
cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  research/validity/ppl_local_official_transfer/ppl_local_official_transfer.py --measure
```

- **local PPL "summary.json" fields:** `ppl=2.376682786480556`, completed=**128/128** PPL records, `num_tokens=61797`, NLL=53498.016889621984, corpus=official `ppl_ground_truth_tokens.jsonl`. (No `tps` / `run_prefix`: **0 TPS, no HF Job** — analysis-only.)
- **Peak GPU memory:** ~19.4 GB observed (GPU_MEMORY_UTILIZATION=0.90 budget; KV cache 9.46 GiB), server freed the GPU cleanly after scoring. No OOM / NaN / failures.
- **W&B run:** `i1e5054m` (group `ppl-local-official-transfer`), `wandb-applied-ai-team/gemma-challenge-senpai`.
- Code + artifacts: `research/validity/ppl_local_official_transfer/` (`ppl_local_official_transfer.py`, `ppl_local_official_transfer_report.json`, `reserve-20260615T034300Z/ppl_summary.json`).

### Public evidence used

- **Leaderboard** (digest `as=senpai`): the deployed config sits in the fa2sw/splitkv/lmhead12k/precache family (frantic-penguin #1 `…skv64-fp-v1` 489.63 TPS; openevolve `splitkv-lmhead12k-precache-oe-v1` 487.87).
- **Result `20260615-022507-320_openevolve.md`** (job-backed, current-image): a sibling family config scored **official PPL 2.377421611888618** — a **third** official PPL anchor that lands **inside** my `[2.3772, 2.3777]` band. **Extends** this public evidence: I use the three independent official family PPLs (2.3772 / 2.3774 / 2.3777, range 0.0005) to ground the τ_ppl residual band — they confirm the family's official PPL clusters at 2.377 ± 0.0005, exactly the jitter I fold into the safe bar.
- vidraft-darwin's byte-diff posts (033518/033955) confirm the leaderboard's TPS best-of-N variance is the τ_lo/clock axis, not PPL — consistent with PPL carrying zero local run-to-run variance here.

### Honest framing

- **0 TPS.** This certifies that a LOCAL PPL screen is official-gate-meaningful; it does **not** produce a ≥500 build and does **not** change the served checkpoint. The launch gate stays land #245's MEASURED ≥500 at λ̂≥0.9780 **AND** PPL≤2.42 (human-approval-gated).
- **Corpus-proxy caveat (resolved):** the official corpus is locally available, so the safe bar is **not** proxy-conservative. Had a proxy been required, the bar would carry an extra corpus-mismatch margin; it does not.
- **int4 body bit-exact (#276):** the transfer is about harness/corpus/hardware, not body numerics — and harness+corpus are identical, leaving only ~0.0005 cross-hardware/measurement jitter.
- Cross-refs: my #267 (τ_lo TPS) + #276 (τ_acc λ̂) — this completes the three-leg framework; fern #287 (the read-reduction PPL Pareto this grounds); denken #283 (the HBM-bound ceiling whose read-reduction lever is PPL-gated); land #245 (the live build).

### Suggested follow-ups

1. **fern #287 imports `safe_local_ppl_bar = 2.4185`** directly as the gate-meaningful local-PPL ceiling for the read-reduction Pareto (0.04177 budget above the deployed 2.376683).
2. If any future read-reduction build serves on a **different corpus** for its local PPL screen (not the official mirror), re-introduce the corpus-mismatch term — but the framework already shows that screening on the official mirror is the tight path.
3. The official family PPL band (0.0005) is the residual's empirical floor; a 4th+ official family measurement would tighten `tau_ppl_residual` further, but the bar is already only ~0.001 PPL conservative.
