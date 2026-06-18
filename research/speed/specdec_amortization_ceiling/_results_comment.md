STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"fires":false,"wandb_run_ids":["hj2afh4j"],"verdict":"SPECDEC_CEILING_CLEARS_10_INSCOPE","primary_metric":{"name":"specdec_tps_ceiling_official_equiv","value":150.29},"test_metric":{"name":"e_accept_star_for_plus10","value":3.343}}

## Results — Spec-dec amortization ceiling (capstone of the #674/#676 line)

**Verdict: `SPECDEC_CEILING_CLEARS_10_INSCOPE`** — a publishable-drafter acceptance clears +10 at the optimal K (top_k64 `E=3.38 → 137.14` official-equiv `> 136.378`), **pending the strict-#319 rescue → flag for #481**. But the in-scope clear is **razor-thin (+0.76 official ≈ +0.55%)**; `E*=3.343` sits only 0.037 below top_k64 and **above** stock (3.33, which misses by 0.26). The body-read is *not* the binding constraint — the strict-#319 **rescue tax** is.

### 1. The spec-dec speed law

`TPS(E,K) = E[accept] / T_step(K)`, where the verify forward reads the body+head weights **once per step** (M-invariant) and spec-dec amortizes that read over `E[accept]` accepted tokens. Anchored to my #676 measurement: **2.38 GB/token** (2.034 body + 0.346 lm_head), the dominant 86% of per-token bytes.

### 2. `K_max_useful ≈ 52` — the verify forward is free-in-width for every realistic K

The verify forward stays **weight-read (memory) bound** while the int4-weight arithmetic intensity `4·M` is below the sm_86 ridge `AI = 125 TFLOP / 600 GB/s = 208.3` → **M_knee = 208.3/4 = 52** (`K_max_useful ≈ 51`). Sensitivity band: 29 (FP32-accum tensor) … 60 (achievable-BW AI).

**Empirically confirmed** by a local M-sweep of the self-built strict int4-Marlin verify shapes (#676 only ever measured M=1). Verify-body `us(M)/us(1)`:

| M | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|---|
| ratio | 1.000 | 0.997 | 1.010 | 1.009 | 1.016 | **1.046** | 1.238 | **1.468** | 2.154 |

Flat (±5%) to **M=32**, knee at **M=48–64** — matching the roofline. Per-shape: the dominant `gate_up`/`down`/`lm_head` GEMMs are M-flat then go compute-bound at M=48–64 (`lm_head` 676→748us at M=32, 1325us at M=64); the small `qkv`/`o` GEMMs are dead-flat (latency-bound). **Realistic spec K=3–7 → verify width M=4–8 sits deep in the flat region** → the verify is free-in-width; **K is limited by *draft cost*, not verify width.**

### 3. `T_step(K)` cost model (calibrated to the committed strict batch-invariant walltps K-sweep)

`int4_mtp_batchinv` (`VLLM_BATCH_INVARIANT=1` + recompute-rescue), qat drafter, `T_step = E/localTPS`:

| K | E[accept] | local TPS | ×0.870 official-equiv | clears +10? |
|---|---|---|---|---|
| 3 | 2.856 | 165.71 | 144.17 | ✅ |
| 4 | 3.204 | 171.68 | 149.36 | ✅ |
| **5** | **3.474** | **172.74** | **150.29** | ✅ **(optimal)** |
| 6 | 3.657 | 170.18 | **148.06** | ✅ |
| 7 | 3.825 | 152.26 | 132.47 | ❌ |

`T_step(K) = 12.98ms + 1.421ms/draft` (linear K=3–6; K=7 hits a vLLM M=8 CUDA-graph bucket jump, not the roofline knee). **Optimal K=5.** The strict base `T0=12.98ms` carries a **+4.84ms batch-invariant + rescue tax** over the deployed AR step (8.14ms, #676 d674) — a ~1.6× per-step inflation. **K=6 → 148.06 cross-checks the PR's "qat 3.66 → 148" exactly**, validating the ×0.870 basis end-to-end.

### 4. The body-read PERMITS far more than +10 — the rescue tax is the wall

- **RAW body-read ceiling (rescue-free, instruction-1 law):** `E=3.38 / (2.38GB read + cheap draft)` → **~580 official-equiv = 4.25× the +10 bar** (matches the public ~481–508 spec-dec frontier). The body-read law says spec-dec *could* go ≫ +10.
- **STRICT realized ceiling (rescue tax):** collapses the raw ~580 to **137–150**. The +10 bar (136.378) sits **right at** the collapsed strict ceiling.
  - **OOS qat** (3.66, gated on a human publish decision): clears comfortably, max **150.29 @ K=5**.
  - **In-scope top_k64** (3.38, publishable): **137.14 → clears +10 by +0.76.**
  - **In-scope stock** (3.33, publishable): **136.12 → misses by 0.26.**

### 5. `E[accept]*` for +10 and the verdict

- `e_accept_star_for_plus10 = 3.343` (realized in-scope envelope) — between stock (3.33, misses) and top_k64 (3.38, clears).
- `e_accept_star_optimalK_bestcase = 3.153` (if an in-scope drafter could match the qat draft cost at K=5 — then even stock clears).
- **Cheapest publishable drafter that clears +10 at the optimal K: top_k64 (3.38).** → `CLEARS_10_INSCOPE`.

**Fragility (honest):** the in-scope clear is +0.76 official (~0.55%, inside the ~1% projection/measurement band) and depends on (a) the rescue being **near-lossless** — land #670's `rescued-equiv` assumes the stark #669 recompute-rescue restores #319 identity at ~0 throughput cost beyond the measured walltps; and (b) the **conservative ×0.870** projection. Both pull the right way for a "clears" claim: the deployed projection is ×1.06 and the AR anchor ×1.029 (126.378/122.87) — under either, in-scope clears by more. So `CLEARS_10_INSCOPE` is robust to the projection but **marginal on the drafter** (needs ≥ top_k64, not stock) and **contingent on the rescue tax staying near-lossless**.

### What this bounds (the advisor's question)

> *"how much the stark #669 recompute-rescue and the land lossless-verify-GEMM unlocks are actually worth"*

The gap between the raw body-read ceiling (~580) and the strict realized envelope (137–150) **is** the rescue tax. The land lossless-verify-GEMM unlock moves you *up* within that band; every TPS it recovers is a TPS off the +4.84ms strict base tax. At today's strict envelope the in-scope margin over +10 is only +0.76, so the unlock's value is **decisive for in-scope robustness**: it's the difference between "top_k64-only, razor-thin" and "stock clears comfortably."

### Command
```bash
cd target/ && CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
  research/speed/specdec_amortization_ceiling/specdec_amortization_ceiling.py \
  --self-test --wandb_name denken/specdec-amortization-ceiling \
  --wandb_group specdec-amortization-ceiling-denken
```
- **Self-test: 14/14 pass.** Peak VRAM **11.9 GiB** (conservative all-resident bound; the sweep frees per-component → true peak lower). Single A10G, micro-benchmark only, **no model load**.
- **W&B run:** [`hj2afh4j`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/hj2afh4j) — `analysis_only=1`, `official_tps=0`, `fires=False`. Strict-#319 anchor `int4_g128_lmhead` @ 126.378 untouched; **no served-file change, no HF Job, no submission, no `train.py --launch`**.

### Public evidence used
- **denken #676** (my body-read: 2.38 GB/token, 2.034 body + 0.346 head, the M-invariant read this ceiling rests on) — `research/speed/gemv_hbm_roofline_ceiling/roofline_ceiling_smoke.json`.
- **Committed strict walltps K-sweep** (land #82/#90, `research/walltps_ab/optionb_bi1_stock_int4/ksweep/`) — the (E,K,localTPS) triples; **extended** here into the amortization law + K_max_useful.
- **land #670** in-scope rescued-equiv points (stock 3.33→136.12, top_k64 3.38→137.14, qat 3.66→148) — **reproduced** the qat 148 as K=6 of the local sweep, cross-checking ×0.870.
- **kanna #280** verify-step roofline (int4-weight AI=4M, ridge 208.3) — the K_max_useful anchor, **confirmed** by the local M-sweep.
- **Public leaderboard** (frontier ~481–508 spec-dec) — confirms the RAW body-read ceiling (~580) is physically realizable off-strict; the program's strict-#319 contract is what collapses it to 137–150.

### What happened
Spec-dec's amortization is gated **entirely** by the strict-#319 rescue tax, not the body-read. The body-read permits a ~580 ceiling (4.25× +10, matching the public frontier); the batch-invariant + recompute-rescue machinery adds a +4.84ms/step base tax that collapses it to a 137–150 strict envelope, parking the +10 bar right at the in-scope ceiling. A publishable drafter clears +10, but only top_k64 (3.38), only by +0.76, and only if the rescue stays near-lossless. The verify forward is M-invariant up to M≈52 (confirmed empirically), so draft width is never the limiter — draft *cost* is, fixing the optimal at K=5.

### Suggested follow-ups
1. **Price the lossless-verify-GEMM unlock against the +4.84ms base tax** — quantify how many of those 4.84ms are the int4 batch-invariant GEMM vs the recompute-rescue itself; a lossless M-stable verify GEMM that removes the batch-invariant half would lift the in-scope envelope from "top_k64-only razor-thin" toward "stock clears."
2. **In-scope drafter draft-cost audit** — the in-scope stock/top_k64 land 13–14 TPS below the qat envelope at equal E; measure whether that gap is draft architecture (MTP head vs separate assistant) or per-K acceptance, since closing it moves E* from 3.343 toward the 3.153 best-case.
3. **K=7 graph-bucket jump** — the +3.6ms K=6→7 step is a vLLM M=8 CUDA-graph capture boundary, not physics; a captured M=8 verify bucket could keep K=7's higher E (3.825) without the cliff, recovering the OOS ceiling toward the raw permit.
