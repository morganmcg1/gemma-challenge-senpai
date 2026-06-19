STUDENT fern:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["qbtzipp2"],"primary_metric":{"name":"bi_tax_decode_ms_per_tok","value":3.30802},"test_metric":{"name":"bi_tax_operating_point_robust","value":1}}

## Results

**Verdict: `bi_tax_operating_point_robust = 1`.** The per-token **decode** BI tax is gen-length-FLAT at **32.68% ± 0.11pp** across {128, 256, 512, 1024} new tokens (spread 0.09pp, max deviation from the #759 anchor 0.11pp ≪ the ±0.65pp = ±2%-of-31.72% tolerance). It reproduces my merged #759 device anchor (32.73%) almost exactly. The 31.72% tax is a **decode steady-state property, not an artifact of one operating point.**

New, load-bearing finding: the **prefill** BI tax is only **13.09%** — far below the decode tax — so the BI tax is *decode-concentrated*. That means the only way the official ~157 number legitimately drifts is via **prefill amortization** (a benchmark-operating-point property that moves the BI=0 baseline identically), **not** tax fragility.

This is an **analysis/calibration card** on my own #759 COST measurement (`analysis_only=1, official_tps=0, no_hf_job=1, fires=0`). The locked `int4_g128_lmhead @ 126.378` baseline is untouched. No submission, no HF Job, no served-file change.

### 1. Decode-vs-prefill split (the headline)

| component | BI=0 | BI=1 | added | BI tax % |
|---|---|---|---|---|
| **decode** (per output tok, @GEN=512) | 2.22689 ms/tok | **3.30802 ms/tok** | +1.08113 ms/tok | **32.68%** |
| **prefill** (one-time, cold, P=564 tok) | 100.093 ms | 115.172 ms | +15.078 ms | **13.09%** |
| prefill normalized | 0.17747 ms/prompt-tok | 0.20420 ms/prompt-tok | +0.02673 | 13.09% |

Single-stream `output_tps` is decode-dominated, so the 31.72% is driven by the decode component (confirmed). The prefill GEMM subsplit shows why prefill is cheaper-taxed: its int4 Marlin **body** GEMM (78.3 ms, the un-swapped custom CUDA op) dominates and **cancels** across arms; the whole prefill tax is the bf16 GEMM swap (7.5→22.0 ms) + a little attention — same mechanism as decode, but a smaller *share* of prefill because the large-M body GEMM dilutes it.

### 2. Generation-length sweep (decode tax is flat)

| GEN | BI0 ms/tok | BI1 ms/tok | added | decode tax % | BI1 wall-proxy tps |
|---|---|---|---|---|---|
| 128 | 2.26823 | 3.37082 | 1.10259 | 32.71% | 269.2 |
| 256 | 2.24627 | 3.33822 | 1.09196 | 32.71% | 281.0 |
| 512 | 2.22689 | 3.30802 | 1.08113 | **32.68%** | 291.4 |
| 1024 | 2.24699 | 3.33498 | 1.08799 | 32.62% | 291.8 |

Decode tax %: **min 32.62 / max 32.71 / mean 32.68 / spread 0.09pp**. The per-token decode *time* is itself near-flat (≤1.5% wobble) — expected from the model physics: `sliding_window=512` with a 5:1 sliding:full layer pattern (35 sliding / 7 full of 42), so KV growth only loads the 7 full-attn layers, and the GEMM family (87–89% of the tax) is per-token-flat by construction. (The rising wall-proxy 269→292 is fixed per-request overhead amortizing, which is exactly why the tax is read off the device trace, not the wall.)

### 3. Prediction band around ~157 (implied official `output_tps`)

Modeled per-request `output_tps(G) = G / (T_prefill + G·t_decode(G))`, with each arm anchored to its #750 official number at GEN=512 (BI1=156.949, BI0=229.847) and the profiled prefill/decode shape extrapolated to other G (prefill scaled to a consistent P_ref=328-tok prompt; ratio is profiler-overhead-robust):

| benchmark GEN | BI1 (fire) implied tps | BI0 (baseline) implied tps | op-point tax % |
|---|---|---|---|
| 128 | **138.60** | 197.57 | 29.85% |
| 256 | 149.93 | 217.48 | 31.06% |
| 512 (= #319 / #750 anchor) | **156.95** | 229.85 | 31.72% |
| 1024 | 158.72 | 233.51 | 32.03% |
| →∞ (decode asymptote) | ~161.84 | ~239.42 | ~32.4% |

- **Which operating point gives ~157:** GEN=512 — i.e. the #319 protocol's **512 new tokens**, exactly where fern #750 measured 156.95. The band reproduces it by construction.
- **How far it can move:** a **short-gen / prefill-heavy** official benchmark lands the fire **lower in absolute TPS (~139 @ 128 tok)** because the one-time prefill amortizes over fewer output tokens — but at a **lower** blended tax (~30%, since prefill is only 13%-taxed); a **long-gen** benchmark approaches the **decode asymptote ~162** at ~32% tax. **Absolute-TPS band: [~139, ~162]. Tax band: [29.85%, 32.03%].**
- **Critical:** this absolute drift is *not* BI-tax fragility — BI=0 drifts proportionally (197.6→239.4). The two arms move together; the tax between them is bounded `[13% prefill floor, 32.7% decode ceiling]`.

### Anchor-length family ledger (GEN=512, matches #319 protocol)
`matmul_gemm` added +0.94372 ms/tok (**share 87.3%**), `attention` +0.13436 (12.4%), rest ~0 — the #759 GEMM-dominance (88.6% @ GEN=256) holds at the 512-token anchor too.

### Engine, operating points, repro
- **Engine:** local A10G (SM86), `/tmp/senpai-venvs/20f658587e8a6643`, vLLM 0.22.0, `submissions/int4_mtp_batchinv` fire config **verbatim** (spec ON, `NUM_SPECULATIVE_TOKENS=6`, `MAX_NUM_SEQS=1`, drafter `gemma-4-E4B-it-qat-q4_0-...-assistant`, TRITON_ATTN), + `--profiler-config` torch device-trace only (no kernel/quant/cudagraph/BI change). Reuses #759's `launch_prof_server.sh` + `parse_traces.py` classifier.
- **Protocol:** one server boot per BI arm; repeated `/start_profile`+`/stop_profile` (no trace accumulation — event counts scale linearly 23.7k→178.7k with GEN). DECODE windows = warm-prompt prefix-cache hit → GEN pure-decode steps; PREFILL window = fresh nonce-salted prompt (guaranteed cold prefill) at `max_tokens=1`, minus one decode step.
- **Operating points swept:** GEN ∈ {128,256,512,1024} new tokens, τ=0, ignore_eos, prompt~328 tok; cold prefill P=564 tok. BI ∈ {0,1}.
- **Peak GPU memory:** ~19.65 GiB / 23 GiB (both arms).
- **W&B:** `qbtzipp2` (group `bi_tax_operating_point`).
- **Commands:**
  ```bash
  cd research/validity/bi_tax_operating_point_765
  bash run_sweep_all.sh 0      # BI=0 arm
  bash run_sweep_all.sh 1      # BI=1 arm + analyze -> runs/sweep_ledger.json
  /usr/bin/python3 log_wandb_765.py --ledger runs/sweep_ledger.json \
    --bi0-summary runs/arm_bi0_summary.json --bi1-summary runs/arm_bi1_summary.json \
    --group bi_tax_operating_point
  ```

### What happened — honest analysis
The hypothesis held: the per-token decode BI tax **is** gen-length-flat in steady state (≤0.1pp across an 8× gen-length range), so the 31.72% is operating-point-robust and the **tax band is tight**. The one place the hypothesis was incomplete — and where the board post needs a caveat — is the **absolute** ~157: it is *not* a single fixed number across benchmark operating points. It is fixed **at the #319 protocol (512 new tokens)**, and moves with gen-length via prefill amortization to ~139 (128 tok) … ~162 (≥1024 tok). Because the prefill tax (13%) < decode tax (32.7%), the fire's *relative* standing actually **improves** at shorter gens. So the safe board statement is: **"~157 holds for ~512-token generations; the BI tax is gen-length-invariant at 32.7% ±0.1pp; the absolute figure ranges ~139–162 across plausible gen-lengths, and that range is shared by the BI=0 baseline, not a fragility of the tax."**

### Public evidence used
Internal calibration of my own merged work — fern #759 (`BI_TAX_GEMM_DOMINATED`, W&B `9hf7gvzd`) and fern #750 (~157, W&B `cdkvekkn`); companion land #760 (`min_strict_bi_tps`=156.95, W&B `2rmeroz8`). No external method reproduced.

### Suggested follow-ups
1. **Confirm the official benchmark's generation length.** The entire absolute-vs-tax distinction hinges on it. If the HF-Job benchmark generates ≪512 tokens, the board's ~157 should be restated toward ~139–150 (while the 32.7% tax claim stays valid). This is the single highest-leverage disclosure before the organizer measures.
2. **Prompt-length axis of prefill amortization.** I fixed P_ref=328; a prefill-length sweep would turn the 1-D gen-length band into the full 2-D (prompt × gen) operating-point surface for `output_tps`.
3. **Since prefill is only 13%-taxed,** a partial-BI scheme that keeps deterministic kernels in *decode* but uses fast kernels in *prefill* would shed ~13% of the prefill tax at zero decode-identity cost — only relevant if prefill ever becomes a non-trivial wall fraction (short-gen benchmark), and only if prefill numerics don't feed greedy identity (needs a separate check).
