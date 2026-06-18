STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"fires":false,"verdict":"CHEAP_ATTENTION_PIN_EXISTS","wandb_run_ids":["026h4xrx"],"primary_metric":{"name":"attention_pin_tps_cost_frac","value":-0.11559},"test_metric":{"name":"lossless_specdec_margin_vs_plus10","value":50.969}}

## Results

**Verdict: `CHEAP_ATTENTION_PIN_EXISTS`.** A config-reachable, verified-lossless attention pin exists, it is **free** (slightly beneficial: −0.843 ms/step, −11.6% of the AR-step transfer), and lossless spec-dec at K=5 clears +10 by a wide margin **with no recompute-rescue tax**. The attention pin is the cheaper of the two lossless routes by 25–50 TPS — a second, much-cheaper-than-rescue lossless path. Flagging for the #481 surface.

### Key decision scalars (denken #677 K=5, e_accept*=3.343, ×0.870)

| scalar | value |
|---|---|
| `attention_pin_tps_cost_frac` (cheapest lossless = fixed2d) | **−0.1156** (free) |
| `lossless_specdec_tps_at_k5` (served, pinned, NO rescue) | **187.35** |
| `lossless_specdec_margin_vs_plus10` (− 136.378) | **+50.97** |
| `fixedsplit_is_lossless` (argmax / GREEDY_IDENTICAL) | **1** |
| byte-identical tier (bi1) margin @ deployed −16% | **+25.94** |
| rescue (stark #669) margin @ K=5 | +0.762 |
| `attention_pin_beats_rescue` | **True** |
| `bracket_all_clear_plus10` | 1 |
| `bi1_blanket_cost_frac` (full-vocab measured; deployed = lawine −16%) | 0.4002 |

### Per-config measured (M=1 AR, full-vocab QAT int4, ctx≈512)

| config | path knob | M=1 path | attn-step ms | Δattn vs base | argmax flips | bitdiff | lossless | bit-exact |
|---|---|---|---|---|---|---|---|---|
| (a) baseline | deployed occupancy split-KV | **3D (16-split)** | 6.123 | 0.0 | **1/768** | 0.902 | ✗ | ✗ |
| (b) bi1 `VLLM_BATCH_INVARIANT=1` | aten BI | 2D (1-split) | 5.273 | −0.850 | 0/768 | **0.0000** | ✓ | **✓** |
| (c) fixed2d `MIN_LAUNCH_GRID_SIZE_2D=0` | path-selector pin | 2D (1-split) | 5.280 | **−0.843** | 0/768 | 0.0143 | **✓** | ✗ |

(Path probe: baseline `path_m1_use_3d=[True]`, `path_verify_use_3d=[False]`; both pinned configs `[False]`/`[False]`. `ar_vs_ar=1.0` all configs.)

### Mechanism (confirms #680)

The strict-#319 break **is** the M-dependent 3D-vs-2D Triton split-KV attention reduction — proven directly by the path probe. Baseline M=1 decode takes the **3D segmented** path (16-way split-KV); M=6 verify takes the **2D single-pass** path. That path mismatch alone produces bitdiff 0.902 and the 1/768 argmax flip → `is_lossless=False`. Pinning *both* M=1 and M=6 onto the same 2D path (configs b and c) closes the break: **0/768 flips, both lossless**.

### Two losslessness tiers (the honest caveat the PR's "byte-identical" wording requires)

- **(b) bi1 — byte-identical.** bitdiff = **0.0000**, `is_bitexact=True`, 0/768 flips. This is the literal byte-identity tier. The deployed authoritative cost is lawine #675's **−16%** (≈ +1.55 ms). Lossless spec-dec at K=5 = **162.32**, margin **+25.94**.
- **(c) fixed2d — argmax-lossless and FREE.** 0/768 argmax flips, `is_lossless=True` → strict-#319's actual GREEDY_IDENTICAL contract holds. bitdiff = 0.0143 (`is_bitexact=False`), but this is **NOT a path difference** — both M=1 and M=6 run the identical 2D single-split reduction. It is the irreducible int4 exact-tie non-determinism from my #654 (two M=1 runs disagree at ties; `near_tie` is a don't-care band). Attention-step cost = **−0.843 ms** (−11.6% of the deployed AR-step transfer) → the 16-way 3D split is net *overhead* at M=1 decode / ctx≈512.

### Decomposition — the −16% blanket is the aten swaps, NOT the attention pin

bi1 and fixed2d pin the attention reduction to the same 2D path and give the **same attention-only delta** (−0.850 vs −0.843 ms by the CUDA-event timer). So the attention pin itself is free in both. I could *not* reproduce lawine's −16% as a single full-step number on my harness because it loads the full **262k** vocab, where the BI `log_softmax` swap inflates the blanket to +40% (`bi1_blanket_cost_frac=0.4002`). But the attention-only timer isolates the actual pin at −0.85 ms (free) — identical to fixed2d — proving the blanket's cost is the **aten / 262k-`log_softmax` swaps**, not the attention reduction (and a no-op on `ops.marlin_gemm`, per my #680). I therefore take lawine #675's deployed **−16%** as authoritative for the byte-identical blanket tier. This refines lawine: the −16% = aten-swap overhead; the *targeted* attention pin (ubel #491/#484's lever) is actually **free**, beating ubel's ≈5.1% expectation.

### Decision scalar — does lossless spec-dec clear +10 without the rescue tax?

My spec-dec model is anchored to denken #677's **published** rescue official 137.14 and reproduces it exactly (`rescue_reproduces_denken_137p14=1`, rescue margin +0.762 — denken's razor-thin binding wall). Applying the cheapest lossless pin (fixed2d, −0.843 ms) to the 8.14 ms AR step → t_step_local 15.524 ms → `lossless_specdec_tps_at_k5` = **187.35**, margin **+50.97**. The byte-identical tier (bi1, deployed −16%) → **162.32**, margin **+25.94**. Both clear +10 comfortably and **dominate the rescue path by 25–50 TPS** (`attention_pin_beats_rescue=True`).

### Robustness bracket

Across the entire plausible pin-cost bracket — fixed2d −0.84 ms (free) … ubel-targeted +0.43 ms … lawine-blanket +1.55 ms — **every point clears +10** (`bracket_all_clear_plus10=1`). Conservative margin at the −16% blanket bound = **+25.94**; ubel-targeted = +36.77. The verdict is robust to which losslessness tier and which cost estimate you demand.

### Command

```bash
# 3 fresh-process configs (BI / monkeypatch are process-global, snapshotted at import)
research/validity/strict319_attention_pin_684/runs/full_run.sh
# decision scalar + W&B
/usr/bin/python research/validity/strict319_attention_pin_684/decide_and_log.py
```
COMMON per config: `--verify-width 6 --n-prompts 24 --n-new 32 --ctx-cap 512 --det-prompts 10 --tps-warmup 24 --tps-long 80 --tps-short 16 --tps-reps 4 --tps-ctx-prompts 3 --attn-tokens 64 --attn-warmup 16 --attn-reps 12`

**Peak memory:** 18.96 GiB (full-vocab QAT int4 `gemma-4-E4B-it-qat-w4a16-ct`, vLLM 0.22.0 in-proc engine, `gpu_memory_utilization=0.90`).
**W&B:** `026h4xrx` (group `strict319-attention-pin-cost-land`) — https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/026h4xrx

### What happened

The hypothesis holds and is **stronger** than priced. Not only does a lossless attention pin exist that clears +10 without the rescue tax — the *targeted* pin (configs b/c both pin attention to 2D) is **free** at the deployed M=1 / ctx≈512 regime: the occupancy heuristic's 16-way split is net overhead at decode, so removing it *gains* 0.84 ms. The only real cost is the aten-swap blanket if you demand strict byte-identity (bi1), and even that clears +10 by +25.9. The attention pin is a strictly cheaper lossless route than stark #669's recompute-rescue (which clears by only +0.76).

### Caveats (honesty)

1. **Full-vocab losslessness.** The argmax-identity ran on the full-vocab QAT ckpt (the deployed pruned-16k head can't load into vLLM's in-process `LLM()`). The attention-path mechanism and the head-independent absolute-ms attention delta transfer cleanly (the delta is isolated by a CUDA-event timer and is head-independent), but the 0/768 flip was checked on full-vocab logits. Deployed 16k-head losslessness is covered separately by lawine #544 / my #552.
2. **fixed2d's 0.0143 bitdiff is the #654 tie residual** (don't-care near-tie band), not a path break. If the bar is literal byte-identity rather than GREEDY_IDENTICAL, use the bi1 tier (byte-identical, deployed −16%, still +25.9).
3. **Context dependence.** The −0.84 ms attention delta is measured at ctx≈512 (the deployed regime). The 3D-split overhead-vs-benefit crossover shifts with sequence length; at very long contexts the 16-way split could turn net-beneficial and the pin cost positive — but still bounded above by the lawine −16% blanket, which clears +10.
4. **Reachability — not a kernel build.** bi1 is a pure env var (`VLLM_BATCH_INVARIANT=1`). fixed2d is a boot-time module-constant override (`triton_attn.MIN_LAUNCH_GRID_SIZE_2D=0`) — config-reachable via a sitecustomize/serve-boot patch (the kanna #177 precedent), **not a kernel rebuild** (the kernel already supports both paths; we only force the selector). So `NEEDS_KERNEL_BUILD` does not apply.
5. `official_tps=0` — analysis_only, no HF Job, served file untouched. The 187.35 / 162.32 / 137.14 are modelled served projections on denken's ×0.870 basis, not measured official benchmarks.

### Suggested follow-ups

- **Price the fixed-split pin on the served 16k-head stack directly** (sitecustomize `MIN_LAUNCH_GRID_SIZE_2D=0` at serve boot, kanna #177 style) and run the strict-#319 GREEDY_IDENTICAL gate end-to-end — converts the modelled +50.97 into a measured official number and closes caveat 1.
- **Long-context sweep of the 3D-vs-2D crossover** (ctx 512 → 4k → 16k) to map where the pin flips from free to costly, bounding the worst-case deployed pin cost.
- **Hand the measured free-pin tax to denken #683** for their parametric ceiling table — the attention-pin tax input is ≈0 (or −16% if byte-identity is demanded), setting their lossless-spec-dec ceiling at the +51 (or +26) margin.
