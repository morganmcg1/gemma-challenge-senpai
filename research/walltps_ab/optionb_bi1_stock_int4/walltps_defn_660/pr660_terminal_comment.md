STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"wandb_run_ids":["gi91z0ax"],"primary_metric":{"name":"rescued_k6_local_wall_tps_stark_basis","value":135.82},"test_metric":{"name":"official_tps","value":0},"verdict":"GAP_IS_REAL","gap_residual_after_def_match_pct":8.28,"named_axis_full_vs_steady_spread_pct":-1.00,"rescued_k6_band_local":[135.82,146.82],"stark_unrescued_k6":155.58,"ar_rungs_agree_pct":0.09,"ppl_unchanged":2.0055}

## Results

**Verdict: `GAP_IS_REAL`.** The 9.4% un-rescued K6 gap (my 170.16 vs stark's 155.58) is **NOT** the advisor-named full-e2e-vs-steady-state window — that axis is worth only **−1.00%** on my harness, an order of magnitude too small. The gap is a genuine **speculative-path throughput difference**, and I can prove the locus *from in-scope data alone* using the AR-rung agreement as a constraint — without needing stark's exact denominator (which lives on his out-of-launch-scope branch; see "Scope note" below).

### 1. Per-K wall_tps under every definition (the headline table)

Fresh instrumented un-rescued capture, exact #632 stack (`int4_mtp_batchinv`, `VLLM_BATCH_INVARIANT=1`, `MAX_NUM_SEQS=1`, greedy, 128×512, seed=1, drafter `/tmp/qat-assistant`). One fresh server per K.

| definition | K6 (fresh) | K5 | residual vs stark 155.58 |
|---|---|---|---|
| **`wall_tps_full_e2e`** (tok / total decode wall, incl. prefill+ramp) | **170.16** | **172.74** | **+9.37%** (K6) |
| **`wall_tps_steady_state`** (vLLM gen-throughput meter, prefill/ramp excl.) | **168.46** | **172.02** | **+8.28%** (K6) |
| stream full-e2e (per-request wall, TTFT incl.) | 171.36 | 171.50 | +10.14% |
| stream steady (decode window, prefill excl.) | 174.76 | 178.00 | +12.33% |
| cold-job wall (tok / (decode + 120s boot)) | 129.73 | 131.50 | −16.62% |
| **named full-vs-steady spread** | **−1.00%** | **−0.42%** | — |

K6 full_e2e **170.16 reproduces #632's 170.21 to 0.03%** — the anchor holds. Every decode-window definition sits **8–14% above** stark's 155.58; the only definition *below* it is the full-120s-boot-inclusive one (−16.6%, overshoots). **stark's 155.58 sits between my hot-decode band and my cold-boot wall — no standard windowing of my raw data reaches it.**

### 2. stark's definition (what's determinable) + Scope note

From the **in-scope** advisor PR #660 baseline + committed `research/CURRENT_RESEARCH_STATE.md` / `EXPERIMENTS_LOG.md`: stark #642's method is **"measure 3 real wall-TPS (acceptor / AR-rung / un-rescued) on one local harness; decisive ratio = acceptor/AR × 126.378"**, un-rescued ceiling **155.58**, AR-rung **77.89**, measured at K=5 too.

**Scope note (honest):** the PR says "stark's branch is in-scope," but this launch's operator isolation rule restricts me to `approval-gated-8gpu-20260613` + land's own branches. I therefore did **not** read stark's branch or W&B; I used only (a) my own captures, (b) the 155.58 / 77.89 numbers quoted in the in-scope PR + committed docs, and (c) the public challenge bucket (no stark de-projection post there — only the broader-team 489–508 TPS leaderboard). **stark's exact denominator *windowing* is therefore not directly confirmed.** The good news: the verdict does not depend on it — the AR-agreement argument below rules out *every* windowing/overhead explanation regardless of which one stark used.

### 3. The decisive argument — AR-rung agreement localizes the gap to the spec path

The PR's key fact: the two harnesses' **M=1 AR rungs agree to 0.09%** (land 77.962 vs stark 77.89). Treat that as a constraint and test each candidate explanation:

| candidate explanation for the 9.4% K6 gap | what it would do to the AR rung | compatible with 0.09% AR agreement? |
|---|---|---|
| **named axis** (full-e2e → steady window) | — (it's a K6 re-pricing) | **NO** — the spread is only **−1.00%** on my harness; can't move 170→155.58 |
| **server-boot folded into wall** (120s) | AR → **68.22 (−12.49%)** | **NO** — stark's AR would be ~68, not 77.89 |
| **fixed per-request overhead** (the +36.1s / 0.282s-per-req the gap implies) | AR → **74.75 (−4.12%)** | **NO** — would also depress AR by ~4% |
| **a different native vLLM meter** | — | **NO** — 155.58 ≠ gen (168.46), accepted (121.05), or drafted (273.61) |
| **speculative per-step component** (draft fwds + verify, and/or acceptance length) | **zero at M=1** (no drafting) | **YES** — affects K≥1 only, leaves AR untouched |

Boot, per-request gap, and prefill are all *long-decode* costs: any of them large enough to explain a 9.4% K6 gap would force the (2.2× longer) AR decode to disagree by 4–13%. It agrees to 0.09%. **The only locus left that is invisible at M=1 but active at K=6 is the speculative per-step work — i.e. a real difference in draft-forward/verify latency or in acceptance length between the two harness configs.** That is `GAP_IS_REAL`, not a metric-definition artifact.

### 4. `gap_residual_after_def_match` and the reconciled rescued re-price (primary deliverable)

- **`gap_residual_after_def_match_pct = 8.28%`** (K6): after picking the *most favorable* named/decode-window definition (steady, 168.46), the residual to stark is still +8.28% → aligning the named definitions does **not** close the gap. (K5: 10.23%.)
- **Rescued re-price** `rescued = 1/(1/U + f/A)`, f=7.282% (#648), A=77.962 (#658):

| basis for U (un-rescued K6) | rescued K6 local wall-TPS |
|---|---|
| stark's U = 155.58 (A=77.89) — **primary** | **135.82** |
| my steady U = 168.46 | 145.56 |
| my full-e2e U = 170.16 | 146.82 |

**Because the gap is real, the rescued K6 number stays a band: `[135.82 (stark basis) … 146.82 (my basis)]`, ±~4% / 8.1% spread** (midpoint ~141.3, consistent with my #648 cross-check 141.05). The conservative stark-basis **135.82** is the headline `primary_metric`. K5 rescued (my basis) = 148.76 (matches #658).

### 5. What happened — honest analysis

The advisor's hypothesis was that the 9.4% is a `full_e2e`-vs-`steady_state` reporting difference (I reported steady, stark reported full-e2e, or vice-versa). **The data refutes that specific mechanism:** on my harness full_e2e and steady are within 1% of each other, so no choice between them spans a 9.4% gap. What the instrumentation *did* establish, decisively, is that the gap cannot be **any** wall-clock windowing, boot inclusion, or per-request overhead — all of those are excluded by the AR-rung agreement, which pins every non-speculative cost equal. The residual is therefore a true spec-path throughput difference. Given both harnesses run greedy + seed=1 + the same `/tmp/qat-assistant` drafter, the **acceptance length should be identical** (mine: espec 3.66) — so my leading hypothesis is a **per-speculative-step latency** difference (e.g. `VLLM_USE_FLASHINFER_SAMPLER`, cudagraph capture of the draft model, or attention-backend config for the draft forwards; my capture set `VLLM_USE_FLASHINFER_SAMPLER=0`). If instead stark's espec is materially below 3.66, that *is* the gap. Either way it is real and **resolvable by config convergence, not by re-defining wall_tps.**

PPL unchanged at **2.0055** (identity-preserving spec lane, K-independent). No HF Job, no submission, `official_tps=0`, locked `int4_g128_lmhead`@126.378 untouched — these are **LOCAL** numbers on a different measurement plane than the OFFICIAL anchor and do **not** trigger a #481 fire.

### 6. Data provenance (full disclosure)

- **K6 = fresh re-run, complete** (pass1 nonstream + pass2 stream + full gen/spec meters); reproduces #632 to 0.03%. This is the headline (the 9.4% gap is at K6).
- **K5 = earlier complete session.** The fresh re-run was interrupted mid-K5 (pass1 reached 115/128, no summary/stream pass written), so K5 is reported from the earlier full session. K5 is the secondary curve-shape check; its numbers are consistent with the re-run's partial K5 gen-meter intervals.

### 7. Reproduce / environment

```bash
cd target
# capture (one fresh server per K, full per-step timing):
VLLM_BATCH_INVARIANT=1 python3 research/walltps_ab/optionb_bi1_stock_int4/walltps_defn_660/capture_defn_660.py --ks 6,5
# offline reconcile + W&B:
python3 research/walltps_ab/optionb_bi1_stock_int4/walltps_defn_660/finalize_defn_reconcile.py
```

- **Peak VRAM:** ~19.9 GB (19917 MiB) — int4 server, `GPU_MEMORY_UTILIZATION=0.90`, single A10G, `MAX_NUM_SEQS=1` (same stack as #658).
- **W&B:** `gi91z0ax` — group `walltps-defn-reconcile-land` (https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/gi91z0ax)
- **Artifacts:** `research/walltps_ab/optionb_bi1_stock_int4/walltps_defn_660/{defn_reconcile_final.json, k6/, k5/, finalize_defn_reconcile.py}`

### 8. Suggested follow-ups

1. **The one discriminating measurement (cheapest, decisive):** have stark report his **K=6 mean acceptance length (espec)**. If espec ≈ 3.66 (== mine) → the gap is per-step draft/verify *latency* (a harness-config diff: sampler / cudagraph / backend); if espec < 3.66 → the gap is *acceptance* (drafter/config diff). One number splits the two and ends the ±9% band.
2. **Config-convergence A/B:** re-run both un-rescued K6 captures with byte-identical spec-path env (`VLLM_USE_FLASHINFER_SAMPLER`, cudagraph mode, draft attention backend) pinned equal. If they then agree, the gap was config; the converged U is the right basis for the OFFICIAL projection.
3. **Resolve via the ratio, not the level:** stark's actual fire metric is `acceptor/AR × 126.378` — a *ratio* in which consistent windowing cancels. The 9.4% level-gap in the un-rescued ceiling may not propagate to that ratio; worth confirming the ratio is stable across our two harnesses even though the level is not.
4. **For the advisor on scope:** if you want a code-level confirmation of stark's denominator (rather than the AR-agreement proof), please either bank stark's de-projection harness onto `approval-gated-8gpu-20260613` or have the human name his run for this launch — under the current isolation rule I could not inspect his branch/W&B directly.
