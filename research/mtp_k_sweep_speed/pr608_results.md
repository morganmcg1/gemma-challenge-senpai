STUDENT fern:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["3sqore0w"],"primary_metric":{"name":"fastest_official_proxy_tps","value":481.25},"test_metric":{"name":"k7_proxy_tps_reproduces_597_427.7","value":428.23}}

## Results — Option-B spec speed Pareto: MTP draft-depth K-sweep

**Deliverable: the fastest spec config is K=13 at 481.25 official-proxy TPS** (fastest in the swept range). But the headline hides the real shape — read the caveat: the proxy frontier is **monotone-rising-but-decelerating** through K=13 and never peaks, while the *realistic* (diverse-prompt) acceptance saturates by ~K=9. The physically-meaningful serve frontier sits near **K=9**, not K=13.

### Setup (no rebuild)
Reused the #597 build verbatim — `int4_g128_lmhead` (model) + `/tmp/qat-assistant` (MTP drafter `gemma4_assistant`→`Gemma4MTPModel`), `VLLM_BATCH_INVARIANT=1`, MAX_NUM_SEQS=1, FlashInfer sampler off, MAX_MODEL_LEN=4096. **K is a free serve-time param** (`NUM_SPECULATIVE_TOKENS` → `--speculative-config num_speculative_tokens`), confirmed in `submissions/int4_mtp_batchinv/serve.py`, so no rebuild was needed. The ~163G build persisted on disk; I did **not** rebuild.

**Protocol = the exact #597 protocol:** `harness.probe_tps` single-stream decode_tps (512 decode tokens, median of 3 repeats) × τ=1.035 (banked, lawine #594) = official-proxy TPS. `official_tps=0`, `analysis_only=true`. Validation: my **K=7 = 428.23** reproduces #597's **427.7** (|Δ|=0.5 TPS, within tolerance) → protocol faithful.

### Speed Pareto (swept K ∈ {3,5,7,9,11,13})

| K | local decode TPS | **official-proxy TPS** | realistic accept/step | realistic e_accept | realistic accept-rate | probe accept/step | probe accept-rate | peak VRAM (GB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 308.37 | 319.16 | 2.123 | 3.123 | 0.708 | 2.871 | 0.957 | 19.45 |
| 5 | 371.66 | 384.67 | 3.039 | 4.039 | 0.608 | 4.659 | 0.932 | 19.47 |
| 7 | 413.75 | **428.23** | 3.643 | 4.643 | 0.520 | 6.357 | 0.908 | 19.45 |
| 9 | 438.18 | 453.52 | 3.922 | 4.922 | 0.436 | 7.965 | 0.885 | 19.46 |
| 11 | 453.45 | 469.32 | 4.044 | 5.044 | 0.368 | 9.490 | 0.863 | 19.45 |
| **13** | **464.98** | **481.25** | 4.219 | 5.219 | 0.325 | 10.977 | 0.844 | 19.46 |

- **`realistic`** = mean over 8 diverse ShareGPT eval prompts × 256 tokens each (Prometheus counter-delta).
- **`probe`** = the single ignore_eos probe prompt that the proxy-TPS is measured on (Prometheus counter-delta).
- accept/step = accepted draft tok per draft step; e_accept = +1 bonus token (tok per forward step); accept-rate = accepted/drafted.

### What happened

1. **K=7 is NOT the speed optimum.** Every K≥9 beats the #597 candidate (427.7) on the proxy: **K9 +25.8, K11 +41.6, K13 +53.5 TPS.** Deeper draft is strictly faster on this metric across the whole range.
2. **The proxy frontier never peaks in-range — it decelerates hard.** Per-step proxy gains: +65.5 (3→5), +43.6 (5→7), +25.3 (7→9), +15.8 (9→11), **+11.9 (11→13)**. Monotone but clearly saturating; it had not turned over by K=13.
3. **CRITICAL caveat — the proxy reads on a degenerate prompt.** The proxy-TPS is measured on a single `ignore_eos` probe prompt where acceptance stays near-degenerate-high (probe rate 0.96→0.84, accept/step 2.9→11.0). On **realistic** diverse prompts, acceptance is far lower and **falls** with K (rate 0.708→0.325), while **realistic accept/step saturates**: 3.92 → 4.04 → 4.22 across K9/11/13 (+0.12, then +0.18 per +2K). So on a realistic workload the marginal throughput from K>9 is small — the probe-basis proxy keeps climbing mostly because the degenerate probe sustains high acceptance even at K=13. **The probe-basis "fastest = K=13" is partly an artifact of the probe; the realistic-throughput frontier sits near K=9.**
4. **local↔submission TPS faithfulness is lawine #606's axis, not mine** — I report local single-stream proxies only and defer the verified-TPS gap. Identity is wirbel #607's (this build fails byte-exact greedy identity per my #597), quality is stark #605's.
5. **VRAM is flat (~19.46 GB) across all K** — deeper draft costs no extra memory; the only cost is verify compute, which is exactly why the realistic gain saturates.

### Recommendation for the #481 A/B "option B = up to X TPS" number
- **Aggressive headline (probe-basis proxy, #597-faithful protocol): up to ~481 TPS at K=13** (still rising, so a literal probe-proxy ceiling is even higher).
- **Honest serve-realistic frontier: ~K=9 (proxy 453.5)** — beyond K=9 the realistic acceptance gain is <0.3 tok/step per +2K while acceptance-rate craters, so verified TPS will saturate well before the probe proxy does.
- Either way the answer to the card's question is **yes, a different K beats 427.7** (K≥9), and **K=7 is not the optimum**.

### Public evidence (per program.md citation requirement)
- **Reproducing/extending** senpai #597 spec candidate `int4_g128+MTP-K7` @ 427.7 official-proxy (W&B `p7jo2ap4`) — my K=7=428.23 reproduces it, and I extend the K axis.
- Public **AR ceiling / option A**: senpai `int4_g128_lmhead` **126.378 TPS, PPL 2.0057, 128/128** (bucket result `20260617-121233-993_senpai.md`, W&B `905tbujn`).
- Public **speed frontier** (different lane): firfir-cast `hayai-ctk48-w256-v1` **507.07 TPS / PPL 2.3813 / 128-128** (`20260616-212031-987_firfir-cast.md`) and openevolve `splitkv-lmhead12k-precache` **496.35 TPS / PPL 2.3735 / 128-128** (`20260616-070216-110_openevolve.md`). Those are **sliding-window/split-KV attention-approximation** lanes that trade PPL up to ~2.37–2.38. **My spec lane reaches K=13 proxy 481.25 while preserving the int4 PPL (~2.006-class)** — i.e. near-frontier proxy TPS *without* the +0.37 PPL tax — though the proxy↔verified gap (lawine #606) and the acceptance inflation above are unresolved here.

### Command
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/mtp_k_sweep_speed/k_sweep.py \
  --ks 3,5,7,9,11,13 --decode-tokens 512 --repeats 3 --accept-tokens 512 \
  --real-prompts 8 --real-tokens 256 --port 8021 \
  --out-dir research/mtp_k_sweep_speed/runs \
  --wandb-name "fern/mtp-k-sweep-speed-pareto" --wandb-group "mtp-k-sweep-speed-pareto"
```
- **Peak VRAM:** 19.47 GB (max across all K; flat).
- **W&B run:** `3sqore0w` (group `mtp-k-sweep-speed-pareto`); logs the full K-sweep Pareto table + per-K local/proxy TPS and probe/realistic acceptance series. `official_tps=0`, `analysis_only=true`.
- **Disk:** stayed at 87% (132G free) throughout — reused the build, eval scratch <100KB. No ENOSPC risk hit.

### Suggested follow-ups (NOT implemented — out of scope)
1. **Realistic-acceptance speed Pareto:** re-measure proxy TPS on a *diverse multi-prompt* decode (not the single ignore_eos probe) so the speed metric and the acceptance metric use the same workload. This would likely show the proxy peaking near K=9 and is the number the fire decision actually wants.
2. **Locate the literal probe-basis proxy peak** (extend K=15,17,…) — cheap (~2 min/K) but characterizes the degenerate-probe artifact, not the serve frontier; only worth it if the advisor wants the absolute probe-proxy ceiling for the "up to X" headline.
3. Hand K=9 (and K=13) to lawine #606 for a verified-TPS faithfulness check, since the probe→verified gap is the binding uncertainty on any "option B = X TPS" claim.
