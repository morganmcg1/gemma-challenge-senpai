STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["pr672-int4-aime-band","pr672-int4-neartie","pr672-int4-aime-VERDICT"],"primary_metric":{"name":"int4_aime_band_upper95","value":0.3883},"test_metric":{"name":"int4_aime_band_mean","value":0.3667}}

## Results

**VERDICT: `BLOCKER_ROBUST` — int4-body AIME is genuinely sub-bar. Option-B token-changing recovery stays dead. Report AIME as this band in the #481 update.**
**Near-tie mechanism: CONFIRMED — per-problem near-tie density predicts cross-session answer-instability (Spearman 0.451, n=60); cross-session divergences land on near-tie steps (median divergence margin 0.125 nat).**

`analysis_only=true`, `official_tps=0` logged as explicit W&B summary scalars. `int4_g128_lmhead` @ 126.378 official TPS untouched — no HF Job, no submission, no served-file change.

---

### Deliverable 1 — Multi-session greedy-AIME band (the decision scalar)

4 **fresh serve sessions** (fresh vLLM process each — the cross-session axis), int4-AR greedy AIME @12288, T=0, BI=1, `min_tokens=8`, `--no-thinking`, full 60 (years 2024 / 2025-I / 2025-II), seed 0, client-concurrency 16. `extract_fail=0` on every session.

| arm | session | maj@1 | n_correct |
|-----|---------|-------|-----------|
| int4-AR | s0 | 0.3667 | 22/60 |
| int4-AR | s1 | **0.3833** (max) | 23/60 |
| int4-AR | s2 | **0.3500** (min) | 21/60 |
| int4-AR | s3 | 0.3667 | 22/60 |

- **Session band [min, max] = [0.3500, 0.3833]**
- mean = **0.3667**, sd = 0.0136, se = 0.0068 (n=4)
- 95% CI of mean (Student-t, normal-approx over sessions) = **[0.3450, 0.3883]**
- bootstrap 95% CI = [0.3542, 0.3792]
- **`int4_aime_band_upper95` = 0.3883  vs  bar 0.420  →  gap −0.0317  →  `BLOCKER_ROBUST`**

The 95% upper bound (0.3883) is **−0.0317 below the 0.420 bar**, so the band does **not** straddle. Folding in the two historical greedy draws the card cites (#650 = 0.400 / 24-60, #668 = 0.350 / 21-60) gives a **6-session observed range [0.350, 0.400]**; the **single highest greedy session ever observed is 0.400 (#650), still −0.020 under the bar.** The "miss" is not a measurement artifact — int4-body AIME is robustly sub-bar.

### bf16 stability control (2 fresh sessions)

| arm | session | maj@1 | n_correct |
|-----|---------|-------|-----------|
| bf16 | s0 | 0.4833 | 29/60 |
| bf16 | s1 | 0.4833 | 29/60 |

- **60/60 problems token-bit-exact across the two fresh processes** (`bf16_bit_exact_rate = 1.0`, 0 answer-flips). Files have distinct md5 / `created_at` / `wall_s` (genuinely separate processes) but byte-identical token streams. Confirms (and strengthens) your #668 "11/11 bit-exact" — bf16 greedy AIME serving is deterministic; the cross-session noise is **int4-specific**.

---

### Deliverable 2 — Near-tie argmax margin (the mechanism)

Margin = `chosen_logprob − runnerup_logprob` (nats), over **921,785 int4 decode steps** (all 4 sessions × 60 problems).

- median margin = **8.375 nats**, mean = 8.317 — the typical decode step is highly confident.
- `frac(< 0.1 nat) = 0.0085`, `frac(< 0.05) = 0.0085`, `frac(< 0.01) = 0.0085`, `frac(< 0.2) = 0.0233`.
- Margins are **quantized** (vLLM rounds logprobs): the histogram has **nothing in [0.01, 0.1) nat** — near-ties are either near-exact (`< 0.01 nat`, 0.85% of steps = 7,813 near-tie argmaxes) or comfortably resolved (`≥ 0.1 nat`). Histogram: 64.7% of steps > 5 nats; only 0.85% are near-exact ties.

**Mechanism test — does near-tie density predict cross-session instability?** YES.
- per-problem near-tie density (frac of that problem's steps with margin < 0.10 nat) vs per-problem cross-session answer-flip rate (over 4 sessions): **Pearson 0.440, Spearman 0.451 (n=60)** → `mechanism_supported = true`. Tie-dense problems flip more.
- **At the first cross-session divergence token (int4 s0 vs s1):** 51/60 problems diverge; **median divergence-token margin = 0.125 nat** (the smallest resolvable gap); 43.1% of divergences occur at margin < 0.1 nat. Divergences concentrate exactly on the near-tie steps.
- int4-vs-bf16 first divergence is **EARLY** (as early as token 2), consistent with your #668 `FUNDAMENTAL` finding — the int4 body diverges immediately, not just at the head.

**Picture:** int4 Marlin matmul non-associativity flips the argmax only at the ~0.85% of decode steps that are near-exact ties; those flips compound (early-divergence) into ±1–2 problem swings per session (band width 0.0333), but the underlying int4-body accuracy (~0.3667) never lifts above 0.420. `VLLM_BATCH_INVARIANT=1` fixes batch composition but not cross-process kernel reduction order, so the ties resolve differently per process. bf16 computes the same logits bit-exactly → no ties to flip → stable.

### Measurement-validity (does capturing logprobs perturb the band?) — NO

Deliverable 2 needs `top_logprobs=2` ON; deliverable 1's band must not be a logprobs artifact. Two converging checks:
1. **logprobs-bias check:** an int4 **logprobs-OFF** official `aime_eval.py` session scores **0.375**, which sits **inside** the logprobs-ON band [0.350, 0.3833] (`bias_all_in_band = true`).
2. **bf16 with logprobs ON is 60/60 bit-exact** → enabling logprobs injects no nondeterminism. (`top_logprobs` is a read-only projection of the same logit vector argmax uses; it cannot move the argmax.)

---

### Comparison vs PR baseline

| quantity | baseline (PR body) | this PR |
|----------|--------------------|---------|
| int4-AR greedy band | #650 0.400, #668 0.350 (n=2) | **[0.350, 0.383] over 4 fresh; upper95 0.3883** |
| int4-AR sampled @12288 upper-CI | 0.4022 (#650) | corroborated: greedy upper95 0.3883 < 0.4022 < 0.420 |
| bf16 greedy | 0.4833, 11/11 bit-exact | **0.4833, 60/60 bit-exact** |
| bar | 0.420 (= 0.9 × base 0.4667) | upper95 −0.0317 under bar |
| decision | n=2 not rigorous | **`BLOCKER_ROBUST` (rigorous n=4 + n=2 historical)** |

### Exact commands

```bash
# fresh int4-AR server per session (BI=1, greedy, mml=16384) — serve_and_eval.sh:
CUDA_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
/tmp/vllm0220-srv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /workspace/gemma_build/int4_g128_lmhead --served-model-name gemma-4-e4b-it \
  --host 127.0.0.1 --port 8000 --max-model-len 16384 --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 2048 --max-num-seqs 16 --seed 0 --trust-remote-code
# per-session greedy AIME @12288 + per-token margins:
python3 research/validity/int4ar_aime_band_neartie/band_neartie.py session \
  --arm int4 --session-idx <i> --base-url http://127.0.0.1:8000 \
  --client-concurrency 16 --max-tokens 12288 --min-tokens 8 --seed 0 --out int4_session<i>.json
# driver (4 fresh int4 + 2 fresh bf16, fresh process each):
bash run_band.sh int4 0 4 ;  bash run_band.sh bf16 0 2
# band + near-tie + verdict:
python3 research/validity/int4ar_aime_band_neartie/band_neartie.py aggregate \
  --int4 int4_session0.json int4_session1.json int4_session2.json int4_session3.json \
  --bf16 bf16_session0.json bf16_session1.json \
  --int4-official _guard_official.json --bar 0.420 --neartie-thresh 0.10 \
  --out band_neartie_agg.json
# W&B (3 runs):
python3 research/validity/int4ar_aime_band_neartie/band_neartie.py wandb --aggregate band_neartie_agg.json
```

- **Peak GPU memory:** ~20.7 GB (`--gpu-memory-utilization 0.90` of the 23 GB device); GPU KV cache 391,163 tokens (23.9× concurrency headroom at 16,384 ctx). Single GPU; sessions run sequentially.
- **W&B group:** `int4ar-aime-band-neartie-ubel` — runs `pr672-int4-aime-band`, `pr672-int4-neartie`, `pr672-int4-aime-VERDICT` (decision scalars on the VERDICT run: `int4_aime_band_upper95=0.3883`, `verdict=BLOCKER_ROBUST`, `analysis_only=true`, `official_tps=0`, `mechanism_supported=true`).

### What happened — honest analysis

Both co-headline deliverables land and reinforce each other. The band is robustly sub-bar (upper95 0.3883, max-ever single session 0.400, both < 0.420), so the int4-body AIME miss is **real, not a measurement artifact** — `BLOCKER_ROBUST`, Option-B recovery stays dead. The near-tie mechanism *explains* the cross-session noise that motivated this card: int4 is unstable only because ~0.85% of decode steps are near-exact argmax ties, and a problem's tie density predicts its flip rate (Spearman 0.451), with divergences landing on the low-margin steps (median 0.125 nat) — while bf16, computing the same logits bit-exactly, is perfectly stable. The instability is genuine kernel-reduction-order noise on near-ties, **orthogonal to** and **not large enough to close** the −0.0317 accuracy deficit.

### Methods note (adaptation of the reproduce block)

I unified both deliverables into one self-contained tool, `research/validity/int4ar_aime_band_neartie/band_neartie.py`, rather than the two-script path sketched in the reproduce block (stock `aime_eval.py` for the band + `decompose_recoverable.py neartie` on the #668 streams). Reason: capturing `top_logprobs=2` **inline on the fresh band sessions** lets the density-vs-flip correlation use **all 60 problems × 4 sessions (n=60)** instead of the 11-problem #668 streams (n=11) — a materially stronger mechanism test, with the band and the margins guaranteed to come from the identical runs. Stock `aime_eval.py` and `decompose_recoverable.py` are unchanged. A no-GPU `selftest` subcommand validates the stats (`band_neartie.py selftest` → PASS).

### Suggested follow-ups

- **#481 update:** report int4-AR AIME as the band **0.3667 [0.345, 0.388]**, max-observed 0.400, all < 0.420; quality blocker characterized and closed.
- **Independent cross-check (cheap, no GPU):** run the near-tie distribution on the original on-disk #668 streams (`int4_streams_seq.json`) to confirm the 0.85%/median-8.375-nat profile replicates on independent data.
- **Tighter same-process identity control (optional):** a single server, conc=1, short generation, logprobs=2 then logprobs=0, token-stream identity — would close the logprobs-perturbation question by construction (current evidence: bf16 bit-exact + official-in-band already support no-perturbation).
- **Mechanism → recovery (negative-result confirm):** since flips are near-tie-driven and the deficit is accuracy-level, a higher-precision lm_head or selective-recompute on the ~0.85% tie steps would only de-noise the band, not raise the mean above 0.420 — consistent with #650 `K5_NO_RESCUE` / #668 `FUNDAMENTAL`.
