<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->
# Fast-kernel numerics tax (PR #540)

**Does the fast-kernel speed stack cost reasoning?** An OFFLINE, per-position divergence
mechanism analysis. fern #535 saw the served fast-kernel ship drop AIME `0.267 -> 0.167`
(n=30) below the `>=0.240` validity gate. This card answers the WHY/WHERE: is there a real,
concentrated per-position distributional tax that the fast-kernel stack levies on the native
262144-vocab int4 Gemma head — one that could flip answer tokens — or is fern's gap n=30
sampling noise (her own `delta=0.10` is `1.24 se`)?

`analysis_only=true`, `official_tps=0`. NO served-file change, NO HF job, NO submission. The
served config + baseline are UNCHANGED; the distribution is MEASURED, never altered.

## What the fast stack reduces to on the native head

The four levers, audited against `submissions/fa2sw_strict_m1ar_int4`:

| lever | effect on next-token logits | status |
|-------|-----------------------------|--------|
| bf16 compute | both arms load `dtype=bfloat16` | COMMON-MODE — cancels exactly |
| PLE fold | folds `embed_scale = sqrt(256) = 16.0` (a power of two) into bf16 weights; `x * 16.0` is exact in bf16 | BIT-IDENTICAL (and not even in the offline native path) |
| split-KV | a decode-time log-sum-exp KV reduction | exact up to fp associativity; DORMANT under teacher-forced prefill scoring |
| **surgical 2D attn** (`fa_sliding`) | swaps eligible sliding-window (head_dim=256) layers from the model's uniform `TRITON_ATTN` backend to FlashAttention | **the only lever that can move logits** |

vLLM forces uniform `TRITON_ATTN` on this heterogeneous-head-dim model
(`head_dim=256` local / `512` global) *"to prevent mixed-backend numerical divergence"* —
exactly the divergence `fa_sliding` re-introduces. **So on the native head the fast-kernel
tax == the `fa_sliding` backend swap.** We measure it.

## The as-run finding vs the as-designed counterfactual (blind-spot contrast)

`fa_sliding_patch` only swaps when `hf_config.model_type == "gemma4"`. The native int4
checkpoint's *text* config reports **`gemma4_text`**, so the guard never matches and the swap
is silently inert (`fa_flips = 0`). The as-run fast stack is therefore **bit-identical** to
plain on the native head. To prove the `TV = 0` we measure is a real property and not a blind
probe, a POSITIVE CONTROL arm relaxes the guard so the swap engages.

Four arms, all scoring the **same** base CoT trajectory (plain's own generation, seed 1234,
official sampling `T=1.0 / top_k=64 / top_p=0.95`):

| arm | wiring | role |
|-----|--------|------|
| **P** (`plain`) | native int4, default backend | generates + scores the base trajectory |
| **P'** (`plainB`) | plain, re-score | DETERMINISM / NOISE FLOOR — must be `TV=0` |
| **F_asrun** (`fast_asrun`) | fern recipe: `fa_sliding_patch`, `FA_SLIDING=1` | the as-run answer (inert by the `mt` guard) |
| **F_forced** (`fast_forced`) | `_forced_fa_patch.py`: guard relaxed to `startswith("gemma4")` | LATENT as-designed tax + probe sensitivity |

The contrast `P|P' = 0` **and** `P|F_asrun = 0` while `P|F_forced > 0` proves: the probe can
see a kernel tax, and the as-run stack has none **because the swap never fires** — not
because the measurement is blind. Each arm is a separate subprocess (the backend swap is a
global monkeypatch applied at `Attention.__init__`), so each gets its own real forward pass;
the offline `LLM` kwargs and prefill chunking are byte-identical across arms (so the int4
batch-variance confound is held fixed and the only difference is the patch).

## KEY OUTPUTS

From the as-run pair `P|F_asrun` (the question fern actually ran):

`mean_tv_plain_vs_fast`, `max_tv`, `tv_histogram` (diffuse|bimodal), `kernel_argmax_flip_rate`,
`answer_vs_filler_concentration`, `tax_locus` (diffuse|concentrated),
`kernel_numerics_tax` (real-concentrated|real-diffuse|negligible), `corroborates_fern_aime`
(yes|no), one-line verdict. Plus `F_forced` reported as the latent as-designed tax
(fa_flips, max TV, flip rate, locus). NaN-clean.

## Reproduce

GPU-free self-test (deterministic transform + locus + probe-sensitivity controls):

```bash
python research/validity/fast_kernel_divergence/fast_kernel_divergence.py --self-test
```

Full divergence run (orchestrator under the repo `.venv`; spawns the four arm subprocesses
under the submission server venv, `CUDA_VISIBLE_DEVICES=0`):

```bash
.venv/bin/python research/validity/fast_kernel_divergence/fast_kernel_divergence.py \
    --prompts research/validity/served_benchmark_divergence/prompts.jsonl \
    --n-new 320 --topk 256 \
    --wandb_name denken/fast-kernel-divergence \
    --wandb_group fast-kernel-divergence
```

Re-compose from cached arm captures (no GPU): add `--skip-gpu`.

## Files

- `fast_kernel_divergence.py` — orchestrator + per-arm GPU phase + CPU compose + self-test + W&B.
- `_forced_fa_patch.py` — the positive-control patch (forced `fa_sliding`, guard relaxed). Used
  ONLY by the `fast_forced` arm; never a served file.
- Reuses `research/validity/served_benchmark_divergence/prompts.jsonl` (135 MMLU-STEM + MATH +
  GSM8K proxy prompts) and the merged `gen_config / tv / kl` transform + `rc` model helpers.

## Result

**NEGLIGIBLE — the as-run fast-kernel stack is BIT-IDENTICAL to plain on the native int4
head; fern #535's AIME drop is n=30 noise.** W&B
[`1qpd61cb`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/1qpd61cb).
`analysis_only=true`, `official_tps=0`, no HF job / submission / served-file change.
n=41,962 positions across 135 prompts (MMLU-STEM + MATH + GSM8K), official sampling
`T=1.0 / top_k=64 / top_p=0.95`, seed 1234, peak 20.5 GiB, NaN-clean, self-test 7/7.

| pair | mean TV | max TV | argmax-flip | locus | n |
|------|--------:|-------:|------------:|-------|--:|
| **P\|P'** (determinism floor) | 0 | 0 | 0 | — | 41962 |
| **P\|F_asrun** (the as-run answer) | **0** | **0** | **0** | diffuse | 41962 |
| P\|F_forced (positive control, guard relaxed) | 3.28e-3 | 0.185 | 3.17e-3 (133) | diffuse | 41962 |

- `mean_tv_plain_vs_fast = 0`, `max_tv = 0`, `kernel_argmax_flip_rate = 0`. TV histogram:
  **all 41,962 positions in the `0` bin** — not diffuse-small, *exactly zero*. Contrast
  #529 base│ship (a `-inf` mask): mean TV 0.066, 6.5% killed, bimodal, ~27× answer-concentrated.
- `answer_vs_filler_concentration = n/a` (no divergence to concentrate), `tax_locus = diffuse`,
  `kernel_numerics_tax = negligible`, `corroborates_fern_aime = NO`.
- **Mechanism:** the as-run `fa_sliding` swap NEVER fires — native text config
  `model_type = "gemma4_text"` ≠ patch guard `"gemma4"` → `fa_flips = 0`. PLE-fold (×16.0
  power-of-two, exact in bf16), bf16 (common-mode), split-KV (dormant under teacher-forced
  prefill) contribute nothing. The as-run fast stack is bit-identical.
- **Probe is provably sensitive (positive control F_forced):** relax the guard to
  `startswith("gemma4")` → swap fires on **18 eligible sliding layers** (eligibility-drift=0)
  → max TV 0.185, 133 flips (0.317%). So `TV=0` is a real property, not a blind probe. Even
  this *latent as-designed* tax is small + **diffuse** (no position > 0.19; every flip a
  near-50/50 tie reorder) and **digit/answer tokens are the LEAST affected** (digit mean TV
  4.3e-4, flip 2.2e-4) — it would not explain a 0.10 AIME collapse either.

**Verdict:** *the fast kernels DO NOT cost reasoning on the native head; fern's AIME drop is
n=30 noise.*
