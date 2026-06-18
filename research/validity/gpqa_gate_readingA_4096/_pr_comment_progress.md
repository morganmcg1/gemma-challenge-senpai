STUDENT kanna: progress + two transparency items (NOT terminal — sweep in flight)

Setting up the Option-B GPQA-D Reading-A sweep at the #631 dev307/conc=1 gate point, I hit two things worth flagging before I report numbers. Server is up and seed 12345 is running now (W&B `cr3c4y3q`, group `gpqa-gate-readingA-4096-kanna`).

## 1. Bug fix: mt=4096 collides with the #631 serve `model_len=6144` (fixed → 8192)

#631 served `--max-model-len 6144` with `max_tokens=3072` (max input that fits = 3072). PR #643's one deliberate change is `max_tokens 3072→4096`. That tips **1 GPQA-D item over the context cap**: idx 127 has **2429 input tokens**, and 2429 + 4096 = **6525 > 6144**. vLLM returns HTTP 400 for that item, and `score_on_error=True` scores it **wrong** — a ~0.5% headwind landing right at the bar (0.4864). I confirmed it empirically: a greedy run at 6144 had `n_error=1` (idx 127) with `accuracy=87/198`.

**Fix:** serve `--max-model-len 6144→8192`. The Gemma build supports 131072; 6525 < 8192 fits `max_tokens=4096` for **all 198** items (max input across the set = 2429). At conc=1 + BI=1, `model_len` changes only KV-cache capacity, not per-sequence compute, so this does not move the numerics for the 197 items that already fit. Collision verified cleared (idx 127 now 200, not 400). All seeds in this sweep run at 8192.

## 2. Determinism nuance: the gate point is ANSWER-near-deterministic, not byte-deterministic

The PR frames #631's conc=1 as "the near-deterministic gate point." I built a pre-sweep byte-determinism probe (`_determinism_check.py`: POST the same greedy item twice serially at conc=1, compare completions byte-for-byte). Finding:

| serve config | model_len | byte-divergence @ conc=1 |
|---|---|---|
| body-alone | 8192 | 5/5 probe items differ |
| body + drafter K=7 + BI=1 (Option-B) | 8192 | ~4/5 differ |
| body + drafter K=7 + BI=1 (Option-B, **unmodified**) | 6144 | 3/4 differ |

So the deployed Option-B config is **byte-NONdeterministic** run-to-run at conc=1 — **and so is body-alone — at both 6144 and 8192** (i.e. my model_len bump is NOT the cause; this is the intrinsic dev307 + int4-Marlin-GEMM + greedy A10G nondeterminism).

**This does not contradict #631.** #631's own `determinism_summary.json` (run `zk9zffp5`) reports `byte_identical=false` at the conc=1 gate point — it is "near-deterministic" at the **ANSWER** level (union of pairwise answer flips = **1/198**), not byte-identical. So the gate point's near-determinism was always answer-level, and my probe (stricter, byte-level) is consistent with it.

**Implication for the verdict (unchanged):** each `--seed` is a deterministic GPQA choice-shuffle, run once; the multi-seed t-CI captures between-seed variance and conservatively absorbs within-seed decode variance (it inflates the between-seed SD). The accuracy verdict is sound — I'm just reporting the byte-nondeterminism as a documented caveat rather than silently assuming byte-determinism.

## 3. Early positive signal on the cap-artifact hypothesis

First mt=4096 greedy reads show **finish_length_rate ≈ 3.5%** (down from #631's ~13% at mt=3072), and the 3-item smoke had 0/3 length-truncated (all stopped naturally). Consistent with PR #643's prediction that the ~13% is a 3072 **cap artifact**. I'll confirm at 8192 across seeds and back-derive the implied 3072 rate within the same runs.

## Status / plan

- Server: int4_g128_lmhead body + Gemma4-MTP K=7 drafter, BI=1, dev307, conc=1, model_len 8192. `analysis_only=true`, `official_tps=0`. LOCAL single A10G. No HF Job, no submission.
- Running 3 seeds × {greedy, sampled} (lewtun #31 sampled = temp 1.0 / top_p 0.95 / top_k 64), **chunked one-seed-per-invocation** to keep each background job under the 90-min per-run bound; fully resumable (skips completed result JSONs, resumes W&B by id).
- Will extend to 5 seeds if `SENPAI_TIMEOUT_MINUTES` budget allows, then post the terminal `SENPAI-RESULT` with `optionb_gpqa_sampled_mean` / `optionb_gpqa_greedy_mean`, CIs, `pct_of_base_sampled`, `finish_length_at_4096`, n_seeds, and the Reading-A verdict.

Keeping `status:wip` and continuing — no blocking question. Flag here if you'd rather I change the determinism handling or seed count.
