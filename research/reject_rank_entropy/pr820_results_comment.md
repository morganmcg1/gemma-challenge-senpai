STUDENT fern:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["0r80mau9"],"decision":"GO","b_dominates_a":false,"recommended_variant":"uniform_topk","recommended_k_start":2,"primary_metric":{"name":"projected_best_uniform_tps","value":415.82},"test_metric":{"name":"measured_strict_E_accept","value":3.380}}

## Results

**Viability gate verdict: GO for uniform top-k (A) — stark #816. FLy entropy-gating (B) does NOT dominate; it is counterproductive on this model.** This is a 0-submission, local-only profiling + projection (no kernel change, no HF job), as instructed.

| | answer |
|---|---|
| Does the lever clear a meaningful gain over ~253? | **YES** — projected **+19.9%** (k=2 → 303 TPS) up to **+64.4%** (k=16 → 416 TPS) |
| Does entropy-gating (B) dominate uniform (A)? | **NO** — no threshold keeps ≥90% of uniform's gain while removing risky relaxes |
| Which variant to pursue? | **Uniform top-k (A)**, start at **k=2**; entropy is the *wrong* gating signal here |
| Recommended `H_thresh`? | **None** — FLy's premise is inverted on Gemma's peaked verifier (see below) |

### Method + faithfulness validation
Env-gated probe (`REJECTRANK_ENABLE=1`, default OFF) wraps `RejectionSampler.forward` in the shipped `int4_mtp_bi0_int4head` submission, reads the full-vocab verifier logits the sampler already computed, and logs per draft position: strict accept (`draft==argmax`), 1-indexed **rank** of the draft token, verifier **entropy** `H=-Σp·logp`, draft prob mass, argmax prob. It only ADDS logging and calls the original `forward` unchanged → greedy tokens stay byte-identical to production. Drove the **official 128-prompt greedy decode** (conc=1, temp=0, `ignore_eos`, output_len 512, seed 1) → **19,538 verify blocks / 117,228 draft positions**, vocab 262,144.

**The probe is output-neutral, proven three ways (all agree within 0.01):**

| source | E_accept | accept-rate r | note |
|---|---|---|---|
| PR baseline (`7ntx4nrn`) | 3.379 | 0.397 | target |
| probe records (this run) | **3.3800** | **0.3967** | self-consistency ✓ |
| vLLM server-log counters (37 intervals, independent) | **3.3711** | **0.3952** | 45409/114906 drafted ✓ |

> ⚠️ The probe run's **wall TPS = 174.8** (65,536 tok / 375 s) is *depressed by the probe's per-step full-262k-vocab softmax + entropy + CPU-sync + JSONL write* — it is **NOT a TPS measurement**. The acceptance *structure* (E_accept, ranks, entropy) is faithful (matches baseline above); TPS is read off stark's independent oracle curve, never from this run.

Oracle TPS-vs-acceptance-length fit (stark `bi0-int4head-accept-oracle`): `TPS = 43.60 + 61.37·L`, **R²=0.9989**, anchors L∈[4.355, 7.0]. At measured strict E_accept=3.380 it gives 251.0 TPS (≈ the ~253 baseline; gains below are vs 253).

### (2a) Reject-rank CDF — fraction of strict rejects with draft at verifier-rank ≤ k
n = 40,638 reject positions.

| subset | ≤2 | ≤3 | ≤5 | ≤8 | ≤16 |
|---|---|---|---|---|---|
| all rejects | 0.323 | 0.475 | 0.618 | 0.719 | 0.823 |

→ ~32% of rejects are the verifier's **2nd choice** (draft just missed argmax); ~62% are rank≤5; ~18% sit beyond rank 16 (genuinely wrong, never rescued by top-k).

### (2b) Entropy-conditioned reject-rank CDF — the FLy-killer
Verifier entropy is tiny: **all-position median = 0.202 nats** (p90 1.643, max 4.906); **reject-position median = 1.015 nats**. FLy's normalized θ=0.3 → **3.74 nats**, above essentially the entire distribution (only **15 of 40,638** rejects exceed it) → FLy literal is **inert** (retention −3.9%, near-strict).

Splitting the reject-rank CDF by entropy shows the rescuable mass is at **LOW** entropy — the **opposite** of FLy's premise:

| H_thresh | low-H (H≤thr) rank≤2 | high-H (H>thr) rank≤2 | low-H n | high-H n |
|---|---|---|---|---|
| 1.0 nats | **0.395** | 0.254 | 20,036 | 20,602 |
| 2.0 nats | **0.348** | 0.151 | 35,557 | 5,081 |

FLy relaxes **only at high entropy** — but high-entropy reject positions are where the draft is *least* likely to be a cheap rank-2 rescue (the verifier is flat, so the draft scatters to high ranks). FLy refuses the abundant low-entropy rank-2 near-misses that carry the speed.

### (3) Projected E_accept(k) / TPS(k)

**(A) Uniform top-k** (accept if rank≤k everywhere):

| k | E_accept | TPS | Δ% vs 253 | non-argmax tokens emitted (quality surface) |
|---|---|---|---|---|
| 2 | 4.234 | **303.4** | +19.9% | 7,793 |
| 3 | 4.713 | 332.8 | +31.5% | 12,772 |
| 5 | 5.219 | 363.9 | +43.8% | 18,360 |
| 8 | 5.616 | 388.2 | +53.5% | 22,978 |
| 16 | 6.065 | **415.8** | +64.4% | 28,419 |

Within-block replay is **exactly faithful** (vLLM computes all K draft-position logits in one teacher-forced pass, so rank/entropy at depth d is fixed regardless of accept decisions at earlier depths); the only approximation is cross-block context drift — the same assumption stark's oracle curve makes.

**(B) Entropy-gated top-k** (relax to top-k only where H>H_thresh, keep strict argmax where confident) vs (A), with the gate threshold set equal to the risk boundary:

| k | H_thresh | TPS_A | TPS_B | retention of A's gain | risky low-H relaxes B refuses | safe high-H relaxes |
|---|---|---|---|---|---|---|
| 2 | 1.0 | 303.4 | 265.5 | 25% | 5,104 | 2,689 |
| 16 | 1.0 | 415.8 | 305.7 | 32% | 14,833 | 13,586 |
| 16 | 0.10 (lowest) | 415.8 | 387.6 | 83% | only 3,249 of 28,419 removed | — |

**Recommended gated point: NONE.** To keep ≥83% of the gain you must set H_thresh so low it eliminates only ~12% of the relaxes (it barely gates); to actually remove the "risky" low-entropy relaxes you sacrifice 68–75% of the speed. The speed lives in the low-entropy region.

### (3b) Quality proxy
At the reject-position median (H=1.0 nats), **52–65% of uniform's relaxed-accepts land at LOW (confident) verifier entropy** — the positions FLy labels risky (k=2: 5,104/7,793; k=16: 14,833/28,419). The quality cost of accepting these is **NOT evaluated here** (no eval was run, per scope) and **cannot be projected** — it must be measured by stark #816 on the quality panel (PPL≤2.42, AIME≥0.090, MMLU-Pro≥0.572, GPQA≥0.471, GSM8K≥0.807, 128/128). Note many low-entropy rank-2 rejects may be benign *near-ties* (two near-interchangeable tokens in a peaked distribution), which only a real grade can disambiguate from genuine errors.

### Go/no-go + recommendation
- **GO** on the uniform-k acceptance-relaxation lever (stark #816). The lever is real and large (+20% to +64% projected), and it directly attacks the binding constraint (r≈0.397 vs oracle 1.0).
- **Pursue uniform top-k (A), NOT FLy entropy-gating (B).** On Gemma's peaked 262k verifier, entropy **anti-correlates** with rescuability, so gating refuses the cheap rank-2 rescues and is strictly worse at every (k, H_thresh).
- **Start the kernel at k=2** (+19.9% → 303 TPS, smallest quality exposure: 7,793 non-argmax emissions, rescuing 32% of rejects). The gain-vs-relaxes tradeoff is smooth (no free knee — each ~5k extra non-argmax emissions buys ~+10–12%), so k is a pure speed↔quality dial the quality panel must set. If quality holds at k=2, climb k.

### Repro
```bash
# capture (GPU, env-gated probe; serves shipped int4_mtp_bi0_int4head):
python -m scripts.profiler.reject_rank_capture --num-prompts 128 --output-len 512 \
  --out-dir research/reject_rank_entropy/int4head
# offline projection (0-GPU; re-runnable without re-serving):
python -m scripts.profiler.reject_rank_project --in-dir research/reject_rank_entropy/int4head \
  --wandb-group bi0-reject-rank-entropy
```
- **Peak GPU:** ~19.7 GiB on the 23 GiB A10G (model 10.22 GiB + KV cache 8.1 GiB; probe transient tensors negligible). GPU_MEMORY_UTILIZATION=0.90.
- **W&B:** run [`0r80mau9`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/0r80mau9), group `bi0-reject-rank-entropy` (full projection JSON uploaded as artifact).
- **Artifacts:** `research/reject_rank_entropy/int4head/{reject_rank_projection.md,reject_rank_projection.json,capture.out,decode.summary.json,server.log}`.

### What happened
The acceptance frontier is recoverable cheaply: a third of all rejects are rank-2 near-misses, so even k=2 buys ~+20%. **But the FLy hypothesis fails on this substrate for a measured, structural reason** — Gemma-4-E4B's verifier is extremely peaked (median 0.202 nats), so a "low-entropy reject" is overwhelmingly a confident distribution where the int4 drafter proposed a high-logit rank-2 alternative, while genuine high-entropy ambiguity is rare *and* rank-scattered. FLy's "relax only where uncertain" therefore targets the wrong positions. The result cleanly **de-risks the team's fork**: build the simple uniform-k kernel (stark #816), don't spend effort on an entropy-gated/FLy lane. The remaining open question is purely the quality cost of uniform-k, which this projection deliberately does not (and cannot) answer.

### Suggested follow-ups
1. **stark #816 should measure the quality panel at k=2 first** (smallest exposure), then climb k while quality holds — there is no free knee, so the ship-k is whatever the panel tolerates.
2. **If a quality-safe gate is still wanted, gate on logit/prob MARGIN, not entropy.** I already capture `dp` (draft prob) and `ap` (argmax prob) per position; accept a rank-2 token only when `dp/ap` ≈ 1 (a true near-tie) — this directly separates benign near-ties from real errors, which entropy cannot do on a peaked verifier. (Not implemented — flagging per scope.)
3. The tail beyond rank-16 (~18% of rejects) is unrescuable by any top-k; those need a better drafter, consistent with the K-sweep finding that depth is exhausted at K=6.

### Bug fix (separate, please review)
While wiring W&B logging I found `scripts/profiler/reject_rank_project.py::_maybe_log_wandb` (my own new script for this PR) called the `scripts/wandb_logging` helpers with **stale signatures** — `flatten_numeric(summary)` (missing the required `prefix`), `log_summary(run, …)` (missing required `step=`), and `log_json_artifact(run, r, name=…)` (wrong arg shape; missing `artifact_type=`/`data=`). It only "passed" before because in a wandb-less venv `init_wandb_run` returns None and the calls short-circuit; under system python (which has wandb) it would have raised `TypeError`. Fixed to match the current API and enriched the logged summary (per-k TPS/E_accept/gain/relaxes + reject-rank CDF). Verified end-to-end (run `0r80mau9`).

### Public evidence used
- **Leaderboard frontier ≈ 506–514 TPS** (digest as=senpai, 2026-06-20: rank-1 `w160-ctk42-noprecache-gemma-slayer-lean` @ 513.8) vs our **shipped bi0 @ 218.02** (`s63tb03x`, #769) → the +20–64% acceptance lever is materially relevant to closing that gap.
- **FLy**: arXiv:2511.22972 (ICLR 2026), GitHub AMD-AGI/FLy — entropy-gated top-k acceptance; its high-entropy-only relaxation premise is what this gate tests and finds inverted on Gemma.
- Baselines from the PR body: int4head `7ntx4nrn` (E_accept 3.379 / r 0.397), oracle `j60r68os`/`zc76n7xz`.
