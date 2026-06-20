STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["tvxku5vw","qviiadib","49jrlpor","zmuc7v32"],"primary_metric":{"name":"local_decode_wall_tps_byte_identical","value":217.63},"test_metric":{"name":"mean_acceptance_length_byte_identical","value":3.336}}

## Results — NO FIRE (clean null on the primary lever; byte-identity premise REFUTED)

**Verdict: the primary greedy-safe lever does not fire, and it additionally refutes the PR's output-neutrality premise.** Widening the MTP drafter's `centroid_intermediate_top_k` (32→64→128) (a) moves acceptance only marginally (+1.5% E_accept), (b) leaves decode TPS flat within run-to-run noise, and (c) **breaks byte-identity** on ~7% of prompts — the red flag you flagged, now investigated and explained.

### Acceptance / TPS curve (K=6 fixed, MAX_NUM_SEQS=1, 128×512 conc=1 official workload)

| centroid_top_k | wall_tps | E_accept | accept_rate | cycle_ms | steady_tps | PPL | 128/128 | ==control |
|---|---|---|---|---|---|---|---|---|
| **32** (native = shipped bi0 = control) | **217.63** | 3.3364 | 0.3894 | 15.331 | 210.74 | 2.0053 | ✅ | **128/128 ✅** |
| 64 | 219.63 | 3.3758 | 0.3960 | 15.370 | 218.45 | 2.0053 | ✅ | 119/128 ❌ |
| 128 | 217.58 | 3.3857¹ | 0.3976¹ | 15.561 | 210.53 | 2.0053 | ✅ | 118/128 ❌ |

¹ ctk128's driver died before inline finalize; values above are the original inline-computed run (W&B `49jrlpor`). A post-hoc rebuild from on-disk logs gave E_accept 3.391 / accept 0.3985 — within rounding, conclusion unchanged.

- **Acceptance moves, barely:** E_accept 3.336 → 3.386 (**+1.5%**), accept_rate 0.3894 → 0.3976 (+2.1% rel). Monotone but tiny, with diminishing returns.
- **TPS is flat:** wall_tps 217.63 / 219.63 / 217.58. The lone nominal bump (ctk64, +0.9%) is within run-to-run noise **and** breaks identity. Reason TPS doesn't follow acceptance: `cycle_wall_ms` rises monotonically (15.33 → 15.37 → 15.56) — the wider centroid candidate set makes the latency-bound drafter step slightly costlier per #786 (drafter is 17.2% of GPU), and that cost eats the marginal acceptance gain. Net ≈ zero.
- **Control reproduces the #786 anchor:** steady 210.7 vs 210.1 TPS, accept 0.389 vs 0.386, E_accept 3.34 vs 3.31. Setup validated.

### Greedy-identity break — investigated (the red flag)
ctk64/128 are **not** byte-identical to the shipped ctk32 control: 9/128 (ctk64) and 10/128 (ctk128) prompts diverge. I ran a **determinism control** (`repeat_det.py`) to separate the two candidate causes — re-ran the ctk32 control against itself (same drafter, same env, same workload):

> **ctk32b vs ctk32 = 128/128 identical → STACK DETERMINISTIC given the drafter.** The divergences are **genuinely drafter-induced (cause A)**, not run-to-run nondeterminism. (W&B `zmuc7v32`.)

Divergence localization (`analyze_identity.py`):
- **Deep, not at the start:** first-divergence token indices 285–504 (median ~370) of 512. Sequences are byte-identical for hundreds of tokens, then a single near-tie flips and cascades.
- **Concentrated in reasoning:** 5 gpqa_diamond + 4–5 mmlu_pro; **zero aime** divergences.
- **A fixed set of knife's-edge positions:** the first-divergence index is nearly identical between ctk64 and ctk128 on shared prompts (e.g. 338/338, 367/367, 285/285, 447/447) — i.e. *any* drafter perturbation trips the *same* handful of positions.

**Mechanism.** bi0 serves `VLLM_BATCH_INVARIANT=0` — only attention is surgically byte-exact (force-2D patch); the Marlin W4A16 GEMM / RMSNorm are **not** batch-invariant (the un-taxed-matmul residual flip source, cf. #500/#496 census: ~5 matmul-induced ULP ties). The shipped drafter (ctk32) is validated byte-exact to plain AR, but those ~5–10 logit positions sit within ~1 ULP of a tie. Greedy spec-decode is output-neutral **only in exact arithmetic**; on bi0's deliberately-non-BI kernels, changing the drafter's proposals perturbs the verify-GEMM numerics just enough to flip those near-ties → the drafter leaks into the emitted sequence. **Corollary: the only byte-safe drafter config on bi0 is the exact shipped one.** You can't tune the bi0 drafter without either paying the full batch-invariance tax (which kills bi0's speed edge) or re-validating the new drafter+target as a byte-exact unit.

**PPL is decode-path-blind.** PPL stayed *exactly* 2.0053 across all three configs while the decoded output diverged — because PPL is teacher-forced over fixed ground-truth tokens and never runs the drafter. Only the greedy-identity (byte-exact decode) gate catches this drift. (Directly answers the open board question "@senpai is there a separate validation other than perplexity?" — yes, and PPL alone is insufficient for spec-decode submissions.)

### Gates
| Gate | Result |
|---|---|
| local TPS > control (proj > 218.02 official) | ❌ best byte-identical config (ctk32) = 217.63 = shipped; only ctk64 is nominally higher (+0.9%, within noise) and it fails identity |
| greedy-identical OR panel-certified | ❌ only control (ctk32) is identical; variants are drafter-induced-divergent; secondary panel not pursued (see follow-ups) |
| PPL ≤ 2.42, 128/128 | ✅ PPL 2.0053, 128/128 (all configs) |

No config clears all gates → **no fire.** This is the "moves but costs TPS / won't move usefully" clean null you described.

### Public evidence used
- **Leaderboard** (digest `as=senpai`): rank 1–6 are all ~505–506 TPS using `centroid_top_k` **44–48** (`w192-ctk48`, `vidraft…ctk44`, `osoi5…ctk44`) on the frontier **w192** stack; akira's inbox note confirms the private-stable frontier is `W192 + CENTROID_TOP_K=44 + noprecache`. This independently corroborates that *this drafter's* acceptance optimum sits ~44–64 (matches stark #786: "64 optimal, topk128 = −3.9 TPS" on fa2sw). My bi0 sweep is consistent (ctk64 best wall TPS, ctk128 flat-to-worse) — but on **bi0's byte-exact stack even ctk64 breaks the identity gate**, so the frontier's ctk44–48 optimum cannot be byte-safely ported here.
- Inherited axis from denken **#783**; adjacent to **#787** spec-verify internals; control anchor **#786**; quality anchor **#773**.

### Commands
```bash
# Primary sweep (stages /tmp drafters per top_k, never edits the shipped submission):
cd target && .venv/bin/python -m research.bi0_mtp_accept.sweep 32 64 128
# Determinism control (re-run ctk32 vs itself):
cd target && .venv/bin/python -m research.bi0_mtp_accept.repeat_det
```
Serve config (per point): `MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct`, `VLLM_BATCH_INVARIANT=0`, `DRAFTER_MODEL=/tmp/drafter_ctk{K}` (config.json `centroid_intermediate_top_k` patched, weights symlinked from HF cache), `NUM_SPECULATIVE_TOKENS=6`, `MAX_MODEL_LEN=4096`, `GPU_MEMORY_UTILIZATION=0.90`, `MAX_NUM_BATCHED_TOKENS=512`, `MAX_NUM_SEQS=1`. Decode: official `eval_prompts_sharegpt.json`, 128×512, seed 1. PPL: official `ppl_ground_truth_tokens.jsonl`. **LOCAL ONLY — no HF job.**

### Peak memory
~19.7 GiB on the A10G (23 GB) at `--gpu-memory-utilization 0.90`; EngineCore RSS 19,664 MiB during serve. GPU confirmed clean (0 MiB) before and after; no fleet leak (#780 check passed).

### W&B (group `bi0-mtp-accept`)
`tvxku5vw` (ctk32) · `qviiadib` (ctk64) · `49jrlpor` (ctk128) · `zmuc7v32` (determinism control). [project](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai)

### What happened
The EV thesis (amortize the M=7 verify GEMM by lifting acceptance) is sound, but the **drafter's proposal quality — not the acceptance criterion — is the binding constraint**, and `centroid_top_k` is too weak a lever to move it: the candidate-set widening that buys acceptance also slows the drafter, and the net is zero TPS. Worse, on bi0's byte-exact (non-BI) stack the lever isn't even free — *any* drafter change forfeits the strict greedy-identity gate via a fixed set of reasoning-prompt ULP near-ties. The acceptance/TPS axis on bi0 via runtime drafter knobs is exhausted.

### Suggested follow-ups
1. **Secondary (relaxed acceptance) — assessed, low EV; recommend NOT pursuing as a quick win.** A research pass confirms: vLLM 0.22.0 has a `TypicalAcceptanceSampler` but **hard-routes to the strict `RejectionSampler` at temp=0**, so relaxed greedy acceptance needs a ~5–10 line patch to the sampler-selection branch in `spec_decode_worker.py` (it is *not* an exposed `--speculative-config` knob). EV is low because (a) my primary already shows drafter proposal quality is the binding constraint, so realistic threshold relaxation adds only ~+0.2–0.5 acceptance length (~+8–13% TPS, far below the +35% a real win needs); (b) the eval is 100% reasoning (mmlu_pro 57 / gpqa 57 / aime 14) and reasoning chains are fragile to non-argmax tokens, putting the 5% band on GPQA/AIME at high risk; (c) it converts bi0 from byte-exact to approximate-greedy and *requires* the full MMLU-Pro/GSM8K/AIME/GPQA panel to certify. **Recommend scoping as its own PR only if you want to explore the byte-exact→approximate-greedy trade** — with an entropy-gated threshold (accept non-argmax only where target entropy is low) as the safest first parameterization.
2. **The real acceptance lever is a better-trained drafter, not a runtime knob** (consistent with #282/#532). That's a cluster-training request, not local work — happy to draft the issue if you want to pursue E[T] on bi0.
3. **Byte-exact tuning headroom is structurally blocked on bi0.** If the program wants to keep the strict identity gate *and* tune the drafter, the only route is the full batch-invariance tax (forfeits bi0's speed edge) or unit re-validation of a new drafter+target pair.
