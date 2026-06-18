STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"wandb_run_ids":["8sfauo3i","uo6netrr","0206qiry","x6yyuglx","obfvs9ma"],"primary_metric":{"name":"tps_k_star_local_walltps","value":172.74},"test_metric":{"name":"ppl_k_star","value":2.0055},"k_star":5,"tps_k_star_local":172.74,"tps_k7_local":152.31,"k_star_beats_k7":true,"k_star_byte_exact_319":false,"ppl_k_star":2.0055,"verdict":"K_SWEEP_RECOVERS_TPS__BUT_SPEC_NOT_GREEDY_IDENTICAL"}

## Results

Two findings, one expected and one not:

- **(A) TPS — the asked question, answered:** the net-TPS-optimal drafter depth under BI=1 is **K\*=5**, netting **172.74 local wall_tps** — **+13.4% over the #623 K=7 anchor (152.31 ≈ 152.29)** and **+36.7% over the locked 126.378 rung**. My #623 follow-up #3 is confirmed: lowering K shrinks the BI tax (fewer M=1 BI-taxed draft forwards/cycle) faster than it loses acceptance.
- **(B) CRITICAL, unexpected — the byte-identity gate fails for the whole Option-B BI=1 spec lane, at *every* K.** The first strict greedy-token-identity gate ever run on this spec stack (instruction #3) shows it is **NOT token-identical to plain greedy AR** — K\*=5 and K=7 each diverge from the int4 M=1 AR reference on **~84% of prompts**. **PPL is unchanged (2.0055)**, so this is floating-point near-tie divergence in the speculative verify forward, **not** a quality regression — but `program.md` line 27-28 makes greedy-token-identity a **hard validity gate**, so this is material to whether Option-B can ever be a strict-#319 submission. It is **K-independent** (not introduced by the K-sweep; K=7 / the #623 152.29 anchor has it too).

**Verdict: `K_SWEEP_RECOVERS_TPS__BUT_SPEC_NOT_GREEDY_IDENTICAL`.** The TPS lever is real and the best #319-*shaped* operating point is K=5, but it does **not** yet feed an HF-Job approval packet because the greedy-identity gate is unresolved for the spec lane independent of K.

### 1. The K-sweep curve (BI=1 fixed, single-stream batch=1, 128×512, n=3 fresh servers/arm, median-of-3)

| K | net wall_tps (median) | mean ± std | e_accept (acc len) | acc/draft | cycle ms | M=1 draft-fwd/tok |
|---|---|---|---|---|---|---|
| 3 | 165.75 | 165.71 ± 0.07 | 2.856 | 0.619 | 17.23 | 1.050 |
| 4 | 171.72 | 171.68 ± 0.08 | 3.204 | 0.551 | 18.66 | 1.248 |
| **5** | **172.74** | **172.74 ± 0.01** | **3.474** | **0.495** | **20.11** | **1.439** |
| 6 | 170.21 | 170.18 ± 0.04 | 3.657 | 0.443 | 21.49 | 1.641 |
| 7 (anchor) | 152.31 | 152.26 ± 0.08 | 3.825 | 0.404 | 25.12 | 1.830 |

- **K\* = 5** by argmax median net wall_tps. The optimum is **interior and well-bracketed** (K=4 171.72 < K=5 172.74 > K=6 170.21), σ ≤ 0.08 TPS, so the hump is far beyond noise (K5−K4 = +1.0 TPS ≈ 13σ). K=2 was unnecessary: K=3 (165.75) already shows the left arm turning down, so the optimum cannot be < 4.
- **K=7 reproduces the #623 anchor exactly** (152.31 vs banked 152.29) — harness/engine stable across PRs.

### 2. Why a *lower* K nets MORE under BI=1 (instruction #2 — the mechanism)

Net TPS = `e_accept / cycle_time`. Two competing effects, from the marginal table:

| step | Δ e_accept (acceptance gained) | Δ cycle (ms, BI-taxed M=1 fwd) |
|---|---|---|
| K3→4 | +0.348 | +1.43 |
| K4→5 | +0.270 | +1.45 |
| K5→6 | +0.183 | +1.38 |
| **K6→7** | **+0.168** | **+3.63** |

- Each added draft token buys **less acceptance** (acc/draft falls 0.619 → 0.404) while costing **one M=1 BI-taxed forward** (~1.4 ms for K≤6; empirical fit `cycle_time(K)=1.860·K+11.22 ms`, draft/verify ratio 0.166).
- **The 7th draft token is a cliff**: 3.63 ms (2.5× the K4–6 marginal) for the *smallest* acceptance gain (+0.168) — that's why K=7 craters 170.21 → 152.31. (Drafter = gemma4 MTP head `/tmp/qat-assistant`; depth 7 hits a per-step cost discontinuity, later heads add almost no acceptance but still pay a full/super-full BI-taxed M=1 forward.)
- Cross-over: K=7→K=5 cuts cycle time −20% (25.12→20.11 ms) but acceptance only −9% (3.825→3.474). Cycle wins → K=5 nets more. This is the #623 −40% BI tax (which lives in the M=1 draft forwards) paid **fewer times per cycle**.

### 3. #319 byte-identity (instruction #3) — **FAILS, K-independently** ⚠️

Greedy capture (n=128, temp=0, `ignore_eos`, seed 1) vs a **BI=1 AR M=1 reference** (drafter OFF — confirmed `speculative_config=None`, `SENPAI_REFERENCE_MODE=1` forced `num_speculative_tokens=0`), served through the **same** `submissions/int4_mtp_batchinv` engine/venv (`/tmp/senpai-venvs/20f658587e8a6643`, vLLM 0.22.0) so the gate isolates speculation vs cross-engine FP noise. Verifier: the official `greedy_identity` module (`scripts/local_validation/greedy_gate.py`):

| comparison | verdict | identical / 128 | divergent tokens | onset min/median/max |
|---|---|---|---|---|
| K=7 (control) vs AR M=1 | **DIVERGENT** | 20 | 36,573 | 0 / 92 / 510 |
| **K\*=5 vs AR M=1** | **DIVERGENT** | 22 | 34,559 | 0 / 108 / 509 |
| K=5 vs K=7 (direct) | **DIVERGENT** | 20 | 33,234 | 0 / 134 / 509 |

- **Every pairwise comparison among {AR M=1, K=5, K=7} diverges at ~84%.** So a K change **does** perturb the exact greedy tokens (K=5 ≠ K=7), at the same rate either diverges from AR — instruction #3's "assert 0 flips" is **false on all counts**.
- **This is FP near-tie divergence in the spec verify forward, not a lossy bug:** first-token divergence is rare (1–3/128 onset=0); onsets spread broadly (buckets 1-50:~32, 51-150:~34, 151-300:~24, 301-511:~15) i.e. cascades that start when a near-tie argmax flips; the "safe" prompt sets only partially overlap across K (K5∩K7 = 12/30) → stochastic, not a fixed prompt property; and **PPL is bit-identical (§4)**. A lossy optimization diverges early and on nearly all prompts with a PPL hit — none of which we see.
- **Mechanism:** the verify forward processes **M=K+1 query positions in one call** (varlen seqlen_q>1); the M=1 AR forward processes one. `VLLM_BATCH_INVARIANT=1` makes reductions invariant to **batch size** (number of sequences), **not** to **query length M** — so the verify forward's per-position target logits differ from the AR forward's by FP rounding, and near-tie argmaxes flip. This is exactly the batch-invariant-≠-M-invariant gap (cf. the cuBLASLt/Triton M-invariance result: 15/15 M-inv is required, and the served attention/GEMM path here is not M-invariant under BI). The drafter is irrelevant — it only proposes; the divergence is in the **target's** verify logits.
- **Contract impact (`program.md` line 27-28):** "Greedy decode must remain token-identical to plain greedy autoregressive decode for the submitted checkpoint." A literal read ⇒ the Option-B BI=1 spec config is **invalid at every K**, despite passing PPL. This needs advisor adjudication (see follow-ups) — it is **not** specific to K\* and it retroactively qualifies the #623 "#319-candidate" label, which was based on BI determinism + PPL identity, never a greedy-token gate.

### 4. PPL sanity at K\* (instruction #4) — PASSES

Teacher-forced `prompt_logprobs=1, max_tokens=1` over the official 128-prompt `ppl_ground_truth_tokens.jsonl`, BI=1, K=5 (full spec stack):

- **PPL @ K\*=5 = 2.005501** vs ≤ 2.42 gate → **PASSES** with huge margin; vs #623 K=7 PPL 2.0055, **Δ = +1.0e-6** (bit-identical).
- Teacher-forced PPL reads only the **target** model's logprobs on ground-truth tokens; the drafter/K never enters the scored forward, so PPL is **K-independent by construction**. Its identity (2.0055) is the proof that §3's divergence is FP-near-tie, *not* a quality loss: the spec stack's continuations are equally-good greedy trajectories that simply differ from AR at near-ties.

### Config / commands

- **Submission:** `submissions/int4_mtp_batchinv` (int4 W4A16 `google/gemma-4-E4B-it-qat-w4a16-ct` + gemma4 MTP drafter `/tmp/qat-assistant`), vLLM 0.22.0 / dev307, `VLLM_BATCH_INVARIANT=1` fixed.
- **Sweep:** `research/walltps_ab/optionb_bi1_stock_int4/ksweep/run_k456.sh` → `scripts/profiler/paired_tps_ab.py`, only `NUM_SPECULATIVE_TOKENS` varied (K∈{3,4,5,6,7}); K=4,5,6 reuse the K=7 baseline (PR #72 restart-invariance). `--wandb_group optionb-bi1-k-sweep`, n=3, 128×512, seed 1, `GPU_MEMORY_UTILIZATION=0.90`.
- **Analysis:** `ksweep/analyze_ksweep.py`. **Finalize (AR ref + identity gates + PPL):** `ksweep/finalize_kstar.sh 5`. **W&B closeout:** `ksweep/log_ksweep_wandb.py`.
- **Peak memory:** **19,917 MiB** (~19.5 GiB; 0.90 of the 22.5 GiB A10G) — identical across K. No OOM.
- **Local only — no HF Job, no submission.** `analysis_only=true`, `official_tps=0`.

### W&B

- Closeout (curve Table + `net wall_tps vs K` line plot + decision + identity gates + PPL): **`8sfauo3i`** ([link](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/8sfauo3i)), group `optionb-bi1-k-sweep`.
- Per-K A/B runs: K=3 `0206qiry`, K=4 `x6yyuglx`, K=5 `uo6netrr`, K=6 `obfvs9ma` (K=7 anchor = the K=3 run's reused baseline arm).

### What happened — honest analysis

- **The TPS hypothesis is confirmed:** K\*=5 (172.74) is the BI=1 net-TPS optimum, +13.4% over 152.29 and +36.7% over 126.378, via exactly the #623-predicted mechanism (cut the count of M=1 BI-taxed draft forwards faster than acceptance falls), with a bonus surprise (the depth-7 cliff makes K=7 strictly bad).
- **But instruction #3 changed the story:** running the strict greedy-token gate for the first time shows the Option-B BI=1 **spec** stack is not token-identical to plain greedy AR at any K. I went in expecting "the verify step guarantees identity" (as the PR states) and found it does **not** — because BI=1 is batch-invariant, not M-invariant, and the M=K+1 verify forward FP-perturbs the target logits enough to flip near-tie argmaxes. PPL being bit-identical (2.0055) tells us it's quality-neutral, but the contract's gate is *token-identity*, not PPL.
- **Net:** the 172.74 number is a real, reproducible local speed result, but it is **not** a validated #319 operating point under a literal contract read — and neither was 152.29. The strict-#319-valid config remains the **AR locked rung (126.378)**; the entire Option-B BI=1 spec lane's #319 status is now an open question that supersedes the K choice.

### Suggested follow-ups

1. **Advisor adjudication (blocks everything):** does FP-near-tie spec divergence with **bit-identical PPL** violate `program.md` line 27-28, or does the gate's intent (catch *lossy* optimizations) treat an equally-good greedy trajectory as compliant? This decides whether Option-B spec is submittable at all. I did not assume either way — reporting the literal failure.
2. **If strict token-identity is required → make the verify forward M-invariant:** the lever is an M-invariant (query-length-invariant) GEMM + attention path so the M=K+1 verify logits bit-match M=1 AR. This is the same "batch-invariant GEMM tax" family from #623/§2 plus M-invariance — a real kernel lift, but the only way to get a strict-#319 *and* fast Option-B.
3. **Cheap diagnostic of the depth-7 cliff** (independent of the above): profile MTP per-head step cost K=6 vs 7 — CUDA-graph/padding shape boundary vs genuine compute. Not blocking (K=5 already wins on TPS), but explains the cliff.
4. **Do not spend HF-Job quota on Option-B spec** until #1 resolves — if token-identity is strict, the locked AR 126.378 stands and the spec-lane speed numbers (152.29 / 172.74) are off-leaderboard.
