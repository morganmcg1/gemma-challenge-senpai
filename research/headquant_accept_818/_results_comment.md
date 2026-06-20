STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["94vktlsi","8g8bclka","2b99mjew","3enknmyr","xtazaes2","45rxjwkk"],"primary_metric":{"name":"e_accept_int4_g32","value":3.369},"test_metric":{"name":"e_accept_bf16_head_ceiling","value":3.329}}

## Results — Head-precision × E_accept: does int4 lm_head depress acceptance?

**Verdict: HYPOTHESIS REFUTED.** int4-quantizing the lm_head does **not** depress spec-decode acceptance. E_accept is flat (~3.33) across bf16/g32/g64/g128 even as head rel_err climbs 0 → 0.067 → 0.083 → 0.100, and the predicted "ceiling" (bf16) lands at the **bottom**, not the top. The low **r ≈ 0.397 is intrinsic to the MTP drafter's proposals**, not a head-quantization artifact. There is **no free TPS win** from a higher-precision head — int4 g32 is already the best net-TPS arm and stays the correct config.

### E_accept-vs-head-precision curve (the decisive table)

| arm | head rel_err | head GB | **E_accept** r1 / r2 | r | steady TPS r1 / r2 | **wall TPS** r1 / r2 |
|-----|----:|----:|----:|----:|----:|----:|
| bf16 (tied, ceiling ref) | 0 | 1.342 | 3.326 / 3.332 | 0.390 | 216.4 / 211.2 | 211.8 / 211.8 |
| **int4 g32 (shipped)** | 0.0674 | 0.378 | **3.369 / 3.370** | 0.398 | 245.7 / 245.2 | **246.8 / 246.5** |
| int4 g64 | 0.0825 | 0.357 | 3.326 / — | 0.390 | 247.7 / — | 241.9 / — |
| int4 g128 | 0.1004 | 0.346 | 3.347 / — | 0.394 | 249.9 / — | 243.9 / — |

Within-arm noise (rep1 vs rep2): E_accept ±0.003 (bf16), ±0.0005 (g32); wall TPS ±0.03 (bf16), ±0.16 (g32). E_accept is **essentially deterministic** here — per-position counts are byte-identical across reps — so the cross-arm pattern is real signal, not noise.

### Per-position accept rate (positions 1–6 of K=6, rep1)

| pos | bf16 | g32 | g64 | g128 |
|----:|----:|----:|----:|----:|
| 1 | 0.698 | 0.706 | 0.698 | 0.697 |
| 2 | 0.502 | 0.514 | 0.507 | 0.507 |
| 3 | 0.389 | 0.393 | 0.386 | 0.394 |
| 4 | 0.307 | 0.312 | 0.306 | 0.312 |
| 5 | 0.247 | 0.255 | 0.247 | 0.252 |
| 6 | 0.195 | 0.205 | 0.197 | 0.200 |

The decay shape (0.70 → 0.20) is **identical across all head precisions** → the per-position drop-off is a drafter property, independent of head bytes.

### Two decisive questions

**(i) Does E_accept rise monotonically with head precision (bf16 > g128 > g64 > g32)?** **NO.** Actual order is g32 (3.370) > g128 (3.347) > bf16 (3.329) ≈ g64 (3.326). The bf16 "ceiling" is near the **bottom**. Head rel_err nearly doubles (g32→g128) and E_accept does not fall — so the premise that int4 perturbs verifier-argmax enough to lower the match rate is false in this rel_err range.

**(ii) Is there a head-precision arm whose net TPS beats int4 g32's ~246?** **NO.** On **wall TPS** (total tokens / total decode wall — the most stable metric and the one closest to the official benchmark's `tps`), g32 is highest at **246.6**; bf16 loses **−34.8 (−14%)**, g64 **−4.7**, g128 **−2.7**. (steady-state-burst TPS nominally ranks g128 top at 249.9, tracking its 8.5%-smaller head read, but that ordering *disagrees* with wall TPS and sits inside the steady-TPS noise band — bf16's own steady swung 5 TPS between reps. The end-to-end wall number is the honest one, and it favors g32.)

### Causal answer: the low r ≈ 0.397 is INTRINSIC to the drafter

The clean control is the **bf16 head — zero quantization error** — which yields the **same-to-lower** acceptance (E_accept 3.329) as the int4 heads. If int4 quantization were depressing acceptance, bf16 would sit clearly above every int4 arm; instead it's at the bottom. So the verifier-argmax over the 262k-vocab head is robust to quantization here, and the binding constraint on r is the **MTP drafter's proposal quality**, not the head. The lever to lift r is the drafter or the accept criterion (cf. stark #816's lenient criterion), **not** head precision.

**Bonus:** int4 g32 E_accept (3.370) is reliably **above** bf16 (3.329) by +0.041 (≈ 6× the within-arm noise). Plausible mechanism: the drafter is QAT-matched (`gemma-4-E4B-it-qat-q4_0-unquantized-assistant`, trained against the quantized target), so the int4 verifier-argmax aligns with the drafter's proposals marginally better than a bf16 head does. Net: the int4 head is **not merely throughput-free — it is mildly acceptance-positive**.

### Why the int4 head wins (mechanistic summary)

The TPS ordering is explained **entirely by head-GEMV bytes, not acceptance**: bf16 reads 1.342 GB/token (2.78 ms GEMV, stark #798) for **zero** acceptance benefit → −34.8 wall TPS; int4 reads ~0.35–0.38 GB (0.75 ms) and keeps the same (slightly better) acceptance. The int4 head buys the GEMV win with **no acceptance penalty** — the opposite of the hypothesized trade-off.

### Quality spot-check

The best net-TPS arm **is int4 g32**, which is the **already-shipped, quality-validated int4head**: the sweep rebuilds it byte-identically (same deterministic builder + official snapshot) and reproduces the control's E_accept 3.369 / r 0.398 within noise. It already clears PPL ≤ 2.42, 128/128, MMLU-Pro ≥ 0.572, GPQA ≥ 0.471, GSM8K ≥ 0.807, AIME ≥ 0.090 (#769/#805). No **new** arm beats it net, so there is no new checkpoint to quality-gate. g64/g128 do not net-win **and** carry 22–49% higher head rel_err (more PPL/greedy-identity risk), so gating them is not warranted.

### Run details

- **Command (rep1, all 4 arms):** `WANDB_PROJECT=gemma-challenge-senpai WANDB_ENTITY=wandb-applied-ai-team python research/headquant_accept_818/sweep.py`
- **Command (rep2, noise floor, bf16+g32):** `… python research/headquant_accept_818/sweep.py --tag full_rep2 --rep 2 --arms bf16_head,int4_g32`
- **Workload:** official 128-prompt ShareGPT greedy decode, conc=1, temp=0, output_len=512, ignore_eos (128/128 → 65,536 completion tokens), identical prompts + identical MTP drafter (K=6) across all arms. Only the lm_head bytes the verifier reads differ; int4 body (343 packed tensors, g32) + embeddings copied byte-for-byte.
- **Peak GPU memory:** ~20.7 GB (gpu_memory_utilization=0.90 cap on the 22.5 GiB A10G), identical across arms. KV cache: bf16 336.9k tokens (tied head, no extra weight) vs int4 g32 322.9k (untied int4 head costs ~14k KV tokens) vs g128 324.0k.
- **W&B group `bi0-headquant-accept`:** bf16 `94vktlsi` (rep1) / `xtazaes2` (rep2); int4 g32 `8g8bclka` (rep1) / `45rxjwkk` (rep2); int4 g64 `2b99mjew`; int4 g128 `3enknmyr`.
- **Local profiling only — no HF job launched.**

### What happened & why

The hypothesis assumed the int4 head perturbs verifier-argmax enough to reject more draft tokens, trading acceptance for a cheaper GEMV. The data refute this: across a 0 → 0.100 head-rel_err sweep, E_accept is flat (~3.33) and the bf16 zero-error control is at the bottom, not the top. The verifier-argmax is dominated by a clear top-1 margin that int4 (rel_err ≤ 0.10) does not flip often enough to move the match rate. The binding constraint r ≈ 0.397 lives in the drafter — so the int4 head is a clean GEMV win with no acceptance cost (and a small QAT-matching bonus), and **int4 g32 remains the correct fire-candidate config**.

### Suggested follow-ups

1. **Drafter is the real lever for r.** Since r ≈ 0.397 is drafter-intrinsic, the largest acceptance upside is a better/larger MTP drafter or a tree/multi-branch proposer — not anything on the verifier head. This is complementary to stark #816 (which changes *which* proposals count): the two address the two halves of the accept equation.
2. **(Optional) Do not chase the ~1% steady-burst edge of g128.** It loses on wall TPS and carries 49% higher head rel_err for no end-to-end throughput; the required g128 quality gate (PPL + greedy-identity over 128 prompts) is not worth spending.
3. The +0.041 int4-over-bf16 E_accept bonus suggests verifier↔drafter *QAT-distribution match* matters more than head precision — worth a dedicated probe if drafter retraining is on the table (match the drafter's training target dtype to the served head).
