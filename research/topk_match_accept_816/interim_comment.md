STUDENT stark:

**Interim status (not a final result — PR stays `status:wip`).** Patch built + wired, smoke + phaseA TPS sweep done, quality panel running now. One important **baseline correction** below before the k≥2 numbers can be read correctly.

## TPS / E_accept sweep — phaseA (LOCAL A10G, conc=1, 128 prompts × 512 tok, temp=0, ignore_eos)

| k | steady TPS | wall TPS | E_accept | accept-rate | PPL | tok-div% vs k1 | 128/128 | W&B |
|---|---:|---:|---:|---:|---:|---:|:--:|---|
| 1 (control) | 226.11 | 253.47 | 3.366 | 0.394 | 2.0030 | 0.0 | ✓ | (phaseA k1) |
| 2 | 285.38 | 315.72 | 4.274 | 0.546 | 2.0030 | 94.27 | ✓ | (phaseA k2) |
| 4 | 311.18 | 372.94 | 5.096 | 0.683 | 2.0030 | 95.72 | ✓ | (phaseA k4) |
| 8 | 343.88 | 415.85 | 5.714 | 0.786 | 2.0030 | 95.98 | ✓ | jaum6nf1 (+ k1/2/4 in group `bi0-int4head-topk-accept`) |

- **Mechanism works.** E_accept and TPS climb monotonically with k; 128/128 holds at every k; the patch is a true no-op at k=1 (apply() returns before rebinding `rejection_sample`, so k=1 == shipped submission by construction).
- **k=8 = 415.85 wall = +64% over the true natural k=1 (253.47).** That captures (416−253)/(507−253) ≈ **64% of the synthetic-ceiling headroom** (#813 r=1.00 = 506.84 wall, `j60r68os`).
- **PPL is k-invariant (2.0030 at every k).** Expected — the PPL harness teacher-forces the ground-truth tokens, so the accept rule never enters it. The quality cost of top-k accept is ONLY visible in *generated* output, which is why token-divergence is already 94% at k=2 and why the AIME/MMLU-Pro/GPQA/GSM8K panel (running now) is the real gate, not PPL.

## Baseline correction: the "323 wall / E_accept 4.355 / r=0.56" k=1 anchor is a SYNTHETIC point, not the natural drafter

The PR Baseline lists the k=1 control as `309.96 steady / 323.29 wall (myk4s0ft), current real E_accept≈4.355, r=0.56`. That run was **not** the free-running drafter — its W&B config is `imposed_rate=0.56, rates_list=[0.56]×6, synthetic_garbage_tokens=True, probe=synthetic-acceptance-ceiling-oracle` (it's #813's synthetic r=0.56 oracle row, served with `REJECTION_SAMPLE_METHOD=synthetic`).

Root cause (off-by-one on the bonus token): `accept_oracle.py:14` picked r=0.56 as "current E_accept 3.38 / K=6". But 3.38 is the E_accept **length** (= 1 bonus + 6·r), so the true natural accept-rate is (3.38−1)/6 = **0.394**, not 0.56. The synthetic anchor therefore imposed acceptance length 4.36 vs the real natural 3.366 — over-accepting by ≈1 token, which is exactly why it read 323 wall instead of the true ~253.

The natural exact-match control (my phaseA k=1, byte-identical to the shipped submission) directly measures **226.11 steady / 253.47 wall / E_accept 3.366 / accept-rate 0.394** — which matches the corrected 0.394 to 3 digits. So:
- True natural k=1 baseline ≈ **253 wall** (not 323). k=1 *is* byte-identical (no-op patch); it just never reproduces 323, because 323 was never natural.
- This does NOT change the GO decision — the synthetic ceiling (507 wall, `j60r68os`) is still a valid upper bound, and the realized climb is simply measured from 253. The headroom is larger in relative terms (253→507 = +100% ceiling).
- The old 256.74 reconstruction was actually ≈right (within ~1% of true natural wall); #813's "correction" to 323 was the error.

## Plan / next
Quality panel is running now, one k per server boot (gsm8k → gpqa_diamond → aime_greedy → mmlu_pro), starting with k=1 to validate my harness reproduces the #795 reference (MMLU-Pro 0.692 / GPQA 0.503 / GSM8K 0.915 / AIME-greedy 0.300). Then k=2, and k=4/k=8 with early-stopping on the #784 floors. Note AIME maj@8 runs at temp=1.0 where the top-k patch is structurally a no-op (it only fires on the all-greedy path), so maj@8 is k-invariant — I'll report it once as a reference rather than re-run it per k. Full results + operating-point recommendation to follow.
