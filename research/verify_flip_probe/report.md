# int4 spec-verify greedy flip-rate probe (PR #23)

Per-token argmax flip between **M=1 (AR greedy)** and **M=K+1 (batched verify)** on the
merged int4 QAT base (`google/gemma-4-E4B-it-qat-w4a16-ct`), and whether fp32-logit accumulation or
deterministic reduction removes it.

## Result table

flip_rate_per_token = flips / (contexts × positions); cell shows `rate (flips/total)`.

| config | flip_rate (M=2) | flip_rate (M=4) | flip_rate (M=6) | flip_rate (M=8) | latency overhead |
|---|---|---|---|---|---|
| baseline | 0.00521 (3/576) | 0.00521 (3/576) | 0.00521 (3/576) | 0.00521 (3/576) | 0% |
| fp32-logit | 0.00174 (1/576) | 0.00174 (1/576) | 0.00174 (1/576) | 0.00174 (1/576) | +0.2% |
| deterministic | 0.00521 (3/576) | 0.00521 (3/576) | 0.00521 (3/576) | 0.00521 (3/576) | +14.0% |
| fp32-plus-det | 0.00174 (1/576) | 0.00174 (1/576) | 0.00174 (1/576) | 0.00174 (1/576) | +14.7% |

- contexts = 12, positions/context = 48 (spaced 4 tokens apart,
  starting right after the 256-token prompt — a shallow dense decode-region window), k-sweep =
  [1, 3, 5, 7] (M = K+1). Total compared (context, position) pairs per config = 576 (12×48).
- **Cross-process determinism noise floor** (two independent M=1 runs): 0/576 (rate 0.00000). A ~0 floor means the
  flips above are the genuine batch-shape (M) effect, not process-boundary noise.
- Latency overhead = single-sequence decode ms/token vs `baseline` at the verify size (M=8).

## Decision

**CHEAP FIXES RULED OUT as a standalone linchpin resolution.** No config reaches
`flip_rate_per_token = 0.000`, and the position-level diagnostic below shows the verdict is
*proven* (a concrete batch-induced flip survives every fix), not merely a small-sample artifact:

- **`deterministic` is a complete no-op AND costs +14% latency.** Its argmaxes are byte-identical
  to `baseline` at *both* M=1 and M=8 (0 diffs / 576) for every M — confirming
  `torch.use_deterministic_algorithms` + `CUBLAS_WORKSPACE_CONFIG` cannot reach the custom Marlin
  int4 CUDA kernel or the Triton attention kernel (they only constrain cuBLAS/cuDNN/aten). Never
  ship it: pure latency loss, zero greedy-identity gain. (Consistent with kanna #19's note that
  `VLLM_BATCH_INVARIANT=1` is aten-scoped and cannot reach the Marlin `_C` op.)
- **`fp32-logit` does not remove the batch-variance; it reshuffles which near-ties bf16 rounding
  masks vs exposes.** It costs only +0.2% latency and *nets* fewer flips on this sample (1 vs 3), but
  it is not a real reduction of the underlying effect — see 7:268 below.
- **The irreducible source is the decoder-layer Marlin int4 GEMM**, which is genuinely batch-variant
  (M=1 hidden states differ from M≥2). This is reachable only by a source-level batch-invariant
  Marlin kernel — i.e. **kanna #19's lane is the required fix**, not a post-GEMM precision/reduction
  trick.

### Position-level mechanism (why no cheap fix can reach zero)

Every flip is the *same* M=1-vs-M≥2 disagreement; M=2,4,6,8 argmaxes are bit-identical to each other
(hence the table is flat in K). So the effect is **binary: batched (M≥2) vs not (M=1)** — it does
**not** grow with draft width K. Implication for spec decode: a longer draft is no worse for
greedy-identity than a short one; the penalty is incurred the moment you batch the verify at all.

The flips live in a tiny set of near-tie positions whose argmax depends on the exact logit values,
which in turn depend on lm_head precision:

| position | baseline (bf16): M=1 / M≥2 | fp32-logit: M=1 / M≥2 | what it shows |
|---|---|---|---|
| 1:296 | 68535 / 236764 → **flip** | 236764 / 236764 → ok | bf16 rounding *introduced* a flip; fp32 resolves it |
| 1:384 | 236769 / 5471 → **flip** | 5471 / 5471 → ok | same — bf16 rounding artifact |
| 7:340 | 5143 / 11081 → **flip** | 5143 / 5143 → ok | same — bf16 rounding artifact |
| **7:268** | 11082 / 11082 → ok | **6816 / 11082 → flip** | **smoking gun:** faithful fp32 logits expose a real batch flip that bf16 rounding had *masked* |

At **7:268** the more-faithful fp32 logits disagree between M=1 (token 6816) and M≥2 (token 11082).
Since fp32 lm_head removes the output-rounding confound, this disagreement can only come from the
**hidden state feeding lm_head differing by batch size** — i.e. the decoder Marlin GEMM is
batch-variant. fp32 lm_head therefore cannot zero the flip; it only changes which near-ties the
rounding happens to mask (3 here) vs expose (1 here). The "3→1" is a near-tie reshuffle, not a
reduction of the batch-dependence. Hence the linchpin is not the logit-accumulation step.

**Statistical note.** Absolute counts are small (576 forced-decode positions, 12 contexts). The
*rate* estimates are therefore coarse (a 0/576 cell would only bound the true rate to ≲0.6%). But the
*verdict* does not rest on the counts: 7:268 is a single, reproducible-across-M (M=2/4/6/8 all =11082
vs M=1 =6816) existence proof that a batch-induced flip survives the strongest cheap fix. A
higher-context confirmation would tighten the rates; it cannot make a batch-invariant decoder GEMM
appear. (Re-run with `--num-contexts 64 --num-steps 32 --resume` to extend.)

## Method (why this isolates the GEMM batch dimension M)

- Real int4 Marlin engine (vLLM 0.22, compressed-tensors W4A16, TRITON_ATTN), `enforce_eager=True`
  so a chunked-prefill chunk of width M runs the decoder GEMMs at batch-dim exactly M.
- `max_num_batched_tokens = M`; `prompt_logprobs` recomputes every interior position's logit at
  batch-dim = M. The rank-1 token of `prompt_logprobs[c+1]` is the forced argmax of the logit at
  position c given the real context S[:c].
- M=1 (`max_num_batched_tokens=1`) is the AR-greedy reference; M=K+1 are the verify forwards, both
  forced on the identical causal context — only the GEMM batch shape differs.
- vLLM V1 in-process teardown does not free GPU memory, so each (config-family, M) runs in its own
  fresh worker subprocess. The path is process-to-process deterministic (eager, fixed shapes, no
  Marlin atomic-add, seed=0), verified by the noise floor above, so cross-process comparison isolates
  only the batch-shape effect.
- deterministic configs run with `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set before the cuBLAS handle is
  created) and `torch.use_deterministic_algorithms(True, warn_only=True)`.

### fp32-logit semantics
`fp32-logit` recomputes the lm_head projection in fp32 (`F.linear(hidden.float(), weight.float())`),
i.e. fp32 accumulation with no bf16 output rounding. The trivial `logits.to(float32)` cast is a
mathematical no-op for argmax (monotone, and the Gemma final-logit soft-cap is also monotone), so the
fp32-accumulation form is the only faithful test of hypothesis (a).

## Caveats
- Deterministic-mode overhead is measured cross-process (group A vs group B); treat as approximate.
  fp32-logit overhead is within-process (patch on/off, same engine).
- `torch.use_deterministic_algorithms` / `CUBLAS_WORKSPACE_CONFIG` do not alter the custom Marlin int4
  CUDA kernel or the Triton attention kernel; they only constrain cuBLAS/cuDNN/torch ops.
