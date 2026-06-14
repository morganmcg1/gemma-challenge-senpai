# PR #144 — lm_head verify-candidate shortcut: AUDIT + Step-1 quantification

**Verdict: audit = LIVE (verify GEMM reads full 12288), Step-1 = NO-GO.**
Projection metric (official TPS) stays **481.53**. Local A10G profiling only; no HF Job.

## Step 0 — AUDIT (hard gate)

**Question:** does the verify-step lm_head GEMM column-**read** already restrict to
≤ {drafter-K ∪ ~256 bonus} (≈263) columns, or does it project the full 12k pruned vocab
then sparse-argmax?

**Answer:** the verify GEMM reads the **full K=12288-row int4 head**, then scatters to
262144. NOT pre-restricted.

- `lmhead_verify_gemm_cols_read = 12288`
- `lmhead_verify_read_already_pruned = False`  → **LIVE**

### Code-citation evidence

1. **Verify logits = `Gemma4ForCausalLM.compute_logits` (PCK-04 patched).**
   `serve_patch_pck04.py:335-342` `compute_logits_pck04` calls
   `original_compute_logits(self_model, hidden_states)` → `[M, K]` pruned logits, then
   `_scatter_to_full_vocab` → `[M, 262144]`. The dense projection onto the K-row head IS
   the verify GEMM.

2. **The head is a dense K-row `ParallelLMHead`, not a candidate gather.**
   `serve_patch_pck04.py:310-316` rebuilds `self.lm_head = ParallelLMHead(K, hidden, …)`
   with `K = len(keep_ids)`. The weight is `[K × hidden]` int4, read in full each call.

3. **Served K = 12288.** `serve.py:632-634` `_prune_lm_head_rows` row-slices
   `lm_head.weight_packed`/`weight_scale` to `len(keep_ids)` rows; `serve.py:705-708`
   repoints `LOCAL_MODEL_DIR`/`PCK04_KEEPSET` to `/tmp/osoi5-12k-baked`. That dir’s
   `pck04_keepset.json` has `len(keep_ids)=12288`, `full_vocab=262144`; safetensors
   `lm_head.weight_packed [12288, 320] I32` = 12288 rows × 2560 hidden int4 (15.7 MB).

4. **The 8192-candidate FUSED_SPARSE_ARGMAX is the DRAFTER, not the verifier.**
   `sitecustomize.py:771-772, 901-902` patches `Gemma4MTPMaskedEmbedder.get_top_tokens`
   (the MTP drafter’s embedder), invoked in the drafter propose loop at
   `sitecustomize.py:190, 202` (`self.model.get_top_tokens(...)`). Different class from the
   target `Gemma4ForCausalLM`.

5. **`target_argmax` is an INPUT to the accept kernel — computed upstream by the full
   verify path.** `sitecustomize.py:927, 945` `_dixie_fused_accept_prep_kernel` loads
   `target_argmax_ptr`; `:948` compares `draft_token_id != target_argmax_id`; `:950-951`
   emits `target_argmax`. The accept kernel does NOT compute logits — it consumes the
   argmax produced by `compute_logits` over the scattered 262144 (which came from the full
   12288 GEMM).

6. **Design intent: verifier output is independent of the drafter candidate set.**
   `sitecustomize.py:49-50` — "Verifier output is untouched (greedy spec decode emits the
   target argmax regardless of drafter proposals)." The drafter’s sparse candidate set is
   an approximation for *its own* sampling; the verifier emits the TARGET argmax.

7. **Runtime confirmation** (`research/lmhead_verify_audit/bench.log`):
   `[pck04] rebuilt lm_head: ParallelLMHead(num_embeddings=12288, embedding_dim=2560, …)`,
   `Using MarlinLinearKernel for CompressedTensorsWNA16`, and my synthetic
   `compute_logits(M=8)` returns `out=(8, 262144)`. Same code path the real verify uses.

## Step 1 — share quantification + GO/NO-GO (LIVE ⇒ unlocked)

Faithful local microbench on the served stack (A10G, submission vLLM wheel
0.22.1rc1.dev307, real int4 Marlin head). M = K+1 = 8 verify rows.

| stage (M=8) | time | note |
|---|---|---|
| **int4 Marlin lm_head GEMM only → [8,12288]** | **38.27 µs** | the lever’s actual target (12288-col read, 15.7 MB) |
| scatter only `index_copy_ [8,12288]→[8,262144]` | 8.15 µs | NOT the lever (argmax/logprobs still need it) |
| full `compute_logits` (GEMM+scatter+LogitsProcessor) | 135.82 µs | flat across M=8/16/32 ⇒ BW/overhead-bound |
| **candidate per-row gather-GEMM+argmax (8×263, bf16 UB)** | **80.67 µs** | optimistic: no int4 unpack, no candidate-selection cost |

**Share of the decode step** (per-accepted-token budget @481.53 TPS = 2076.7 µs; step = E[T]×budget):

| E[T] | lm_head GEMM share (38.27 µs) |
|---|---|
| 2.5 | 0.74% |
| 3.0 | 0.61% |
| 3.5 | 0.53% |
| 4.0 | 0.46% |

### NO-GO — two independent grounds

**(A) Perf: the candidate read is NET SLOWER.** The optimistic candidate gather-GEMM
(80.67 µs) is **2.1× slower** than the full dense int4 Marlin GEMM (38.27 µs). Reason: at
M=8 the dense GEMM is BW-bound on a 15.7 MB int4 read that Marlin streams efficiently; the
per-row candidate path materializes 8×263×2560 bf16 = 10.8 MB of *gathered* embeddings
(more bytes than the int4 weight) with uncoalesced access, and the small irregular einsum
underutilizes the SMs. Even ignoring (B), restricting the column-read removes at most a
0.46–0.74% slice of the step *and replaces it with a slower kernel* ⇒ net `wall_tps` < 0.
Same outcome as the permanently-closed GEMM-BW lane (#117/#130/#108).

**(B) Correctness: a candidate-restricted argmax can’t preserve exact greedy.** Greedy
spec-decode emits `target_argmax` over the full 12288 vocab at every verify/bonus row
(accept kernel `sitecustomize.py:948,950-951`). For token-identity, every emitted token
must equal the true full-vocab argmax. Computing argmax over only {drafter-K ∪ top-256}
yields the true argmax **only if it lies in that 263-set** — which no cheap proxy can
certify (the centroid top-k that the drafter uses is approximate, tuned for sampling that
need not be exact). A miss emits the wrong token ⇒ breaks greedy token-identity and PPL≤2.42
(both hard gates). Certifying the argmax is in-set requires either the full projection
(defeats the lever) or a per-row upper bound on the 12025 non-candidate logits (a loose
high-dim Cauchy–Schwarz bound that prunes few columns and adds cost). This is precisely why
"active-vocabulary" speculative methods restrict the **drafter**, not the exact-greedy
**verifier**.

**Conclusion:** the verify lm_head GEMM-read lane is technically LIVE (full 12288, distinct
from a #121-style MOOT) but cannot be reclaimed: it is a <1% slice, a candidate read is
empirically slower, and it is incompatible with exact greedy. **Projection stays 481.53.**

### Durable lesson
Verify-GEMM column-read pruning is incompatible with exact-greedy spec decode — and even
the optimistic prototype is net-slower at M=8. Restrict the drafter, not the verifier.
