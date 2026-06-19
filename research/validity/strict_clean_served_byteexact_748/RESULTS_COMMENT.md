STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["fikec7di"],"primary_metric":{"name":"served_spec_bi1_greedy_identity","value":0.1641},"test_metric":{"name":"served_spec_bi1_tps","value":82.26}}

## Results

**VERDICT: `SERVED_RESIDUAL_DIVERGENCE`** — `VLLM_BATCH_INVARIANT=1` does **not** make the live batched-verify served spec-dec *literally* 128/128 byte-exact (strict-#319 greedy identity vs same-BI served AR ref = **21/128**, frac 0.164). **But — per the advisor reframe (#752/lawine) — that bare number is not the headline.** The honest read is: the residual is **quality-neutral** (PPL BI-delta **0.001 %**) and **predominantly a benign bf16-ULP-tie cascade** (104/107 divergence onsets at ≤2 ULP). It does **NOT**, however, clear the *strict* τ=0.3 self-consistency gate the way lawine's publishable config did: there are **3/107 confident genuine flips** (vs lawine's 0), all sitting at the minimal 3-ULP gap. So the precise classification is **`predominantly_benign_ulp_marginal_confident_tail`**, and the bottom line is **BI=1 is necessary but not sufficient for literal served byte-exactness.**

### 1. Identity (the load-bearing measurement)

| comparison | identical | per-token flip hazard | meaning |
|---|---|---|---|
| **BI=1 spec vs BI=1 AR** (PRIMARY / strict-#319) | **21/128** (0.164) | **0.383 %/tok** | BI=1 batched-verify spec is NOT literally byte-exact |
| BI=0 spec vs BI=0 AR (CONTROL, deployed order) | 24/128 (0.188) | 0.342 %/tok | deployed non-byte-exact behavior, reproduced |
| BI=1 AR vs BI=0 AR (x-check, **spec OFF both**) | 18/128 (0.141) | 0.416 %/tok | even pure-AR diverges under a BI toggle |
| **DETERMINISM FLOOR** (BI=1 AR vs BI=1 AR, **same config twice**) | **128/128** (1.000) | 0.000 %/tok | **stack is bit-reproducible within-config** |

The **determinism floor = 128/128** is the control that makes the 21/128 interpretable: the served greedy stack is bit-reproducible when nothing changes, so the 21/128 is a **real, reproducible per-step reduction-order divergence**, not run-to-run noise. BI=1 engagement was confirmed live (the BI=1 servers register the `batch_invariant.py` custom kernels; absent under BI=0). Each spec arm is compared against its **same-BI** AR reference, so the only within-pair difference is spec on/off.

The raw hazards (0.383 % BI=1 vs 0.342 % BI=0) have overlapping Poisson CIs — BI=1 gives **no measurable reduction in the bare flip hazard**, because a 512-token greedy rollout is fragile to *any* single reduction-order site and BI=1 pins only one (the M=1 decode split-KV). `(1−0.0038)^512 ≈ 0.16` reproduces the ~16 % full-completion identity. **This is exactly why the bare 21/128 must be read through self-consistency + PPL, not as a pass/fail of literal byte-exactness.**

### 2. Self-consistency (τ=0.3 gap-probe, #720 protocol) — honest deviation from lawine's clean pass

Re-scoring the 107 divergence onsets under one clean BI=1 eager forward (gap_nat = top1_logprob − top2_logprob at the onset):

| gap_nat | # onsets | bf16 ULPs | reading |
|---|---|---|---|
| 0.000 | 39 | 0 (exact tie) | benign |
| 0.125 | 52 | 1 | benign |
| 0.250 | 13 | 2 | benign |
| **0.375** | **3** | **3** | **confident (≥ τ=0.3)** |

- **`confident_genuine_flips = 3 / 107` → `self_consistent_pass = False`.** I am **not** reporting this as lawine's clean 0-flip result; the K=6 fire-config served residual keeps a small confident tail that the publishable config did not.
- **Every gap is bf16-ULP-quantized** (multiples of 0.125), the signature of numeric near-ties. All 3 confident flips sit at **exactly 3 ULP (0.375 nat)** — the *smallest* gap above τ — so the gate is **τ-sensitive: 0 confident flips at τ ≥ 0.4.**
- **96.3 % of onsets** have the spec/AR token pair as the model's actual top-2 (a genuine near-tie between the two leading candidates). Of the 3 confident flips, 2 are still top-1↔top-2 near-ties; only 1 (`mmlu_pro-012ddc1ffc`) flips to the model's rank-2 token.

So: `confident_flip_frac = 2.8 %`, all marginal (≤ τ+1 ULP) → **`predominantly_benign_ulp_marginal_confident_tail`**. The residual is overwhelmingly benign ULP ties, but I will not overclaim a clean self-consistent pass — it is not one.

### 3. PPL BI-neutrality (quality is preserved by the toggle)

| arm | PPL (61,797 tok) |
|---|---|
| BI=1 (num_splits=1) | 2.00641 |
| BI=0 (deployed) | 2.00639 |
| **abs delta** | **0.0010 %** |

Teacher-forced PPL (official `ppl_endpoint` math; spec-decode does not enter scoring) is **BI-neutral to 0.001 %** → the reduction-order difference is **quality-neutral**. (Absolute PPL is on the loadable full-vocab QAT proxy `gemma-4-E4B-it-qat-w4a16-ct`, which brackets — not exactly hits — the deployed pruned-16k anchor 2.019; the load-bearing claim is the **BI=1-vs-BI=0 parity**, which holds.)

### 4. Mechanism — where the residual lives (`enforce_eager` arms)

Two extra BI=1 arms with `--enforce-eager` (replicating #743's offline execution mode but with the **real** online batched spec path, not the `prompt_logprobs` proxy) decompose the residual by hazard removed:

| arm | identical | hazard | note |
|---|---|---|---|
| BI=1 spec, **CUDA graphs** (primary) | 21/128 | 0.383 %/tok | served default |
| BI=1 spec, **enforce_eager** | **84/128** | **0.073 %/tok** | no CUDA graphs |

- **`mechanism = MIXED_CUDAGRAPH_DOMINANT_VERIFY_RESIDUAL`.** Disabling CUDA graphs removes **81 %** of the per-token flip hazard (84/128 eager vs 21/128 captured). So **CUDA-graph capture asymmetry is the DOMINANT contributor** — the M=1 decode is graph-captured (sizes [1,2,4,8]) but the M=K+1=7 verify is not, so captured-decode vs eager-verify take different attention reduction orders. A **residual M=K adaptive split-KV verify reduction** (num_splits>1 at served seq-len, which plain BI=1 does not align) remains as the secondary ~19 %.
- This is **NOT the GEMM** (land #680: int4 Marlin GEMM is byte-identical across M). It explains why #743's *offline* byte-exactness held: the `prompt_logprobs` M=1-shaped proxy avoids **both** the capture asymmetry and the M=K verify shape.

### 5. TPS deliverables

| arm | output_tps | peak mem |
|---|---|---|
| **BI=1 spec (batched verify)** | **82.26** | 19543 MiB |
| BI=0 spec | 88.35 | 19537 MiB |
| BI=1 AR | 78.26 | 19513 MiB |
| BI=0 AR | 95.28 | 19489 MiB |

- **Decode tax (BI=1 vs BI=0, spec on): 6.90 %** (82.26 vs 88.35) — the price of `num_splits=1` on the batched-verify path.
- **`bi1_spec` clears 126.378? NO** (82.26) — **but this is not a meaningful comparison.** The loadable full-vocab QAT proxy on a bare `api_server` lacks every deployed optimization (pruned-16k lm_head, fa2sw, precache, onegraph); absolute TPS does not transfer. The PR uses this proxy because the attention reduction-order property is head-/quant-independent (#743), so only the **identity / self-consistency / PPL** results and the **BI=1-vs-BI=0 decode tax** transfer. (`official_tps=0`.)

### Command

```bash
# 4 served arms (resumable), each = one live vLLM 0.22.0 api_server, MAX_NUM_SEQS=1, TRITON_ATTN,
# 128 reasoning prompts, temp 0, ignore_eos, 512 tok; ngram K=6 batched verify for spec arms.
cd target/ && research/validity/strict_clean_served_byteexact_748/run_all.sh
# eager mechanism arms (BI=1, enforce_eager, AR + spec) + determinism rep + selfconsist/PPL:
research/validity/strict_clean_served_byteexact_748/run_eager.sh
research/validity/strict_clean_served_byteexact_748/phase2b.sh   # tau=0.3 gap-probe + offline PPL (chunked prefill)
research/validity/strict_clean_served_byteexact_748/analyze.py
research/validity/strict_clean_served_byteexact_748/wandb_log.py
```

- **Peak GPU mem:** ~19.5 GiB (bi1_spec 19543 MiB) on the single A10G.
- **W&B run:** `fikec7di` (group `strict-clean-served-byteexact-land`).
- **`analysis_only=1, official_tps=0, no_hf_job=1, fires=0`.** Locked `int4_g128_lmhead`@126.378 untouched.

### What happened — honest analysis

The literal hypothesis (does BI=1 make served spec **128/128** byte-exact?) is **NO** (21/128), and the determinism floor (128/128) proves that is a real reduction-order effect, not noise. But the advisor-mandated deeper read changes the *meaning*: the residual is **quality-neutral** (PPL Δ 0.001 %) and **predominantly a benign bf16-ULP-tie cascade** (104/107 onsets ≤ 2 ULP, 96 % top-2 near-ties). **Where I diverge honestly from lawine's precedent:** the strict τ=0.3 gate does **not** cleanly pass here — there are **3/107 confident genuine flips**, all at the minimal 3-ULP gap (τ-sensitive: 0 at τ≥0.4). So I classify this `predominantly_benign_ulp_marginal_confident_tail`, not `benign_ulp_tie_cascade`. **BI=1 is necessary but not sufficient for literal served byte-exactness.** The mechanism arms localize the dominant residual to **CUDA-graph capture asymmetry** (81 % of hazard, decode captured / verify not), with a secondary M=K split-KV verify reduction — neither of which plain BI=1 aligns. The 6.9 % decode tax is real but buys no literal byte-exactness on the K=6 fire path.

### Suggested follow-ups

- **Close the marginal tail via the `num_splits=1` verify fix** (the land #743 finding; the advisor routed this to **lawine #755** on the publishable config). It should align the residual M=K split-KV verify reduction and likely drive the 3 confident flips → 0, matching lawine's clean pass. Worth confirming whether it also closes the K=6 fire-config tail or whether the CUDA-graph capture asymmetry (the dominant 81 %) needs the **decode-graph captured under BI=1** (or verify added to the capture set) as a separate fix.
- **A don't-care near-tie band acceptance** (cf. land #654's tie-tolerant residual): since 104/107 onsets are pure ULP ties and PPL is neutral, a tie-tolerant identity gate would pass batched-verify spec without bit-exact reduction alignment for the benign 97 % — but the 3/107 marginally-confident flips mean the *strict* #319 contract still needs the verify fix, so this band would have to be justified as quality-equivalent (the PPL-neutrality supports that).

### Public evidence used

Challenge leaderboard frontier is ~489 TPS (`hayai-ctk48-mwfix-v1`, `osoi5-feopt2…lmhead12k-fa2sw-precache-skv64`), a different lane from this DQ-risk-free byte-exact route over the locked 126.378 anchor. This card refines the land #743 (`rwk498ve`) offline finding for the **served** stack: BI=1 transfers as a **quality-neutral, predominantly-benign** residual but **not** as literal served byte-exactness, and names the dominant residual sub-op (CUDA-graph capture asymmetry > M=K split-KV verify).
