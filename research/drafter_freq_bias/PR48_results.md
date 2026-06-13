STUDENT kanna:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["96pn3c43","rrp0xc6e","rggrg6r6","l32wjlig"],"primary_metric":{"name":"tps","value":463.49},"test_metric":{"name":"ppl","value":2.3767}}

## Results

**Verdict: NEGATIVE for the goal (TPS), with a small *real* acceptance signal.** A static unigram logit bias on the drafter **regresses TPS at every tested bias** (−1.7% / −2.2% / −4.1% vs the bias=0 baseline) and so fails the success criteria. The acceptance-rate sub-claim is *weakly confirmed* — bias=0.5 gives a reproducible **+0.56% E_accept** — but the gain is ~4× too small to overcome the per-step cost of the bias re-rank, and it reverses at higher bias (double-counting the prior the trained drafter already encodes).

### What I built (and deviations from the PR pseudocode)

Two deviations were forced by the actual serving architecture; both make the test *more* favourable to the hypothesis, not less:

1. **`train.py --local-only --env` does not exist** in this target. I ran the experiment through the local serving harness (`harness.LocalServer` + the official `sglang.bench_serving` protocol, reused from my `private_gap_probe.py`, PR #44). Driver: `scripts/validity/drafter_bias_screen.py`. Each arm = one **fresh server** with exactly one changed variable (`DRAFTER_FREQ_BIAS`), benched on the public 32-prompt set.
2. **The drafter is not a dense `[batch, 262144]` logit vector.** It is a centroid-sparse MTP head (`Gemma4MTPMaskedEmbedder`, num_centroids=2048, top-k=64 → ~8k candidate tokens/step) whose argmax runs in a **fused Triton kernel** (`FUSED_SPARSE_ARGMAX`). So `logits[:, top_k_tokens] += bias` is not directly expressible. I implemented the bias as a static additive table over the **sparse candidate set**, re-ranking inside the drafter's own proposer (`submissions/fa2sw_precache_kenyan/sitecustomize.py`, `_biased_top_tokens`). It is **drafter-only** — the verifier argmax path is untouched — and at bias=0 the hook is bypassed entirely, so the baseline arm is byte-identical to the leaderboard submission (and stays on the fused kernel).

**Frequency histogram** (`scripts/validity/build_freq_bias_tokens.py`): built over the verifier's own greedy **output** tokens (`completion_token_ids`) from the committed fa2sw greedy reference — the distribution the drafter is actually trying to predict, the most favourable corpus for the claim. 128 records, 65,536 output tokens, 4,864 unique. Top-500 covers **76%** of output mass. `<eos>`(1) and `<end_of_turn>`(106) were **excluded** from the boost set — boosting the drafter toward premature stops would make it propose tokens the verifier rejects, unfairly dooming the hypothesis.

### Decision metric: E_accept

E_accept (mean accepted tokens / target step) is read straight from vLLM's spec-decode counters. It is **onegraph/CUDA-graph invariant** (the graph changes *how* drafted tokens are produced, not *which*), so it is the clean test of the claim: **if the bias does not raise E_accept it cannot raise TPS.** Baseline (PR #44, fused, precache-off public): **4.071**; this screen measured **3.956** on the public-32 subset (consistent — fewer prompts).

### Results — K=500, n=32 public prompts, 2 independent runs

| bias | E_accept (run1 = run2) | ΔE_acc | accept_rate | TPS run1 / run2 | ΔTPS% | per-step lat |
|---|---|---|---|---|---|---|
| **0.0** (fused, =leaderboard) | **3.95587** | — | 0.42227 | **471.35 / 471.08** | — | **8.29 ms** |
| 0.5 | **3.97793** | **+0.56%** | 0.42542 | 463.49 / 463.44 | **−1.67%** | 8.48 ms |
| 1.0 | 3.95160 | −0.11% | 0.42166 | 461.15 / 461.15 | **−2.16%** | 8.47 ms |
| 2.0 | 3.87126 | −2.14% | 0.41018 | 451.94 / 452.21 | **−4.12%** | 8.47 ms |

- **PPL: 2.3767 (unchanged, by construction)** — the bias is drafter-only; the verifier still emits its own argmax, so every committed token id is identical to the bias=0 greedy reference. Greedy-exactness holds. ✅ (≤ 2.42)
- **Determinism note:** E_accept, accept_rate, and draft-step counts are **bit-identical across the two fresh-server runs** (only wall-clock TPS jitters ±0.27 tok/s). Generation is deterministic in this config, so the +0.56% at bias=0.5 is **real and reproducible, not noise**.

### K-sweep at the best bias (b=0.5) — per PR Step 2, K ∈ {100, 500, 1000}

The acceptance gain grows modestly with K (more frequent tokens boosted) but **never approaches the +2.2%/step penalty**, so TPS regresses at every K. The per-step cost is K-independent, confirming a K-sweep cannot rescue the hypothesis as implemented.

| K (output-mass covered) | E_accept | ΔE_acc vs base | TPS | ΔTPS% |
|---|---|---|---|---|
| 100 (58%) | 3.96699 | +0.28% | 461.89 | −2.01% |
| 500 (76%) | 3.97793 | +0.56% | 463.49 | −1.67% |
| 1000 (83%) | 3.96673 | +0.27% | 462.42 | −1.89% |

(baseline bias=0 is K-independent: E_accept 3.95587, TPS 471.35.) The acceptance gain is **non-monotonic in K, peaking at K=500** — boosting the rarer tokens past rank 500 re-introduces the same double-counting erosion seen in the bias-magnitude sweep. So **K=500 / b=0.5 is the global optimum of the entire (K × bias) grid tested**, at +0.56% E_accept — and it still loses −1.67% on TPS.

### Why a real acceptance gain still loses on TPS

TPS ≈ E_accept / per-step-latency. The bias moves *both* terms, in opposite directions:

- **Numerator (E_accept):** best case +0.56% (bias=0.5 only). The acceptance-vs-bias curve **peaks near b≈0.5 and degrades** (b=1.0 flat, b=2.0 −2.1%). This is exactly the double-counting prediction: the FT drafter is a *trained* MTP head that already encodes the unigram marginal, so adding an external unigram prior pulls it **away** from the verifier's *conditional* argmax — the more bias, the worse.
- **Denominator (per-step latency):** the biased arms leave the fused Triton kernel for an eager re-rank (`_select_and_score` → materialize `[T,S,hidden]` embeddings + `[T,S]` logits → gather/add/argmax/gather). This costs a **constant +2.2%/step (8.47–8.48 ms vs 8.29 ms), independent of bias value** — the smoking gun that it's an implementation cost, not a bias effect. All arms kept onegraph capture (`onegraph_captured=true`), so the comparison is fair.

The +2.2% per-step penalty is **~4× larger** than the +0.56% best-case acceptance gain → net TPS regresses everywhere.

### Success criteria

| Criterion | Result |
|---|---|
| Local TPS ≥ 432 for ≥1 bias arm | ✅ trivially (all arms 452–463) — but **all biased arms < the 471.35 bias=0 baseline** |
| PPL ≤ 2.42 for all arms | ✅ 2.3767 (unchanged by construction) |
| Acceptance rate **increases** vs no-bias | ⚠️ only at bias=0.5 (+0.56%, real); flat/negative at b≥1.0 |
| Private-gap does NOT increase | ➖ not run — see note below |
| **TPS improves (the actual goal)** | ❌ **regresses −1.67% to −4.12%** |

**Private-gap (not run, deliberate):** the hypothesis already fails decisively on the *primary* metric (public TPS), and the dominant failure mechanism — the +2.2%/step eager-re-rank penalty — is **prompt-set-independent**, so it regresses the private side identically. Running the per-arm probe cannot change the verdict. The probe is ready (`scripts/validity/private_gap_probe.py`) if you want the number anyway; say the word.

### Reproduce

```bash
cd target/
# 1. histogram over the greedy-reference output tokens (top-500, specials excluded)
python scripts/validity/build_freq_bias_tokens.py --top-k 1000
# 2. 4-arm E_accept+TPS screen (fresh server per arm; one changed var = DRAFTER_FREQ_BIAS)
python scripts/validity/drafter_bias_screen.py \
    --arms 0,0.5,1.0,2.0 --num-prompts 32 --topk 500 \
    --wandb-group token-freq-bias --wandb-name kanna/drafter-freq-bias-k500
```

- **Peak memory:** ~19.4 GiB / 22.5 GiB usable (under the 0.90 util cap; model load 8.85 GiB, KV cache 9.46 GiB). The bias table is one `[262144]` bf16 tensor ≈ **0.5 MB** — memory is unchanged vs baseline.
- **W&B:** `96pn3c43` (K=500 run 1), `rrp0xc6e` (K=500 run 2 replicate), `rggrg6r6` (K=100), `l32wjlig` (K=1000), group `token-freq-bias`.

### What happened — honest analysis

The hypothesis is **right in spirit but wrong in magnitude, and dominated by implementation cost.** A small frequency bias *does* nudge the drafter toward the verifier's mode (+0.56% E_accept at b=0.5, reproducible to the bit) — but:
1. the headroom is tiny because the drafter is already FT'd on this distribution (it encodes the unigram prior; PR #25 found drafter acceptance is data-bottlenecked and plateaus ~0.73), and the gain reverses at higher bias (double-counting);
2. expressing the bias in this serving stack means leaving the fused sparse-argmax kernel, and that per-step cost (+2.2%) is several times the acceptance benefit.

Net: every biased arm is slower than the existing fused baseline. The PR's projected "5–15 TPS / 1–3% acceptance" is off by ~5× on the acceptance side and has the wrong sign on TPS.

### Suggested follow-ups (NOT implemented — flagging per boundaries)

1. **Fuse the bias into the Triton kernel** to remove the +2.2% per-step penalty. The acceptance ceiling is now *empirically bounded across the whole (K × bias) grid*: the best E_accept anywhere is K=500/b=0.5 = 3.97793 (+0.56%). At zero added cost that maps to TPS ≈ 471.35 × (3.97793/3.95587) ≈ **474.0 tok/s (+2.6 tok/s)** — the hard ceiling, greedy-exact, but far below the PR's 5–15 tok/s target and requiring a modification + re-validation of the exact fused sparse-argmax kernel for +0.6%. My recommendation: **don't pursue** unless +2–3 tok/s is independently valuable.
2. **Bias the *acceptance threshold*, not the drafter** — if the goal is "accept more of what the drafter already proposes," that lives in the verifier accept path and would break greedy-exactness (out of scope / changes identity).
3. **Spend the effort on the drafter's data bottleneck instead** (more/closer FT data), which PR #25 identified as the real ceiling on E_accept. A bias can only re-rank within the candidate set; it cannot add the conditional information the drafter is missing.
