STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["rf0gitmc","gkv90epe","f4fkl7dv"],"primary_metric":{"name":"served_local_tps_pinned","value":159.79},"test_metric":{"name":"realization_gap_vs_model_frac","value":0.898}}

## Results — PR #697: does the served stack realize the +50.97 pin model?

**VERDICT: `REALIZATION_GAP`.** On the *real* served vLLM spec stack (LOCAL A10G, analysis_only), the attention/identity pin does **not** realize the modelled "≈free +25.94 (bi1) / +50.97 (fixed2d)" speedup tier. It does the opposite: turning the pin on **costs −29.09% wall-TPS** (−65.55 local TPS). ~90% of the modelled +25.94 margin is unrealized. This **confirms denken #698's drafter-M=1-degeneration prediction** (predicted ≈136.71 official-equiv; measured 139.02, surplus +2.31 ≈ 1.7%, not material → no large verify-side surplus).

`analysis_only=1 · official_tps=0 · no_hf_job=1 · fires=0` — no submission, served file untouched, locked `int4_g128_lmhead`@126.378 untouched.

### Headline: mapping-free served A/B (the robust deliverable)

Fresh server per run, paired A/B, single-stream greedy, 128 prompts × 512 out-len, seed 1, N=3.

| arm (served, local A10G) | wall_tps median | cv% | e_accept |
|---|---|---|---|
| unpinned spec (BI=0, K=5) | **225.34** | 0.015 | 3.190 |
| **bi1-pinned spec (BI=1, K=5)** | **159.79** | 0.071 | 3.196 |
| Δ pin | **−65.55 TPS / −29.09%** | | +0.006 |
| AR M=1 anchor (K=0), N=2 | 77.96 | ~0 | — |
| bi1-pinned xcheck (vs AR, N=2) | 159.94 | 0.020 | 3.200 |

The bi1-pinned xcheck (paired against the AR anchor, independent runs) lands **159.94** — within **0.15 TPS (0.10%)** of the core's 159.79, so the pinned number is cross-validated across two different pairings.

The pin's cost is **sign-flipped** vs the model: the #691 model placed bi1 **+25.94 above** the AR bar (a net-speedup tier); the served measurement shows the pinned spec stack is **−29% slower than the unpinned spec stack**. This finding is mapping-free and needs no basis to stand.

**Mechanism (confirms denken #698).** e_accept is essentially identical pinned vs unpinned (3.196 vs 3.190): the pin does **not** change *what* gets accepted per iteration, so the entire −29% is per-iteration **time**. The pin (`VLLM_BATCH_INVARIANT=1`) swaps in slower deterministic kernels *globally*, including the M=1 drafter forward that runs K=5× per step. The free-verify-side win the static model credited is dominated by the drafter-decode slowdown → "drafter-M=1-degeneration." **Pin provably live, not silently ignored:** the bi1 server log registers `batch_invariant.py` `matmul_persistent` (deterministic fixed-reduction kernels) in the Dynamo trace; the unpinned server log has **zero** batch-invariant references. The −29% is itself causal proof — only `VLLM_BATCH_INVARIANT` differs between the two arms (same K=5, same drafter, fresh server each), and e_accept is unchanged, so the slowdown is the pinned kernels, not a config artifact.

### Official-equiv mapping (×0.870) — reported per PR, but soft

Per the PR's #691 ×0.870 basis: bi1-pinned 159.79 → **139.02 official-equiv**; unpinned → 196.05; AR → 67.83.

- vs **+10 bar 136.378**: measured margin **+2.64**; modelled bi1 margin +25.94 → **`realization_gap_vs_model_frac` = 0.898** (90% unrealized). Level-realization frac = 0.856 (shortfall 23.30 below modelled 162.32).
- **Self-calibration caveat (important):** the AR self-cal **fails** — AR served-local 77.96 × 0.870 = 67.83, but the locked official AR is 126.378 (implied multiplier 1.621, not 0.870). So ×0.870 does **not** transfer to served wall-TPS; the absolute 139.02 is **not** a certification that the pinned stack clears the bar. The robust deliverable is the head-independent **−29%**, not the absolute official number. Neither ×0.870 nor an AR-derived multiplier cleanly maps the *spec* stack served→official (different serving/head overhead structure), so I report 139.02 as the PR-prescribed figure only.
- **denken #698 check (per your flag request):** measured official-equiv **139.02** vs denken's predicted **136.71** → **surplus +2.31 (1.7%)**. denken's mechanism (fixed2d ≡ blanket `num_splits=1` on the M=1 drafter) applies equally to the bi1 arm I measured, so this is apples-to-apples. **My read: this surplus is NOT a material verify-side effect** — it is far smaller than the ×0.870 basis's own modeling error (the AR self-cal misses by 58.5 TPS / 46%), so +2.31 in that soft space is well inside the mapping uncertainty, not signal. The mapping-free −29% is the authoritative realization number; denken's drafter-degeneration is confirmed. I'm surfacing the +2.31 explicitly per your instruction — **the joint-review trigger is yours**; I don't think the data warrants it, but flagging as asked.

### Decomposition (official-equiv, ×0.870)
- spec speedup served (U−A)×0.870 = **+128.22** (spec dec itself is a large served win)
- **PIN delta served (P−U)×0.870 = −57.03** (the pin gives almost all of it back)

### Caveats / scope
- **K=5 vs deployed K=6.** I ran K=5 (NUM_SPECULATIVE_TOKENS=5, M=6 verify width is K+1) to match the #691 modelled tier exactly. The deployed submission default is NUM_SPECULATIVE_TOKENS=6. The −29% is a per-iteration-time effect (e_accept-invariant), so it is expected to hold or worsen at K=6 (more drafter forwards/step under the slow kernels). Not separately measured.
- **Head:** full-vocab head (served proxy), heavier than the deployed pruned-16k head. The pin's −29% is **head-independent** (it slows kernels, not the head), so this is the right knob to isolate. The 16k-head **absolute** number (capstone) would need stark #690's 16k-head boot cert or an HF job — **out of scope** here (cross-read blocked by launch isolation; analysis_only, no HF job).

### Command
```bash
# Run A (core, decisive): unpinned spec vs bi1-pinned spec, N=3
./.venv/bin/python scripts/profiler/paired_tps_ab.py \
  --baseline int4_mtp_batchinv --candidate int4_mtp_batchinv \
  --baseline-env VLLM_BATCH_INVARIANT=0 --baseline-env NUM_SPECULATIVE_TOKENS=5 \
  --candidate-env VLLM_BATCH_INVARIANT=1 --candidate-env NUM_SPECULATIVE_TOKENS=5 \
  --n 3 --num-prompts 128 --output-len 512 --seed 1 \
  --tag pin_served_realization_697/core --no-project \
  --wandb-name land/pin-realization-unpinned-vs-bi1 --wandb-group pin-served-tps-realization-land
# Run B (anchor): AR M=1 vs bi1-pinned xcheck, N=2  (--baseline-env NUM_SPECULATIVE_TOKENS=0)
# Analysis: /usr/bin/python research/validity/pin_served_realization_697/analyze_realization.py --wandb
```

### Environment / cost
- LOCAL A10G (CUDA_VISIBLE_DEVICES=0), served vLLM (server venv `/tmp/senpai-venvs/20f658587e8a6643`), gpu_memory_utilization=0.9.
- **Peak GPU memory:** ≈20.7 GiB envelope (0.9 × 23 GiB A10G); measured KV cache 8.51 GiB available / 338,956 tokens (82.75× concurrency @ 4096-ctx). No OOM.
- W&B group `pin-served-tps-realization-land`: canonical analysis `rf0gitmc` (land/pin-served-realization-697); harness `gkv90epe` (core A/B), `f4fkl7dv` (AR anchor).

### What happened
The static #691 model said the pin is ~free-to-cheap at deployed ctx (bi1 +25.94 over bar). The served realization refutes that **for the spec stack**: the pin is a **−29% wall-TPS tax** because it slows the per-step M=1 drafter forward (run K× per accepted block), and e_accept is unchanged so there's no acceptance win to offset it. The #691 picture was a single-forward attention-cost model; it did not price the drafter-decode multiplier that dominates in a served spec loop. denken #698 called this exactly. The pin is real and live; the modelled credit is not realized.

### Suggested follow-ups
1. **K=6 confirmation** at the deployed NUM_SPECULATIVE_TOKENS to quantify the tax at the shipped width (expect ≥ −29%).
2. **16k-head capstone**: once stark #690 certifies a 16k-head boot (or with human-approved HF job), measure the *absolute* deployed-head pinned served TPS vs the 126.378 bar — the only number that certifies ship/no-ship.
3. **Selective pinning**: pin only the verify forward (greedy-identity-critical) and leave the M=1 drafter on fast kernels — if identity is preserved (drafter is a proposal, verify decides), this could recover most of the −57 official-equiv. Needs an identity proof that drafter non-determinism can't change accepted tokens.
