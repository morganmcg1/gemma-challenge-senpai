STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"analysis_only":true,"official_tps":0,"fires":false,"wandb_run_ids":["o3pa7bbj","ai34kyj6","gq5c3uud","ujwxjfam","4rt69y8r"],"primary_metric":{"name":"espec_edge_ftv1@64_minus_stock@32","value":0.3132},"test_metric":{"name":"ppl_guard_drafter_independent","value":2.019},"verdict":"DRAFTER_NOT_PUBLISHABLE"}

## Results — Local-drafter de-risk (#670): the +0.32 espec edge is REAL & ROBUST, but it is the RETRAIN, not a free knob — and the retrain is not in-scope publishable

**Verdict: `DRAFTER_NOT_PUBLISHABLE` (binding), with `DRAFTER_EDGE_ROBUST` on the robustness axis.**
The +0.32 espec edge is **not a small-sample artifact** — it holds at the full 128-prompt eval population and across prompt-resampling. But the 2×2 decomposition shows the edge is **~98% the kenyan-duma retrain weights**, only ~11% the free `top_k` knob. The retrain artifact (`/tmp/qat-assistant`) is **not reproducible/publishable inside land's launch scope** (cross-student checkpoint + out-of-scope recipe + `/tmp` ephemeral). The one in-scope-free lever (`top_k=64` on the publishable stock Google drafter) only **grazes** the +10 bar (+8.5%, +0.76 TPS over), it is not a robust clear. So there is **no robustly-in-scope-publishable +10 drafter lever**; realizing the projected +17% requires a human-gated decision on a cross-student artifact.

### Design correction vs the PR card (already flagged 17:00Z, advisor proceeding)
1. **"512–1024 prompts" is infeasible** — `eval_prompts_sharegpt.json` is a hard **128-prompt population** (`decode_outputs.py` raises `ValueError` for `--num-prompts>128`, no resampling). So **128 = the entire scored eval set**, espec@128 is the population value (no sampling error to shrink). The reproduce command's `--num-prompts 512` would crash as written. Larger-N robustness is therefore answered by a **64-prompt subsample-seed sweep** (genuine prompt-resampling).
2. **The #664 A/B conflated two changes**, not one: deployed `/tmp/qat-assistant` differs from stock in BOTH the retrain weights AND `centroid_intermediate_top_k` (stock-native **32** vs deployed **64**). I ran the full **2×2 {stock,ftv1}×{top_k 32,64}** to split them.

### Arm B — 2×2 espec / wall_tps decomposition (128×512 population, BI=1, K6, `int4_mtp_batchinv`, vllm 0.22.0)

| drafter cell | espec (`e_accept_exact`) | un-rescued K6 wall_tps | rescued official-equiv (×0.870) | vs locked 126.378 | clears +10 bar (136.378)? |
|---|---|---|---|---|---|
| **stock@32** (ships; byte-exact Google Hub) | 3.3446 ±0.005 (n=3) | 156.46 ±0.02 | 136.12 | +7.71% | no (sits at bar) |
| **stock@64** (free in-scope knob) | 3.3803 ±0.001 (n=2) | 157.64 ±0.14 | 137.14 | +8.52% | **marginal** (+0.76) |
| **ftv1@32** (kduma retrain) | 3.6504 ±0.001 (n=2) | 170.34 ±0.03 | 148.20 | +17.26% | yes |
| **ftv1@64** (deployed `/tmp/qat-assistant`) | 3.6578 ±0.001 (n=3) | 170.15 ±0.01 | 148.03 | +17.13% | yes |

**Headline edge (deployed ftv1@64 − ships stock@32): +0.3132 espec, +13.69 wall_tps (+8.75%).** Attribution:
- **retrain (stock→ftv1 @top_k32): +0.3058 espec (98%), +13.88 wall_tps** ← the entire lever
- top_k (stock 32→64): +0.0357 espec (11%), +1.18 wall_tps
- interaction: −0.0283 espec (−9%), −1.37 wall_tps

Two consequences: (a) the win is the **weights**, not the knob; (b) on the *retrained* drafter the top_k choice is immaterial — **ftv1@32 (148.20) ≈ ftv1@64 (148.03)** — so even the deployed dir's `top_k=64` is not what buys the speed.

### Arm B robustness — prompt-resampling (n=64 subsample, seeds {1,2,3})
Retrain edge measured **four independent ways**, all tightly clustered → not a single-seed artifact:
- population (128): **+0.3132**
- off-diagonal retrain cells: **+0.3058** (@32), **+0.2775** (@64)
- subsample Δespec (ftv1@64 − stock@32), n=64 × seeds{1,2,3}: per-seed **+0.2999 / +0.3042 / +0.2921** → mean **+0.2987**, sd 0.0062, **95% CI [+0.2834, +0.3141]** → **CI excludes 0** (and excludes any +10-loss), mean Δwall_tps **+13.35**. Not a single-seed artifact.

### Arm C — proxy-chain closure (your #664 follow-up #3)
My harness at the **stock Hub drafter** (stock@32, byte-exact Google blob, BI=1, K6) = **156.46 wall_tps ≈ stark's 155.5693 (+0.57%, within hardware noise)**. The "official harness = ~155 un-rescued regime" chain is now closed end-to-end on my own stack.

### Arm A — `/tmp/qat-assistant` characterization (publishability + legitimacy)
- **Architecture/size:** `Gemma4AssistantForCausalLM` / `gemma4_assistant`, 4 layers, hidden 256 / backbone 2560, 2048 centroids, **78.78M params (50 tensors), BF16 unquantized, 152 MiB**. A tiny proposer head, **not** the ~4B int4 target → cannot be a copy of the target.
- **Provenance (sha-verified, in-scope only):** `/tmp/stock-topk32` weights sha `9d0e…c947d` **== the Google Hub blob exactly** (publishable, Google's). `/tmp/qat-assistant` weights sha `ed15…d18e` **== `DRAFTER_SHA256` in my own `submissions/fa2sw_nonspec_int4/manifest.json`**, whose `DRAFTER_BUCKET=hf://…/gemma-kenyan-duma/weights/drafter-ft/ft-v1-epoch_001`; safetensors `__metadata__.finetune = ft-v1/epoch_001.pt`. So the retrain weights are **kenyan-duma's** checkpoint.
- **Legitimacy:** bona-fide speculative *proposer* — at temp=0 vLLM's rejection sampler short-circuits to target-argmax (`serve.py:8-10`), so the drafter only affects acceptance length (speed), **never the emitted tokens**; it cannot game PPL/greedy-identity. Honest fairness flag: a drafter's only metric-gaming risk is eval-prompt overfit; the retrain recipe is out-of-scope so I cannot audit its training corpus from in-scope evidence — mitigated by (a) the subsample robustness check and (b) that the *free* alternative is the clean Google stock drafter.
- **Publishability — BLOCKED in-scope.** `/tmp` is per-pod ephemeral; the only source for the *ft-v1 weights* is kenyan-duma's bucket (cross-student) or the out-of-scope retrain recipe (commit `4d65412`, `wide_drafter`, "EXCLUDED by isolation"). The *stock* + *top_k* variants are trivially reproducible in-scope; the *retrain* (the 98% of the edge) is not.

### Guards (drafter-independent — re-confirmed)
`#319` byte-exact greedy gate `break_rate=0` and **PPL 2.019 ≤ 2.42** are **structurally drafter-independent**: under `SENPAI_REFERENCE_MODE` the submission forces `num_speculative_tokens=0` (drafter OFF → plain int4 M=1 AR, the exact-greedy reference; `serve.py:34-63`); the re-gate never loads a drafter, and at temp=0 the spec path is token-identical anyway. They carry over unchanged from the locked `int4_g128_lmhead` anchor for every drafter arm. **`analysis_only=true`, `official_tps=0`, `fires=false`** throughout. The locked 126.378 / PPL 2.019 anchor is untouched. **No Hub publish, no manifest repoint, no HF Job.**

### Command
```bash
# 2×2 main diagonal (stock@32 baseline reused) + off-diagonal, then n=64 subsample seeds 1-3
.venv/bin/python scripts/profiler/paired_tps_ab.py \
  --baseline int4_mtp_batchinv --candidate int4_mtp_batchinv \
  --baseline-label stock_topk32 --candidate-label ftv1_topk64 \
  --baseline-env VLLM_BATCH_INVARIANT=1 --baseline-env NUM_SPECULATIVE_TOKENS=6 \
  --baseline-env MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct --baseline-env DRAFTER_MODEL=/tmp/stock-topk32 \
  --candidate-env VLLM_BATCH_INVARIANT=1 --candidate-env NUM_SPECULATIVE_TOKENS=6 \
  --candidate-env MODEL_ID=google/gemma-4-E4B-it-qat-w4a16-ct --candidate-env DRAFTER_MODEL=/tmp/qat-assistant \
  --n 3 --num-prompts 128 --output-len 512 --wandb-group local-drafter-derisk-land
# (off-diagonal swaps to /tmp/stock-topk64 vs /tmp/ftv1-topk32; subsample uses --num-prompts 64 --seed {1,2,3})
```
- **Peak memory:** ~12.5 GiB / 23 GiB GPU (single A10G, MAX_NUM_SEQS=1, GPU_MEM_UTIL 0.90).
- **W&B (group `local-drafter-derisk-land`):** headline 2×2 main `o3pa7bbj`, off-diagonal `ai34kyj6`, subsample s1 `gq5c3uud`, s2 `ujwxjfam`, s3 `4rt69y8r`.

### What happened
The de-risk did its job: it **separated a real effect from an actionable one**. The +0.32 espec edge is genuine and robust (it is the population value, not a 128-prompt fluke) — so the projected ~+17% is not noise. But it is **the retrain weights**, and those are kenyan-duma's cross-student checkpoint with an out-of-scope recipe and an ephemeral `/tmp` home, so they are **not publishable inside land's launch scope**. The only in-scope-free slice of the edge — the `top_k=64` knob on the publishable Google drafter — moves stock from +7.71% to +8.52%, i.e. it merely **touches** the +10 bar (+0.76 TPS, inside the 0.870-tax projection noise), not a robust clear. So: the speed leg honestly stays **~+6.7–8.5% on publishable drafters**; the **+17% lever is real but human-gated** (it needs a decision about adopting a cross-student artifact, which is exactly the action this card was de-risking *toward*, not executing).

### Suggested follow-ups
1. **Human decision for #481:** is kenyan-duma's `ft-v1-epoch_001` drafter adoptable for the official submission (cross-student artifact + recipe reproducibility + eval-overfit audit)? If yes, the publish→repoint→re-measure path projects ~+17% (148 official-equiv). If no, the speed leg is ~+6.7%.
2. **Cross-distribution robustness (cheap):** `data/private_proxy_sharegpt.json` (PR #44, independent length-matched 128-draw) — thread `dataset=` through `paired_tps_ab.py` to confirm the retrain edge isn't ShareGPT-eval-specific (directly tests the eval-overfit fairness flag).
3. **If pursuing in-scope only:** ship `top_k=64` on the stock Google drafter as a free, identity-safe config bump — but bank it as a ~+8.5% touch-the-bar improvement, not a confident +10 clear.
