STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["oyqek0ou"],"primary_metric":{"name":"served_min_tau_for_zero_break","value":0.25},"test_metric":{"name":"frac_sub_0p5","value":1.0}}

## Results

**VERDICT: `FLAG_COVERS_ALL`.** Every one of the 108 served root forks sits at a verify-margin **< 0.5 nat** (max 0.25). Stark #636's τ=0.5 flag provably catches 100% of the #632 served divergences — **no coverage hole**, with 2× headroom (worst served fork 0.25 vs his threshold 0.5).

### Deliverable

| field | value |
|---|---|
| `n_prompts_diverged` | **108** (of 128; 20 never fork) |
| `n_root_forks` | **108** (one root per diverged prompt) |
| `frac_sub_0p5` `[0,0.5)` → caught | **1.0 (108/108)** |
| `frac_0p5_to_1p0` `[0.5,1.0)` → hole | **0.0 (0/108)** |
| `frac_ge_1p0` `≥1.0` → loud hole | **0.0 (0/108)** |
| `margin_median` | 0.125 nat |
| `margin_p95` | 0.125 nat |
| `margin_max` | **0.25 nat** |
| `served_min_tau_for_zero_break` | **0.25 nat** |

Margin = log-prob gap between the AR-greedy (M=1 argmax) token **B** and the spec-emitted (M=8 argmax) token **A**, read from the served K=7 verify forward (vLLM `/v1/completions` `logprobs`). Reported both ways — they coincide:
- **AR-token margin** `logp(A)−logp(B)` (PR instr #2): min 0.0 / median 0.125 / p95 0.125 / **max 0.25**, mean 0.069.
- **M=8 top1−top2 gap** `logp(top1)−logp(top2)` (stark's literal threshold quantity): identical bins, **max 0.25**.

Every margin lands exactly on the **bf16-ULP grid** {0.0, 0.125, 0.25} = {0, 1, 2 ULPs} at logit magnitude [16,32) — the near-tie-flip signature. This is FP precision in the M-dependent int4-Marlin reduction, not quality (consistent with #632 PPL 2.0055 unchanged, #122/#576 `genuine_precision`).

### Cross-check vs stark #636 (PR instr #4)

His `min_tau=0.5` was measured **teacher-forced** (`break_rate=0/14035`). My census is on the **served** trajectory. Served `served_min_tau_for_zero_break = 0.25 < 0.5` → **his threshold transfers to the served path with margin to spare** (`transfers_to_served=True`). The served margins are *tighter* than his TF measurement implied; there is no served position at/above 0.5, so no teacher-forced/served disagreement to name. His flag (which fires on 7.80% of TF positions) recomputes every served root fork → served stream byte-exact by construction.

**Runner-up identity (sanity that the margin is the right quantity):** at a root fork the spec-accepted prefix is byte-identical to AR, so the M=8 verify slot shares the M=1 AR causal context (differs only by varlen width). Prediction: the AR token B is the M=8 *runner-up*. Confirmed empirically on **107/108 (99.1%)** forks (the 1 exception is a perfect 0.0-nat A↔B tie where "runner-up" is degenerate). So `margin_AB ≡ margin_AC` and stark's top1−top2 threshold is exactly the AR-token margin.

Worst forks (all near-ties of adjacent tokens): `gpqa_diamond-183797b844` p139 A=`')-'` B=`'),'` 0.25; `mmlu_pro-00996c6808` p103 A=`'4'` B=`'3'` 0.25; `mmlu_pro-00a3fcc287` p89 A=`' being'` B=`' made'` 0.25.

### Method / faithfulness

- **Root forks (no new generation):** reused #632 on-disk captures — AR M=1 ref (`ar_ref_bi1/decode_outputs.jsonl`, served spec-OFF, BASELINE.md L10) and K=7 spec lane (`k3/k7/decode/run00.jsonl`), with `gate_k7.json` first-divergence. Independently recomputed each first-fork and asserted prefix byte-identity up to the root, A≠B (108 forks; matches #632's 20-identical/108-divergent gate exactly).
- **Verify margins (faithful replay):** re-served the **exact** #632 payload (stored `prompt_token_ids`, temp=0, `add_special_tokens=false`, `ignore_eos=true`, `return_token_ids=true`) + `logprobs=20`, verifying the completion sha256 reproduces #632. **126/128 prompts replayed byte-exact** → 106/108 root forks read on a byte-identical served trajectory.
- **2 near-tie-sensitive prompts (idx 0,1):** requesting logprobs is *not* perfectly trajectory-neutral at a perfect ULP tie (the #632 capture requested none), so 2 prompts re-serve to a stable but different trajectory. Handled faithfully:
  - **fork1** `mmlu_pro-006f3a2112` p161: replay diverges *exactly at 161* (prefix[:161] byte-identical to #632) → **on-#632-context**. A=`'-'`, B=`' bits'` are **perfectly tied (gap 0.0)**; the argmax flips A↔B purely by reduction order. margin = 0.0, faithful.
  - **fork2** `mmlu_pro-012f0d5c8d` p214: replay diverges at 54 (off-context at 214). Best **stored-context estimate** via forced-prefix forward (feed stored prefix[:214]): gap **0.125**, A on top. Tagged `offtraj_estimate`; < 0.5 by every measurement and by root-fork construction.
  - **Robustness:** the strict 107-fork subset (drop the 1 estimate) gives the **identical** result — `frac_sub_0p5=1.0`, max 0.25, `FLAG_COVERS_ALL`. Fork2 cannot change any bin or the verdict.

### Exact commands

```bash
# Engine point = #632 served Option-B BI=1 spec lane (int4 w4a16, K=7), 1×GPU (CUDA_VISIBLE_DEVICES=0)
VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
/tmp/senpai-venvs/20f658587e8a6643/bin/python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-E4B-it-qat-w4a16-ct --served-model-name gemma-4-e4b-it \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.90 --max-num-seqs 1 --trust-remote-code \
  --no-enable-log-requests --max-num-batched-tokens 512 \
  --speculative-config '{"model": "/tmp/qat-assistant", "num_speculative_tokens": 7}'

cd research/walltps_ab/optionb_bi1_stock_int4/margin_census
python extract_root_forks.py            # 108 root forks from #632 data
python capture_margins.py               # faithful re-serve + logprobs (126/128 sha-exact)
python rematch_flipped.py --attempts 12 # confirm 2 near-tie prompts perturb under logprobs
python forced_prefix_margins.py         # stored-context measure for fork2 (+anchor validation)
/tmp/senpai-venvs/20f658587e8a6643/bin/python analyze_margins.py   # histogram + verdict (needs transformers)
WANDB_PROJECT=gemma-challenge-senpai WANDB_ENTITY=wandb-applied-ai-team \
  .venv/bin/python log_margin_census_wandb.py                      # → W&B
```

- **W&B run:** `oyqek0ou` — group `served-margin-census-land` ([link](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/oyqek0ou)). Logs per-fork table, both margin histograms, the 3-bin coverage table, and the verdict.
- **Peak VRAM:** 19921 MiB (1× GPU, served K=7 spec engine).
- **`analysis_only=true`, `official_tps=0`** — no HF Job, no submission, no served-file change. Pure offline analysis of the #632 served streams.

### Public evidence used

- **#632** (my prior PR, this branch lineage): served Option-B BI=1 spec diverges from served M=1 AR on 84% (108/128), K-independent, PPL 2.0055 unchanged. Provides the AR ref + K=7 spec captures + gate reused here. W&B K=5 `uo6netrr`, K=7 `8sfauo3i`.
- **stark #636** (advisor-provided in PR body): gap-flagged M=1-recompute acceptor, `min_tau=0.5`, `break_rate=0/14035` teacher-forced, flag fires 7.80% of positions (`ukiyyuca`). Used only as the cross-check target (numbers taken from the PR body, not re-derived).
- **Merged mechanism** (BASELINE.md / merged history): #114 spec-vs-AR divergence, #122 BI=1 ≠ M-invariant (sole M-dependent reduction = int4 Marlin GEMM), #576 `root_cause=genuine_precision`. Explains why root-fork margins are ULP-quantized.

### What happened

The two results compose cleanly and the answer is a clean **yes, the flag covers the 84%**. The reason is structural, not lucky: a *root* fork is by definition the first position where served spec leaves the AR trajectory, so its preceding context is byte-identical to AR. The M=8 verify forward there shares the M=1 AR causal context up to varlen-width FP — so the only thing that can make spec pick a different argmax than AR is a sub-ULP perturbation, which by construction bounds the margin to a few bf16 ULPs (≤ 0.25 nat here). There is no mechanism that could produce a served root fork at ≥ 0.5 nat (that would require A and B to be genuinely far apart in probability, contradicting "the prefix was identical and greedy"). So `FLAG_COVERS_ALL` is not just observed — it's the expected shape, and stark's 0.5 threshold has built-in headroom over the entire served population.

### Suggested follow-ups

1. **Non-root coverage:** I measured root forks (sufficient — catching the root prevents the cascade). If the advisor wants belt-and-suspenders, census the *compounding* forks too (post-root positions where context already differs); those can in principle exceed 0.5 nat, but stark's flag still recomputes them per-position so it shouldn't matter.
2. **Tighten the trajectory-neutral capture:** the 2 perturbed prompts show `logprobs=20` can tip a perfect ULP tie. A logprob-free capture that emits per-position top-2 logits via a side channel (or a teacher-forced single-step probe at each stored root) would read all 108 fully on-trajectory — only relevant if an exact value for fork2 is ever needed (it doesn't change the verdict).
3. **Fire de-risk:** coverage (this PR) + recompute speed (stark #642) are the two gates before Option-B fires. With coverage proven hole-free, the remaining question is purely whether the 7.80%-fire recompute survives serving overhead at competitive TPS — stark's lane, not mine.
