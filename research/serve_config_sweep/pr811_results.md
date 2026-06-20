STUDENT fern:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["zzd2u9uf","p7h9e9el","33fj96l5","8v9jek10","g20uxfxz","9bqdnbzp","a0y5apgj","xdwa9w23","etkpijrn"],"primary_metric":{"name":"best_byteexact_local_tps_delta_pct_vs_control","value":0.0},"test_metric":{"name":"control_local_decode_tps_proxy_conc1","value":254.2}}

## Results

**Verdict: double NULL — (1) no serve-config knob yields free TPS, and (2) the lane's "byte-exact by construction" premise is false because the served int4head stack is not boot-deterministic at the token level.** Capping the lane per #784. **Recommended manifest diff: none — keep the shipped config.** No HF Job launched (LOCAL A10G only); no `summary.json`/`run_prefix` exists because no submission was run — all numbers below are the local single-stream decode proxy.

### Per-knob sweep — LOCAL A10G, conc=1 / `MAX_NUM_SEQS=1`, `output_len=512`, 128 ShareGPT prompts (seed=1)

TPS = decode `num_completion_tokens / duration_s` (local proxy, **NOT** the official a10g-small number). Every prompt emits exactly 512 tokens (`ignore_eos`), so length is constant across cells and ΔTPS is purely duration. "byte-exact (warm)" = the warm-cache (95 s) rep's per-prompt completion-token sha256 set equals control's.

| knob (env override) | n | median TPS | Δ% vs ctrl | valid | byte-exact (warm boot) | cold-boot stream |
|---|---:|---:|---:|:--:|:--:|---|
| **control** (shipped) | 4 | **254.20** | — | ✓ | ✓ (4/4 identical) | n/a — warm 95 s, spread 0.03% |
| `MAX_MODEL_LEN=3072` | 2 | 253.88 | −0.13% | ✓ | ✓ (rep1) | ✗ rep0 diverged **97/128** prompts |
| `MAX_MODEL_LEN=2048` | 1 | — | — | ✗ | — | **HTTP 400** — truncates the 2427-tok prompt |
| `MAX_MODEL_LEN=1024` | 1 | — | — | ✗ | — | **HTTP 400** — truncates 6/128 prompts |
| `MAX_NUM_BATCHED_TOKENS=2048` | 2 | 253.56 | −0.25% | ✓ | ✗ (both reps diverge) | rep0 105/128, rep1 106/128 — two *different* streams |
| `GPU_MEMORY_UTILIZATION=0.95` | 1 | 253.89 | −0.12% | ✓ | ✓ | — (non-graph; KV idle at conc=1) |
| `CUDAGRAPH_CAPTURE_SIZES="1 8"` | 2 | 253.94 | −0.10% | ✓ | ✓ (rep1) | ✗ rep0 diverged **98/128** prompts |
| `VLLM_MARLIN_USE_ATOMIC_ADD=1` | 1 | 251.24 | −1.16%* | ✓ | ✓ (output bit-identical) | — |

*marlin_atomic −1.16% is a single **cold-boot** (155 s) timing artifact, **not** a real regression: its output hash is byte-identical to control, so the knob is functionally inert (see below).

**No knob clears the +1% byte-exact bar — every valid knob is within −0.25%..−0.13% of the 254.20 control (control warm-rep spread is 0.03%). The serve config is already optimal for this workload.**

### Why there is no free TPS
At `conc=1 / MAX_NUM_SEQS=1` the engine reserves an 8.1 GiB KV cache = **322,912 tokens = 78.84× concurrency headroom** for a workload whose longest single request is 2,939 tokens. The KV cache is essentially idle, so the knobs the PR hoped would help are inert at this operating point:
- **`MAX_MODEL_LEN` right-sizing is a non-lever** (and the PR's premise was wrong): the speed-benchmark prompts are ShareGPT, **input up to 2427 tok** (not ~128), so the true floor is **2939**. `1024`/`2048` truncate real prompts → `VLLMValidationError ... maximum context length is 2048 tokens ... prompt contains 2427 input tokens` → HTTP 400 (invalid, not "right-sized"). The only legal right-size is `≥2939`; `3072` works but moves TPS −0.13% because KV physical size at conc=1 is `GPU_MEMORY_UTILIZATION`-bound, not `max_model_len`-bound — the attention kernel reads only the actual ≤2939 KV either way.
- **`GPU_MEMORY_UTILIZATION 0.90→0.95`** just enlarges the already-idle reservation (8.1→9.21 GiB): −0.12%, byte-exact.
- **CUDA-graph capture set** `{1,8}` vs default: −0.10%. The conc=1 M=7 verify path is already captured; pruning never-hit sizes doesn't help.
- **`MAX_NUM_BATCHED_TOKENS 512→2048`**: −0.25%. Irrelevant at conc=1 as expected.
- **Marlin launch config**: confirmed **structural null** (see below).

### The "byte-exact by construction" premise is FALSE — the served stack is not boot-deterministic
This is the more important finding. The PR assumed config-only changes preserve the token stream "by construction." They do not, because the served **int4-Marlin GEMM + K=6 MTP spec-decode + `VLLM_BATCH_INVARIANT=0`** stack is **not boot-deterministic**:

- The **same config on different boots produces wholesale-different token streams.** `MAX_MODEL_LEN=3072` (which truncates nothing and cannot change the math) was byte-identical to control on its warm boot (rep1) but diverged on **97/128 prompts** on its cold boot (rep0). `MAX_NUM_BATCHED_TOKENS=2048` diverged **two different ways** across its two reps (105/128 then 106/128, distinct hashes).
- Across the sweep I observed **≥5 distinct valid streams** for the same checkpoint. Divergence is **not corruption**: spot-checking a divergent completion, it is identical to control for the first ~108 tokens, then a near-tie logit flips the greedy argmax and the (still coherent, valid) completion cascades apart.
- **Mechanism**: graph-shape-altering knobs (`MAX_MODEL_LEN`, `MAX_NUM_BATCHED_TOKENS`, `CUDAGRAPH_CAPTURE_SIZES`) force a cold recompile / CUDA-graph re-capture (~155 s vs ~95 s warm). The recompiled boot can land on a different FP-reduction attractor. **9/10 warm-cache boots reproduce control; graph-altering cold boots usually don't** (and `batched2048` didn't reconverge even warm). Non-graph knobs (`GPU_MEMORY_UTILIZATION`, and the inert `VLLM_MARLIN_USE_ATOMIC_ADD`) reuse the identical compiled artifact → byte-exact even cold.
- **Consequence for this lane's methodology:** byte-exact parity cannot certify a config knob here, because the control reference is only *empirically* stable (4/4), not *provably* boot-invariant (the `maxlen3072` A-vs-divergent split proves it). The parity signal is dominated by boot-attractor selection, not by the knob.

### Greedy-identity implication — flagging for advisor review
`program.md` validity gate (line 27): *"Greedy decode must remain token-identical to plain greedy autoregressive decode for the submitted checkpoint."* If the served int4head stream is boot-unstable (shown above), then **at most one boot-attractor can equal the plain-greedy reference R**, so greedy-identity is *boot-fragile*: a fresh-container cold boot on the a10g-small runner is exactly the recompile case most likely to land on a non-control attractor. This connects directly to the open board question *"@senpai is there a separate validation other than perplexity?"* (human-dhruv-mishra, 2026-06-20 11:06 UTC) — the answer is yes (greedy-identity + 128/128 + multimodal-intact), and greedy-identity is the gate this finding touches.

**Scope/caveats (not over-claiming):** I measured served-stream *self-consistency across boots*, **not** the served stream vs plain-greedy R — so I am **not** asserting the current submission passes or fails the gate. PPL itself is unaffected (it is teacher-forced on fixed ground-truth tokens, attractor-independent), and 128/128 completion is unaffected (every attractor emits 512 tokens). The new fact is only that the *decode stream* is not boot-reproducible, which makes the greedy-identity gate boot-dependent. I did not implement the R-comparison (out of #811 scope) — see follow-ups.

### Marlin atomic-add sub-lever — structural null confirmed (source + empirical)
`VLLM_MARLIN_USE_ATOMIC_ADD=1` produced a **byte-identical output hash** to control → functionally inert, exactly as predicted by source read: `marlin_utils.should_use_atomic_add_reduce` returns `False` when `device_capability[0] < 9 and dtype == bfloat16` (sm_86 A10G + bf16 → split-K atomic-add path is hard-gated off). vLLM 0.22 exposes **no** user-tunable Marlin tile/num-warps/split-K knob. The body-MLP 74–79% HBM-BW at M=7 (stark #798) is the genuine auto-tile 1-wave floor, not a flag — the lever is fewer bytes, not a launch-config knob. **This sub-lever is dead.**

### Code changes in this PR
- `submissions/int4_mtp_bi0_int4head/serve.py`: added an **optional, default-OFF** `CUDAGRAPH_CAPTURE_SIZES` env → `--cudagraph-capture-sizes` passthrough (the instrument used to A/B the capture set). Unset ⇒ vLLM's default selection ⇒ **shipped behavior is unchanged byte-for-byte**. Kept so the cgsizes cell is reproducible from committed code; safe to prune on merge if preferred (verdict is "no config lever").
- `research/serve_config_sweep/`: sweep harness (`sweep.py`, `analyze.py`, `wandb_log.py`, `promptlen.py`), `results.jsonl`, per-cell decode outputs + serve logs (byte-exact evidence). No manifest/weight/quantization/drafter changes.

### Command (LOCAL only — no HF Job)
```bash
cd target/research/serve_config_sweep
# control + each knob is one fresh-server run; repeat a label for reps:
/tmp/senpai-venvs/20f658587e8a6643/bin/python sweep.py \
  control control control control maxlen3072 maxlen3072 maxlen2048 maxlen1024 \
  batched2048 batched2048 gpumem095 cgsizes_1_8 cgsizes_1_8 marlin_atomic \
  --num-prompts 128 --output-len 512 --port 8033
# sweep.py launches submissions/int4_mtp_bi0_int4head/serve.py (MODEL_ID=/workspace/gemma_build/bi0_int4head_g32)
# with per-knob env, scores with the official decode_outputs.py, hashes per-prompt completion tokens.
/usr/bin/python3 analyze.py        # per-knob TPS delta + byte-exact verdict
cd /tmp && /usr/bin/python3 .../wandb_log.py   # log to W&B group serve-config-tps-sweep
```

### Peak memory
~**19.7 GiB** used of 22.5 GiB usable (A10G): model weights 10.22 GiB + KV reservation 8.1 GiB (at `GPU_MEMORY_UTILIZATION=0.90`) + ~0.05 GiB CUDA graphs. `gpumem095` raises the (idle) KV reservation to 9.21 GiB. No OOM.

### W&B
Group **`serve-config-tps-sweep`** (`wandb-applied-ai-team/gemma-challenge-senpai`). Summary run **`zzd2u9uf`** (master per-knob table + verdict `NULL_CONFIG_ALREADY_OPTIMAL`); per-knob cells: control `p7h9e9el`, maxlen3072 `33fj96l5`, batched2048 `8v9jek10`, cgsizes_1_8 `g20uxfxz`, gpumem095 `9bqdnbzp`, marlin_atomic `a0y5apgj`, maxlen1024 `xdwa9w23`, maxlen2048 `etkpijrn`. Each cell logs `steady_state_byte_exact`, `cold_rep0_diverged`, both parity hashes, and `graph_altering`.

### What happened
The hypothesis was that the bi0-inherited serve config was untuned for the 640-token single-stream workload and held free byte-exact TPS. Both halves failed: (a) the config is already at its floor for conc=1 (the KV cache is 78.84× oversized and idle, so the proposed knobs are inert — every valid one is within noise, several slightly negative); and (b) the prompts are not short (ShareGPT, up to 2427 tok), so the headline `MAX_MODEL_LEN` right-size is half-invalid (1024/2048 truncate) and the legal `3072` does nothing. The unexpected payoff was discovering the served stack is **not boot-deterministic** — "byte-exact config tuning" is ill-posed on this stack — which matters for the greedy-identity validity gate.

### Suggested follow-ups
1. **Measure served-vs-plain-greedy on a COLD boot** (the real gate, line 27): run plain greedy autoregressive decode of the int4head checkpoint (no spec, no CUDA graphs) to get reference R, then compare a *cold-boot* served stream to R. This resolves whether the greedy-identity gate is actually at risk on a fresh a10g-small container, and whether attractor "A" == R. (Out of #811 scope — flagging for a dedicated PR.)
2. **If boot-reproducibility is wanted**, the lever is `VLLM_BATCH_INVARIANT=1` (batch-invariant kernels) — but note it does **not** cover int4-Marlin GEMM (prior finding), so it may not fully stabilize this stack, and it costs TPS. Worth a scoped test only if the gate proves at-risk.
3. TPS gains on this stack require **fewer body-MLP bytes** (sub-int4 / W4A8), not serve-config flags — config lane is exhausted.

### Public evidence used
- `program.md` validity gates (PPL ≤ ~2.42, 128/128, **greedy token-identity**, multimodal-intact) — motivates the byte-exact framing and the greedy-identity flag.
- Board inbox: open `@senpai` question on validation-beyond-PPL (human-dhruv-mishra, 2026-06-20 11:06 UTC) — directly relevant to the greedy-identity finding.
- Did not borrow from any other agent's leaderboard method (frontier ~513 TPS is out-of-scope for this byte-exact int4head config lane).
