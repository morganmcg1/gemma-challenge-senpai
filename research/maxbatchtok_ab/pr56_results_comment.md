STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["3756geng","3vvsjm10","q28zoru2","k76d5d0a"],"primary_metric":{"name":"maxbatchtok_parity_confirmed","value":1},"test_metric":{"name":"ppl_control_512","value":2.3767}}

## Results — `max_num_batched_tokens` served A/B on the split-KV #1 stack

**Verdict: outcome (b), strengthened — parity + the launch warning is benign and structurally un-silenceable. No config change. Keep `MAX_NUM_BATCHED_TOKENS=512`.**

Single-variable served A/B on the deployed `fa2sw_precache_kenyan` (linear MTP K=7 + #43 split-KV). Only `MAX_NUM_BATCHED_TOKENS` was swept (extra-env override → `serve.py` → `--max-num-batched-tokens`); every other served arg held at the manifest. 128 public prompts × 512 tok, conc=1, seed 1. Steady-state TPS = vLLM's own per-interval "Avg generation throughput" meter (the canonical metric behind the 428.37 baseline), snapshotted on the decode phase before the PPL pass.

### A/B table

| `max_num_batched_tokens` | steady-state TPS (n) | Δ vs control | wall TPS | PPL | completion | split-KV engaged | startup warning | valid? |
|---|---|---|---|---|---|---|---|---|
| **512 (control / deployed)** | **448.01** (14) | — | 454.35 | **2.3767** | 128/128 | ✅ 5× M=8→3D, 0 fb | `vllm.py:1597` spec-decode | ✅ |
| 2048 | 445.92 (14) | −2.09 (−0.47%) | 450.83 | **OOM** | 128 decode, PPL crash | ✅ 5× M=8→3D, 0 fb | `vllm.py:1597` spec-decode | ❌¹ |
| 4096 | 453.40 (14) | +5.39 (+1.20%) | 452.72 | **OOM** | 128 decode, PPL crash | ✅ 5× M=8→3D, 0 fb | `vllm.py:1597` spec-decode | ❌¹ |
| 8192 | 449.56 (14) | +1.55 (+0.35%) | 453.09 | **OOM** | 128 decode, PPL crash | ✅ 5× M=8→3D, 0 fb | `scheduler.py:281` exceeds max_model_len | ❌¹ |

¹ The decode pass completed 128/128 on every arm; the PPL/`prompt_logprobs` pass OOMs at `mbt ≥ 2048` (see below).

### Comparison vs baseline (PR body)
- **Deployed frontier:** 481.53 official / **428.37 local steady** / PPL 2.3767 / 128-128.
- **My control (512) reproduces it:** steady **448.01**, PPL **2.3767** (exact), 128/128, split-KV engaged. (The 448.01 vs 428.37 gap is run-to-run noise — see noise floor below.)

### Decode TPS is parity (noise), not a frontier bump
- Inter-arm steady-TPS spread is **445.9–453.4, all within ±1.2%** of control.
- **The control's own run-to-run variance is larger than any inter-arm delta:** two independent runs of the *identical* 512 config gave **429.04** and **448.01** steady TPS (+4.4%). That +4.4% same-config swing dwarfs the max inter-arm delta (+1.2% at 4096), so **no arm is distinguishable from control.**
- **Why it must be parity:** at conc=1 / `max_num_seqs=1` with spec decode, each decode step verifies only **M=8** tokens (confirmed: split-KV redirect `M=8` on all arms) — far below any `mbt`. `max_num_batched_tokens` only governs **prefill chunking** (`scheduler.py:239 "Chunked prefill is enabled with max_num_batched_tokens=…"`), and prefill is a small fraction of a 512-token decode-dominated run. The decode throughput is structurally insensitive to this knob. Acceptance E[accept] ≈ 3.85–3.88 tok/step was flat across arms too.

### Raising `mbt` OOMs the PPL validity pass
- At `mbt ≥ 2048`, the `prompt_logprobs` path (`_get_prompt_logprobs_dict → compute_logprobs → logits.log_softmax(dim=-1, dtype=torch.float32)`) tries to allocate **+1.34 GiB** over the larger prefill chunk and OOMs → `EngineDeadError`.
- Process footprint **grows monotonically with `mbt`**: 20.92 → 20.96 → 21.02 GiB (free 1.13 → 1.09 → 1.03 GiB) on this 22.06 GiB A10G. Decode (no `prompt_logprobs`) is unaffected; the **validity gate** is what crashes.
- **Caveat:** my local A10G is 22.06 GiB; the official **a10g-small is ~24 GiB**, and the shortfall is only ~0.2–0.3 GiB — so this OOM *might not* reproduce on the official GPU. But decode TPS shows **no upside** to raising `mbt` regardless of GPU size, so raising it only buys PPL-pass memory risk for zero benefit.

### The #52 warning is benign AND cannot be silenced
Two mutually-exclusive vLLM warnings bracket this operating point:
- `vllm.py:1597` (the #52 warning) fires when `max_num_scheduled_tokens` (≈ `mbt`) **< 8192** → silent only at `mbt ≥ 8192`.
- `scheduler.py:281` fires when `mbt > max_num_seqs × max_model_len` (= 1×4096) → silent only at `mbt ≤ 4096`.
- The silence regions (`≥8192` vs `≤4096`) **never overlap**, so *some* warning always fires. Confirmed empirically: 512/2048/4096 → `vllm.py:1597`; **8192 → `vllm.py:1597` count = 0 (silenced) but `scheduler.py:281` appears instead.** The only value that silences the spec-decode warning (8192) also OOMs PPL.
- At `max_num_seqs=1` the decode step needs only 8 token slots, so the warning's premise ("suboptimal scheduling") is a false-positive heuristic here.

**Net:** `MAX_NUM_BATCHED_TOKENS=512` is already an *explicit, deliberate* value in `manifest.json` (it is **not** "whatever default vLLM picked" — correcting the hypothesis premise; the manifest set it). It is decode-optimal and the **only** `mbt` that passes the PPL gate. The launch warning is harmless and unavoidable without changing load-bearing params (`max_num_seqs`, `max_model_len`, or the spec config). **No `manifest.json`/`serve.py` change is warranted** — so I did not touch any served arg (no `validate_submission` rerun needed). **No HF job launched.**

### Split-KV guardrail (PR Step 3)
Split-KV stayed engaged on **all four arms**: `[splitkv-verify] … verify batch M=8 … -> 3D split-KV`, 5 redirects, **0 fallbacks** each. No arm silently dropped to the 2D path, so every TPS reading is valid (just parity).

### Exact command
```
/workspace/senpai/target/.venv/bin/python research/maxbatchtok_ab/maxbatchtok_ab.py \
  --arms 512,2048,4096,8192 --num-prompts 128 --output-len 512 \
  --wandb-group maxbatchtok-served-ab --out-dir research/maxbatchtok_ab
```
(Orchestrator runs under repo `.venv` — the per-submission serve venv ships no `wandb`; serve/decode/PPL still run under the serve venv via the fixed `SERVER_PYTHON`. See bug-fix note below.)

### Peak memory
- Control (512, valid): reserves to `gpu_memory_utilization=0.90` (~19.85 GiB of 22.06 GiB); **9.46 GiB KV cache**, GPU KV cache size 376,880 tokens, 92.01× max concurrency for a 4096-token request. No OOM.
- Invalid arms (2048/4096/8192): peak **20.92 / 20.96 / 21.02 GiB**, then OOM on the +1.34 GiB `prompt_logprobs` log-softmax.

### W&B (group `maxbatchtok-served-ab`)
512 → `3756geng` · 2048 → `3vvsjm10` · 4096 → `q28zoru2` · 8192 → `k76d5d0a`

### What happened
The hypothesis resolves cleanly to outcome (b) — parity — but with two extra teeth: (1) the knob has **no decode-TPS leverage** at this operating point (decode batches only M=8 regardless), and (2) raising it **actively OOMs the validity pass** while still not producing a warning-free config. The shipped 512 is correct. The launch warning is a benign spec-decode heuristic that no single `mbt` value can silence.

*Internal note (not a gate):* greedy token-identity is uninformative here — even control-vs-its-own-reference is only 10/128 identical, i.e. the spec stack is intrinsically run-to-run nondeterministic, so cross-arm greedy divergence reflects that, not the knob. PPL (distributional) is the meaningful validity signal and is unchanged at 512 (2.3767). Consistent with greedy-identity being internal-only (kanna #38).

### Bug fix made (please review)
My A/B harness (`research/maxbatchtok_ab/maxbatchtok_ab.py`, research-only — not a served file) crashed the whole sweep on the first full run because the per-submission **serve venv has no `wandb`** (and the local `./wandb` run-logs dir shadows the import → `module 'wandb' has no attribute 'init'`). Fixes: (a) run the orchestrator under repo `.venv` (which has wandb); (b) made `_log_wandb` and the PPL pass **non-fatal** so a logging error or a PPL OOM can never discard a completed arm (the PPL OOM at `mbt ≥ 2048` is now captured as data, `engine_oom=true`). No served files touched.

### Suggested follow-ups
- **None on this knob** — it is closed (parity + invalid-above-512). 
- If the `vllm.py:1597` log line is undesirable for submission hygiene, the only lever is `num_speculative_tokens` / `max_num_seqs` — both load-bearing for the K=7 stack — so it's not worth touching; recommend documenting the warning as benign instead.
- Orthogonal: the `prompt_logprobs` OOM headroom is a reminder that the validity pass, not decode, is the memory-tight phase at 0.90 util — worth keeping in mind for any future change that grows activation buffers.
