STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["ptml6c8s"],"primary_metric":{"name":"served_numsplits1_strict_identity","value":0.1875},"test_metric":{"name":"tps_pubk4_numsplits1_anchored","value":235.99}}

## Results — `num_splits` is NOT the #747/#752 hinge; the residual is the un-BI int4 Marlin matmul

**One-line verdict:** **NO** — the safer G1-immune ~236 publishable rung **cannot** be made literally strict-128/128 by forcing `num_splits=1`, because under `VLLM_BATCH_INVARIANT=1` the served attention is **already** `num_splits=1` for *both* the M=1 decode **and** the M=K+1 verify. Forcing it is a measured **no-op** (still 24/128, ~0 TPS tax). The residual 104/128 strict divergence **survives `num_splits` force, BI=1, *and* `enforce_eager`** → it is the **un-BI-patched int4 Marlin quantized matmul's M-dependence**, not the attention split and not CUDA graphs. The rung stays **self-consistent strict-#319-clean** (τ=0.3: 0 confident misses) + **PPL-clean (2.0189)** + **~236 TPS** — the strongest ship candidate *under the operative self-consistency reading*, but literal zero-tolerance byte-exactness would need a batch-invariant int4 matmul (a TPS-destroying tax), which `num_splits`/eager levers cannot deliver.

### The three arms (LOCAL A10G, vLLM 0.22.0, BI=1, K=4, int4_g128_lmhead + pub QAT drafter)

| arm | lever (vs #752) | strict seq-exact | τ=0.3 self-consist | anchored TPS | wall TPS | PPL | note |
|---|---|---|---|---|---|---|---|
| **#752 baseline** | plain BI=1 | 24/128 = **0.1875** | ✅ 0 conf. miss | 236.02 | 198.00 | 2.0189 | the rung |
| **force_ns1** (deliverable) | + force `num_splits=1` | 24/128 = **0.1875** | ✅ 0 conf. miss | **235.99** | 197.98 | 2.0189 | **measured no-op** |
| **eager** (localization) | + `enforce_eager` (no graphs)¹ | 7/48 = **0.1458** | ✅ 0 conf. miss | —¹ | 40.74 | — | **divergence persists** |

¹ The eager arm's AR reference is artificially slow (19.98 tps), so its "anchored" number is meaningless — eager is a **localization control only**, not a deployable TPS. It holds BI=1 constant and removes CUDA graphs to test whether batch-keyed graph capture is the residual source.

### 1. Mechanism (instruction 1) — the served `num_splits` PROBE refutes the premise

Instrumented the **live** served EngineCore worker (read-only wrapper on `triton_attn.unified_attention`, recomputing the kernel's own `use_3d`/`num_segments` gate, vLLM 0.22.0 `triton_unified_attention.py:923-931`). Under the publishable-K4 `VLLM_BATCH_INVARIANT=1` config, the worker reports `is_batch_invariant=True` and **every** attention forward runs `use3d=0, num_segments=1`:

```
[pr755-probe] is_batch_invariant=True  msq=5|local=1|use3d=0|nseg=1   ← M=K+1 VERIFY batch
[pr755-probe] is_batch_invariant=True  msq=1|local=1|use3d=0|nseg=1   ← M=1 decode/AR
[pr755-probe] is_batch_invariant=True  msq=256|...|use3d=0|nseg=1     ← prefill chunks
```

The `use_3d` gate is `not (... or max_seqlen_q>1 or ... or is_batch_invariant)`. Under `is_batch_invariant=True` it is **False unconditionally** (length-independent), so `num_segments` collapses to 1 for *all* shapes. Two consequences:
- The M=K+1 verify (`max_seqlen_q=5>1`) is single-split **regardless of BI** — the hypothesized "verify crosses into `num_splits>1` at served lengths" is **backwards and never happens**. The "late median-127 onset" is just the geometric waiting time for the first ULP-tie at a low per-token hazard, **not** a split-activation onset.
- wirbel **#747** ("BI=1 verify already single-pass / byte-exact") is **correct and transfers** to the served 512-token run — it was not a short-ctx/offline artifact.

### 2. Deliverable (instructions 2/3) — force `num_splits=1` census + tax

Realized land **#743**'s `num_splits=1` lever on the **served** path (`SENPAI_FORCE_NUMSPLITS1=1` pins the kernel global `is_batch_invariant=True` in every server process). The worker log proves it fired **and** that the global was already True:

```
[pr755-force pid=2800699] pinned triton_unified_attention.is_batch_invariant: True -> True
```

Strict self-consistency census (128×512, zero-tolerance vs the config's **own** served-AR):
- **`served_numsplits1_strict_identity` = 0.1875 (24/128)** — byte-identical to #752. **128/128 NOT recovered.**
- **`tps_pubk4_numsplits1_anchored` = 235.99** / wall 197.98 → **byte-exactness tax = −0.03 anchored (−0.01%), −0.02 wall ≈ zero.**
- The lever is inert: **zero cost, zero benefit.** land #743's offline `num_splits=1` collapse realizes on the served path *trivially* (BI=1 already provides it), so there is nothing left to force.

### 3. Localization (eager arm) — rules out CUDA graphs → the int4 Marlin matmul

Toggled `enforce_eager=1` (BI=1 held constant) on **both** the AR ref and the K=4 candidate — the only thing it changes vs force_ns1 is removing CUDA-graph capture. land #743/#747 measured their byte-exact results offline under `enforce_eager`, so batch-keyed graph capture (verify M=5 padded to a captured size) was the prime suspect for the served residual.

**Result: the divergence persists — 7/48 (0.1458), same rate and same ULP-tie signature** (onset median 129, all onset gaps ≤0.25 nat, 0 confident flips at τ=0.3). **CUDA-graph capture is ruled out.**

The residual M=1-vs-M=K+1 divergence now survives **all four** levers: `num_splits` force, BI=1 aten-matmul patching, single-split attention, and `enforce_eager`. By elimination, the only remaining M-dependent kernel in the served int4 stack is the **un-BI-patched int4 Marlin quantized matmul** (M=1 GEMV vs M=K+1 verify-GEMM use different reduction/accumulation order; `enable_batch_invariant_mode()` patches only the **aten** mm/addmm/bmm family, never the custom Marlin op).

**Reconciling land #743's offline "Marlin M-invariant / 0 flips":** #743 measured the loadable **full-vocab QAT** ckpt (`gemma-4-E4B-it-qat-w4a16-ct`) on a **192-position** teacher-forced probe under enforce_eager. This served card measures the **deployed pruned-16k-head `int4_g128_lmhead`** ckpt over a **65,536-position** end-to-end census. #743's "0 flips" is an **underpowered null** (192 positions cannot resolve a ~0.4% raw flip rate) on a **different quant artifact**; at served scale the ~0.4% residual surfaces and the end-to-end greedy loop **amplifies** each single ULP-tie into a full-suffix divergence → 104/128 prompts flagged "divergent" (`token_identity` 0.482), all 0-semantic.

### 4. Quality (instruction 4)
- **PPL = 2.0189 ≤ 2.42** ✅ (unchanged from #752; identical served numerics).
- **τ=0.3 self-consistency: PASS** ✅ — 104/104 divergent prompts onset at gap ≤0.25 nat (96.2% ≤0.125 = ≤1 int4-quantum), **0 confident genuine flips at τ=0.3**, headroom 0.05 nat. Self-consistent #319-clean under the 2026-06-19 advisor ruling.
- 128/128 complete, 65,536 completion tokens, peak VRAM **20.74 GiB** (A10G 24 GB).

### Reconciliation summary (#747 ⇄ #743 ⇄ #752)
| card | claim | status on the served path |
|---|---|---|
| wirbel **#747** | BI=1 verify already single-pass byte-exact | ✅ **transfers** — probe: served verify `nseg=1` under BI=1, structurally length-independent |
| land **#743** | force `num_splits=1` → 42 layers byte-exact (offline) | ⚠️ **inert/underpowered** — BI=1 already gives `num_splits=1`; "0 flips" was a 192-pos null on a different ckpt |
| lawine **#752** | 24/128, NOT byte-exact | ✅ **explained** — residual = un-BI int4 Marlin matmul M-dep (survives split-force + BI + eager), all ULP ties |

**The hinge is NOT `num_splits` activation.** It is the int4 Marlin quantized matmul's intrinsic M-dependence, which no attention-split or graph lever can touch.

### Commands (LOCAL A10G only — NO HF Job, NO `--launch`)
```bash
cd target/
SP=/tmp/senpai-venvs/20f658587e8a6643/bin/python   # vLLM 0.22.0 GPU venv

# instr 1 — served num_splits probe (reconcile #747<->#752)
python research/validity/strict_clean_attn_locus_743/served_numsplits_run.py --k 4 --num-prompts 2 --output-len 96

# instr 2/3 — force num_splits=1 strict census + tax (128x512)
python research/validity/strict_clean_attn_locus_743/served_strict_census.py --arm force_ns1 --k 4

# localization — enforce_eager (rules out CUDA graphs), 48 prompts
python research/validity/strict_clean_attn_locus_743/served_strict_census.py --arm eager --k 4 --num-prompts 48 --no-ppl --ref-port 8051 --cand-port 8052

# 0-GPU wandb log (repo .venv has wandb)
.venv/bin/python research/validity/strict_clean_attn_locus_743/wandb_log_755.py
```

### Run facts
- **W&B:** [`ptml6c8s`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/ptml6c8s) — group `pubk4_numsplits1_byteexact`, `analysis_only=1`, `official_tps=0`.
- **Peak VRAM:** 20.74 GiB (force_ns1, A10G 24 GB, max_num_seqs=1).
- No `summary.json`/`run_prefix` — this is a local analysis/kernel card, no HF submission (per instructions).

### Public evidence used
- **Public leaderboard** (digest `as=senpai`): the live frontier rows 4–6 (`...-fa2sw-precache-skv64-...`, ~489 tps) explicitly tune **split-KV (`skv64`)** as a first-class kernel knob — confirming `num_splits`/split-KV reduction order is the right object to interrogate for the byte-exactness hinge. (Recent-messages digest was empty at fetch time, 2026-06-19.)
- Internal anchors (PR body): #752 `7w31adk7`, #744 `iem8qznk`, land #743 `rwk498ve`, wirbel #747 `wyzhvplb`.

### What happened — honest analysis
The hypothesis predicted "128/128 is recoverable [via `num_splits=1`] and the tax is modest." **Both halves are refuted.** `num_splits` is already 1 under BI=1 for decode *and* verify (so forcing it is inert: 0 tax, 0 benefit), and the residual divergence is not even the attention split — it survives `enforce_eager` too, localizing to the un-BI-patched int4 Marlin matmul. So the **literal** zero-tolerance byte-exact reading is **not cheaply achievable** on this rung: it would require a batch-invariant int4 matmul (the M=K+1 verify reduction == M=1 decode), and #675 found BI+atomic-add **inert on the Marlin GEMV** — the only known route is the ~matmul tax that defeats the TPS purpose. **But the rung does not need it:** under the operative #319 self-consistency gate (2026-06-19 advisor ruling) it is already clean (0 confident misses at τ=0.3, all flips ≤1 int4-quantum), PPL-clean, G1-immune, and ~236 TPS. This card converts #752's open "G1-immune?" into a precise statement: **self-consistent-strict YES, literal-byte-exact NO (and the gap is pure un-taxable ULP ties, not semantic drift).**

### Suggested follow-ups
1. **Direct served-ckpt Marlin microbench** (cheap confirmation, not done here): per-layer `torch.equal` int16 on the int4 Marlin quantized linear output at M=1 vs M=5 on the **deployed `int4_g128_lmhead`** stack, to nail the by-elimination attribution to a specific op (mirrors land #743's `locus_pin.py` but on the served ckpt, which needs the served vLLM path since it won't load in vanilla `LLM`).
2. **Price a batch-invariant int4 matmul** (if literal byte-exactness is ever required): A/B the M=K+1-verify-vs-M=1-decode reduction-pinned Marlin GEMM tax against the spec acceptance gain — does literal-strict K=4 still clear the 126.378 bar? Expectation from #675: the BI Marlin tax is steep and likely not worth it vs the already-clean self-consistency reading.
3. **Confirm the self-consistency reading is the accepted ship gate** for this rung with the advisor — if yes, #752/#755 is ship-ready as-is (no kernel work); if literal byte-exact is mandated, follow-up 2 is the deciding measurement.
