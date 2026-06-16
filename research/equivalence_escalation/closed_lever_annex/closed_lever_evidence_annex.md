<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Closed-Lever Evidence Annex — every strict TPS lever and its closure

**PR #456 · denken · `equivalence-escalation-anchors` · analysis-only (0 TPS, no HF job, no submission, no served-file change) · PPL 2.3772 (gate ≤ 2.42) · 2026-06-16**

This annex is the auditable appendix for the **relax-strict-equivalence decision** the program is bringing to the human ([issue #407](https://github.com/morganmcg1/gemma-challenge-senpai/issues/407)). It is the complete closed-lever ledger that extends denken #447's verify-wall map ([`crrq2e1y`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/crrq2e1y)) into **every strict TPS lever we have tried, the W&B run that closed it, and the physics/measurement that closes it** — so a skeptical reviewer can verify: *the strict frontier is genuinely closed; the only material headroom is greedy-unsafe.*

Every cited run id below is a **live, finished** W&B run (all 18 verified for this annex).

---

## TL;DR verdict

- **Both equivalence axes are closed.** SUPPLY (faster verify) and DEMAND (more accepted tokens) each have every cheap lever realized-NULL or build-NO-GO.
- **The best realized byte-exact lever is +0.2613 TPS** (#447 verify-wall Triton retune) — noise-level, far below the σ_hw ≈ 4.8 measurement floor.
- **The realized blanket-strict frontier sits at 467.14 TPS — 14.39 below the deployed (non-equivalent) 481.53** (−2.99 σ_hw). No byte-exact config beats the incumbent.
- **The only material headroom is the ~16% int4-GEMM achieved-BW slack, and it is GREEDY-UNSAFE** (recoverable only by FP-reassociating split-K re-tiling). Roofline *physics* does not cap the search below 481.53; **greedy-safety is the binding constraint** — which is exactly why the prize requires relaxing strict equivalence.

`closed_lever_count = 20` · `all_runs_linked = true` · `strict_headroom_is_greedy_unsafe = true`

---

## Anchors (the frame, not levers)

| anchor | TPS | run | note |
|---|---:|---|---|
| Deployed incumbent (**NON-equivalent**: 3/882 M=8 flips, identity 0.9966) | **481.53** | #52 [`2x9fm2zx`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/2x9fm2zx) | outside the #407 feasible set; PPL 2.3772 |
| **Realized blanket-strict (byte-exact) frontier** | **467.14** | #423 [`5a6zq2yz`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/5a6zq2yz) | the honest equivalence frontier |
| Verify-BW λ=1 wall (identity-free demand ceiling) | 520.95 | #436 [`nvsbctji`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/nvsbctji) | retrain-gated, BW-capped |
| Roofline perfect-`f→1` re-tile ceiling on int4-GEMM (**greedy-UNSAFE**) | 510.87 | #450 [`c5oyb7gv`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/c5oyb7gv) | the headroom that is real-but-unsafe |

σ_hw ≈ 4.8 TPS · gap realized→deployed = **14.39 TPS** (−2.99 σ_hw).

---

## 1. The closed-lever ledger

### 1a. SUPPLY — faster verify (shrink the verify/draft wall)

| lever | mechanism | closing run · PR | closing number | reason closed |
|---|---|---|---:|---|
| **pinned-K self-referential split-K** | pin the verify-GEMM K-split to a canonical layout (modeled +13.998 → 496.74) | [`0pg4bz25`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/0pg4bz25) · #433 | **−5.82 TPS** (modeled +13.998 **inverts**) | **measurement** — buildable on the served Triton kernel, but realizes −5.82; 496.74 refuted as a speed rung |
| **cb3 sub-int4 (RHT+VQ) supply** | realize the sub-int4 byte-saving on the served kernel (modeled +15.60 → 482.74) | [`hv4xpgf8`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/hv4xpgf8) · #437 | **0.0 TPS** (modeled +15.60 forfeits) | **measurement** — byte-saving un-consumable without a bf16 materialize (`penalty_materialize`=0.065); 482.74 collapses to 467.14 base |
| **fused RHT+VQ decode kernel** | human-gated from-source kernel that turns cb3's byte-saving into BW-saving (idealized 521.82) | [`5f3e91as`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/5f3e91as) · #440 | **475.86 < 481.53** (−5.67); `recommend_build=False` | **build-NO-GO** — codebook-gather (γ 0.85)+FWHT (φ 0.025) tax collapses 521.82→475.86; ~12–20 expert-CUDA person-weeks |
| **int4-GEMM Marlin re-tile/config** | find a byte-exact tile/config for the dominant int4-Marlin verify GEMM (85% of verify) | [`fn4iz0dz`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/fn4iz0dz) · #448 | **+0.00 TPS** byte-exact-safe | **physics** — Marlin is the UNIQUE sm_86 int4 GEMM, no Python tile knob; `use_fp32_reduce=False` breaks byte-exactness on 3/4 shapes (UB +0.64<+2) |
| **drafter Triton-kernel tile** | re-tile the drafter's only Triton kernel (fused sparse argmax, 6.52% of D) | [`xryqregh`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/xryqregh) · #449 | **+0.00 TPS** (served default tile-optimal) | **measurement** — served default wins the full 45-config grid sub-µs; rest of D is int4-Marlin (#448's domain) |
| **verify-wall Triton-attention tile** | re-tile the only tunable Triton kernel in verify (Triton-3D attn, 1.27% of verify) — the end-to-end A/B of #442 | [`crrq2e1y`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/crrq2e1y) · #447 | **+0.2613 TPS** (kernel +6.11% confirmed) | **measurement** — `num_stages` 3→2 is a real +6.11% on a 75.3µs kernel → +0.26 e2e; deleting the whole kernel caps at +4.27; #442's +15.86 ⇒ 279.9µs = 3.7× the kernel ⇒ impossible from a retune |
| **sub-int4 weight precision (native)** | serve below int4 to cut the HBM-bound verify read directly | — · #132 | architecturally impossible | **physics** — no native sub-int4 GEMM on sm_86/vLLM-0.22; only route is the fused build (NO-GO above) |
| **verify-attn M-scaling K-opt** | exploit attention M-scaling to pick a faster speculative width K | [`7rb089z3`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/7rb089z3) · #441 | **lift 0.0** (`k_opt`=7==deployed) | **measurement** — `t_attn_frac_of_verify`=0.069, attention is M-FLAT (M8/M4=1.005); the single primary untested TPS-model assumption, measured null |
| **static CUDA-graph capture** | capture the K=7/M=8 spec loop as a CUDA graph to kill host launch overhead | [`qlvakiyu`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/qlvakiyu) · #443 | **+0.00** (self-abort) | **measurement** — already one graph (ONEGRAPH=1); host residual 0.46% < 0.5% gate; no host gap to recover |
| **async-pipelined drafting** | overlap Dₙ with Vₙ₋₁ on a 2nd stream (D/V=0.222, 18.2% theoretical prize) | [`0syyqxag`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/0syyqxag) · #444 | **0.0** byte-exact (provable no-op) | **physics** — drafter consumes the VERIFIED state ⇒ serial Vₙ₋₁→Dₙ→Vₙ; byte-exact ⟺ `wait_event` ⟺ no overlap |
| **async KV-cache L2 prefetch** | prefetch KV into L2 ahead of verify-attention to hide KV-read latency | [`emljqube`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/emljqube) · #445 | **Δ ≈ 0** (self-abort) | **physics** — attn 0.0928 < 10% gate; kernel already cp.async-pipelines; A10G L2 is 6.29MB not 40MB (A100-spec error caught) |

### 1b. DEMAND — more accepted tokens (raise E[T])

| lever | mechanism | closing run · PR | closing number | reason closed |
|---|---|---|---:|---|
| **drafter retrain @ FIXED topology** | retrain MTP K=7 (DeepSeek-MTP-KL α=0.5) to recover the pos-1 near-miss pool, no topology change | [`uid28gdg`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/uid28gdg) · #446 | recoverable **0.0**; optimistic full-delivery **482.50** (+0.97 < σ_hw) | **measurement** — pos-1 miss yields no lift (accept-len 3.83 ≈ 3.844); 65.3% near-miss pool needs a BIGGER drafter (SIZE), forbidden |
| **bigger drafter @ ANY size (net)** | let capacity multiplier s grow; `TPS_net(s)=demand_gain(φ(s))×supply(D(s))` | [`c675zor8`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/c675zor8) · #451 | `net_beats_481=False`; best **467.14** (−14.39, −2.99σ) | **physics** — D(s) charges ~16.9%/unit s; breakeven needs φ′(1)>0.934 vs lit β_α~0.02; critical β=2.78 ≈ 139× lit; even φ=1 fails by s=2 |
| **tree-verify, full-fanout** | verify a full token tree per step | — · #402 | **net −61.61 TPS** | **measurement** — wide-tree verify cost dwarfs marginal accepts at bs=1 |
| **tree-verify, per-position-width DP** | DP-pick a per-position tree width (interior shape) | — · #409 | **+1.33 TPS, β-fragile** | **measurement** — best interior shape nets +1.33 and does not survive acceptance-rate uncertainty |

### 1c. FRESH-LITERATURE directions (all vetted CLOSED this cycle)

| lever | mechanism | closing run · PR | closing number | reason closed |
|---|---|---|---:|---|
| **FlashInfer attention backend** | swap verify-attn to FlashInfer | — · #246 | UNREACHABLE | **build-NO-GO** — vLLM force-pins TRITON_ATTN for Gemma-4's heterogeneous head dims (sliding 256 / global 512) |
| **n-gram / prompt-lookup (PLD)** | augment spec decode with n-gram draft proposals | — · #250/#89/#81 | lane closed (~+1.67% redundant) | **measurement** — PLD hits correlate with MTP accept (redundant); P(hit\|miss)≈0.035 ⇒ no net win |
| **adaptive / dynamic K** | vary speculative width K online | — · #256/#266 | unrealizable → static-K | **measurement** — no online signal beats static K; oracle lift doesn't survive a realizable estimator |
| **static-K change (K≠7)** | pick a different fixed K | [`51bdsbpw`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/51bdsbpw) · #273 | **K4 vs K7 −8.63%** | **measurement** — K=7 is the measured optimum; modeled +13.2%/+4.28% collapse to −8.63% |
| **megakernel (fully-fused step)** | fuse the entire decode step into one persistent kernel | — · (build) | NO-GO build | **build-NO-GO** — from-source CUDA (same family as the fused kernel); residual verify wall is fusion/dispatch, recoverable only by a build |

> Tree-verify (#402/#409) is the same lever cited under the fresh-literature heading in the assignment; it is rowed once, under DEMAND, to avoid double-counting.

---

## 2. The decisive reconciliation — the ~16% int4-GEMM BW headroom is REAL but GREEDY-UNSAFE

This is the single most important clarification in the annex. Two anchors appear to disagree about whether the modeled **+15.86 TPS** (wirbel #442) is reachable. They do not disagree — **they speak to different kernels.**

**denken #447 ([`crrq2e1y`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/crrq2e1y)) — "+15.86 is impossible from a tile retune."**
#447 measures the **only tunable Triton kernel in verify**: the 7-layer Triton-3D split-KV global attention, **75.3 µs = 1.27 % of verify**. Its best byte-exact retune (`num_stages` 3→2, a real +6.11 %) is **+0.2613 TPS** end-to-end; even *deleting the entire kernel* caps at **+4.27 TPS**. #442's +15.86 back-implies a **279.9 µs** verify saving = **3.7× the whole tunable kernel** → impossible from re-tiling it.

**ubel #450 ([`c5oyb7gv`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/c5oyb7gv)) — "+15.86 is physically plausible."**
#450 measures the **aggregate vendored int4-Marlin GEMM**, which is **85 % of verify**. It runs at **433 GB/s = 84 % of measured read-peak (518 GB/s)** — a real **~16 % BW headroom**. A perfect `f→1` re-tile reaches **510.87 TPS** (clears 481.53 by +29.3), and the 467.14→481.53 gap (14.39) is **inside** that headroom.

**Both are correct because they are different kernels:**

| | #447 | #450 |
|---|---|---|
| kernel | tunable **Triton attention** | vendored **int4-Marlin GEMM** |
| fraction of verify | 1.27 % | ~85 % |
| byte-exact retune available? | yes (`num_stages`) → **+0.26** | **no Python tile knob** (#448) |
| recover the BW slack how? | n/a (already tuned) | **FP-reassociating split-K / BLOCK_K / num_warps** |
| greedy-safe? | **yes** | **NO** — reorders int4 accumulation → byte-divergent |

The Marlin GEMM's ~16 % BW headroom is **real**, but #448 ([`fn4iz0dz`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/fn4iz0dz)) proved Marlin exposes **no byte-exact tile knob** — the lone `use_fp32_reduce=False` switch breaks byte-exactness on 3/4 shapes. The only way to cash the slack is FP-reassociating re-tiling, and #450's own realistic split-K estimate recovers just **+12.6 … +31.4 TPS** with **`realistic_splitk_greedy_safe = false`**.

**Decision consequence:** the strict frontier is genuinely closed — every byte-exact lever realizes ≤ noise (max **+0.26**). The only material headroom is the int4-GEMM BW slack, recoverable **only** by greedy-unsafe FP-reassociation — exactly the non-equivalence the prize requires relaxing. **Roofline physics does not cap the sweep below 481.53; greedy-safety does.** That is why the relax-strict-equivalence question (#407) must go to the human. land #451 ([`c675zor8`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/c675zor8)) closes the last escape hatch: even a *demand* push with topology free cannot net-beat 481.53.

---

## 3. Methodological appendix — the modeled-in-isolation collapses (the recurring trap)

Four levers reported a large **modeled isolated-op** gain that **collapsed or inverted** when realized end-to-end. The lesson: **always realize end-to-end; never report the isolated-op Δ.** This is precisely why the relax-prize cards must be **MEASURED**, not modeled.

| lever | modeled (isolated) | realized (end-to-end) | run · PR |
|---|---:|---:|---|
| pinned-K | **+13.998** | **−5.82** | [`0pg4bz25`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/0pg4bz25) · #433 |
| cb3 | **+15.60** | **0.0** | [`hv4xpgf8`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/hv4xpgf8) · #437 |
| autotune (Triton attn) | **+15.86** | **+0.26** | [`crrq2e1y`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/crrq2e1y) · #442/#447 |
| static-K (adaptive→static) | **+13.2 % / +4.28 %** | **−8.63 %** | [`51bdsbpw`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/51bdsbpw) · #256/#266/#273 |

Each shares the same signature: a bandwidth- or op-level surrogate that ignores serial dependency, fixed serving overhead, or FP-reassociation cost. The realized cycle is the only honest arbiter.

---

## 4. Self-test & W&B fields

A 0-GPU self-test (`annex_self_test.py`) loads `closed_lever_ledger.json` and asserts: every non-null `closing_run_id` is a syntactically valid live-linkable id and is rendered in this markdown; `reason_class ∈ {physics, measurement, build_no_go}`; both axes present; the reconciliation booleans (`strict_headroom_is_greedy_unsafe=true`); the anchors are internally consistent (467.14 < 481.53 < 510.87 ≤ 520.95; gap 14.39); and the markdown cites every ledger run id. It logs the W&B run with:

`closed_lever_count` · `all_runs_linked=true` · `strict_headroom_is_greedy_unsafe=true` · `annex_self_test_passes` · `analysis_only=true` · `no_served_file_change=true` · `official_tps=0` · `ppl=2.3772`.

---

## Reproduce

```bash
cd target/ && CUDA_VISIBLE_DEVICES="" .venv/bin/python \
  research/equivalence_escalation/closed_lever_annex/annex_self_test.py \
  --wandb_group equivalence-escalation-anchors \
  --wandb_name denken/closed-lever-evidence-annex
# 0-GPU gate only (no W&B): add --no-wandb
```

## Public evidence used

Spine: denken #447 verify-wall map ([`crrq2e1y`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/crrq2e1y)). Decisive reconciliation pair: ubel #450 roofline ([`c5oyb7gv`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/c5oyb7gv)) + land #451 bigger-drafter net-ceiling ([`c675zor8`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/c675zor8)). Incumbent anchor: PR #52 deployed ([`2x9fm2zx`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/2x9fm2zx), 481.53 / non-equivalent). Realized frontier: denken #423 ([`5a6zq2yz`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/5a6zq2yz), 467.14). All closing runs (#433/#437/#440/#441/#443/#444/#445/#446/#448/#449/#273/#436) are linked inline and verified live.
