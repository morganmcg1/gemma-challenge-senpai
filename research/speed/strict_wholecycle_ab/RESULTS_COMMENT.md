STUDENT stark:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["wfggu51k"],"primary_metric":{"name":"realized_strict_frontier_best_estimate_tps","value":457.55},"test_metric":{"name":"ppl","value":2.3772}}

## Results — whole-cycle strict A/B, overlap-captured (#472)

**Bottom line: the honest, overlap-captured realized strict frontier is `457.55 TPS` (headline
L=640) — essentially #466's lower bound, NOT the composed 467.14. In-graph overlap hides only
+4.7% of the strict tax at the deployed-faithful KV, far below the ≥28% needed to reach 467.**
`whole_cycle_holds_within_sigma_hw=False` (drift +9.59 > σ_hw 4.8153). Strict 2D byte-exact
(identity 1.0000, 0 flips). 21/21 self-test checks pass. W&B `wfggu51k`.

### Headline (L=640, deployed-faithful longest KV — #466's headline convention)

| metric | value |
|---|---|
| `realized_strict_frontier_best_estimate_tps` | **457.55** (σ_lo 457.53 / σ_hi 457.57) |
| `whole_cycle_strict_tps` | 457.55 |
| `whole_cycle_perm_tps` (deployed anchor) | 481.53 |
| `whole_cycle_strict_delta_us` | **+401.90 µs/cycle** (σ 0.36, N=21) |
| in-harness isolated Δ (the #466 locus, co-measured) | +421.77 µs/cycle (σ 0.32) |
| `overlap_recovery_fraction` | **+4.7%** (vs ≥28% needed to reach 467.14) |
| `whole_cycle_holds_within_sigma_hw` | **False** |
| `composed_vs_wholecycle_drift` | +9.59 (> σ_hw 4.8153) |
| `realized_eta_attn_decode` | 5.24% (composed 3.08%) |
| `whole_cycle_strict_identity_fraction` / `token_flips` | **1.0000 / 0** |
| permissive identity (reproduces non-equivalence) | 0.0000 byte / 1.0 argmax |
| PPL anchor | 2.3772 (≤ 2.42 gate) |
| peak VRAM | 2.51 GiB |

### Full KV sweep (the overlap recovery shrinks as KV grows)

| L | whole_perm µs | whole_strict µs | whole Δ µs (σ) | iso Δ µs (σ) | overlap recover | **whole_strict_tps** | iso_strict_tps |
|---|---|---|---|---|---|---|---|
| 128 | 4685.76 | 4752.26 | +66.54 (0.44) | +78.68 (0.25) | **+15.4%** | 477.39 | 476.64 |
| 384 | 4843.21 | 5073.82 | +230.63 (0.53) | +252.91 (0.28) | **+8.8%** | 467.47 | 466.15 |
| **640** | 4935.31 | 5337.09 | **+401.90 (0.36)** | +421.77 (0.32) | **+4.7%** | **457.55** | 456.42 |
| | | | | | cluster-mean | 467.47 | 466.40 |

### What happened (honest analysis)

**The strict frontier does NOT reach the composed 467.14 once overlap is accounted for — it stays
at #466's ~457.5 lower bound.** At the deployed-faithful headline L=640 the whole-cycle strict tax
(+401.9 µs/cycle) sits only ~20 µs below the isolated #466 locus Δ (+421.8 µs), i.e. in-graph
overlap recovers just **+4.7%** of the tax — well under the **≥28%** that would have been needed to
lift the realized number to 467.14. So `realized_strict_frontier_best_estimate_tps=457.55`, +1.2
above the #466 isolated lower bound (456.42 in-harness) and **+9.59 below the composition** — and
9.59 > σ_hw (4.8153), so the optimistic-composition verdict **does not hold**.

**Why so little overlap — and why this confirms #466 was tight, not loose.** The deployed decode
cycle is a single-stream ONEGRAPH CUDA graph; CUDA stream-ordering semantics mean kernels in one
stream do not overlap. The strict 2D reduction is a genuine *serialization* tax (single-segment
sequential-KV), and a single-stream graph has no second stream to hide it under. The ~20 µs the
whole-cycle Δ comes in below the isolated Δ is marginal scheduling/L2-residency, not real compute
overlap. **#466's isolated lower bound was therefore tight, not loose** — exactly the conservative
read it claimed.

**The L-dependence is worth flagging.** Overlap recovery shrinks monotonically with KV length
(15.4% → 8.8% → 4.7% as L goes 128 → 384 → 640): the strict serial reduction tax scales with KV,
while the fixed GEMM body it could hide under does not, so relative headroom vanishes as KV grows.
The cluster-mean (467.47 whole / 466.40 iso) is buoyed entirely by the short-L=128 point — which is
**not** the deployed-faithful decode KV. I therefore quote the **L=640 headline (457.55)** as the
best estimate, matching #466's headline convention, and report the cluster transparently above so
you can choose a KV-distribution-weighted number if you prefer for the board.

**Calibration guards passed (`whole_cycle_perm_tps=481.53` is definitional, so these are the real
checks):** in-harness isolated Δ +421.77 µs reproduces #466's banked +422.9 µs within 12% (True);
body GEMM 4125.6 µs reproduces #450's 4152.96 µs within 20% (True). The M=8 strict-reduction verify
attention **CAPTURES + REPLAYS** at every L (no M=1 collapse → `strict_frontier_collapses_to_m1=False`).

**Reconcile vs #466 and #452.**
- **vs #466 (isolated lower bound):** my in-harness isolated arm reproduces #466's per-L almost
  exactly (476.64 / 466.15 / 456.42 vs #466's 476.63 / 466.09 / 456.36) — strong cross-validation.
  The whole-cycle (457.55) sits just +1.2 above it, confirming the lower bound was tight.
- **vs #452 (relax side held: composed 498.6, realized −0.94):** the strict-side whole-cycle does
  **not** hold (drift +9.59 > σ_hw). The asymmetry is physical: the relax lever (fp32 split-K
  reduce) is nearly free under overlap, whereas the strict lever (serial single-segment reduction)
  is a real serialization tax that a single-stream graph cannot hide. Same #452 *method*, opposite
  *verdict* — and the method is exactly what surfaces the difference.

**For the approval issue:** quote the board's predicted strict TPS at **~457.5 (headline L=640)**,
not the composed 467.14. The whole-cycle A/B tightens the #466 interval [456.5, ≤467.14] down to a
point estimate of **457.55** with the composition refuted (overlap recovers < 5% at deployed KV).

### Command

```bash
cd target/ && CUDA_VISIBLE_DEVICES=0 /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
  research/speed/strict_wholecycle_ab/strict_wholecycle_ab.py --self-test \
  --wandb_group equivalence-escalation-anchors --wandb_name stark/strict-wholecycle-ab
# W&B logged from the repo .venv (GPU tool-venv has no usable wandb):
cd target/ && .venv/bin/python research/speed/strict_wholecycle_ab/wandb_log.py \
  --json research/speed/strict_wholecycle_ab/strict_wholecycle_ab.json \
  --wandb_group equivalence-escalation-anchors --wandb_name stark/strict-wholecycle-ab
```

- **Constraints honored:** `analysis_only=true`, `no_served_file_change=true`, `no_kernel_rebuild=true`,
  `official_tps=0`, no HF job, no submission. Strict lever is config-reachable (== `VLLM_BATCH_INVARIANT=1`).
- **Public evidence used:** none beyond banked repo anchors (#466 `sxigz7dp`/`gmd8v9sw`, #452, #450,
  #455 `0r0ounl8`, #423 `5a6zq2yz`, PR #52 `2x9fm2zx`) and the served `osoi5-v0-baked/config.json`
  geometry. No external code or weights pulled.

### Suggested follow-ups

1. **Multi-stream overlap probe (the only path to >457.5).** The single-stream graph forecloses
   overlap by construction. If a future serve change split the M=8 verify attention onto a second
   CUDA stream concurrent with the body GEMMs, *some* of the +401.9 µs could hide — but that is a
   served-kernel/scheduler change (gated, out of scope here). Worth a measurement-only feasibility
   probe before anyone assumes 467 is reachable.
2. **KV-distribution-weighted board estimate.** If the leaderboard workload's KV mix is shorter than
   L=640 on average, the effective strict TPS rises toward the cluster-mean (467.47). A weighting
   over the real 128–640 prompt-length distribution would give a single defensible board number.
3. **Coordinate the point estimate with denken #471's 128-prompt served census** before the approval
   issue quotes a board TPS — this A/B fixes the *speed* point; #471 fixes the *equivalence* census.
