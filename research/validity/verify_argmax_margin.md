# Verify-GEMM argmax-margin: greedy-safety gate for SplitK + tree M-width (PR #87)

**Verdict: GREEN.** 0 / 65,536 emitted positions flip their greedy argmax under
either tested perturbation. Both in-flight TPS levers clear the gate:

- **land #71 (tree-verify M=16/32)** — DIRECT, no emulation. The real int4 W4A16
  Marlin lm_head kernel at **M=16 is bit-identical to M=8** (max|Δlogit| = 0),
  and at **M=32** perturbs by ≤ 0.25 logit but flips **0** argmaxes.
- **ubel #84 (SplitK W4A16)** — isolated reduction-order change flips **0**
  argmaxes (split-count-independent across S ∈ {2,4,8}), max|Δlogit| = 0.125
  (one bf16-ULP). GREEN under the FP32-reduce / atomic-add-off regime that the
  12,288-row vocab head forces; residual = 907 enumerated exact-tie positions
  (below) that ubel's real kernel should confirm bit-preserved.

W&B run `875cujdk` (group `verify-gemm-argmax-margin`). Capture dir
`research/validity/verify_argmax_margin/20260614T041541Z/`.

---

## Why this gate

Two TPS levers rest on a "lossless by construction / bit-identical" claim:
- **ubel #84** decomposes the K-reduction across more SMs → partial sums combined
  across splits (a reduction-ORDER change).
- **land #71** widens the verify batch M=8 → M=16/32 → changes the Marlin
  tile / M-template (a kernel-SELECTION change).

FP accumulation is non-associative, so both produce logits that are bit-*close*,
not bit-*identical*, to the deployed Marlin-M8 verify. The official gate is
**greedy-token-identity 128/128 — a single flipped argmax DISQUALIFIES.** The
decisive question: at how many emitted positions is the top-2 logit margin thin
enough that a reduction-order / M-width perturbation flips the argmax? My #73
atomic-add positive control already proved the *mechanism* (a gross
reduction-order swap, `VLLM_MARLIN_USE_ATOMIC_ADD=1`, flips ~36% of tokens); this
gate quantifies the headroom and tests whether a *claimed-lossless* swap stays
inside it.

## Method

Reuse the deployed `fa2sw_precache_kenyan` stack UNCHANGED (no served-file
change). `scripts/validity/verify_argmax_margin.py` loads it in-process
(pck04 12k head-prune + PLE fold + Gemma4 softcap=30, exactly as `serve.py`
applies them), runs the official 128-prompt × 512-token greedy decode, and hooks
`Gemma4ForCausalLM.compute_logits` to capture the real hidden state `h` feeding
the lm_head at every one of the **65,536** emitted positions. From the real `h`:

- **PHASE 1 (margin map).** ref = the REAL int4 W4A16 Marlin kernel at **M=8**
  (the deployed verify width) → softcapped full-vocab logits → top-2 margin
  Δ = logit[argmax] − logit[2nd] at every position. Reported at the bf16 native
  rung (what the sampler sees) and the FP32-accumulator rung (where reduction
  noise lives ~2⁻²³). This is the authoritative safety-headroom map.

- **PHASE 2a (SplitK, ubel #84).** Recover the EXACT dequantized head
  W [12288, 2560] via `apply(lm_head, I)` (one nonzero product per output → no
  accumulation rounding), then recompute logits with K=2560 split into S ∈ {2,4,8}
  FP32-accumulated chunks vs S=1 single accumulation. This isolates **only the
  reduction order** — ubel's perturbation — with the true weights.
  `flip = argmax(S) ≠ argmax(S=1)`.

- **PHASE 2b (M-widen, land #71).** Call the REAL Marlin kernel at M ∈ {16,32} on
  the same `h` grouped wider; compare each row's argmax to the M=8 reference. A
  GEMM's per-row output is row-independent, so M-widening is purely a
  tile/template-selection numeric change — exercised here exactly as land #71
  would on the real kernel.

`scripts/validity/analyze_argmax_margin.py` (CPU, repo venv) turns the compact
`.npz` into the ULP histogram, the flip-proof bound, the GREEN/RED gate, and the
W&B record.

## Results (full official config, 65,536 positions)

### Phase 1 — top-2 margin map
| stat | value |
|---|---|
| median Δ | 4.875 (39 bf16-ULP) — healthy typical headroom |
| min positive Δ | 0.03125 (0.5 bf16-ULP) |
| exact bf16 ties (Δ ≤ 0) | **907** (1.38%) |
| frac Δ < 16 bf16-ULP | 28.4% |
| frac Δ < 16 **fp32**-ULP | **1.38%** (≈ only the bf16-cast ties are thin at the accumulator scale) |
| max \|logit\| (softcapped) | 29.875 (ceiling = 30) |

The fp32-rung column is the key Phase-1 finding: at the FP32 accumulator scale
(where a reduction-order change actually perturbs), **only the 1.38% of positions
that are bf16-cast ties fall within even 16 ULP** — every non-tie position has
enormous fp32 headroom. The thinness is a bf16-*final-cast* artifact, not an
accumulator-scale ambiguity.

The 907 exact ties are **all genuine sub-ceiling bf16 ties** (top1 ∈ [8.875,
29.0], median 25.75) — *not* tanh-saturation artifacts. At magnitude ~25, the
bf16 ULP is 0.125, so two tokens within 0.125 round to the same bf16 value; the
argmax is then decided by the deterministic lowest-index tie-break.

### Phase 2 — kernel-swap sensitivity
| swap | flips | max\|Δlogit\| | notes |
|---|---|---|---|
| SplitK S=2 (vs S=1) | **0** | 0.125 | isolated reduction order |
| SplitK S=4 (vs S=1) | **0** | 0.125 | identical to S=2 |
| SplitK S=8 (vs S=1) | **0** | 0.125 | identical to S=2 |
| SplitK S=2/4/8 (vs real M=8) | 186 | 0.125 | = the emu fidelity gap (see below) |
| M-widen M=16 (real, vs M=8) | **0** | **0** | **bit-identical** |
| M-widen M=32 (real, vs M=8) | **0** | 0.25 | |
| emu S=1 vs real M=8 (fidelity) | 186 (0.28%) | — | FP32-emu vs FP16-MAC path |

**Reading the 186.** The SplitK-vs-real-M8 flip count is *exactly* 186 for every
split count (2, 4, 8), identical to the emu-S1-vs-real-M8 fidelity gap. Since
argmax(emu S) ≡ argmax(emu S=1) at all 65,536 positions (Phase 2a = 0 flips), the
186 is *entirely* the FP32-emulation-vs-FP16-Marlin-MAC path difference and the
reduction-order change adds **zero** incremental flips. ubel's real kernel runs
the same Marlin FP16-MAC path as the deployed M=8 (just split across more SMs),
so it does **not** pay the 186 emulation gap — it pays only the reduction-order
change, which is measured 0-flip.

### Thinnest-margin audit (provable bound + measured residual)
- **98.13%** (64,310 / 65,536) are **provably flip-proof**: margin > 2·max|Δlogit|,
  so the runner-up cannot overtake regardless of perturbation direction — no
  measurement needed.
- **1,226** residual positions rely on direct measurement → **0 measured flips**.
  Buckets (all 0 flips): exact_tie 907, below_1ulp 908, below_2ulp 2402,
  below_4ulp 5298. At the tie positions the perturbation reaches 1 bf16-ULP
  (0.125) yet preserves every argmax.

## Numerics regime (why the deployed kernel is in the safe band)

The deployed int4 W4A16 Marlin lm_head GEMM (vocab n=12,288) already runs a
deterministic split-K reduction: vLLM tiles K=2560 into 128-wide slices,
accumulates partials in FP32 (`global_reduce_fp32`, `C_tmp=at::kFloat`), and casts
FP32→bf16 **once** at the final output write. For n ≥ 2048,
`should_use_atomic_add_reduce()` hard-returns False, so atomic-add is OFF and
`use_fp32_reduce` defaults True. Consequence: the only lossy step is the single
final bf16 cast, so any FP32-reduce-regime reduction-order change (the deployed
20-slice kernel, the S=1 emulation, ubel's SplitK) perturbs the pre-cast FP32
accumulator by ~2⁻²³ relative (negligible) and surfaces only as an occasional
±1 bf16-ULP final-cast difference — exactly the measured 0.125 ceiling. This
bound HOLDS ONLY in that regime; #73's atomic-add control (out-of-regime) flips
~36%. (Source: vLLM `marlin_utils.py` L33-36 / L445-465; `gptq_marlin.cu`
`global_reduce_fp32`; Marlin paper arXiv:2408.11743 §4.)

## Scope and limitations (honest)

- **Audits the lm_head projection** — the GEMM where the greedy argmax is
  computed ("the tensor feeding the argmax", per the PR). Captured `h` is the
  REAL deployed-stack hidden state from the real M=8 forward.
- Both perturbations are **row-independent GEMM-tiling changes**, so testing them
  on the lm_head exercises the *same class* of numeric perturbation the upstream
  attention/MLP GEMMs would incur under SplitK/M-width. The margin map quantifies
  the tolerance.
- **Not directly measured:** upstream per-layer perturbations COMPOUND (the `h`
  fed to the lm_head would itself drift if SplitK/M-width were applied network-
  wide). This is bounded by the per-layer ≤1-ULP regime argument above and is
  ultimately settled by the official 128/128 greedy gate on ubel's/land's real
  submissions — which this gate de-risks from "unbounded worry" to "907
  enumerated positions + a regime bound."
- The SplitK arm is an **emulation** (FP32 reassociation), not ubel's tuned
  kernel; its fidelity to reality is anchored by the 0.28% emu-vs-real argmax
  agreement and the regime bound. M-widen uses the **real** kernel (authoritative).

## Hand-off to the levers

- **land #71:** GREEN with no residual on the verify projection — M=16 is
  bit-identical, M=32 is 0-flip on the real kernel. The 907 ties are bit-preserved
  (M=16 Δ=0 everywhere). Proceed to quota on the M-width numerics.
- **ubel #84:** GREEN on isolated reduction order (0 flips, ≤1-ULP) + the regime
  bound. Before quota, confirm the **907 enumerated exact-tie positions**
  (`margin_perturb.npz`, `ref_argmax` where `ref_top1−ref_top2 ≤ 0`) stay
  bit-preserved under the real SplitK kernel, or rely on the atomic-off /
  fp32-reduce regime (forced for the n=12,288 head) that caps the perturbation at
  ±1 bf16-ULP. The margin map tells ubel exactly how much headroom the SplitK
  output has before it threatens a token: 98.13% provably safe, the rest 0-flip
  under a faithful reduction-order emulation.

## Reproduce

```bash
cd target/
# GPU capture (server venv, single A10G, ~45 min decode + ~3 min reduce):
python scripts/validity/verify_argmax_margin.py
# CPU analysis + W&B (repo venv):
python scripts/validity/analyze_argmax_margin.py \
  --capture-dir research/validity/verify_argmax_margin/<ts>/ \
  --wandb-name "kanna/verify-gemm-argmax-margin-full"
# plumbing smoke (4 prompts x 32 tok):
python scripts/validity/verify_argmax_margin.py --smoke
```
