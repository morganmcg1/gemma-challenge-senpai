# PR #302 — Is the #287 read-cut a free body-side companion to the EAGLE-3 build?

**Verdict: NO — doubly-closed.** The #287 read-cut FITS the build's PPL headroom but
REGRESSES on the wall (realization ratio ≤ 0). Bank it as a CLOSED companion; build alone.

- **PRIMARY** `read_cut_build_companion_self_test_passes` = **True** (10/10 checks)
- **TEST** `read_cut_is_free_build_companion` = **False**
- `read_cut_fits_ppl_headroom` = **True** (cost 0.0203 ≤ headroom 0.0428)
- `read_cut_realization_ratio` = **−2.018** → classification **`regresses`**
- `read_cut_wall_realized_tps_credit` = **−33.43 TPS** (build-floor anchor; −32.2 deployed / −34.8 λ-ceiling)
- W&B run `8jewx2ur` · 0 TPS · analytic over banked ratios · BASELINE stays 481.53

## Question

#287 (`17en3hus`) closed the read-side **STANDALONE** door: the max PPL-safe body-read
cut is 8.431% (`mixed_int3_demote16L`, proj PPL 2.3975), and on denken #283's MEASURED
38% read-fraction it does **not** reach 500 alone (`read_reduction_lever_clears_500 = False`).
The sole >500 path is the human-gated EAGLE-3 build, which raises E[T] via a better
greedy-token-identical drafter and leaves the verify-path **body unchanged**. "Does not
clear 500 standalone" ≠ "is a free COMPANION to the build." Is the read-cut a free
body-side companion that stacks TPS ON TOP of the build (read analog of lawine's SAM
companion #296/#300), or does it under-realize/REGRESS the way stark #273 (`51bdsbpw`)
found every body-side deviation from the K=7-optimized ONEGRAPH graph regresses
(K4-vs-K7 = −8.629%, realization ratio K4 = −2.018 / K5 = −0.864, NEGATIVE)?

## Method (analytic; NOT a fresh wall A/B — that is stark's #298 lane)

1. **PPL-headroom fit.** Greedy-token-identical drafter ⇒ drafter precision moves E[T],
   not PPL; body unchanged ⇒ build PPL = deployed **2.3772**, headroom to gate 2.42 =
   **0.0428**. Read-cut cost = 2.3975 − 2.3772 = **0.0203 ≤ 0.0428** → **FITS** (it spends
   the body-only headroom, never touching the binding acceptance clause).
2. **Compose the read-cut TPS credit** = 8.43% byte cut × denken #283's 38% read-fraction
   = **3.208%** wall-fraction (**+3.314%** TPS-uplift). Deployed cross-check reproduces the
   #287 bank exactly (481.53 → **497.488 TPS**). On the build the same absolute body read is
   a *smaller* wall-fraction (~31.6%, the step grows with E[T]=4.9029) → the idealized credit
   is already optimistic on the build (diluted ≈ 2.67%).
3. **Realization-discount** by stark #273's banked body-side ratios. The read-cut
   re-quantizes verify-path weights (int4→int3 on 16 layers) → a body-side ONEGRAPH
   deviation in stark #273's class. `realized = composed × ratio`. Every banked body-side
   ratio is **negative** (K3 −6.18, K4 −2.018, K5 −0.864, K6 −0.610; mean −2.42).

## Result

| quantity | value |
|---|---|
| build PPL (body unchanged) | 2.3772 |
| PPL headroom to gate 2.42 | **0.0428** |
| read-cut PPL cost | **0.0203** (fits; residual 0.0225) |
| composed credit (8.43%×38%) | +3.208% wall / **+3.314% TPS-uplift** |
| deployed cross-check vs #287 | 497.488 TPS ✓ (exact) |
| build read-fraction (diluted) | ~31.6% (vs 38% deployed) |
| `read_cut_realization_ratio` | **−2.018** (K4 precedent; range [−6.18, −0.61], all <0) |
| realized uplift | **−6.69%** (ratio × composed) |
| `read_cut_wall_realized_tps_credit` | **−33.43 TPS** (build-floor; −32.2/−34.8 bracket) |
| classification | **regresses** |

The composition *credits* +3.31%, but transferred onto the wall via stark #273's body-side
prior the read-cut **regresses ~6.7%** — exactly the K4 pattern (composition predicted
+4.28% → clears 500, wall REFUTED to −8.63%). The credit's sign is robust to anchor and to
ratio choice (every banked ratio is negative).

## Portfolio implication

**BUILD ALONE.** Bank the #287 read-cut as a **CLOSED companion** (not an integration
target): it fits the build's 0.0428 PPL headroom but regresses on the deployed K=7 graph
the same way static-K did. The read-cut is **doubly-closed** — not standalone (#287), not a
companion (this leg).

## Honest scope / caveats

- **0 TPS.** No >500 build, no served-file change, served checkpoint stays `fa2sw_precache_kenyan`.
- **Analytic, not a wall A/B.** The realization discount is stark #273's banked PRIOR
  applied over banked numbers; a fresh wall A/B of the read-cut is stark's #298 harness.
- **assumes_body_unchanged = True.** If the build co-quantizes the body for VRAM (ubel
  #299's lane), the 0.0428 headroom is shared with the body-quant.
- **Read-fraction dilutes on the build** (38% deployed → ~31.6%), so even the idealized
  companion credit is optimistic before the discount.
- **Launch gate stays** land #245 MEASURED ≥500 at λ̂≥0.9780 AND PPL≤2.42 AND VRAM≤24 GB,
  human-approval-gated.

## Cross-references

#287 (read-frontier priced as a companion), stark #273 (realization-ratio method +
body-side regression precedent), **stark #298** (ORTHOGONAL — wall-realization of the FREE
CEILING lever; this leg prices the READ-CUT analytically over banked ratios), denken #283
(38% read-fraction), ubel #299 (VRAM lane the body-quant headroom would share),
wirbel #290/#293 (the build target the companion would stack on).

## Reproduce

```bash
cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  research/validity/read_cut_build_companion/read_cut_build_companion.py \
  --self-test --wandb_group read-cut-build-companion --wandb_name fern/read-cut-build-companion
```
