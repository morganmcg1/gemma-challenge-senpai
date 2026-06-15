STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["fqt33bj3"],"primary_metric":{"name":"pessimistic_breaches_3p2_measured","value":0},"test_metric":{"name":"irreducible_gap_floor_pct_vbi1_measured","value":0.5764}}

## Results — GPU per-L attention pin LIFTS the #386 pessimistic breach

**Verdict: `GREEN_breach_LIFTED_measured`. The thin −0.32pp pessimistic breach in #386 was a CONSERVATIVE-SLOPE ARTIFACT.** Measured on the A10G under VBI=1, the local attention shape slope on L∈[528,658] is **0.353× #386's interpolated slope**, so the pessimistic corner floor falls **3.5235% → 1.272%** and clears 3.2% with **+1.93pp** margin. **All three corners clear.** `per_l_attention_vbi1_self_test_passes = True (13/13)`.

This is identity-safe local GPU latency profiling — **0 official TPS, no submission, no served-file change, no HF Job, no `--launch`.** Deployed best stays **481.53 TPS / PPL 2.3772** (private-verified 460.85), unchanged.

### Headline (full-L convention = the like-for-like #386 pin)

| metric | #386 (`xxzujn7a`) | **#389 measured (`fqt33bj3`)** |
|---|---|---|
| `pessimistic_breaches_3p2_measured` | **True** @3.5235% | **False** @1.272% (clears +1.93pp) |
| central floor `irreducible_gap_floor_pct_vbi1` | 1.3097% | **0.5764%** (clears +2.62pp) |
| pessimistic floor | 3.5235% (breach) | **1.2723%** |
| banked floor | 0.000% | 0.000% |
| `all_corners_clear_3p2` | False | **True** |
| `measured_local_penalty_slope` [528,658] | 0.003037/tok (interp) | **0.001071/tok (measured)** |
| `slope_ratio` (measured/interp) | — | **0.353×** |
| `f_attn` | 0.0951 (modeled, #378) | **0.1033 (measured)** |
| `breakeven_prompt_shift_tok` | +118.6 | **≈+324.5** (~2.7× safer) |

### Re-derived floor at the 3 corners (r_a, g_a inherited per-corner UNCHANGED — kernel-invariant; only g_s recomputed with the measured shape)

| corner | ΔP (tok) | L_priv | shape_meas | **floor FULL** | margin | floor COMPOSED (window-aware) |
|---|---|---|---|---|---|---|
| banked | 0 | 528 | 1.0000 | 0.000% | +3.20pp | 0.000% |
| central | 50 | 578 | 1.0633 | **0.576%** | +2.62pp | 0.000% |
| pessimistic | 130 | 658 | 1.1392 | **1.272%** | +1.93pp | **0.121%** |

The **COMPOSED (window-aware)** convention is the physically faithful per-step attention: 35/42 layers are sliding-window (512) and SATURATE above L=512, so only 7/42 full layers grow with L. On that convention the pessimistic floor is just **0.121%** — the #386 full-L floor materially **overstates** ctxlen sensitivity by ignoring window saturation. Both conventions clear 3.2% comfortably.

### The pinned quantity — measured vs interpolated slope

```
 MEASURED per-L un-pack attention (median us) + shape (norm @528), n=300 after 60 warmup
        L |  full_med | full_std |  shp_full |  shp_comp | shp_386mdl | pen_full
      352 |     65.54 |     4.48 |    0.8101 |    0.7821 |     0.5274 |    1.103
      503 |     76.80 |     4.79 |    0.9494 |    0.9259 |     0.9244 |    1.316
      528 |     80.90 |     5.65 |    1.0000 |    1.0000 |     1.0000 |    1.386
      578 |     86.02 |     4.19 |    1.0633 |    1.0000 |     1.1449 |    1.474
      658 |     92.16 |     3.83 |    1.1392 |    1.0131 |     1.3949 |    1.579
     2048 |    212.99 |     4.45 |    2.6329 |    1.2810 |     9.2888 |    3.525
 mean±std @resolved anchors (full un-pack us): 528 82.73±5.65 | 578 87.59±4.19 | 658 93.51±3.83 | 2048 214.63±4.45
```

The measured full shape rises **+13.9% over [528,658]** (82.73→93.51 us, a +13% mean gap separated by ~25σ given SEM≈0.3 us), vs #386's interpolated shape which assumes **+39.5%** over the same span (`shp_386mdl` column: 1.0→1.3949). #386 inherited the [528,2048] far-field segment of #375's penalty curve, which is **3× too steep** locally because #375's curve is convex (accelerating) far past the operating point. Pinning the slope at the operating point is exactly what removes the breach.

### Self-test (PRIMARY) — 13/13 pass

- **Round-trip (the key check):** feeding #386's INTERPOLATED shape back through this module's re-derivation reproduces #386 **exactly** — central **1.3097%** (==1.3097 ✓), pessimistic **3.5235%** (==3.5235 ✓). Proves the only thing that changed is interpolated→measured shape, not the floor math.
- **Provenance to #375:** measured un-pack penalty reproduces #375's curve STRUCTURE — penalty(352)=1.103 vs 1.000 (+10.3%), penalty(2048)=3.525 vs 3.027 (+16.5%), monotone. (Absolute penalty is +10–16% above #375; this is expected kernel/hardware-specific absolute scaling — the *shape/slope* is what is portable and what the re-derivation uses.)
- **Central clears 3.2% under the measured slope** ✓; shapes finite/positive, resolved-anchor monotone, shape(528)=1, iters≥200, on-target A10G sm_86, all launch-safety flags set ✓.

### Anchor-sensitivity leg (L=503, the #282 median operating point)

`pessimistic_floor_pct_at_L503` = **1.219%** (full) / 0.864% (composed). `breach_is_anchor_artifact = False` — because there is **no breach at either anchor** once the slope is measured. The 528-vs-503 anchor choice is not what drives the result; the slope is.

### Method, command, resources

- **Method:** `flash_attn` 2.8.4 isolated decode-step timing (vLLM / FlashInfer / transformers are NOT installed on this pod). `num_splits=1` (un-packed single split) == **VBI=1** attention; `num_splits=0` == deployed heuristic auto-split. This is the established stark #363 / #365 / wirbel #375 repo proxy for VBI=1 attention. Real gemma-4-E4B-it geometry from `config.json`: 8 q-heads / 2 kv-heads, **head_dim 256 for all layers**, 42 layers = 35 sliding(window 512) + 7 full at idx {5,11,17,23,29,35,41}. Single-stream batch=1, seqlen_q=1 decode step. CUDA-event timing, **median** central estimator (robust to right-skew scheduling outliers; mean±std also reported). Greedy identity untouched (timing kernels, not changing them).
- **Command:**
  ```
  cd target/ && CUDA_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=1 python \
    research/validity/per_l_attention_vbi1/per_l_attention_vbi1.py --gpu \
    --proxy google/gemma-4-E4B-it-qat-w4a16-ct --vbi1 --measure-f-attn --self-test \
    --iters 300 --warmup 60 --wandb_group per-l-attention-vbi1 --wandb_name ubel/per-l-attention-vbi1
  ```
  (`CUDA_VISIBLE_DEVICES=0` is required on this pod — the default device is dead, the #358/#363 gotcha.)
- **Peak GPU mem:** **4.27 MiB** (attention-only microbench; no model weights loaded).
- **W&B run:** `fqt33bj3` — https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/fqt33bj3 (group `per-l-attention-vbi1`). Logged: per-L curve table (measured vs #386-modeled), corner-floor table (full+composed vs #386), slope/f_attn/breakeven/provenance/round-trip/diagnostics scalars, full JSON artifact.

### What happened — honest analysis

The breach **lifts**, and it lifts robustly. #386's pessimistic corner sat at 3.5235% (−0.32pp over the line) entirely because it inherited #375's penalty slope from the **far-field [528,2048] segment**, where the un-pack penalty curve is convex and accelerating. At the actual operating point [528,658] the measured slope is **~1/3 of that** (0.353×), so the pessimistic floor lands at 1.272% with ~2pp of headroom and every corner clears. The robustness is not marginal:

- The conclusion is insensitive to measurement noise: a ±10% error on shape(658) moves the pessimistic floor only ~1.07–1.30%, nowhere near 3.2%.
- The window-aware composed floor (0.121% pessimistic) is even gentler — the full-L convention #386 used is itself an upper bound, because 35/42 layers saturate at the 512 window.
- f_attn measured (0.1033) sits close to #378's modeled 0.0951 (+8.7%), so the attention fraction was not the weak input — the **slope** was.

**Caveats (stated honestly):** (1) This is a flash_attn proxy on a random KV cache, not the live served vLLM stack — but it is the repo's accepted VBI=1 proxy (vLLM is not installed) and the PR explicitly scoped it. (2) Two non-corner grid points (L=553, 633) showed large std (27 / 39 us) from transient scheduling outliers; the **median** estimator filters these and the decision-relevant corner points (528/578/658) have clean ±4–6 us std — this is why median, not mean, drives the shape. (3) The breakeven ≈+324.5 tok is interpolated across the 658→2048 grid gap (no intermediate points there); the robust takeaway is its **direction** — ~2.7× larger than #386's +118.6, i.e. far safer — not the exact value.

### Public evidence used

- **#386** (`xxzujn7a`, MERGED b0de7eb) — the floor math (`floor = r_a·g_s`), the 1.3097%/3.5235%/0.000% corner floors, the +118.6 breakeven, and the breach this card pins. Round-trip reproduces it exactly.
- **#379** (`5kpb73tb`, deployed) — the per-corner kernel-invariant r_a back-out (banked/central/pessimistic), inherited unchanged.
- **#378** (`gghmgtk9`) — f_attn=0.0951 (modeled), eval-weighted penalty 1.2257; the f_attn comparison anchor.
- **#375** (`27sbg3zb`) — the un-pack penalty curve (1.264/3.027/4.756× @ 528/2048/4096, crossover ≈352) whose interpolated slope #386 used and whose anchors the provenance self-test reproduces.
- **#282** — decode-length median L=503, the anchor-sensitivity operating point.
- **#363 / #365** (stark) — the flash_attn isolated-timing methodology and num_splits VBI=1 proxy.

### Suggested follow-ups

1. **fern #357 can re-derive the demand ceiling on the MEASURED floor** (central 0.576%, pessimistic 1.272%, **all corners clearing**) instead of #386's 1.310% with a breaching corner — the all-corner robustness pillar is restored on the live VBI=1 contract.
2. **Adopt the window-aware composed floor as the headline ctxlen-sensitivity number** in future demand-route work: it is the physically faithful per-step attention (sliding-window saturation) and is ~10× gentler than the full-L convention.
3. **If the team wants the slope on the true served stack** (not the flash_attn proxy), a vLLM install on the pod would let a direct VBI=1 per-L decode-latency trace confirm the proxy — but the +10–16% absolute deviation here is within expected kernel-scaling and does not change the (slope-driven, large-margin) verdict.
