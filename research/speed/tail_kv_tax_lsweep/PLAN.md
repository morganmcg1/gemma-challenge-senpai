# PR #479 — Tail-KV tax L-sweep: measure L>640, tighten the 461.80 center

**stark · group `equivalence-escalation-anchors` · LOCAL A10G (sm_86) · MEASUREMENT + analysis ONLY.
NO HF job, NO submission, NO served-file change, NO `train.py --launch`, NO kernel rebuild.
`analysis_only=true`, `official_tps=0`, `no_served_file_change=true`.**

## The decision-critical question (follow-up to stark #475)

stark #475 (`qkcev1pt`) put the honest strict center at `kv_weighted_strict_tps = 461.80`
(band [461.8, 463.6]). But that number rests on a **linear-tax extrapolation across the fat
tail**: the #472 whole-cycle strict A/B L-sweep STOPPED at **L=640**, while **24.2% of real
decode tokens sit at KV>640 (max KV 2938)**. The tail tax is the only un-measured uncertainty
left in the #474 number. This card extends the measured sweep into the tail to replace the
>640 extrapolation with direct measurement, and to **confirm or refute** the linear-tax
assumption.

## Physical expectation (why this is worth measuring, not assuming)

The strict 2D reduction is a **single-segment sequential-KV** reduction → cost O(KV), linear in L.
The deployed permissive 3D path parallelizes the KV reduction across `num_par=16` segments → its
cost grows **sub-linearly** with KV (more parallelism amortizes longer KV). The measured tax is
`whole_strict − whole_perm`. If perm grows sub-linearly while strict grows linearly, the **delta
could grow slightly super-linearly** in the tail — which would make the true honest center *below*
461.80. The head slope (128→640) was ~0.655 µs/token; the tail slope (640→2048) is the unknown.

## Plan

1. **Extend the #472 harness** (`research/speed/strict_wholecycle_ab/strict_wholecycle_ab.py`)
   to L∈{896, 1280, 2048} on the EXACT GO config (`fa2sw_precache_kenyan` +
   `VLLM_BATCH_INVARIANT=1`: natural M=8 → `use_3d=False` → 2D single-segment order-preserving
   sequential-KV). `--Ls 128,384,640,896,1280,2048 --L 640` (headline stays L=640 so the
   `iso_delta_reproduces_466` calibration guard — anchored at L=640 — still passes).
2. **Per-L strict identity** at each tail L (probe currently runs only at `args.L`): confirm
   identity stays **1.0000 / 0 semantic flips** deep in the tail (the tie-only census must not
   degrade into semantic flips at long KV — that would be a submission-relevant finding).
3. **Tail-tax linearity fit**: head_slope (128→640) vs tail_slope (640→2048) →
   `tail_tax_slope_ratio = tail/head`, `tail_tax_is_linear` (|ratio−1| ≤ 0.15).
4. **L=2048 session repeats** (2–3 fresh processes) so the deepest tail anchor carries its own
   between-session σ, not a single draw.
5. **Re-aggregate** (`research/speed/kv_weighted_strict_tps/kv_weighted_strict_tps.py`): the
   harmonic over the 6-point measured tax replaces the >640 extrapolation →
   `tail_kv_weighted_tps` (PRIMARY) + `updated_band_low/high` (propagating the tail σ).

## Reported (instruction 4)

`tail_kv_weighted_tps` (PRIMARY), `l896_tps`, `l1280_tps`, `l2048_tps`, `l2048_identity`,
`tail_tax_is_linear`, `tail_tax_slope_ratio`, `updated_band_low`, `updated_band_high`,
`ppl=2.3772`, `analysis_only=true`, `official_tps=0`, `no_served_file_change=true`.

## Anchors (banked)

- strict center: `kv_weighted_strict_tps` **461.80** (stark #475 `qkcev1pt`), band [461.8, 463.6],
  `kv_trajectory_mean_L` 527.7, 24.2% of decode tokens at KV>640 / max 2938.
- per-L tax map (head): L=128→**477.39**, L=384→**467.47**, L=640→**457.55** (stark #472 `wfggu51k`).
- σ_hw: within-session 0.3494 / between-session 4.864 (lawine #467 `jb1a0lab`).
- GO config: `fa2sw_precache_kenyan` + `VLLM_BATCH_INVARIANT=1`. PPL gate 2.42 (carry 2.3772).
  M=1 AR floor 161.70. Deployed (non-equiv) 481.53.
