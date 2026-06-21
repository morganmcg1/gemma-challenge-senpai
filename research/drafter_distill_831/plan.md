# PR #831 — Drafter distillation (raise r≈0.397 via better drafter proposals)

**Lane:** The only remaining lever to raise speculative acceptance r (accept-relaxation
half of the frontier is closed: #816/#823/#820 refuted). `TPS = 43.60 + 61.37·accepted_len`.

## Baseline (validate against PLE-dequant, #817)
- PLE-dequant faithful **268.26 TPS**, CI [260.70, 278.88], PPL 2.0027, GSM8K 0.925,
  **E_accept 3.525±0.712 (r≈0.397)**. W&B cwjfuci4 / zm79qum6.
- Oracle: `TPS = 43.60 + 61.37·accepted_len`; r=1.0 ceiling ~471–507 TPS.

## Load-bearing prior art (this branch, research/drafter_accept_objective/pr95_report.md)
- **#80**: retrained drafter under CE / KL-distill / recipe sweeps → **all MTP parity** (no gain);
  concluded ceiling is architectural (single-layer head capacity).
- **#95 (fern)**: re-ranking channel = **+0.0%** (head argmax already acceptance-ordered);
  corrected greedy LK ceiling = **+1.0–2.4% E[T]** (NOT +8–10%), AMBER, #80 capacity argues for floor.
- Implication: +0.1 accepted-tok/step STOP-gate (≈+2.8% on base 3.525) is **above** the prior ceiling.

## The one genuinely novel angle (why this isn't a strict dup of #80/#95)
- The drafter (`gemma-4-E4B-it-qat-q4_0-unquantized-assistant`, bf16) was trained to match a
  **bf16/QAT** teacher. At serve time the **verifier is int4 W4A16** (pledequant). Any int4-vs-bf16
  argmax mismatch is a train/serve distribution gap the drafter was never trained on.
- **Cheap discriminator (Stage 0):** measure how often the int4-verifier argmax differs from the
  bf16/QAT argmax, and the drafter's top-1 match-rate vs the int4 argmax. If int4≈bf16 (no gap),
  the lane is dead (corroborates pr95). If there's a systematic gap, distilling on int4 argmax may help.

## Drafter architecture (vllm gemma4_mtp.py + assistant config)
- Tiny: hidden_size=256, 4 decoder layers (Q-only attn, KV-shared with target; layer_types
  [sliding×3, full]), centroid head (num_centroids=2048, top_k=32 → 4096/262144 tok/step).
- embed_tokens replaced w/ target 2560-dim embedding at setup; lm_head stays 256-dim (centroid token emb).
- Trainable (per advisor): **dense projections (pre_projection 5120→256, post_projection 256→2560)
  + centroid embeddings (centroids 256→2048)**; FREEZE verifier (body + lm_head) entirely.

## Staged plan (cheap STOP-gate; #828 "fast analyses, decide, move on")
- **Stage 0** (~30 min, fwd-only): broad general distribution (NOT 128 public prompts). Capture
  (drafter inputs, verifier int4 argmax) on a held-out slice. Baseline per-position accept + E_accept.
  Plus the int4-vs-bf16 argmax-gap discriminator above.
- **Stage 1** (~45 min): fine-tune MTP drafter 1 epoch, CE: drafter logits → verifier argmax,
  Adam lr≈1e-4. **STOP-gate:** held-out E_accept must rise ≥ +0.1 acc-tok/step (≈+6 TPS) else report
  negative + STOP (cross-validates #826 hard-ceiling).
- **Stage 2** (only if Stage 1 lifts): scale, serveable checkpoint (verifier bytes unchanged),
  register pledequant_distill, `measure_faithful.py --candidate pledequant --reps 5`
  (need clears_250_robust=1), held-out vs public E_accept (private-stability proof), PPL≤2.42 + 4-axis panel.

## Quality floors (#784)
PPL ≤ 2.42, AIME ≥ 0.090, MMLU-Pro ≥ 0.572, GPQA-Diamond ≥ 0.471, GSM8K ≥ 0.807, 128/128, private Δ ≤ 5%.

## Constraints
- Local/exploratory only on A10G. NO HF launches from this card. Does NOT displace official bi0 218.02.
- Quality-safe by construction: spec-decode is lossless w.r.t. verifier; drafter only changes speed.
