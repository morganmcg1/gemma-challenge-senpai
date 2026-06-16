STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["prb1h0ys"],"primary_metric":{"name":"closed_lever_count","value":21},"test_metric":{"name":"ppl","value":2.3772}}

## Results — the greedy-safety boundary (byte-exact ⟺ reduction-order-preserving) + annex v2 (lever #21)

**Verdict:** the #456 thesis is now *principled*, not just enumerated. The auditable WHY a skeptical reviewer demands:

> **A change to the served verify path is byte-exact (greedy-identical) IFF it preserves the order of every floating-point reduction. Every material BW-saving lever recovers its bandwidth by *reassociating* a reduction → greedy-unsafe.**

Analysis-only: **0 TPS, no served-file change, no submission, no HF job.** PPL anchor **2.3772 ≤ 2.42**. `greedy_safety_boundary_self_test_passes = 45/45`. W&B [`prb1h0ys`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/prb1h0ys) (group `equivalence-escalation-anchors`).

### 1. The greedy-safety boundary table — a total dichotomy (no third class)

| kernel | % verify | PRESERVING (byte-exact) | REASSOCIATING (greedy-unsafe) |
|---|---|---|---|
| **int4-Marlin GEMM** | ~85% | tile/config — no Python knob → deployed order, **+0.00** (#448 [`fn4iz0dz`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/fn4iz0dz)) | split-K / BLOCK_K / num_warps → **the ~16% BW slack** (#450 [`c5oyb7gv`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/c5oyb7gv)); `fp32_reduce=False` breaks 3/4 shapes (#448) |
| **Triton 3D split-KV attn** (head-256 sliding ×35 + head-512 global ×7) | ~14.19% | `num_stages` (pipeline depth, partition-invariant) → **+0.26** (#447 [`crrq2e1y`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/crrq2e1y), ref #428) | **BLOCK_M/BLOCK_Q tile** (re-partitions the split-KV reduction) → **greedy-UNSAFE, lever #21** (#442) |
| **lm_head vocab GEMM** | ~0.64% | full-hidden read, pinned — candidate-restrict 2.1× slower + correctness-impossible (#144) | (split-K immaterial: 0.64% slice) |
| **RMSNorm** | <0.1% | deployed Σx² order, pinned — memory-trivial at bs=1 | (tree-reduce immaterial) |
| drafter fused-sparse-argmax | (draft) | tile sweep → **+0.00** (#449 [`xryqregh`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/xryqregh)) | — |

Every PRESERVING lever caps at **≤ +0.26 TPS**; every material BW headroom is on the REASSOCIATING side. `byte_exact_levers_all_capped_le_0p26=true`, `every_material_headroom_is_reassociating=true`.

### 2. The decisive new lever (#21) — #442 Triton-attn tile retune is greedy-UNSAFE

wirbel #442 ran a served bm4 (`{block_m:4, tile:32, warps:4, stages:2}`) A/B + greedy census + a default-vs-default floor:

| measurement | result | run |
|---|---|---|
| served greedy census (bm4 vs default, 64×512) | **34/64 = 53.1% identical** (`byte_exact=0`, 30 diverge, first flip @ tok 290) | [`grrc3zms`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/grrc3zms) |
| **floor control** (default vs default, fresh procs) | **64/64 = 100% identical** → the 53.1% is a **REAL** break, not eager-FP noise (#438 confound dead) | [`cy0ijlit`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/cy0ijlit) |
| served wall A/B (2 seeds × 3 reps) | **−5.65 TPS** (475.88 < 481.53, `classification=evaporates`) | [`gyw2ksvs`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/gyw2ksvs) |
| isolated autotune | `deployable_cfg_maxdiff=0` (the **a-priori byte-exact** that the served census refuted) | [`e5n9a2dc`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/e5n9a2dc) |

**#450's split-K FP-reassociation hazard is NOT confined to the Marlin GEMM** — changing the BLOCK_M/BLOCK_Q partition reassociates the 3D split-KV online-softmax merge (`reduce_segments`) → flipped argmax. The clean attribution holds because the knobs are isolable: `num_stages` alone is byte-exact (#447 +0.26), so the break is owned by the **partition** change. This is the **4th kernel-tiling strict-NULL** (#447/#448/#449/#442) and the **5th** isolated-op collapse (+15.86 isolated → −5.65 served).

### 3. Completeness — 4 reduction families exhaust the verify path

`reduction_count_enumerated = 4`. R1 GEMM K-reduction (85%, ~16% BW slack, reassoc → #450/#448) · R2 attention split-KV merge (14.19%, byte-exact `num_stages` +0.26 / material BW only via partition reassoc → #442) · R3 lm_head vocab GEMM (0.64%, pinned, no BW → #144) · R4 RMSNorm Σx² (<0.1%, pinned, no BW). **Every** reduction is either (a) pinned = byte-exact with no material BW, or (b) reassociable-for-BW-but-greedy-unsafe. The answer to *"are you SURE there's no byte-exact lever you missed?"* is **no** — here is every reduction, and the only BW-bearing ones reassociate.

### 4. The trap — PPL-pass ≠ greedy-identical

#442 **passed PPL** (2.3767 ≤ 2.42, [`grrc3zms`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/grrc3zms) `ppl_bm4`) while its greedy census was **53.1% identical** (`byte_exact_and_ppl_pass=False`). PPL is teacher-forced (robust to sub-ULP perturbation); greedy identity is free-running (a single argmax flip cascades). A PPL-only gate would have waved bm4 through. **Lesson: certify identity by a served same-path census, never by PPL or an a-priori assertion** — #442's assertion was wrong; only the census caught it.

### 5. FLAG-2 — parameterized slot (NOT blocking), robust

#442 FLAG-2: head-256 **sliding** also routes through Triton 3D split-KV in verify → the tunable surface is **larger** than #447's head-512-only **1.27%**. wirbel is reseated on `triton_verify_attn_frac_of_verify` (group `equivalence-escalation-anchors`) — **not yet landed** as of 2026-06-16, so I carry **1.27% (lower bound, #447)** as a parameterized slot, upper bound **14.19%** (full attention). **Robust either way:** even at 14.19%, (a) the byte-exact knob (`num_stages`) is partition-invariant → ceiling stays +0.26 (and #442's wall A/B on the full head-256+512 surface realized −5.65, occupancy-saturated), and (b) the material BW is recoverable only by a partition change → greedy-unsafe (#442 0.531). The correction only enlarges the greedy-unsafe column; `closed_lever_count` stays **21**, conclusion intact.

### Consequence for #407
**Roofline physics does NOT close the frontier below 481.53; greedy-safety does.** The strict byte-exact frontier is closed at realized **467.14** (best byte-exact lever +0.26). The only way past it reassociates a reduction → the non-equivalence the prize requires relaxing.

---

### Deliverables (all under `research/equivalence_escalation/greedy_safety_boundary/`)
- `greedy_safety_boundary.md` — boundary table + 4-family reduction enumeration + principle + PPL-trap + FLAG-2 slot.
- `closed_lever_ledger_v2.json` — annex v2: #456 base 20 + lever #21 + boundary/reduction sections + FLAG-2 slot (machine-readable).
- `greedy_safety_boundary_self_test.py` — 0-GPU self-test (45/45) + W&B logger; loads the #456 base ledger and derives 20+1=21 (proves continuity).

### Command
```bash
cd target/ && CUDA_VISIBLE_DEVICES="" .venv/bin/python \
  research/equivalence_escalation/greedy_safety_boundary/greedy_safety_boundary_self_test.py \
  --wandb_group equivalence-escalation-anchors --wandb_name denken/greedy-safety-boundary
```

### summary.json-equivalent fields
- **official_tps:** 0 (analysis-only, no served-file change, no HF job). **ppl:** 2.3772 (anchor, gate ≤ 2.42).
- **closed_lever_count:** 21 · **reduction_count_enumerated:** 4 · `byte_exact_levers_all_capped_le_0p26`=true · `every_material_headroom_is_reassociating`=true · `greedy_safety_boundary_self_test_passes`=45/45.
- **Peak memory:** 0 GPU (CPU self-test). **W&B:** [`prb1h0ys`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/prb1h0ys).

### What happened
The boundary is a clean dichotomy with no third class, and the new #442 evidence slots in exactly where the principle predicts: the reassociating sibling of #447's byte-exact `num_stages` is the greedy-unsafe BLOCK_M/BLOCK_Q retile. Every material BW headroom in the verify path is reassociating → greedy-unsafe.

### Honest caveats / suggested follow-ups
1. **num_stages-only byte-exactness** rests on #447's realized +0.26 (treated byte-exact) + #428's bit-identical supply ceiling + the partition-invariance physics, **not** a fresh *isolated* served census (#442's census bundled num_stages with BLOCK_M/BLOCK_Q). A cheap num_stages-only served census would close it definitively — but the conclusion does not depend on it (worst case moves more headroom into the greedy-unsafe column).
2. **FLAG-2 number:** when wirbel's `triton_verify_attn_frac_of_verify` lands, replace the parameterized 1.27% slot with the measured fraction (revise the #447 spine). Robustness pre-proven; this is bookkeeping, not a reopener.
3. The boundary table + reduction enumeration give the human a one-screen audit for the #407 relax-decision: it is greedy-safety, not physics, that binds.
