STUDENT lawine:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["wndiyzxk"],"primary_metric":{"name":"base_fullhead_spec_tps","value":253.99},"test_metric":{"name":"official_tps","value":0},"spec_drafter":"mtp_k7","acceptance_length":3.844,"greedy_identity_vs_base_fullhead":false,"exceeds_ship":false,"gap_to_ship":121.87,"beats_capstone_floor":false,"quality_gate_passes_by_construction":true,"self_det":0.188,"analysis_only":true,"official_tps":0}

## Results — base_fullhead + spec-dec ceiling: **NO FIRE (clean miss on both gates)**

`base_fullhead` + the ship's own **MTP K=7** drafter, measured served on the idle pod A10G at `MAX_NUM_SEQS=1`, lands at **253.99 TPS** — **below the 311.25 capstone floor and 121.87 TPS short of the 375.857 ship**. The last un-bounded lever-class (spec-dec) is now MEASURED, and the `verdict_flip_condition` is **NOT met**. **No HF Job / submission / served-file change. `analysis_only=true`, `official_tps=0`.**

### Key outputs

| output | value |
|---|---|
| `base_fullhead_spec_tps` (local, warm-median of 2) | **253.99** (runs 252.35 / 255.62; steady-gen 243.04) |
| `spec_drafter` | `mtp_k7` (ship surgical-357 `SPECULATIVE_CONFIG` verbatim, K=7) |
| `acceptance_length` | **3.844** (e_accept_exact; 71 intervals; 127 972 acc / 314 951 draft; accept-rate 0.406) |
| `official_projected_tps` (×1.03524, #267) | 262.94 |
| `exceeds_ship` (≥375.857) | **false** — `gap_to_ship` **121.87** |
| `beats_capstone_floor` (>311.25) | **false** |
| `quality_gate_passes_by_construction` | **true** (base_fullhead's 95.2% MMLU / 99.9% GPQA / 97.3% GSM8K / 118% AIME) |
| `greedy_identity_vs_base_fullhead` (LIGHT) | false — **noise-limited, see below**; denken #576 owns the rigorous census |
| `self_det` spec / nospec | 0.188 / 0.454 |
| `analysis_only` / `official_tps` / NaN-clean | true / 0 / **clean** |

W&B: [`wndiyzxk`](https://wandb.ai/wandb-applied-ai-team/gemma-challenge-senpai/runs/wndiyzxk) (group `base-fullhead-specdec-ceiling`). Peak GPU **≈19.4 GiB** (19 409 MiB, identical-substrate smoke; the full run's GPU sampler died with the disk crash below).

### Mechanism — why spec can't escape the bound in absolute terms

At `MAX_NUM_SEQS=1` each verify step is **memory-bound by the per-step weight load** (int4 body + the full **bf16 262k head**). Spec amortizes the head's *compute* (K+1=8 candidate positions cost ≈ the same memory traffic as 1), but the per-step weight *traffic* is paid once per step regardless of K. So MTP K=7 converts the heavy-head no-spec penalty into ~254 TPS via acceptance 3.844 — a genuine **+204% lift over the real no-spec (83→254)** — but the absolute ceiling is still set by one memory-bound step per ~1/acceptance tokens. The ship beats this (375.857) precisely by **lightening the per-step load** — 12k head-prune + osoi5 body-bake — *not* by better acceptance (it uses the same drafter). base_fullhead by definition keeps the full head and unbaked body, so it cannot. **The per-token-decode weight-load that the capstone ceilings bound is NOT escaped by spec in absolute terms** — spec just trades the head's compute amortization on top of it.

### ⚠️ Finding to reconcile: the **no-spec anchor 252.69 is not reproduced on this pod** (measured 83.44)

Same-pod, same-kernel `base_fullhead` no-spec measures **83.44 TPS** (wall == steady "Avg generation throughput" == 82–84), not the card's 252.69 (wirbel #553). This is well-supported and physically grounded — flagging because the **311.25 capstone floor is derived from 252.69 and inherits the caveat**:

- **Reference mode changes only `SPECULATIVE_CONFIG`.** `serve.py:disable_speculation_for_reference_mode()` clears the drafter and nothing else; both server logs show identical PIECEWISE cudagraph capture, `enforce_eager` off — the **only** config delta between the 83 and 254 arms is `speculative_config` mtp-vs-None. So 83.44 *is* the legitimate same-kernel no-spec baseline (and the +204% spec lift over it is real).
- **The stock head is bf16.** `gemma-4-E4B-it-qat-w4a16-ct` keeps `lm_head` in the quant `ignore` list → the full native 262 144-row head is **bf16 (~1.34 GB)**, not int4. Loading that per M=1 decode step is memory-bound at ~12 ms/token → ~83 TPS. **252.69 TPS = 3.96 ms/token is below the memory floor for a bf16 262k head** — unreachable as *plain* no-spec; it must come from a lighter/quantized head or a different metric.

The PRIMARY verdict is robust on either basis: 253.99 < 311.25 < 375.857 is a **clean miss** whether the no-spec reference is 83 or 252.69.

### Greedy identity (light gate — deferred to denken #576)

Measured spec-on vs no-spec (same kernels, drafter off) over 128×512: `seq_frac` 0.156, first-divergence onset median frac 0.26 (late+spread). **This is noise-limited, not a spec-specific identity break:** the engine has heavy intrinsic int4 FP/ULP nondeterminism — **no-spec greedy only reproduces 45.4% of 512-token sequences run-to-run** (sampler is PyTorch-native argmax, `VLLM_USE_FLASHINFER_SAMPLER=0`, so this is kernel-level, not the fern #566 flashinfer/tie path). Sequence-level identity therefore *cannot exceed* the 0.454 self-det ceiling and **cannot certify or refute #319 here**. Per your relay, **denken #576** owns the rigorous served byte-exact census; this light check corroborates that base_fullhead+spec is not trivially divergent (onset is late, not at token 0 for most seqs).

### What happened

The hypothesis asked whether spec — the one lever the per-token-decode ceiling doesn't bound — escapes it on a quality-safe config. **It does not, in absolute terms.** Spec gives a large *relative* lift on the heavy head (the card's "heavier head → spec helps more relatively" intuition is correct: no-spec is crippled to 83, spec recovers to 254), but the absolute ceiling 254 is still set by the memory-bound per-verify-step weight load of the full bf16 head + unbaked body. To clear 375.857 you must lighten that per-step load (prune/bake) — i.e. leave base_fullhead. So the `verdict_flip_condition` is **closed with a hard served number** on the spec-dec axis: NO FIRE. The clean, publishable form is "spec helps a lot (+204%) but lands at 254 — short of the 311.25 floor and 121.87 short of the ship."

### Exact commands

```bash
cd /workspace/senpai/target
# serve base_fullhead substrate (stock int4 + native bf16 262k head, NO bake/prune) + ship MTP K=7,
# warm + 2 timed decodes per arm (128x512, seed 1); + no-spec M=1 AR reference arm (SENPAI_REFERENCE_MODE=1):
.venv/bin/python research/base_fullhead_specdec_ceiling/probe_specdec_ceiling.py \
    --tag full --num-prompts 128 --output-len 512 \
    --wandb-group base-fullhead-specdec-ceiling
# ^ ARM A measured cleanly, then crashed disk-full (Errno 28, 2026-06-17 ~09:09 UTC, shared node hit 100%)
#   BEFORE the final report. All load-bearing artifacts (ARM A r1/r2 + server logs, ARM B r1) are complete;
#   assembled the final report from them (no re-serve — avoids another disk crash on the 95%-full node):
cd research/base_fullhead_specdec_ceiling && /workspace/senpai/target/.venv/bin/python assemble_report.py
```

### Suggested follow-ups

1. **Reconcile the 252.69 no-spec anchor** (wirbel #553) against this pod's measured 83.44 — most likely the anchor used a quantized/lighter head, not the true native bf16 262k head. If so, the 311.25 "magically-free-head" floor (derived from 252.69) should be re-grounded on the true bf16-head no-spec (~83), which would move the floor substantially.
2. **The interesting absolute lever stays head/body weight, not spec.** Spec on base_fullhead is a wash against the ship because the ship's win is per-step load reduction. Any future "quality-safe + fast" route needs a lighter *quality-preserving* head (e.g. an int4 head that holds #319 + PPL) — spec is then additive on top, not a substitute.
3. **MTP K sweep is unlikely to help** here: acceptance 3.844 at K=7 is already near the drafter's design point, and raising K only adds draft-forward cost against a fixed memory-bound verify step. Not worth a run unless the head is first lightened.
