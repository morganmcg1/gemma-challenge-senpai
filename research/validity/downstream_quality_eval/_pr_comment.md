STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["pfu3vy7c"],"primary_metric":{"name":"ship_mmlu_pro","value":0.274},"test_metric":{"name":"ship_gpqa_diamond","value":0.2323}}

## Results

**VERDICT — MOAT REFUTED.** The served **surgical-357 ship does NOT reproduce base downstream quality.** On byte-identical prompts it collapses into the same pruned-substrate regime dixie-flatline measured in #483, and **fails the quality gate** on both tasks. The asserted "byte-exact greedy-equivalent / operative-1.0" moat does **not** hold on the broad MMLU-Pro / GPQA-Diamond free-generation distribution.

### KEY OUTPUTS

| output | value |
|---|---|
| `base_mmlu_pro` | **0.668** (334/500) |
| `base_gpqa_diamond` | **0.4444** (88/198, 1 err) |
| `ship_mmlu_pro` | **0.274** (137/500) |
| `ship_gpqa_diamond` | **0.2323** (46/198, 1 err) |
| `mmlu_pro_delta_ship_minus_base` | **−0.394** |
| `gpqa_delta_ship_minus_base` | **−0.2121** |
| `ship_passes_quality_gate` | **false** (needs MMLU‑Pro ≥ 0.60 AND GPQA‑Diamond ≥ 0.42; ship = 0.274 / 0.2323 — both fail) |
| `reproduces_dixie_base_anchors` | **true** |
| `eval_subset_n` | **500** (MMLU‑Pro, seed 12345) · GPQA‑Diamond run **full = 198** |

**One-line verdict:** served ship = pruned substrate (collapses, gate FAIL) — the zero-degradation moat is refuted, not confirmed.

### A/B on byte-identical prompts (PRIMARY deliverable)

| task | base | ship | Δ (ship−base) | frac retained | gate thresh | ship gate |
|---|---|---|---|---|---|---|
| MMLU‑Pro (n=500) | 0.668 | 0.274 | **−0.394** | 41% | ≥0.60 | **FAIL** |
| GPQA‑Diamond (n=198) | 0.4444 | 0.2323 | **−0.2121** | 52% | ≥0.42 | **FAIL** |

- **Prompts byte-identical** (`prompt_sha` match, **0 mismatches** in both tasks) → the entire gap is **model-driven**, not a prompt/subset artifact.
- MMLU‑Pro is 10-choice (chance 0.10); GPQA‑Diamond is 4-choice (**chance 0.25** → ship 0.2323 is **at chance**).
- **Moat-break** (base-correct → ship-wrong): MMLU **212/500**, GPQA **58/198**. Ship-only-correct: 15 / 16. Answer-agreement: 0.352 / 0.328.
- Output pathology (ship vs base): MMLU **empty-answer 8.4% → 26%**, **truncation@2048 8.6% → 26%** — ship far more often rambles to the cap or emits no parseable `ANSWER`, consistent with dixie's "answers fluently and wrong."

### Harness validity — dual cross-validation (SECONDARY)

The harness is faithful because **two independent harnesses (mine + dixie #483) agree on BOTH endpoints**:

| anchor | dixie #483 | my measurement | match |
|---|---|---|---|
| base MMLU‑Pro | 0.668 | 0.668 | **exact** |
| base GPQA‑Diamond | 0.470 | 0.4444 (CI95 0.375–0.514) | **within CI95** |
| substrate MMLU‑Pro | 0.330 | 0.274 (collapsed regime) | same regime |
| substrate GPQA‑Diamond | 0.283 | 0.2323 (CI95 0.173–0.291) | **within CI95** |

Base reproduces the **base** anchors and ship reproduces the **substrate** anchors → the collapse is the **substrate**, not the harness. A 5-question smoke scored base correctly before each full run (self-test passed).

### Why this does NOT contradict the operative-1.0 / byte-exact census

There is **no contradiction** — the identity was measured on axes that never exercise the failure mode:

1. **The operative-1.0 census (wirbel #510 / stark #509) is ship-vs-ship.** "9/128 flips, 0 semantic" compares surgical-357 vs the 222-flag variant — **both on the same osoi5-12k substrate**. It's an attention-order identity between two pruned models, never ship-vs-**base**.
2. **Byte-exact greedy-equivalence was asserted on the 128-sample speed/ppl smoke** — a **narrow calibration distribution** where the 12k keepset is sufficient (those prompts stay in-keepset). It was never measured on the broad MMLU/GPQA distribution.
3. **PPL 2.3767 is teacher-forced** — the model is fed the gold token, so it never has to *pick* from the keepset; teacher-forcing structurally cannot surface keepset-insufficiency. Free-generation accuracy is a different axis.

**Mechanism of collapse:** the 12k "reachable-token" keepset was computed on a narrow calibration set. On broad MMLU/GPQA prompts the model frequently wants a token *outside* the keepset; the head-prune scatter `-inf`s it, so the model emits its best *in-keepset* token instead → coherent on-topic CoT with the wrong final answer. I confirmed this by reading raw completions (fluent vector-addition / chemistry reasoning, wrong choice), not garbage.

### Faithfulness of the ship serve (verified)

- `/tmp/osoi5-12k-baked` `config.json`: `quant_method=compressed-tensors`, **int4 pack-quantized** on the backbone `Linear` group **and** on `lm_head` (`re:.*lm_head$`, channel-wise, num_bits 4) → served at official int4 precision, **not** a bf16 proxy (`--dtype bfloat16` is the activation dtype; weights are int4).
- `pck04_keepset.json`: **12288 unique** ids over the 262144 vocab; ship arm rebuilds `lm_head` to K=12288 and scatters via the **real** submission `serve_patch_pck04.py` (not a re-implementation). Manifest `LM_HEAD_KEEPSET_BUCKET=dixie-flatline/int4-pck04c-12k` → ship literally uses dixie's substrate keepset.

**Caveat (honest):** my serve uses standard `TRITON_ATTN` + **no** spec-dec; the official serve adds surgical-attn/fa-sliding/splitkv + MTP‑K7. Those are **speed/ULP-level** (spec-dec is output-verified-equivalent; surgical-attn yields the sub-ULP tie flips of the operative census) and **cannot rescue a 0.39 collapse** — if anything my unsped, unperturbed reference decode is a clean **upper bound** on ship quality. The dual anchor (ship reproduces dixie's *independent* substrate measurement) corroborates this.

### Exact commands

```bash
# Ship server (CUDA_VISIBLE_DEVICES=0, VLLM_ATTENTION_BACKEND=TRITON_ATTN, VLLM_USE_FLASHINFER_SAMPLER=0,
#   PCK04_KEEPSET set, PYTHONPATH=pck04_inject:submissions/fa2sw_strict_surgical357)
bash start_server.sh ship /tmp/osoi5-12k-baked /tmp/osoi5-12k-baked/pck04_keepset.json
#  -> python -m vllm.entrypoints.openai.api_server --model /tmp/osoi5-12k-baked \
#       --served-model-name gemma-4-e4b-it --host 127.0.0.1 --port 8000 --dtype bfloat16 \
#       --max-model-len 4096 --gpu-memory-utilization 0.90 --max-num-seqs 16 --trust-remote-code \
#       --disable-log-stats --override-generation-config '{"temperature":0.0,"top_p":1.0,"top_k":0}'

# Ship eval (greedy)
python run_eval.py --task mmlu_pro    --arm ship --out ship_mmlu_pro.json \
  --n 500 --seed 12345 --max-tokens 2048 --max-connections 16 \
  --base-url http://127.0.0.1:8000/v1 --model gemma-4-e4b-it
python run_eval.py --task gpqa_diamond --arm ship --out ship_gpqa.json \
  --seed 12345 --max-tokens 3072 --max-connections 16 \
  --base-url http://127.0.0.1:8000/v1 --model gemma-4-e4b-it

# Base eval used the identical run_eval.py invocations with --arm base against the
# vanilla-vLLM base server (same n/seed/max-tokens), giving byte-identical prompts.

# Aggregate + supplement
python3 aggregate.py        # system python3 (wandb) -> aggregate.json + W&B pfu3vy7c
/tmp/eval-serve-venv/bin/python _supplement.py   # inspect_ai (truncation/empty analysis) -> _supplement.json
```

- **W&B run:** `pfu3vy7c` (project `wandb-applied-ai-team/gemma-challenge-senpai`, group `downstream-quality-eval`).
- **Peak GPU memory:** ~**19.6 GB** / 24 GB A10G serving footprint (`--gpu-memory-utilization 0.90`, single GPU, `CUDA_VISIBLE_DEVICES=0`).
- **Run mode:** `analysis_only=true`, `official_tps=0`, **no HF Jobs job, no `--launch`, no submission, no served-file change.**

### What happened — honest analysis

The byte-exact moat was **asserted, not measured on this axis**, and it is **false** here. Surgical-357 *is* dixie's collapsing substrate (it literally loads dixie's 12k keepset). The operative-1.0 census proved a **ship-vs-ship** attention-order identity and PPL proved a teacher-forced likelihood match — **neither tests free-generation ship-vs-base capability**, which is exactly where the 12k head-prune breaks (keepset insufficient on the broad task distribution). On greedy MMLU/GPQA the ship loses ~59% of MMLU-Pro and ~48% of GPQA capability and drops to chance on GPQA. **This PR's A/B is the first ship-vs-base downstream measurement, and it refutes the zero-degradation claim.**

### Suggested follow-ups

1. **The fix is a distribution-matched (or unpruned) `lm_head`.** Recompute the keepset over an MMLU/GPQA-representative calibration set, or drop the head-prune entirely (keep layer-reduction + int4) and re-measure. The head-prune is the dominant capability lever — sweep quality-vs-speed for keepset ∈ {12k, 24k, 48k, full}.
2. **Belt-and-suspenders:** re-run the ship arm through the FULL official stack (surgical-attn + MTP‑K7 spec-dec) to confirm the collapse is substrate-driven, not an artifact of the simplified serve (expected: no change — the dual anchor already cross-validates).
3. **Optional generation_config-sampled secondary pass** (lewtun #31 leaderboard protocol uses sampling, not greedy) — quantifies the leaderboard-protocol number; won't fix keepset-insufficiency.
4. **Cross-link the identity legs** (stark #509, wirbel #510, denken #505): their ship-vs-ship operative-identity, while true, does **not** transfer to a ship-vs-base capability claim. The quality gate needs a ship-vs-base downstream measurement — which this harness now provides and can be reused as the pre-submission gate.

### Public evidence used

This harness **reproduces and extends dixie-flatline Issue #483** (MMLU-Pro 0.668→0.330, GPQA-Diamond 0.470→0.283 substrate collapse) under the **same** inspect_evals / greedy / pinned challenge vLLM wheel (0.22.1rc1) protocol. dixie's #483 is the evidence anchor: my **base** row reproduces dixie's base anchors and my **ship** row reproduces dixie's substrate regime, cross-validating both endpoints. No external web sources beyond the challenge's own inspect_evals MMLU-Pro / GPQA-Diamond task definitions were used.
