# /// script
# requires-python = ">=3.10"
# dependencies = ["llmcompressor", "compressed-tensors", "torch", "transformers==5.9.0"]
# ///
"""LEAD BIG-SWING BET: 2:4 sparsity + W4A16 (Sparse-Marlin) for gemma-4-E4B-it.

Sparse-Marlin is A10-PROVEN: ~30% extra throughput / 20% lower latency from the
2:4 sparsity on top of int4, and the combined int4+2:4 kernel runs ~3.3x vs fp16
in vLLM [IST-DASLab/Sparse-Marlin; vLLM #10260, cap>=8.0]. At batch-1
(bandwidth-bound) 2:4 halves the int4 weight bytes again -> the largest hardware-
matched lever after int4 itself. Also quantizes lm_head (drop it from `ignore`)
to fold in that bandwidth win.

One-shot recipe (SparseGPT 2:4 -> GPTQ W4A16). Calibration-only (no training), so
it fits a single GPU run. **Quality is the gate**: one-shot 2:4 on a 4.5B model
may exceed the 2.42 PPL cap; if so, the proven fix is short 2:4 sparse-aware
fine-tuning (Sparse-Llama precedent) — a heavier GPU job. Validate PPL first.

Run on GPU: `hf jobs uv run --flavor a10g-large build_2of4_w4a16.py`  (a10g-large
for headroom), or embed the build in a benchmark serve.py to validate in one shot.
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import SparseGPTModifier
from llmcompressor.modifiers.quantization import GPTQModifier

BASE = os.environ.get("BASE_MODEL", "google/gemma-4-E4B-it")  # bf16 base
OUT = os.environ.get("OUT_DIR", "gemma4-e4b-2of4-w4a16")
NCALIB = int(os.environ.get("NCALIB", "512"))
SEQLEN = int(os.environ.get("SEQLEN", "2048"))

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="auto", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(BASE)

# calibration data (generic; for the challenge, calibrating on eval-prompt-style
# CoT text would track the target distribution better)
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=42).select(range(NCALIB))
def prep(ex):
    return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}
ds = ds.map(prep)

recipe = [
    # 1) 2:4 structured sparsity (SparseGPT), skip embeddings/lm_head for the prune
    SparseGPTModifier(sparsity=0.5, mask_structure="2:4", targets=["Linear"],
                      ignore=["lm_head", "re:.*vision_tower.*", "re:.*audio.*", "re:.*embed.*"]),
    # 2) W4A16 int4 group-128 over Linear INCLUDING lm_head (fold in lm_head bandwidth)
    GPTQModifier(targets=["Linear"], scheme="W4A16", group_size=128,
                 ignore=["re:.*vision_tower.*", "re:.*audio.*", "re:.*embed_tokens.*"]),
]

oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=SEQLEN,
        num_calibration_samples=NCALIB, output_dir=OUT)
print(f"[build] wrote {OUT} — 2:4 sparse + W4A16-g128 (incl lm_head). VALIDATE PPL <= 2.42.")
