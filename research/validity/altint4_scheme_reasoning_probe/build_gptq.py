#!/usr/bin/env python
"""PR #647 — build an ALTERNATIVE int4-g128 PTQ body (GPTQ) for the scheme-vs-scheme
reasoning probe.

The live body (`/workspace/gemma_build/int4_g128_lmhead`) is Google's QAT-int4
(W4A16, group_size 128, symmetric int, pack-quantized) over every language-model
Linear + lm_head, towers left bf16. This script produces a DIFFERENT int4 recipe at
the *same* bit-width and group size — GPTQ (Hessian-weighted PTQ) via llm-compressor —
so PR #647 can ask whether the graduate-reasoning loss (GPQA-D, AIME) is specific to
the QAT recipe or intrinsic to 4-bit at g128.

Match to the live QAT footprint:
  * scheme W4A16  -> num_bits 4, type int, symmetric, strategy group, group_size 128
    (verified identical to the live body's quantization_config).
  * targets="Linear" over the language model -> all 343 decoder Linears
    (self_attn q/k/v/o, mlp gate/up/down, per_layer_input_gate, per_layer_projection,
    per_layer_model_projection); KV-shared layers have no own k/v_proj, exactly as QAT.
  * ignore the vision_tower + audio_tower + embed_vision/embed_audio (towers stay bf16,
    same as QAT).

ONE deliberate deviation from the live body: **lm_head is left bf16 (tied embedding),
NOT int4.** Rationale: (1) PR #600 established the int4 *body* drives the quality break
and the head contribution is negligible; this isolates the BODY recipe (the #647
question) instead of confounding it with a head-precision change. (2) The base ships
tied embeddings (no separate lm_head tensor); GPTQ-on-tied-lm_head is fragile. (3) It is
conservative: a bf16 head can only *help* the alt scheme, so an alt that still loses
reasoning ~equally to QAT makes the "intrinsic" verdict stronger, not weaker. Flagged in
the write-up; matching the int4 head is a suggested follow-up.

Calibration is **reasoning-rich, non-leaking**: GSM8K *train* chain-of-thought
(question -> full CoT solution), chat-templated. GSM8K is a distinct instrument from the
two failing bars (GPQA-D, AIME), so there is no eval leakage; CoT math is the lever most
likely to separate a reasoning-preserving int4 from a knowledge-only one (PR #647).

ANALYSIS-ONLY artifact build. Local A10G. NO HF Job, NO submission.

Usage:
  build_gptq.py --out /workspace/gemma_build/altint4_gptq_g128 --num-calib 256 --max-seq 1024
  build_gptq.py --smoke            # 8 samples, seq 256, pipeline+arch validation
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = "google/gemma-4-E4B-it"
DECODER_LAYER = "Gemma4TextDecoderLayer"
# towers + multimodal projectors stay bf16 (exactly the live QAT body's ignore set);
# lm_head stays bf16 (tied) — see module docstring.
IGNORE = [
    "lm_head",
    "re:.*vision_tower.*",
    "re:.*audio_tower.*",
    "re:.*embed_vision.*",
    "re:.*embed_audio.*",
]


_LETTERS = "ABCDEFGHIJKLMNOP"


def _patch_gptq_weight_observer() -> str:
    """Work around an llmcompressor-0.12.0 + Gemma4 sequential-partition bug.

    In the SequentialPipeline, ``GPTQModifier.on_event(SEQUENTIAL_EPOCH_END)`` runs
    ``observe(get_modules(subgraph.submodules(model)), "weight")`` (populates each
    module's ``weight_observer`` with the per-group weight min/max) and THEN
    ``compress_modules()`` (iterates every module that fired a calibration hook,
    i.e. ``self._num_samples.keys()``, and calls ``quantize_weight`` ->
    ``observer.get_qparams()``).

    On the FX-traced Gemma4 multimodal graph those two module sets diverge: a
    module's Hessian hook can fire during a subgraph's forward even though the
    module is not in that subgraph's *declared* ``submodules()`` list, so it gets
    compressed without ever being observed -> ``get_qparams()`` asserts
    "No statistics available." (per_layer_model_projection compresses fine, the
    next Linear dies). See smoke_build.log.

    Fix: wrap ``quantize_weight`` so that, immediately before it computes qparams,
    the module's weight observer is guaranteed populated. Weight min/max is a pure
    function of ``module.weight`` (data-independent), and at call time the weight is
    still uncompressed (sequential compress pops modules one at a time), so this is
    byte-equivalent to the intended ``observe(...)`` and leaves ``actorder=None``
    (the QAT-matching scheme) untouched. Idempotent: skips if stats already present.
    """
    from llmcompressor.modifiers.gptq import base as _gptq_base
    from llmcompressor.modifiers.quantization.calibration import observe as _observe

    _orig = _gptq_base.quantize_weight

    def _quantize_weight(module, *a, **k):
        obs = getattr(module, "weight_observer", None)
        if obs is not None and not obs.has_statistics:
            _observe(module, base_name="weight")
        return _orig(module, *a, **k)

    _gptq_base.quantize_weight = _quantize_weight
    return "patched gptq.base.quantize_weight: ensure weight_observer before get_qparams"


def _tok_turn(tok, q: str, a: str, max_seq: int) -> list[int] | None:
    """user(q)->assistant(a) rendered through the chat template then tokenized.

    transformers 5.9 apply_chat_template(tokenize=True) returns an Encoding object; render
    to a string (template already injects <bos> + <|turn> markers) then tokenize with
    add_special_tokens=False to avoid a double BOS."""
    messages = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if not ids:
        return None
    return ids[:max_seq]


def build_calibration(tok, num_samples: int, max_seq: int, seed: int = 1234):
    """Reasoning-rich CoT calibration -> list[{"input_ids": [...]}].

    A MIX of two non-leaking reasoning instruments (neither is GPQA-D or AIME — the two
    failing bars — so there is no eval leakage), rendered as user->assistant CoT turns in
    the model's own chat template:
      * MMLU-Pro validation `cot_content` (all 70: graduate-breadth multi-domain worked CoT
        — the lever closest to GPQA-D's reasoning).
      * GSM8K train answers (math word-problem CoT — closest to AIME), filling the rest.
    Calibration data is the most likely lever separating a reasoning-preserving int4 from a
    knowledge-only one (PR #647), so we deliberately use CoT, not generic web text."""
    import random

    from datasets import load_dataset

    rng = random.Random(seed)
    samples: list[list[int]] = []

    # --- MMLU-Pro validation CoT (graduate breadth) ---
    n_mmlu = 0
    try:
        mp = load_dataset("TIGER-Lab/mmlu-pro", split="validation")
        for r in mp:
            opts = r.get("options") or []
            opt_txt = "\n".join(f"{_LETTERS[i]}. {o}" for i, o in enumerate(opts))
            q = f"{r['question'].strip()}\n{opt_txt}"
            a = str(r.get("cot_content") or "").strip()
            if not a:
                continue
            ids = _tok_turn(tok, q, a, max_seq)
            if ids:
                samples.append(ids)
                n_mmlu += 1
    except Exception as exc:  # noqa: BLE001
        print(f"[build] MMLU-Pro calib skipped: {repr(exc)[:160]}", flush=True)

    # --- GSM8K train CoT (math), fill the remainder ---
    n_gsm = 0
    gs = load_dataset("openai/gsm8k", "main", split="train")
    order = list(range(len(gs)))
    rng.shuffle(order)
    for i in order:
        if len(samples) >= num_samples:
            break
        ids = _tok_turn(tok, gs[i]["question"].strip(), gs[i]["answer"].strip(), max_seq)
        if ids:
            samples.append(ids)
            n_gsm += 1

    rng.shuffle(samples)
    samples = samples[:num_samples]
    print(f"[build] calib mix: mmlu_pro={n_mmlu} gsm8k={n_gsm} -> {len(samples)} used",
          flush=True)
    return [{"input_ids": s} for s in samples]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/gemma_build/altint4_gptq_g128")
    ap.add_argument("--num-calib", type=int, default=256)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--smoke", action="store_true",
                    help="8 samples / seq 256 — validate arch trace + per-layer timing")
    args = ap.parse_args()
    if args.smoke:
        args.num_calib, args.max_seq = 8, 256
        args.out = args.out + "_smoke"

    from scripts.local_validation import paths
    for note in paths.prepare_local_gpu_env():
        print(f"[gpu] {note}", flush=True)

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    print(f"[build] {_patch_gptq_weight_observer()}", flush=True)

    t0 = time.time()
    # Load the full bf16 model (~16 GB) DIRECTLY on the A10G. The basic pipeline's
    # dispatch_model keeps an already-GPU-resident model in place; a CPU-resident model
    # is left on CPU and the calibration forward runs ~17 s/sample (CPU) instead of
    # <1 s (GPU). offload_hessians keeps the ~20 GB of Hessians on host RAM, so GPU peak
    # is model + transients, well under 23.7 GB.
    dev = "cuda:0" if torch.cuda.is_available() else None
    print(f"[build] loading tokenizer + base {BASE} (bf16, device_map={dev}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=dev
    )
    model.eval()
    print(f"[build] model loaded in {time.time()-t0:.0f}s; type={type(model).__name__} "
          f"dev={next(model.parameters()).device}", flush=True)

    print(f"[build] building calibration: MMLU-Pro-val + GSM8K-train CoT mix "
          f"n={args.num_calib} max_seq={args.max_seq} ...", flush=True)
    calib = build_calibration(tok, args.num_calib, args.max_seq)
    ds = Dataset.from_list(calib)
    lens = [len(c["input_ids"]) for c in calib]
    print(f"[build] calib ready: {len(calib)} samples, tok-len [{min(lens)},{max(lens)}] "
          f"mean {sum(lens)/len(lens):.0f}", flush=True)

    def collator(batch):
        # batch_size=1 -> no padding needed; return the single sequence as [1, T].
        return {"input_ids": torch.tensor([batch[0]["input_ids"]], dtype=torch.long)}

    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",                 # int4, group_size 128, symmetric int (== QAT body)
        ignore=IGNORE,
        offload_hessians=True,          # ~20 GB of Hessians live on CPU (709 GB host RAM)
    )
    # BASIC (non-sequential) pipeline: ONE full-model calibration forward, then a single
    # compress pass over every calibrated Linear. We cannot use the auto-inferred
    # SequentialPipeline because it partitions at Gemma4TextDecoderLayer boundaries and
    # Gemma4's 18 KV-shared tail layers read `shared_kv_states[layer_type]` populated by a
    # source layer in an EARLIER subgraph — across the partition that dict is empty
    # (KeyError 'sliding_attention'). The full-model forward keeps KV-sharing intact. The
    # only cost is no sequential error propagation -> independent per-layer GPTQ (still
    # full Hessian-weighted OBS rounding, far stronger than RTN). Flagged in the write-up.
    print(f"[build] GPTQ recipe: W4A16 g128 sym, targets=Linear, pipeline=basic "
          f"(independent, KV-share-safe), offload_hessians=True, ignore={IGNORE}", flush=True)

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        data_collator=collator,
        num_calibration_samples=len(calib),
        max_seq_length=args.max_seq,
        batch_size=1,
        pad_to_max_length=False,
        pipeline="basic",
    )
    print(f"[build] GPTQ oneshot done in {time.time()-t0:.0f}s total", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), save_compressed=True)
    tok.save_pretrained(str(out))
    # save_pretrained writes config/generation_config/tokenizer + the chat template, but NOT
    # the multimodal processor config: we loaded AutoTokenizer, not AutoProcessor. gemma-4-E4B
    # is multimodal, so vLLM's loader needs processor_config.json (the single file defining the
    # Gemma4Processor — tokenizer + image/audio/video processors) or serve dies with
    # "Can't load feature extractor ... preprocessor_config.json". Carry the base's verbatim so
    # the checkpoint is self-contained and serves exactly like the live QAT body.
    import shutil

    from huggingface_hub import hf_hub_download
    for fname in ("processor_config.json",):
        try:
            shutil.copyfile(hf_hub_download(BASE, fname), out / fname)
            print(f"[build] carried {fname} from base", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[build] WARN could not carry {fname}: {repr(exc)[:160]}", flush=True)
    print(f"[build] saved compressed checkpoint -> {out}", flush=True)

    # provenance
    (out / "_build_meta.json").write_text(json.dumps({
        "base": BASE, "scheme": "W4A16", "group_size": 128, "symmetric": True,
        "method": "gptq", "targets": "Linear", "ignore": IGNORE,
        "pipeline": "basic", "error_propagation": False, "offload_hessians": True,
        "pipeline_note": "independent per-layer GPTQ; basic pipeline avoids Gemma4 "
                         "KV-share break under sequential layer partitioning",
        "lm_head_quantized": False, "lm_head_note": "bf16 tied (body-isolation, #600)",
        "calib": "mmlu_pro-val-cot + gsm8k-train-cot (reasoning-rich, non-leaking)",
        "num_calib": len(calib), "max_seq": args.max_seq,
        "build_seconds": round(time.time() - t0, 1),
        "analysis_only": True, "official_tps": 0,
    }, indent=2))
    print(f"[build] ALL DONE {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
