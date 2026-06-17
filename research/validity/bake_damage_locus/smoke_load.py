#!/usr/bin/env python3
"""Feasibility smoke for PR #539 bake-damage-locus.

Verifies HF transformers can load each substrate on the A10G and expose per-layer
hidden states + logits, for the offline divergence analysis. NO divergence math here
-- just: does it load, what's the model class, how many hidden states, what shapes,
is there logit softcapping, and what is the peak memory.

Run (assigned GPU):
  CUDA_VISIBLE_DEVICES=0 .venvs/vllm022/bin/python \
      research/validity/bake_damage_locus/smoke_load.py --which base_bf16
  CUDA_VISIBLE_DEVICES=0 .venvs/vllm022/bin/python \
      research/validity/bake_damage_locus/smoke_load.py --which base_int4
  CUDA_VISIBLE_DEVICES=0 .venvs/vllm022/bin/python \
      research/validity/bake_damage_locus/smoke_load.py --which osoi5 --path /tmp/osoi5-v0-baked
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODELS = {
    "base_bf16": "google/gemma-4-E4B-it",
    "base_int4": "google/gemma-4-E4B-it-qat-w4a16-ct",
    "osoi5": None,  # local path via --path
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=list(MODELS))
    ap.add_argument("--path", default=None, help="local checkpoint path (osoi5)")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_id = args.path if args.which == "osoi5" else MODELS[args.which]
    print(f"[smoke] which={args.which} model_id={model_id}", flush=True)

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"[smoke] config class: {type(cfg).__name__}", flush=True)
    print(f"[smoke] architectures: {getattr(cfg, 'architectures', None)}", flush=True)
    tcfg = getattr(cfg, "text_config", cfg)
    for k in ("num_hidden_layers", "hidden_size", "vocab_size", "head_dim",
              "num_kv_shared_layers", "final_logit_softcapping",
              "tie_word_embeddings", "layer_types"):
        v = getattr(tcfg, k, getattr(cfg, k, "<none>"))
        if k == "layer_types" and isinstance(v, list):
            full = [i for i, t in enumerate(v) if "full" in str(t)]
            print(f"[smoke]   {k}: len={len(v)} full_attn_idx={full}", flush=True)
        else:
            print(f"[smoke]   {k}: {v}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    torch.cuda.reset_peak_memory_stats()
    t_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=t_dtype, trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    load_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"[smoke] loaded; model class: {type(model).__name__}; "
          f"weight mem ~{load_mem:.2f} GB", flush=True)

    # tiny text-only forward, teacher-forced
    ids = tok("The capital of France is Paris. 2 + 2 =", return_tensors="pt").input_ids.to("cuda")
    print(f"[smoke] input_ids shape {tuple(ids.shape)}", flush=True)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    print(f"[smoke] num hidden_states = {len(hs)} (expect num_layers+1)", flush=True)
    print(f"[smoke] hs[0] shape {tuple(hs[0].shape)} dtype {hs[0].dtype}", flush=True)
    print(f"[smoke] hs[-1] shape {tuple(hs[-1].shape)}", flush=True)
    print(f"[smoke] logits shape {tuple(out.logits.shape)} dtype {out.logits.dtype}", flush=True)

    # softcap sanity: if final_logit_softcapping set, logits should be bounded by it
    cap = getattr(tcfg, "final_logit_softcapping", None)
    lg = out.logits[0, -1].float()
    print(f"[smoke] last-pos logits min/max = {lg.min().item():.3f}/{lg.max().item():.3f} "
          f"(softcap={cap})", flush=True)
    top = torch.topk(lg, 5)
    print(f"[smoke] top5 ids {top.indices.tolist()} vals {[round(v,3) for v in top.values.tolist()]}",
          flush=True)
    print(f"[smoke] decoded top1: {tok.decode(top.indices[:1])!r}", flush=True)

    # is hs[-1] pre- or post-final-norm? compare to manual lm_head on hs[-1]
    try:
        lm_head = model.get_output_embeddings()
        if lm_head is not None:
            man = lm_head(hs[-1][0, -1].to(t_dtype)).float()
            # cos between manual (no softcap) and model logits (maybe softcapped)
            cos = torch.nn.functional.cosine_similarity(
                man.unsqueeze(0), lg.unsqueeze(0)).item()
            print(f"[smoke] cos(lm_head@hs[-1], model.logits) = {cos:.6f} "
                  f"(1.0 => hs[-1] is post-final-norm & logits pre-softcap)", flush=True)
            print(f"[smoke] manual logits min/max = {man.min().item():.3f}/{man.max().item():.3f}",
                  flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[smoke] manual lm_head check skipped: {e!r}", flush=True)

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[smoke] PEAK mem ~{peak:.2f} GB", flush=True)
    print("[smoke] OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
