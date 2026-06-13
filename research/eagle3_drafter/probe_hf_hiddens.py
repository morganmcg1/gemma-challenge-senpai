"""Probe: confirm HF transformers exposes Gemma-4 E4B text hidden states for
EAGLE-3 corpus generation (PR #16, Step 2 de-risk).

Confirms on the real model:
  - the module path to the 42-layer text tower,
  - output_hidden_states returns a tuple of length 43 ([emb] + 42 layers),
  - aux indices (2, 21, 39) have shape [B, T, 2560] with no NaN,
  - tied embeddings (embed_tokens.weight is the lm_head),
  - rope_theta / config knobs needed by the draft head,
  - peak GPU memory for a single 512-token forward.

Run (from target/):
  HF_HOME=/senpai-run/home/student-fern/.cache/huggingface \
  /tmp/server-venv/bin/python research/eagle3_drafter/probe_hf_hiddens.py
"""

import os
import json

import torch

MODEL = os.environ.get("PROBE_MODEL", "google/gemma-4-E4B-it")
AUX = (2, 21, 39)


def find_text_model(model):
    """Locate the submodule whose .layers has 42 entries and has embed_tokens."""
    candidates = []
    for name, mod in model.named_modules():
        layers = getattr(mod, "layers", None)
        emb = getattr(mod, "embed_tokens", None)
        if layers is not None and emb is not None:
            try:
                n = len(layers)
            except TypeError:
                continue
            candidates.append((name, n, mod))
    return candidates


def main():
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    print(f"[probe] loading {MODEL} (bf16) ...", flush=True)
    cfg = AutoConfig.from_pretrained(MODEL)
    print("[probe] config class:", type(cfg).__name__, flush=True)

    # Load the full model (no device_map -> avoids the accelerate dep); we only
    # exercise the text path, then move to GPU.
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    except Exception as e:  # noqa: BLE001
        print("[probe] AutoModelForCausalLM failed:", repr(e), flush=True)
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16)
    model = model.to("cuda")
    model.eval()
    print("[probe] model class:", type(model).__name__, flush=True)

    cands = find_text_model(model)
    print("[probe] (module, n_layers) candidates:", [(n, k) for n, k, _ in cands], flush=True)
    text = None
    for name, n, mod in cands:
        if n == 42:
            text = mod
            print(f"[probe] using text module '{name}' with {n} layers", flush=True)
            break
    assert text is not None, "could not find 42-layer text module"

    # Tied embeddings? Gemma ties input/output.
    emb_w = text.embed_tokens.weight
    lm_head = getattr(model, "lm_head", None)
    tied = lm_head is not None and (lm_head.weight.data_ptr() == emb_w.data_ptr())
    print("[probe] embed_tokens.weight shape:", tuple(emb_w.shape), "dtype", emb_w.dtype, flush=True)
    print("[probe] lm_head tied to embed_tokens:", tied, flush=True)

    # rope/config knobs we need for the draft head.
    tcfg = getattr(cfg, "text_config", cfg)
    knobs = {
        k: getattr(tcfg, k, None)
        for k in [
            "hidden_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "intermediate_size",
            "rms_norm_eps", "rope_theta", "rope_local_base_freq",
            "vocab_size", "max_position_embeddings", "query_pre_attn_scalar",
        ]
    }
    print("[probe] text knobs:", json.dumps(knobs, default=str), flush=True)

    # Forward over a 512-token random-but-valid sequence using the text module.
    torch.cuda.reset_peak_memory_stats()
    T = 512
    ids = torch.randint(0, knobs["vocab_size"], (1, T), device="cuda")
    with torch.no_grad():
        out = text(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    print("[probe] num hidden_states:", len(hs), "(expect 43)", flush=True)
    for i in AUX:
        h = hs[i]
        print(
            f"[probe] hidden_states[{i}]: shape {tuple(h.shape)} dtype {h.dtype} "
            f"nan={bool(torch.isnan(h).any())} std={h.float().std().item():.3f} "
            f"absmax={h.float().abs().max().item():.2f}",
            flush=True,
        )
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[probe] peak GPU mem for 1x512 forward: {peak:.2f} GB", flush=True)
    print("[probe] done.", flush=True)


if __name__ == "__main__":
    main()
