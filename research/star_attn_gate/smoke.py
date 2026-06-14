"""Smoke test for PR #93 star-attention greedy-equivalence gate.

Loads the deployed int4 verifier (osoi5-v0-baked), inspects the text decoder
self-attention module tree (to locate the attention-output tensor the star-
attention kernel would perturb), and teacher-forces ONE #86 greedy prompt to
confirm the plain-AR post-softcap argmax reproduces the known served greedy
completion tokens (reference-frame alignment, program.md 27-28).
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

MODEL = "/tmp/osoi5-v0-baked"
CORPUS = "research/rank_coverage/pr86/decode_rank_coverage.jsonl"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_model():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL)
    print("[smoke] architectures:", cfg.architectures, "model_type:", cfg.model_type)
    from transformers import Gemma4ForConditionalGeneration
    t0 = time.time()
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"},
    )
    model.eval()
    print(f"[smoke] loaded in {time.time()-t0:.1f}s")
    return model


def find_text_layers(model):
    """Return the ModuleList of text decoder layers regardless of nesting."""
    candidates = []
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList):
            # heuristic: text layers have self_attn with q_proj/o_proj
            first = mod[0]
            if hasattr(first, "self_attn"):
                candidates.append((name, mod))
    for name, mod in candidates:
        print(f"[smoke] layer stack: {name}  (len={len(mod)})")
    # text stack is the longest one (37) that is NOT vision/audio
    text = None
    for name, mod in candidates:
        if "vision" not in name and "audio" not in name:
            if text is None or len(mod) > len(text[1]):
                text = (name, mod)
    return text


def main():
    model = load_model()
    text = find_text_layers(model)
    if text is None:
        print("[smoke] FAILED to find text layers")
        sys.exit(1)
    name, layers = text
    print(f"[smoke] TEXT layer stack = {name}, n={len(layers)}")
    attn = layers[0].self_attn
    print("[smoke] self_attn type:", type(attn).__name__)
    print("[smoke] self_attn children:", [n for n, _ in attn.named_children()])
    # identify the output projection (o_proj or post)
    for cand in ("o_proj", "post"):
        if hasattr(attn, cand):
            print(f"[smoke] output projection submodule = self_attn.{cand}")
            sub = getattr(attn, cand)
            print(f"[smoke]   {cand} type:", type(sub).__name__)

    # restricted-head keepset: head-index -> real vocab token id
    ks = json.load(open(os.path.join(MODEL, "pck04_keepset.json")))
    keep_ids = torch.tensor(ks["keep_ids"], device="cuda")  # (16384,)
    print(f"[smoke] keepset K={ks['pruned_vocab_K']} full_vocab={ks['full_vocab']}")
    softcap = getattr(model.config.text_config, "final_logit_softcapping", None)
    print("[smoke] final_logit_softcapping:", softcap)

    # ---- teacher-force a few prompts, map argmax through keepset, check alignment ----
    recs = [json.loads(l) for l in open(CORPUS)][:4]
    tot_match = tot = 0
    for rec in recs:
        p = rec["prompt_token_ids"]
        c = rec["completion_token_ids"]
        ids = torch.tensor([p + c], device="cuda")
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
        logits = out.logits[0]  # (T, K=16384) post-softcap
        start = len(p) - 1
        pred = logits[start:start + len(c)].float()
        top2 = pred.topk(2, dim=-1)
        head_arg = top2.indices[:, 0]
        real_arg = keep_ids[head_arg].tolist()
        gold = c
        match = sum(int(a == g) for a, g in zip(real_arg, gold))
        tot_match += match
        tot += len(c)
        # restricted-head margin (top1-top2) at mismatches
        margin = (top2.values[:, 0] - top2.values[:, 1])
        mism_idx = [i for i in range(len(c)) if real_arg[i] != gold[i]]
        mm = [(i, round(margin[i].item(), 4)) for i in mism_idx[:6]]
        print(f"[smoke] {rec['id']}: match {match}/{len(c)}={match/len(c):.4f}; "
              f"max|logit|={pred.abs().max().item():.2f}; "
              f"margin median={margin.median().item():.3f} min={margin.min().item():.4f}; "
              f"mismatch(i,margin)[:6]={mm}")
        # are gold tokens even in the keepset?
        not_in = sum(1 for g in gold if g not in set(ks["keep_ids"]))
        if not_in:
            print(f"[smoke]    gold tokens NOT in keepset: {not_in}/{len(c)}")
    print(f"[smoke] OVERALL clean argmax==served greedy: {tot_match}/{tot} = {tot_match/tot:.4f}")
    print(f"[smoke] peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
