"""Precision/timing diagnostic for the star-attention greedy gate (PR #93).

Answers, on 2 #86 prompts:
  1. dtype + magnitude of the o_proj INPUT (the attention-output tensor the star-
     attention kernel produces) per layer; bf16 ULP at that magnitude.
  2. realized relerr after injecting eps-relative fp32 noise then casting back to
     the deployed dtype (does a 1e-3 perturbation survive bf16 rounding?).
  3. true fp32 final-logit top1-top2 margin vs bf16 margin; bf16 tie fraction.
  4. per-forward wall time (to budget the full 128-prompt sweep).
"""
from __future__ import annotations
import json, os, time, math
import torch

MODEL = "/tmp/osoi5-v0-baked"
CORPUS = "research/rank_coverage/pr86/decode_rank_coverage.jsonl"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main():
    from transformers import Gemma4ForConditionalGeneration
    t0 = time.time()
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"})
    model.eval()
    print(f"[diag] loaded {time.time()-t0:.1f}s, attn_impl={model.config.text_config._attn_implementation}")
    layers = model.model.language_model.layers
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.model.language_model.lm_head if hasattr(model.model.language_model, 'lm_head') else None
    print("[diag] lm_head:", type(getattr(model, 'lm_head', None)).__name__,
          "weight dtype:", getattr(getattr(model, 'lm_head', None), 'weight', torch.tensor(0)).dtype)
    softcap = float(model.config.text_config.final_logit_softcapping)

    # capture o_proj-input stats via forward_pre_hooks
    stats = {}
    def mk(i):
        def hook(mod, args):
            a = args[0]
            stats[i] = (str(a.dtype), float(a.norm()), int(a.numel()),
                        float(a.abs().mean()), float(a.abs().max()))
            return None
        return hook
    hs = [layers[i].self_attn.o_proj.register_forward_pre_hook(mk(i)) for i in range(len(layers))]

    recs = [json.loads(l) for l in open(CORPUS)][:2]
    rec = recs[0]
    ids = torch.tensor([rec["prompt_token_ids"] + rec["completion_token_ids"]], device="cuda")
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    torch.cuda.synchronize()
    print(f"[diag] forward(1x768) = {time.time()-t:.3f}s")
    for h in hs: h.remove()

    print("[diag] o_proj INPUT stats (layer: dtype norm numel mean|.| max|.|):")
    for i in [0, 1, 2, 18, 36]:
        if i in stats:
            print(f"   L{i}: {stats[i]}")
    # bf16 ULP at the typical magnitude
    samp = layers[0].self_attn  # just to compute ulp on a sample value
    v = torch.tensor([stats[0][3]], dtype=torch.bfloat16)  # mean|.|
    ulp = (torch.nextafter(v, v + 1) - v).item()
    print(f"[diag] bf16 ULP at mean|a|={stats[0][3]:.4f} is ~{ulp:.5f} (rel {ulp/max(stats[0][3],1e-9):.4f})")

    # realized relerr after eps-relative fp32 noise then cast back to a.dtype
    print("[diag] realized relerr after inject+cast-back (per layer-0 attn output):")
    a0_dtype = torch.bfloat16
    # reconstruct a representative tensor scale: use stats norm
    for eps in (1e-4, 1e-3, 1e-2):
        # synthetic per-row test on a bf16 tensor of similar scale
        torch.manual_seed(0)
        a = torch.randn(768, 2560, device="cuda", dtype=torch.bfloat16) * stats[0][3] * 1.25
        af = a.float()
        rownorm = af.norm(dim=-1, keepdim=True)
        z = torch.randn_like(af)
        delta = eps * rownorm / math.sqrt(af.shape[-1]) * z
        a_pert = (af + delta).to(torch.bfloat16)
        realized = ((a_pert.float() - af).norm() / af.norm()).item()
        changed = (a_pert != a).float().mean().item()
        print(f"   eps={eps:.0e}: realized_relerr={realized:.2e}  frac_elements_changed={changed:.4f}")

    # bf16 vs fp32 margin on the clean pass
    keep = json.load(open(os.path.join(MODEL, "pck04_keepset.json")))["keep_ids"]
    keep_t = torch.tensor(keep, device="cuda")
    # capture lm_head input (final normed hidden, bf16)
    cap = {}
    def lmhook(mod, args):
        cap["h"] = args[0].detach()
        return None
    hh = model.lm_head.register_forward_pre_hook(lmhook)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    hh.remove()
    bf16_logits = out.logits[0].float()  # already softcapped, but bf16-rounded -> upcast
    h = cap["h"][0]  # (T, hidden) bf16
    # fp32 logits: need fp32 lm_head weight
    W = model.lm_head.weight  # may be bf16 (decompressed) or packed
    print("[diag] lm_head.weight dtype/shape:", W.dtype, tuple(W.shape))
    fp32_ok = W.dtype in (torch.float16, torch.bfloat16, torch.float32)
    start = len(rec["prompt_token_ids"]) - 1
    L = len(rec["completion_token_ids"])
    # bf16-readout margin
    seg_bf = bf16_logits[start:start+L]
    t2b = seg_bf.topk(2, dim=-1)
    mb = (t2b.values[:,0]-t2b.values[:,1])
    print(f"[diag] BF16 margin: median={mb.median():.4f} min={mb.min():.4f} "
          f"frac==0:{(mb==0).float().mean():.4f} frac<=0.125:{(mb<=0.1251).float().mean():.4f}")
    if fp32_ok:
        Wf = W.float()
        bias = model.lm_head.bias.float() if getattr(model.lm_head,'bias',None) is not None else None
        hf = h.float()[start-0:]  # all positions; we slice below
        # compute fp32 logits for the completion positions only
        hseg = h.float()[ (start):(start+L) ] if h.shape[0]>=start+L else h.float()
        # h is aligned to ids positions; lm_head input position p predicts token p+1
        # we already sliced bf16 by start; mirror with h positions start..start+L-1
        logits_fp = hseg @ Wf.t()
        if bias is not None: logits_fp = logits_fp + bias
        logits_fp = softcap * torch.tanh(logits_fp / softcap)
        t2f = logits_fp.topk(2, dim=-1)
        mf = (t2f.values[:,0]-t2f.values[:,1])
        top1mag = t2f.values[:,0].abs()
        relmf = mf/top1mag.clamp_min(1e-6)
        print(f"[diag] FP32 margin: median={mf.median():.5f} min={mf.min():.6f} p1={mf.kthvalue(max(1,int(0.01*L))).values:.6f}")
        for thr in (1e-2,1e-3,1e-4):
            print(f"   frac rel-margin < {thr:.0e} of top1: {(relmf<thr).float().mean():.4f}")
        print(f"[diag] FP32 tie (margin==0): {(mf==0).float().mean():.4f}")
    print(f"[diag] peak mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
