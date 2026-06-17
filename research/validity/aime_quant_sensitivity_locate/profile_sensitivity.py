#!/usr/bin/env python
"""PRIMARY: per-layer bf16-restore AIME-sensitivity profile (#586).

DIAGNOSTIC ONLY (analysis_only=true, official_tps=0). NO HF Job / submission /
served-file change. Localizes the int4->AIME quantization sensitivity WITHIN the
42-layer body: concentrated in a few decoder layers, or diffuse across all 42?

Method (teacher-forced, NO generation in the ablation):
  * Build a fixed batch of AIME-style reasoning prompts (chat-template, thinking)
    + bf16-greedy reference completions ("reasoning traces").
  * bf16 reference = the unquantized google/gemma-4-E4B-it (100% reference).
  * Working model = the int4-QAT gemma-4-E4B-it-qat-w4a16-ct DECOMPRESSED to dense
    bf16 (run_compressed=False) -> dense weight == library-exact dequant of the
    served int4 weights (isolates WEIGHT-quant error, not Marlin kernel numerics).
  * For each decoder layer L, restore ONLY layer L's params to bf16, teacher-force
    over the batch, and measure the reduction in final-logit divergence vs the
    full int4 baseline:  s_L = D_int4 - D_restore_L  (D = mean KL(P_bf16||Q) and,
    separately, argmax-flip rate on the reasoning tokens).
  * Concentration: top{1,5,10}_divergence_fraction = sum(topk max(s_L,0)) / sum(max(s_L,0)).
  * Cross-check: per-layer hidden-state relative-L2 int4-vs-bf16 (one paired fwd).

Verdict: aime_collapse_concentrated = (top5_divergence_fraction >= CONC_THRESH).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # repo root for research.downstream_quality_aime
from research.downstream_quality_aime.aime_eval import (  # noqa: E402
    load_aime, build_messages,
)

BF16_DIR = "/senpai-run/home/student-ubel/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/fee6332c1abaafb77f6f9624236c63aa2f1d0187"
INT4_ID = "google/gemma-4-E4B-it-qat-w4a16-ct"
DEVICE = "cuda:0"
CONC_THRESH = 0.60   # top5_divergence_fraction >= this => concentrated


def get_layers(model):
    return model.model.language_model.layers


def layer_shared_sd(layer):
    """State-dict keys that exist on BOTH bf16 and int4-dense layers (i.e. the
    dense `weight`s + RMSNorm `weight`s + buffers) — excludes int4-only
    weight_scale / weight_shape leftovers, which are vestigial once dense."""
    return layer.state_dict()


def build_batch(tokenizer, problems, max_new_tokens, min_new_tokens, bf16_model):
    """Greedy-generate bf16 reference completions; return per-problem token seqs
    and the scored (completion) position slice."""
    batch = []
    eos_ids = []
    gcfg = bf16_model.generation_config
    if gcfg.eos_token_id is not None:
        eos_ids = gcfg.eos_token_id if isinstance(gcfg.eos_token_id, list) else [gcfg.eos_token_id]
    for prob in problems:
        msgs = build_messages(prob["problem"])
        enc = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            enable_thinking=True, return_tensors="pt", return_dict=True,
        ).to(DEVICE)
        P = enc["input_ids"].shape[1]
        with torch.inference_mode():
            out = bf16_model.generate(
                **enc, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                do_sample=False, num_beams=1,
                pad_token_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
            )
        seq = out[0]                      # [P + C]
        C = seq.shape[0] - P
        if C < 1:
            continue
        batch.append({
            "id": prob["id"], "seq": seq.to(DEVICE), "P": P, "C": C,
            "scored": (P - 1, P + C - 1),  # logit positions predicting completion tokens
        })
        print(f"  [batch] id={prob['id']} P={P} C={C}", flush=True)
    return batch


@torch.inference_mode()
def teacher_forced_logprobs(model, seq, scored, want_hidden=False):
    """Return log_softmax logits at scored positions [C, V] (bf16) and, optionally,
    per-layer hidden states at scored positions [C, n_hidden, H]."""
    a, b = scored
    out = model(input_ids=seq[None], use_cache=False, output_hidden_states=want_hidden)
    logits = out.logits[0, a:b].float()          # [C, V]
    logp = F.log_softmax(logits, dim=-1)         # [C, V] fp32
    hid = None
    if want_hidden:
        hs = out.hidden_states                   # tuple(n_hidden) of [1, L, H]
        hid = torch.stack([h[0, a:b] for h in hs], dim=1).to(torch.bfloat16)  # [C, n_hidden, H]
    return logp.to(torch.bfloat16), hid




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2024,2025")
    ap.add_argument("--n-problems", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--min-new-tokens", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="2 problems, 24 tok, layers 0,20,41")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", default=str(HERE / "profile_result.json"))
    args = ap.parse_args()

    if args.smoke:
        args.n_problems, args.max_new_tokens = 2, 24

    torch.manual_seed(0)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(INT4_ID, trust_remote_code=True)

    problems_all = load_aime([y.strip() for y in args.years.split(",")], limit=None)
    # deterministic subset SPREAD across both years (first-N would be all-2024)
    if args.n_problems and args.n_problems < len(problems_all):
        stride = len(problems_all) / args.n_problems
        idx = sorted({int(i * stride) for i in range(args.n_problems)})
        problems = [problems_all[i] for i in idx]
    else:
        problems = problems_all
    yrs = sorted({p["year"] for p in problems})
    print(f"[load] {len(problems)} problems (years={yrs}); {time.time()-t0:.1f}s", flush=True)

    # ---------- Phase 1: bf16 reference (GPU) ----------
    print("[phase1] loading bf16 base ...", flush=True)
    bf16 = AutoModelForCausalLM.from_pretrained(BF16_DIR, dtype=torch.bfloat16, trust_remote_code=True).to(DEVICE).eval()
    batch = build_batch(tokenizer, problems, args.max_new_tokens, args.min_new_tokens, bf16)
    assert batch, "empty batch"
    n_scored = sum(b["C"] for b in batch)
    print(f"[phase1] batch={len(batch)} total_scored_tokens={n_scored}", flush=True)

    ref_logp, ref_argmax, bf16_hidden = [], [], []
    for b in batch:
        lp, hid = teacher_forced_logprobs(bf16, b["seq"], b["scored"], want_hidden=True)
        ref_logp.append(lp)                       # GPU bf16 [C,V]
        ref_argmax.append(lp.argmax(dim=-1))      # GPU [C]
        bf16_hidden.append(hid.cpu())             # CPU [C,n_hidden,H]
    # extract bf16 per-layer weights to CPU
    bf16_sd = [{k: v.detach().cpu() for k, v in layer_shared_sd(L).items()} for L in get_layers(bf16)]
    n_layers = len(bf16_sd)
    del bf16
    gc.collect(); torch.cuda.empty_cache()
    print(f"[phase1] done; n_layers={n_layers}; GPU freed; {time.time()-t0:.1f}s", flush=True)

    # ---------- Phase 2: int4-dense working model (GPU) ----------
    print("[phase2] loading int4 (run_compressed=False) ...", flush=True)
    qc = CompressedTensorsConfig(run_compressed=False)
    int4 = AutoModelForCausalLM.from_pretrained(
        INT4_ID, dtype=torch.bfloat16, trust_remote_code=True, quantization_config=qc
    ).to(DEVICE).eval()
    layers = get_layers(int4)
    assert len(layers) == n_layers

    def eval_config(want_hidden=False):
        """Online KL(P_ref||Q_var) + argmax-flip over scored positions; memory-flat
        (never holds all variant logits at once)."""
        kl_sum = 0.0
        flips = 0
        n = 0
        vhid = []
        for i, b in enumerate(batch):
            lp, hid = teacher_forced_logprobs(int4, b["seq"], b["scored"], want_hidden=want_hidden)
            rlp = ref_logp[i].float()
            p = rlp.exp()
            kl = (p * (rlp - lp.float())).sum(dim=-1)   # [C]
            kl_sum += kl.sum().item()
            flips += (lp.argmax(dim=-1) != ref_argmax[i]).sum().item()
            n += lp.shape[0]
            if want_hidden:
                vhid.append(hid.cpu())
            del lp, rlp, p, kl
        return kl_sum / n, flips / n, vhid

    # int4 baseline + hidden-L2 cross-check
    D_int4_kl, D_int4_flip, int4_hidden = eval_config(want_hidden=True)
    print(f"[phase2] D_int4: KL={D_int4_kl:.5f} flip={D_int4_flip:.4f}", flush=True)

    # per-layer hidden relative-L2 (accumulated divergence up to each hidden index)
    n_hidden = bf16_hidden[0].shape[1]
    hid_rel_l2 = []
    for hi in range(n_hidden):
        num = den = 0.0
        for bh, ih in zip(bf16_hidden, int4_hidden):
            d = (ih[:, hi].float() - bh[:, hi].float())
            num += (d * d).sum().item()
            den += (bh[:, hi].float() ** 2).sum().item()
        hid_rel_l2.append(math.sqrt(num / den) if den > 0 else 0.0)
    del int4_hidden, bf16_hidden
    gc.collect(); torch.cuda.empty_cache()

    # self-determinism: re-run int4 baseline, assert byte-identical divergence
    D_int4_kl2, D_int4_flip2, _ = eval_config(want_hidden=False)
    self_det = bool(D_int4_kl == D_int4_kl2 and D_int4_flip == D_int4_flip2)
    print(f"[phase2] self_det={self_det} (re-run KL={D_int4_kl2:.5f} flip={D_int4_flip2:.4f})", flush=True)

    # per-layer bf16-restore ablation
    layer_ids = list(range(n_layers))
    if args.smoke:
        layer_ids = [0, n_layers // 2, n_layers - 1]
    s_kl, s_flip, D_restore_kl, D_restore_flip = {}, {}, {}, {}
    for li in layer_ids:
        L = layers[li]
        orig = {k: v.detach().clone() for k, v in L.state_dict().items()}   # GPU snapshot
        L.load_state_dict(bf16_sd[li], strict=False)                        # restore -> bf16
        kl, flip, _ = eval_config(want_hidden=False)
        L.load_state_dict(orig, strict=False)                              # restore -> int4
        del orig
        D_restore_kl[li], D_restore_flip[li] = kl, flip
        s_kl[li] = D_int4_kl - kl
        s_flip[li] = D_int4_flip - flip
        print(f"  [layer {li:2d}] D_restore KL={kl:.5f} flip={flip:.4f} | s_kl={s_kl[li]:+.5f} s_flip={s_flip[li]:+.5f}", flush=True)

    # ---------- concentration metrics (KL-based primary) ----------
    def topk_fraction(s_map, k):
        vals = sorted((max(v, 0.0) for v in s_map.values()), reverse=True)
        tot = sum(vals)
        return (sum(vals[:k]) / tot) if tot > 0 else 0.0

    top1 = topk_fraction(s_kl, 1)
    top5 = topk_fraction(s_kl, 5)
    top10 = topk_fraction(s_kl, 10)
    ranked = sorted(s_kl.items(), key=lambda kv: kv[1], reverse=True)
    concentrated = bool(top5 >= CONC_THRESH)

    result = {
        "analysis_only": True, "official_tps": 0,
        "models": {"bf16": "google/gemma-4-E4B-it", "int4": INT4_ID,
                   "int4_decompress": "run_compressed=False (dense bf16 dequant)"},
        "n_problems": len(batch), "n_scored_tokens": n_scored,
        "max_new_tokens": args.max_new_tokens, "n_layers": n_layers,
        "D_int4_kl": D_int4_kl, "D_int4_flip": D_int4_flip,
        "self_det": self_det,
        "top1_divergence_fraction": top1,
        "top5_divergence_fraction": top5,
        "top10_divergence_fraction": top10,
        "aime_collapse_concentrated": concentrated,
        "conc_thresh": CONC_THRESH,
        "ranked_layers_by_s_kl": [[int(li), float(v)] for li, v in ranked],
        "per_layer": {
            str(li): {"s_kl": s_kl[li], "s_flip": s_flip[li],
                      "D_restore_kl": D_restore_kl[li], "D_restore_flip": D_restore_flip[li]}
            for li in layer_ids
        },
        "hidden_rel_l2": hid_rel_l2,
        "elapsed_s": time.time() - t0,
        "smoke": args.smoke,
    }
    nan_clean = all(math.isfinite(x) for x in [D_int4_kl, D_int4_flip, top1, top5, top10]
                    + [v for li in layer_ids for v in (s_kl[li], s_flip[li])])
    result["nan_clean"] = bool(nan_clean)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in
          ["top1_divergence_fraction", "top5_divergence_fraction", "top10_divergence_fraction",
           "aime_collapse_concentrated", "self_det", "nan_clean", "D_int4_kl", "D_int4_flip"]}, indent=2), flush=True)
    print(f"[done] wrote {args.out}; {time.time()-t0:.1f}s", flush=True)

    # ---------- W&B ----------
    if not args.no_wandb:
        import wandb
        run = wandb.init(
            entity="wandb-applied-ai-team", project="gemma-challenge-senpai",
            name="stark/aime-quant-sensitivity-locate",
            group="aime-quant-sensitivity-locate",
            config={k: result[k] for k in
                    ["analysis_only", "official_tps", "n_problems", "n_scored_tokens",
                     "max_new_tokens", "n_layers", "conc_thresh", "models", "smoke"]},
        )
        wandb.summary.update({k: result[k] for k in
            ["D_int4_kl", "D_int4_flip", "self_det", "nan_clean",
             "top1_divergence_fraction", "top5_divergence_fraction", "top10_divergence_fraction",
             "aime_collapse_concentrated"]})
        tbl = wandb.Table(columns=["layer", "s_kl", "s_flip", "D_restore_kl", "D_restore_flip", "hidden_rel_l2"])
        for li in layer_ids:
            tbl.add_data(li, s_kl[li], s_flip[li], D_restore_kl[li], D_restore_flip[li],
                         hid_rel_l2[li + 1] if li + 1 < len(hid_rel_l2) else float("nan"))
        wandb.log({"per_layer_sensitivity": tbl})
        for li in layer_ids:
            wandb.log({"layer": li, "s_kl": s_kl[li], "s_flip": s_flip[li],
                       "hidden_rel_l2": hid_rel_l2[li + 1] if li + 1 < len(hid_rel_l2) else float("nan")})
        art = wandb.Artifact("aime_sensitivity_profile", type="analysis")
        art.add_file(args.out)
        run.log_artifact(art)
        run.finish()
        print(f"[wandb] run id: {run.id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
