#!/usr/bin/env python
"""COMPLEMENT to the restore-direction profile (#586): per-layer QUANTIZE-FROM-BF16
sensitivity. Start from the full bf16 model and quantize ONLY layer L (swap in the
int4-dequant weights), measure the divergence INTRODUCED: d_L = KL(bf16 || bf16+quant_L)
on the same teacher-forced AIME reasoning tokens. Unlike single-layer restore-INTO-int4
(confounded by QAT co-adaptation -> negative s), this is all-positive and is the
canonical layer-wise quant-sensitivity, giving a clean ranking + an independent
concentration check. Verdict input: is the quant DAMAGE concentrated in a few layers?

DIAGNOSTIC ONLY (analysis_only=true, official_tps=0). NO HF Job / submission / ship.
"""
from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
from profile_sensitivity import (  # noqa: E402  (reuse validated helpers)
    BF16_DIR, INT4_ID, DEVICE, get_layers, teacher_forced_logprobs, build_batch,
)
from research.downstream_quality_aime.aime_eval import load_aime  # noqa: E402

CONC_THRESH = 0.60
N_PROBLEMS = 16
MAX_NEW = 320


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=N_PROBLEMS)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW)
    ap.add_argument("--years", default="2024,2025")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", default=str(HERE / "quantize_result.json"))
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(INT4_ID, trust_remote_code=True)
    pall = load_aime([y.strip() for y in args.years.split(",")], limit=None)
    if args.n_problems and args.n_problems < len(pall):
        stride = len(pall) / args.n_problems
        idx = sorted({int(i * stride) for i in range(args.n_problems)})
        problems = [pall[i] for i in idx]
    else:
        problems = pall
    print(f"[load] {len(problems)} problems; {time.time()-t0:.1f}s", flush=True)

    # bf16 = reference AND working model (on GPU)
    print("[q] loading bf16 (ref+working) ...", flush=True)
    bf16 = AutoModelForCausalLM.from_pretrained(BF16_DIR, dtype=torch.bfloat16, trust_remote_code=True).to(DEVICE).eval()
    batch = build_batch(tok, problems, args.max_new_tokens, 8, bf16)
    n_scored = sum(b["C"] for b in batch)
    print(f"[q] batch={len(batch)} scored={n_scored}", flush=True)
    ref_logp, ref_argmax = [], []
    for b in batch:
        lp, _ = teacher_forced_logprobs(bf16, b["seq"], b["scored"], want_hidden=False)
        ref_logp.append(lp)
        ref_argmax.append(lp.argmax(dim=-1))
    # working model IS bf16; the per-layer bf16 weights are captured just-in-time
    # inside the loop (orig clone), so no full second copy on GPU (was OOM).
    n_layers = len(get_layers(bf16))

    # int4-dequant layer weights -> CPU (load int4-dense on CPU, extract, free)
    print("[q] loading int4-dense on CPU to extract dequant layer weights ...", flush=True)
    qc = CompressedTensorsConfig(run_compressed=False)
    int4_cpu = AutoModelForCausalLM.from_pretrained(INT4_ID, dtype=torch.bfloat16, trust_remote_code=True, quantization_config=qc)
    int4_sd = [{k: v.detach().cpu() for k, v in L.state_dict().items()} for L in get_layers(int4_cpu)]
    del int4_cpu
    gc.collect(); torch.cuda.empty_cache()

    layers = get_layers(bf16)  # working == bf16

    def eval_div():
        kl_sum = 0.0; flips = 0; n = 0
        for i, b in enumerate(batch):
            lp, _ = teacher_forced_logprobs(bf16, b["seq"], b["scored"], want_hidden=False)
            rlp = ref_logp[i].float(); p = rlp.exp()
            kl_sum += (p * (rlp - lp.float())).sum(dim=-1).sum().item()
            flips += (lp.argmax(dim=-1) != ref_argmax[i]).sum().item()
            n += lp.shape[0]
            del lp, rlp, p
        return kl_sum / n, flips / n

    base_kl, base_flip = eval_div()  # bf16 vs bf16 == 0 sanity
    print(f"[q] baseline (bf16 vs bf16): KL={base_kl:.6f} flip={base_flip:.5f} (expect ~0)", flush=True)

    d_kl, d_flip = {}, {}
    for li in range(n_layers):
        L = layers[li]
        orig = {k: v.detach().clone() for k, v in L.state_dict().items()}
        L.load_state_dict(int4_sd[li], strict=False)        # quantize layer li
        kl, flip = eval_div()
        L.load_state_dict(orig, strict=False)               # restore bf16
        del orig
        d_kl[li], d_flip[li] = kl, flip
        print(f"  [L{li:02d}] d_kl={kl:.5f} d_flip={flip:.4f}", flush=True)

    # self-determinism: re-quantize one layer, assert identical
    L = layers[0]; orig = {k: v.detach().clone() for k, v in L.state_dict().items()}
    L.load_state_dict(int4_sd[0], strict=False); k2, f2 = eval_div(); L.load_state_dict(orig, strict=False)
    self_det = bool(k2 == d_kl[0] and f2 == d_flip[0])

    vals = sorted(d_kl.values(), reverse=True)
    tot = sum(vals)
    top1 = vals[0] / tot if tot else 0.0
    top5 = sum(vals[:5]) / tot if tot else 0.0
    top10 = sum(vals[:10]) / tot if tot else 0.0
    ranked = sorted(d_kl.items(), key=lambda kv: kv[1], reverse=True)
    concentrated = bool(top5 >= CONC_THRESH)

    result = {
        "analysis_only": True, "official_tps": 0, "direction": "quantize_from_bf16",
        "n_problems": len(batch), "n_scored_tokens": n_scored, "n_layers": n_layers,
        "baseline_bf16_vs_bf16_kl": base_kl,
        "sum_single_layer_d_kl": tot, "self_det": self_det,
        "top1_divergence_fraction": top1, "top5_divergence_fraction": top5,
        "top10_divergence_fraction": top10, "conc_thresh": CONC_THRESH,
        "quantize_concentrated": concentrated,
        "ranked_layers_by_d_kl": [[int(li), float(v)] for li, v in ranked],
        "per_layer": {str(li): {"d_kl": d_kl[li], "d_flip": d_flip[li]} for li in range(n_layers)},
        "elapsed_s": time.time() - t0,
    }
    result["nan_clean"] = bool(all(math.isfinite(v) for v in d_kl.values()) and math.isfinite(top5))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["top1_divergence_fraction", "top5_divergence_fraction",
          "top10_divergence_fraction", "quantize_concentrated", "self_det", "sum_single_layer_d_kl",
          "nan_clean"]}, indent=2), flush=True)
    print("top sensitive layers (quantize):", [li for li, _ in ranked[:8]], flush=True)
    print(f"[done] wrote {args.out}; {time.time()-t0:.1f}s", flush=True)

    if not args.no_wandb:
        import wandb
        run = wandb.init(entity="wandb-applied-ai-team", project="gemma-challenge-senpai",
                         name="stark/aime-quant-sensitivity-locate-quantdir",
                         group="aime-quant-sensitivity-locate",
                         config={"analysis_only": True, "official_tps": 0, "direction": "quantize_from_bf16",
                                 "n_problems": len(batch), "n_scored_tokens": n_scored})
        wandb.summary.update({k: result[k] for k in ["top1_divergence_fraction", "top5_divergence_fraction",
              "top10_divergence_fraction", "quantize_concentrated", "self_det", "sum_single_layer_d_kl", "nan_clean"]})
        tbl = wandb.Table(columns=["layer", "d_kl", "d_flip"])
        for li in range(n_layers):
            tbl.add_data(li, d_kl[li], d_flip[li]); wandb.log({"layer": li, "d_kl": d_kl[li], "d_flip": d_flip[li]})
        wandb.log({"quantize_sensitivity": tbl})
        run.finish(); print(f"[wandb] run id: {run.id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
