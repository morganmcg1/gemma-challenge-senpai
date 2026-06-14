#!/usr/bin/env python
"""PR #140 — Marlin group-size scale-BW: offline PPL scan over servable int4 body group sizes.

Step-1 of the staged build-or-kill gate. Step-0 (static) already established that the
pinned vLLM-0.22 Marlin W4A16 path serves group_size in {-1, 32, 64, 128} only
(MARLIN_SUPPORTED_GROUP_SIZES, commit 3e8afdf78) -> g=256 is UNSERVABLE, so the only
"coarser-than-128 = fewer scale bytes" option is g=-1 (per-channel).

This scan measures the offline fake-quant PPL for each servable body group size, isolating
the body group-size as the ONLY variable:
  source  : cached official google/gemma-4-E4B-it-qat-w4a16-ct (g=32 int4, the QAT base)
  dequant : unpack int4 + per-group(g=32) scale -> bf16 QAT body weights (343 lang modules)
  requant : fake-quant each body module at the target g (compressed-tensors primitives,
            identical math to submissions/int4_g128_lmhead/build_quant.py)
  head    : lm_head stays bf16-tied to embed_tokens (constant across arms -> body-isolated)
  score   : teacher-forced NLL over ppl_ground_truth_tokens.jsonl, ppl=exp(sum_nll/sum_tok)
            (exact ppl_endpoint.py / validate_offline.py convention)

Hard gate: PPL <= 2.42 (served #52 PPL 2.3772; offline g=128 anchor 2.3812 per wirbel #118).
LOCAL only: no HF Job, no submission. GPU forward (CUDA_VISIBLE_DEVICES=0).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import time
from pathlib import Path

import torch
from safetensors import safe_open

from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import quantize, dequantize
from compressed_tensors.quantization.utils.helpers import calculate_qparams
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

LANG_PREFIX = "model.language_model."
PPL_CAP = 2.42


def make_qargs(group_size: int) -> QuantizationArgs:
    if group_size == -1:
        return QuantizationArgs(num_bits=4, type="int", strategy="channel",
                                symmetric=True, observer="minmax")
    return QuantizationArgs(num_bits=4, type="int", strategy="group", group_size=group_size,
                            symmetric=True, observer="minmax")


def dequant_module(packed: torch.Tensor, scale: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """int32-packed int4 + per-group bf16 scale -> fp32 dense weight (symmetric, zp=0)."""
    out_dim, in_dim = int(shape[0]), int(shape[1])
    q = unpack_from_int32(packed, 4, torch.Size([out_dim, in_dim]), packed_dim=1).to(torch.float32)
    assert int(q.min()) >= -8 and int(q.max()) <= 7, f"int4 range [{q.min()},{q.max()}]"
    n_groups = scale.shape[1]
    assert in_dim % n_groups == 0, f"in_dim {in_dim} not divisible by n_groups {n_groups}"
    gs = in_dim // n_groups
    qg = q.reshape(out_dim, n_groups, gs)
    return (qg * scale.float().unsqueeze(-1)).reshape(out_dim, in_dim)


def fake_quant(w_f32: torch.Tensor, group_size: int) -> torch.Tensor:
    """Quantize->dequantize a weight at the target int4 group size (build_quant.py math)."""
    out_dim, in_dim = w_f32.shape
    qargs = make_qargs(group_size)
    if group_size == -1:
        min_vals = w_f32.amin(dim=-1, keepdim=True)
        max_vals = w_f32.amax(dim=-1, keepdim=True)
    else:
        assert in_dim % group_size == 0, f"in_dim {in_dim} not divisible by {group_size}"
        ng = in_dim // group_size
        wg = w_f32.reshape(out_dim, ng, group_size)
        min_vals = wg.amin(dim=-1)
        max_vals = wg.amax(dim=-1)
    scale, zp = calculate_qparams(min_vals, max_vals, qargs)
    q = quantize(w_f32, scale, zp, qargs)
    deq = dequantize(q, scale, zp, qargs)
    rel = float((w_f32 - deq).norm() / w_f32.norm().clamp_min(1e-9))
    return deq, scale, rel


def load_source(snap: Path):
    """Return (base_bf16{base->fp32 dequant body}, plain_text{key->tensor}, scale_groups{base->n_groups})."""
    base_bf16: dict[str, torch.Tensor] = {}
    plain_text: dict[str, torch.Tensor] = {}
    parts: dict[str, dict[str, torch.Tensor]] = {}
    g32_scale_bytes = 0
    with safe_open(str(snap / "model.safetensors"), framework="pt", device="cpu") as f:
        for name in f.keys():
            if name.endswith((".weight_packed", ".weight_scale", ".weight_shape")):
                base, kind = name.rsplit(".", 1)
                parts.setdefault(base, {})[kind] = f.get_tensor(name)
            elif name.startswith(LANG_PREFIX):
                plain_text["model." + name[len(LANG_PREFIX):]] = f.get_tensor(name)
            # vision_tower / audio_tower / projectors: not part of the text PPL model
    scale_elems_g32 = 0
    for base, p in parts.items():
        w = dequant_module(p["weight_packed"], p["weight_scale"], p["weight_shape"])
        base_bf16[base] = w.to(torch.bfloat16)  # bf16 base (== qat_unq); upcast per-arm
        scale_elems_g32 += p["weight_scale"].numel()
    return base_bf16, plain_text, scale_elems_g32


def build_model(snap: Path, plain_text: dict, device: str, head: str, head_g: int):
    """head='bf16_tied' (clean body isolation) or 'int4' (untied int4 head at head_g,
    matches the cap-comparable int4_g128_lmhead anchor 2.3812). Head is held constant
    across all body arms so the body group size is the only variable."""
    from transformers import Gemma4ForCausalLM
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    cfg_full = json.load(open(snap / "config.json"))
    tc = dict(cfg_full["text_config"])
    tc["tie_word_embeddings"] = (head == "bf16_tied")
    cfg = Gemma4TextConfig(**tc)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            model = Gemma4ForCausalLM(cfg)
    finally:
        torch.set_default_dtype(prev)
    res = model.load_state_dict({k: v.to(device) for k, v in plain_text.items()},
                                strict=False, assign=True)
    if head == "bf16_tied":
        model.tie_weights()
        head_rel = 0.0
    else:
        embed = plain_text["model.embed_tokens.weight"].float()
        lm_w, _, head_rel = fake_quant(embed, head_g)   # untie + int4 head from embed (== int4_g128_lmhead)
        model.load_state_dict({"lm_head.weight": lm_w.to(torch.bfloat16).to(device)},
                              strict=False, assign=False)
    print(f"[model] built | head={head}{'' if head=='bf16_tied' else f' g={head_g} rel={head_rel:.4f}'} "
          f"| unexpected={len(res.unexpected_keys)} missing={len(res.missing_keys)}", flush=True)
    model = model.eval()
    return model


def apply_body(model, base_bf16: dict, group_size: int, device: str):
    """Fake-quant every body module at group_size and copy into the model in place.

    In-place per-module copy (no full on-device state-dict) holds at most one ~50 MB
    module temp at a time, so the untied int4 head fits the A10G 22 GiB budget (the
    old full-body_sd build was a ~7 GB 2x duplicate of the body that OOM'd)."""
    rels = []
    scale_elems = 0
    params = dict(model.named_parameters())
    with torch.no_grad():
        for base, w_bf16 in base_bf16.items():
            key = "model." + base[len(LANG_PREFIX):] + ".weight"
            if group_size == 32:
                w = w_bf16.to(torch.bfloat16)            # native g=32 dequant (no re-round)
                scale_elems += (w_bf16.shape[1] // 32) * w_bf16.shape[0]
            else:
                deq, scale, rel = fake_quant(w_bf16.float(), group_size)  # f32 requant (== build_quant.py)
                w = deq.to(torch.bfloat16)
                rels.append(rel)
                scale_elems += scale.numel()
            p = params.get(key)
            assert p is not None, f"missing body param {key}"
            p.copy_(w.to(device))
            del w
    rel_stats = (min(rels), sum(rels) / len(rels), max(rels)) if rels else (0.0, 0.0, 0.0)
    return scale_elems, rel_stats


@torch.no_grad()
def ppl_sweep(model, records, device: str, limit: int | None) -> dict:
    total_nll, total_tok = 0.0, 0
    rec_ppls = []
    recs = records[:limit] if limit else records
    for i, r in enumerate(recs):
        ctx = r["context_token_ids"]
        tgt = r["target_token_ids"]
        ids = torch.tensor([ctx + tgt], dtype=torch.long, device=device)
        score_start = max(len(ctx), 1)
        out = model(input_ids=ids)
        logits = out.logits[0]
        sl = logits[score_start - 1:ids.shape[1] - 1].float()
        lp = torch.log_softmax(sl, dim=-1)
        tgt_ids = ids[0, score_start:ids.shape[1]]
        tok_lp = lp.gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        nll = float(-tok_lp.sum().item())
        n = int(tgt_ids.numel())
        total_nll += nll
        total_tok += n
        rec_ppls.append(math.exp(nll / n))
        if (i + 1) % 32 == 0:
            print(f"    scored {i+1}/{len(recs)} (running ppl={math.exp(total_nll/total_tok):.4f})", flush=True)
    return {
        "ppl": math.exp(total_nll / total_tok),
        "mean_record_ppl": sum(rec_ppls) / len(rec_ppls),
        "num_records": len(recs),
        "num_tokens": total_tok,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    default_snap = next(Path(
        "/senpai-run/home/student-ubel/.cache/huggingface/hub/"
        "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots").glob("*"))
    ap.add_argument("--snap", default=str(default_snap))
    ap.add_argument("--ppl", default="official/main_bucket/shared_resources/speed_benchmark/"
                                      "data/ppl_ground_truth_tokens.jsonl")
    ap.add_argument("--arms", default="32,128,-1", help="comma list of body group sizes")
    ap.add_argument("--head", default="int4", choices=["int4", "bf16_tied"],
                    help="int4 untied head (cap-comparable, == int4_g128_lmhead anchor) or bf16_tied")
    ap.add_argument("--head-g", type=int, default=128, help="int4 head group size (held constant)")
    ap.add_argument("--limit", type=int, default=0, help="0=all 128 records; >0 = smoke")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--out", default="research/marlin_groupsize_scalebw/ppl_scan_results.json")
    args = ap.parse_args()

    arms = [int(x) for x in args.arms.split(",")]
    limit = args.limit or None
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    snap = Path(args.snap)
    print(f"[init] snap={snap.name} device={device} torch={torch.__version__} arms={arms} "
          f"limit={limit or 'all'}", flush=True)

    records = [json.loads(l) for l in open(args.ppl) if l.strip()]
    t0 = time.time()
    base_bf16, plain_text, scale_elems_g32 = load_source(snap)
    print(f"[source] dequantized {len(base_bf16)} body modules | plain text tensors={len(plain_text)} "
          f"| {time.time()-t0:.1f}s", flush=True)

    model = build_model(snap, plain_text, device, args.head, args.head_g)

    # coverage: every text weight/bias must come from plain_text or a body module (lm_head is tied)
    body_keys = {"model." + b[len(LANG_PREFIX):] + ".weight" for b in base_bf16}
    covered = set(plain_text) | body_keys | {"lm_head.weight", "model.embed_tokens.weight"}
    uncovered = [k for k, _ in model.named_parameters()
                 if (k.endswith(".weight") or k.endswith(".bias")) and k not in covered]
    assert not uncovered, f"{len(uncovered)} text weights left at random init: {uncovered[:8]}"
    print(f"[coverage] all {len(body_keys)} body + {len(plain_text)} plain text weights resolved "
          f"(head={args.head}); 0 random-init weights", flush=True)

    results = {}
    for g in arms:
        ta = time.time()
        scale_elems, rel_stats = apply_body(model, base_bf16, g, device)
        torch.cuda.synchronize() if device == "cuda" else None
        summ = ppl_sweep(model, records, device, limit)
        scale_mb = scale_elems * 2 / 1e6  # bf16 scales = 2 bytes
        summ.update(group_size=g, scale_elems=scale_elems, scale_mb=round(scale_mb, 3),
                    rel_err_mean=round(rel_stats[1], 5), rel_err_max=round(rel_stats[2], 5),
                    verdict="PASS" if summ["ppl"] <= PPL_CAP else "FAIL",
                    secs=round(time.time() - ta, 1))
        results[str(g)] = summ
        print(f"[arm g={g}] ppl={summ['ppl']:.4f} cap={PPL_CAP} -> {summ['verdict']} | "
              f"scale={scale_mb:.2f}MB ({scale_elems} elems) | rel_err mean={rel_stats[1]:.4f} "
              f"max={rel_stats[2]:.4f} | {summ['secs']}s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"[done] wrote {args.out}", flush=True)

    if args.wandb:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
                         entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
                         name=f"ubel/marlin-groupsize-ppl-{args.head}head", group="marlin-groupsize-scalebw",
                         config={"arms": arms, "ppl_cap": PPL_CAP, "source": snap.name,
                                 "num_records": len(records),
                                 "head": args.head, "head_g": args.head_g, "device": device})
        tbl = wandb.Table(columns=["group_size", "ppl", "verdict", "scale_mb", "scale_elems",
                                   "rel_err_mean", "rel_err_max"])
        for g in arms:
            s = results[str(g)]
            wandb.log({f"ppl_g{g}": s["ppl"], f"scale_mb_g{g}": s["scale_mb"],
                       f"rel_err_mean_g{g}": s["rel_err_mean"]})
            tbl.add_data(g, s["ppl"], s["verdict"], s["scale_mb"], s["scale_elems"],
                         s["rel_err_mean"], s["rel_err_max"])
        wandb.summary["best_ppl_passing_groupsize"] = min(
            [g for g in arms if results[str(g)]["verdict"] == "PASS"], key=lambda g: (g == -1, g),
            default=128)
        wandb.log({"groupsize_ppl_scan": tbl})
        run.finish()
        print(f"[wandb] run {run.id}", flush=True)


if __name__ == "__main__":
    main()
